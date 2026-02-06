#!/usr/bin/env python3
"""Run your Colab Groq transcription workflow locally (or in Colab) with resume + rate-limit handling.

What this replaces from your notebook
- No google.colab.drive mount
- No google.colab.userdata secrets
- Uses environment variables / a keys file for API keys
- CLI args for input/output folders

Key features
- Recursively transcribes all audio files under --input_dir
- Saves one JSON envelope per input file (same idea as your notebook)
- Resume: skips if the output JSON already exists
- Rate-limit handling: respects 429 + retry-after; optional conservative RPM throttling
- Optional decode-check (catches corrupted Takeout files before you waste API calls)
- Progress bars + beep when done

Groq docs (why this matters)
- Whisper free-tier limits include RPM/RPD and audio-seconds limits; rate limits apply at org level.
- Batch API exists, but for audio it uses URL inputs (not local paths), so your local Takeout files
  need direct upload unless you host them somewhere.

Usage (Windows)
  I:\Whisper-training-env\Scripts\python.exe groq_transcribe_local_main.py \
    --input_dir "I:\Record_harsha\groq_batches" \
    --out_dir   "I:\Record_harsha\groq_transcriptions" \
    --model whisper-large-v3 \
    --language en \
    --timestamp_granularities word segment \
    --decode_check

API key options
1) Single key via env var (recommended):
   setx GROQ_API_KEY "gsk_..."
   (new terminal) then run script

2) Multiple keys via env var names (only helps if they are in different orgs/projects):
   setx GROQ_API_KEY "..."
   setx GROQ_API_KEY_2 "..."
   python groq_transcribe_local_main.py --key_env_names GROQ_API_KEY GROQ_API_KEY_2 ...

3) Keys file (one per line):
   keys.txt contents:
     main=gsk_...
     alt=gsk_...
   python groq_transcribe_local_main.py --keys_file keys.txt ...

Notes
- For huge workloads, you can also consider Groq Batch API, but it expects a URL per audio request.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from tqdm import tqdm
from dotenv import load_dotenv


# ----------------------------- helpers (local module bootstrap) -----------------------------
# You asked for modular code. Canvas only gives me one file per turn, so this script will
# write its helper module next to itself on first run.

_UTILS_NAME = "groq_transcribe_local_utils.py"

_UTILS_CODE = r'''
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

AUDIO_EXTS = {
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".mpeg", ".mpga", ".aac"
}

GROQ_TRANSCRIBE_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


@dataclass(frozen=True)
class Key:
    name: str
    value: str


class KeyPool:
    """Round-robin key pool with cooldown.

    Important: Groq rate limits apply at the organisation level, so multiple keys from the same
    organisation won't increase throughput. This is still useful if you genuinely have keys under
    different orgs/projects, or if you want failover.
    """

    def __init__(self, keys: Sequence[Key], default_cooldown: float = 60.0, max_backoff_factor: int = 8):
        if not keys:
            raise ValueError("KeyPool: no keys provided")
        self.keys = list(keys)
        self.default_cooldown = float(default_cooldown)
        self.max_backoff_factor = int(max_backoff_factor)
        self.idx = 0
        self.next_ok: Dict[str, float] = {k.name: 0.0 for k in self.keys}
        self.fail_streak: Dict[str, int] = {k.name: 0 for k in self.keys}

    def get(self) -> Key:
        now = time.time()
        for _ in range(len(self.keys)):
            k = self.keys[self.idx]
            self.idx = (self.idx + 1) % len(self.keys)
            if now >= self.next_ok[k.name]:
                return k
        # all keys cooling down -> wait for earliest
        soonest = min(self.next_ok.values())
        time.sleep(max(1.0, soonest - now))
        return self.get()

    def cooldown(self, key_name: str, retry_after: Optional[float] = None) -> None:
        self.fail_streak[key_name] += 1
        base = float(retry_after) if retry_after else float(self.default_cooldown)
        factor = min(self.max_backoff_factor, 2 ** (self.fail_streak[key_name] - 1))
        wait = base * factor
        self.next_ok[key_name] = time.time() + wait


def which_or_die(exe: str) -> str:
    p = shutil.which(exe)
    if not p:
        raise SystemExit(f"ERROR: '{exe}' not found on PATH.")
    return p


def sha256_file(path: Path, block: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def list_audio_files(input_dir: Path, recursive: bool = True, exts: Optional[Sequence[str]] = None) -> List[Path]:
    ext_set = set((e.lower() if e.startswith(".") else "." + e.lower()) for e in (exts or AUDIO_EXTS))
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in ext_set]
    else:
        files = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in ext_set]
    files.sort(key=lambda p: str(p).lower())
    return files


def ffmpeg_decode_check(ffmpeg_bin: str, audio_path: Path, seconds: float = 0.0) -> Tuple[bool, str]:
    cmd = [ffmpeg_bin, "-hide_banner", "-v", "error", "-i", str(audio_path)]
    if seconds and seconds > 0:
        cmd += ["-t", f"{seconds}"]
    cmd += ["-f", "null", "-"]

    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode == 0:
        return True, ""
    err = (p.stderr or "").strip()
    return False, err[:2500]


def groq_transcribe_requests(
    audio_path: Path,
    api_key: str,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
    response_format: str,
    timestamp_granularities: Sequence[str],
    timeout_s: float,
) -> requests.Response:

    headers = {"Authorization": f"Bearer {api_key}"}

    # multipart form; repeated keys need list-of-tuples
    data: List[Tuple[str, str]] = [
        ("model", model),
        ("response_format", response_format),
        ("temperature", str(float(temperature))),
    ]
    if language:
        data.append(("language", language))
    if prompt:
        data.append(("prompt", prompt))
    for g in timestamp_granularities:
        # Groq docs: timestamp_granularities requires verbose_json
        data.append(("timestamp_granularities[]", g))

    with audio_path.open("rb") as f:
        files = {"file": (audio_path.name, f)}
        r = requests.post(
            GROQ_TRANSCRIBE_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=timeout_s,
        )
    return r


def beep_done() -> None:
    try:
        import winsound  # type: ignore

        winsound.Beep(1000, 350)
        winsound.Beep(1200, 350)
    except Exception:
        sys.stdout.write("\a")
        sys.stdout.flush()
'''


def _ensure_utils_written() -> None:
    here = Path(__file__).resolve().parent
    utils_path = here / _UTILS_NAME
    if utils_path.exists():
        return
    utils_path.write_text(_UTILS_CODE, encoding="utf-8")


_ensure_utils_written()

# now we can import our helper module
from groq_transcribe_local_utils import (  # type: ignore
    AUDIO_EXTS,
    GROQ_TRANSCRIBE_URL,
    Key,
    KeyPool,
    beep_done,
    ffmpeg_decode_check,
    groq_transcribe_requests,
    list_audio_files,
    sha256_file,
    which_or_die,
)


# ----------------------------- main -----------------------------


def _load_keys_from_env_names(env_names: Sequence[str]) -> List[Key]:
    keys: List[Key] = []
    for name in env_names:
        v = os.environ.get(name)
        if v:
            # Handle comma-separated keys (for GROQ_API_KEYS)
            if "," in v:
                for i, key_value in enumerate(v.split(",")):
                    key_value = key_value.strip()
                    if key_value:
                        keys.append(Key(name=f"{name}_{i+1}", value=key_value))
            else:
                keys.append(Key(name=name, value=v))
    return keys


def _load_keys_from_file(keys_file: Path) -> List[Key]:
    keys: List[Key] = []
    for line in keys_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, val = line.split("=", 1)
            name = name.strip()
            val = val.strip()
        else:
            name = f"key{len(keys)+1}"
            val = line
        if val:
            keys.append(Key(name=name, value=val))
    return keys


def _derive_output_json_path(out_dir: Path, audio_path: Path) -> Path:
    # Keep filenames unique even if there are duplicates: append a short hash of the full path.
    stem = audio_path.stem
    path_hash = hashlib.sha1(str(audio_path).encode("utf-8", errors="ignore")).hexdigest()[:10]
    out_name = f"{stem}__{path_hash}.json"
    return out_dir / out_name


def main() -> None:
    # Load environment variables from .env file
    load_dotenv()
    
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--recursive", action="store_true", default=True)
    ap.add_argument("--no_recursive", action="store_false", dest="recursive")

    ap.add_argument("--model", default="whisper-large-v3-turbo")
    ap.add_argument("--language", default="en")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--temperature", type=float, default=0.0)

    ap.add_argument("--response_format", default="verbose_json", choices=["json", "verbose_json", "text"])
    ap.add_argument("--timestamp_granularities", nargs="*", default=["word", "segment"], help="e.g. word segment")

    ap.add_argument("--timeout_s", type=float, default=float(os.getenv("GROQ_TIMEOUT_S", "300.0")))

    # Rate limiting: conservative local throttle (in addition to server-side 429 handling)
    ap.add_argument("--max_rpm", type=float, default=float(os.getenv("GROQ_MAX_RPM", "18.0")), help="Client-side throttle; set 0 to disable")
    ap.add_argument("--jitter_s", type=float, default=float(os.getenv("GROQ_JITTER_S", "0.25")), help="Random jitter added to sleeps")

    # Resume
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", action="store_false", dest="resume")

    # Decode check for Takeout corruption
    ap.add_argument("--decode_check", action="store_true", default=False)
    ap.add_argument("--decode_check_seconds", type=float, default=0.0, help="0=full file; else first N seconds")

    # API key inputs
    ap.add_argument("--api_key", default=None, help="Direct key (not recommended; prefer env)")
    ap.add_argument("--key_env_names", nargs="*", default=["GROQ_API_KEY", "GROQ_API_KEYS"], help="Env var names to look up")
    ap.add_argument("--keys_file", type=Path, default=None)

    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Keys
    keys: List[Key] = []
    if args.api_key:
        keys.append(Key("api_key", args.api_key))
    if args.keys_file:
        keys.extend(_load_keys_from_file(args.keys_file))
    keys.extend(_load_keys_from_env_names(args.key_env_names))

    # de-dup by value
    seen = set()
    uniq: List[Key] = []
    for k in keys:
        if k.value in seen:
            continue
        seen.add(k.value)
        uniq.append(k)
    keys = uniq

    if not keys:
        raise SystemExit(
            "No Groq API keys found. Set GROQ_API_KEY in your environment, or pass --keys_file / --api_key."
        )

    pool = KeyPool(keys)

    # Validate timestamp granularities + response_format
    if args.timestamp_granularities and args.response_format != "verbose_json":
        print("NOTE: timestamp_granularities requires response_format=verbose_json on Groq.")
        args.response_format = "verbose_json"

    # List inputs
    audio_files = list_audio_files(args.input_dir, recursive=args.recursive)
    if not audio_files:
        raise SystemExit(f"No audio files found under: {args.input_dir}")

    ffmpeg_bin = None
    if args.decode_check:
        ffmpeg_bin = which_or_die("ffmpeg")

    # Logs
    bad_decode_log = args.out_dir / "bad_files_decode_fail.txt"
    bad_api_log = args.out_dir / "bad_files_api_fail.txt"

    # Client-side throttle tracking
    min_interval = 60.0 / args.max_rpm if args.max_rpm and args.max_rpm > 0 else 0.0
    last_request_t = 0.0

    ok_count = 0
    skip_count = 0
    fail_count = 0

    for audio_path in tqdm(audio_files, desc="Transcribing", unit="file"):
        out_json = _derive_output_json_path(args.out_dir, audio_path)
        if args.resume and out_json.exists() and out_json.stat().st_size > 0:
            skip_count += 1
            continue

        # Decode check (catches corruption before wasting an API call)
        if args.decode_check and ffmpeg_bin:
            ok, err = ffmpeg_decode_check(ffmpeg_bin, audio_path, seconds=args.decode_check_seconds)
            if not ok:
                fail_count += 1
                with bad_decode_log.open("a", encoding="utf-8") as f:
                    f.write(f"{audio_path}\n{err}\n---\n")
                continue

        # Throttle
        if min_interval > 0:
            now = time.time()
            to_wait = (last_request_t + min_interval) - now
            if to_wait > 0:
                time.sleep(to_wait + random.uniform(0, args.jitter_s))

        attempts = 0
        max_attempts = 5

        while True:
            attempts += 1
            key = pool.get()

            t0 = time.time()
            try:
                r = groq_transcribe_requests(
                    audio_path=audio_path,
                    api_key=key.value,
                    model=args.model,
                    language=args.language,
                    prompt=args.prompt,
                    temperature=args.temperature,
                    response_format=args.response_format,
                    timestamp_granularities=args.timestamp_granularities,
                    timeout_s=args.timeout_s,
                )
                last_request_t = time.time()

                # Handle rate limit / transient errors
                if r.status_code == 429:
                    retry_after = r.headers.get("retry-after")
                    ra_s = float(retry_after) if retry_after and retry_after.replace('.', '', 1).isdigit() else None
                    pool.cooldown(key.name, retry_after=ra_s)
                    if attempts < max_attempts:
                        continue
                    r.raise_for_status()

                if 500 <= r.status_code < 600:
                    # server hiccup: backoff
                    pool.cooldown(key.name, retry_after=10)
                    if attempts < max_attempts:
                        continue
                    r.raise_for_status()

                r.raise_for_status()
                resp_json = r.json()

                # Save envelope (like your notebook)
                file_hash = sha256_file(audio_path)
                envelope = {
                    "request": {
                        "endpoint": GROQ_TRANSCRIBE_URL,
                        "model": args.model,
                        "language": args.language,
                        "prompt": args.prompt,
                        "temperature": args.temperature,
                        "response_format": args.response_format,
                        "timestamp_granularities": list(args.timestamp_granularities),
                        "api_key_name": key.name,
                    },
                    "input_file": {
                        "path": str(audio_path),
                        "name": audio_path.name,
                        "bytes": audio_path.stat().st_size,
                        "sha256": file_hash,
                    },
                    "http": {
                        "status_code": r.status_code,
                        "elapsed_seconds": round(time.time() - t0, 3),
                        "response_headers": dict(r.headers),
                    },
                    "groq_response": resp_json,
                }

                out_json.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
                ok_count += 1
                break

            except Exception as e:
                fail_count += 1
                # log the failure details
                with bad_api_log.open("a", encoding="utf-8") as f:
                    f.write(f"{audio_path}\nkey={key.name}\nattempt={attempts}\nerror={repr(e)}\n")
                    if hasattr(e, "response") and e.response is not None:  # type: ignore
                        try:
                            f.write(f"status={e.response.status_code}\ntext={e.response.text[:2000]}\n")  # type: ignore
                        except Exception:
                            pass
                    f.write("---\n")

                # cooldown + retry a bit for transient stuff
                pool.cooldown(key.name, retry_after=15)
                if attempts < max_attempts:
                    continue
                break

    print("\nDone")
    print(f"OK: {ok_count} | Skipped(existing): {skip_count} | Failed: {fail_count}")
    print(f"Output folder: {args.out_dir}")
    if bad_decode_log.exists():
        print(f"Decode failures log: {bad_decode_log}")
    if bad_api_log.exists():
        print(f"API failures log: {bad_api_log}")

    beep_done()


if __name__ == "__main__":
    main()
