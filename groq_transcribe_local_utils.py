
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
