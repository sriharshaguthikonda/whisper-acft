#!/usr/bin/env python3
"""stage_10_add_speech_tempo_perturbation_idempotent.py

Idempotent Stage 10: speech-optimised tempo (speed) perturbation using SoX:

  sox in.wav out.wav tempo -s <factor>

This changes tempo (duration) WITHOUT changing pitch, using WSOLA.
It's a good approximation to "naturally faster" (but not a perfect one).

Typical use (mild factors): 1.05–1.20

Example:

& "I:\Whisper-training-env\Scripts\python.exe" "I:\whisper-acft\stage_10_add_speech_tempo_perturbation_idempotent.py" `
  --in_manifest  "I:\Record_chunks\pairs_manifest_combined_all_datasets.jsonl" `
  --out_manifest "I:\Record_chunks\pairs_manifest_stage10_tempo.jsonl" `
  --out_dir      "I:\Record_chunks_tempo" `
  --ratio 0.30 `
  --copies 1 `
  --tempo_min 1.05 --tempo_max 1.20 `
  --tempo_factors "1.05,1.10,1.15,1.20" `
  --mode choice `
  --workers 8 `
  --stage_name tempo_speech `
  --seen_db "I:\Record_chunks\seen_stage10_tempo.sqlite"

Notes
- Requires SoX installed and available on PATH (or pass --sox "C:\\path\\to\\sox.exe").
- Output rows are appended to --out_manifest (idempotent via --seen_db + output validation).
- Designed to match the idempotent patterns of your Stages 8/9.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from tqdm import tqdm

from pipeline_uid_utils import (
    SQLiteSeenSet,
    default_seen_db,
    is_valid_wav,
    safe_unlink,
    make_aug_uid,
    rng_for,
    safe_beep,
    should_select,
)


def is_original_row(row: Dict[str, Any]) -> bool:
    # Same logic as Stage 8/9
    if row.get("aug_stage"):
        return False
    base_uid = row.get("base_uid")
    uid = row.get("uid") or base_uid
    if base_uid and uid and uid != base_uid:
        return False
    return True


def _parse_factors(s: str) -> List[float]:
    out: List[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            continue
    # keep stable ordering for deterministic RNG.choice
    out = [x for x in out if x > 0]
    return out


def _quantise_factor(x: float) -> float:
    # Keep filenames/UIDs stable + avoid silly float noise.
    return float(round(float(x), 3))


def choose_tempo_factor(base_uid: str, stage_name: str, copy_idx: int, args) -> float:
    rng = rng_for(base_uid, stage_name, copy_idx)

    if args.mode == "choice":
        factors = _parse_factors(args.tempo_factors)
        if not factors:
            # fallback
            lo = float(args.tempo_min)
            hi = float(args.tempo_max)
            return _quantise_factor(rng.uniform(lo, hi))
        return _quantise_factor(rng.choice(factors))

    # random_uniform
    lo = float(args.tempo_min)
    hi = float(args.tempo_max)
    if hi <= 0 or lo <= 0 or hi < lo:
        lo, hi = 1.05, 1.20
    return _quantise_factor(rng.uniform(lo, hi))


def build_out_wav_name(
    row: Dict[str, Any],
    stage_name: str,
    new_uid: str,
    copy_idx: int,
    tempo_factor: float,
    out_dir: Path,
) -> str:
    base_uid = (row.get("base_uid") or row.get("uid") or "")[:12]
    aug_uid = (new_uid or "")[:12]
    ftag = f"{tempo_factor:.2f}".replace(".", "p")  # 1.10 -> 1p10
    fname = f"{base_uid}_{aug_uid}__{stage_name}__t{ftag}__c{copy_idx:02d}.wav"
    return str(out_dir / fname)


def _sox_tempo_speech(
    sox_exe: str,
    in_wav: Path,
    out_tmp: Path,
    tempo_factor: float,
    sample_rate: int,
    channels: int,
    bit_depth: int,
) -> Tuple[bool, str]:
    """Run SoX tempo -s into out_tmp.

    Returns (ok, status).
    """
    # SoX file-format options apply to the *next* file token.
    # Pattern: sox IN [output-format-opts] OUT tempo -s FACTOR
    cmd = [
        sox_exe,
        str(in_wav),
        "-r",
        str(int(sample_rate)),
        "-c",
        str(int(channels)),
        "-b",
        str(int(bit_depth)),
        str(out_tmp),
        "tempo",
        "-s",
        f"{float(tempo_factor):.3f}",
    ]

    try:
        cp = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
        if cp.returncode != 0:
            err = (cp.stderr or cp.stdout or "").strip()
            if len(err) > 300:
                err = err[:300] + " …"
            return False, f"sox-failed({cp.returncode}): {err}"
        return True, "ok"
    except FileNotFoundError:
        return False, f"sox-not-found:{sox_exe}"
    except Exception as e:
        return False, f"sox-exception:{type(e).__name__}:{e}"


def process_one(
    row: Dict[str, Any],
    stage_name: str,
    copy_idx: int,
    out_dir: Path,
    args,
    seen: SQLiteSeenSet,
    lock: threading.Lock,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    base_uid = row.get("base_uid") or row.get("uid")
    if not base_uid:
        return False, None, "missing base_uid"

    tempo = choose_tempo_factor(base_uid, stage_name, copy_idx, args)
    extra = f"tempo{tempo:.3f}"

    aug_key = f"{base_uid}:{stage_name}:{copy_idx}:{tempo:.3f}"
    if seen.contains(aug_key):
        return True, None, "already-seen"

    new_uid = make_aug_uid(base_uid, stage_name, copy_idx, extra=extra)
    out_wav = build_out_wav_name(row, stage_name, new_uid, copy_idx, tempo, out_dir)
    out_wav_p = Path(out_wav)

    # If output already exists and looks valid, mark seen and move on.
    if out_wav_p.exists() and out_wav_p.stat().st_size > 0:
        if is_valid_wav(out_wav_p, min_frames=16):
            seen.add(aug_key)
            return True, None, "already-exists"
        safe_unlink(out_wav_p)

    in_ap = Path(row.get("audio_path", ""))
    if not in_ap.exists():
        return False, None, f"missing-audio:{in_ap}"

    if args.dry_run:
        out_row = dict(row)
        out_row["parent_uid"] = row.get("uid") or base_uid
        out_row["base_uid"] = base_uid
        out_row["uid"] = new_uid
        out_row["aug_stage"] = stage_name
        out_row["aug_copy_idx"] = copy_idx
        out_row["out_wav"] = out_wav
        out_row["audio_path"] = out_wav
        out_row.setdefault("aug_meta", {})
        out_row["aug_meta"] = {**out_row["aug_meta"], "tempo_factor": float(tempo), "dry_run": True}
        seen.add(aug_key)
        return True, out_row, "dry-run"

    # Atomic-ish write: render to tmp, validate, then os.replace.
    tmp = out_wav_p.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}.wav")
    safe_unlink(tmp)

    ok, status = _sox_tempo_speech(
        args.sox,
        in_ap,
        tmp,
        tempo_factor=float(tempo),
        sample_rate=int(args.sample_rate),
        channels=int(args.channels),
        bit_depth=int(args.bit_depth),
    )
    if not ok:
        safe_unlink(tmp)
        return False, None, status

    if not is_valid_wav(tmp, min_frames=16):
        safe_unlink(tmp)
        return False, None, "write-produced-invalid-wav"

    try:
        os.replace(str(tmp), str(out_wav_p))
    except Exception as e:
        safe_unlink(tmp)
        return False, None, f"replace-failed:{type(e).__name__}:{e}"

    if not is_valid_wav(out_wav_p, min_frames=16):
        safe_unlink(out_wav_p)
        return False, None, "final-invalid-wav"

    out_row = dict(row)
    out_row["parent_uid"] = row.get("uid") or base_uid
    out_row["base_uid"] = base_uid
    out_row["uid"] = new_uid
    out_row["aug_stage"] = stage_name
    out_row["aug_copy_idx"] = copy_idx
    out_row["out_wav"] = str(out_wav_p)
    out_row["audio_path"] = str(out_wav_p)

    out_row.setdefault("aug_meta", {})
    out_row["aug_meta"] = {
        **out_row["aug_meta"],
        "tempo_factor": float(tempo),
        "sox_effect": "tempo -s",
    }

    seen.add(aug_key)
    return True, out_row, "ok"


def _ensure_sox_available(sox_exe: str) -> str:
    # allow explicit full path
    p = Path(sox_exe)
    if p.exists():
        return str(p)
    found = shutil.which(sox_exe)
    if found:
        return found
    raise SystemExit(
        "SoX was not found. Install it or pass --sox <path-to-sox.exe>. "
        "On Windows you can use Chocolatey: choco install sox.portable"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--stage_name", default="tempo_speech")
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)

    ap.add_argument("--sox", default="sox", help="Path to sox.exe or 'sox' if on PATH")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--bit_depth", type=int, default=16)

    ap.add_argument("--tempo_min", type=float, default=1.05)
    ap.add_argument("--tempo_max", type=float, default=1.20)
    ap.add_argument("--tempo_factors", default="1.05,1.10,1.15,1.20")
    ap.add_argument("--mode", choices=["random_uniform", "choice"], default="choice")

    ap.add_argument("--seen_db", default="")
    ap.add_argument("--allow_augmented_input", action="store_true")
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    in_path = Path(args.in_manifest)
    out_path = Path(args.out_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    args.sox = _ensure_sox_available(args.sox)

    seen_db = args.seen_db or default_seen_db(out_path, args.stage_name)
    seen = SQLiteSeenSet(seen_db)

    # Streaming + bounded pending futures (prevents holding 100k futures in RAM)
    max_pending = max(8, int(args.workers) * 4)
    pending: List[Future] = []
    lock = threading.Lock()

    n_total = 0
    n_selected = 0
    n_submitted = 0

    def flush_one(fut: Future, f_out) -> None:
        ok, out_row, status = fut.result()
        if out_row:
            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            pbar = tqdm(total=0, desc=f"{args.stage_name} augment", unit="job")
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                n_total += 1

                if not args.allow_augmented_input and not is_original_row(row):
                    continue

                base_uid = row.get("base_uid") or row.get("uid")
                if not base_uid:
                    continue

                if not should_select(base_uid, args.stage_name, float(args.ratio)):
                    continue

                n_selected += 1
                for copy_idx in range(1, int(args.copies) + 1):
                    fut = ex.submit(process_one, row, args.stage_name, copy_idx, out_dir, args, seen, lock)
                    pending.append(fut)
                    n_submitted += 1

                    # Backpressure
                    if len(pending) >= max_pending:
                        done = pending.pop(0)
                        flush_one(done, f_out)
                        pbar.total = n_submitted
                        pbar.update(1)

            # Flush remaining
            for fut in pending:
                flush_one(fut, f_out)
                pbar.total = n_submitted
                pbar.update(1)
            pbar.close()

    seen.commit()
    seen.close()

    print(
        f"Stage {args.stage_name}: scanned {n_total} rows; selected {n_selected} base rows; "
        f"submitted {n_submitted} job(s) (copies={int(args.copies)})."
    )
    safe_beep()


if __name__ == "__main__":
    main()
