#!/usr/bin/env python3
"""stage_11_add_frequency_manipulation_idempotent.py

Idempotent Stage 11: frequency manipulation via mild pitch shifting.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from pipeline_uid_utils import (
    SQLiteSeenSet,
    default_seen_db,
    is_valid_wav,
    safe_unlink,
    atomic_write_wav_pcm16,
    make_aug_uid,
    rng_for,
    safe_beep,
    should_select,
)


def _ensure_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def _parse_choices(s: str) -> List[float]:
    out: List[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            continue
    return out


def _choose_semitones(rng, args) -> float:
    if args.mode == "choice" and args.semitones_choices:
        return float(rng.choice(args.semitones_choices))
    lo = float(args.semitones_min)
    hi = float(args.semitones_max)
    if hi < lo:
        lo, hi = hi, lo
    return float(rng.uniform(lo, hi))


def build_out_wav_name(row: Dict[str, Any], stage_name: str, new_uid: str, copy_idx: int, out_dir: Path) -> str:
    """Build stable output filename with shorter format to avoid path length issues."""
    base_uid = row.get("base_uid", "")[:12]
    aug_uid = new_uid[:12]
    fname = f"{base_uid}_{aug_uid}__{stage_name}__copy{copy_idx:02d}.wav"
    return str(out_dir / fname)


def is_original_row(row: Dict[str, Any]) -> bool:
    if row.get("aug_stage"):
        return False
    base_uid = row.get("base_uid")
    uid = row.get("uid") or base_uid
    if base_uid and uid and uid != base_uid:
        return False
    return True


def process_one(
    row: Dict[str, Any],
    stage_name: str,
    copy_idx: int,
    out_dir: Path,
    args,
    seen: SQLiteSeenSet,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    base_uid = row.get("base_uid") or row.get("uid")
    if not base_uid:
        return False, None, "missing base_uid"

    aug_key = f"{base_uid}:{stage_name}:{copy_idx}"
    if seen.contains(aug_key):
        return True, None, "already-seen"

    new_uid = make_aug_uid(base_uid, stage_name, copy_idx)
    out_wav = build_out_wav_name(row, stage_name, new_uid, copy_idx, out_dir)
    out_wav_p = Path(out_wav)
    if out_wav_p.exists() and out_wav_p.stat().st_size > 0:
        if is_valid_wav(out_wav_p, min_frames=16):
            seen.add(aug_key)
            return True, None, "already-exists"
        safe_unlink(out_wav_p)

    ap = Path(row.get("audio_path", ""))
    if not ap.exists():
        return False, None, f"missing-audio:{ap}"

    rng = rng_for(base_uid, stage_name, copy_idx)
    x, sr = sf.read(str(ap), dtype="float32", always_2d=False)
    x = _ensure_mono(x).astype(np.float32)
    if x.size < 16:
        return False, None, "too-short"

    semitones = _choose_semitones(rng, args)
    y = librosa.effects.pitch_shift(
        x,
        sr=int(sr),
        n_steps=float(semitones),
        bins_per_octave=int(args.bins_per_octave),
        res_type=args.res_type,
    ).astype(np.float32, copy=False)

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak * 0.98

    atomic_write_wav_pcm16(Path(out_wav), y, int(sr))
    if not is_valid_wav(out_wav, min_frames=16):
        safe_unlink(out_wav)
        return False, None, "write-produced-invalid-wav"

    out_row = dict(row)
    out_row["parent_uid"] = row.get("uid") or base_uid
    out_row["base_uid"] = base_uid
    out_row["uid"] = new_uid
    out_row["aug_stage"] = stage_name
    out_row["aug_copy_idx"] = copy_idx
    out_row["out_wav"] = out_wav
    out_row["audio_path"] = out_wav
    out_row.setdefault("aug_meta", {})
    out_row["aug_meta"] = {
        **out_row["aug_meta"],
        "pitch_semitones": float(semitones),
        "res_type": str(args.res_type),
    }

    seen.add(aug_key)
    return True, out_row, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stage_name", default="frequency_shift")
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--semitones_min", type=float, default=-1.0)
    ap.add_argument("--semitones_max", type=float, default=1.0)
    ap.add_argument("--mode", choices=["uniform", "choice"], default="uniform")
    ap.add_argument("--semitones_choices", default="-1.5,-1.0,-0.5,0.5,1.0,1.5")
    ap.add_argument("--bins_per_octave", type=int, default=12)
    ap.add_argument("--res_type", default="kaiser_fast")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seen_db", default="")
    ap.add_argument("--allow_augmented_input", action="store_true")
    args = ap.parse_args()

    args.semitones_choices = _parse_choices(args.semitones_choices)

    in_path = Path(args.in_manifest)
    out_path = Path(args.out_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_db = args.seen_db or default_seen_db(out_path, args.stage_name)
    seen = SQLiteSeenSet(seen_db)

    futures = []
    n_total = 0
    n_selected = 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
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
                    futures.append(ex.submit(process_one, row, args.stage_name, copy_idx, out_dir, args, seen))

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{args.stage_name} augment"):
                ok, out_row, status = fut.result()
                if out_row:
                    f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    seen.commit()
    seen.close()
    print(f"Stage {args.stage_name}: scanned {n_total} rows; selected {n_selected} base rows; emitted up to {n_selected*int(args.copies)}")
    safe_beep()


if __name__ == "__main__":
    main()
