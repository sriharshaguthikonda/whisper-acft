#!/usr/bin/env python3
"""stage_9_add_reverb_idempotent.py

Idempotent Stage 9: apply RIR convolution (reverb) to selected rows.




& "I:\Whisper-training-env\Scripts\python.exe" "I:\whisper-acft\stage_9_add_reverb_idempotent.py" `
  --in_manifest  "I:\Record_chunks\pairs_manifest_stereo_english_only_filtered.jsonl" `
  --out_manifest "I:\Record_chunks\pairs_manifest_stage7_plus_stage9_reverb.jsonl" `
  --rir_dir "I:\RIRS_NOISES\real_rirs_isotropic_noises\RIRS_NOISES\simulated_rirs" `
  --out_dir "I:\Record_chunks_reverb" `
  --ratio 0.3 `
  --copies 1 `
  --workers 4 `
  --stage_name reverb `
  --seen_db "I:\Record_chunks\seen_stage9_reverb.sqlite"




Assumptions:
- Input audio is already 16 kHz mono WAV (recommended)
- RIR files are WAV/FLAC/OGG (usually 16 kHz)
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

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


def _safe_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    atomic_write_wav_pcm16(path, audio, sr)


def _fft_convolve_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Fast-ish convolution returning same length as x."""
    try:
        import scipy.signal

        y = scipy.signal.fftconvolve(x, h, mode="full")
    except Exception:
        y = np.convolve(x, h, mode="full")
    return y[: len(x)].astype(np.float32)


def _resample_poly(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    try:
        import scipy.signal

        return scipy.signal.resample_poly(x, sr_out, sr_in).astype(np.float32)
    except Exception:
        # Fallback: deterministic linear interpolation
        if x.size == 0:
            return x.astype(np.float32)
        t_in = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
        n_out = int(round(x.size * (sr_out / float(sr_in))))
        t_out = np.linspace(0.0, 1.0, num=max(1, n_out), endpoint=False)
        return np.interp(t_out, t_in, x).astype(np.float32)


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
    rir_files: list[Path],
    seen: SQLiteSeenSet,
) -> tuple[bool, Optional[Dict[str, Any]], str]:
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

    rir_path = rng.choice(rir_files)
    rir, sr_r = sf.read(rir_path, dtype="float32", always_2d=False)
    if sr_r != sr:
        # RIR corpora often mix sampling rates; resample instead of dropping the file.
        rir = _resample_poly(rir.astype(np.float32), sr_r, sr).astype(np.float32)
        sr_r = sr

    # RIR datasets are often 48k; resample deterministically to match speech SR.
    if int(sr_r) != int(sr):
        rir = _resample_poly(rir, int(sr_r), int(sr))

    # Energy normalise RIR
    rir = rir / (float(np.sqrt(np.sum(rir * rir))) + 1e-12)

    wet = rng.uniform(float(args.wet_min), float(args.wet_max))
    y_conv = _fft_convolve_same(x, rir)
    y = (1.0 - wet) * x + wet * y_conv

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = y / peak * 0.98

    _safe_write_wav(Path(out_wav), y, int(sr))
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
    out_row["aug_meta"] = {**out_row["aug_meta"], "rir_source": str(rir_path), "wet": float(wet)}

    seen.add(aug_key)
    return True, out_row, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--rir_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stage_name", default="reverb")
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--wet_min", type=float, default=0.2)
    ap.add_argument("--wet_max", type=float, default=0.8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seen_db", default="")
    ap.add_argument("--allow_augmented_input", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.in_manifest)
    out_path = Path(args.out_manifest)
    rir_dir = Path(args.rir_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rir_files = [p for p in rir_dir.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".ogg"}]
    if not rir_files:
        raise SystemExit(f"No RIR audio files found in: {rir_dir}")

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
                    futures.append(ex.submit(process_one, row, args.stage_name, copy_idx, out_dir, args, rir_files, seen))

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
