#!/usr/bin/env python3
"""stage_7_add_others_voices_to_my_audio_fast_idempotent.py

Idempotent Stage 7: mix OTHER speaker audio under TARGET speaker chunks.

Key properties
--------------
- Only augments original rows by default (uid==base_uid, aug_stage empty)
- Deterministic selection per base_uid (should_select)
- Deterministic RNG per (base_uid, stage, copy_idx)
- SQLite seen index prevents duplicates on resume/re-run
- Stable filenames with embedded uid

Notes
-----
- Best performance is to preconvert OTHER voices to 16 kHz mono WAV.
- If you pass --scores_csv, only rows whose file is decision==TARGET will be mixed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

from pipeline_uid_utils import (
    SQLiteSeenSet,
    canonicalise_path,
    default_seen_db,
    is_valid_wav,
    safe_unlink,
    atomic_write_wav_pcm16,
    make_aug_uid,
    rng_for,
    safe_beep,
    should_select,
)


_TARGET_KEYS: Set[str] = set()  # canonical paths + basenames for TARGET rows


def _find_ffmpeg() -> Optional[str]:
    cand = shutil.which("ffmpeg")
    if cand:
        return cand
    for p in [
        r"C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe",
        r"C:\\ffmpeg\\bin\\ffmpeg.exe",
    ]:
        if os.path.exists(p):
            return p
    return None


def _ffmpeg_read_full_f32le(path: Path, target_sr: int) -> np.ndarray:
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH (or common locations)")
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(target_sr)),
        "-f",
        "f32le",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    if not raw:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(raw, dtype=np.float32)


def _resample_poly(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    try:
        import scipy.signal

        return scipy.signal.resample_poly(x, sr_out, sr_in).astype(np.float32)
    except Exception:
        # Fallback: deterministic linear interpolation (good enough for mixing)
        if x.size == 0:
            return x.astype(np.float32)
        t_in = np.linspace(0.0, 1.0, num=x.size, endpoint=False)
        n_out = int(round(x.size * (sr_out / float(sr_in))))
        t_out = np.linspace(0.0, 1.0, num=max(1, n_out), endpoint=False)
        return np.interp(t_out, t_in, x).astype(np.float32)


def _read_audio_mono_to_sr(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    """Read any audio into mono float32 at target_sr.

    - WAV/FLAC/OGG via soundfile (with resample if needed)
    - Everything else via ffmpeg decode/resample
    """
    suf = path.suffix.lower()
    if suf in {".wav", ".flac", ".ogg"}:
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        y = _ensure_mono(y).astype(np.float32)
        if int(sr) != int(target_sr):
            y = _resample_poly(y, int(sr), int(target_sr))
        return y, int(target_sr)

    y = _ffmpeg_read_full_f32le(path, int(target_sr)).astype(np.float32)
    return y, int(target_sr)


def _ensure_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def _rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))


def _safe_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    atomic_write_wav_pcm16(path, audio, sr)


def duck_bad_under_good(bad: np.ndarray, good: np.ndarray, max_ratio: float, good_floor_db: float) -> np.ndarray:
    """Samplewise limiter: ensures |bad| <= max_ratio * max(|good|, floor)."""
    bad = np.asarray(bad, dtype=np.float32)
    good = np.asarray(good, dtype=np.float32)
    floor = float(10.0 ** (float(good_floor_db) / 20.0))
    allowed = float(max_ratio) * np.maximum(np.abs(good), floor)
    return np.clip(bad, -allowed, allowed)


def _load_target_keys(scores_csv: Path) -> None:
    global _TARGET_KEYS
    _TARGET_KEYS = set()
    if not scores_csv.exists():
        return
    with scores_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return
        file_col = "file" if "file" in reader.fieldnames else ("audio_path" if "audio_path" in reader.fieldnames else None)
        decision_col = "decision" if "decision" in reader.fieldnames else None
        if not file_col:
            return
        for r in reader:
            decision = (r.get(decision_col) or "").strip().upper() if decision_col else "TARGET"
            if decision != "TARGET":
                continue
            fp = (r.get(file_col) or "").strip()
            if not fp:
                continue
            _TARGET_KEYS.add(canonicalise_path(fp))
            _TARGET_KEYS.add(Path(fp).name.lower())


def _is_target_row(row: Dict[str, Any], scores_csv: Optional[Path]) -> bool:
    if not scores_csv:
        return True
    if not _TARGET_KEYS:
        return True
    ap = row.get("audio_path") or row.get("out_wav") or ""
    return (canonicalise_path(ap) in _TARGET_KEYS) or (Path(ap).name.lower() in _TARGET_KEYS)


def _pick_and_fit_other(
    rng: random.Random,
    other_files: list[Path],
    n_samples: int,
    target_sr: int,
) -> Tuple[np.ndarray, str]:
    p_other = Path(other_files[rng.randrange(len(other_files))])
    y, sr = _read_audio_mono_to_sr(p_other, target_sr)
    if y.size == 0:
        return np.zeros((n_samples,), dtype=np.float32), str(p_other)

    if len(y) < n_samples:
        reps = int(math.ceil(n_samples / max(1, len(y))))
        y = np.tile(y, reps)
    if len(y) > n_samples:
        start = rng.randrange(0, len(y) - n_samples + 1)
        y = y[start : start + n_samples]
    return y.astype(np.float32), str(p_other)


def build_out_wav_name(row: Dict[str, Any], stage_name: str, new_uid: str, copy_idx: int, out_dir: Path) -> str:
    """Build stable output filename with shorter format to avoid path length issues."""
    base_uid = row.get("base_uid", "")[:12]
    aug_uid = new_uid[:12]
    fname = f"{base_uid}_{aug_uid}__{stage_name}__copy{copy_idx:02d}.wav"
    return str(out_dir / fname)


def is_original_row(row: Dict[str, Any]) -> bool:
    if row.get("aug_stage"):
        return False
    base_uid = row.get("base_uid") or row.get("uid")
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
    other_files: list[Path],
    seen: SQLiteSeenSet,
) -> tuple[bool, Optional[Dict[str, Any]], str]:
    base_uid = row.get("base_uid") or row.get("uid")
    if not base_uid:
        return False, None, "missing base_uid"

    aug_key = f"{base_uid}:{stage_name}:{copy_idx}"
    if seen.contains(aug_key):
        return True, None, "already-seen"

    if args.scores_csv and not _is_target_row(row, Path(args.scores_csv)):
        return False, None, "not-target"

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

    # Read good audio and enforce target_sr deterministically.
    good, sr = _read_audio_mono_to_sr(ap, int(args.target_sr))

    if good.size < 16:
        return False, None, "too-short"

    bad, other_src = _pick_and_fit_other(rng, other_files, len(good), args.target_sr)

    snr_db = rng.uniform(float(args.snr_db_min), float(args.snr_db_max))
    g_rms = _rms(good)
    b_rms = _rms(bad)
    if b_rms <= 1e-12 or g_rms <= 1e-12:
        bad_scaled = np.zeros_like(good)
    else:
        # SNR = 20 log10(g_rms / n_rms)
        target_b_rms = g_rms / (10.0 ** (snr_db / 20.0))
        bad_scaled = bad * (target_b_rms / b_rms)

    bad_scaled = duck_bad_under_good(bad_scaled, good, float(args.max_bad_to_good_ratio), float(args.good_floor_db))
    y = good + bad_scaled

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
    out_row["aug_meta"] = {
        **out_row["aug_meta"],
        "snr_db": float(snr_db),
        "other_source": other_src,
        "max_bad_to_good_ratio": float(args.max_bad_to_good_ratio),
        "good_floor_db": float(args.good_floor_db),
    }

    seen.add(aug_key)
    return True, out_row, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--other_voices_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scores_csv", default="")
    ap.add_argument("--stage_name", default="voice_mix")
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--snr_db_min", type=float, default=5.0)
    ap.add_argument("--snr_db_max", type=float, default=10.0)
    ap.add_argument("--max_bad_to_good_ratio", type=float, default=1.0)
    ap.add_argument("--good_floor_db", type=float, default=-45.0)
    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seen_db", default="")
    ap.add_argument("--allow_augmented_input", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.in_manifest)
    out_path = Path(args.out_manifest)
    other_dir = Path(args.other_voices_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.scores_csv:
        _load_target_keys(Path(args.scores_csv))
        print(f"Loaded {len(_TARGET_KEYS)} TARGET keys")

    # Accept common audio types; decode/resample handled in _read_audio_mono_to_sr
    allowed = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".mp4", ".opus", ".wma"}
    other_files = [p for p in other_dir.rglob("*") if p.suffix.lower() in allowed]
    if not other_files:
        raise SystemExit(f"No audio files found in other_voices_dir: {other_dir}")

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
                    futures.append(
                        ex.submit(process_one, row, args.stage_name, copy_idx, out_dir, args, other_files, seen)
                    )

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
