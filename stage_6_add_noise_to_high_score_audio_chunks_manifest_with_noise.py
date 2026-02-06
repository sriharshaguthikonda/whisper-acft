#!/usr/bin/env python3
"""stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise.py

Idempotent version of Stage 6 noise augmentation using stable UIDs.
- Only augments original rows (uid==base_uid, aug_stage empty)
- Deterministic selection per base_uid
- Deterministic RNG per (base_uid, stage, copy_idx)
- SQLite seen index prevents duplicates on resume/re-run
- Stable output filenames with embedded UIDs

Dependencies
------------
pip install numpy soundfile tqdm
(Optional for resampling) pip install scipy

Example
-------
python stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise.py \
  --in_manifest "i:/Record_chunks/pairs_manifest_stereo_english_only_filtered_with_uids_score_bottom_filtered.jsonl" \
  --out_manifest "i:/Record_chunks/only_noise_mixed.jsonl" \
  --noises_dir "i:/noise/RIRS_NOISES/pointsource_noises" \
  --scores_csv "i:/whisper-acft/speaker_sort_scores.csv" \
  --out_dir "i:/Record_chunks_noisy_mixed" \
  --stage_name "noise_mix" \
  --seen_db "I:\Record_chunks\seen_stage6_noise_mix.sqlite" \
  --ratio 0.5 \
  --copies 4 \
  --snr_db_min 5 --snr_db_max 20 \
  --max_bad_to_good_ratio 1.0 --good_floor_db -125 \
  --workers 4
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
    is_valid_wav,
    safe_unlink,
    atomic_write_wav_pcm16,
    make_aug_uid,
    rng_for,
    safe_beep,
    should_select,
)

# --- Audio utilities (copied from original) ---

_G_TARGET_SR = 16000
_G_NOISE_CHUNKS = []
_G_TARGET_KEYS: Set[str] = set()  # canonical paths + basenames for TARGET rows


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


def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.shape[1] == 1:
        return audio[:, 0]
    if audio.shape[1] >= 2:
        return np.mean(audio, axis=1)
    raise ValueError(f"Unexpected audio shape: {audio.shape}")


def _resample_if_needed(audio: np.ndarray, sr_in: int, target_sr: int) -> np.ndarray:
    if sr_in == target_sr:
        return audio
    # You can swap this out for a better resampler if you like.
    try:
        import scipy.signal
        
        num = target_sr
        den = sr_in
        audio = scipy.signal.resample_poly(audio, num, den)
        return audio
    except Exception:
        # Fallback: linear interpolation (keeps the stage running even if scipy isn't installed).
        # For noise augmentation this is perfectly acceptable.
        x = audio.astype(np.float32)
        n_out = int(round(len(x) * (target_sr / float(sr_in))))
        if n_out <= 1 or len(x) <= 1:
            return np.zeros((0,), dtype=np.float32)
        t_in = (np.arange(len(x), dtype=np.float32) / float(sr_in))
        t_out = (np.arange(n_out, dtype=np.float32) / float(target_sr))
        return np.interp(t_out, t_in, x).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio * audio)))


def _safe_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    atomic_write_wav_pcm16(path, audio, sr)


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
            y = _resample_if_needed(y, int(sr), int(target_sr))
        return y, int(target_sr)

    y = _ffmpeg_read_full_f32le(path, int(target_sr)).astype(np.float32)
    return y, int(target_sr)


def _load_noise_chunks(noises_dir: Path, max_chunk_sec: float = 30.0) -> None:
    """Load all noise files and chunk them."""
    global _G_NOISE_CHUNKS
    _G_NOISE_CHUNKS = []
    
    exts = ["*.wav", "*.flac", "*.ogg", "*.mp3", "*.m4a", "*.aac", "*.mp4", "*.opus", "*.wma"]
    noise_files = []
    for pat in exts:
        noise_files.extend(list(noises_dir.rglob(pat)))
    print(f"Found {len(noise_files)} noise files")
    
    for noise_file in tqdm(noise_files, desc="Loading noise files"):
        try:
            audio, sr = _read_audio_mono_to_sr(noise_file, _G_TARGET_SR)
            # audio is already at target_sr, no need to resample
            
            chunk_samples = int(max_chunk_sec * _G_TARGET_SR)
            for start in range(0, len(audio), chunk_samples):
                end = min(start + chunk_samples, len(audio))
                if end - start >= 0.5 * _G_TARGET_SR:  # At least 0.5 seconds
                    _G_NOISE_CHUNKS.append({
                        "audio_path": str(noise_file),
                        "source_audio": str(noise_file),
                        "chunk_start": float(start) / _G_TARGET_SR,
                        "chunk_end": float(end) / _G_TARGET_SR,
                        "chunk_samples": end - start,
                    })
        except Exception as e:
            print(f"Failed to load {noise_file}: {e}")
    
    print(f"Created {len(_G_NOISE_CHUNKS)} noise chunks")


def _is_target_speaker(row: Dict[str, Any], scores_csv: Optional[Path]) -> bool:
    """Check if row is from target speaker based on scores CSV."""
    if not scores_csv or not scores_csv.exists():
        return True
    if not _G_TARGET_KEYS:
        # If CSV provided but we failed to load it, do NOT accidentally exclude everything.
        return True
    ap = row.get("audio_path") or row.get("out_wav") or ""
    ap_can = canonicalise_path(ap)
    base = Path(ap).name.lower() if ap else ""
    return (ap_can in _G_TARGET_KEYS) or (base in _G_TARGET_KEYS)


def _load_target_keys(scores_csv: Path) -> None:
    """Load TARGET file keys once.

    Expects the Stage 3 CSV with columns: file, score, decision, reason, ...
    """
    global _G_TARGET_KEYS
    _G_TARGET_KEYS = set()
    if not scores_csv.exists():
        return

    with scores_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return

        # Common column names
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
            _G_TARGET_KEYS.add(canonicalise_path(fp))
            _G_TARGET_KEYS.add(Path(fp).name.lower())


def do_augmentation(row: Dict[str, Any], rng: random.Random, out_wav: str, args) -> Dict[str, Any]:
    """Implement noise augmentation for a single row."""
    if not _G_NOISE_CHUNKS:
        raise ValueError("No noise chunks loaded")
    
    ap = Path(row.get("audio_path", ""))
    if not ap.exists():
        raise FileNotFoundError(f"Audio file not found: {ap}")
    
    # Load speech
    speech, sr_in = _read_audio_mono_to_sr(ap, _G_TARGET_SR)
    
    speech_dur = float(len(speech)) / float(_G_TARGET_SR)
    if speech_dur <= 0.01:
        raise ValueError(f"Audio too short: {speech_dur:.3f}s")
    
    # Select noise chunk (deterministic via rng)
    noise_pick = None
    for _ in range(30):
        cand = rng.choice(_G_NOISE_CHUNKS)
        cand_len = float(cand["chunk_end"]) - float(cand["chunk_start"])
        if cand_len >= speech_dur:
            noise_pick = cand
            break
    if noise_pick is None:
        noise_pick = rng.choice(_G_NOISE_CHUNKS)
    
    # Load and prepare noise
    noise_audio_path = Path(noise_pick["audio_path"])
    if not noise_audio_path.exists():
        raise FileNotFoundError(f"Noise file not found: {noise_audio_path}")
    
    noise, nsr = _read_audio_mono_to_sr(noise_audio_path, _G_TARGET_SR)
    
    # Match noise length to speech
    if len(noise) < len(speech):
        reps = int(math.ceil(len(speech) / max(1, len(noise))))
        noise = np.tile(noise, reps)[: len(speech)]
    else:
        max_start = len(noise) - len(speech)
        if max_start > 0:
            start = rng.randint(0, max_start)
            noise = noise[start : start + len(speech)]
        else:
            noise = noise[: len(speech)]
    
    # Apply SNR
    snr_db = rng.uniform(args.snr_db_min, args.snr_db_max)
    rs = _rms(speech)
    rn = _rms(noise)
    target_rn = rs / (10.0 ** (snr_db / 20.0))
    gain = target_rn / max(rn, 1e-12)
    noise_scaled = noise * gain
    
    # Apply instantaneous ducker if available
    try:
        from audio_instant_ducker_utils import duck_bad_under_good
        noise_scaled = duck_bad_under_good(
            noise_scaled,
            good=speech,
            max_ratio=args.max_bad_to_good_ratio,
            good_floor_db=args.good_floor_db,
        )
    except ImportError:
        print("Warning: audio_instant_ducker_utils not available, skipping ducking")
    
    # Mix and normalize
    mixed = speech + noise_scaled
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    
    # Write output
    _safe_write_wav(Path(out_wav), mixed.astype(np.float32, copy=False), _G_TARGET_SR)
    if not is_valid_wav(out_wav, min_frames=16):
        safe_unlink(out_wav)
        raise RuntimeError("write-produced-invalid-wav")
    
    # Return metadata
    return {
        "snr_db": round(snr_db, 2),
        "noise_source": noise_pick.get("source_audio"),
        "noise_chunk_start": noise_pick["chunk_start"],
        "noise_chunk_end": noise_pick["chunk_end"],
        "max_bad_to_good_ratio": args.max_bad_to_good_ratio,
        "good_floor_db": args.good_floor_db,
    }


def build_out_wav_name(row: Dict[str, Any], stage_name: str, new_uid: str, copy_idx: int, out_dir: Path) -> str:
    """Build stable output filename with shorter format to avoid path length issues."""
    base_uid = row.get("base_uid", "")[:12]
    aug_uid = new_uid[:12]
    fname = f"{base_uid}_{aug_uid}__{stage_name}__copy{copy_idx:02d}.wav"
    return str(out_dir / fname)


def is_original_row(row: Dict[str, Any]) -> bool:
    """Check if row is an original (not augmented)."""
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
) -> tuple[bool, Optional[Dict[str, Any]], str]:
    """Process one row through augmentation."""
    base_uid = row.get("base_uid") or row.get("uid")
    if not base_uid:
        return False, None, "missing base_uid"
    
    aug_key = f"{base_uid}:{stage_name}:{copy_idx}"
    if seen.contains(aug_key):
        return True, None, "already-seen"
    
    new_uid = make_aug_uid(base_uid, stage_name, copy_idx)
    out_wav = build_out_wav_name(row, stage_name, new_uid, copy_idx, out_dir)
    
    # Idempotent check on disk (robust)
    out_wav_p = Path(out_wav)
    if out_wav_p.exists() and out_wav_p.stat().st_size > 0:
        if is_valid_wav(out_wav_p, min_frames=16):
            seen.add(aug_key)
            return True, None, "already-exists"
        safe_unlink(out_wav_p)
    
    # Check if this is a target speaker (if scores CSV provided)
    if args.scores_csv and not _is_target_speaker(row, Path(args.scores_csv)):
        return False, None, "not-target-speaker"
    
    # Perform augmentation
    try:
        rng = rng_for(base_uid, stage_name, copy_idx)
        meta = do_augmentation(row, rng, out_wav, args)
    except Exception as e:
        return False, None, f"augmentation-failed: {e}"
    
    # Build output row
    out_row = dict(row)
    out_row["parent_uid"] = row.get("uid") or base_uid
    out_row["base_uid"] = base_uid
    out_row["uid"] = new_uid
    out_row["aug_stage"] = stage_name
    out_row["aug_copy_idx"] = copy_idx
    out_row["out_wav"] = out_wav
    out_row["audio_path"] = out_wav  # Update audio_path to new file
    out_row.setdefault("aug_meta", {})
    out_row["aug_meta"] = {**out_row["aug_meta"], **(meta or {})}
    
    seen.add(aug_key)
    return True, out_row, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True, help="Input jsonl")
    ap.add_argument("--out_manifest", required=True, help="Output jsonl (augmented rows appended)")
    ap.add_argument("--noises_dir", required=True, help="Directory containing noise files")
    ap.add_argument("--scores_csv", help="CSV with speaker scores for target selection")
    ap.add_argument("--out_dir", required=True, help="Output directory for augmented audio")
    ap.add_argument("--stage_name", default="noise_mix", help="Stage name for UIDs")
    ap.add_argument("--ratio", type=float, default=0.5, help="Fraction of rows to augment")
    ap.add_argument("--copies", type=int, default=1, help="Number of augmented copies per selected row")
    ap.add_argument("--snr_db_min", type=float, default=5.0, help="Minimum SNR in dB")
    ap.add_argument("--snr_db_max", type=float, default=20.0, help="Maximum SNR in dB")
    ap.add_argument("--max_bad_to_good_ratio", type=float, default=1.0, help="Max noise-to-speech ratio")
    ap.add_argument("--good_floor_db", type=float, default=-120.0, help="Noise floor during speech")
    ap.add_argument("--max_chunk_sec", type=float, default=30.0, help="Maximum noise chunk duration")
    ap.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    ap.add_argument("--seen_db", default="", help="SQLite file; default is out_manifest + .seen.sqlite")
    ap.add_argument("--allow_augmented_input", action="store_true", help="Allow augmenting already augmented rows")
    args = ap.parse_args()
    
    # Setup paths
    in_path = Path(args.in_manifest)
    out_path = Path(args.out_manifest)
    out_dir = Path(args.out_dir)
    noises_dir = Path(args.noises_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup seen database (consistent with stages 7 and 9 pattern)
    if not args.seen_db:
        # Create explicit path like stages 7 and 9: manifest_parent/seen_stage6_{stage_name}.sqlite
        args.seen_db = str(out_path.parent / f"seen_stage6_{args.stage_name}.sqlite")
    seen = SQLiteSeenSet(args.seen_db)
    
    # Load target keys once (if provided)
    if args.scores_csv:
        try:
            _load_target_keys(Path(args.scores_csv))
            print(f"Loaded {len(_G_TARGET_KEYS)} TARGET keys from scores CSV")
        except Exception as e:
            print(f"Warning: failed to load scores CSV; treating all rows as target. Error: {e}")

    # Load noise chunks
    print("Loading noise chunks...")
    _load_noise_chunks(noises_dir, args.max_chunk_sec)
    if not _G_NOISE_CHUNKS:
        raise ValueError("No noise chunks loaded successfully")
    
    # Process rows
    futures = []
    n_total = 0
    n_selected = 0
    
    print(f"Processing manifest: {in_path}")
    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            # Schedule all tasks
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                n_total += 1
                
                # Skip augmented rows unless explicitly allowed
                if not args.allow_augmented_input and not is_original_row(row):
                    continue
                
                base_uid = row.get("base_uid") or row.get("uid")
                if not base_uid:
                    continue
                
                # Deterministic selection
                if not should_select(base_uid, args.stage_name, args.ratio):
                    continue
                
                n_selected += 1
                for copy_idx in range(1, args.copies + 1):
                    futures.append(ex.submit(process_one, row, args.stage_name, copy_idx, out_dir, args, seen))
            
            # Process results
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{args.stage_name} augment"):
                ok, out_row, status = fut.result()
                if out_row:
                    f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    
    seen.commit()
    seen.close()
    
    print(f"Stage {args.stage_name}: scanned {n_total} rows; selected {n_selected} base rows; emitted up to {n_selected*args.copies} aug rows")
    safe_beep()


if __name__ == "__main__":
    main()
