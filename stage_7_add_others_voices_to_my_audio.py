#!/usr/bin/env python3
"""stage_7_add_others_voices_to_my_audio.py

Goal
----
Given a Whisper-style JSONL manifest and speaker scores, augment TARGET audio files
by mixing them with OTHER voice files (poor scores but non-zero):

1) Load TARGET files (high scores) and OTHER files (poor scores but non-zero)
2) Chunk OTHER voice files into <= 30s WAV segments (16 kHz by default)
3) Mix OTHER voice chunks onto TARGET voice files at random SNRs
4) Create new audio files with TARGET transcripts (not OTHER transcripts)
5) Add mixed files to manifest with appropriate metadata

This script is designed for your manifest schema:
  {
    "audio_path": "...wav",
    "raw_transcription": "...",
    "source_audio": "...m4a",
    "chunk_index": 0,
    "chunk_start": 0.0,
    "chunk_end": 2.40,
    "transcript_json": "...json"
  }

Mixed rows will use:
  - TARGET transcript and metadata
  - Additional fields for voice mixing info
  - is_voice_mixed: true

Dependencies
------------
pip install numpy soundfile tqdm
(Optional for resampling) pip install scipy

Usage examples
--------------
# Mix OTHER voices onto TARGET files using I:\Record_others directory:



# Use custom other voices directory:
cd "i:\whisper-acft" && python stage_7_add_others_voices_to_my_audio.py \
  --manifest "i:/Record_chunks/pairs_manifest_filtered_with_noises_with_mix.jsonl" \
  --scores_csv "i:/whisper-acft/speaker_sort_scores.csv" \
  --other_voices_dir "I:/Custom/Other/Voices" \
  --out_manifest "i:/Record_chunks/pairs_manifest_filtered_with_noises_and_others_voices_mixed.jsonl" \
  --out_mix_dir "i:/Record_chunks/voice_mixed" \
  --mix_ratio 0.5 \
  --snr_db_min 5 \
  --snr_db_max 20 \
  --seed 1337 \
  --shuffle
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------

def _norm_path(p: str | Path) -> str:
    # Keep Windows-style paths readable in JSONL
    return str(Path(p))


def _is_audio_file(p: Path) -> bool:
    # soundfile supports wav/flac/ogg and some others depending on libsndfile.
    # Keep this conservative but add m4a support for this pipeline.
    return p.suffix.lower() in {".wav", ".flac", ".ogg", ".m4a"}


def _rms(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))


def _ensure_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    # Average channels
    return np.mean(x, axis=1)


def _resample_if_needed(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    try:
        from scipy.signal import resample_poly
    except Exception as e:
        raise RuntimeError(
            "Sample rate mismatch (sr_in != sr_out) but SciPy is not available. "
            "Install scipy or ensure all audio is already at the target sample rate."
        ) from e

    # Polyphase resampling: up/down ratio reduced by gcd
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    y = resample_poly(x, up=up, down=down)
    return y.astype(np.float32, copy=False)


def _safe_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # PCM_16 keeps files small and consistent
    sf.write(str(path), audio, sr, subtype="PCM_16")


def load_target_and_other_files(scores_csv: Path, other_voices_dir: Path = None) -> Tuple[set[str], List[Tuple[Path, float]]]:
    """Load TARGET files set and OTHER files from either CSV scores or a directory.
    
    If other_voices_dir is provided, scan it for audio files instead of using CSV.
    Returns OTHER files sorted by score (low to high) - lower scores = more likely other speakers.
    """
    target_files = set()
    other_files_with_scores = []
    
    # Load target files from CSV
    with scores_csv.open("r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                file_path = parts[0]
                score_str = parts[1]
                decision = parts[2]
                
                # Normalize path: lowercase and forward slashes
                norm_path = file_path.lower().replace("\\", "/")
                
                if decision == "TARGET":
                    target_files.add(norm_path)
    
    # Load other voice files
    if other_voices_dir and other_voices_dir.exists():
        # Scan directory for audio files
        print(f"Scanning {other_voices_dir} for other voice files...")
        for audio_file in other_voices_dir.rglob("*"):
            if audio_file.is_file() and _is_audio_file(audio_file):
                # Assign a default score (0.5) for directory files
                other_files_with_scores.append((audio_file, 0.5))
        print(f"Found {len(other_files_with_scores)} audio files in {other_voices_dir}")
    else:
        # Load from CSV (original behavior)
        with scores_csv.open("r", encoding="utf-8") as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 3:
                    file_path = parts[0]
                    score_str = parts[1]
                    decision = parts[2]
                    
                    if decision == "OTHER" and score_str and score_str != "":
                        try:
                            score = float(score_str)
                            # Only include OTHER files with non-zero scores
                            if score > 0:
                                other_files_with_scores.append((Path(file_path), score))
                        except ValueError:
                            # Skip if score is not a valid number
                            continue
    
    # Sort OTHER files by score (low to high) - lower scores = more likely other speakers
    other_files_with_scores.sort(key=lambda x: x[1])
    
    return target_files, other_files_with_scores


def load_manifest(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Failed parsing JSON on line {i} of {path}: {e}")
    return rows


def write_manifest(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_wav_any(path: Path) -> Tuple[np.ndarray, int]:
    # Use soundfile; assumes wav/flac/ogg supported
    # Convert m4a to wav first if needed
    if path.suffix.lower() == ".m4a":
        wav_path = path.with_suffix(".wav")
        if not wav_path.exists() or wav_path.stat().st_mtime < path.stat().st_mtime:
            # Convert m4a to wav using ffmpeg
            import subprocess
            cmd = ["ffmpeg", "-y", "-i", str(path), str(wav_path)]
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"Converted {path} to {wav_path}")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to convert {path} to wav: {e}")
        path = wav_path
    
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = _ensure_mono(audio)
    return audio, int(sr)


def _extract_random_segment(audio: np.ndarray, sr: int, target_duration: float, rng: random.Random) -> Tuple[np.ndarray, float]:
    """Extract a random segment of target_duration from audio, or entire audio if shorter."""
    audio_duration = len(audio) / sr
    if audio_duration <= target_duration:
        return audio, audio_duration
    
    # Extract random segment
    max_start = audio_duration - target_duration
    start_time = rng.uniform(0, max_start)
    start_sample = int(start_time * sr)
    end_sample = int((start_time + target_duration) * sr)
    
    return audio[start_sample:end_sample], target_duration


def mix_other_voices_into_target(
    base_rows: List[Dict],
    other_files_with_scores: List[Tuple[Path, float]],
    out_mix_dir: Path,
    target_sr: int,
    mix_ratio: float,
    snr_db_min: float,
    snr_db_max: float,
    allow_overwrite: bool,
    target_files: set[str],
) -> List[Dict]:
    """Create voice-mixed copies for a subset of TARGET base_rows and return new rows.
    
    Uses score-based selection: lower scores = more likely other speakers.
    Prioritizes OTHER files with lowest scores for mixing.
    """

    out_mix_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only target files first
    target_rows = [row for row in base_rows if row["audio_path"].lower().replace("\\", "/") in target_files]
    if not target_rows:
        print("[WARN] No target files found in manifest for voice mixing", file=sys.stderr)
        return []

    print(f"Found {len(target_rows)} target files out of {len(base_rows)} total rows")
    print(f"Using {len(other_files_with_scores)} OTHER files sorted by score (low to high)")

    n_to_mix = int(round(len(target_rows) * float(mix_ratio)))
    n_to_mix = max(0, min(len(target_rows), n_to_mix))
    if n_to_mix == 0:
        return []

    chosen_idx = set(random.sample(range(len(target_rows)), k=n_to_mix))

    new_rows: List[Dict] = []

    for i, row in tqdm(list(enumerate(target_rows)), desc="Mixing OTHER voices into TARGET speech", unit="clip"):
        if i not in chosen_idx:
            continue

        target_ap = Path(row["audio_path"])
        if not target_ap.exists():
            print(f"[WARN] Missing TARGET speech audio_path, skipping: {target_ap}", file=sys.stderr)
            continue

        try:
            # Load TARGET speech
            target_speech, sr_in = _read_wav_any(target_ap)
            target_speech = _resample_if_needed(target_speech, sr_in, target_sr)
            target_dur = float(len(target_speech)) / float(target_sr)
            
            if target_dur <= 0.01:
                continue

            # Find suitable OTHER voice files (any audio file can work now since we extract segments)
            suitable_other_files = []
            checked_count = 0
            max_checks = min(50, len(other_files_with_scores))  # Limit checks for performance
            
            for other_path, other_score in other_files_with_scores:
                if checked_count >= max_checks:
                    break
                checked_count += 1
                
                if not other_path.exists():
                    continue
                    
                try:
                    other_audio, other_sr = _read_wav_any(other_path)
                    other_dur = float(len(other_audio)) / other_sr
                    # Any audio file with duration > 1 second can work (we'll extract segments)
                    if other_dur > 1.0:
                        suitable_other_files.append((other_path, other_score, other_audio, other_sr))
                        # If we found a few good options, stop searching
                        if len(suitable_other_files) >= 5:
                            break
                except Exception:
                    continue
            
            if not suitable_other_files:
                print(f"[WARN] No suitable OTHER voice files found for TARGET {target_ap}", file=sys.stderr)
                continue
            
            # Pick from suitable files, prioritizing lowest scores
            # Sort by score again to ensure we pick the lowest scoring suitable files
            suitable_other_files.sort(key=lambda x: x[1])
            chosen_other = suitable_other_files[0]
            other_path, other_score, other_audio, other_sr = chosen_other
            
            # Extract a random segment from the other voice that matches target duration
            rng = random.Random(1337)  # Use fixed seed for reproducibility
            other_voice_segment, actual_duration = _extract_random_segment(other_audio, other_sr, target_dur, rng)
            other_voice = _resample_if_needed(other_voice_segment, other_sr, target_sr)

            # Choose SNR and scale OTHER voice
            snr_db = random.uniform(float(snr_db_min), float(snr_db_max))
            target_rms = _rms(target_speech)
            other_rms = _rms(other_voice)

            # amplitude-ratio SNR: snr_db = 20*log10(target_rms / other_voice_scaled)
            # => other_voice_scaled = target_rms / 10^(snr_db/20)
            target_other_rms = target_rms / (10.0 ** (snr_db / 20.0))
            gain = target_other_rms / max(other_rms, 1e-12)
            voice_scaled = other_voice * gain

            mixed = target_speech + voice_scaled

            # Avoid hard clipping by scaling down if needed
            peak = float(np.max(np.abs(mixed)))
            if peak > 0.99:
                mixed = mixed * (0.99 / peak)

            # Build output name
            target_stem = target_ap.stem
            safe_target_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in target_stem)
            other_stem = other_path.stem
            safe_other_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in other_stem)

            out_name = f"{safe_target_stem}__voice_mixed__snr{snr_db:.1f}dB__score{other_score:.3f}__{safe_other_stem}.wav"
            out_path = out_mix_dir / out_name

            if out_path.exists() and not allow_overwrite:
                # still add manifest row referencing existing file
                pass
            else:
                _safe_write_wav(out_path, mixed.astype(np.float32, copy=False), target_sr)

            # Create new row with TARGET transcript and metadata
            new_row = dict(row)
            new_row["audio_path"] = _norm_path(out_path)
            new_row["aug_voice_source"] = str(other_path)
            new_row["aug_voice_score"] = float(other_score)
            new_row["aug_snr_db"] = float(snr_db)
            new_row["is_voice_mixed"] = True
            new_rows.append(new_row)

        except Exception as e:
            print(f"[WARN] Failed voice mixing for row {i} ({row.get('audio_path')}): {e}", file=sys.stderr)

    return new_rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Input JSONL manifest path")
    p.add_argument("--scores_csv", required=True, help="Speaker sort scores CSV with TARGET/OTHER files")
    p.add_argument("--other_voices_dir", default="I:\\Record_others", help="Directory with other voice audio files (default: I:\\Record_others)")
    
    p.add_argument("--out_manifest", required=True, help="Output JSONL manifest path")
    p.add_argument("--out_mix_dir", required=True, help="Where to write voice-mixed copies")
    
    p.add_argument("--target_sr", type=int, default=16000)
    
    p.add_argument("--mix_ratio", type=float, default=0.5, help="Fraction of TARGET rows to create voice-mixed copies for")
    p.add_argument("--snr_db_min", type=float, default=5.0)
    p.add_argument("--snr_db_max", type=float, default=20.0)
    
    p.add_argument("--shuffle", action="store_true", help="Shuffle output manifest rows")
    p.add_argument("--seed", type=int, default=1337)
    
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting generated audio files")
    
    args = p.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    manifest_path = Path(args.manifest)
    scores_csv = Path(args.scores_csv)
    other_voices_dir = Path(args.other_voices_dir)
    out_manifest = Path(args.out_manifest)
    out_mix_dir = Path(args.out_mix_dir)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not scores_csv.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv}")
    
    # Load target files and other voice files from scores CSV or directory
    target_files, other_files_with_scores = load_target_and_other_files(scores_csv, other_voices_dir)
    print(f"Loaded {len(target_files)} target files and {len(other_files_with_scores)} OTHER voice files")
    
    if not other_files_with_scores:
        raise RuntimeError(f"No OTHER voice files found in {other_voices_dir} or {scores_csv}")
    
    base_rows = load_manifest(manifest_path)
    if not base_rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")
    
    # Mix OTHER voices into TARGET speech
    new_rows = mix_other_voices_into_target(
        base_rows=base_rows,
        other_files_with_scores=other_files_with_scores,
        out_mix_dir=out_mix_dir,
        target_sr=int(args.target_sr),
        mix_ratio=float(args.mix_ratio),
        snr_db_min=float(args.snr_db_min),
        snr_db_max=float(args.snr_db_max),
        allow_overwrite=bool(args.overwrite),
        target_files=target_files,
    )
    
    # Original TARGET files are kept in base_rows, new mixed files are added
    out_rows = list(base_rows) + list(new_rows)
    
    if args.shuffle:
        random.shuffle(out_rows)
    
    write_manifest(out_manifest, out_rows)
    
    print("\nDone")
    print(f"  Base rows (original TARGET files):     {len(base_rows)}")
    print(f"  Added rows (voice-mixed files):        {len(new_rows)}")
    print(f"  Total output rows:                     {len(out_rows)}")
    print(f"  Output file:                           {out_manifest}")
    print(f"  Mixed files saved to:                  {out_mix_dir}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
