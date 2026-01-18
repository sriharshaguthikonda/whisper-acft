#!/usr/bin/env python3
"""augment_manifest_with_noise.py

Goal
----
Given a Whisper-style JSONL manifest (one JSON object per line), augment it using a folder
of *non-speech* noise audio:

1) Chunk ALL noise files into <= 30s WAV segments (16 kHz by default)
2) Add a random subset of those noise-only chunks as extra manifest rows (true negatives)
3) (Optional) Also create noisy MIXED versions of existing speech clips at random SNRs

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

Noise-only rows will use:
  raw_transcription: ""  (empty)
  transcript_json: null
  is_noise_only: true

Dependencies
------------
pip install numpy soundfile tqdm
(Optional for resampling) pip install scipy

Usage examples
--------------
# 1) Add noise-only rows (~20% of dataset size) + shuffle output

cd "i:\whisper-acft" && python stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise.py \
  --manifest "i:/Record_chunks/pairs_manifest_sorted_by_scores_english_only_filtered_with_noises.jsonl" \
  --noises_dir "i:/noise/RIRS_NOISES/pointsource_noises" \
  --scores_csv "i:/whisper-acft/speaker_sort_scores.csv" \
  --out_manifest "i:/Record_chunks/pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix.jsonl" \
  --out_noise_dir "i:/Record_chunks/noise_chunks" \
  --out_mix_dir "i:/Record_chunks/noisy_mixed" \
  --mode mix \
  --mix_ratio 0.5 \
  --snr_db_min 5 \
  --snr_db_max 20 \
  --seed 1337 \
  --shuffle


  python stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise.py --manifest "i:\Record_chunks\pairs_manifest_sorted_by_scores_english_only_filtered_with_noises.jsonl" --noises_dir "i:\noise\RIRS_NOISES\pointsource_noises" --scores_csv "i:\whisper-acft\speaker_sort_scores.csv" --out_manifest "i:\Record_chunks\pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix.jsonl" --out_noise_dir "i:\Record_chunks\noise_chunks" --out_mix_dir "i:\Record_chunks\noisy_mixed" --mode mix --mix_ratio 0.5 --snr_db_min 5 --snr_db_max 20 --seed 1337 --shuffle
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    # Keep this conservative.
    return p.suffix.lower() in {".wav", ".flac", ".ogg"}


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


@dataclass
class NoiseChunk:
    audio_path: str
    source_audio: str
    chunk_index: int
    chunk_start: float
    chunk_end: float


def chunk_noises(
    noises_dir: Path,
    out_noise_dir: Path,
    target_sr: int,
    max_chunk_sec: float,
    min_chunk_sec: float,
    random_chunking: bool,
    random_min_sec: float,
    random_max_sec: float,
    reuse_index: bool,
) -> List[NoiseChunk]:
    """Chunk every audio file in noises_dir into <= max_chunk_sec pieces.

    Writes WAV chunks to out_noise_dir and returns an index of chunks.
    """

    index_path = out_noise_dir / "noise_chunks_index.jsonl"
    if reuse_index and index_path.exists():
        chunks: List[NoiseChunk] = []
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunks.append(
                    NoiseChunk(
                        audio_path=obj["audio_path"],
                        source_audio=obj["source_audio"],
                        chunk_index=int(obj["chunk_index"]),
                        chunk_start=float(obj["chunk_start"]),
                        chunk_end=float(obj["chunk_end"]),
                    )
                )
        return chunks

    # Discover noise audio files
    noise_files = [p for p in noises_dir.rglob("*") if p.is_file() and _is_audio_file(p)]
    if not noise_files:
        raise FileNotFoundError(f"No audio files (.wav/.flac/.ogg) found under: {noises_dir}")

    out_noise_dir.mkdir(parents=True, exist_ok=True)

    chunks_out: List[NoiseChunk] = []
    with index_path.open("w", encoding="utf-8") as index_f:
        for nf in tqdm(noise_files, desc="Chunking noise files", unit="file"):
            try:
                with sf.SoundFile(str(nf), "r") as snd:
                    sr_in = int(snd.samplerate)
                    n_frames = int(len(snd))

                    # Walk through file with chunks in frames
                    cursor = 0
                    chunk_i = 0
                    while cursor < n_frames:
                        if random_chunking:
                            dur = random.uniform(random_min_sec, random_max_sec)
                            dur = min(dur, max_chunk_sec)
                        else:
                            dur = max_chunk_sec

                        frames = int(round(dur * sr_in))
                        if frames <= 0:
                            break

                        end = min(cursor + frames, n_frames)
                        frames_to_read = end - cursor

                        # Discard too-short tail chunks (unless it is the only chunk)
                        if frames_to_read / sr_in < min_chunk_sec and (end != n_frames):
                            cursor = end
                            continue

                        snd.seek(cursor)
                        audio = snd.read(frames_to_read, dtype="float32", always_2d=False)
                        audio = _ensure_mono(audio)
                        audio = _resample_if_needed(audio, sr_in, target_sr)

                        # Final duration in seconds after resample
                        dur_out = float(len(audio)) / float(target_sr)
                        if dur_out < min_chunk_sec:
                            cursor = end
                            continue

                        # Build output name
                        stem = nf.stem
                        # Make filename safe-ish
                        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
                        out_name = f"noise__{safe_stem}__chunk{chunk_i:05d}.wav"
                        out_path = out_noise_dir / out_name

                        _safe_write_wav(out_path, audio, target_sr)

                        start_sec = float(cursor) / float(sr_in)
                        end_sec = float(end) / float(sr_in)

                        nc = NoiseChunk(
                            audio_path=_norm_path(out_path),
                            source_audio=_norm_path(nf),
                            chunk_index=chunk_i,
                            chunk_start=start_sec,
                            chunk_end=end_sec,
                        )
                        chunks_out.append(nc)

                        index_f.write(
                            json.dumps(
                                {
                                    "audio_path": nc.audio_path,
                                    "source_audio": nc.source_audio,
                                    "chunk_index": nc.chunk_index,
                                    "chunk_start": nc.chunk_start,
                                    "chunk_end": nc.chunk_end,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                        chunk_i += 1
                        cursor = end

            except Exception as e:
                print(f"[WARN] Skipping noise file (read/chunk failed): {nf} :: {e}", file=sys.stderr)

    if not chunks_out:
        raise RuntimeError("Chunking produced zero usable noise chunks. Check your noises_dir contents.")

    return chunks_out


def load_target_files(scores_csv: Path) -> set[str]:
    """Load set of target audio files from speaker sort scores CSV."""
    target_files = set()
    with scores_csv.open("r", encoding="utf-8") as f:
        next(f)  # Skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3 and parts[2] == "TARGET":
                # Normalize path: lowercase and forward slashes
                file_path = parts[0].lower().replace("\\", "/")
                target_files.add(file_path)
    return target_files


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


def add_noise_only_rows(
    base_rows: List[Dict],
    noise_chunks: List[NoiseChunk],
    noise_ratio: float,
    n_noise: Optional[int],
) -> List[Dict]:
    if n_noise is None:
        if noise_ratio <= 0:
            return []
        n_noise = int(round(len(base_rows) * noise_ratio))

    n_noise = max(0, int(n_noise))
    if n_noise == 0:
        return []

    if n_noise > len(noise_chunks):
        n_noise = len(noise_chunks)

    picked = random.sample(noise_chunks, k=n_noise)
    out: List[Dict] = []
    for nc in picked:
        out.append(
            {
                "audio_path": nc.audio_path,
                "raw_transcription": "",
                "source_audio": nc.source_audio,
                "chunk_index": nc.chunk_index,
                "chunk_start": nc.chunk_start,
                "chunk_end": nc.chunk_end,
                "transcript_json": None,
                "is_noise_only": True,
            }
        )
    return out


def _read_wav_any(path: Path) -> Tuple[np.ndarray, int]:
    # Use soundfile; assumes wav/flac/ogg supported
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = _ensure_mono(audio)
    return audio, int(sr)


def mix_noise_into_speech(
    base_rows: List[Dict],
    noise_chunks: List[NoiseChunk],
    out_mix_dir: Path,
    target_sr: int,
    mix_ratio: float,
    snr_db_min: float,
    snr_db_max: float,
    allow_overwrite: bool,
    target_files: set[str],
) -> List[Dict]:
    """Create noisy mixed copies for a subset of TARGET base_rows and return new rows."""

    out_mix_dir.mkdir(parents=True, exist_ok=True)

    # Filter to only target files first
    target_rows = [row for row in base_rows if row["audio_path"].lower().replace("\\", "/") in target_files]
    if not target_rows:
        print("[WARN] No target files found in manifest for noise mixing", file=sys.stderr)
        return []

    print(f"Found {len(target_rows)} target files out of {len(base_rows)} total rows")

    n_to_mix = int(round(len(target_rows) * float(mix_ratio)))
    n_to_mix = max(0, min(len(target_rows), n_to_mix))
    if n_to_mix == 0:
        return []

    chosen_idx = set(random.sample(range(len(target_rows)), k=n_to_mix))

    new_rows: List[Dict] = []

    for i, row in tqdm(list(enumerate(target_rows)), desc="Mixing noise into TARGET speech", unit="clip"):
        if i not in chosen_idx:
            continue

        ap = Path(row["audio_path"])
        if not ap.exists():
            print(f"[WARN] Missing speech audio_path, skipping: {ap}", file=sys.stderr)
            continue

        try:
            speech, sr_in = _read_wav_any(ap)
            speech = _resample_if_needed(speech, sr_in, target_sr)

            speech_dur = float(len(speech)) / float(target_sr)
            if speech_dur <= 0.01:
                continue

            # Pick a noise chunk at least as long as speech; try a few times.
            noise_pick: Optional[NoiseChunk] = None
            for _ in range(30):
                cand = random.choice(noise_chunks)
                cand_len = cand.chunk_end - cand.chunk_start
                if cand_len >= speech_dur:
                    noise_pick = cand
                    break
            if noise_pick is None:
                # fallback: just pick any and tile
                noise_pick = random.choice(noise_chunks)

            noise_audio_path = Path(noise_pick.audio_path)
            noise, nsr = _read_wav_any(noise_audio_path)
            noise = _resample_if_needed(noise, nsr, target_sr)

            # Ensure noise length matches speech length
            if len(noise) < len(speech):
                reps = int(math.ceil(len(speech) / max(1, len(noise))))
                noise = np.tile(noise, reps)[: len(speech)]
            else:
                # Random crop
                max_start = len(noise) - len(speech)
                if max_start > 0:
                    start = random.randint(0, max_start)
                    noise = noise[start : start + len(speech)]
                else:
                    noise = noise[: len(speech)]

            # Choose SNR and scale noise
            snr_db = random.uniform(float(snr_db_min), float(snr_db_max))
            rs = _rms(speech)
            rn = _rms(noise)

            # amplitude-ratio SNR: snr_db = 20*log10(rs / rnoise_scaled)
            # => rnoise_scaled = rs / 10^(snr_db/20)
            target_rn = rs / (10.0 ** (snr_db / 20.0))
            gain = target_rn / max(rn, 1e-12)
            noise_scaled = noise * gain

            mixed = speech + noise_scaled

            # Avoid hard clipping by scaling down if needed
            peak = float(np.max(np.abs(mixed)))
            if peak > 0.99:
                mixed = mixed * (0.99 / peak)

            # Build output name
            base_stem = ap.stem
            safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_stem)
            noise_stem = Path(noise_pick.source_audio).stem
            noise_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in noise_stem)

            out_name = f"{safe_stem}__noisy__snr{snr_db:.1f}dB__{noise_stem}.wav"
            out_path = out_mix_dir / out_name

            if out_path.exists() and not allow_overwrite:
                # still add manifest row referencing existing file
                pass
            else:
                _safe_write_wav(out_path, mixed.astype(np.float32, copy=False), target_sr)

            new_row = dict(row)
            new_row["audio_path"] = _norm_path(out_path)
            new_row["aug_noise_source"] = noise_pick.source_audio
            new_row["aug_noise_chunk_audio"] = noise_pick.audio_path
            new_row["aug_snr_db"] = float(snr_db)
            new_row["is_noisy_mixed"] = True
            new_rows.append(new_row)

        except Exception as e:
            print(f"[WARN] Failed mixing for row {i} ({row.get('audio_path')}): {e}", file=sys.stderr)

    return new_rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Input JSONL manifest path")
    p.add_argument("--noises_dir", required=True, help="Folder containing noise audio (wav/flac/ogg)")
    p.add_argument("--scores_csv", required=True, help="Speaker sort scores CSV with TARGET files")

    p.add_argument("--out_manifest", required=True, help="Output JSONL manifest path")
    p.add_argument("--out_noise_dir", required=True, help="Where to write <=30s noise chunks")
    p.add_argument("--out_mix_dir", default=None, help="Where to write noisy mixed copies (mode=mix)")

    p.add_argument("--mode", choices=["noise_only", "mix", "both"], default="noise_only")

    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--max_chunk_sec", type=float, default=29.5)
    p.add_argument("--min_chunk_sec", type=float, default=1.0)

    p.add_argument("--random_chunking", action="store_true", help="Random chunk lengths instead of fixed chunks")
    p.add_argument("--random_min_sec", type=float, default=2.0)
    p.add_argument("--random_max_sec", type=float, default=29.5)

    p.add_argument("--noise_ratio", type=float, default=0.15, help="How many noise-only rows to add, as a fraction of manifest")
    p.add_argument("--n_noise", type=int, default=None, help="Exact number of noise-only rows to add (overrides noise_ratio)")

    p.add_argument("--mix_ratio", type=float, default=0.0, help="Fraction of TARGET rows to create noisy mixed copies for")
    p.add_argument("--snr_db_min", type=float, default=5.0)
    p.add_argument("--snr_db_max", type=float, default=20.0)

    p.add_argument("--shuffle", action="store_true", help="Shuffle output manifest rows")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--reuse_noise_index", action="store_true", help="Reuse noise chunk index if it exists")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting generated audio files")

    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    manifest_path = Path(args.manifest)
    noises_dir = Path(args.noises_dir)
    scores_csv = Path(args.scores_csv)
    out_manifest = Path(args.out_manifest)
    out_noise_dir = Path(args.out_noise_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not noises_dir.exists():
        raise FileNotFoundError(f"Noises directory not found: {noises_dir}")
    if not scores_csv.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv}")

    # Load target files from scores CSV
    target_files = load_target_files(scores_csv)
    print(f"Loaded {len(target_files)} target files from {scores_csv}")

    base_rows = load_manifest(manifest_path)
    if not base_rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    # 1) Chunk noises
    noise_chunks = chunk_noises(
        noises_dir=noises_dir,
        out_noise_dir=out_noise_dir,
        target_sr=int(args.target_sr),
        max_chunk_sec=float(args.max_chunk_sec),
        min_chunk_sec=float(args.min_chunk_sec),
        random_chunking=bool(args.random_chunking),
        random_min_sec=float(args.random_min_sec),
        random_max_sec=float(args.random_max_sec),
        reuse_index=bool(args.reuse_noise_index),
    )

    new_rows: List[Dict] = []

    # 2) Add noise-only rows
    if args.mode in {"noise_only", "both"}:
        new_rows.extend(
            add_noise_only_rows(
                base_rows=base_rows,
                noise_chunks=noise_chunks,
                noise_ratio=float(args.noise_ratio),
                n_noise=args.n_noise,
            )
        )

    # 3) Mix noise into speech (optional)
    if args.mode in {"mix", "both"}:
        if not args.out_mix_dir:
            raise ValueError("--out_mix_dir is required when mode is mix or both")
        out_mix_dir = Path(args.out_mix_dir)
        new_rows.extend(
            mix_noise_into_speech(
                base_rows=base_rows,
                noise_chunks=noise_chunks,
                out_mix_dir=out_mix_dir,
                target_sr=int(args.target_sr),
                mix_ratio=float(args.mix_ratio),
                snr_db_min=float(args.snr_db_min),
                snr_db_max=float(args.snr_db_max),
                allow_overwrite=bool(args.overwrite),
                target_files=target_files,
            )
        )

    out_rows = list(base_rows) + list(new_rows)

    if args.shuffle:
        random.shuffle(out_rows)

    write_manifest(out_manifest, out_rows)

    print("\nDone")
    print(f"  Base rows:     {len(base_rows)}")
    print(f"  Added rows:    {len(new_rows)}")
    print(f"  Output rows:   {len(out_rows)}")
    print(f"  Output file:   {out_manifest}")
    print(f"  Noise chunks:  {len(noise_chunks)} (index at {out_noise_dir / 'noise_chunks_index.jsonl'})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
