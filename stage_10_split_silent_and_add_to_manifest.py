#!/usr/bin/env python3
"""stage_10_split_silent_and_add_to_manifest.py

Goal
----
Split a silent recording into separate channels, chunk them into 0-30 second segments,
and add them to the manifest from stage 9 at random positions.

Why it helps
------------
Adding silent segments helps the model learn to handle silence and non-speech audio,
improving robustness to real-world scenarios where silence is common.

Key design choices
------------------
1) Split multi-channel silent recording into individual channel WAV files
2) Chunk each channel into 0-30 second segments (configurable range)
3) Add silent chunks as noise-only entries to the existing manifest
4) Randomly interleave silent chunks with existing manifest entries

Dependencies
------------
- pydub (for reading M4A files)
- numpy
- soundfile (for saving WAV files)
- tqdm (for progress bars)

Usage
-----
i:\\Whisper-training-env\\Scripts\\python.exe i:\\whisper-acft\\stage_10_split_silent_and_add_to_manifest.py `
  --silent_audio "I:\\Silence_from_phone\\silent recording baseline.m4a" `
  --in_manifest "I:\\Record_chunks\\pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real.jsonl" `
  --out_manifest "I:\\Record_chunks\\pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_silent.jsonl" `
  --out_audio_dir "I:\\Record_chunks\\silent_chunks" `
  --min_chunk_sec 1.0 `
  --max_chunk_sec 30.0 `
  --silent_ratio 0.05 `
  --seed 1337
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Iterable

import numpy as np

try:
    from pydub import AudioSegment
except ImportError as e:
    raise SystemExit("Missing dependency: pydub. Install with: pip install pydub") from e

try:
    import soundfile as sf
except ImportError as e:
    raise SystemExit("Missing dependency: soundfile. Install with: pip install soundfile") from e


# ---------------------------
# Audio processing utilities
# ---------------------------

def load_silent_audio(path: Path) -> Tuple[AudioSegment, int]:
    """Load silent audio file using pydub."""
    audio = AudioSegment.from_file(str(path))
    return audio, audio.frame_rate


def split_channels(audio: AudioSegment) -> List[AudioSegment]:
    """Split multi-channel audio into separate mono channels."""
    channels = []
    for channel_idx in range(audio.channels):
        # Extract single channel
        channel_audio = audio.split_to_mono()[channel_idx]
        channels.append(channel_audio)
    return channels


def audiosegment_to_numpy(audio: AudioSegment) -> np.ndarray:
    """Convert pydub AudioSegment to numpy array."""
    # Get samples as numpy array
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    
    # Normalize to [-1.0, 1.0] range
    if audio.sample_width == 2:  # 16-bit
        samples = samples / 32768.0
    elif audio.sample_width == 4:  # 32-bit
        samples = samples / 2147483648.0
    else:
        # For other bit depths, normalize by max value
        max_val = 2 ** (8 * audio.sample_width - 1)
        samples = samples / max_val
    
    return samples


def save_wav_pcm16(path: Path, audio: np.ndarray, sr: int) -> None:
    """Save numpy array as 16-bit PCM WAV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_16")


# ---------------------------
# Chunking utilities
# ---------------------------

def chunk_audio_silent(
    audio: np.ndarray,
    sr: int,
    min_chunk_sec: float,
    max_chunk_sec: float,
    channel_idx: int,
    out_dir: Path
) -> List[Dict[str, Any]]:
    """Chunk silent audio into segments and save them."""
    chunks = []
    cursor = 0
    chunk_idx = 0
    
    while cursor < len(audio):
        # Random chunk duration within range
        chunk_duration = random.uniform(min_chunk_sec, max_chunk_sec)
        chunk_samples = int(chunk_duration * sr)
        
        # Don't go beyond audio length
        end_cursor = min(cursor + chunk_samples, len(audio))
        actual_chunk_samples = end_cursor - cursor
        
        # Skip if too short
        actual_duration = actual_chunk_samples / sr
        if actual_duration < min_chunk_sec and end_cursor < len(audio):
            cursor = end_cursor
            continue
        
        # Extract chunk
        chunk_audio = audio[cursor:end_cursor]
        
        # Save chunk
        chunk_name = f"silent_channel{channel_idx}_chunk{chunk_idx:05d}.wav"
        chunk_path = out_dir / chunk_name
        save_wav_pcm16(chunk_path, chunk_audio, sr)
        
        # Create chunk metadata
        chunk_info = {
            "audio_path": str(chunk_path),
            "source_audio": f"silent_recording_channel{channel_idx}",
            "channel_index": channel_idx,
            "chunk_index": chunk_idx,
            "chunk_start": cursor / sr,
            "chunk_end": end_cursor / sr,
            "chunk_duration": actual_duration,
            "is_silent": True
        }
        
        chunks.append(chunk_info)
        chunk_idx += 1
        cursor = end_cursor
    
    return chunks


# ---------------------------
# Manifest utilities
# ---------------------------

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Iterate over JSONL manifest file."""
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no}: {path}") from e


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write rows to JSONL manifest file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def create_silent_manifest_rows(silent_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create manifest rows for silent chunks."""
    rows = []
    for chunk in silent_chunks:
        row = {
            "audio_path": chunk["audio_path"],
            "raw_transcription": "<|nospeech|>",  # Empty transcription for silence
            "source_audio": chunk["source_audio"],
            "channel_index": chunk["channel_index"],
            "chunk_index": chunk["chunk_index"],
            "chunk_start": chunk["chunk_start"],
            "chunk_end": chunk["chunk_end"],
            "chunk_duration": chunk["chunk_duration"],
            "transcript_json": None,
            "is_silent": True,
            "is_noise_only": False  # Differentiate from regular noise
        }
        rows.append(row)
    return rows


def interleave_silent_chunks(
    base_rows: List[Dict[str, Any]], 
    silent_rows: List[Dict[str, Any]],
    silent_ratio: float,
    rng: random.Random
) -> List[Dict[str, Any]]:
    """Interleave silent chunks with base manifest rows."""
    if not silent_rows:
        return base_rows
    
    # Calculate how many silent chunks to add
    n_silent_to_add = min(len(silent_rows), int(len(base_rows) * silent_ratio))
    
    if n_silent_to_add == 0:
        return base_rows
    
    # Randomly select silent chunks
    selected_silent = rng.sample(silent_rows, n_silent_to_add)
    rng.shuffle(selected_silent)
    
    # Interleave with base rows
    result = []
    silent_idx = 0
    
    # Calculate spacing for interleaving
    if n_silent_to_add > 0:
        spacing = max(1, len(base_rows) // n_silent_to_add)
    else:
        spacing = 1
    
    for i, base_row in enumerate(base_rows):
        result.append(base_row)
        
        # Add silent chunk at regular intervals with some randomness
        if silent_idx < len(selected_silent):
            if i % spacing == 0 and rng.random() > 0.3:  # 70% chance to add at spacing interval
                result.append(selected_silent[silent_idx])
                silent_idx += 1
            elif i == len(base_rows) - 1:  # Add remaining at the end
                while silent_idx < len(selected_silent):
                    result.append(selected_silent[silent_idx])
                    silent_idx += 1
    
    return result


# ---------------------------
# Main function
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Split silent recording into channels and add chunks to manifest"
    )
    
    ap.add_argument("--silent_audio", required=True, help="Path to silent recording file")
    ap.add_argument("--in_manifest", required=True, help="Input JSONL manifest from stage 9")
    ap.add_argument("--out_manifest", required=True, help="Output JSONL manifest with silent chunks")
    ap.add_argument("--out_audio_dir", required=True, help="Directory to save silent chunk WAV files")
    
    ap.add_argument("--min_chunk_sec", type=float, default=5.0, help="Minimum chunk duration in seconds")
    ap.add_argument("--max_chunk_sec", type=float, default=30.0, help="Maximum chunk duration in seconds")
    ap.add_argument("--target_sr", type=int, default=16000, help="Target sample rate for output chunks")
    
    ap.add_argument("--silent_ratio", type=float, default=0.15, 
                   help="Ratio of silent chunks to add relative to base manifest size")
    
    ap.add_argument("--seed", type=int, default=1337, help="Random seed")
    
    args = ap.parse_args()
    
    # Set random seeds
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    
    # Validate paths
    silent_audio_path = Path(args.silent_audio)
    in_manifest_path = Path(args.in_manifest)
    out_manifest_path = Path(args.out_manifest)
    out_audio_dir = Path(args.out_audio_dir)
    
    if not silent_audio_path.exists():
        raise FileNotFoundError(f"Silent audio file not found: {silent_audio_path}")
    if not in_manifest_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {in_manifest_path}")
    
    print(f"Loading silent audio from: {silent_audio_path}")
    
    # Load and analyze silent audio
    audio, sr = load_silent_audio(silent_audio_path)
    print(f"Audio info: {audio.channels} channels, {sr} Hz, {len(audio)/1000:.2f}s duration")
    
    # Split into channels
    channels = split_channels(audio)
    print(f"Split into {len(channels)} channels")
    
    # Process each channel
    all_silent_chunks = []
    
    for channel_idx, channel_audio in enumerate(channels):
        print(f"Processing channel {channel_idx + 1}/{len(channels)}...")
        
        # Convert to numpy
        channel_samples = audiosegment_to_numpy(channel_audio)
        
        # Resample if needed
        if sr != args.target_sr:
            try:
                from scipy.signal import resample_poly
                from math import gcd
                
                g = gcd(sr, args.target_sr)
                up = args.target_sr // g
                down = sr // g
                channel_samples = resample_poly(channel_samples, up, down).astype(np.float32)
                print(f"  Resampled from {sr} Hz to {args.target_sr} Hz")
            except ImportError:
                print(f"  Warning: scipy not available, keeping original sample rate {sr} Hz")
        
        # Create channel-specific output directory
        channel_dir = out_audio_dir / f"channel_{channel_idx}"
        
        # Chunk the channel audio
        chunks = chunk_audio_silent(
            channel_samples,
            args.target_sr,  # Always use target sample rate
            args.min_chunk_sec,
            args.max_chunk_sec,
            channel_idx,
            channel_dir
        )
        
        all_silent_chunks.extend(chunks)
        print(f"  Created {len(chunks)} chunks from channel {channel_idx}")
    
    print(f"Total silent chunks created: {len(all_silent_chunks)}")
    
    # Load base manifest
    print(f"Loading base manifest from: {in_manifest_path}")
    base_rows = list(iter_jsonl(in_manifest_path))
    print(f"Base manifest has {len(base_rows)} rows")
    
    # Create manifest rows for silent chunks
    silent_rows = create_silent_manifest_rows(all_silent_chunks)
    print(f"Created {len(silent_rows)} silent manifest rows")
    
    # Interleave silent chunks with base manifest
    final_rows = interleave_silent_chunks(
        base_rows, 
        silent_rows, 
        args.silent_ratio,
        rng
    )
    
    # Write output manifest
    print(f"Writing output manifest to: {out_manifest_path}")
    write_jsonl(out_manifest_path, final_rows)
    
    print("\nDone!")
    print(f"  Base rows:           {len(base_rows)}")
    print(f"  Silent chunks:       {len(silent_rows)}")
    print(f"  Silent chunks added: {len([r for r in final_rows if r.get('is_silent')])}")
    print(f"  Total output rows:   {len(final_rows)}")
    print(f"  Output manifest:     {out_manifest_path}")
    print(f"  Silent chunks dir:   {out_audio_dir}")


if __name__ == "__main__":
    main()
