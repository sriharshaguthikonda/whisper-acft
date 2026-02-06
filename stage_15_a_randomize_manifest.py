#!/usr/bin/env python3
"""stage_15_a_randomize_manifest.py

Goal
----
Randomize the order of lines in a Whisper-style JSONL manifest.

Why it helps
------------
Randomizing the manifest order ensures that:
1) The model doesn't learn any ordering patterns from the data
2) Training batches are more diverse in terms of audio characteristics
3) Any systematic biases in the original ordering are eliminated
4) The subsequent train/test split gets a truly random distribution

Key design choices
------------------
1) Use a fixed seed for reproducible shuffling
2) Preserve all JSON fields and structure exactly
3) Support dry-run mode for testing
4) Show progress with tqdm for large manifests
5) Validate input JSON format

Usage
-----
i:\\Whisper-training-env\\Scripts\\python.exe i:\\whisper-acft\\stage_15_a_randomize_manifest.py `
  --input_manifest "I:\\Record_chunks\\pairs_pending_stereo_english_only_filtered_with_others_voice_mix_aug_rir_real_bottom_filtered.jsonl" `
  --output_manifest "I:\\Record_chunks\\pairs_pending_stereo_english_only_filtered_with_others_voice_mix_aug_rir_real_bottom_filtered_randomized.jsonl" `
  --seed 1337

Optional:
  --dry_run (show what would be done without writing)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List
from tqdm import tqdm


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file with progress tracking."""
    rows: List[Dict[str, Any]] = []
    
    # Count lines first for progress bar
    line_count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                line_count += 1
    
    # Read with progress bar
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(tqdm(f, total=line_count, desc="Reading manifest"), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no}: {path}") from e
    
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write JSONL file with progress tracking."""
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to list to get count for progress bar
    rows_list = list(rows)
    
    with path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows_list, desc="Writing manifest"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def shuffle_manifest(rows: List[Dict[str, Any]], seed: int) -> List[Dict[str, Any]]:
    """Shuffle manifest rows using the provided seed."""
    rng = random.Random(seed)
    shuffled = rows.copy()
    rng.shuffle(shuffled)
    return shuffled


def validate_manifest_rows(rows: List[Dict[str, Any]]) -> None:
    """Basic validation of manifest structure."""
    if not rows:
        raise ValueError("Manifest is empty")
    
    # Check first few rows for expected fields
    sample_size = min(5, len(rows))
    for i, row in enumerate(rows[:sample_size]):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i+1} is not a JSON object")
        
        # Check for common manifest fields (optional, just for validation)
        if "audio_path" not in row:
            print(f"Warning: Row {i+1} missing 'audio_path' field")
        
        if "raw_transcription" not in row and "text" not in row:
            print(f"Warning: Row {i+1} missing transcription fields")


def main() -> None:
    p = argparse.ArgumentParser(description="Randomize the order of lines in a Whisper JSONL manifest")
    p.add_argument("--input_manifest", required=True, type=Path, 
                   help="Input JSONL manifest file (output from stage 10)")
    p.add_argument("--output_manifest", required=True, type=Path,
                   help="Output JSONL manifest file (input for stage 11b)")
    p.add_argument("--seed", type=int, default=1337,
                   help="Random seed for reproducible shuffling (default: 1337)")
    p.add_argument("--dry_run", action="store_true",
                   help="Show what would be done without writing files")
    
    args = p.parse_args()
    
    # Validate input file exists
    if not args.input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {args.input_manifest}")
    
    # Read input manifest
    print(f"Reading manifest from: {args.input_manifest}")
    rows = read_jsonl(args.input_manifest)
    print(f"Loaded {len(rows):,} rows")
    
    # Validate manifest structure
    print("Validating manifest structure...")
    validate_manifest_rows(rows)
    
    # Show sample of original order
    print("\nFirst 3 rows (original order):")
    for i, row in enumerate(rows[:3]):
        audio_path = row.get("audio_path", "N/A")
        text = row.get("raw_transcription") or row.get("text", "")
        print(f"  {i+1}. {audio_path}")
        print(f"     {text[:80]}{'...' if len(text) > 80 else ''}")
    
    # Shuffle the manifest
    print(f"\nShuffling manifest with seed {args.seed}...")
    shuffled_rows = shuffle_manifest(rows, args.seed)
    
    # Show sample of shuffled order
    print("\nFirst 3 rows (shuffled order):")
    for i, row in enumerate(shuffled_rows[:3]):
        audio_path = row.get("audio_path", "N/A")
        text = row.get("raw_transcription") or row.get("text", "")
        print(f"  {i+1}. {audio_path}")
        print(f"     {text[:80]}{'...' if len(text) > 80 else ''}")
    
    # Check if order actually changed
    order_changed = any(
        rows[i].get("audio_path") != shuffled_rows[i].get("audio_path") 
        for i in range(min(len(rows), 10))
    )
    
    if not order_changed:
        print("\nWarning: Manifest order appears unchanged (possible small dataset or seed issue)")
    else:
        print("\nManifest order successfully randomized")
    
    if args.dry_run:
        print(f"\nDry run: Would write {len(shuffled_rows):,} rows to {args.output_manifest}")
        return
    
    # Write shuffled manifest
    print(f"\nWriting shuffled manifest to: {args.output_manifest}")
    write_jsonl(args.output_manifest, shuffled_rows)
    
    print(f"\nSuccessfully wrote {len(shuffled_rows):,} shuffled rows")
    print("\nNext step: Use this output as input for stage 11b")
    print(f"Example: python stage_11_b_group_split_train_test.py --input_manifest \"{args.output_manifest}\" ...")


if __name__ == "__main__":
    main()
