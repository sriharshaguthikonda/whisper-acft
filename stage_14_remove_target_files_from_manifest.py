#!/usr/bin/env python3
"""stage_14_remove_target_files_from_manifest.py

Goal
----
Remove manifest rows for files that are marked as "TARGET" in the speaker_sort_scores.csv.

Why it helps
------------
1) Removes high-quality target speaker files from training manifest to prevent data leakage
2) Ensures target speaker files are reserved for testing/evaluation only
3) Prevents model from overfitting to target speaker during training
4) Maintains clean separation between training and testing data

Key design choices
------------------
1) Use the decision column from speaker_sort_scores.csv to identify TARGET files
2) Case-insensitive file path matching between manifest and CSV
3) Preserve manifest order for remaining entries
4) Show detailed statistics about what's being removed
5) Support dry-run mode for testing
6) Handle missing or malformed entries gracefully

Usage
-----
i:\\Whisper-training-env\\Scripts\\python.exe i:\\whisper-acft\\stage_14_remove_target_files_from_manifest.py `
  --input_manifest "I:\\Record_chunks\\pairs_manifest_sorted_by_scores_english_only_filtered_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_randomized_filtered_train.jsonl" `
  --output_manifest "I:\\Record_chunks\\pairs_manifest_sorted_by_scores_english_only_filtered_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_randomized_filtered_train_no_targets.jsonl" `
  --speaker_scores_csv "i:\\whisper-acft\\speaker_sort_scores.csv"

Optional:
  --dry_run (show what would be done without writing)
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
import re
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
from tqdm import tqdm
import winsound  # For beep notification


def canonical_key(p: str) -> str:
    if not p:
        return ""
    p = str(p).strip().strip('"').strip("'")
    p = p.replace("\\", "/")
    p = re.sub(r"/+", "/", p)
    return p.casefold()


def canonical_rel_key(p: str) -> str:
    if not p:
        return ""
    p = canonical_key(p)
    for marker in ("/record_chunks/", "/record_harsha/"):
        idx = p.find(marker)
        if idx != -1:
            return p[idx:]
    return p


def load_target_files(csv_path: Path) -> Set[str]:
    """Load target files from CSV based on decision column."""
    print(f"Loading target files from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError as e:
        print(f"CSV parsing error: {e}")
        print("Attempting to read with more robust settings...")
        df = pd.read_csv(csv_path, on_bad_lines='skip', quoting=3)
    
    # Filter for TARGET decisions and normalize file paths to lowercase
    target_files = set()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing scores"):
        file_path = row['file']
        decision = row['decision']
        
        # Skip rows where file_path is NaN or not a string
        if pd.isna(file_path) or not isinstance(file_path, str):
            continue
            
        # Add to target set if decision is TARGET
        if pd.notna(decision) and str(decision).upper() == 'TARGET':
            ck = canonical_key(file_path)
            target_files.add(ck)
            rk = canonical_rel_key(file_path)
            if rk:
                target_files.add(rk)
    
    print(f"Found {len(target_files)} target files")
    return target_files


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file with progress tracking."""
    rows: List[Dict[str, Any]] = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Reading manifest"), 1):
            try:
                row = json.loads(line.strip())
                rows.append(row)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return rows


def filter_manifest_by_target_files(
    manifest_rows: List[Dict[str, Any]], 
    target_files: Set[str]
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter manifest entries to remove target files."""
    
    # Filter entries
    kept_entries = []
    removed_entries = []
    
    stats = {
        'total': len(manifest_rows),
        'kept': 0,
        'removed': 0,
        'removed_target': 0,
        'kept_non_target': 0,
        'unknown_files': 0
    }
    
    for entry in tqdm(manifest_rows, desc="Filtering entries"):
        audio_path = entry.get('audio_path', '')
        source_audio = entry.get('source_audio', '')
        keys = []
        if audio_path:
            keys.extend([canonical_key(audio_path), canonical_rel_key(audio_path)])
        if source_audio:
            keys.extend([canonical_key(source_audio), canonical_rel_key(source_audio)])

        if any(k in target_files for k in keys if k):
            # This is a target file - remove it
            removed_entries.append(entry)
            stats['removed'] += 1
            stats['removed_target'] += 1
        else:
            # This is not a target file - keep it
            kept_entries.append(entry)
            stats['kept'] += 1
            stats['kept_non_target'] += 1
            
            # Check if this file was in the CSV at all
            if audio_path and not any(audio_path == target_file for target_file in normalized_target_files):
                # We don't have info about this file from the CSV
                pass  # This is normal, many files won't be in the scores CSV
    
    return kept_entries, stats


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write rows to JSONL file with progress tracking."""
    with open(path, 'w', encoding='utf-8') as f:
        for row in tqdm(rows, desc="Writing manifest"):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print filtering statistics."""
    print("=" * 60)
    print("FILTERING STATISTICS")
    print("=" * 60)
    print(f"Total entries processed: {stats['total']:,}")
    print(f"Entries kept: {stats['kept']:,} ({stats['kept']/stats['total']*100:.1f}%)")
    print(f"Entries removed: {stats['removed']:,} ({stats['removed']/stats['total']*100:.1f}%)")
    
    if stats['removed'] > 0:
        print("\nRemoval breakdown:")
        print(f"  - Target files removed: {stats['removed_target']:,}")
    
    print("\nKeep breakdown:")
    print(f"  - Non-target files kept: {stats['kept_non_target']:,}")
    print("=" * 60)


def beep_notification():
    """Play beep notification when script completes."""
    try:
        # Play a beep sound (frequency: 1000Hz, duration: 500ms)
        winsound.Beep(1000, 500)
    except Exception:
        # Fallback if winsound is not available
        print("\a")  # Terminal bell


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove target files from manifest based on speaker scores CSV"
    )
    parser.add_argument(
        "--input_manifest",
        type=Path,
        required=True,
        help="Path to input JSONL manifest file"
    )
    parser.add_argument(
        "--output_manifest",
        type=Path,
        required=True,
        help="Path to output filtered JSONL manifest file"
    )
    parser.add_argument(
        "--speaker_scores_csv",
        type=Path,
        default=Path("i:\\whisper-acft\\speaker_sort_scores.csv"),
        help="Path to speaker scores CSV file"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without writing output file"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {args.input_manifest}")
    
    if not args.speaker_scores_csv.exists():
        raise FileNotFoundError(f"Speaker scores CSV not found: {args.speaker_scores_csv}")
    
    print("=" * 60)
    print("STAGE 14: Remove Target Files from Manifest")
    print("=" * 60)
    print(f"Input manifest: {args.input_manifest}")
    print(f"Output manifest: {args.output_manifest}")
    print(f"Speaker scores CSV: {args.speaker_scores_csv}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)
    
    # Load data
    print("Loading target files from speaker scores...")
    target_files = load_target_files(args.speaker_scores_csv)
    
    print("Loading manifest...")
    manifest_rows = read_jsonl(args.input_manifest)
    print(f"Loaded {len(manifest_rows):,} manifest entries")
    
    # Filter manifest
    print("Filtering manifest to remove target files...")
    kept_entries, stats = filter_manifest_by_target_files(manifest_rows, target_files)
    
    # Print statistics
    print_statistics(stats)
    
    # Write output if not dry run
    if not args.dry_run:
        print(f"\nWriting filtered manifest to {args.output_manifest}...")
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(kept_entries, args.output_manifest)
        print(f"Successfully wrote {len(kept_entries):,} entries to output file")
    else:
        print(f"\nDRY RUN: Would write {len(kept_entries):,} entries to {args.output_manifest}")
    
    print("=" * 60)
    print("STAGE 14 COMPLETED")
    print("=" * 60)
    
    # Play completion beep
    beep_notification()


if __name__ == "__main__":
    main()
