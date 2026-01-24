#!/usr/bin/env python3
"""stage_12_remove_bottom_percent_by_speaker_scores.py

Goal
----
Remove the bottom X percent of manifest entries based on speaker scores from the CSV.

Why it helps
------------
1) Removes low-quality audio chunks that scored poorly on speaker analysis
2) Improves training data quality by filtering out problematic segments
3) Reduces noise in training data from segments that don't meet quality thresholds
4) Focuses training on higher-quality speaker samples

Key design choices
------------------
1) Use configurable percentage threshold (default 10%)
2) Handle NaN scores and missing scores appropriately
3) Preserve manifest order for remaining entries
4) Show detailed statistics about what's being removed
5) Support dry-run mode for testing
6) Use case-insensitive file path matching

Usage
-----
i:\\Whisper-training-env\\Scripts\\python.exe i:\\whisper-acft\\stage_12_remove_bottom_percent_by_speaker_scores.py `
  --input_manifest "I:\\Record_chunks\\pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_silent_randomized.jsonl" `
  --output_manifest "I:\\Record_chunks\\pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_silent_randomized_filtered.jsonl" `
  --speaker_scores_csv "i:\\whisper-acft\\speaker_sort_scores.csv" `
  --bottom_percent 30

Optional:
  --dry_run (show what would be done without writing)
  --min_score (minimum score threshold, overrides bottom_percent)
"""

from __future__ import annotations

import argparse
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple
from tqdm import tqdm


def load_speaker_scores(csv_path: Path) -> Dict[str, float]:
    """Load speaker scores from CSV file into a dictionary."""
    print(f"Loading speaker scores from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Create a dictionary mapping file paths to scores
    score_dict = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing scores"):
        file_path = row['file']
        score = row['score']
        
        # Skip rows where file_path is NaN or not a string
        if pd.isna(file_path) or not isinstance(file_path, str):
            continue
            
        # Store score as float, keep NaN as special value
        score_dict[file_path] = score
    
    print(f"Loaded {len(score_dict)} file scores")
    return score_dict


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


def calculate_score_threshold(scores: List[float], bottom_percent: float) -> float:
    """Calculate the score threshold for removing bottom percentage."""
    if not scores:
        return float('-inf')
    
    # Sort scores to find threshold
    sorted_scores = sorted(scores)
    remove_count = int(len(sorted_scores) * (bottom_percent / 100.0))
    
    if remove_count == 0:
        return sorted_scores[0]
    
    threshold = sorted_scores[remove_count - 1]
    return threshold


def filter_manifest_by_scores(
    manifest_rows: List[Dict[str, Any]], 
    score_dict: Dict[str, float],
    bottom_percent: float,
    min_score: float = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Filter manifest entries by speaker scores."""
    
    # Normalize score dictionary keys to lowercase for case-insensitive matching
    normalized_score_dict = {k.lower(): v for k, v in score_dict.items()}
    
    # Collect scores for threshold calculation
    valid_scores = []
    for score in normalized_score_dict.values():
        if not pd.isna(score):
            valid_scores.append(score)
    
    # Determine threshold
    if min_score is not None:
        threshold = min_score
        print(f"Using minimum score threshold: {threshold}")
    else:
        threshold = calculate_score_threshold(valid_scores, bottom_percent)
        print(f"Calculated threshold for bottom {bottom_percent}%: {threshold}")
    
    # Filter entries
    kept_entries = []
    removed_entries = []
    
    stats = {
        'total': len(manifest_rows),
        'kept': 0,
        'removed': 0,
        'removed_by_threshold': 0,
        'removed_no_score': 0,
        'removed_nan_score': 0,
        'kept_with_score': 0,
        'kept_no_score': 0
    }
    
    for entry in tqdm(manifest_rows, desc="Filtering entries"):
        audio_path = entry.get('audio_path', '').lower()
        
        if audio_path in normalized_score_dict:
            score = normalized_score_dict[audio_path]
            
            if pd.isna(score):
                # NaN scores - remove these
                removed_entries.append(entry)
                stats['removed'] += 1
                stats['removed_nan_score'] += 1
            elif min_score is not None:
                # Use minimum score threshold
                if score >= min_score:
                    kept_entries.append(entry)
                    stats['kept'] += 1
                    stats['kept_with_score'] += 1
                else:
                    removed_entries.append(entry)
                    stats['removed'] += 1
                    stats['removed_by_threshold'] += 1
            else:
                # Use bottom percentage threshold
                if score > threshold:
                    kept_entries.append(entry)
                    stats['kept'] += 1
                    stats['kept_with_score'] += 1
                else:
                    removed_entries.append(entry)
                    stats['removed'] += 1
                    stats['removed_by_threshold'] += 1
        else:
            # No score found - keep these (they weren't scored)
            kept_entries.append(entry)
            stats['kept'] += 1
            stats['kept_no_score'] += 1
    
    return kept_entries, stats


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write rows to JSONL file with progress tracking."""
    with open(path, 'w', encoding='utf-8') as f:
        for row in tqdm(rows, desc="Writing manifest"):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def print_statistics(stats: Dict[str, Any], bottom_percent: float) -> None:
    """Print filtering statistics."""
    print("=" * 60)
    print("FILTERING STATISTICS")
    print("=" * 60)
    print(f"Total entries processed: {stats['total']:,}")
    print(f"Entries kept: {stats['kept']:,} ({stats['kept']/stats['total']*100:.1f}%)")
    print(f"Entries removed: {stats['removed']:,} ({stats['removed']/stats['total']*100:.1f}%)")
    
    if stats['removed'] > 0:
        print("\nRemoval breakdown:")
        print(f"  - Below threshold: {stats['removed_by_threshold']:,}")
        print(f"  - NaN scores: {stats['removed_nan_score']:,}")
        print(f"  - No scores found: {stats['removed_no_score']:,}")
    
    print("\nKeep breakdown:")
    print(f"  - With valid scores: {stats['kept_with_score']:,}")
    print(f"  - No scores found: {stats['kept_no_score']:,}")
    print(f"Target bottom {bottom_percent}% removal: {stats['total'] * (bottom_percent/100):.0f} entries")
    print(f"Actual removal: {stats['removed']:,} entries")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove bottom percentage of manifest entries by speaker scores"
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
        "--bottom_percent",
        type=float,
        default=10.0,
        help="Percentage of bottom-scoring entries to remove (default: 10.0)"
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=None,
        help="Minimum score threshold (overrides bottom_percent if specified)"
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
    
    if args.bottom_percent < 0 or args.bottom_percent > 100:
        raise ValueError("bottom_percent must be between 0 and 100")
    
    print("=" * 60)
    print("STAGE 12: Remove Bottom Percentage by Speaker Scores")
    print("=" * 60)
    print(f"Input manifest: {args.input_manifest}")
    print(f"Output manifest: {args.output_manifest}")
    print(f"Speaker scores CSV: {args.speaker_scores_csv}")
    print(f"Bottom percentage to remove: {args.bottom_percent}%")
    if args.min_score is not None:
        print(f"Minimum score threshold: {args.min_score}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)
    
    # Load data
    print("Loading speaker scores...")
    score_dict = load_speaker_scores(args.speaker_scores_csv)
    
    print("Loading manifest...")
    manifest_rows = read_jsonl(args.input_manifest)
    print(f"Loaded {len(manifest_rows):,} manifest entries")
    
    # Filter manifest
    print("Filtering manifest by speaker scores...")
    kept_entries, stats = filter_manifest_by_scores(
        manifest_rows, score_dict, args.bottom_percent, args.min_score
    )
    
    # Print statistics
    print_statistics(stats, args.bottom_percent)
    
    # Write output if not dry run
    if not args.dry_run:
        print(f"\nWriting filtered manifest to {args.output_manifest}...")
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(kept_entries, args.output_manifest)
        print(f"Successfully wrote {len(kept_entries):,} entries to output file")
    else:
        print(f"\nDRY RUN: Would write {len(kept_entries):,} entries to {args.output_manifest}")
    
    print("=" * 60)
    print("STAGE 12 COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
