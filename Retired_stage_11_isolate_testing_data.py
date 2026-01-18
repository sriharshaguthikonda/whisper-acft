"""
Stage 11: Isolate Testing Data

Extract 10% of manifest files for testing by MOVING (not copying) them to a separate testing manifest.
This ensures no data leakage between training and testing sets.

Usage:
python stage_11_isolate_testing_data.py --input_manifest "I:\Record_chunks\pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_silent.jsonl" --test_manifest "I:\Record_chunks\pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_silent_test.jsonl" --train_manifest "I:\Record_chunks\pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_silent_train.jsonl" --test_ratio 0.1 --seed 1337
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import tqdm


def iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read all JSONL lines into a list."""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no}: {path}") from e
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Write rows to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split_train_test(
    rows: List[Dict[str, Any]], 
    test_ratio: float = 0.1,
    seed: int = 1337
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split manifest into train and test sets.
    
    Args:
        rows: List of manifest rows
        test_ratio: Fraction of data to allocate to test set
        seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (train_rows, test_rows)
    """
    # Set random seed for reproducible split
    rng = random.Random(seed)
    
    # Shuffle the rows
    shuffled_rows = rows.copy()
    rng.shuffle(shuffled_rows)
    
    # Calculate split point
    n_test = int(len(shuffled_rows) * test_ratio)
    n_test = max(1, n_test)  # Ensure at least 1 test sample
    
    # Split
    test_rows = shuffled_rows[:n_test]
    train_rows = shuffled_rows[n_test:]
    
    return train_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Split manifest into train/test sets")
    parser.add_argument("--input_manifest", required=True, type=Path, 
                       help="Input JSONL manifest file")
    parser.add_argument("--test_manifest", required=True, type=Path,
                       help="Output JSONL for test data")
    parser.add_argument("--train_manifest", required=True, type=Path,
                       help="Output JSONL for training data (remaining 90%)")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                       help="Fraction of data to move to test set (default: 0.1)")
    parser.add_argument("--seed", type=int, default=1337,
                       help="Random seed for reproducible split")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show split plan without writing files")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {args.input_manifest}")
    
    if not 0 < args.test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    
    # Load manifest
    print(f"Loading manifest from: {args.input_manifest}")
    rows = iter_jsonl(args.input_manifest)
    print(f"Loaded {len(rows):,} rows")
    
    # Split data
    print(f"Splitting with test_ratio={args.test_ratio}, seed={args.seed}")
    train_rows, test_rows = split_train_test(rows, args.test_ratio, args.seed)
    
    # Show split summary
    print(f"\nSplit Summary:")
    print(f"  Total rows:     {len(rows):,}")
    print(f"  Test rows:      {len(test_rows):,} ({len(test_rows)/len(rows)*100:.1f}%)")
    print(f"  Train rows:     {len(train_rows):,} ({len(train_rows)/len(rows)*100:.1f}%)")
    
    if args.dry_run:
        print(f"\nDry run - not writing files.")
        print(f"Would write test manifest to: {args.test_manifest}")
        print(f"Would write train manifest to: {args.train_manifest}")
        return
    
    # Write output files
    print(f"\nWriting test manifest: {args.test_manifest}")
    write_jsonl(args.test_manifest, test_rows)
    
    print(f"Writing train manifest: {args.train_manifest}")
    write_jsonl(args.train_manifest, train_rows)
    
    print(f"\n✅ Successfully split manifest:")
    print(f"  Test:  {args.test_manifest} ({len(test_rows):,} rows)")
    print(f"  Train: {args.train_manifest} ({len(train_rows):,} rows)")
    
    # Show sample of test files for verification
    print(f"\nSample test files:")
    for i, row in enumerate(test_rows[:5]):
        audio_path = row.get("audio_path", "N/A")
        transcription = row.get("raw_transcription", "N/A")
        print(f"  {i+1}. {audio_path}")
        print(f"     Transcription: {transcription[:100]}{'...' if len(transcription) > 100 else ''}")


if __name__ == "__main__":
    main()
