#!/usr/bin/env python3
"""
Randomly sample and move entries from noise-mixed manifest to combined test manifest.
"""

import argparse
import json
import random
import sys
from pathlib import Path


def count_lines(file_path):
    """Count lines in a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def read_jsonl(file_path):
    """Read all entries from a JSONL file."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def write_jsonl(file_path, entries):
    """Write entries to a JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Randomly sample and move entries from noise-mixed to test manifest')
    parser.add_argument('--source', required=True, help='Source noise-mixed manifest')
    parser.add_argument('--target', required=True, help='Target combined test manifest')
    parser.add_argument('--count', type=int, required=True, help='Number of entries to move')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be moved without actually moving')
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    target_path = Path(args.target)
    
    if not source_path.exists():
        print(f"Error: Source file {source_path} does not exist")
        sys.exit(1)
    
    if not target_path.exists():
        print(f"Error: Target file {target_path} does not exist")
        sys.exit(1)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Count total lines
    source_count = count_lines(source_path)
    target_count = count_lines(target_path)
    
    print(f"Source file has {source_count} entries")
    print(f"Target file has {target_count} entries")
    print(f"Will move {args.count} entries")
    
    if args.count > source_count:
        print(f"Error: Cannot move {args.count} entries from source with only {source_count} entries")
        sys.exit(1)
    
    # Read all entries
    print("Reading source entries...")
    source_entries = read_jsonl(source_path)
    print("Reading target entries...")
    target_entries = read_jsonl(target_path)
    
    # Randomly sample entries from source
    print(f"Randomly sampling {args.count} entries...")
    sampled_indices = random.sample(range(len(source_entries)), args.count)
    sampled_entries = [source_entries[i] for i in sampled_indices]
    
    # Show sample of what will be moved
    print("\nSample of entries to be moved:")
    for i, entry in enumerate(sampled_entries[:3]):
        print(f"  {i+1}. {entry.get('raw_transcription', 'N/A')[:50]}...")
        print(f"     Audio: {entry.get('audio_path', 'N/A')}")
        print(f"     SNR: {entry.get('aug_meta', {}).get('snr_db', 'N/A')} dB")
        print()
    
    if args.dry_run:
        print("DRY RUN: Not actually moving files")
        return
    
    # Create new source list without sampled entries
    new_source_entries = [entry for i, entry in enumerate(source_entries) if i not in sampled_indices]
    
    # Combine target entries with sampled entries
    new_target_entries = target_entries + sampled_entries
    
    # Write back to files
    print(f"Writing {len(new_source_entries)} entries back to source...")
    write_jsonl(source_path, new_source_entries)
    
    print(f"Writing {len(new_target_entries)} entries to target...")
    write_jsonl(target_path, new_target_entries)
    
    print("Completed!")
    print(f"Source now has {len(new_source_entries)} entries (removed {args.count})")
    print(f"Target now has {len(new_target_entries)} entries (added {args.count})")


if __name__ == "__main__":
    main()
