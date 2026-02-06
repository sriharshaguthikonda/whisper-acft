#!/usr/bin/env python3
"""
Append noise-mixed entries to the combined manifest.
"""

import argparse
import sys
from pathlib import Path
from tqdm import tqdm


def count_lines(file_path):
    """Count lines in a JSONL file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def append_jsonl(source_path, target_path):
    """Append entries from source to target JSONL file."""
    source_count = 0
    target_count_before = count_lines(target_path)
    
    print(f"Source file: {source_path}")
    print(f"Target file: {target_path}")
    print(f"Target has {target_count_before:,} entries before appending")
    
    # Append entries
    with open(source_path, 'r', encoding='utf-8') as src, \
         open(target_path, 'a', encoding='utf-8') as tgt:
        
        for line in tqdm(src, desc="Appending entries"):
            line = line.strip()
            if line:
                tgt.write(line + '\n')
                source_count += 1
    
    target_count_after = count_lines(target_path)
    
    print(f"\nAppended {source_count:,} entries")
    print(f"Target now has {target_count_after:,} entries")
    print(f"Increase: {target_count_after - target_count_before:,} entries")
    
    return source_count


def main():
    parser = argparse.ArgumentParser(description='Append noise-mixed entries to combined manifest')
    parser.add_argument('--source', required=True, help='Source noise-mixed manifest')
    parser.add_argument('--target', required=True, help='Target combined manifest')
    parser.add_argument('--dry_run', action='store_true', help='Show counts without actually appending')
    
    args = parser.parse_args()
    
    source_path = Path(args.source)
    target_path = Path(args.target)
    
    if not source_path.exists():
        print(f"Error: Source file {source_path} does not exist")
        sys.exit(1)
    
    if not target_path.exists():
        print(f"Error: Target file {target_path} does not exist")
        sys.exit(1)
    
    # Count entries
    source_count = count_lines(source_path)
    target_count = count_lines(target_path)
    
    print(f"Source has {source_count:,} entries")
    print(f"Target has {target_count:,} entries")
    print(f"After appending, target will have {source_count + target_count:,} entries")
    
    if args.dry_run:
        print("DRY RUN: Not actually appending files")
        return
    
    # Append the entries
    appended_count = append_jsonl(source_path, target_path)
    
    print(f"\nSuccessfully appended {appended_count:,} entries to combined manifest!")


if __name__ == "__main__":
    main()
