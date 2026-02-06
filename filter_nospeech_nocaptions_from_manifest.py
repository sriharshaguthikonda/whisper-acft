#!/usr/bin/env python3
"""
filter_nospeech_nocaptions_from_manifest.py

Goal
----
Remove lines containing no speech or no captions tokens from a Whisper-style JSONL manifest.

This script filters out entries that contain:
- "<|nospeech|>" in raw_transcription
- "<|nocaptions|>" in raw_transcription

Usage
-----
python filter_nospeech_nocaptions_from_manifest.py --input "I:\Record_chunks\pairs_manifest_local_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_silent_randomized_filtered_train_no_targets.jsonl" --output "I:\Record_chunks\pairs_manifest_local_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_silent_randomized_filtered_train_no_targets_filtered.jsonl"
"""

import argparse
import json
import sys
from pathlib import Path
from tqdm import tqdm


def filter_manifest(input_path: str, output_path: str) -> None:
    """
    Filter out lines containing nospeech or nocaptions tokens from a JSONL manifest.
    
    Args:
        input_path: Path to input JSONL manifest file
        output_path: Path to output filtered JSONL manifest file
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Count total lines first for progress bar
    print("Counting total lines...")
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    print(f"Total lines to process: {total_lines:,}")
    
    kept_count = 0
    removed_count = 0
    nospeech_count = 0
    nocaptions_count = 0
    
    print("Filtering manifest...")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(tqdm(infile, total=total_lines, desc="Processing"), 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                raw_transcription = data.get('raw_transcription', '')
                
                # Check for tokens to filter out
                has_nospeech = '<|nospeech|>' in raw_transcription
                has_nocaptions = '<|nocaptions|>' in raw_transcription
                
                if has_nospeech:
                    nospeech_count += 1
                    removed_count += 1
                    continue
                elif has_nocaptions:
                    nocaptions_count += 1
                    removed_count += 1
                    continue
                else:
                    # Keep this line
                    outfile.write(line + '\n')
                    kept_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"\nFiltering complete!")
    print(f"Total lines processed: {total_lines:,}")
    print(f"Lines kept: {kept_count:,}")
    print(f"Lines removed: {removed_count:,}")
    print(f"  - <|nospeech|> entries: {nospeech_count:,}")
    print(f"  - <|nocaptions|> entries: {nocaptions_count:,}")
    print(f"Output written to: {output_path}")
    
    # Calculate percentages
    if total_lines > 0:
        kept_percentage = (kept_count / total_lines) * 100
        removed_percentage = (removed_count / total_lines) * 100
        print(f"Kept: {kept_percentage:.2f}%")
        print(f"Removed: {removed_percentage:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Filter out nospeech and nocaptions entries from a JSONL manifest"
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input JSONL manifest file path'
    )
    parser.add_argument(
        '--output', '-o',
        required=True,
        help='Output JSONL manifest file path'
    )
    
    args = parser.parse_args()
    
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    
    filter_manifest(args.input, args.output)


if __name__ == '__main__':
    main()
