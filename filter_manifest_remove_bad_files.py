#!/usr/bin/env python3
"""
Script to filter out manifest entries that reference files in the bad folder.
Removes entries where audio_path contains 'Record_chunks_others_voices_mixed' and creates a clean manifest.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def filter_manifest(input_manifest_path, output_manifest_path, bad_folder_path):
    """
    Filter out entries that reference files in the bad folder.
    
    Args:
        input_manifest_path: Path to input JSONL manifest file
        output_manifest_path: Path to output filtered JSONL manifest file
        bad_folder_path: Path to bad folder to filter out
    """
    # Normalize the bad folder path for consistent comparison
    bad_folder_normalized = os.path.normpath(bad_folder_path).lower()
    
    total_entries = 0
    filtered_entries = 0
    kept_entries = 0
    
    print(f"Reading manifest from: {input_manifest_path}")
    print(f"Filtering out entries with paths containing: {bad_folder_normalized}")
    print(f"Writing filtered manifest to: {output_manifest_path}")
    
    # Count total lines first for progress bar
    with open(input_manifest_path, 'r', encoding='utf-8') as f:
        total_entries = sum(1 for _ in f)
    
    with open(input_manifest_path, 'r', encoding='utf-8') as infile, \
         open(output_manifest_path, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, total=total_entries, desc="Filtering manifest"):
            try:
                entry = json.loads(line.strip())
                total_entries += 1
                
                # Get the audio path and normalize it
                audio_path = entry.get('audio_path', '')
                audio_path_normalized = os.path.normpath(audio_path).lower()
                
                # Check if the audio path contains the bad folder
                if bad_folder_normalized in audio_path_normalized:
                    filtered_entries += 1
                    continue  # Skip this entry
                
                # Write the entry to the output file
                outfile.write(line)
                kept_entries += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {total_entries + 1}: {e}")
                continue
    
    print(f"\nFiltering complete!")
    print(f"Total entries processed: {total_entries}")
    print(f"Entries filtered out: {filtered_entries}")
    print(f"Entries kept: {kept_entries}")
    print(f"Filtered manifest saved to: {output_manifest_path}")

def main():
    parser = argparse.ArgumentParser(description='Filter manifest to remove bad folder entries')
    parser.add_argument('--input', required=True, help='Input manifest JSONL file')
    parser.add_argument('--output', required=True, help='Output filtered manifest JSONL file')
    parser.add_argument('--bad-folder', required=True, help='Bad folder path to filter out')
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filter_manifest(args.input, args.output, args.bad_folder)
    return 0

if __name__ == "__main__":
    exit(main())
