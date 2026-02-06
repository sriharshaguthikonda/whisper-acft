#!/usr/bin/env python3
"""
Sort manifest JSONL file by speaker sort scores CSV.
High score audios and their sister audios should be at the top.
"""

import csv
import json
import os
from tqdm import tqdm
import argparse
from collections import defaultdict

def load_speaker_scores(csv_path):
    """Load speaker scores from CSV file."""
    scores = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = row['file']
            score_str = row['score'].strip()
            if score_str:  # Skip empty scores
                try:
                    score = float(score_str)
                    scores[file_path] = score
                except ValueError:
                    print(f"Warning: Invalid score '{score_str}' for file {file_path}, skipping")
            else:
                print(f"Warning: Empty score for file {file_path}, skipping")
    return scores

def extract_base_audio_name(audio_path):
    """Extract base audio name from chunk path to identify sister audios."""
    # Remove chunk identifier like _sent0000
    basename = os.path.basename(audio_path)
    if '_sent' in basename:
        return basename.split('_sent')[0]
    return basename

def load_manifest(manifest_path):
    """Load manifest from JSONL file."""
    manifest = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                manifest.append(json.loads(line.strip()))
    return manifest

def group_by_base_audio(manifest):
    """Group manifest entries by base audio name."""
    groups = defaultdict(list)
    for entry in manifest:
        audio_path = entry['audio_path']
        base_name = extract_base_audio_name(audio_path)
        groups[base_name].append(entry)
    return groups

def calculate_group_score(group_entries, speaker_scores):
    """Calculate score for a group based on max score among its entries."""
    max_score = float('-inf')
    for entry in group_entries:
        audio_path = entry['audio_path']
        if audio_path in speaker_scores:
            max_score = max(max_score, speaker_scores[audio_path])
    return max_score if max_score != float('-inf') else 0

def sort_manifest_by_scores(manifest_path, scores_csv_path, output_path):
    """Sort manifest by speaker scores."""
    print("Loading speaker scores...")
    speaker_scores = load_speaker_scores(scores_csv_path)
    print(f"Loaded {len(speaker_scores)} speaker scores")
    
    print("Loading manifest...")
    manifest = load_manifest(manifest_path)
    print(f"Loaded {len(manifest)} manifest entries")
    
    print("Grouping by base audio...")
    groups = group_by_base_audio(manifest)
    print(f"Found {len(groups)} base audio groups")
    
    # Calculate scores for each group
    group_scores = []
    for base_name, entries in groups.items():
        group_score = calculate_group_score(entries, speaker_scores)
        group_scores.append((group_score, base_name, entries))
    
    # Sort groups by score (descending)
    print("Sorting groups by score...")
    group_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Create sorted manifest
    sorted_manifest = []
    for score, base_name, entries in tqdm(group_scores, desc="Processing groups"):
        # Sort entries within group by score too
        entries_with_scores = []
        for entry in entries:
            audio_path = entry['audio_path']
            entry_score = speaker_scores.get(audio_path, 0)
            entries_with_scores.append((entry_score, entry))
        
        # Sort entries within group by score (descending)
        entries_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Add to sorted manifest
        for entry_score, entry in entries_with_scores:
            sorted_manifest.append(entry)
    
    # Write sorted manifest
    print(f"Writing sorted manifest to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in sorted_manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Done! Sorted manifest written to {output_path}")
    print(f"Total entries: {len(sorted_manifest)}")
    
    # Print some statistics
    print("\nTop 10 groups by score:")
    for i, (score, base_name, entries) in enumerate(group_scores[:10]):
        print(f"{i+1}. {base_name}: {score:.3f} ({len(entries)} chunks)")

def main():
    parser = argparse.ArgumentParser(description='Sort manifest by speaker scores')
    parser.add_argument('--manifest', required=True, help='Path to input manifest JSONL file')
    parser.add_argument('--scores', required=True, help='Path to speaker scores CSV file')
    parser.add_argument('--output', required=True, help='Path to output sorted manifest JSONL file')
    
    args = parser.parse_args()
    
    sort_manifest_by_scores(args.manifest, args.scores, args.output)

if __name__ == "__main__":
    main()
