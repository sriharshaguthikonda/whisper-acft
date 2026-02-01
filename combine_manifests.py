#!/usr/bin/env python3
"""Combine multiple manifest files into one"""

import json
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

def read_manifest(manifest_path: Path) -> List[Dict[str, Any]]:
    """Read manifest file and return list of entries"""
    entries = []
    print(f"Reading {manifest_path}...")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc=f"Reading {manifest_path.name}"), 1):
            try:
                row = json.loads(line.strip())
                entries.append(row)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return entries

def write_manifest(entries: List[Dict[str, Any]], output_path: Path) -> None:
    """Write entries to JSONL manifest file"""
    print(f"Writing {len(entries):,} entries to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in tqdm(entries, desc="Writing manifest"):
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

def main():
    # Input manifests
    manifests = [
        Path("I:\\Record_chunks\\only_stage9_reverb.jsonl"),
        Path("I:\\Record_chunks\\only_others_voices_mixed.jsonl"),
        Path("I:\\Record_chunks\\pairs_manifest_stereo_english_only_filtered_with_uids_score_filtered.jsonl")
    ]
    
    # Output manifest
    output_path = Path("I:\\Record_chunks\\pairs_manifest_combined_all_datasets.jsonl")
    
    print("="*80)
    print("COMBINING MANIFEST FILES")
    print("="*80)
    
    # Read all manifests
    all_entries = []
    total_files = 0
    
    for manifest in manifests:
        if not manifest.exists():
            print(f"WARNING: {manifest} not found!")
            continue
        
        entries = read_manifest(manifest)
        all_entries.extend(entries)
        total_files += len(entries)
        print(f"Added {len(entries):,} entries from {manifest.name}")
    
    print(f"\nTotal entries combined: {len(all_entries):,}")
    
    # Write combined manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(all_entries, output_path)
    
    print(f"\nSuccessfully created combined manifest: {output_path}")
    print(f"Combined manifest contains {len(all_entries):,} total entries")
    print("="*80)

if __name__ == "__main__":
    main()
