#!/usr/bin/env python3
"""Check overlap between manifest files"""

import json
from pathlib import Path
from typing import Set, Dict, List
from tqdm import tqdm

def read_manifest_paths(manifest_path: Path) -> Set[str]:
    """Read manifest and return set of audio paths"""
    paths = set()
    print(f"Reading {manifest_path}...")
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc=f"Reading {manifest_path.name}"), 1):
            try:
                row = json.loads(line.strip())
                audio_path = row.get('audio_path', '')
                if audio_path:
                    paths.add(audio_path)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return paths

def main():
    manifests = [
        Path("I:\\Record_chunks\\pairs_manifest_stage7_plus_stage9_reverb.jsonl"),
        Path("I:\\Record_chunks\\pairs_manifest_stereo_english_only_filtered_others_voices_mixed.jsonl"),
        Path("I:\\Record_chunks\\pairs_manifest_stereo_english_only_filtered_with_uids_score_filtered.jsonl")
    ]
    
    names = [
        "stage7_plus_stage9_reverb",
        "stereo_english_only_filtered_others_voices_mixed", 
        "stereo_english_only_filtered_with_uids_score_filtered"
    ]
    
    # Read all manifests
    manifest_paths = {}
    for manifest, name in zip(manifests, names):
        if not manifest.exists():
            print(f"WARNING: {manifest} not found!")
            continue
        manifest_paths[name] = read_manifest_paths(manifest)
    
    print("\n" + "="*80)
    print("MANIFEST OVERLAP ANALYSIS")
    print("="*80)
    
    # Print counts
    for name, paths in manifest_paths.items():
        print(f"{name}: {len(paths):,} unique audio files")
    
    print("\n" + "-"*60)
    
    # Check overlaps
    manifest_list = list(manifest_paths.keys())
    for i in range(len(manifest_list)):
        for j in range(i+1, len(manifest_list)):
            name1, name2 = manifest_list[i], manifest_list[j]
            paths1, paths2 = manifest_paths[name1], manifest_paths[name2]
            
            overlap = paths1.intersection(paths2)
            overlap_pct = (len(overlap) / min(len(paths1), len(paths2))) * 100 if paths1 and paths2 else 0
            
            print(f"\n{name1} vs {name2}:")
            print(f"  Overlap: {len(overlap):,} files ({overlap_pct:.1f}% of smaller)")
            print(f"  Unique to {name1}: {len(paths1 - paths2):,}")
            print(f"  Unique to {name2}: {len(paths2 - paths1):,}")
    
    # Three-way overlap
    if len(manifest_paths) == 3:
        three_way_overlap = set.intersection(*manifest_paths.values())
        print(f"\nThree-way overlap: {len(three_way_overlap):,} files")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
