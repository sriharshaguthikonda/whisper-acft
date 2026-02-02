#!/usr/bin/env python3
"""verify_randomization.py

Quick verification script to compare original vs randomized manifest quality.
"""

import json
from collections import defaultdict
from pathlib import Path

def analyze_clustering(manifest_path, sample_size=1000):
    """Analyze clustering in a manifest file."""
    consecutive_same_source = 0
    source_changes = 0
    source_counts = defaultdict(int)
    
    rows = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if line_num >= sample_size:
                break
            if line.strip():
                rows.append(json.loads(line))
    
    for i in range(1, len(rows)):
        curr_source = rows[i].get("source_audio", "")
        prev_source = rows[i-1].get("source_audio", "")
        
        source_counts[curr_source] += 1
        
        if curr_source == prev_source:
            consecutive_same_source += 1
        else:
            source_changes += 1
    
    clustering_score = consecutive_same_source / len(rows)
    return clustering_score, source_changes, len(source_counts)

def main():
    original = Path(r"I:\Record_chunks\pairs_manifest_combined_all_datasets_randomized_train_no_reverb.jsonl")
    fixed = Path(r"I:\Record_chunks\pairs_manifest_combined_all_datasets_randomized_train_no_reverb_fixed.jsonl")
    
    print("COMPARING RANDOMIZATION QUALITY")
    print("=" * 50)
    
    # Analyze original
    orig_cluster, orig_changes, orig_sources = analyze_clustering(original)
    print(f"Original Manifest:")
    print(f"  Clustering score: {orig_cluster:.4f}")
    print(f"  Source changes: {orig_changes}")
    print(f"  Unique sources: {orig_sources}")
    
    # Analyze fixed
    fixed_cluster, fixed_changes, fixed_sources = analyze_clustering(fixed)
    print(f"\nFixed Manifest:")
    print(f"  Clustering score: {fixed_cluster:.4f}")
    print(f"  Source changes: {fixed_changes}")
    print(f"  Unique sources: {fixed_sources}")
    
    # Show improvement
    improvement = (orig_cluster - fixed_cluster) / orig_cluster * 100
    print(f"\nImprovement:")
    print(f"  Clustering reduced by: {improvement:.1f}%")
    print(f"  Source diversity increased by: {fixed_changes - orig_changes} changes")
    
    if fixed_cluster < orig_cluster:
        print("\n✅ RANDOMIZATION SUCCESSFUL!")
    else:
        print("\n❌ Randomization needs improvement")

if __name__ == "__main__":
    main()
