#!/usr/bin/env python3
"""analyze_manifest_randomization.py

Analyze the randomization quality of combined_all_manifests.jsonl
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm

def read_jsonl_sample(path: Path, sample_size: int = 5000):
    """Read a sample of lines from JSONL file for analysis."""
    rows = []
    
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(tqdm(f, desc="Reading manifest sample")):
            if line_no >= sample_size:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Bad JSON on line {line_no + 1}: {e}")
                continue
    
    return rows

def analyze_clustering(rows, window_size=100):
    """Analyze clustering by counting consecutive entries from same source."""
    if len(rows) < window_size:
        window_size = len(rows)
    
    consecutive_same_source = 0
    source_changes = 0
    source_sequences = defaultdict(list)
    
    for i in range(1, window_size):
        curr_source = rows[i].get("source_audio", "").split("\\")[-1] if "\\" in rows[i].get("source_audio", "") else rows[i].get("source_audio", "")
        prev_source = rows[i-1].get("source_audio", "").split("\\")[-1] if "\\" in rows[i-1].get("source_audio", "") else rows[i-1].get("source_audio", "")
        
        if curr_source == prev_source:
            consecutive_same_source += 1
        else:
            source_changes += 1
            
        # Track sequences
        source_sequences[curr_source].append(i)
    
    clustering_score = consecutive_same_source / window_size
    return clustering_score, source_changes, source_sequences

def analyze_augmentation_distribution(rows):
    """Analyze distribution of different augmentation types."""
    aug_types = defaultdict(int)
    original_count = 0
    
    for row in rows:
        audio_path = row.get("audio_path", "")
        if "reverb" in audio_path:
            aug_types["reverb"] += 1
        elif "noise_mix" in audio_path:
            aug_types["noise_mix"] += 1
        elif "voice_mix" in audio_path:
            aug_types["voice_mix"] += 1
        elif "tempo" in audio_path:
            aug_types["tempo"] += 1
        else:
            original_count += 1
    
    return aug_types, original_count

def check_chunk_index_sequencing(rows):
    """Check if chunks from same recording are in sequential order."""
    source_chunks = defaultdict(list)
    
    for i, row in enumerate(rows):
        source = row.get("source_audio", "")
        chunk_idx = row.get("chunk_index", 0)
        source_chunks[source].append((i, chunk_idx))
    
    sequencing_issues = 0
    total_sequences = 0
    
    for source, positions in source_chunks.items():
        if len(positions) > 1:
            total_sequences += 1
            # Sort by position in file
            positions.sort()
            # Check if chunk indices are mostly sequential
            chunk_indices = [chunk_idx for pos, chunk_idx in positions]
            
            # Count how many times the next chunk is +1 from current
            sequential_count = 0
            for i in range(1, len(chunk_indices)):
                if chunk_indices[i] == chunk_indices[i-1] + 1:
                    sequential_count += 1
            
            # If more than 50% are sequential, that's a sign of poor randomization
            if sequential_count / len(chunk_indices) > 0.5:
                sequencing_issues += 1
    
    return sequencing_issues, total_sequences

def main():
    manifest_path = Path("I:/Record_chunks/combined_all_manifests.jsonl")
    
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return
    
    print("Analyzing manifest randomization quality...")
    print(f"Manifest: {manifest_path}")
    print()
    
    # Read sample for analysis
    sample_size = 10000
    rows = read_jsonl_sample(manifest_path, sample_size)
    print(f"Analyzed {len(rows):,} entries from manifest")
    
    # Get total count
    total_lines = 0
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                total_lines += 1
    
    print(f"Total manifest size: {total_lines:,} entries")
    print()
    
    # Analyze clustering
    print("=== CLUSTERING ANALYSIS ===")
    clustering_score, source_changes, source_sequences = analyze_clustering(rows)
    
    print(f"Clustering score (lower is better): {clustering_score:.4f}")
    print(f"Source changes in first 100 entries: {source_changes}")
    print(f"Expected random changes: ~95-100")
    
    # Show top sources by frequency in sample
    sorted_sources = sorted(source_sequences.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    print("\nTop 10 sources in sample:")
    for i, (source, positions) in enumerate(sorted_sources, 1):
        source_name = source.split("\\")[-1] if "\\" in source else source
        print(f"  {i:2d}. {source_name[:40]:<40}: {len(positions)} entries")
    
    # Analyze augmentation distribution
    print("\n=== AUGMENTATION DISTRIBUTION ===")
    aug_types, original_count = analyze_augmentation_distribution(rows)
    
    total_analyzed = len(rows)
    print(f"Original entries: {original_count:,} ({original_count/total_analyzed*100:.1f}%)")
    for aug_type, count in aug_types.items():
        print(f"{aug_type:12s}: {count:,} ({count/total_analyzed*100:.1f}%)")
    
    # Check chunk sequencing
    print("\n=== CHUNK SEQUENCING ANALYSIS ===")
    sequencing_issues, total_sequences = check_chunk_index_sequencing(rows)
    
    print(f"Sources with sequential chunks: {sequencing_issues}/{total_sequences} ({sequencing_issues/total_sequences*100:.1f}%)")
    print("Lower percentage indicates better randomization")
    
    # Overall assessment
    print("\n=== RANDOMIZATION ASSESSMENT ===")
    
    # Scoring criteria
    clustering_good = clustering_score < 0.1  # Less than 10% consecutive same sources
    sequencing_good = sequencing_issues/total_sequences < 0.3  # Less than 30% sequential
    
    print(f"Clustering quality: {'✓ GOOD' if clustering_good else '✗ POOR'} (score: {clustering_score:.4f})")
    print(f"Sequencing quality: {'✓ GOOD' if sequencing_good else '✗ POOR'} ({sequencing_issues/total_sequences*100:.1f}% sequential)")
    
    if clustering_good and sequencing_good:
        print("\n🎉 MANIFEST APPEARS WELL RANDOMIZED")
    else:
        print("\n⚠️  MANIFEST MAY NEED BETTER RANDOMIZATION")
        if not clustering_good:
            print("   - High clustering detected (same sources appear consecutively)")
        if not sequencing_good:
            print("   - Sequential chunk ordering detected")

if __name__ == "__main__":
    main()
