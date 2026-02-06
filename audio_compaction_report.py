#!/usr/bin/env python3
"""
Generate a report showing the compression results from the audio compaction process.
"""

from pathlib import Path

def generate_report(original_dir: Path, compacted_dir: Path):
    """Generate a compression report."""
    print("=== Audio Compaction Report ===\n")
    
    original_files = list(original_dir.glob("*.m4a"))
    compacted_files = list(compacted_dir.glob("*_compact.m4a"))
    
    total_original_size = 0
    total_compacted_size = 0
    processed_files = 0
    
    print(f"{'File Name':<50} {'Original (MB)':<15} {'Compacted (MB)':<15} {'Saved (MB)':<12} {'Saved %':<8}")
    print("-" * 105)
    
    for compacted_file in compacted_files:
        # Find corresponding original file
        original_name = compacted_file.stem.replace("_compact", "")
        original_file = original_dir / f"{original_name}.m4a"
        
        if original_file.exists():
            original_size = original_file.stat().st_size
            compacted_size = compacted_file.stat().st_size
            saved_size = original_size - compacted_size
            saved_percent = (saved_size / original_size) * 100
            
            total_original_size += original_size
            total_compacted_size += compacted_size
            processed_files += 1
            
            print(f"{original_name:<50} {original_size/1024/1024:>13.1f} {compacted_size/1024/1024:>13.1f} {saved_size/1024/1024:>10.1f} {saved_percent:>6.1f}%")
    
    print("-" * 105)
    
    total_saved = total_original_size - total_compacted_size
    total_saved_percent = (total_saved / total_original_size) * 100
    
    print(f"{'TOTAL':<50} {total_original_size/1024/1024:>13.1f} {total_compacted_size/1024/1024:>13.1f} {total_saved/1024/1024:>10.1f} {total_saved_percent:>6.1f}%")
    print(f"\nFiles processed: {processed_files}")
    print(f"Total space saved: {total_saved/1024/1024:.1f} MB ({total_saved_percent:.1f}%)")
    
    # Check for files that weren't processed
    processed_names = {f.stem.replace("_compact", "") for f in compacted_files}
    original_names = {f.stem for f in original_files}
    not_processed = original_names - processed_names
    
    if not_processed:
        print(f"\nFiles not processed ({len(not_processed)}):")
        for name in sorted(not_processed):
            print(f"  - {name}.m4a")

if __name__ == "__main__":
    original_dir = Path(r"I:\Record_others")
    compacted_dir = Path(r"I:\Record_others_compacted")
    
    generate_report(original_dir, compacted_dir)
    
    # Beep to notify completion
    print('\a')
