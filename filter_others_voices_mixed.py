#!/usr/bin/env python3
"""
Filter manifest to remove lines containing files from Record_chunks_others_voices_mixed folder
and create separate manifests.
"""

import json
import os

def filter_manifest(input_path, output_filtered_path, output_others_path):
    """
    Filter manifest file to separate files from Record_chunks_others_voices_mixed folder.
    
    Args:
        input_path: Path to input manifest file
        output_filtered_path: Path to output manifest without others_voices_mixed files
        output_others_path: Path to output manifest with only others_voices_mixed files
    """
    # Counters
    total_lines = 0
    filtered_lines = 0
    others_lines = 0
    
    # Get total line count
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"Processing {total_lines} lines from {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_filtered_path, 'w', encoding='utf-8') as filtered_file, \
         open(output_others_path, 'w', encoding='utf-8') as others_file:
        
        for i, line in enumerate(infile, 1):
            if i % 1000 == 0:
                print(f"Processed {i}/{total_lines} lines...")
                
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                audio_path = data.get('audio_path', '')
                
                # Check if audio path contains the target folder
                if 'Record_chunks_others_voices_mixed' in audio_path:
                    others_file.write(line + '\n')
                    others_lines += 1
                else:
                    filtered_file.write(line + '\n')
                    filtered_lines += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line: {e}")
                continue
    
    print("\nFiltering complete:")
    print(f"  Total lines processed: {total_lines}")
    print(f"  Filtered manifest (without others): {filtered_lines} lines")
    print(f"  Others manifest (only others_voices_mixed): {others_lines} lines")
    print(f"  Filtered manifest saved to: {output_filtered_path}")
    print(f"  Others manifest saved to: {output_others_path}")

def main():
    # Input and output paths
    input_manifest = r"I:\Record_chunks\pairs_manifest_combined_train_with_tempo_pause_randomized_updated.jsonl"
    output_filtered = r"I:\Record_chunks\pairs_manifest_combined_train_with_tempo_pause_randomized_updated_filtered.jsonl"
    output_others = r"I:\Record_chunks\pairs_manifest_combined_train_with_tempo_pause_randomized_updated_others_only.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_manifest):
        print(f"Error: Input file not found: {input_manifest}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filtered), exist_ok=True)
    os.makedirs(os.path.dirname(output_others), exist_ok=True)
    
    # Filter the manifest
    filter_manifest(input_manifest, output_filtered, output_others)
    
    # Beep when done
    try:
        import winsound
        winsound.Beep(1000, 300)
        winsound.Beep(1200, 300)
        winsound.Beep(1500, 500)
    except ImportError:
        print("Task completed!")

if __name__ == "__main__":
    main()
