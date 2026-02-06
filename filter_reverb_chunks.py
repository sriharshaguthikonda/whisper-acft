#!/usr/bin/env python3
"""
Filter out lines with files from I:\Record_chunks_reverb from manifest file
and create separate files for filtered and reverb-only chunks.
"""

import json
from pathlib import Path
from tqdm import tqdm

def filter_manifest_file(input_file_path, output_filtered_path, output_reverb_path):
    """
    Filter manifest file to separate reverb chunks from regular chunks.
    
    Args:
        input_file_path: Path to input manifest file
        output_filtered_path: Path for filtered output (without reverb)
        output_reverb_path: Path for reverb-only output
    """
    input_path = Path(input_file_path)
    filtered_path = Path(output_filtered_path)
    reverb_path = Path(output_reverb_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")
    
    # Count total lines for progress bar
    total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8'))
    
    filtered_count = 0
    reverb_count = 0
    total_count = 0
    
    print(f"Processing {total_lines:,} lines from {input_file_path}")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(filtered_path, 'w', encoding='utf-8') as filtered_out, \
         open(reverb_path, 'w', encoding='utf-8') as reverb_out:
        
        for line in tqdm(infile, total=total_lines, desc="Filtering lines"):
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                audio_path = data.get('audio_path', '')
                
                total_count += 1
                
                # Check if audio path contains Record_chunks_reverb
                if 'Record_chunks_reverb' in audio_path or 'I:\\Record_chunks_reverb' in audio_path:
                    reverb_out.write(line + '\n')
                    reverb_count += 1
                else:
                    filtered_out.write(line + '\n')
                    filtered_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {total_count + 1}: {e}")
                continue
    
    print("Filtering complete!")
    print(f"Total lines processed: {total_count:,}")
    print(f"Filtered (no reverb): {filtered_count:,} lines")
    print(f"Reverb chunks: {reverb_count:,} lines")
    print(f"Filtered output: {output_filtered_path}")
    print(f"Reverb output: {output_reverb_path}")
    
    return filtered_count, reverb_count

def main():
    # Input and output file paths
    input_file = r"I:\Record_chunks\pairs_manifest_combined_all_datasets_randomized_train.jsonl"
    output_filtered = r"I:\Record_chunks\pairs_manifest_combined_all_datasets_randomized_train_no_reverb.jsonl"
    output_reverb = r"I:\Record_chunks\pairs_manifest_combined_all_datasets_randomized_train_reverb_only.jsonl"
    
    try:
        filtered_count, reverb_count = filter_manifest_file(
            input_file, output_filtered, output_reverb
        )
        
        # Beep when done
        import winsound
        winsound.Beep(1000, 300)
        winsound.Beep(1200, 300)
        winsound.Beep(1500, 500)
        
    except Exception as e:
        print(f"Error: {e}")
        import winsound
        winsound.Beep(800, 500)  # Error beep

if __name__ == "__main__":
    main()
