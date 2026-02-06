#!/usr/bin/env python3
"""
Clean up rename_and_corrected_transcript_report.json by removing entries where:
1. original_transcript and corrected_transcript are almost the same
2. corrected_transcript is empty
"""

import json
import difflib
from pathlib import Path

def are_texts_similar(text1, text2, threshold=0.95):
    """Check if two texts are similar using sequence matcher"""
    if not text1 or not text2:
        return False
    
    # Normalize texts (remove extra whitespace, lowercase)
    norm1 = ' '.join(text1.lower().split())
    norm2 = ' '.join(text2.lower().split())
    
    # Calculate similarity ratio
    similarity = difflib.SequenceMatcher(None, norm1, norm2).ratio()
    return similarity >= threshold

def clean_transcript_report(input_file, output_file, similarity_threshold=0.95):
    """Clean the transcript report file"""
    print(f"Loading transcript report from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    cleaned_data = []
    removed_similar = 0
    removed_empty = 0
    
    print(f"Processing {original_count} entries...")
    
    for entry in data:
        original_transcript = entry.get('original_transcript', '')
        corrected_transcript = entry.get('corrected_transcript', '')
        
        # Skip if corrected_transcript is empty
        if not corrected_transcript.strip():
            removed_empty += 1
            continue
        
        # Skip if original and corrected transcripts are too similar
        if are_texts_similar(original_transcript, corrected_transcript, similarity_threshold):
            removed_similar += 1
            continue
        
        # Keep the entry
        cleaned_data.append(entry)
    
    # Write cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    
    final_count = len(cleaned_data)
    
    print(f"\n=== Summary ===")
    print(f"Original entries: {original_count}")
    print(f"Entries removed (similar transcripts): {removed_similar}")
    print(f"Entries removed (empty corrected): {removed_empty}")
    print(f"Total removed: {removed_similar + removed_empty}")
    print(f"Final entries: {final_count}")
    print(f"Cleaned file saved to: {output_file}")
    
    return cleaned_data

if __name__ == "__main__":
    input_file = r"i:\whisper-acft\rename_and_corrected_transcript_report.json"
    output_file = r"i:\whisper-acft\rename_and_corrected_transcript_report_cleaned.json"
    
    # Create backup first
    backup_file = r"i:\whisper-acft\rename_and_corrected_transcript_report_backup.json"
    print(f"Creating backup at {backup_file}...")
    
    import shutil
    shutil.copy2(input_file, backup_file)
    
    # Clean the file
    clean_transcript_report(input_file, output_file, similarity_threshold=0.95)
    
    # Beep when done
    import winsound
    winsound.Beep(1000, 300)
    winsound.Beep(1200, 300)
    winsound.Beep(1500, 500)
