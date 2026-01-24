#!/usr/bin/env python3
"""
Script to move audio files from test manifest to I:\Record_test_chunks\ 
and update the manifest with new paths.
"""

import json
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def move_test_chunks():
    # Paths
    manifest_path = r"I:\Record_chunks\pairs_manifest_sorted_by_scores_english_only_filtered_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_randomized_filtered_test.jsonl"
    target_dir = r"I:\Record_test_chunks"
    backup_manifest = manifest_path + ".backup"
    
    # Create target directory if it doesn't exist
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Reading manifest from: {manifest_path}")
    print(f"Target directory: {target_dir}")
    
    # Read all lines from manifest
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} entries in manifest")
    
    # Create backup
    shutil.copy2(manifest_path, backup_manifest)
    print(f"Created backup: {backup_manifest}")
    
    updated_lines = []
    moved_files = 0
    skipped_files = 0
    
    for line_num, line in enumerate(tqdm(lines, desc="Processing entries")):
        try:
            entry = json.loads(line.strip())
            audio_path = entry["audio_path"]
            
            # Check if file exists
            if not os.path.exists(audio_path):
                print(f"Warning: File not found: {audio_path}")
                skipped_files += 1
                updated_lines.append(line)
                continue
            
            # Get filename
            filename = os.path.basename(audio_path)
            new_audio_path = os.path.join(target_dir, filename)
            
            # Move file if it's not already in target directory
            if audio_path != new_audio_path:
                # Check if target file already exists
                if os.path.exists(new_audio_path):
                    print(f"Warning: Target file already exists: {new_audio_path}")
                    # Generate unique name
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(os.path.join(target_dir, f"{base}_{counter}{ext}")):
                        counter += 1
                    new_audio_path = os.path.join(target_dir, f"{base}_{counter}{ext}")
                
                # Move the file
                shutil.move(audio_path, new_audio_path)
                moved_files += 1
                
                # Update the entry
                entry["audio_path"] = new_audio_path
            
            # Also update transcript_json path if it exists and is related
            if entry.get("transcript_json"):
                transcript_path = entry["transcript_json"]
                if os.path.exists(transcript_path):
                    # For now, we'll keep transcript paths as they are
                    # Only move if they're in the same directory structure as audio
                    pass
            
            updated_lines.append(json.dumps(entry, ensure_ascii=False) + "\n")
            
        except json.JSONDecodeError as e:
            print(f"Error parsing line {line_num + 1}: {e}")
            updated_lines.append(line)
        except Exception as e:
            print(f"Error processing line {line_num + 1}: {e}")
            updated_lines.append(line)
    
    # Write updated manifest
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)
    
    print(f"\nSummary:")
    print(f"- Total entries: {len(lines)}")
    print(f"- Files moved: {moved_files}")
    print(f"- Files skipped: {skipped_files}")
    print(f"- Updated manifest: {manifest_path}")
    print(f"- Backup created: {backup_manifest}")
    
    # Add beep notification
    print('\a')

if __name__ == "__main__":
    move_test_chunks()
