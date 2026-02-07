#!/usr/bin/env python3
import os
from tqdm import tqdm

def delete_matching_files_from_record_harsha():
    """Delete mixed_true_positive_*.wav files from Record_harsha that match between hotword folders"""
    
    # Directories
    dir1 = r'J:\Openwakeword_whisper_keyboard_training_data_hotword\train\Hey_computer_True_positives'
    dir2 = r'J:\Openwakeword_whisper_keyboard_training_data_hotword\train\Hey_llama_true_positives'
    target_dir = r'I:\P2GPT_google_drive\My Drive\Record_harsha'
    
    # Get matching files between the two hotword directories
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))
    matching = files1.intersection(files2)
    
    # Filter only mixed_true_positive_*.wav files
    mixed_files = [f for f in matching if f.startswith('mixed_true_positive_') and f.endswith('.wav')]
    
    print(f"Found {len(mixed_files)} mixed_true_positive files to delete from Record_harsha")
    
    # Check which files exist in target directory and delete them
    deleted_count = 0
    not_found_count = 0
    
    for filename in tqdm(mixed_files, desc="Deleting files"):
        target_path = os.path.join(target_dir, filename)
        
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
                deleted_count += 1
                print(f"Deleted: {filename}")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
        else:
            not_found_count += 1
            print(f"Not found in Record_harsha: {filename}")
    
    print(f"\nSummary:")
    print(f"Total matching files: {len(mixed_files)}")
    print(f"Successfully deleted: {deleted_count}")
    print(f"Not found in Record_harsha: {not_found_count}")

if __name__ == "__main__":
    delete_matching_files_from_record_harsha()
