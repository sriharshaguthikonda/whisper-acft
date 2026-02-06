#!/usr/bin/env python3
import os

def find_matching_files(dir1, dir2):
    """Find files with matching names in two directories"""
    
    # Get files from both directories
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))
    
    # Find matching files
    matching = files1.intersection(files2)
    
    return sorted(matching)

if __name__ == "__main__":
    dir1 = r'J:\Openwakeword_whisper_keyboard_training_data_hotword\train\Hey_computer_True_positives'
    dir2 = r'J:\Openwakeword_whisper_keyboard_training_data_hotword\train\Hey_llama_true_positives'
    
    matching_files = find_matching_files(dir1, dir2)
    
    print("Files with matching names in both directories:")
    print("==========================================")
    for filename in matching_files:
        print(filename)
    
    print(f"\nTotal matching files: {len(matching_files)}")
    
    # Also show file sizes for comparison
    print("\nFile size comparison:")
    print("====================")
    for filename in matching_files:
        file1_path = os.path.join(dir1, filename)
        file2_path = os.path.join(dir2, filename)
        
        size1 = os.path.getsize(file1_path)
        size2 = os.path.getsize(file2_path)
        
        print(f"{filename}:")
        print(f"  Hey_computer: {size1:,} bytes")
        print(f"  Hey_llama:    {size2:,} bytes")
        print(f"  Same size:    {size1 == size2}")
        print()
