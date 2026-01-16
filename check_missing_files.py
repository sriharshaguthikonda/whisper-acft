#!/usr/bin/env python3
"""
Script to check for files that are in speaker_sort_state.json but missing from speaker_sort_scores.csv
"""
import pandas as pd
import json
from pathlib import Path

def check_missing_files():
    """Compare files in state.json vs scores.csv"""
    
    # Read the CSV file
    print("Reading speaker_sort_scores.csv...")
    df_scores = pd.read_csv("I:\\whisper-acft\\speaker_sort_scores.csv")
    csv_files = set(df_scores['file'].tolist())
    print(f"CSV contains {len(csv_files)} files")
    
    # Read the state JSON file
    print("Reading speaker_sort_state.json...")
    with open("I:\\whisper-acft\\speaker_sort_state.json", 'r') as f:
        state_data = json.load(f)
    
    # Get all files from state (assuming it's a list of file paths)
    if isinstance(state_data, list):
        state_files = set(state_data)
    elif isinstance(state_data, dict) and 'files' in state_data:
        state_files = set(state_data['files'])
    else:
        # Try to extract files from the structure
        state_files = set()
        for key, value in state_data.items():
            if isinstance(value, list):
                state_files.update(value)
            elif isinstance(value, str) and value.endswith('.wav'):
                state_files.add(value)
    
    print(f"State JSON contains {len(state_files)} files")
    
    # Find files in state but not in CSV
    missing_files = state_files - csv_files
    print(f"\nFiles in state.json but missing from scores.csv: {len(missing_files)}")
    
    if missing_files:
        print("\nFirst 20 missing files:")
        for i, file in enumerate(sorted(missing_files)[:20]):
            print(f"  {i+1}. {file}")
        
        if len(missing_files) > 20:
            print(f"  ... and {len(missing_files) - 20} more")
        
        # Save missing files list
        with open("I:\\whisper-acft\\missing_files.txt", 'w') as f:
            for file in sorted(missing_files):
                f.write(f"{file}\n")
        print(f"\nSaved complete list to: I:\\whisper-acft\\missing_files.txt")
    else:
        print("No missing files found!")
    
    # Also check for files in CSV but not in state
    extra_files = csv_files - state_files
    if extra_files:
        print(f"\nFiles in scores.csv but not in state.json: {len(extra_files)}")
        print("First 10 extra files:")
        # Convert to list and sort safely
        extra_list = list(extra_files)
        try:
            sorted_extra = sorted(extra_list)
        except TypeError:
            # Handle mixed types by converting to string for sorting
            sorted_extra = sorted([str(f) for f in extra_list])
        
        for i, file in enumerate(sorted_extra[:10]):
            print(f"  {i+1}. {file}")

if __name__ == "__main__":
    check_missing_files()
