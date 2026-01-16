import pandas as pd
import json

# Load speaker scores to identify "OTHER" category files
df = pd.read_csv('i:\\whisper-acft\\speaker_sort_scores.csv')
other_files = set()
score_files = set()

for _, row in df.iterrows():
    file_path = row['file']
    decision = row['decision']
    
    if pd.notna(file_path) and isinstance(file_path, str):
        if pd.notna(decision) and decision == 'OTHER':
            other_files.add(file_path.lower())
        else:
            score_files.add(file_path.lower())

print(f"Files marked as 'OTHER' in CSV: {len(other_files)}")
print(f"Files with scores (not 'OTHER'): {len(score_files)}")

# Check positions in sorted manifest
other_positions = []
score_positions = []
nan_positions = []

with open('I:\\Record_chunks\\pairs_manifest_sorted_by_scores.jsonl', 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        entry = json.loads(line)
        audio_path = entry['audio_path'].lower()
        
        if audio_path in other_files:
            other_positions.append(line_num)
        elif audio_path in score_files:
            score_positions.append(line_num)
        else:
            nan_positions.append(line_num)

print(f"\nPOSITIONS IN SORTED MANIFEST:")
print(f"Total manifest entries: {line_num}")
print(f"'OTHER' category files: {len(other_positions)} positions")
print(f"Scored files: {len(score_positions)} positions")  
print(f"Files not in CSV (NaN): {len(nan_positions)} positions")

print(f"\n'OTHER' CATEGORY FILE POSITIONS:")
if other_positions:
    print(f"First 'OTHER' file at position: {min(other_positions)}")
    print(f"Last 'OTHER' file at position: {max(other_positions)}")
    print(f"First 10 'OTHER' positions: {other_positions[:10]}")
    print(f"Last 10 'OTHER' positions: {other_positions[-10:]}")
    
    # Check if they're clustered at the end
    total_entries = line_num
    end_threshold = total_entries - 1000  # Last 1000 entries
    others_at_end = sum(1 for pos in other_positions if pos >= end_threshold)
    print(f"'OTHER' files in last 1000 entries: {others_at_end}/{len(other_positions)} ({100*others_at_end/len(other_positions):.1f}%)")
else:
    print("No 'OTHER' category files found in manifest!")

print(f"\nSCORED FILE POSITIONS:")
if score_positions:
    print(f"First scored file at position: {min(score_positions)}")
    print(f"Last scored file at position: {max(score_positions)}")
    print(f"First 10 scored positions: {score_positions[:10]}")
else:
    print("No scored files found in manifest!")

print(f"\nNaN FILE POSITIONS:")
if nan_positions:
    print(f"First NaN file at position: {min(nan_positions)}")
    print(f"Last NaN file at position: {max(nan_positions)}")
    print(f"First 10 NaN positions: {nan_positions[:10]}")
else:
    print("No NaN files found in manifest!")

# Show actual examples from different positions
print(f"\nSAMPLE ENTRIES FROM DIFFERENT POSITIONS:")
print("-" * 50)

with open('I:\\Record_chunks\\pairs_manifest_sorted_by_scores.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
    # Show first 5 entries (should be highest scores)
    print("FIRST 5 ENTRIES (highest scores):")
    for i in range(min(5, len(lines))):
        entry = json.loads(lines[i])
        audio_path = entry['audio_path'].lower()
        category = "OTHER" if audio_path in other_files else ("SCORED" if audio_path in score_files else "NaN")
        print(f"  {i+1}: {audio_path} - {category}")
    
    # Show entries around the transition
    if other_positions and score_positions:
        transition_point = max(score_positions)  # Last scored file
        print(f"\nENTRIES AROUND TRANSITION (around position {transition_point}):")
        start = max(0, transition_point - 3)
        end = min(len(lines), transition_point + 7)
        for i in range(start, end):
            entry = json.loads(lines[i])
            audio_path = entry['audio_path'].lower()
            category = "OTHER" if audio_path in other_files else ("SCORED" if audio_path in score_files else "NaN")
            marker = " <-- LAST SCORED" if i == transition_point else ""
            print(f"  {i+1}: {audio_path} - {category}{marker}")
    
    # Show last 5 entries (should be NaN/OTHER)
    print(f"\nLAST 5 ENTRIES (should be NaN/OTHER):")
    for i in range(max(0, len(lines)-5), len(lines)):
        entry = json.loads(lines[i])
        audio_path = entry['audio_path'].lower()
        category = "OTHER" if audio_path in other_files else ("SCORED" if audio_path in score_files else "NaN")
        print(f"  {i+1}: {audio_path} - {category}")
