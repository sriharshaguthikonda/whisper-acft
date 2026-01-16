import pandas as pd
import json

# Load speaker scores
df = pd.read_csv('i:\\whisper-acft\\speaker_sort_scores.csv')
score_dict = dict(zip(df['file'], df['score']))

print("First 10 entries from CSV with scores:")
for i in range(min(10, len(df))):
    file_path = df.iloc[i]['file']
    score = df.iloc[i]['score']
    print(f"{file_path}: {score}")

print("\n" + "="*60)
print("First 10 entries from sorted manifest with their scores:")
print("="*60)

with open('I:\\Record_chunks\\pairs_manifest_sorted_by_scores.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 10:
            break
        entry = json.loads(line)
        audio_path = entry['audio_path']
        score = score_dict.get(audio_path, 'NO_SCORE')
        print(f"{audio_path}: {score}")

print("\n" + "="*60)
print("Checking if any manifest entries have scores:")
print("="*60)

found_with_score = 0
found_without_score = 0
sample_with_scores = []

with open('I:\\Record_chunks\\pairs_manifest_sorted_by_scores.jsonl', 'r') as f:
    for i, line in enumerate(f):
        if i >= 100:  # Check first 100 entries
            break
        entry = json.loads(line)
        audio_path = entry['audio_path']
        if audio_path in score_dict:
            found_with_score += 1
            if len(sample_with_scores) < 5:
                sample_with_scores.append((audio_path, score_dict[audio_path]))
        else:
            found_without_score += 1

print(f"In first 100 manifest entries:")
print(f"With scores: {found_with_score}")
print(f"Without scores: {found_without_score}")

if sample_with_scores:
    print("\nSample entries that have scores:")
    for audio_path, score in sample_with_scores:
        print(f"  {audio_path}: {score}")

# Check path format differences
print("\n" + "="*60)
print("Checking path format differences:")
print("="*60)

# Get a sample from each
sample_csv_path = df.iloc[0]['file']
with open('I:\\Record_chunks\\pairs_manifest_sorted_by_scores.jsonl', 'r') as f:
    first_entry = json.loads(f.readline())
    sample_manifest_path = first_entry['audio_path']

print(f"Sample CSV path: {sample_csv_path}")
print(f"Sample manifest path: {sample_manifest_path}")
print(f"Are they equal? {sample_csv_path == sample_manifest_path}")
