import pandas as pd
import json

# Load speaker scores
df = pd.read_csv('i:\\whisper-acft\\speaker_sort_scores.csv')
score_dict = {}
for _, row in df.iterrows():
    file_path = row['file']
    if pd.notna(file_path) and isinstance(file_path, str):
        score_dict[file_path.lower()] = row['score']

print("VERIFICATION OF SORTED MANIFEST")
print("=" * 60)

# Check first 10 entries from sorted manifest
with open('I:\\Record_chunks\\pairs_manifest_sorted_by_scores.jsonl', 'r', encoding='utf-8') as f:
    print("FIRST 10 ENTRIES (should have highest scores):")
    print("-" * 40)
    for i, line in enumerate(f):
        if i >= 10:
            break
        entry = json.loads(line)
        audio_path = entry['audio_path'].lower()
        score = score_dict.get(audio_path, 'NO_SCORE')
        print(f"{i+1:2d}. {audio_path}")
        print(f"    Score: {score}")
        print()

# Check last 5 entries from sorted manifest
print("\nLAST 5 ENTRIES (should have lowest scores):")
print("-" * 40)
lines = []
with open('I:\\Record_chunks\\pairs_manifest_sorted_by_scores.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for i, line in enumerate(lines[-5:]):
    entry = json.loads(line)
    audio_path = entry['audio_path'].lower()
    score = score_dict.get(audio_path, 'NO_SCORE')
    line_num = len(lines) - 5 + i + 1
    print(f"{line_num:5d}. {audio_path}")
    print(f"        Score: {score}")
    print()

# Get top 10 scores from CSV
print("\nTOP 10 SCORES FROM CSV (for comparison):")
print("-" * 40)
sorted_scores = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
for i, (file_path, score) in enumerate(sorted_scores[:10]):
    print(f"{i+1:2d}. {file_path}")
    print(f"    Score: {score}")
    print()

# Check if sorting is correct
print("\nSORTING VERIFICATION:")
print("-" * 40)
manifest_scores = []
with open('I:\\Record_chunks\\pairs_manifest_sorted_by_scores.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        audio_path = entry['audio_path'].lower()
        score = score_dict.get(audio_path, float('-inf'))
        manifest_scores.append(score)

# Check if scores are in descending order (excluding NaN values)
valid_scores = [s for s in manifest_scores if not pd.isna(s) and s != float('-inf')]
is_sorted = all(valid_scores[i] >= valid_scores[i+1] for i in range(len(valid_scores)-1))

print(f"Total entries in manifest: {len(manifest_scores)}")
print(f"Entries with valid scores: {len(valid_scores)}")
print(f"Entries with NaN scores: {len([s for s in manifest_scores if pd.isna(s)])}")
print(f"Entries without scores: {len([s for s in manifest_scores if s == float('-inf')])}")
print(f"Scores are correctly sorted (descending): {is_sorted}")

if valid_scores:
    print(f"Highest score: {max(valid_scores)}")
    print(f"Lowest score: {min(valid_scores)}")
    
# Show first few valid scores to confirm descending order
print(f"\nFirst 10 valid scores in manifest:")
for i, score in enumerate(valid_scores[:10]):
    print(f"  {i+1}: {score}")
