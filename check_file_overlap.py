import pandas as pd
import json

# Load speaker scores
df = pd.read_csv('i:\\whisper-acft\\speaker_sort_scores.csv')
score_files = set()
for _, row in df.iterrows():
    file_path = row['file']
    if pd.notna(file_path) and isinstance(file_path, str):
        score_files.add(file_path.lower())

print(f"Files in speaker scores CSV: {len(score_files)}")
print("Sample files from scores CSV:")
for i, f in enumerate(list(score_files)[:5]):
    print(f"  {f}")

# Load bad quality manifest
manifest_files = set()
with open('I:\\Record_chunks_bad_quality\\pairs_manifest_bad_quality.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line)
        audio_path = entry['audio_path'].lower()
        manifest_files.add(audio_path)

print(f"\nFiles in bad quality manifest: {len(manifest_files)}")
print("Sample files from manifest:")
for i, f in enumerate(list(manifest_files)[:5]):
    print(f"  {f}")

# Check overlap
overlap = score_files.intersection(manifest_files)
print(f"\nOverlap: {len(overlap)} files")
if overlap:
    print("Sample overlapping files:")
    for i, f in enumerate(list(overlap)[:5]):
        print(f"  {f}")
else:
    print("No overlapping files found!")

# Check if there are any similar files
print("\nChecking for similar file patterns...")
score_prefixes = set()
for f in score_files:
    parts = f.split('\\')
    if len(parts) > 2:
        score_prefixes.add(parts[-2].split('_')[0])

manifest_prefixes = set()
for f in manifest_files:
    parts = f.split('\\')
    if len(parts) > 2:
        manifest_prefixes.add(parts[-2].split('_')[0])

print(f"Score file prefixes: {sorted(list(score_prefixes))[:10]}")
print(f"Manifest file prefixes: {sorted(list(manifest_prefixes))[:10]}")
