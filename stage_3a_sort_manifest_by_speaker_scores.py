import os
import json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# ----------------------------
# USER SETTINGS
# ----------------------------
SPEAKER_SCORES_CSV = "i:\\whisper-acft\\speaker_sort_scores.csv"
INPUT_MANIFEST = "I:\\Record_chunks\\pairs_manifest.jsonl"
OUTPUT_MANIFEST = "I:\\Record_chunks\\pairs_manifest_sorted_by_scores.jsonl"

# ----------------------------
# MAIN SCRIPT
# ----------------------------

def load_speaker_scores(csv_path):
    """Load speaker scores from CSV file into a dictionary"""
    print(f"Loading speaker scores from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Create a dictionary mapping file paths to scores
    score_dict = {}
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing scores"):
        file_path = row['file']
        score = row['score']
        
        # Skip rows where file_path is NaN or not a string
        if pd.isna(file_path) or not isinstance(file_path, str):
            continue
            
        score_dict[file_path] = score
    
    print(f"Loaded {len(score_dict)} file scores")
    return score_dict

def load_manifest(manifest_path):
    """Load manifest from JSONL file"""
    print(f"Loading manifest from {manifest_path}...")
    manifest_entries = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading manifest"), 1):
            try:
                entry = json.loads(line.strip())
                manifest_entries.append(entry)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(manifest_entries)} manifest entries")
    return manifest_entries

def sort_manifest_by_scores(manifest_entries, score_dict):
    """Sort manifest entries by speaker scores (highest first)"""
    print("Sorting manifest entries by speaker scores...")
    print(f"DEBUG: score_dict type: {type(score_dict)}")
    print(f"DEBUG: First item in score_dict: {next(iter(score_dict.items())) if score_dict else 'Empty'}")
    
    # Normalize score dictionary keys to lowercase for case-insensitive matching
    normalized_score_dict = {k.lower(): v for k, v in score_dict.items()}
    
    # Count entries with and without scores
    with_score = 0
    without_score = 0
    with_nan_score = 0
    
    # Create list of (entry, score) tuples
    entries_with_scores = []
    for entry in manifest_entries:
        audio_path = entry.get('audio_path', '').lower()  # Convert to lowercase for matching
        if audio_path in normalized_score_dict:
            score = normalized_score_dict[audio_path]
            # Handle NaN scores - put them at the end with very low score
            if pd.isna(score):
                entries_with_scores.append((entry, float('-inf')))
                with_nan_score += 1
            else:
                entries_with_scores.append((entry, score))
                with_score += 1
        else:
            # Put entries without scores at the end with score -inf-1
            entries_with_scores.append((entry, float('-inf') - 1))
            without_score += 1
    
    print(f"Entries with valid scores: {with_score}")
    print(f"Entries with NaN scores: {with_nan_score}")
    print(f"Entries without scores: {without_score}")
    
    # Sort by score (descending)
    entries_with_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Extract just the entries
    sorted_entries = [entry for entry, score in entries_with_scores]
    
    return sorted_entries

def save_sorted_manifest(sorted_entries, output_path):
    """Save sorted manifest to JSONL file"""
    print(f"Saving sorted manifest to {output_path}...")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in tqdm(sorted_entries, desc="Saving manifest"):
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(sorted_entries)} entries to {output_path}")

def main():
    print("=" * 60)
    print("STAGE 3a: Sorting Manifest by Speaker Scores")
    print("=" * 60)
    
    # Check input files exist
    if not os.path.exists(SPEAKER_SCORES_CSV):
        raise FileNotFoundError(f"Speaker scores CSV not found: {SPEAKER_SCORES_CSV}")
    
    if not os.path.exists(INPUT_MANIFEST):
        raise FileNotFoundError(f"Input manifest not found: {INPUT_MANIFEST}")
    
    # Load data
    score_dict = load_speaker_scores(SPEAKER_SCORES_CSV)
    manifest_entries = load_manifest(INPUT_MANIFEST)
    
    # Sort manifest
    sorted_entries = sort_manifest_by_scores(manifest_entries, score_dict)
    
    # Save sorted manifest
    save_sorted_manifest(sorted_entries, OUTPUT_MANIFEST)
    
    print("=" * 60)
    print("STAGE 3a COMPLETED SUCCESSFULLY")
    print(f"Sorted manifest saved to: {OUTPUT_MANIFEST}")
    print("=" * 60)

if __name__ == "__main__":
    main()
