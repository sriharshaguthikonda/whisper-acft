#!/usr/bin/env python3
"""
Script to move recordings from i:\Record to i:\Record_harsha based on keyword search results.
"""

import json
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse

def load_keyword_results(state_file_path):
    """Load the keyword search state file and extract matched file paths."""
    with open(state_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all JSON file paths from the results
    matched_json_paths = []
    for result in data.get('results', []):
        json_path = result.get('path')
        if json_path:
            matched_json_paths.append(json_path)
    
    return matched_json_paths

def json_to_audio_filename(json_path):
    """Convert JSON transcription filename to expected audio filename."""
    # Remove .json extension
    base_name = Path(json_path).stem
    
    # Common audio extensions to try
    audio_extensions = ['.mp3', '.wav', '.m4a', '.mp4', '.avi', '.mov', '.mkv']
    
    return base_name, audio_extensions

def find_audio_file(json_path, record_dir):
    """Find the corresponding audio file for a given JSON transcription file."""
    base_name, audio_extensions = json_to_audio_filename(json_path)
    
    # Try different audio extensions
    for ext in audio_extensions:
        audio_file = record_dir / f"{base_name}{ext}"
        if audio_file.exists():
            return audio_file
    
    # If not found with exact name, try case-insensitive search
    try:
        for file_path in record_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                if file_path.stem.lower() == base_name.lower():
                    return file_path
    except Exception:
        pass
    
    return None

def move_recordings(state_file, source_dir, target_dir, dry_run=False):
    """Move recordings based on keyword search results."""
    
    # Convert to Path objects
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    state_file = Path(state_file)
    
    # Create target directory if it doesn't exist
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # Load matched JSON files
    print("Loading keyword search results...")
    matched_json_paths = load_keyword_results(state_file)
    print(f"Found {len(matched_json_paths)} matched transcription files")
    
    # Track statistics
    moved_count = 0
    not_found_count = 0
    already_exists_count = 0
    
    # Process each matched JSON file
    print(f"\nSearching for audio files in: {source_dir}")
    print(f"Moving to: {target_dir}")
    
    for json_path in tqdm(matched_json_paths, desc="Processing files"):
        # Find corresponding audio file
        audio_file = find_audio_file(json_path, source_dir)
        
        if audio_file is None:
            print(f"⚠️  Audio file not found for: {Path(json_path).name}")
            not_found_count += 1
            continue
        
        # Check if file already exists in target
        target_file = target_dir / audio_file.name
        if target_file.exists():
            print(f"⚠️  File already exists in target: {audio_file.name}")
            already_exists_count += 1
            continue
        
        # Move the file
        if dry_run:
            print(f"🔄 Would move: {audio_file.name}")
        else:
            try:
                shutil.move(str(audio_file), str(target_file))
                print(f"✅ Moved: {audio_file.name}")
                moved_count += 1
            except Exception as e:
                print(f"❌ Error moving {audio_file.name}: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"SUMMARY ({'DRY RUN' if dry_run else 'ACTUAL'}):")
    print(f"Total matched transcriptions: {len(matched_json_paths)}")
    print(f"Files moved: {moved_count}")
    print(f"Files not found: {not_found_count}")
    print(f"Files already in target: {already_exists_count}")
    print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(
        description="Move recordings to Record_harsha based on keyword search results"
    )
    parser.add_argument(
        "--state-file",
        default=r"i:\P2GPT_google_drive\My Drive\Transcriptions\keyword_search_state.json",
        help="Path to keyword search state JSON file"
    )
    parser.add_argument(
        "--source-dir",
        default=r"i:\Record",
        help="Source directory containing recordings"
    )
    parser.add_argument(
        "--target-dir", 
        default=r"i:\Record_harsha",
        help="Target directory to move recordings to"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be moved without actually moving files"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.state_file).exists():
        print(f"❌ State file not found: {args.state_file}")
        return
    
    if not Path(args.source_dir).exists():
        print(f"❌ Source directory not found: {args.source_dir}")
        return
    
    print(f"🔍 Processing keyword search results...")
    print(f"📁 State file: {args.state_file}")
    print(f"📂 Source: {args.source_dir}")
    print(f"📂 Target: {args.target_dir}")
    
    move_recordings(
        args.state_file,
        args.source_dir, 
        args.target_dir,
        dry_run=args.dry_run
    )

if __name__ == "__main__":
    main()
