#!/usr/bin/env python3
"""
Script to delete WAV files that are no longer present in the JSONL manifest file.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_audio_paths_from_jsonl(jsonl_path):
    """Extract all audio paths from JSONL file."""
    audio_paths = set()
    
    print(f"Reading JSONL file: {jsonl_path}")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Processing JSONL lines"), 1):
            try:
                data = json.loads(line.strip())
                audio_path = data.get('audio_path')
                if audio_path:
                    # Normalize path for case-insensitive comparison
                    normalized_path = str(Path(audio_path)).lower()
                    audio_paths.add(normalized_path)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")
                continue
    
    print(f"Found {len(audio_paths)} audio paths in JSONL")
    return audio_paths

def get_wav_files_in_directory(directory):
    """Get all WAV files in directory."""
    wav_files = set()
    
    print(f"Scanning directory for WAV files: {directory}")
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory {directory} does not exist")
        return wav_files
    
    wav_files = {str(f).lower() for f in directory_path.rglob("*.wav")}
    print(f"Found {len(wav_files)} WAV files in directory")
    
    return wav_files

def delete_orphaned_files(orphaned_files, dry_run=True):
    """Delete orphaned files."""
    if not orphaned_files:
        print("No orphaned files to delete.")
        return
    
    print(f"\n{'DRY RUN - ' if dry_run else ''}Found {len(orphaned_files)} orphaned files to delete")
    
    # Show only first 10 files as examples
    print("Example files to be deleted:")
    for file_path in sorted(orphaned_files)[:10]:
        print(f"  {file_path}")
    
    if len(orphaned_files) > 10:
        print(f"  ... and {len(orphaned_files) - 10} more files")
    
    if dry_run:
        print("\nDRY RUN MODE - No files were actually deleted.")
        print("Run again with --execute to actually delete the files.")
        return
    
    print(f"\nDeleting {len(orphaned_files)} orphaned files...")
    deleted_count = 0
    error_count = 0
    
    for file_path in tqdm(orphaned_files, desc="Deleting files"):
        try:
            os.remove(file_path)
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
            error_count += 1
    
    print(f"Successfully deleted: {deleted_count} files")
    if error_count > 0:
        print(f"Errors encountered: {error_count} files")

def main():
    parser = argparse.ArgumentParser(description="Delete WAV files not present in JSONL manifest")
    parser.add_argument("--jsonl", required=True, help="Path to JSONL manifest file")
    parser.add_argument("--audio-dir", required=True, help="Directory containing WAV files")
    parser.add_argument("--execute", action="store_true", help="Actually delete files (default is dry run)")
    parser.add_argument("--show-kept", action="store_true", help="Show files that will be kept")
    
    args = parser.parse_args()
    
    # Extract audio paths from JSONL
    jsonl_paths = extract_audio_paths_from_jsonl(args.jsonl)
    
    # Get all WAV files in directory (both lowercase for comparison and original for deletion)
    directory_path = Path(args.audio_dir)
    all_wav_files = list(directory_path.rglob("*.wav"))
    wav_files_lower = {str(f).lower() for f in all_wav_files}
    
    print(f"Found {len(wav_files_lower)} WAV files in directory")
    
    # Find orphaned files (WAV files not in JSONL)
    orphaned_files_lower = wav_files_lower - jsonl_paths
    
    # Convert back to original case paths for deletion
    orphaned_files_original = []
    lower_to_original = {str(f).lower(): str(f) for f in all_wav_files}
    for lower_path in orphaned_files_lower:
        orphaned_files_original.append(lower_to_original[lower_path])
    
    # Show kept files if requested
    if args.show_kept:
        kept_files_lower = wav_files_lower & jsonl_paths
        kept_files_original = [lower_to_original[lower_path] for lower_path in kept_files_lower]
        print(f"\nFiles that will be kept ({len(kept_files_original)}):")
        for file_path in sorted(kept_files_original):
            print(f"  {file_path}")
    
    # Delete orphaned files
    delete_orphaned_files(orphaned_files_original, dry_run=not args.execute)

if __name__ == "__main__":
    main()
