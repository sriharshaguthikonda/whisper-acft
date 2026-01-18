#!/usr/bin/env python3
"""Convert JSON transcripts to ASS files for audio files in Record_harsha folder.

This script:
1. Scans audio files in i:\Record_harsha
2. Finds corresponding JSON transcript files in i:\Transcriptions
3. Converts JSON files to ASS subtitle format
4. Saves ASS files in i:\Record_harsha alongside the audio files
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple
import tqdm

# Add the current directory to path to import from convert_json_transcripts_to_vtt_and_ass
sys.path.insert(0, str(Path(__file__).parent))

from convert_json_transcripts_to_vtt_and_ass import (
    extract_segments,
    lp_norm_bounds,
    segments_to_ass_text,
    load_trivial_patterns,
    should_keep_segment
)


def find_matching_files(record_dir: Path, transcriptions_dir: Path) -> List[Tuple[Path, Path]]:
    """Find matching audio-transcript pairs."""
    audio_files = list(record_dir.glob("*.m4a")) + list(record_dir.glob("*.mp3")) + list(record_dir.glob("*.wav"))
    matching_pairs = []
    
    for audio_file in audio_files:
        # Look for JSON with same name (case-insensitive)
        json_name = audio_file.stem + ".json"
        json_file = transcriptions_dir / json_name
        
        # Try case-insensitive search if exact match not found
        if not json_file.exists():
            for json_candidate in transcriptions_dir.glob("*.json"):
                if json_candidate.stem.lower() == audio_file.stem.lower():
                    json_file = json_candidate
                    break
        
        if json_file.exists():
            matching_pairs.append((audio_file, json_file))
        else:
            print(f"Warning: No transcript found for {audio_file.name}")
    
    return matching_pairs


def process_single_pair(
    audio_file: Path,
    json_file: Path,
    output_dir: Path,
    overwrite: bool,
    exact_trivial: set,
    regex_trivial: list,
    trivial_bad_score_max: int,
    trivial_good_score_min: int,
    keep_good_trivial_fraction: float,
    keep_seed: str
) -> Tuple[bool, str]:
    """Process a single audio-transcript pair."""
    try:
        ass_output = output_dir / (audio_file.stem + ".ass")
        
        # Skip if exists and not overwriting
        if ass_output.exists() and not overwrite:
            return True, f"Skipped (already exists): {audio_file.name}"
        
        # Load JSON transcript
        with json_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        
        # Extract segments
        segments_all = list(extract_segments(payload))
        if not segments_all:
            return False, f"No segments found in {json_file.name}"
        
        # Calculate normalization bounds
        lo, hi = lp_norm_bounds(segments_all)
        file_key = json_file.stem
        
        # Filter segments
        segments = [
            s for s in segments_all
            if should_keep_segment(
                s, lo, hi, file_key, exact_trivial, regex_trivial,
                trivial_bad_score_max, trivial_good_score_min,
                keep_good_trivial_fraction, keep_seed
            )
        ]
        
        if not segments:
            return False, f"No valid segments after filtering for {audio_file.name}"
        
        # Generate ASS content
        ass_content = segments_to_ass_text(
            segments, lo, hi, f"Transcript for {audio_file.stem}"
        )
        
        # Write ASS file
        ass_output.write_text(ass_content, encoding="utf-8")
        
        return True, f"Created: {audio_file.name} -> {ass_output.name}"
        
    except Exception as e:
        return False, f"Error processing {audio_file.name}: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON transcripts to ASS files for Record_harsha audio files"
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=Path(r"i:\Record_harsha"),
        help="Directory containing audio files"
    )
    parser.add_argument(
        "--transcriptions-dir",
        type=Path,
        default=Path(r"i:\Transcriptions"),
        help="Directory containing JSON transcript files"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ASS files"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of worker threads"
    )
    parser.add_argument(
        "--trivial-bad-score-max",
        type=int,
        default=35,
        help="Drop ALL trivial/junk cues with score <= this value"
    )
    parser.add_argument(
        "--trivial-good-score-min",
        type=int,
        default=80,
        help="Treat trivial/junk cues with score >= this as 'good'"
    )
    parser.add_argument(
        "--keep-good-trivial-fraction",
        type=float,
        default=0.10,
        help="Keep only this fraction of good trivial/junk cues"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without creating files"
    )
    
    args = parser.parse_args()
    
    # Validate directories
    if not args.record_dir.exists():
        print(f"Error: Record directory not found: {args.record_dir}")
        sys.exit(1)
    
    if not args.transcriptions_dir.exists():
        print(f"Error: Transcriptions directory not found: {args.transcriptions_dir}")
        sys.exit(1)
    
    # Load trivial patterns
    exact_trivial, regex_trivial = load_trivial_patterns(None, False)
    
    # Find matching files
    print("Scanning for matching audio-transcript pairs...")
    matching_pairs = find_matching_files(args.record_dir, args.transcriptions_dir)
    
    if not matching_pairs:
        print("No matching audio-transcript pairs found!")
        sys.exit(1)
    
    print(f"Found {len(matching_pairs)} matching pairs")
    
    if args.dry_run:
        print("Dry run - files that would be processed:")
        for audio_file, json_file in matching_pairs:
            print(f"  {audio_file.name} <- {json_file.name}")
        return
    
    # Process files
    created = 0
    failed = 0
    skipped = 0
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_pair = {
            executor.submit(
                process_single_pair,
                audio_file,
                json_file,
                args.record_dir,  # Output to same directory as audio
                args.overwrite,
                exact_trivial,
                regex_trivial,
                args.trivial_bad_score_max,
                args.trivial_good_score_min,
                args.keep_good_trivial_fraction,
                "v1"  # keep_seed
            ): (audio_file, json_file)
            for audio_file, json_file in matching_pairs
        }
        
        with tqdm.tqdm(total=len(matching_pairs), desc="Converting to ASS", unit="file") as pbar:
            for future in as_completed(future_to_pair):
                audio_file, json_file = future_to_pair[future]
                try:
                    success, message = future.result()
                    if success:
                        if "Skipped" in message:
                            skipped += 1
                        else:
                            created += 1
                        tqdm.tqdm.write(f"✓ {message}")
                    else:
                        failed += 1
                        tqdm.tqdm.write(f"✗ {message}")
                except Exception as e:
                    failed += 1
                    tqdm.tqdm.write(f"✗ Unexpected error for {audio_file.name}: {e}")
                pbar.update(1)
    
    print(f"\nDone! Created: {created}, Skipped: {skipped}, Failed: {failed}")
    
    # Add completion beep
    print("\a")  # Bell character


if __name__ == "__main__":
    main()
