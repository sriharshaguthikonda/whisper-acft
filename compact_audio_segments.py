#!/usr/bin/env python3
"""
Remove silent segments from audio files using transcription timestamps.
Compacts audio by extracting only the segments where speech is present.
"""

import json
import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import subprocess
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_transcription(transcription_path: Path) -> Dict:
    """Load transcription JSON file."""
    try:
        with open(transcription_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading transcription {transcription_path}: {e}")
        return None

def find_matching_transcription(audio_path: Path, transcription_dir: Path) -> Optional[Path]:
    """
    Find matching transcription file for an audio file.
    Handles various naming patterns:
    - Exact match: audio.wav -> audio.json
    - With suffix: audio_sent0128.wav -> audio.json
    - With chunk suffix: audio_chunk0000.wav -> audio.json
    - With other suffixes: audio__L.wav -> audio.json
    """
    audio_stem = audio_path.stem
    
    # Try exact match first
    exact_match = transcription_dir / f"{audio_stem}.json"
    if exact_match.exists():
        return exact_match
    
    # Try removing common suffixes
    patterns_to_remove = [
        r'_sent\d+$',           # _sent0128
        r'_chunk\d+$',          # _chunk0000
        r'__L$',                # __L
        r'_Trim\d+$',           # _Trim1
        r'_\w+_\w+_\w+_\w+_\w+_\w+$',  # __c3231fec39_chunk0000
    ]
    
    for pattern in patterns_to_remove:
        cleaned_stem = re.sub(pattern, '', audio_stem)
        if cleaned_stem != audio_stem:
            match_path = transcription_dir / f"{cleaned_stem}.json"
            if match_path.exists():
                logger.info(f"Matched {audio_path.name} -> {match_path.name}")
                return match_path
    
    # Try fuzzy matching - find files that start with the base name
    base_name = re.split(r'[_-]', audio_stem)[0]
    for json_file in transcription_dir.glob("*.json"):
        json_stem = json_file.stem
        if json_stem.startswith(base_name):
            logger.info(f"Fuzzy matched {audio_path.name} -> {json_file.name}")
            return json_file
    
    return None

def get_speech_segments(transcription_data: Dict) -> List[Tuple[float, float]]:
    """Extract speech segments from transcription data."""
    segments = []
    
    # Handle different possible structures
    if 'groq_response' in transcription_data and 'segments' in transcription_data['groq_response']:
        segment_data = transcription_data['groq_response']['segments']
    elif 'segments' in transcription_data:
        segment_data = transcription_data['segments']
    else:
        logger.warning("No segments found in transcription data")
        return segments
    
    for segment in segment_data:
        start_time = segment['start']
        end_time = segment['end']
        # Only include segments with actual speech (skip very short silence segments)
        if end_time - start_time > 0.1:  # Minimum 100ms to avoid tiny fragments
            segments.append((start_time, end_time))
    
    return segments

def merge_close_segments(segments: List[Tuple[float, float]], gap_threshold: float = 0.5) -> List[Tuple[float, float]]:
    """Merge segments that are close to each other to avoid too many cuts."""
    if not segments:
        return segments
    
    merged = [segments[0]]
    
    for current in segments[1:]:
        last = merged[-1]
        # If gap between segments is small, merge them
        if current[0] - last[1] <= gap_threshold:
            merged[-1] = (last[0], current[1])
        else:
            merged.append(current)
    
    return merged

def create_compact_audio(input_audio: Path, output_audio: Path, segments: List[Tuple[float, float]]) -> bool:
    """Create compacted audio by extracting speech segments."""
    if not segments:
        logger.warning(f"No speech segments found for {input_audio}")
        return False
    
    # Create a temporary file for the concatenated segments
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as concat_file:
        concat_list_path = concat_file.name
        
        # Write FFmpeg concat demuxer file
        for i, (start, end) in enumerate(segments):
            concat_file.write(f"file '{input_audio}'\n")
            concat_file.write(f"inpoint {start:.3f}\n")
            concat_file.write(f"outpoint {end:.3f}\n")
    
    try:
        # Use FFmpeg to concatenate segments
        cmd = [
            'ffmpeg', '-y',  # Overwrite output file
            '-f', 'concat',  # Use concat demuxer
            '-safe', '0',    # Allow unsafe file paths
            '-i', concat_list_path,
            '-c', 'copy',    # Copy codec without re-encoding
            str(output_audio)
        ]
        
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully created compacted audio: {output_audio}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error for {input_audio}: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("FFmpeg not found. Please install FFmpeg and add it to PATH.")
        return False
    finally:
        # Clean up temporary file
        try:
            os.unlink(concat_list_path)
        except OSError:
            pass

def process_audio_file(audio_path: Path, transcription_dir: Path, output_dir: Path, 
                      gap_threshold: float = 0.5) -> Tuple[bool, str]:
    """Process a single audio file."""
    # Find corresponding transcription file with enhanced matching
    transcription_path = find_matching_transcription(audio_path, transcription_dir)
    
    if not transcription_path:
        return False, f"No matching transcription file found for: {audio_path.name}"
    
    # Load transcription
    transcription_data = load_transcription(transcription_path)
    if not transcription_data:
        return False, f"Failed to load transcription: {transcription_path}"
    
    # Get speech segments
    segments = get_speech_segments(transcription_data)
    if not segments:
        return False, f"No speech segments found in: {transcription_path}"
    
    # Merge close segments
    merged_segments = merge_close_segments(segments, gap_threshold)
    
    # Create output path
    output_path = output_dir / f"{audio_path.stem}_compact{audio_path.suffix}"
    
    # Create compacted audio
    success = create_compact_audio(audio_path, output_path, merged_segments)
    
    if success:
        # Calculate time saved
        original_duration = sum(end - start for start, end in segments)
        total_duration = merged_segments[-1][1] - merged_segments[0][1] if merged_segments else 0
        time_saved = total_duration - original_duration
        
        return True, f"Success - Saved {time_saved:.1f}s of silence"
    else:
        return False, "Failed to create compacted audio"

def main():
    parser = argparse.ArgumentParser(description="Remove silent segments from audio files using transcriptions")
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio files")
    parser.add_argument("--transcription-dir", required=True, help="Directory containing transcription JSON files")
    parser.add_argument("--output-dir", required=True, help="Directory to save compacted audio files")
    parser.add_argument("--gap-threshold", type=float, default=0.5, 
                       help="Maximum gap between segments to merge (default: 0.5 seconds)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--audio-ext", default=".m4a", help="Audio file extension (default: .m4a)")
    
    args = parser.parse_args()
    
    # Convert to Path objects
    audio_dir = Path(args.audio_dir)
    transcription_dir = Path(args.transcription_dir)
    output_dir = Path(args.output_dir)
    
    # Validate directories
    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        sys.exit(1)
    
    if not transcription_dir.exists():
        logger.error(f"Transcription directory not found: {transcription_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all audio files
    audio_files = list(audio_dir.glob(f"*{args.audio_ext}"))
    if not audio_files:
        logger.error(f"No audio files found with extension {args.audio_ext} in {audio_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Process files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_audio_file, audio_file, transcription_dir, output_dir, args.gap_threshold): audio_file
            for audio_file in audio_files
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_file), total=len(audio_files), desc="Processing audio files"):
            audio_file = future_to_file[future]
            try:
                success, message = future.result()
                results.append((audio_file.name, success, message))
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {e}")
                results.append((audio_file.name, False, str(e)))
    
    # Print summary
    logger.info("\n=== Processing Summary ===")
    successful = sum(1 for _, success, _ in results if success)
    logger.info(f"Successfully processed: {successful}/{len(results)} files")
    
    if successful < len(results):
        logger.info("\nFailed files:")
        for filename, success, message in results:
            if not success:
                logger.info(f"  {filename}: {message}")
    
    logger.info(f"\nCompacted audio files saved to: {output_dir}")
    
    # Add beep to notify completion
    print('\a')

if __name__ == "__main__":
    main()
