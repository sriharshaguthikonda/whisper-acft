#!/usr/bin/env python3
"""Debug script to test stage 10b processing on a single file"""

import json
import sys
from pathlib import Path

# Add the current directory to Python path to import from the main script
sys.path.insert(0, r"I:\whisper-acft")

from stage_10_b_add_speech_tempo_pause_aware_idempotent import (
    process_one, SQLiteSeenSet, Path
)

def main():
    # Load a sample row from the manifest
    with open(r"I:\Record_chunks\pairs_manifest_stereo_english_only_filtered_with_uids_score_bottom_filtered.jsonl", "r") as f:
        first_line = f.readline().strip()
        row = json.loads(first_line)
    
    print(f"Processing row: {row.get('audio_path')}")
    print(f"Base UID: {row.get('base_uid')}")
    
    # Create a mock args object
    class Args:
        def __init__(self):
            self.sox = "sox"
            self.ffmpeg = "ffmpeg"
            self.sample_rate = 16000
            self.channels = 1
            self.bit_depth = 16
            self.tempo_min = 1.05
            self.tempo_max = 1.20
            self.tempo_factors = "1.05,1.07,1.09,1.10,1.12,1.14,1.16,1.18,1.20"
            self.mode = "choice"
            self.pause_policy = "truncate"
            self.silence_factor = 2.8
            self.silence_noise_db = -35
            self.silence_min_dur = 0.15
            self.edge_pad_sec = 0.1
            self.min_segment_sec = 0.1
            self.min_effect_sec = 0.1
            self.drop_silence_below_sec = 0.05
            self.dry_run = False
    
    args = Args()
    
    # Create output directory
    out_dir = Path(r"I:\Record_chunks_debug_test")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create seen database
    seen = SQLiteSeenSet(r"I:\Record_chunks\seen_debug.sqlite")
    
    # Process the row
    success, out_row, status = process_one(
        row, "tempo_speech_pause", 1, out_dir, args, seen
    )
    
    print(f"Success: {success}")
    print(f"Status: {status}")
    if out_row:
        print(f"Output audio: {out_row.get('audio_path')}")
    else:
        print("No output row generated")
    
    seen.close()

if __name__ == "__main__":
    main()
