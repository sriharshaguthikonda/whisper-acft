#!/usr/bin/env python3
"""Fix bad header silent chunks by resampling them to 16000 Hz"""

import os
import json
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import numpy as np

def resample_audio(audio_data, orig_sr, target_sr):
    """Resample audio data from orig_sr to target_sr"""
    try:
        from scipy.signal import resample_poly
        from math import gcd
        
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        resampled = resample_poly(audio_data, up, down).astype(np.float32)
        return resampled
    except ImportError:
        print("Warning: scipy not available, cannot resample")
        return None

def fix_file(file_path):
    """Fix a single audio file by resampling to 16000 Hz if needed"""
    try:
        # Read the file
        audio_data, sr = sf.read(file_path, dtype="float32", always_2d=False)
        
        # Convert to mono if stereo
        if audio_data.ndim == 2:
            audio_data = audio_data.mean(axis=-1)
        
        # Check if resampling is needed
        if sr != 16000:
            print(f"Resampling {Path(file_path).name} from {sr} Hz to 16000 Hz")
            resampled = resample_audio(audio_data, sr, 16000)
            if resampled is None:
                return False, "scipy not available"
            
            # Save the resampled audio
            sf.write(file_path, resampled, 16000, subtype="PCM_16")
            return True, f"resampled from {sr} Hz"
        else:
            return True, "already correct"
            
    except Exception as e:
        return False, str(e)

def main():
    manifest_path = "I:/Record_chunks/train_manifest.jsonl"
    trained_path = "i:/Record_chunks/trained_stage1.jsonl"
    
    print("=" * 80)
    print("FIXING BAD HEADER SILENT CHUNKS")
    print("=" * 80)
    
    # Load trained files to skip them
    trained_files = set()
    if os.path.exists(trained_path):
        with open(trained_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        ap = row.get("audio_path")
                        if ap:
                            trained_files.add(ap)
                    except:
                        pass
    
    print(f"Already trained files: {len(trained_files)}")
    
    # Find bad header files (silent chunks with wrong sample rate)
    bad_header_files = []
    
    print(f"\nScanning manifest for bad header silent chunks...")
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                row = json.loads(line)
                audio_path = row.get("audio_path")
                if not audio_path:
                    continue
                
                # Skip already trained files
                if audio_path in trained_files:
                    continue
                
                # Only process silent chunks
                if not row.get("is_silent", False):
                    continue
                
                # Check if file exists
                if not os.path.exists(audio_path):
                    continue
                
                # Check if it's a bad header file
                try:
                    info = sf.info(audio_path)
                    if info.samplerate != 16000:
                        bad_header_files.append(audio_path)
                    elif float(info.frames) / float(info.samplerate) > 30.0:
                        bad_header_files.append(audio_path)
                except:
                    bad_header_files.append(audio_path)
                    
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
    
    print(f"Found {len(bad_header_files)} bad header silent chunks to fix")
    
    if not bad_header_files:
        print("No bad header files found!")
        return
    
    # Fix the files
    fixed = 0
    failed = 0
    
    print(f"\nFixing files...")
    for file_path in tqdm(bad_header_files, desc="Fixing files"):
        success, message = fix_file(file_path)
        if success:
            fixed += 1
        else:
            failed += 1
            print(f"Failed to fix {file_path}: {message}")
    
    print(f"\n" + "=" * 80)
    print("FIXING SUMMARY:")
    print(f"Files to fix: {len(bad_header_files)}")
    print(f"Successfully fixed: {fixed}")
    print(f"Failed to fix: {failed}")
    
    if failed == 0:
        print("\nAll bad header silent chunks have been fixed!")
        print("You can now re-run stage 13 training.")
    else:
        print(f"\n{failed} files could not be fixed. Check the errors above.")

if __name__ == "__main__":
    main()
