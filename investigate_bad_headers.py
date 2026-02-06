#!/usr/bin/env python3
"""Investigate bad_header files from stage 13 training"""

import soundfile as sf
from pathlib import Path
import json

def investigate_file(file_path):
    """Check what's wrong with a specific audio file"""
    result = {
        "path": str(file_path),
        "exists": file_path.exists(),
        "readable": False,
        "samplerate": None,
        "duration": None,
        "frames": None,
        "error": None
    }
    
    if not file_path.exists():
        result["error"] = "File does not exist"
        return result
    
    try:
        info = sf.info(str(file_path))
        result["readable"] = True
        result["samplerate"] = info.samplerate
        result["frames"] = info.frames
        result["duration"] = float(info.frames) / float(info.samplerate)
        
        # Check specific issues
        issues = []
        if info.samplerate != 16000:
            issues.append(f"Wrong sample rate: {info.samplerate} (expected 16000)")
        
        dur = result["duration"]
        if dur <= 0.0:
            issues.append(f"Non-positive duration: {dur}")
        elif dur > 30.0:
            issues.append(f"Duration too long: {dur:.2f}s (max 30.0s)")
        
        if issues:
            result["error"] = "; ".join(issues)
        
    except Exception as e:
        result["error"] = f"SoundFile error: {str(e)}"
    
    return result

def main():
    # Read the train manifest to find files that might be problematic
    manifest_path = "I:/Record_chunks/train_manifest.jsonl"
    
    print("Scanning manifest for potentially problematic files...")
    
    bad_files = []
    total_files = 0
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                row = json.loads(line)
                total_files += 1
                
                audio_path = row.get("audio_path")
                if not audio_path:
                    continue
                
                file_path = Path(audio_path)
                result = investigate_file(file_path)
                
                # Check if this would be flagged as bad_header
                if (not result["exists"] or 
                    not result["readable"] or 
                    result["samplerate"] != 16000 or 
                    result["duration"] <= 0.0 or 
                    result["duration"] > 29.0):
                    
                    bad_files.append(result)
                    if len(bad_files) <= 50:  # Show first 50
                        print(f"\n{len(bad_files)}. {audio_path}")
                        print(f"   Exists: {result['exists']}")
                        print(f"   Readable: {result['readable']}")
                        print(f"   Sample rate: {result['samplerate']}")
                        print(f"   Duration: {result['duration']:.3f}s" if result['duration'] else "   Duration: N/A")
                        print(f"   Issue: {result['error']}")
                
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
    
    print(f"\n" + "="*60)
    print(f"SUMMARY:")
    print(f"Total files scanned: {total_files}")
    print(f"Problematic files found: {len(bad_files)}")
    
    if len(bad_files) > 50:
        print(f"(Showing first 50, {len(bad_files)-50} more not shown)")
    
    # Categorize issues
    issues_by_type = {}
    for bad_file in bad_files:
        error = bad_file.get("error", "Unknown")
        if error not in issues_by_type:
            issues_by_type[error] = 0
        issues_by_type[error] += 1
    
    print(f"\nIssues by type:")
    for error, count in sorted(issues_by_type.items(), key=lambda x: x[1], reverse=True):
        print(f"  {count}: {error}")

if __name__ == "__main__":
    main()
