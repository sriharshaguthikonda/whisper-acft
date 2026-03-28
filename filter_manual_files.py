#!/usr/bin/env python3
"""
Filter manual_ files to find only those longer than 30 seconds
"""

import os
import json
from pathlib import Path

def get_audio_duration(audio_path):
    """Get audio file duration using librosa"""
    try:
        import librosa
        duration = librosa.get_duration(filename=audio_path)
        return duration
    except:
        return 0

def main():
    record_dir = Path("I:/Record_harsha")
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
    
    manual_files = []
    for file_path in record_dir.iterdir():
        if file_path.is_file() and file_path.name.lower().startswith('manual_'):
            if file_path.suffix.lower() in audio_extensions:
                duration = get_audio_duration(str(file_path))
                if duration > 30:  # Only files longer than 30 seconds
                    manual_files.append({
                        "path": str(file_path),
                        "name": file_path.name,
                        "duration": duration
                    })
    
    # Sort by duration (longest first)
    manual_files.sort(key=lambda x: x["duration"], reverse=True)
    
    print(f"Found {len(manual_files)} manual_ files longer than 30 seconds:")
    print(f"Total duration: {sum(f['duration'] for f in manual_files)/3600:.1f} hours")
    print()
    
    # Save file list for processing
    with open("I:/whisper-acft/long_manual_files.json", "w") as f:
        json.dump(manual_files, f, indent=2)
    
    # Create command line argument list
    file_paths = [f['path'] for f in manual_files]
    print("File paths for processing:")
    for path in file_paths[:10]:  # Show first 10
        print(f'  "{path}"')
    if len(file_paths) > 10:
        print(f"  ... and {len(file_paths) - 10} more files")
    
    return file_paths

if __name__ == "__main__":
    main()
