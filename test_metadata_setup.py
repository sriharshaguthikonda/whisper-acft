#!/usr/bin/env python3
"""
Quick test script to check audio metadata functionality
"""

import subprocess
import sys

def check_dependencies():
    """Check if required dependencies are available"""
    print("🔍 Checking dependencies...")
    
    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        print("✅ ffmpeg is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ ffmpeg not found")
        print("📥 Please install ffmpeg:")
        print("   Windows: choco install ffmpeg")
        print("   Or download from: https://ffmpeg.org/download.html")
        return False

def main():
    print("🎵 Audio Metadata Updater - Dependency Check")
    print("="*50)
    
    if check_dependencies():
        print("\n✅ All dependencies satisfied!")
        print("\n🚀 You can now run the metadata updater:")
        print("   python c:\\Windows_software\\whisper-acft\\update_audio_metadata.py --dry-run")
        print("   python c:\\Windows_software\\whisper-acft\\update_audio_metadata.py")
    else:
        print("\n❌ Please install missing dependencies first")
        sys.exit(1)

if __name__ == "__main__":
    main()
