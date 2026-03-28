#!/usr/bin/env python3
"""
Audio Segment Analyzer - Detect actual audio content in large audio files
Identifies segments with real audio vs silence/quiet segments
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install librosa soundfile")
    import librosa
    import soundfile as sf

@dataclass
class AudioSegment:
    """Represents a detected audio segment"""
    start_time: float  # seconds
    end_time: float    # seconds
    duration: float    # seconds
    avg_amplitude: float
    max_amplitude: float
    is_speech: bool = False

class AudioSegmentAnalyzer:
    def __init__(self, 
                 silence_threshold: float = 0.01,
                 min_segment_duration: float = 0.5,
                 hop_length: int = 512,
                 sr: int = 16000):
        """
        Initialize the audio analyzer
        
        Args:
            silence_threshold: Amplitude threshold below which is considered silence (0-1)
            min_segment_duration: Minimum duration in seconds for a valid audio segment
            hop_length: Hop length for audio processing
            sr: Sample rate to load audio at
        """
        self.silence_threshold = silence_threshold
        self.min_segment_duration = min_segment_duration
        self.hop_length = hop_length
        self.sr = sr
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate"""
        try:
            # Load audio with librosa
            audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
            return audio, sr
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None
    
    def detect_audio_segments(self, audio: np.ndarray, sr: int) -> List[AudioSegment]:
        """
        Detect audio segments in the audio data
        
        Args:
            audio: Audio data array
            sr: Sample rate
            
        Returns:
            List of detected audio segments
        """
        # Calculate RMS energy in windows
        frame_length = self.hop_length * 4
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=self.hop_length)[0]
        
        # Convert frame indices to time
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=self.hop_length)
        
        # Find segments above threshold
        above_threshold = rms > self.silence_threshold
        
        # Find contiguous segments
        segments = []
        in_segment = False
        segment_start = 0
        
        for i, is_above in enumerate(above_threshold):
            if is_above and not in_segment:
                # Start of new segment
                in_segment = True
                segment_start = times[i]
            elif not is_above and in_segment:
                # End of segment
                segment_end = times[i]
                duration = segment_end - segment_start
                
                if duration >= self.min_segment_duration:
                    # Extract audio for this segment
                    start_sample = int(segment_start * sr)
                    end_sample = int(segment_end * sr)
                    segment_audio = audio[start_sample:end_sample]
                    
                    # Calculate amplitude stats
                    avg_amplitude = float(np.mean(np.abs(segment_audio)))
                    max_amplitude = float(np.max(np.abs(segment_audio)))
                    
                    segments.append(AudioSegment(
                        start_time=segment_start,
                        end_time=segment_end,
                        duration=duration,
                        avg_amplitude=avg_amplitude,
                        max_amplitude=max_amplitude,
                        is_speech=self._is_likely_speech(segment_audio, sr)
                    ))
                
                in_segment = False
        
        # Handle case where audio ends while still in segment
        if in_segment:
            segment_end = times[-1]
            duration = segment_end - segment_start
            
            if duration >= self.min_segment_duration:
                start_sample = int(segment_start * sr)
                end_sample = int(segment_end * sr)
                segment_audio = audio[start_sample:end_sample]
                
                avg_amplitude = float(np.mean(np.abs(segment_audio)))
                max_amplitude = float(np.max(np.abs(segment_audio)))
                
                segments.append(AudioSegment(
                    start_time=segment_start,
                    end_time=segment_end,
                    duration=duration,
                    avg_amplitude=avg_amplitude,
                    max_amplitude=max_amplitude,
                    is_speech=self._is_likely_speech(segment_audio, sr)
                ))
        
        return segments
    
    def _is_likely_speech(self, audio: np.ndarray, sr: int) -> bool:
        """
        Simple heuristic to determine if audio is likely speech
        """
        if len(audio) < sr * 0.1:  # Less than 0.1 seconds
            return False
        
        # Calculate spectral features
        try:
            # Zero crossing rate - speech typically has higher ZCR
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            avg_zcr = np.mean(zcr)
            
            # Spectral centroid - speech typically has certain range
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            avg_centroid = np.mean(spectral_centroids)
            
            # Simple heuristic based on these features
            # These thresholds might need tuning
            speech_like = (avg_zcr > 0.05) and (avg_centroid > 500 and avg_centroid < 4000)
            
            return speech_like
        except:
            return False
    
    def analyze_file(self, audio_path: str) -> Dict:
        """
        Analyze a single audio file and return results
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with analysis results
        """
        print(f"\nAnalyzing: {Path(audio_path).name}")
        
        # Load audio
        audio, sr = self.load_audio(audio_path)
        if audio is None:
            return {"error": f"Could not load {audio_path}"}
        
        total_duration = len(audio) / sr
        
        # Detect segments
        segments = self.detect_audio_segments(audio, sr)
        
        # Calculate statistics
        total_audio_time = sum(seg.duration for seg in segments)
        silence_time = total_duration - total_audio_time
        
        # Sort segments by start time
        segments.sort(key=lambda x: x.start_time)
        
        results = {
            "file_path": str(audio_path),
            "file_name": Path(audio_path).name,
            "total_duration": total_duration,
            "total_audio_time": total_audio_time,
            "silence_time": silence_time,
            "audio_percentage": (total_audio_time / total_duration) * 100 if total_duration > 0 else 0,
            "num_segments": len(segments),
            "segments": [
                {
                    "start_time": float(seg.start_time),
                    "end_time": float(seg.end_time),
                    "duration": float(seg.duration),
                    "avg_amplitude": float(seg.avg_amplitude),
                    "max_amplitude": float(seg.max_amplitude),
                    "is_speech": bool(seg.is_speech),
                    "start_time_formatted": self._format_time(seg.start_time),
                    "end_time_formatted": self._format_time(seg.end_time),
                    "duration_formatted": self._format_time(seg.duration)
                }
                for seg in segments
            ]
        }
        
        return results
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS.mmm format"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
    
    def print_summary(self, results: Dict):
        """Print a summary of the analysis results"""
        if "error" in results:
            print(f"❌ {results['error']}")
            return
        
        print(f"📁 File: {results['file_name']}")
        print(f"⏱️  Total Duration: {self._format_time(results['total_duration'])}")
        print(f"🔊 Audio Content: {self._format_time(results['total_audio_time'])} ({results['audio_percentage']:.1f}%)")
        print(f"🔇 Silence: {self._format_time(results['silence_time'])}")
        print(f"📊 Segments Found: {results['num_segments']}")
        
        if results['segments']:
            print(f"\n📋 Audio Segments:")
            print("   # | Start    | End      | Duration | Speech | Amp (Avg/Max)")
            print("   ---|----------|----------|----------|--------|-------------")
            
            for i, seg in enumerate(results['segments'], 1):
                speech_marker = "✓" if seg['is_speech'] else "✗"
                print(f"   {i:2d} | {seg['start_time_formatted']} | {seg['end_time_formatted']} | {seg['duration_formatted']} |   {speech_marker}    | {seg['avg_amplitude']:.4f}/{seg['max_amplitude']:.4f}")

def find_manual_audio_files(directory: str) -> List[str]:
    """Find all audio files starting with 'manual_' in the directory"""
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
    manual_files = []
    
    for file_path in Path(directory).iterdir():
        if file_path.is_file() and file_path.name.lower().startswith('manual_'):
            if file_path.suffix.lower() in audio_extensions:
                manual_files.append(str(file_path))
    
    return sorted(manual_files)

def main():
    parser = argparse.ArgumentParser(description="Analyze audio files to detect actual audio segments")
    parser.add_argument("--input-dir", "-i", help="Directory containing audio files")
    parser.add_argument("--files", "-f", nargs="+", help="Specific audio files to analyze")
    parser.add_argument("--threshold", "-t", type=float, default=0.01, 
                       help="Silence threshold (0-1, default: 0.01)")
    parser.add_argument("--min-duration", "-m", type=float, default=0.5,
                       help="Minimum segment duration in seconds (default: 0.5)")
    parser.add_argument("--output-json", "-o", help="Output JSON file for results")
    parser.add_argument("--manual-only", action="store_true", 
                       help="Only analyze files starting with 'manual_'")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = AudioSegmentAnalyzer(
        silence_threshold=args.threshold,
        min_segment_duration=args.min_duration
    )
    
    # Determine which files to analyze
    if args.files:
        files_to_analyze = args.files
    elif args.input_dir:
        if args.manual_only:
            files_to_analyze = find_manual_audio_files(args.input_dir)
        else:
            # Find all audio files
            audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
            files_to_analyze = []
            for file_path in Path(args.input_dir).iterdir():
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    files_to_analyze.append(str(file_path))
            files_to_analyze.sort()
    else:
        print("Either --input-dir or --files must be specified")
        return
    
    if not files_to_analyze:
        print("No audio files found")
        return
    
    print(f"Found {len(files_to_analyze)} audio files to analyze")
    
    # Analyze files
    all_results = []
    for audio_file in tqdm(files_to_analyze, desc="Analyzing files"):
        results = analyzer.analyze_file(audio_file)
        analyzer.print_summary(results)
        all_results.append(results)
    
    # Save results to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {args.output_json}")
    
    # Print overall summary
    print(f"\n📈 Overall Summary:")
    total_files = len([r for r in all_results if "error" not in r])
    total_duration = sum(r.get('total_duration', 0) for r in all_results if "error" not in r)
    total_audio = sum(r.get('total_audio_time', 0) for r in all_results if "error" not in r)
    
    print(f"   Files analyzed: {total_files}")
    print(f"   Total duration: {analyzer._format_time(total_duration)}")
    print(f"   Total audio content: {analyzer._format_time(total_audio)} ({(total_audio/total_duration)*100:.1f}%)")

if __name__ == "__main__":
    main()
