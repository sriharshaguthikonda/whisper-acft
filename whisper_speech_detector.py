#!/usr/bin/env python3
"""
Whisper Speech Detector - Use Whisper tiny model to detect actual spoken audio segments
Identifies where real speech occurs in audio files with timestamps
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
    import whisper
    import torch
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install openai-whisper torch")
    import whisper
    import torch

@dataclass
class SpeechSegment:
    """Represents a detected speech segment from Whisper"""
    start_time: float  # seconds
    end_time: float    # seconds
    duration: float    # seconds
    text: str
    confidence: float
    language: str = "en"

class WhisperSpeechDetector:
    def __init__(self, model_size: str = "tiny", device: Optional[str] = None):
        """
        Initialize the Whisper speech detector
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cuda, cpu, or None for auto)
        """
        self.model_size = model_size
        
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        print(f"Loading Whisper {model_size} model on {device}...")
        self.model = whisper.load_model(model_size, device=device)
        print(f"Model loaded successfully!")
        
    def transcribe_audio(self, audio_path: str, language: str = None) -> Dict:
        """
        Transcribe audio file using Whisper
        
        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en') or None for auto-detect
            
        Returns:
            Dictionary with transcription results
        """
        try:
            print(f"\n🎙️  Transcribing: {Path(audio_path).name}")
            
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                verbose=False,
                fp16=(self.device == "cuda")
            )
            
            # Extract speech segments
            segments = []
            for segment in result.get("segments", []):
                for word in segment.get("words", []):
                    if word:  # Check if word dict exists
                        speech_seg = SpeechSegment(
                            start_time=word.get("start", 0),
                            end_time=word.get("end", 0),
                            duration=word.get("end", 0) - word.get("start", 0),
                            text=word.get("word", ""),
                            confidence=word.get("probability", 0),
                            language=result.get("language", "en")
                        )
                        segments.append(speech_seg)
            
            # If no word-level timestamps, use segment-level
            if not segments:
                for segment in result.get("segments", []):
                    speech_seg = SpeechSegment(
                        start_time=segment.get("start", 0),
                        end_time=segment.get("end", 0),
                        duration=segment.get("end", 0) - segment.get("start", 0),
                        text=segment.get("text", ""),
                        confidence=0.5,  # Default confidence for segment-level
                        language=result.get("language", "en")
                    )
                    segments.append(speech_seg)
            
            return {
                "file_path": str(audio_path),
                "file_name": Path(audio_path).name,
                "detected_language": result.get("language", "unknown"),
                "full_text": result.get("text", ""),
                "segments": segments,
                "total_speech_time": sum(seg.duration for seg in segments),
                "num_words": len(segments)
            }
            
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return {"error": str(e), "file_path": str(audio_path)}
    
    def analyze_speech_patterns(self, transcription_result: Dict) -> Dict:
        """
        Analyze speech patterns in the transcription results
        
        Args:
            transcription_result: Results from transcribe_audio
            
        Returns:
            Dictionary with speech pattern analysis
        """
        if "error" in transcription_result:
            return transcription_result
        
        segments = transcription_result["segments"]
        if not segments:
            return {
                **transcription_result,
                "speech_analysis": {
                    "total_speech_percentage": 0,
                    "avg_confidence": 0,
                    "speech_density": 0,
                    "longest_speech_gap": 0,
                    "speech_segments_summary": []
                }
            }
        
        # Calculate speech statistics
        total_speech_time = transcription_result["total_speech_time"]
        avg_confidence = np.mean([seg.confidence for seg in segments])
        
        # Find speech gaps (silence between speech segments)
        speech_gaps = []
        for i in range(len(segments) - 1):
            gap = segments[i + 1].start_time - segments[i].end_time
            if gap > 0:
                speech_gaps.append(gap)
        
        longest_gap = max(speech_gaps) if speech_gaps else 0
        
        # Group consecutive speech into meaningful segments
        speech_segments_summary = []
        if segments:
            current_start = segments[0].start_time
            current_end = segments[0].end_time
            current_text = segments[0].text
            
            for i in range(1, len(segments)):
                gap = segments[i].start_time - current_end
                # If gap is small (< 2 seconds), consider it continuous speech
                if gap < 2.0:
                    current_end = segments[i].end_time
                    current_text += " " + segments[i].text
                else:
                    # End current segment and start new one
                    speech_segments_summary.append({
                        "start_time": current_start,
                        "end_time": current_end,
                        "duration": current_end - current_start,
                        "text": current_text.strip(),
                        "word_count": len(current_text.split())
                    })
                    current_start = segments[i].start_time
                    current_end = segments[i].end_time
                    current_text = segments[i].text
            
            # Add the last segment
            speech_segments_summary.append({
                "start_time": current_start,
                "end_time": current_end,
                "duration": current_end - current_start,
                "text": current_text.strip(),
                "word_count": len(current_text.split())
            })
        
        return {
            **transcription_result,
            "speech_analysis": {
                "total_speech_percentage": 0,  # Will be calculated later
                "avg_confidence": float(avg_confidence),
                "speech_density": len(segments) / total_speech_time if total_speech_time > 0 else 0,
                "longest_speech_gap": longest_gap,
                "speech_segments_summary": speech_segments_summary,
                "total_speech_segments": len(speech_segments_summary)
            }
        }
    
    def print_summary(self, analysis_result: Dict, total_duration: float = None):
        """Print a summary of the speech analysis results"""
        if "error" in analysis_result:
            print(f"❌ {analysis_result['error']}")
            return
        
        file_name = analysis_result["file_name"]
        detected_lang = analysis_result["detected_language"]
        full_text = analysis_result["full_text"].strip()
        segments = analysis_result["segments"]
        analysis = analysis_result["speech_analysis"]
        
        print(f"\n📁 File: {file_name}")
        print(f"🌐 Detected Language: {detected_lang}")
        
        if total_duration:
            speech_percentage = (analysis_result["total_speech_time"] / total_duration) * 100
            print(f"⏱️  Total Duration: {self._format_time(total_duration)}")
            print(f"🗣️  Speech Time: {self._format_time(analysis_result['total_speech_time'])} ({speech_percentage:.1f}%)")
            print(f"🔇 Silence: {self._format_time(total_duration - analysis_result['total_speech_time'])}")
        else:
            print(f"🗣️  Total Speech Time: {self._format_time(analysis_result['total_speech_time'])}")
        
        print(f"📊 Words Detected: {analysis_result['num_words']}")
        print(f"📈 Speech Segments: {analysis.get('speech_analysis', {}).get('total_speech_segments', 0)}")
        print(f"🎯 Avg Confidence: {analysis.get('speech_analysis', {}).get('avg_confidence', 0):.3f}")
        print(f"⏸️  Longest Speech Gap: {self._format_time(analysis.get('speech_analysis', {}).get('longest_speech_gap', 0))}")
        
        if full_text:
            print(f"\n📝 Full Transcription:")
            print(f"   \"{full_text[:200]}{'...' if len(full_text) > 200 else ''}\"")
        
        if analysis.get("speech_segments_summary"):
            print(f"\n📋 Speech Segments:")
            print("   # | Start    | End      | Duration | Words | Text Preview")
            print("   ---|----------|----------|----------|-------|-------------")
            
            for i, seg in enumerate(analysis.get("speech_segments_summary", [])[:10], 1):  # Show first 10
                preview = seg["text"][:50] + "..." if len(seg["text"]) > 50 else seg["text"]
                preview = preview.replace("\n", " ")
                print(f"   {i:2d} | {self._format_time(seg['start_time'])} | {self._format_time(seg['end_time'])} | {self._format_time(seg['duration'])} | {seg['word_count']:5d} | {preview}")
            
            if len(analysis.get("speech_segments_summary", [])) > 10:
                print(f"   ... and {len(analysis.get('speech_segments_summary', [])) - 10} more segments")
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to HH:MM:SS or MM:SS format"""
        if seconds < 3600:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes:02d}:{secs:06.3f}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = seconds % 60
            return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def get_audio_duration(audio_path: str) -> float:
    """Get audio file duration using whisper (faster than librosa for this purpose)"""
    try:
        info = whisper.load_audio(audio_path)
        # Using librosa for duration calculation
        import librosa
        duration = librosa.get_duration(filename=audio_path)
        return duration
    except:
        return 0

def main():
    parser = argparse.ArgumentParser(description="Use Whisper to detect actual speech in audio files")
    parser.add_argument("--input-dir", "-i", help="Directory containing audio files")
    parser.add_argument("--files", "-f", nargs="+", help="Specific audio files to analyze")
    parser.add_argument("--model", "-m", default="tiny", choices=["tiny", "base", "small", "medium", "large"],
                       help="Whisper model size (default: tiny)")
    parser.add_argument("--device", "-d", choices=["cuda", "cpu"], help="Device to use (default: auto)")
    parser.add_argument("--language", "-l", help="Language code (e.g., en) or None for auto-detect")
    parser.add_argument("--output-json", "-o", help="Output JSON file for results")
    parser.add_argument("--manual-only", action="store_true", 
                       help="Only analyze files starting with 'manual_'")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = WhisperSpeechDetector(model_size=args.model, device=args.device)
    
    # Determine which files to analyze
    if args.files:
        files_to_analyze = args.files
    elif args.input_dir:
        if args.manual_only:
            audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}
            files_to_analyze = []
            for file_path in Path(args.input_dir).iterdir():
                if file_path.is_file() and file_path.name.lower().startswith('manual_'):
                    if file_path.suffix.lower() in audio_extensions:
                        files_to_analyze.append(str(file_path))
            files_to_analyze.sort()
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
    for audio_file in tqdm(files_to_analyze, desc="Transcribing files"):
        # Get audio duration
        duration = get_audio_duration(audio_file)
        
        # Transcribe and analyze
        transcription = detector.transcribe_audio(audio_file, language=args.language)
        analysis = detector.analyze_speech_patterns(transcription)
        
        # Calculate speech percentage
        if "speech_analysis" in analysis and duration > 0:
            speech_time = analysis["total_speech_time"]
            analysis["speech_analysis"]["total_speech_percentage"] = (speech_time / duration) * 100
        
        detector.print_summary(analysis, duration)
        all_results.append(analysis)
    
    # Save results to JSON if requested
    if args.output_json:
        # Convert dataclasses to dicts for JSON serialization
        json_results = []
        for result in all_results:
            if "error" not in result:
                json_result = {
                    "file_path": result["file_path"],
                    "file_name": result["file_name"],
                    "detected_language": result["detected_language"],
                    "full_text": result["full_text"],
                    "total_speech_time": result["total_speech_time"],
                    "num_words": result["num_words"],
                    "speech_analysis": result["speech_analysis"],
                    "segments": [
                        {
                            "start_time": seg.start_time,
                            "end_time": seg.end_time,
                            "duration": seg.duration,
                            "text": seg.text,
                            "confidence": seg.confidence,
                            "language": seg.language
                        }
                        for seg in result["segments"]
                    ]
                }
            else:
                json_result = result
            json_results.append(json_result)
        
        with open(args.output_json, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to: {args.output_json}")
    
    # Print overall summary
    print(f"\n📈 Overall Summary:")
    successful_results = [r for r in all_results if "error" not in r]
    total_files = len(successful_results)
    total_duration = sum(get_audio_duration(r["file_path"]) for r in successful_results)
    total_speech = sum(r["total_speech_time"] for r in successful_results)
    
    print(f"   Files analyzed: {total_files}")
    print(f"   Total duration: {detector._format_time(total_duration)}")
    print(f"   Total speech detected: {detector._format_time(total_speech)} ({(total_speech/total_duration)*100:.1f}%)")
    print(f"   Total words detected: {sum(r['num_words'] for r in successful_results)}")

if __name__ == "__main__":
    main()
