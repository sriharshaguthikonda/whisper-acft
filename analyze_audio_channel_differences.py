"""
Audio Channel Difference Analyzer

This script analyzes the actual audio signal differences between left and right channels
using various signal processing metrics, not speaker identification.

Metrics used:
1. RMS level difference
2. Spectral centroid difference  
3. Zero crossing rate difference
4. Correlation coefficient
5. Phase difference
6. Energy distribution difference

Usage:
    python analyze_audio_channel_differences.py --input-dir "I:\Record" --output-json "audio_channel_diff_report.json"
"""

import argparse
import json
import pathlib
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from scipy.stats import pearsonr


@dataclass
class ChannelDifferenceMetrics:
    filename: str
    rms_diff_db: float
    rms_left: float
    rms_right: float
    spectral_centroid_diff: float
    spectral_centroid_left: float
    spectral_centroid_right: float
    zcr_diff: float
    zcr_left: float
    zcr_right: float
    correlation: float
    phase_diff_mean: float
    phase_diff_std: float
    energy_ratio: float
    overall_difference_score: float
    significant_difference: bool


def load_audio(path: pathlib.Path, target_sr: int = 16000) -> Tuple[torch.Tensor, int]:
    """Load audio and resample if needed"""
    try:
        # Try torchaudio first
        wav, sr = torchaudio.load(str(path))
    except Exception as e:
        # Fallback to ffmpeg
        try:
            import subprocess
            import tempfile
            
            # Create temporary wav file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            
            # Convert using ffmpeg
            cmd = [
                'ffmpeg', '-y', '-i', str(path), 
                '-acodec', 'pcm_s16le', '-ar', str(target_sr), 
                '-ac', '2', tmp_path
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            
            # Load the converted file
            wav, sr = torchaudio.load(tmp_path)
            
            # Clean up
            import os
            os.unlink(tmp_path)
            
        except Exception as e2:
            print(f"Could not load {path.name}: {e}")
            raise e2
    
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        wav = resampler(wav)
    
    return wav, target_sr


def compute_rms(audio: torch.Tensor) -> float:
    """Compute RMS level"""
    return torch.sqrt(torch.mean(audio ** 2)).item()


def compute_spectral_centroid(audio: torch.Tensor, sr: int) -> float:
    """Compute spectral centroid"""
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    
    # Use STFT to compute spectral centroid
    stft = torch.stft(audio, n_fft=1024, hop_length=512, return_complex=True)
    magnitude = torch.abs(stft)
    freqs = torch.linspace(0, sr/2, magnitude.shape[0])
    
    # Weighted average of frequencies
    centroid = torch.sum(freqs.unsqueeze(1) * magnitude, dim=0) / torch.sum(magnitude, dim=0)
    return torch.mean(centroid).item()


def compute_zero_crossing_rate(audio: torch.Tensor) -> float:
    """Compute zero crossing rate"""
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    
    signs = torch.sign(audio)
    zero_crossings = torch.sum(torch.abs(signs[:, 1:] - signs[:, :-1]), dim=1)
    zcr = zero_crossings / (2 * audio.shape[1])
    return torch.mean(zcr).item()


def compute_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute correlation coefficient between channels"""
    left_np = left.numpy().flatten()
    right_np = right.numpy().flatten()
    
    # Use Pearson correlation
    corr, _ = pearsonr(left_np, right_np)
    return corr if not np.isnan(corr) else 0.0


def compute_phase_difference(left: torch.Tensor, right: torch.Tensor, sr: int) -> Tuple[float, float]:
    """Compute phase difference between channels"""
    # Use STFT to get phase information
    stft_left = torch.stft(left, n_fft=1024, hop_length=512, return_complex=True)
    stft_right = torch.stft(right, n_fft=1024, hop_length=512, return_complex=True)
    
    phase_left = torch.angle(stft_left)
    phase_right = torch.angle(stft_right)
    
    phase_diff = phase_left - phase_right
    
    # Wrap to [-pi, pi]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    
    mean_diff = torch.mean(phase_diff).item()
    std_diff = torch.std(phase_diff).item()
    
    return mean_diff, std_diff


def compute_energy_ratio(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute energy ratio between channels"""
    energy_left = torch.sum(left ** 2).item()
    energy_right = torch.sum(right ** 2).item()
    
    if energy_right == 0:
        return float('inf')
    return energy_left / energy_right


def analyze_channel_differences(audio_path: pathlib.Path) -> Optional[ChannelDifferenceMetrics]:
    """Analyze differences between left and right channels"""
    try:
        wav, sr = load_audio(audio_path)
        
        # Ensure we have exactly 2 channels
        if wav.shape[0] != 2:
            print(f"Skipping {audio_path.name}: not stereo (has {wav.shape[0]} channels)")
            return None
        
        left = wav[0]
        right = wav[1]
        
        # Compute metrics
        rms_left = compute_rms(left)
        rms_right = compute_rms(right)
        rms_diff_db = 20 * np.log10(rms_left / rms_right) if rms_right > 0 else float('inf')
        
        spec_centroid_left = compute_spectral_centroid(left, sr)
        spec_centroid_right = compute_spectral_centroid(right, sr)
        spec_centroid_diff = spec_centroid_left - spec_centroid_right
        
        zcr_left = compute_zero_crossing_rate(left)
        zcr_right = compute_zero_crossing_rate(right)
        zcr_diff = zcr_left - zcr_right
        
        correlation = compute_correlation(left, right)
        phase_mean, phase_std = compute_phase_difference(left, right, sr)
        energy_ratio = compute_energy_ratio(left, right)
        
        # Compute overall difference score (normalized)
        # Higher score means more different channels
        rms_score = min(abs(rms_diff_db) / 20.0, 1.0) if not np.isnan(rms_diff_db) and not np.isinf(rms_diff_db) else 0.0  # Normalize to 0-1, cap at 20dB
        spec_score = min(abs(spec_centroid_diff) / 2000.0, 1.0) if not np.isnan(spec_centroid_diff) else 0.0  # Normalize to 0-1, cap at 2kHz
        zcr_score = min(abs(zcr_diff) / 0.1, 1.0) if not np.isnan(zcr_diff) else 0.0  # Normalize to 0-1, cap at 0.1
        corr_score = 1.0 - abs(correlation) if not np.isnan(correlation) else 0.0  # Lower correlation = more different
        phase_score = min(phase_std / np.pi, 1.0) if not np.isnan(phase_std) else 0.0  # Normalize to 0-1
        energy_score = min(abs(np.log10(energy_ratio)) / 2.0, 1.0) if energy_ratio > 0 and not np.isnan(energy_ratio) and not np.isinf(energy_ratio) else 0.0  # Normalize to 0-1, cap at factor 100
        
        overall_score = (rms_score + spec_score + zcr_score + corr_score + phase_score + energy_score) / 6.0
        
        # Determine if significantly different (threshold can be adjusted)
        significant = overall_score > 0.3  # 30% difference threshold
        
        return ChannelDifferenceMetrics(
            filename=audio_path.name,
            rms_diff_db=float(rms_diff_db) if not np.isnan(rms_diff_db) and not np.isinf(rms_diff_db) else 0.0,
            rms_left=float(rms_left),
            rms_right=float(rms_right),
            spectral_centroid_diff=float(spec_centroid_diff) if not np.isnan(spec_centroid_diff) else 0.0,
            spectral_centroid_left=float(spec_centroid_left) if not np.isnan(spec_centroid_left) else 0.0,
            spectral_centroid_right=float(spec_centroid_right) if not np.isnan(spec_centroid_right) else 0.0,
            zcr_diff=float(zcr_diff) if not np.isnan(zcr_diff) else 0.0,
            zcr_left=float(zcr_left),
            zcr_right=float(zcr_right),
            correlation=float(correlation) if not np.isnan(correlation) else 0.0,
            phase_diff_mean=float(phase_mean) if not np.isnan(phase_mean) else 0.0,
            phase_diff_std=float(phase_std) if not np.isnan(phase_std) else 0.0,
            energy_ratio=float(energy_ratio) if not np.isnan(energy_ratio) and not np.isinf(energy_ratio) else 1.0,
            overall_difference_score=float(overall_score),
            significant_difference=bool(significant)
        )
        
    except Exception as e:
        print(f"Error processing {audio_path.name}: {e}")
        return None


def discover_audio_files(root: pathlib.Path, exts: List[str]) -> List[pathlib.Path]:
    """Find all audio files in directory"""
    audio_files = []
    for ext in exts:
        audio_files.extend(root.rglob(f"*{ext}"))
    return sorted(audio_files)


def main():
    parser = argparse.ArgumentParser(description="Analyze audio channel differences")
    parser.add_argument("--input-dir", required=True, help="Directory containing audio files")
    parser.add_argument("--output-json", required=True, help="Output JSON report file")
    parser.add_argument("--extensions", default=".wav,.mp3,.m4a,.flac", help="Audio file extensions")
    parser.add_argument("--min-score", type=float, default=0.3, help="Minimum difference score to consider significant")
    parser.add_argument("--force", action="store_true", help="Force reprocess all files (ignore existing results)")
    
    args = parser.parse_args()
    
    input_dir = pathlib.Path(args.input_dir)
    output_file = pathlib.Path(args.output_json)
    extensions = [ext.strip() for ext in args.extensions.split(",")]
    
    # Load existing results if not forcing
    existing_results = {}
    if output_file.exists() and not args.force:
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
            # Create lookup by filename
            for file_data in existing_data.get('all_files', []):
                existing_results[file_data['filename']] = file_data
            print(f"Loaded {len(existing_results)} existing results from {output_file}")
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
            existing_results = {}
    
    # Find audio files
    audio_files = discover_audio_files(input_dir, extensions)
    print(f"Found {len(audio_files)} audio files")
    
    # Filter files that need processing
    files_to_process = []
    for audio_path in audio_files:
        if args.force or audio_path.name not in existing_results:
            files_to_process.append(audio_path)
    
    print(f"Need to process {len(files_to_process)} files ({len(audio_files) - len(files_to_process)} already done)")
    
    # Analyze each file
    results = []
    significant_files = []
    
    # Add existing results first
    for existing_file_data in existing_results.values():
        # Reconstruct metrics object from saved data
        metrics = ChannelDifferenceMetrics(**existing_file_data)
        results.append(metrics)
        if metrics.significant_difference:
            significant_files.append(metrics)
    
    # Process new files
    for audio_path in tqdm(files_to_process, desc="Analyzing channels"):
        metrics = analyze_channel_differences(audio_path)
        if metrics:
            results.append(metrics)
            if metrics.significant_difference:
                significant_files.append(metrics)
            
            # Add to existing results to save progress incrementally
            existing_results[metrics.filename] = asdict(metrics)
    
    # Sort results by overall difference score (descending)
    results.sort(key=lambda x: x.overall_difference_score, reverse=True)
    significant_files.sort(key=lambda x: x.overall_difference_score, reverse=True)
    
    # Print summary
    print(f"\nAnalyzed {len(results)} stereo files total")
    print(f"Found {len(significant_files)} files with significant channel differences")
    
    print("\nTop 10 files with most different channels:")
    print("=" * 80)
    for i, metrics in enumerate(results[:10], 1):  # Top 10
        print(f"{i:2d}. {metrics.filename}")
        print(f"    Overall Score: {metrics.overall_difference_score:.3f}")
        print(f"    RMS Difference: {metrics.rms_diff_db:.2f} dB")
        print(f"    Correlation: {metrics.correlation:.3f}")
        print(f"    Energy Ratio: {metrics.energy_ratio:.2f}")
        print()
    
    # Save results to JSON (already sorted)
    report = {
        "metadata": {
            "total_files_analyzed": len(results),
            "significant_files": len(significant_files),
            "min_score_threshold": args.min_score,
            "extensions_analyzed": extensions,
            "resumable": True
        },
        "all_files": [asdict(r) for r in results],  # Already sorted by score
        "significant_files": [asdict(r) for r in significant_files]  # Already sorted by score
    }
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {output_file}")
    if len(files_to_process) == 0:
        print("All files already processed. Use --force to reprocess all files.")


if __name__ == "__main__":
    main()
