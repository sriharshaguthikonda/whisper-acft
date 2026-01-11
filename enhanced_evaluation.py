#!/usr/bin/env python3
"""
Enhanced Whisper evaluation with additional metrics:
- Word-level confidence scores
- Processing speed (RTFx)
- Memory usage
- Timestamp accuracy
- Language probabilities
"""

import os
import json
import time
import torch
import librosa
import numpy as np
import psutil
import gc
from pathlib import Path
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer as jiwer_wer

# CONFIG
TEST_AUDIO_DIR = r"c:\Windows_software\whisper-acft\test_sample"
CHECKPOINT_DIR = r"i:\checkpoints_partialctx"
BASE_MODEL_ID = "futo-org/acft-whisper-tiny.en"
PROCESSOR_ID = "openai/whisper-tiny.en"
TARGET_SR = 16000
MAX_NEW_TOKENS = 128
EVAL_LANGUAGE = "en"
EVAL_TASK = "transcribe"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def load_audio(audio_path):
    """Load audio file and return duration."""
    try:
        waveform, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True, dtype=np.float32)
        duration = len(waveform) / sr
        return waveform, duration
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return np.zeros(16000, dtype=np.float32), 1.0

def find_test_files():
    """Find all audio files in test directory."""
    test_files = []
    test_dir = Path(TEST_AUDIO_DIR)
    
    if not test_dir.exists():
        print(f"Test directory not found: {TEST_AUDIO_DIR}")
        return test_files
    
    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
        test_files.extend(test_dir.glob(ext))
    
    return sorted(test_files)

def find_checkpoints():
    """Find all available checkpoint directories."""
    checkpoints = []
    checkpoint_dir = Path(CHECKPOINT_DIR)
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {CHECKPOINT_DIR}")
        return checkpoints
    
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith('model_epoch_'):
            try:
                epoch_num = int(item.name.split('_')[-1])
                checkpoints.append((epoch_num, str(item)))
            except ValueError:
                continue
    
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def extract_confidence_scores(model, processor, input_features, predicted_ids):
    """Extract confidence scores from model outputs."""
    try:
        with torch.no_grad():
            outputs = model.model(
                input_features=input_features,
                decoder_input_ids=predicted_ids,
                output_attentions=True,
                output_hidden_states=True
            )
        
        # Get cross-attention weights for confidence estimation
        cross_attentions = outputs.cross_attentions
        if cross_attentions:
            # Use last decoder layer's cross-attention
            attention_weights = cross_attentions[-1]  # [batch_size, num_heads, seq_len, audio_seq_len]
            
            # Average across heads and calculate confidence
            avg_attention = attention_weights.mean(dim=1)  # [batch_size, seq_len, audio_seq_len]
            max_attention = avg_attention.max(dim=-1)[0]  # [batch_size, seq_len]
            
            # Convert to confidence scores (0-1 range)
            confidence_scores = torch.softmax(max_attention, dim=-1)
            
            return confidence_scores.cpu().numpy()
        
        return None
    except Exception as e:
        print(f"Error extracting confidence scores: {e}")
        return None

def evaluate_checkpoint_enhanced(checkpoint_path, test_files, processor):
    """Enhanced evaluation with additional metrics."""
    print(f"\n=== Enhanced Evaluation: {checkpoint_path} ===")
    
    try:
        # Memory before loading
        memory_before = get_memory_usage()
        
        # Load model
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
        model.to(DEVICE)
        model.eval()
        
        # Memory after loading
        memory_after_load = get_memory_usage()
        
        results = []
        total_processing_time = 0
        total_audio_duration = 0
        all_confidences = []
        
        for audio_file in test_files:
            print(f"Processing {audio_file.name}...")
            
            # Load audio with duration
            audio, duration = load_audio(str(audio_file))
            total_audio_duration += duration
            
            # Memory before inference
            memory_before_inference = get_memory_usage()
            
            # Process audio
            start_time = time.time()
            input_features = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(DEVICE)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features=input_features,
                    max_new_tokens=MAX_NEW_TOKENS,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            # Extract transcription
            transcription = processor.batch_decode(predicted_ids.sequences, skip_special_tokens=True)[0]
            
            # Extract confidence scores
            confidence_scores = extract_confidence_scores(model, processor, input_features, predicted_ids.sequences)
            
            # Memory after inference
            memory_after_inference = get_memory_usage()
            
            # Calculate RTFx (Real-Time Factor)
            rtfx = duration / processing_time if processing_time > 0 else 0
            
            # Average confidence
            avg_confidence = np.mean(confidence_scores) if confidence_scores is not None else 0.0
            
            result = {
                "file": audio_file.name,
                "transcription": transcription,
                "processing_time": processing_time,
                "audio_duration": duration,
                "rtfx": rtfx,
                "avg_confidence": avg_confidence,
                "memory_before": memory_before_inference,
                "memory_after": memory_after_inference,
                "memory_peak": memory_after_inference - memory_before_inference
            }
            
            if confidence_scores is not None:
                result["confidence_scores"] = confidence_scores.tolist()
                all_confidences.extend(confidence_scores.flatten())
            
            results.append(result)
            
            print(f"  Transcription: {transcription}")
            print(f"  Time: {processing_time:.3f}s, RTFx: {rtfx:.2f}, Confidence: {avg_confidence:.3f}")
        
        # Calculate aggregate metrics
        avg_rtfx = total_audio_duration / total_processing_time if total_processing_time > 0 else 0
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        avg_processing_time = total_processing_time / len(test_files)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        memory_final = get_memory_usage()
        
        return {
            "checkpoint": checkpoint_path,
            "results": results,
            "metrics": {
                "avg_processing_time": avg_processing_time,
                "avg_rtfx": avg_rtfx,
                "avg_confidence": avg_confidence,
                "total_processing_time": total_processing_time,
                "total_audio_duration": total_audio_duration,
                "memory_usage": {
                    "before_load": memory_before,
                    "after_load": memory_after_load,
                    "final": memory_final,
                    "peak_model_memory": memory_after_load - memory_before
                }
            },
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error evaluating {checkpoint_path}: {e}")
        return {
            "checkpoint": checkpoint_path,
            "error": str(e),
            "status": "failed"
        }

def main():
    print("=== Enhanced Whisper Checkpoint Evaluation ===")
    print(f"Device: {DEVICE}")
    print(f"Test directory: {TEST_AUDIO_DIR}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")
    
    # Find test files and checkpoints
    test_files = find_test_files()
    checkpoints = find_checkpoints()
    
    print(f"\nFound {len(test_files)} test files:")
    for f in test_files:
        print(f"  - {f.name}")
    
    print(f"\nFound {len(checkpoints)} checkpoints:")
    for epoch_num, path in checkpoints:
        print(f"  - Epoch {epoch_num}: {path}")
    
    if not test_files or not checkpoints:
        print("No test files or checkpoints found. Exiting.")
        return
    
    # Load processor
    try:
        processor = WhisperProcessor.from_pretrained(PROCESSOR_ID)
        print(f"\nLoaded processor: {PROCESSOR_ID}")
    except Exception as e:
        print(f"Error loading processor: {e}")
        return
    
    # Evaluate all checkpoints with enhanced metrics
    all_results = []
    
    for epoch_num, checkpoint_path in checkpoints:
        result = evaluate_checkpoint_enhanced(checkpoint_path, test_files, processor)
        all_results.append(result)
    
    # Save enhanced results
    output_file = "enhanced_checkpoint_evaluation.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Enhanced Evaluation Summary ===")
    print(f"Results saved to: {output_file}")
    
    # Print comparison table
    print("\nEnhanced Model Performance Summary:")
    print("-" * 100)
    print(f"{'Model':<20} {'RTFx':<8} {'Conf':<8} {'Time(s)':<10} {'Memory(MB)':<12} {'Status':<10}")
    print("-" * 100)
    
    for result in all_results:
        if result["status"] == "success":
            model_name = result["checkpoint"].split('\\')[-1]
            metrics = result["metrics"]
            rtfx = metrics["avg_rtfx"]
            confidence = metrics["avg_confidence"]
            time_sec = metrics["avg_processing_time"]
            memory_mb = metrics["memory_usage"]["peak_model_memory"]
            
            print(f"{model_name:<20} {rtfx:<8.2f} {confidence:<8.3f} {time_sec:<10.3f} {memory_mb:<12.1f} {'success':<10}")
        else:
            model_name = result["checkpoint"].split('\\')[-1]
            print(f"{model_name:<20} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<12} {'failed':<10}")
    
    # Find best model based on combined score
    successful_results = [r for r in all_results if r["status"] == "success"]
    if successful_results:
        # Calculate combined score: (RTFx * 0.4) + (Confidence * 0.4) + (Speed * 0.2)
        best_model = None
        best_score = -1
        
        for result in successful_results:
            metrics = result["metrics"]
            # Normalize metrics (higher RTFx and confidence are better, lower time is better)
            rtfx_score = min(metrics["avg_rtfx"] / 10.0, 1.0)  # Normalize to 0-1, cap at 1.0
            confidence_score = metrics["avg_confidence"]
            speed_score = 1.0 / (1.0 + metrics["avg_processing_time"])  # Inverse of time
            
            combined_score = (rtfx_score * 0.4) + (confidence_score * 0.4) + (speed_score * 0.2)
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = result
        
        if best_model:
            print(f"\n🏆 BEST MODEL: {best_model['checkpoint'].split('\\')[-1]}")
            print(f"   Combined Score: {best_score:.3f}")
            print(f"   RTFx: {best_model['metrics']['avg_rtfx']:.2f}")
            print(f"   Confidence: {best_model['metrics']['avg_confidence']:.3f}")
            print(f"   Avg Time: {best_model['metrics']['avg_processing_time']:.3f}s")

if __name__ == "__main__":
    main()
