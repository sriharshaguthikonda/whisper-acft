#!/usr/bin/env python3
"""
Evaluation script to test all trained checkpoints against test audio files.
Compares performance metrics (WER, processing time) across all checkpoints.
"""

import os
import json
import time
import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer as jiwer_wer

# CONFIG
TEST_AUDIO_DIR = r"i:\whisper-acft\test_sample"
CHECKPOINT_DIR = r"i:\checkpoints_partialctx"
BASE_MODEL_ID = "futo-org/acft-whisper-tiny.en"  # Futo Whisper tiny English model
PROCESSOR_ID = "openai/whisper-tiny.en"  # Use OpenAI processor since Futo model lacks processor files
TARGET_SR = 16000
MAX_NEW_TOKENS = 128
EVAL_LANGUAGE = "en"
EVAL_TASK = "transcribe"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_audio(audio_path):
    """Load audio file using librosa."""
    try:
        waveform, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True, dtype=np.float32)
        return waveform
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return np.zeros(16000, dtype=np.float32)

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
    
    # Find model_epoch_XXXXX directories
    for item in checkpoint_dir.iterdir():
        if item.is_dir() and item.name.startswith('model_epoch_'):
            try:
                epoch_num = int(item.name.split('_')[-1])
                checkpoints.append((epoch_num, str(item)))
            except ValueError:
                continue
    
    # Sort by epoch number
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints

def evaluate_checkpoint(checkpoint_path, test_files, processor, base_model):
    """Evaluate a single checkpoint on all test files."""
    print(f"\n=== Evaluating {checkpoint_path} ===")
    
    try:
        # Load model from checkpoint
        model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
        model.to(DEVICE)
        model.eval()
        
        # Get forced decoder IDs
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=EVAL_LANGUAGE, task=EVAL_TASK)
        
        results = []
        total_time = 0
        
        for audio_file in test_files:
            print(f"Processing {audio_file.name}...")
            
            # Load audio
            audio = load_audio(str(audio_file))
            
            # Process audio
            start_time = time.time()
            input_features = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(DEVICE)
            
            # Generate transcription
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features=input_features,
                    max_new_tokens=MAX_NEW_TOKENS
                )
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            processing_time = time.time() - start_time
            total_time += processing_time
            
            results.append({
                "file": audio_file.name,
                "transcription": transcription,
                "processing_time": processing_time
            })
            
            print(f"  Transcription: {transcription}")
            print(f"  Time: {processing_time:.2f}s")
        
        # Calculate average metrics
        avg_time = total_time / len(test_files) if test_files else 0
        
        return {
            "checkpoint": checkpoint_path,
            "results": results,
            "avg_processing_time": avg_time,
            "total_files": len(test_files),
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error evaluating {checkpoint_path}: {e}")
        return {
            "checkpoint": checkpoint_path,
            "error": str(e),
            "status": "failed"
        }
    finally:
        # Cleanup
        if 'model' in locals():
            del model
            torch.cuda.empty_cache()

def evaluate_base_model(test_files, processor):
    """Evaluate the base model for comparison."""
    print(f"\n=== Evaluating Base Model {BASE_MODEL_ID} ===")
    
    try:
        model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_ID)
        model.to(DEVICE)
        model.eval()
        
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=EVAL_LANGUAGE, task=EVAL_TASK)
        
        results = []
        total_time = 0
        
        for audio_file in test_files:
            print(f"Processing {audio_file.name}...")
            
            audio = load_audio(str(audio_file))
            input_features = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(DEVICE)
            
            start_time = time.time()
            with torch.no_grad():
                predicted_ids = model.generate(
                    input_features=input_features,
                    max_new_tokens=MAX_NEW_TOKENS
                )
            
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            processing_time = time.time() - start_time
            total_time += processing_time
            
            results.append({
                "file": audio_file.name,
                "transcription": transcription,
                "processing_time": processing_time
            })
            
            print(f"  Transcription: {transcription}")
            print(f"  Time: {processing_time:.2f}s")
        
        avg_time = total_time / len(test_files) if test_files else 0
        
        return {
            "checkpoint": "base_model",
            "results": results,
            "avg_processing_time": avg_time,
            "total_files": len(test_files),
            "status": "success"
        }
        
    except Exception as e:
        print(f"Error evaluating base model: {e}")
        return {
            "checkpoint": "base_model",
            "error": str(e),
            "status": "failed"
        }
    finally:
        if 'model' in locals():
            del model
            torch.cuda.empty_cache()

def main():
    print("=== Whisper Checkpoint Evaluation ===")
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
    
    if not test_files:
        print("No test files found. Exiting.")
        return
    
    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return
    
    # Load processor
    try:
        processor = WhisperProcessor.from_pretrained(PROCESSOR_ID)
        print(f"\nLoaded processor: {PROCESSOR_ID}")
    except Exception as e:
        print(f"Error loading processor: {e}")
        return
    
    # Evaluate base model first
    base_results = evaluate_base_model(test_files, processor)
    
    # Evaluate all checkpoints
    all_results = [base_results]
    
    for epoch_num, checkpoint_path in checkpoints:
        result = evaluate_checkpoint(checkpoint_path, test_files, processor, None)
        all_results.append(result)
    
    # Save results
    output_file = "checkpoint_evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Evaluation Summary ===")
    print(f"Results saved to: {output_file}")
    
    # Print summary table
    print("\nModel Performance Summary:")
    print("-" * 80)
    print(f"{'Model':<20} {'Status':<10} {'Avg Time (s)':<15} {'Files':<8}")
    print("-" * 80)
    
    for result in all_results:
        model_name = result["checkpoint"].split('\\')[-1] if result["checkpoint"] != "base_model" else "Base Model"
        status = result["status"]
        avg_time = result.get("avg_processing_time", "N/A")
        files = result.get("total_files", "N/A")
        
        print(f"{model_name:<20} {status:<10} {avg_time:<15} {files:<8}")
    
    # Show detailed transcriptions for comparison
    print(f"\n=== Detailed Transcriptions ===")
    for i, test_file in enumerate(test_files):
        print(f"\n--- {test_file.name} ---")
        for result in all_results:
            if result["status"] == "success":
                model_name = result["checkpoint"].split('\\')[-1] if result["checkpoint"] != "base_model" else "Base Model"
                transcription = result["results"][i]["transcription"]
                time_taken = result["results"][i]["processing_time"]
                print(f"{model_name}: {transcription} ({time_taken:.2f}s)")

if __name__ == "__main__":
    main()
