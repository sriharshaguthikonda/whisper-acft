"""
Stage 14: Evaluate Test Manifest

Evaluate the trained model on the test manifest created in stage 11.
This provides WER metrics on the held-out test set.

Usage:
python stage_14_evaluate_test_manifest.py --test_manifest "I:\Record_chunks\pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_silent_test.jsonl" --checkpoint_dir "I:\Dynamic_n_ctx_checkpoints_partialctx" [--percentage 100]
"""

from __future__ import annotations

import argparse
import json
import os
import gc
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import jiwer
from tqdm import tqdm
import numpy as np


def get_gpu_memory_info():
    """Get current GPU memory usage info."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        return allocated, reserved, total
    return 0, 0, 0


def estimate_model_memory(model_name: str, is_base_model: bool = False):
    """Estimate memory needed for a model."""
    # Rough estimates based on model size
    if is_base_model:
        return 2.0  # GB for base tiny.en model
    else:
        return 1.5  # GB for fine-tuned checkpoint (slightly smaller)


def load_jsonl(path: Path):
    """Load JSONL file."""
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def evaluate_model(model_name: str, test_data: list, device: str, is_base_model: bool = False):
    """Evaluate model on test data."""
    print(f"Loading model: {model_name}")
    
    # Load processor and model
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    
    if is_base_model:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    else:
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    model.to(device)
    model.eval()
    
    predictions = []
    references = []
    
    print(f"Evaluating on {len(test_data)} test samples...")
    
    for item in tqdm(test_data, desc="Evaluating"):
        try:
            audio_path = Path(item["audio_path"])
            if not audio_path.exists():
                print(f"Warning: Audio file not found: {audio_path}")
                continue
            
            # Load and process audio
            import soundfile as sf
            audio, sr = sf.read(str(audio_path))
            if sr != 16000:
                # Resample if needed (simple linear interpolation)
                import scipy.signal
                audio = scipy.signal.resample_poly(audio, 16000, sr)
            
            # Process with processor
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate transcription
            with torch.no_grad():
                generated_ids = model.generate(**inputs)
                transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Clean up transcription
            transcription = transcription.strip()
            reference = item.get("raw_transcription", "").strip()
            
            predictions.append(transcription)
            references.append(reference)
            
        except Exception as e:
            print(f"Error processing {item.get('audio_path', 'unknown')}: {e}")
            continue
    
    return predictions, references, model_name


def calculate_wer(predictions: list, references: list):
    """Calculate WER metrics."""
    if not predictions or not references:
        return {"wer": 0.0, "cer": 0.0, "total_samples": 0}
    
    # Calculate WER
    wer = jiwer.wer(references, predictions)
    
    # Calculate CER (Character Error Rate)
    cer = jiwer.cer(references, predictions)
    
    return {
        "wer": wer,
        "cer": cer,
        "total_samples": len(predictions)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test manifest")
    parser.add_argument("--test_manifest", required=True, type=Path,
                       help="Test manifest JSONL file")
    parser.add_argument("--checkpoint_dir", required=True, type=Path,
                       help="Directory containing model checkpoints")
    parser.add_argument("--percentage", type=float, default=100.0,
                       help="Percentage of test data to use (0-100, default: 100)")
    parser.add_argument("--base_model", default="futo-org/acft-whisper-tiny.en",
                       help="Base model to evaluate (default: futo-org/acft-whisper-tiny.en)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.test_manifest.exists():
        raise FileNotFoundError(f"Test manifest not found: {args.test_manifest}")
    
    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    
    if not 0 <= args.percentage <= 100:
        raise ValueError(f"Percentage must be between 0 and 100, got: {args.percentage}")
    
    print(f"Device: {args.device}")
    print(f"Test manifest: {args.test_manifest}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Using {args.percentage}% of test data")
    
    # Load test data
    test_data = load_jsonl(args.test_manifest)
    print(f"Loaded {len(test_data)} test samples")
    
    # Apply percentage subset
    if args.percentage < 100:
        import random
        subset_size = max(1, int(len(test_data) * args.percentage / 100))
        random.seed(42)  # For reproducible results
        test_data = random.sample(test_data, subset_size)
        print(f"Using subset of {len(test_data)} samples ({args.percentage}% of total)")
    
    # Find all checkpoints and sort by epoch
    checkpoints = list(args.checkpoint_dir.glob("model_epoch_*"))
    checkpoints.sort(key=lambda x: int(x.name.split("_")[2]))
    
    # Create evaluation list: base model + all checkpoints
    models_to_evaluate = [(args.base_model, True)]  # (model_path/name, is_base_model)
    models_to_evaluate.extend([(str(cp), False) for cp in checkpoints])
    
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Will evaluate {len(models_to_evaluate)} models total (base model + {len(checkpoints)} checkpoints)")
    
    all_results = []
    
    # Evaluate each model with preloading
    for i, (model_path, is_base) in enumerate(models_to_evaluate):
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_path}")
        print(f"{'='*60}")
        
        # Check GPU memory before loading
        alloc, reserv, total = get_gpu_memory_info()
        print(f"GPU memory before: {alloc:.2f}GB allocated, {reserv:.2f}GB reserved, {total:.2f}GB total")
        
        # Preload next model if possible
        next_model = None
        if i < len(models_to_evaluate) - 1:
            next_path, next_is_base = models_to_evaluate[i + 1]
            next_model_memory = estimate_model_memory(next_path, next_is_base)
            available_memory = total - reserv
            
            if available_memory > next_model_memory + 1.0:  # Leave 1GB buffer
                print(f"Preloading next model: {next_path}")
                try:
                    if next_is_base:
                        next_model = WhisperForConditionalGeneration.from_pretrained(next_path)
                    else:
                        next_model = WhisperForConditionalGeneration.from_pretrained(next_path)
                    next_model.to(args.device)
                    next_model.eval()
                    print("Successfully preloaded next model")
                except Exception as e:
                    print(f"Failed to preload next model: {e}")
                    next_model = None
            else:
                print(f"Not enough memory to preload next model (need {next_model_memory:.1f}GB, have {available_memory:.1f}GB)")
        
        predictions, references, model_name = evaluate_model(model_path, test_data, args.device, is_base)
        
        # Calculate metrics
        metrics = calculate_wer(predictions, references)
        
        print(f"\n📊 Results for {model_name}:")
        print(f"  Total samples evaluated: {metrics['total_samples']}")
        print(f"  Word Error Rate (WER): {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
        print(f"  Character Error Rate (CER): {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
        
        # Store results
        result = {
            "model_name": model_name,
            "model_path": model_path,
            "is_base_model": is_base,
            "metrics": metrics,
            "samples_evaluated": len(predictions),
            "examples": [
                {
                    "reference": ref,
                    "predicted": pred,
                    "match": ref.strip().lower() == pred.strip().lower()
                }
                for ref, pred in zip(references[:5], predictions[:5])  # Save first 5 examples
            ]
        }
        all_results.append(result)
        
        # Save incremental results after each model
        incremental_results = {
            "base_model": args.base_model,
            "percentage_used": args.percentage,
            "total_available_samples": len(load_jsonl(args.test_manifest)),
            "current_model_index": i,
            "total_models": len(models_to_evaluate),
            "models_evaluated": len(all_results),
            "results_by_model": all_results,
            "summary": {
                "best_wer_model": min(all_results, key=lambda x: x["metrics"]["wer"])["model_name"],
                "best_cer_model": min(all_results, key=lambda x: x["metrics"]["cer"])["model_name"],
                "base_model_wer": next((r["metrics"]["wer"] for r in all_results if r["is_base_model"]), None),
                "latest_checkpoint_wer": all_results[-1]["metrics"]["wer"] if all_results else None
            }
        }
        
        # Save incremental results
        incremental_file = args.checkpoint_dir / "evaluation_results_incremental.json"
        with incremental_file.open("w", encoding="utf-8") as f:
            json.dump(incremental_results, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Incremental results saved to: {incremental_file}")
        
        # Show current summary
        print(f"\n📊 Current Summary ({i+1}/{len(models_to_evaluate)} models):")
        print(f"  Best WER so far: {incremental_results['summary']['best_wer_model']}")
        print(f"  Best CER so far: {incremental_results['summary']['best_cer_model']}")
        if incremental_results['summary']['base_model_wer'] is not None:
            print(f"  Base model WER: {incremental_results['summary']['base_model_wer']:.4f} ({incremental_results['summary']['base_model_wer']*100:.2f}%)")
        if incremental_results['summary']['latest_checkpoint_wer'] is not None:
            print(f"  Latest checkpoint WER: {incremental_results['summary']['latest_checkpoint_wer']:.4f} ({incremental_results['summary']['latest_checkpoint_wer']*100:.2f}%)")
        
        # Clean up current model
        torch.cuda.empty_cache()
        gc.collect()
        
        # If we preloaded the next model, keep it for the next iteration
        if next_model is not None:
            print("Keeping preloaded model for next iteration")
        else:
            # Regular cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    # Save final results (same as last incremental)
    results = incremental_results
    results_file = args.checkpoint_dir / "evaluation_results_final.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Final results saved to: {results_file}")
    
    # Print final summary
    print("\n🏆 FINAL SUMMARY:")
    print(f"  Total models evaluated: {results['models_evaluated']}")
    print(f"  Best WER: {results['summary']['best_wer_model']}")
    print(f"  Best CER: {results['summary']['best_cer_model']}")
    if results['summary']['base_model_wer'] is not None:
        print(f"  Base model WER: {results['summary']['base_model_wer']:.4f} ({results['summary']['base_model_wer']*100:.2f}%)")
    if results['summary']['latest_checkpoint_wer'] is not None:
        print(f"  Final checkpoint WER: {results['summary']['latest_checkpoint_wer']:.4f} ({results['summary']['latest_checkpoint_wer']*100:.2f}%)")
        improvement = results['summary']['base_model_wer'] - results['summary']['latest_checkpoint_wer']
        print(f"  Total improvement: {improvement:.4f} ({improvement*100:.2f}% absolute)")


if __name__ == "__main__":
    main()
