#!/usr/bin/env python3
"""
Simple script to calculate WER from existing prediction JSON files.
This script reads the evaluation JSON files and recalculates WER metrics.
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from jiwer import wer as jiwer_wer, cer as jiwer_cer

def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate."""
    if not reference.strip() or not hypothesis.strip():
        return 1.0 if reference.strip() else 0.0
    return jiwer_wer(reference, hypothesis)

def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate."""
    if not reference.strip() or not hypothesis.strip():
        return 1.0 if reference.strip() else 0.0
    return jiwer_cer(reference, hypothesis)

def load_predictions(json_path: str) -> Dict[str, Any]:
    """Load predictions from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_metrics(predictions_data) -> Dict[str, float]:
    """Calculate WER and CER metrics from predictions."""
    total_wer_target = 0.0
    total_wer_other = 0.0
    total_cer_target = 0.0
    total_cer_other = 0.0
    total_samples = 0
    wer_target_list = []
    wer_other_list = []
    cer_target_list = []
    cer_other_list = []
    wins = 0
    
    # Handle both list and dict formats
    if isinstance(predictions_data, list):
        # Skip metadata entries
        predictions = [item for item in predictions_data if 'mix_key' in item and 'predictions' in item]
    else:
        predictions = predictions_data.values() if isinstance(predictions_data, dict) else []
    
    # Process each prediction
    for data in predictions:
        if isinstance(data, dict) and 'predictions' in data:
            # Get the first model's predictions
            model_predictions = next(iter(data['predictions'].values()))
            if isinstance(model_predictions, dict):
                # Extract existing metrics
                if 'wer_target' in model_predictions:
                    wer_target_list.append(model_predictions['wer_target'])
                    total_wer_target += model_predictions['wer_target']
                
                if 'wer_other' in model_predictions:
                    wer_other_list.append(model_predictions['wer_other'])
                    total_wer_other += model_predictions['wer_other']
                
                if 'cer_target' in model_predictions:
                    cer_target_list.append(model_predictions['cer_target'])
                    total_cer_target += model_predictions['cer_target']
                
                if 'cer_other' in model_predictions:
                    cer_other_list.append(model_predictions['cer_other'])
                    total_cer_other += model_predictions['cer_other']
                
                if 'win_target_closer' in model_predictions:
                    if model_predictions['win_target_closer']:
                        wins += 1
                
                total_samples += 1
    
    # Calculate averages
    avg_wer_target = total_wer_target / total_samples if total_samples > 0 else 0.0
    avg_wer_other = total_wer_other / total_samples if total_samples > 0 else 0.0
    avg_cer_target = total_cer_target / total_samples if total_samples > 0 else 0.0
    avg_cer_other = total_cer_other / total_samples if total_samples > 0 else 0.0
    win_rate = wins / total_samples if total_samples > 0 else 0.0
    
    # Calculate percentiles for WER
    wer_target_percentiles = {}
    wer_other_percentiles = {}
    if wer_target_list:
        wer_target_percentiles = {
            'wer_target_50th': np.percentile(wer_target_list, 50),
            'wer_target_75th': np.percentile(wer_target_list, 75),
            'wer_target_90th': np.percentile(wer_target_list, 90),
            'wer_target_95th': np.percentile(wer_target_list, 95)
        }
    if wer_other_list:
        wer_other_percentiles = {
            'wer_other_50th': np.percentile(wer_other_list, 50),
            'wer_other_75th': np.percentile(wer_other_list, 75),
            'wer_other_90th': np.percentile(wer_other_list, 90),
            'wer_other_95th': np.percentile(wer_other_list, 95)
        }
    
    return {
        'avg_wer_target': avg_wer_target,
        'avg_wer_other': avg_wer_other,
        'avg_cer_target': avg_cer_target,
        'avg_cer_other': avg_cer_other,
        'win_rate': win_rate,
        'total_samples': total_samples,
        'wer_target_std': np.std(wer_target_list) if wer_target_list else 0.0,
        'wer_other_std': np.std(wer_other_list) if wer_other_list else 0.0,
        'cer_target_std': np.std(cer_target_list) if cer_target_list else 0.0,
        'cer_other_std': np.std(cer_other_list) if cer_other_list else 0.0,
        **wer_target_percentiles,
        **wer_other_percentiles
    }

def main():
    parser = argparse.ArgumentParser(description='Calculate WER/CER from prediction JSON')
    parser.add_argument('--input_json', required=True, help='Input JSON file with predictions')
    parser.add_argument('--output_json', required=True, help='Output JSON file for metrics')
    parser.add_argument('--compare_to', help='Optional: Compare to another JSON file')
    
    args = parser.parse_args()
    
    # Load predictions
    print(f"Loading predictions from {args.input_json}")
    predictions = load_predictions(args.input_json)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions)
    
    # Add file info
    metrics['input_file'] = args.input_json
    metrics['total_predictions'] = len(predictions) if isinstance(predictions, (list, dict)) else 0
    
    # If comparison file provided, calculate comparison metrics
    if args.compare_to:
        print(f"Loading comparison from {args.compare_to}")
        compare_predictions = load_predictions(args.compare_to)
        compare_metrics = calculate_metrics(compare_predictions)
        
        # Calculate improvement
        wer_improvement = ((compare_metrics['avg_wer_target'] - metrics['avg_wer_target']) / compare_metrics['avg_wer_target']) * 100 if compare_metrics['avg_wer_target'] > 0 else 0
        cer_improvement = ((compare_metrics['avg_cer_target'] - metrics['avg_cer_target']) / compare_metrics['avg_cer_target']) * 100 if compare_metrics['avg_cer_target'] > 0 else 0
        
        metrics['comparison'] = {
            'compare_file': args.compare_to,
            'compare_avg_wer_target': compare_metrics['avg_wer_target'],
            'compare_avg_wer_other': compare_metrics['avg_wer_other'],
            'compare_avg_cer_target': compare_metrics['avg_cer_target'],
            'compare_avg_cer_other': compare_metrics['avg_cer_other'],
            'compare_win_rate': compare_metrics['win_rate'],
            'wer_improvement_percent': wer_improvement,
            'cer_improvement_percent': cer_improvement
        }
    
    # Save results
    print(f"Saving results to {args.output_json}")
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n=== Results ===")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Average WER (Target): {metrics['avg_wer_target']:.4f}")
    print(f"Average WER (Other): {metrics['avg_wer_other']:.4f}")
    print(f"Average CER (Target): {metrics['avg_cer_target']:.4f}")
    print(f"Average CER (Other): {metrics['avg_cer_other']:.4f}")
    print(f"Win Rate (Target closer): {metrics['win_rate']:.4f}")
    print(f"WER Target Std Dev: {metrics['wer_target_std']:.4f}")
    print(f"WER Other Std Dev: {metrics['wer_other_std']:.4f}")
    print(f"CER Target Std Dev: {metrics['cer_target_std']:.4f}")
    print(f"CER Other Std Dev: {metrics['cer_other_std']:.4f}")
    
    if 'wer_target_50th' in metrics:
        print(f"WER Target 50th percentile: {metrics['wer_target_50th']:.4f}")
        print(f"WER Target 75th percentile: {metrics['wer_target_75th']:.4f}")
        print(f"WER Target 90th percentile: {metrics['wer_target_90th']:.4f}")
        print(f"WER Target 95th percentile: {metrics['wer_target_95th']:.4f}")
    
    if 'wer_other_50th' in metrics:
        print(f"WER Other 50th percentile: {metrics['wer_other_50th']:.4f}")
        print(f"WER Other 75th percentile: {metrics['wer_other_75th']:.4f}")
        print(f"WER Other 90th percentile: {metrics['wer_other_90th']:.4f}")
        print(f"WER Other 95th percentile: {metrics['wer_other_95th']:.4f}")
    
    if 'comparison' in metrics:
        print(f"\n=== Comparison ===")
        print(f"WER improvement: {metrics['comparison']['wer_improvement_percent']:.2f}%")
        print(f"CER improvement: {metrics['comparison']['cer_improvement_percent']:.2f}%")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
