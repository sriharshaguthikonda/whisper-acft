#!/usr/bin/env python3
"""
Analyze worst performing audio files from evaluation results
"""

import json
from collections import defaultdict

def analyze_worst_performing_audio(evaluation_file):
    """Analyze evaluation results to find worst performing audio files"""
    
    with open(evaluation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    worst_cases = []
    file_stats = defaultdict(list)
    
    for checkpoint_data in data:
        checkpoint_name = checkpoint_data.get('checkpoint', 'unknown')
        results = checkpoint_data.get('results', [])
        
        for result in results:
            file_name = result.get('file', 'unknown')
            wer_raw = result.get('wer_raw', 0)
            wer_norm = result.get('wer_norm', 0)
            cer_norm = result.get('cer_norm', 0)
            ground_truth = result.get('ground_truth', '')
            transcription = result.get('transcription', '')
            avg_confidence = result.get('avg_confidence', 0)
            
            worst_cases.append({
                'file': file_name,
                'checkpoint': checkpoint_name,
                'wer_raw': wer_raw,
                'wer_norm': wer_norm,
                'cer_norm': cer_norm,
                'ground_truth': ground_truth,
                'transcription': transcription,
                'avg_confidence': avg_confidence
            })
            
            file_stats[file_name].append({
                'checkpoint': checkpoint_name,
                'wer_raw': wer_raw,
                'wer_norm': wer_norm,
                'cer_norm': cer_norm,
                'transcription': transcription
            })
    
    # Sort by worst WER
    worst_cases.sort(key=lambda x: x['wer_raw'], reverse=True)
    
    # Calculate file-level statistics
    file_summary = {}
    for file_name, stats in file_stats.items():
        wer_values = [s['wer_raw'] for s in stats]
        file_summary[file_name] = {
            'avg_wer': sum(wer_values) / len(wer_values),
            'max_wer': max(wer_values),
            'min_wer': min(wer_values),
            'num_checkpoints': len(stats),
            'stats': stats
        }
    
    return worst_cases, file_summary

def main():
    evaluation_file = "i:\\whisper-acft\\enhanced_checkpoint_evaluation_projout_fixed.json"
    
    print("Analyzing worst performing audio files...")
    worst_cases, file_summary = analyze_worst_performing_audio(evaluation_file)
    
    print("\n" + "="*80)
    print("TOP 20 WORST PERFORMING AUDIO CASES (WER = 1.0)")
    print("="*80)
    
    wer_1_cases = [case for case in worst_cases if case['wer_raw'] == 1.0]
    
    for i, case in enumerate(wer_1_cases[:20], 1):
        print(f"\n{i}. File: {case['file']}")
        print(f"   Checkpoint: {case['checkpoint']}")
        print(f"   WER: {case['wer_raw']:.3f}")
        print(f"   Confidence: {case['avg_confidence']:.3f}")
        print(f"   Ground Truth: {case['ground_truth'][:100]}...")
        print(f"   Transcription: '{case['transcription']}'")
    
    print("\n" + "="*80)
    print("FILES WITH HIGHEST AVERAGE WER ACROSS ALL CHECKPOINTS")
    print("="*80)
    
    # Sort files by average WER
    sorted_files = sorted(file_summary.items(), key=lambda x: x[1]['avg_wer'], reverse=True)
    
    for i, (file_name, stats) in enumerate(sorted_files[:15], 1):
        print(f"\n{i}. File: {file_name}")
        print(f"   Average WER: {stats['avg_wer']:.3f}")
        print(f"   Max WER: {stats['max_wer']:.3f}")
        print(f"   Min WER: {stats['min_wer']:.3f}")
        print(f"   Number of checkpoints: {stats['num_checkpoints']}")
        
        # Show worst transcription for this file
        worst_stat = max(stats['stats'], key=lambda x: x['wer_raw'])
        print(f"   Worst transcription: '{worst_stat['transcription']}'")
    
    print("\n" + "="*80)
    print("COMMON PATTERNS IN WORST TRANSCRIPTIONS")
    print("="*80)
    
    # Analyze common patterns in worst transcriptions
    worst_transcriptions = [case['transcription'] for case in wer_1_cases[:50]]
    pattern_counts = defaultdict(int)
    
    for trans in worst_transcriptions:
        if not trans.strip():
            pattern_counts['EMPTY_TRANSCRIPTION'] += 1
        elif trans.lower() in ['i am not a fan', 'i am not a man', 'i am not a person']:
            pattern_counts['NEGATIVE_FAN_PATTERN'] += 1
        elif len(trans.strip()) <= 3:
            pattern_counts['VERY_SHORT'] += 1
        elif 'not a' in trans.lower():
            pattern_counts['NEGATION_PATTERN'] += 1
        else:
            pattern_counts['OTHER'] += 1
    
    print("\nPattern analysis for WER=1.0 cases:")
    for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {pattern}: {count} cases")
    
    # Save detailed results
    output_file = "worst_audio_analysis.json"
    results = {
        'top_worst_cases': wer_1_cases[:50],
        'file_summary': dict(sorted_files[:20]),
        'pattern_analysis': dict(pattern_counts)
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
