"""
Stage 3b: Filter manifest to keep only English language entries

Usage:
python stage_3b_filter_english_only.py --input I:\Record_chunks\pairs_manifest_sorted_by_scores.jsonl --output I:\Record_chunks\pairs_manifest_sorted_by_scores_english_only.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from tqdm.auto import tqdm

def is_english_text(text, min_english_ratio=0.7):
    """
    Check if text is primarily English.
    
    Args:
        text: Text to check
        min_english_ratio: Minimum ratio of English characters to consider text as English
    
    Returns:
        bool: True if text is primarily English
    """
    if not text or not isinstance(text, str):
        return False
    
    # Remove whitespace and convert to lowercase for analysis
    clean_text = text.strip().lower()
    if not clean_text:
        return False
    
    # Count English characters (letters, numbers, common punctuation)
    english_chars = 0
    total_chars = len(clean_text)
    
    for char in clean_text:
        # English letters (a-z)
        if 'a' <= char <= 'z':
            english_chars += 1
        # Numbers
        elif '0' <= char <= '9':
            english_chars += 1
        # Common English punctuation and medical symbols
        elif char in ".,!?;:'\"()-/[]{}@#$%&*+=<>\\":
            english_chars += 1
    
    # Calculate ratio of English characters
    if total_chars == 0:
        return False
    
    english_ratio = english_chars / total_chars
    
    # Additional check: Look for non-ASCII characters which often indicate other languages
    non_ascii_count = sum(1 for char in clean_text if ord(char) > 127)
    non_ascii_ratio = non_ascii_count / total_chars if total_chars > 0 else 0
    
    # Text is English if:
    # 1. High ratio of English characters, AND
    # 2. Low ratio of non-ASCII characters
    return english_ratio >= min_english_ratio and non_ascii_ratio <= 0.3

def contains_non_english_words(text):
    """
    Check for common non-English words/patterns.
    This is a supplementary check for obvious non-English content.
    """
    if not text or not isinstance(text, str):
        return False
    
    text_lower = text.lower()
    
    # Common indicators of non-English text
    non_english_patterns = [
        # Hindi/Devanagari script range (Unicode)
        r'[\u0900-\u097F]',
        # Arabic script range
        r'[\u0600-\u06FF]',
        # Chinese/Japanese/Korean script ranges
        r'[\u4E00-\u9FFF]',
        r'[\u3040-\u309F]',  # Hiragana
        r'[\u30A0-\u30FF]',  # Katakana
        # Cyrillic script
        r'[\u0400-\u04FF]',
        # Thai script
        r'[\u0E00-\u0E7F]',
        # Common non-English words that might appear in English text
        r'\b(अगर|आप|संपर्ट|करेंगे|हमें|एक|द्रौड़ी|हो|नामा|सीनिकों|के)\b',
    ]
    
    for pattern in non_english_patterns:
        if re.search(pattern, text):
            return True
    
    return False

def filter_english_manifest(input_path, output_path, min_english_ratio=0.7):
    """
    Filter manifest to keep only English entries.
    
    Args:
        input_path: Path to input manifest file
        output_path: Path to output manifest file
        min_english_ratio: Minimum ratio of English characters
    """
    print(f"Loading manifest from: {input_path}")
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_entries = 0
    english_entries = 0
    non_english_entries = 0
    samples_removed = []
    
    print("Filtering manifest for English-only content...")
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(tqdm(infile, desc="Processing entries")):
            total_entries += 1
            
            try:
                entry = json.loads(line.strip())
                transcription = entry.get('raw_transcription', '')
                
                # Check if transcription is primarily English
                is_english = is_english_text(transcription, min_english_ratio)
                has_non_english_patterns = contains_non_english_words(transcription)
                
                if is_english and not has_non_english_patterns:
                    # Keep English entry
                    outfile.write(line)
                    english_entries += 1
                else:
                    # Skip non-English entry
                    non_english_entries += 1
                    # Store some examples of removed entries
                    if len(samples_removed) < 10:
                        samples_removed.append({
                            'line': line_num + 1,
                            'transcription': transcription,
                            'audio_path': entry.get('audio_path', ''),
                            'reason': 'Non-English patterns' if has_non_english_patterns else f'Low English ratio ({min_english_ratio})'
                        })
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num + 1}: {e}")
                continue
    
    print(f"\n" + "="*60)
    print("FILTERING RESULTS")
    print("="*60)
    print(f"Total entries processed: {total_entries}")
    print(f"English entries kept: {english_entries} ({100*english_entries/total_entries:.1f}%)")
    print(f"Non-English entries removed: {non_english_entries} ({100*non_english_entries/total_entries:.1f}%)")
    print(f"Output saved to: {output_path}")
    
    if samples_removed:
        print(f"\nSAMPLE REMOVED ENTRIES:")
        print("-" * 40)
        for i, sample in enumerate(samples_removed, 1):
            print(f"{i}. Line {sample['line']} ({sample['reason']}):")
            print(f"   Audio: {sample['audio_path']}")
            print(f"   Text: {sample['transcription'][:100]}{'...' if len(sample['transcription']) > 100 else ''}")
            print()
    
    return english_entries, non_english_entries

def main():
    parser = argparse.ArgumentParser(description='Filter manifest to keep only English entries')
    parser.add_argument('--input', required=True, help='Input manifest file path')
    parser.add_argument('--output', required=True, help='Output manifest file path')
    parser.add_argument('--min-english-ratio', type=float, default=0.7, 
                       help='Minimum ratio of English characters (default: 0.7)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("STAGE 3b: Filter Manifest for English-Only Content")
    print("="*60)
    
    # Validate input file exists
    if not Path(args.input).exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Filter manifest
    english_entries, non_english_entries = filter_english_manifest(
        args.input, 
        args.output, 
        args.min_english_ratio
    )
    
    print("="*60)
    print("STAGE 3b COMPLETED SUCCESSFULLY")
    print(f"English-only manifest: {args.output}")
    print("="*60)

if __name__ == "__main__":
    main()
