#!/usr/bin/env python3
"""
Check if chunk files referenced in a manifest JSONL file actually exist.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm
import argparse

def check_manifest_files(manifest_path, sample_size=None):
    """Check if audio files in manifest exist."""
    
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        print(f"❌ Manifest file not found: {manifest_path}")
        return
    
    print(f"📋 Checking manifest: {manifest_path}")
    
    # Count total lines first
    with open(manifest_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)
    
    print(f"📊 Total entries: {total_lines}")
    
    missing_files = []
    existing_files = []
    invalid_entries = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        lines_to_check = total_lines if sample_size is None else min(sample_size, total_lines)
        
        for line_num, line in enumerate(tqdm(f, total=lines_to_check, desc="Checking files"), 1):
            if sample_size and line_num > sample_size:
                break
                
            try:
                data = json.loads(line.strip())
                audio_path = data.get('audio_path')
                
                if not audio_path:
                    invalid_entries.append((line_num, "No audio_path field"))
                    continue
                
                audio_file = Path(audio_path)
                if audio_file.exists():
                    existing_files.append(audio_path)
                else:
                    missing_files.append((line_num, audio_path))
                    
            except json.JSONDecodeError as e:
                invalid_entries.append((line_num, f"JSON decode error: {e}"))
            except Exception as e:
                invalid_entries.append((line_num, f"Error: {e}"))
    
    # Print results
    print(f"\n📈 RESULTS:")
    print(f"✅ Existing files: {len(existing_files):,}")
    print(f"❌ Missing files: {len(missing_files):,}")
    print(f"⚠️  Invalid entries: {len(invalid_entries):,}")
    
    if missing_files:
        print(f"\n❌ MISSING FILES (first 20):")
        for line_num, path in missing_files[:20]:
            print(f"  Line {line_num}: {path}")
        if len(missing_files) > 20:
            print(f"  ... and {len(missing_files) - 20} more")
    
    if invalid_entries:
        print(f"\n⚠️  INVALID ENTRIES:")
        for line_num, error in invalid_entries[:10]:
            print(f"  Line {line_num}: {error}")
        if len(invalid_entries) > 10:
            print(f"  ... and {len(invalid_entries) - 10} more")
    
    # Check file size statistics for existing files
    if existing_files:
        sizes = []
        for path in existing_files[:1000]:  # Sample first 1000 for speed
            try:
                sizes.append(Path(path).stat().st_size)
            except:
                pass
        
        if sizes:
            print(f"\n📊 FILE SIZE STATS (sample of {len(sizes)} files):")
            print(f"  Min size: {min(sizes):,} bytes")
            print(f"  Max size: {max(sizes):,} bytes")
            print(f"  Avg size: {sum(sizes)/len(sizes):,.0f} bytes")
    
    return {
        'total_entries': total_lines,
        'existing_files': len(existing_files),
        'missing_files': len(missing_files),
        'invalid_entries': len(invalid_entries),
        'missing_file_list': missing_files,
        'invalid_entry_list': invalid_entries
    }

def main():
    parser = argparse.ArgumentParser(description='Check if chunk files in manifest exist')
    parser.add_argument('manifest', help='Path to manifest JSONL file')
    parser.add_argument('--sample', type=int, help='Check only first N entries')
    parser.add_argument('--output', help='Save results to JSON file')
    
    args = parser.parse_args()
    
    results = check_manifest_files(args.manifest, args.sample)
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            # Convert lists to smaller samples for JSON output
            output_data = results.copy()
            output_data['missing_file_list'] = output_data['missing_file_list'][:100]
            output_data['invalid_entry_list'] = output_data['invalid_entry_list'][:50]
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")

if __name__ == "__main__":
    main()
