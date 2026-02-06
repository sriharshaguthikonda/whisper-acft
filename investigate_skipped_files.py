#!/usr/bin/env python3
"""Investigate skipped files and bad headers from stage 13 training"""

import os
import json
import soundfile as sf
from pathlib import Path
from collections import defaultdict, Counter
import hashlib

def get_file_stem(path):
    """Get the filename without extension"""
    return Path(path).stem

def get_file_stem_with_dir(path):
    """Get a more detailed stem including parent directory"""
    p = Path(path)
    return f"{p.parent.name}/{p.stem}" if p.parent.name else p.stem

def analyze_file_groups(file_paths):
    """Analyze files by stems to find similar groups"""
    stem_groups = defaultdict(list)
    dir_stem_groups = defaultdict(list)
    
    for path in file_paths:
        stem = get_file_stem(path)
        dir_stem = get_file_stem_with_dir(path)
        
        stem_groups[stem].append(path)
        dir_stem_groups[dir_stem].append(path)
    
    # Find stems with multiple files (potential duplicates/versions)
    multi_file_stems = {stem: files for stem, files in stem_groups.items() if len(files) > 1}
    multi_file_dir_stems = {stem: files for stem, files in dir_stem_groups.items() if len(files) > 1}
    
    return {
        'stem_groups': stem_groups,
        'dir_stem_groups': dir_stem_groups,
        'multi_file_stems': multi_file_stems,
        'multi_file_dir_stems': multi_file_dir_stems
    }

def investigate_file_details(file_path):
    """Check what's wrong with a specific audio file"""
    result = {
        "path": str(file_path),
        "exists": os.path.exists(file_path),
        "readable": False,
        "samplerate": None,
        "duration": None,
        "frames": None,
        "error": None,
        "file_size": None
    }
    
    if os.path.exists(file_path):
        try:
            result["file_size"] = os.path.getsize(file_path)
        except:
            pass
    
    if not os.path.exists(file_path):
        result["error"] = "File does not exist"
        return result
    
    try:
        info = sf.info(file_path)
        result["readable"] = True
        result["samplerate"] = info.samplerate
        result["frames"] = info.frames
        result["duration"] = float(info.frames) / float(info.samplerate)
        
        # Check specific issues that would cause bad_header
        issues = []
        if info.samplerate != 16000:
            issues.append(f"Wrong sample rate: {info.samplerate} (expected 16000)")
        
        dur = result["duration"]
        if dur <= 0.0:
            issues.append(f"Non-positive duration: {dur}")
        elif dur > 30.0:  # MAX_AUDIO_SECONDS from training script
            issues.append(f"Duration too long: {dur:.2f}s (max 30.0s)")
        
        if issues:
            result["error"] = "; ".join(issues)
        
    except Exception as e:
        result["error"] = f"SoundFile error: {str(e)}"
    
    return result

def main():
    manifest_path = "I:/Record_chunks/train_manifest.jsonl"
    trained_path = "i:/Record_chunks/trained_stage1.jsonl"
    
    print("=" * 80)
    print("INVESTIGATION OF SKIPPED FILES AND BAD HEADERS")
    print("=" * 80)
    
    # Load trained files
    trained_files = set()
    if os.path.exists(trained_path):
        with open(trained_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        ap = row.get("audio_path")
                        if ap:
                            trained_files.add(ap)
                    except:
                        pass
    
    print(f"Already trained files: {len(trained_files)}")
    
    # Analyze manifest files and print bad headers immediately
    total_files = 0
    missing_files = []
    bad_header_files = []
    trained_skipped_files = []
    
    print(f"\nScanning manifest and printing bad headers immediately:")
    print("-" * 50)
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                row = json.loads(line)
                total_files += 1
                
                audio_path = row.get("audio_path")
                if not audio_path:
                    continue
                
                # Check if already trained
                if audio_path in trained_files:
                    trained_skipped_files.append(audio_path)
                    continue
                
                # Check if file exists
                if not os.path.exists(audio_path):
                    missing_files.append(audio_path)
                    print(f"MISSING: {audio_path}")
                    continue
                
                # Check audio header (same logic as training script)
                try:
                    info = sf.info(audio_path)
                    if info.samplerate != 16000:
                        bad_header_files.append(audio_path)
                        print(f"BAD HEADER (wrong sample rate {info.samplerate}): {audio_path}")
                        continue
                    
                    dur = float(info.frames) / float(info.samplerate)
                    if dur <= 0.0 or dur > 30.0:
                        bad_header_files.append(audio_path)
                        print(f"BAD HEADER (duration {dur:.2f}s): {audio_path}")
                        continue
                        
                except Exception as e:
                    bad_header_files.append(audio_path)
                    print(f"BAD HEADER (soundfile error): {audio_path} - {str(e)}")
                    continue
                
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
    
    print("-" * 50)
    print(f"\nFINAL SUMMARY:")
    print(f"Total files in manifest: {total_files}")
    print(f"Already trained (skipped): {len(trained_skipped_files)}")
    print(f"Missing files: {len(missing_files)}")
    print(f"Bad header files: {len(bad_header_files)}")
    print(f"Usable files: {total_files - len(trained_skipped_files) - len(missing_files) - len(bad_header_files)}")
    
    # Analyze file stems for patterns
    print(f"\n" + "=" * 80)
    print("STEM ANALYSIS FOR SKIPPED FILES")
    print("=" * 80)
    
    # Analyze trained_skipped files
    if trained_skipped_files:
        print(f"\nTRAINED SKIPPED FILES ({len(trained_skipped_files)}):")
        trained_analysis = analyze_file_groups(trained_skipped_files)
        
        print(f"Unique stems: {len(trained_analysis['stem_groups'])}")
        print(f"Stems with multiple files: {len(trained_analysis['multi_file_stems'])}")
        
        if trained_analysis['multi_file_stems']:
            print(f"\nTop 20 stems with most trained files:")
            sorted_stems = sorted(trained_analysis['multi_file_stems'].items(), 
                                key=lambda x: len(x[1]), reverse=True)[:20]
            for stem, files in sorted_stems:
                print(f"  {stem}: {len(files)} files")
                if len(files) <= 5:
                    for f in files:
                        print(f"    - {f}")
                else:
                    print(f"    - {files[0]}")
                    print(f"    - ... and {len(files)-1} more")
    
    # Analyze missing files
    if missing_files:
        print(f"\nMISSING FILES ({len(missing_files)}):")
        missing_analysis = analyze_file_groups(missing_files)
        
        print(f"Unique stems: {len(missing_analysis['stem_groups'])}")
        print(f"Stems with multiple files: {len(missing_analysis['multi_file_stems'])}")
        
        if missing_analysis['multi_file_stems']:
            print(f"\nTop 10 stems with most missing files:")
            sorted_stems = sorted(missing_analysis['multi_file_stems'].items(), 
                                key=lambda x: len(x[1]), reverse=True)[:10]
            for stem, files in sorted_stems:
                print(f"  {stem}: {len(files)} files")
    
    # Analyze bad header files
    if bad_header_files:
        print(f"\nBAD HEADER FILES ({len(bad_header_files)}):")
        bad_header_analysis = analyze_file_groups(bad_header_files)
        
        print(f"Unique stems: {len(bad_header_analysis['stem_groups'])}")
        print(f"Stems with multiple files: {len(bad_header_analysis['multi_file_stems'])}")
        
        if bad_header_analysis['multi_file_stems']:
            print(f"\nTop 10 stems with most bad header files:")
            sorted_stems = sorted(bad_header_analysis['multi_file_stems'].items(), 
                                key=lambda x: len(x[1]), reverse=True)[:10]
            for stem, files in sorted_stems:
                print(f"  {stem}: {len(files)} files")
        
        # Print all bad header files with their issues
        print(f"\nALL BAD HEADER FILES ({len(bad_header_files)}):")
        issue_counts = Counter()
        for i, file_path in enumerate(bad_header_files):
            details = investigate_file_details(file_path)
            issue = details.get("error", "Unknown")
            issue_counts[issue] += 1
            print(f"  {i+1:3d}. {file_path}")
            print(f"       Issue: {issue}")
        
        print(f"\nBad header issue summary:")
        for issue, count in issue_counts.most_common():
            print(f"  {count}: {issue}")
    
    # Check for overlap between categories
    print(f"\n" + "=" * 80)
    print("OVERLAP ANALYSIS")
    print("=" * 80)
    
    # Check if same stems appear in multiple categories
    all_skipped = trained_skipped_files + missing_files + bad_header_files
    all_skipped_analysis = analyze_file_groups(all_skipped)
    
    problematic_stems = set()
    for stem, files in all_skipped_analysis['multi_file_stems'].items():
        categories = set()
        for f in files:
            if f in trained_skipped_files:
                categories.add("trained")
            if f in missing_files:
                categories.add("missing")
            if f in bad_header_files:
                categories.add("bad_header")
        
        if len(categories) > 1:  # Stem appears in multiple skip categories
            problematic_stems.add((stem, files, categories))
    
    if problematic_stems:
        print(f"\nStems appearing in multiple skip categories:")
        for stem, files, categories in sorted(problematic_stems, key=lambda x: len(x[1]), reverse=True)[:10]:
            print(f"  {stem}: {len(files)} files, categories: {', '.join(sorted(categories))}")
    
    # Save detailed report
    report = {
        "summary": {
            "total_files": total_files,
            "trained_skipped": len(trained_skipped_files),
            "missing": len(missing_files),
            "bad_header": len(bad_header_files),
            "usable": total_files - len(trained_skipped_files) - len(missing_files) - len(bad_header_files)
        },
        "trained_skipped_stems": len(analyze_file_groups(trained_skipped_files)['stem_groups']) if trained_skipped_files else 0,
        "missing_stems": len(analyze_file_groups(missing_files)['stem_groups']) if missing_files else 0,
        "bad_header_stems": len(analyze_file_groups(bad_header_files)['stem_groups']) if bad_header_files else 0,
        "problematic_multi_category_stems": len(problematic_stems)
    }
    
    with open("skip_investigation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: skip_investigation_report.json")
    print(f"\nInvestigation complete!")

if __name__ == "__main__":
    main()
