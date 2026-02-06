#!/usr/bin/env python3
"""Quick randomization check on different sections of the manifest"""

import json
from pathlib import Path

def check_section(manifest_path, start_line, section_name):
    """Check a specific section of the manifest"""
    print(f"\n=== {section_name} (lines {start_line}-{start_line+19}) ===")
    
    sources = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= start_line and i < start_line + 20:
                if line.strip():
                    try:
                        row = json.loads(line)
                        source = row.get("source_audio", "").split("\\")[-1] if "\\" in row.get("source_audio", "") else row.get("source_audio", "")
                        sources.append(source)
                    except:
                        pass
    
    # Check for duplicates
    unique_sources = set(sources)
    duplicates = len(sources) - len(unique_sources)
    
    print(f"Sources in this section: {len(sources)}")
    print(f"Unique sources: {len(unique_sources)}")
    print(f"Duplicates: {duplicates}")
    print(f"Duplicate rate: {duplicates/len(sources)*100:.1f}%")
    
    # Show first few sources
    print("First 10 sources:")
    for i, source in enumerate(sources[:10]):
        print(f"  {i+1:2d}. {source[:50]}")

def main():
    manifest_path = Path("I:/Record_chunks/combined_all_manifests.jsonl")
    
    print("Quick randomization check on different sections...")
    
    # Check different sections
    sections = [
        (0, "Beginning"),
        (50000, "Middle"),
        (100000, "Later"),
        (140000, "End")
    ]
    
    for start_line, name in sections:
        check_section(manifest_path, start_line, name)

if __name__ == "__main__":
    main()
