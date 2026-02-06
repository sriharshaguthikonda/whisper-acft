"""
Stage 4: Delete common filler words and segments from manifest

Usage:
python stage_4_Delete_common_fillers_words_from_manifest.py --input I:\Record_chunks\pairs_manifest_stereo_english_only.jsonl --output I:\Record_chunks\pairs_manifest_stereo_english_only_filtered.jsonl --state-file I:\Transcriptions_patched_corrected\Most_common_segments_state.json --min-frequency 3

This script removes 90% of manifest lines whose raw_transcription matches segments 
from the state file with frequency >= min_frequency (keeps 10% for diversity).
"""


import argparse
import json
import random
import re
from pathlib import Path

from tqdm import tqdm


def normalize_text(text: str) -> str:
    """Lowercase, strip, remove punctuation, and collapse internal whitespace."""
    text = text.lower().strip()
    # Remove punctuation characters
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_segments_from_state(state_path: Path, min_frequency: int = 4) -> set[str]:
    """Load segments from state file with frequency >= min_frequency."""
    if not state_path.exists():
        print(f"Warning: State file not found: {state_path}")
        return set()
    
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        counts = state_data.get('counts', {})
        segments = set()
        
        for segment, count in counts.items():
            if count >= min_frequency:
                normalized = normalize_text(segment)
                segments.add(normalized)
        
        print(f"Loaded {len(segments)} segments from {state_path} with frequency >= {min_frequency}")
        return segments
    
    except Exception as e:
        print(f"Error loading state file {state_path}: {e}")
        return set()


def filter_manifest(input_path: Path, output_path: Path, trivial_phrases: set[str]) -> tuple[int, int]:
    total_lines = sum(1 for _ in input_path.open("r", encoding="utf-8"))
    kept = 0
    removed = 0

    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in tqdm(infile, total=total_lines, desc="Filtering lines"):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("raw_transcription", "")
            normalized = normalize_text(text)
            if normalized in trivial_phrases:
                # Remove 90% of trivial phrases, keep 10%
                if random.random() < 0.9:
                    removed += 1
                    continue
            outfile.write(line)
            kept += 1
    return kept, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove 90% of manifest lines whose raw_transcription is a trivial phrase (keeps 10%)."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input JSONL manifest (one JSON object per line).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write cleaned manifest. Defaults to <input>.filtered.jsonl.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        help="Path to state file containing segment counts.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=4,
        help="Minimum frequency for segments to be included (default: 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output or input_path.with_name(f"{input_path.stem}.filtered.jsonl")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Load segments from state file if provided
    trivial_phrases = set()
    if args.state_file:
        trivial_phrases = load_segments_from_state(args.state_file, args.min_frequency)
    else:
        print("No state file provided. No filtering will be applied.")

    kept, removed = filter_manifest(input_path, output_path, trivial_phrases)
    print(f"Completed. Kept {kept} lines, removed {removed}. Output: {output_path}")
    try:
        # Windows beep when done (frequency, duration ms)
        import winsound

        winsound.Beep(880, 400)
    except Exception:
        print("\a")


if __name__ == "__main__":
    main()
