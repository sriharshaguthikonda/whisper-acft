import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

# Built-in exact matches (normalized forms)
DEFAULT_TRIVIAL_PHRASES = {
    # Core acknowledgement junk
    "ok",
    "okay",
    "ok doctor",
    "okay doctor",
    "ok doc",
    "okay doc",
    "ok dr",
    "okay dr",
    "yes",
    "yeah",
    "yep",
    "ya",
    "yes doctor",
    "yeah doctor",
    "yep doctor",
    "yes doc",
    "yeah doc",
    "yes dr",
    "yeah dr",
    "no",
    "nope",
    "nah",
    "no doctor",
    "no doc",
    "no dr",
    "right",
    "alright",
    "all right",
    "sure",
    "fine",
    # From your frequency list
    "hello",
    "hello doctor",
    "hello there",
    "hello there doctor",
    "thank you",
    "thank you doctor",
    "thank you very much",
    "okay all right",
    "okay alright",
    "okay thank you",
    "yes yes",
    "yeah yeah",
    "yeah yeah yeah",
    "no no",
    "no no no",
    "nothing",
    "nothing else",
    "no nothing",
    "normal",
    "i dont know",
    "i dont know doctor",
    "i dont think so",
    "i dont think so doctor",
    "i see",
    "i understand",
    "oh",
    # single-word junk that often becomes its own cue
    "and",
    "so",
    "you",
    # station/meta cues
    "enter the room",
    "two minutes remaining",
    "move on to the next station",
    "begin",
    # repeated prompts you listed (treat as junk by same policy)
    "could you please confirm your age for me",
    "could you please confirm your name and age for me",
    "what would you like me to call you",
    "any allergies",
    "any allergies by any chance",
    "any fever",
    "do you smoke",
    "do you drink alcohol",
    "does that make sense",
    "is that okay",
    "could you tell me a little bit more",
    "what do you do for a living",
    "what do you want to know",
    "thats it",
    "thats good",
    "not exactly",
    "like what",
    "what is that",
    "what is that doctor",
    "like what doctor",
}


def normalize_text(text: str) -> str:
    """Lowercase, strip, remove punctuation, and collapse internal whitespace."""
    text = text.lower().strip()
    # Remove punctuation characters
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
                removed += 1
                continue
            outfile.write(line)
            kept += 1
    return kept, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove manifest lines whose raw_transcription is a trivial phrase."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input
    output_path: Path = args.output or input_path.with_name(f"{input_path.stem}.filtered.jsonl")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    kept, removed = filter_manifest(input_path, output_path, DEFAULT_TRIVIAL_PHRASES)
    print(f"Completed. Kept {kept} lines, removed {removed}. Output: {output_path}")
    try:
        # Windows beep when done (frequency, duration ms)
        import winsound

        winsound.Beep(880, 400)
    except Exception:
        print("\a")


if __name__ == "__main__":
    main()
