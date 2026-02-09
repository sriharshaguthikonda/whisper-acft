#!/usr/bin/env python3
"""Stage 5: Convert number words to digits in manifest raw_transcription.

Examples:
  "twenty one" -> "21"
  "five three two one" -> "5321"
  "twenty-one" -> "21"
  "five, three, two, one" -> "5321"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm


_UNITS: Dict[str, int] = {
    "zero": 0,
    "oh": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

_TEENS: Dict[str, int] = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}

_TENS: Dict[str, int] = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_SCALES: Dict[str, int] = {
    "hundred": 100,
    "thousand": 1_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
}

_NUM_WORDS = set(_UNITS) | set(_TEENS) | set(_TENS) | set(_SCALES) | {"and"}
_NUM_START_WORDS = _NUM_WORDS - {"and"}
_DIGIT_WORDS = set(_UNITS)

_NUM_PATTERN = re.compile(
    r"\b(?:"
    + "|".join(sorted(_NUM_START_WORDS, key=len, reverse=True))
    + r")\b(?:[\s,-]+\b(?:"
    + "|".join(sorted(_NUM_WORDS, key=len, reverse=True))
    + r")\b)*",
    re.IGNORECASE,
)


def _words_to_number(words: List[str]) -> Optional[int]:
    total = 0
    current = 0
    for w in words:
        if w == "and":
            continue
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TEENS:
            current += _TEENS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w in _SCALES:
            scale = _SCALES[w]
            if current == 0:
                current = 1
            if scale == 100:
                current *= scale
            else:
                total += current * scale
                current = 0
        else:
            return None
    return total + current


def _replace_match(match: re.Match) -> str:
    chunk = match.group(0)
    words = re.findall(r"[A-Za-z']+", chunk.lower())
    if not words:
        return chunk

    cleaned = [w for w in words if w in _NUM_WORDS]
    if not cleaned:
        return chunk

    if all(w in _DIGIT_WORDS for w in cleaned):
        return "".join(str(_UNITS[w]) for w in cleaned)

    num = _words_to_number(cleaned)
    if num is None:
        return chunk
    return str(num)


def normalize_number_words(text: str) -> str:
    return _NUM_PATTERN.sub(_replace_match, text)


def process_manifest(input_path: Path, output_path: Path, field: str) -> tuple[int, int]:
    total = 0
    changed = 0
    total_lines = sum(1 for _ in input_path.open("r", encoding="utf-8"))

    with input_path.open("r", encoding="utf-8") as infile, output_path.open("w", encoding="utf-8") as outfile:
        for line in tqdm(infile, total=total_lines, desc="Converting numbers"):
            if not line.strip():
                continue
            obj = json.loads(line)
            total += 1
            txt = obj.get(field, "")
            if isinstance(txt, str) and txt:
                new_txt = normalize_number_words(txt)
                if new_txt != txt:
                    obj[field] = new_txt
                    changed += 1
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return total, changed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert number words to digits in manifest text.")
    p.add_argument("--input", required=True, type=Path, help="Input JSONL manifest.")
    p.add_argument("--output", required=True, type=Path, help="Output JSONL manifest.")
    p.add_argument("--field", default="raw_transcription", help="Field to normalize (default: raw_transcription).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total, changed = process_manifest(args.input, args.output, args.field)
    print(f"Completed. Lines: {total}, changed: {changed}. Output: {args.output}")
    try:
        import winsound  # type: ignore

        winsound.Beep(880, 400)
    except Exception:
        print("\a")


if __name__ == "__main__":
    main()
