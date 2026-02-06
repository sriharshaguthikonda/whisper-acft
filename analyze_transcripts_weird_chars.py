#!/usr/bin/env python3
"""
Scan transcription JSONs for non-language characters in transcript text fields.

Targets fields inside the transcription response (groq_response/response/result):
- text
- words[].word
- segments[].text
- segments[].words[].word

Outputs:
- weird_chars_report.jsonl (per-hit detail)
- weird_chars_summary.csv (aggregate counts)
"""

from __future__ import annotations

import argparse
import csv
import json
import string
import unicodedata
from collections import Counter
from pathlib import Path


ASCII_SYMBOLS = set(string.punctuation)
ASCII_LETTERS = set(string.ascii_letters)


def pick_response_node(data: dict):
    for key in ("groq_response", "response", "result", "transcription", "output"):
        node = data.get(key)
        if isinstance(node, dict):
            return node
    return data


def iter_text_fields(node, prefix: str = ""):
    if isinstance(node, dict):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else k
            if k in ("text", "word", "transcript") and isinstance(v, str):
                yield path, v
            if k in ("words", "segments") and isinstance(v, list):
                for i, item in enumerate(v):
                    yield from iter_text_fields(item, f"{path}[{i}]")
            elif isinstance(v, dict):
                # Allow nested objects inside response.
                yield from iter_text_fields(v, path)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            yield from iter_text_fields(item, f"{prefix}[{i}]")


def is_allowed_char(ch: str, allow_ascii_symbols: bool, strict_english: bool) -> bool:
    if ch in ("\t", "\n", "\r"):
        return True
    if strict_english:
        if ch in ASCII_LETTERS:
            return True
        if ch in ASCII_SYMBOLS:
            return True
        if ch.isspace():
            return True
        return False
    cat = unicodedata.category(ch)
    if cat[0] in ("L", "M", "N", "P"):
        return True
    if cat.startswith("Z"):
        return True
    if allow_ascii_symbols and ch in ASCII_SYMBOLS:
        return True
    return False


def scan_text(s: str, allow_ascii_symbols: bool, strict_english: bool):
    for idx, ch in enumerate(s):
        if not is_allowed_char(ch, allow_ascii_symbols, strict_english):
            yield idx, ch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with transcription JSON files")
    ap.add_argument("--out_dir", default=None, help="Output folder (default: in_dir)")
    ap.add_argument("--max_hits_per_file", type=int, default=50)
    ap.add_argument("--allow_ascii_symbols", type=int, default=1)
    ap.add_argument(
        "--strict_english",
        type=int,
        default=0,
        help="If 1: allow only ASCII letters + punctuation + whitespace",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "weird_chars_report.jsonl"
    summary_path = out_dir / "weird_chars_summary.csv"

    allow_ascii_symbols = bool(int(args.allow_ascii_symbols))
    strict_english = bool(int(args.strict_english))

    counts = Counter()
    sample_file = {}

    with report_path.open("w", encoding="utf-8") as report_f:
        for json_path in sorted(in_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                rec = {
                    "file": str(json_path),
                    "error": f"failed_to_read_json: {e}",
                }
                report_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            resp = pick_response_node(data)
            hits_in_file = 0

            for field_path, text in iter_text_fields(resp):
                for idx, ch in scan_text(text, allow_ascii_symbols, strict_english):
                    cat = unicodedata.category(ch)
                    name = unicodedata.name(ch, "UNKNOWN")
                    cp = f"U+{ord(ch):04X}"
                    key = (ch, cp, cat, name)
                    counts[key] += 1
                    sample_file.setdefault(key, json_path.name)

                    if hits_in_file < int(args.max_hits_per_file):
                        start = max(0, idx - 20)
                        end = min(len(text), idx + 20)
                        rec = {
                            "file": json_path.name,
                            "path": field_path,
                            "index": idx,
                            "char": ch,
                            "codepoint": cp,
                            "category": cat,
                            "name": name,
                            "context": text[start:end],
                        }
                        report_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        hits_in_file += 1

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["char", "codepoint", "category", "name", "count", "example_file"])
        for (ch, cp, cat, name), count in sorted(
            counts.items(), key=lambda x: (-x[1], x[0][1])
        ):
            w.writerow([ch, cp, cat, name, count, sample_file.get((ch, cp, cat, name), "")])

    print("Wrote:")
    print(" -", report_path)
    print(" -", summary_path)


if __name__ == "__main__":
    main()
