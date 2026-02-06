#!/usr/bin/env python3
"""
Remove non-language characters from transcription JSONs in-place (or to out_dir).

Targets fields inside the transcription response (groq_response/response/result):
- text
- words[].word
- segments[].text
- segments[].words[].word
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


def clean_text(s: str, allow_ascii_symbols: bool, strict_english: bool):
    removed = []
    kept = []
    for ch in s:
        if is_allowed_char(ch, allow_ascii_symbols, strict_english):
            kept.append(ch)
        else:
            removed.append(ch)
    return "".join(kept), removed


def clean_node(node, *, allow_ascii_symbols: bool, strict_english: bool, path: str, removed_log: list):
    if isinstance(node, dict):
        for k, v in node.items():
            sub_path = f"{path}.{k}" if path else k
            if k in ("text", "word", "transcript") and isinstance(v, str):
                cleaned, removed = clean_text(v, allow_ascii_symbols, strict_english)
                if removed:
                    node[k] = cleaned
                    removed_log.append((sub_path, v, removed))
            elif isinstance(v, dict):
                clean_node(
                    v,
                    allow_ascii_symbols=allow_ascii_symbols,
                    strict_english=strict_english,
                    path=sub_path,
                    removed_log=removed_log,
                )
            elif isinstance(v, list) and k in ("words", "segments"):
                for i, item in enumerate(v):
                    clean_node(
                        item,
                        allow_ascii_symbols=allow_ascii_symbols,
                        strict_english=strict_english,
                        path=f"{sub_path}[{i}]",
                        removed_log=removed_log,
                    )
    elif isinstance(node, list):
        for i, item in enumerate(node):
            clean_node(
                item,
                allow_ascii_symbols=allow_ascii_symbols,
                strict_english=strict_english,
                path=f"{path}[{i}]",
                removed_log=removed_log,
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Folder with transcription JSON files")
    ap.add_argument("--out_dir", default=None, help="Write cleaned files here (default: in-place)")
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
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    allow_ascii_symbols = bool(int(args.allow_ascii_symbols))
    strict_english = bool(int(args.strict_english))

    report_path = (out_dir or in_dir) / "weird_chars_removed.jsonl"
    summary_path = (out_dir or in_dir) / "weird_chars_removed_summary.csv"

    counts = Counter()
    sample_file = {}

    with report_path.open("w", encoding="utf-8") as report_f:
        for json_path in sorted(in_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                rec = {"file": json_path.name, "error": f"failed_to_read_json: {e}"}
                report_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            resp = pick_response_node(data)
            removed_log = []
            clean_node(
                resp,
                allow_ascii_symbols=allow_ascii_symbols,
                strict_english=strict_english,
                path="",
                removed_log=removed_log,
            )

            hits_in_file = 0
            for field_path, original, removed in removed_log:
                for ch in removed:
                    cat = unicodedata.category(ch)
                    name = unicodedata.name(ch, "UNKNOWN")
                    cp = f"U+{ord(ch):04X}"
                    key = (ch, cp, cat, name)
                    counts[key] += 1
                    sample_file.setdefault(key, json_path.name)
                if hits_in_file < int(args.max_hits_per_file):
                    rec = {
                        "file": json_path.name,
                        "path": field_path,
                        "removed_count": len(removed),
                        "removed_chars": "".join(removed),
                        "original_snippet": original[:200],
                    }
                    report_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    hits_in_file += 1

            # Write cleaned JSON
            out_path = (out_dir / json_path.name) if out_dir else json_path
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

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
