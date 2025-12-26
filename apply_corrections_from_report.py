from __future__ import annotations

import argparse
import json
import sys
import difflib
from pathlib import Path
from typing import Dict, List


"""
usage : python apply_corrections_from_report.py --report rename_and_corrected_transcript_report.json --transcripts-dir "i:\P2GPT_google_drive\My Drive\Transcriptions" --output-dir "i:\P2GPT_google_drive\My Drive\Transcriptions_corrected"

"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply corrected transcripts from a rename_and_corrected_transcript_report.json to transcript JSON files."
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Path to rename_and_corrected_transcript_report.json",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        required=True,
        help="Directory containing the original transcript JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to write corrected transcript JSON files (defaults to in-place overwrite)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip files that already exist in output-dir (default: on)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Process all files even if already present in output-dir",
    )
    parser.add_argument(
        "--progress-width",
        type=int,
        default=30,
        help="Width of the ASCII progress bar",
    )
    return parser.parse_args()


def load_report(report_path: Path) -> Dict[str, str]:
    data = json.loads(report_path.read_text(encoding="utf-8"))
    mapping: Dict[str, str] = {}
    for entry in data:
        transcript_path = entry.get("transcript")
        corrected = entry.get("corrected_transcript")
        if not transcript_path or not corrected:
            continue
        # normalize path case for Windows
        mapping[Path(transcript_path).resolve().as_posix().lower()] = corrected
    return mapping


def proportional_split(text: str, counts: List[int]) -> List[str]:
    """Split text into len(counts) parts proportional to counts values."""
    if not counts or sum(counts) == 0:
        return [text]
    total = sum(counts)
    pieces: List[str] = []
    start = 0
    for i, c in enumerate(counts):
        if i == len(counts) - 1:
            pieces.append(text[start:].strip())
            break
        end = int(round((start + (len(text) - start) * c / (total if total else 1))))
        pieces.append(text[start:end].strip())
        start = end
    return pieces


def align_corrected_to_segments(corrected_text: str, segments: List[Dict]) -> List[str]:
    """
    Slice corrected_text back into segment-sized pieces using a character-level alignment
    against the original concatenated segment text. This prevents words drifting across
    segment boundaries.
    """
    if not segments:
        return [corrected_text]

    orig_parts = [seg.get("text", "") for seg in segments]
    sep = "\u241f"  # unlikely sentinel to keep boundaries distinct
    orig_text = sep.join(orig_parts)

    # Precompute original segment char ranges in the joined string
    ranges: List[tuple[int, int]] = []
    cursor = 0
    for i, part in enumerate(orig_parts):
        start = cursor
        end = start + len(part)
        ranges.append((start, end))
        cursor = end + (len(sep) if i < len(orig_parts) - 1 else 0)

    sm = difflib.SequenceMatcher(None, orig_text, corrected_text, autojunk=False)

    # Map each original character index to slices of corrected text
    char_to_spans: List[List[tuple[float, float]]] = [[] for _ in orig_text]
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            # attach insertion to previous char if possible, else next char, else first
            attach_idx = i1 - 1 if i1 > 0 else (i1 if i1 < len(char_to_spans) else len(char_to_spans) - 1)
            if 0 <= attach_idx < len(char_to_spans):
                char_to_spans[attach_idx].append((j1, j2))
            continue
        if tag == "delete":
            continue

        # equal or replace: distribute proportionally across the original slice
        o_len = max(1, i2 - i1)
        c_len = j2 - j1
        for k in range(i1, i2):
            rel_start = (k - i1) / o_len
            rel_end = (k - i1 + 1) / o_len
            c_start = j1 + rel_start * c_len
            c_end = j1 + rel_end * c_len
            char_to_spans[k].append((c_start, c_end))

    results: List[str] = []
    for seg_start, seg_end in ranges:
        spans = []
        for idx in range(seg_start, seg_end):
            spans.extend(char_to_spans[idx])
        if spans:
            c_start = int(min(s for s, _ in spans))
            c_end = int(max(e for _, e in spans))
            results.append(corrected_text[c_start:c_end].strip())
        else:
            results.append("")

    return results


def apply_correction(transcript_path: Path, corrected_text: str) -> Dict:
    obj = json.loads(transcript_path.read_text(encoding="utf-8"))
    groq = obj.get("groq_response") or {}
    segments = groq.get("segments") or []

    if segments:
        new_texts = align_corrected_to_segments(corrected_text, segments)
        # If any non-empty original segment became empty, fallback to proportional split for that slot
        lengths = [len(seg.get("text", "")) for seg in segments]
        fallback = proportional_split(corrected_text, lengths)
        for seg, new_text in zip(segments, new_texts):
            if not new_text and seg.get("text"):
                seg["text"] = fallback[segments.index(seg)]
            else:
                seg["text"] = new_text
        groq["segments"] = segments
    else:
        groq["segments"] = [{"text": corrected_text, "start": None, "end": None}]

    groq["text"] = corrected_text
    obj["groq_response"] = groq
    return obj


def render_progress(current: int, total: int, bar_width: int) -> None:
    if total <= 0:
        return
    fraction = min(1.0, current / total)
    filled = int(bar_width * fraction)
    bar = "#" * filled + "-" * (bar_width - filled)
    sys.stdout.write(f"\r[{bar}] {current}/{total}")
    sys.stdout.flush()


def main() -> None:
    args = parse_args()
    if not args.report.exists():
        raise SystemExit(f"Report not found: {args.report}")
    if not args.transcripts_dir.exists():
        raise SystemExit(f"Transcripts directory not found: {args.transcripts_dir}")

    out_dir = args.output_dir or args.transcripts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_report(args.report)
    if not mapping:
        raise SystemExit("No corrections found in report.")

    transcript_files = sorted(args.transcripts_dir.glob("*.json"))
    total = len(transcript_files)
    render_progress(0, total, args.progress_width)

    for idx, tfile in enumerate(transcript_files, start=1):
        key = tfile.resolve().as_posix().lower()
        out_path = out_dir / tfile.name
        if args.resume and out_path.exists():
            render_progress(idx, total, args.progress_width)
            continue
        if key not in mapping:
            render_progress(idx, total, args.progress_width)
            continue

        corrected = mapping[key]
        updated = apply_correction(tfile, corrected)
        out_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")
        render_progress(idx, total, args.progress_width)

    sys.stdout.write("\nDone.\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
