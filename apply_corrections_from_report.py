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
    sep = " "
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
    builders = [""] * len(ranges)

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "insert":
            # Assign insertion to the segment that ends at i1 (previous), otherwise next
            seg_idx = None
            for idx, (s, e) in enumerate(ranges):
                if e == i1:
                    seg_idx = idx
                    break
                if s <= i1 < e:
                    seg_idx = idx
                    break
                if i1 < s:
                    seg_idx = max(idx - 1, 0)
                    break
            if seg_idx is None:
                seg_idx = len(ranges) - 1
            builders[seg_idx] += corrected_text[j1:j2]
            continue

        if tag == "delete":
            # nothing to add from corrected text
            continue

        # tag in {"equal", "replace"}
        o_len = i2 - i1
        c_len = j2 - j1
        if o_len == 0:
            continue

        # map portions proportionally to overlapping segments
        o_cursor = i1
        c_cursor = j1
        while o_cursor < i2:
            # find segment containing this origin position
            seg_idx = next((idx for idx, (s, e) in enumerate(ranges) if s <= o_cursor < e), len(ranges) - 1)
            seg_start, seg_end = ranges[seg_idx]
            take = min(i2, seg_end) - o_cursor
            # proportional slice of corrected span
            proportion = take / o_len
            c_take = max(1, int(round(c_len * proportion))) if (o_cursor + take) >= i2 else int(c_len * proportion)
            c_take = min(c_take, j2 - c_cursor)
            builders[seg_idx] += corrected_text[c_cursor : c_cursor + c_take]
            o_cursor += take
            c_cursor += c_take

    # Fallback: ensure each piece stripped
    return [b.strip() for b in builders]


def apply_correction(transcript_path: Path, corrected_text: str) -> Dict:
    obj = json.loads(transcript_path.read_text(encoding="utf-8"))
    groq = obj.get("groq_response") or {}
    segments = groq.get("segments") or []

    if segments:
        new_texts = align_corrected_to_segments(corrected_text, segments)
        for seg, new_text in zip(segments, new_texts):
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
