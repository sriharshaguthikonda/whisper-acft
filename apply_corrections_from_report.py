from __future__ import annotations

import argparse
import json
import sys
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
        default=Path("rename_and_corrected_transcript_report.json"),
        help="Path to rename_and_corrected_transcript_report.json (default: rename_and_corrected_transcript_report.json)",
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path(r"I:\P2GPT_google_drive\My Drive\Transcriptions"),
        help="Directory containing the original transcript JSON files (default: I:\\P2GPT_google_drive\\My Drive\\Transcriptions)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"I:\P2GPT_google_drive\My Drive\Transcriptions_corrected"),
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


def _smith_waterman_affine(orig: str, corr: str, match: int = 2, mismatch: int = -1, gap_open: int = -2, gap_extend: int = -1):
    """Local alignment (Smith–Waterman) with affine gaps. Returns aligned index pairs."""
    n, m = len(orig), len(corr)
    M = [[0] * (m + 1) for _ in range(n + 1)]
    Ix = [[0] * (m + 1) for _ in range(n + 1)]  # gap in corr (deletion from corr)
    Iy = [[0] * (m + 1) for _ in range(n + 1)]  # gap in orig (insertion into corr)

    trace_M = [[0] * (m + 1) for _ in range(n + 1)]
    trace_Ix = [[0] * (m + 1) for _ in range(n + 1)]
    trace_Iy = [[0] * (m + 1) for _ in range(n + 1)]

    best_score = 0
    best_pos = (0, 0, "M")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Ix: gap in corr (orig consumes a char)
            op1 = M[i - 1][j] + gap_open
            op2 = Ix[i - 1][j] + gap_extend
            if op1 >= op2 and op1 > 0:
                Ix[i][j] = op1
                trace_Ix[i][j] = 1  # from M
            elif op2 > 0:
                Ix[i][j] = op2
                trace_Ix[i][j] = 2  # from Ix
            else:
                Ix[i][j] = 0
                trace_Ix[i][j] = 0

            # Iy: gap in orig (corr consumes a char)
            op1 = M[i][j - 1] + gap_open
            op2 = Iy[i][j - 1] + gap_extend
            if op1 >= op2 and op1 > 0:
                Iy[i][j] = op1
                trace_Iy[i][j] = 1  # from M
            elif op2 > 0:
                Iy[i][j] = op2
                trace_Iy[i][j] = 2  # from Iy
            else:
                Iy[i][j] = 0
                trace_Iy[i][j] = 0

            score = match if orig[i - 1] == corr[j - 1] else mismatch
            diag_M = M[i - 1][j - 1] + score
            diag_Ix = Ix[i - 1][j - 1] + score
            diag_Iy = Iy[i - 1][j - 1] + score
            best_diag = diag_M
            source = 1
            if diag_Ix > best_diag:
                best_diag = diag_Ix
                source = 2
            if diag_Iy > best_diag:
                best_diag = diag_Iy
                source = 3

            if best_diag > 0:
                M[i][j] = best_diag
                trace_M[i][j] = source
            else:
                M[i][j] = 0
                trace_M[i][j] = 0

            # track best over all matrices
            if M[i][j] > best_score:
                best_score = M[i][j]
                best_pos = (i, j, "M")
            if Ix[i][j] > best_score:
                best_score = Ix[i][j]
                best_pos = (i, j, "Ix")
            if Iy[i][j] > best_score:
                best_score = Iy[i][j]
                best_pos = (i, j, "Iy")

    # Traceback
    i, j, state = best_pos
    alignment = []
    while i > 0 and j > 0:
        if state == "M":
            if M[i][j] == 0:
                break
            source = trace_M[i][j]
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
            if source == 1:
                state = "M"
            elif source == 2:
                state = "Ix"
            else:
                state = "Iy"
        elif state == "Ix":
            if Ix[i][j] == 0:
                break
            source = trace_Ix[i][j]
            alignment.append((i - 1, None))  # orig char aligned to gap in corr
            i -= 1
            if source == 1:
                state = "M"
            else:
                state = "Ix"
        else:  # Iy
            if Iy[i][j] == 0:
                break
            source = trace_Iy[i][j]
            alignment.append((None, j - 1))  # corr char aligned to gap in orig
            j -= 1
            if source == 1:
                state = "M"
            else:
                state = "Iy"

    alignment.reverse()
    return alignment


def align_corrected_to_segments(corrected_text: str, segments: List[Dict]) -> List[str]:
    """
    Align corrected_text to the concatenated original segment text using Smith–Waterman
    with affine gaps, then slice corrected_text back into per-segment pieces while
    preserving segment order.
    """
    if not segments:
        return [corrected_text]

    orig_parts = [seg.get("text", "") for seg in segments]
    sep = "\u241f"  # unlikely sentinel to keep boundaries distinct
    orig_text = sep.join(orig_parts)

    alignment = _smith_waterman_affine(orig_text, corrected_text)

    spans: List[List[int]] = [[] for _ in orig_text]
    pending_ins: List[int] = []
    last_orig = None

    for o_idx, c_idx in alignment:
        if o_idx is not None and c_idx is not None:
            if pending_ins:
                spans[o_idx].extend(pending_ins)
                pending_ins = []
            spans[o_idx].append(c_idx)
            last_orig = o_idx
        elif o_idx is None and c_idx is not None:
            if last_orig is not None:
                spans[last_orig].append(c_idx)
            else:
                pending_ins.append(c_idx)
        elif o_idx is not None:
            if pending_ins and last_orig is not None:
                spans[last_orig].extend(pending_ins)
                pending_ins = []
            last_orig = o_idx

    if pending_ins and last_orig is not None:
        spans[last_orig].extend(pending_ins)

    # Precompute original segment char ranges in the joined string
    ranges: List[tuple[int, int]] = []
    cursor = 0
    for i, part in enumerate(orig_parts):
        start = cursor
        end = start + len(part)
        ranges.append((start, end))
        cursor = end + (len(sep) if i < len(orig_parts) - 1 else 0)

    results: List[str] = []
    for seg_start, seg_end in ranges:
        seg_indices: List[int] = []
        for idx in range(seg_start, seg_end):
            seg_indices.extend(spans[idx])
        if seg_indices:
            c_start = min(seg_indices)
            c_end = max(seg_indices) + 1
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
