#!/usr/bin/env python3
"""Unpack Groq batch transcriptions back into per-original-file transcripts.

Why this exists
--------------
You packed many small audio files into <=25MB "batch" files (with silence gaps) to:
  - avoid Groq request-per-day limits
  - avoid the 10-second minimum billed length per-request
  - keep uploads under size limits

After Groq returns a `verbose_json` transcription for each batch file (ideally with
`timestamp_granularities=["word","segment"]`), this script:
  - reads your `batch_map.csv` (original_file -> batch_file + [start,end] seconds)
  - loads each batch's Groq JSON
  - slices words/segments that fall inside each original window
  - shifts timestamps so each original transcript starts at t=0
  - writes ONE transcript JSON per original file, in the same schema your Stage 1
    manifest script expects:

      {
        "input_file": {"path": "<original audio path>"},
        "groq_response": {
          "text": "...",
          "segments": [...],
          "words": [...]  # optional
        },
        "source_batch": {...}
      }

Requirements
------------
- Python 3.9+
- pip install tqdm

Typical usage (Windows):
  I:\Whisper-training-env\Scripts\python.exe unpack_groq_batches_to_original_transcripts.py \
    --batch_map_csv "I:\\Record_harsha\\groq_batches\\batch_map.csv" \
    --groq_json_dir  "I:\\Record_harsha\\groq_batch_transcripts" \
    --out_dir        "I:\\Record_harsha\\Groq_unpacked_transcripts" \
    --prefer_words \
    --gap_split_sec 1.0 \
    --resume

Notes
-----
- If your Groq JSONs are nested or named differently, this script tries hard to find
  the right JSON for each batch file.
- If word timestamps are present, we rebuild per-file segments from words (more robust
  around the inserted silence gaps).
- If words are missing, we fall back to clipping Groq segments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm


@dataclass(frozen=True)
class MapRow:
    original_file: str
    batch_file: str
    start_sec: float
    end_sec: float
    gap_sec: float


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _norm_path(p: str) -> str:
    # Normalise for matching; keep as string (Windows paths are case-insensitive).
    return str(Path(p)).replace("/", "\\").lower()


def read_batch_map(csv_path: Path) -> List[MapRow]:
    rows: List[MapRow] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required = {"original_file", "batch_file", "start_sec", "end_sec"}
        missing = required - set(r.fieldnames or [])
        if missing:
            raise SystemExit(f"batch_map.csv missing columns: {sorted(missing)}")

        for row in r:
            rows.append(
                MapRow(
                    original_file=row["original_file"],
                    batch_file=row["batch_file"],
                    start_sec=_as_float(row["start_sec"]),
                    end_sec=_as_float(row["end_sec"]),
                    gap_sec=_as_float(row.get("gap_sec", 0.0)),
                )
            )
    return rows


def find_batch_json(groq_json_dir: Path, batch_audio_path: str) -> Optional[Path]:
    """Find the Groq transcription JSON corresponding to a batch audio file."""
    bpath = Path(batch_audio_path)
    stem = bpath.stem
    name = bpath.name

    # Common patterns
    candidates = [
        groq_json_dir / f"{stem}.json",
        groq_json_dir / f"{name}.json",  # e.g. batch_00001.ogg.json
        groq_json_dir / f"{stem}.verbose_json.json",
        groq_json_dir / f"{stem}_verbose.json",
    ]

    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Fallback: recursive search by stem
    # (Limit work by scanning only *.json)
    matches: List[Path] = []
    for p in groq_json_dir.rglob("*.json"):
        if p.stem == stem or p.name == f"{name}.json":
            matches.append(p)

    if not matches:
        return None

    # Prefer shortest path (usually closest) and newest
    matches.sort(key=lambda p: (len(str(p)), -p.stat().st_mtime))
    return matches[0]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_groq_response(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Handle both:
    - wrapper schema: { ..., "groq_response": {...} }
    - raw Groq response: {"text":..., "segments":...}
    """
    if "groq_response" in obj and isinstance(obj["groq_response"], dict):
        return obj["groq_response"]
    return obj


def extract_words(resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return word list if present.

    Groq docs: if you request timestamp_granularities=['word','segment'] with
    response_format='verbose_json', you should get word-level timestamps.
    """
    words = resp.get("words")
    if isinstance(words, list) and words and isinstance(words[0], dict):
        # Expected items: {word,start,end}
        return words

    # Some variants embed words inside segments
    segs = resp.get("segments")
    if isinstance(segs, list):
        embedded: List[Dict[str, Any]] = []
        for s in segs:
            if isinstance(s, dict) and isinstance(s.get("words"), list):
                embedded.extend([w for w in s["words"] if isinstance(w, dict)])
        if embedded:
            return embedded

    return []


def slice_words(
    words: List[Dict[str, Any]],
    start: float,
    end: float,
    tol: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    dur = max(0.0, end - start)

    for w in words:
        ws = _as_float(w.get("start"))
        we = _as_float(w.get("end"))
        if we <= ws:
            continue

        mid = (ws + we) / 2.0
        if mid < (start - tol) or mid > (end + tol):
            continue

        # Shift to original-file timeline
        nws = max(0.0, ws - start)
        nwe = min(dur, we - start)
        if nwe <= nws:
            continue

        out.append(
            {
                "word": str(w.get("word", "")),
                "start": round(nws, 3),
                "end": round(nwe, 3),
            }
        )

    return out


def words_to_text(words: List[Dict[str, Any]]) -> str:
    # OpenAI/Groq often include leading spaces in word tokens.
    # Safest: concat raw "word" strings and strip at end.
    return "".join([w.get("word", "") for w in words]).strip()


def words_to_segments(
    words: List[Dict[str, Any]],
    gap_split_sec: float,
    max_segment_sec: float,
) -> List[Dict[str, Any]]:
    """Rebuild segments from word timestamps.

    - Split when there's a big silence gap between consecutive words.
    - Also split if a segment grows too long.

    This gives stable segment boundaries per original file (since your batch audio
    has explicit silence gaps between files).
    """
    if not words:
        return []

    segs: List[Dict[str, Any]] = []
    cur_words: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal cur_words
        if not cur_words:
            return
        s = float(cur_words[0]["start"])
        e = float(cur_words[-1]["end"])
        segs.append(
            {
                "id": len(segs),
                "seek": 0,
                "start": round(s, 3),
                "end": round(e, 3),
                "text": words_to_text(cur_words),
            }
        )
        cur_words = []

    for w in words:
        if not cur_words:
            cur_words.append(w)
            continue

        prev_end = float(cur_words[-1]["end"])
        cur_start = float(w["start"])
        gap = cur_start - prev_end

        seg_len = float(cur_words[-1]["end"]) - float(cur_words[0]["start"])
        if gap >= gap_split_sec or seg_len >= max_segment_sec:
            flush()

        cur_words.append(w)

    flush()
    return segs


def clip_segments(
    segments: List[Dict[str, Any]],
    start: float,
    end: float,
    tol: float,
) -> List[Dict[str, Any]]:
    """Fallback slicing when word timestamps are unavailable."""
    out: List[Dict[str, Any]] = []
    dur = max(0.0, end - start)

    for s in segments:
        if not isinstance(s, dict):
            continue
        ss = _as_float(s.get("start"))
        se = _as_float(s.get("end"))
        if se <= ss:
            continue

        # overlap test
        if se < (start - tol) or ss > (end + tol):
            continue

        cs = max(start, ss)
        ce = min(end, se)
        if ce <= cs:
            continue

        new_seg = dict(s)
        new_seg["start"] = round(max(0.0, cs - start), 3)
        new_seg["end"] = round(min(dur, ce - start), 3)
        out.append(new_seg)

    # Re-id segments for cleanliness
    for i, seg in enumerate(out):
        seg["id"] = i
        seg.setdefault("seek", 0)

    return out


def safe_filename_for_original(original_path: str) -> str:
    """Make a stable, unique filename for an original audio path.

    We include a short hash of the full path to avoid collisions when two files
    share the same basename.
    """
    p = Path(original_path)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem)[:120]
    h = hashlib.sha1(original_path.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{stem}__{h}.json"


def beep_done() -> None:
    try:
        import winsound  # type: ignore

        winsound.Beep(1000, 300)
        winsound.Beep(1200, 300)
    except Exception:
        sys.stdout.write("\a")
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch_map_csv", type=Path, required=True)
    ap.add_argument("--groq_json_dir", type=Path, required=True, help="Folder containing Groq batch transcript JSONs")
    ap.add_argument("--out_dir", type=Path, required=True, help="Where to write per-original transcript JSONs")

    ap.add_argument("--prefer_words", action="store_true", help="Prefer word timestamps if present (recommended)")
    ap.add_argument("--gap_split_sec", type=float, default=1.0, help="When rebuilding segments from words, split if silence gap >= this")
    ap.add_argument("--max_segment_sec", type=float, default=25.0, help="When rebuilding segments from words, also cap segment length")
    ap.add_argument("--tol_sec", type=float, default=0.25, help="Tolerance when assigning words/segments to a window")

    ap.add_argument("--resume", action="store_true", default=True, help="Skip outputs that already exist")
    ap.add_argument("--no_resume", action="store_false", dest="resume")

    args = ap.parse_args()

    if not args.batch_map_csv.exists():
        raise SystemExit(f"Missing batch_map_csv: {args.batch_map_csv}")
    if not args.groq_json_dir.exists():
        raise SystemExit(f"Missing groq_json_dir: {args.groq_json_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_batch_map(args.batch_map_csv)
    if not rows:
        raise SystemExit("batch_map.csv is empty")

    # Group by batch file
    by_batch: Dict[str, List[MapRow]] = defaultdict(list)
    for r in rows:
        by_batch[_norm_path(r.batch_file)].append(r)

    # Track missing batch JSONs
    missing_batches: List[str] = []
    written = 0
    skipped = 0

    for batch_key in tqdm(list(by_batch.keys()), desc="Unpacking batches", unit="batch"):
        batch_rows = by_batch[batch_key]
        # Recover original (non-normalised) path string from first row
        batch_file = batch_rows[0].batch_file

        bj = find_batch_json(args.groq_json_dir, batch_file)
        if bj is None:
            missing_batches.append(batch_file)
            continue

        wrapper = load_json(bj)
        resp = extract_groq_response(wrapper)

        all_segments = resp.get("segments") if isinstance(resp.get("segments"), list) else []
        all_words = extract_words(resp)

        for r in tqdm(batch_rows, desc=f"  {Path(batch_file).name}", unit="file", leave=False):
            out_name = safe_filename_for_original(r.original_file)
            out_path = args.out_dir / out_name

            if args.resume and out_path.exists() and out_path.stat().st_size > 0:
                skipped += 1
                continue

            # Slice words/segments
            use_words = args.prefer_words and bool(all_words)
            if use_words:
                words = slice_words(all_words, r.start_sec, r.end_sec, tol=args.tol_sec)
                segments = words_to_segments(words, gap_split_sec=args.gap_split_sec, max_segment_sec=args.max_segment_sec)
                text = words_to_text(words)
            else:
                segments = clip_segments(all_segments, r.start_sec, r.end_sec, tol=args.tol_sec)
                text = " ".join([str(s.get("text", "")).strip() for s in segments]).strip()
                words = []

            out_obj: Dict[str, Any] = {
                "input_file": {"path": r.original_file},
                "groq_response": {
                    "task": resp.get("task", "transcribe"),
                    "language": resp.get("language"),
                    "duration": round(max(0.0, r.end_sec - r.start_sec), 3),
                    "text": text,
                    "segments": segments,
                },
                "source_batch": {
                    "batch_file": batch_file,
                    "batch_json": str(bj),
                    "start_sec": round(r.start_sec, 3),
                    "end_sec": round(r.end_sec, 3),
                    "gap_sec": round(r.gap_sec, 3),
                },
            }

            if use_words:
                out_obj["groq_response"]["words"] = words

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(out_obj, f, ensure_ascii=False, indent=2)

            written += 1

    # Write a small report
    report = {
        "batch_map_csv": str(args.batch_map_csv),
        "groq_json_dir": str(args.groq_json_dir),
        "out_dir": str(args.out_dir),
        "written": written,
        "skipped_existing": skipped,
        "missing_batch_jsons": missing_batches,
        "missing_batch_jsons_count": len(missing_batches),
    }
    report_path = args.out_dir / "unpack_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Wrote:   {written} transcript JSON(s)")
    print(f"Skipped: {skipped} existing transcript JSON(s)")
    print(f"Report:  {report_path}")
    if missing_batches:
        print("\nMissing batch transcript JSON for these batch audio files:")
        for b in missing_batches[:20]:
            print(f"  - {b}")
        if len(missing_batches) > 20:
            print(f"  ... and {len(missing_batches) - 20} more")

    beep_done()


if __name__ == "__main__":
    main()
