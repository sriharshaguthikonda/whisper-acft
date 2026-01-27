#!/usr/bin/env python3
"""stage_0_apply_corrections_from_report.py

What you asked for
------------------
When you push a corrected (punctuation-fixed) full transcript back into segment texts,
mark any segment that has a mismatch/conflict with a boolean flag so your subtitle
exporter can highlight it.

This script:
1) Loads rename_and_corrected_transcript_report.json
2) For each transcript JSON file, applies corrected transcript to groq_response.segments[*].text
   using a robust tiered aligner:
   - Tier 1: strict word-token gate + wordcount slicing (fastest, best when punctuation-only)
   - Tier 2: jiwer word-level alignment (optional dependency)
   - Tier 3: edlib global char alignment (optional dependency)
3) Adds per-segment keys:
   - transcript_conflict: true/false
   - transcript_conflict_reasons: list[str]
   - tokens_stale: true if we changed text (because Whisper tokens won't match any more)
4) Writes:
   - corrected JSON transcript files
   - conflicts_files.txt : basenames of transcript JSONs that had ANY conflicts
   - conflicts_segments.csv : per-segment conflict details

Install optional deps (recommended)
----------------------------------
  pip install jiwer edlib

Usage
-----
i:\Whisper-training-env\Scripts\python.exe "i:\whisper-acft\stage_0_apply_corrections_from_report.py"\
  --report "rename_and_corrected_transcript_report.json" \
  --transcripts-dir "I:\P2GPT_google_drive\My Drive\nfa_corrected\patched_json" \
  --output-dir "I:\\P2GPT_google_drive\\My Drive\\Transcriptions_patched_corrected" \
  --mode auto \
  --strict_words \
  --conflicts-files "I:\\P2GPT_google_drive\\My Drive\\Transcriptions_patched_corrected\\stage_0_apply_corrections_from_report_conflicts_files.txt" \
  --conflicts-csv "I:\\P2GPT_google_drive\\My Drive\\Transcriptions_patched_corrected\\stage_0_apply_corrections_from_report_conflicts_segments.csv"

Notes
-----
- If --strict_words is set, any FILE whose full corrected transcript changes word tokens
  is treated as a file-level conflict. We still attempt alignment, but segments will be
  flagged and you can patch them later via subtitles.
- Your subtitle exporter can prepend e.g. "[CONFLICT]" to any segment with transcript_conflict=true.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
import winsound
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

WORD_RE = re.compile(r"[A-Za-z0']+")

SMART_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"',
    "\u2026": "...",
    "\u2013": "-", "\u2014": "-",
    "\u00A0": " ",
}


def clean_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    for k, v in SMART_PUNCT_MAP.items():
        s = s.replace(k, v)

    # drop control/format chars
    out = []
    for ch in s:
        if unicodedata.category(ch) in ("Cc", "Cf"):
            continue
        out.append(ch)
    s = "".join(out)

    # collapse whitespace (kills newlines/tabs)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_tokens(s: str) -> List[str]:
    return WORD_RE.findall((s or "").lower())


def word_spans(s: str) -> List[Tuple[int, int]]:
    return [(m.start(), m.end()) for m in WORD_RE.finditer(s)]


@dataclass
class ReportRow:
    corrected: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--transcripts-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["auto", "wordcount", "jiwer", "edlib"], default="auto")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--strict_words", action="store_true", help="Flag file-level conflict when words changed")
    ap.add_argument("--conflicts-files", type=Path, required=True)
    ap.add_argument("--conflicts-csv", type=Path, required=True)
    return ap.parse_args()


def load_report(report_path: Path) -> Dict[str, ReportRow]:
    data = json.loads(report_path.read_text("utf-8", errors="replace"))
    if not isinstance(data, list):
        raise SystemExit("Report JSON must be a list.")

    mapping: Dict[str, ReportRow] = {}
    for e in tqdm(data, desc="Loading report", unit="entries"):
        if not isinstance(e, dict):
            continue
        t = e.get("transcript")
        c = e.get("corrected_transcript")
        if not isinstance(t, str) or not isinstance(c, str) or not c:
            continue

        # Match by basename ONLY (robust across different drive letters/roots)
        try:
            base = PureWindowsPath(t).name
        except Exception:
            base = Path(t).name
        mapping[base.lower()] = ReportRow(corrected=c)

    return mapping


def get_segments(obj: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    groq = obj.get("groq_response")
    if not isinstance(groq, dict):
        groq = {}
    segs = groq.get("segments")
    if not isinstance(segs, list):
        segs = []
    segs = [s for s in segs if isinstance(s, dict)]
    return groq, segs


def rebuild_full_from_segments(segs: List[Dict[str, Any]]) -> str:
    return " ".join(str(s.get("text") or "").strip() for s in segs).strip()


# ---------------- Tier 1: wordcount slicing ----------------

def slice_by_wordcount(corrected_full: str, segs: List[Dict[str, Any]]) -> List[str]:
    corrected_full = clean_text(corrected_full)
    corr_spans = word_spans(corrected_full)
    corr_tokens = word_tokens(corrected_full)

    counts = [len(WORD_RE.findall(str(s.get("text") or ""))) for s in segs]
    if sum(counts) != len(corr_tokens):
        raise ValueError("wordcount mismatch")

    out: List[str] = []
    w = 0
    for seg, n in zip(segs, counts):
        if n <= 0:
            out.append(clean_text(str(seg.get("text") or "")).strip())
            continue
        start_word = w
        end_word = w + n
        start_char = corr_spans[start_word][0]
        end_char = corr_spans[end_word][0] if end_word < len(corr_spans) else len(corrected_full)
        out.append(corrected_full[start_char:end_char].strip())
        w = end_word

    return out


# ---------------- Tier 2: jiwer word alignment -------------

def slice_with_jiwer(original_full: str, corrected_full: str, segs: List[Dict[str, Any]]) -> List[str]:
    try:
        import jiwer
    except Exception as e:
        raise RuntimeError("jiwer not installed: pip install jiwer") from e

    original_full = clean_text(original_full)
    corrected_full = clean_text(corrected_full)

    ref_words = WORD_RE.findall(original_full)
    hyp_words = WORD_RE.findall(corrected_full)

    ref_str = " ".join(ref_words)
    hyp_str = " ".join(hyp_words)

    out = jiwer.process_words(ref_str, hyp_str)

    ref_to_hyp: List[Optional[int]] = [None] * len(ref_words)
    for ch in out.alignments[0]:
        if ch.type == "equal":
            for r_i, h_i in zip(range(ch.ref_start_idx, ch.ref_end_idx), range(ch.hyp_start_idx, ch.hyp_end_idx)):
                ref_to_hyp[r_i] = h_i

    hyp_spans = word_spans(corrected_full)

    seg_ref_counts = [len(WORD_RE.findall(str(s.get("text") or ""))) for s in segs]

    ref_cursor = 0
    results: List[str] = []

    for n in seg_ref_counts:
        if n <= 0:
            results.append("")
            continue
        r_start = ref_cursor
        r_end = ref_cursor + n

        hyp_indices = [i for i in ref_to_hyp[r_start:r_end] if i is not None]
        if not hyp_indices:
            results.append("")
        else:
            h_start = min(hyp_indices)
            h_end = max(hyp_indices) + 1
            start_char = hyp_spans[h_start][0]
            end_char = hyp_spans[h_end][0] if h_end < len(hyp_spans) else len(corrected_full)
            results.append(corrected_full[start_char:end_char].strip())

        ref_cursor = r_end

    return results


# ---------------- Tier 3: edlib global char alignment -------

def _edlib_alignment_path(orig: str, corr: str) -> str:
    try:
        import edlib
    except Exception as e:
        raise RuntimeError("edlib not installed: pip install edlib") from e

    res = edlib.align(orig, corr, mode="NW", task="path")
    cigar = res.get("cigar")
    if not cigar:
        raise RuntimeError("edlib returned no cigar")
    return cigar


def _cigar_to_mapping(orig: str, corr: str, cigar: str) -> List[Optional[int]]:
    mapping: List[Optional[int]] = [None] * len(orig)
    o = 0
    c = 0

    for length_s, op in re.findall(r"(\d+)([=XID])", cigar):
        n = int(length_s)
        if op in ("=", "X"):
            for _ in range(n):
                if o < len(orig):
                    mapping[o] = c
                o += 1
                c += 1
        elif op == "I":
            c += n
        elif op == "D":
            o += n

    return mapping


def slice_with_edlib(original_full: str, corrected_full: str, segs: List[Dict[str, Any]]) -> List[str]:
    original_full = clean_text(original_full)
    corrected_full = clean_text(corrected_full)

    cigar = _edlib_alignment_path(original_full, corrected_full)
    mapping = _cigar_to_mapping(original_full, corrected_full, cigar)

    # compute original spans by concatenating cleaned segment texts with single spaces
    parts = [clean_text(str(s.get("text") or "")).strip() for s in segs]
    spans: List[Tuple[int, int]] = []
    cur = 0
    for i, p in enumerate(parts):
        start = cur
        end = start + len(p)
        spans.append((start, end))
        cur = end + (1 if i < len(parts) - 1 else 0)

    out: List[str] = []
    for (s, e) in spans:
        corr_idxs = [mapping[i] for i in range(s, min(e, len(mapping))) if mapping[i] is not None]
        if not corr_idxs:
            out.append("")
        else:
            c_start = min(corr_idxs)
            c_end = max(corr_idxs) + 1
            out.append(corrected_full[c_start:c_end].strip())

    return out


def apply_alignment(original_full: str, corrected_full: str, segs: List[Dict[str, Any]], mode: str) -> Tuple[str, List[str]]:
    if mode == "wordcount":
        return "wordcount", slice_by_wordcount(corrected_full, segs)
    if mode == "jiwer":
        return "jiwer", slice_with_jiwer(original_full, corrected_full, segs)
    if mode == "edlib":
        return "edlib", slice_with_edlib(original_full, corrected_full, segs)

    # auto
    last_err = None
    for m in ("wordcount", "jiwer", "edlib"):
        try:
            return apply_alignment(original_full, corrected_full, segs, m)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"auto failed: {last_err}")


def mark_conflicts(
    filename: str,
    segs: List[Dict[str, Any]],
    old_texts: List[str],
    new_texts: List[str],
    *,
    file_word_changed: bool,
    align_method: str,
    conflict_rows: List[Dict[str, Any]],
    original_full: Optional[str] = None,
    corrected_full: Optional[str] = None,
) -> bool:
    """Return True if file has any conflicts.

    IMPORTANT CHANGE:
    - file_word_changed no longer forces every segment to be a conflict.
    - file_word_changed is still counted as a FILE conflict, and we optionally write a
      file-level CSV row (segment_index = -1) so you can find these quickly.
    """

    file_has_conflict = False

    # (Optional) write ONE file-level conflict row
    if file_word_changed:
        file_has_conflict = True
        conflict_rows.append(
            {
                "file": filename,
                "segment_index": -1,
                "segment_id": None,
                "start": None,
                "end": None,
                "align_method": align_method,
                "reasons": "file_word_tokens_changed",
                "old_text": clean_text(original_full or ""),
                "new_text": clean_text(corrected_full or ""),
            }
        )

    for idx, (seg, old, new) in enumerate(zip(segs, old_texts, new_texts)):
        reasons: List[str] = []

        old_c = clean_text(old)
        new_c = clean_text(new)

        # Segment-level checks ONLY
        if old_c and not new_c:
            reasons.append("became_empty")

        if len(word_tokens(old_c)) != len(word_tokens(new_c)):
            reasons.append("word_count_changed")

        if word_tokens(old_c) != word_tokens(new_c):
            reasons.append("word_tokens_changed")

        if reasons:
            file_has_conflict = True
            seg["transcript_conflict"] = True
            seg["transcript_conflict_reasons"] = reasons

            conflict_rows.append(
                {
                    "file": filename,
                    "segment_index": idx,
                    "segment_id": seg.get("id"),
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "align_method": align_method,
                    "reasons": ";".join(reasons),
                    "old_text": old_c,
                    "new_text": new_c,
                }
            )
        else:
            seg["transcript_conflict"] = False
            seg["transcript_conflict_reasons"] = []

        # Tokens are stale if text changed at all (set explicitly either way)
        seg["tokens_stale"] = (old_c != new_c)

    return file_has_conflict



def main() -> None:
    args = parse_args()

    mapping = load_report(args.report)
    if not mapping:
        raise SystemExit("No corrected_transcript entries found in report.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.conflicts_files.parent.mkdir(parents=True, exist_ok=True)
    args.conflicts_csv.parent.mkdir(parents=True, exist_ok=True)

    conflict_rows: List[Dict[str, Any]] = []
    conflict_files: List[str] = []

    files = sorted(args.transcripts_dir.glob("*.json"))

    wrote = 0
    for f in tqdm(files, desc="Processing transcripts", unit="files"):
        key = f.name.lower()
        if key not in mapping:
            continue

        out_path = args.output_dir / f.name
        if args.resume and out_path.exists():
            continue

        try:
            obj = json.loads(f.read_text("utf-8", errors="replace"))
            if not isinstance(obj, dict):
                continue

            groq, segs = get_segments(obj)
            corrected_full = clean_text(mapping[key].corrected)

            if not segs:
                # still write file-level corrected text
                groq["text"] = corrected_full
                groq["segments"] = [{"text": corrected_full, "start": None, "end": None, "transcript_conflict": False, "transcript_conflict_reasons": [], "tokens_stale": True}]
                obj["groq_response"] = groq
                out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")
                wrote += 1
                continue

            old_texts = [str(s.get("text") or "") for s in segs]
            original_full = rebuild_full_from_segments(segs)

            file_word_changed = False
            if args.strict_words and (word_tokens(original_full) != word_tokens(corrected_full)):
                file_word_changed = True

            align_method, new_texts = apply_alignment(original_full, corrected_full, segs, args.mode)

            # Apply new segment texts
            for seg, new_t in zip(segs, new_texts):
                seg["text"] = clean_text(new_t)

            # Mark conflicts + collect anomalies
            if mark_conflicts(
                f.name, 
                segs, 
                old_texts, 
                [s.get("text", "") for s in segs], 
                file_word_changed=file_word_changed, 
                align_method=align_method, 
                conflict_rows=conflict_rows,
                original_full=original_full,
                corrected_full=corrected_full
            ):
                conflict_files.append(f.name)

            groq["segments"] = segs
            groq["text"] = rebuild_full_from_segments(segs)
            obj["groq_response"] = groq

            out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), "utf-8")
            wrote += 1

        except Exception as e:
            print(f"FAIL {f.name}: {e}", file=sys.stderr)
            # Treat as a conflict file so you can quickly inspect
            conflict_files.append(f.name)

    # Write conflicts list (unique, stable order)
    seen = set()
    uniq_files = []
    for n in conflict_files:
        if n not in seen:
            seen.add(n)
            uniq_files.append(n)

    args.conflicts_files.write_text("\n".join(uniq_files) + ("\n" if uniq_files else ""), "utf-8")

    # Write conflict details CSV
    with args.conflicts_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "segment_index",
                "segment_id",
                "start",
                "end",
                "align_method",
                "reasons",
                "old_text",
                "new_text",
            ],
        )
        w.writeheader()
        for r in conflict_rows:
            w.writerow(r)

    print(f"Done. Wrote {wrote} corrected JSON files.")
    print(f"Conflict files: {len(uniq_files)}  -> {args.conflicts_files}")
    print(f"Conflict segments: {len(conflict_rows)} -> {args.conflicts_csv}")
    
    # Beep notification
    try:
        winsound.Beep(1000, 500)  # 1000Hz for 500ms
    except (ImportError, OSError, RuntimeError):
        print("\a")  # Fallback to system bell


if __name__ == "__main__":
    main()
