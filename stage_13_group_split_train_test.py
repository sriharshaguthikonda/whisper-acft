"""stage_13_group_split_train_test.py

Group-aware train/test split for Whisper-style JSONL manifests.

Why you want this
-----------------
Your Stage 11 script splits *rows* randomly.
That leaks data when you have augmentations (gain/noise/reverb/mix) because the
same underlying chunk (same transcript) can end up in BOTH train and test.

This version splits by GROUPS so that all augmented variants of the same base
chunk stay together.

Grouping strategy
-----------------
For each row we build a stable group key using whichever fields exist:
  1) transcript_json + chunk_index
  2) source_audio + chunk_index
  3) (fallback) basename(audio_path) with augmentation suffixes stripped

You can change the group_key() function if you want a different rule.

Usage
-----


python stage_13_group_split_train_test.py --input_manifest "I:\Record_chunks\pairs_manifest_stage7_plus_stage9_reverb_bottom_filtered.jsonl" --test_manifest "I:\Record_chunks\pairs_manifest_stage7_plus_stage9_reverb_bottom_filtered_test.jsonl" --train_manifest "I:\Record_chunks\pairs_manifest_stage7_plus_stage9_reverb_bottom_filtered_train.jsonl" --test_ratio 0.1 --seed 1337

Optional:
  --dry_run



python stage_13_group_split_train_test.py --input_manifest "I:\Record_chunks\pairs_manifest_local_randomised.jsonl" --test_manifest "I:\Record_chunks\pairs_manifest_sorted_by_scores_english_only_filtered_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_test.jsonl" --train_manifest "I:\Record_chunks\pairs_manifest_sorted_by_scores_english_only_filtered_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_train.jsonl" --test_ratio 0.1 --seed 1337
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _canon_pathish(x: Any) -> str:
    """Canonicalise Windows paths so different slash styles don't create different groups."""
    s = str(x)
    return s.replace("\\", "/").strip().lower()


_UID_FROM_STEM_RE = re.compile(r"^([0-9a-f]{8,40})_[0-9a-f]{8,40}__", re.IGNORECASE)


def _base_uid_from_filename(ap: str) -> str | None:
    """Best-effort: for files like '<baseuid>_<auguuid>__stage__copy00.wav' return <baseuid>."""
    stem = Path(str(ap)).stem
    m = _UID_FROM_STEM_RE.match(stem)
    return m.group(1).lower() if m else None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no}: {path}") from e
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_AUG_SUFFIX_RE = re.compile(
    r"(__gain[-+]?\d+(?:\.\d+)?dB|__gainp\d+|__rir[^\\/]*|__mix[^\\/]*|__noise[^\\/]*|__reverb[^\\/]*|__aug[^\\/]*|__dup\d+)"  # common suffix styles
)


def _strip_aug_suffixes(stem: str) -> str:
    # remove repeated augmentation suffixes from a filename stem
    prev = None
    cur = stem
    while prev != cur:
        prev = cur
        cur = _AUG_SUFFIX_RE.sub("", cur)
    return cur


def group_key(row: Dict[str, Any]) -> str:
    """Return a stable group key so all augmented variants stay together."""

    # 0) If your pipeline has stable IDs, ALWAYS use them first.
    #    This is the most reliable way to avoid leakage across augmentations.
    bu = row.get("base_uid")
    if bu:
        return f"uid::{str(bu).lower()}"

    pu = row.get("parent_uid")
    if pu:
        return f"uid::{str(pu).lower()}"

    tj = row.get("transcript_json")
    ci = row.get("chunk_index")
    if tj and ci is not None:
        return f"tj::{_canon_pathish(tj)}::ci::{int(ci)}"

    sa = row.get("source_audio")
    if sa and ci is not None:
        return f"sa::{_canon_pathish(sa)}::ci::{int(ci)}"

    # Fallback: strip known augmentation suffixes from audio_path basename
    ap = row.get("audio_path") or row.get("out_wav") or ""

    # If this looks like your idempotent-augmentation naming, group by the embedded base uid.
    bu2 = _base_uid_from_filename(str(ap))
    if bu2:
        return f"uid::{bu2}"

    p = Path(str(ap))
    base = _strip_aug_suffixes(p.stem.lower())
    base = re.sub(r"__copy\d+$", "", base)  # common tail
    return f"ap::{base}"


def underlying_key_for_leak_audit(row: Dict[str, Any]) -> str:
    """A stricter key used only to detect leakage after splitting."""
    # Prefer stable ids
    for k in ("base_uid", "parent_uid", "uid"):
        v = row.get(k)
        if v:
            return f"uid::{str(v).lower()}"

    # Next best: transcript_json/source_audio + chunk_index (with canonicalised path)
    ci = row.get("chunk_index")
    tj = row.get("transcript_json")
    if tj and ci is not None:
        return f"tj::{_canon_pathish(tj)}::ci::{int(ci)}"
    sa = row.get("source_audio")
    if sa and ci is not None:
        return f"sa::{_canon_pathish(sa)}::ci::{int(ci)}"

    # Last resort: try embedded uid from filename, else stem
    ap = row.get("audio_path") or row.get("out_wav") or ""
    bu = _base_uid_from_filename(str(ap))
    if bu:
        return f"uid::{bu}"
    return f"ap::{Path(str(ap)).stem.lower()}"


def split_groups(
    rows: List[Dict[str, Any]],
    test_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1")

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[group_key(r)].append(r)

    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    n_test_groups = max(1, int(round(len(keys) * test_ratio)))
    test_keys = set(keys[:n_test_groups])

    test_rows: List[Dict[str, Any]] = []
    train_rows: List[Dict[str, Any]] = []

    for k, rs in groups.items():
        (test_rows if k in test_keys else train_rows).extend(rs)

    return train_rows, test_rows


def main() -> None:
    p = argparse.ArgumentParser(description="Group-aware train/test split for Whisper JSONL manifests")
    p.add_argument("--input_manifest", required=True, type=Path)
    p.add_argument("--test_manifest", required=True, type=Path)
    p.add_argument("--train_manifest", required=True, type=Path)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    if not args.input_manifest.exists():
        raise FileNotFoundError(f"Input manifest not found: {args.input_manifest}")

    rows = read_jsonl(args.input_manifest)
    print(f"Loaded {len(rows):,} rows from {args.input_manifest}")

    train_rows, test_rows = split_groups(rows, args.test_ratio, args.seed)

    # Hard fail if we detect the same underlying chunk in both sets.
    train_u = {underlying_key_for_leak_audit(r) for r in train_rows}
    test_u = {underlying_key_for_leak_audit(r) for r in test_rows}
    overlap = train_u & test_u
    if overlap:
        ex = list(sorted(overlap))[:10]
        print("\n[LEAK DETECTED] Underlying groups appear in BOTH train and test.")
        print(f"Examples (up to 10): {ex}")
        raise SystemExit(2)

    # Group counts (for sanity)
    uniq_groups = len({group_key(r) for r in rows})
    uniq_train = len({group_key(r) for r in train_rows})
    uniq_test = len({group_key(r) for r in test_rows})

    print("\nSplit summary (group-aware):")
    print(f"  Total rows:     {len(rows):,}")
    print(f"  Total groups:   {uniq_groups:,}")
    print(f"  Train rows:     {len(train_rows):,} | groups: {uniq_train:,}")
    print(f"  Test rows:      {len(test_rows):,} | groups: {uniq_test:,}")

    if args.dry_run:
        print("\nDry run: not writing files.")
        return

    write_jsonl(args.test_manifest, test_rows)
    write_jsonl(args.train_manifest, train_rows)

    print("\nWrote:")
    print(f"  Test:  {args.test_manifest}")
    print(f"  Train: {args.train_manifest}")

    print("\nSample test rows:")
    for r in test_rows[:5]:
        ap = r.get("audio_path", "N/A")
        tx = (r.get("raw_transcription") or "")
        print(f"  - {ap}")
        print(f"    text: {tx[:90]}{'...' if len(tx) > 90 else ''}")


if __name__ == "__main__":
    main()
