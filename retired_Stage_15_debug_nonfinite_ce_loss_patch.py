#!/usr/bin/env python3
"""debug_nonfinite_ce_loss_patch.py

Goal
----
Stop (and explain) the "Non-finite CE loss. Skipping batch" events.

There are two *very* common causes:
1) **All labels are ignore_index (-100)** in a batch.
   PyTorch CrossEntropyLoss with reduction='mean' becomes NaN if it has *zero* valid targets.
2) **Numerical overflow** (often FP16 / AMP) -> logits become NaN/Inf.

This file gives you:
A) A manifest audit to find samples that become empty targets.
B) A safe CE computation + debug prints to drop the exact offending items.

Usage
-----
python stage_15_debug_nonfinite_ce_loss_patch.py --manifest I:\Record_chunks\pairs_manifest_local_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_silent_randomized_filtered_train_no_targets_filtered.jsonl --show 30

Then patch your training loop using the helper functions at the bottom.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


SPECIAL_EMPTY = {"", "<|nospeech|>", "<|nocaptions|>"}


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def basic_whisperish_normalize(text: str) -> str:
    """Keep this aligned with whatever you use during WER evaluation.

    Key thing: if you map <|nospeech|>/<|nocaptions|> to "", then a batch of these becomes
    'all labels ignored' and CE loss becomes NaN.
    """
    t = (text or "").strip()
    if t.lower() in SPECIAL_EMPTY:
        return ""
    return t


def audit_manifest(manifest_path: Path, show: int = 20) -> None:
    empty_raw = 0
    empty_norm = 0
    total = 0

    examples_raw: List[Tuple[str, str]] = []
    examples_norm: List[Tuple[str, str]] = []

    for ex in iter_jsonl(manifest_path):
        total += 1
        txt = ex.get("raw_transcription", "")
        ap = ex.get("audio_path", "")

        if (txt or "").strip() == "":
            empty_raw += 1
            if len(examples_raw) < show:
                examples_raw.append((ap, repr(txt)))

        norm = basic_whisperish_normalize(txt)
        if norm == "":
            empty_norm += 1
            if len(examples_norm) < show:
                examples_norm.append((ap, repr(txt)))

    print(f"Total rows: {total}")
    print(f"Empty/whitespace raw_transcription: {empty_raw}")
    print(f"Becomes empty after normalize(): {empty_norm}")

    if examples_raw:
        print("\n--- Examples: empty raw_transcription ---")
        for ap, r in examples_raw:
            print(ap)
            print("  ", r)

    if examples_norm:
        print("\n--- Examples: empty-after-normalize (danger for CE) ---")
        for ap, r in examples_norm:
            print(ap)
            print("  ", r)


# =========================
# Training-loop patch bits
# =========================

# Drop-in helper (PyTorch) to prevent NaN CE loss when there are zero valid targets.

def safe_cross_entropy_mean(logits, labels, ignore_index: int = -100):
    """Compute CE safely even when every target is ignore_index.

    Returns (loss, valid_count)
    - loss is a scalar tensor
    - valid_count is an int

    If valid_count == 0, loss is returned as 0 (so you can skip the step cleanly).
    """
    import torch
    import torch.nn.functional as F

    # logits: [B, T, V] or [N, V]
    if logits.dim() == 3:
        V = logits.size(-1)
        logits_2d = logits.reshape(-1, V)
    else:
        logits_2d = logits

    labels_1d = labels.reshape(-1)
    valid = labels_1d.ne(ignore_index)
    valid_count = int(valid.sum().item())

    if valid_count == 0:
        return logits_2d.new_tensor(0.0), 0

    # sum reduction avoids divide-by-zero; then we do the mean ourselves
    loss_sum = F.cross_entropy(logits_2d, labels_1d, ignore_index=ignore_index, reduction="sum")
    return loss_sum / valid_count, valid_count


def print_bad_batch(batch_rows: List[Dict[str, Any]], max_items: int = 20) -> None:
    print("\n=== BAD BATCH DETAILS ===")
    for i, ex in enumerate(batch_rows[:max_items]):
        ap = ex.get("audio_path", "")
        txt = ex.get("raw_transcription", "")
        print(f"[{i}] {ap}")
        print(f"     {repr(txt)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--show", type=int, default=30)
    args = p.parse_args()

    audit_manifest(args.manifest, show=args.show)


if __name__ == "__main__":
    main()
