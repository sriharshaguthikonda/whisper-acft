#!/usr/bin/env python3
r"""stage_21_rank_target_severity_confidence_weighted.py

Rank TARGET files by "severity" of TARGET transcript vs prediction, using
confidence-weighted scoring (if confidence/logprob is present) or heuristics
when it is not.

Input:
  evaluation_per_sample_predictions_targetmix_sweep.json

Filtering:
  Uses ONLY the least overlap + highest SNR condition available in the input.

Output:
  JSON only (sorted by aggregate severity).

Usage (PowerShell):
  I:\Whisper-training-env\Scripts\python.exe i:\whisper-acft\stage_21_rank_target_severity_confidence_weighted.py `
    --in_json "I:\Stage_17_aug_futo_wer_rank64_dora_dyn_ctx_chkpts_small_en_26\evaluation_per_sample_predictions_targetmix_sweep.json" `
    --out_json "I:\Stage_17_aug_futo_wer_rank64_dora_dyn_ctx_chkpts_small_en_26\target_severity_ranked.json"
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _basic_whisperish_normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"(?!\B'\b)[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in ("<|nospeech|>", "<|nocaptions|>"):
        return ""
    return s


def _tokenize_words(s: str) -> List[str]:
    s = _basic_whisperish_normalize(s)
    if not s:
        return []
    return s.split()


def _char_tokens(s: str) -> List[str]:
    s = _basic_whisperish_normalize(s)
    return list(s)


def _edit_distance(a: List[str], b: List[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[n]


def _wer(ref: str, hyp: str) -> float:
    rt = _tokenize_words(ref)
    ht = _tokenize_words(hyp)
    if not rt:
        return 0.0 if not ht else 1.0
    return float(_edit_distance(rt, ht)) / float(len(rt))


def _cer(ref: str, hyp: str) -> float:
    rc = _char_tokens(ref)
    hc = _char_tokens(hyp)
    if not rc:
        return 0.0 if not hc else 1.0
    return float(_edit_distance(rc, hc)) / float(len(rc))


def _ngram_repeat_ratio(tokens: List[str], n: int) -> float:
    if len(tokens) < n + 1:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    return float(repeats) / float(max(1, len(ngrams)))


def _repetition_penalty(tokens: List[str]) -> float:
    if len(tokens) < 4:
        return 0.0
    rep2 = _ngram_repeat_ratio(tokens, 2)
    rep3 = _ngram_repeat_ratio(tokens, 3)
    return min(1.0, 0.5 * rep2 + 0.5 * rep3)


def _length_penalty(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    diff = abs(len(hyp_tokens) - len(ref_tokens))
    return min(1.0, float(diff) / float(max(1, len(ref_tokens))))


def _extract_confidence(pred_entry: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    for key in (
        "avg_logprob",
        "logprob",
        "avg_logprob_whisper",
        "confidence",
        "avg_confidence",
        "prob",
        "avg_prob",
    ):
        if key in pred_entry:
            try:
                return key, float(pred_entry.get(key))
            except Exception:
                continue
    return None


def _normalize_confidence(key: str, val: float) -> float:
    if "logprob" in key:
        try:
            return float(max(0.0, min(1.0, math.exp(val))))
        except Exception:
            return 0.0
    if val > 1.0 and val <= 100.0:
        return float(max(0.0, min(1.0, val / 100.0)))
    return float(max(0.0, min(1.0, val)))


def _compute_severity(
    *,
    pred_entry: Dict[str, Any],
    target_ref: str,
    pred: str,
    wer_weight: float,
    cer_weight: float,
    len_weight: float,
    rep_weight: float,
    cap_weight: float,
    conf_weight: float,
) -> Dict[str, Any]:
    wer = pred_entry.get("wer_target")
    cer = pred_entry.get("cer_target")
    if wer is None:
        wer = _wer(target_ref, pred)
    if cer is None:
        cer = _cer(target_ref, pred)
    base_error = (float(wer_weight) * float(wer)) + (float(cer_weight) * float(cer))

    tokens_ref = _tokenize_words(target_ref)
    tokens_pred = _tokenize_words(pred)
    len_pen = _length_penalty(tokens_ref, tokens_pred)
    rep_pen = _repetition_penalty(tokens_pred)
    cap_pen = 1.0 if bool(pred_entry.get("likely_hit_max_token_cap")) else 0.0

    conf = _extract_confidence(pred_entry)
    if conf is not None:
        conf_norm = _normalize_confidence(conf[0], conf[1])
        multiplier = 1.0 + float(conf_weight) * conf_norm
        used_mode = "confidence"
        conf_payload = {"key": conf[0], "value": conf[1], "normalized": conf_norm}
        len_pen_used, rep_pen_used, cap_pen_used = 0.0, 0.0, 0.0
    else:
        multiplier = 1.0 + (float(len_weight) * len_pen) + (float(rep_weight) * rep_pen) + (float(cap_weight) * cap_pen)
        used_mode = "heuristic"
        conf_payload = None
        len_pen_used, rep_pen_used, cap_pen_used = len_pen, rep_pen, cap_pen

    severity = base_error * multiplier
    return {
        "severity": severity,
        "base_error": base_error,
        "wer_target": float(wer),
        "cer_target": float(cer),
        "mode": used_mode,
        "confidence": conf_payload,
        "len_penalty": len_pen_used,
        "rep_penalty": rep_pen_used,
        "cap_penalty": cap_pen_used,
        "multiplier": multiplier,
        "pred_tokens": len(tokens_pred),
        "ref_tokens": len(tokens_ref),
    }


def _iter_items(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, dict):
        items = data.get("items") or data.get("samples")
        if isinstance(items, list):
            for it in items:
                if isinstance(it, dict) and "predictions" in it:
                    yield it
            return
    if isinstance(data, list):
        for it in data:
            if not isinstance(it, dict):
                continue
            if "__meta__" in it:
                continue
            if "run_args" in it and "mix_key" not in it:
                continue
            if "predictions" in it:
                yield it


def _select_condition(items: List[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    overlaps: List[float] = []
    snrs: List[float] = []
    for it in items:
        ov = it.get("overlap", None)
        snr = it.get("snr_db", None)
        try:
            if ov is not None:
                overlaps.append(float(ov))
        except Exception:
            pass
        try:
            if snr is not None:
                snrs.append(float(snr))
        except Exception:
            pass
    min_ov = min(overlaps) if overlaps else None
    max_snr = max(snrs) if snrs else None
    return min_ov, max_snr


def _aggregate(values: List[float], mode: str) -> Optional[float]:
    if not values:
        return None
    vals = sorted(values)
    if mode == "max":
        return max(vals)
    if mode == "mean":
        return float(sum(vals)) / float(len(vals))
    if mode == "median":
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            return float(vals[mid])
        return float(vals[mid - 1] + vals[mid]) / 2.0
    if mode == "p90":
        idx = int(math.ceil(0.9 * len(vals))) - 1
        idx = max(0, min(idx, len(vals) - 1))
        return float(vals[idx])
    return max(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--aggregate", default="max", choices=["max", "mean", "median", "p90"])
    ap.add_argument("--wer_weight", type=float, default=0.7)
    ap.add_argument("--cer_weight", type=float, default=0.3)
    ap.add_argument("--len_weight", type=float, default=0.4)
    ap.add_argument("--rep_weight", type=float, default=0.4)
    ap.add_argument("--cap_weight", type=float, default=0.2)
    ap.add_argument("--conf_weight", type=float, default=0.5)
    args = ap.parse_args()

    in_path = Path(args.in_json)
    out_path = Path(args.out_json)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    items = list(_iter_items(data))
    if not items:
        raise SystemExit("No per-sample predictions found in input JSON.")

    min_overlap, max_snr = _select_condition(items)
    filtered: List[Dict[str, Any]] = []
    for it in items:
        try:
            snr_val = float(it.get("snr_db")) if it.get("snr_db") is not None else None
        except Exception:
            snr_val = None
        try:
            ov_val = float(it.get("overlap")) if it.get("overlap") is not None else None
        except Exception:
            ov_val = None

        if max_snr is not None and snr_val is not None and snr_val != max_snr:
            continue
        if min_overlap is not None and ov_val is not None and ov_val != min_overlap:
            continue
        filtered.append(it)

    if not filtered:
        raise SystemExit("No samples matched the least-overlap + highest-SNR filter.")

    # Group by target audio path; take worst per model across pairs for that target.
    targets: Dict[str, Dict[str, Any]] = {}
    for it in filtered:
        tpath = str(it.get("target_audio_path") or "")
        if not tpath:
            continue
        target_ref = it.get("target_reference") or it.get("target_ref") or ""
        preds = it.get("predictions", {})
        if not isinstance(preds, dict):
            continue

        tgt = targets.setdefault(
            tpath,
            {
                "target_audio_path": tpath,
                "target_reference": target_ref,
                "per_model": {},
            },
        )
        if not tgt.get("target_reference") and target_ref:
            tgt["target_reference"] = target_ref

        for model_name, pred_entry in preds.items():
            if not isinstance(pred_entry, dict):
                continue
            pred = pred_entry.get("pred", "") or ""
            sev = _compute_severity(
                pred_entry=pred_entry,
                target_ref=str(target_ref),
                pred=str(pred),
                wer_weight=float(args.wer_weight),
                cer_weight=float(args.cer_weight),
                len_weight=float(args.len_weight),
                rep_weight=float(args.rep_weight),
                cap_weight=float(args.cap_weight),
                conf_weight=float(args.conf_weight),
            )

            existing = tgt["per_model"].get(model_name)
            if existing is None or sev["severity"] > existing.get("severity", -1):
                tgt["per_model"][model_name] = {
                    "model": model_name,
                    "severity": sev["severity"],
                    "base_error": sev["base_error"],
                    "wer_target": sev["wer_target"],
                    "cer_target": sev["cer_target"],
                    "mode": sev["mode"],
                    "confidence": sev["confidence"],
                    "len_penalty": sev["len_penalty"],
                    "rep_penalty": sev["rep_penalty"],
                    "cap_penalty": sev["cap_penalty"],
                    "multiplier": sev["multiplier"],
                    "pred_tokens": sev["pred_tokens"],
                    "ref_tokens": sev["ref_tokens"],
                    "pred": pred,
                    "mix_key": it.get("mix_key"),
                    "snr_db": it.get("snr_db"),
                    "overlap": it.get("overlap"),
                    "other_audio_path": it.get("other_audio_path"),
                    "other_reference": it.get("other_reference") or it.get("other_ref"),
                }

    output_items: List[Dict[str, Any]] = []
    for tpath, rec in targets.items():
        per_model = list(rec.get("per_model", {}).values())
        severities = [float(x.get("severity")) for x in per_model if x.get("severity") is not None]
        agg = _aggregate(severities, str(args.aggregate))
        if agg is None:
            continue
        output_items.append(
            {
                "target_audio_path": tpath,
                "target_reference": rec.get("target_reference", ""),
                "severity_aggregate": agg,
                "aggregate_mode": str(args.aggregate),
                "per_model": sorted(per_model, key=lambda x: float(x.get("severity", 0.0)), reverse=True),
            }
        )

    output_items.sort(key=lambda x: float(x.get("severity_aggregate", 0.0)), reverse=True)

    out_obj = {
        "meta": {
            "input_json": str(in_path),
            "selected_overlap": min_overlap,
            "selected_snr_db": max_snr,
            "aggregate": str(args.aggregate),
            "weights": {
                "wer_weight": float(args.wer_weight),
                "cer_weight": float(args.cer_weight),
                "len_weight": float(args.len_weight),
                "rep_weight": float(args.rep_weight),
                "cap_weight": float(args.cap_weight),
                "conf_weight": float(args.conf_weight),
            },
            "targets": len(output_items),
            "filtered_pairs": len(filtered),
            "total_pairs": len(items),
        },
        "items": output_items,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
