#!/usr/bin/env python3
r"""stage_21b_apply_corrections_to_per_sample_predictions.py

Apply corrected TARGET transcripts from target_severity_ranked.json
back into evaluation_per_sample_predictions_targetmix_sweep.json files.

Usage (PowerShell):
  I:\Whisper-training-env\Scripts\python.exe i:\whisper-acft\stage_21b_apply_corrections_to_per_sample_predictions.py `
    --corrections_json "I:\Stage_17_aug_futo_wer_rank32_dora_dyn_ctx_chkpts_small_en_25\target_severity_ranked.json" `
    --in_jsons "I:\Stage_17_aug_futo_wer_rank32_dora_dyn_ctx_chkpts_small_en_25\evaluation_per_sample_predictions_targetmix_sweep.json"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def canonical_audio_key(p: str) -> str:
    if not p:
        return ""
    return os.path.normpath(p).replace("\\", "/").lower()


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    return i


def _best_effort_parse_array(text: str) -> List[Any]:
    items: List[Any] = []
    dec = json.JSONDecoder()
    i = _skip_ws(text, 0)
    if i >= len(text) or text[i] != "[":
        return items
    i += 1
    while True:
        i = _skip_ws(text, i)
        if i >= len(text):
            break
        if text[i] == "]":
            break
        try:
            obj, j = dec.raw_decode(text, i)
            items.append(obj)
            i = _skip_ws(text, j)
            if i < len(text) and text[i] == ",":
                i += 1
                continue
            if i < len(text) and text[i] == "]":
                break
        except json.JSONDecodeError:
            break
    return items


def load_json_any(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[warn] JSONDecodeError: {e.msg} at line {e.lineno} col {e.colno}")
        text = path.read_text(encoding="utf-8", errors="replace")
        items = _best_effort_parse_array(text)
        if items:
            print(f"[warn] Proceeding with best-effort parse: {len(items)} items recovered.")
            return items
        raise


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _extract_items(root: Any) -> Tuple[List[Dict[str, Any]], str]:
    if isinstance(root, dict):
        items = root.get("items")
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)], "dict_items"
        items = root.get("samples")
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)], "dict_samples"
    if isinstance(root, list):
        return [x for x in root if isinstance(x, dict)], "list"
    return [], "unknown"


def _extract_corrected_map(corrections_root: Any) -> Tuple[Dict[str, str], Dict[str, Any]]:
    items, _ = _extract_items(corrections_root)
    corrected: Dict[str, str] = {}
    conflicts = 0
    total = 0
    for item in items:
        if item.get("__meta__"):
            continue
        path = item.get("target_audio_path") or item.get("target_path")
        if not isinstance(path, str) or not path.strip():
            continue
        key = canonical_audio_key(path)
        if not key:
            continue
        # Prefer explicit corrected fields if present, else use target_reference.
        ref = (
            item.get("corrected_target_reference")
            or item.get("target_reference_corrected")
            or item.get("target_reference")
            or item.get("target_ref")
        )
        if not isinstance(ref, str):
            continue
        ref = ref.strip()
        if not ref:
            continue
        total += 1
        if key in corrected and corrected[key] != ref:
            conflicts += 1
        corrected[key] = ref
    stats = {"total_entries": total, "unique_targets": len(corrected), "conflicts": conflicts}
    return corrected, stats


def _resolve_input_paths(paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if not p:
            continue
        path = Path(p)
        if path.is_dir():
            candidate = path / "evaluation_per_sample_predictions_targetmix_sweep.json"
            if candidate.exists():
                out.append(candidate)
            else:
                print(f"[warn] No per-sample JSON found in dir: {path}")
            continue
        if path.exists():
            out.append(path)
        else:
            print(f"[warn] Missing input path: {path}")
    # de-dup
    seen = set()
    uniq: List[Path] = []
    for p in out:
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def apply_corrections_to_file(
    in_path: Path,
    corrected_map: Dict[str, str],
    *,
    inplace: bool,
    out_dir: Optional[Path],
    dry_run: bool,
) -> Dict[str, Any]:
    root = load_json_any(in_path)
    items, mode = _extract_items(root)
    if not items:
        return {"file": str(in_path), "updated": 0, "matched": 0, "mode": mode, "skipped": 0}

    updated = 0
    matched = 0
    for item in items:
        if item.get("__meta__"):
            continue
        path = item.get("target_audio_path") or item.get("target_path")
        if not isinstance(path, str) or not path.strip():
            continue
        key = canonical_audio_key(path)
        if key not in corrected_map:
            continue
        matched += 1
        new_ref = corrected_map[key]
        cur_ref = item.get("target_reference")
        if cur_ref != new_ref:
            updated += 1
            if not dry_run:
                item["target_reference"] = new_ref
                if "target_ref" in item:
                    item["target_ref"] = new_ref

    if dry_run:
        return {"file": str(in_path), "updated": updated, "matched": matched, "mode": mode, "skipped": 0}

    if out_dir is not None:
        out_path = out_dir / in_path.name
        write_json(out_path, root)
    elif inplace:
        backup = in_path.with_suffix(in_path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(in_path.read_text(encoding="utf-8"), encoding="utf-8")
        write_json(in_path, root)

    return {"file": str(in_path), "updated": updated, "matched": matched, "mode": mode, "skipped": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrections_json", required=True)
    ap.add_argument("--in_jsons", nargs="+", required=True, help="Per-sample JSON files or directories.")
    ap.add_argument("--out_dir", default=None, help="Optional output directory (writes new files).")
    ap.add_argument("--inplace", action="store_true", default=True, help="Modify inputs in place (default).")
    ap.add_argument("--dry_run", action="store_true", help="Report counts without writing changes.")
    args = ap.parse_args()

    corrections_path = Path(args.corrections_json)
    if not corrections_path.exists():
        raise SystemExit(f"Missing corrections_json: {corrections_path}")

    corrections_root = load_json_any(corrections_path)
    corrected_map, stats = _extract_corrected_map(corrections_root)
    print(f"Corrections: {stats['unique_targets']} targets (conflicts={stats['conflicts']})")
    if not corrected_map:
        raise SystemExit("No corrected target references found in corrections_json.")

    in_paths = _resolve_input_paths(args.in_jsons)
    if not in_paths:
        raise SystemExit("No valid input JSONs to update.")

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    total_updated = 0
    total_matched = 0
    for p in in_paths:
        res = apply_corrections_to_file(
            p,
            corrected_map,
            inplace=bool(args.inplace) and out_dir is None,
            out_dir=out_dir,
            dry_run=bool(args.dry_run),
        )
        total_updated += int(res.get("updated", 0))
        total_matched += int(res.get("matched", 0))
        print(f"{res['file']}: matched={res['matched']} updated={res['updated']}")

    print(f"Done. matched={total_matched} updated={total_updated}")


if __name__ == "__main__":
    main()
