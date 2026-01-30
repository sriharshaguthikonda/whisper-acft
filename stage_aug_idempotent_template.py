"""Template for *any* augmentation stage.

Key behaviour:
- Only augments originals by default (uid==base_uid and aug_stage empty)
- Selection is deterministic per base_uid
- Each output copy has a stable uid and stable file name
- Uses SQLiteSeenSet to skip already-created outputs (resume-friendly)

You must implement `do_augmentation(row, rng, out_wav, args)`.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from tqdm import tqdm

from pipeline_uid_utils import (
    SQLiteSeenSet,
    default_seen_db,
    is_valid_wav,
    safe_unlink,
    make_aug_uid,
    rng_for,
    safe_beep,
    should_select,
)


def do_augmentation(row: Dict[str, Any], rng, out_wav: str, args) -> Dict[str, Any]:
    """Implement the stage-specific augmentation.

    Must create `out_wav` on disk and return extra metadata to store in JSON.

    Example return:
      {"snr_db": 8.1, "noise_file": "...", "mix_ratio": 0.35}
    """
    raise NotImplementedError


def build_out_wav_name(row: Dict[str, Any], stage_name: str, new_uid: str, copy_idx: int, out_dir: Path) -> str:
    """Build stable output filename with shorter format to avoid path length issues."""
    # 8 hex chars (32 bits) can collide once you scale; 12 chars (48 bits) is effectively safe here.
    base_uid = row.get("base_uid", "")[:12]
    aug_uid = new_uid[:12]
    fname = f"{base_uid}_{aug_uid}__{stage_name}__c{copy_idx:02d}.wav"
    return str(out_dir / fname)


def is_original_row(row: Dict[str, Any]) -> bool:
    # Treat as original if no aug_stage and uid==base_uid (or uid missing)
    if row.get("aug_stage"):
        return False
    base_uid = row.get("base_uid")
    uid = row.get("uid") or base_uid
    if base_uid and uid and uid != base_uid:
        return False
    return True


def process_one(
    row: Dict[str, Any],
    stage_name: str,
    copy_idx: int,
    out_dir: Path,
    args,
    seen: SQLiteSeenSet,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    base_uid = row.get("base_uid") or row.get("uid")
    if not base_uid:
        return False, None, "missing base_uid"

    aug_key = f"{base_uid}:{stage_name}:{copy_idx}"
    if seen.contains(aug_key):
        return True, None, "already-seen"

    new_uid = make_aug_uid(base_uid, stage_name, copy_idx)
    out_wav = build_out_wav_name(row, stage_name, new_uid, copy_idx, out_dir)

    # Idempotent on disk (robust): only skip if existing WAV is actually readable.
    if os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
        if is_valid_wav(out_wav, min_frames=16):
            seen.add(aug_key)
            return True, None, "already-exists"
        # Bad/corrupt partial output: delete and redo.
        safe_unlink(out_wav)

    rng = rng_for(base_uid, stage_name, copy_idx)
    meta = do_augmentation(row, rng, out_wav, args)

    # Post-write validation: if writer crashed/bugged, do NOT mark seen.
    if not (os.path.exists(out_wav) and os.path.getsize(out_wav) > 0 and is_valid_wav(out_wav, min_frames=16)):
        safe_unlink(out_wav)
        return False, None, "write-produced-invalid-wav"

    out_row = dict(row)
    out_row["parent_uid"] = row.get("uid") or base_uid
    out_row["base_uid"] = base_uid
    out_row["uid"] = new_uid
    out_row["aug_stage"] = stage_name
    out_row["aug_copy_idx"] = copy_idx
    out_row["out_wav"] = out_wav
    out_row["audio_path"] = out_wav  # keep downstream stages simple
    out_row.setdefault("aug_meta", {})
    out_row["aug_meta"] = {**out_row["aug_meta"], **(meta or {})}

    seen.add(aug_key)
    return True, out_row, "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True, help="Input jsonl")
    ap.add_argument("--out_manifest", required=True, help="Output jsonl (augmented rows appended)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--stage_name", required=True)
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seen_db", default="", help="SQLite file; default is out_manifest + .seen.sqlite")
    ap.add_argument("--allow_augmented_input", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.in_manifest)
    out_path = Path(args.out_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_db = args.seen_db or default_seen_db(out_path, args.stage_name)
    seen = SQLiteSeenSet(seen_db)

    # Read input rows (streaming) -> schedule futures
    futures = []
    n_total = 0
    n_selected = 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                n_total += 1

                if not args.allow_augmented_input and not is_original_row(row):
                    continue

                base_uid = row.get("base_uid") or row.get("uid")
                if not base_uid:
                    continue

                if not should_select(base_uid, args.stage_name, args.ratio):
                    continue

                n_selected += 1
                for copy_idx in range(1, args.copies + 1):
                    futures.append(ex.submit(process_one, row, args.stage_name, copy_idx, out_dir, args, seen))

            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{args.stage_name} augment"):
                ok, out_row, status = fut.result()
                if out_row:
                    f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    seen.commit()
    seen.close()

    print(f"Stage {args.stage_name}: scanned {n_total} rows; selected {n_selected} base rows; emitted up to {n_selected*args.copies} aug rows")
    safe_beep()


if __name__ == "__main__":
    main()
