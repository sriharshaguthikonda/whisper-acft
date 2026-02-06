"""Run this once if you already have manifests without base_uid.

Example:
  python backfill_uids_in_manifest.py --in_jsonl I:\\Record_chunks\\tasks_pending.jsonl --out_jsonl I:\\Record_chunks\\tasks_pending__uid.jsonl

- Streams line-by-line (handles huge files)
- Writes a new jsonl (doesn't destroy your old one)
"""

import json
from pathlib import Path

from pipeline_uid_utils import make_base_uid, safe_beep

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm

    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--uid_extra", default="")
    args = ap.parse_args()

    in_path = Path(args.in_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_added = 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, desc="Backfilling base_uid"):
            line = line.strip()
            if not line:
                continue
            n_in += 1
            row = json.loads(line)

            # Try to find stable chunk identity fields
            orig = row.get("orig_audio_path") or row.get("audio_path") or row.get("source_audio_path") or ""
            chunk_idx = int(row.get("chunk_index") or row.get("chunk_idx") or 0)
            core_start = float(row.get("core_start") or row.get("start") or 0.0)
            core_end = float(row.get("core_end") or row.get("end") or 0.0)

            if not row.get("base_uid"):
                row["base_uid"] = make_base_uid(orig, chunk_idx, core_start, core_end, extra=args.uid_extra)
                n_added += 1

            # If uid missing, set uid==base_uid for original rows
            if not row.get("uid"):
                row["uid"] = row["base_uid"]

            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. Read {n_in} rows; added base_uid to {n_added} rows.")
    safe_beep()
