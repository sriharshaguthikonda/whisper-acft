import json
import os
from pathlib import Path

cases = {
  "train_only": [r"I:\Record_chunks\pairs_manifest_stage15_train_no_targets_randomized.jsonl"],
  "train_plus_test": [
    r"I:\Record_chunks\pairs_manifest_stage15_train_no_targets_randomized.jsonl",
    r"I:\Record_chunks\pairs_manifest_stage13_test_randomized.jsonl",
  ],
}

for name, manifests in cases.items():
    seen = set()
    total = 0
    missing = 0
    rows = 0
    for mp in manifests:
        p = Path(mp)
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows += 1
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                ap = str(obj.get("audio_path") or "").strip()
                if not ap:
                    continue
                key = os.path.normpath(ap).replace("\\", "/").lower()
                if key in seen:
                    continue
                seen.add(key)
                if os.path.exists(ap):
                    total += os.path.getsize(ap)
                else:
                    missing += 1
    print(name, "rows", rows, "unique", len(seen), "missing", missing, "gb", round(total/(1024**3), 3))
