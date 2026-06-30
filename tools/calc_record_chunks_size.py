import os
from pathlib import Path

root = Path(r"I:\Record_chunks")
if not root.exists():
    print("exists=False")
    raise SystemExit(0)

total = 0
count = 0
for p in root.rglob("*"):
    if p.is_file():
        count += 1
        try:
            total += p.stat().st_size
        except Exception:
            pass

print(f"exists=True")
print(f"files={count}")
print(f"bytes={total}")
print(f"gb={total/(1024**3):.3f}")
