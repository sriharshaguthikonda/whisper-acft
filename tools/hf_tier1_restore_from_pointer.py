#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_env_file(env_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not env_path.exists():
        return out
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            out[key] = value
    return out


def resolve_hf_token(pointer_path: Path) -> str:
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    repo_root = Path(r"I:\whisper-acft")
    env_values = parse_env_file(repo_root / ".env")
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        value = env_values.get(key, "").strip()
        if value:
            return value
    env_values = parse_env_file(pointer_path.parent.parent / ".env")
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        value = env_values.get(key, "").strip()
        if value:
            return value
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="Restore archived model files from MODEL_POINTER.json")
    ap.add_argument("--pointer", required=True, help="Path to MODEL_POINTER.json")
    ap.add_argument("--cache-dir", default=r"I:\hf_model_cache", help="Central HF cache directory")
    ap.add_argument("--local-dir", default="", help="Optional restore directory")
    ap.add_argument("--force-download", action="store_true", help="Force fresh download")
    args = ap.parse_args()

    pointer_path = Path(args.pointer)
    if not pointer_path.exists():
        raise SystemExit(f"Pointer file not found: {pointer_path}")

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    repo_id = pointer.get("repo_id", "").strip()
    revision = pointer.get("revision", "").strip() or "main"
    if not repo_id:
        raise SystemExit(f"Invalid pointer file (missing repo_id): {pointer_path}")

    token = resolve_hf_token(pointer_path)
    if not token:
        raise SystemExit("HF token missing in env/.env")

    local_dir = args.local_dir.strip() or pointer.get("cache_strategy", {}).get("local_restore_dir_default", "")
    local_dir = local_dir.strip()
    local_dir_arg = local_dir if local_dir else None

    path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        token=token,
        cache_dir=args.cache_dir,
        local_dir=local_dir_arg,
        force_download=bool(args.force_download),
    )

    print(f"repo_id={repo_id}")
    print(f"revision={revision}")
    print(f"cache_dir={args.cache_dir}")
    if local_dir_arg:
        print(f"local_dir={local_dir_arg}")
    print(f"restored_path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
