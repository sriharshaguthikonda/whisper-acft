#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


try:
    from huggingface_hub import HfApi
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Failed to import huggingface_hub. Run with the training venv python and "
        "set PYTHONNOUSERSITE=1.\n"
        f"Import error: {exc}"
    )


MODEL_EXTENSIONS = {
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".ggml",
    ".onnx",
}


@dataclass
class LocalModelFile:
    abs_path: str
    rel_path: str
    size_bytes: int


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def resolve_hf_token(repo_root: Path) -> str:
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    env_values = parse_env_file(repo_root / ".env")
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        value = env_values.get(key, "").strip()
        if value:
            return value
    return ""


def safe_slug(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "unk"


def parse_run_name_tokens(run_name: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for piece in run_name.split("__")[1:]:
        if "-" not in piece:
            continue
        key, value = piece.split("-", 1)
        if key:
            tokens[key] = value
    return tokens


def build_repo_name(prefix: str, run_name: str) -> str:
    prefix_slug = safe_slug(prefix)
    tokens = parse_run_name_tokens(run_name)
    stage = safe_slug(tokens.get("s", "unk"))
    base = safe_slug(tokens.get("b", "unk"))[:18]
    method = safe_slug(tokens.get("m", "unk"))[:24]
    run_id = safe_slug(tokens.get("id", "unk"))[:20]
    digest = hashlib.blake2s(run_name.encode("utf-8"), digest_size=5).hexdigest()
    core = safe_slug(f"{prefix_slug}-s-{stage}-{base}-{method}-{run_id}")
    max_len = 96 - (len(digest) + 1)
    core = core[:max_len].rstrip("-")
    if not core:
        core = prefix_slug
    return f"{core}-{digest}"


def discover_run_folders(root: Path) -> list[Path]:
    folders = []
    for item in root.iterdir():
        if item.is_dir() and item.name.startswith("RUN__"):
            folders.append(item)
    return sorted(folders, key=lambda p: p.name.lower())


def discover_model_files(run_folder: Path, min_size_mb: int) -> list[LocalModelFile]:
    threshold = int(min_size_mb * 1024 * 1024)
    out: list[LocalModelFile] = []
    for p in run_folder.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in MODEL_EXTENSIONS:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size < threshold:
            continue
        rel = p.relative_to(run_folder).as_posix()
        out.append(LocalModelFile(abs_path=str(p), rel_path=rel, size_bytes=size))
    return sorted(out, key=lambda x: x.rel_path)


def repo_file_size_map(api: HfApi, repo_id: str, token: str) -> tuple[dict[str, int], str]:
    info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True, token=token)
    size_map: dict[str, int] = {}
    for sibling in info.siblings or []:
        if sibling.rfilename:
            size_map[sibling.rfilename] = int(getattr(sibling, "size", -1) or -1)
    revision = getattr(info, "sha", "") or ""
    return size_map, revision


def write_pointer_files(
    run_folder: Path,
    pointer: dict,
    cache_dir: Path,
) -> None:
    pointer_path = run_folder / "MODEL_POINTER.json"
    readme_path = run_folder / "MODEL_POINTER_README.md"
    url_path = run_folder / "HF_MODEL_REPO.url"

    pointer_path.write_text(json.dumps(pointer, indent=2), encoding="utf-8")

    repo_id = pointer["repo_id"]
    revision = pointer.get("revision", "main")
    restore_cmd = (
        f'python "I:\\whisper-acft\\tools\\hf_tier1_restore_from_pointer.py" '
        f'--pointer "{pointer_path}" --cache-dir "{cache_dir}" --local-dir "{run_folder}\\restored_model"'
    )
    readme = (
        "# Model Pointer\n\n"
        f"- Repo: https://huggingface.co/{repo_id}\n"
        f"- Revision: `{revision}`\n"
        "- Privacy: private\n"
        f"- Pointer JSON: `{pointer_path.name}`\n\n"
        "## Restore (hybrid cache)\n\n"
        "Use central cache + optional local restore:\n\n"
        "```powershell\n"
        f"{restore_cmd}\n"
        "```\n\n"
        "## Direct Hub CLI restore\n\n"
        "```powershell\n"
        f'hf download "{repo_id}" --repo-type model --revision "{revision}" --cache-dir "{cache_dir}" --local-dir "{run_folder}\\restored_model"\n'
        "```\n"
    )
    readme_path.write_text(readme, encoding="utf-8")

    url_path.write_text(
        "[InternetShortcut]\n"
        f"URL=https://huggingface.co/{repo_id}\n",
        encoding="utf-8",
    )


def bytes_to_gb(value: int) -> float:
    return round(value / (1024**3), 3)


def load_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def choose_run_folders(
    runs: list[Path],
    include: Iterable[str],
    exclude: Iterable[str],
    max_runs: int,
) -> list[Path]:
    include_set = {item.strip() for item in include if item.strip()}
    exclude_set = {item.strip() for item in exclude if item.strip()}
    selected: list[Path] = []
    for run in runs:
        name = run.name
        if include_set and name not in include_set:
            continue
        if name in exclude_set:
            continue
        selected.append(run)
    if max_runs > 0:
        return selected[:max_runs]
    return selected


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier-1 private archival for local RUN__ folders.")
    ap.add_argument("--root", default=r"I:\\", help="Root path that contains RUN__ folders.")
    ap.add_argument("--repo-root", default=r"I:\\whisper-acft", help="Repo root (for .env/report files).")
    ap.add_argument("--repo-prefix", default="Whisper-acft", help="HF repo name prefix.")
    ap.add_argument("--cache-dir", default=r"I:\\hf_model_cache", help="Central restore cache path.")
    ap.add_argument("--min-size-mb", type=int, default=50, help="Only archive files >= this size.")
    ap.add_argument("--max-runs", type=int, default=0, help="Limit number of runs (0 = all).")
    ap.add_argument("--include-run", action="append", default=[], help="Include only specific run folder names.")
    ap.add_argument("--exclude-run", action="append", default=[], help="Exclude specific run folder names.")
    ap.add_argument("--dry-run", action="store_true", help="Do not upload/delete; only report.")
    ap.add_argument(
        "--cleanup-local",
        action="store_true",
        help="Delete local archived model files after upload verification.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    repo_root = Path(args.repo_root)
    cache_dir = Path(args.cache_dir)
    report_dir = repo_root / "hf_tier1_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    token = resolve_hf_token(repo_root)
    if not token:
        raise SystemExit("HF token not found in env/.env")
    os.environ["HF_TOKEN"] = token

    api = HfApi()
    who = api.whoami(token=token)
    username = who.get("name", "").strip()
    if not username:
        raise SystemExit(f"Unable to resolve HF username from whoami(): {who}")

    all_runs = discover_run_folders(root)
    selected_runs = choose_run_folders(all_runs, args.include_run, args.exclude_run, args.max_runs)

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_report_path = report_dir / f"hf_tier1_archive_{session_id}.json"
    csv_report_path = report_dir / f"hf_tier1_archive_{session_id}.csv"
    registry_path = repo_root / "HF_PRIVATE_TIER1_REGISTRY.json"

    rows: list[dict] = []
    total_uploaded = 0
    total_deleted = 0
    total_freed = 0

    print(f"[info] username={username}")
    print(f"[info] runs_found={len(all_runs)} selected={len(selected_runs)}")
    print(f"[info] dry_run={args.dry_run} cleanup_local={args.cleanup_local}")

    for idx, run in enumerate(selected_runs, start=1):
        files = discover_model_files(run, args.min_size_mb)
        if not files:
            rows.append(
                {
                    "run_folder": run.name,
                    "repo_id": "",
                    "status": "skip_no_files",
                    "file_count": 0,
                    "upload_bytes": 0,
                    "deleted_bytes": 0,
                    "revision": "",
                    "note": f"no model files >= {args.min_size_mb}MB",
                }
            )
            continue

        repo_name = build_repo_name(args.repo_prefix, run.name)
        repo_id = f"{username}/{repo_name}"
        print(f"[{idx}/{len(selected_runs)}] {run.name} -> {repo_id} ({len(files)} files)")

        if args.dry_run:
            upload_bytes = sum(f.size_bytes for f in files)
            rows.append(
                {
                    "run_folder": run.name,
                    "repo_id": repo_id,
                    "status": "dry_run",
                    "file_count": len(files),
                    "upload_bytes": upload_bytes,
                    "deleted_bytes": 0,
                    "revision": "",
                    "note": "",
                }
            )
            continue

        try:
            api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True, token=token)
            commit_info = api.upload_folder(
                folder_path=str(run),
                path_in_repo="",
                repo_id=repo_id,
                repo_type="model",
                allow_patterns=[f.rel_path for f in files],
                commit_message=f"Archive model artifacts from {run.name}",
                token=token,
            )

            remote_sizes, repo_revision = repo_file_size_map(api, repo_id, token)
            missing = [f.rel_path for f in files if f.rel_path not in remote_sizes]
            size_mismatch = [
                f.rel_path for f in files if f.rel_path in remote_sizes and remote_sizes[f.rel_path] != f.size_bytes
            ]
            if missing or size_mismatch:
                rows.append(
                    {
                        "run_folder": run.name,
                        "repo_id": repo_id,
                        "status": "verify_failed",
                        "file_count": len(files),
                        "upload_bytes": sum(f.size_bytes for f in files),
                        "deleted_bytes": 0,
                        "revision": repo_revision or getattr(commit_info, "oid", ""),
                        "note": f"missing={len(missing)} mismatch={len(size_mismatch)}",
                    }
                )
                continue

            deleted_bytes = 0
            deleted_paths: list[str] = []
            if args.cleanup_local:
                for f in files:
                    p = Path(f.abs_path)
                    if not p.exists():
                        continue
                    p.unlink()
                    deleted_bytes += f.size_bytes
                    deleted_paths.append(f.rel_path)

            revision = repo_revision or getattr(commit_info, "oid", "")
            pointer = {
                "pointer_version": "1.0",
                "created_at_utc": utc_now(),
                "run_folder": run.name,
                "local_run_path": str(run),
                "repo_id": repo_id,
                "repo_url": f"https://huggingface.co/{repo_id}",
                "repo_type": "model",
                "private": True,
                "revision": revision or "main",
                "cache_strategy": {
                    "mode": "hybrid",
                    "central_cache_dir": str(cache_dir),
                    "local_restore_dir_default": str(run / "restored_model"),
                },
                "archive_policy": {
                    "min_size_mb": args.min_size_mb,
                    "extensions": sorted(MODEL_EXTENSIONS),
                    "cleanup_local_enabled": bool(args.cleanup_local),
                },
                "archived_files": [asdict(f) for f in files],
                "deleted_local_files": deleted_paths,
            }
            write_pointer_files(run, pointer, cache_dir)

            upload_bytes = sum(f.size_bytes for f in files)
            total_uploaded += len(files)
            total_deleted += len(deleted_paths)
            total_freed += deleted_bytes

            rows.append(
                {
                    "run_folder": run.name,
                    "repo_id": repo_id,
                    "status": "uploaded",
                    "file_count": len(files),
                    "upload_bytes": upload_bytes,
                    "deleted_bytes": deleted_bytes,
                    "revision": revision,
                    "note": "",
                }
            )
        except Exception as exc:  # pragma: no cover
            rows.append(
                {
                    "run_folder": run.name,
                    "repo_id": repo_id,
                    "status": "error",
                    "file_count": len(files),
                    "upload_bytes": sum(f.size_bytes for f in files),
                    "deleted_bytes": 0,
                    "revision": "",
                    "note": str(exc),
                }
            )

    summary = {
        "created_at_utc": utc_now(),
        "root": str(root),
        "repo_root": str(repo_root),
        "repo_prefix": args.repo_prefix,
        "username": username,
        "min_size_mb": args.min_size_mb,
        "dry_run": bool(args.dry_run),
        "cleanup_local": bool(args.cleanup_local),
        "runs_found": len(all_runs),
        "runs_selected": len(selected_runs),
        "files_uploaded_count": total_uploaded,
        "files_deleted_count": total_deleted,
        "bytes_freed": total_freed,
        "gb_freed": bytes_to_gb(total_freed),
        "rows": rows,
    }

    json_report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_folder", "repo_id", "status", "file_count", "upload_bytes", "deleted_bytes", "revision", "note"],
        )
        writer.writeheader()
        writer.writerows(rows)

    existing_registry = load_json_file(registry_path)
    merged: dict[str, dict] = {}
    for item in existing_registry.get("entries", []):
        if isinstance(item, dict) and item.get("run_folder"):
            merged[item["run_folder"]] = item
    for row in rows:
        run_folder = row.get("run_folder", "")
        repo_id = row.get("repo_id", "")
        if not run_folder or not repo_id:
            continue
        merged[run_folder] = {
            "run_folder": run_folder,
            "repo_id": repo_id,
            "status": row.get("status", ""),
            "revision": row.get("revision", ""),
            "updated_at_utc": utc_now(),
        }

    report_history = existing_registry.get("report_history", [])
    if not isinstance(report_history, list):
        report_history = []
    report_history.append(
        {
            "json": str(json_report_path),
            "csv": str(csv_report_path),
            "created_at_utc": utc_now(),
            "dry_run": bool(args.dry_run),
            "cleanup_local": bool(args.cleanup_local),
        }
    )
    report_history = report_history[-200:]

    registry = {
        "updated_at_utc": utc_now(),
        "repo_prefix": args.repo_prefix,
        "username": username,
        "min_size_mb": args.min_size_mb,
        "entries": sorted(merged.values(), key=lambda x: x.get("run_folder", "")),
        "latest_report_json": str(json_report_path),
        "latest_report_csv": str(csv_report_path),
        "report_history": report_history,
    }
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    print(f"[done] report_json={json_report_path}")
    print(f"[done] report_csv={csv_report_path}")
    print(f"[done] registry={registry_path}")
    print(
        "[done] summary: "
        f"uploaded_files={total_uploaded} deleted_files={total_deleted} freed_gb={bytes_to_gb(total_freed)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
