#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import shutil
import math
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cloud_eval_common import parse_env_file, resolve_hf_token

try:
    from huggingface_hub import HfApi
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing huggingface_hub. Install in your environment first:\n"
        "python -m pip install huggingface_hub"
    ) from exc


def _utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(value: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-") or "pack"


def _norm_key(path_value: str) -> str:
    p = (path_value or "").strip().strip('"').strip("'")
    p = p.replace("/", "\\")
    try:
        p = str(PureWindowsPath(p))
    except Exception:
        pass
    return p.lower()


def _iter_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_input_path(raw: str, source_root: Path) -> Path:
    p = Path(str(raw))
    if p.is_absolute():
        return p
    return (source_root / p).resolve()


def _portable_relpath(src_path: Path, source_root: Path) -> str:
    try:
        rel = src_path.resolve().relative_to(source_root.resolve())
        return rel.as_posix()
    except Exception:
        pass

    # Keep drive letter for absolute paths outside source_root.
    anchor = src_path.anchor
    if anchor:
        drive = anchor.replace("\\", "").replace("/", "").replace(":", "")
        try:
            tail = src_path.resolve().relative_to(src_path.anchor)
        except Exception:
            tail = src_path.name
        if isinstance(tail, Path):
            tail_str = tail.as_posix()
        else:
            tail_str = str(tail).replace("\\", "/")
        return f"{drive}/{tail_str}".strip("/")
    return src_path.name


def _rel_to(base: Path, p: Path) -> str:
    try:
        return p.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return str(p).replace("\\", "/")


def _iter_dirs(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    dirs = [root]
    dirs.extend(sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda x: x.as_posix().lower()))
    return dirs


def _immediate_file_count(dir_path: Path) -> int:
    return sum(1 for p in dir_path.iterdir() if p.is_file())


def _collect_directory_file_counts(root: Path, base: Path, *, non_zero_only: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dir_path in _iter_dirs(root):
        file_count = _immediate_file_count(dir_path)
        if non_zero_only and file_count <= 0:
            continue
        rows.append(
            {
                "directory": _rel_to(base, dir_path),
                "file_count": int(file_count),
            }
        )
    rows.sort(key=lambda r: (-int(r["file_count"]), str(r["directory"]).lower()))
    return rows


def _validate_directory_file_limit(root: Path, base: Path, *, max_files_per_dir: int) -> list[dict[str, Any]]:
    offenders: list[dict[str, Any]] = []
    for dir_path in _iter_dirs(root):
        file_count = _immediate_file_count(dir_path)
        if file_count > max_files_per_dir:
            offenders.append(
                {
                    "directory": _rel_to(base, dir_path),
                    "file_count": int(file_count),
                }
            )
    offenders.sort(key=lambda r: (-int(r["file_count"]), str(r["directory"]).lower()))
    return offenders


def _apply_audio_dir_sharding(
    *,
    staging_root: Path,
    audio_root: Path,
    path_map: dict[str, str],
    max_files_per_dir: int,
    shard_prefix: str,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "max_files_per_dir": int(max_files_per_dir),
        "shard_prefix": shard_prefix,
        "directories_sharded": [],
    }
    if not audio_root.exists():
        return report

    reverse_map: dict[str, list[str]] = {}
    for src_key, dst_rel in path_map.items():
        nk = str(dst_rel).replace("\\", "/").strip("/")
        reverse_map.setdefault(nk, []).append(src_key)

    for dir_path in _iter_dirs(audio_root):
        files = [p for p in dir_path.iterdir() if p.is_file()]
        if len(files) <= max_files_per_dir:
            continue

        files_sorted = sorted(files, key=lambda p: (p.name.lower(), p.name))
        shard_rows: list[dict[str, Any]] = []
        shard_total = int(math.ceil(len(files_sorted) / float(max_files_per_dir)))
        for shard_idx in range(shard_total):
            start = shard_idx * max_files_per_dir
            end = min(start + max_files_per_dir, len(files_sorted))
            shard_name = f"{shard_prefix}_{shard_idx:04d}"
            shard_dir = dir_path / shard_name
            shard_dir.mkdir(parents=True, exist_ok=True)

            for src in files_sorted[start:end]:
                dst = shard_dir / src.name
                old_rel = _rel_to(staging_root, src).replace("\\", "/").strip("/")
                new_rel = _rel_to(staging_root, dst).replace("\\", "/").strip("/")
                shutil.move(str(src), str(dst))
                keys = reverse_map.pop(old_rel, [])
                if keys:
                    reverse_map.setdefault(new_rel, [])
                for key in keys:
                    path_map[key] = new_rel
                    reverse_map[new_rel].append(key)

            shard_rows.append(
                {
                    "shard": _rel_to(staging_root, shard_dir),
                    "file_count": int(end - start),
                }
            )

        report["directories_sharded"].append(
            {
                "directory": _rel_to(staging_root, dir_path),
                "files_before": int(len(files_sorted)),
                "shards_created": int(shard_total),
                "shards": shard_rows,
            }
        )

    return report


def _copy_audio_files(
    audio_paths: list[Path],
    source_root: Path,
    pack_root: Path,
) -> tuple[dict[str, str], list[str], int]:
    mapping: dict[str, str] = {}
    missing: list[str] = []
    copied = 0
    for src in audio_paths:
        key = _norm_key(str(src))
        if not src.exists():
            missing.append(str(src))
            continue
        rel = _portable_relpath(src, source_root)
        dst_rel = Path("audio") / rel
        dst = pack_root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        mapping[key] = dst_rel.as_posix()
        copied += 1
    return mapping, missing, copied


def _copy_entire_folder(
    folder: Path,
    source_root: Path,
    pack_root: Path,
    *,
    target_root: str = "audio",
) -> tuple[int, int]:
    copied = 0
    skipped = 0
    for src in folder.rglob("*"):
        if not src.is_file():
            continue
        rel = _portable_relpath(src, source_root)
        dst = pack_root / target_root / rel
        if dst.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def _rewrite_manifest_rows(rows: list[dict], path_map: dict[str, str]) -> tuple[list[dict], int]:
    rewritten: list[dict] = []
    changed = 0
    for row in rows:
        out = dict(row)
        audio_path = str(row.get("audio_path", "") or "").strip()
        mapped = path_map.get(_norm_key(audio_path))
        if mapped:
            out["audio_path"] = mapped
            changed += 1
        rewritten.append(out)
    return rewritten, changed


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _rewrite_speaker_scores(src_csv: Path, dst_csv: Path, path_map: dict[str, str]) -> tuple[int, int]:
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    with src_csv.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        fieldnames = list(reader.fieldnames or [])
        if "file" not in fieldnames:
            raise ValueError(f"speaker_scores CSV missing 'file' column: {src_csv}")
        rows = list(reader)

    changed = 0
    for row in rows:
        if None in row:
            row.pop(None, None)
        raw = str(row.get("file", "") or "")
        mapped = path_map.get(_norm_key(raw))
        if mapped:
            row["file"] = mapped
            changed += 1

    with dst_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            clean_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(clean_row)
    return len(rows), changed


def _collect_audio_paths_from_manifest(rows: list[dict], source_root: Path) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for row in rows:
        raw = str(row.get("audio_path", "") or "").strip()
        if not raw:
            continue
        p = _resolve_input_path(raw, source_root)
        key = _norm_key(str(p))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _main() -> int:
    hf_hard_limit = 10_000
    ap = argparse.ArgumentParser(description="Create/upload private HF eval-pack for cloud testing")
    ap.add_argument("--repo-id", required=True, help="HF dataset repo, e.g. user/private-eval-pack")
    ap.add_argument("--repo-root", default=r"I:\whisper-acft", help="Repo root for .env fallback")
    ap.add_argument("--source-root", default=r"I:\\", help="Root used to resolve relative audio paths")
    ap.add_argument("--pack-tag", default="stage13-indian-accent-en", help="Human tag for this pack")
    ap.add_argument("--manifest", required=True, help="Input test manifest JSONL")
    ap.add_argument("--extra-manifest", action="append", default=[], help="Additional manifest(s) to include and rewrite")
    ap.add_argument("--speaker-scores-csv", required=True, help="Input speaker_sort_scores.csv")
    ap.add_argument("--others-manifest", default="", help="Optional OTHER-manifest JSONL")
    ap.add_argument("--extra-folder", action="append", default=[], help="Optional extra folder(s) to include under audio/")
    ap.add_argument("--private", type=int, default=1, help="Create repo as private if it doesn't exist")
    ap.add_argument("--revision", default="main", help="Branch/revision for upload")
    ap.add_argument("--dry-run", action="store_true", help="Build staging pack only, no upload")
    ap.add_argument("--keep-staging", action="store_true", help="Do not delete staging folder")
    ap.add_argument("--path-in-repo-prefix", default="eval_packs", help="Repo folder prefix")
    ap.add_argument("--max-files-per-dir", type=int, default=9000, help="Shard any destination directory above this many files")
    ap.add_argument("--shard-prefix", default="shard", help="Shard directory name prefix (e.g., shard -> shard_0000)")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    source_root = Path(args.source_root).resolve()
    manifest_path = _resolve_input_path(args.manifest, repo_root)
    speaker_scores_path = _resolve_input_path(args.speaker_scores_csv, repo_root)
    others_manifest_path = _resolve_input_path(args.others_manifest, repo_root) if args.others_manifest else None
    extra_manifest_paths = [_resolve_input_path(p, repo_root) for p in (args.extra_manifest or []) if str(p).strip()]
    extra_folder_paths = [_resolve_input_path(p, repo_root) for p in (args.extra_folder or []) if str(p).strip()]
    shard_prefix = str(args.shard_prefix or "").strip()

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not speaker_scores_path.exists():
        raise SystemExit(f"Speaker scores CSV not found: {speaker_scores_path}")
    if others_manifest_path and not others_manifest_path.exists():
        raise SystemExit(f"Others manifest not found: {others_manifest_path}")
    for mp in extra_manifest_paths:
        if not mp.exists():
            raise SystemExit(f"Extra manifest not found: {mp}")
    for fp in extra_folder_paths:
        if not fp.exists() or not fp.is_dir():
            raise SystemExit(f"Extra folder not found or not a directory: {fp}")
    if int(args.max_files_per_dir) <= 0:
        raise SystemExit("--max-files-per-dir must be > 0")
    if int(args.max_files_per_dir) > hf_hard_limit:
        raise SystemExit(f"--max-files-per-dir must be <= {hf_hard_limit}")
    if not shard_prefix:
        raise SystemExit("--shard-prefix must be non-empty")
    if "/" in shard_prefix or "\\" in shard_prefix:
        raise SystemExit("--shard-prefix must not contain path separators")

    pack_id = f"{_utc_compact()}__{_safe_slug(args.pack_tag)}"
    staging_root = repo_root / ".hf_eval_pack_staging" / pack_id
    if staging_root.exists():
        shutil.rmtree(staging_root, ignore_errors=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    test_rows = _iter_jsonl(manifest_path)
    other_rows = _iter_jsonl(others_manifest_path) if others_manifest_path else []
    extra_rows_by_path: dict[Path, list[dict]] = {mp: _iter_jsonl(mp) for mp in extra_manifest_paths}
    audio_paths = _collect_audio_paths_from_manifest(test_rows, source_root)
    if other_rows:
        audio_paths.extend(_collect_audio_paths_from_manifest(other_rows, source_root))
    for extra_rows in extra_rows_by_path.values():
        audio_paths.extend(_collect_audio_paths_from_manifest(extra_rows, source_root))

    path_map, missing_audio, copied_count = _copy_audio_files(
        audio_paths=audio_paths,
        source_root=source_root,
        pack_root=staging_root,
    )
    extra_folder_stats: list[dict[str, Any]] = []
    for folder in extra_folder_paths:
        copied_folder, skipped_folder = _copy_entire_folder(
            folder=folder,
            source_root=source_root,
            pack_root=staging_root,
            target_root="audio",
        )
        extra_folder_stats.append(
            {
                "folder": str(folder),
                "copied_files": int(copied_folder),
                "skipped_existing_files": int(skipped_folder),
            }
        )
    sharding_report = _apply_audio_dir_sharding(
        staging_root=staging_root,
        audio_root=staging_root / "audio",
        path_map=path_map,
        max_files_per_dir=int(args.max_files_per_dir),
        shard_prefix=shard_prefix,
    )
    print(f"Audio files copied: {copied_count} | missing: {len(missing_audio)}")

    rewritten_test_rows, changed_test = _rewrite_manifest_rows(test_rows, path_map)
    test_manifest_out = staging_root / "manifests" / "pairs_manifest_stage13_test.jsonl"
    _write_jsonl(test_manifest_out, rewritten_test_rows)

    others_manifest_out = None
    changed_others = 0
    if other_rows:
        rewritten_other_rows, changed_others = _rewrite_manifest_rows(other_rows, path_map)
        others_manifest_out = staging_root / "manifests" / "others_manifest.jsonl"
        _write_jsonl(others_manifest_out, rewritten_other_rows)

    extra_manifest_outputs: list[dict[str, Any]] = []
    for extra_path, extra_rows in extra_rows_by_path.items():
        rewritten_extra_rows, changed_extra = _rewrite_manifest_rows(extra_rows, path_map)
        out_name = f"{extra_path.stem}.jsonl"
        out_path = staging_root / "manifests" / out_name
        _write_jsonl(out_path, rewritten_extra_rows)
        extra_manifest_outputs.append(
            {
                "source": str(extra_path),
                "output": f"manifests/{out_name}",
                "rows": int(len(extra_rows)),
                "paths_rewritten": int(changed_extra),
                "sha256": _sha256_file(out_path),
            }
        )

    speaker_scores_out = staging_root / "metadata" / "speaker_sort_scores.csv"
    speaker_rows, speaker_changed = _rewrite_speaker_scores(
        src_csv=speaker_scores_path,
        dst_csv=speaker_scores_out,
        path_map=path_map,
    )

    audio_file_counts = _collect_directory_file_counts(staging_root / "audio", base=staging_root, non_zero_only=True)
    hard_limit_offenders = _validate_directory_file_limit(
        staging_root,
        base=staging_root,
        max_files_per_dir=hf_hard_limit,
    )
    if hard_limit_offenders:
        top = "\n".join(
            f"  - {x['directory']}: {x['file_count']} files"
            for x in hard_limit_offenders[:10]
        )
        raise RuntimeError(
            "Pre-upload directory file-count validation failed. "
            f"Hub hard limit is {hf_hard_limit} files per directory.\n"
            f"Top offenders:\n{top}"
        )

    metadata = {
        "pack_id": pack_id,
        "pack_tag": args.pack_tag,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "manifest": str(manifest_path),
            "extra_manifests": [str(p) for p in extra_manifest_paths],
            "speaker_scores_csv": str(speaker_scores_path),
            "others_manifest": str(others_manifest_path) if others_manifest_path else None,
            "source_root": str(source_root),
            "extra_folders": [str(p) for p in extra_folder_paths],
        },
        "counts": {
            "test_rows": len(test_rows),
            "other_rows": len(other_rows),
            "extra_manifest_rows_total": int(sum(len(v) for v in extra_rows_by_path.values())),
            "audio_paths_total": len(audio_paths),
            "audio_files_copied": copied_count,
            "audio_files_missing": len(missing_audio),
            "test_manifest_paths_rewritten": changed_test,
            "others_manifest_paths_rewritten": changed_others,
            "speaker_rows": speaker_rows,
            "speaker_paths_rewritten": speaker_changed,
        },
        "files": {
            "test_manifest": "manifests/pairs_manifest_stage13_test.jsonl",
            "speaker_scores_csv": "metadata/speaker_sort_scores.csv",
            "others_manifest": "manifests/others_manifest.jsonl" if others_manifest_out else None,
            "extra_manifests": [x["output"] for x in extra_manifest_outputs],
            "others_dir": "audio",
        },
        "checksums": {
            "test_manifest_sha256": _sha256_file(test_manifest_out),
            "speaker_scores_sha256": _sha256_file(speaker_scores_out),
        },
        "extra_manifest_outputs": extra_manifest_outputs,
        "extra_folder_copy_stats": extra_folder_stats,
        "layout": {
            "hf_hard_limit_files_per_dir": int(hf_hard_limit),
            "max_files_per_dir_requested": int(args.max_files_per_dir),
            "shard_prefix": shard_prefix,
            "directories_sharded": sharding_report["directories_sharded"],
            "audio_directory_file_counts": audio_file_counts,
        },
    }
    if others_manifest_out is not None:
        metadata["checksums"]["others_manifest_sha256"] = _sha256_file(others_manifest_out)
    if missing_audio:
        metadata["missing_audio_examples"] = missing_audio[:200]

    metadata_path = staging_root / "PACK_METADATA.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print(f"[dry-run] Staging pack ready at: {staging_root}")
        return 0

    token = resolve_hf_token(repo_root, env_var="HF_TOKEN")
    if not token:
        env_values = parse_env_file(repo_root / ".env")
        token = env_values.get("HF_TOKEN", "").strip()
    if not token:
        raise SystemExit("HF token missing (set HF_TOKEN env var or add to .env)")
    os.environ["HF_TOKEN"] = token

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=bool(args.private),
        exist_ok=True,
        token=token,
    )

    path_in_repo = f"{args.path_in_repo_prefix.strip('/')}/{pack_id}"
    print(f"Uploading pack to dataset:{args.repo_id}/{path_in_repo}")
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(staging_root),
        path_in_repo=path_in_repo,
        token=token,
        revision=args.revision,
        commit_message=f"Add eval pack {pack_id}",
    )

    latest_ref = {
        "pack_id": pack_id,
        "pack_tag": args.pack_tag,
        "path_in_repo": path_in_repo,
        "repo_id": args.repo_id,
        "revision": args.revision,
        "updated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "commit": getattr(commit, "oid", None),
        "files": metadata["files"],
    }
    latest_tmp = staging_root / "LATEST_PACK.json"
    latest_tmp.write_text(json.dumps(latest_ref, indent=2), encoding="utf-8")
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=str(latest_tmp),
        path_in_repo=f"{args.path_in_repo_prefix.strip('/')}/LATEST_PACK.json",
        token=token,
        revision=args.revision,
        commit_message=f"Update latest eval pack -> {pack_id}",
    )

    print(f"Upload complete. Repo: https://huggingface.co/datasets/{args.repo_id}")
    print(f"Pack path: {path_in_repo}")
    print(f"Latest pointer: {args.path_in_repo_prefix.strip('/')}/LATEST_PACK.json")

    if not args.keep_staging:
        shutil.rmtree(staging_root, ignore_errors=True)
        print("Staging folder removed.")
    else:
        print(f"Staging folder kept: {staging_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
