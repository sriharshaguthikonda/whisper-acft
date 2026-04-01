#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import shutil
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
        raw = str(row.get("file", "") or "")
        mapped = path_map.get(_norm_key(raw))
        if mapped:
            row["file"] = mapped
            changed += 1

    with dst_csv.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
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
    ap = argparse.ArgumentParser(description="Create/upload private HF eval-pack for cloud testing")
    ap.add_argument("--repo-id", required=True, help="HF dataset repo, e.g. user/private-eval-pack")
    ap.add_argument("--repo-root", default=r"I:\whisper-acft", help="Repo root for .env fallback")
    ap.add_argument("--source-root", default=r"I:\\", help="Root used to resolve relative audio paths")
    ap.add_argument("--pack-tag", default="stage13-indian-accent-en", help="Human tag for this pack")
    ap.add_argument("--manifest", required=True, help="Input test manifest JSONL")
    ap.add_argument("--speaker-scores-csv", required=True, help="Input speaker_sort_scores.csv")
    ap.add_argument("--others-manifest", default="", help="Optional OTHER-manifest JSONL")
    ap.add_argument("--private", type=int, default=1, help="Create repo as private if it doesn't exist")
    ap.add_argument("--revision", default="main", help="Branch/revision for upload")
    ap.add_argument("--dry-run", action="store_true", help="Build staging pack only, no upload")
    ap.add_argument("--keep-staging", action="store_true", help="Do not delete staging folder")
    ap.add_argument("--path-in-repo-prefix", default="eval_packs", help="Repo folder prefix")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    source_root = Path(args.source_root).resolve()
    manifest_path = _resolve_input_path(args.manifest, repo_root)
    speaker_scores_path = _resolve_input_path(args.speaker_scores_csv, repo_root)
    others_manifest_path = _resolve_input_path(args.others_manifest, repo_root) if args.others_manifest else None

    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    if not speaker_scores_path.exists():
        raise SystemExit(f"Speaker scores CSV not found: {speaker_scores_path}")
    if others_manifest_path and not others_manifest_path.exists():
        raise SystemExit(f"Others manifest not found: {others_manifest_path}")

    pack_id = f"{_utc_compact()}__{_safe_slug(args.pack_tag)}"
    staging_root = repo_root / ".hf_eval_pack_staging" / pack_id
    if staging_root.exists():
        shutil.rmtree(staging_root, ignore_errors=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    test_rows = _iter_jsonl(manifest_path)
    other_rows = _iter_jsonl(others_manifest_path) if others_manifest_path else []
    audio_paths = _collect_audio_paths_from_manifest(test_rows, source_root)
    if other_rows:
        audio_paths.extend(_collect_audio_paths_from_manifest(other_rows, source_root))

    path_map, missing_audio, copied_count = _copy_audio_files(
        audio_paths=audio_paths,
        source_root=source_root,
        pack_root=staging_root,
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

    speaker_scores_out = staging_root / "metadata" / "speaker_sort_scores.csv"
    speaker_rows, speaker_changed = _rewrite_speaker_scores(
        src_csv=speaker_scores_path,
        dst_csv=speaker_scores_out,
        path_map=path_map,
    )

    metadata = {
        "pack_id": pack_id,
        "pack_tag": args.pack_tag,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "source": {
            "manifest": str(manifest_path),
            "speaker_scores_csv": str(speaker_scores_path),
            "others_manifest": str(others_manifest_path) if others_manifest_path else None,
            "source_root": str(source_root),
        },
        "counts": {
            "test_rows": len(test_rows),
            "other_rows": len(other_rows),
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
            "others_dir": "audio",
        },
        "checksums": {
            "test_manifest_sha256": _sha256_file(test_manifest_out),
            "speaker_scores_sha256": _sha256_file(speaker_scores_out),
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
