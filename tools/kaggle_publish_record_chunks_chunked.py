#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


KAGGLE_DATASET_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$")
KAGGLE_OWNER_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def canonical_path(path: str) -> str:
    return os.path.normpath(path).replace("\\", "/").lower()


def to_native_path(path: Path | str) -> str:
    s = os.path.abspath(str(path))
    if os.name != "nt":
        return s
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def slug_to_title(slug: str) -> str:
    parts = [p for p in slug.replace("_", "-").split("-") if p]
    titled = " ".join(p.upper() if p.isdigit() else p.capitalize() for p in parts)
    if not titled:
        return "Dataset"
    return titled


def normalize_subtitle(subtitle: str) -> str:
    s = " ".join(str(subtitle).split()).strip()
    if not s:
        s = "Dataset package for Kaggle upload"
    if len(s) < 20:
        s = f"{s} dataset package"
    if len(s) < 20:
        s = "Dataset package for Kaggle upload"
    if len(s) > 80:
        s = s[:77].rstrip() + "..."
    return s


def audio_chunk_title(chunk_index: int) -> str:
    return f"ACFT Moonshine Record Chunks Audio Chunk {int(chunk_index):03d}"


def audio_chunk_subtitle(kaggle_top_dir: str, chunk_index: int, chunk_file_limit: int) -> str:
    return f"{kaggle_top_dir} audio chunk {int(chunk_index):03d}; max {int(chunk_file_limit)} files."


def manifest_title() -> str:
    return "ACFT Moonshine Record Chunks Manifests Chunked"


def manifest_subtitle() -> str:
    return "Kaggle-ready manifests and path index for chunked Record_chunks datasets."


def validate_kaggle_dataset_id(dataset_id: str) -> tuple[str, str]:
    if dataset_id.count("/") != 1:
        raise ValueError(f"Dataset id must be owner/slug: {dataset_id}")
    owner, slug = dataset_id.split("/", 1)
    if not owner or not slug:
        raise ValueError(f"Dataset id must be owner/slug: {dataset_id}")
    if not KAGGLE_OWNER_RE.match(owner):
        raise ValueError(f"Kaggle owner contains unsafe characters: {owner}")
    if len(slug) < 6 or len(slug) > 50:
        raise ValueError(f"The dataset slug must be between 6 and 50 characters: {slug} ({len(slug)})")
    if not KAGGLE_DATASET_SLUG_RE.match(slug):
        raise ValueError(
            "The dataset slug must contain only lowercase letters, numbers, and hyphens, "
            f"and must start/end with a letter or number: {slug}"
        )
    return owner, slug


def validate_kaggle_text_lengths(title: str, subtitle: str | None = None) -> None:
    clean_title = " ".join(str(title).split()).strip()
    if len(clean_title) < 6 or len(clean_title) > 50:
        raise ValueError(f"The dataset title must be between 6 and 50 characters: {clean_title!r} ({len(clean_title)})")
    if subtitle:
        clean_subtitle = " ".join(str(subtitle).split()).strip()
        if len(clean_subtitle) < 20 or len(clean_subtitle) > 80:
            raise ValueError(
                f"The dataset subtitle must be between 20 and 80 characters: "
                f"{clean_subtitle!r} ({len(clean_subtitle)})"
            )


def build_name_preflight(
    chunk_details: list[dict],
    *,
    manifest_dataset_id: str,
    kaggle_top_dir: str,
    chunk_file_limit: int,
) -> list[dict]:
    rows: list[dict] = []
    for c in chunk_details:
        dataset_id = str(c["dataset_id"])
        _, slug = validate_kaggle_dataset_id(dataset_id)
        title = audio_chunk_title(int(c["chunk_index"]))
        subtitle = normalize_subtitle(audio_chunk_subtitle(kaggle_top_dir, int(c["chunk_index"]), chunk_file_limit))
        validate_kaggle_text_lengths(title, subtitle)
        rows.append(
            {
                "dataset_id": dataset_id,
                "slug": slug,
                "slug_len": len(slug),
                "title": title,
                "title_len": len(title),
                "subtitle_len": len(subtitle),
                "status": "ok",
            }
        )

    _, manifest_slug = validate_kaggle_dataset_id(manifest_dataset_id)
    title = manifest_title()
    subtitle = normalize_subtitle(manifest_subtitle())
    validate_kaggle_text_lengths(title, subtitle)
    rows.append(
        {
            "dataset_id": manifest_dataset_id,
            "slug": manifest_slug,
            "slug_len": len(manifest_slug),
            "title": title,
            "title_len": len(title),
            "subtitle_len": len(subtitle),
            "status": "ok",
        }
    )
    return rows


def build_assignment_signature(entries: list[dict]) -> dict[str, Any]:
    h = hashlib.sha256()
    for row in sorted(entries, key=lambda r: (int(r["chunk_index"]), str(r["rel"]))):
        line = f"{row['chunk_index']}\t{row['rel']}\t{row['size']}\n"
        h.update(line.encode("utf-8", errors="replace"))
    return {
        "sha256": h.hexdigest(),
        "entries": int(len(entries)),
        "total_bytes": int(sum(int(r["size"]) for r in entries)),
    }


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stream: bool = True,
) -> tuple[int, str]:
    if stream:
        print("$ " + " ".join([f'"{c}"' if " " in c else c for c in cmd]))
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line)
        if stream:
            print(line, end="")
    rc = proc.wait()
    output = "".join(lines)
    return rc, output


def run_kaggle_dataset_cmd(cmd: list[str], *, env: dict[str, str]) -> tuple[int, str]:
    rc, out = run_cmd(cmd, env=env, stream=True)
    out_l = out.lower()
    needs_fallback = (
        "unrecognized arguments: --dir-mode" in out_l
        or "unrecognized arguments: --keep-tabular" in out_l
        or "no such option: --dir-mode" in out_l
        or "no such option: --keep-tabular" in out_l
    )
    if not needs_fallback:
        return rc, out

    fallback: list[str] = []
    i = 0
    while i < len(cmd):
        token = cmd[i]
        if token == "--keep-tabular":
            i += 1
            continue
        if token == "--dir-mode":
            mode = "tar"
            if i + 1 < len(cmd):
                mode = cmd[i + 1]
            fallback.extend(["-r", mode])
            i += 2
            continue
        fallback.append(token)
        i += 1

    print("[warn] Kaggle CLI flag compatibility fallback: retrying command with legacy flags.")
    return run_cmd(fallback, env=env, stream=True)


def enumerate_source_files(source_root: Path) -> list[dict]:
    files: list[dict] = []
    for p in source_root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(source_root).as_posix()
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            files.append({"src": str(p.resolve()), "rel": rel, "size": int(size)})
    return files


def assign_chunks(entries: list[dict], chunk_file_limit: int, chunk_bytes_limit: int = 0) -> tuple[list[dict], dict]:
    if chunk_file_limit <= 0:
        raise ValueError("chunk_file_limit must be > 0")
    if not entries:
        return [], {
            "chunks_total": 0,
            "files_total": 0,
            "bytes_total": 0,
            "chunk_file_limit": chunk_file_limit,
            "chunk_bytes_limit": int(chunk_bytes_limit),
            "chunks": [],
        }

    if chunk_bytes_limit and chunk_bytes_limit > 0:
        ordered = sorted(entries, key=lambda r: (-int(r["size"]), str(r["rel"])))
        chunk_bytes: list[int] = []
        chunk_files: list[int] = []
        out: list[dict] = []

        for row in ordered:
            size = int(row["size"])
            if size > chunk_bytes_limit:
                raise ValueError(
                    f"File exceeds chunk_bytes_limit: {row.get('rel')} size={size} limit={chunk_bytes_limit}"
                )

            target = -1
            target_bytes = sys.maxsize
            for idx, current_bytes in enumerate(chunk_bytes):
                if chunk_files[idx] >= chunk_file_limit:
                    continue
                if current_bytes + size > chunk_bytes_limit:
                    continue
                if current_bytes < target_bytes:
                    target = idx
                    target_bytes = current_bytes

            if target < 0:
                chunk_bytes.append(0)
                chunk_files.append(0)
                target = len(chunk_bytes) - 1

            chunk_files[target] += 1
            chunk_bytes[target] += size
            out.append({**row, "chunk_index": target + 1})

        out.sort(key=lambda r: (int(r["chunk_index"]), str(r["rel"])))
        chunks = [
            {
                "chunk_index": i + 1,
                "files": int(chunk_files[i]),
                "bytes": int(chunk_bytes[i]),
                "gb": round(chunk_bytes[i] / (1024**3), 3),
            }
            for i in range(len(chunk_bytes))
        ]
        return out, {
            "chunk_file_limit": int(chunk_file_limit),
            "chunk_bytes_limit": int(chunk_bytes_limit),
            "chunks_total": int(len(chunk_bytes)),
            "files_total": int(sum(chunk_files)),
            "bytes_total": int(sum(chunk_bytes)),
            "total_gb": round(sum(chunk_bytes) / (1024**3), 3),
            "chunks": chunks,
        }

    # Largest-first greedy packing across fixed-count bins keeps chunk sizes closer.
    ordered = sorted(entries, key=lambda r: (-int(r["size"]), str(r["rel"])))
    chunk_count = int(math.ceil(len(ordered) / chunk_file_limit))
    chunk_bytes = [0] * chunk_count
    chunk_files = [0] * chunk_count

    out: list[dict] = []
    for row in ordered:
        target = -1
        target_bytes = sys.maxsize
        for idx in range(chunk_count):
            if chunk_files[idx] >= chunk_file_limit:
                continue
            b = chunk_bytes[idx]
            if b < target_bytes:
                target = idx
                target_bytes = b
        if target < 0:
            raise RuntimeError("Failed to assign chunk (all chunks full unexpectedly).")
        chunk_files[target] += 1
        chunk_bytes[target] += int(row["size"])
        out.append({**row, "chunk_index": target + 1})

    out.sort(key=lambda r: (int(r["chunk_index"]), str(r["rel"])))

    chunks: list[dict] = []
    for i in range(chunk_count):
        chunks.append(
            {
                "chunk_index": i + 1,
                "files": int(chunk_files[i]),
                "bytes": int(chunk_bytes[i]),
                "gb": round(chunk_bytes[i] / (1024**3), 3),
            }
        )
    summary = {
        "chunk_file_limit": int(chunk_file_limit),
        "chunk_bytes_limit": int(chunk_bytes_limit),
        "chunks_total": int(chunk_count),
        "files_total": int(sum(chunk_files)),
        "bytes_total": int(sum(chunk_bytes)),
        "total_gb": round(sum(chunk_bytes) / (1024**3), 3),
        "chunks": chunks,
    }
    return out, summary


def copy_if_needed(src: Path, dst: Path, max_attempts: int = 6) -> tuple[bool, str]:
    src_s = to_native_path(src)
    dst_s = to_native_path(dst)
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            if os.path.exists(dst_s):
                try:
                    if os.path.getsize(dst_s) == os.path.getsize(src_s):
                        return True, "skipped"
                except Exception:
                    pass
                try:
                    os.remove(dst_s)
                except Exception:
                    pass

            os.makedirs(os.path.dirname(dst_s), exist_ok=True)
            shutil.copy2(src_s, dst_s)
            return True, "copied"
        except (PermissionError, FileNotFoundError) as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(0.08 * attempt)
                continue
            break
        except Exception as exc:
            last_exc = exc
            break

    assert last_exc is not None
    return False, f"{type(last_exc).__name__}: {last_exc}"


def materialize_chunk_roots(
    entries: list[dict],
    stage_root: Path,
    *,
    top_dir_name: str,
    workers: int,
) -> dict:
    copied = 0
    skipped = 0
    failed = 0
    failures: list[dict] = []
    total = len(entries)
    pending = set()
    idx = 0
    progress_last = time.time()

    def submit_one(ex: ThreadPoolExecutor, item: dict):
        src = Path(item["src"])
        dst = stage_root / str(item["chunk_slug"]) / top_dir_name / Path(str(item["rel"]))
        fut = ex.submit(copy_if_needed, src, dst)
        fut._kaggle_item = item  # type: ignore[attr-defined]
        return fut

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        while idx < total and len(pending) < max(1, int(workers) * 4):
            pending.add(submit_one(ex, entries[idx]))
            idx += 1

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                item = getattr(fut, "_kaggle_item", None)
                ok, status = fut.result()
                if ok:
                    if status == "copied":
                        copied += 1
                    else:
                        skipped += 1
                else:
                    failed += 1
                    if item:
                        failures.append(
                            {
                                "src": item.get("src"),
                                "rel": item.get("rel"),
                                "chunk_index": item.get("chunk_index"),
                                "chunk_slug": item.get("chunk_slug"),
                                "error": status,
                            }
                        )

                if idx < total:
                    pending.add(submit_one(ex, entries[idx]))
                    idx += 1

            now = time.time()
            if now - progress_last >= 5.0:
                done_count = copied + skipped + failed
                print(f"[copy] done={done_count}/{total} copied={copied} skipped={skipped} failed={failed}")
                progress_last = now

    return {
        "copied": copied,
        "skipped": skipped,
        "failed": failed,
        "failures": failures,
    }


def rewrite_manifest(
    source_manifest: Path,
    out_manifest: Path,
    path_map: dict[str, dict],
) -> dict:
    ensure_dir(out_manifest.parent)
    rewritten = 0
    missing = 0
    rows_total = 0
    out_rows: list[str] = []

    with source_manifest.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows_total += 1
            try:
                obj = json.loads(line)
            except Exception:
                out_rows.append(line)
                continue
            if not isinstance(obj, dict):
                out_rows.append(line)
                continue
            ap = str(obj.get("audio_path") or "").strip()
            if not ap:
                out_rows.append(json.dumps(obj, ensure_ascii=False))
                continue
            key = canonical_path(ap)
            mapped = path_map.get(key)
            if not mapped:
                try:
                    mapped = path_map.get(canonical_path(str(Path(ap).resolve())))
                except Exception:
                    mapped = None
            if not mapped:
                missing += 1
                out_rows.append(json.dumps(obj, ensure_ascii=False))
                continue
            obj["audio_path_original"] = ap
            obj["audio_path"] = str(mapped["kaggle_path"])
            rewritten += 1
            out_rows.append(json.dumps(obj, ensure_ascii=False))

    with out_manifest.open("w", encoding="utf-8") as w:
        for row in out_rows:
            w.write(row + "\n")

    return {
        "source_manifest": str(source_manifest),
        "out_manifest": str(out_manifest),
        "rows_total": int(rows_total),
        "rewritten": int(rewritten),
        "missing": int(missing),
    }


def parse_manifest_specs(args: argparse.Namespace) -> list[tuple[Path, str]]:
    specs: list[tuple[Path, str]] = []
    for raw in args.manifest or []:
        if "=" not in raw:
            raise SystemExit(f"Invalid --manifest spec, expected output_name=path: {raw}")
        out_name, src = raw.split("=", 1)
        out_name = out_name.strip()
        src = src.strip()
        if not out_name or not src:
            raise SystemExit(f"Invalid --manifest spec, expected output_name=path: {raw}")
        if any(sep in out_name for sep in ("/", "\\")) or not out_name.endswith(".jsonl"):
            raise SystemExit(f"Manifest output name must be a JSONL file name only: {out_name}")
        specs.append((Path(src), out_name))
    if specs:
        return specs

    train_manifest = Path(args.train_manifest)
    test_manifest = Path(args.test_manifest)
    if not train_manifest.exists() or not test_manifest.exists():
        raise SystemExit("Train/test manifests are required and must exist unless --manifest is supplied.")
    return [
        (train_manifest, "pairs_manifest_stage15_train_no_targets_randomized_kaggle.jsonl"),
        (test_manifest, "pairs_manifest_stage13_test_randomized_kaggle.jsonl"),
    ]


def build_kaggle_path_record(
    row: dict,
    *,
    kaggle_top_dir: str,
    audio_path_root_template: str = "/kaggle/working/acft_chunks/{top_dir}",
) -> dict:
    slug = str(row["chunk_slug"])
    rel = str(row["rel"]).replace("\\", "/")
    dataset_id = str(row["dataset_id"])
    archive_name = f"{kaggle_top_dir}.tar"
    try:
        audio_root = audio_path_root_template.format(
            chunk_slug=slug,
            top_dir=kaggle_top_dir,
            dataset_id=dataset_id,
        )
    except KeyError as exc:
        raise ValueError(
            "--audio_path_root_template may only use {chunk_slug}, {top_dir}, and {dataset_id}"
        ) from exc
    audio_root = audio_root.rstrip("/")

    return {
        "source_abs": str(Path(str(row["src"])).resolve()),
        "rel": rel,
        "chunk_index": int(row["chunk_index"]),
        "chunk_slug": slug,
        "dataset_id": dataset_id,
        "size": int(row["size"]),
        "kaggle_path": f"{audio_root}/{rel}",
        "kaggle_uploaded_archive_name": archive_name,
        "kaggle_input_extracted_path": f"/kaggle/input/{slug}/{rel}",
        "archive_member": rel,
        "desired_relative_path": f"{kaggle_top_dir}/{rel}",
    }


def ensure_dataset_metadata(folder: Path, dataset_id: str, title: str, subtitle: str, license_name: str) -> Path:
    md_path = folder / "dataset-metadata.json"
    validate_kaggle_dataset_id(dataset_id)
    clean_title = str(title).strip()
    if not clean_title:
        slug_part = dataset_id.split("/", 1)[-1]
        clean_title = slug_to_title(slug_part)
    clean_subtitle = normalize_subtitle(subtitle)
    validate_kaggle_text_lengths(clean_title, clean_subtitle)
    payload = {
        "title": clean_title,
        "id": dataset_id,
        "licenses": [{"name": license_name}],
        "subtitle": clean_subtitle,
        "description": str(subtitle).strip() or clean_subtitle,
    }
    write_json(md_path, payload)
    return md_path


def dataset_exists(kaggle_exe: str, dataset_id: str, env: dict[str, str], probe_dir: Path) -> bool:
    run_probe = probe_dir / f"probe_{int(time.time() * 1000)}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    ensure_dir(run_probe)
    rc, _ = run_cmd([kaggle_exe, "datasets", "metadata", dataset_id, "-p", str(run_probe)], env=env, stream=False)
    return rc == 0


def wait_for_dataset_exists(
    kaggle_exe: str,
    dataset_id: str,
    env: dict[str, str],
    probe_dir: Path,
    *,
    max_attempts: int = 12,
    sleep_seconds: float = 5.0,
) -> bool:
    for attempt in range(1, max_attempts + 1):
        if dataset_exists(kaggle_exe, dataset_id, env, probe_dir / f"attempt_{attempt:02d}"):
            return True
        if attempt < max_attempts:
            time.sleep(sleep_seconds)
    return False


def classify_upload_failure(out_l: str) -> str:
    if "too many requests" in out_l or "429 client error" in out_l:
        return "rate_limit"
    if "not found" in out_l or "404 client error" in out_l:
        return "not_found"
    if "already exists" in out_l or "cannot create a dataset with this slug" in out_l:
        return "already_exists"
    if "dataset creation error" in out_l or "dataset versioning error" in out_l or "slug can only contain" in out_l:
        return "hard_error"
    return "unknown"


def create_dataset_only(
    kaggle_exe: str,
    dataset_id: str,
    seed_folder: Path,
    env: dict[str, str],
    probe_dir: Path,
    *,
    upload_retry_max: int,
    upload_retry_backoff_seconds: float,
) -> str:
    if dataset_exists(kaggle_exe, dataset_id, env, probe_dir / ("exists_" + dataset_id.replace("/", "__"))):
        return "exists"

    cmd = [kaggle_exe, "datasets", "create", "-p", str(seed_folder), "--dir-mode", "tar", "--keep-tabular"]
    retries = max(1, int(upload_retry_max))
    for attempt in range(1, retries + 1):
        rc, out = run_kaggle_dataset_cmd(cmd, env=env)
        out_l = out.lower()
        reason = classify_upload_failure(out_l)

        if rc == 0 and reason not in {"hard_error", "rate_limit"}:
            break
        if reason == "already_exists":
            return "exists"
        if reason == "rate_limit" and attempt < retries:
            delay = min(float(upload_retry_backoff_seconds) * (2 ** (attempt - 1)), 900.0)
            print(f"[warn] create rate-limited for {dataset_id}; retrying in {delay:.0f}s (attempt {attempt}/{retries}).")
            time.sleep(delay)
            continue
        raise RuntimeError(f"Create failed for {dataset_id}. Last output: {out[:400]}")

    visible = wait_for_dataset_exists(
        kaggle_exe,
        dataset_id,
        env,
        probe_dir / ("postcreate_" + dataset_id.replace("/", "__")),
        max_attempts=60,
        sleep_seconds=5.0,
    )
    if not visible:
        print(f"[warn] create accepted for {dataset_id}, but metadata is not visible yet.")
        return "created_pending"
    return "created"


def version_dataset_only(
    kaggle_exe: str,
    dataset_id: str,
    folder: Path,
    message: str,
    env: dict[str, str],
    probe_dir: Path,
    *,
    upload_retry_max: int,
    upload_retry_backoff_seconds: float,
) -> str:
    retries = max(1, int(upload_retry_max))
    cmd = [kaggle_exe, "datasets", "version", "-p", str(folder), "-m", message, "--dir-mode", "tar", "--keep-tabular"]
    for attempt in range(1, retries + 1):
        rc, out = run_kaggle_dataset_cmd(cmd, env=env)
        out_l = out.lower()
        reason = classify_upload_failure(out_l)

        if rc == 0 and reason not in {"hard_error", "rate_limit", "not_found"}:
            break
        if reason in {"rate_limit", "not_found"} and attempt < retries:
            delay = min(float(upload_retry_backoff_seconds) * (2 ** (attempt - 1)), 900.0)
            print(f"[warn] version blocked for {dataset_id} ({reason}); retrying in {delay:.0f}s (attempt {attempt}/{retries}).")
            time.sleep(delay)
            continue
        raise RuntimeError(f"Version failed for {dataset_id}. Last output: {out[:400]}")

    if not wait_for_dataset_exists(
        kaggle_exe,
        dataset_id,
        env,
        probe_dir / ("postversion_" + dataset_id.replace("/", "__")),
        max_attempts=30,
        sleep_seconds=5.0,
    ):
        raise RuntimeError(f"Version uploaded but dataset not visible: {dataset_id}")
    return "version"


def upload_dataset(
    kaggle_exe: str,
    dataset_id: str,
    folder: Path,
    message: str,
    env: dict[str, str],
    probe_dir: Path,
    *,
    upload_retry_max: int,
    upload_retry_backoff_seconds: float,
) -> tuple[str, bool]:
    exists = dataset_exists(kaggle_exe, dataset_id, env, probe_dir / dataset_id.replace("/", "__"))
    mode = "version" if exists else "create"

    def cmd_for_mode(which: str) -> list[str]:
        if which == "version":
            return [kaggle_exe, "datasets", "version", "-p", str(folder), "-m", message, "--dir-mode", "tar", "--keep-tabular"]
        return [kaggle_exe, "datasets", "create", "-p", str(folder), "--dir-mode", "tar", "--keep-tabular"]

    for attempt in range(1, max(1, int(upload_retry_max)) + 1):
        rc, out = run_kaggle_dataset_cmd(cmd_for_mode(mode), env=env)
        out_l = out.lower()

        if mode == "create" and ("already exists" in out_l or "cannot create a dataset with this slug" in out_l) and rc != 0:
            print(f"[warn] create reported existing slug for {dataset_id}; falling back to version.")
            mode = "version"
            continue

        rate_limited = "too many requests" in out_l or "429 client error" in out_l
        if rate_limited:
            maybe_visible = wait_for_dataset_exists(
                kaggle_exe,
                dataset_id,
                env,
                probe_dir / ("rate_limit_probe_" + dataset_id.replace("/", "__")),
                max_attempts=6,
                sleep_seconds=5.0,
            )
            if maybe_visible and mode == "create":
                print(f"[warn] 429 received but {dataset_id} became visible; treating create as accepted.")
                return "create", True

            if attempt < max(1, int(upload_retry_max)):
                delay = float(upload_retry_backoff_seconds) * (2 ** (attempt - 1))
                delay = min(delay, 900.0)
                print(f"[warn] Kaggle rate limit for {dataset_id} ({mode}); retrying in {delay:.0f}s (attempt {attempt}/{upload_retry_max}).")
                time.sleep(delay)
                if dataset_exists(kaggle_exe, dataset_id, env, probe_dir / ("recheck_" + dataset_id.replace("/", "__"))):
                    mode = "version"
                continue
            raise RuntimeError(f"Upload failed with repeated rate limits for {dataset_id} ({mode})")

        if rc != 0 or "dataset creation error" in out_l or "dataset versioning error" in out_l or "slug can only contain" in out_l:
            raise RuntimeError(f"Upload failed for {dataset_id} ({mode})")
        break
    if mode == "create":
        visible = wait_for_dataset_exists(
            kaggle_exe,
            dataset_id,
            env,
            probe_dir / ("postcheck_" + dataset_id.replace("/", "__")),
            max_attempts=12,
            sleep_seconds=5.0,
        )
        if not visible:
            print(f"[warn] create accepted for {dataset_id}, but metadata not visible yet; continuing.")
        return mode, visible

    if not wait_for_dataset_exists(
        kaggle_exe,
        dataset_id,
        env,
        probe_dir / ("postcheck_" + dataset_id.replace("/", "__")),
        max_attempts=30,
        sleep_seconds=5.0,
    ):
        raise RuntimeError(f"Upload did not materialize dataset {dataset_id} ({mode})")
    return mode, True


def dataset_files_listing_ready(out: str, *, expected_fragment: str | None = None) -> bool:
    if "No files found" in out:
        return False
    if expected_fragment and expected_fragment not in out:
        return False
    return True


def verify_dataset_files(
    kaggle_exe: str,
    dataset_id: str,
    env: dict[str, str],
    *,
    expected_fragment: str | None = None,
) -> None:
    last_out = ""
    for attempt in range(1, 31):
        rc, out = run_cmd([kaggle_exe, "datasets", "files", dataset_id, "--page-size", "200"], env=env, stream=(attempt == 1))
        last_out = out
        if rc == 0 and dataset_files_listing_ready(out, expected_fragment=expected_fragment):
            if attempt > 1:
                print(f"[verify] {dataset_id} visible after attempt {attempt}")
            return
        if attempt < 30:
            time.sleep(5.0)
    raise RuntimeError(f"Verification failed for {dataset_id}. Last output: {last_out[:400]}")


def build_env_with_token() -> dict[str, str]:
    env = dict(os.environ)
    if env.get("KAGGLE_API_TOKEN"):
        return env
    if env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"):
        return env

    user_tok = ""
    username = ""
    api_key = ""

    if not user_tok:
        try:
            import winreg  # type: ignore

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                user_tok = winreg.QueryValueEx(k, "KAGGLE_API_TOKEN")[0]
        except Exception:
            user_tok = ""

    cfg_dir = env.get("KAGGLE_CONFIG_DIR") or os.path.join(str(Path.home()), ".kaggle")
    cfg_path = Path(cfg_dir) / "kaggle.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            username = str(cfg.get("username") or "").strip()
            api_key = str(cfg.get("key") or "").strip()
            if not user_tok:
                user_tok = str(cfg.get("apiToken") or cfg.get("api_token") or "").strip()
        except Exception:
            username = ""
            api_key = ""

    if user_tok:
        env["KAGGLE_API_TOKEN"] = user_tok
    if username and api_key:
        env["KAGGLE_USERNAME"] = username
        env["KAGGLE_KEY"] = api_key
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish Record_chunks to Kaggle private chunked datasets")
    ap.add_argument("--source_root", default=r"I:\Record_chunks")
    ap.add_argument("--stage_root", default=r"J:\kaggle_publish\acft-moonshine-record-chunks-chunked-publish")
    ap.add_argument("--train_manifest", default=r"I:\Record_chunks\pairs_manifest_stage15_train_no_targets_randomized.jsonl")
    ap.add_argument("--test_manifest", default=r"I:\Record_chunks\pairs_manifest_stage13_test_randomized.jsonl")
    ap.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Rewrite explicit manifest as output_name.jsonl=path; repeatable. Replaces default train/test rewriting when supplied.",
    )
    ap.add_argument("--kaggle_top_dir", default="", help="Top-level folder name inside each Kaggle dataset; defaults to source root name")
    ap.add_argument(
        "--audio_path_root_template",
        default="/kaggle/working/acft_chunks/{top_dir}",
        help=(
            "Template for rewritten manifest audio_path roots. Available fields: {chunk_slug}, {top_dir}, {dataset_id}. "
            "Default points to a Kaggle working directory where notebooks reconstruct the original top-level folder."
        ),
    )
    ap.add_argument("--kaggle_exe", default=r"C:\Users\deletable\AppData\Roaming\Python\Python311\Scripts\kaggle.exe")
    ap.add_argument("--owner", default="drsriharshaguthik")
    ap.add_argument("--audio_slug_prefix", default="acft-moonshine-record-chunks-audio-chunk")
    ap.add_argument("--manifest_dataset", default="drsriharshaguthik/acft-moonshine-record-chunks-manifests-chunked")
    ap.add_argument("--chunk_file_limit", type=int, default=5000)
    ap.add_argument(
        "--chunk_bytes_limit_mb",
        type=float,
        default=0.0,
        help="Optional max bytes per staged audio chunk, in MiB. 0 keeps file-count-only chunking.",
    )
    ap.add_argument("--license_name", default="CC0-1.0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max_chunks_upload", type=int, default=0, help="0 means upload all chunks")
    ap.add_argument("--start_chunk_index", type=int, default=1, help="1-based chunk index to start uploads from")
    ap.add_argument("--end_chunk_index", type=int, default=0, help="1-based inclusive chunk index; 0 means last chunk")
    ap.add_argument("--upload_retry_max", type=int, default=8, help="Retries per chunk when Kaggle returns 429")
    ap.add_argument(
        "--upload_retry_backoff_seconds",
        type=float,
        default=90.0,
        help="Base backoff seconds for 429 (90, 180, 360, ... capped)",
    )
    ap.add_argument(
        "--sleep_between_uploads_seconds",
        type=float,
        default=15.0,
        help="Delay between chunk uploads to reduce Kaggle create rate pressure",
    )
    ap.add_argument(
        "--publish_phase",
        choices=["all", "precreate", "upload"],
        default="all",
        help="all=legacy create/version per dataset, precreate=create empty seeds only, upload=version selected datasets only",
    )
    ap.add_argument("--skip_copy", action="store_true", help="Skip copy/materialization and use existing stage folders")
    ap.add_argument("--skip_upload", action="store_true")
    ap.add_argument("--only_preflight", action="store_true")
    args = ap.parse_args()

    source_root = Path(args.source_root)
    if not source_root.exists():
        raise SystemExit(f"Missing source root: {source_root}")
    manifest_specs = parse_manifest_specs(args)
    for manifest_path, _ in manifest_specs:
        if not manifest_path.exists():
            raise SystemExit(f"Manifest is required and must exist: {manifest_path}")
    kaggle_top_dir = (args.kaggle_top_dir or source_root.name).strip().strip("/\\")
    if not kaggle_top_dir or any(sep in kaggle_top_dir for sep in ("/", "\\")):
        raise SystemExit(f"Invalid --kaggle_top_dir: {args.kaggle_top_dir!r}")

    stage_root = Path(args.stage_root)
    ensure_dir(stage_root)
    probe_root = stage_root / "_probe"
    ensure_dir(probe_root)

    env = build_env_with_token()
    rc, out = run_cmd([args.kaggle_exe, "config", "view"], env=env, stream=True)
    if rc != 0:
        raise SystemExit(f"Kaggle auth preflight failed. Output: {out[:400]}")
    if args.owner.lower() not in out.lower():
        raise SystemExit(
            f"Authenticated Kaggle username mismatch. Expected to include: {args.owner}. config view output: {out[:400]}"
        )

    print("[scan] enumerating source files...")
    files = enumerate_source_files(source_root)
    if not files:
        raise SystemExit("No source files found.")
    source_total = sum(int(r["size"]) for r in files)
    print(f"[scan] files={len(files)} total_gb={source_total / (1024**3):.3f}")

    chunk_bytes_limit = int(float(args.chunk_bytes_limit_mb) * 1024 * 1024) if float(args.chunk_bytes_limit_mb) > 0 else 0
    chunk_entries, chunk_summary = assign_chunks(files, args.chunk_file_limit, chunk_bytes_limit)
    chunk_count = int(chunk_summary["chunks_total"])
    width = max(3, len(str(chunk_count)))
    chunk_slug_map = {i: f"{args.audio_slug_prefix}-{i:0{width}d}" for i in range(1, chunk_count + 1)}

    for row in chunk_entries:
        idx = int(row["chunk_index"])
        slug = chunk_slug_map[idx]
        row["chunk_slug"] = slug
        row["dataset_id"] = f"{args.owner}/{slug}"

    chunk_details: list[dict] = []
    for c in chunk_summary["chunks"]:
        idx = int(c["chunk_index"])
        slug = chunk_slug_map[idx]
        chunk_details.append(
            {
                **c,
                "chunk_slug": slug,
                "dataset_id": f"{args.owner}/{slug}",
            }
        )
    chunk_summary["chunks"] = chunk_details
    chunk_listing_fragments: dict[int, str] = {}
    for row in chunk_entries:
        idx = int(row["chunk_index"])
        if idx not in chunk_listing_fragments:
            chunk_listing_fragments[idx] = Path(str(row["rel"]).replace("\\", "/")).name
    manifest_dataset_id = args.manifest_dataset
    name_preflight = build_name_preflight(
        chunk_details,
        manifest_dataset_id=manifest_dataset_id,
        kaggle_top_dir=kaggle_top_dir,
        chunk_file_limit=int(args.chunk_file_limit),
    )
    write_json(
        stage_root / "kaggle_name_preflight.json",
        {
            "generated_utc": now_utc(),
            "datasets": name_preflight,
        },
    )
    print(f"[preflight] Kaggle dataset names validated: {len(name_preflight)} datasets")
    current_signature = build_assignment_signature(chunk_entries)
    signature_path = stage_root / "chunk_plan_signature.json"
    previous_signature: dict[str, Any] | None = None
    if signature_path.exists():
        try:
            previous_signature = json.loads(signature_path.read_text(encoding="utf-8"))
        except Exception:
            previous_signature = None

    if args.skip_copy:
        if not previous_signature:
            raise SystemExit(
                f"--skip_copy requested but no prior {signature_path.name} found. "
                "Run once without --skip_copy to materialize staged chunk folders."
            )
        if previous_signature.get("sha256") != current_signature.get("sha256"):
            raise SystemExit(
                "--skip_copy requested but current chunk plan differs from staged plan. "
                f"staged={previous_signature.get('sha256')} current={current_signature.get('sha256')}"
            )

    write_json(signature_path, {"generated_utc": now_utc(), **current_signature})
    write_json(stage_root / "chunk_summary.json", chunk_summary)
    write_jsonl(stage_root / "chunk_assignment.jsonl", chunk_entries)
    print(f"[chunk] chunks={chunk_count} limit={args.chunk_file_limit}")

    if args.only_preflight:
        print("[done] preflight only.")
        return

    if args.publish_phase == "precreate":
        copy_stats = {"copied": 0, "skipped": len(chunk_entries), "failed": 0, "failures": []}
        print("[copy] skipped in precreate phase")
    elif args.skip_copy:
        for slug in chunk_slug_map.values():
            p = stage_root / slug / kaggle_top_dir
            if not p.exists():
                raise SystemExit(f"--skip_copy requested but missing staged folder: {p}")
        copy_stats = {"copied": 0, "skipped": len(chunk_entries), "failed": 0, "failures": []}
        print("[copy] skipped by --skip_copy")
    else:
        print("[copy] materializing chunk trees...")
        copy_stats = materialize_chunk_roots(
            chunk_entries,
            stage_root=stage_root,
            top_dir_name=kaggle_top_dir,
            workers=int(args.workers),
        )
        print(f"[copy] stats copied={copy_stats['copied']} skipped={copy_stats['skipped']} failed={copy_stats['failed']}")
        if copy_stats["failed"] > 0:
            fail_path = stage_root / "copy_failures.jsonl"
            write_jsonl(fail_path, copy_stats["failures"])
            raise SystemExit(f"Copy failed for {copy_stats['failed']} files. See {fail_path}")

    manifest_slug = manifest_dataset_id.split("/", 1)[1]
    manifests_root = stage_root / manifest_slug
    ensure_dir(manifests_root)

    path_map: dict[str, dict] = {}
    map_rows: list[dict] = []
    for row in chunk_entries:
        rec = build_kaggle_path_record(
            row,
            kaggle_top_dir=kaggle_top_dir,
            audio_path_root_template=str(args.audio_path_root_template),
        )
        key = canonical_path(str(rec["source_abs"]))
        path_map[key] = rec
        map_rows.append(rec)
    write_jsonl(manifests_root / "audio_path_index.jsonl", map_rows)

    rewrite_stats: list[dict] = []
    copied_manifests: set[Path] = set()
    for manifest_path, out_name in manifest_specs:
        rewrite_stats.append(rewrite_manifest(manifest_path, manifests_root / out_name, path_map))
        if manifest_path not in copied_manifests:
            shutil.copy2(manifest_path, manifests_root / manifest_path.name)
            copied_manifests.add(manifest_path)
    write_json(manifests_root / "manifest_rewrite_stats.json", {"generated_utc": now_utc(), "stats": rewrite_stats})
    shutil.copy2(stage_root / "chunk_summary.json", manifests_root / "chunk_summary.json")

    for c in chunk_details:
        slug = str(c["chunk_slug"])
        dataset_id = str(c["dataset_id"])
        root = stage_root / slug
        ensure_dataset_metadata(
            root,
            dataset_id,
            audio_chunk_title(int(c["chunk_index"])),
            audio_chunk_subtitle(kaggle_top_dir, int(c["chunk_index"]), int(args.chunk_file_limit)),
            args.license_name,
        )
    ensure_dataset_metadata(
        manifests_root,
        manifest_dataset_id,
        manifest_title(),
        manifest_subtitle(),
        args.license_name,
    )
    print(f"[meta] wrote metadata for {len(chunk_details)} audio chunks + manifests")

    report: dict[str, Any] = {
        "generated_utc": now_utc(),
        "source_root": str(source_root),
        "stage_root": str(stage_root),
        "kaggle_top_dir": kaggle_top_dir,
        "audio_path_root_template": str(args.audio_path_root_template),
        "owner": args.owner,
        "audio_slug_prefix": args.audio_slug_prefix,
        "manifest_dataset": manifest_dataset_id,
        "name_preflight": name_preflight,
        "chunk_summary": chunk_summary,
        "copy_stats": {k: v for k, v in copy_stats.items() if k != "failures"},
        "rewrite_stats": rewrite_stats,
    }

    max_upload = int(args.max_chunks_upload)
    start_idx = max(1, int(args.start_chunk_index))
    end_idx = int(args.end_chunk_index) if int(args.end_chunk_index) > 0 else len(chunk_details)
    end_idx = min(len(chunk_details), end_idx)
    if start_idx > end_idx:
        raise SystemExit(f"Invalid chunk range: start={start_idx} end={end_idx}")
    candidate_chunks = [c for c in chunk_details if start_idx <= int(c["chunk_index"]) <= end_idx]
    upload_chunks = candidate_chunks if max_upload <= 0 else candidate_chunks[:max_upload]
    if not upload_chunks:
        raise SystemExit("No chunks selected after applying range/max filters.")

    if args.publish_phase in {"precreate", "upload"}:
        msg = f"Record_chunks chunked publish refresh {now_utc()}"
        seed_root = stage_root / "_precreate_seed"
        ensure_dir(seed_root)

        if args.publish_phase == "precreate":
            create_results: dict[str, str] = {}
            for c in upload_chunks:
                dataset_id = str(c["dataset_id"])
                slug = str(c["chunk_slug"])
                seed_folder = seed_root / slug
                ensure_dataset_metadata(
                    seed_folder,
                    dataset_id,
                    audio_chunk_title(int(c["chunk_index"])),
                    f"Seed placeholder for chunk {int(c['chunk_index']):03d}; upload via version.",
                    args.license_name,
                )
                (seed_folder / "_seed.txt").write_text(
                    f"seed dataset for {dataset_id}\ncreated_utc={now_utc()}\n",
                    encoding="utf-8",
                )
                print(f"[precreate] {dataset_id}")
                mode = create_dataset_only(
                    args.kaggle_exe,
                    dataset_id,
                    seed_folder,
                    env,
                    probe_root,
                    upload_retry_max=int(args.upload_retry_max),
                    upload_retry_backoff_seconds=float(args.upload_retry_backoff_seconds),
                )
                create_results[dataset_id] = mode
                if float(args.sleep_between_uploads_seconds) > 0:
                    time.sleep(float(args.sleep_between_uploads_seconds))

            seed_folder_manifest = seed_root / manifest_slug
            ensure_dataset_metadata(
                seed_folder_manifest,
                manifest_dataset_id,
                manifest_title(),
                "Seed placeholder for manifests; upload full files via version.",
                args.license_name,
            )
            (seed_folder_manifest / "_seed.txt").write_text(
                f"seed dataset for {manifest_dataset_id}\ncreated_utc={now_utc()}\n",
                encoding="utf-8",
            )
            print(f"[precreate] {manifest_dataset_id}")
            create_results[manifest_dataset_id] = create_dataset_only(
                args.kaggle_exe,
                manifest_dataset_id,
                seed_folder_manifest,
                env,
                probe_root,
                upload_retry_max=int(args.upload_retry_max),
                upload_retry_backoff_seconds=float(args.upload_retry_backoff_seconds),
            )
            report["upload_modes"] = create_results
            report["uploaded_chunk_count"] = 0
            report["uploaded_chunk_indices"] = []
            write_json(stage_root / "publish_report.json", report)
            print("[done] precreate phase completed.")
            return

        if args.publish_phase == "upload":
            upload_modes: dict[str, str] = {}
            for c in upload_chunks:
                dataset_id = str(c["dataset_id"])
                slug = str(c["chunk_slug"])
                root = stage_root / slug
                if not dataset_exists(args.kaggle_exe, dataset_id, env, probe_root / ("exists_before_upload_" + dataset_id.replace("/", "__"))):
                    raise SystemExit(f"Upload phase requires dataset to exist first: {dataset_id}")
                print(f"[upload] {dataset_id}")
                mode = version_dataset_only(
                    args.kaggle_exe,
                    dataset_id,
                    root,
                    msg,
                    env,
                    probe_root,
                    upload_retry_max=int(args.upload_retry_max),
                    upload_retry_backoff_seconds=float(args.upload_retry_backoff_seconds),
                )
                verify_dataset_files(
                    args.kaggle_exe,
                    dataset_id,
                    env,
                    expected_fragment=chunk_listing_fragments.get(int(c["chunk_index"])),
                )
                upload_modes[dataset_id] = mode
                if float(args.sleep_between_uploads_seconds) > 0:
                    time.sleep(float(args.sleep_between_uploads_seconds))

            print(f"[upload] {manifest_dataset_id}")
            if not dataset_exists(
                args.kaggle_exe,
                manifest_dataset_id,
                env,
                probe_root / ("exists_before_upload_" + manifest_dataset_id.replace("/", "__")),
            ):
                raise SystemExit(f"Upload phase requires dataset to exist first: {manifest_dataset_id}")
            mode_m = version_dataset_only(
                args.kaggle_exe,
                manifest_dataset_id,
                manifests_root,
                msg,
                env,
                probe_root,
                upload_retry_max=int(args.upload_retry_max),
                upload_retry_backoff_seconds=float(args.upload_retry_backoff_seconds),
            )
            verify_dataset_files(args.kaggle_exe, manifest_dataset_id, env, expected_fragment="manifest_rewrite_stats.json")
            upload_modes[manifest_dataset_id] = mode_m
            report["upload_modes"] = upload_modes
            report["uploaded_chunk_count"] = len(upload_chunks)
            report["uploaded_chunk_indices"] = [int(c["chunk_index"]) for c in upload_chunks]
            write_json(stage_root / "publish_report.json", report)
            print("[done] upload phase completed.")
            return

    if not args.skip_upload:
        msg = f"Record_chunks chunked publish refresh {now_utc()}"
        upload_modes: dict[str, str] = {}

        for c in upload_chunks:
            dataset_id = str(c["dataset_id"])
            slug = str(c["chunk_slug"])
            root = stage_root / slug
            print(f"[upload] {dataset_id}")
            mode, visible = upload_dataset(
                args.kaggle_exe,
                dataset_id,
                root,
                msg,
                env,
                probe_root,
                upload_retry_max=int(args.upload_retry_max),
                upload_retry_backoff_seconds=float(args.upload_retry_backoff_seconds),
            )
            if visible or mode == "version":
                verify_dataset_files(
                    args.kaggle_exe,
                    dataset_id,
                    env,
                    expected_fragment=chunk_listing_fragments.get(int(c["chunk_index"])),
                )
            else:
                print(f"[warn] skipped immediate files verification for {dataset_id} (create not yet visible).")
            upload_modes[dataset_id] = mode
            if float(args.sleep_between_uploads_seconds) > 0:
                time.sleep(float(args.sleep_between_uploads_seconds))

        print(f"[upload] {manifest_dataset_id}")
        mode_m, visible_m = upload_dataset(
            args.kaggle_exe,
            manifest_dataset_id,
            manifests_root,
            msg,
            env,
            probe_root,
            upload_retry_max=int(args.upload_retry_max),
            upload_retry_backoff_seconds=float(args.upload_retry_backoff_seconds),
        )
        if visible_m or mode_m == "version":
            verify_dataset_files(args.kaggle_exe, manifest_dataset_id, env, expected_fragment="manifest_rewrite_stats.json")
        else:
            print(f"[warn] skipped immediate files verification for {manifest_dataset_id} (create not yet visible).")
        upload_modes[manifest_dataset_id] = mode_m
        report["upload_modes"] = upload_modes
        report["uploaded_chunk_count"] = len(upload_chunks)
        report["uploaded_chunk_indices"] = [int(c["chunk_index"]) for c in upload_chunks]
    else:
        report["upload_modes"] = "skipped"

    write_json(stage_root / "publish_report.json", report)
    (stage_root / "publish_report.md").write_text(
        "\n".join(
            [
                "# Kaggle Chunked Publish Report",
                "",
                f"- generated_utc: {report['generated_utc']}",
                f"- source_root: {report['source_root']}",
                f"- stage_root: {report['stage_root']}",
                f"- kaggle_top_dir: {report['kaggle_top_dir']}",
                f"- total_files: {report['chunk_summary']['files_total']}",
                f"- total_gb: {report['chunk_summary']['total_gb']}",
                f"- chunk_file_limit: {report['chunk_summary']['chunk_file_limit']}",
                f"- chunks_total: {report['chunk_summary']['chunks_total']}",
                f"- manifest_dataset: {manifest_dataset_id}",
                f"- upload_modes: {report.get('upload_modes')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] report: {stage_root / 'publish_report.json'}")


if __name__ == "__main__":
    main()
