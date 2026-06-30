from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_INCLUDES = [
    "Record_harsha",
    "Transcriptions_corrected",
    "Record_only_by_harsha",
    "Record_others_compacted",
    "noise/RIRS_NOISES",
]

DEFAULT_EXTRA_FILES = [
    "whisper-acft/speaker_sort_scores.csv",
    "whisper-acft/most_commonly_spoken_segments_state.json",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: object) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def win_long(path: Path) -> str:
    s = str(path)
    if os.name != "nt":
        return s
    p = path.resolve()
    sp = str(p)
    if sp.startswith("\\\\?\\"):
        return sp
    if sp.startswith("\\\\"):
        return "\\\\?\\UNC\\" + sp.lstrip("\\")
    return "\\\\?\\" + sp


def normalize_rel(value: str) -> str:
    return value.strip().replace("\\", "/").strip("/")


def iter_files(root: Path, rel_root: str) -> list[dict]:
    src_root = root / Path(rel_root)
    if src_root.is_file():
        st = src_root.stat()
        rel_path = src_root.relative_to(root).as_posix()
        return [
            {
                "rel_path": rel_path,
                "source_path": str(src_root),
                "bytes": int(st.st_size),
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                "kind": classify_path(rel_path),
            }
        ]
    rows: list[dict] = []
    for p in src_root.rglob("*"):
        if not p.is_file():
            continue
        st = p.stat()
        rows.append(
            {
                "rel_path": p.relative_to(root).as_posix(),
                "source_path": str(p),
                "bytes": int(st.st_size),
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                "kind": classify_path(p.relative_to(root).as_posix()),
            }
        )
    return rows


def classify_path(rel: str) -> str:
    r = rel.lower()
    if r.startswith("record_harsha/"):
        return "primary_audio"
    if r.startswith("transcriptions_corrected/"):
        return "primary_transcript"
    if r.startswith("record_only_by_harsha/"):
        return "speaker_reference_target"
    if r.startswith("record_others_compacted/"):
        return "speaker_reference_other"
    if r.startswith("noise/rirs_noises/"):
        return "augmentation_noise_rir_source"
    if r.startswith("whisper-acft/"):
        return "small_pipeline_state"
    return "included"


def summarize(rows: list[dict]) -> dict:
    total_bytes = sum(int(r["bytes"]) for r in rows)
    by_kind: dict[str, dict[str, int]] = {}
    by_root: dict[str, dict[str, int]] = {}
    by_ext: dict[str, dict[str, int]] = {}
    for r in rows:
        rel = str(r["rel_path"])
        root = rel.split("/", 1)[0]
        ext = Path(rel).suffix.lower() or "<none>"
        for bucket, key in ((by_kind, str(r["kind"])), (by_root, root), (by_ext, ext)):
            entry = bucket.setdefault(key, {"files": 0, "bytes": 0})
            entry["files"] += 1
            entry["bytes"] += int(r["bytes"])
    return {
        "generated_utc": now_utc(),
        "files": len(rows),
        "bytes": total_bytes,
        "gb": round(total_bytes / (1024**3), 3),
        "by_kind": by_kind,
        "by_root": by_root,
        "by_ext": dict(sorted(by_ext.items(), key=lambda kv: kv[1]["files"], reverse=True)),
    }


def chunk_rows(rows: list[dict], file_limit: int, byte_limit: int = 0) -> list[list[dict]]:
    if file_limit <= 0 and byte_limit <= 0:
        raise ValueError("chunk_file_limit or chunk_byte_limit_gb must be > 0")
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0
    for row in rows:
        row_bytes = int(row["bytes"])
        would_exceed_files = file_limit > 0 and len(current) >= file_limit
        would_exceed_bytes = byte_limit > 0 and current and current_bytes + row_bytes > byte_limit
        if would_exceed_files or would_exceed_bytes:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(row)
        current_bytes += row_bytes
    if current:
        chunks.append(current)
    return chunks


def copy_one(row: dict, dataset_root: Path, source_root: Path) -> tuple[str, int, str]:
    rel = str(row["rel_path"])
    src = source_root / Path(rel)
    dst = dataset_root / Path(rel)
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size == int(row["bytes"]):
        return rel, int(row["bytes"]), "skipped"
    tmp = dst.with_suffix(dst.suffix + ".tmp_copy")
    if tmp.exists():
        tmp.unlink()
    shutil.copy2(win_long(src), win_long(tmp))
    os.replace(win_long(tmp), win_long(dst))
    return rel, int(row["bytes"]), "copied"


def materialize(rows: list[dict], dataset_root: Path, source_root: Path, workers: int) -> dict:
    copied = skipped = failed = 0
    bytes_copied = bytes_skipped = 0
    failures: list[dict] = []
    with futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(copy_one, row, dataset_root, source_root) for row in rows]
        for idx, fut in enumerate(futures.as_completed(futs), 1):
            try:
                rel, size, status = fut.result()
                if status == "copied":
                    copied += 1
                    bytes_copied += size
                else:
                    skipped += 1
                    bytes_skipped += size
            except Exception as e:
                failed += 1
                failures.append({"error": repr(e)})
            if idx % 1000 == 0:
                print(f"[copy] {idx}/{len(rows)} copied={copied} skipped={skipped} failed={failed}")
    return {
        "copied": copied,
        "skipped": skipped,
        "failed": failed,
        "bytes_copied": bytes_copied,
        "bytes_skipped": bytes_skipped,
        "failures": failures[:200],
    }


def archive_name(rel: str) -> str:
    normalized = normalize_rel(rel)
    if Path(normalized).suffix:
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
        return f"source_payload_{digest}.tar"
    return normalized.replace("/", "__") + ".tar"


def create_archive_payload(
    includes: list[str],
    extra_files: list[str],
    dataset_root: Path,
    source_root: Path,
    *,
    force: bool,
) -> dict:
    archives: list[dict] = []
    errors: list[dict] = []

    def build_one(name: str, paths: list[str]) -> None:
        archive_path = dataset_root / name
        if archive_path.exists() and archive_path.stat().st_size > 0 and not force:
            archives.append({"archive": name, "status": "skipped", "bytes": int(archive_path.stat().st_size), "paths": paths})
            return
        tmp = archive_path.with_suffix(archive_path.suffix + ".tmp")
        if tmp.exists():
            tmp.unlink()
        cmd = ["tar", "-cf", str(tmp), "-C", str(source_root), *paths]
        print("$ " + " ".join(f"'{c}'" if " " in c else c for c in cmd))
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            errors.append({"archive": name, "paths": paths, "returncode": proc.returncode, "output": proc.stdout[-2000:]})
            if tmp.exists():
                tmp.unlink()
            return
        os.replace(win_long(tmp), win_long(archive_path))
        archives.append({"archive": name, "status": "created", "bytes": int(archive_path.stat().st_size), "paths": paths})

    for rel in includes:
        build_one(archive_name(rel), [rel])
    if extra_files:
        build_one("extra_files.tar", extra_files)

    return {
        "archive_payload": True,
        "archives": archives,
        "failed": len(errors),
        "failures": errors,
        "bytes_archived": sum(int(a["bytes"]) for a in archives),
    }


def kaggle_text(value: str | None, fallback: str, min_len: int, max_len: int) -> str:
    text = " ".join((value or fallback).split())
    if len(text) > max_len:
        text = text[:max_len].rstrip(" -_.,")
    if len(text) < min_len:
        text = (text + " dataset")[:max_len]
    return text


def ensure_dataset_metadata(folder: Path, dataset_id: str, license_name: str, title: str | None, subtitle: str) -> None:
    slug = dataset_id.split("/", 1)[-1]
    metadata = {
        "id": dataset_id,
        "title": kaggle_text(title, slug.replace("-", " "), 6, 50),
        "subtitle": kaggle_text(subtitle, "Primary source files for training rebuilds.", 20, 80),
        "licenses": [{"name": license_name}],
    }
    write_json(folder / "dataset-metadata.json", metadata)


def write_readme(dataset_root: Path, summary: dict, dataset_id: str, includes: list[str], extra_files: list[str]) -> None:
    include_lines = "\n".join(f"- `{rel}/`" for rel in includes) or "- None"
    extra_lines = "\n".join(f"- `{rel}`" for rel in extra_files) or "- None"
    readme = f"""# ACFT Moonshine Primary Training Data

Dataset ID: `{dataset_id}`

Generated UTC: `{summary["generated_utc"]}`

This package keeps primary/rebuild inputs and excludes generated chunk WAVs,
augmented WAVs, checkpoints, run folders, caches, and virtual environments.

## Included

{include_lines}

## Extra Files

{extra_lines}

## Excluded

- `Record_chunks/` and `Record_chunks_*`: generated chunk and augmentation outputs.
- `Record_test_chunks/`: generated held-out chunk copy.
- `Stage_*`, `RUN__*`, `checkpoints_*`, `Dynamic_n_ctx_*`: checkpoints and run outputs.
- `cache/`, model caches, virtual environments, and temporary folders.

## Rebuild Anchor

The local pipeline detected `DATA_ROOT` by requiring `Transcriptions_corrected`
and `Record_harsha`. Stage 1 creates `Record_chunks/tasks_pending.jsonl` and
`Record_chunks/pairs_pending.jsonl`; stage 2 creates chunk WAVs and the first
manifest; later stages create augmented chunks and stage manifests.

See `manifests/primary_source_inventory.jsonl` and `manifests/source_summary.json`
for exact payload inventory.
"""
    (dataset_root / "README.md").write_text(readme, encoding="utf-8", newline="\n")


def run_cmd(cmd: list[str], env: dict[str, str], stream: bool = True) -> tuple[int, str]:
    print("$ " + " ".join(f"'{c}'" if " " in c else c for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    assert proc.stdout is not None
    lines: list[str] = []
    for line in proc.stdout:
        lines.append(line)
        if stream:
            print(line, end="")
    rc = proc.wait()
    return rc, "".join(lines)


def run_kaggle_dataset_cmd(cmd: list[str], env: dict[str, str]) -> tuple[int, str]:
    rc, out = run_cmd(cmd, env=env, stream=True)
    out_l = out.lower()
    if "unrecognized arguments: --dir-mode" not in out_l and "no such option: --dir-mode" not in out_l:
        return rc, out
    fallback: list[str] = []
    i = 0
    while i < len(cmd):
        token = cmd[i]
        if token == "--keep-tabular":
            i += 1
            continue
        if token == "--dir-mode":
            mode = cmd[i + 1] if i + 1 < len(cmd) else "tar"
            fallback.extend(["-r", mode])
            i += 2
            continue
        fallback.append(token)
        i += 1
    print("[kaggle] retrying with legacy archive flags")
    return run_cmd(fallback, env=env, stream=True)


def build_env_with_token() -> dict[str, str]:
    env = dict(os.environ)
    if env.get("KAGGLE_API_TOKEN") or (env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY")):
        return env
    user_tok = ""
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
            if not user_tok:
                user_tok = str(cfg.get("apiToken") or cfg.get("api_token") or "").strip()
            username = str(cfg.get("username") or "").strip()
            api_key = str(cfg.get("key") or "").strip()
            if username and api_key:
                env["KAGGLE_USERNAME"] = username
                env["KAGGLE_KEY"] = api_key
        except Exception:
            pass
    if user_tok:
        env["KAGGLE_API_TOKEN"] = user_tok
    return env


def dataset_exists(kaggle_exe: str, dataset_id: str, env: dict[str, str], probe_dir: Path) -> bool:
    ensure_dir(probe_dir)
    rc, _ = run_cmd([kaggle_exe, "datasets", "metadata", dataset_id, "-p", str(probe_dir)], env=env, stream=False)
    return rc == 0


def wait_for_dataset_exists(
    kaggle_exe: str,
    dataset_id: str,
    env: dict[str, str],
    probe_dir: Path,
    *,
    max_attempts: int = 60,
    sleep_seconds: float = 5.0,
) -> bool:
    for attempt in range(1, max_attempts + 1):
        if dataset_exists(kaggle_exe, dataset_id, env, probe_dir / f"attempt_{attempt:03d}"):
            return True
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


def prepare_seed_folder(seed_folder: Path, dataset_id: str, license_name: str, title: str | None, subtitle: str) -> None:
    ensure_dir(seed_folder)
    ensure_dataset_metadata(seed_folder, dataset_id, license_name, title, subtitle)
    (seed_folder / "README.md").write_text(
        "# ACFT Moonshine Primary Training Data\n\nSeed placeholder. Full payload is uploaded as a dataset version.\n",
        encoding="utf-8",
        newline="\n",
    )
    (seed_folder / "seed.txt").write_text(f"seed for {dataset_id}\n", encoding="utf-8", newline="\n")


def create_dataset_only(
    kaggle_exe: str,
    dataset_id: str,
    seed_folder: Path,
    env: dict[str, str],
    probe_root: Path,
    *,
    upload_retry_max: int,
    upload_retry_backoff_seconds: float,
) -> str:
    if dataset_exists(kaggle_exe, dataset_id, env, probe_root / ("exists_" + dataset_id.replace("/", "__"))):
        return "exists"
    cmd = [kaggle_exe, "datasets", "create", "-p", str(seed_folder), "--dir-mode", "tar", "--keep-tabular"]
    for attempt in range(1, max(1, upload_retry_max) + 1):
        rc, out = run_kaggle_dataset_cmd(cmd, env=env)
        reason = classify_upload_failure(out.lower())
        if rc == 0 and reason not in {"hard_error", "rate_limit"}:
            break
        if reason == "already_exists":
            return "exists"
        if reason == "rate_limit" and attempt < upload_retry_max:
            delay = min(upload_retry_backoff_seconds * (2 ** (attempt - 1)), 900.0)
            print(f"[warn] create rate-limited for {dataset_id}; retrying in {delay:.0f}s")
            time.sleep(delay)
            continue
        raise RuntimeError(f"Kaggle seed create failed for {dataset_id}")
    if not wait_for_dataset_exists(kaggle_exe, dataset_id, env, probe_root / ("postcreate_" + dataset_id.replace("/", "__"))):
        raise RuntimeError(f"Kaggle seed create accepted but dataset is not visible: {dataset_id}")
    return "created_seed"


def version_dataset_only(
    kaggle_exe: str,
    dataset_id: str,
    folder: Path,
    env: dict[str, str],
    probe_root: Path,
    *,
    upload_retry_max: int,
    upload_retry_backoff_seconds: float,
) -> str:
    cmd = [
        kaggle_exe,
        "datasets",
        "version",
        "-p",
        str(folder),
        "-m",
        f"primary training data refresh {now_utc()}",
        "--dir-mode",
        "tar",
        "--keep-tabular",
    ]
    for attempt in range(1, max(1, upload_retry_max) + 1):
        rc, out = run_kaggle_dataset_cmd(cmd, env=env)
        reason = classify_upload_failure(out.lower())
        if rc == 0 and reason not in {"hard_error", "rate_limit", "not_found"}:
            break
        if reason in {"rate_limit", "not_found"} and attempt < upload_retry_max:
            delay = min(upload_retry_backoff_seconds * (2 ** (attempt - 1)), 900.0)
            print(f"[warn] version blocked for {dataset_id} ({reason}); retrying in {delay:.0f}s")
            time.sleep(delay)
            continue
        raise RuntimeError(f"Kaggle version upload failed for {dataset_id}")
    if not wait_for_dataset_exists(kaggle_exe, dataset_id, env, probe_root / ("postversion_" + dataset_id.replace("/", "__")), max_attempts=30):
        raise RuntimeError(f"Kaggle version uploaded but dataset is not visible: {dataset_id}")
    return "version"


def dataset_files_contain(kaggle_exe: str, dataset_id: str, env: dict[str, str], needles: list[str]) -> tuple[bool, str]:
    rc, out = run_cmd([kaggle_exe, "datasets", "files", dataset_id, "--page-size", "200"], env=env, stream=False)
    if rc != 0:
        return False, out
    return any(n in out for n in needles), out


def wait_for_dataset_files(
    kaggle_exe: str,
    dataset_id: str,
    env: dict[str, str],
    needles: list[str],
    *,
    max_attempts: int = 30,
    sleep_seconds: float = 10.0,
) -> bool:
    for attempt in range(1, max_attempts + 1):
        found, _ = dataset_files_contain(kaggle_exe, dataset_id, env, needles)
        if found:
            return True
        print(f"[verify] waiting for {dataset_id} files ({attempt}/{max_attempts})")
        time.sleep(sleep_seconds)
    return False


def upload_dataset(
    kaggle_exe: str,
    dataset_id: str,
    dataset_root: Path,
    env: dict[str, str],
    probe_root: Path,
    *,
    seed_version: bool,
    license_name: str,
    title: str | None,
    subtitle: str,
    upload_retry_max: int,
    upload_retry_backoff_seconds: float,
) -> str:
    exists = dataset_exists(kaggle_exe, dataset_id, env, probe_root / dataset_id.replace("/", "__"))
    if seed_version and not exists:
        seed_folder = probe_root / "_seed" / dataset_id.split("/", 1)[1]
        prepare_seed_folder(seed_folder, dataset_id, license_name, title, subtitle)
        create_dataset_only(
            kaggle_exe,
            dataset_id,
            seed_folder,
            env,
            probe_root,
            upload_retry_max=upload_retry_max,
            upload_retry_backoff_seconds=upload_retry_backoff_seconds,
        )
        return version_dataset_only(
            kaggle_exe,
            dataset_id,
            dataset_root,
            env,
            probe_root,
            upload_retry_max=upload_retry_max,
            upload_retry_backoff_seconds=upload_retry_backoff_seconds,
        )
    if exists:
        cmd = [
            kaggle_exe,
            "datasets",
            "version",
            "-p",
            str(dataset_root),
            "-m",
            f"primary training data refresh {now_utc()}",
            "--dir-mode",
            "tar",
            "--keep-tabular",
        ]
        mode = "version"
    else:
        cmd = [
            kaggle_exe,
            "datasets",
            "create",
            "-p",
            str(dataset_root),
            "--dir-mode",
            "tar",
            "--keep-tabular",
        ]
        mode = "create"
    for attempt in range(1, max(1, upload_retry_max) + 1):
        rc, out = run_kaggle_dataset_cmd(cmd, env=env)
        reason = classify_upload_failure(out.lower())
        if rc == 0 and reason not in {"hard_error", "rate_limit"}:
            break
        if reason == "rate_limit" and attempt < upload_retry_max:
            delay = min(upload_retry_backoff_seconds * (2 ** (attempt - 1)), 900.0)
            print(f"[warn] upload rate-limited for {dataset_id}; retrying in {delay:.0f}s")
            time.sleep(delay)
            continue
        raise RuntimeError(f"Kaggle upload failed for {dataset_id} ({mode})")
    return mode


def write_dataset_package(
    dataset_root: Path,
    rows: list[dict],
    dataset_id: str,
    license_name: str,
    title: str | None,
    subtitle: str,
    includes: list[str],
    extra_files: list[str],
) -> dict:
    summary = summarize(rows)
    manifests_root = dataset_root / "manifests"
    write_jsonl(manifests_root / "primary_source_inventory.jsonl", rows)
    write_json(manifests_root / "source_summary.json", summary)
    ensure_dataset_metadata(dataset_root, dataset_id, license_name, title, subtitle)
    write_readme(dataset_root, summary, dataset_id, includes, extra_files)
    return summary


def run_chunked_publish(args: argparse.Namespace, source_root: Path, rows: list[dict], includes: list[str], extra_files: list[str]) -> None:
    stage_root = Path(args.stage_root)
    ensure_dir(stage_root)
    byte_limit = int(float(args.chunk_byte_limit_gb) * (1024**3))
    chunks = chunk_rows(rows, int(args.chunk_file_limit), byte_limit)
    width = max(3, len(str(len(chunks))))
    owner = args.owner or args.dataset_id.split("/", 1)[0]
    prefix = args.chunk_slug_prefix or (args.dataset_id.split("/", 1)[1] + "-chunk")
    start_idx = max(1, int(args.chunk_start_index))
    end_idx = int(args.chunk_end_index) if int(args.chunk_end_index) > 0 else len(chunks)
    end_idx = min(end_idx, len(chunks))
    selected = [(idx, chunks[idx - 1]) for idx in range(start_idx, end_idx + 1)]
    if int(args.max_chunks_upload) > 0:
        selected = selected[: int(args.max_chunks_upload)]
    if not selected:
        raise SystemExit("No chunks selected")

    env = build_env_with_token() if args.upload else {}
    reports: list[dict] = []
    for idx, chunk in selected:
        slug = f"{prefix}-{idx:0{width}d}"
        dataset_id = f"{owner}/{slug}"
        dataset_root = stage_root / slug
        ensure_dir(dataset_root)
        title = f"{args.title or slug} {idx:0{width}d}"
        print(f"[chunk] {idx}/{len(chunks)} files={len(chunk)} dataset={dataset_id}")
        summary = write_dataset_package(dataset_root, chunk, dataset_id, args.license_name, title, args.subtitle, includes, extra_files)

        copy_stats = {"plan_only": True}
        if not args.plan_only:
            if args.skip_copy:
                print("[copy] skipped by --skip_copy")
                copy_stats = {"skip_copy": True}
            else:
                copy_stats = materialize(chunk, dataset_root, source_root, args.workers)
                if copy_stats["failed"]:
                    write_json(stage_root / f"{slug}_copy_failures.json", copy_stats["failures"])
                    raise SystemExit(f"Copy failed for {copy_stats['failed']} files in {slug}")

        report: dict[str, object] = {
            "chunk_index": idx,
            "chunks_total": len(chunks),
            "dataset_id": dataset_id,
            "dataset_root": str(dataset_root),
            "includes": includes,
            "extra_files": extra_files,
            "summary": summary,
            "copy_stats": copy_stats,
            "uploaded": False,
        }
        if args.upload:
            if args.plan_only:
                raise SystemExit("--upload cannot be combined with --plan_only")
            mode = upload_dataset(
                args.kaggle_exe,
                dataset_id,
                dataset_root,
                env,
                stage_root / "_probe",
                seed_version=bool(args.seed_version),
                license_name=args.license_name,
                title=title,
                subtitle=args.subtitle,
                upload_retry_max=int(args.upload_retry_max),
                upload_retry_backoff_seconds=float(args.upload_retry_backoff_seconds),
            )
            needles = [f"{rel}/" for rel in includes] or [str(row["rel_path"]) for row in chunk]
            verified = wait_for_dataset_files(
                args.kaggle_exe,
                dataset_id,
                env,
                needles,
                max_attempts=int(args.verify_attempts),
                sleep_seconds=float(args.verify_sleep_seconds),
            )
            report["uploaded"] = True
            report["upload_mode"] = mode
            report["verified_files"] = verified
            report["url"] = f"https://www.kaggle.com/datasets/{dataset_id}"
            if not verified:
                print(f"[warn] uploaded but files not visible yet: {dataset_id}")
        reports.append(report)
        write_json(stage_root / "chunked_publish_report.json", {"generated_utc": now_utc(), "chunks": reports})
        if args.upload and float(args.sleep_between_uploads_seconds) > 0 and idx != selected[-1][0]:
            time.sleep(float(args.sleep_between_uploads_seconds))

    print(f"[done] chunked report: {stage_root / 'chunked_publish_report.json'}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage and optionally upload primary speech-training inputs to Kaggle")
    ap.add_argument("--source_root", default=r"I:\\")
    ap.add_argument("--stage_root", default=r"J:\kaggle_publish\acft-moonshine-primary-training-data-publish")
    ap.add_argument("--dataset_id", default="drsriharshaguthik/acft-moonshine-primary-training-data")
    ap.add_argument("--kaggle_exe", default=r"C:\Users\deletable\AppData\Roaming\Python\Python311\Scripts\kaggle.exe")
    ap.add_argument("--license_name", default="CC0-1.0")
    ap.add_argument("--title", default=None)
    ap.add_argument("--subtitle", default="Primary speech-training inputs for generated chunk rebuilds.")
    ap.add_argument("--include", action="append", default=[], help="Repo-relative source dir to include; repeatable")
    ap.add_argument("--extra_file", action="append", default=[], help="Repo-relative small file to include; repeatable")
    ap.add_argument("--extra_file_list", action="append", default=[], help="Text file with one repo-relative extra file per line; repeatable")
    ap.add_argument("--extra_manifest_jsonl", action="append", default=[], help="Manifest JSONL with rel_path entries to include; repeatable")
    ap.add_argument("--exclude_rel_path", action="append", default=[], help="Repo-relative path to exclude after selection; repeatable")
    ap.add_argument("--exclude_rel_path_list", action="append", default=[], help="Text file with one repo-relative exclude path per line; repeatable")
    ap.add_argument("--no_default_includes", action="store_true")
    ap.add_argument("--no_default_extra_files", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip_copy", action="store_true")
    ap.add_argument("--archive_payload", action="store_true", help="Upload source roots as top-level tar archives instead of individual files")
    ap.add_argument("--force_archive", action="store_true")
    ap.add_argument("--plan_only", action="store_true")
    ap.add_argument("--upload", action="store_true", help="Perform live Kaggle create/version after staging")
    ap.add_argument("--seed_version", action="store_true", help="Create a tiny seed dataset first, then upload full payload as a version")
    ap.add_argument("--upload_retry_max", type=int, default=4)
    ap.add_argument("--upload_retry_backoff_seconds", type=float, default=30.0)
    ap.add_argument("--verify_attempts", type=int, default=30)
    ap.add_argument("--verify_sleep_seconds", type=float, default=10.0)
    ap.add_argument("--chunk_file_limit", type=int, default=0, help="When >0, publish selected files as multiple bounded Kaggle datasets")
    ap.add_argument("--chunk_byte_limit_gb", type=float, default=0.0, help="When >0, start a new chunk before this many GiB")
    ap.add_argument("--chunk_slug_prefix", default="")
    ap.add_argument("--owner", default="")
    ap.add_argument("--chunk_start_index", type=int, default=1)
    ap.add_argument("--chunk_end_index", type=int, default=0, help="Inclusive; 0 means final chunk")
    ap.add_argument("--max_chunks_upload", type=int, default=0, help="0 means all selected chunks")
    ap.add_argument("--sleep_between_uploads_seconds", type=float, default=15.0)
    ap.add_argument("--row_start_index", type=int, default=1, help="1-based row slice start after deterministic sorting")
    ap.add_argument("--row_end_index", type=int, default=0, help="1-based inclusive row slice end; 0 means final row")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    if not source_root.exists():
        raise SystemExit(f"Missing source_root: {source_root}")
    if "/" not in args.dataset_id:
        raise SystemExit("--dataset_id must be owner/slug")
    include_args = args.include or ([] if args.no_default_includes else DEFAULT_INCLUDES)
    includes = [normalize_rel(x) for x in include_args]
    default_extra_files = [] if args.no_default_extra_files else DEFAULT_EXTRA_FILES
    extra_file_args = list(args.extra_file or default_extra_files)
    for list_path in args.extra_file_list or []:
        with open(list_path, "r", encoding="utf-8") as fh:
            for line in fh:
                rel = line.strip()
                if rel and not rel.startswith("#"):
                    extra_file_args.append(rel)
    for manifest_path in args.extra_manifest_jsonl or []:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rel = json.loads(line).get("rel_path")
                if rel:
                    extra_file_args.append(rel)
    extra_files = [normalize_rel(x) for x in extra_file_args]
    exclude_args = list(args.exclude_rel_path or [])
    for list_path in args.exclude_rel_path_list or []:
        with open(list_path, "r", encoding="utf-8") as fh:
            for line in fh:
                rel = line.strip()
                if rel and not rel.startswith("#"):
                    exclude_args.append(rel)
    exclude_rel_paths = {normalize_rel(x) for x in exclude_args}
    dataset_slug = args.dataset_id.split("/", 1)[1]
    stage_root = Path(args.stage_root)
    dataset_root = stage_root / dataset_slug
    probe_root = stage_root / "_probe"
    ensure_dir(dataset_root)

    rows: list[dict] = []
    missing: list[str] = []
    for rel in includes:
        p = source_root / Path(rel)
        if not p.exists():
            missing.append(rel)
            continue
        print(f"[scan] {rel}")
        rows.extend(iter_files(source_root, rel))
    for rel in extra_files:
        p = source_root / Path(rel)
        if not p.exists() or not p.is_file():
            missing.append(rel)
            continue
        st = p.stat()
        rows.append(
            {
                "rel_path": p.relative_to(source_root).as_posix(),
                "source_path": str(p),
                "bytes": int(st.st_size),
                "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
                "kind": classify_path(p.relative_to(source_root).as_posix()),
            }
        )
    if missing:
        raise SystemExit("Missing required inputs:\n" + "\n".join(f"- {x}" for x in missing))
    rows.sort(key=lambda r: str(r["rel_path"]).lower())
    if exclude_rel_paths:
        before_exclude = len(rows)
        rows = [row for row in rows if normalize_rel(str(row["rel_path"])) not in exclude_rel_paths]
        print(f"[exclude] removed={before_exclude - len(rows)} remaining={len(rows)}")
    row_start = max(1, int(args.row_start_index))
    row_end = int(args.row_end_index) if int(args.row_end_index) > 0 else len(rows)
    if row_start > 1 or row_end < len(rows):
        rows = rows[row_start - 1 : row_end]
        print(f"[slice] rows={row_start}..{row_end} selected={len(rows)}")
    if int(args.chunk_file_limit) > 0 or float(args.chunk_byte_limit_gb) > 0:
        run_chunked_publish(args, source_root, rows, includes, extra_files)
        return
    summary = summarize(rows)
    print(f"[summary] files={summary['files']} gb={summary['gb']} dataset={args.dataset_id}")

    manifests_root = dataset_root / "manifests"
    write_jsonl(manifests_root / "primary_source_inventory.jsonl", rows)
    write_json(manifests_root / "source_summary.json", summary)
    ensure_dataset_metadata(dataset_root, args.dataset_id, args.license_name, args.title, args.subtitle)
    write_readme(dataset_root, summary, args.dataset_id, includes, extra_files)

    copy_stats = {"plan_only": True}
    if not args.plan_only:
        if args.skip_copy:
            print("[copy] skipped by --skip_copy")
            copy_stats = {"skip_copy": True}
        elif args.archive_payload:
            copy_stats = create_archive_payload(
                includes,
                extra_files,
                dataset_root,
                source_root,
                force=bool(args.force_archive),
            )
            if copy_stats["failed"]:
                write_json(stage_root / "archive_failures.json", copy_stats["failures"])
                raise SystemExit(f"Archive failed for {copy_stats['failed']} archives; see {stage_root / 'archive_failures.json'}")
        else:
            copy_stats = materialize(rows, dataset_root, source_root, args.workers)
            if copy_stats["failed"]:
                write_json(stage_root / "copy_failures.json", copy_stats["failures"])
                raise SystemExit(f"Copy failed for {copy_stats['failed']} files; see {stage_root / 'copy_failures.json'}")

    report = {
        "generated_utc": now_utc(),
        "source_root": str(source_root),
        "stage_root": str(stage_root),
        "dataset_root": str(dataset_root),
        "dataset_id": args.dataset_id,
        "includes": includes,
        "extra_files": extra_files,
        "summary": summary,
        "copy_stats": copy_stats,
        "uploaded": False,
    }

    if args.upload:
        if args.plan_only:
            raise SystemExit("--upload cannot be combined with --plan_only")
        env = build_env_with_token()
        if not env.get("KAGGLE_API_TOKEN") and not (env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY")):
            raise SystemExit("No Kaggle credentials found in env or kaggle.json")
        mode = upload_dataset(
            args.kaggle_exe,
            args.dataset_id,
            dataset_root,
            env,
            probe_root,
            seed_version=bool(args.seed_version),
            license_name=args.license_name,
            title=args.title,
            subtitle=args.subtitle,
            upload_retry_max=int(args.upload_retry_max),
            upload_retry_backoff_seconds=float(args.upload_retry_backoff_seconds),
        )
        report["uploaded"] = True
        report["upload_mode"] = mode
        report["seed_version"] = bool(args.seed_version)
        report["url"] = f"https://www.kaggle.com/datasets/{args.dataset_id}"

    write_json(stage_root / "publish_report.json", report)
    print(f"[done] report: {stage_root / 'publish_report.json'}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
