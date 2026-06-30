#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any


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


def run_cmd(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, stream: bool = True) -> tuple[int, str]:
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


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def enumerate_source_files(source_root: Path) -> list[dict]:
    files: list[dict] = []
    for p in source_root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(source_root).as_posix()
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            files.append({"src": str(p), "rel": rel, "size": int(size)})
    return files


def assign_shards(files: list[dict]) -> tuple[list[dict], dict]:
    ordered = sorted(files, key=lambda r: (-int(r["size"]), str(r["rel"])))
    a_total = 0
    b_total = 0
    out: list[dict] = []
    for row in ordered:
        if a_total <= b_total:
            shard = "a"
            a_total += int(row["size"])
        else:
            shard = "b"
            b_total += int(row["size"])
        out.append({**row, "shard": shard})
    summary = {
        "files_total": len(out),
        "bytes_total": int(a_total + b_total),
        "a_files": sum(1 for r in out if r["shard"] == "a"),
        "b_files": sum(1 for r in out if r["shard"] == "b"),
        "a_bytes": int(a_total),
        "b_bytes": int(b_total),
        "a_gb": round(a_total / (1024**3), 3),
        "b_gb": round(b_total / (1024**3), 3),
        "total_gb": round((a_total + b_total) / (1024**3), 3),
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


def materialize_shards(
    entries: list[dict],
    shard_a_root: Path,
    shard_b_root: Path,
    *,
    workers: int,
) -> dict:
    copied = 0
    skipped = 0
    failed = 0
    failures: list[dict] = []

    ensure_dir(shard_a_root)
    ensure_dir(shard_b_root)

    total = len(entries)
    pending = set()
    idx = 0
    progress_last = time.time()

    def submit_one(ex: ThreadPoolExecutor, item: dict):
        src = Path(item["src"])
        root = shard_a_root if item["shard"] == "a" else shard_b_root
        dst = root / "Record_chunks" / Path(item["rel"])
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
                        failures.append({"src": item.get("src"), "rel": item.get("rel"), "shard": item.get("shard"), "error": status})

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
    shard_a_slug: str,
    shard_b_slug: str,
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
                missing += 1
                out_rows.append(json.dumps(obj, ensure_ascii=False))
                continue
            slug = shard_a_slug if mapped["shard"] == "a" else shard_b_slug
            rel = str(mapped["rel"]).replace("\\", "/")
            kaggle_path = f"/kaggle/input/{slug}/Record_chunks/{rel}"
            obj["audio_path_original"] = ap
            obj["audio_path"] = kaggle_path
            rewritten += 1
            out_rows.append(json.dumps(obj, ensure_ascii=False))

    with out_manifest.open("w", encoding="utf-8") as w:
        for row in out_rows:
            w.write(row + "\n")

    return {
        "source_manifest": str(source_manifest),
        "out_manifest": str(out_manifest),
        "rows_total": rows_total,
        "rewritten": rewritten,
        "missing": missing,
    }


def ensure_dataset_metadata(folder: Path, dataset_id: str, title: str, subtitle: str, license_name: str) -> Path:
    md_path = folder / "dataset-metadata.json"
    payload = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": license_name}],
        "subtitle": subtitle,
        "description": subtitle,
    }
    write_json(md_path, payload)
    return md_path


def dataset_exists(kaggle_exe: str, dataset_id: str, env: dict[str, str], probe_dir: Path) -> bool:
    ensure_dir(probe_dir)
    rc, _out = run_cmd(
        [kaggle_exe, "datasets", "metadata", dataset_id, "-p", str(probe_dir)],
        env=env,
        stream=False,
    )
    return rc == 0


def upload_dataset(kaggle_exe: str, dataset_id: str, folder: Path, message: str, env: dict[str, str], probe_dir: Path) -> str:
    exists = dataset_exists(kaggle_exe, dataset_id, env, probe_dir / dataset_id.replace("/", "__"))
    if exists:
        cmd = [kaggle_exe, "datasets", "version", "-p", str(folder), "-m", message, "-r", "tar", "-t"]
        mode = "version"
    else:
        cmd = [kaggle_exe, "datasets", "create", "-p", str(folder), "-r", "tar", "-t"]
        mode = "create"
    rc, out = run_cmd(cmd, env=env, stream=True)
    out_l = out.lower()
    if rc != 0 or "dataset creation error" in out_l or "dataset versioning error" in out_l or "slug can only contain" in out_l:
        raise RuntimeError(f"Upload failed for {dataset_id} ({mode})")
    if not dataset_exists(kaggle_exe, dataset_id, env, probe_dir / ("postcheck_" + dataset_id.replace("/", "__"))):
        raise RuntimeError(f"Upload did not materialize dataset {dataset_id} ({mode})")
    return mode


def verify_dataset_files(kaggle_exe: str, dataset_id: str, env: dict[str, str]) -> None:
    rc, out = run_cmd([kaggle_exe, "datasets", "files", dataset_id, "--page-size", "200"], env=env, stream=True)
    if rc != 0:
        raise RuntimeError(f"Verification failed for {dataset_id}")
    if "No files found" in out:
        raise RuntimeError(f"No files listed for {dataset_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Publish Record_chunks to Kaggle private sharded datasets")
    ap.add_argument("--source_root", default=r"I:\Record_chunks")
    ap.add_argument("--stage_root", default=r"J:\kaggle_publish\acft-moonshine-Record_chunks_publish")
    ap.add_argument("--train_manifest", default=r"I:\Record_chunks\pairs_manifest_stage15_train_no_targets_randomized.jsonl")
    ap.add_argument("--test_manifest", default=r"I:\Record_chunks\pairs_manifest_stage13_test_randomized.jsonl")
    ap.add_argument("--kaggle_exe", default=r"C:\Users\deletable\AppData\Roaming\Python\Python311\Scripts\kaggle.exe")
    ap.add_argument("--dataset_a", default="drsriharshaguthik/acft-moonshine-record-chunks-audio-shard-a")
    ap.add_argument("--dataset_b", default="drsriharshaguthik/acft-moonshine-record-chunks-audio-shard-b")
    ap.add_argument("--dataset_m", default="drsriharshaguthik/acft-moonshine-record-chunks-manifests")
    ap.add_argument("--license_name", default="CC0-1.0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip_upload", action="store_true")
    ap.add_argument("--only_preflight", action="store_true")
    args = ap.parse_args()

    source_root = Path(args.source_root)
    if not source_root.exists():
        raise SystemExit(f"Missing source root: {source_root}")

    train_manifest = Path(args.train_manifest)
    test_manifest = Path(args.test_manifest)
    if not train_manifest.exists() or not test_manifest.exists():
        raise SystemExit("Train/test manifests are required and must exist.")

    stage_root = Path(args.stage_root)
    shard_a_slug = args.dataset_a.split("/", 1)[1]
    shard_b_slug = args.dataset_b.split("/", 1)[1]
    man_slug = args.dataset_m.split("/", 1)[1]
    shard_a_root = stage_root / shard_a_slug
    shard_b_root = stage_root / shard_b_slug
    manifests_root = stage_root / man_slug
    probe_root = stage_root / "_probe"
    ensure_dir(stage_root)
    ensure_dir(probe_root)

    env = dict(os.environ)
    if not env.get("KAGGLE_API_TOKEN"):
        user_tok = os.environ.get("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_API_TOKEN")
        if not user_tok:
            try:
                import winreg  # type: ignore

                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                    user_tok = winreg.QueryValueEx(k, "KAGGLE_API_TOKEN")[0]
            except Exception:
                user_tok = ""
        if user_tok:
            env["KAGGLE_API_TOKEN"] = user_tok

    rc, out = run_cmd([args.kaggle_exe, "config", "view"], env=env, stream=True)
    if rc != 0:
        raise SystemExit("Kaggle auth preflight failed.")
    if "drsriharshaguthik" not in out:
        raise SystemExit("Authenticated Kaggle username mismatch.")

    print("[scan] enumerating source files...")
    files = enumerate_source_files(source_root)
    if not files:
        raise SystemExit("No source files found.")
    source_total = sum(int(r["size"]) for r in files)
    print(f"[scan] files={len(files)} total_gb={source_total / (1024**3):.3f}")

    shard_entries, shard_summary = assign_shards(files)
    print(f"[shard] summary={json.dumps(shard_summary, ensure_ascii=False)}")
    write_json(stage_root / "shard_summary.json", shard_summary)
    write_jsonl(stage_root / "shard_assignment.jsonl", shard_entries)

    if args.only_preflight:
        print("[done] preflight only.")
        return

    print("[copy] materializing shard A/B trees...")
    copy_stats = materialize_shards(
        shard_entries,
        shard_a_root=shard_a_root,
        shard_b_root=shard_b_root,
        workers=int(args.workers),
    )
    print(f"[copy] stats copied={copy_stats['copied']} skipped={copy_stats['skipped']} failed={copy_stats['failed']}")
    if copy_stats["failed"] > 0:
        fail_path = stage_root / "copy_failures.jsonl"
        write_jsonl(fail_path, copy_stats["failures"])
        raise SystemExit(f"Copy failed for {copy_stats['failed']} files. See {fail_path}")

    ensure_dir(manifests_root)
    path_map: dict[str, dict] = {}
    map_rows: list[dict] = []
    for row in shard_entries:
        src = str(Path(row["src"]).resolve())
        key = canonical_path(src)
        rec = {
            "source_abs": src,
            "rel": row["rel"],
            "shard": row["shard"],
            "size": int(row["size"]),
            "kaggle_path": f"/kaggle/input/{shard_a_slug if row['shard']=='a' else shard_b_slug}/Record_chunks/{row['rel']}",
        }
        path_map[key] = rec
        map_rows.append(rec)
    write_jsonl(manifests_root / "audio_path_index.jsonl", map_rows)

    rewrite_stats: list[dict] = []
    rewrite_stats.append(
        rewrite_manifest(
            train_manifest,
            manifests_root / "pairs_manifest_stage15_train_no_targets_randomized_kaggle.jsonl",
            path_map,
            shard_a_slug,
            shard_b_slug,
        )
    )
    rewrite_stats.append(
        rewrite_manifest(
            test_manifest,
            manifests_root / "pairs_manifest_stage13_test_randomized_kaggle.jsonl",
            path_map,
            shard_a_slug,
            shard_b_slug,
        )
    )
    write_json(manifests_root / "manifest_rewrite_stats.json", {"generated_utc": now_utc(), "stats": rewrite_stats})

    # Include originals for audit.
    shutil.copy2(train_manifest, manifests_root / train_manifest.name)
    shutil.copy2(test_manifest, manifests_root / test_manifest.name)
    shutil.copy2(stage_root / "shard_summary.json", manifests_root / "shard_summary.json")

    md_a = ensure_dataset_metadata(
        shard_a_root,
        args.dataset_a,
        "acft-moonshine-Record_chunks_audio-shard-a",
        "Shard A of full Record_chunks publish for moonshine training.",
        args.license_name,
    )
    md_b = ensure_dataset_metadata(
        shard_b_root,
        args.dataset_b,
        "acft-moonshine-Record_chunks_audio-shard-b",
        "Shard B of full Record_chunks publish for moonshine training.",
        args.license_name,
    )
    md_m = ensure_dataset_metadata(
        manifests_root,
        args.dataset_m,
        "acft-moonshine-Record_chunks_manifests",
        "Kaggle-ready manifests and path index for Record_chunks shard datasets.",
        args.license_name,
    )
    print(f"[meta] wrote {md_a}")
    print(f"[meta] wrote {md_b}")
    print(f"[meta] wrote {md_m}")

    report = {
        "generated_utc": now_utc(),
        "source_root": str(source_root),
        "stage_root": str(stage_root),
        "dataset_ids": {
            "audio_shard_a": args.dataset_a,
            "audio_shard_b": args.dataset_b,
            "manifests": args.dataset_m,
        },
        "shard_summary": shard_summary,
        "copy_stats": {k: v for k, v in copy_stats.items() if k != "failures"},
        "rewrite_stats": rewrite_stats,
    }

    if not args.skip_upload:
        msg = f"Record_chunks full publish refresh {now_utc()}"
        mode_a = upload_dataset(args.kaggle_exe, args.dataset_a, shard_a_root, msg, env, probe_root)
        mode_b = upload_dataset(args.kaggle_exe, args.dataset_b, shard_b_root, msg, env, probe_root)
        mode_m = upload_dataset(args.kaggle_exe, args.dataset_m, manifests_root, msg, env, probe_root)
        verify_dataset_files(args.kaggle_exe, args.dataset_a, env)
        verify_dataset_files(args.kaggle_exe, args.dataset_b, env)
        verify_dataset_files(args.kaggle_exe, args.dataset_m, env)
        report["upload_modes"] = {"audio_shard_a": mode_a, "audio_shard_b": mode_b, "manifests": mode_m}
        report["urls"] = {
            "audio_shard_a": f"https://www.kaggle.com/datasets/{args.dataset_a}",
            "audio_shard_b": f"https://www.kaggle.com/datasets/{args.dataset_b}",
            "manifests": f"https://www.kaggle.com/datasets/{args.dataset_m}",
        }
    else:
        report["upload_modes"] = "skipped"

    write_json(stage_root / "publish_report.json", report)
    (stage_root / "publish_report.md").write_text(
        "\n".join(
            [
                "# Kaggle Publish Report",
                "",
                f"- generated_utc: {report['generated_utc']}",
                f"- source_root: {report['source_root']}",
                f"- stage_root: {report['stage_root']}",
                f"- total_files: {report['shard_summary']['files_total']}",
                f"- total_gb: {report['shard_summary']['total_gb']}",
                f"- shard_a_gb: {report['shard_summary']['a_gb']}",
                f"- shard_b_gb: {report['shard_summary']['b_gb']}",
                f"- dataset_a: {args.dataset_a}",
                f"- dataset_b: {args.dataset_b}",
                f"- dataset_m: {args.dataset_m}",
                f"- upload_modes: {report.get('upload_modes')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[done] report: {stage_root / 'publish_report.json'}")


if __name__ == "__main__":
    main()
