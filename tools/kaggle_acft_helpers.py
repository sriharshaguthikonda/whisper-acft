#!/usr/bin/env python3
"""Shared helpers for resumable Kaggle ACFT notebooks.

Keep notebooks thin. This module owns filesystem reconstruction, resume state,
dataset metadata/publish plumbing, and conservative public-ASR mixing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


KAGGLE_DATASET_SLUG_MAX = 50
DEFAULT_TOP_LEVELS = (
    "Record_harsha",
    "Transcriptions_corrected",
    "Record_only_by_harsha",
    "Record_others_compacted",
    "noise",
    "whisper-acft",
)


@dataclass(frozen=True)
class ReconstructResult:
    copied_files: int
    skipped_existing_files: int
    missing_dataset_roots: list[str]
    scanned_dataset_roots: list[str]
    inventory_path: str


def _as_path(path: str | os.PathLike[str]) -> Path:
    return path if isinstance(path, Path) else Path(path)


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = _as_path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"bad JSONL at {p}:{line_no}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"JSONL row is not an object at {p}:{line_no}")
            rows.append(obj)
    return rows


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[dict[str, Any]]) -> None:
    p = _as_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def limit_jsonl(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    max_rows: int | None,
) -> int:
    rows = read_jsonl(src)
    if max_rows is not None and max_rows > 0:
        rows = rows[: int(max_rows)]
    write_jsonl(dst, rows)
    return len(rows)


def parse_canonical_dataset_handles(doc_path: str | os.PathLike[str]) -> list[str]:
    """Parse only the Canonical Kaggle Datasets section from export docs."""
    text = _as_path(doc_path).read_text(encoding="utf-8")
    in_canonical = False
    handles: list[str] = []
    seen: set[str] = set()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        h2 = re.match(r"^##\s+(.+?)\s*$", line)
        if h2:
            title = h2.group(1).strip().lower()
            if title == "canonical kaggle datasets":
                in_canonical = True
                continue
            if in_canonical:
                break
        if not in_canonical:
            continue

        for handle in re.findall(r"`([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)`", line):
            normalized = handle.lower()
            if normalized not in seen:
                handles.append(normalized)
                seen.add(normalized)

    return handles


def dataset_slug_from_handle(handle: str) -> str:
    if "/" not in handle:
        raise ValueError(f"Kaggle dataset handle must be owner/slug: {handle!r}")
    owner, slug = handle.split("/", 1)
    if not owner or not slug:
        raise ValueError(f"Kaggle dataset handle must be owner/slug: {handle!r}")
    return slug


def _iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _same_existing_file(src: Path, dst: Path) -> bool:
    if not dst.exists() or not dst.is_file():
        return False
    try:
        return src.stat().st_size == dst.stat().st_size
    except OSError:
        return False


def reconstruct_kaggle_sources(
    *,
    input_root: str | os.PathLike[str] = "/kaggle/input",
    output_root: str | os.PathLike[str] = "/kaggle/working/acft_data",
    dataset_handles: Sequence[str],
    top_levels: Sequence[str] = DEFAULT_TOP_LEVELS,
    dry_run: bool = False,
) -> ReconstructResult:
    """Rebuild old top-level paths from attached Kaggle datasets.

    Kaggle exposes each attached dataset as /kaggle/input/<slug>. The export
    datasets already preserve old top-level names inside each slug; this copies
    those trees back under one working root.
    """
    in_root = _as_path(input_root)
    out_root = _as_path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    inventory_path = out_root / "_kaggle_source_inventory.jsonl"

    copied = 0
    skipped = 0
    missing: list[str] = []
    scanned: list[str] = []
    inventory_rows: list[dict[str, Any]] = []

    for handle in dataset_handles:
        slug = dataset_slug_from_handle(handle)
        dataset_root = in_root / slug
        if not dataset_root.exists():
            missing.append(str(dataset_root))
            continue
        scanned.append(str(dataset_root))

        for top_level in top_levels:
            source_top = dataset_root / top_level
            if not source_top.exists():
                continue
            for src in _iter_files(source_top):
                rel = src.relative_to(dataset_root)
                dst = out_root / rel
                action = "skipped_existing" if _same_existing_file(src, dst) else "copied"
                if action == "copied":
                    copied += 1
                    if not dry_run:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, dst)
                else:
                    skipped += 1

                inventory_rows.append(
                    {
                        "dataset_handle": handle,
                        "dataset_slug": slug,
                        "source_path": str(src),
                        "dest_path": str(dst),
                        "relative_path": rel.as_posix(),
                        "top_level": rel.parts[0] if rel.parts else top_level,
                        "bytes": src.stat().st_size,
                        "action": "would_copy" if dry_run and action == "copied" else action,
                    }
                )

    write_jsonl(inventory_path, inventory_rows)
    return ReconstructResult(
        copied_files=copied,
        skipped_existing_files=skipped,
        missing_dataset_roots=missing,
        scanned_dataset_roots=scanned,
        inventory_path=str(inventory_path),
    )


def _hash_file(path: Path, hasher: "hashlib._Hash") -> None:
    hasher.update(path.as_posix().encode("utf-8"))
    hasher.update(b"\0")
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    hasher.update(b"\0")


def stage_signature(
    *,
    inputs: Iterable[str | os.PathLike[str]] = (),
    config: dict[str, Any] | None = None,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(json.dumps(config or {}, sort_keys=True, default=str).encode("utf-8"))
    hasher.update(b"\n")
    for raw in sorted((_as_path(p) for p in inputs), key=lambda p: p.as_posix()):
        if raw.is_file():
            hasher.update(b"file:")
            _hash_file(raw, hasher)
        elif raw.is_dir():
            hasher.update(b"dir:")
            hasher.update(raw.as_posix().encode("utf-8"))
            for child in _iter_files(raw):
                rel = child.relative_to(raw).as_posix()
                stat = child.stat()
                hasher.update(f"{rel}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
        else:
            hasher.update(f"missing:{raw.as_posix()}\n".encode("utf-8"))
    return hasher.hexdigest()


def _stage_state_path(stage_name: str, state_dir: str | os.PathLike[str]) -> Path:
    safe = slugify_component(stage_name)
    return _as_path(state_dir) / f"{safe}.resume.json"


def should_run_stage(
    stage_name: str,
    *,
    outputs: Iterable[str | os.PathLike[str]],
    signature: str,
    state_dir: str | os.PathLike[str],
) -> bool:
    output_paths = [_as_path(p) for p in outputs]
    if any(not p.exists() for p in output_paths):
        return True

    state_path = _stage_state_path(stage_name, state_dir)
    if not state_path.exists():
        return True
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return True
    return state.get("signature") != signature


def mark_stage_done(
    stage_name: str,
    *,
    outputs: Iterable[str | os.PathLike[str]],
    signature: str,
    state_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    state_path = _stage_state_path(stage_name, state_dir)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta = []
    for raw in outputs:
        p = _as_path(raw)
        stat = p.stat() if p.exists() else None
        output_meta.append(
            {
                "path": str(p),
                "exists": p.exists(),
                "bytes": stat.st_size if stat else None,
                "mtime_ns": stat.st_mtime_ns if stat else None,
            }
        )
    state = {
        "stage": stage_name,
        "signature": signature,
        "completed_unix": time.time(),
        "outputs": output_meta,
    }
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return state


def run_resumable_stage(
    stage_name: str,
    command: Sequence[str],
    *,
    inputs: Iterable[str | os.PathLike[str]],
    outputs: Iterable[str | os.PathLike[str]],
    config: dict[str, Any],
    state_dir: str | os.PathLike[str],
    dry_run: bool = False,
) -> dict[str, Any]:
    sig = stage_signature(inputs=inputs, config=config)
    output_list = list(outputs)
    if not should_run_stage(stage_name, outputs=output_list, signature=sig, state_dir=state_dir):
        return {"stage": stage_name, "skipped": True, "signature": sig, "command": list(command)}
    if dry_run:
        return {"stage": stage_name, "skipped": False, "dry_run": True, "signature": sig, "command": list(command)}
    subprocess.run(list(command), check=True)
    mark_stage_done(stage_name, outputs=output_list, signature=sig, state_dir=state_dir)
    return {"stage": stage_name, "skipped": False, "signature": sig, "command": list(command)}


def slugify_component(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug or "acft"


def make_dataset_handle(owner: str, prefix: str, run_tag: str, suffix: str = "") -> str:
    slug = slugify_component("-".join(part for part in (prefix, run_tag, suffix) if part))
    if len(slug) > KAGGLE_DATASET_SLUG_MAX:
        digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:8]
        keep = KAGGLE_DATASET_SLUG_MAX - len(digest) - 1
        slug = f"{slug[:keep].rstrip('-')}-{digest}"
    slug = slug.strip("-")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*[a-z0-9]", slug):
        raise ValueError(f"invalid Kaggle dataset slug after normalization: {slug!r}")
    return f"{owner}/{slug}"


def write_dataset_metadata(
    *,
    local_dir: str | os.PathLike[str],
    handle: str,
    title: str,
    subtitle: str = "",
    keywords: Sequence[str] = (),
    licenses: Sequence[dict[str, str]] | None = None,
) -> dict[str, Any]:
    directory = _as_path(local_dir)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": handle,
        "title": title,
        "subtitle": subtitle,
        "isPrivate": True,
        "licenses": list(licenses or [{"name": "unknown"}]),
        "keywords": list(keywords),
    }
    path = directory / "dataset-metadata.json"
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"metadata_path": str(path), "metadata": metadata}


def publish_dataset(
    *,
    local_dir: str | os.PathLike[str],
    handle: str,
    version_notes: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    directory = _as_path(local_dir)
    plan = {
        "handle": handle,
        "local_dir": str(directory),
        "version_notes": version_notes,
        "dry_run": bool(dry_run),
    }
    plan_path = directory / "_publish_plan.json"
    directory.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if dry_run:
        return plan

    try:
        import kagglehub  # type: ignore

        upload = getattr(kagglehub, "dataset_upload", None)
        if upload is not None:
            try:
                result = upload(handle, str(directory), version_notes=version_notes)
            except TypeError:
                result = upload(handle, str(directory))
            return {**plan, "backend": "kagglehub", "result": str(result)}
    except Exception as exc:
        plan["kagglehub_error"] = repr(exc)

    version_cmd = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(directory),
        "-m",
        version_notes,
        "--dir-mode",
        "tar",
    ]
    create_cmd = [
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(directory),
        "--dir-mode",
        "tar",
        "--private",
    ]
    version = subprocess.run(version_cmd, text=True, capture_output=True)
    if version.returncode == 0:
        return {**plan, "backend": "kaggle-cli-version", "stdout": version.stdout}
    create = subprocess.run(create_cmd, text=True, capture_output=True)
    if create.returncode != 0:
        raise RuntimeError(
            "Kaggle publish failed. "
            f"version stderr={version.stderr!r}; create stderr={create.stderr!r}"
        )
    return {**plan, "backend": "kaggle-cli-create", "stdout": create.stdout}


def _max_public_for_ratio(private_count: int, public_ratio: float) -> int:
    if private_count <= 0:
        return 0
    if not (0 <= public_ratio < 1):
        raise ValueError("public_ratio must be in [0, 1)")
    return int(math.floor(private_count * public_ratio / max(1e-12, 1.0 - public_ratio)))


def mix_private_and_public_rows(
    private_rows: Sequence[dict[str, Any]],
    public_rows: Sequence[dict[str, Any]],
    *,
    public_ratio: float = 0.30,
    seed: int = 17,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    private = [dict(row, dataset_scope="private_acft", exclude_from_private_eval=False) for row in private_rows]
    public_cap = _max_public_for_ratio(len(private), public_ratio)
    public_pool = [dict(row) for row in public_rows]
    rng.shuffle(public_pool)
    public = [
        dict(
            row,
            dataset_scope="public_asr",
            exclude_from_private_eval=True,
            public_asr=True,
        )
        for row in public_pool[:public_cap]
    ]
    combined = private + public
    rng.shuffle(combined)
    return combined


def build_public_asr_manifest_from_hf(
    *,
    output_dir: str | os.PathLike[str],
    specs: Sequence[dict[str, Any]],
    max_rows: int,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Create a small public-ASR manifest from HF Datasets streaming rows.

    Specs are dictionaries with name/config/split/text_column/audio_column/source.
    Audio is decoded by datasets and written as WAV so Stage 17 sees local files.
    """
    from datasets import load_dataset  # type: ignore
    import soundfile as sf  # type: ignore

    rng = random.Random(seed)
    out_root = _as_path(output_dir)
    rows: list[dict[str, Any]] = []
    if max_rows <= 0:
        return rows

    per_spec = max(1, int(math.ceil(max_rows / max(1, len(specs)))))
    for spec in specs:
        name = spec["name"]
        config = spec.get("config")
        split = spec.get("split", "train")
        text_column = spec.get("text_column", "text")
        audio_column = spec.get("audio_column", "audio")
        source = slugify_component(spec.get("source") or name)
        stream = load_dataset(name, config, split=split, streaming=True)
        for idx, row in enumerate(stream.shuffle(seed=seed, buffer_size=1000).take(per_spec)):
            audio = row.get(audio_column)
            text = (row.get(text_column) or row.get("sentence") or row.get("transcription") or "").strip()
            if not text or not isinstance(audio, dict) or "array" not in audio or "sampling_rate" not in audio:
                continue
            wav_path = out_root / source / f"{len(rows):06d}.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(wav_path), audio["array"], int(audio["sampling_rate"]))
            rows.append(
                {
                    "audio_path": str(wav_path),
                    "raw_transcription": text,
                    "source_audio": str(wav_path),
                    "source_dataset": spec.get("source") or name,
                    "dataset_scope": "public_asr",
                    "exclude_from_private_eval": True,
                }
            )
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break

    rng.shuffle(rows)
    return rows[:max_rows]


def write_stage12_smoke_fixture(root: str | os.PathLike[str]) -> dict[str, str]:
    """Create a tiny Stage 1/2 compatible source tree without private data."""
    base = _as_path(root)
    audio_dir = base / "Record_harsha"
    transcript_dir = base / "Transcriptions_corrected"
    chunks_dir = base / "Record_chunks"
    audio_dir.mkdir(parents=True, exist_ok=True)
    transcript_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    wav_path = audio_dir / "synthetic_smoke.wav"
    sample_rate = 16000
    total_frames = sample_rate * 3
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\0\0" * total_frames)

    transcript_path = transcript_dir / "synthetic_smoke.json"
    transcript = {
        "input_file": {"path": str(wav_path)},
        "groq_response": {
            "segments": [
                {
                    "start": 0.25,
                    "end": 2.75,
                    "text": "this is a synthetic smoke test",
                    "no_speech_prob": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 1.0,
                }
            ]
        },
    }
    transcript_path.write_text(json.dumps(transcript, indent=2) + "\n", encoding="utf-8")

    return {
        "audio_dir": str(audio_dir),
        "transcript_dir": str(transcript_dir),
        "chunks_dir": str(chunks_dir),
        "audio_path": str(wav_path),
        "transcript_path": str(transcript_path),
    }


def as_jsonable_dataclass(value: Any) -> dict[str, Any]:
    return asdict(value)
