#!/usr/bin/env python3
"""
Verify NFA timestamp corrections by re-transcribing per-segment audio and comparing to JSON transcripts.

High-level flow (per file):
1) Load original + patched JSON (Groq envelopes or raw segments)
2) Extract segments + timestamps
3) Cut audio by segment timestamps
4) Transcribe each segment with Groq Whisper
5) Compare Groq text vs JSON segment text (WER/CER)
6) Report whether patched timestamps improve alignment vs original
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

try:
    from groq_transcribe_local_utils import (
        AUDIO_EXTS,
        GROQ_TRANSCRIBE_URL,
        Key,
        KeyPool,
        groq_transcribe_requests,
        which_or_die,
    )
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Missing local helper module: groq_transcribe_local_utils.py ({exc})")


SEGMENT_TIME_KEYS = ("start", "end")


@dataclass(frozen=True)
class SegmentRef:
    idx: int
    seg_id: int
    start: float
    end: float
    text: str


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_segments(data: object) -> List[dict]:
    if isinstance(data, dict):
        gr = data.get("groq_response")
        if isinstance(gr, dict) and isinstance(gr.get("segments"), list):
            return gr["segments"]
        if isinstance(data.get("segments"), list):
            return data["segments"]
        # fall back to first list-of-dicts that looks like segments
        for v in data.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                if "start" in v[0] and "end" in v[0]:
                    return v
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data
    raise ValueError("Could not locate segments in JSON.")


def _extract_request(data: object) -> dict:
    if isinstance(data, dict):
        req = data.get("request")
        if isinstance(req, dict):
            return req
        if isinstance(data.get("groq_response"), dict):
            return {}
    return {}


def _segment_id(seg: dict, idx: int) -> int:
    try:
        return int(seg.get("id", idx))
    except Exception:
        return idx


def _segment_text(seg: dict) -> str:
    t = seg.get("text", "")
    return "" if t is None else str(t)


def _segment_times(seg: dict) -> Optional[Tuple[float, float]]:
    try:
        s = float(seg.get("start"))
        e = float(seg.get("end"))
        return s, e
    except Exception:
        return None


def _normalize_text(text: str) -> str:
    t = text.lower()
    t = t.replace("’", "'").replace("‘", "'")
    t = re.sub(r"[^a-z0-9\s']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _wer(ref: str, hyp: str) -> float:
    ref_norm = _normalize_text(ref)
    hyp_norm = _normalize_text(hyp)
    ref_words = ref_norm.split() if ref_norm else []
    hyp_words = hyp_norm.split() if hyp_norm else []
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    dist = _levenshtein(ref_words, hyp_words)
    return dist / max(1, len(ref_words))


def _cer(ref: str, hyp: str) -> float:
    ref_norm = _normalize_text(ref).replace(" ", "")
    hyp_norm = _normalize_text(hyp).replace(" ", "")
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    dist = _levenshtein(list(ref_norm), list(hyp_norm))
    return dist / max(1, len(ref_norm))


def _find_audio_files(audio_dir: Path, recursive: bool) -> List[Path]:
    exts = set((e if e.startswith(".") else "." + e) for e in AUDIO_EXTS)
    if recursive:
        files = [p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in audio_dir.glob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: str(p).lower())
    return files


def _index_audio_by_stem(audio_files: Iterable[Path]) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for p in audio_files:
        key = p.stem.lower()
        idx.setdefault(key, []).append(p)
    return idx


def _pick_audio_for_json(json_path: Path, audio_index: Dict[str, List[Path]]) -> Optional[Path]:
    key = json_path.stem.lower()
    matches = audio_index.get(key, [])
    if not matches:
        return None
    # Prefer exact stem match (case-sensitive) if available
    exact = [p for p in matches if p.stem == json_path.stem]
    if exact:
        return exact[0]
    return matches[0]


def _safe_name(s: str) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("._-") or "segment"


def _ffmpeg_cut_segment(
    ffmpeg_bin: str,
    audio_path: Path,
    start_s: float,
    end_s: float,
    out_path: Path,
    sample_rate: int,
    channels: int,
    accurate_seek: bool,
) -> None:
    duration = max(0.0, end_s - start_s)
    if duration <= 0:
        raise ValueError("Segment duration <= 0")
    if accurate_seek:
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(audio_path),
            "-ss",
            f"{start_s:.3f}",
            "-t",
            f"{duration:.3f}",
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            "-y",
            str(out_path),
        ]
    else:
        cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            "-y",
            str(out_path),
        ]
    subprocess.run(cmd, check=True)


def _load_keys_from_env_names(env_names: Sequence[str]) -> List[Key]:
    keys: List[Key] = []
    for name in env_names:
        v = os.environ.get(name)
        if v:
            if "," in v:
                for i, key_value in enumerate(v.split(",")):
                    key_value = key_value.strip()
                    if key_value:
                        keys.append(Key(name=f"{name}_{i+1}", value=key_value))
            else:
                keys.append(Key(name=name, value=v))
    return keys


def _load_keys_from_file(keys_file: Path) -> List[Key]:
    keys: List[Key] = []
    for line in keys_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, val = line.split("=", 1)
            name = name.strip()
            val = val.strip()
        else:
            name = f"key{len(keys)+1}"
            val = line
        if val:
            keys.append(Key(name=name, value=val))
    return keys


def _dedupe_keys(keys: List[Key]) -> List[Key]:
    seen = set()
    uniq: List[Key] = []
    for k in keys:
        if k.value in seen:
            continue
        seen.add(k.value)
        uniq.append(k)
    return uniq


def _parse_groq_text(response: requests.Response, response_format: str) -> Tuple[str, object]:
    if response_format == "text":
        return (response.text or "").strip(), response.text
    data = response.json()
    text = data.get("text", "")
    return ("" if text is None else str(text)), data


def _transcribe_segment(
    *,
    segment_audio: Path,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
    response_format: str,
    timestamp_granularities: Sequence[str],
    timeout_s: float,
    pool: KeyPool,
    max_attempts: int,
    min_interval: float,
    jitter_s: float,
    last_request_t: List[float],
) -> Tuple[str, dict]:
    attempts = 0
    while True:
        attempts += 1
        if min_interval > 0:
            now = time.time()
            wait = (last_request_t[0] + min_interval) - now
            if wait > 0:
                time.sleep(wait + random.uniform(0, jitter_s))
        key = pool.get()
        try:
            r = groq_transcribe_requests(
                audio_path=segment_audio,
                api_key=key.value,
                model=model,
                language=language,
                prompt=prompt,
                temperature=temperature,
                response_format=response_format,
                timestamp_granularities=timestamp_granularities,
                timeout_s=timeout_s,
            )
            last_request_t[0] = time.time()
            if r.status_code == 429:
                retry_after = r.headers.get("retry-after")
                ra_s = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
                pool.cooldown(key.name, retry_after=ra_s)
                if attempts < max_attempts:
                    continue
                r.raise_for_status()
            if 500 <= r.status_code < 600:
                pool.cooldown(key.name, retry_after=10)
                if attempts < max_attempts:
                    continue
                r.raise_for_status()
            r.raise_for_status()
            text, payload = _parse_groq_text(r, response_format)
            meta = {
                "api_key_name": key.name,
                "status_code": r.status_code,
                "response_headers": dict(r.headers),
            }
            return text, {"response": payload, "meta": meta}
        except Exception:
            pool.cooldown(key.name, retry_after=15)
            if attempts < max_attempts:
                continue
            raise


def _build_segment_refs(segments: List[dict]) -> List[SegmentRef]:
    refs: List[SegmentRef] = []
    for idx, seg in enumerate(segments):
        times = _segment_times(seg)
        if not times:
            continue
        start, end = times
        refs.append(
            SegmentRef(
                idx=idx,
                seg_id=_segment_id(seg, idx),
                start=start,
                end=end,
                text=_segment_text(seg),
            )
        )
    return refs


def _align_segments(
    orig_refs: List[SegmentRef],
    patched_refs: List[SegmentRef],
) -> Tuple[List[Tuple[int, SegmentRef, SegmentRef]], List[int], List[int]]:
    if len(orig_refs) == len(patched_refs):
        pairs = [(i, o, p) for i, (o, p) in enumerate(zip(orig_refs, patched_refs))]
        return pairs, [], []
    orig_by_id = {r.seg_id: r for r in orig_refs}
    pat_by_id = {r.seg_id: r for r in patched_refs}
    shared_ids = sorted(set(orig_by_id).intersection(pat_by_id))
    pairs = [(sid, orig_by_id[sid], pat_by_id[sid]) for sid in shared_ids]
    missing_orig = sorted(set(pat_by_id).difference(orig_by_id))
    missing_pat = sorted(set(orig_by_id).difference(pat_by_id))
    return pairs, missing_orig, missing_pat


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _maybe_load_cached(cache_path: Path, start: float, end: float, request_sig: Optional[dict]) -> Optional[dict]:
    if not cache_path.exists():
        return None
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    seg = data.get("segment") if isinstance(data, dict) else None
    if not isinstance(seg, dict):
        return None
    if abs(float(seg.get("start", -1)) - start) > 1e-6 or abs(float(seg.get("end", -1)) - end) > 1e-6:
        return None
    if request_sig:
        req = data.get("request")
        if isinstance(req, dict) and req != request_sig:
            return None
    return data


def _compare_texts(ref_text: str, hyp_text: str) -> Dict[str, object]:
    norm_ref = _normalize_text(ref_text)
    norm_hyp = _normalize_text(hyp_text)
    return {
        "norm_match": norm_ref == norm_hyp,
        "wer": _wer(ref_text, hyp_text),
        "cer": _cer(ref_text, hyp_text),
    }


def process_file_pair(
    *,
    orig_json: Path,
    patched_json: Path,
    audio_path: Path,
    out_dir: Path,
    ffmpeg_bin: str,
    sample_rate: int,
    channels: int,
    accurate_seek: bool,
    pad_start: float,
    pad_end: float,
    min_duration: float,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
    response_format: str,
    timestamp_granularities: Sequence[str],
    timeout_s: float,
    pool: KeyPool,
    max_attempts: int,
    min_interval: float,
    jitter_s: float,
    keep_segments: bool,
    dry_run: bool,
) -> dict:
    orig_data = _load_json(orig_json)
    patched_data = _load_json(patched_json)

    orig_segments = _extract_segments(orig_data)
    patched_segments = _extract_segments(patched_data)

    orig_refs = _build_segment_refs(orig_segments)
    pat_refs = _build_segment_refs(patched_segments)

    pairs, missing_orig, missing_pat = _align_segments(orig_refs, pat_refs)

    file_out_dir = out_dir / _safe_name(orig_json.stem)
    file_out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = file_out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    segment_dir = file_out_dir / "segments"
    if keep_segments:
        segment_dir.mkdir(parents=True, exist_ok=True)

    last_request_t = [0.0]

    results: List[dict] = []

    for idx, orig_ref, pat_ref in pairs:
        seg_key = f"{orig_ref.seg_id}_{idx}"

        def _run_one(ref: SegmentRef, tag: str) -> dict:
            start = max(0.0, ref.start - pad_start)
            end = max(start, ref.end + pad_end)
            if (end - start) < min_duration:
                end = start + min_duration
            request_sig = {
                "model": model,
                "language": language,
                "prompt": prompt,
                "temperature": temperature,
                "response_format": response_format,
                "timestamp_granularities": list(timestamp_granularities),
            }
            cache_path = cache_dir / f"{_safe_name(seg_key)}__{tag}.json"
            cached = _maybe_load_cached(cache_path, start, end, request_sig)
            if cached:
                return cached

            if dry_run:
                return {
                    "segment": {"start": start, "end": end, "text": ref.text},
                    "groq_text": "",
                    "compare": {},
                    "status": "dry_run",
                }

            with tempfile.TemporaryDirectory(prefix="nfa_seg_") as tmpdir:
                tmpdir_path = Path(tmpdir)
                seg_name = f"{_safe_name(seg_key)}__{tag}.wav"
                out_seg = (segment_dir if keep_segments else tmpdir_path) / seg_name
                _ffmpeg_cut_segment(
                    ffmpeg_bin=ffmpeg_bin,
                    audio_path=audio_path,
                    start_s=start,
                    end_s=end,
                    out_path=out_seg,
                    sample_rate=sample_rate,
                    channels=channels,
                    accurate_seek=accurate_seek,
                )
                groq_text, payload = _transcribe_segment(
                    segment_audio=out_seg,
                    model=model,
                    language=language,
                    prompt=prompt,
                    temperature=temperature,
                    response_format=response_format,
                    timestamp_granularities=timestamp_granularities,
                    timeout_s=timeout_s,
                    pool=pool,
                    max_attempts=max_attempts,
                    min_interval=min_interval,
                    jitter_s=jitter_s,
                    last_request_t=last_request_t,
                )
                compare = _compare_texts(ref.text, groq_text)
                data = {
                    "segment": {"start": start, "end": end, "text": ref.text},
                    "request": request_sig,
                    "groq_text": groq_text,
                    "compare": compare,
                    "groq_payload": payload,
                    "status": "ok",
                }
                _write_json(cache_path, data)
                return data

        try:
            orig_out = _run_one(orig_ref, "original")
        except Exception as exc:
            orig_out = {
                "segment": {"start": orig_ref.start, "end": orig_ref.end, "text": orig_ref.text},
                "groq_text": "",
                "compare": {},
                "status": f"error: {exc}",
            }

        try:
            pat_out = _run_one(pat_ref, "patched")
        except Exception as exc:
            pat_out = {
                "segment": {"start": pat_ref.start, "end": pat_ref.end, "text": pat_ref.text},
                "groq_text": "",
                "compare": {},
                "status": f"error: {exc}",
            }

        wer_orig = orig_out.get("compare", {}).get("wer")
        wer_pat = pat_out.get("compare", {}).get("wer")
        better = "tie"
        delta_wer = None
        if isinstance(wer_orig, (int, float)) and isinstance(wer_pat, (int, float)):
            delta_wer = wer_orig - wer_pat
            if wer_pat + 1e-9 < wer_orig:
                better = "patched"
            elif wer_orig + 1e-9 < wer_pat:
                better = "original"

        results.append(
            {
                "segment_index": idx,
                "segment_id": orig_ref.seg_id,
                "original": orig_out,
                "patched": pat_out,
                "better": better,
                "delta_wer": delta_wer,
            }
        )

    # summary
    patched_better = sum(1 for r in results if r.get("better") == "patched")
    original_better = sum(1 for r in results if r.get("better") == "original")
    ties = sum(1 for r in results if r.get("better") == "tie")

    wer_orig_vals = [
        r["original"]["compare"]["wer"]
        for r in results
        if isinstance(r.get("original", {}).get("compare", {}).get("wer"), (int, float))
    ]
    wer_pat_vals = [
        r["patched"]["compare"]["wer"]
        for r in results
        if isinstance(r.get("patched", {}).get("compare", {}).get("wer"), (int, float))
    ]

    def _avg(vals: List[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "file": orig_json.name,
        "audio": str(audio_path),
        "segments_total_original": len(orig_segments),
        "segments_total_patched": len(patched_segments),
        "segments_compared": len(results),
        "missing_in_original": missing_orig,
        "missing_in_patched": missing_pat,
        "patched_better": patched_better,
        "original_better": original_better,
        "ties": ties,
        "avg_wer_original": _avg(wer_orig_vals),
        "avg_wer_patched": _avg(wer_pat_vals),
    }

    report = {
        "summary": summary,
        "segments": results,
    }
    _write_json(file_out_dir / "verification_report.json", report)
    _write_json(file_out_dir / "verification_summary.json", summary)
    return summary


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Verify NFA-corrected timestamps via Groq segment re-transcription.")
    ap.add_argument("--patched_dir", type=Path, required=True, help="Folder with corrected JSONs")
    ap.add_argument("--original_dir", type=Path, required=True, help="Folder with original JSONs")
    ap.add_argument("--audio_dir", type=Path, required=True, help="Folder with audio files")
    ap.add_argument("--out_dir", type=Path, default=Path("nfa_verification_out"))

    ap.add_argument("--file_glob", default="*.json", help="Glob to select JSON files in patched_dir")
    ap.add_argument("--limit_files", type=int, default=0, help="Limit number of files (0 = no limit)")

    ap.add_argument("--model", default=None, help="Groq Whisper model (default: from JSON or whisper-large-v3)")
    ap.add_argument("--language", default=None, help="Language code (default: from JSON or None)")
    ap.add_argument("--prompt", default=None, help="Prompt (optional)")
    ap.add_argument("--temperature", type=float, default=None, help="Temperature (default: from JSON or 0)")
    ap.add_argument("--response_format", default="json", choices=["json", "verbose_json", "text"])
    ap.add_argument("--timestamp_granularities", nargs="*", default=[], help="e.g. word segment")
    ap.add_argument("--timeout_s", type=float, default=float(os.getenv("GROQ_TIMEOUT_S", "300.0")))

    ap.add_argument("--api_key", default=None, help="Direct Groq API key (not recommended)")
    ap.add_argument("--key_env_names", nargs="*", default=["GROQ_API_KEY", "GROQ_API_KEYS"])
    ap.add_argument("--keys_file", type=Path, default=None)

    ap.add_argument("--max_rpm", type=float, default=float(os.getenv("GROQ_MAX_RPM", "18.0")))
    ap.add_argument("--jitter_s", type=float, default=float(os.getenv("GROQ_JITTER_S", "0.25")))
    ap.add_argument("--max_attempts", type=int, default=5)

    ap.add_argument("--pad_start", type=float, default=0.0, help="Seconds to extend segment start earlier")
    ap.add_argument("--pad_end", type=float, default=0.0, help="Seconds to extend segment end later")
    ap.add_argument("--min_duration", type=float, default=0.2, help="Minimum segment duration in seconds")

    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--accurate_seek", action="store_true", default=True)
    ap.add_argument("--fast_seek", action="store_false", dest="accurate_seek")

    ap.add_argument("--keep_segments", action="store_true", default=False)
    ap.add_argument("--dry_run", action="store_true", default=False)

    ap.add_argument("--audio_recursive", action="store_true", default=True)
    ap.add_argument("--no_audio_recursive", action="store_false", dest="audio_recursive")

    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    if load_dotenv:
        load_dotenv()

    if args.timestamp_granularities and args.response_format != "verbose_json":
        print("NOTE: timestamp_granularities requires response_format=verbose_json; switching.")
        args.response_format = "verbose_json"

    if not args.patched_dir.exists():
        raise SystemExit(f"Patched dir not found: {args.patched_dir}")
    if not args.original_dir.exists():
        raise SystemExit(f"Original dir not found: {args.original_dir}")
    if not args.audio_dir.exists():
        raise SystemExit(f"Audio dir not found: {args.audio_dir}")

    # Build audio index
    audio_files = _find_audio_files(args.audio_dir, args.audio_recursive)
    if not audio_files:
        raise SystemExit(f"No audio files found under: {args.audio_dir}")
    audio_index = _index_audio_by_stem(audio_files)

    # Gather JSON pairs
    patched_jsons = sorted(args.patched_dir.glob(args.file_glob))
    if args.limit_files and args.limit_files > 0:
        patched_jsons = patched_jsons[: args.limit_files]
    if not patched_jsons:
        raise SystemExit(f"No JSON files matched in {args.patched_dir} with glob {args.file_glob}")

    # Groq keys
    keys: List[Key] = []
    if args.api_key:
        keys.append(Key("api_key", args.api_key))
    if args.keys_file:
        keys.extend(_load_keys_from_file(args.keys_file))
    keys.extend(_load_keys_from_env_names(args.key_env_names))
    keys = _dedupe_keys(keys)
    if not keys:
        raise SystemExit("No Groq API keys found. Set GROQ_API_KEY or pass --keys_file / --api_key.")
    pool = KeyPool(keys)

    ffmpeg_bin = which_or_die("ffmpeg")

    min_interval = 60.0 / args.max_rpm if args.max_rpm and args.max_rpm > 0 else 0.0

    summaries: List[dict] = []
    missing_audio: List[str] = []
    missing_original: List[str] = []

    for idx, patched_json in enumerate(patched_jsons, start=1):
        original_json = args.original_dir / patched_json.name
        if not original_json.exists():
            missing_original.append(str(patched_json))
            print(f"[{idx}/{len(patched_jsons)}] Missing original JSON for {patched_json.name}")
            continue
        audio_path = _pick_audio_for_json(patched_json, audio_index)
        if not audio_path:
            missing_audio.append(patched_json.name)
            print(f"[{idx}/{len(patched_jsons)}] Missing audio for {patched_json.stem}")
            continue

        # derive defaults from JSON if not provided
        try:
            patched_data = _load_json(patched_json)
            req = _extract_request(patched_data)
        except Exception:
            req = {}
        model = args.model or req.get("model") or "whisper-large-v3"
        language = args.language if args.language is not None else req.get("language")
        prompt = args.prompt if args.prompt is not None else req.get("prompt")
        temperature = args.temperature if args.temperature is not None else float(req.get("temperature", 0.0))

        print(f"[{idx}/{len(patched_jsons)}] Processing {patched_json.name}")
        summary = process_file_pair(
            orig_json=original_json,
            patched_json=patched_json,
            audio_path=audio_path,
            out_dir=args.out_dir,
            ffmpeg_bin=ffmpeg_bin,
            sample_rate=args.sample_rate,
            channels=args.channels,
            accurate_seek=args.accurate_seek,
            pad_start=args.pad_start,
            pad_end=args.pad_end,
            min_duration=args.min_duration,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=args.response_format,
            timestamp_granularities=args.timestamp_granularities,
            timeout_s=args.timeout_s,
            pool=pool,
            max_attempts=args.max_attempts,
            min_interval=min_interval,
            jitter_s=args.jitter_s,
            keep_segments=args.keep_segments,
            dry_run=args.dry_run,
        )
        summaries.append(summary)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "patched_dir": str(args.patched_dir),
        "original_dir": str(args.original_dir),
        "audio_dir": str(args.audio_dir),
        "out_dir": str(args.out_dir),
        "missing_original": missing_original,
        "missing_audio": missing_audio,
        "files": summaries,
    }

    _write_json(args.out_dir / "verification_summary_all.json", report)
    print(f"Done. Summary written to {args.out_dir / 'verification_summary_all.json'}")


if __name__ == "__main__":
    main()
