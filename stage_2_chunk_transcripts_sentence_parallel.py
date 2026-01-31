"""Stage 2b: Cut audio chunks from tasks_pending.jsonl, with optional stereo split + drop-dupe.

Your existing pipeline already precomputes cut-jobs in:
  I:\Record_chunks\tasks_pending.jsonl
Each JSONL line looks like:
  {"audio_path": "...mp3", "out_wav": "...chunk0000.wav", "core_start": 0.0, "core_end": 1.88, "target_out_sec": 2.28}

This script:
  1) Reads tasks_pending.jsonl (cut jobs)
  2) Cuts audio with ffmpeg
  3) If source is stereo and policy demands it:
       - write L/R as separate files when channels differ
       - drop one channel when near-identical
  4) Writes an UPDATED pairs manifest (pairs_pending-like JSONL) where each chunk can become:
       - 1 row (mono or duplicate stereo)
       - 2 rows (true stereo: L + R)

Resumable:
  - Tracks done jobs in a state JSON
  - Optionally rewrites a remaining-tasks file


  usage

I:\Whisper-training-env\Scripts\python.exe i:\whisper-acft\stage_2_chunk_transcripts_sentence_parallel.py ^
  --tasks_pending_path "I:\Record_chunks\tasks_pending.jsonl" ^
  --pairs_pending_path "I:\Record_chunks\pairs_pending.jsonl" ^
  --out_pairs_path "I:\Record_chunks\pairs_manifest_stereo.jsonl" ^
  --ffmpeg_workers 4 --task_workers 4 --ffmpeg_threads 1 ^
  --stereo_policy split_drop_dupes



Windows note:
  - Paths are treated case-insensitively.

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional: numpy speeds up correlation math
try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


# -------------------------
# Small utilities
# -------------------------

def _bell() -> None:
    try:
        if sys.platform.startswith("win"):
            import winsound  # type: ignore

            winsound.MessageBeep(winsound.MB_OK)
        else:
            sys.stdout.write("\a")
            sys.stdout.flush()
    except Exception:
        pass


def json_dumps(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=False).encode("utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="ignore"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(tmp), str(path))


def norm_path(p: str) -> str:
    # Normalise for Windows case-insensitive comparisons
    return str(p).replace("/", "\\").strip().lower()


def with_suffix_before_ext(path_str: str, suffix: str) -> str:
    p = Path(path_str)
    if p.suffix.lower() != ".wav":
        # Still append safely
        return str(p) + suffix
    return str(p.with_name(p.stem + suffix + p.suffix))


def sha1_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


# -------------------------
# Data models
# -------------------------

@dataclass
class CutTask:
    audio_path: str
    out_wav: str
    core_start: float
    core_end: float
    target_out_sec: float


@dataclass
class TaskResult:
    ok: bool
    key: str
    wrote: int
    emitted_rows: int
    error: Optional[str] = None


# -------------------------
# Reading inputs
# -------------------------

def load_tasks_pending(tasks_path: Path) -> List[CutTask]:
    tasks: List[CutTask] = []
    with tasks_path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line.decode("utf-8", errors="ignore"))
            if not isinstance(obj, dict):
                continue
            ap = obj.get("audio_path")
            ow = obj.get("out_wav")
            cs = obj.get("core_start")
            ce = obj.get("core_end")
            to = obj.get("target_out_sec")
            if not (isinstance(ap, str) and isinstance(ow, str)):
                continue
            try:
                tasks.append(CutTask(ap, ow, float(cs), float(ce), float(to)))
            except Exception:
                continue
    return tasks


def load_pairs_pending(pairs_path: Path) -> Dict[str, Dict[str, Any]]:
    """Map audio_path/out_wav -> base row dict."""
    m: Dict[str, Dict[str, Any]] = {}
    if not pairs_path.exists():
        return m

    with pairs_path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8", errors="ignore"))
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            ap = obj.get("audio_path")
            if isinstance(ap, str) and ap:
                m[norm_path(ap)] = obj
            ow = obj.get("out_wav")
            if isinstance(ow, str) and ow:
                m[norm_path(ow)] = obj
    return m


# -------------------------
# Cut window computation
# -------------------------

def compute_cut_window(task: CutTask) -> Tuple[float, float]:
    """Compute actual cut start/end from (core_start, core_end, target_out_sec).

    Old pipeline pattern strongly suggests:
      target_out_sec = (core_end-core_start) + 0.4  (i.e., ~0.2s pad each side)

    We centre padding around the core region where possible, clamping at 0.
    """
    core_dur = max(0.0, task.core_end - task.core_start)
    target = max(core_dur, float(task.target_out_sec))
    pad_total = max(0.0, target - core_dur)
    pad_each = pad_total / 2.0

    start = max(0.0, task.core_start - pad_each)
    end = start + target

    # Ensure core region is inside the cut; if start was clamped, extend end if needed
    if end < task.core_end:
        end = task.core_end
    return float(start), float(end)


# -------------------------
# FFprobe + stereo similarity
# -------------------------

def probe_audio_channels(ffprobe_path: str, src_audio: str) -> int:
    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=channels",
        "-of",
        "default=nw=1:nk=1",
        src_audio,
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffprobe failed ({r.returncode}): {msg[:400]}")
    out = r.stdout.decode("utf-8", errors="ignore").strip()
    return int(out)


def decode_channel_snippet_f32le(
    ffmpeg_path: str,
    src_audio: str,
    channel_index: int,
    sec: float,
    sample_rate: int,
) -> "np.ndarray | List[float]":
    sec = max(0.2, float(sec))
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        "0",
        "-t",
        f"{sec:.3f}",
        "-i",
        src_audio,
        "-vn",
        "-map_channel",
        f"0.0.{int(channel_index)}",
        "-ar",
        str(int(sample_rate)),
        "-ac",
        "1",
        "-f",
        "f32le",
        "pipe:1",
    ]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg snippet decode failed ({r.returncode}): {msg[:400]}")
    b = r.stdout
    if not b:
        return np.array([], dtype=np.float32) if np is not None else []

    if np is not None:
        return np.frombuffer(b, dtype=np.float32)

    # Fallback without numpy
    import array

    a = array.array("f")
    a.frombytes(b)
    return list(a)


def rms_vec(x: "np.ndarray | List[float]") -> float:
    if np is not None and isinstance(x, np.ndarray):
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(x * x)))
    if not x:
        return 0.0
    s = 0.0
    for v in x:  # type: ignore
        s += float(v) * float(v)
    return math.sqrt(s / max(1, len(x)))


def max_abs_corr_with_lag(x: "np.ndarray | List[float]", y: "np.ndarray | List[float]", max_lag_samples: int) -> float:
    # If numpy is available, do a proper lagged corr. Otherwise, return 0 (we’ll rely on diff RMS).
    if np is None or not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        return 0.0

    n = int(min(x.size, y.size))
    if n < 400:
        return 0.0

    xa = x[:n].astype(np.float32, copy=False)
    ya = y[:n].astype(np.float32, copy=False)
    xa = xa - xa.mean()
    ya = ya - ya.mean()

    L = int(max(0, max_lag_samples))
    maxc = 0.0

    for lag in range(-L, L + 1):
        if lag < 0:
            a = xa[-lag:]
            b = ya[: n + lag]
        elif lag > 0:
            a = xa[: n - lag]
            b = ya[lag:]
        else:
            a = xa
            b = ya

        if a.size < 400:
            continue

        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        c = float(np.dot(a, b) / denom)
        if abs(c) > maxc:
            maxc = abs(c)

    return float(maxc)


def stereo_plan_for_audio(
    ffprobe_path: str,
    ffmpeg_path: str,
    src_audio: str,
    check_sec: float,
    corr_thresh: float,
    diff_rms_ratio_thresh: float,
    lag_ms: float,
    sample_rate: int,
) -> Dict[str, Any]:
    ch = probe_audio_channels(ffprobe_path, src_audio)
    plan: Dict[str, Any] = {
        "source_channels": int(ch),
        "keep": ["M"],
        "duplicate": False,
        "dropped": None,
        "reason": "mono_or_downmix",
    }

    if ch < 2:
        plan["reason"] = "mono"
        return plan

    # Decode snippet L/R
    left = decode_channel_snippet_f32le(ffmpeg_path, src_audio, 0, check_sec, sample_rate)
    right = decode_channel_snippet_f32le(ffmpeg_path, src_audio, 1, check_sec, sample_rate)

    rmsL = rms_vec(left)
    rmsR = rms_vec(right)
    eps = 1e-4

    if rmsL < eps and rmsR < eps:
        plan.update({"keep": ["L"], "duplicate": True, "dropped": "R", "reason": "both_silent"})
        return plan
    if rmsL < eps and rmsR >= eps:
        plan.update({"keep": ["R"], "duplicate": False, "dropped": "L", "reason": "left_silent"})
        return plan
    if rmsR < eps and rmsL >= eps:
        plan.update({"keep": ["L"], "duplicate": False, "dropped": "R", "reason": "right_silent"})
        return plan

    # Diff RMS ratio
    if np is not None and isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        n = int(min(left.size, right.size))
        diff = left[:n] - right[:n]
        diff_rms = float(np.sqrt(np.mean(diff * diff)))
    else:
        n = int(min(len(left), len(right)))  # type: ignore
        diff_rms = rms_vec([float(left[i]) - float(right[i]) for i in range(n)])  # type: ignore

    denom = max(rmsL, rmsR) + 1e-12
    diff_ratio = float(diff_rms / denom)

    # Lagged corr (optional)
    max_lag_samples = int(max(0.0, lag_ms) * (sample_rate / 1000.0))
    corr = max_abs_corr_with_lag(left, right, max_lag_samples)

    if (corr >= corr_thresh or corr == 0.0) and diff_ratio <= diff_rms_ratio_thresh:
        plan.update({"keep": ["L"], "duplicate": True, "dropped": "R", "reason": "near_identical"})
    else:
        plan.update({"keep": ["L", "R"], "duplicate": False, "dropped": None, "reason": "different"})

    plan.update({
        "corr": float(corr),
        "diff_rms_ratio": float(diff_ratio),
        "rmsL": float(rmsL),
        "rmsR": float(rmsR),
    })
    return plan


def get_stereo_plan_cached(
    src_audio: str,
    args: argparse.Namespace,
    stereo_cache: Dict[str, Any],
    stereo_lock: threading.Lock,
    ffmpeg_sem: threading.Semaphore,
) -> Dict[str, Any]:
    key = norm_path(src_audio)
    with stereo_lock:
        cached = stereo_cache.get(key)
    if isinstance(cached, dict) and "source_channels" in cached:
        # Apply policy override only
        return apply_stereo_policy(dict(cached), str(args.stereo_policy))

    ffmpeg_sem.acquire()
    try:
        plan = stereo_plan_for_audio(
            ffprobe_path=args.ffprobe,
            ffmpeg_path=args.ffmpeg,
            src_audio=src_audio,
            check_sec=float(args.stereo_check_sec),
            corr_thresh=float(args.stereo_corr_thresh),
            diff_rms_ratio_thresh=float(args.stereo_diff_rms_ratio_thresh),
            lag_ms=float(args.stereo_lag_ms),
            sample_rate=int(args.sample_rate),
        )
    finally:
        ffmpeg_sem.release()

    with stereo_lock:
        stereo_cache[key] = plan

    return apply_stereo_policy(dict(plan), str(args.stereo_policy))


def apply_stereo_policy(plan: Dict[str, Any], policy: str) -> Dict[str, Any]:
    ch = int(plan.get("source_channels", 1) or 1)
    if policy == "none":
        plan["keep"] = ["M"]
        plan["reason"] = str(plan.get("reason", "")) + "|policy_none"
    elif policy == "split":
        if ch >= 2:
            plan["keep"] = ["L", "R"]
            plan["duplicate"] = False
            plan["dropped"] = None
            plan["reason"] = str(plan.get("reason", "")) + "|policy_split"
    # split_drop_dupes uses computed keep
    return plan


# -------------------------
# FFmpeg cutting
# -------------------------

def _exists_ok(p: Path, min_bytes: int = 1024) -> bool:
    try:
        return p.exists() and p.stat().st_size >= min_bytes
    except Exception:
        return False


def ffmpeg_cut_mono(
    ffmpeg_path: str,
    src_audio: str,
    out_wav: str,
    start: float,
    end: float,
    sample_rate: int,
    ffmpeg_threads: int,
    overwrite: bool,
    fast_seek: bool,
) -> None:
    dur = max(0.01, end - start)
    outp = Path(out_wav)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [ffmpeg_path, "-hide_banner", "-loglevel", "error"]
    cmd.append("-y" if overwrite else "-n")
    if ffmpeg_threads and ffmpeg_threads > 0:
        cmd += ["-threads", str(ffmpeg_threads)]

    if fast_seek:
        cmd += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src_audio]
    else:
        cmd += ["-i", src_audio, "-ss", f"{start:.3f}", "-t", f"{dur:.3f}"]

    cmd += [
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        out_wav,
    ]

    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg cut mono failed ({r.returncode}): {msg[:800]}")


def ffmpeg_cut_channel(
    ffmpeg_path: str,
    src_audio: str,
    out_wav: str,
    start: float,
    end: float,
    channel_index: int,
    sample_rate: int,
    ffmpeg_threads: int,
    overwrite: bool,
    fast_seek: bool,
) -> None:
    dur = max(0.01, end - start)
    outp = Path(out_wav)
    outp.parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [ffmpeg_path, "-hide_banner", "-loglevel", "error"]
    cmd.append("-y" if overwrite else "-n")
    if ffmpeg_threads and ffmpeg_threads > 0:
        cmd += ["-threads", str(ffmpeg_threads)]

    if fast_seek:
        cmd += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src_audio]
    else:
        cmd += ["-i", src_audio, "-ss", f"{start:.3f}", "-t", f"{dur:.3f}"]

    cmd += [
        "-vn",
        "-map_channel",
        f"0.0.{int(channel_index)}",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        out_wav,
    ]

    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg cut channel failed ({r.returncode}): {msg[:800]}")


def ffmpeg_cut_lr_pair(
    ffmpeg_path: str,
    src_audio: str,
    out_left: str,
    out_right: str,
    start: float,
    end: float,
    sample_rate: int,
    ffmpeg_threads: int,
    overwrite: bool,
    fast_seek: bool,
) -> None:
    dur = max(0.01, end - start)
    Path(out_left).parent.mkdir(parents=True, exist_ok=True)
    Path(out_right).parent.mkdir(parents=True, exist_ok=True)

    cmd: List[str] = [ffmpeg_path, "-hide_banner", "-loglevel", "error"]
    cmd.append("-y" if overwrite else "-n")
    if ffmpeg_threads and ffmpeg_threads > 0:
        cmd += ["-threads", str(ffmpeg_threads)]

    if fast_seek:
        cmd += ["-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", src_audio]
    else:
        cmd += ["-i", src_audio, "-ss", f"{start:.3f}", "-t", f"{dur:.3f}"]

    # Left output
    cmd += [
        "-vn",
        "-map_channel",
        "0.0.0",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        out_left,
    ]

    # Right output
    cmd += [
        "-vn",
        "-map_channel",
        "0.0.1",
        "-ac",
        "1",
        "-ar",
        str(int(sample_rate)),
        "-c:a",
        "pcm_s16le",
        out_right,
    ]

    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        msg = r.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg cut L/R failed ({r.returncode}): {msg[:800]}")


# -------------------------
# Writer thread
# -------------------------

def writer_thread_main(q: "queue.Queue[Optional[bytes]]", out_path: Path, flush_every: int = 2000) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("ab") as f:
        while True:
            item = q.get()
            if item is None:
                break
            f.write(item)
            f.write(b"\n")
            count += 1
            if count % flush_every == 0:
                f.flush()


# -------------------------
# Task worker
# -------------------------

def task_key(task: CutTask) -> str:
    # Unique key per original chunk job
    return sha1_short(norm_path(task.audio_path) + "|" + norm_path(task.out_wav) + f"|{task.core_start:.3f}|{task.core_end:.3f}|{task.target_out_sec:.3f}", 16)


def process_one_task(
    task: CutTask,
    args: argparse.Namespace,
    pairs_map: Dict[str, Dict[str, Any]],
    ffmpeg_sem: threading.Semaphore,
    stereo_cache: Dict[str, Any],
    stereo_lock: threading.Lock,
    done_set: set,
    done_lock: threading.Lock,
) -> Tuple[TaskResult, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    key = task_key(task)

    with done_lock:
        if key in done_set:
            return TaskResult(True, key, 0, 0, None), rows

    try:
        src = task.audio_path
        out = task.out_wav

        # Compute cut window
        start, end = compute_cut_window(task)

        # Decide channel plan
        plan = get_stereo_plan_cached(src, args, stereo_cache, stereo_lock, ffmpeg_sem)
        source_channels = int(plan.get("source_channels", 1) or 1)
        keep = list(plan.get("keep", ["M"]))

        base_pair = pairs_map.get(norm_path(out)) or pairs_map.get(norm_path(with_suffix_before_ext(out, "")))

        # Base row template (pairs_pending-compatible)
        def base_row_for(audio_path: str, channel: str) -> Dict[str, Any]:
            if isinstance(base_pair, dict):
                r = dict(base_pair)
            else:
                r = {
                    "raw_transcription": "",
                    "source_audio": src,
                    "chunk_index": None,
                    "transcript_json": None,
                    "sr": int(args.sample_rate),
                    "model": "",
                }
            r["audio_path"] = audio_path
            r["source_audio"] = src
            r["duration_sec_target"] = float(task.target_out_sec)
            r["sr"] = int(args.sample_rate)
            r["channel"] = channel
            r["source_channels"] = source_channels
            r["core_start"] = float(task.core_start)
            r["core_end"] = float(task.core_end)
            r["chunk_start"] = float(start)
            r["chunk_end"] = float(end)
            r["stereo_policy"] = str(args.stereo_policy)
            r["stereo_duplicate"] = bool(plan.get("duplicate", False))
            r["stereo_dropped_channel"] = plan.get("dropped")
            r["stereo_reason"] = plan.get("reason")
            r["stereo_corr"] = plan.get("corr")
            r["stereo_diff_rms_ratio"] = plan.get("diff_rms_ratio")
            return r

        wrote = 0

        # Completion checks
        outp = Path(out)
        left_out = with_suffix_before_ext(out, "__L")
        right_out = with_suffix_before_ext(out, "__R")

        # Decide what to cut and where
        if keep == ["M"] or source_channels < 2:
            # Classic behaviour: downmix to mono
            if int(args.skip_existing) == 1 and _exists_ok(outp):
                pass
            else:
                ffmpeg_sem.acquire()
                try:
                    ffmpeg_cut_mono(
                        ffmpeg_path=args.ffmpeg,
                        src_audio=src,
                        out_wav=out,
                        start=start,
                        end=end,
                        sample_rate=int(args.sample_rate),
                        ffmpeg_threads=int(args.ffmpeg_threads),
                        overwrite=(int(args.skip_existing) == 0),
                        fast_seek=bool(int(args.fast_seek)),
                    )
                finally:
                    ffmpeg_sem.release()
                wrote += 1

            rows.append(base_row_for(out, "M"))

        else:
            keep_set = set(keep)
            keep_L = "L" in keep_set
            keep_R = "R" in keep_set

            # If near-identical -> keep only LEFT in the ORIGINAL filename (keeps pipeline expectations sane)
            if keep_L and (not keep_R):
                if int(args.skip_existing) == 1 and _exists_ok(outp):
                    pass
                else:
                    ffmpeg_sem.acquire()
                    try:
                        ffmpeg_cut_channel(
                            ffmpeg_path=args.ffmpeg,
                            src_audio=src,
                            out_wav=out,
                            start=start,
                            end=end,
                            channel_index=0,
                            sample_rate=int(args.sample_rate),
                            ffmpeg_threads=int(args.ffmpeg_threads),
                            overwrite=(int(args.skip_existing) == 0),
                            fast_seek=bool(int(args.fast_seek)),
                        )
                    finally:
                        ffmpeg_sem.release()
                    wrote += 1

                rows.append(base_row_for(out, "L"))

            elif keep_R and (not keep_L):
                # Rare (left silent): keep RIGHT in the original filename
                if int(args.skip_existing) == 1 and _exists_ok(outp):
                    pass
                else:
                    ffmpeg_sem.acquire()
                    try:
                        ffmpeg_cut_channel(
                            ffmpeg_path=args.ffmpeg,
                            src_audio=src,
                            out_wav=out,
                            start=start,
                            end=end,
                            channel_index=1,
                            sample_rate=int(args.sample_rate),
                            ffmpeg_threads=int(args.ffmpeg_threads),
                            overwrite=(int(args.skip_existing) == 0),
                            fast_seek=bool(int(args.fast_seek)),
                        )
                    finally:
                        ffmpeg_sem.release()
                    wrote += 1

                rows.append(base_row_for(out, "R"))

            else:
                # True stereo difference: write TWO files (L/R) and update manifest accordingly
                lp = Path(left_out)
                rp = Path(right_out)

                left_ok = int(args.skip_existing) == 1 and _exists_ok(lp)
                right_ok = int(args.skip_existing) == 1 and _exists_ok(rp)

                if left_ok and right_ok:
                    pass
                else:
                    ffmpeg_sem.acquire()
                    try:
                        # One call writes both
                        ffmpeg_cut_lr_pair(
                            ffmpeg_path=args.ffmpeg,
                            src_audio=src,
                            out_left=left_out,
                            out_right=right_out,
                            start=start,
                            end=end,
                            sample_rate=int(args.sample_rate),
                            ffmpeg_threads=int(args.ffmpeg_threads),
                            overwrite=(int(args.skip_existing) == 0),
                            fast_seek=bool(int(args.fast_seek)),
                        )
                    finally:
                        ffmpeg_sem.release()
                    wrote += 2

                rows.append(base_row_for(left_out, "L"))
                rows.append(base_row_for(right_out, "R"))

        with done_lock:
            done_set.add(key)

        return TaskResult(True, key, wrote, len(rows), None), rows

    except Exception as e:
        return TaskResult(False, key, 0, 0, str(e)), rows


# -------------------------
# State handling
# -------------------------

def load_done_state(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        obj = read_json(path)
        done = obj.get("done")
        if isinstance(done, list):
            return set(map(str, done))
    except Exception:
        pass
    return set()


def save_done_state(path: Path, done_set: set, extra: Dict[str, Any]) -> None:
    obj = {"done": sorted(done_set), **extra, "updated_at": time.time()}
    write_json(path, obj)


def write_remaining_tasks(path: Path, remaining: Iterable[CutTask]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        for t in remaining:
            f.write(json_dumps({
                "audio_path": t.audio_path,
                "out_wav": t.out_wav,
                "core_start": t.core_start,
                "core_end": t.core_end,
                "target_out_sec": t.target_out_sec,
            }))
            f.write(b"\n")
    os.replace(str(tmp), str(path))


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cut tasks_pending.jsonl into WAV chunks with stereo splitting")

    p.add_argument("--tasks_pending_path", type=str, required=True, help="Path to tasks_pending.jsonl")
    p.add_argument(
        "--pairs_pending_path",
        type=str,
        default="",
        help="Optional: existing pairs_pending.jsonl (to carry over raw_transcription etc.)",
    )
    p.add_argument(
        "--out_pairs_path",
        type=str,
        required=True,
        help="Output pairs manifest JSONL (updated for stereo splitting)",
    )
    p.add_argument(
        "--remaining_tasks_path",
        type=str,
        default="",
        help="Write remaining failed tasks JSONL here (default: alongside out_pairs_path)",
    )

    # Tools
    p.add_argument("--ffmpeg", type=str, default="ffmpeg", help="Path to ffmpeg")
    p.add_argument("--ffprobe", type=str, default="ffprobe", help="Path to ffprobe")

    # Cutting
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--ffmpeg_workers", type=int, default=8, help="Max concurrent ffmpeg/ffprobe operations")
    p.add_argument("--ffmpeg_threads", type=int, default=1, help="ffmpeg -threads value per process")
    p.add_argument("--task_workers", type=int, default=24, help="ThreadPool workers for scheduling tasks")
    p.add_argument("--skip_existing", type=int, default=1, help="1=skip already-cut outputs (size>1KB)")
    p.add_argument("--fast_seek", type=int, default=1, help="1=put -ss before -i (faster, less accurate)")

    # Stereo policy
    p.add_argument(
        "--stereo_policy",
        type=str,
        default="split_drop_dupes",
        choices=["none", "split", "split_drop_dupes"],
    )
    p.add_argument("--stereo_check_sec", type=float, default=8.0)
    p.add_argument("--stereo_corr_thresh", type=float, default=0.8)
    p.add_argument("--stereo_diff_rms_ratio_thresh", type=float, default=0.1)
    p.add_argument("--stereo_lag_ms", type=float, default=20.0)
    p.add_argument(
        "--stereo_cache_path",
        type=str,
        default="",
        help="Stereo plan cache JSON (default: out_pairs_path dir / stage2_stereo_cache.json)",
    )

    # State
    p.add_argument(
        "--state_path",
        type=str,
        default="",
        help="State JSON for done tasks (default: out_pairs_path dir / stage2_cut_state.json)",
    )
    p.add_argument("--save_every", type=int, default=250)

    return p.parse_args()


# -------------------------
# Main
# -------------------------

def main() -> int:
    args = parse_args()

    tasks_path = Path(args.tasks_pending_path)
    if not tasks_path.exists():
        print(f"ERROR: tasks_pending_path not found: {tasks_path}")
        return 2

    out_pairs = Path(args.out_pairs_path)
    out_dir = out_pairs.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    state_path = Path(args.state_path) if args.state_path else (out_dir / "stage2_cut_state.json")
    stereo_cache_path = Path(args.stereo_cache_path) if args.stereo_cache_path else (out_dir / "stage2_stereo_cache.json")

    remaining_tasks_path = Path(args.remaining_tasks_path) if args.remaining_tasks_path else (out_dir / "tasks_remaining.jsonl")

    # Load tasks
    tasks = load_tasks_pending(tasks_path)
    print(f"[cut] tasks loaded: {len(tasks):,} from {tasks_path}")

    # Load pairs metadata
    pairs_map: Dict[str, Dict[str, Any]] = {}
    if args.pairs_pending_path:
        pairs_map = load_pairs_pending(Path(args.pairs_pending_path))
        print(f"[cut] pairs loaded: {len(pairs_map):,} keys from {args.pairs_pending_path}")

    # State
    done_set = load_done_state(state_path)
    done_lock = threading.Lock()

    # Stereo cache
    stereo_cache: Dict[str, Any] = {}
    if stereo_cache_path.exists():
        try:
            stereo_cache = read_json(stereo_cache_path)
            if not isinstance(stereo_cache, dict):
                stereo_cache = {}
        except Exception:
            stereo_cache = {}
    stereo_lock = threading.Lock()

    ffmpeg_sem = threading.Semaphore(int(args.ffmpeg_workers))

    # Output writer
    # Truncate output if you want a clean rebuild; otherwise append.
    # Here: we rebuild from scratch unless you explicitly want append.
    out_pairs.write_bytes(b"")

    q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=50000)
    wt = threading.Thread(target=writer_thread_main, args=(q, out_pairs), daemon=True)
    wt.start()

    # Errors CSV
    errors_csv = out_dir / "stage2_cut_errors.csv"
    write_header = not errors_csv.exists()
    err_f = errors_csv.open("a", newline="", encoding="utf-8")
    err_w = csv.writer(err_f)
    if write_header:
        err_w.writerow(["task_key", "audio_path", "out_wav", "error"])

    # Filter tasks that are not done
    todo: List[CutTask] = []
    for t in tasks:
        k = task_key(t)
        if k not in done_set:
            todo.append(t)

    print(f"[cut] already done (state): {len(done_set):,}")
    print(f"[cut] to do now: {len(todo):,}")

    if not todo:
        print("[cut] Nothing to do.")
        q.put(None)
        wt.join()
        err_f.close()
        _bell()
        return 0

    from concurrent.futures import ThreadPoolExecutor, as_completed

    bar = tqdm(total=len(todo), desc="Cut tasks", unit="task") if tqdm is not None else None

    start_time = time.time()
    done_count = 0
    ok_count = 0
    wrote_total = 0
    rows_total = 0

    extra_state = {
        "tasks_pending_path": str(tasks_path),
        "pairs_pending_path": str(args.pairs_pending_path),
        "out_pairs_path": str(out_pairs),
        "stereo_policy": str(args.stereo_policy),
        "ffmpeg_workers": int(args.ffmpeg_workers),
        "task_workers": int(args.task_workers),
    }

    save_done_state(state_path, done_set, {**extra_state, "started_at": time.time()})
    write_json(stereo_cache_path, stereo_cache)

    with ThreadPoolExecutor(max_workers=int(args.task_workers)) as ex:
        futures = [
            ex.submit(
                process_one_task,
                t,
                args,
                pairs_map,
                ffmpeg_sem,
                stereo_cache,
                stereo_lock,
                done_set,
                done_lock,
            )
            for t in todo
        ]

        for fut in as_completed(futures):
            res, rows = fut.result()
            done_count += 1

            if res.ok:
                ok_count += 1
                wrote_total += res.wrote
                rows_total += len(rows)
                for r in rows:
                    q.put(json_dumps(r))
            else:
                err_w.writerow([res.key, "", "", res.error or "unknown_error"])
                err_f.flush()

            if bar is not None:
                bar.update(1)
                bar.set_postfix({"ok": ok_count, "wav": wrote_total, "rows": rows_total})

            if done_count % int(args.save_every) == 0:
                save_done_state(
                    state_path,
                    done_set,
                    {
                        **extra_state,
                        "done": done_count,
                        "ok": ok_count,
                        "wrote": wrote_total,
                        "rows": rows_total,
                        "elapsed_sec": time.time() - start_time,
                    },
                )
                # stereo cache can be large; still save periodically
                try:
                    write_json(stereo_cache_path, stereo_cache)
                except Exception:
                    pass

    if bar is not None:
        bar.close()

    # Finish writer
    q.put(None)
    wt.join()
    err_f.close()

    elapsed = time.time() - start_time

    # Remaining tasks (those not marked done)
    with done_lock:
        done_snapshot = set(done_set)

    remaining = [t for t in tasks if task_key(t) not in done_snapshot]
    write_remaining_tasks(remaining_tasks_path, remaining)

    save_done_state(
        state_path,
        done_set,
        {
            **extra_state,
            "done": done_count,
            "ok": ok_count,
            "wrote": wrote_total,
            "rows": rows_total,
            "elapsed_sec": elapsed,
            "finished_at": time.time(),
        },
    )

    print(
        f"[cut] finished ok={ok_count:,}/{done_count:,} | wav_written={wrote_total:,} | rows={rows_total:,} | elapsed={elapsed/60:.1f} min"
    )
    print(f"[cut] out pairs:   {out_pairs}")
    print(f"[cut] state:       {state_path}")
    print(f"[cut] stereo cache:{stereo_cache_path}")
    print(f"[cut] remaining:   {remaining_tasks_path}")
    print(f"[cut] errors:      {errors_csv}")

    _bell()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
