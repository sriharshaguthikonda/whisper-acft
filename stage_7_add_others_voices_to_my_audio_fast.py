#!/usr/bin/env python3
"""stage_7_add_others_voices_to_my_audio_FAST.py

You said: it should handle M4A files as well.

Key idea
--------
- `soundfile` (libsndfile) generally **cannot read .m4a**.
- So we keep `soundfile` for WAV/FLAC/OGG (fast), and **fall back to ffmpeg** for MP3/M4A/AAC/MP4/OPUS/etc.
- Best performance is still: **one-time preconvert OTHER voices to 16 kHz mono WAV** via ffmpeg.

Modes
-----
1) One-time OTHER cache (fastest + reliable for M4A):
   python stage_7_add_others_voices_to_my_audio_FAST.py --prepare_other_cache \
     --other_voices_dir "I:/Record_others" \
     --other_cache_dir "I:/Record_others_16k_wav" \
     --target_sr 16000 --workers 5

2) Mixing using the cached WAVs (recommended):
   python stage_7_add_others_voices_to_my_audio_FAST.py \
     --manifest "i:/Record_chunks/pairs_pending_stereo_english_only_filtered_with_mix.jsonl" \
     --scores_csv "i:/whisper-acft/speaker_sort_scores.csv" \
     --other_voices_dir "I:/Record_others_16k_wav" \
     --out_manifest "i:/Record_chunks/pairs_pending_stereo_english_only_filtered_with_others_voice_mix.jsonl" \
     --out_mix_dir "i:/Record_chunks_voice_mixed" \
     --mix_ratio 0.8 --snr_db_min 5 --snr_db_max 10 \
     --max_bad_to_good_ratio 1.0 --good_floor_db -45 \
     --seed 1337 --shuffle --workers 5 --pool thread

3) Mixing directly from a folder that contains M4A/MP3 etc:
   - works (ffmpeg fallback) but slower than cached WAVs.

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------

def _norm_path(p: str | Path) -> str:
    return str(Path(p))


def _norm_key(p: str | Path) -> str:
    return str(p).lower().replace("\\", "/")


def _stable_hex(s: str, n: int = 10) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _is_audio_file_any(p: Path) -> bool:
    # Anything ffmpeg can read (broad)
    return p.suffix.lower() in {
        ".wav",
        ".flac",
        ".ogg",
        ".mp3",
        ".m4a",
        ".aac",
        ".mp4",
        ".opus",
        ".wma",
    }


def _is_audio_file_soundfile_safe(p: Path) -> bool:
    # soundfile/libsndfile is reliably happy with these
    return p.suffix.lower() in {".wav", ".flac", ".ogg"}


def _rms(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))


def _ensure_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def _safe_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def _beep() -> None:
    try:
        import winsound

        winsound.MessageBeep()
    except Exception:
        pass


# -----------------------------
# ffmpeg / ffprobe
# -----------------------------

def _find_ffmpeg() -> Optional[str]:
    cand = shutil.which("ffmpeg")
    if cand:
        return cand
    for p in [
        r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]:
        if os.path.exists(p):
            return p
    return None


def _find_ffprobe() -> Optional[str]:
    cand = shutil.which("ffprobe")
    if cand:
        return cand
    for p in [
        r"C:\ProgramData\chocolatey\bin\ffprobe.exe",
        r"C:\ffmpeg\bin\ffprobe.exe",
    ]:
        if os.path.exists(p):
            return p
    return None


def _ffprobe_duration_sec(path: Path) -> Optional[float]:
    ffprobe = _find_ffprobe()
    if not ffprobe:
        return None
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, check=True)
        s = (p.stdout or "").strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def _ffmpeg_read_segment_f32le(path: Path, start_sec: float, dur_sec: float, target_sr: int) -> np.ndarray:
    """Decode a mono float32 segment using ffmpeg (works for M4A/MP3/etc).

    Returns: float32 mono array at target_sr.
    """
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH (or common locations)")

    # Use -ss before -i for speed (seeking), -t for duration.
    # Output raw float32 little-endian.
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{max(0.0, float(start_sec)):.6f}",
        "-t",
        f"{max(0.0, float(dur_sec)):.6f}",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(target_sr)),
        "-f",
        "f32le",
        "pipe:1",
    ]

    raw = subprocess.check_output(cmd)
    if not raw:
        return np.zeros((0,), dtype=np.float32)

    y = np.frombuffer(raw, dtype=np.float32)
    return y


def _ffmpeg_read_full_f32le(path: Path, target_sr: int) -> np.ndarray:
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH (or common locations)")
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(target_sr)),
        "-f",
        "f32le",
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    if not raw:
        return np.zeros((0,), dtype=np.float32)
    return np.frombuffer(raw, dtype=np.float32)


# -----------------------------
# Ducker
# -----------------------------

try:
    from audio_instant_ducker_utils import duck_bad_under_good as _duck_external
except Exception:
    _duck_external = None


def duck_bad_under_good(bad: np.ndarray, good: np.ndarray, max_ratio: float, good_floor_db: float) -> np.ndarray:
    """Ensures |bad| <= max_ratio * max(|good|, floor)."""
    if _duck_external is not None:
        return _duck_external(bad, good=good, max_ratio=max_ratio, good_floor_db=good_floor_db)

    bad = np.asarray(bad, dtype=np.float32)
    good = np.asarray(good, dtype=np.float32)

    floor = float(10.0 ** (float(good_floor_db) / 20.0))
    good_abs = np.abs(good)
    allowed = float(max_ratio) * np.maximum(good_abs, floor)
    return np.clip(bad, -allowed, allowed)


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class OtherVoice:
    path: Path
    score: float
    duration_sec: float
    use_ffmpeg: bool


# -----------------------------
# CSV + manifest helpers
# -----------------------------

def load_target_and_other_from_csv(scores_csv: Path) -> Tuple[set[str], List[OtherVoice]]:
    """Compatibility loader.

    We keep it, but we typically ignore the CSV OTHER list because your OTHER pool is the folder.
    """
    target_files: set[str] = set()
    others: List[OtherVoice] = []

    with scores_csv.open("r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 3:
                continue
            file_path, score_str, decision = parts[0], parts[1], parts[2]
            npath = _norm_key(file_path)

            if decision == "TARGET":
                target_files.add(npath)
                continue

            if decision == "OTHER":
                try:
                    score = float(score_str)
                except Exception:
                    continue
                if score > 0:
                    p = Path(file_path)
                    # duration unknown at this point; we fill later if we actually use this list
                    others.append(OtherVoice(p, score, duration_sec=0.0, use_ffmpeg=not _is_audio_file_soundfile_safe(p)))

    others.sort(key=lambda o: o.score)
    return target_files, others


def load_manifest(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_manifest(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Reading audio
# -----------------------------

def read_full_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    """Read full audio (mono) at target_sr.

    Tries soundfile first; if it fails (common for M4A/MP3), falls back to ffmpeg.
    """
    try:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        audio = _ensure_mono(audio)
        if int(sr) != int(target_sr):
            # For TARGET clips this should normally already match.
            # If not, use ffmpeg to resample (no SciPy dependency).
            raise RuntimeError("sr mismatch -> use ffmpeg")
        return audio
    except Exception:
        return _ffmpeg_read_full_f32le(path, int(target_sr))


def read_random_segment_soundfile(path: Path, target_duration_sec: float, rng: random.Random) -> Tuple[np.ndarray, int, float]:
    with sf.SoundFile(str(path), "r") as f:
        sr = int(f.samplerate)
        n_frames = int(f.frames)
        seg_frames = max(1, int(round(target_duration_sec * sr)))

        if seg_frames >= n_frames:
            f.seek(0)
            audio = f.read(n_frames, dtype="float32", always_2d=False)
            return _ensure_mono(audio), sr, 0.0

        max_start = n_frames - seg_frames
        start_frame = rng.randint(0, max_start)
        f.seek(start_frame)
        audio = f.read(seg_frames, dtype="float32", always_2d=False)
        return _ensure_mono(audio), sr, float(start_frame) / float(sr)


def read_random_segment_any(other: OtherVoice, target_duration_sec: float, target_sr: int, rng: random.Random) -> Tuple[np.ndarray, float]:
    """Read a random segment from OTHER at target_sr.

    Returns: (mono float32 samples at target_sr, start_sec)
    """
    dur = float(other.duration_sec)
    if dur <= 0.0:
        # last-ditch attempt
        dur = _ffprobe_duration_sec(other.path) or 0.0

    if dur <= 0.0:
        # We can still attempt: start at 0
        start_sec = 0.0
    else:
        max_start = max(0.0, dur - float(target_duration_sec))
        start_sec = rng.uniform(0.0, max_start) if max_start > 0 else 0.0

    if not other.use_ffmpeg and _is_audio_file_soundfile_safe(other.path):
        y, sr, sf_start = read_random_segment_soundfile(other.path, target_duration_sec, rng)
        if int(sr) != int(target_sr):
            # avoid SciPy: use ffmpeg resampling instead
            y = _ffmpeg_read_segment_f32le(other.path, sf_start, target_duration_sec, int(target_sr))
            return y, float(sf_start)
        return y.astype(np.float32, copy=False), float(sf_start)

    # ffmpeg path (M4A/MP3/etc)
    y = _ffmpeg_read_segment_f32le(other.path, start_sec, target_duration_sec, int(target_sr))
    return y.astype(np.float32, copy=False), float(start_sec)


# -----------------------------
# OTHER cache (ffmpeg -> WAV)
# -----------------------------

def _ffmpeg_convert_one(args: Tuple[str, str, int]) -> Tuple[bool, str]:
    src, dst, sr = args
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return False, "ffmpeg not found"

    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True, "skipped"

    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        src,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(int(sr)),
        "-c:a",
        "pcm_s16le",
        dst,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True, "ok"
    except subprocess.CalledProcessError:
        return False, "ffmpeg failed"


def prepare_other_cache(other_voices_dir: Path, other_cache_dir: Path, target_sr: int, workers: int) -> Path:
    other_cache_dir.mkdir(parents=True, exist_ok=True)
    index_path = other_cache_dir / "other_cache_index.jsonl"

    src_files = [p for p in other_voices_dir.rglob("*") if p.is_file() and _is_audio_file_any(p)]
    if not src_files:
        raise RuntimeError(f"No audio found under {other_voices_dir}")

    jobs: List[Tuple[str, str, int]] = []
    for p in src_files:
        sid = _stable_hex(_norm_key(p), 12)
        out_name = f"{p.stem}__{sid}.wav"
        out_path = other_cache_dir / out_name
        jobs.append((_norm_path(p), _norm_path(out_path), int(target_sr)))

    w = max(1, int(workers))
    ok = 0
    bad = 0
    with ProcessPoolExecutor(max_workers=w) as ex:
        futs = [ex.submit(_ffmpeg_convert_one, j) for j in jobs]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Preparing OTHER cache (ffmpeg, workers={w})", unit="file"):
            success, _msg = fut.result()
            if success:
                ok += 1
            else:
                bad += 1

    wavs = [p for p in other_cache_dir.glob("*.wav") if p.is_file()]
    with index_path.open("w", encoding="utf-8") as f:
        for wav in wavs:
            try:
                info = sf.info(str(wav))
                rec = {
                    "audio_path": _norm_path(wav),
                    "samplerate": int(info.samplerate),
                    "frames": int(info.frames),
                    "duration_sec": float(info.frames) / float(info.samplerate) if info.samplerate else 0.0,
                }
                f.write(json.dumps(rec) + "\n")
            except Exception:
                continue

    print(f"Cache ready: {other_cache_dir} (converted ok={ok}, failed={bad})")
    print(f"Index: {index_path}")
    return index_path


# -----------------------------
# OTHER scanning + sampling
# -----------------------------

def scan_other_voices_dir(other_voices_dir: Path, default_score: float = 0.5, require_ffmpeg_for_non_sf: bool = True) -> List[OtherVoice]:
    """Build OTHER pool.

    - WAV/FLAC/OGG: duration via soundfile
    - Everything else (M4A/MP3/etc): duration via ffprobe (needs ffmpeg tools installed)
    """
    out: List[OtherVoice] = []

    if require_ffmpeg_for_non_sf:
        if _find_ffmpeg() is None:
            raise RuntimeError("ffmpeg not found, but OTHER dir includes formats that need ffmpeg (e.g., .m4a/.mp3)")
        if _find_ffprobe() is None:
            # We can still decode segments without ffprobe, but random start becomes weak.
            # We’ll allow it, but warn.
            print("WARNING: ffprobe not found; random start for M4A/MP3 will be less accurate.")

    for p in other_voices_dir.rglob("*"):
        if not p.is_file() or not _is_audio_file_any(p):
            continue

        use_ff = not _is_audio_file_soundfile_safe(p)

        dur = 0.0
        if not use_ff:
            try:
                info = sf.info(str(p))
                dur = float(info.frames) / float(info.samplerate) if info.samplerate else 0.0
            except Exception:
                # if soundfile fails unexpectedly, fall back to ffmpeg path
                use_ff = True

        if use_ff:
            d = _ffprobe_duration_sec(p)
            dur = float(d) if d is not None else 0.0

        # Skip tiny files
        if dur and dur < 0.05:
            continue

        out.append(OtherVoice(p, float(default_score), float(dur), bool(use_ff)))

    out.sort(key=lambda o: _norm_key(o.path))
    return out


def build_weighted_sampler(others: Sequence[OtherVoice], eps: float = 1e-3) -> Tuple[List[OtherVoice], List[float]]:
    items = list(others)
    weights: List[float] = []
    for o in items:
        s = float(o.score)
        weights.append(1.0 / max(s, eps))
    return items, weights


def pick_other(items: Sequence[OtherVoice], weights: Sequence[float], rng: random.Random) -> OtherVoice:
    return rng.choices(list(items), weights=list(weights), k=1)[0]


# -----------------------------
# Mixing worker
# -----------------------------

_GLOBAL_OTHERS_ITEMS: Optional[Sequence[OtherVoice]] = None
_GLOBAL_OTHERS_WEIGHTS: Optional[Sequence[float]] = None


def _init_pool(others_items: Sequence[OtherVoice], others_weights: Sequence[float]) -> None:
    global _GLOBAL_OTHERS_ITEMS, _GLOBAL_OTHERS_WEIGHTS
    _GLOBAL_OTHERS_ITEMS = others_items
    _GLOBAL_OTHERS_WEIGHTS = others_weights


def mix_one(
    *,
    row: Dict,
    row_uid: int,
    out_mix_dir: Path,
    target_sr: int,
    snr_db_min: float,
    snr_db_max: float,
    allow_overwrite: bool,
    seed: int,
    max_bad_to_good_ratio: float,
    good_floor_db: float,
    max_other_pick_tries: int,
    others_items: Optional[Sequence[OtherVoice]] = None,
    others_weights: Optional[Sequence[float]] = None,
) -> Optional[Dict]:

    if others_items is None:
        others_items = _GLOBAL_OTHERS_ITEMS
    if others_weights is None:
        others_weights = _GLOBAL_OTHERS_WEIGHTS

    if not others_items or not others_weights:
        return None

    local_seed = int(seed) ^ (row_uid * 10007) ^ int(_stable_hex(_norm_key(row.get("audio_path", "")), 8), 16)
    rng = random.Random(local_seed)

    target_ap = Path(row.get("audio_path", ""))
    if not target_ap.exists():
        return None

    try:
        target_speech = read_full_audio_mono(target_ap, int(target_sr)).astype(np.float32, copy=False)
        target_len = len(target_speech)
        if target_len < int(0.01 * target_sr):
            return None
        target_dur = float(target_len) / float(target_sr)

        other = None
        other_scaled = None
        other_start_sec = 0.0
        snr_db = 0.0

        for _ in range(max(1, int(max_other_pick_tries))):
            cand = pick_other(others_items, others_weights, rng)
            if not cand.path.exists():
                continue

            try:
                other_seg, other_start_sec = read_random_segment_any(cand, target_dur, int(target_sr), rng)

                # pad/trim to exact length
                if len(other_seg) < target_len:
                    other_seg = np.pad(other_seg, (0, target_len - len(other_seg)), mode="constant")
                elif len(other_seg) > target_len:
                    other_seg = other_seg[:target_len]

                snr_db = rng.uniform(float(snr_db_min), float(snr_db_max))
                rs = _rms(target_speech)
                rn = _rms(other_seg)
                target_rn = rs / (10.0 ** (snr_db / 20.0))
                gain = target_rn / max(rn, 1e-12)
                other_scaled = other_seg * gain

                other = cand
                break
            except Exception:
                continue

        if other is None or other_scaled is None:
            return None

        other_scaled = duck_bad_under_good(
            other_scaled,
            good=target_speech,
            max_ratio=float(max_bad_to_good_ratio),
            good_floor_db=float(good_floor_db),
        )

        mixed = target_speech + other_scaled
        peak = float(np.max(np.abs(mixed)))
        if peak > 0.99:
            mixed = mixed * (0.99 / peak)

        safe_target = "".join(c if c.isalnum() or c in "-_" else "_" for c in target_ap.stem)
        safe_other = "".join(c if c.isalnum() or c in "-_" else "_" for c in other.path.stem)
        tid = _stable_hex(_norm_key(target_ap), 6)
        oid = _stable_hex(_norm_key(other.path), 6)

        out_name = (
            f"{safe_target}__t{tid}__row{row_uid:07d}__voice_mixed__"
            f"snr{snr_db:.1f}dB__oscore{other.score:.4f}__{safe_other}_o{oid}.wav"
        )
        out_path = out_mix_dir / out_name

        if (not out_path.exists()) or allow_overwrite:
            _safe_write_wav(out_path, mixed.astype(np.float32, copy=False), int(target_sr))

        new_row = dict(row)
        new_row["audio_path"] = _norm_path(out_path)
        new_row["orig_audio_path"] = row.get("audio_path")
        new_row["aug_voice_source"] = _norm_path(other.path)
        new_row["aug_voice_score"] = float(other.score)
        new_row["aug_snr_db"] = float(snr_db)
        new_row["aug_voice_start_sec"] = float(other_start_sec)
        new_row["aug_voice_dur_sec"] = float(target_dur)
        new_row["aug_voice_used_ffmpeg"] = bool(other.use_ffmpeg)
        new_row["is_voice_mixed"] = True
        return new_row

    except Exception:
        return None


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--prepare_other_cache", action="store_true", help="Only build OTHER cache (ffmpeg -> 16k mono WAV) and exit")
    ap.add_argument("--other_cache_dir", default="", help="When preparing cache: output directory for converted WAVs")

    ap.add_argument("--manifest", help="Input JSONL manifest path")
    ap.add_argument("--scores_csv", help="Speaker sort scores CSV with TARGET/OTHER files")
    ap.add_argument("--other_voices_dir", required=True, help="Directory with OTHER voice audio files (can include .m4a/.mp3)")
    ap.add_argument("--out_manifest", help="Output JSONL manifest path")
    ap.add_argument("--out_mix_dir", default="", help="Where to write voice-mixed copies")

    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--mix_ratio", type=float, default=0.5)
    ap.add_argument("--snr_db_min", type=float, default=5.0)
    ap.add_argument("--snr_db_max", type=float, default=20.0)

    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--workers", type=int, default=0, help="Workers for pool (0=auto)")
    ap.add_argument("--pool", choices=["thread", "process"], default="thread", help="Use threads or processes")

    ap.add_argument("--max_bad_to_good_ratio", type=float, default=1.0)
    ap.add_argument("--good_floor_db", type=float, default=-120.0)
    ap.add_argument("--max_other_pick_tries", type=int, default=8)

    args = ap.parse_args()

    other_voices_dir = Path(args.other_voices_dir)
    if not other_voices_dir.exists():
        raise FileNotFoundError(f"OTHER voices dir not found: {other_voices_dir}")

    workers = int(args.workers)
    if workers <= 0:
        workers = min(32, (os.cpu_count() or 4) + 4)

    if args.prepare_other_cache:
        other_cache_dir = Path(args.other_cache_dir) if args.other_cache_dir else (other_voices_dir.parent / (other_voices_dir.name + "_16k_wav"))
        prepare_other_cache(other_voices_dir, other_cache_dir, int(args.target_sr), workers)
        _beep()
        return 0

    if not args.manifest or not args.scores_csv or not args.out_manifest or not args.out_mix_dir:
        raise SystemExit("Mixing mode requires --manifest, --scores_csv, --out_manifest, --out_mix_dir (or use --prepare_other_cache)")

    manifest_path = Path(args.manifest)
    scores_csv = Path(args.scores_csv)
    out_manifest = Path(args.out_manifest)
    out_mix_dir = Path(args.out_mix_dir)
    out_mix_dir.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not scores_csv.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv}")

    target_files, _csv_others = load_target_and_other_from_csv(scores_csv)

    others = scan_other_voices_dir(other_voices_dir, default_score=0.5)
    if not target_files:
        raise RuntimeError("No TARGET entries found in scores CSV.")
    if not others:
        raise RuntimeError("No OTHER voice files found.")

    others_items, others_weights = build_weighted_sampler(others)

    base_rows = load_manifest(manifest_path)
    if not base_rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    target_rows: List[Tuple[int, Dict]] = []
    for idx, row in enumerate(base_rows):
        apath = row.get("audio_path")
        if apath and _norm_key(apath) in target_files:
            target_rows.append((idx, row))

    if not target_rows:
        raise RuntimeError("No TARGET rows found in manifest that match scores CSV paths.")

    rng_master = random.Random(int(args.seed))
    n_to_mix = int(round(len(target_rows) * float(args.mix_ratio)))
    n_to_mix = max(0, min(len(target_rows), n_to_mix))
    if n_to_mix == 0:
        write_manifest(out_manifest, base_rows)
        print("mix_ratio produced 0 mixes; wrote base manifest only.")
        _beep()
        return 0

    chosen = set(rng_master.sample(range(len(target_rows)), k=n_to_mix))
    tasks: List[Tuple[int, Dict]] = [target_rows[i] for i in sorted(chosen)]

    new_rows: List[Dict] = []
    failed = 0

    Executor = ThreadPoolExecutor if args.pool == "thread" else ProcessPoolExecutor

    initargs = (others_items, others_weights) if Executor is ProcessPoolExecutor else None

    with Executor(max_workers=workers, initializer=_init_pool if initargs else None, initargs=initargs or ()) as ex:
        futs = []
        for orig_idx, row in tasks:
            futs.append(
                ex.submit(
                    mix_one,
                    row=row,
                    row_uid=int(orig_idx),
                    out_mix_dir=out_mix_dir,
                    target_sr=int(args.target_sr),
                    snr_db_min=float(args.snr_db_min),
                    snr_db_max=float(args.snr_db_max),
                    allow_overwrite=bool(args.overwrite),
                    seed=int(args.seed),
                    max_bad_to_good_ratio=float(args.max_bad_to_good_ratio),
                    good_floor_db=float(args.good_floor_db),
                    max_other_pick_tries=int(args.max_other_pick_tries),
                    others_items=None if Executor is ProcessPoolExecutor else others_items,
                    others_weights=None if Executor is ProcessPoolExecutor else others_weights,
                )
            )

        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Mixing OTHER voices ({args.pool}, workers={workers})", unit="clip"):
            r = fut.result()
            if r is None:
                failed += 1
            else:
                new_rows.append(r)

    out_rows = list(base_rows) + list(new_rows)
    if args.shuffle:
        rng_master.shuffle(out_rows)

    write_manifest(out_manifest, out_rows)

    print("Done")
    print(f"  Base rows:                 {len(base_rows)}")
    print(f"  TARGET rows in manifest:   {len(target_rows)}")
    print(f"  Requested mixes:           {len(tasks)}")
    print(f"  Successful mixes:          {len(new_rows)}")
    print(f"  Failed mixes:              {failed}")
    print(f"  Output rows:               {len(out_rows)}")
    print(f"  Output manifest:           {out_manifest}")
    print(f"  Mixed audio dir:           {out_mix_dir}")

    _beep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
