#!/usr/bin/env python3
"""stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise_NO_NOISE_ONLY.py

What changed vs your current Stage 6
-----------------------------------
- **Noise-only rows are disabled by default** and will NOT be added to the output manifest.
- Backwards compatible: you can still request `--mode noise_only` / `--mode both`, but
  **nothing will be added** unless you explicitly pass `--enable_noise_only_rows`.

Why
---
Noise-only rows typically have an empty transcript (or a placeholder). Depending on your
collator/normaliser, they can become batches with zero valid label tokens -> CrossEntropyLoss
can produce NaN/Inf when the loss is averaged over zero valid targets.

Goal
----
Given a Whisper-style JSONL manifest and a folder of noise audio:

1) Chunk ALL noise files into <= max_chunk_sec WAV segments (16 kHz default)
2) Create noisy MIXED versions of existing speech clips at random SNRs,
   BUT only for rows whose audio files are marked as TARGET in speaker_sort_scores.csv.

(Optionally) Add noise-only rows ONLY if you *explicitly* enable them.

Dependencies
------------
pip install numpy soundfile tqdm
(Optional for resampling) pip install scipy

Example (recommended)
---------------------
python stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise.py \
  --manifest "i:/Record_chunks/pairs_manifest.jsonl" \
  --noises_dir "i:/noise/RIRS_NOISES/pointsource_noises" \
  --scores_csv "i:/whisper-acft/speaker_sort_scores.csv" \
  --out_manifest "i:/Record_chunks/pairs_manifest_with_mix.jsonl" \
  --out_noise_dir "i:/Record_chunks/noise_chunks" \
  --out_mix_dir "i:/Record_chunks/noisy_mixed" \
  --mode mix \
  --mix_ratio 0.5 \
  --snr_db_min 5 --snr_db_max 20 \
  --seed 1337 \
  --shuffle \
  --workers 8

If you *really* want noise-only rows (not recommended):
------------------------------------------------------
Add:
  --mode both --enable_noise_only_rows --noise_ratio 0.05

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Avoid CPU oversubscription when using multiprocessing + OpenMP/BLAS
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import soundfile as sf
from tqdm import tqdm


# -----------------------------
# Small helpers
# -----------------------------

def beep() -> None:
    """Audible notification when done."""
    try:
        import winsound

        winsound.Beep(1000, 350)
        winsound.Beep(1400, 250)
    except Exception:
        print("\a")


def _norm_path(p: str | Path) -> str:
    return str(Path(p))


def _is_audio_file(p: Path) -> bool:
    # soundfile supports wav/flac/ogg and some others depending on libsndfile.
    return p.suffix.lower() in {".wav", ".flac", ".ogg"}


def _rms(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))


def _ensure_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    return np.mean(x, axis=1)


def _resample_if_needed(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    try:
        from scipy.signal import resample_poly
    except Exception as e:
        raise RuntimeError(
            "Sample rate mismatch (sr_in != sr_out) but SciPy is not available. "
            "Install scipy or ensure all audio is already at the target sample rate."
        ) from e

    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    y = resample_poly(x, up=up, down=down)
    return y.astype(np.float32, copy=False)


def _safe_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def _mp_context_spawn() -> mp.context.BaseContext:
    return mp.get_context("spawn")


def _suggest_workers(user_workers: int) -> int:
    if int(user_workers) > 0:
        return int(user_workers)
    n = os.cpu_count() or 4
    return max(1, n - 1)


def _stable_id(s: str, n: int = 8) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _norm_key(path_str: str) -> str:
    return str(path_str).lower().replace("\\", "/")


@dataclass
class NoiseChunk:
    audio_path: str
    source_audio: str
    chunk_index: int
    chunk_start: float
    chunk_end: float


# -----------------------------
# Parallel noise chunking
# -----------------------------

def _chunk_one_noise_file(task: Dict) -> List[Dict]:
    """Worker: chunk a single noise file."""
    nf = Path(task["noise_file"])
    out_noise_dir = Path(task["out_noise_dir"])
    target_sr = int(task["target_sr"])
    max_chunk_sec = float(task["max_chunk_sec"])
    min_chunk_sec = float(task["min_chunk_sec"])
    random_chunking = bool(task["random_chunking"])
    random_min_sec = float(task["random_min_sec"])
    random_max_sec = float(task["random_max_sec"])
    base_seed = int(task["seed"])

    per_file_seed = base_seed ^ int(hashlib.md5(str(nf).lower().encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(per_file_seed)

    out: List[Dict] = []

    try:
        with sf.SoundFile(str(nf), "r") as snd:
            sr_in = int(snd.samplerate)
            n_frames = int(len(snd))

            cursor = 0
            chunk_i = 0
            noise_id = _stable_id(str(nf).lower(), 8)

            while cursor < n_frames:
                dur = rng.uniform(random_min_sec, random_max_sec) if random_chunking else max_chunk_sec
                dur = min(dur, max_chunk_sec)

                frames = int(round(dur * sr_in))
                if frames <= 0:
                    break

                end = min(cursor + frames, n_frames)
                frames_to_read = end - cursor

                # Discard too-short tail chunks unless final chunk
                if (frames_to_read / sr_in) < min_chunk_sec and (end != n_frames):
                    cursor = end
                    continue

                snd.seek(cursor)
                audio = snd.read(frames_to_read, dtype="float32", always_2d=False)
                audio = _ensure_mono(audio)
                audio = _resample_if_needed(audio, sr_in, target_sr)

                dur_out = float(len(audio)) / float(target_sr)
                if dur_out < min_chunk_sec:
                    cursor = end
                    continue

                stem = nf.stem
                safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
                out_name = f"noise__{safe_stem}__{noise_id}__chunk{chunk_i:05d}.wav"
                out_path = out_noise_dir / out_name

                _safe_write_wav(out_path, audio, target_sr)

                start_sec = float(cursor) / float(sr_in)
                end_sec = float(end) / float(sr_in)

                out.append(
                    {
                        "audio_path": _norm_path(out_path),
                        "source_audio": _norm_path(nf),
                        "chunk_index": int(chunk_i),
                        "chunk_start": float(start_sec),
                        "chunk_end": float(end_sec),
                    }
                )

                chunk_i += 1
                cursor = end

    except Exception:
        return []

    return out


def chunk_noises(
    noises_dir: Path,
    out_noise_dir: Path,
    target_sr: int,
    max_chunk_sec: float,
    min_chunk_sec: float,
    random_chunking: bool,
    random_min_sec: float,
    random_max_sec: float,
    reuse_index: bool,
    seed: int,
    workers: int,
) -> List[NoiseChunk]:
    """Chunk every audio file in noises_dir into <= max_chunk_sec pieces (parallel)."""

    index_path = out_noise_dir / "noise_chunks_index.jsonl"
    if reuse_index and index_path.exists():
        chunks: List[NoiseChunk] = []
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                chunks.append(
                    NoiseChunk(
                        audio_path=obj["audio_path"],
                        source_audio=obj["source_audio"],
                        chunk_index=int(obj["chunk_index"]),
                        chunk_start=float(obj["chunk_start"]),
                        chunk_end=float(obj["chunk_end"]),
                    )
                )
        if chunks:
            return chunks

    noise_files = [p for p in noises_dir.rglob("*") if p.is_file() and _is_audio_file(p)]
    if not noise_files:
        raise FileNotFoundError(f"No audio files (.wav/.flac/.ogg) found under: {noises_dir}")

    out_noise_dir.mkdir(parents=True, exist_ok=True)

    w = _suggest_workers(workers)

    tasks: List[Dict] = [
        {
            "noise_file": str(nf),
            "out_noise_dir": str(out_noise_dir),
            "target_sr": int(target_sr),
            "max_chunk_sec": float(max_chunk_sec),
            "min_chunk_sec": float(min_chunk_sec),
            "random_chunking": bool(random_chunking),
            "random_min_sec": float(random_min_sec),
            "random_max_sec": float(random_max_sec),
            "seed": int(seed),
        }
        for nf in noise_files
    ]

    flat: List[Dict] = []
    with ProcessPoolExecutor(max_workers=w, mp_context=_mp_context_spawn()) as ex:
        futures = [ex.submit(_chunk_one_noise_file, t) for t in tasks]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Chunking noise files (proc={w})",
            unit="file",
        ):
            res = fut.result()
            if res:
                flat.extend(res)

    if not flat:
        raise RuntimeError("Chunking produced zero usable noise chunks. Check your noises_dir contents.")

    flat.sort(key=lambda o: (o["source_audio"], int(o["chunk_index"]), o["audio_path"]))

    tmp = index_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for obj in flat:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp.replace(index_path)

    return [
        NoiseChunk(
            audio_path=o["audio_path"],
            source_audio=o["source_audio"],
            chunk_index=int(o["chunk_index"]),
            chunk_start=float(o["chunk_start"]),
            chunk_end=float(o["chunk_end"]),
        )
        for o in flat
    ]


def load_target_files(scores_csv: Path) -> set[str]:
    """Load set of target audio files from speaker sort scores CSV."""
    target_files: set[str] = set()
    with scores_csv.open("r", encoding="utf-8") as f:
        next(f)  # header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3 and parts[2] == "TARGET":
                target_files.add(_norm_key(parts[0]))
    return target_files


def load_manifest(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"Failed parsing JSON on line {i} of {path}: {e}")
    return rows


def write_manifest(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -----------------------------
# Optional noise-only rows (now opt-in)
# -----------------------------

def add_noise_only_rows(
    base_rows: List[Dict],
    noise_chunks: List[NoiseChunk],
    noise_ratio: float,
    n_noise: Optional[int],
    seed: int,
) -> List[Dict]:
    rng = random.Random(int(seed))

    if n_noise is None:
        if noise_ratio <= 0:
            return []
        n_noise = int(round(len(base_rows) * noise_ratio))

    n_noise = max(0, int(n_noise))
    if n_noise == 0:
        return []

    if n_noise > len(noise_chunks):
        n_noise = len(noise_chunks)

    picked = rng.sample(noise_chunks, k=n_noise)
    out: List[Dict] = []
    for nc in picked:
        out.append(
            {
                "audio_path": nc.audio_path,
                "raw_transcription": "",
                "source_audio": nc.source_audio,
                "chunk_index": nc.chunk_index,
                "chunk_start": nc.chunk_start,
                "chunk_end": nc.chunk_end,
                "transcript_json": None,
                "is_noise_only": True,
            }
        )
    return out


# -----------------------------
# Parallel mixing for TARGET rows
# -----------------------------

_G_NOISE_CHUNKS: List[Dict] = []
_G_TARGET_SR: int = 16000


def _mix_init(noise_index_path: str, target_sr: int) -> None:
    global _G_NOISE_CHUNKS, _G_TARGET_SR
    _G_TARGET_SR = int(target_sr)
    idx_path = Path(noise_index_path)
    chunks: List[Dict] = []
    with idx_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    _G_NOISE_CHUNKS = chunks


def _read_wav_any(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = _ensure_mono(audio)
    return audio, int(sr)


def _mix_one_target_row(task: Dict) -> Optional[Dict]:
    """Worker: mix noise into a single speech clip and return a new manifest row."""
    global _G_NOISE_CHUNKS, _G_TARGET_SR

    row = task["row"]
    row_uid = str(task["row_uid"])
    out_mix_dir = Path(task["out_mix_dir"])
    snr_db_min = float(task["snr_db_min"])
    snr_db_max = float(task["snr_db_max"])
    allow_overwrite = bool(task["allow_overwrite"])
    seed = int(task["seed"])

    rng = random.Random(seed)

    ap = Path(row.get("audio_path", ""))
    if not ap.exists():
        return None

    try:
        speech, sr_in = _read_wav_any(ap)
        speech = _resample_if_needed(speech, sr_in, _G_TARGET_SR)

        speech_dur = float(len(speech)) / float(_G_TARGET_SR)
        if speech_dur <= 0.01:
            return None

        noise_pick: Optional[Dict] = None
        for _ in range(30):
            cand = rng.choice(_G_NOISE_CHUNKS)
            cand_len = float(cand["chunk_end"]) - float(cand["chunk_start"])
            if cand_len >= speech_dur:
                noise_pick = cand
                break
        if noise_pick is None:
            noise_pick = rng.choice(_G_NOISE_CHUNKS)

        noise_audio_path = Path(noise_pick["audio_path"])
        if not noise_audio_path.exists():
            return None

        noise, nsr = _read_wav_any(noise_audio_path)
        noise = _resample_if_needed(noise, nsr, _G_TARGET_SR)

        if len(noise) < len(speech):
            reps = int(math.ceil(len(speech) / max(1, len(noise))))
            noise = np.tile(noise, reps)[: len(speech)]
        else:
            max_start = len(noise) - len(speech)
            if max_start > 0:
                start = rng.randint(0, max_start)
                noise = noise[start : start + len(speech)]
            else:
                noise = noise[: len(speech)]

        snr_db = rng.uniform(snr_db_min, snr_db_max)
        rs = _rms(speech)
        rn = _rms(noise)
        target_rn = rs / (10.0 ** (snr_db / 20.0))
        gain = target_rn / max(rn, 1e-12)
        noise_scaled = noise * gain
        
        # Apply instantaneous ducker to ensure noise never exceeds speech
        from audio_instant_ducker_utils import duck_bad_under_good
        noise_scaled = duck_bad_under_good(
            noise_scaled,
            good=speech,
            max_ratio=float(task.get("max_bad_to_good_ratio", 1.0)),
            good_floor_db=float(task.get("good_floor_db", -120.0)),
        )
        
        mixed = speech + noise_scaled

        peak = float(np.max(np.abs(mixed)))
        if peak > 0.99:
            mixed = mixed * (0.99 / peak)

        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in ap.stem)
        speech_id = _stable_id(_norm_key(str(ap)), 6)
        noise_src = str(noise_pick.get("source_audio", "noise"))
        noise_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in Path(noise_src).stem)
        noise_id = _stable_id(_norm_key(noise_src), 6)

        out_name = (
            f"{safe_stem}__{speech_id}__row{row_uid}__noisy__snr{snr_db:.1f}dB__{noise_stem}_{noise_id}.wav"
        )
        out_path = out_mix_dir / out_name
        out_mix_dir.mkdir(parents=True, exist_ok=True)

        if (not out_path.exists()) or allow_overwrite:
            _safe_write_wav(out_path, mixed.astype(np.float32, copy=False), _G_TARGET_SR)

        new_row = dict(row)
        new_row["audio_path"] = _norm_path(out_path)
        new_row["aug_noise_source"] = noise_pick.get("source_audio")
        new_row["aug_noise_chunk_audio"] = noise_pick.get("audio_path")
        new_row["aug_snr_db"] = float(snr_db)
        new_row["is_noisy_mixed"] = True
        return new_row

    except Exception:
        return None


def mix_noise_into_speech(
    base_rows: List[Dict],
    noise_chunks: List[NoiseChunk],
    out_mix_dir: Path,
    target_sr: int,
    mix_ratio: float,
    snr_db_min: float,
    snr_db_max: float,
    allow_overwrite: bool,
    target_files: set[str],
    seed: int,
    workers: int,
    max_bad_to_good_ratio: float = 1.0,
    good_floor_db: float = -120.0,
) -> List[Dict]:
    """Create noisy mixed copies for a subset of TARGET base_rows and return new rows (parallel)."""

    out_mix_dir.mkdir(parents=True, exist_ok=True)

    target_rows: List[Dict] = []
    for row in base_rows:
        ap = row.get("audio_path")
        if ap and _norm_key(ap) in target_files:
            target_rows.append(row)

    if not target_rows:
        print("[WARN] No target files found in manifest for noise mixing", file=sys.stderr)
        return []

    print(f"Found {len(target_rows)} TARGET rows out of {len(base_rows)} total rows")

    n_to_mix = int(round(len(target_rows) * float(mix_ratio)))
    n_to_mix = max(0, min(len(target_rows), n_to_mix))
    if n_to_mix == 0:
        return []

    rng = random.Random(int(seed))
    chosen_idx = sorted(rng.sample(range(len(target_rows)), k=n_to_mix))

    w = _suggest_workers(workers)

    noise_index_path = out_mix_dir / "temp_noise_index.jsonl"
    with noise_index_path.open("w", encoding="utf-8") as f:
        for nc in noise_chunks:
            f.write(
                json.dumps(
                    {
                        "audio_path": nc.audio_path,
                        "source_audio": nc.source_audio,
                        "chunk_index": nc.chunk_index,
                        "chunk_start": nc.chunk_start,
                        "chunk_end": nc.chunk_end,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    tasks: List[Dict] = []
    for local_i in chosen_idx:
        tasks.append(
            {
                "row": target_rows[local_i],
                "row_uid": str(local_i),
                "out_mix_dir": str(out_mix_dir),
                "snr_db_min": float(snr_db_min),
                "snr_db_max": float(snr_db_max),
                "allow_overwrite": bool(allow_overwrite),
                "max_bad_to_good_ratio": float(max_bad_to_good_ratio),
                "good_floor_db": float(good_floor_db),
                "seed": int(seed + local_i * 10007),
            }
        )

    new_rows: List[Dict] = []

    with ProcessPoolExecutor(
        max_workers=w,
        mp_context=_mp_context_spawn(),
        initializer=_mix_init,
        initargs=(str(noise_index_path), int(target_sr)),
    ) as ex:
        futures = [ex.submit(_mix_one_target_row, t) for t in tasks]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Mixing noise into TARGET speech (proc={w})",
            unit="clip",
        ):
            r = fut.result()
            if r is not None:
                new_rows.append(r)

    noise_index_path.unlink(missing_ok=True)

    return new_rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Input JSONL manifest path")
    p.add_argument("--noises_dir", required=True, help="Folder containing noise audio (wav/flac/ogg)")
    p.add_argument("--scores_csv", required=True, help="Speaker sort scores CSV with TARGET files")

    p.add_argument("--out_manifest", required=True, help="Output JSONL manifest path")
    p.add_argument("--out_noise_dir", required=True, help="Where to write <=max_chunk_sec noise chunks (used as mix sources)")
    p.add_argument("--out_mix_dir", default=None, help="Where to write noisy mixed copies (mode=mix or both)")

    # Keep backwards compatible modes, but default is now mix
    p.add_argument("--mode", choices=["noise_only", "mix", "both"], default="mix")

    p.add_argument("--enable_noise_only_rows", action="store_true", help="Explicitly allow adding noise-only rows to manifest (NOT recommended)")

    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--max_chunk_sec", type=float, default=29.5)
    p.add_argument("--min_chunk_sec", type=float, default=1.0)

    p.add_argument("--random_chunking", action="store_true", help="Random chunk lengths instead of fixed chunks")
    p.add_argument("--random_min_sec", type=float, default=2.0)
    p.add_argument("--random_max_sec", type=float, default=29.5)

    # Noise-only parameters (only used if --enable_noise_only_rows)
    p.add_argument("--noise_ratio", type=float, default=0.15, help="Fraction of manifest rows to add as noise-only")
    p.add_argument("--n_noise", type=int, default=None, help="Exact number of noise-only rows to add (overrides noise_ratio)")

    p.add_argument("--mix_ratio", type=float, default=0.0, help="Fraction of TARGET rows to create noisy mixed copies for")
    p.add_argument("--snr_db_min", type=float, default=5.0)
    p.add_argument("--snr_db_max", type=float, default=20.0)

    p.add_argument("--shuffle", action="store_true", help="Shuffle output manifest rows")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--reuse_noise_index", action="store_true", help="Reuse noise chunk index if it exists")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting generated audio files")

    p.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto)")
    
    # Instantaneous ducker parameters
    p.add_argument('--max_bad_to_good_ratio', type=float, default=1.0, help='Maximum ratio of bad to good signal at any sample (1.0 = bad never louder)')
    p.add_argument('--good_floor_db', type=float, default=-120.0, help='Floor for good signal in dB (strict: -120, relaxed: -45)')

    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    manifest_path = Path(args.manifest)
    noises_dir = Path(args.noises_dir)
    scores_csv = Path(args.scores_csv)
    out_manifest = Path(args.out_manifest)
    out_noise_dir = Path(args.out_noise_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not noises_dir.exists():
        raise FileNotFoundError(f"Noises directory not found: {noises_dir}")
    if not scores_csv.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv}")

    target_files = load_target_files(scores_csv)
    print(f"Loaded {len(target_files)} TARGET files from {scores_csv}")

    base_rows = load_manifest(manifest_path)
    if not base_rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    noise_chunks = chunk_noises(
        noises_dir=noises_dir,
        out_noise_dir=out_noise_dir,
        target_sr=int(args.target_sr),
        max_chunk_sec=float(args.max_chunk_sec),
        min_chunk_sec=float(args.min_chunk_sec),
        random_chunking=bool(args.random_chunking),
        random_min_sec=float(args.random_min_sec),
        random_max_sec=float(args.random_max_sec),
        reuse_index=bool(args.reuse_noise_index),
        seed=int(args.seed),
        workers=int(args.workers),
    )

    added_rows: List[Dict] = []

    # Noise-only rows are now OPT-IN
    if args.mode in {"noise_only", "both"}:
        if not args.enable_noise_only_rows:
            print("[INFO] Noise-only rows requested by --mode, but they are DISABLED by default.")
            print("       If you really want them, add: --enable_noise_only_rows")
        else:
            added_rows.extend(
                add_noise_only_rows(
                    base_rows=base_rows,
                    noise_chunks=noise_chunks,
                    noise_ratio=float(args.noise_ratio),
                    n_noise=args.n_noise,
                    seed=int(args.seed),
                )
            )

    # Mixing
    if args.mode in {"mix", "both"}:
        if not args.out_mix_dir:
            raise ValueError("--out_mix_dir is required when mode is mix or both")
        out_mix_dir = Path(args.out_mix_dir)
        added_rows.extend(
            mix_noise_into_speech(
                base_rows=base_rows,
                noise_chunks=noise_chunks,
                out_mix_dir=out_mix_dir,
                target_sr=int(args.target_sr),
                mix_ratio=float(args.mix_ratio),
                snr_db_min=float(args.snr_db_min),
                snr_db_max=float(args.snr_db_max),
                allow_overwrite=bool(args.overwrite),
                target_files=target_files,
                seed=int(args.seed),
                workers=int(args.workers),
                max_bad_to_good_ratio=float(args.max_bad_to_good_ratio),
                good_floor_db=float(args.good_floor_db),
            )
        )

    out_rows = list(base_rows) + list(added_rows)

    if args.shuffle:
        random.shuffle(out_rows)

    write_manifest(out_manifest, out_rows)

    print("\nDone")
    print(f"  Base rows:     {len(base_rows)}")
    print(f"  Added rows:    {len(added_rows)}")
    print(f"  Output rows:   {len(out_rows)}")
    print(f"  Output file:   {out_manifest}")
    print(f"  Noise chunks:  {len(noise_chunks)} (index at {out_noise_dir / 'noise_chunks_index.jsonl'})")

    beep()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
