#!/usr/bin/env python3
"""stage_5_augment_manifest_with_noise.py

Goal
----
Given a Whisper-style JSONL manifest (one JSON object per line), augment it using a folder
of *non-speech* noise audio:

1) Chunk ALL noise files into <= max_chunk_sec WAV segments (16 kHz by default)
2) Add a random subset of those noise-only chunks as extra manifest rows (true negatives)
3) (Optional) Also create noisy MIXED versions of existing speech clips at random SNRs

Parallel upgrade (this version)
-------------------------------
- Noise chunking is parallelised per noise file using ProcessPoolExecutor (Windows-safe spawn).
- Mixing is parallelised per selected speech clip using ProcessPoolExecutor with an initializer
  that loads the noise chunk index once per worker.

Dependencies
------------
pip install numpy soundfile tqdm
(Optional for resampling) pip install scipy

Usage examples
--------------
# Stage 5: Augment manifest with noise
python stage_5_augment_manifest_with_noise.py ^
  --manifest "I:/Record_chunks/pairs_manifest_filtered.jsonl" ^
  --noises_dir "I:/noise/RIRS_NOISES/pointsource_noises" ^
  --out_manifest "I:/Record_chunks/pairs_manifest_filtered_with_noises.jsonl" ^
  --out_noise_dir "I:/Record_chunks/noise_chunks" ^
  --mode noise_only --noise_ratio 0.2 --seed 1337 --shuffle ^
  --workers 8

# Also create noisy mixes
python stage_5_augment_manifest_with_noise.py ^
  --manifest "I:/Record_chunks/pairs_manifest_filtered.jsonl" ^
  --noises_dir "I:/noise/RIRS_NOISES/pointsource_noises" ^
  --out_manifest "I:/Record_chunks/pairs_manifest_filtered_with_noises_and_mixes.jsonl" ^
  --out_noise_dir "I:/Record_chunks/noise_chunks" ^
  --out_mix_dir "I:/Record_chunks/noisy_mixes" ^
  --mode both --noise_ratio 0.15 --mix_ratio 0.30 --snr_db_min 5 --snr_db_max 20 ^
  --seed 1337 --shuffle --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Prevent CPU oversubscription when using multiprocessing + BLAS/OpenMP stacks.
# Set BEFORE importing numpy/scipy-heavy code.
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
# Utilities
# -----------------------------

def _norm_path(p: str | Path) -> str:
    # Keep Windows-style paths readable in JSONL
    return str(Path(p))


def _is_audio_file(p: Path) -> bool:
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

    g = math.gcd(int(sr_in), int(sr_out))
    up = int(sr_out) // g
    down = int(sr_in) // g
    y = resample_poly(x, up=up, down=down)
    return y.astype(np.float32, copy=False)


def _safe_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr, subtype="PCM_16")


def _stable_id(s: str, n: int = 8) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _mp_context_spawn() -> mp.context.BaseContext:
    return mp.get_context("spawn")


def _suggest_workers(user_workers: int) -> int:
    if int(user_workers) > 0:
        return int(user_workers)
    n = os.cpu_count() or 4
    return max(1, n - 1)


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
                if random_chunking:
                    dur = rng.uniform(random_min_sec, random_max_sec)
                    dur = min(dur, max_chunk_sec)
                else:
                    dur = max_chunk_sec

                frames = int(round(dur * sr_in))
                if frames <= 0:
                    break

                end = min(cursor + frames, n_frames)
                frames_to_read = end - cursor

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

    chunks_flat: List[Dict] = []
    with ProcessPoolExecutor(max_workers=w, mp_context=_mp_context_spawn()) as ex:
        futures = [ex.submit(_chunk_one_noise_file, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Chunking noise files (proc={w})", unit="file"):
            res = fut.result()
            if res:
                chunks_flat.extend(res)

    if not chunks_flat:
        raise RuntimeError("Chunking produced zero usable noise chunks. Check your noises_dir contents.")

    chunks_flat.sort(key=lambda o: (o["source_audio"], int(o["chunk_index"]), o["audio_path"]))

    tmp_path = index_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w", encoding="utf-8") as index_f:
        for obj in chunks_flat:
            index_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    tmp_path.replace(index_path)

    return [
        NoiseChunk(
            audio_path=o["audio_path"],
            source_audio=o["source_audio"],
            chunk_index=int(o["chunk_index"]),
            chunk_start=float(o["chunk_start"]),
            chunk_end=float(o["chunk_end"]),
        )
        for o in chunks_flat
    ]


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
                "raw_transcription": "<|nocaptions|>",
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
# Parallel mixing
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


def _mix_one_row(task: Dict) -> Optional[Dict]:
    global _G_NOISE_CHUNKS, _G_TARGET_SR
    row = task["row"]
    out_mix_dir = Path(task["out_mix_dir"])
    snr_db_min = float(task["snr_db_min"])
    snr_db_max = float(task["snr_db_max"])
    allow_overwrite = bool(task["allow_overwrite"])
    seed = int(task["seed"])

    rng = random.Random(seed)

    ap = Path(row["audio_path"])
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
        mixed = speech + (noise * gain)

        peak = float(np.max(np.abs(mixed)))
        if peak > 0.99:
            mixed = mixed * (0.99 / peak)

        base_stem = ap.stem
        safe_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_stem)
        speech_id = _stable_id(str(ap).lower(), 6)

        noise_src = str(noise_pick.get("source_audio", "noise"))
        noise_stem = Path(noise_src).stem
        noise_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in noise_stem)
        noise_id = _stable_id(noise_src.lower(), 6)

        out_name = f"{safe_stem}__{speech_id}__noisy__snr{snr_db:.1f}dB__{noise_stem}_{noise_id}.wav"
        out_path = out_mix_dir / out_name
        out_mix_dir.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not allow_overwrite:
            pass
        else:
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
    noise_index_path: Path,
    out_mix_dir: Path,
    target_sr: int,
    mix_ratio: float,
    snr_db_min: float,
    snr_db_max: float,
    allow_overwrite: bool,
    seed: int,
    workers: int,
) -> List[Dict]:
    n_to_mix = int(round(len(base_rows) * float(mix_ratio)))
    n_to_mix = max(0, min(len(base_rows), n_to_mix))
    if n_to_mix == 0:
        return []

    rng = random.Random(int(seed))
    chosen_idx = sorted(rng.sample(range(len(base_rows)), k=n_to_mix))

    w = _suggest_workers(workers)

    tasks: List[Dict] = []
    for i in chosen_idx:
        tasks.append(
            {
                "row": base_rows[i],
                "out_mix_dir": str(out_mix_dir),
                "snr_db_min": float(snr_db_min),
                "snr_db_max": float(snr_db_max),
                "allow_overwrite": bool(allow_overwrite),
                "seed": int(seed + i * 10007),
            }
        )

    new_rows: List[Dict] = []
    with ProcessPoolExecutor(
        max_workers=w,
        mp_context=_mp_context_spawn(),
        initializer=_mix_init,
        initargs=(str(noise_index_path), int(target_sr)),
    ) as ex:
        futures = [ex.submit(_mix_one_row, t) for t in tasks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Mixing noise into speech (proc={w})", unit="clip"):
            r = fut.result()
            if r is not None:
                new_rows.append(r)

    return new_rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Input JSONL manifest path")
    p.add_argument("--noises_dir", required=True, help="Folder containing noise audio (wav/flac/ogg)")

    p.add_argument("--out_manifest", required=True, help="Output JSONL manifest path")
    p.add_argument("--out_noise_dir", required=True, help="Where to write <=30s noise chunks")
    p.add_argument("--out_mix_dir", default=None, help="Where to write noisy mixed copies (mode=mix)")

    p.add_argument("--mode", choices=["noise_only", "mix", "both"], default="noise_only")

    p.add_argument("--target_sr", type=int, default=16000)
    p.add_argument("--max_chunk_sec", type=float, default=29.5)
    p.add_argument("--min_chunk_sec", type=float, default=1.0)

    p.add_argument("--random_chunking", action="store_true", help="Random chunk lengths instead of fixed chunks")
    p.add_argument("--random_min_sec", type=float, default=2.0)
    p.add_argument("--random_max_sec", type=float, default=29.5)

    p.add_argument("--noise_ratio", type=float, default=0.15, help="How many noise-only rows to add, as a fraction of manifest")
    p.add_argument("--n_noise", type=int, default=None, help="Exact number of noise-only rows to add (overrides noise_ratio)")

    p.add_argument("--mix_ratio", type=float, default=0.0, help="Fraction of base rows to create noisy mixed copies for")
    p.add_argument("--snr_db_min", type=float, default=5.0)
    p.add_argument("--snr_db_max", type=float, default=20.0)

    p.add_argument("--shuffle", action="store_true", help="Shuffle output manifest rows")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--reuse_noise_index", action="store_true", help="Reuse noise chunk index if it exists")
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting generated audio files")

    p.add_argument("--workers", type=int, default=0, help="Parallel workers for chunking/mixing (0=auto)")

    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    manifest_path = Path(args.manifest)
    noises_dir = Path(args.noises_dir)
    out_manifest = Path(args.out_manifest)
    out_noise_dir = Path(args.out_noise_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not noises_dir.exists():
        raise FileNotFoundError(f"Noises directory not found: {noises_dir}")

    base_rows = load_manifest(manifest_path)
    if not base_rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    # 1) Chunk noises
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

    new_rows: List[Dict] = []

    # 2) Add noise-only rows
    if args.mode in {"noise_only", "both"}:
        new_rows.extend(
            add_noise_only_rows(
                base_rows=base_rows,
                noise_chunks=noise_chunks,
                noise_ratio=float(args.noise_ratio),
                n_noise=args.n_noise,
                seed=int(args.seed),
            )
        )

    # 3) Mix noise into speech (optional)
    if args.mode in {"mix", "both"}:
        if not args.out_mix_dir:
            raise ValueError("--out_mix_dir is required when mode is mix or both")
        out_mix_dir = Path(args.out_mix_dir)
        index_path = out_noise_dir / "noise_chunks_index.jsonl"
        if not index_path.exists():
            raise RuntimeError(f"Noise index missing (expected at {index_path}). Chunking step failed?")

        new_rows.extend(
            mix_noise_into_speech(
                base_rows=base_rows,
                noise_index_path=index_path,
                out_mix_dir=out_mix_dir,
                target_sr=int(args.target_sr),
                mix_ratio=float(args.mix_ratio),
                snr_db_min=float(args.snr_db_min),
                snr_db_max=float(args.snr_db_max),
                allow_overwrite=bool(args.overwrite),
                seed=int(args.seed),
                workers=int(args.workers),
            )
        )

    out_rows = list(base_rows) + list(new_rows)

    if args.shuffle:
        random.shuffle(out_rows)

    write_manifest(out_manifest, out_rows)

    print("\nDone")
    print(f"  Base rows:     {len(base_rows)}")
    print(f"  Added rows:    {len(new_rows)}")
    print(f"  Output rows:   {len(out_rows)}")
    print(f"  Output file:   {out_manifest}")
    print(f"  Noise chunks:  {len(noise_chunks)} (index at {out_noise_dir / 'noise_chunks_index.jsonl'})")
    if args.mode in {"mix", "both"}:
        print(f"  Mixed clips:   {sum(1 for r in new_rows if r.get('is_noisy_mixed'))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
