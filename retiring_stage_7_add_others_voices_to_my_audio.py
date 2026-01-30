#!/usr/bin/env python3
"""stage_7_add_others_voices_to_my_audio.py



https://chatgpt.com/g/g-p-6969433d33d4819187ec3158a8f3745f-whisper-training/c/69746fed-bf70-8324-85ae-514723de0e1e

Goal
----
Given a Whisper-style JSONL manifest and speaker scores, augment TARGET audio files
by mixing them with OTHER voice files (poor scores but non-zero):

1) Load TARGET files (high scores) and OTHER files (poor scores but non-zero)
2) Chunk OTHER voice files into <= 30s WAV segments (16 kHz by default)
3) Mix OTHER voice chunks onto TARGET voice files at random SNRs
4) Create new audio files with TARGET transcripts (not OTHER transcripts)
5) Add mixed files to manifest with appropriate metadata

This script is designed for your manifest schema:
  {
    "audio_path": "...wav",
    "raw_transcription": "...",
    "source_audio": "...m4a",
    "chunk_index": 0,
    "chunk_start": 0.0,
    "chunk_end": 2.40,
    "transcript_json": "...json"
  }

Mixed rows will use:
  - TARGET transcript and metadata
  - Additional fields for voice mixing info
  - is_voice_mixed: true

Dependencies
------------
pip install numpy soundfile tqdm librosa
(Optional for resampling) pip install scipy

Usage examples
--------------
# Mix OTHER voices onto TARGET files using I:\Record_others directory:



# Use custom other voices directory:
cd "i:\whisper-acft" && python stage_7_add_others_voices_to_my_audio.py \
  --manifest "i:/Record_chunks/pairs_pending_stereo_english_only_filtered_with_mix.jsonl" \
  --scores_csv "i:/whisper-acft/speaker_sort_scores.csv" \

  --other_voices_dir "I:/Record_others" \
  --out_manifest "i:/Record_chunks/pairs_pending_stereo_english_only_filtered_with_mix_and_others_voices_mixed.jsonl" \
  --out_mix_dir "i:/Record_chunks_voice_mixed" \
  --mix_ratio 0.8 \
  --snr_db_min 5 \
  --snr_db_max 10 \
  --max_bad_to_good_ratio 1.0 --good_floor_db -45 \
  --seed 1337 \
  --shuffle

# With relaxed ducker (allows some other voice during silence):
# Add: --good_floor_db -45

python stage_7_add_others_voices_to_my_audio.py --manifest "i:\Record_chunks\pairs_pending_stereo_english_only_filtered_with_mix.jsonl" --scores_csv "i:\whisper-acft\speaker_sort_scores.csv" --other_voices_dir "i:\Custom\Other\Voices" --out_manifest "i:\Record_chunks\pairs_pending_stereo_english_only_filtered_with_mix_and_others_voices_mixed.jsonl" --out_mix_dir "i:\Record_chunks_voice_mixed" --mix_ratio 0.5 --snr_db_min 5 --snr_db_max 10 --max_bad_to_good_ratio 1.0 --good_floor_db -120 --seed 1337 --shuffle
  
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Try to import librosa for compressed audio support
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not available. M4A/MP3 files will not be supported.")


# -----------------------------
# Utilities
# -----------------------------

def _norm_path(p: str | Path) -> str:
    return str(Path(p))


def _norm_key(p: str | Path) -> str:
    return str(p).lower().replace("\\", "/")


def _stable_hex(s: str, n: int = 10) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _is_audio_file(p: Path) -> bool:
    # soundfile supports wav/flac/ogg and some others depending on libsndfile.
    # librosa adds support for m4a/mp3 through audioread/ffmpeg.
    return p.suffix.lower() in {".wav", ".flac", ".ogg", ".m4a", ".mp3"}


def _rms(x: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x)
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))


def _ensure_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    # Average channels
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

    # Polyphase resampling: up/down ratio reduced by gcd
    g = math.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    y = resample_poly(x, up=up, down=down)
    return y.astype(np.float32, copy=False)


def _safe_write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # PCM_16 keeps files small and consistent
    sf.write(str(path), audio, sr, subtype="PCM_16")


@dataclass(frozen=True)
class OtherVoice:
    path: Path
    score: float


def load_target_and_other_from_csv(scores_csv: Path) -> Tuple[set[str], List[OtherVoice]]:
    """Reads speaker_sort_scores.csv and returns:
    - target_files: set of normalised audio paths marked TARGET
    - other_files:  list of OTHER voices with score>0 (lower score = more OTHER)

    NOTE: We keep paths as in CSV; they must exist on disk.
    """
    target_files: set[str] = set()
    others: List[OtherVoice] = []

    with scores_csv.open("r", encoding="utf-8") as f:
        # Skip header line
        next(f, None)
        # tolerant parsing: assume first 3 columns are file, score, decision
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
                    others.append(OtherVoice(Path(file_path), score))

    others.sort(key=lambda o: o.score)
    return target_files, others


def scan_other_voices_dir(other_voices_dir: Path, default_score: float = 0.5) -> List[OtherVoice]:
    out: List[OtherVoice] = []
    for p in other_voices_dir.rglob("*"):
        if p.is_file() and _is_audio_file(p):
            out.append(OtherVoice(p, float(default_score)))
    # stable order
    out.sort(key=lambda o: _norm_key(o.path))
    return out


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


def read_full_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Read whole file to mono float32."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    audio = _ensure_mono(audio)
    return audio, int(sr)


def read_random_segment(path: Path, target_duration_sec: float, rng: random.Random) -> Tuple[np.ndarray, int, float]:
    """Read a random segment (in frames) using optimal library for format.

    Returns: (segment_audio, sr, start_sec)
    """
    suffix = path.suffix.lower()
    
    # Use librosa for compressed formats (M4A, MP3)
    if suffix in {'.m4a', '.mp3'} and HAS_LIBROSA:
        try:
            # Get duration first without loading full audio
            duration = librosa.get_duration(path=str(path))
            if duration <= target_duration_sec:
                # Load entire file if shorter than target duration
                audio, sr = librosa.load(str(path), sr=None, mono=True)
                return audio, sr, 0.0
            
            # Calculate random start time
            max_start = duration - target_duration_sec
            start_sec = rng.uniform(0, max_start)
            
            # Load segment with offset and duration
            audio, sr = librosa.load(
                str(path), 
                sr=None, 
                mono=True,
                offset=start_sec,
                duration=target_duration_sec
            )
            return audio, sr, start_sec
            
        except Exception as e:
            print(f"Warning: librosa failed to load {path}: {e}")
            # Fall back to soundfile if librosa fails
    
    # Use soundfile for uncompressed formats (WAV, FLAC, OGG) - faster
    try:
        with sf.SoundFile(str(path), "r") as f:
            sr = int(f.samplerate)
            n_frames = int(f.frames)
            seg_frames = max(1, int(round(target_duration_sec * sr)))

            if seg_frames >= n_frames:
                f.seek(0)
                audio = f.read(n_frames, dtype="float32", always_2d=False)
                audio = _ensure_mono(audio)
                return audio, sr, 0.0

            max_start = n_frames - seg_frames
            start_frame = rng.randint(0, max_start)
            f.seek(start_frame)
            audio = f.read(seg_frames, dtype="float32", always_2d=False)
            audio = _ensure_mono(audio)
            return audio, sr, float(start_frame) / float(sr)
            
    except Exception as e:
        raise RuntimeError(f"Failed to load audio with both librosa and soundfile: {path} - {e}")


def build_weighted_sampler(others: Sequence[OtherVoice], eps: float = 1e-3) -> Tuple[List[OtherVoice], List[float]]:
    """Weights: inverse of score (lower score -> higher probability)."""
    items = list(others)
    weights: List[float] = []
    for o in items:
        s = float(o.score)
        weights.append(1.0 / max(s, eps))
    return items, weights


def pick_other(items: Sequence[OtherVoice], weights: Sequence[float], rng: random.Random) -> OtherVoice:
    # random.choices supports weights
    return rng.choices(list(items), weights=list(weights), k=1)[0]


def mix_one(
    *,
    row: Dict,
    row_uid: int,
    out_mix_dir: Path,
    target_sr: int,
    snr_db_min: float,
    snr_db_max: float,
    allow_overwrite: bool,
    others_items: Sequence[OtherVoice],
    others_weights: Sequence[float],
    seed: int,
    max_bad_to_good_ratio: float = 1.0,
    good_floor_db: float = -120.0,
) -> Optional[Dict]:
    """Mix one OTHER voice segment on top of TARGET speech for one row."""

    local_seed = int(seed) ^ (row_uid * 10007) ^ int(_stable_hex(_norm_key(row.get("audio_path", "")), 8), 16)
    rng = random.Random(local_seed)

    target_ap = Path(row.get("audio_path", ""))
    if not target_ap.exists():
        return None

    try:
        # TARGET speech
        target_speech, sr_in = read_full_audio(target_ap)
        target_speech = _resample_if_needed(target_speech, sr_in, target_sr)
        target_len = len(target_speech)
        if target_len < int(0.01 * target_sr):
            return None
        target_dur = float(target_len) / float(target_sr)

        # Pick OTHER file
        other = pick_other(others_items, others_weights, rng)
        other_path = other.path

        if not other_path.exists():
            return None

        # Read random segment of OTHER matching target duration (random-access)
        other_seg, other_sr, other_start_sec = read_random_segment(other_path, target_dur, rng)
        other_seg = _resample_if_needed(other_seg, other_sr, target_sr)

        # ensure same length
        if len(other_seg) < target_len:
            other_seg = np.pad(other_seg, (0, target_len - len(other_seg)), mode="constant")
        elif len(other_seg) > target_len:
            other_seg = other_seg[:target_len]

        # SNR scaling
        snr_db = rng.uniform(float(snr_db_min), float(snr_db_max))
        rs = _rms(target_speech)
        rn = _rms(other_seg)
        target_rn = rs / (10.0 ** (snr_db / 20.0))
        gain = target_rn / max(rn, 1e-12)
        other_scaled = other_seg * gain
        
        # Apply instantaneous ducker to ensure other voice never exceeds target speech
        from audio_instant_ducker_utils import duck_bad_under_good
        other_scaled = duck_bad_under_good(
            other_scaled,
            good=target_speech,
            max_ratio=max_bad_to_good_ratio,
            good_floor_db=good_floor_db,
        )
        
        mixed = target_speech + other_scaled

        # avoid clipping
        peak = float(np.max(np.abs(mixed)))
        if peak > 0.99:
            mixed = mixed * (0.99 / peak)

        # output filename
        safe_target = "".join(c if c.isalnum() or c in "-_" else "_" for c in target_ap.stem)
        safe_other = "".join(c if c.isalnum() or c in "-_" else "_" for c in other_path.stem)
        tid = _stable_hex(_norm_key(target_ap), 6)
        oid = _stable_hex(_norm_key(other_path), 6)

        out_name = (
            f"{safe_target}__t{tid}__row{row_uid:07d}__voice_mixed__"
            f"snr{snr_db:.1f}dB__oscore{other.score:.4f}__{safe_other}_o{oid}.wav"
        )
        out_path = out_mix_dir / out_name

        if (not out_path.exists()) or allow_overwrite:
            _safe_write_wav(out_path, mixed.astype(np.float32, copy=False), target_sr)

        # New manifest row keeps TARGET transcript
        new_row = dict(row)
        new_row["audio_path"] = _norm_path(out_path)
        new_row["orig_audio_path"] = row.get("audio_path")
        new_row["aug_voice_source"] = _norm_path(other_path)
        new_row["aug_voice_score"] = float(other.score)
        new_row["aug_snr_db"] = float(snr_db)
        new_row["aug_voice_start_sec"] = float(other_start_sec)
        new_row["aug_voice_dur_sec"] = float(target_dur)
        new_row["is_voice_mixed"] = True
        return new_row

    except Exception:
        # keep it quiet; caller will count failures
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, help="Input JSONL manifest path")
    p.add_argument("--scores_csv", required=True, help="Speaker sort scores CSV with TARGET/OTHER files")
    p.add_argument(
        "--other_voices_dir",
        default=r"I:\\Record_others",
        help="Directory with other voice audio files; if missing, fall back to CSV OTHER rows",
    )

    p.add_argument("--out_manifest", required=True, help="Output JSONL manifest path")
    p.add_argument("--out_mix_dir", required=True, help="Where to write voice-mixed copies")

    p.add_argument("--target_sr", type=int, default=16000)

    p.add_argument("--mix_ratio", type=float, default=0.5, help="Fraction of TARGET rows to create voice-mixed copies for")
    p.add_argument("--snr_db_min", type=float, default=5.0)
    p.add_argument("--snr_db_max", type=float, default=20.0)

    p.add_argument("--shuffle", action="store_true", help="Shuffle output manifest rows")
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--overwrite", action="store_true", help="Allow overwriting generated audio files")
    p.add_argument("--workers", type=int, default=0, help="Thread workers for mixing (0=auto)")
    
    # Instantaneous ducker parameters
    p.add_argument('--max_bad_to_good_ratio', type=float, default=1.0, help='Maximum ratio of bad to good signal at any sample (1.0 = bad never louder)')
    p.add_argument('--good_floor_db', type=float, default=-120.0, help='Floor for good signal in dB (strict: -120, relaxed: -45)')

    args = p.parse_args()

    manifest_path = Path(args.manifest)
    scores_csv = Path(args.scores_csv)
    other_voices_dir = Path(args.other_voices_dir)
    out_manifest = Path(args.out_manifest)
    out_mix_dir = Path(args.out_mix_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not scores_csv.exists():
        raise FileNotFoundError(f"Scores CSV not found: {scores_csv}")

    out_mix_dir.mkdir(parents=True, exist_ok=True)

    # Load targets + (fallback) others from CSV
    target_files, csv_others = load_target_and_other_from_csv(scores_csv)

    # Prefer directory scan for others if it exists and has audio
    others: List[OtherVoice]
    if other_voices_dir.exists():
        dir_others = scan_other_voices_dir(other_voices_dir, default_score=0.5)
        if dir_others:
            others = dir_others
        else:
            others = csv_others
    else:
        others = csv_others

    if not target_files:
        raise RuntimeError("No TARGET entries found in scores CSV.")
    if not others:
        raise RuntimeError("No OTHER voice files found (directory scan empty and CSV OTHER empty).")

    # Set up workers
    workers = int(args.workers)
    if workers <= 0:
        workers = min(32, (os.cpu_count() or 4) + 4)

    # Build sampler
    others_items, others_weights = build_weighted_sampler(others)

    # Load manifest
    base_rows = load_manifest(manifest_path)
    if not base_rows:
        raise RuntimeError(f"Manifest is empty: {manifest_path}")

    # Filter to TARGET rows in manifest
    target_rows: List[Tuple[int, Dict]] = []
    for idx, row in enumerate(base_rows):
        ap = row.get("audio_path")
        if ap and _norm_key(ap) in target_files:
            target_rows.append((idx, row))

    if not target_rows:
        raise RuntimeError("No TARGET rows found in manifest that match scores CSV paths.")

    # Choose subset to mix
    rng_master = random.Random(int(args.seed))
    n_to_mix = int(round(len(target_rows) * float(args.mix_ratio)))
    n_to_mix = max(0, min(len(target_rows), n_to_mix))
    if n_to_mix == 0:
        # just copy original manifest
        write_manifest(out_manifest, base_rows)
        print("mix_ratio produced 0 mixes; wrote base manifest only.")
        return 0

    chosen = set(rng_master.sample(range(len(target_rows)), k=n_to_mix))
    tasks: List[Tuple[int, Dict]] = [target_rows[i] for i in sorted(chosen)]

    # Run mixing in parallel (threads)
    new_rows: List[Dict] = []
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for j, (orig_idx, row) in enumerate(tasks):
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
                    others_items=others_items,
                    others_weights=others_weights,
                    seed=int(args.seed),
                    max_bad_to_good_ratio=float(args.max_bad_to_good_ratio),
                    good_floor_db=float(args.good_floor_db),
                )
            )

        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"Mixing OTHER voices (threads={workers})", unit="clip"):
            r = fut.result()
            if r is None:
                failed += 1
            else:
                new_rows.append(r)

    # Combine rows
    out_rows = list(base_rows) + list(new_rows)
    if args.shuffle:
        rng_master.shuffle(out_rows)

    write_manifest(out_manifest, out_rows)

    print("\nDone")
    print(f"  Base rows:                 {len(base_rows)}")
    print(f"  TARGET rows in manifest:   {len(target_rows)}")
    print(f"  Requested mixes:           {len(tasks)}")
    print(f"  Successful mixes:          {len(new_rows)}")
    print(f"  Failed mixes:              {failed}")
    print(f"  Output rows:               {len(out_rows)}")
    print(f"  Output manifest:           {out_manifest}")
    print(f"  Mixed audio dir:           {out_mix_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
