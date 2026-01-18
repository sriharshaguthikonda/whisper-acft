"""stage_8_add_random_gain_to_high_score_voices_parallel.py

Multi-threaded version of random_gain_augmentation.py for faster processing.

Goal
----
Augment an existing Whisper-style JSONL manifest by creating *new* audio files that differ only
by random gain (volume), using multiple threads for parallel processing.

Why this works
--------------
Random gain makes the model less sensitive to differences in:
- how loudly the person speaks
- microphone distance
- device / driver gain

Multi-threading benefits
------------------------
- Audio I/O operations are I/O-bound and benefit from parallelization
- Multiple files can be processed simultaneously
- Progress tracking with thread-safe tqdm

Usage examples
--------------
1) Add 1 augmented copy per original row with 8 threads:

# Use 8 worker threads
python stage_8_add_random_gain_to_high_score_voices_parallel.py \
  --in_manifest I:/Record_chunks/pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed.jsonl \
  --out_audio_dir I:/Record_chunks_aug_gain \
  --out_manifest I:/Record_chunks/pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain.jsonl \
  --keep_original \
  --copies 1 \
  --min_db -12 --max_db 12 \
  --p 0.1 \
  --workers 8

  python stage_8_add_random_gain_to_high_score_voices_parallel.py \
    --in_manifest I:/Record_chunks/pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed.jsonl \
    --out_audio_dir I:/Record_chunks_aug_gain \
    --out_manifest I:/Record_chunks/pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain.jsonl \
    --keep_original \
    --copies 1 \
    --min_db -12 --max_db 12 \
    --p 0.1 \
    --mode peak_safe \
    --workers 8

2) Process exactly 0.1% of all files with 4 threads:

  python stage_8_add_random_gain_to_high_score_voices_parallel.py \
    --in_manifest ... \
    --out_audio_dir ... \
    --out_manifest ... \
    --copies 1 \
    --exact_percentage 0.001 \
    --workers 4

"""

from __future__ import annotations

import argparse
import json
import random
import threading
import winsound  # For beep notification
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List

import numpy as np

try:
    import soundfile as sf
except ImportError as e:
    raise SystemExit(
        "Missing dependency: soundfile. Install with: pip install soundfile"
    ) from e

try:
    from tqdm import tqdm
except ImportError as e:
    raise SystemExit(
        "Missing dependency: tqdm. Install with: pip install tqdm"
    ) from e


# ---------------------------
# Audio helpers (thread-safe)
# ---------------------------

def load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio as float32 mono in [-1, 1] (typical for WAV PCM)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        # average channels to mono
        audio = audio.mean(axis=1)
    return audio, int(sr)


def save_wav_float(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use PCM_16 because that's what most Whisper pipelines use.
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_16")


def db_to_amplitude(gain_db: float) -> float:
    return float(10.0 ** (gain_db / 20.0))


def apply_random_gain(
    audio: np.ndarray,
    *,
    min_db: float,
    max_db: float,
    mode: str = "peak_safe",
    clip_prob: float = 0.0,
    peak_target: float = 0.98,
    rng: random.Random,
) -> Tuple[np.ndarray, float, bool]:
    """Apply random gain.

    Parameters
    ----------
    mode:
      - peak_safe: reduce gain if needed so we never clip (best when you only want gain invariance)
      - clamp: apply gain then hard-clamp to [-1, 1] (introduces some clipping-ish distortion)
      - allow_clip: apply gain; with probability clip_prob hard-clamp, else peak_safe

    Returns
    -------
    (aug_audio, gain_db, clipped)
    """
    assert max_db >= min_db
    gain_db = rng.uniform(min_db, max_db)
    g = db_to_amplitude(gain_db)

    if mode not in {"peak_safe", "clamp", "allow_clip"}:
        raise ValueError(f"Unknown mode: {mode}")

    # Decide whether to intentionally clip
    want_clip = False
    if mode == "clamp":
        want_clip = True
    elif mode == "allow_clip":
        want_clip = rng.random() < clip_prob

    x = audio * g

    if want_clip:
        # Hard clipping
        y = np.clip(x, -1.0, 1.0)
        clipped = bool(np.max(np.abs(x)) > 1.0 + 1e-7)
        return y.astype(np.float32), gain_db, clipped

    # peak_safe (default): never clip; if gain would clip, scale down to keep peak under peak_target
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > peak_target and peak > 0.0:
        scale = peak_target / peak
        x = x * scale
        clipped = True  # we prevented clipping by rescaling
    else:
        clipped = False

    return x.astype(np.float32), gain_db, clipped


# ---------------------------
# Manifest helpers
# ---------------------------

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no}: {path}") from e


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_out_name(src: Path, suffix: str) -> str:
    # Keep original stem for traceability; add suffix before extension
    return f"{src.stem}{suffix}{src.suffix}"  # e.g. sent0003__gainp05.wav


# ---------------------------
# Parallel processing
# ---------------------------

def process_single_file(
    idx: int,
    row: Dict[str, Any],
    args: argparse.Namespace,
    out_audio_dir: Path,
    rng_seed: int,
    should_process: bool,
) -> List[Dict[str, Any]]:
    """Process a single file and return augmented rows."""
    if not should_process:
        return []
    
    # Create thread-local random generator
    rng = random.Random(rng_seed + idx)
    
    src_path = Path(row["audio_path"])
    
    # Load audio once
    try:
        audio, sr = load_audio_mono(src_path)
    except Exception as e:
        print(f"Error loading {src_path}: {e}")
        return []
    
    augmented_rows = []
    
    for k in range(args.copies):
        aug_audio, gain_db, clipped_or_scaled = apply_random_gain(
            audio,
            min_db=args.min_db,
            max_db=args.max_db,
            mode=args.mode,
            clip_prob=args.clip_prob,
            peak_target=args.peak_target,
            rng=rng,
        )

        suffix = f"__gain{gain_db:+.1f}dB"
        out_name = make_out_name(src_path, suffix)
        out_path = out_audio_dir / out_name
        
        try:
            save_wav_float(out_path, aug_audio, sr)
        except Exception as e:
            print(f"Error saving {out_path}: {e}")
            continue

        new_row = dict(row)
        new_row["audio_path"] = str(out_path)
        new_row["augmentation"] = {
            "type": "random_gain",
            "gain_db": round(gain_db, 3),
            "mode": args.mode,
            "clipped_or_scaled": bool(clipped_or_scaled),
        }
        augmented_rows.append(new_row)
    
    return augmented_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--out_audio_dir", required=True)
    ap.add_argument("--out_manifest", required=True)

    ap.add_argument("--copies", type=int, default=1, help="How many gain-augmented copies per original row")
    ap.add_argument("--p", type=float, default=0.001, help="Probability to create each copy (per copy)")
    ap.add_argument("--exact_percentage", type=float, help="Exact percentage of files to process (0.0-1.0). Overrides --p if specified.")

    ap.add_argument("--min_db", type=float, default=-12.0)
    ap.add_argument("--max_db", type=float, default=12.0)

    ap.add_argument("--mode", choices=["peak_safe", "clamp", "allow_clip"], default="peak_safe")
    ap.add_argument("--clip_prob", type=float, default=0.15, help="Only used when mode=allow_clip")
    ap.add_argument("--peak_target", type=float, default=0.98)

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--keep_original", action="store_true", help="Also write the original rows into out_manifest")
    ap.add_argument("--workers", type=int, default=None, help="Number of worker threads (default: CPU count)")

    args = ap.parse_args()

    in_manifest = Path(args.in_manifest)
    out_audio_dir = Path(args.out_audio_dir)
    out_manifest = Path(args.out_manifest)

    # Set number of workers
    if args.workers is None:
        import multiprocessing
        args.workers = multiprocessing.cpu_count()
    
    print(f"Using {args.workers} worker threads")

    rng = random.Random(args.seed)

    # First pass: count total rows and collect them
    print("Loading manifest...")
    all_rows = list(iter_jsonl(in_manifest))
    total_rows = len(all_rows)
    
    # Determine processing strategy
    use_exact_percentage = args.exact_percentage is not None
    if use_exact_percentage:
        num_to_process = max(1, int(total_rows * args.exact_percentage))
        # Randomly select which rows to process
        selected_indices = set(rng.sample(range(total_rows), num_to_process))
        print(f"Processing exactly {num_to_process:,} files ({args.exact_percentage*100:.2f}% of {total_rows:,} total)")
    else:
        print(f"Processing with probability {args.p:.3f} per file")
        selected_indices = None

    # Prepare work items
    work_items = []
    for idx, row in enumerate(all_rows):
        should_process = False
        if use_exact_percentage:
            should_process = idx in selected_indices
        else:
            should_process = rng.random() <= args.p
        
        work_items.append((idx, row, should_process))

    # Thread-safe list for results
    out_rows = []
    out_rows_lock = threading.Lock()
    
    # Process files in parallel
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_file,
                idx,
                row,
                args,
                out_audio_dir,
                args.seed,
                should_process
            ): idx for idx, row, should_process in work_items
        }
        
        # Process completed tasks with progress bar
        completed = 0
        with tqdm(total=len(work_items), desc="Processing files", unit="files") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    augmented_rows = future.result()
                    with out_rows_lock:
                        if args.keep_original or not use_exact_percentage or idx in selected_indices:
                            # Add original row if requested
                            if args.keep_original:
                                out_rows.append(all_rows[idx])
                        out_rows.extend(augmented_rows)
                except Exception as e:
                    print(f"Error processing file {idx}: {e}")
                
                completed += 1
                pbar.update(1)
                
                # Update description with progress
                if use_exact_percentage:
                    processed_count = sum(1 for _, _, should in work_items[:completed] if should)
                    pbar.set_description(f"Processing selected file {processed_count}/{num_to_process}")

    # Sort rows to maintain order (optional but helpful)
    if args.keep_original:
        # Separate original and augmented rows
        original_rows = [r for r in out_rows if "augmentation" not in r]
        augmented_rows = [r for r in out_rows if "augmentation" in r]
        # Sort original rows by their original order
        original_indices = {str(row["audio_path"]): i for i, row in enumerate(all_rows)}
        original_rows.sort(key=lambda r: original_indices.get(str(r["audio_path"]), float('inf')))
        out_rows = original_rows + augmented_rows

    write_jsonl(out_manifest, out_rows)

    print(f"Wrote {len(out_rows):,} rows -> {out_manifest}")
    print(f"Audio written under: {out_audio_dir}")
    
    # Beep notification when done
    try:
        winsound.Beep(1000, 500)  # 1000Hz for 500ms
    except Exception:
        pass  # Ignore if beep fails


if __name__ == "__main__":
    main()
