"""rir_convolution_augmentation.py

Goal
----
Augment a Whisper-style JSONL manifest by creating reverberant / far-field-ish versions
of each utterance using Room Impulse Response (RIR) convolution.

Why it helps
------------
Convolving clean/close-talk audio with RIRs is a standard way to inject realistic
reverberation and room coloration, improving robustness to distance/room acoustics.
- audiomentations: ApplyImpulseResponse is explicitly described as a common augmentation
  to add realistic reverb and improve robustness to acoustic environments/distances.
- torchaudio has an augmentation tutorial that includes RIR effects.

Key design choices
------------------
1) Wet/dry mix: y = (1-wet)*x + wet*(x * rir)
   - wet in [0.2, 1.0] is typical.

2) RIR normalisation:
   - We normalise the RIR energy so loud RIR files don’t explode amplitude.

3) Length handling:
   - We keep the output the SAME length as the input (trim tail).
     (Good for fixed-duration chunks; reverb tail won’t increase segment duration.)

4) Clipping protection:
   - Peak-safe scaling to keep output under peak_target.

5) RIR pre-trim (optional but recommended):
   - Many RIRs include leading silence.
   - We can trim leading silence while keeping a small random pre-delay
     to simulate distance.

RIR sources
-----------
- OpenSLR SLR28 includes simulated + real RIRs and is already 16 kHz.
- BUT ReverbDB is another real-RIR dataset (paper + dataset).

Usage
-----


i:\Whisper-training-env\Scripts\python.exe i:\whisper-acft\stage_9_add_reverb.py `
  --in_manifest "I:\Record_chunks\pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain.jsonl" `
  --rir_dir "I:\noise\RIRS_NOISES\real_rirs_isotropic_noises" `
  --out_audio_dir "I:\Record_chunks_aug_rir_real" `
  --out_manifest "I:\Record_chunks\pairs_manifest_sorted_by_scores_english_only_filtered_with_noises_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real.jsonl" `
  --copies 1 --p 0.1 `
  --wet_min 0.25 --wet_max 0.9 `
  --trim_leading_silence --silence_thresh 1e-4 `
  --pre_delay_ms_min 0 --pre_delay_ms_max 30




Tip: Start with p≈0.5–0.8. If you RIR every sample, you can hurt clean-condition accuracy.

Manifest format expected
------------------------
Each line is JSON like:
{"audio_path": "...wav", "raw_transcription": "...", ...}

We preserve all fields and change:
- audio_path (new augmented wav)
- adds: augmentation (metadata)

Dependencies
------------
- numpy
- soundfile
Optional (for resampling RIRs that don’t match SR):
- scipy (preferred) OR torchaudio

If all your RIRs are 16k and your audio is 16k, you won’t need resampling.

"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    print("Warning: tqdm not installed. Install with: pip install tqdm")
    tqdm = None

try:
    import soundfile as sf
except ImportError as e:
    raise SystemExit("Missing dependency: soundfile. Install with: pip install soundfile") from e


# ---------------------------
# I/O
# ---------------------------

def load_audio_mono(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def save_wav_pcm16(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), sr, subtype="PCM_16")


# ---------------------------
# Resampling (only if needed)
# ---------------------------


def try_resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x

    # Prefer scipy if available (good quality, fast).
    try:
        from scipy.signal import resample_poly

        # Use rational approximation for polyphase resampling
        # resample_poly expects int up/down
        from math import gcd

        g = gcd(sr_in, sr_out)
        up = sr_out // g
        down = sr_in // g
        y = resample_poly(x, up, down).astype(np.float32)
        return y
    except Exception:
        pass

    # Fallback: torchaudio (also good), if available.
    try:
        import torch
        import torchaudio

        xt = torch.from_numpy(x).unsqueeze(0)  # [1, T]
        y = torchaudio.functional.resample(xt, sr_in, sr_out)
        return y.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        raise SystemExit(
            f"Need resampling for sr {sr_in}->{sr_out}, but neither scipy nor torchaudio is available.\n"
            "Install one: pip install scipy   OR   pip install torchaudio"
        )


# ---------------------------
# RIR processing
# ---------------------------


def trim_leading_silence_with_predelay(
    rir: np.ndarray,
    *,
    thresh: float,
    pre_delay_samples: int,
) -> np.ndarray:
    if rir.size == 0:
        return rir

    idx = int(np.argmax(np.abs(rir) > thresh))
    if np.max(np.abs(rir)) <= thresh:
        # Entire RIR is basically silence -> keep as-is
        return rir

    start = max(0, idx - pre_delay_samples)
    return rir[start:]


def normalize_rir_energy(rir: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # Remove DC offset
    rir = rir - float(np.mean(rir))
    # Energy norm (L2)
    e = float(np.sqrt(np.sum(rir * rir)) + eps)
    return (rir / e).astype(np.float32)


# ---------------------------
# Convolution
# ---------------------------


def fft_convolve_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Linear convolution via FFT, output trimmed to len(x)."""
    n = int(x.size)
    m = int(h.size)
    if n == 0 or m == 0:
        return x

    # Next power-of-two for speed
    L = 1
    need = n + m - 1
    while L < need:
        L <<= 1

    X = np.fft.rfft(x, n=L)
    H = np.fft.rfft(h, n=L)
    Y = X * H
    y = np.fft.irfft(Y, n=L)

    # Trim to input length
    return y[:n].astype(np.float32)


def peak_safe(audio: np.ndarray, peak_target: float = 0.98, eps: float = 1e-8) -> Tuple[np.ndarray, bool]:
    p = float(np.max(np.abs(audio))) if audio.size else 0.0
    if p <= peak_target + eps:
        return audio, False
    scale = peak_target / max(p, eps)
    return (audio * scale).astype(np.float32), True


# ---------------------------
# Manifest
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


def list_rir_files(rir_dir: Path) -> List[Path]:
    exts = {".wav", ".flac", ".ogg"}
    files = [p for p in rir_dir.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise SystemExit(f"No RIR audio files found under: {rir_dir}")
    return files


def make_out_name(src: Path, suffix: str) -> str:
    return f"{src.stem}{suffix}{src.suffix}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--rir_dir", required=True, help="Folder containing RIR wav/flac etc. (recursively scanned)")
    ap.add_argument("--out_audio_dir", required=True)
    ap.add_argument("--out_manifest", required=True)

    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--p", type=float, default=0.7, help="Probability to create each copy (per copy)")

    ap.add_argument("--wet_min", type=float, default=0.25)
    ap.add_argument("--wet_max", type=float, default=0.90)

    ap.add_argument("--peak_target", type=float, default=0.98)

    ap.add_argument("--trim_leading_silence", action="store_true")
    ap.add_argument("--silence_thresh", type=float, default=1e-4)
    ap.add_argument("--pre_delay_ms_min", type=float, default=0.0)
    ap.add_argument("--pre_delay_ms_max", type=float, default=30.0)

    ap.add_argument("--max_rir_seconds", type=float, default=2.0, help="Cap RIR length to speed up convolution")

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--workers", type=int, default=0, help="Number of worker threads (0 = auto-detect CPU count)")

    args = ap.parse_args()

    in_manifest = Path(args.in_manifest)
    rir_dir = Path(args.rir_dir)
    out_audio_dir = Path(args.out_audio_dir)
    out_manifest = Path(args.out_manifest)

    rng = random.Random(args.seed)

    rir_files = list_rir_files(rir_dir)
    
    # Determine number of workers
    if args.workers <= 0:
        import os
        args.workers = min(32, (os.cpu_count() or 1))  # Don't overwhelm system
    
    print(f"Using {args.workers} worker threads")

    out_rows: List[Dict[str, Any]] = []
    augmented_rows: List[Dict[str, Any]] = []
    
    # Thread-safe storage for results
    results_lock = threading.Lock()
    rir_cache: Dict[Path, Tuple[np.ndarray, int]] = {}
    cache_lock = threading.Lock()

    def get_rir(path: Path) -> Tuple[np.ndarray, int]:
        with cache_lock:
            if path in rir_cache:
                return rir_cache[path]
        rir, sr = load_audio_mono(path)
        with cache_lock:
            rir_cache[path] = (rir, sr)
        return rir, sr
    
    def process_row(row: Dict[str, Any], row_idx: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Process a single manifest row and return original and augmented rows"""
        src_path = Path(row["audio_path"])
        
        # Always keep the original file
        original_row = row
        augmented_rows_for_this = []
        
        try:
            x, sr = load_audio_mono(src_path)
            
            # Create a separate RNG for this thread to avoid race conditions
            thread_rng = random.Random(args.seed + row_idx)
            
            for k in range(args.copies):
                if thread_rng.random() > args.p:
                    continue
                
                rir_path = thread_rng.choice(rir_files)
                h, rir_sr = get_rir(rir_path)
                
                # Resample RIR if needed
                h = try_resample(h, rir_sr, sr)
                
                # Optionally cap RIR length
                max_h = int(max(1, round(args.max_rir_seconds * sr)))
                if h.size > max_h:
                    h = h[:max_h]
                
                # Optional leading silence trim + random pre-delay
                pre_delay_ms = thread_rng.uniform(args.pre_delay_ms_min, args.pre_delay_ms_max)
                pre_delay_samples = int(round((pre_delay_ms / 1000.0) * sr))
                if args.trim_leading_silence:
                    h = trim_leading_silence_with_predelay(
                        h, thresh=args.silence_thresh, pre_delay_samples=pre_delay_samples
                    )
                
                # Normalise RIR energy
                h = normalize_rir_energy(h)
                
                # Convolve
                y_rev = fft_convolve_same(x, h)
                
                # Wet/dry mix
                wet = thread_rng.uniform(args.wet_min, args.wet_max)
                y = (1.0 - wet) * x + wet * y_rev
                y = y.astype(np.float32)
                
                # Peak-safe scaling
                y, scaled = peak_safe(y, peak_target=args.peak_target)
                
                suffix = f"__rir_w{wet:.2f}"
                out_name = make_out_name(src_path, suffix)
                out_path = out_audio_dir / out_name
                save_wav_pcm16(out_path, y, sr)
                
                new_row = dict(row)
                new_row["audio_path"] = str(out_path)
                new_row["augmentation"] = {
                    "type": "rir_convolution",
                    "rir_path": str(rir_path),
                    "wet": round(float(wet), 4),
                    "trim_leading_silence": bool(args.trim_leading_silence),
                    "pre_delay_ms": round(float(pre_delay_ms), 3),
                    "max_rir_seconds": float(args.max_rir_seconds),
                    "peak_safe_scaled": bool(scaled),
                }
                augmented_rows_for_this.append(new_row)
                
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
        
        return original_row, augmented_rows_for_this

    # Process rows in parallel
    all_rows = list(iter_jsonl(in_manifest))
    
    if tqdm:
        progress_bar = tqdm(total=len(all_rows), desc="Processing manifest", unit="files")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(process_row, row, idx): idx for idx, row in enumerate(all_rows)}
        
        # Collect results as they complete
        for future in as_completed(future_to_idx):
            try:
                original_row, augmented_for_row = future.result()
                
                with results_lock:
                    out_rows.append(original_row)
                    augmented_rows.extend(augmented_for_row)
                    
                if tqdm:
                    progress_bar.set_postfix({"augmented": len(augmented_rows)})
                    progress_bar.update(1)
                    
            except Exception as e:
                idx = future_to_idx[future]
                print(f"Error processing row {idx}: {e}")
                if tqdm:
                    progress_bar.update(1)
    
    if tqdm:
        progress_bar.close()
    
    # Progress bar for interleaving augmented files
    if augmented_rows and tqdm:
        print("Interleaving augmented files with originals...")
        interleave_progress = tqdm(total=len(augmented_rows), desc="Interleaving files", unit="files")
    
    # Randomly interleave augmented files with originals
    if augmented_rows:
        rng.shuffle(augmented_rows)
        final_rows = []
        aug_idx = 0
        total_originals = len(out_rows)
        total_augmented = len(augmented_rows)
        
        # Calculate approximate spacing for augmented files
        if total_augmented > 0:
            spacing = max(1, total_originals // total_augmented)
        else:
            spacing = 1
            
        for i, original_row in enumerate(out_rows):
            final_rows.append(original_row)
            
            # Add augmented file at regular intervals with some randomness
            if aug_idx < total_augmented:
                if i % spacing == 0 and rng.random() > 0.3:  # 70% chance to add at spacing interval
                    final_rows.append(augmented_rows[aug_idx])
                    aug_idx += 1
                    if tqdm:
                        interleave_progress.update(1)
                elif i == total_originals - 1:  # Add remaining at the end
                    while aug_idx < total_augmented:
                        final_rows.append(augmented_rows[aug_idx])
                        aug_idx += 1
                        if tqdm:
                            interleave_progress.update(1)
        
        out_rows = final_rows
        
        if tqdm:
            interleave_progress.close()

    # Progress bar for writing output
    if tqdm:
        write_progress = tqdm(total=len(out_rows), desc="Writing manifest", unit="rows")
    
    def write_with_progress(rows):
        for r in rows:
            yield r
            if tqdm:
                write_progress.update(1)
    
    write_jsonl(out_manifest, write_with_progress(out_rows))
    
    if tqdm:
        write_progress.close()

    print(f"\n{'='*60}")
    print(f"RIR files found: {len(rir_files):,}")
    print(f"Total rows processed: {len(out_rows):,}")
    print(f"Augmented files created: {len(augmented_rows):,}")
    print(f"Output manifest: {out_manifest}")
    print(f"Audio files written to: {out_audio_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
