"""sort_audio_files_by_speaker_v3_resumable.py

Same goal as v2, but with RESUMABLE capability by default.

Key changes:
- Step-by-step status messages (model loading / ref embedding / scoring)
- Progress mode: auto|tqdm|print|none
  - 'print' works even when tqdm bars are suppressed by an IDE console
- Optional per-file logging every N files
- **RESUMABLE by default** - saves state and can resume from interruptions
- State file tracks processed files and results
- Auto-resumes from where it left off



Usage (Windows PowerShell)
-------------------------


$refs = @(Get-ChildItem "I:\Record_only_by_harsha" -Recurse -File  -Include *.wav,*.m4a,*.mp3,*.flac,*.aac,*.ogg,*.opus,*.mka,*.mp4 | ForEach-Object { $_.FullName })
$refs
$refs.Count
I:\Whisper-training-env\Scripts\Activate.ps1
python "i:\whisper-acft\stage_3_sort_audio_files_by_speaker.py" `
  --in "I:\Record_chunks" `
  --ref $refs[0] $refs[1] $refs[2] $refs[3] $refs[4] `
  --target_out "I:\Record_chunks_sorted\target" `
  --other_out "I:\Record_chunks_sorted\other" `
  --device cuda `
  --dry_run `
  --progress print `
  --batch_size 32 `
  --workers 4



  python -u sort_audio_files_by_speaker_v3_resumable.py --in ./audio --ref ./refs/doctor1.m4a --target_out ./keep --other_out ./others --dry_run --progress print

Then actually move:
  python -u sort_audio_files_by_speaker_v3_resumable.py --in ./audio --ref ./refs/doctor1.m4a --target_out ./keep --other_out ./others --move --threshold 0.65 --progress print

Resume after interruption:
  python -u sort_audio_files_by_speaker_v3_resumable.py --in ./audio --ref ./refs/doctor1.m4a --target_out ./keep --other_out ./others --dry_run --progress print
  # Script auto-detects and resumes from state file

Notes
-----
- File-level speaker verification (not diarization).
- If your files contain both doctor and patient, classification will be less reliable.
- State file: speaker_sort_state.json (auto-created)
- CSV file: speaker_sort_scores.csv (updated incrementally)

"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".aac", ".flac", ".ogg", ".opus", ".mka", ".mp4"}


def _which(exe: str) -> Optional[str]:
    return shutil.which(exe)


def require_ffmpeg() -> None:
    if _which("ffmpeg") is None:
        raise SystemExit(
            "ffmpeg not found on PATH. Install ffmpeg and try again.\n"
            "Windows: winget install Gyan.FFmpeg  (or use choco/scoop)\n"
            "macOS: brew install ffmpeg\n"
            "Linux: sudo apt-get install ffmpeg"
        )


def ffmpeg_to_wav_16k_mono(src: Path, dst: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(dst),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}:\n{p.stderr}")


def load_audio_direct(audio_path: Path) -> Optional[np.ndarray]:
    """Load audio directly using librosa if possible, faster than ffmpeg."""
    if not HAS_LIBROSA:
        return None
    
    try:
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        return audio
    except Exception:
        return None


def get_duration_fast(audio_path: Path) -> Optional[float]:
    """Get duration quickly using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-show_entries", 
            "format=duration", "-of", "csv=p=0", str(audio_path)
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return None


def iter_audio_files(in_path: Path) -> List[Path]:
    if in_path.is_file():
        return [in_path]
    return sorted([p for p in in_path.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS])


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def audio_duration_seconds(wav_path: Path) -> float:
    info = sf.info(str(wav_path))
    if info.samplerate <= 0:
        return 0.0
    return float(info.frames) / float(info.samplerate)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_move_or_copy(src: Path, dst: Path, do_move: bool) -> None:
    ensure_dir(dst.parent)
    if do_move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def compute_reference_embedding(ref_files_wav16: List[Path], inference) -> np.ndarray:
    embs = []
    for rf in ref_files_wav16:
        e = inference(str(rf))  # (1, D)
        e = np.asarray(e).reshape(1, -1)
        e = l2_normalize(e)[0]
        embs.append(e)
    if not embs:
        raise ValueError("No reference embeddings computed. Provide at least one reference file.")
    ref = np.mean(np.stack(embs, axis=0), axis=0)
    return l2_normalize(ref.reshape(1, -1))[0]


def process_audio_batch(audio_files: List[Path], tmp_dir: Path, batch_size: int = 32) -> List[tuple]:
    """Process audio files in batches for better GPU utilization."""
    results = []
    
    def convert_file(af: Path) -> tuple:
        """Convert single file and return (path, audio_array or None, error)"""
        # Try direct loading first
        audio = load_audio_direct(af)
        if audio is not None:
            return af, audio, None
        
        # Fallback to ffmpeg conversion
        wav_path = tmp_dir / f"in_{af.stem}_{abs(hash(str(af))) % 10_000_000}.wav"
        try:
            ffmpeg_to_wav_16k_mono(af, wav_path)
            audio, _ = sf.read(str(wav_path), dtype='float32')
            return af, audio, None
        except Exception as e:
            return af, None, str(e)
    
    # Process in parallel using ThreadPool for I/O bound conversion
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(convert_file, af) for af in audio_files]
        for future in as_completed(futures):
            results.append(future.result())
    
    return results


def compute_batch_embeddings(audio_arrays: List[np.ndarray], inference, device: torch.device, batch_size: int = 32) -> Tuple[List[np.ndarray], List[str]]:
    """Compute embeddings for a batch of audio arrays with OOM protection."""
    if not audio_arrays:
        return [], []
    
    # Convert to tensors and batch
    tensors = []
    for audio in audio_arrays:
        if audio is None:
            continue
        tensor = torch.from_numpy(audio).float()
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)  # Add channel dimension if needed
        tensors.append(tensor)
    
    if not tensors:
        return [], []
    
    # Try processing with current batch size, reduce if OOM
    current_batch_size = len(tensors)
    embeddings_list = []
    error_messages = []
    
    while current_batch_size > 0:
        try:
            # Process in sub-batches
            sub_embeddings = []
            for i in range(0, len(tensors), current_batch_size):
                sub_batch = tensors[i:i + current_batch_size]
                
                # Pad sequences to same length if needed
                max_len = max(t.shape[-1] for t in sub_batch)
                padded_tensors = []
                for t in sub_batch:
                    if t.shape[-1] < max_len:
                        padding = torch.zeros(1, max_len - t.shape[-1])
                        t = torch.cat([t, padding], dim=-1)
                    padded_tensors.append(t)
                
                batch_tensor = torch.stack(padded_tensors).to(device)
                
                with torch.no_grad():
                    embeddings = inference.model(batch_tensor)
                    if len(embeddings.shape) == 3:
                        embeddings = embeddings.mean(dim=1)  # Average over time dimension
                    
                    sub_embeddings.extend(embeddings.cpu().numpy())
            
            embeddings_list = sub_embeddings
            break  # Success!
            
        except torch.cuda.OutOfMemoryError:
            # Reduce batch size and retry
            current_batch_size = current_batch_size // 2
            if current_batch_size == 0:
                error_messages.append("OOM: Could not process even smallest batch")
                break
            print(f"OOM detected, reducing batch size to {current_batch_size}...", flush=True)
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
        except Exception as e:
            error_messages.append(f"Error computing embeddings: {str(e)}")
            break
    
    return embeddings_list, error_messages


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # assumes both l2-normalized


def load_state(state_file: Path) -> Dict[str, any]:
    """Load existing state if it exists."""
    if state_file.exists():
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load state file {state_file}: {e}", flush=True)
    return {}


def save_state(state_file: Path, state: Dict[str, any]) -> None:
    """Save current state to file."""
    try:
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save state file {state_file}: {e}", flush=True)


def load_existing_csv(csv_path: Path) -> Dict[str, Dict[str, str]]:
    """Load existing CSV results into a dict for fast lookup."""
    results = {}
    if csv_path.exists():
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results[row['file']] = row
        except Exception as e:
            print(f"Warning: Could not load existing CSV {csv_path}: {e}", flush=True)
    return results


def append_csv_row(csv_path: Path, row: Dict[str, str]) -> None:
    """Append a single row to CSV file."""
    file_exists = csv_path.exists()
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['file', 'score', 'decision', 'reason'])
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"Warning: Could not append to CSV {csv_path}: {e}", flush=True)


def make_progress(progress_mode: str, total: int) -> Callable[[int, str], None]:
    """Return a function update(i, msg) that reports progress."""
    progress_mode = progress_mode.lower()
    if progress_mode == "none":
        return lambda i, msg: None

    if progress_mode == "print":
        def upd(i: int, msg: str) -> None:
            # prints are line-based (works in any console)
            print(f"[{i}/{total}] {msg}", flush=True)
        return upd

    # tqdm or auto
    use_tqdm = progress_mode == "tqdm" or (progress_mode == "auto" and sys.stderr.isatty())
    if not use_tqdm:
        # fallback to print if stderr is not a tty
        def upd(i: int, msg: str) -> None:
            print(f"[{i}/{total}] {msg}", flush=True)
        return upd

    try:
        from tqdm import tqdm
    except Exception:
        def upd(i: int, msg: str) -> None:
            print(f"[{i}/{total}] {msg}", flush=True)
        return upd

    bar = tqdm(total=total, desc="Scoring", unit="file", dynamic_ncols=True, file=sys.stderr)

    last_i = 0

    def upd(i: int, msg: str) -> None:
        nonlocal last_i
        # i is 1-based
        step = max(0, i - last_i)
        if step:
            bar.update(step)
            last_i = i
        bar.set_postfix_str(msg[:120])

    return upd


def main() -> None:
    require_ffmpeg()

    ap = argparse.ArgumentParser(description="Sort audio files by whether they match a target speaker.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input file or directory containing many audio files")
    ap.add_argument("--ref", dest="ref_files", nargs="+", required=True, help="Reference audio files for the target speaker")
    ap.add_argument("--target_out", required=True, help="Directory to put files matching target speaker")
    ap.add_argument("--other_out", required=True, help="Directory to put all other files")
    ap.add_argument("--threshold", type=float, default=0.5, help="Cosine similarity threshold for match (tune this)")
    ap.add_argument("--min_seconds", type=float, default=2.0, help="Route to OTHER files shorter than this")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Use cuda if you have a GPU")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    ap.add_argument("--copy", action="store_true", help="Copy files (default if neither --move nor --copy set)")
    ap.add_argument("--dry_run", action="store_true", help="Do not move/copy; just print decisions and write CSV")
    ap.add_argument("--progress", choices=["auto", "tqdm", "print", "none"], default="auto", help="Progress display mode")
    ap.add_argument("--log_every", type=int, default=0, help="Also print a line every N files (0=off)")
    ap.add_argument(
        "--hf_token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Optional Hugging Face token (only needed if a model requires it).",
    )
    ap.add_argument("--batch_size", type=int, default=32, help="Initial batch size for processing (auto-reduced on OOM)")
    ap.add_argument("--workers", type=int, default=4, help="Number of worker threads for audio conversion (default: 4)")
    ap.add_argument("--min_batch_size", type=int, default=1, help="Minimum batch size to attempt (default: 1)")
    ap.add_argument(
        "--state_file",
        default="speaker_sort_state.json",
        help="State file for resuming (default: speaker_sort_state.json)",
    )
    ap.add_argument(
        "--no_resume",
        action="store_true",
        help="Force restart from beginning (ignore existing state)",
    )

    args = ap.parse_args()

    in_path = Path(args.in_path)
    ref_files = [Path(p) for p in args.ref_files]
    target_out = Path(args.target_out)
    other_out = Path(args.other_out)

    do_move = bool(args.move)
    if args.copy:
        do_move = False

    if not in_path.exists():
        raise SystemExit(f"Input path not found: {in_path}")

    for rf in ref_files:
        if not rf.exists():
            raise SystemExit(f"Reference file not found: {rf}")

    audio_files = iter_audio_files(in_path)
    if not audio_files:
        raise SystemExit(f"No audio files found under: {in_path}")

    # Initialize resumable state
    state_file = Path(args.state_file)
    csv_path = Path.cwd() / "speaker_sort_scores.csv"
    
    # Load existing state and results
    existing_state = load_state(state_file) if not args.no_resume else {}
    
    # Check if we're resuming
    processed_files = set(existing_state.get('processed_files', []))
    
    if processed_files and not args.no_resume:
        print(f"Resuming from previous run: {len(processed_files)} files already processed.", flush=True)
        print(f"State file: {state_file}", flush=True)
    
    # Filter out already processed files
    remaining_files = [f for f in audio_files if str(f) not in processed_files]
    
    if not remaining_files:
        print("All files have already been processed!", flush=True)
        return
    
    print(f"Found {len(audio_files)} total audio file(s).", flush=True)
    print(f"Processing {len(remaining_files)} remaining file(s).", flush=True)

    # Lazy import to keep start-up errors clean
    print("Loading speaker embedding model... (first run may download weights)", flush=True)
    import torch
    from pyannote.audio import Model, Inference

    device = torch.device(args.device)

    model = Model.from_pretrained(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        token=args.hf_token or None,
    )
    inference = Inference(model, window="whole")
    inference.to(device)

    ensure_dir(target_out)
    ensure_dir(other_out)

    progress_upd = make_progress(args.progress, total=len(remaining_files))
    
    # Initialize state tracking
    current_processed_files = list(processed_files)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        print("Converting reference audio and computing target voice profile...", flush=True)
        ref_wavs: List[Path] = []
        for rf in ref_files:
            out = tmp / f"ref_{rf.stem}.wav"
            ffmpeg_to_wav_16k_mono(rf, out)
            ref_wavs.append(out)

        ref_emb = compute_reference_embedding(ref_wavs, inference)

        # Process remaining files in batches
        initial_batch_size = args.batch_size
        total_batches = (len(remaining_files) + initial_batch_size - 1) // initial_batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * initial_batch_size
            end_idx = min(start_idx + initial_batch_size, len(remaining_files))
            batch_files = remaining_files[start_idx:end_idx]
            
            print(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_files)} files)...", flush=True)
            
            # Pre-filter by duration to skip short files early
            filtered_files = []
            for af in batch_files:
                duration = get_duration_fast(af)
                if duration is not None and duration < float(args.min_seconds):
                    # Short file - add result directly
                    row_result = {
                        "file": str(af),
                        "score": "",
                        "decision": "OTHER",
                        "reason": f"too_short({duration:.2f}s)",
                    }
                    append_csv_row(csv_path, row_result)
                    current_processed_files.append(str(af))
                    progress_upd(start_idx + batch_files.index(af) + 1, af.name)
                else:
                    filtered_files.append(af)
            
            if not filtered_files:
                continue  # All files in this batch were too short
            
            # Process audio batch
            batch_results = process_audio_batch(filtered_files, tmp, args.workers)
            
            # Separate successful conversions from errors
            successful = []
            audio_arrays = []
            
            for af, audio, error in batch_results:
                if error is not None:
                    # Conversion error
                    row_result = {
                        "file": str(af),
                        "score": "",
                        "decision": "ERROR_CONVERT",
                        "reason": error,
                    }
                    append_csv_row(csv_path, row_result)
                    current_processed_files.append(str(af))
                    progress_upd(start_idx + batch_files.index(af) + 1, af.name)
                else:
                    successful.append(af)
                    audio_arrays.append(audio)
            
            if not audio_arrays:
                continue  # No successful conversions in this batch
            
            # Compute embeddings for the entire batch with OOM protection
            embeddings, error_messages = compute_batch_embeddings(audio_arrays, inference, device, len(filtered_files))
            
            if error_messages:
                # Handle embedding errors
                for i, af in enumerate(successful):
                    if i < len(error_messages) and error_messages[i]:
                        row_result = {
                            "file": str(af),
                            "score": "",
                            "decision": "ERROR_EMBEDDING",
                            "reason": error_messages[i],
                        }
                        append_csv_row(csv_path, row_result)
                        current_processed_files.append(str(af))
                        progress_upd(start_idx + batch_files.index(af) + 1, af.name)
                continue  # Skip to next batch
            
            # Process results
            for i, (af, embedding) in enumerate(zip(successful, embeddings)):
                if embedding is None:
                    # Skip if embedding failed
                    continue
                    
                emb = l2_normalize(embedding.reshape(1, -1))[0]
                s = cosine_sim(ref_emb, emb)
                score = f"{s:.6f}"
                decision = "TARGET" if s >= float(args.threshold) else "OTHER"
                reason = ""
                
                row_result = {
                    "file": str(af),
                    "score": score,
                    "decision": decision,
                    "reason": reason,
                }
                
                append_csv_row(csv_path, row_result)
                current_processed_files.append(str(af))
                
                # Update progress
                file_idx = start_idx + batch_files.index(af) + 1
                progress_upd(file_idx, af.name)
                
                if args.log_every and file_idx % args.log_every == 0:
                    print(f"[{file_idx}/{len(remaining_files)}] {af.name} -> {decision} score={score} {reason}", flush=True)
                
                # Handle file moving/copying
                if not args.dry_run:
                    rel = af.relative_to(in_path) if in_path.is_dir() else Path(af.name)
                    dst = (target_out / rel) if decision == "TARGET" else (other_out / rel)
                    safe_move_or_copy(af, dst, do_move=do_move)
            
            # Save checkpoint after each batch
            save_state(state_file, {'processed_files': current_processed_files})
            if batch_idx % 5 == 0:  # Less frequent checkpoint messages
                print(f"Checkpoint saved: {len(current_processed_files)}/{len(audio_files)} files processed", flush=True)

        # Final state save
        save_state(state_file, {'processed_files': current_processed_files})
        print(f"Final state saved: {len(current_processed_files)} files processed", flush=True)

    print("\nDone.", flush=True)
    print(f"CSV log: {csv_path}", flush=True)
    if args.dry_run:
        print("Dry-run only: no files moved/copied.", flush=True)
    else:
        print("Files have been sorted.", flush=True)
        print("If results look wrong: re-run with --dry_run and tune --threshold.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
