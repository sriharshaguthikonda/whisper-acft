r"""stage_3_sort_audio_files_by_speaker_target_other.py

Sort audio files by whether they match a TARGET speaker, optionally using an OTHERS folder
as a negative reference cohort.

✅ Backwards compatible with downstream pipeline expectations:
- CSV first columns stay EXACTLY: file,score,decision,reason
- Extra columns are appended AFTER reason.

If you provide --other_ref_dir:
- score_target = cosine(target_centroid, emb)
- score_other_max = max cosine(other_ref_i, emb)
- compound score (written to CSV as "score") = score_target - score_other_max
- decision = TARGET if compound_score >= --threshold else OTHER

If you do NOT provide --other_ref_dir:
- score (CSV) = score_target
- decision = TARGET if score_target >= --threshold else OTHER

Typical threshold guidance:
- Without others: 0.50–0.75 (depends heavily on your data)
- With others (margin): often 0.00–0.20 (you must tune)

Usage (PowerShell)
------------------
I:\Whisper-training-env\Scripts\Activate.ps1

python "i:\whisper-acft\stage_3_sort_audio_files_by_speaker_target_other.py" 
  --in "I:\Record_chunks" 
  --target_ref_dir "I:\Record_only_by_harsha" 
  --other_ref_dir "I:\Record_others_compacted" 
  --target_out "I:\Record_chunks_sorted\target" 
  --other_out "I:\Record_chunks_sorted\other" 
  --threshold 0.10 
  --device cuda 
  --dry_run 
  --progress print 
  --batch_size 1 `
  --workers 1

Then run again without --dry_run to actually move/copy.

Resume:
- State file default: speaker_sort_state.json
- CSV default: speaker_sort_scores.csv (written in the CURRENT working directory)

"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
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
MODEL_ID = "pyannote/wespeaker-voxceleb-resnet34-LM"


def beep() -> None:
    """Audible notification when done."""
    try:
        import winsound

        winsound.Beep(1000, 250)
        winsound.Beep(1400, 200)
    except Exception:
        # terminal bell fallback
        print("\a", end="", flush=True)


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
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
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


DIRECT_LIBROSA_EXTS = {".wav", ".flac"}  # boring + reliable on Windows


def load_audio_direct(audio_path: Path) -> Optional[np.ndarray]:
    """
    Load audio directly using librosa ONLY for formats that SoundFile/libsndfile
    tends to handle cleanly.

    For MP3/M4A/etc we return None so we fall back to ffmpeg.
    """
    if (not HAS_LIBROSA) or (audio_path.suffix.lower() not in DIRECT_LIBROSA_EXTS):
        return None
    try:
        audio, _sr = librosa.load(str(audio_path), sr=16000, mono=True)
        if audio is None or len(audio) == 0:
            return None
        return np.asarray(audio, dtype=np.float32)
    except Exception:
        return None


def _tmp_wav_path(tmp_dir: Path, af: Path) -> Path:
    """Stable temp path without collisions (avoid built-in hash() + modulo)."""
    h = hashlib.sha1(str(af).encode("utf-8", "ignore")).hexdigest()[:16]
    return tmp_dir / f"in_{h}.wav"


def get_duration_fast(audio_path: Path) -> Optional[float]:
    """Get duration quickly using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(audio_path),
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


def expand_files_or_dirs(items: List[Path]) -> List[Path]:
    out: List[Path] = []
    for p in items:
        if p.is_dir():
            out.extend(iter_audio_files(p))
        else:
            out.append(p)
    # filter + dedupe
    seen = set()
    filtered: List[Path] = []
    for p in out:
        if not p.exists():
            continue
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            sp = str(p)
            if sp not in seen:
                seen.add(sp)
                filtered.append(p)
    return filtered


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.maximum(denom, eps)
    return x / denom


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_move_or_copy(src: Path, dst: Path, do_move: bool) -> None:
    ensure_dir(dst.parent)
    if do_move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # assumes both l2-normalized


def load_state(state_file: Path) -> Dict[str, object]:
    if state_file.exists():
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load state file {state_file}: {e}", flush=True)
    return {}


def save_state(state_file: Path, state: Dict[str, object]) -> None:
    try:
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Could not save state file {state_file}: {e}", flush=True)


def compute_signature(payload: Dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def ensure_csv_header(csv_path: Path, fieldnames: List[str]) -> None:
    """If csv exists with different header, rewrite it with new header, padding missing fields."""
    if not csv_path.exists():
        return
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if not existing_header:
            return
        if existing_header == fieldnames:
            return

        # Rewrite
        tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
        with csv_path.open("r", encoding="utf-8", newline="") as fin, tmp_path.open(
            "w", encoding="utf-8", newline=""
        ) as fout:
            old_reader = csv.DictReader(fin)
            new_writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
            new_writer.writeheader()
            for row in old_reader:
                out_row = {k: "" for k in fieldnames}
                for k, v in row.items():
                    if k in out_row and v is not None:
                        out_row[k] = v
                new_writer.writerow(out_row)

        tmp_path.replace(csv_path)
        print(f"Rewrote CSV header to include new columns: {csv_path}", flush=True)

    except Exception as e:
        print(f"Warning: Could not ensure CSV header for {csv_path}: {e}", flush=True)


def append_csv_row(csv_path: Path, fieldnames: List[str], row: Dict[str, str]) -> None:
    file_exists = csv_path.exists()
    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
            out_row = {k: "" for k in fieldnames}
            for k, v in row.items():
                if k in out_row:
                    out_row[k] = v
            writer.writerow(out_row)
    except Exception as e:
        print(f"Warning: Could not append to CSV {csv_path}: {e}", flush=True)


def make_progress(progress_mode: str, total: int) -> Callable[[int, str], None]:
    progress_mode = progress_mode.lower()
    if progress_mode == "none":
        return lambda i, msg: None

    if progress_mode == "print":
        def upd(i: int, msg: str) -> None:
            print(f"[{i}/{total}] {msg}", flush=True)
        return upd

    use_tqdm = progress_mode == "tqdm" or (progress_mode == "auto" and sys.stderr.isatty())
    if not use_tqdm:
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
        step = max(0, i - last_i)
        if step:
            bar.update(step)
            last_i = i
        bar.set_postfix_str(msg[:120])

    return upd


def process_audio_batch(audio_files: List[Path], tmp_dir: Path, workers: int = 4) -> List[Tuple[Path, Optional[np.ndarray], Optional[str]]]:
    """Convert/load audio files in parallel, preserving input order."""

    def convert_file(af: Path) -> Tuple[Path, Optional[np.ndarray], Optional[str]]:
        audio = load_audio_direct(af)
        if audio is not None:
            return af, audio, None
        wav_path = _tmp_wav_path(tmp_dir, af)
        try:
            ffmpeg_to_wav_16k_mono(af, wav_path)
            audio, _ = sf.read(str(wav_path), dtype="float32")
            try:
                wav_path.unlink(missing_ok=True)
            except Exception:
                pass
            return af, audio, None
        except Exception as e:
            return af, None, str(e)

    results: List[Tuple[Path, Optional[np.ndarray], Optional[str]]] = [
        (p, None, "not_started") for p in audio_files
    ]

    max_workers = max(1, int(workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(convert_file, af): i for i, af in enumerate(audio_files)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = (audio_files[i], None, str(e))

    return results


def compute_batch_embeddings(
    audio_arrays: List[np.ndarray],
    inference,
    device: torch.device,
    batch_size: int = 32,
    min_batch_size: int = 1,
) -> Tuple[List[Optional[np.ndarray]], List[str]]:
    """Compute embeddings for audio arrays, with OOM backoff."""
    if not audio_arrays:
        return [], []

    tensors: List[torch.Tensor] = []
    for audio in audio_arrays:
        if audio is None:
            tensors.append(None)  # placeholder
            continue
        t = torch.from_numpy(np.asarray(audio)).float()
        if len(t.shape) == 1:
            t = t.unsqueeze(0)
        tensors.append(t)

    # Keep index mapping to preserve per-item results
    idx_map = [i for i, t in enumerate(tensors) if t is not None]
    valid_tensors = [tensors[i] for i in idx_map]
    if not valid_tensors:
        return [None] * len(audio_arrays), ["all_empty"]

    current_bs = min(int(batch_size), len(valid_tensors))
    current_bs = max(int(min_batch_size), current_bs)

    out_embeddings: List[Optional[np.ndarray]] = [None] * len(audio_arrays)
    errors: List[str] = []

    while current_bs >= int(min_batch_size):
        try:
            with torch.no_grad():
                for start in range(0, len(valid_tensors), current_bs):
                    sub = valid_tensors[start : start + current_bs]
                    max_len = max(t.shape[-1] for t in sub)
                    padded = []
                    for t in sub:
                        if t.shape[-1] < max_len:
                            pad = torch.zeros(1, max_len - t.shape[-1])
                            t = torch.cat([t, pad], dim=-1)
                        padded.append(t)

                    batch = torch.stack(padded).to(device)
                    emb = inference.model(batch)
                    if len(emb.shape) == 3:
                        emb = emb.mean(dim=1)
                    emb_np = emb.detach().cpu().numpy()

                    for j, e in enumerate(emb_np):
                        orig_idx = idx_map[start + j]
                        out_embeddings[orig_idx] = e

            return out_embeddings, []

        except torch.cuda.OutOfMemoryError:
            current_bs = current_bs // 2
            if current_bs < int(min_batch_size):
                errors.append("OOM: could not process even at min batch size")
                break
            print(f"OOM detected, reducing batch size to {current_bs}...", flush=True)
            torch.cuda.empty_cache()
        except Exception as e:
            errors.append(f"Error computing embeddings: {e}")
            break

    return out_embeddings, errors


def build_ref_embeddings(
    ref_paths: List[Path],
    tmp_dir: Path,
    inference,
    device: torch.device,
    workers: int,
    batch_size: int,
    min_batch_size: int,
    label: str,
    progress_mode: str = "print",
) -> Tuple[List[Path], np.ndarray]:
    """Return (paths_used, embeddings_matrix[N,D]) with L2-normalised embeddings."""
    if not ref_paths:
        return [], np.zeros((0, 1), dtype=np.float32)

    print(f"Loading {len(ref_paths)} {label} reference file(s) for embedding...", flush=True)

    conv = process_audio_batch(ref_paths, tmp_dir, workers=workers)
    good_paths: List[Path] = []
    audios: List[np.ndarray] = []

    for p, audio, err in conv:
        if err is None and audio is not None:
            good_paths.append(p)
            audios.append(audio)
        else:
            print(f"[ref:{label}] skip {p.name}: {err}", flush=True)

    if not audios:
        raise ValueError(f"No valid {label} reference audio could be loaded.")

    # embeddings per ref (aligned to good_paths)
    emb_list, emb_errors = compute_batch_embeddings(
        audios, inference, device, batch_size=batch_size, min_batch_size=min_batch_size
    )

    if emb_errors:
        raise RuntimeError(f"Failed computing {label} reference embeddings: {emb_errors}")

    embs: List[np.ndarray] = []
    used_paths: List[Path] = []

    for p, e in zip(good_paths, emb_list):
        if e is None:
            continue
        e2 = l2_normalize(np.asarray(e).reshape(1, -1))[0].astype(np.float32)
        embs.append(e2)
        used_paths.append(p)

    if not embs:
        raise ValueError(f"No usable {label} reference embeddings computed.")

    mat = np.stack(embs, axis=0)
    return used_paths, mat


def sample_paths(paths: List[Path], mode: str, max_n: int, seed: int) -> List[Path]:
    if max_n <= 0 or len(paths) <= max_n:
        return paths
    if mode == "random":
        rng = random.Random(int(seed))
        idx = list(range(len(paths)))
        rng.shuffle(idx)
        return [paths[i] for i in idx[:max_n]]
    # default: first
    return paths[:max_n]


def main() -> None:
    require_ffmpeg()

    ap = argparse.ArgumentParser(description="Sort audio files by whether they match a target speaker (with optional others cohort).")
    ap.add_argument("--in", dest="in_path", required=True, help="Input file or directory containing many audio files")

    # Target refs
    ap.add_argument("--target_ref_dir", default=None, help="Directory of TARGET speaker reference audio files (preferred)")
    ap.add_argument("--ref", dest="ref_files", nargs="*", default=None, help="Legacy: TARGET reference audio files or directories")

    # Others cohort
    ap.add_argument("--other_ref_dir", default=None, help="Directory of OTHER speakers reference audio files (optional)")
    ap.add_argument("--other_ref_max", type=int, default=200, help="Max number of OTHER reference files to use (0=all)")
    ap.add_argument("--other_ref_sample", choices=["first", "random"], default="first", help="How to choose other refs if capped")
    ap.add_argument("--seed", type=int, default=1337, help="Seed for random sampling")

    ap.add_argument("--target_out", required=True, help="Directory to put files matching target speaker")
    ap.add_argument("--other_out", required=True, help="Directory to put all other files")

    ap.add_argument("--threshold", type=float, default=0.5, help="Threshold for TARGET decision. If --other_ref_dir is set, this is a margin threshold.")
    ap.add_argument("--min_seconds", type=float, default=2.0, help="Route to OTHER files shorter than this")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Use cuda if you have a GPU")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    ap.add_argument("--copy", action="store_true", help="Copy files (default if neither --move nor --copy set)")
    ap.add_argument("--dry_run", action="store_true", help="Do not move/copy; just print decisions and write CSV")
    ap.add_argument("--progress", choices=["auto", "tqdm", "print", "none"], default="auto", help="Progress display mode")
    ap.add_argument("--log_every", type=int, default=0, help="Also print a line every N files (0=off)")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""), help="Optional Hugging Face token")

    ap.add_argument("--batch_size", type=int, default=16, help="Initial batch size for embedding (auto-reduced on OOM)")
    ap.add_argument("--min_batch_size", type=int, default=1, help="Minimum batch size to attempt")
    ap.add_argument("--workers", type=int, default=8, help="Number of worker threads for audio conversion")

    ap.add_argument("--state_file", default="speaker_sort_state.json", help="State file for resuming")
    ap.add_argument("--no_resume", action="store_true", help="Force restart from beginning")
    ap.add_argument("--force_resume_mismatch", action="store_true", help="Allow resume even if refs/params changed (not recommended)")

    args = ap.parse_args()

    in_path = Path(args.in_path)
    target_out = Path(args.target_out)
    other_out = Path(args.other_out)

    do_move = bool(args.move)
    if args.copy:
        do_move = False

    if not in_path.exists():
        raise SystemExit(f"Input path not found: {in_path}")

    # Build target refs from folder OR legacy list
    target_refs: List[Path] = []
    if args.target_ref_dir:
        td = Path(args.target_ref_dir)
        if not td.exists():
            raise SystemExit(f"target_ref_dir not found: {td}")
        target_refs = iter_audio_files(td)
    else:
        legacy = [Path(p) for p in (args.ref_files or [])]
        if legacy:
            target_refs = expand_files_or_dirs(legacy)

    if not target_refs:
        raise SystemExit("No TARGET reference files found. Provide --target_ref_dir or --ref ...")

    # Build other refs
    other_refs: List[Path] = []
    if args.other_ref_dir:
        od = Path(args.other_ref_dir)
        if not od.exists():
            raise SystemExit(f"other_ref_dir not found: {od}")
        other_refs = iter_audio_files(od)
        other_refs = sample_paths(other_refs, args.other_ref_sample, int(args.other_ref_max), int(args.seed))

    audio_files = iter_audio_files(in_path)
    if not audio_files:
        raise SystemExit(f"No audio files found under: {in_path}")

    # State + CSV
    state_file = Path(args.state_file)
    csv_path = Path.cwd() / "speaker_sort_scores.csv"

    # Fieldnames (keep first 4 stable)
    base_fields = ["file", "score", "decision", "reason"]
    extra_fields = [
        "score_target",
        "score_other_max",
        "other_top1_ref",
        "compound_mode",
    ]
    use_compound = bool(other_refs)
    fieldnames = base_fields + (extra_fields if use_compound else [])

    ensure_csv_header(csv_path, fieldnames)

    existing_state = load_state(state_file) if not args.no_resume else {}
    processed_files = set(existing_state.get("processed_files", []))

    # Signature to prevent mixing runs
    sig_payload = {
        "model_id": MODEL_ID,
        "use_compound": use_compound,
        "threshold": float(args.threshold),
        "min_seconds": float(args.min_seconds),
        "target_refs": [str(p) for p in target_refs],
        "other_refs": [str(p) for p in other_refs],
        "other_ref_max": int(args.other_ref_max),
        "other_ref_sample": str(args.other_ref_sample),
    }
    run_sig = compute_signature(sig_payload)

    old_sig = existing_state.get("run_signature")
    if processed_files and not args.no_resume and old_sig and old_sig != run_sig and not args.force_resume_mismatch:
        raise SystemExit(
            "State file exists but refs/params changed.\n"
            "Run with --no_resume to restart, or --force_resume_mismatch (not recommended).\n"
            f"state_file: {state_file}"
        )

    remaining_files = [f for f in audio_files if str(f) not in processed_files]
    if not remaining_files:
        print("All files have already been processed!", flush=True)
        return

    print(f"Found {len(audio_files)} total audio file(s).", flush=True)
    print(f"Processing {len(remaining_files)} remaining file(s).", flush=True)

    # Lazy import
    print("Loading speaker embedding model... (first run may download weights)", flush=True)
    from pyannote.audio import Model, Inference

    device = torch.device(args.device)
    model = Model.from_pretrained(MODEL_ID, token=args.hf_token or None)
    inference = Inference(model, window="whole")
    inference.to(device)

    ensure_dir(target_out)
    ensure_dir(other_out)

    progress_upd = make_progress(args.progress, total=len(remaining_files))

    # Track state
    current_processed_files = list(processed_files)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # TARGET centroid
        used_target_paths, target_embs = build_ref_embeddings(
            target_refs,
            tmp,
            inference,
            device,
            workers=int(args.workers),
            batch_size=int(args.batch_size),
            min_batch_size=int(args.min_batch_size),
            label="TARGET",
        )
        target_centroid = l2_normalize(np.mean(target_embs, axis=0, keepdims=True))[0]

        # OTHERS embeddings matrix
        other_mat = None
        other_paths_used: List[Path] = []
        if use_compound:
            other_paths_used, other_mat0 = build_ref_embeddings(
                other_refs,
                tmp,
                inference,
                device,
                workers=int(args.workers),
                batch_size=int(args.batch_size),
                min_batch_size=int(args.min_batch_size),
                label="OTHERS",
            )
            other_mat = other_mat0  # shape (N,D)
            print(f"Using {other_mat.shape[0]} OTHER reference embeddings (max-sim cohort).", flush=True)
        else:
            print("No --other_ref_dir set (or empty). Using TARGET-only scoring.", flush=True)

        # Process files in batches (conversion parallel, embedding batched)
        bs = int(args.batch_size)
        total = len(remaining_files)
        total_batches = (total + bs - 1) // bs

        for batch_idx in range(total_batches):
            start = batch_idx * bs
            end = min(start + bs, total)
            batch_files = remaining_files[start:end]

            # Pre-filter short files
            to_process: List[Path] = []
            for af in batch_files:
                dur = get_duration_fast(af)
                if dur is not None and dur < float(args.min_seconds):
                    row = {
                        "file": str(af),
                        "score": "",
                        "decision": "OTHER",
                        "reason": f"too_short({dur:.2f}s)",
                    }
                    if use_compound:
                        row.update(
                            {
                                "score_target": "",
                                "score_other_max": "",
                                "other_top1_ref": "",
                                "compound_mode": "target_minus_othermax",
                            }
                        )
                    append_csv_row(csv_path, fieldnames, row)
                    current_processed_files.append(str(af))
                    progress_upd(start + batch_files.index(af) + 1, af.name)
                else:
                    to_process.append(af)

            if not to_process:
                save_state(state_file, {"processed_files": current_processed_files, "run_signature": run_sig})
                continue

            conv = process_audio_batch(to_process, tmp, workers=int(args.workers))

            good_files: List[Path] = []
            audios: List[np.ndarray] = []
            for af, audio, err in conv:
                if err is not None or audio is None:
                    row = {
                        "file": str(af),
                        "score": "",
                        "decision": "ERROR_CONVERT",
                        "reason": (err or "convert_failed"),
                    }
                    if use_compound:
                        row.update(
                            {
                                "score_target": "",
                                "score_other_max": "",
                                "other_top1_ref": "",
                                "compound_mode": "target_minus_othermax",
                            }
                        )
                    append_csv_row(csv_path, fieldnames, row)
                    current_processed_files.append(str(af))
                    progress_upd(start + batch_files.index(af) + 1, af.name)
                else:
                    good_files.append(af)
                    audios.append(audio)

            if not audios:
                save_state(state_file, {"processed_files": current_processed_files, "run_signature": run_sig})
                continue

            emb_list, emb_errors = compute_batch_embeddings(
                audios,
                inference,
                device,
                batch_size=int(args.batch_size),
                min_batch_size=int(args.min_batch_size),
            )

            if emb_errors:
                # mark all as embedding error
                for af in good_files:
                    row = {
                        "file": str(af),
                        "score": "",
                        "decision": "ERROR_EMBEDDING",
                        "reason": ";".join(emb_errors)[:500],
                    }
                    if use_compound:
                        row.update(
                            {
                                "score_target": "",
                                "score_other_max": "",
                                "other_top1_ref": "",
                                "compound_mode": "target_minus_othermax",
                            }
                        )
                    append_csv_row(csv_path, fieldnames, row)
                    current_processed_files.append(str(af))
                    progress_upd(start + batch_files.index(af) + 1, af.name)

                save_state(state_file, {"processed_files": current_processed_files, "run_signature": run_sig})
                continue

            for af, e in zip(good_files, emb_list):
                if e is None:
                    continue

                emb = l2_normalize(np.asarray(e).reshape(1, -1))[0]

                s_target = cosine_sim(target_centroid, emb)

                if use_compound and other_mat is not None and other_mat.shape[0] > 0:
                    sims = other_mat @ emb  # (N,)
                    j = int(np.argmax(sims))
                    s_other = float(sims[j])
                    other_top = str(other_paths_used[j]) if j < len(other_paths_used) else ""
                    compound = float(s_target - s_other)
                    score_for_decision = compound
                    score_str = f"{compound:.6f}"
                    decision = "TARGET" if compound >= float(args.threshold) else "OTHER"
                    row = {
                        "file": str(af),
                        "score": score_str,
                        "decision": decision,
                        "reason": "",
                        "score_target": f"{s_target:.6f}",
                        "score_other_max": f"{s_other:.6f}",
                        "other_top1_ref": other_top,
                        "compound_mode": "target_minus_othermax",
                    }
                else:
                    score_for_decision = s_target
                    score_str = f"{s_target:.6f}"
                    decision = "TARGET" if s_target >= float(args.threshold) else "OTHER"
                    row = {
                        "file": str(af),
                        "score": score_str,
                        "decision": decision,
                        "reason": "",
                    }

                append_csv_row(csv_path, fieldnames, row)
                current_processed_files.append(str(af))

                # progress
                file_idx = start + batch_files.index(af) + 1
                progress_upd(file_idx, af.name)

                if args.log_every and file_idx % int(args.log_every) == 0:
                    if use_compound:
                        print(
                            f"[{file_idx}/{len(remaining_files)}] {af.name} -> {decision} score={score_str} (target={row.get('score_target','')}, other={row.get('score_other_max','')})",
                            flush=True,
                        )
                    else:
                        print(
                            f"[{file_idx}/{len(remaining_files)}] {af.name} -> {decision} score={score_str}",
                            flush=True,
                        )

                # move/copy
                if not args.dry_run:
                    rel = af.relative_to(in_path) if in_path.is_dir() else Path(af.name)
                    dst = (target_out / rel) if decision == "TARGET" else (other_out / rel)
                    safe_move_or_copy(af, dst, do_move=do_move)

            save_state(state_file, {"processed_files": current_processed_files, "run_signature": run_sig})

    print("\nDone.", flush=True)
    print(f"CSV log: {csv_path}", flush=True)
    print(f"State file: {state_file}", flush=True)
    if args.dry_run:
        print("Dry-run only: no files moved/copied.", flush=True)
    else:
        print("Files have been sorted.", flush=True)

    beep()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
