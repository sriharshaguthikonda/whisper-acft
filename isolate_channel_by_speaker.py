"""isolate_channel_by_speaker_v2.py

Isolate the speaker-matching channel from multi-channel .m4a files and save mono WAVs.

Fixes vs prior version:
- Model is downloaded once and loaded once per worker process (NOT per file).
- If CUDA is used, forces a single worker (one GPU != many processes).
- Batches channel embeddings in a single encode call for speed.
- Adds acceptance rule using BOTH threshold and margin (best - second_best).
- Adds robust audio loading with optional FFmpeg fallback.
- Resumable state JSON, with optional reprocess when input file changed.

Usage (PowerShell):
  python c:\Windows_software\whisper-acft\isolate_channel_by_speaker_v2.py `
    --reference "c:\path\to\reference_voice.wav" `
    --input-dir "I:\Record" `
    --output-dir "I:\Record_Channel_selected" `
    --state-file "c:\Windows_software\whisper-acft\channel_isolation_state.json" `
    --report-json "c:\Windows_software\whisper-acft\channel_isolation_report.json" `
    --threshold 0.80 `
    --margin 0.05 `
    --workers 0

Usage (cmd.exe one line):
  python c:\Windows_software\whisper-acft\isolate_channel_by_speaker_v2.py --reference "c:\path\to\reference_voice.wav" --input-dir "I:\Record" --output-dir "I:\Record_Channel_selected" --state-file "c:\Windows_software\whisper-acft\channel_isolation_state.json" --report-json "c:\Windows_software\whisper-acft\channel_isolation_report.json" --threshold 0.80 --margin 0.05 --workers 1 --segment-seconds 20 --segments 5

Requirements:
  pip install torch torchaudio speechbrain huggingface_hub tqdm

Notes:
- This script assumes one channel is predominantly speaker A and the other channel is predominantly speaker B.
- If your audio is long, embeddings are computed from multiple short segments for stability and speed.

"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import multiprocessing
import os
import pathlib
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torchaudio
from huggingface_hub import snapshot_download
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm


AUDIO_EXTS_DEFAULT = {".m4a"}
DEFAULT_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_THRESHOLD = 0.80
DEFAULT_MARGIN = 0.05
TARGET_SR = 16_000


# -------------------------
# Data structures
# -------------------------

@dataclass
class Decision:
    channel_index: int
    similarity: float
    second_similarity: float
    margin: float
    accepted: bool
    output_path: str
    num_channels: int
    input_mtime: float
    error: str = ""


# -------------------------
# Globals for worker processes
# -------------------------

_G_ENCODER: Optional[EncoderClassifier] = None
_G_DEVICE: Optional[str] = None
_G_REF_VEC: Optional[torch.Tensor] = None
_G_RESAMPLERS: Dict[int, torchaudio.transforms.Resample] = {}
_G_ARGS: Dict[str, Any] = {}


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isolate target speaker channel from multichannel m4a files.")
    p.add_argument("--reference", required=True, help="Path to reference audio of the target speaker (wav/m4a/etc).")
    p.add_argument("--input-dir", required=True, help="Directory containing multichannel m4a files.")
    p.add_argument("--output-dir", required=True, help="Directory to write mono wav files (mirrors input tree).")
    p.add_argument("--state-file", required=True, help="Path to JSON state for resumable runs.")
    p.add_argument("--report-json", help="Where to write a JSON report of decisions.")

    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Cosine similarity threshold.")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="Require best-second_best >= margin.")

    p.add_argument("--workers", type=int, default=0, help="Worker processes (0 => CPU count). CUDA forces 1.")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")

    p.add_argument("--cache-dir", default=r"I:\\pretrained_models", help="Cache directory for model files.")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id.")

    p.add_argument("--audio-exts", default=",".join(sorted(AUDIO_EXTS_DEFAULT)), help="Comma-separated extensions to scan.")

    p.add_argument("--segment-seconds", type=float, default=10.0, help="Seconds per segment used for embedding.")
    p.add_argument("--segments", type=int, default=3, help="Number of segments sampled across the file.")

    p.add_argument("--reprocess-changed", action="store_true", help="Reprocess files if mtime changed since last run.")

    p.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg executable for fallback decode.")
    p.add_argument("--no-ffmpeg-fallback", action="store_true", help="Disable ffmpeg fallback decoding.")

    return p.parse_args()


def pick_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Model prep / loading
# -------------------------

def prepare_model_local_dir(model_id: str, cache_dir: str) -> str:
    """Download model snapshot once, with Windows-friendly (no symlink) settings."""
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("SPEECHBRAIN_CACHE", cache_dir)
    os.environ.setdefault("SPEECHBRAIN_LOCAL_STRATEGY", "copy")

    cache_p = pathlib.Path(cache_dir)
    cache_p.mkdir(parents=True, exist_ok=True)

    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_dir=str(cache_p / model_id.replace("/", "_")),
        local_dir_use_symlinks=False,
        allow_patterns="*",
    )

    # SpeechBrain occasionally expects label_encoder.ckpt; ensure it exists.
    label_txt = pathlib.Path(local_dir) / "label_encoder.txt"
    label_ckpt = pathlib.Path(local_dir) / "label_encoder.ckpt"
    if label_txt.exists() and not label_ckpt.exists():
        shutil.copyfile(label_txt, label_ckpt)

    return str(local_dir)


def load_embedder(device: str, local_dir: str) -> EncoderClassifier:
    return EncoderClassifier.from_hparams(
        source=local_dir,
        run_opts={"device": device},
        savedir=local_dir,
    )


# -------------------------
# Audio IO
# -------------------------

def _load_audio_torchaudio(path: pathlib.Path) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return wav, int(sr)


def _load_audio_ffmpeg(path: pathlib.Path, ffmpeg_exe: str) -> Tuple[torch.Tensor, int]:
    """Decode with ffmpeg to a temporary WAV (preserving channel count), then load."""
    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        cmd = [
            ffmpeg_exe,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-ar",
            str(TARGET_SR),
            "-c:a",
            "pcm_s16le",
            tmp_wav,
        ]
        subprocess.run(cmd, check=True)
        wav, sr = _load_audio_torchaudio(pathlib.Path(tmp_wav))
        return wav, sr
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass


def load_audio_any(path: pathlib.Path, ffmpeg_exe: str, allow_ffmpeg_fallback: bool) -> Tuple[torch.Tensor, int]:
    try:
        return _load_audio_torchaudio(path)
    except Exception:
        if not allow_ffmpeg_fallback:
            raise
        return _load_audio_ffmpeg(path, ffmpeg_exe)


# -------------------------
# Embeddings
# -------------------------

def get_resampler(orig_sr: int) -> Optional[torchaudio.transforms.Resample]:
    if orig_sr == TARGET_SR:
        return None
    r = _G_RESAMPLERS.get(orig_sr)
    if r is None:
        r = torchaudio.transforms.Resample(orig_sr, TARGET_SR)
        _G_RESAMPLERS[orig_sr] = r
    return r


def normalize_audio(wavs: torch.Tensor) -> torch.Tensor:
    # Light amplitude normalization helps when levels vary wildly
    # (doesn't change relative channel content).
    peak = wavs.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return wavs / peak


def make_segments(wavs_ct: torch.Tensor, segment_samples: int, segments: int) -> List[torch.Tensor]:
    """Return a list of (C, segment_samples) tensors sampled across time."""
    c, t = wavs_ct.shape
    if t <= segment_samples or segments <= 1:
        return [wavs_ct[:, :segment_samples] if t > segment_samples else wavs_ct]

    max_start = t - segment_samples
    # Evenly spaced start indices
    starts = torch.linspace(0, max_start, steps=segments)
    starts = starts.round().to(torch.int64)
    out: List[torch.Tensor] = []
    for s in starts.tolist():
        s = int(s)
        out.append(wavs_ct[:, s : s + segment_samples])
    return out


def encode_channel_embeddings(
    encoder: EncoderClassifier,
    wavs_ct_16k: torch.Tensor,
    segment_seconds: float,
    segments: int,
) -> torch.Tensor:
    """Return embeddings per channel as tensor [C, D] on encoder.device."""
    # wavs_ct_16k: (C, T) at 16k
    wavs_ct_16k = normalize_audio(wavs_ct_16k)

    seg_samples = int(max(1, round(segment_seconds * TARGET_SR)))
    seg_list = make_segments(wavs_ct_16k, seg_samples, segments)

    # Accumulate embeddings across segments
    emb_sum: Optional[torch.Tensor] = None
    for seg in seg_list:
        # Treat channels as batch: (C, T)
        seg = seg.to(encoder.device)
        with torch.no_grad():
            embs = encoder.encode_batch(seg)  # [C, 1, D] (typically)
        if embs.ndim == 3:
            embs = embs.squeeze(1)  # [C, D]
        if emb_sum is None:
            emb_sum = embs
        else:
            emb_sum = emb_sum + embs

    assert emb_sum is not None
    emb_avg = emb_sum / float(len(seg_list))
    return emb_avg


def cosine_sims(embs_cd: torch.Tensor, ref_d: torch.Tensor) -> torch.Tensor:
    # Normalize then cosine sim via dot product
    embs = torch.nn.functional.normalize(embs_cd, p=2, dim=1)
    ref = torch.nn.functional.normalize(ref_d, p=2, dim=0)
    return torch.matmul(embs, ref.unsqueeze(1)).squeeze(1)  # [C]


# -------------------------
# State
# -------------------------

def load_state(state_path: pathlib.Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {"meta": {}, "decisions": {}}
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path: pathlib.Path, state: Dict[str, Any]) -> None:
    tmp = state_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_path)


# -------------------------
# File discovery
# -------------------------

def discover_audio_files(root: pathlib.Path, exts: Iterable[str]) -> List[pathlib.Path]:
    exts_l = {e.lower().strip() for e in exts if e.strip()}
    out: List[pathlib.Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_l:
            out.append(p)
    return out


# -------------------------
# Worker init + task
# -------------------------

def worker_init(init_payload: Dict[str, Any]) -> None:
    """Initializer: load model once per worker process.

    ProcessPoolExecutor supports initializer + initargs (no kwargs),
    so we pass a single dict as initargs.
    """
    global _G_ENCODER, _G_DEVICE, _G_REF_VEC, _G_ARGS

    # Avoid CPU oversubscription when using many workers.
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    device = str(init_payload["device"])
    model_dir = str(init_payload["model_dir"])
    ref_vec_list = init_payload["ref_vec_list"]

    _G_DEVICE = device
    _G_ENCODER = load_embedder(device, model_dir)
    _G_REF_VEC = torch.tensor(ref_vec_list, dtype=torch.float32, device=device)

    _G_ARGS = {
        "segment_seconds": float(init_payload["segment_seconds"]),
        "segments": int(init_payload["segments"]),
        "ffmpeg_exe": str(init_payload["ffmpeg_exe"]),
        "allow_ffmpeg_fallback": bool(init_payload["allow_ffmpeg_fallback"]),
    }


def save_selected_channel(wavs_ct_16k: torch.Tensor, channel: int, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected = wavs_ct_16k[channel].unsqueeze(0).cpu()
    torchaudio.save(str(out_path), selected, TARGET_SR)


def process_one_file(
    audio_path_str: str,
    input_dir_str: str,
    output_dir_str: str,
    threshold: float,
    margin: float,
    reprocess_changed: bool,
    prev_entry: Optional[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """Run inside worker; returns a serializable decision dict."""
    global _G_ENCODER, _G_REF_VEC

    path = pathlib.Path(audio_path_str)
    input_dir = pathlib.Path(input_dir_str)
    output_dir = pathlib.Path(output_dir_str)

    mtime = path.stat().st_mtime
    if prev_entry and (not reprocess_changed) and float(prev_entry.get("input_mtime", -1)) == float(mtime):
        # unchanged; skip
        return audio_path_str, prev_entry

    try:
        assert _G_ENCODER is not None and _G_REF_VEC is not None

        wavs, sr = load_audio_any(
            path,
            ffmpeg_exe=_G_ARGS["ffmpeg_exe"],
            allow_ffmpeg_fallback=_G_ARGS["allow_ffmpeg_fallback"],
        )
        # wavs: (C, T)
        if wavs.ndim == 1:
            wavs = wavs.unsqueeze(0)

        # Resample once for all channels
        if sr != TARGET_SR:
            resampler = get_resampler(int(sr))
            if resampler is None:
                wavs_16k = wavs
            else:
                wavs_16k = resampler(wavs)
        else:
            wavs_16k = wavs

        num_channels = int(wavs_16k.shape[0])

        embs = encode_channel_embeddings(
            _G_ENCODER,
            wavs_16k,
            segment_seconds=_G_ARGS["segment_seconds"],
            segments=_G_ARGS["segments"],
        )

        sims = cosine_sims(embs, _G_REF_VEC).detach().cpu()

        if num_channels == 1:
            best_idx = 0
            best_sim = float(sims[0].item())
            second_sim = -1.0
        else:
            top2 = torch.topk(sims, k=2)
            best_sim = float(top2.values[0].item())
            second_sim = float(top2.values[1].item())
            best_idx = int(top2.indices[0].item())

        m = float(best_sim - second_sim) if num_channels > 1 else 0.0
        accepted = (best_sim >= float(threshold)) and (m >= float(margin) if num_channels > 1 else True)

        rel = path.relative_to(input_dir)
        out_path = (output_dir / rel).with_suffix(".wav")
        save_selected_channel(wavs_16k, best_idx, out_path)

        decision = Decision(
            channel_index=best_idx,
            similarity=best_sim,
            second_similarity=second_sim,
            margin=m,
            accepted=accepted,
            output_path=str(out_path),
            num_channels=num_channels,
            input_mtime=float(mtime),
            error="",
        )

        return audio_path_str, dataclasses.asdict(decision)

    except Exception as e:
        # Record failure but keep run going
        fail = Decision(
            channel_index=0,
            similarity=-1.0,
            second_similarity=-1.0,
            margin=0.0,
            accepted=False,
            output_path="",
            num_channels=0,
            input_mtime=float(mtime),
            error=f"{type(e).__name__}: {e}",
        )
        return audio_path_str, dataclasses.asdict(fail)


# -------------------------
# Main run
# -------------------------

def compute_reference_vector(reference_path: pathlib.Path, device: str, model_dir: str, ffmpeg_exe: str, allow_ffmpeg_fallback: bool, segment_seconds: float, segments: int) -> List[float]:
    encoder = load_embedder(device, model_dir)
    wavs, sr = load_audio_any(reference_path, ffmpeg_exe=ffmpeg_exe, allow_ffmpeg_fallback=allow_ffmpeg_fallback)
    if wavs.ndim == 1:
        wavs = wavs.unsqueeze(0)

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(int(sr), TARGET_SR)
        wavs_16k = resampler(wavs)
    else:
        wavs_16k = wavs

    # If reference has multiple channels, average them first (safe for a reference snippet)
    if wavs_16k.shape[0] > 1:
        wav_mono = wavs_16k.mean(dim=0, keepdim=True)
    else:
        wav_mono = wavs_16k

    emb = encode_channel_embeddings(encoder, wav_mono, segment_seconds=segment_seconds, segments=segments)[0]
    emb = emb.detach().cpu().to(torch.float32)
    return emb.tolist()


def main() -> None:
    args = parse_args()

    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    state_path = pathlib.Path(args.state_file)
    report_path = pathlib.Path(args.report_json) if args.report_json else None

    exts = [e.strip() for e in args.audio_exts.split(",") if e.strip()]
    device = pick_device(args.device)

    # If CUDA is in use, run single-worker to avoid multiple CUDA contexts/models.
    workers = args.workers or multiprocessing.cpu_count()
    if device == "cuda" and workers != 1:
        print(f"[device] CUDA detected; forcing workers=1 (was {workers})")
        workers = 1

    allow_ffmpeg_fallback = not args.no_ffmpeg_fallback

    model_dir = prepare_model_local_dir(args.model_id, args.cache_dir)

    print(f"[device] {device}")
    print(f"[model_dir] {model_dir}")
    print(f"[threshold] {args.threshold}")
    print(f"[margin] {args.margin}")
    print(f"[segments] {args.segments} x {args.segment_seconds:.2f}s")
    print(f"[workers] {workers}")

    ref_vec_list = compute_reference_vector(
        reference_path=pathlib.Path(args.reference),
        device=device,
        model_dir=model_dir,
        ffmpeg_exe=args.ffmpeg,
        allow_ffmpeg_fallback=allow_ffmpeg_fallback,
        segment_seconds=args.segment_seconds,
        segments=max(1, min(args.segments, 3)),  # reference: keep it modest
    )

    state = load_state(state_path)
    state.setdefault("meta", {})
    state.setdefault("decisions", {})
    state["meta"].update(
        {
            "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "threshold": float(args.threshold),
            "margin": float(args.margin),
            "target_sr": int(TARGET_SR),
            "model_id": str(args.model_id),
            "device": str(device),
            "segment_seconds": float(args.segment_seconds),
            "segments": int(args.segments),
            "allow_ffmpeg_fallback": bool(allow_ffmpeg_fallback),
        }
    )

    all_files = discover_audio_files(input_dir, exts)

    # Determine which files need processing
    to_process: List[pathlib.Path] = []
    for p in all_files:
        k = str(p.resolve())
        prev = state["decisions"].get(k)
        if prev is None:
            to_process.append(p)
            continue
        if args.reprocess_changed:
            try:
                mtime = p.stat().st_mtime
                if float(prev.get("input_mtime", -1)) != float(mtime):
                    to_process.append(p)
            except FileNotFoundError:
                pass

    print(f"[discovery] total={len(all_files)} to_process={len(to_process)} already_done={len(all_files) - len(to_process)}")

    if not to_process:
        print("No new/changed files to process.")
    else:
        # Run pool
        init_payload = {
            "device": device,
            "model_dir": model_dir,
            "ref_vec_list": ref_vec_list,
            "segment_seconds": float(args.segment_seconds),
            "segments": int(args.segments),
            "ffmpeg_exe": str(args.ffmpeg),
            "allow_ffmpeg_fallback": allow_ffmpeg_fallback,
        }

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=worker_init,
            initargs=(init_payload,),
        ) as ex:
            futures = {}
            for p in to_process:
                abs_key = str(p.resolve())
                prev_entry = state["decisions"].get(abs_key)
                fut = ex.submit(
                    process_one_file,
                    abs_key,
                    str(input_dir),
                    str(output_dir),
                    float(args.threshold),
                    float(args.margin),
                    bool(args.reprocess_changed),
                    prev_entry,
                )
                futures[fut] = abs_key

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring/isolating", unit="file"):
                k, decision_dict = fut.result()
                state["decisions"][k] = decision_dict
                save_state(state_path, state)

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)


if __name__ == "__main__":
    main()