"""find_files_without_speaker.py


https://chatgpt.com/c/69510f30-919c-8324-af65-19d4576b359d



find_files_without_speaker.py

What you asked for (DONE):
- Scan each file segment-by-segment.
- As soon as the target speaker has been detected for >= X% of the *entire file duration* (default 10%),
  STOP scanning that file and mark it as "KEEP" (do NOT move).
- If it becomes impossible to reach that X% with the remaining unscanned audio, STOP early and mark as "MOVE".
- Move happens immediately file-by-file (streaming), not at the end.
- Resumable: JSON state is written after every processed file; reruns skip already-processed files.

How "speaker duration" is estimated:
- We walk through the file in non-overlapping chunks (segment-seconds, default 10s).
- For each chunk:
  - If chunk is silence (RMS dBFS < vad-energy-db), we skip speaker embedding and do not count it.
  - Otherwise, compute cosine similarity vs reference speaker embedding (MAX across channels).
  - If similarity >= threshold, we count that chunk's duration as target_present_seconds.

Decision rule:
- required_seconds = max(min_presence_fraction * file_duration_seconds, min_presence_seconds)
- If target_present_seconds >= required_seconds -> KEEP (present=True)
- Else if target_present_seconds + (file_duration_seconds - scanned_seconds) < required_seconds -> MOVE early (present=False)
- Else scan next segment

Dependencies:
  pip install torch torchaudio speechbrain huggingface_hub tqdm
FFmpeg recommended for .m4a decode.

Recommended usage:
- Use --dry-run first.
- If it keeps moving too much:
  - Lower --threshold (e.g. 0.74 / 0.72)
  - Lower (more permissive) --vad-energy-db (e.g. -40)
  - Reduce --segment-seconds to get finer counting (e.g. 6)




  python c:\Windows_software\whisper-acft\find_files_without_speaker.py `
  --reference "I:\My_voice\CRISPR Data analysis final (slow paced) - Dr Sri Harsha Guthikonda.mp3" `
  --input-dir "I:\Record" `
  --move-nonmajority-to "I:\Record_others" `
  --out-txt "I:\Record_others\speaker_absent.txt" `
  --out-json "I:\Record_others\speaker_presence_report.json" `
  --min-presence-fraction 0.05 `
  --segment-seconds 10 `
  --threshold 0.78 `
  --vad-energy-db -35 `
  --device cuda `
  --workers 1 `
  --dry-run

"""



from __future__ import annotations

import argparse
import datetime as _dt
import json
import multiprocessing as mp
import os
import pathlib
import shutil
import signal
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


# -------------------------
# Defaults
# -------------------------

TARGET_SR = 16_000
DEFAULT_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_EXTS = {".m4a", ".wav", ".mp3"}
DEFAULT_THRESHOLD = 0.78


# -------------------------
# Result struct
# -------------------------

@dataclass
class PresenceResult:
    # present=True means: speaker present for >= required_seconds (over voiced-only denominator) => KEEP
    present: bool

    # classic diagnostics
    max_similarity: float
    best_channel: int
    best_segment_index: int

    # duration-based logic
    file_duration_sec: float
    total_voiced_seconds: float
    required_seconds: float

    scanned_seconds: float  # how far (in timeline seconds) we scanned embeddings before early-stop
    voiced_scanned_seconds: float

    target_present_seconds: float
    presence_fraction_of_voiced: float

    voiced_segments_total: int
    voiced_segments_scanned: int
    present_voiced_segments_scanned: int

    # optional debug (capped)
    segment_best_sims: List[float]
    segment_voiced_flags: List[bool]
    segment_present_flags: List[bool]

    stop_reason: str
    input_mtime: float
    error: str = ""
    moved_to: str = ""  # set by main process if moved


# -------------------------
# Worker globals
# -------------------------

_G_ENCODER: Optional[EncoderClassifier] = None
_G_REF_VEC: Optional[torch.Tensor] = None
_G_DEVICE: str = "cpu"
_G_ARGS: Dict[str, Any] = {}
_G_RESAMPLERS: Dict[int, torchaudio.transforms.Resample] = {}
_G_STOP_FLAG: Optional[Any] = None


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Move files where the target speaker is present for < X% of voiced (non-silent) audio."
    )

    p.add_argument("--reference", required=True, help="Reference audio of target speaker (wav/m4a/mp3).")
    p.add_argument("--input-dir", required=True, help="Directory to scan recursively.")

    # Speaker match + segmentation
    p.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Voiced segment counts as target-present if MAX cosine similarity across channels >= threshold.",
    )

    p.add_argument(
        "--segment-seconds",
        type=float,
        default=10.0,
        help="Segment length in seconds (non-overlapping scan).",
    )

    # Presence rule (voiced-only)
    p.add_argument(
        "--min-presence-fraction",
        type=float,
        default=0.10,
        help="KEEP if target_present_seconds >= this * total_voiced_seconds (default 0.10 = 10%).",
    )

    p.add_argument(
        "--min-presence-seconds",
        type=float,
        default=0.0,
        help="Optional absolute floor in seconds (required_seconds = max(fraction*voiced, this)).",
    )

    # VAD-ish gate (silence ignored in denominator)
    p.add_argument(
        "--vad-energy-db",
        type=float,
        default=-35.0,
        help="Segments with RMS dBFS below this are treated as silence (ignored).",
    )

    p.add_argument(
        "--min-voiced-segments",
        type=int,
        default=1,
        help="If fewer voiced segments than this, mark error and do NOT move.",
    )

    # Debug storage cap (avoid giant JSON on long files)
    p.add_argument(
        "--max-debug-segments",
        type=int,
        default=200,
        help="Store per-segment debug arrays for at most this many segments (0 => store none).",
    )

    p.add_argument("--audio-exts", default=",".join(sorted(DEFAULT_EXTS)), help="Comma-separated extensions.")

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")
    p.add_argument("--workers", type=int, default=0, help="Worker processes (0 => CPU count). CUDA forces 1.")

    p.add_argument("--cache-dir", default=r"I:\pretrained_models", help="Cache dir for HF/SpeechBrain.")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Hugging Face model id.")

    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable path.")
    p.add_argument("--no-ffmpeg-fallback", action="store_true", help="Disable ffmpeg fallback decoding.")

    p.add_argument(
        "--out-txt",
        default=None,
        help="Write MOVED file list to this .txt (deduped on resume).",
    )

    p.add_argument(
        "--out-json",
        default=None,
        help="Write/maintain JSON state+report here (recommended for resume).",
    )

    p.add_argument(
        "--state-json",
        default=None,
        help="Optional separate JSON state file path. If not set, uses --out-json.",
    )

    p.add_argument("--reprocess", action="store_true", help="Ignore any existing state and reprocess everything.")

    p.add_argument("--min-duration-sec", type=float, default=2.0, help="Skip files shorter than this (sec).")

    # Move options
    p.add_argument(
        "--move-nonmajority-to",
        default=None,
        help="Move files where target is below required presence to this directory.",
    )

    p.add_argument(
        "--preserve-relative",
        action="store_true",
        help="Preserve input-dir relative subfolders under move destination.",
    )

    p.add_argument(
        "--on-collision",
        choices=["rename", "skip", "overwrite"],
        default="rename",
        help="What to do if destination filename already exists.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not actually move files; just report what WOULD be moved.",
    )

    return p.parse_args()


def pick_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Model
# -------------------------

def prepare_model_local_dir(model_id: str, cache_dir: str) -> str:
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

    # SpeechBrain sometimes expects label_encoder.ckpt
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
            "-c:a",
            "pcm_s16le",
            tmp_wav,
        ]
        subprocess.run(cmd, check=True)
        return _load_audio_torchaudio(pathlib.Path(tmp_wav))
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


def get_resampler(orig_sr: int) -> Optional[torchaudio.transforms.Resample]:
    if orig_sr == TARGET_SR:
        return None
    r = _G_RESAMPLERS.get(orig_sr)
    if r is None:
        r = torchaudio.transforms.Resample(orig_sr, TARGET_SR)
        _G_RESAMPLERS[orig_sr] = r
    return r


def peak_normalize(wavs_ct: torch.Tensor) -> torch.Tensor:
    peak = wavs_ct.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return wavs_ct / peak


def rms_dbfs(seg_ct: torch.Tensor) -> float:
    """RMS dBFS on mono mix (expects float audio around [-1, 1])."""
    mono = seg_ct.mean(dim=0)
    rms = torch.sqrt(torch.mean(mono * mono) + 1e-12)
    db = 20.0 * torch.log10(rms + 1e-12)
    return float(db.detach().cpu().item())


# -------------------------
# Speaker similarity
# -------------------------

def cosine_sims(embs_cd: torch.Tensor, ref_d: torch.Tensor) -> torch.Tensor:
    embs = torch.nn.functional.normalize(embs_cd, p=2, dim=1)
    ref = torch.nn.functional.normalize(ref_d, p=2, dim=0)
    return (embs @ ref.unsqueeze(1)).squeeze(1)


# -------------------------
# Worker init
# -------------------------

def worker_init(init_payload: Dict[str, Any]) -> None:
    global _G_ENCODER, _G_REF_VEC, _G_DEVICE, _G_ARGS, _G_STOP_FLAG

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    _G_DEVICE = str(init_payload["device"])
    model_dir = str(init_payload["model_dir"])
    _G_ENCODER = load_embedder(_G_DEVICE, model_dir)

    _G_REF_VEC = torch.tensor(init_payload["ref_vec_list"], dtype=torch.float32, device=_G_DEVICE)

    _G_ARGS = {
        "threshold": float(init_payload["threshold"]),
        "segment_seconds": float(init_payload["segment_seconds"]),
        "ffmpeg_exe": str(init_payload["ffmpeg_exe"]),
        "allow_ffmpeg_fallback": bool(init_payload["allow_ffmpeg_fallback"]),
        "min_duration_sec": float(init_payload["min_duration_sec"]),
        "min_presence_fraction": float(init_payload["min_presence_fraction"]),
        "min_presence_seconds": float(init_payload["min_presence_seconds"]),
        "vad_energy_db": float(init_payload["vad_energy_db"]),
        "min_voiced_segments": int(init_payload["min_voiced_segments"]),
        "max_debug_segments": int(init_payload["max_debug_segments"]),
    }

    _G_STOP_FLAG = init_payload.get("stop_flag")


def _should_stop() -> bool:
    try:
        return _G_STOP_FLAG is not None and bool(_G_STOP_FLAG.is_set())
    except Exception:
        return False


# -------------------------
# Per-file scoring (worker)
# -------------------------

def score_presence_one_file(abs_path_str: str) -> Tuple[str, Dict[str, Any]]:
    if _should_stop():
        return abs_path_str, {}

    global _G_ENCODER, _G_REF_VEC

    path = pathlib.Path(abs_path_str)
    mtime = path.stat().st_mtime

    try:
        assert _G_ENCODER is not None and _G_REF_VEC is not None

        wavs, sr = load_audio_any(path, _G_ARGS["ffmpeg_exe"], _G_ARGS["allow_ffmpeg_fallback"])
        if wavs.ndim == 1:
            wavs = wavs.unsqueeze(0)

        file_dur = wavs.shape[1] / float(sr)
        if file_dur < _G_ARGS["min_duration_sec"]:
            res = PresenceResult(
                present=False,
                max_similarity=-1.0,
                best_channel=0,
                best_segment_index=0,
                file_duration_sec=float(file_dur),
                total_voiced_seconds=0.0,
                required_seconds=0.0,
                scanned_seconds=0.0,
                voiced_scanned_seconds=0.0,
                target_present_seconds=0.0,
                presence_fraction_of_voiced=0.0,
                voiced_segments_total=0,
                voiced_segments_scanned=0,
                present_voiced_segments_scanned=0,
                segment_best_sims=[],
                segment_voiced_flags=[],
                segment_present_flags=[],
                stop_reason="TooShort",
                input_mtime=float(mtime),
                error=f"TooShort: {file_dur:.2f}s",
            )
            return abs_path_str, res.__dict__

        # Resample
        if sr != TARGET_SR:
            rs = get_resampler(int(sr))
            wavs_16k = wavs if rs is None else rs(wavs)
        else:
            wavs_16k = wavs

        seg_samples = int(max(1, round(float(_G_ARGS["segment_seconds"]) * TARGET_SR)))
        c, t = wavs_16k.shape

        vad_db = float(_G_ARGS["vad_energy_db"])
        thr = float(_G_ARGS["threshold"])
        max_debug = int(_G_ARGS["max_debug_segments"])

        # Build segment boundaries
        boundaries: List[Tuple[int, int]] = []
        start = 0
        while start < t:
            end = min(t, start + seg_samples)
            boundaries.append((start, end))
            start = end

        # PASS 1 (cheap): VAD only -> total voiced seconds + voiced flags
        voiced_flags: List[bool] = []
        total_voiced_seconds = 0.0
        voiced_segments_total = 0

        for (s, e) in boundaries:
            seg_raw = wavs_16k[:, s:e]
            seg_len_sec = (e - s) / float(TARGET_SR)
            seg_db = rms_dbfs(seg_raw)
            voiced = bool(seg_db >= vad_db)
            voiced_flags.append(voiced)
            if voiced:
                voiced_segments_total += 1
                total_voiced_seconds += seg_len_sec

        if voiced_segments_total < int(_G_ARGS["min_voiced_segments"]):
            res = PresenceResult(
                present=False,
                max_similarity=-1.0,
                best_channel=0,
                best_segment_index=0,
                file_duration_sec=float(file_dur),
                total_voiced_seconds=float(total_voiced_seconds),
                required_seconds=0.0,
                scanned_seconds=0.0,
                voiced_scanned_seconds=0.0,
                target_present_seconds=0.0,
                presence_fraction_of_voiced=0.0,
                voiced_segments_total=int(voiced_segments_total),
                voiced_segments_scanned=0,
                present_voiced_segments_scanned=0,
                segment_best_sims=[],
                segment_voiced_flags=[],
                segment_present_flags=[],
                stop_reason="NotEnoughVoicedSegments",
                input_mtime=float(mtime),
                error=f"NotEnoughVoicedSegments: {voiced_segments_total}",
            )
            return abs_path_str, res.__dict__

        required = max(
            float(_G_ARGS["min_presence_fraction"]) * float(total_voiced_seconds),
            float(_G_ARGS["min_presence_seconds"]),
        )

        # PASS 2: embeddings only on voiced segments until early stop
        scanned_seconds = 0.0
        voiced_scanned_seconds = 0.0
        present_seconds = 0.0

        voiced_segments_scanned = 0
        present_voiced_segments_scanned = 0

        best = -1e9
        best_ch = 0
        best_seg = 0

        # debug arrays (capped)
        segment_best_sims: List[float] = []
        segment_voiced_flags: List[bool] = []
        segment_present_flags: List[bool] = []

        stop_reason = "EndOfFile"

        for seg_idx, ((s, e), voiced) in enumerate(zip(boundaries, voiced_flags)):
            if _should_stop():
                stop_reason = "Stopped"
                break

            seg_raw = wavs_16k[:, s:e]
            seg_len_sec = (e - s) / float(TARGET_SR)
            scanned_seconds = (e / float(TARGET_SR))  # timeline position of scan

            # record voiced flag in debug (capped)
            if max_debug > 0 and seg_idx < max_debug:
                segment_voiced_flags.append(bool(voiced))

            if not voiced:
                # silence ignored entirely for denominator and for embeddings
                if max_debug > 0 and seg_idx < max_debug:
                    segment_best_sims.append(-1.0)
                    segment_present_flags.append(False)
                continue

            voiced_segments_scanned += 1
            voiced_scanned_seconds += seg_len_sec

            # embeddings on normalized segment
            seg = peak_normalize(seg_raw).to(_G_ENCODER.device)
            with torch.no_grad():
                embs = _G_ENCODER.encode_batch(seg)
            if embs.ndim == 3:
                embs = embs.squeeze(1)

            sims = cosine_sims(embs, _G_REF_VEC).detach().cpu().tolist()
            sims_f = [float(x) for x in sims]
            seg_best = max(sims_f) if sims_f else -1e9
            seg_present = bool(seg_best >= thr)

            # update global best
            for ch, sv in enumerate(sims_f):
                if sv > best:
                    best = sv
                    best_ch = ch
                    best_seg = seg_idx

            if seg_present:
                present_voiced_segments_scanned += 1
                present_seconds += seg_len_sec

            # debug
            if max_debug > 0 and seg_idx < max_debug:
                segment_best_sims.append(float(seg_best))
                segment_present_flags.append(bool(seg_present))

            # EARLY STOP KEEP
            if present_seconds >= required:
                stop_reason = "EnoughPresence"
                break

            # EARLY STOP MOVE (cannot reach even if all remaining voiced is present)
            remaining_voiced = max(0.0, float(total_voiced_seconds) - float(voiced_scanned_seconds))
            if present_seconds + remaining_voiced < required:
                stop_reason = "CannotReachThreshold"
                break

        keep = bool(present_seconds >= required)
        frac_voiced = float(present_seconds) / float(total_voiced_seconds) if total_voiced_seconds > 0 else 0.0

        res = PresenceResult(
            present=keep,
            max_similarity=float(best if best > -1e8 else -1.0),
            best_channel=int(best_ch),
            best_segment_index=int(best_seg),
            file_duration_sec=float(file_dur),
            total_voiced_seconds=float(total_voiced_seconds),
            required_seconds=float(required),
            scanned_seconds=float(scanned_seconds),
            voiced_scanned_seconds=float(voiced_scanned_seconds),
            target_present_seconds=float(present_seconds),
            presence_fraction_of_voiced=float(frac_voiced),
            voiced_segments_total=int(voiced_segments_total),
            voiced_segments_scanned=int(voiced_segments_scanned),
            present_voiced_segments_scanned=int(present_voiced_segments_scanned),
            segment_best_sims=segment_best_sims,
            segment_voiced_flags=segment_voiced_flags,
            segment_present_flags=segment_present_flags,
            stop_reason=str(stop_reason),
            input_mtime=float(mtime),
            error="",
        )
        return abs_path_str, res.__dict__

    except Exception as e:
        res = PresenceResult(
            present=False,
            max_similarity=-1.0,
            best_channel=0,
            best_segment_index=0,
            file_duration_sec=0.0,
            total_voiced_seconds=0.0,
            required_seconds=0.0,
            scanned_seconds=0.0,
            voiced_scanned_seconds=0.0,
            target_present_seconds=0.0,
            presence_fraction_of_voiced=0.0,
            voiced_segments_total=0,
            voiced_segments_scanned=0,
            present_voiced_segments_scanned=0,
            segment_best_sims=[],
            segment_voiced_flags=[],
            segment_present_flags=[],
            stop_reason="Error",
            input_mtime=float(mtime),
            error=f"{type(e).__name__}: {e}",
        )
        return abs_path_str, res.__dict__


# -------------------------
# Reference vector
# -------------------------

def compute_reference_vector(
    reference_path: pathlib.Path,
    device: str,
    model_dir: str,
    ffmpeg_exe: str,
    allow_ffmpeg_fallback: bool,
    segment_seconds: float,
) -> List[float]:
    encoder = load_embedder(device, model_dir)
    wavs, sr = load_audio_any(reference_path, ffmpeg_exe=ffmpeg_exe, allow_ffmpeg_fallback=allow_ffmpeg_fallback)
    if wavs.ndim == 1:
        wavs = wavs.unsqueeze(0)

    if sr != TARGET_SR:
        rs = torchaudio.transforms.Resample(int(sr), TARGET_SR)
        wavs_16k = rs(wavs)
    else:
        wavs_16k = wavs

    mono = wavs_16k.mean(dim=0, keepdim=True)
    mono = peak_normalize(mono)

    seg_samples = int(max(1, round(segment_seconds * TARGET_SR)))
    seg = mono[:, : min(mono.shape[1], seg_samples)]

    with torch.no_grad():
        emb = encoder.encode_batch(seg.to(encoder.device))
    if emb.ndim == 3:
        emb = emb.squeeze(1)
    emb0 = emb[0]
    return emb0.detach().cpu().to(torch.float32).tolist()


# -------------------------
# Discovery
# -------------------------

def discover_audio_files(
    root: pathlib.Path,
    exts: Iterable[str],
    exclude_roots: Optional[List[pathlib.Path]] = None,
) -> List[pathlib.Path]:
    exts_l = {e.lower().strip() for e in exts if e.strip()}
    exclude_abs: List[pathlib.Path] = []
    if exclude_roots:
        for x in exclude_roots:
            try:
                exclude_abs.append(x.resolve())
            except Exception:
                exclude_abs.append(x)

    out: List[pathlib.Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        if exclude_abs:
            skip = False
            for exr in exclude_abs:
                try:
                    if str(rp).lower().startswith(str(exr).lower() + os.sep):
                        skip = True
                        break
                except Exception:
                    pass
            if skip:
                continue
        if p.suffix.lower() in exts_l:
            out.append(p)
    return out


# -------------------------
# State I/O (resumable)
# -------------------------

def _atomic_write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def load_state(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def rebuild_txt_from_state(out_txt: pathlib.Path, state: Dict[str, Any]) -> None:
    results = state.get("results", {}) if isinstance(state, dict) else {}
    lines: List[str] = []
    if isinstance(results, dict):
        for src, r in results.items():
            if not isinstance(r, dict):
                continue
            if r.get("error"):
                continue
            # moved condition => present == False
            if bool(r.get("present")):
                continue
            lines.append(str(r.get("moved_to") or src))

    lines = sorted(set(lines))
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w", encoding="utf-8", newline="") as f:
        for line in lines:
            f.write(line + " ")


# -------------------------
# Move helpers
# -------------------------

def _ensure_unique_path(dst: pathlib.Path) -> pathlib.Path:
    if not dst.exists():
        return dst
    stem = dst.stem
    suf = dst.suffix
    parent = dst.parent
    n = 1
    while True:
        cand = parent / f"{stem} ({n}){suf}"
        if not cand.exists():
            return cand
        n += 1


def move_one_file(
    src: pathlib.Path,
    input_dir: pathlib.Path,
    dest_dir: pathlib.Path,
    preserve_relative: bool,
    on_collision: str,
    dry_run: bool,
) -> str:
    dest_dir.mkdir(parents=True, exist_ok=True)

    if preserve_relative:
        try:
            rel = src.resolve().relative_to(input_dir.resolve())
        except Exception:
            rel = pathlib.Path(src.name)
        dst = (dest_dir / rel).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
    else:
        dst = (dest_dir / src.name).resolve()

    if dst.exists():
        if on_collision == "skip":
            return str(dst)
        if on_collision == "overwrite":
            try:
                dst.unlink()
            except Exception:
                dst = _ensure_unique_path(dst)
        else:
            dst = _ensure_unique_path(dst)

    if dry_run:
        return str(dst)

    final_path = shutil.move(str(src), str(dst))
    return str(pathlib.Path(final_path).resolve())


# -------------------------
# Main
# -------------------------

def main() -> None:
    args = parse_args()

    input_dir = pathlib.Path(args.input_dir)
    exts = [e.strip() for e in args.audio_exts.split(",") if e.strip()]

    device = pick_device(args.device)
    workers = args.workers or mp.cpu_count()
    if device == "cuda" and workers != 1:
        print(f"[device] CUDA detected; forcing workers=1 (was {workers})")
        workers = 1

    allow_ffmpeg_fallback = not args.no_ffmpeg_fallback

    # State path: prefer explicit --state-json, else use --out-json
    state_path: Optional[pathlib.Path] = None
    if args.state_json:
        state_path = pathlib.Path(args.state_json)
    elif args.out_json:
        state_path = pathlib.Path(args.out_json)

    # Load state (resume)
    state: Dict[str, Any] = {}
    if state_path and state_path.exists() and not args.reprocess:
        state = load_state(state_path)
        if state:
            print(f"[resume] loaded state: {state_path}")

    if args.reprocess and state_path and state_path.exists():
        print(f"[reprocess] ignoring existing state: {state_path}")
        state = {}

    prior_results = state.get("results", {}) if isinstance(state, dict) else {}
    if not isinstance(prior_results, dict):
        prior_results = {}

    # If out-txt exists and we're resuming, rebuild it to avoid duplicates
    if args.out_txt and state and not args.reprocess:
        rebuild_txt_from_state(pathlib.Path(args.out_txt), state)
        print(f"[resume] rebuilt txt: {args.out_txt}")

    model_dir = prepare_model_local_dir(args.model_id, args.cache_dir)

    print(f"[device] {device}")
    print(f"[model_dir] {model_dir}")
    print(f"[threshold] {args.threshold}")
    print(f"[min_presence_fraction (of voiced)] {args.min_presence_fraction}")
    print(f"[min_presence_seconds] {args.min_presence_seconds}")
    print(f"[segment_seconds] {args.segment_seconds}")
    print(f"[vad_energy_db] {args.vad_energy_db}")
    print(f"[workers] {workers}")
    if args.move_nonmajority_to:
        print(f"[move_to] {args.move_nonmajority_to} (dry_run={args.dry_run})")

    ref_vec_list = compute_reference_vector(
        reference_path=pathlib.Path(args.reference),
        device=device,
        model_dir=model_dir,
        ffmpeg_exe=args.ffmpeg,
        allow_ffmpeg_fallback=allow_ffmpeg_fallback,
        segment_seconds=args.segment_seconds,
    )

    # Exclude move destination if it sits inside input-dir
    exclude_roots: List[pathlib.Path] = []
    if args.move_nonmajority_to:
        try:
            dest = pathlib.Path(args.move_nonmajority_to).resolve()
            root = input_dir.resolve()
            if str(dest).lower().startswith(str(root).lower() + os.sep):
                exclude_roots.append(dest)
        except Exception:
            pass

    files = discover_audio_files(input_dir, exts, exclude_roots=exclude_roots)

    # Skip already processed files
    todo: List[pathlib.Path] = []
    for p in files:
        rp = str(p.resolve())
        if rp in prior_results and not args.reprocess:
            continue
        todo.append(p)

    print(f"[discovery] total={len(files)} todo={len(todo)} already_done={len(files) - len(todo)}")

    # Prepare stop flag for Ctrl+C
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    stop_flag = manager.Event()

    def _handle_sigint(sig, frame):
        stop_flag.set()
        print("[ctrl+c] stopping…")

    signal.signal(signal.SIGINT, _handle_sigint)

    init_payload = {
        "device": device,
        "model_dir": model_dir,
        "ref_vec_list": ref_vec_list,
        "threshold": float(args.threshold),
        "segment_seconds": float(args.segment_seconds),
        "ffmpeg_exe": str(args.ffmpeg),
        "allow_ffmpeg_fallback": bool(allow_ffmpeg_fallback),
        "min_duration_sec": float(args.min_duration_sec),
        "min_presence_fraction": float(args.min_presence_fraction),
        "min_presence_seconds": float(args.min_presence_seconds),
        "vad_energy_db": float(args.vad_energy_db),
        "min_voiced_segments": int(args.min_voiced_segments),
        "max_debug_segments": int(args.max_debug_segments),
        "stop_flag": stop_flag,
    }

    # Ensure state skeleton
    if not state:
        state = {
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "device": device,
            "threshold": float(args.threshold),
            "min_presence_fraction": float(args.min_presence_fraction),
            "min_presence_seconds": float(args.min_presence_seconds),
            "segment_seconds": float(args.segment_seconds),
            "vad_energy_db": float(args.vad_energy_db),
            "min_voiced_segments": int(args.min_voiced_segments),
            "max_debug_segments": int(args.max_debug_segments),
            "input_dir": str(input_dir),
            "move_to": str(args.move_nonmajority_to) if args.move_nonmajority_to else "",
            "preserve_relative": bool(args.preserve_relative),
            "on_collision": str(args.on_collision),
            "dry_run": bool(args.dry_run),
            "results": {},
            "moved_map": {},
            "move_errors": [],
        }

    if "results" not in state or not isinstance(state.get("results"), dict):
        state["results"] = {}
    if "moved_map" not in state or not isinstance(state.get("moved_map"), dict):
        state["moved_map"] = {}
    if "move_errors" not in state or not isinstance(state.get("move_errors"), list):
        state["move_errors"] = []

    # Open txt in append mode for streaming writes (rebuild on resume + rebuild at end)
    txt_fh = None
    if args.out_txt:
        out_txt_p = pathlib.Path(args.out_txt)
        out_txt_p.parent.mkdir(parents=True, exist_ok=True)
        txt_fh = out_txt_p.open("a", encoding="utf-8", newline="")

    ex: Optional[ProcessPoolExecutor] = None
    processed_now = 0

    try:
        ex = ProcessPoolExecutor(
            max_workers=workers,
            initializer=worker_init,
            initargs=(init_payload,),
            mp_context=ctx,
        )

        futs = {ex.submit(score_presence_one_file, str(p.resolve())): str(p.resolve()) for p in todo}

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Scoring + moving", unit="file"):
            if stop_flag.is_set():
                break

            src_path, res = fut.result()
            if not res:
                continue

            # STREAMING MOVE (main process): move if KEEP is False and error-free
            try:
                is_error_free = (not res.get("error"))
                keep = bool(res.get("present"))

                if args.move_nonmajority_to and is_error_free and (not keep):
                    src_p = pathlib.Path(src_path)
                    dest_dir = pathlib.Path(args.move_nonmajority_to)
                    moved_to = move_one_file(
                        src=src_p,
                        input_dir=input_dir,
                        dest_dir=dest_dir,
                        preserve_relative=bool(args.preserve_relative),
                        on_collision=str(args.on_collision),
                        dry_run=bool(args.dry_run),
                    )
                    res["moved_to"] = moved_to
                    state["moved_map"][src_path] = moved_to

                    if txt_fh is not None:
                        # CRLF per line to avoid one-line concatenation on Windows
                        txt_fh.write((moved_to or src_path) + "\n")
                        txt_fh.flush()

            except Exception as e:
                err_line = f"{src_path} | MoveError {type(e).__name__}: {e}"
                state["move_errors"].append(err_line)

            # Save result into state
            state["results"][src_path] = res
            processed_now += 1

            # Persist state after each file (resumable)
            if state_path:
                state["updated_at"] = _dt.datetime.now().isoformat(timespec="seconds")
                state["processed_now"] = processed_now
                _atomic_write_json(state_path, state)

    except KeyboardInterrupt:
        stop_flag.set()

    finally:
        if txt_fh is not None:
            try:
                txt_fh.close()
            except Exception:
                pass
        if ex is not None:
            try:
                ex.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    # Final summary from state
    results = state.get("results", {}) if isinstance(state, dict) else {}

    kept: List[str] = []
    moved: List[str] = []
    errors: List[str] = []

    if isinstance(results, dict):
        for k, r in results.items():
            if not isinstance(r, dict):
                continue
            if r.get("error"):
                errors.append(k)
            if bool(r.get("present")):
                kept.append(k)
            else:
                moved.append(k)

    kept.sort()
    moved.sort()
    errors.sort()

    moved_count = len(state.get("moved_map", {}) or {})
    move_err_count = len(state.get("move_errors", []) or [])

    print(
        f"[summary] keep={len(kept)} move_candidates={len(moved)} "
        f"errors={len(errors)} moved={moved_count} move_errors={move_err_count}"
    )

    # Rebuild txt at end (dedup + correct moved paths)
    if args.out_txt:
        rebuild_txt_from_state(pathlib.Path(args.out_txt), state)

    # Final write
    if args.out_json:
        out_json = pathlib.Path(args.out_json)
        final_payload = dict(state)
        final_payload.update(
            {
                "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
                "keep_count": len(kept),
                "move_candidate_count": len(moved),
                "error_count": len(errors),
                "moved_count": moved_count,
                "move_error_count": move_err_count,
                "kept_files": kept,
                "move_candidate_files": moved,
            }
        )
        _atomic_write_json(out_json, final_payload)
        print("[write] report/state ->", out_json)


if __name__ == "__main__":
    main()