#!/usr/bin/env python3
r"""stage_19e_edge_and_moonshine_targetmix_sweep_with_cer.py

Stage 19 evaluation (TARGET vs OTHER mixtures) with SWEPT conditions.

Why this exists
---------------
A single SNR / single overlap setting can lie to you.
This version runs a grid (sweep) over:
  - SNR (TARGET over OTHER, dB)
  - Overlap placement (where OTHER starts inside TARGET)

For every mixture, we score the model output against:
  - TARGET transcript (should be LOW WER)
  - OTHER transcript  (should be HIGH WER)
And compute:
  - win_rate = P(WER_target < WER_other)
  - avg_margin = mean(WER_other - WER_target)

Key constraints
---------------
- The interfering OTHER audio is scaled so that peak(|OTHER|) never exceeds peak(|TARGET|) * other_peak_ratio.
  (Default ratio=1.0 => OTHER is never louder than TARGET at any instant, in peak amplitude terms.)

Resume-safe
-----------
- Each mixture has a stable key: target+other+condition.
- If you rerun with --resume, already-computed (pair,model) predictions are skipped.
- If you rerun with --force_resume, metrics are recalculated from existing predictions.


usage
---------------------------------------------------

i:\Whisper-training-env\Scripts\python.exe "i:\whisper-acft\stage_19e_edge_and_moonshine_targetmix_sweep_with_cer.py" `
  --test_manifest "I:\Record_chunks\pairs_manifest_stage13_test_randomized.jsonl" `
  --speaker_scores_csv "I:\whisper-acft\speaker_sort_scores.csv" `
  --checkpoint_dir "I:\asr_edge_eval_runs\run_01" `
  --mix_per_target 5 `
  --other_peak_ratio 1.0 `
  --sweep_snr_db "15,5,0" `
  --sweep_overlap "0.25,0.75,1" `
  --batch_size 1 `
  --auto_batch 0 `
  --resume `
  --others_dir "I:\Record_others_chunks" `
  --others_manifest "I:\Record_others_chunks\pairs_pending_stereo.jsonl" `
  --ffmpeg_path "ffmpeg" `
  --percentage 100 `
  --vad_filter 0 `
  --models_csv "openai/whisper-tiny.en,openai/whisper-base.en,openai/whisper-small.en,distil-whisper/distil-small.en,UsefulSensors/moonshine-tiny,UsefulSensors/moonshine-base,UsefulSensors/moonshine-streaming-tiny,UsefulSensors/moonshine-streaming-small,UsefulSensors/moonshine-streaming-medium"


Outputs (in checkpoint_dir by default)
-------------------------------------
- evaluation_results_futo_like_targetmix_sweep.json
- evaluation_per_sample_predictions_targetmix_sweep.json

Dependencies
------------
pip install transformers torch soundfile jiwer tqdm numpy
Optional (better resample): scipy
Optional (VAD): torch hub will pull Silero-VAD
For Moonshine Streaming with latest support:
pip install --upgrade git+https://github.com/huggingface/transformers.git
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import hashlib
import os
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Tuple, Optional, Iterable, Any
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

import soundfile as sf
import jiwer
from transformers import (
    AutoConfig,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperForConditionalGeneration,
)


DEFAULT_EDGE_MODEL_PACK: List[str] = [
    "openai/whisper-tiny.en",
    "openai/whisper-base.en",
    "openai/whisper-small.en",
    "distil-whisper/distil-small.en",
    "UsefulSensors/moonshine-tiny",
    "UsefulSensors/moonshine-base",
    "UsefulSensors/moonshine-streaming-tiny",
    "UsefulSensors/moonshine-streaming-small",
    "UsefulSensors/moonshine-streaming-medium",
]


# ----------------------------
# Text normalisation
# ----------------------------

def _basic_whisperish_normalize(s: str) -> str:
    """Pragmatic normaliser for WER."""
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    # remove most punctuation, keep apostrophes inside words
    s = re.sub(r"(?!\B'\b)[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in ("<|nospeech|>", "<|nocaptions|>"):
        return ""
    return s


# ----------------------------
# Path matching helpers
# ----------------------------

def _norm_windows_key(p: str) -> str:
    p = (p or "").strip().strip('"').strip("'")
    p = p.replace("/", "\\")
    try:
        p = str(PureWindowsPath(p))
    except Exception:
        p = os.path.normpath(p)
    return p.lower()


def _basename(p: str) -> str:
    p = (p or "").strip().strip('"').strip("'")
    p = p.replace("\\", "/")
    return p.split("/")[-1].lower()


# ----------------------------
# Deterministic subsampling helpers
# ----------------------------

def _stable_u64(s: str) -> int:
    """Stable 64-bit hash for deterministic sampling across runs."""
    h = hashlib.sha1((s or "").encode("utf-8", errors="ignore")).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def _deterministic_subsample(items: List[dict], key_fn, percentage: float, seed: int) -> List[dict]:
    """Return a deterministic subset (exact count) based on stable hashing + sorting."""
    if percentage >= 100.0:
        return list(items)
    if percentage <= 0.0:
        return []

    tagged = [(_stable_u64(f"{int(seed)}||{key_fn(x)}"), x) for x in items]
    tagged.sort(key=lambda t: t[0])
    k = max(1, int(len(tagged) * (float(percentage) / 100.0)))
    return [x for _, x in tagged[:k]]


def _deterministic_cap(items: List[dict], key_fn, max_n: int, seed: int) -> List[dict]:
    """Return a deterministic cap (first N after stable hash sort)."""
    if max_n <= 0 or len(items) <= max_n:
        return list(items)
    tagged = [(_stable_u64(f"{int(seed)}||{key_fn(x)}"), x) for x in items]
    tagged.sort(key=lambda t: t[0])
    return [x for _, x in tagged[: int(max_n)]]


# ----------------------------
# CSV: speaker_sort_scores.csv
# ----------------------------

@dataclass
class SpeakerLabelDB:
    by_path: Dict[str, str]        # normalised full path -> decision
    by_basename: Dict[str, str]    # basename -> decision (ONLY if unique)

    def decision_for(self, audio_path: str) -> Optional[str]:
        k = _norm_windows_key(audio_path)
        if k in self.by_path:
            return self.by_path[k]
        b = _basename(audio_path)
        return self.by_basename.get(b)


def load_speaker_sort_scores(csv_path: Path) -> SpeakerLabelDB:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    by_path: Dict[str, str] = {}
    basename_counts: Dict[str, Dict[str, int]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"file", "decision"}
        if not required.issubset(set((reader.fieldnames or []))):
            raise ValueError(f"CSV missing required columns: {required}. Found: {reader.fieldnames}")

        for row in reader:
            fp = row.get("file", "")
            decision = (row.get("decision", "") or "").strip().upper()
            if not fp:
                continue
            if decision not in {"TARGET", "OTHER"}:
                continue

            nk = _norm_windows_key(fp)
            by_path[nk] = decision

            b = _basename(fp)
            basename_counts.setdefault(b, {})
            basename_counts[b][decision] = basename_counts[b].get(decision, 0) + 1

    by_basename: Dict[str, str] = {}
    for b, counts in basename_counts.items():
        if len(counts) == 1:
            by_basename[b] = next(iter(counts.keys()))

    return SpeakerLabelDB(by_path=by_path, by_basename=by_basename)

# ---- Extended speaker CSV with scores (for score-sorted target selection when mixing with external others)

@dataclass
class SpeakerScoreDB:
    decision_by_path: Dict[str, str]
    decision_by_basename: Dict[str, str]
    score_by_path: Dict[str, float]
    score_by_basename: Dict[str, float]

    def decision_for(self, audio_path: str) -> Optional[str]:
        k = _norm_windows_key(audio_path)
        if k in self.decision_by_path:
            return self.decision_by_path[k]
        b = _basename(audio_path)
        return self.decision_by_basename.get(b)

    def score_for(self, audio_path: str) -> Optional[float]:
        k = _norm_windows_key(audio_path)
        if k in self.score_by_path:
            return self.score_by_path[k]
        b = _basename(audio_path)
        return self.score_by_basename.get(b)


def load_speaker_scores_csv(csv_path: Path) -> SpeakerScoreDB:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    decision_by_path: Dict[str, str] = {}
    score_by_path: Dict[str, float] = {}
    basename_decisions: Dict[str, Dict[str, int]] = {}
    basename_scores: Dict[str, List[float]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        if "file" not in reader.fieldnames or "decision" not in reader.fieldnames:
            raise ValueError(f"CSV must contain 'file' and 'decision' columns. Found: {reader.fieldnames}")

        for row in reader:
            fp = (row.get("file") or "").strip()
            if not fp:
                continue

            decision = (row.get("decision") or "").strip().upper()
            if decision not in {"TARGET", "OTHER"}:
                continue

            score_raw = (row.get("score") or "").strip()
            score: Optional[float] = None
            if score_raw:
                try:
                    score = float(score_raw)
                except Exception:
                    score = None

            nk = _norm_windows_key(fp)
            decision_by_path[nk] = decision
            if score is not None:
                score_by_path[nk] = score

            b = _basename(fp)
            basename_decisions.setdefault(b, {})
            basename_decisions[b][decision] = basename_decisions[b].get(decision, 0) + 1
            if score is not None:
                basename_scores.setdefault(b, []).append(score)

    decision_by_basename: Dict[str, str] = {}
    score_by_basename: Dict[str, float] = {}

    for b, counts in basename_decisions.items():
        if len(counts) == 1:
            decision_by_basename[b] = next(iter(counts.keys()))
            if b in basename_scores and basename_scores[b]:
                score_by_basename[b] = float(max(basename_scores[b]))

    return SpeakerScoreDB(
        decision_by_path=decision_by_path,
        decision_by_basename=decision_by_basename,
        score_by_path=score_by_path,
        score_by_basename=score_by_basename,
    )


# ----------------------------
# Audio: load/cache/resample
# ----------------------------

class AudioCacheLRU:
    """LRU cache for mono 16k float32 audio arrays."""

    def __init__(self, max_bytes: int):
        self.max_bytes = int(max(0, max_bytes))
        self._od: "OrderedDict[str, Tuple[np.ndarray, int, int]]" = OrderedDict()
        self._bytes = 0

    def get(self, key: str) -> Optional[Tuple[np.ndarray, int]]:
        if self.max_bytes <= 0:
            return None
        k = _norm_windows_key(key)
        if k not in self._od:
            return None
        audio, sr, b = self._od.pop(k)
        self._od[k] = (audio, sr, b)
        return audio, sr

    def put(self, key: str, audio: np.ndarray, sr: int) -> None:
        if self.max_bytes <= 0:
            return
        k = _norm_windows_key(key)
        b = int(audio.nbytes)
        if b > self.max_bytes:
            return
        if k in self._od:
            _, _, oldb = self._od.pop(k)
            self._bytes -= oldb
        self._od[k] = (audio, sr, b)
        self._bytes += b
        while self._bytes > self.max_bytes and self._od:
            _, (_, _, evb) = self._od.popitem(last=False)
            self._bytes -= evb


def load_audio_mono_16k(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)

    # stereo -> mono
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if sr != 16000:
        try:
            import scipy.signal  # type: ignore
            audio = scipy.signal.resample_poly(audio, 16000, sr).astype(np.float32)
            sr = 16000
        except Exception:
            x_old = np.linspace(0, 1, num=len(audio), endpoint=False)
            new_len = int(round(len(audio) * (16000.0 / float(sr))))
            x_new = np.linspace(0, 1, num=new_len, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
            sr = 16000

    return audio, sr


def make_loader_with_ffmpeg(ffmpeg_path: str, core_load_orig):
    def _load_audio_mono_16k(p: Path) -> Tuple[np.ndarray, int]:
        try:
            audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if isinstance(audio, np.ndarray) and audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio = np.asarray(audio, dtype=np.float32)
            if not np.isfinite(audio).all():
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            if sr != 16000:
                return core_load_orig(p)
            return audio, int(sr)
        except Exception:
            pass

        cmd = [
            ffmpeg_path,
            "-v",
            "error",
            "-i",
            str(p),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg failed for {p}: {err}")

        audio = np.frombuffer(proc.stdout, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError(f"ffmpeg produced empty audio for {p}")
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return audio.astype(np.float32), 16000

    return _load_audio_mono_16k


def seconds_from_audio(audio: np.ndarray, sr: int) -> float:
    return float(len(audio)) / float(sr)


def rms(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + eps))


# ----------------------------
# Mixing with overlap placement
# ----------------------------

@dataclass(frozen=True)
class MixCondition:
    snr_db: float
    overlap: Optional[float]  # None => use other_offset_mode; else ratio in [0,1]

    def cond_id(self) -> str:
        if self.overlap is None:
            return f"snr{self.snr_db:+g}dB_ovNA"
        # stable compact formatting
        ov = float(self.overlap)
        return f"snr{self.snr_db:+g}dB_ov{ov:.2f}"


def _stable_uint64_from_str(s: str, seed: int) -> int:
    b = (str(int(seed)) + "||" + (s or "")).encode("utf-8", errors="ignore")
    h = hashlib.md5(b).digest()
    return int.from_bytes(h[:8], "little", signed=False)


def mix_target_with_other(
    target: np.ndarray,
    other: np.ndarray,
    snr_db_target_over_other: float,
    other_offset_mode: str,
    rng: np.random.Generator,
    other_peak_ratio: float = 1.0,
    max_abs: float = 0.999,
    overlap_ratio: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """Return mixture where TARGET dominates OTHER by desired SNR (RMS).

    If overlap_ratio is provided:
        - It defines where OTHER starts inside TARGET as a fraction in [0,1].
          0.0 => start aligned, 1.0 => end aligned, 0.5 => centred.
        - If OTHER is longer than TARGET, we crop OTHER with the same ratio logic.

    Enforces: peak(|other_scaled|) <= peak(|target|) * other_peak_ratio.
    """
    t = np.asarray(target, dtype=np.float32)
    o = np.asarray(other, dtype=np.float32)

    t_len = len(t)
    if t_len == 0:
        return t, {"note": "empty_target"}

    if len(o) == 0:
        return t, {"note": "empty_other"}

    meta: Dict = {
        "snr_db_target_over_other_req": float(snr_db_target_over_other),
        "other_offset_mode": other_offset_mode,
        "overlap_ratio": None if overlap_ratio is None else float(overlap_ratio),
        "other_peak_ratio": float(other_peak_ratio),
    }

    # 1) Crop/align OTHER into TARGET-length buffer
    # Crop if other longer than target.
    if len(o) > t_len:
        if overlap_ratio is not None:
            r = float(max(0.0, min(1.0, overlap_ratio)))
            start_o = int(round(r * (len(o) - t_len)))
            start_o = max(0, min(start_o, len(o) - t_len))
            o = o[start_o:start_o + t_len]
            meta["other_crop_start"] = int(start_o)
        elif other_offset_mode == "random":
            start_o = int(rng.integers(0, len(o) - t_len + 1))
            o = o[start_o:start_o + t_len]
            meta["other_crop_start"] = int(start_o)
        else:
            o = o[:t_len]
            meta["other_crop_start"] = 0

    o_aligned = np.zeros_like(t)

    # Align start position inside TARGET
    if len(o) < t_len:
        if overlap_ratio is not None:
            r = float(max(0.0, min(1.0, overlap_ratio)))
            start_t = int(round(r * (t_len - len(o))))
            start_t = max(0, min(start_t, t_len - len(o)))
        elif other_offset_mode == "random":
            start_t = int(rng.integers(0, t_len - len(o) + 1))
        else:
            start_t = 0
    else:
        start_t = 0

    o_aligned[start_t:start_t + len(o)] = o
    meta["other_start_sample"] = int(start_t)

    # 2) Scale OTHER to requested SNR (RMS)
    rt = rms(t)
    ro = rms(o_aligned)
    meta["rms_target"] = float(rt)
    meta["rms_other_raw"] = float(ro)

    if ro < 1e-8 or rt < 1e-8:
        mix = np.clip(t + o_aligned, -max_abs, max_abs).astype(np.float32)
        meta["note"] = "degenerate_rms"
        return mix, meta

    snr_lin = 10.0 ** (float(snr_db_target_over_other) / 20.0)
    desired_ro = rt / snr_lin
    scale = desired_ro / ro
    o_scaled = o_aligned * float(scale)

    # 3) Enforce peak constraint (other never louder than target peak)
    peak_t = float(np.max(np.abs(t))) + 1e-12
    peak_o = float(np.max(np.abs(o_scaled)))
    meta["peak_target"] = peak_t
    meta["peak_other_scaled_prepeakcap"] = peak_o

    if peak_o > peak_t * float(other_peak_ratio):
        scale2 = (peak_t * float(other_peak_ratio)) / (peak_o + 1e-12)
        o_scaled = o_scaled * float(scale2)
        scale = scale * float(scale2)
        meta["peakcap_applied"] = True
        meta["peakcap_scale2"] = float(scale2)
    else:
        meta["peakcap_applied"] = False

    # 4) Sum and prevent clipping
    mix = t + o_scaled
    peak_mix = float(np.max(np.abs(mix))) + 1e-12
    if peak_mix > max_abs:
        mix = mix * float(max_abs / peak_mix)
        meta["mix_clip_scaling_applied"] = True
        meta["mix_clip_scale"] = float(max_abs / peak_mix)
    else:
        meta["mix_clip_scaling_applied"] = False

    ro2 = rms(o_scaled)
    snr_actual = 20.0 * math.log10((rt + 1e-12) / (ro2 + 1e-12))

    meta.update(
        {
            "scale_other": float(scale),
            "rms_other_scaled": float(ro2),
            "snr_db_target_over_other_actual": float(snr_actual),
            "peak_other_scaled": float(np.max(np.abs(o_scaled))),
            "peak_mix": float(np.max(np.abs(mix))),
        }
    )

    return mix.astype(np.float32), meta


# ----------------------------
# Dynamic audio context (mel cropping)
# ----------------------------

FULL_MEL_FRAMES = 3000  # Whisper pads/truncates to 30s => ~3000 frames


def mel_frames_for_duration(duration_sec: float) -> int:
    d = max(0.0, min(30.0, float(duration_sec)))
    frames = int(round((FULL_MEL_FRAMES / 30.0) * d))
    return max(1, min(FULL_MEL_FRAMES, frames))


def crop_input_features_for_duration(input_features: torch.Tensor, duration_sec: float) -> torch.Tensor:
    frames = mel_frames_for_duration(duration_sec)
    if input_features.ndim != 3:
        return input_features
    b, d1, d2 = input_features.shape
    if d1 == 80:
        return input_features[:, :, :frames]
    if d2 == 80:
        return input_features[:, :frames, :]
    return input_features


# ----------------------------
# Optional: VAD trimming (Silero)
# ----------------------------

@dataclass
class VADConfig:
    enabled: bool
    policy: str  # skip | keep | empty
    threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int


class SileroVADTrimmer:
    def __init__(self):
        self._model = None
        self._get_speech_timestamps = None
        self._load_error: Optional[str] = None

    def _ensure_loaded(self) -> bool:
        if self._model is not None and self._get_speech_timestamps is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
            )
            get_speech_timestamps = None
            if isinstance(utils, (list, tuple)) and len(utils) >= 1:
                get_speech_timestamps = utils[0]
            elif isinstance(utils, dict) and "get_speech_timestamps" in utils:
                get_speech_timestamps = utils["get_speech_timestamps"]
            if get_speech_timestamps is None:
                raise RuntimeError("Unexpected Silero utils format; can't find get_speech_timestamps")
            self._model = model
            self._get_speech_timestamps = get_speech_timestamps
            return True
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
            return False

    def trim(self, audio_16k: np.ndarray, sr: int, cfg: VADConfig) -> Tuple[np.ndarray, Dict]:
        if not cfg.enabled:
            return audio_16k, {"vad_applied": False}
        if sr != 16000:
            return audio_16k, {"vad_applied": False, "note": "sr_not_16k"}
        if not self._ensure_loaded():
            return audio_16k, {"vad_applied": False, "vad_error": self._load_error}

        wav = torch.from_numpy(audio_16k)
        try:
            speech_ts = self._get_speech_timestamps(
                wav,
                self._model,
                sampling_rate=sr,
                threshold=float(cfg.threshold),
                min_speech_duration_ms=int(cfg.min_speech_duration_ms),
                min_silence_duration_ms=int(cfg.min_silence_duration_ms),
                speech_pad_ms=int(cfg.speech_pad_ms),
            )
        except Exception as e:
            return audio_16k, {"vad_applied": False, "vad_error": f"{type(e).__name__}: {e}"}

        if not speech_ts:
            return np.zeros((0,), dtype=np.float32), {"vad_applied": True, "speech_segments": 0}

        start = int(speech_ts[0]["start"])
        end = int(speech_ts[-1]["end"])
        start = max(0, start)
        end = min(len(audio_16k), end)
        if end <= start:
            return np.zeros((0,), dtype=np.float32), {"vad_applied": True, "speech_segments": len(speech_ts), "note": "bad_span"}

        return audio_16k[start:end].astype(np.float32), {
            "vad_applied": True,
            "speech_segments": len(speech_ts),
            "trim_start": start,
            "trim_end": end,
        }


# ----------------------------
# Eval config
# ----------------------------

@dataclass
class EvalConfig:
    base_processor_id: str
    language: str
    task: str
    num_beams: int
    temperature: float
    max_new_tokens: int

    dynamic_audio_ctx: bool
    vad: VADConfig
    normalize_mode: str

    device: str

    batch_size: int
    auto_batch: bool
    batch_min: int
    batch_max: int
    fp16: bool
    mem_low: float
    mem_high: float
    cleanup_interval: int
    moonshine_token_limit_tps: float

    # pairing / sweep
    mix_per_target: int
    pairing_mode: str
    seed: int

    # mixing rules
    other_offset_mode: str
    other_peak_ratio: float

    # sweep grid
    conditions: List[MixCondition]

    # caching
    audio_cache_bytes: int


# ----------------------------
# JSONL IO
# ----------------------------

def load_jsonl(p: Path) -> List[dict]:
    rows: List[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

# ----------------------------
# Transcript helpers for external OTHERs
# ----------------------------

@dataclass
class TranscriptDB:
    by_path: Dict[str, str]
    by_basename: Dict[str, str]

    def text_for(self, audio_path: str) -> Optional[str]:
        k = _norm_windows_key(audio_path)
        if k in self.by_path:
            return self.by_path[k]
        b = _basename(audio_path)
        return self.by_basename.get(b)


def _row_transcript(row: dict) -> str:
    for k in ("raw_transcription", "transcript", "text"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def build_transcript_db_from_manifest(manifest_path: Path) -> TranscriptDB:
    rows = load_jsonl(manifest_path)
    by_path: Dict[str, str] = {}
    basename_candidates: Dict[str, List[str]] = {}

    for r in rows:
        ap = str(r.get("audio_path", "") or "").strip()
        if not ap:
            continue
        tx = _row_transcript(r)
        if not tx:
            continue
        nk = _norm_windows_key(ap)
        by_path[nk] = tx
        b = _basename(ap)
        basename_candidates.setdefault(b, []).append(tx)

    by_basename: Dict[str, str] = {}
    for b, txs in basename_candidates.items():
        if len(txs) == 1:
            by_basename[b] = txs[0]
        else:
            by_basename[b] = max(txs, key=len)

    return TranscriptDB(by_path=by_path, by_basename=by_basename)


def scan_audio_files(root: Path) -> List[Path]:
    exts = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus"}
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort(key=lambda p: _norm_windows_key(str(p)))
    return out


# ----------------------------
# Generate kwargs
# ----------------------------

def build_generate_kwargs(cfg: EvalConfig) -> Dict:
    return {
        "num_beams": int(cfg.num_beams),
        "temperature": float(cfg.temperature),
        "max_new_tokens": int(cfg.max_new_tokens),
    }


def _pad_stack_input_features(features_list: List[torch.Tensor]) -> torch.Tensor:
    ts = [t.squeeze(0) for t in features_list]
    max_t = max(int(t.shape[-1]) for t in ts)
    out = torch.zeros((len(ts), 80, max_t), dtype=ts[0].dtype)
    for i, t in enumerate(ts):
        out[i, :, : t.shape[-1]] = t
    return out


def _pad_stack_input_features_and_mask(
    features_list: List[torch.Tensor],
    masks_list: List[Optional[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad/stack Whisper log-mel features and matching attention_mask.

    - features_list: list of (1, 80, T)
    - masks_list:    list of (1, T) or None

    Returns:
    - feats_out: (B, 80, Tmax)
    - mask_out:  (B, Tmax) with 1 for real frames, 0 for padding
    """
    ts = [t.squeeze(0) for t in features_list]
    max_t = max(int(t.shape[-1]) for t in ts)

    feats_out = torch.zeros((len(ts), 80, max_t), dtype=ts[0].dtype)
    mask_out = torch.zeros((len(ts), max_t), dtype=torch.long)

    for i, (feat, m) in enumerate(zip(ts, masks_list)):
        T = int(feat.shape[-1])
        feats_out[i, :, :T] = feat

        if m is None:
            mask_out[i, :T] = 1
        else:
            mm = m.squeeze(0)
            if mm.numel() < T:
                # Defensive: if mask is shorter than features, treat remaining as real
                mask_out[i, : mm.numel()] = mm.to(mask_out.dtype)
                mask_out[i, mm.numel() : T] = 1
            else:
                mask_out[i, :T] = mm[:T].to(mask_out.dtype)

    return feats_out, mask_out


def _cuda_mem_ratio() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return float(used) / float(total)
    except Exception:
        return 0.0


# ----------------------------
# Pair building
# ----------------------------

def select_targets_sorted(
    rows: List[dict],
    scores: SpeakerScoreDB,
    target_take: int,
    target_percent: float,
) -> Tuple[List[dict], Dict]:
    labelled = 0
    unknown = 0
    targets: List[Tuple[float, dict]] = []

    for r in rows:
        ap = str(r.get("audio_path", "") or "").strip()
        if not ap:
            continue
        dec = scores.decision_for(ap)
        if dec is None:
            unknown += 1
            continue
        labelled += 1
        if dec != "TARGET":
            continue

        tx = _row_transcript(r)
        if not tx:
            continue

        sc = scores.score_for(ap)
        scv = float(sc) if sc is not None else float("-inf")
        targets.append((scv, r))

    targets.sort(key=lambda t: (-(t[0]), _norm_windows_key(str(t[1].get("audio_path", "")))))

    if not (0.0 <= float(target_percent) <= 100.0):
        raise ValueError("--target_percent must be 0..100")

    if float(target_percent) < 100.0:
        k = max(1, int(len(targets) * (float(target_percent) / 100.0)))
        targets = targets[:k]

    if int(target_take) > 0:
        targets = targets[: int(target_take)]

    info = {
        "rows_total": len(rows),
        "rows_labelled": labelled,
        "rows_unknown": unknown,
        "targets_selected": len(targets),
    }

    return [r for _, r in targets], info


def build_base_pairs_targets_vs_others(
    target_rows: List[dict],
    other_paths: List[Path],
    other_tx: Optional[TranscriptDB],
    mix_per_target: int,
    pairing_mode: str,
    seed: int,
    allow_missing_other_ref: bool,
) -> Tuple[List[dict], Dict]:

    if not target_rows:
        raise RuntimeError("No TARGET rows selected. Check CSV matching and transcripts in --test_manifest.")
    if not other_paths:
        raise RuntimeError("No OTHER audio files found in --others_dir.")

    usable_others: List[Path] = []
    missing_tx = 0
    for p in other_paths:
        if other_tx is None:
            usable_others.append(p)
            continue
        tx = other_tx.text_for(str(p))
        if tx and tx.strip():
            usable_others.append(p)
        else:
            missing_tx += 1

    if other_tx is not None and not allow_missing_other_ref:
        if not usable_others:
            raise RuntimeError(
                "You provided --others_manifest but none of the files in --others_dir matched a transcript.\n"
                "Fix matching (same basenames or full paths) OR set --allow_missing_other_ref=1."
            )

    others = usable_others if (other_tx is not None and not allow_missing_other_ref) else other_paths

    rng = random.Random(int(seed))

    pairs: List[dict] = []
    for i, trow in enumerate(target_rows):
        t_ap = str(trow.get("audio_path", "") or "")
        t_ref = _row_transcript(trow)
        t_key = _norm_windows_key(t_ap)

        for k in range(int(mix_per_target)):
            if pairing_mode == "round_robin":
                oi = (i * int(mix_per_target) + k) % len(others)
            elif pairing_mode == "random":
                oi = rng.randrange(0, len(others))
            else:  # hash (default)
                oi = int(_stable_u64(f"{t_key}||k{k}||seed{seed}")) % len(others)

            o_path = others[oi]
            o_ap = str(o_path)
            o_ref = ""
            if other_tx is not None:
                o_ref = (other_tx.text_for(o_ap) or "").strip()

            if not o_ref and not allow_missing_other_ref:
                found = False
                for j in range(1, min(25, len(others))):
                    oi2 = (oi + j) % len(others)
                    o_ap2 = str(others[oi2])
                    o_ref2 = (other_tx.text_for(o_ap2) or "").strip() if other_tx is not None else ""
                    if o_ref2:
                        o_ap = o_ap2
                        o_ref = o_ref2
                        found = True
                        break
                if not found:
                    continue

            if _norm_windows_key(o_ap) == t_key:
                continue

            base_key = f"{t_key}||{_norm_windows_key(o_ap)}||k{k}"

            pairs.append(
                {
                    "base_key": base_key,
                    "target_audio_path": t_ap,
                    "other_audio_path": o_ap,
                    "target_ref": t_ref,
                    "other_ref": o_ref,
                }
            )

    info = {
        "targets": len(target_rows),
        "others": len(other_paths),
        "others_missing_transcript": int(missing_tx),
        "pairs": len(pairs),
        "pairing_mode": pairing_mode,
        "mix_per_target": int(mix_per_target),
    }

    if not pairs:
        raise RuntimeError("No pairs were generated. Likely because other transcripts were missing.")

    return pairs, info


def build_target_other_base_pairs(
    rows: List[dict],
    labels: SpeakerLabelDB,
    mix_per_target: int,
    pairing_mode: str,
    seed: int,
    target_percentage: float = 100.0,
    target_max: int = 0,
) -> Tuple[List[dict], Dict]:
    labelled: List[Tuple[dict, str]] = []
    unknown = 0
    for r in rows:
        ap = str(r.get("audio_path", ""))
        d = labels.decision_for(ap)
        if d is None:
            unknown += 1
            continue
        labelled.append((r, d))

    targets = [r for r, d in labelled if d == "TARGET"]
    others = [r for r, d in labelled if d == "OTHER"]

    info = {
        "rows_total": len(rows),
        "rows_labelled": len(labelled),
        "rows_unknown": unknown,
        "targets": len(targets),
        "others": len(others),
    }

    if not others:
        raise RuntimeError("No OTHER rows found in test manifest after CSV matching.")

    # Optional: deterministic subsample of TARGET rows BEFORE pairing.
    # This gives you "X% of TARGET samples" (and then each kept TARGET will still be paired mix_per_target times).
    targets_before = len(targets)
    if float(target_percentage) < 100.0:
        targets = _deterministic_subsample(
            targets,
            key_fn=lambda r: _norm_windows_key(str(r.get("audio_path", ""))),
            percentage=float(target_percentage),
            seed=int(seed),
        )
    if int(target_max) > 0:
        targets = _deterministic_cap(
            targets,
            key_fn=lambda r: _norm_windows_key(str(r.get("audio_path", ""))),
            max_n=int(target_max),
            seed=int(seed),
        )
    info.update(
        {
            "target_percentage": float(target_percentage),
            "target_max": int(target_max),
            "targets_before_subsample": int(targets_before),
            "targets_after_subsample": int(len(targets)),
        }
    )

    if not targets:
        raise RuntimeError(
            "No TARGET rows left after --target_percentage/--target_max subsampling. Increase percentage/max." 
        )

    rng = random.Random(int(seed))

    other_indices = list(range(len(others)))
    rng.shuffle(other_indices)
    rr_ptr = 0

    pairs: List[dict] = []
    for trow in targets:
        t_ap = str(trow.get("audio_path", ""))
        t_ref = (trow.get("raw_transcription") or "").strip()

        for k in range(int(mix_per_target)):
            if pairing_mode == "round_robin":
                oi = other_indices[rr_ptr % len(other_indices)]
                rr_ptr += 1
            elif pairing_mode == "hash":
                oi = int(_stable_u64(f"{_norm_windows_key(t_ap)}||k{k}||seed{seed}")) % len(others)
            else:
                oi = rng.randrange(0, len(others))

            orow = others[oi]
            o_ap = str(orow.get("audio_path", ""))
            o_ref = (orow.get("raw_transcription") or "").strip()

            if _norm_windows_key(o_ap) == _norm_windows_key(t_ap):
                oi2 = rng.randrange(0, len(others))
                orow = others[oi2]
                o_ap = str(orow.get("audio_path", ""))
                o_ref = (orow.get("raw_transcription") or "").strip()

            base_key = f"{_norm_windows_key(t_ap)}||{_norm_windows_key(o_ap)}||k{k}"

            pairs.append(
                {
                    "base_key": base_key,
                    "target_audio_path": t_ap,
                    "other_audio_path": o_ap,
                    "target_ref": t_ref,
                    "other_ref": o_ref,
                }
            )

    return pairs, info


def expand_pairs_with_conditions(base_pairs: List[dict], conditions: List[MixCondition]) -> List[dict]:
    out: List[dict] = []
    for bp in base_pairs:
        for cond in conditions:
            cond_id = cond.cond_id()
            mix_key = f"{bp['base_key']}||{cond_id}"
            out.append(
                {
                    "mix_key": mix_key,
                    "base_key": bp["base_key"],
                    "cond_id": cond_id,
                    "snr_db": float(cond.snr_db),
                    "overlap": cond.overlap,
                    "target_audio_path": bp["target_audio_path"],
                    "other_audio_path": bp["other_audio_path"],
                    "target_ref": bp["target_ref"],
                    "other_ref": bp["other_ref"],
                }
            )
    return out


# ----------------------------
# Resume IO
# ----------------------------

RUN_ARGS_VERSION = 1
RUN_ARGS_IGNORE_KEYS = {"resume", "force_resume", "recalc_metrics", "import_per_sample_json", "import_force"}
RUN_ARGS_IMPORT_IGNORE_KEYS = RUN_ARGS_IGNORE_KEYS | {
    "out_json",
    "checkpoint_dir",
    "pairs_manifest",
    "save_every",
    "device",
    "batch_size",
    "auto_batch",
    "batch_min",
    "batch_max",
    "mem_low",
    "mem_high",
    "cleanup_interval",
    "audio_cache_gb",
    "ffmpeg_path",
    "compare_openai_tiny",
    "groq_verify",
    "normalize",
}


def _json_friendly(obj):
    if isinstance(obj, (Path, PureWindowsPath)):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_friendly(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_friendly(v) for v in obj]
    if isinstance(obj, set):
        return [_json_friendly(v) for v in sorted(obj)]
    try:
        # numpy scalars
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    return obj


def _build_run_args(args: argparse.Namespace, out_json: Path) -> dict:
    args_dict = _json_friendly(vars(args))
    args_dict["out_json"] = str(out_json)
    return {
        "version": RUN_ARGS_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": [str(a) for a in sys.argv],
        "python": sys.executable,
        "cwd": os.getcwd(),
        "args": args_dict,
    }


def _normalize_run_args_for_compare(run_args: Optional[dict], ignore_keys: Optional[set] = None) -> Dict[str, object]:
    if not run_args or not isinstance(run_args, dict):
        return {}
    if ignore_keys is None:
        ignore_keys = RUN_ARGS_IGNORE_KEYS
    args_dict = run_args.get("args")
    if not isinstance(args_dict, dict):
        return {}
    normed = {}
    for k, v in args_dict.items():
        if k in ignore_keys:
            continue
        normed[k] = v
    return normed


def _diff_run_args(prev: Optional[dict], current: Optional[dict], ignore_keys: Optional[set] = None) -> Dict[str, dict]:
    prev_norm = _normalize_run_args_for_compare(prev, ignore_keys=ignore_keys)
    curr_norm = _normalize_run_args_for_compare(current, ignore_keys=ignore_keys)
    if not prev_norm or not curr_norm:
        return {}
    diffs: Dict[str, dict] = {}
    for k in sorted(set(prev_norm.keys()) | set(curr_norm.keys())):
        if prev_norm.get(k) != curr_norm.get(k):
            diffs[k] = {"previous": prev_norm.get(k), "current": curr_norm.get(k)}
    return diffs


def warn_if_run_args_changed(prev: Optional[dict], current: Optional[dict]) -> None:
    diffs = _diff_run_args(prev, current)
    if not diffs:
        return
    print("⚠ Detected argument changes since last run:")
    shown = 0
    for k, v in diffs.items():
        print(f"  - {k}: {v.get('previous')} -> {v.get('current')}")
        shown += 1
        if shown >= 20:
            break
    if len(diffs) > shown:
        print(f"  ... {len(diffs) - shown} more changes")


def _print_run_args_diff_summary(diffs: Dict[str, dict], header: str) -> None:
    if not diffs:
        return
    print(header)
    shown = 0
    for k, v in diffs.items():
        print(f"  - {k}: {v.get('previous')} -> {v.get('current')}")
        shown += 1
        if shown >= 20:
            break
    if len(diffs) > shown:
        print(f"  ... {len(diffs) - shown} more changes")


def _prompt_yes_no(question: str, default: bool = False) -> bool:
    prompt = " [y/N] " if not default else " [Y/n] "
    if not sys.stdin or not sys.stdin.isatty():
        print(f"{question}{prompt} (non-interactive, default={'yes' if default else 'no'})")
        return default
    while True:
        resp = input(f"{question}{prompt}").strip().lower()
        if not resp:
            return default
        if resp in {"y", "yes"}:
            return True
        if resp in {"n", "no"}:
            return False
        print("Please answer yes or no.")


def _extract_run_args_from_meta(meta: Optional[dict]) -> Optional[dict]:
    if not meta or not isinstance(meta, dict):
        return None
    if "run_args" in meta and isinstance(meta.get("run_args"), dict):
        return meta.get("run_args")
    return meta


def _select_previous_run_args(results: dict, per_sample_meta: Optional[dict]) -> Optional[dict]:
    if isinstance(results, dict):
        history = results.get("run_history")
        if isinstance(history, list) and history:
            last = history[-1]
            if isinstance(last, dict):
                return last
        if isinstance(results.get("run_args"), dict):
            return results.get("run_args")
    return _extract_run_args_from_meta(per_sample_meta)


def _update_run_history(results: dict, current_run_args: dict) -> None:
    if not isinstance(results, dict) or not isinstance(current_run_args, dict):
        return
    history = results.get("run_history")
    if not isinstance(history, list):
        history = []
        if isinstance(results.get("run_args"), dict):
            history.append(results.get("run_args"))
    if not history or history[-1] != current_run_args:
        history.append(current_run_args)
    results["run_history"] = history
    results["run_args"] = current_run_args


def _pack_per_sample_payload(all_predictions: dict, run_args: Optional[dict]) -> List[dict]:
    payload = list(all_predictions.values())
    if not run_args:
        return payload
    meta = {"__meta__": {"run_args": run_args}}
    return [meta] + payload


def load_per_sample_predictions(per_sample_json: Path) -> Tuple[dict, Optional[dict]]:
    all_predictions: Dict[str, dict] = {}
    per_sample_meta: Optional[dict] = None
    if not per_sample_json.exists():
        return all_predictions, None

    try:
        with per_sample_json.open("r", encoding="utf-8") as f:
            per_sample_data = json.load(f)

        if isinstance(per_sample_data, dict):
            per_sample_meta = per_sample_data.get("run_args") or per_sample_data.get("__meta__")
            per_sample_data = (
                per_sample_data.get("items")
                or per_sample_data.get("samples")
                or []
            )
        if isinstance(per_sample_data, list):
            for item in per_sample_data:
                if isinstance(item, dict):
                    if "__meta__" in item and isinstance(item.get("__meta__"), dict):
                        per_sample_meta = item.get("__meta__")
                        if isinstance(per_sample_meta.get("run_args"), dict):
                            per_sample_meta = per_sample_meta.get("run_args")
                        continue
                    if "run_args" in item and "mix_key" not in item:
                        if isinstance(item.get("run_args"), dict):
                            per_sample_meta = item.get("run_args")
                        continue
                    key = item.get("mix_key") or item.get("key")
                    if key:
                        all_predictions[key] = item
    except Exception as e:
        print(f"⚠ Could not load per-sample predictions: {e}")
        return {}, None

    return all_predictions, per_sample_meta


def merge_imported_predictions(
    dest: Dict[str, dict],
    src: Dict[str, dict],
    *,
    active_keys: Optional[set] = None,
    allow_models: Optional[set] = None,
) -> Tuple[int, int]:
    merged_pairs = 0
    merged_preds = 0
    for key, item in src.items():
        if active_keys is not None and key not in active_keys:
            continue
        if not isinstance(item, dict):
            continue
        preds = item.get("predictions", {})
        if not isinstance(preds, dict):
            continue
        if allow_models is not None:
            preds = {k: v for k, v in preds.items() if k in allow_models}
        if not preds:
            continue
        if key not in dest:
            new_item = dict(item)
            new_item["predictions"] = dict(preds)
            dest[key] = new_item
            merged_pairs += 1
            merged_preds += len(preds)
            continue
        dest_preds = dest[key].get("predictions")
        if not isinstance(dest_preds, dict):
            dest_preds = {}
            dest[key]["predictions"] = dest_preds
        added = 0
        for mk, mv in preds.items():
            if mk not in dest_preds:
                dest_preds[mk] = mv
                added += 1
        if added:
            merged_pairs += 1
            merged_preds += added
    return merged_pairs, merged_preds


def load_existing_results(out_json: Path) -> Tuple[dict, dict, Optional[dict]]:
    if not out_json.exists():
        return {}, {}, None

    try:
        with out_json.open("r", encoding="utf-8") as f:
            results = json.load(f)

        per_sample_json = out_json.parent / "evaluation_per_sample_predictions_targetmix_sweep.json"
        all_predictions, per_sample_meta = load_per_sample_predictions(per_sample_json)

        print(f"✓ Loaded existing results from: {out_json}")
        print(f"✓ Found {len(results.get('models', []))} already evaluated models")
        return results, all_predictions, per_sample_meta
    except Exception as e:
        print(f"⚠ Could not load existing results: {e}")
        return {}, {}, None


def save_incremental_results(results: dict, all_predictions: dict, out_json: Path, run_args: Optional[dict] = None) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if run_args:
        _update_run_history(results, run_args)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    per_sample_json = out_json.parent / "evaluation_per_sample_predictions_targetmix_sweep.json"
    with per_sample_json.open("w", encoding="utf-8") as f:
        json.dump(_pack_per_sample_payload(all_predictions, run_args), f, indent=2, ensure_ascii=False)

    print(f"✓ Incremental results saved to: {out_json}")
    print(f"✓ Per-sample predictions saved to: {per_sample_json}")


# ----------------------------
# Metrics
# ----------------------------

def _wer_by_duration(items: List[dict], key_name: str) -> Dict[str, Optional[float]]:
    buckets: Dict[str, List[float]] = {"0-1s": [], "1-2s": [], "2-5s": [], "5-10s": [], "10-30s": []}
    for row in items:
        d = float(row.get("duration_sec_eval", row.get("duration_sec_target", 0.0)) or 0.0)
        w = row.get(key_name, None)
        if w is None:
            continue
        w = float(w)
        if d < 1:
            buckets["0-1s"].append(w)
        elif d < 2:
            buckets["1-2s"].append(w)
        elif d < 5:
            buckets["2-5s"].append(w)
        elif d < 10:
            buckets["5-10s"].append(w)
        else:
            buckets["10-30s"].append(w)
    return {k: (float(np.mean(v)) if v else None) for k, v in buckets.items()}


def compute_metrics_from_items(items: List[dict], normalize_mode: str) -> Dict:
    if not items:
        return {
            "samples": 0,
            "wer_micro_target": None,
            "wer_micro_other": None,
            "cer_micro_target": None,
            "cer_micro_other": None,
            "cer_macro_target": None,
            "cer_macro_other": None,
            "wer_macro_target": None,
            "wer_macro_other": None,
            "win_rate_target_closer": None,
            "avg_margin_other_minus_target": None,
            "avg_margin_cer_other_minus_target": None,
            "likely_hit_max_token_cap_rate": None,
            "wer_by_duration_target": {"0-1s": None, "1-2s": None, "2-5s": None, "5-10s": None, "10-30s": None},
            "wer_by_duration_other": {"0-1s": None, "1-2s": None, "2-5s": None, "5-10s": None, "10-30s": None},
        }

    preds_raw = [str(x.get("pred", "") or "") for x in items]
    trefs_raw = [str(x.get("target_ref", "") or "") for x in items]
    orefs_raw = [str(x.get("other_ref", "") or "") for x in items]

    if normalize_mode in {"whisper_basic", "basic"}:
        preds = [_basic_whisperish_normalize(p) for p in preds_raw]
        trefs = [_basic_whisperish_normalize(r) for r in trefs_raw]
        orefs = [_basic_whisperish_normalize(r) for r in orefs_raw]
    else:
        preds, trefs, orefs = preds_raw, trefs_raw, orefs_raw

    # micro WER/CER
    wer_micro_t = float(jiwer.wer(trefs, preds))
    cer_micro_t = float(jiwer.cer(trefs, preds))
    wer_micro_o = float(jiwer.wer(orefs, preds))
    cer_micro_o = float(jiwer.cer(orefs, preds))

    # per-utt WER (macro)
    wer_utt_t = [float(x.get("wer_target")) for x in items if x.get("wer_target") is not None]
    wer_utt_o = [float(x.get("wer_other")) for x in items if x.get("wer_other") is not None]
    wer_macro_t = float(np.mean(wer_utt_t)) if wer_utt_t else None
    wer_macro_o = float(np.mean(wer_utt_o)) if wer_utt_o else None

    # per-utt CER (macro) - only available if per-item CER was computed
    cer_utt_t = [float(x.get("cer_target")) for x in items if x.get("cer_target") is not None]
    cer_utt_o = [float(x.get("cer_other")) for x in items if x.get("cer_other") is not None]
    cer_macro_t = float(np.mean(cer_utt_t)) if cer_utt_t else None
    cer_macro_o = float(np.mean(cer_utt_o)) if cer_utt_o else None

    wins = sum(1 for x in items if bool(x.get("win_target_closer")))
    win_rate = float(wins) / float(len(items))

    avg_margin = float(np.mean(np.array(wer_utt_o) - np.array(wer_utt_t))) if (wer_utt_t and wer_utt_o) else None
    avg_margin_cer = float(np.mean(np.array(cer_utt_o) - np.array(cer_utt_t))) if (cer_utt_t and cer_utt_o) else None

    # sanity: "not stopping" / repetition detector
    cap_flags = [x.get("likely_hit_max_token_cap") for x in items]
    cap_flags = [bool(v) for v in cap_flags if v is not None]
    cap_rate = float(np.mean(cap_flags)) if cap_flags else None

    return {
        "samples": len(items),
        "wer_micro_target": wer_micro_t,
        "cer_micro_target": cer_micro_t,
        "wer_macro_target": wer_macro_t,
        "cer_macro_target": cer_macro_t,
        "wer_micro_other": wer_micro_o,
        "cer_micro_other": cer_micro_o,
        "wer_macro_other": wer_macro_o,
        "cer_macro_other": cer_macro_o,
        "win_rate_target_closer": win_rate,
        "avg_margin_other_minus_target": avg_margin,
        "avg_margin_cer_other_minus_target": avg_margin_cer,
        "likely_hit_max_token_cap_rate": cap_rate,
        "wer_by_duration_target": _wer_by_duration(items, "wer_target"),
        "wer_by_duration_other": _wer_by_duration(items, "wer_other"),
    }


def group_items_by_condition(items: List[dict]) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for it in items:
        cid = str(it.get("cond_id", "unknown"))
        out.setdefault(cid, []).append(it)
    return out


# ----------------------------
# Eval core
# ----------------------------

def eval_one_model(
    model_id_or_path: str,
    pair_rows: List[dict],
    cfg: EvalConfig,
    vad_trimmer: Optional[SileroVADTrimmer],
    all_predictions: dict,
    out_json: Path,
    *,
    lora_merge: bool,
    lora_base_model: Optional[str],
    results: Optional[dict] = None,
    run_args: Optional[dict] = None,
    save_every: int = 0,
) -> Tuple[Dict, Dict[str, Dict]]:

    processor_id = _resolve_processor_id_for_model(model_id_or_path, cfg.base_processor_id)
    processor = AutoProcessor.from_pretrained(processor_id)

    model = _load_model_for_eval(
        model_id_or_path,
        lora_merge=bool(lora_merge),
        lora_base_model=lora_base_model,
    )
    model.to(cfg.device)
    model.eval()

    model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()
    is_whisper = _is_whisper_family(model_type, model_id_or_path)
    is_moonshine = _is_moonshine_family(model_type, model_id_or_path)

    if is_whisper:
        try:
            fids = processor.get_decoder_prompt_ids(language=cfg.language, task=cfg.task)
            if hasattr(model, "generation_config") and hasattr(model.generation_config, "forced_decoder_ids"):
                model.generation_config.forced_decoder_ids = fids
        except Exception:
            pass

    gen_kwargs_base = build_generate_kwargs(cfg)

    # Used for the "hit token cap" sanity metric.
    eos_id = None
    try:
        eos_id = getattr(model.generation_config, "eos_token_id", None)
    except Exception:
        eos_id = None
    if eos_id is None:
        try:
            eos_id = getattr(model.config, "eos_token_id", None)
        except Exception:
            eos_id = None
    if eos_id is None:
        try:
            eos_id = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
        except Exception:
            eos_id = None
    if eos_id is None:
        eos_id = 50256  # Whisper

    per_item: List[dict] = []
    skipped: List[dict] = []

    if cfg.vad.enabled and vad_trimmer is None:
        vad_trimmer = SileroVADTrimmer()

    model_name = _model_pred_key(model_id_or_path, bool(lora_merge))

    save_every_n = int(save_every) if save_every else 0
    eval_count = 0

    def _maybe_autosave() -> None:
        if save_every_n <= 0 or results is None:
            return
        if eval_count <= 0 or (eval_count % save_every_n) != 0:
            return
        save_incremental_results(results, all_predictions, out_json, run_args=run_args)
        print(f"✓ Auto-saved after {eval_count} evaluated samples for {model_name}")

    audio_cache = AudioCacheLRU(cfg.audio_cache_bytes)

    def get_audio_cached(p: Path) -> Tuple[np.ndarray, int]:
        c = audio_cache.get(str(p))
        if c is not None:
            return c
        a, sr = load_audio_mono_16k(p)
        audio_cache.put(str(p), a, sr)
        return a, sr

    with torch.inference_mode():
        use_cuda = bool(str(cfg.device).startswith("cuda") and torch.cuda.is_available())
        cur_bs = int(max(cfg.batch_min, min(cfg.batch_max, cfg.batch_size)))
        oom_cooldown = 0

        if use_cuda and cfg.fp16:
            try:
                model.half()
            except Exception:
                pass

        batch_buf: List[dict] = []
        batch_count = 0

        def _prepare_model_inputs(audios: List[np.ndarray]) -> Dict[str, torch.Tensor]:
            proc_kwargs = {
                "sampling_rate": 16000,
                "return_tensors": "pt",
            }
            # Some Whisper checkpoints require fixed 3000-frame mel inputs.
            if is_whisper:
                proc_kwargs["padding"] = "max_length"
                proc_kwargs["max_length"] = 3000
                proc_kwargs["truncation"] = True
            else:
                proc_kwargs["padding"] = True

            try:
                inputs = processor(
                    audios,
                    return_attention_mask=True,
                    **proc_kwargs,
                )
            except TypeError:
                inputs = processor(
                    audios,
                    **proc_kwargs,
                )

            out_inputs: Dict[str, torch.Tensor] = {}
            for k, v in inputs.items():
                if not torch.is_tensor(v):
                    continue
                t = v
                if use_cuda and cfg.fp16 and t.is_floating_point():
                    t = t.half()
                out_inputs[k] = t.to(cfg.device, non_blocking=True)

            # Final guard for legacy Whisper variants that still return non-3000 features.
            if is_whisper and "input_features" in out_inputs:
                feats = out_inputs["input_features"]
                if feats.ndim == 3:
                    cur = int(feats.shape[-1])
                    if cur < 3000:
                        pad = torch.zeros(
                            (feats.shape[0], feats.shape[1], 3000 - cur),
                            dtype=feats.dtype,
                            device=feats.device,
                        )
                        out_inputs["input_features"] = torch.cat([feats, pad], dim=-1)
                    elif cur > 3000:
                        out_inputs["input_features"] = feats[..., :3000]
            return out_inputs

        def _moonshine_max_length(inputs: Dict[str, torch.Tensor]) -> Optional[int]:
            if not is_moonshine:
                return None
            attn = inputs.get("attention_mask")
            if attn is None:
                return None
            try:
                seq_len = attn.sum(dim=-1).to(torch.float32).max().item()
                max_len = int(math.ceil(float(seq_len) * (float(cfg.moonshine_token_limit_tps) / 16000.0)))
                return max(8, min(512, max_len))
            except Exception:
                return None

        def _run_chunk(chunk: List[dict]) -> None:
            nonlocal cur_bs, oom_cooldown, batch_count, eval_count

            audios = [np.asarray(c["audio_eval"], dtype=np.float32) for c in chunk]
            model_inputs = _prepare_model_inputs(audios)

            gen_kwargs = dict(gen_kwargs_base)
            moonshine_max_len = _moonshine_max_length(model_inputs)
            if moonshine_max_len is not None:
                gen_kwargs.pop("max_new_tokens", None)
                gen_kwargs["max_length"] = int(moonshine_max_len)

            generated_ids = model.generate(**model_inputs, **gen_kwargs)
            generated_ids_cpu = generated_ids.detach().cpu()
            texts = processor.batch_decode(generated_ids_cpu, skip_special_tokens=True)

            for c, text_out, seq in zip(chunk, texts, generated_ids_cpu):
                pred = (text_out or "").strip()
                tref = (c.get("target_ref") or "").strip()
                oref = (c.get("other_ref") or "").strip()

                ended_by_eos = bool((seq == int(eos_id)).any().item()) if eos_id is not None else True
                likely_hit_cap = (not ended_by_eos)
                pred_token_len = int(seq.numel())

                if cfg.normalize_mode in {"whisper_basic", "basic"}:
                    pred_n = _basic_whisperish_normalize(pred)
                    tref_n = _basic_whisperish_normalize(tref)
                    oref_n = _basic_whisperish_normalize(oref)
                else:
                    pred_n, tref_n, oref_n = pred, tref, oref

                wt = float(jiwer.wer(tref_n, pred_n))
                ct = float(jiwer.cer(tref_n, pred_n))
                wo = float(jiwer.wer(oref_n, pred_n))
                co = float(jiwer.cer(oref_n, pred_n))

                rec = {
                    "mix_key": c["mix_key"],
                    "cond_id": c.get("cond_id", ""),
                    "snr_db": float(c.get("snr_db")),
                    "overlap": c.get("overlap", None),
                    "target_audio_path": c["target_audio_path"],
                    "other_audio_path": c["other_audio_path"],
                    "duration_sec_target": c["duration_sec_target"],
                    "duration_sec_other": c["duration_sec_other"],
                    "duration_sec_eval": c["duration_sec_eval"],
                    "target_ref": tref,
                    "other_ref": oref,
                    "pred": pred,
                    "wer_target": wt,
                    "cer_target": ct,
                    "wer_other": wo,
                    "cer_other": co,
                    "win_target_closer": bool(wt < wo),
                    "ended_by_eos": ended_by_eos,
                    "likely_hit_max_token_cap": likely_hit_cap,
                    "pred_token_len": pred_token_len,
                    "vad": c.get("vad", {"vad_applied": False}),
                    "mix": c.get("mix", {}),
                }
                per_item.append(rec)

                all_predictions.setdefault(
                    c["mix_key"],
                    {
                        "mix_key": c["mix_key"],
                        "cond_id": c.get("cond_id", ""),
                        "snr_db": float(c.get("snr_db")),
                        "overlap": c.get("overlap", None),
                        "target_audio_path": c["target_audio_path"],
                        "other_audio_path": c["other_audio_path"],
                        "target_reference": tref,
                        "other_reference": oref,
                        "predictions": {},
                    },
                )
                all_predictions[c["mix_key"]]["predictions"][model_name] = {
                    "pred": pred,
                    "duration_sec_eval": c.get("duration_sec_eval"),
                    "wer_target": wt,
                    "wer_other": wo,
                    "cer_target": ct,
                    "cer_other": co,
                    "win_target_closer": bool(wt < wo),
                    "ended_by_eos": ended_by_eos,
                    "likely_hit_max_token_cap": likely_hit_cap,
                    "pred_token_len": pred_token_len,
                    "vad": c.get("vad", {"vad_applied": False}),
                    "mix": c.get("mix", {}),
                }
                eval_count += 1
                _maybe_autosave()

            if cfg.auto_batch and use_cuda:
                ratio = _cuda_mem_ratio()
                if oom_cooldown > 0:
                    oom_cooldown -= 1
                else:
                    if ratio < cfg.mem_low and cur_bs < cfg.batch_max:
                        cur_bs = min(cfg.batch_max, max(cur_bs + 1, int(cur_bs * 1.25)))
                    elif ratio > cfg.mem_high and cur_bs > cfg.batch_min:
                        cur_bs = max(cfg.batch_min, int(cur_bs * 0.8))

            batch_count += 1
            if batch_count % max(1, int(cfg.cleanup_interval)) == 0:
                if use_cuda:
                    torch.cuda.empty_cache()
                gc.collect()

        pbar = tqdm(pair_rows, desc=f"eval {Path(model_id_or_path).name}")
        for item in pbar:
            key = item["mix_key"]
            cid = item.get("cond_id", "")

            # Resume: skip already computed
            if key in all_predictions and model_name in all_predictions[key].get("predictions", {}):
                continue

            t_ap = Path(item["target_audio_path"])
            o_ap = Path(item["other_audio_path"])

            if not t_ap.exists():
                skipped.append({"mix_key": key, "reason": "missing_target_file", "target_audio_path": str(t_ap)})
                continue
            if not o_ap.exists():
                skipped.append({"mix_key": key, "reason": "missing_other_file", "other_audio_path": str(o_ap)})
                continue

            try:
                t_audio, sr_t = get_audio_cached(t_ap)
                o_audio, sr_o = get_audio_cached(o_ap)
                if sr_t != 16000 or sr_o != 16000:
                    skipped.append({"mix_key": key, "reason": "sr_not_16k", "target_sr": sr_t, "other_sr": sr_o})
                    continue

                dur_t = seconds_from_audio(t_audio, sr_t)
                dur_o = seconds_from_audio(o_audio, sr_o)

                # Deterministic per-pair RNG (important for resume)
                rng_local = np.random.default_rng(_stable_uint64_from_str(key, cfg.seed))

                mixed, mix_meta = mix_target_with_other(
                    target=t_audio,
                    other=o_audio,
                    snr_db_target_over_other=float(item["snr_db"]),
                    other_offset_mode=cfg.other_offset_mode,
                    rng=rng_local,
                    other_peak_ratio=cfg.other_peak_ratio,
                    overlap_ratio=item.get("overlap", None),
                )

                # VAD trim on the mixture
                vad_info: Dict = {"vad_applied": False}
                audio_eval = mixed
                if cfg.vad.enabled and vad_trimmer is not None:
                    audio_vad, vad_info = vad_trimmer.trim(mixed, 16000, cfg.vad)
                    if len(audio_vad) == 0:
                        if cfg.vad.policy == "skip":
                            skipped.append({"mix_key": key, "reason": "vad_no_speech", "cond_id": cid})
                            continue
                        if cfg.vad.policy == "empty":
                            # No inference; force empty prediction
                            pred = ""
                            tref = (item.get("target_ref") or "").strip()
                            oref = (item.get("other_ref") or "").strip()

                            pred_n = _basic_whisperish_normalize(pred) if cfg.normalize_mode in {"whisper_basic", "basic"} else pred
                            tref_n = _basic_whisperish_normalize(tref) if cfg.normalize_mode in {"whisper_basic", "basic"} else tref
                            oref_n = _basic_whisperish_normalize(oref) if cfg.normalize_mode in {"whisper_basic", "basic"} else oref

                            wt = float(jiwer.wer(tref_n, pred_n))
                            ct = float(jiwer.cer(tref_n, pred_n))
                            wo = float(jiwer.wer(oref_n, pred_n))
                            co = float(jiwer.cer(oref_n, pred_n))

                            rec = {
                                "mix_key": key,
                                "cond_id": cid,
                                "snr_db": float(item["snr_db"]),
                                "overlap": item.get("overlap", None),
                                "target_audio_path": str(t_ap),
                                "other_audio_path": str(o_ap),
                                "duration_sec_target": dur_t,
                                "duration_sec_other": dur_o,
                                "duration_sec_eval": 0.0,
                                "target_ref": tref,
                                "other_ref": oref,
                                "pred": pred,
                                "wer_target": wt,
                                "cer_target": ct,
                                "wer_other": wo,
                                "cer_other": co,
                                "win_target_closer": bool(wt < wo),
                                "vad": vad_info,
                                "mix": mix_meta,
                            }
                            per_item.append(rec)

                            all_predictions.setdefault(
                                key,
                                {
                                    "mix_key": key,
                                    "cond_id": cid,
                                    "snr_db": float(item["snr_db"]),
                                    "overlap": item.get("overlap", None),
                                    "target_audio_path": str(t_ap),
                                    "other_audio_path": str(o_ap),
                                    "target_reference": tref,
                                    "other_reference": oref,
                                    "predictions": {},
                                },
                            )
                            all_predictions[key]["predictions"][model_name] = {
                                "pred": pred,
                                "duration_sec_eval": 0.0,
                                "wer_target": wt,
                                "wer_other": wo,
                                "cer_target": ct,
                                "cer_other": co,
                                "win_target_closer": bool(wt < wo),
                                "vad": vad_info,
                                "mix": mix_meta,
                            }
                            eval_count += 1
                            _maybe_autosave()
                            continue

                        audio_eval = mixed
                    else:
                        audio_eval = audio_vad

                dur_eval = seconds_from_audio(audio_eval, 16000)

                batch_buf.append(
                    {
                        "mix_key": key,
                        "cond_id": cid,
                        "snr_db": float(item["snr_db"]),
                        "overlap": item.get("overlap", None),
                        "target_audio_path": str(t_ap),
                        "other_audio_path": str(o_ap),
                        "duration_sec_target": dur_t,
                        "duration_sec_other": dur_o,
                        "duration_sec_eval": dur_eval,
                        "target_ref": (item.get("target_ref") or "").strip(),
                        "other_ref": (item.get("other_ref") or "").strip(),
                        "vad": vad_info,
                        "mix": mix_meta,
                        "audio_eval": np.asarray(audio_eval, dtype=np.float32),
                    }
                )

            except Exception as e:
                skipped.append({"mix_key": key, "reason": f"exception: {type(e).__name__}: {e}", "cond_id": cid})
                continue

            if len(batch_buf) < cur_bs:
                continue

            pending = batch_buf
            batch_buf = []

            while pending:
                chunk = pending[:cur_bs]
                try:
                    _run_chunk(chunk)
                    pending = pending[cur_bs:]
                except torch.cuda.OutOfMemoryError:
                    if not use_cuda:
                        raise
                    if cur_bs > cfg.batch_min:
                        cur_bs = max(cfg.batch_min, int(cur_bs * 0.5))
                    oom_cooldown = 5
                    if use_cuda:
                        torch.cuda.empty_cache()
                    gc.collect()
                    continue
                except Exception as e:
                    for c in chunk:
                        skipped.append(
                            {
                                "mix_key": c.get("mix_key"),
                                "reason": f"inference_exception: {type(e).__name__}: {e}",
                                "cond_id": c.get("cond_id", ""),
                            }
                        )
                    pending = pending[cur_bs:]

        # flush remaining
        if batch_buf:
            pending = batch_buf
            while pending:
                chunk = pending[:cur_bs]
                try:
                    _run_chunk(chunk)
                    pending = pending[cur_bs:]
                except torch.cuda.OutOfMemoryError:
                    if not use_cuda:
                        raise
                    if cur_bs > cfg.batch_min:
                        cur_bs = max(cfg.batch_min, int(cur_bs * 0.5))
                        continue
                    for c in chunk:
                        skipped.append(
                            {
                                "mix_key": c.get("mix_key"),
                                "reason": "cuda_oom_at_min_batch",
                                "cond_id": c.get("cond_id", ""),
                            }
                        )
                    pending = pending[cur_bs:]
                except Exception as e:
                    for c in chunk:
                        skipped.append(
                            {
                                "mix_key": c.get("mix_key"),
                                "reason": f"inference_exception: {type(e).__name__}: {e}",
                                "cond_id": c.get("cond_id", ""),
                            }
                        )
                    pending = pending[cur_bs:]

    # Overall metrics and per-condition metrics (from saved predictions, supports resume)
    active_keys = {pr["mix_key"] for pr in pair_rows}
    overall, by_cond = recompute_metrics_from_saved_predictions(
        all_predictions,
        model_name,
        cfg.normalize_mode,
        active_keys=active_keys,
    )

    if int(overall.get("samples") or 0) == 0 and len(pair_rows) > 0:
        first_reason = skipped[0].get("reason") if skipped else "no_predictions_saved"
        raise RuntimeError(
            f"No predictions were saved for model '{model_id_or_path}'. "
            f"Skipped={len(skipped)} first_reason={first_reason}"
        )

    try:
        overall["model_type"] = model_type
        overall["processor_id"] = processor_id
        overall["model_num_params"] = int(model.num_parameters())
    except Exception:
        pass

    overall["skipped"] = len(skipped)
    return overall, by_cond


# ----------------------------
# Force-resume metric recompute
# ----------------------------

def recompute_metrics_from_saved_predictions(
    all_predictions: dict,
    model_name: str,
    normalize_mode: str,
    active_keys: Optional[set] = None,
) -> Tuple[Dict, Dict[str, Dict]]:
    items: List[dict] = []

    for key, blob in all_predictions.items():
        if active_keys is not None and key not in active_keys:
            continue
        preds = blob.get("predictions", {})
        if model_name not in preds:
            continue

        pred = (preds[model_name].get("pred", "") or "").strip()
        tref = (blob.get("target_reference", "") or "").strip()
        oref = (blob.get("other_reference", "") or "").strip()

        if normalize_mode in {"whisper_basic", "basic"}:
            pred_n = _basic_whisperish_normalize(pred)
            tref_n = _basic_whisperish_normalize(tref)
            oref_n = _basic_whisperish_normalize(oref)
        else:
            pred_n, tref_n, oref_n = pred, tref, oref

        wt = float(jiwer.wer(tref_n, pred_n))
        ct = float(jiwer.cer(tref_n, pred_n))
        wo = float(jiwer.wer(oref_n, pred_n))
        co = float(jiwer.cer(oref_n, pred_n))

        # Store per-sample metrics back into all_predictions (helps later analysis)
        try:
            preds[model_name]["wer_target"] = wt
            preds[model_name]["wer_other"] = wo
            preds[model_name]["cer_target"] = ct
            preds[model_name]["cer_other"] = co
            preds[model_name]["win_target_closer"] = bool(wt < wo)
        except Exception:
            pass

        items.append(
            {
                "mix_key": key,
                "cond_id": blob.get("cond_id", ""),
                "snr_db": blob.get("snr_db", None),
                "overlap": blob.get("overlap", None),
                "target_audio_path": blob.get("target_audio_path"),
                "other_audio_path": blob.get("other_audio_path"),
                "duration_sec_eval": preds[model_name].get("duration_sec_eval", None),
                "target_ref": tref,
                "other_ref": oref,
                "pred": pred,
                "wer_target": wt,
                "cer_target": ct,
                "wer_other": wo,
                "cer_other": co,
                "win_target_closer": bool(wt < wo),
            }
        )

    overall = compute_metrics_from_items(items, normalize_mode)
    by_cond_items = group_items_by_condition(items)
    by_cond = {cid: compute_metrics_from_items(v, normalize_mode) for cid, v in by_cond_items.items()}
    overall["skipped"] = 0
    return overall, by_cond


# ----------------------------
# Utilities
# ----------------------------

def parse_float_list(s: Optional[str]) -> Optional[List[float]]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    parts = re.split(r"[\s,;]+", s)
    vals: List[float] = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))
    return vals if vals else None


def parse_overlap_list(s: Optional[str]) -> Optional[List[float]]:
    vals = parse_float_list(s)
    if vals is None:
        return None
    out: List[float] = []
    for v in vals:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"overlap values must be within [0,1], got: {v}")
        out.append(float(v))
    return out


def parse_model_list_csv(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    text = str(s).strip()
    if not text:
        return []
    parts = re.split(r"[\n,;]+", text)
    out: List[str] = []
    for p in parts:
        item = p.strip()
        if item:
            out.append(item)
    return out


def _dedupe_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _safe_model_type(model_id_or_path: str) -> str:
    try:
        cfg = AutoConfig.from_pretrained(model_id_or_path)
        return str(getattr(cfg, "model_type", "") or "").lower()
    except Exception:
        return ""


def _is_whisper_family(model_type: str, model_id_or_path: str) -> bool:
    mt = (model_type or "").lower()
    if "whisper" in mt:
        return True
    return "whisper" in str(model_id_or_path).lower()


def _is_moonshine_family(model_type: str, model_id_or_path: str) -> bool:
    mt = (model_type or "").lower()
    if "moonshine" in mt:
        return True
    return "moonshine" in str(model_id_or_path).lower()


def _resolve_processor_id_for_model(model_id_or_path: str, default_processor_id: str) -> str:
    p = Path(model_id_or_path)
    if not p.exists():
        return model_id_or_path

    has_local_processor = (
        (p / "preprocessor_config.json").is_file()
        or (p / "processor_config.json").is_file()
        or (p / "tokenizer_config.json").is_file()
    )
    return str(p) if has_local_processor else str(default_processor_id)


def beep() -> None:
    try:
        import winsound  # type: ignore
        winsound.MessageBeep()
    except Exception:
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass


def _is_peft_adapter_dir(p: Path) -> bool:
    return p.is_dir() and (p / "adapter_config.json").is_file()


def _model_pred_key(model_id_or_path: str, lora_merge: bool) -> str:
    p = Path(model_id_or_path)
    name = p.name if p.exists() else model_id_or_path
    if lora_merge and _is_peft_adapter_dir(p):
        return f"{name}__merged"
    return name


def _model_results_key(model_id_or_path: str, lora_merge: bool) -> str:
    if lora_merge and _is_peft_adapter_dir(Path(model_id_or_path)):
        return f"{model_id_or_path}::merged"
    return model_id_or_path


def _load_model_for_eval(model_id_or_path: str, *, lora_merge: bool, lora_base_model: Optional[str]):
    if lora_merge and _is_peft_adapter_dir(Path(model_id_or_path)):
        if not lora_base_model:
            raise RuntimeError("LoRA merge requested but no --lora_base_model provided.")
        try:
            from peft import PeftModel  # type: ignore
        except Exception as e:
            raise RuntimeError("LoRA merge requested but 'peft' is not installed. pip install -U peft") from e

        print(f"[lora] merging adapter: {model_id_or_path}")
        print(f"[lora] base model: {lora_base_model}")
        base = WhisperForConditionalGeneration.from_pretrained(lora_base_model)
        peft_model = PeftModel.from_pretrained(base, model_id_or_path, is_trainable=False)
        try:
            merged = peft_model.merge_and_unload(progressbar=True, safe_merge=True)
        except TypeError:
            merged = peft_model.merge_and_unload()
        return merged

    try:
        return AutoModelForSpeechSeq2Seq.from_pretrained(model_id_or_path)
    except Exception as e:
        mt = _safe_model_type(model_id_or_path)
        if _is_moonshine_family(mt, model_id_or_path):
            raise RuntimeError(
                "Failed to load Moonshine model. Update transformers first:\n"
                "pip install --upgrade git+https://github.com/huggingface/transformers.git"
            ) from e
        raise


def _run_groq_verify(base_pairs: List[dict], checkpoint_dir: Path) -> None:
    if not base_pairs:
        return

    script = Path(__file__).with_name("update_targetmix_transcripts_with_groq.py")
    if not script.exists():
        print("⚠ Groq verify requested but update_targetmix_transcripts_with_groq.py not found; skipping.")
        return

    payload = []
    for bp in base_pairs:
        payload.append(
            {
                "target_audio_path": bp.get("target_audio_path"),
                "other_audio_path": bp.get("other_audio_path"),
                "target_reference": bp.get("target_ref", ""),
                "other_reference": bp.get("other_ref", ""),
            }
        )

    tmp_json = checkpoint_dir / "groq_verify_pairs.json"
    tmp_json.parent.mkdir(parents=True, exist_ok=True)
    with tmp_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"▶ Groq verify: {len(payload)} pairs (unique audio handled by script)")
    subprocess.run([sys.executable, str(script), "--input_json", str(tmp_json), "--inplace"], check=True)

    data = json.loads(tmp_json.read_text(encoding="utf-8"))
    t_map: Dict[str, str] = {}
    o_map: Dict[str, str] = {}
    for obj in data:
        if not isinstance(obj, dict):
            continue
        tp = obj.get("target_audio_path")
        tr = obj.get("target_reference")
        if isinstance(tp, str) and isinstance(tr, str) and tr.strip():
            t_map[_norm_windows_key(tp)] = tr.strip()
        op = obj.get("other_audio_path")
        orf = obj.get("other_reference")
        if isinstance(op, str) and isinstance(orf, str) and orf.strip():
            o_map[_norm_windows_key(op)] = orf.strip()

    t_updated = 0
    o_updated = 0
    for bp in base_pairs:
        t_key = _norm_windows_key(str(bp.get("target_audio_path", "")))
        o_key = _norm_windows_key(str(bp.get("other_audio_path", "")))
        if t_key in t_map:
            bp["target_ref"] = t_map[t_key]
            t_updated += 1
        if o_key in o_map:
            bp["other_ref"] = o_map[o_key]
            o_updated += 1

    print(f"✓ Groq verify applied refs: target={t_updated} other={o_updated}")


def _apply_verified_refs_to_pairs(base_pairs: List[dict], pair_rows: List[dict], all_predictions: dict) -> None:
    if not base_pairs or not pair_rows:
        return

    base_ref: Dict[str, Tuple[str, str]] = {}
    for bp in base_pairs:
        bk = bp.get("base_key")
        if not bk:
            continue
        base_ref[str(bk)] = (bp.get("target_ref", ""), bp.get("other_ref", ""))

    updated = 0
    for pr in pair_rows:
        bk = pr.get("base_key")
        if bk not in base_ref:
            continue
        t_ref, o_ref = base_ref[bk]
        pr["target_ref"] = t_ref
        pr["other_ref"] = o_ref
        updated += 1

        key = pr.get("mix_key")
        if key in all_predictions:
            all_predictions[key]["target_reference"] = t_ref
            all_predictions[key]["other_reference"] = o_ref

    print(f"✓ Updated refs for {updated} eval pairs after Groq verify")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_manifest", required=True, type=Path)
    ap.add_argument("--speaker_scores_csv", required=True, type=Path)
    ap.add_argument("--checkpoint_dir", required=True, type=Path)
    ap.add_argument("--percentage", type=float, default=100.0, help="Subsample BASE PAIRS after pairing (percentage of pairs).")
    ap.add_argument("--target_percentage", type=float, default=100.0, help="Subsample TARGET rows before pairing (percentage of TARGET samples).")
    ap.add_argument("--target_max", type=int, default=0, help="Optional cap on number of TARGET rows after target subsampling. 0=disabled")

    ap.add_argument("--base_model", default="openai/whisper-small.en")
    ap.add_argument(
        "--models_csv",
        default=",".join(DEFAULT_EDGE_MODEL_PACK),
        help="Comma/semicolon/newline-separated model ids/paths to evaluate.",
    )
    ap.add_argument("--compare_openai_tiny", action="store_true")
    ap.add_argument("--append_checkpoints", type=int, default=0,
                    help="If 1, append checkpoint folders from --checkpoint_dir to the model list.")
    ap.add_argument("--skip_model_failures", type=int, default=1,
                    help="If 1, continue run when a model fails to load/infer; failure is recorded in output JSON.")
    ap.add_argument("--base_processor_id", default="openai/whisper-small.en")
    ap.add_argument("--lora_merge", action="store_true",
                    help="If set, merge PEFT/LoRA adapter checkpoints into the base model before evaluation.")
    ap.add_argument("--lora_base_model", default=None,
                    help="Base model id/path to merge LoRA adapters into (defaults to --base_processor_id).")
    ap.add_argument("--groq_verify", action="store_true",
                    help="Transcribe target/other audio with Groq before evaluation to refresh references.")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="transcribe")

    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--moonshine_token_limit_tps", type=float, default=6.5,
                    help="Moonshine-specific decoding cap in tokens/sec of audio (recommended ~6.5).")

    # batching
    ap.add_argument("--batch_size", type=int, default=0, help="0=auto default (GPU:8, CPU:1)")
    ap.add_argument("--auto_batch", type=int, default=0)
    ap.add_argument("--batch_min", type=int, default=1)
    ap.add_argument("--batch_max", type=int, default=64)

    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--mem_low", type=float, default=0.60)
    ap.add_argument("--mem_high", type=float, default=0.88)
    ap.add_argument("--cleanup_interval", type=int, default=50)

    ap.add_argument("--dynamic_audio_ctx", type=int, default=1)
    ap.add_argument("--normalize", default="whisper_basic", choices=["whisper_basic", "none"])

    # VAD
    ap.add_argument("--vad_filter", type=int, default=1)
    ap.add_argument("--vad_policy", default="skip", choices=["skip", "keep", "empty"])
    ap.add_argument("--vad_threshold", type=float, default=0.5)
    ap.add_argument("--vad_min_speech_ms", type=int, default=250)
    ap.add_argument("--vad_min_silence_ms", type=int, default=100)
    ap.add_argument("--vad_speech_pad_ms", type=int, default=200)

    # pairing
    ap.add_argument("--mix_per_target", type=int, default=1)
    ap.add_argument("--pairing_mode", default="round_robin", choices=["round_robin", "random", "hash"])
    ap.add_argument("--seed", type=int, default=42)

    # mixing rules
    ap.add_argument("--other_offset_mode", default="start", choices=["start", "random"],
                    help="Used only when overlap sweep is not given. If overlap sweep is given, placement is controlled by overlap ratio.")
    ap.add_argument("--other_peak_ratio", type=float, default=1.0)

    # sweep grid
    ap.add_argument("--sweep_snr_db", type=str, default="20,10,5,0,-5", help="Comma/space separated list")
    ap.add_argument("--sweep_overlap", type=str, default="0,0.25,0.5,0.75,1", help="Comma/space separated overlap ratios in [0,1]")
    ap.add_argument("--disable_overlap_sweep", action="store_true", help="If set, overlap is not swept; uses other_offset_mode")

    # caching
    ap.add_argument("--audio_cache_gb", type=float, default=1.0, help="0 disables. Helps a lot for sweep.")
    ap.add_argument("--ffmpeg_path", default="ffmpeg", help="ffmpeg executable (fallback for mp3/m4a).")

    # external OTHER dir mode
    ap.add_argument("--others_dir", type=Path, default=None, help="Folder containing OTHER audio files to mix.")
    ap.add_argument("--others_manifest", type=Path, default=None, help="Optional manifest for OTHER transcripts.")
    ap.add_argument("--allow_missing_other_ref", type=int, default=0,
                    help="If 0, OTHER files without transcripts are skipped when --others_manifest is provided.")
    ap.add_argument("--pairs_manifest", type=Path, default=None,
                    help="Path to save/load constructed base pairs (for resume across runs).")
    ap.add_argument("--rebuild_pairs", action="store_true", help="Force rebuild base pairs even if pairs_manifest exists.")

    # output/resume
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--import_per_sample_json", type=Path, default=None,
                    help="Optional per-sample predictions JSON to import (base model only) when run args match.")
    ap.add_argument("--import_force", action="store_true",
                    help="If set, import base-model predictions even when run args differ.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force_resume", action="store_true")
    ap.add_argument("--recalc_metrics", action="store_true",
                    help="Recompute metrics from saved predictions for already-evaluated models without new inference.")
    ap.add_argument("--save_every", type=int, default=100,
                    help="Auto-save predictions/results every N evaluated samples (0 disables).")

    args = ap.parse_args()

    if args.lora_merge and not args.lora_base_model:
        args.lora_base_model = args.base_processor_id

    if args.batch_size <= 0:
        args.batch_size = 8 if str(args.device).startswith("cuda") and torch.cuda.is_available() else 1

    if not (0.0 <= args.percentage <= 100.0):
        raise ValueError("--percentage must be 0..100")
    if not (0.0 <= args.target_percentage <= 100.0):
        raise ValueError("--target_percentage must be 0..100")
    if args.target_max < 0:
        raise ValueError("--target_max must be >= 0")
    if args.save_every < 0:
        raise ValueError("--save_every must be >= 0")
    if args.moonshine_token_limit_tps <= 0:
        raise ValueError("--moonshine_token_limit_tps must be > 0")

    rows = load_jsonl(args.test_manifest)

    # Optional ffmpeg fallback for broader formats
    core_load_orig = globals()["load_audio_mono_16k"]
    globals()["load_audio_mono_16k"] = make_loader_with_ffmpeg(str(args.ffmpeg_path), core_load_orig)

    if args.others_dir is not None:
        score_db = load_speaker_scores_csv(args.speaker_scores_csv)
        target_rows, target_info = select_targets_sorted(
            rows=rows,
            scores=score_db,
            target_take=int(args.target_max),
            target_percent=float(args.target_percentage),
        )

        other_paths = scan_audio_files(args.others_dir)
        other_tx: Optional[TranscriptDB] = None
        if args.others_manifest is not None:
            other_tx = build_transcript_db_from_manifest(Path(args.others_manifest))

        if args.pairs_manifest is None:
            args.pairs_manifest = args.checkpoint_dir / "pairs_manifest_targetmix_sweep_othersdir.jsonl"

        if (args.resume or args.force_resume) and args.pairs_manifest.exists() and not args.rebuild_pairs:
            base_pairs = load_jsonl(args.pairs_manifest)
            pair_info = {
                "loaded_from": str(args.pairs_manifest),
                "base_pairs": len(base_pairs),
                "target_info": target_info,
            }
            print(f"✓ Loaded base pairs from {args.pairs_manifest} ({len(base_pairs)} pairs)")
        else:
            base_pairs, base_info = build_base_pairs_targets_vs_others(
                target_rows=target_rows,
                other_paths=other_paths,
                other_tx=other_tx,
                mix_per_target=int(args.mix_per_target),
                pairing_mode=str(args.pairing_mode),
                seed=int(args.seed),
                allow_missing_other_ref=bool(args.allow_missing_other_ref),
            )
            pair_info = {"target_info": target_info, "base_info": base_info}
            save_jsonl = args.pairs_manifest or (args.checkpoint_dir / "pairs_manifest_targetmix_sweep_othersdir.jsonl")
            with save_jsonl.open("w", encoding="utf-8") as f:
                for r in base_pairs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"✓ Saved base pairs to {save_jsonl}")
    else:
        labels = load_speaker_sort_scores(args.speaker_scores_csv)

        base_pairs, pair_info = build_target_other_base_pairs(
            rows=rows,
            labels=labels,
            mix_per_target=int(args.mix_per_target),
            pairing_mode=str(args.pairing_mode),
            seed=int(args.seed),
            target_percentage=float(args.target_percentage),
            target_max=int(args.target_max),
        )

    # Subsample base pairs BEFORE sweep
    if args.percentage < 100.0:
        base_pairs = _deterministic_subsample(
            base_pairs,
            key_fn=lambda bp: str(bp.get("base_key", "")),
            percentage=float(args.percentage),
            seed=int(args.seed),
        )

    snr_list = parse_float_list(args.sweep_snr_db) or [10.0]
    if args.disable_overlap_sweep:
        overlap_list = None
    else:
        overlap_list = parse_overlap_list(args.sweep_overlap)

    conditions: List[MixCondition] = []
    if overlap_list is None:
        for snr in snr_list:
            conditions.append(MixCondition(snr_db=float(snr), overlap=None))
    else:
        for snr in snr_list:
            for ov in overlap_list:
                conditions.append(MixCondition(snr_db=float(snr), overlap=float(ov)))

    pair_rows = expand_pairs_with_conditions(base_pairs, conditions)

    # checkpoints
    checkpoints = list(args.checkpoint_dir.glob("model_epoch_*")) + list(args.checkpoint_dir.glob("s20_model_epoch_*"))
    def _ckpt_key(p: Path) -> int:
        try:
            # Handle both "model_epoch_XXXXX" and "s20_model_epoch_XXXXX" patterns
            parts = p.name.split("_")
            if parts[0] == "s20" and parts[1] == "model" and parts[2] == "epoch":
                return int(parts[3])
            elif parts[0] == "model" and parts[1] == "epoch":
                return int(parts[2])
            else:
                return 0
        except Exception:
            return 0
    checkpoints.sort(key=_ckpt_key)

    models: List[str] = []
    models.extend(parse_model_list_csv(args.models_csv))
    if args.compare_openai_tiny:
        models.insert(0, "openai/whisper-tiny.en")
    if args.base_model:
        models.insert(0, str(args.base_model))
    if bool(args.append_checkpoints):
        models.extend([str(p) for p in checkpoints])
    models = _dedupe_keep_order(models)
    if not models:
        raise RuntimeError("No models configured. Set --models_csv and/or --append_checkpoints 1.")

    vad_cfg = VADConfig(
        enabled=bool(args.vad_filter),
        policy=str(args.vad_policy),
        threshold=float(args.vad_threshold),
        min_speech_duration_ms=int(args.vad_min_speech_ms),
        min_silence_duration_ms=int(args.vad_min_silence_ms),
        speech_pad_ms=int(args.vad_speech_pad_ms),
    )

    audio_cache_bytes = int(max(0, args.audio_cache_gb) * (1024**3))

    cfg = EvalConfig(
        base_processor_id=args.base_processor_id,
        language=args.language,
        task=args.task,
        num_beams=args.num_beams,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        dynamic_audio_ctx=bool(args.dynamic_audio_ctx),
        vad=vad_cfg,
        normalize_mode=args.normalize,
        device=args.device,
        batch_size=int(args.batch_size),
        auto_batch=bool(args.auto_batch),
        batch_min=int(args.batch_min),
        batch_max=int(args.batch_max),
        fp16=bool(args.fp16),
        mem_low=float(args.mem_low),
        mem_high=float(args.mem_high),
        cleanup_interval=int(args.cleanup_interval),
        moonshine_token_limit_tps=float(args.moonshine_token_limit_tps),
        mix_per_target=int(args.mix_per_target),
        pairing_mode=str(args.pairing_mode),
        seed=int(args.seed),
        other_offset_mode=str(args.other_offset_mode),
        other_peak_ratio=float(args.other_peak_ratio),
        conditions=conditions,
        audio_cache_bytes=audio_cache_bytes,
    )

    out_json = args.out_json
    if out_json is None:
        out_json = args.checkpoint_dir / "evaluation_results_futo_like_targetmix_sweep.json"
    args.out_json = out_json
    current_run_args = _build_run_args(args, out_json)

    results = {
        "mode": "others_dir" if args.others_dir is not None else "single_manifest",
        "test_manifest": str(args.test_manifest),
        "speaker_scores_csv": str(args.speaker_scores_csv),
        "checkpoint_dir": str(args.checkpoint_dir),
        "others_dir": str(args.others_dir) if args.others_dir else None,
        "others_manifest": str(args.others_manifest) if args.others_manifest else None,
        "pairs_manifest": str(args.pairs_manifest) if args.pairs_manifest else None,
        "percentage": float(args.percentage),
        "pair_info": pair_info,
        "base_pairs": len(base_pairs),
        "conditions": [{"snr_db": c.snr_db, "overlap": c.overlap, "cond_id": c.cond_id()} for c in conditions],
        "pairs_total": len(pair_rows),
        "cfg": {
            "device": cfg.device,
            "language": cfg.language,
            "task": cfg.task,
            "num_beams": cfg.num_beams,
            "temperature": cfg.temperature,
            "max_new_tokens": cfg.max_new_tokens,
            "dynamic_audio_ctx": cfg.dynamic_audio_ctx,
            "normalize_mode": cfg.normalize_mode,
            "vad": vad_cfg.__dict__,
            "batch": {
                "batch_size": cfg.batch_size,
                "auto_batch": cfg.auto_batch,
                "batch_min": cfg.batch_min,
                "batch_max": cfg.batch_max,
                "fp16": cfg.fp16,
                "mem_low": cfg.mem_low,
                "mem_high": cfg.mem_high,
                "cleanup_interval": cfg.cleanup_interval,
            },
            "pairing": {
                "mix_per_target": cfg.mix_per_target,
                "pairing_mode": cfg.pairing_mode,
                "seed": cfg.seed,
                "target_percentage": float(args.target_percentage),
                "target_max": int(args.target_max),
                "percentage_pairs": float(args.percentage),
            },
            "mixing": {
                "other_offset_mode": cfg.other_offset_mode,
                "other_peak_ratio": cfg.other_peak_ratio,
            },
            "audio_cache_gb": float(args.audio_cache_gb),
            "ffmpeg_path": str(args.ffmpeg_path),
        },
        "models_requested": list(models),
        "model_failures": [],
        "models": [],
    }

    all_predictions: Dict[str, dict] = {}

    if args.resume or args.force_resume:
        existing_results, existing_predictions, per_sample_meta = load_existing_results(out_json)
        if existing_results:
            results = existing_results
            all_predictions = existing_predictions
        previous_run_args = _select_previous_run_args(existing_results, per_sample_meta)
        warn_if_run_args_changed(previous_run_args, current_run_args)

    results["models_requested"] = list(models)
    if not isinstance(results.get("model_failures"), list):
        results["model_failures"] = []

    active_keys = {pr["mix_key"] for pr in pair_rows}

    import_base_model_name: Optional[str] = None
    import_ok = False
    if args.import_per_sample_json:
        import_path = Path(args.import_per_sample_json)
        imported_predictions, import_meta = load_per_sample_predictions(import_path)
        if not imported_predictions:
            print(f"⚠ import_per_sample_json had no predictions: {import_path}")
        else:
            import_run_args = _extract_run_args_from_meta(import_meta)
            if not import_run_args:
                print(f"⚠ import_per_sample_json had no run_args metadata; skipping import: {import_path}")
            else:
                diffs = _diff_run_args(import_run_args, current_run_args, ignore_keys=RUN_ARGS_IMPORT_IGNORE_KEYS)
                proceed_import = True
                if diffs:
                    _print_run_args_diff_summary(
                        diffs,
                        f"⚠ import_per_sample_json args mismatch vs current run:",
                    )
                    if args.import_force:
                        proceed_import = True
                    else:
                        proceed_import = _prompt_yes_no(
                            f"Use base-model predictions from {import_path} anyway?",
                            default=True,
                        )
                if proceed_import:
                    import_base_model_name = _model_pred_key(str(args.base_model), bool(args.lora_merge))
                    merged_pairs, merged_preds = merge_imported_predictions(
                        all_predictions,
                        imported_predictions,
                        active_keys=active_keys,
                        allow_models={import_base_model_name},
                    )
                    import_ok = True
                    print(f"✓ Imported {merged_preds} base-model predictions from {import_path} ({merged_pairs} pairs)")
                else:
                    print(f"⏭ Skipped import from {import_path}")

    # Ensure prediction stubs exist for every pair (helps force_resume)
    for pr in pair_rows:
        key = pr["mix_key"]
        if key not in all_predictions or not isinstance(all_predictions.get(key), dict):
            all_predictions[key] = {
                "mix_key": key,
                "cond_id": pr.get("cond_id", ""),
                "snr_db": float(pr.get("snr_db")),
                "overlap": pr.get("overlap", None),
                "target_audio_path": pr["target_audio_path"],
                "other_audio_path": pr["other_audio_path"],
                "target_reference": pr["target_ref"],
                "other_reference": pr["other_ref"],
                "predictions": {},
            }
            continue

        entry = all_predictions[key]
        preds = entry.get("predictions")
        if not isinstance(preds, dict):
            entry["predictions"] = {}
        if not entry.get("cond_id"):
            entry["cond_id"] = pr.get("cond_id", "")
        if entry.get("snr_db") is None:
            entry["snr_db"] = float(pr.get("snr_db"))
        if "overlap" not in entry or entry.get("overlap") is None:
            entry["overlap"] = pr.get("overlap", None)
        if not entry.get("target_audio_path"):
            entry["target_audio_path"] = pr["target_audio_path"]
        if not entry.get("other_audio_path"):
            entry["other_audio_path"] = pr["other_audio_path"]
        if not entry.get("target_reference"):
            entry["target_reference"] = pr["target_ref"]
        if not entry.get("other_reference"):
            entry["other_reference"] = pr["other_ref"]

    if import_ok and import_base_model_name:
        missing_for_import = [
            k for k in active_keys
            if import_base_model_name not in all_predictions.get(k, {}).get("predictions", {})
        ]
        if missing_for_import:
            print(f"▶ Imported base-model predictions for {len(active_keys) - len(missing_for_import)}/{len(active_keys)} pairs; remaining will be evaluated.")
        else:
            model_results_key = _model_results_key(str(args.base_model), bool(args.lora_merge))
            overall, by_cond = recompute_metrics_from_saved_predictions(
                all_predictions,
                import_base_model_name,
                cfg.normalize_mode,
                active_keys=active_keys,
            )
            results["models"] = [mm for mm in results.get("models", []) if mm.get("model") != model_results_key]
            results["models"].append({"model": model_results_key, "metrics_overall": overall, "metrics_by_condition": by_cond})
            print(f"✓ Base model metrics recomputed from imported predictions: {model_results_key}")

    evaluated_models = {m["model"] for m in results.get("models", [])}

    print(f"Device: {args.device}")
    print(f"Test manifest rows: {len(rows)}")
    if pair_info:
        try:
            print(
                f"Targets: {pair_info.get('targets')} (kept {pair_info.get('targets_after_subsample')} after target subsample) | "
                f"Others: {pair_info.get('others')} | Unknown: {pair_info.get('rows_unknown')}"
            )
        except Exception:
            pass
    print(f"Base pairs (TARGET-OTHER): {len(base_pairs)}")
    print(f"Conditions: {len(conditions)}")
    print(f"Total eval pairs (base * conditions): {len(pair_rows)}")
    print(f"Models to eval: {len(models)}")
    print(f"Already evaluated models: {len(evaluated_models)}")

    groq_pending = bool(args.groq_verify)
    groq_done = False

    def maybe_run_groq_verify() -> None:
        nonlocal groq_done
        if groq_done or not groq_pending:
            return
        print("▶ Groq verify requested; running after base model evaluation...")
        _run_groq_verify(base_pairs, args.checkpoint_dir)
        _apply_verified_refs_to_pairs(base_pairs, pair_rows, all_predictions)
        groq_done = True

    vad_trimmer = SileroVADTrimmer() if cfg.vad.enabled else None

    base_model_name = str(args.base_model)

    for m in models:
        model_results_key = _model_results_key(m, bool(args.lora_merge))
        model_already_done = model_results_key in evaluated_models
        model_name = _model_pred_key(m, bool(args.lora_merge))
        wants_recalc = args.recalc_metrics or args.force_resume
        is_base_model = (str(m) == base_model_name)

        if model_already_done and not wants_recalc:
            missing_preds = 0
            for pr in pair_rows:
                key = pr["mix_key"]
                preds = all_predictions.get(key, {}).get("predictions", {})
                if model_name not in preds:
                    missing_preds += 1

            if missing_preds == 0:
                print(f"\n⏭ Skipping already evaluated model: {model_results_key}")
                if is_base_model:
                    maybe_run_groq_verify()
                continue
            else:
                print(f"\n▶ Found {missing_preds} new pairs for {model_results_key}; evaluating only missing predictions.")

        print("\n" + "=" * 80)
        if model_already_done and wants_recalc:
            print(f"Re-evaluating metrics from saved predictions: {model_results_key}")
        elif model_already_done and not wants_recalc:
            print(f"Evaluating missing pairs only: {m}")
        else:
            print(f"Evaluating: {m}")
        print("=" * 80)

        if model_already_done and wants_recalc:
            has_predictions = any(model_name in v.get("predictions", {}) for v in all_predictions.values())
            if has_predictions:
                print(f"📊 Recalculating metrics from existing predictions for {model_name}")
                active_keys = {pr["mix_key"] for pr in pair_rows}
                overall, by_cond = recompute_metrics_from_saved_predictions(
                    all_predictions, model_name, cfg.normalize_mode, active_keys=active_keys
                )
                results["models"] = [mm for mm in results.get("models", []) if mm.get("model") != model_results_key]
                results["models"].append({"model": model_results_key, "metrics_overall": overall, "metrics_by_condition": by_cond})
                save_incremental_results(results, all_predictions, out_json, run_args=current_run_args)
                if is_base_model:
                    maybe_run_groq_verify()
                continue
            else:
                print("⚠ Recalc requested but no saved predictions found; running full evaluation.")

        try:
            overall, by_cond = eval_one_model(
                model_id_or_path=m,
                pair_rows=pair_rows,
                cfg=cfg,
                vad_trimmer=vad_trimmer,
                all_predictions=all_predictions,
                out_json=out_json,
                lora_merge=bool(args.lora_merge),
                lora_base_model=args.lora_base_model,
                results=results,
                run_args=current_run_args,
                save_every=int(args.save_every),
            )
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if bool(args.skip_model_failures):
                print(f"⚠ Model failed and will be skipped: {m} | {msg}")
                results.setdefault("model_failures", [])
                if isinstance(results["model_failures"], list):
                    results["model_failures"].append({"model": str(m), "error": msg})
                save_incremental_results(results, all_predictions, out_json, run_args=current_run_args)
                continue
            raise

        if model_already_done:
            results["models"] = [mm for mm in results.get("models", []) if mm.get("model") != model_results_key]
        results["models"].append({"model": model_results_key, "metrics_overall": overall, "metrics_by_condition": by_cond})
        if isinstance(results.get("model_failures"), list):
            results["model_failures"] = [f for f in results["model_failures"] if str(f.get("model")) != str(m)]
        save_incremental_results(results, all_predictions, out_json, run_args=current_run_args)
        if is_base_model:
            maybe_run_groq_verify()

        print(f"samples={overall.get('samples')} skipped={overall.get('skipped')}")
        print(f"WER target micro={overall.get('wer_micro_target')} | WER other micro={overall.get('wer_micro_other')}")
        print(f"CER target micro={overall.get('cer_micro_target')} | CER other micro={overall.get('cer_micro_other')}")
        print(f"win_rate(target closer)={overall.get('win_rate_target_closer')} avg_margin(other-target)={overall.get('avg_margin_other_minus_target')}")

    print("\nDone.")
    beep()


if __name__ == "__main__":
    main()
