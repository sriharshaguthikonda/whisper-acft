#!/usr/bin/env python3
r"""stage_19c_specific_target_advanced_futo_like_evaluate_targetmix_sweep_with_cer.py

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

i:\Whisper-training-env\Scripts\python.exe "i:\whisper-acft\stage_19c_specific_target_advanced_futo_like_evaluate_targetmix_sweep_with_cer.py" `
  --test_manifest "I:\Record_chunks\pairs_manifest_combined_all_datasets_randomized_test_with_tempo.jsonl" `
  --speaker_scores_csv "I:\whisper-acft\speaker_sort_scores.csv" `
  --checkpoint_dir "I:\Stage_17_aug_futo_wer_dora_dyn_ctx_chkpts_tiny_en_21" `
  --mix_per_target 5 `
  --other_peak_ratio 1.0 `
  --sweep_snr_db "15,5,0" `
  --sweep_overlap "0.25,0.75,1" `
  --dynamic_audio_ctx 1 `
  --batch_size 4 `
  --auto_batch 0 `
  --resume `
  --others_dir "I:\Record_others_chunks" `
  --others_manifest "I:\Record_others_chunks\pairs_pending_stereo.jsonl" `
  --ffmpeg_path "ffmpeg" `
  --percentage 10 `
  --vad_filter 0 `
  --lora_merge `
  --lora_base_model "futo-org/acft-whisper-tiny.en" `
  --groq_verify `

  --recalc_metrics


Outputs (in checkpoint_dir by default)
-------------------------------------
- evaluation_results_futo_like_targetmix_sweep.json
- evaluation_per_sample_predictions_targetmix_sweep.json

Dependencies
------------
pip install transformers torch soundfile jiwer tqdm numpy
Optional (better resample): scipy
Optional (VAD): torch hub will pull Silero-VAD
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
from typing import Dict, List, Tuple, Optional, Iterable
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm

import soundfile as sf
import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration


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
RUN_ARGS_IGNORE_KEYS = {"resume", "force_resume", "recalc_metrics"}


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


def _normalize_run_args_for_compare(run_args: Optional[dict]) -> Dict[str, object]:
    if not run_args or not isinstance(run_args, dict):
        return {}
    args_dict = run_args.get("args")
    if not isinstance(args_dict, dict):
        return {}
    normed = {}
    for k, v in args_dict.items():
        if k in RUN_ARGS_IGNORE_KEYS:
            continue
        normed[k] = v
    return normed


def _diff_run_args(prev: Optional[dict], current: Optional[dict]) -> Dict[str, dict]:
    prev_norm = _normalize_run_args_for_compare(prev)
    curr_norm = _normalize_run_args_for_compare(current)
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


def load_existing_results(out_json: Path) -> Tuple[dict, dict, Optional[dict]]:
    if not out_json.exists():
        return {}, {}, None

    try:
        with out_json.open("r", encoding="utf-8") as f:
            results = json.load(f)

        per_sample_json = out_json.parent / "evaluation_per_sample_predictions_targetmix_sweep.json"
        all_predictions = {}
        per_sample_meta = None
        if per_sample_json.exists():
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
    processor: WhisperProcessor,
    cfg: EvalConfig,
    vad_trimmer: Optional[SileroVADTrimmer],
    all_predictions: dict,
    out_json: Path,
    *,
    lora_merge: bool,
    lora_base_model: Optional[str],
) -> Tuple[Dict, Dict[str, Dict]]:

    model = _load_model_for_eval(
        model_id_or_path,
        lora_merge=bool(lora_merge),
        lora_base_model=lora_base_model,
    )
    model.to(cfg.device)
    model.eval()

    try:
        fids = processor.get_decoder_prompt_ids(language=cfg.language, task=cfg.task)
        if hasattr(model, "generation_config") and hasattr(model.generation_config, "forced_decoder_ids"):
            model.generation_config.forced_decoder_ids = fids
    except Exception:
        pass

    gen_kwargs = build_generate_kwargs(cfg)

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
            eos_id = getattr(processor.tokenizer, "eos_token_id", None)
        except Exception:
            eos_id = None
    if eos_id is None:
        eos_id = 50256  # Whisper

    per_item: List[dict] = []
    skipped: List[dict] = []

    if cfg.vad.enabled and vad_trimmer is None:
        vad_trimmer = SileroVADTrimmer()

    model_name = _model_pred_key(model_id_or_path, bool(lora_merge))

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
                            continue

                        audio_eval = mixed
                    else:
                        audio_eval = audio_vad

                dur_eval = seconds_from_audio(audio_eval, 16000)

                inputs = processor(
                    audio_eval,
                    sampling_rate=16000,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                input_features = inputs["input_features"]
                attn_mask = inputs.get("attention_mask")  # (1, T)

                if cfg.dynamic_audio_ctx:
                    input_features = crop_input_features_for_duration(input_features, dur_eval)
                    if attn_mask is not None:
                        attn_mask = attn_mask[:, : input_features.shape[-1]]

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
                        "input_features": input_features,
                        "attention_mask": attn_mask,
                    }
                )

            except Exception as e:
                skipped.append({"mix_key": key, "reason": f"exception: {type(e).__name__}: {e}", "cond_id": cid})
                continue

            if len(batch_buf) < cur_bs:
                continue

            # Run a batch
            pending = batch_buf
            batch_buf = []

            while pending:
                chunk = pending[:cur_bs]

                try:
                    feats = [c["input_features"] for c in chunk]
                    masks = [c.get("attention_mask") for c in chunk]
                    batch_feats, batch_mask = _pad_stack_input_features_and_mask(feats, masks)
                    batch_feats = batch_feats.to(cfg.device, non_blocking=True)
                    batch_mask = batch_mask.to(cfg.device, non_blocking=True)
                    if use_cuda and cfg.fp16:
                        batch_feats = batch_feats.half()

                    generated_ids = model.generate(
                        input_features=batch_feats,
                        attention_mask=batch_mask,
                        **gen_kwargs,
                    )
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

                    pending = pending[cur_bs:]

                    # Auto batch-size tuning
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

        # flush remaining
        if batch_buf:
            pending = batch_buf
            while pending:
                chunk = pending[:cur_bs]
                feats = [c["input_features"] for c in chunk]
                masks = [c.get("attention_mask") for c in chunk]
                batch_feats, batch_mask = _pad_stack_input_features_and_mask(feats, masks)
                batch_feats = batch_feats.to(cfg.device, non_blocking=True)
                batch_mask = batch_mask.to(cfg.device, non_blocking=True)
                if use_cuda and cfg.fp16:
                    batch_feats = batch_feats.half()
                generated_ids = model.generate(
                    input_features=batch_feats,
                    attention_mask=batch_mask,
                    **gen_kwargs,
                )
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

                pending = pending[cur_bs:]

    # Overall metrics and per-condition metrics
    overall = compute_metrics_from_items(per_item, cfg.normalize_mode)
    by_cond_items = group_items_by_condition(per_item)
    by_cond: Dict[str, Dict] = {}
    for cid, items in by_cond_items.items():
        by_cond[cid] = compute_metrics_from_items(items, cfg.normalize_mode)

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

    return WhisperForConditionalGeneration.from_pretrained(model_id_or_path)


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

    ap.add_argument("--base_model", default="futo-org/acft-whisper-tiny.en")
    ap.add_argument("--compare_openai_tiny", action="store_true")
    ap.add_argument("--base_processor_id", default="openai/whisper-tiny.en")
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
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force_resume", action="store_true")
    ap.add_argument("--recalc_metrics", action="store_true",
                    help="Recompute metrics from saved predictions for already-evaluated models without new inference.")

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

    rows = load_jsonl(args.test_manifest)

    # Optional ffmpeg fallback for broader formats
    core_load_orig = globals()['load_audio_mono_16k']
    load_audio_mono_16k = make_loader_with_ffmpeg(str(args.ffmpeg_path), core_load_orig)

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

    if args.groq_verify:
        _run_groq_verify(base_pairs, args.checkpoint_dir)

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
    if args.compare_openai_tiny:
        models.append("openai/whisper-tiny.en")
    models.append(str(args.base_model))
    models.extend([str(p) for p in checkpoints])

    processor = WhisperProcessor.from_pretrained(args.base_processor_id)

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

    # Ensure prediction stubs exist for every pair (helps force_resume)
    for pr in pair_rows:
        key = pr["mix_key"]
        if key not in all_predictions:
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

    vad_trimmer = SileroVADTrimmer() if cfg.vad.enabled else None

    for m in models:
        model_results_key = _model_results_key(m, bool(args.lora_merge))
        model_already_done = model_results_key in evaluated_models
        model_name = _model_pred_key(m, bool(args.lora_merge))
        wants_recalc = args.recalc_metrics or args.force_resume

        if model_already_done and not wants_recalc:
            print(f"\n⏭ Skipping already evaluated model: {model_results_key}")
            continue

        print("\n" + "=" * 80)
        if model_already_done and wants_recalc:
            print(f"Re-evaluating metrics from saved predictions: {model_results_key}")
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
                continue
            else:
                print("⚠ Recalc requested but no saved predictions found; running full evaluation.")

        overall, by_cond = eval_one_model(
            model_id_or_path=m,
            pair_rows=pair_rows,
            processor=processor,
            cfg=cfg,
            vad_trimmer=vad_trimmer,
            all_predictions=all_predictions,
            out_json=out_json,
            lora_merge=bool(args.lora_merge),
            lora_base_model=args.lora_base_model,
        )

        results["models"].append({"model": model_results_key, "metrics_overall": overall, "metrics_by_condition": by_cond})
        save_incremental_results(results, all_predictions, out_json, run_args=current_run_args)

        print(f"samples={overall.get('samples')} skipped={overall.get('skipped')}")
        print(f"WER target micro={overall.get('wer_micro_target')} | WER other micro={overall.get('wer_micro_other')}")
        print(f"CER target micro={overall.get('cer_micro_target')} | CER other micro={overall.get('cer_micro_other')}")
        print(f"win_rate(target closer)={overall.get('win_rate_target_closer')} avg_margin(other-target)={overall.get('avg_margin_other_minus_target')}")

    print("\nDone.")
    beep()


if __name__ == "__main__":
    main()
