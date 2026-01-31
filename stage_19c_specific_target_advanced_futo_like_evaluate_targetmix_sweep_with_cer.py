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
i:\Whisper-training-env\Scripts\python.exe stage_19c_specific_target_advanced_futo_like_evaluate_targetmix_sweep_with_cer.py ^ 
  --test_manifest "I:\Record_chunks\pairs_manifest_local_english_only_filtered_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_randomized_bottom_filtered_test.jsonl" ^
  --speaker_scores_csv "I:\whisper-acft\speaker_sort_scores_sorted.csv" ^
  --checkpoint_dir "I:\Stage_2_shuffle_Dynamic_n_ctx_checkpoints_partialctx_tiny_en_8" ^
  --mix_per_target 1 ^
  --other_peak_ratio 1.0 ^
  --sweep_snr_db "20,10,5,0,-5" ^
  --sweep_overlap "0,0.25,0.5,0.75,1" ^
  --auto_batch 0
  --resume


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

def load_existing_results(out_json: Path) -> Tuple[dict, dict]:
    if not out_json.exists():
        return {}, {}

    try:
        with out_json.open("r", encoding="utf-8") as f:
            results = json.load(f)

        per_sample_json = out_json.parent / "evaluation_per_sample_predictions_targetmix_sweep.json"
        all_predictions = {}
        if per_sample_json.exists():
            with per_sample_json.open("r", encoding="utf-8") as f:
                per_sample_data = json.load(f)
                for item in per_sample_data:
                    key = item.get("mix_key") or item.get("key")
                    if key:
                        all_predictions[key] = item

        print(f"✓ Loaded existing results from: {out_json}")
        print(f"✓ Found {len(results.get('models', []))} already evaluated models")
        return results, all_predictions
    except Exception as e:
        print(f"⚠ Could not load existing results: {e}")
        return {}, {}


def save_incremental_results(results: dict, all_predictions: dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    per_sample_json = out_json.parent / "evaluation_per_sample_predictions_targetmix_sweep.json"
    with per_sample_json.open("w", encoding="utf-8") as f:
        json.dump(list(all_predictions.values()), f, indent=2, ensure_ascii=False)

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
) -> Tuple[Dict, Dict[str, Dict]]:

    model = WhisperForConditionalGeneration.from_pretrained(model_id_or_path)
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

    model_name = Path(model_id_or_path).name if Path(model_id_or_path).exists() else model_id_or_path

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
    ap.add_argument("--pairing_mode", default="round_robin", choices=["round_robin", "random"])
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

    # output/resume
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force_resume", action="store_true")

    args = ap.parse_args()

    if args.batch_size <= 0:
        args.batch_size = 8 if str(args.device).startswith("cuda") and torch.cuda.is_available() else 1

    if not (0.0 <= args.percentage <= 100.0):
        raise ValueError("--percentage must be 0..100")
    if not (0.0 <= args.target_percentage <= 100.0):
        raise ValueError("--target_percentage must be 0..100")
    if args.target_max < 0:
        raise ValueError("--target_max must be >= 0")

    rows = load_jsonl(args.test_manifest)
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
    checkpoints = list(args.checkpoint_dir.glob("model_epoch_*"))
    def _ckpt_key(p: Path) -> int:
        try:
            return int(p.name.split("_")[2])
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

    results = {
        "test_manifest": str(args.test_manifest),
        "speaker_scores_csv": str(args.speaker_scores_csv),
        "checkpoint_dir": str(args.checkpoint_dir),
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
        },
        "models": [],
    }

    all_predictions: Dict[str, dict] = {}

    if args.resume or args.force_resume:
        existing_results, existing_predictions = load_existing_results(out_json)
        if existing_results:
            results = existing_results
            all_predictions = existing_predictions

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
        if m in evaluated_models:
            print(f"\n⏭ Skipping already evaluated model: {m}")
            continue

        print("\n" + "=" * 80)
        print(f"Evaluating: {m}")
        print("=" * 80)

        model_name = Path(m).name if Path(m).exists() else m

        if args.force_resume:
            has_predictions = any(model_name in v.get("predictions", {}) for v in all_predictions.values())
            if has_predictions:
                print(f"📊 Recalculating metrics from existing predictions for {model_name}")
                active_keys = {pr["mix_key"] for pr in pair_rows}
                overall, by_cond = recompute_metrics_from_saved_predictions(
                    all_predictions, model_name, cfg.normalize_mode, active_keys=active_keys
                )
                results["models"].append({"model": m, "metrics_overall": overall, "metrics_by_condition": by_cond})
                save_incremental_results(results, all_predictions, out_json)
                continue

        overall, by_cond = eval_one_model(
            model_id_or_path=m,
            pair_rows=pair_rows,
            processor=processor,
            cfg=cfg,
            vad_trimmer=vad_trimmer,
            all_predictions=all_predictions,
            out_json=out_json,
        )

        results["models"].append({"model": m, "metrics_overall": overall, "metrics_by_condition": by_cond})
        save_incremental_results(results, all_predictions, out_json)

        print(f"samples={overall.get('samples')} skipped={overall.get('skipped')}")
        print(f"WER target micro={overall.get('wer_micro_target')} | WER other micro={overall.get('wer_micro_other')}")
        print(f"CER target micro={overall.get('cer_micro_target')} | CER other micro={overall.get('cer_micro_other')}")
        print(f"win_rate(target closer)={overall.get('win_rate_target_closer')} avg_margin(other-target)={overall.get('avg_margin_other_minus_target')}")

    print("\nDone.")
    beep()


if __name__ == "__main__":
    main()
