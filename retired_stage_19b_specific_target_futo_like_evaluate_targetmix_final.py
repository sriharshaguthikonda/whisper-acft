#!/usr/bin/env python3
r"""stage_19b_futo_like_evaluate_target_vs_other_mix.py

Stage 19 (fixed) evaluation for "target-speaker" style training.

What this script does (as per your intended logic)
--------------------------------------------------
1) Read the test manifest (JSONL). Each row must at least have:
   - audio_path
   - raw_transcription

2) Read speaker_sort_scores.csv and label each audio file as:
   - TARGET (your voice)
   - OTHER  (someone else's voice)

3) Build evaluation pairs:
   For each TARGET row, pick N OTHER rows and create a mixture:
       mixed_audio = target_audio + scaled(other_audio)

4) Run the ASR model on the mixed audio, then compute:
   - WER vs TARGET transcript  (should be LOW if model follows you)
   - WER vs OTHER transcript   (should be HIGH if model ignores the interferer)

5) Report both sets of metrics, plus a "win rate":
   win = (wer_target < wer_other)



Usage
---------------------------------------------------
i:\Whisper-training-env\Scripts\python.exe stage_19b_futo_like_evaluate_targetmix_final.py ^
  --test_manifest "I:\Record_chunks\pairs_manifest_local_english_only_filtered_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_randomized_bottom_filtered_test.jsonl" ^
  --speaker_scores_csv "I:\whisper-acft\speaker_sort_scores_sorted.csv" ^
  --checkpoint_dir "I:\Stage_2_shuffle_Dynamic_n_ctx_checkpoints_partialctx6" ^
  --percentage 10 ^
  --mix_per_target 1 ^
  --snr_db 10 ^
  --other_peak_ratio 1.0 ^
  --pairing_mode round_robin ^
  --other_offset_mode start ^
  --resume


Outputs
-------
- evaluation_results_futo_like_targetmix.json (default) with per-model metrics
- evaluation_per_sample_predictions_targetmix.json with per-pair predictions




Notes
-----
- Mixing is deterministic given --seed and --pairing_mode.
- By default the interfering OTHER audio is forced to NOT exceed the TARGET peak amplitude
  (so "bad audio is never louder than good audio" at any time, in an amplitude sense).
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
from typing import Dict, List, Tuple, Optional

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
    """A pragmatic normaliser for WER.

    - lowercase
    - collapse whitespace
    - remove most punctuation (keeps apostrophes inside words)
    - map <|nospeech|> / <|nocaptions|> to empty
    """
    s = (s or "").strip().lower()
    s = s.replace("’", "'")
    s = re.sub(r"(?!\B'\b)[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s in ("<|nospeech|>", "<|nocaptions|>"):
        return ""
    return s


# ----------------------------
# Path matching helpers
# ----------------------------

def _norm_windows_key(p: str) -> str:
    """Normalise a Windows-ish path for stable matching (case-insensitive, slash-normalised)."""
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

    # Build basename lookup only when unique (prevents accidental mismatches)
    by_basename: Dict[str, str] = {}
    for b, counts in basename_counts.items():
        if len(counts) == 1:
            by_basename[b] = next(iter(counts.keys()))
        # else ambiguous -> do not include

    return SpeakerLabelDB(by_path=by_path, by_basename=by_basename)


# ----------------------------
# Audio helpers
# ----------------------------

def load_audio_mono_16k(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)

    # stereo -> mono
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = np.asarray(audio, dtype=np.float32)

    # handle NaNs/Infs
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if sr != 16000:
        # Prefer resample_poly if scipy is available.
        try:
            import scipy.signal  # type: ignore
            audio = scipy.signal.resample_poly(audio, 16000, sr).astype(np.float32)
            sr = 16000
        except Exception:
            # Linear interpolation fallback
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


def mix_target_with_other(
    target: np.ndarray,
    other: np.ndarray,
    snr_db_target_over_other: float,
    other_offset_mode: str,
    rng: np.random.Generator,
    other_peak_ratio: float = 1.0,
    max_abs: float = 0.999,
) -> Tuple[np.ndarray, Dict]:
    """Return mixture where TARGET dominates over OTHER by desired SNR (in RMS terms).

    Enforces: peak(|other_scaled|) <= peak(|target|) * other_peak_ratio
    so the "interferer" is never louder than the target in peak amplitude terms.
    """
    t = np.asarray(target, dtype=np.float32)
    o = np.asarray(other, dtype=np.float32)

    # Align other into a same-length buffer as target
    t_len = len(t)
    if t_len == 0:
        return t, {"note": "empty_target"}

    if len(o) == 0:
        return t, {"note": "empty_other"}

    # If other longer than target: crop
    if len(o) > t_len:
        if other_offset_mode == "random":
            start = int(rng.integers(0, len(o) - t_len + 1))
            o = o[start:start + t_len]
        else:
            o = o[:t_len]

    o_aligned = np.zeros_like(t)
    if other_offset_mode == "random" and len(o) < t_len:
        start = int(rng.integers(0, t_len - len(o) + 1))
    else:
        start = 0
    o_aligned[start:start + len(o)] = o

    rt = rms(t)
    ro = rms(o_aligned)

    meta: Dict = {
        "snr_db_target_over_other_req": float(snr_db_target_over_other),
        "other_offset_mode": other_offset_mode,
        "other_start_sample": int(start),
        "rms_target": float(rt),
        "rms_other_raw": float(ro),
        "other_peak_ratio": float(other_peak_ratio),
    }

    if ro < 1e-8 or rt < 1e-8:
        # Degenerate; just sum (or return target)
        mix = np.clip(t + o_aligned, -max_abs, max_abs).astype(np.float32)
        meta["note"] = "degenerate_rms"
        return mix, meta

    snr_lin = 10.0 ** (float(snr_db_target_over_other) / 20.0)
    desired_ro = rt / snr_lin
    scale = desired_ro / ro

    o_scaled = o_aligned * float(scale)

    # Enforce peak constraint (other never louder than target peak)
    peak_t = float(np.max(np.abs(t))) + 1e-12
    peak_o = float(np.max(np.abs(o_scaled)))
    if peak_o > peak_t * float(other_peak_ratio):
        scale2 = (peak_t * float(other_peak_ratio)) / (peak_o + 1e-12)
        o_scaled = o_scaled * float(scale2)
        scale = scale * float(scale2)

    mix = t + o_scaled

    # Prevent clipping by global scaling if needed
    peak_mix = float(np.max(np.abs(mix))) + 1e-12
    if peak_mix > max_abs:
        mix = mix * float(max_abs / peak_mix)

    # Actual SNR after any peak constraint and clipping-scale
    ro2 = rms(o_scaled)
    snr_actual = 20.0 * math.log10((rt + 1e-12) / (ro2 + 1e-12))

    meta.update(
        {
            "scale_other": float(scale),
            "rms_other_scaled": float(ro2),
            "snr_db_target_over_other_actual": float(snr_actual),
            "peak_target": peak_t,
            "peak_other_scaled": float(np.max(np.abs(o_scaled))),
            "peak_mix": float(np.max(np.abs(mix))),
        }
    )

    return mix.astype(np.float32), meta


# ----------------------------
# Dynamic audio context emulation (mel cropping)
# ----------------------------

FULL_MEL_FRAMES = 3000  # Whisper feature extractor pads/truncates to 30s => ~3000 frames


def mel_frames_for_duration(duration_sec: float) -> int:
    d = max(0.0, min(30.0, float(duration_sec)))
    frames = int(round((FULL_MEL_FRAMES / 30.0) * d))
    return max(1, min(FULL_MEL_FRAMES, frames))


def crop_input_features_for_duration(input_features: torch.Tensor, duration_sec: float) -> torch.Tensor:
    """Crop time dimension of Whisper mel features.

    input_features shape either:
    - (B, 80, T)
    - (B, T, 80)

    We crop T.
    """
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
# Optional: Voice Activity Detection (Silero) trimming
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
    """Trim leading/trailing silence using Silero VAD.

    We take the first-to-last speech span (+pad), not splitting into multiple chunks.
    """

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

            try:
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    force_reload=False,
                )
            except TypeError:
                model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", force_reload=False)

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

    # batching / memory
    batch_size: int
    auto_batch: bool
    batch_min: int
    batch_max: int
    fp16: bool
    mem_low: float
    mem_high: float
    cleanup_interval: int

    # mixing
    mix_per_target: int
    snr_db: float
    other_offset_mode: str
    other_peak_ratio: float
    pairing_mode: str
    seed: int


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
# Whisper generate kwargs
# ----------------------------

def build_generate_kwargs(cfg: EvalConfig) -> Dict:
    # Deterministic by default (temperature=0)
    return {
        "num_beams": int(cfg.num_beams),
        "temperature": float(cfg.temperature),
        "max_new_tokens": int(cfg.max_new_tokens),
    }


def _pad_stack_input_features(features_list: List[torch.Tensor]) -> torch.Tensor:
    """Pad a list of (1,80,T) tensors to (B,80,Tmax)."""
    ts = [t.squeeze(0) for t in features_list]
    max_t = max(int(t.shape[-1]) for t in ts)
    out = torch.zeros((len(ts), 80, max_t), dtype=ts[0].dtype)
    for i, t in enumerate(ts):
        out[i, :, : t.shape[-1]] = t
    return out


def _cuda_mem_ratio() -> float:
    if not torch.cuda.is_available():
        return 0.0
    try:
        free, total = torch.cuda.mem_get_info()
        used = total - free
        return float(used) / float(total)
    except Exception:
        return 0.0


def _wer_duration_buckets(per_item: List[dict]) -> Dict[str, Optional[float]]:
    """Mean WER by duration bucket (uses duration_sec_eval, falls back to duration_sec_target)."""
    buckets: Dict[str, List[float]] = {"0-1s": [], "1-2s": [], "2-5s": [], "5-10s": [], "10-30s": []}

    for row in per_item:
        d = float(row.get("duration_sec_eval", row.get("duration_sec_target", 0.0)) or 0.0)
        w = row.get("wer_target", None)
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


def _wer_duration_buckets_other(per_item: List[dict]) -> Dict[str, Optional[float]]:
    """Mean OTHER WER by duration bucket."""
    buckets: Dict[str, List[float]] = {"0-1s": [], "1-2s": [], "2-5s": [], "5-10s": [], "10-30s": []}

    for row in per_item:
        d = float(row.get("duration_sec_eval", row.get("duration_sec_target", 0.0)) or 0.0)
        w = row.get("wer_other", None)
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


def _stable_uint64_from_str(s: str, seed: int) -> int:
    """Deterministic per-item seed (independent of Python's hash randomisation)."""
    b = (str(int(seed)) + "||" + (s or "")).encode("utf-8", errors="ignore")
    h = hashlib.md5(b).digest()
    return int.from_bytes(h[:8], "little", signed=False)


# ----------------------------
# Pair building
# ----------------------------

def build_target_other_pairs(
    rows: List[dict],
    labels: SpeakerLabelDB,
    mix_per_target: int,
    pairing_mode: str,
    seed: int,
) -> Tuple[List[dict], Dict]:
    """Return a list of pair-rows suitable for evaluation."""
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

    if not targets:
        raise RuntimeError("No TARGET rows found in test manifest after CSV matching.")
    if not others:
        raise RuntimeError("No OTHER rows found in test manifest after CSV matching.")

    rng = random.Random(int(seed))

    # prepare OTHER indices for round-robin if needed
    other_indices = list(range(len(others)))
    rng.shuffle(other_indices)
    rr_ptr = 0

    pairs: List[dict] = []
    for ti, trow in enumerate(targets):
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

            mix_key = f"{_norm_windows_key(t_ap)}||{_norm_windows_key(o_ap)}||k{k}"

            pairs.append(
                {
                    "mix_key": mix_key,
                    "target_audio_path": t_ap,
                    "other_audio_path": o_ap,
                    "target_ref": t_ref,
                    "other_ref": o_ref,
                }
            )

    return pairs, info


# ----------------------------
# Resume / persistence
# ----------------------------

def load_existing_results(out_json: Path) -> Tuple[dict, dict]:
    """Load existing results for resume capability."""
    if not out_json.exists():
        return {}, {}

    try:
        with out_json.open("r", encoding="utf-8") as f:
            results = json.load(f)

        per_sample_json = out_json.parent / "evaluation_per_sample_predictions_targetmix.json"
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
    """Save intermediate results after each model evaluation."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    per_sample_json = out_json.parent / "evaluation_per_sample_predictions_targetmix.json"
    with per_sample_json.open("w", encoding="utf-8") as f:
        json.dump(list(all_predictions.values()), f, indent=2, ensure_ascii=False)

    print(f"✓ Incremental results saved to: {out_json}")
    print(f"✓ Per-sample predictions saved to: {per_sample_json}")


# ----------------------------
# Metric recompute (force resume)
# ----------------------------

def recalculate_metrics_from_predictions(all_predictions: dict, model_name: str, cfg: EvalConfig) -> Tuple[Dict, List[dict]]:
    preds_raw: List[str] = []
    trefs_raw: List[str] = []
    orefs_raw: List[str] = []
    per_item: List[dict] = []
    skipped = 0

    per_utt_wer_t: List[float] = []
    per_utt_wer_o: List[float] = []
    wins = 0

    for key, data in all_predictions.items():
        preds = data.get("predictions", {})
        if model_name not in preds:
            skipped += 1
            continue

        pred = (preds[model_name].get("pred", "") or "").strip()
        tref = (data.get("target_reference", data.get("target_ref", "")) or "").strip()
        oref = (data.get("other_reference", data.get("other_ref", "")) or "").strip()

        preds_raw.append(pred)
        trefs_raw.append(tref)
        orefs_raw.append(oref)

        if cfg.normalize_mode in {"whisper_basic", "basic"}:
            pred_n = _basic_whisperish_normalize(pred)
            tref_n = _basic_whisperish_normalize(tref)
            oref_n = _basic_whisperish_normalize(oref)
        else:
            pred_n, tref_n, oref_n = pred, tref, oref

        wt = float(jiwer.wer(tref_n, pred_n))
        wo = float(jiwer.wer(oref_n, pred_n))
        per_utt_wer_t.append(wt)
        per_utt_wer_o.append(wo)
        if wt < wo:
            wins += 1

        per_item.append(
            {
                "mix_key": key,
                "target_audio_path": data.get("target_audio_path"),
                "other_audio_path": data.get("other_audio_path"),
                "duration_sec_eval": preds[model_name].get("duration_sec_eval", 0.0),
                "target_ref": tref,
                "other_ref": oref,
                "pred": pred,
                "target_ref_norm": tref_n,
                "other_ref_norm": oref_n,
                "pred_norm": pred_n,
                "wer_target": wt,
                "wer_other": wo,
                "win_target_closer": bool(wt < wo),
                "vad": preds[model_name].get("vad", {"vad_applied": False}),
                "mix": preds[model_name].get("mix", {}),
            }
        )

    n = len(per_item)
    if n == 0:
        return {
            "samples": 0,
            "skipped": skipped,
            "wer_micro": None,
            "cer_micro": None,
            "wer_macro": None,
            "wer_micro_target": None,
            "wer_micro_other": None,
            "cer_micro_target": None,
            "cer_micro_other": None,
            "wer_macro_target": None,
            "wer_macro_other": None,
            "win_rate_target_closer": None,
            "avg_margin_other_minus_target": None,
            "wer_by_duration": {"0-1s": None, "1-2s": None, "2-5s": None, "5-10s": None, "10-30s": None},
            "wer_by_duration_target": {"0-1s": None, "1-2s": None, "2-5s": None, "5-10s": None, "10-30s": None},
            "wer_by_duration_other": {"0-1s": None, "1-2s": None, "2-5s": None, "5-10s": None, "10-30s": None},
            "normalize_mode": cfg.normalize_mode,
            "dynamic_audio_ctx": bool(cfg.dynamic_audio_ctx),
            "vad": cfg.vad.__dict__,
            "mixing": {
                "snr_db": cfg.snr_db,
                "other_offset_mode": cfg.other_offset_mode,
                "other_peak_ratio": cfg.other_peak_ratio,
                "pairing_mode": cfg.pairing_mode,
                "mix_per_target": cfg.mix_per_target,
                "seed": cfg.seed,
            },
            "decode": {
                "num_beams": int(cfg.num_beams),
                "temperature": float(cfg.temperature),
                "max_new_tokens": int(cfg.max_new_tokens),
                "language": cfg.language,
                "task": cfg.task,
            },
        }, per_item

    # Prepare normalised lists for micro metrics
    if cfg.normalize_mode in {"whisper_basic", "basic"}:
        preds_n = [_basic_whisperish_normalize(p) for p in preds_raw]
        trefs_n = [_basic_whisperish_normalize(r) for r in trefs_raw]
        orefs_n = [_basic_whisperish_normalize(r) for r in orefs_raw]
    else:
        preds_n, trefs_n, orefs_n = preds_raw, trefs_raw, orefs_raw

    wer_micro_t = float(jiwer.wer(trefs_n, preds_n))
    cer_micro_t = float(jiwer.cer(trefs_n, preds_n))
    wer_micro_o = float(jiwer.wer(orefs_n, preds_n))
    cer_micro_o = float(jiwer.cer(orefs_n, preds_n))

    wer_macro_t = float(np.mean(per_utt_wer_t)) if per_utt_wer_t else None
    wer_macro_o = float(np.mean(per_utt_wer_o)) if per_utt_wer_o else None

    win_rate = float(wins) / float(n)
    avg_margin = float(np.mean(np.array(per_utt_wer_o) - np.array(per_utt_wer_t)))

    metrics = {
        "samples": n,
        "skipped": skipped,

        # Backwards-compatible keys (assume "primary" is TARGET)
        "wer_micro": wer_micro_t,
        "cer_micro": cer_micro_t,
        "wer_macro": float(wer_macro_t) if wer_macro_t is not None else None,

        # New keys
        "wer_micro_target": wer_micro_t,
        "cer_micro_target": cer_micro_t,
        "wer_macro_target": float(wer_macro_t) if wer_macro_t is not None else None,

        "wer_micro_other": wer_micro_o,
        "cer_micro_other": cer_micro_o,
        "wer_macro_other": float(wer_macro_o) if wer_macro_o is not None else None,

        "win_rate_target_closer": win_rate,
        "avg_margin_other_minus_target": avg_margin,

        "wer_by_duration": _wer_duration_buckets(per_item),
        "wer_by_duration_target": _wer_duration_buckets(per_item),
        "wer_by_duration_other": _wer_duration_buckets_other(per_item),

        "normalize_mode": cfg.normalize_mode,
        "dynamic_audio_ctx": bool(cfg.dynamic_audio_ctx),
        "vad": cfg.vad.__dict__,
        "mixing": {
            "snr_db": cfg.snr_db,
            "other_offset_mode": cfg.other_offset_mode,
            "other_peak_ratio": cfg.other_peak_ratio,
            "pairing_mode": cfg.pairing_mode,
            "mix_per_target": cfg.mix_per_target,
            "seed": cfg.seed,
        },
        "decode": {
            "num_beams": int(cfg.num_beams),
            "temperature": float(cfg.temperature),
            "max_new_tokens": int(cfg.max_new_tokens),
            "language": cfg.language,
            "task": cfg.task,
        },
    }

    return metrics, per_item


# ----------------------------
# Main eval
# ----------------------------

def eval_one_model(
    model_id_or_path: str,
    pair_rows: List[dict],
    processor: WhisperProcessor,
    cfg: EvalConfig,
    vad_trimmer: Optional[SileroVADTrimmer],
    all_predictions: dict,
    out_json: Path,
) -> Tuple[Dict, List[dict]]:

    model = WhisperForConditionalGeneration.from_pretrained(model_id_or_path)
    model.to(cfg.device)
    model.eval()

    # Set Whisper prompt (language/task) via forced_decoder_ids if available
    try:
        fids = processor.get_decoder_prompt_ids(language=cfg.language, task=cfg.task)
        if hasattr(model, "generation_config") and hasattr(model.generation_config, "forced_decoder_ids"):
            model.generation_config.forced_decoder_ids = fids
    except Exception:
        pass

    gen_kwargs = build_generate_kwargs(cfg)

    preds_raw: List[str] = []
    trefs_raw: List[str] = []
    orefs_raw: List[str] = []

    per_item: List[dict] = []
    skipped: List[dict] = []

    per_utt_wer_t: List[float] = []
    per_utt_wer_o: List[float] = []
    wins = 0

    if cfg.vad.enabled and vad_trimmer is None:
        vad_trimmer = SileroVADTrimmer()

    model_name = Path(model_id_or_path).name if Path(model_id_or_path).exists() else model_id_or_path

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

            # Skip already-evaluated pairs for this model (resume)
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
                t_audio, sr_t = load_audio_mono_16k(t_ap)
                o_audio, sr_o = load_audio_mono_16k(o_ap)
                if sr_t != 16000 or sr_o != 16000:
                    skipped.append({"mix_key": key, "reason": "sr_not_16k", "target_sr": sr_t, "other_sr": sr_o})
                    continue

                dur_t = seconds_from_audio(t_audio, sr_t)
                dur_o = seconds_from_audio(o_audio, sr_o)

                # Mix (per-pair deterministic RNG so resume doesn't change offsets)
                rng_local = np.random.default_rng(_stable_uint64_from_str(key, cfg.seed))
                mixed, mix_meta = mix_target_with_other(
                    target=t_audio,
                    other=o_audio,
                    snr_db_target_over_other=cfg.snr_db,
                    other_offset_mode=cfg.other_offset_mode,
                    rng=rng_local,
                    other_peak_ratio=cfg.other_peak_ratio,
                )

                # VAD trim on the mixture (keyboard-like endpointing)
                vad_info: Dict = {"vad_applied": False}
                audio_eval = mixed
                if cfg.vad.enabled and vad_trimmer is not None:
                    audio_vad, vad_info = vad_trimmer.trim(mixed, 16000, cfg.vad)
                    if len(audio_vad) == 0:
                        if cfg.vad.policy == "skip":
                            skipped.append({"mix_key": key, "reason": "vad_no_speech"})
                            continue
                        if cfg.vad.policy == "empty":
                            # No inference; force empty prediction
                            pred = ""
                            tref = (item.get("target_ref") or "").strip()
                            oref = (item.get("other_ref") or "").strip()

                            preds_raw.append(pred)
                            trefs_raw.append(tref)
                            orefs_raw.append(oref)

                            if cfg.normalize_mode in {"whisper_basic", "basic"}:
                                pred_n = _basic_whisperish_normalize(pred)
                                tref_n = _basic_whisperish_normalize(tref)
                                oref_n = _basic_whisperish_normalize(oref)
                            else:
                                pred_n, tref_n, oref_n = pred, tref, oref

                            wt = float(jiwer.wer(tref_n, pred_n))
                            wo = float(jiwer.wer(oref_n, pred_n))
                            per_utt_wer_t.append(wt)
                            per_utt_wer_o.append(wo)
                            if wt < wo:
                                wins += 1

                            per_item.append(
                                {
                                    "mix_key": key,
                                    "target_audio_path": str(t_ap),
                                    "other_audio_path": str(o_ap),
                                    "duration_sec_target": dur_t,
                                    "duration_sec_other": dur_o,
                                    "duration_sec_eval": 0.0,
                                    "target_ref": tref,
                                    "other_ref": oref,
                                    "pred": pred,
                                    "target_ref_norm": tref_n,
                                    "other_ref_norm": oref_n,
                                    "pred_norm": pred_n,
                                    "wer_target": wt,
                                    "wer_other": wo,
                                    "win_target_closer": bool(wt < wo),
                                    "vad": vad_info,
                                    "mix": mix_meta,
                                }
                            )

                            # Save prediction
                            all_predictions.setdefault(
                                key,
                                {
                                    "mix_key": key,
                                    "target_audio_path": str(t_ap),
                                    "other_audio_path": str(o_ap),
                                    "target_reference": tref,
                                    "other_reference": oref,
                                    "predictions": {},
                                },
                            )
                            all_predictions[key]["predictions"][model_name] = {
                                "pred": pred,
                                "pred_norm": pred_n,
                                "duration_sec_eval": 0.0,
                                "vad": vad_info,
                                "mix": mix_meta,
                            }
                            continue

                        # keep mixture
                        audio_eval = mixed
                    else:
                        audio_eval = audio_vad

                dur_eval = seconds_from_audio(audio_eval, 16000)

                # Feature extraction
                inputs = processor(audio_eval, sampling_rate=16000, return_tensors="pt")
                input_features = inputs["input_features"]  # (1,80,T)

                if cfg.dynamic_audio_ctx:
                    input_features = crop_input_features_for_duration(input_features, dur_eval)

                batch_buf.append(
                    {
                        "mix_key": key,
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
                    }
                )

            except Exception as e:
                skipped.append({"mix_key": key, "reason": f"exception: {type(e).__name__}: {e}"})
                continue

            if len(batch_buf) < cur_bs:
                continue

            pending = batch_buf
            batch_buf = []

            while pending:
                chunk = pending[:cur_bs]

                try:
                    feats = [c["input_features"] for c in chunk]
                    batch_feats = _pad_stack_input_features(feats)  # (B,80,Tmax)

                    batch_feats = batch_feats.to(cfg.device, non_blocking=True)
                    if use_cuda and cfg.fp16:
                        batch_feats = batch_feats.half()

                    generated_ids = model.generate(input_features=batch_feats, **gen_kwargs)
                    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                    for c, text_out in zip(chunk, texts):
                        pred = (text_out or "").strip()
                        tref = (c.get("target_ref") or "").strip()
                        oref = (c.get("other_ref") or "").strip()

                        preds_raw.append(pred)
                        trefs_raw.append(tref)
                        orefs_raw.append(oref)

                        if cfg.normalize_mode in {"whisper_basic", "basic"}:
                            pred_n = _basic_whisperish_normalize(pred)
                            tref_n = _basic_whisperish_normalize(tref)
                            oref_n = _basic_whisperish_normalize(oref)
                        else:
                            pred_n, tref_n, oref_n = pred, tref, oref

                        wt = float(jiwer.wer(tref_n, pred_n))
                        wo = float(jiwer.wer(oref_n, pred_n))
                        per_utt_wer_t.append(wt)
                        per_utt_wer_o.append(wo)
                        if wt < wo:
                            wins += 1

                        per_item.append(
                            {
                                "mix_key": c["mix_key"],
                                "target_audio_path": c["target_audio_path"],
                                "other_audio_path": c["other_audio_path"],
                                "duration_sec_target": c["duration_sec_target"],
                                "duration_sec_other": c["duration_sec_other"],
                                "duration_sec_eval": c["duration_sec_eval"],
                                "target_ref": tref,
                                "other_ref": oref,
                                "pred": pred,
                                "target_ref_norm": tref_n,
                                "other_ref_norm": oref_n,
                                "pred_norm": pred_n,
                                "wer_target": wt,
                                "wer_other": wo,
                                "win_target_closer": bool(wt < wo),
                                "vad": c.get("vad", {"vad_applied": False}),
                                "mix": c.get("mix", {}),
                            }
                        )

                        # Persist prediction for resume/recalc
                        all_predictions.setdefault(
                            c["mix_key"],
                            {
                                "mix_key": c["mix_key"],
                                "target_audio_path": c["target_audio_path"],
                                "other_audio_path": c["other_audio_path"],
                                "target_reference": tref,
                                "other_reference": oref,
                                "predictions": {},
                            },
                        )
                        all_predictions[c["mix_key"]]["predictions"][model_name] = {
                            "pred": pred,
                            "pred_norm": pred_n,
                            "duration_sec_eval": c.get("duration_sec_eval"),
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

                    # Periodic cleanup
                    batch_count += 1
                    if batch_count % max(1, int(cfg.cleanup_interval)) == 0:
                        if use_cuda:
                            torch.cuda.empty_cache()
                        gc.collect()

                except torch.cuda.OutOfMemoryError:
                    if not use_cuda:
                        raise
                    # Backoff
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
            batch_buf = []
            while pending:
                chunk = pending[:cur_bs]
                feats = [c["input_features"] for c in chunk]
                batch_feats = _pad_stack_input_features(feats).to(cfg.device, non_blocking=True)
                if use_cuda and cfg.fp16:
                    batch_feats = batch_feats.half()
                generated_ids = model.generate(input_features=batch_feats, **gen_kwargs)
                texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                for c, text_out in zip(chunk, texts):
                    pred = (text_out or "").strip()
                    tref = (c.get("target_ref") or "").strip()
                    oref = (c.get("other_ref") or "").strip()

                    preds_raw.append(pred)
                    trefs_raw.append(tref)
                    orefs_raw.append(oref)

                    if cfg.normalize_mode in {"whisper_basic", "basic"}:
                        pred_n = _basic_whisperish_normalize(pred)
                        tref_n = _basic_whisperish_normalize(tref)
                        oref_n = _basic_whisperish_normalize(oref)
                    else:
                        pred_n, tref_n, oref_n = pred, tref, oref

                    wt = float(jiwer.wer(tref_n, pred_n))
                    wo = float(jiwer.wer(oref_n, pred_n))
                    per_utt_wer_t.append(wt)
                    per_utt_wer_o.append(wo)
                    if wt < wo:
                        wins += 1

                    per_item.append(
                        {
                            "mix_key": c["mix_key"],
                            "target_audio_path": c["target_audio_path"],
                            "other_audio_path": c["other_audio_path"],
                            "duration_sec_target": c["duration_sec_target"],
                            "duration_sec_other": c["duration_sec_other"],
                            "duration_sec_eval": c["duration_sec_eval"],
                            "target_ref": tref,
                            "other_ref": oref,
                            "pred": pred,
                            "target_ref_norm": tref_n,
                            "other_ref_norm": oref_n,
                            "pred_norm": pred_n,
                            "wer_target": wt,
                            "wer_other": wo,
                            "win_target_closer": bool(wt < wo),
                            "vad": c.get("vad", {"vad_applied": False}),
                            "mix": c.get("mix", {}),
                        }
                    )

                    all_predictions.setdefault(
                        c["mix_key"],
                        {
                            "mix_key": c["mix_key"],
                            "target_audio_path": c["target_audio_path"],
                            "other_audio_path": c["other_audio_path"],
                            "target_reference": tref,
                            "other_reference": oref,
                            "predictions": {},
                        },
                    )
                    all_predictions[c["mix_key"]]["predictions"][model_name] = {
                        "pred": pred,
                        "pred_norm": pred_n,
                        "duration_sec_eval": c.get("duration_sec_eval"),
                        "vad": c.get("vad", {"vad_applied": False}),
                        "mix": c.get("mix", {}),
                    }

                pending = pending[cur_bs:]

    # Micro metrics
    if cfg.normalize_mode in {"whisper_basic", "basic"}:
        preds_n = [_basic_whisperish_normalize(p) for p in preds_raw]
        trefs_n = [_basic_whisperish_normalize(r) for r in trefs_raw]
        orefs_n = [_basic_whisperish_normalize(r) for r in orefs_raw]
    else:
        preds_n, trefs_n, orefs_n = preds_raw, trefs_raw, orefs_raw

    if len(preds_n) == 0:
        metrics = {
            "samples": 0,
            "skipped": len(skipped),
            "wer_micro": None,
            "cer_micro": None,
            "wer_macro": None,
            "wer_micro_target": None,
            "wer_micro_other": None,
            "cer_micro_target": None,
            "cer_micro_other": None,
            "wer_macro_target": None,
            "wer_macro_other": None,
            "win_rate_target_closer": None,
            "avg_margin_other_minus_target": None,
            "wer_by_duration": {"0-1s": None, "1-2s": None, "2-5s": None, "5-10s": None, "10-30s": None},
            "wer_by_duration_target": {"0-1s": None, "1-2s": None, "2-5s": None, "5-10s": None, "10-30s": None},
            "wer_by_duration_other": {"0-1s": None, "1-2s": None, "2-5s": None, "5-10s": None, "10-30s": None},
        }
        return metrics, per_item

    wer_micro_t = float(jiwer.wer(trefs_n, preds_n))
    cer_micro_t = float(jiwer.cer(trefs_n, preds_n))
    wer_micro_o = float(jiwer.wer(orefs_n, preds_n))
    cer_micro_o = float(jiwer.cer(orefs_n, preds_n))

    wer_macro_t = float(np.mean(per_utt_wer_t)) if per_utt_wer_t else None
    wer_macro_o = float(np.mean(per_utt_wer_o)) if per_utt_wer_o else None

    win_rate = float(wins) / float(len(per_item)) if per_item else None
    avg_margin = float(np.mean(np.array(per_utt_wer_o) - np.array(per_utt_wer_t))) if per_item else None

    metrics = {
        "samples": len(per_item),
        "skipped": len(skipped),

        # Backwards-compatible (primary=TARGET)
        "wer_micro": wer_micro_t,
        "cer_micro": cer_micro_t,
        "wer_macro": float(wer_macro_t) if wer_macro_t is not None else None,

        # New breakdown
        "wer_micro_target": wer_micro_t,
        "cer_micro_target": cer_micro_t,
        "wer_macro_target": float(wer_macro_t) if wer_macro_t is not None else None,

        "wer_micro_other": wer_micro_o,
        "cer_micro_other": cer_micro_o,
        "wer_macro_other": float(wer_macro_o) if wer_macro_o is not None else None,

        "win_rate_target_closer": win_rate,
        "avg_margin_other_minus_target": avg_margin,

        "wer_by_duration": _wer_duration_buckets(per_item),
        "wer_by_duration_target": _wer_duration_buckets(per_item),
        "wer_by_duration_other": _wer_duration_buckets_other(per_item),

        "normalize_mode": cfg.normalize_mode,
        "dynamic_audio_ctx": bool(cfg.dynamic_audio_ctx),
        "vad": cfg.vad.__dict__,
        "mixing": {
            "snr_db": cfg.snr_db,
            "other_offset_mode": cfg.other_offset_mode,
            "other_peak_ratio": cfg.other_peak_ratio,
            "pairing_mode": cfg.pairing_mode,
            "mix_per_target": cfg.mix_per_target,
            "seed": cfg.seed,
        },
        "decode": {
            "num_beams": int(cfg.num_beams),
            "temperature": float(cfg.temperature),
            "max_new_tokens": int(cfg.max_new_tokens),
            "language": cfg.language,
            "task": cfg.task,
        },
    }

    return metrics, per_item


# ----------------------------
# Beep
# ----------------------------

def beep() -> None:
    try:
        import winsound  # type: ignore
        winsound.MessageBeep()
    except Exception:
        try:
            print("\a", end="", flush=True)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_manifest", required=True, type=Path)
    ap.add_argument("--speaker_scores_csv", required=True, type=Path)
    ap.add_argument("--checkpoint_dir", required=True, type=Path)
    ap.add_argument("--percentage", type=float, default=100.0)

    ap.add_argument("--base_model", default="futo-org/acft-whisper-tiny.en")
    ap.add_argument("--compare_openai_tiny", action="store_true", help="Also evaluate openai/whisper-tiny.en")
    ap.add_argument("--base_processor_id", default="openai/whisper-tiny.en", help="Processor/tokenizer ID")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="transcribe")

    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)

    # Inference batching
    ap.add_argument("--batch_size", type=int, default=0, help="0=auto default (GPU:8, CPU:1)")
    ap.add_argument("--auto_batch", type=int, default=1, help="1=auto-adjust batch size to use more VRAM safely; 0=static")
    ap.add_argument("--batch_min", type=int, default=1)
    ap.add_argument("--batch_max", type=int, default=64)

    ap.add_argument("--fp16", type=int, default=1, help="1=use float16 model+inputs on CUDA; 0=float32")
    ap.add_argument("--mem_low", type=float, default=0.60)
    ap.add_argument("--mem_high", type=float, default=0.88)
    ap.add_argument("--cleanup_interval", type=int, default=50)

    ap.add_argument("--dynamic_audio_ctx", type=int, default=1, help="1=enable mel cropping by duration; 0=disable")
    ap.add_argument("--normalize", default="whisper_basic", choices=["whisper_basic", "none"])

    # VAD
    ap.add_argument("--vad_filter", type=int, default=1)
    ap.add_argument("--vad_policy", default="skip", choices=["skip", "keep", "empty"])
    ap.add_argument("--vad_threshold", type=float, default=0.5)
    ap.add_argument("--vad_min_speech_ms", type=int, default=250)
    ap.add_argument("--vad_min_silence_ms", type=int, default=100)
    ap.add_argument("--vad_speech_pad_ms", type=int, default=200)

    # Mixing
    ap.add_argument("--mix_per_target", type=int, default=1)
    ap.add_argument("--snr_db", type=float, default=10.0, help="TARGET over OTHER, in dB (RMS)")
    ap.add_argument("--other_offset_mode", default="start", choices=["start", "random"])
    ap.add_argument("--other_peak_ratio", type=float, default=1.0, help="Peak(OTHER_scaled) <= Peak(TARGET)*ratio")

    ap.add_argument("--pairing_mode", default="round_robin", choices=["round_robin", "random"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force_resume", action="store_true")

    args = ap.parse_args()

    if args.batch_size <= 0:
        args.batch_size = 8 if str(args.device).startswith("cuda") and torch.cuda.is_available() else 1

    if not args.test_manifest.exists():
        raise FileNotFoundError(args.test_manifest)
    if not args.speaker_scores_csv.exists():
        raise FileNotFoundError(args.speaker_scores_csv)
    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(args.checkpoint_dir)
    if not (0.0 <= args.percentage <= 100.0):
        raise ValueError("--percentage must be 0..100")

    rows = load_jsonl(args.test_manifest)
    labels = load_speaker_sort_scores(args.speaker_scores_csv)

    pair_rows, pair_info = build_target_other_pairs(
        rows=rows,
        labels=labels,
        mix_per_target=int(args.mix_per_target),
        pairing_mode=str(args.pairing_mode),
        seed=int(args.seed),
    )

    # Subset pairs if requested
    if args.percentage < 100.0:
        rng = random.Random(int(args.seed))
        k = max(1, int(len(pair_rows) * (args.percentage / 100.0)))
        pair_rows = rng.sample(pair_rows, k)

    # Checkpoints
    checkpoints = list(args.checkpoint_dir.glob("model_epoch_*"))
    checkpoints.sort(key=lambda p: int(p.name.split("_")[2]) if p.name.split("_")[2].isdigit() else 0)

    models: List[str] = []
    if args.compare_openai_tiny:
        models.append("openai/whisper-tiny.en")
    models.append(str(args.base_model))
    models.extend([str(p) for p in checkpoints])

    print(f"Device: {args.device}")
    print(f"Test manifest rows: {len(rows)}")
    print(f"Pair build: {pair_info}")
    print(f"Eval pairs: {len(pair_rows)}")
    print(f"Models to eval: {len(models)}")

    processor = WhisperProcessor.from_pretrained(args.base_processor_id)

    vad_cfg = VADConfig(
        enabled=bool(args.vad_filter),
        policy=str(args.vad_policy),
        threshold=float(args.vad_threshold),
        min_speech_duration_ms=int(args.vad_min_speech_ms),
        min_silence_duration_ms=int(args.vad_min_silence_ms),
        speech_pad_ms=int(args.vad_speech_pad_ms),
    )

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
        snr_db=float(args.snr_db),
        other_offset_mode=str(args.other_offset_mode),
        other_peak_ratio=float(args.other_peak_ratio),
        pairing_mode=str(args.pairing_mode),
        seed=int(args.seed),
    )

    out_json = args.out_json
    if out_json is None:
        out_json = args.checkpoint_dir / "evaluation_results_futo_like_targetmix.json"

    results = {
        "test_manifest": str(args.test_manifest),
        "speaker_scores_csv": str(args.speaker_scores_csv),
        "checkpoint_dir": str(args.checkpoint_dir),
        "percentage": float(args.percentage),
        "pair_info": pair_info,
        "models": [],
        "cfg": {
            **cfg.__dict__,
            "vad": vad_cfg.__dict__,
        },
    }
    all_predictions: Dict[str, dict] = {}

    if args.resume or args.force_resume:
        existing_results, existing_predictions = load_existing_results(out_json)
        if existing_results:
            results = existing_results
            all_predictions = existing_predictions

    evaluated_models = {m["model"] for m in results.get("models", [])}
    print(f"Already evaluated models: {len(evaluated_models)}")

    # Ensure prediction entries exist for every pair (so force_resume can work cleanly)
    for pr in pair_rows:
        key = pr["mix_key"]
        if key not in all_predictions:
            all_predictions[key] = {
                "mix_key": key,
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

        print("\n" + "=" * 70)
        print(f"Evaluating: {m}")
        print("=" * 70)

        model_name = Path(m).name if Path(m).exists() else m

        if args.force_resume and all_predictions:
            has_predictions = any(model_name in v.get("predictions", {}) for v in all_predictions.values())
            if has_predictions:
                print(f"📊 Recalculating metrics from existing predictions for {model_name}")
                metrics, _ = recalculate_metrics_from_predictions(all_predictions, model_name, cfg)
                results["models"].append({"model": m, "metrics": metrics})
                save_incremental_results(results, all_predictions, out_json)
                continue

        metrics, _ = eval_one_model(
            model_id_or_path=m,
            pair_rows=pair_rows,
            processor=processor,
            cfg=cfg,
            vad_trimmer=vad_trimmer,
            all_predictions=all_predictions,
            out_json=out_json,
        )
        results["models"].append({"model": m, "metrics": metrics})
        save_incremental_results(results, all_predictions, out_json)

        print(f"samples={metrics.get('samples')} skipped={metrics.get('skipped')}")
        print(f"WER target micro={metrics.get('wer_micro_target')} | WER other micro={metrics.get('wer_micro_other')}")
        print(f"win_rate(target closer)={metrics.get('win_rate_target_closer')} avg_margin(other-target)={metrics.get('avg_margin_other_minus_target')}")

    print("\nDone.")
    beep()


if __name__ == "__main__":
    main()
