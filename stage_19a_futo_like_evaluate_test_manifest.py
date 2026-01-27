# Drop-in replacement for stage_19a_futo_like_evaluate_test_manifest.py
# Adds: batched inference + auto batch-size tuning (backoff on OOM) + optional fp16 on CUDA.
#
# Save as: stage_19a_futo_like_evaluate_test_manifest_dynamic_batch.py
#
# Example:
#   python stage_19a_futo_like_evaluate_test_manifest_dynamic_batch.py \
#     --test_manifest "I:\\Record_chunks\\pairs_manifest_sorted_by_speaker_scores_test.jsonl" \
#     --checkpoint_dir "I:\\Dynamic_n_ctx_checkpoints_partialctx3" \
#     --base_model "futo-org/acft-whisper-tiny.en" \
#     --device cuda \
#     --num_beams 5 \
#     --fp16 1 \
#     --batch_size 16 \
#     --auto_batch 1 \
#     --batch_max 64

r"""stage_19a_futo_like_evaluate_test_manifest.py

Evaluate Whisper checkpoints in a way that's closer to keyboard voice-input usage:
- short dictations
- (optional) dynamic audio context emulation (crop mel frames by duration)
- (optional) VAD trim (Silero) to remove leading/trailing silence (endpointing-like)
- consistent decode config across checkpoints
- micro-WER (corpus) + macro-WER (mean per-utterance)
- duration-bucket breakdown

Example:
# Force resume using existing predictions
python stage_19a_futo_like_evaluate_test_manifest.py \
  --test_manifest "I:\Record_chunks\pairs_manifest_local_english_only_filtered_with_mix_and_others_voices_mixed_aug_gain_aug_rir_real_randomized_bottom_filtered_test.jsonl" \
  --checkpoint_dir "I:\Stage_2_shuffle_Dynamic_n_ctx_checkpoints_partialctx6"\
  --base_model "futo-org/acft-whisper-tiny.en" \
  --percentage 10 --device cuda --num_beams 5 --temperature 0.0 \
  --normalize whisper_basic --dynamic_audio_ctx 0 --vad_filter 0 \
  --vad_policy skip --fp16 1 --batch_size 16 --auto_batch 1 \
  --force_resume

Notes:
- VAD trim is FIRST speech to LAST speech (+pad). We do NOT split into multiple chunks.
- If Silero VAD can't be downloaded/loaded (offline), we fall back to "keep audio" and continue.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
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
            # CPU is recommended.
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
        # Only support 16k here.
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
        except TypeError:
            # Some older Silero versions use slightly different kwarg names.
            speech_ts = self._get_speech_timestamps(
                wav,
                self._model,
                sampling_rate=sr,
                threshold=float(cfg.threshold),
            )

        if not speech_ts:
            info = {"vad_applied": True, "speech_found": False, "speech_segments": 0}
            if cfg.policy == "keep":
                return audio_16k, info
            if cfg.policy == "empty":
                return np.zeros((0,), dtype=np.float32), info
            # skip
            return np.zeros((0,), dtype=np.float32), info

        start = int(speech_ts[0].get("start", 0))
        end = int(speech_ts[-1].get("end", len(audio_16k)))
        start = max(0, start)
        end = min(len(audio_16k), end)

        trimmed = audio_16k[start:end].astype(np.float32, copy=False)
        info = {
            "vad_applied": True,
            "speech_found": True,
            "speech_segments": int(len(speech_ts)),
            "trim_start_sample": int(start),
            "trim_end_sample": int(end),
        }
        return trimmed, info


# ----------------------------
# Evaluation
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

    # Inference batching (evaluation speed)
    batch_size: int
    auto_batch: bool
    batch_min: int
    batch_max: int

    # Memory / dtype knobs
    fp16: bool
    mem_low: float   # if below this ratio, try increasing batch size
    mem_high: float  # if above this ratio, try decreasing batch size
    
    # Periodic cleanup to prevent memory fragmentation
    cleanup_interval: int  # cleanup every N batches


def build_generate_kwargs(cfg: EvalConfig) -> Dict:
    # Keep kwargs minimal/robust across checkpoint configs.
    return {
        "num_beams": int(cfg.num_beams),
        "temperature": float(cfg.temperature),
        "do_sample": False,
        "max_new_tokens": int(cfg.max_new_tokens),
    }


# ----------------------------
# Batched generation helpers (dynamic batch size)
# ----------------------------

def _cuda_mem_ratio() -> float:
    """Return current CUDA memory allocated / total, or 0.0 if not on CUDA."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        dev = torch.cuda.current_device()
        total = torch.cuda.get_device_properties(dev).total_memory
        alloc = torch.cuda.memory_allocated(dev)
        return float(alloc) / float(total) if total else 0.0
    except Exception:
        return 0.0


def _ensure_80_first(feat_2d: torch.Tensor) -> torch.Tensor:
    """Whisper mels should be (80, T). If we see (T, 80) we transpose."""
    if feat_2d.ndim != 2:
        return feat_2d
    if feat_2d.shape[0] == 80:
        return feat_2d
    if feat_2d.shape[1] == 80:
        return feat_2d.transpose(0, 1)
    return feat_2d


def _pad_stack_input_features(feats_3d: List[torch.Tensor]) -> torch.Tensor:
    """Take a list of (1,80,T) or (80,T) tensors and return (B,80,Tmax)."""
    feats_2d: List[torch.Tensor] = []
    tmax = 1
    for f in feats_3d:
        if f.ndim == 3:
            f2 = f[0]
        else:
            f2 = f
        f2 = _ensure_80_first(f2)
        feats_2d.append(f2)
        if f2.ndim == 2:
            tmax = max(tmax, int(f2.shape[1]))

    padded: List[torch.Tensor] = []
    for f2 in feats_2d:
        if f2.ndim != 2:
            padded.append(f2)
            continue
        pad_t = max(0, tmax - int(f2.shape[1]))
        if pad_t:
            f2 = F.pad(f2, (0, pad_t))  # pad time dim on the right
        padded.append(f2)

    return torch.stack(padded, dim=0)  # (B,80,T)


def _adjust_batch_size(
    cur_bs: int,
    avg_dur: float,
    mem_ratio: float,
    cfg: EvalConfig,
    oom_cooldown: int,
) -> Tuple[int, int]:
    """Heuristic: nudge batch size up/down to keep VRAM hot but safe."""
    cur_bs = int(max(cfg.batch_min, min(cfg.batch_max, cur_bs)))

    if not cfg.auto_batch:
        return cur_bs, max(0, oom_cooldown - 1)

    # Cooldown prevents immediate re-growth after an OOM.
    if oom_cooldown > 0:
        return cur_bs, oom_cooldown - 1

    new_bs = cur_bs

    # If we are comfortably under memory, try to grow.
    if mem_ratio > 0.0 and mem_ratio < float(cfg.mem_low):
        new_bs = min(cfg.batch_max, cur_bs + 2)

    # If we are close to the cliff, shrink.
    if mem_ratio > float(cfg.mem_high):
        new_bs = max(cfg.batch_min, max(1, int(cur_bs * 0.7)))

    # Duration heuristic (longer utterances cost more encoder memory when dynamic_audio_ctx is on)
    if avg_dur > 15.0:
        new_bs = max(cfg.batch_min, max(1, int(new_bs * 0.7)))
    elif avg_dur < 6.0 and mem_ratio < float(cfg.mem_low):
        new_bs = min(cfg.batch_max, new_bs + 1)

    return int(max(cfg.batch_min, min(cfg.batch_max, new_bs))), 0


def eval_one_model(
    model_id_or_path: str,
    test_rows: List[dict],
    processor: WhisperProcessor,
    cfg: EvalConfig,
    vad_trimmer: Optional[SileroVADTrimmer] = None,
    results: Optional[dict] = None,
    all_predictions: Optional[dict] = None,
    out_json: Optional[Path] = None,
) -> Tuple[Dict, List[dict]]:

    model = WhisperForConditionalGeneration.from_pretrained(model_id_or_path)
    model.to(cfg.device)
    model.eval()

    # Set Whisper prompt via generation_config (more robust than passing forced_decoder_ids to generate()).
    try:
        fids = processor.get_decoder_prompt_ids(language=cfg.language, task=cfg.task)
        if hasattr(model, "generation_config") and hasattr(model.generation_config, "forced_decoder_ids"):
            model.generation_config.forced_decoder_ids = fids
    except Exception:
        pass

    gen_kwargs = build_generate_kwargs(cfg)

    preds_raw: List[str] = []
    refs_raw: List[str] = []
    per_item: List[dict] = []
    skipped: List[dict] = []
    per_utt_wer: List[float] = []

    if cfg.vad.enabled and vad_trimmer is None:
        vad_trimmer = SileroVADTrimmer()

    with torch.inference_mode():
        use_cuda = bool(str(cfg.device).startswith("cuda") and torch.cuda.is_available())
        cur_bs = int(max(cfg.batch_min, min(cfg.batch_max, cfg.batch_size)))
        oom_cooldown = 0

        # Optional half precision to allow larger batches on GPU
        if use_cuda and cfg.fp16:
            try:
                model.half()
            except Exception:
                pass

        batch_buf: List[dict] = []
        batch_count = 0  # Track batches for periodic cleanup

        pbar = tqdm(test_rows, desc=f"eval {Path(model_id_or_path).name}")
        for item in pbar:
            ap = Path(item.get("audio_path", ""))
            if not ap.exists():
                skipped.append({"audio_path": str(ap), "reason": "missing_file"})
                continue

            try:
                audio, sr = load_audio_mono_16k(ap)
                dur_raw = seconds_from_audio(audio, sr)

                # VAD trim (endpointing-like)
                vad_info: Dict = {"vad_applied": False}
                audio_eval = audio
                if cfg.vad.enabled and vad_trimmer is not None:
                    audio_vad, vad_info = vad_trimmer.trim(audio, sr, cfg.vad)
                    if len(audio_vad) == 0:
                        # no speech found
                        if cfg.vad.policy == "skip":
                            skipped.append({"audio_path": str(ap), "reason": "vad_no_speech"})
                            continue
                        if cfg.vad.policy == "empty":
                            pred = ""
                            ref = (item.get("raw_transcription") or "").strip()

                            preds_raw.append(pred)
                            refs_raw.append(ref)

                            if cfg.normalize_mode in {"whisper_basic", "basic"}:
                                pred_n = _basic_whisperish_normalize(pred)
                                ref_n = _basic_whisperish_normalize(ref)
                            else:
                                pred_n = pred
                                ref_n = ref

                            per_utt_wer.append(jiwer.wer(ref_n, pred_n))

                            per_item.append(
                                {
                                    "audio_path": str(ap),
                                    "duration_sec_raw": dur_raw,
                                    "duration_sec_eval": 0.0,
                                    "ref": ref,
                                    "pred": pred,
                                    "ref_norm": ref_n,
                                    "pred_norm": pred_n,
                                    "vad": vad_info,
                                }
                            )
                            continue

                        # keep
                        audio_eval = audio
                    else:
                        audio_eval = audio_vad

                dur_eval = seconds_from_audio(audio_eval, sr)

                # Feature extraction (CPU for now; we batch-move to GPU later)
                inputs = processor(audio_eval, sampling_rate=sr, return_tensors="pt")
                input_features = inputs["input_features"]  # (1,80,T)

                if cfg.dynamic_audio_ctx:
                    input_features = crop_input_features_for_duration(input_features, dur_eval)

                batch_buf.append(
                    {
                        "audio_path": str(ap),
                        "duration_sec_raw": dur_raw,
                        "duration_sec_eval": dur_eval,
                        "ref": (item.get("raw_transcription") or "").strip(),
                        "vad": vad_info,
                        "input_features": input_features,
                    }
                )

            except Exception as e:
                skipped.append({"audio_path": str(ap), "reason": f"exception: {type(e).__name__}: {e}"})
                continue

            # Flush when buffer reaches batch size
            if len(batch_buf) < cur_bs:
                continue

            pending = batch_buf
            batch_buf = []

            while pending:
                chunk = pending[:cur_bs]

                try:
                    feats = [c["input_features"] for c in chunk]
                    batch_feats = _pad_stack_input_features(feats)  # (B,80,Tmax) on CPU

                    # Move to device
                    batch_feats = batch_feats.to(cfg.device, non_blocking=True)
                    if use_cuda and cfg.fp16:
                        batch_feats = batch_feats.half()

                    # Generate
                    try:
                        generated_ids = model.generate(input_features=batch_feats, **gen_kwargs)
                    except ValueError as e:
                        # Some configs can be strict; retry bare minimum.
                        msg = str(e)
                        if any(k in msg for k in ("forced_decoder_ids", "return_timestamps", "language", "task")):
                            generated_ids = model.generate(input_features=batch_feats)
                        else:
                            raise

                    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                    # Record per-item outputs
                    for c, text_out in zip(chunk, texts):
                        pred = (text_out or "").strip()
                        ref = (c.get("ref") or "").strip()

                        preds_raw.append(pred)
                        refs_raw.append(ref)

                        if cfg.normalize_mode in {"whisper_basic", "basic"}:
                            pred_n = _basic_whisperish_normalize(pred)
                            ref_n = _basic_whisperish_normalize(ref)
                        else:
                            pred_n = pred
                            ref_n = ref

                        per_utt_wer.append(jiwer.wer(ref_n, pred_n))

                        per_item.append(
                            {
                                "audio_path": c["audio_path"],
                                "duration_sec_raw": c["duration_sec_raw"],
                                "duration_sec_eval": c["duration_sec_eval"],
                                "ref": ref,
                                "pred": pred,
                                "ref_norm": ref_n,
                                "pred_norm": pred_n,
                                "vad": c.get("vad", {"vad_applied": False}),
                            }
                        )
                        
                        # Immediately save each prediction to prevent data loss
                        model_name = Path(model_id_or_path).name if Path(model_id_or_path).exists() else model_id_or_path
                        audio_path = c["audio_path"]
                        if audio_path not in all_predictions:
                            all_predictions[audio_path] = {
                                "audio_path": audio_path,
                                "reference": ref,
                                "predictions": {},
                            }
                        all_predictions[audio_path]["predictions"][model_name] = {
                            "pred": pred,
                            "pred_norm": pred_n,
                            "duration_sec_raw": c.get("duration_sec_raw"),
                            "duration_sec_eval": c.get("duration_sec_eval"),
                            "vad": c.get("vad", {"vad_applied": False}),
                        }

                    # Consume chunk
                    pending = pending[len(chunk):]

                    # Heuristic batch-size update
                    if use_cuda:
                        mem_ratio = _cuda_mem_ratio()
                        avg_d = float(np.mean([x["duration_sec_eval"] for x in chunk])) if chunk else 0.0
                        cur_bs, oom_cooldown = _adjust_batch_size(cur_bs, avg_d, mem_ratio, cfg, oom_cooldown)
                        pbar.set_postfix({"bs": cur_bs, "vram": f"{mem_ratio:.0%}"})
                    
                    # Periodic cleanup to prevent memory fragmentation
                    batch_count += 1
                    if batch_count % cfg.cleanup_interval == 0:
                        if use_cuda:
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Incremental save predictions during cleanup
                        save_incremental_results(results, all_predictions, out_json)
                        
                        pbar.set_postfix({"bs": cur_bs, "vram": f"{mem_ratio:.0%}" if use_cuda else "cpu", "cleanup": f"batch_{batch_count}"})
                    
                    # Save every 10 batches to prevent data loss
                    elif batch_count % 10 == 0 and out_json is not None:
                        save_incremental_results(results, all_predictions, out_json)

                except torch.cuda.OutOfMemoryError:
                    if use_cuda:
                        torch.cuda.empty_cache()
                    gc.collect()

                    # Skip the problematic batch and continue
                    old_bs = cur_bs
                    cur_bs = max(cfg.batch_min, max(1, int(cur_bs // 2)))
                    oom_cooldown = 5

                    # Output problematic files to console
                    print(f"\n⚠ OOM Error: Skipping batch of {len(chunk)} items due to insufficient GPU memory")
                    print("📁 Problematic files in this batch:")
                    for i, bad_item in enumerate(chunk, 1):
                        audio_path = bad_item.get("audio_path", "unknown")
                        duration = bad_item.get("duration_sec_eval", bad_item.get("duration_sec_raw", "unknown"))
                        print(f"   {i}. {audio_path} (duration: {duration}s)")
                        skipped.append({"audio_path": audio_path, "reason": "oom_batch_skipped"})
                    
                    print(f"🔧 Reducing batch size from {old_bs} to {cur_bs} and continuing...")
                    print(f"💾 Current progress: {len(preds_raw)}/{len(test_rows)} samples processed\n")
                    
                    # Consume the problematic chunk entirely
                    pending = pending[len(chunk):]
                    continue

                finally:
                    # encourage freeing temporary tensors
                    try:
                        del batch_feats
                    except Exception:
                        pass
                    try:
                        del generated_ids
                    except Exception:
                        pass

        # Flush remaining
        if batch_buf:
            pending = batch_buf
            batch_buf = []

            while pending:
                chunk = pending[:cur_bs]

                try:
                    feats = [c["input_features"] for c in chunk]
                    batch_feats = _pad_stack_input_features(feats)
                    batch_feats = batch_feats.to(cfg.device, non_blocking=True)
                    if use_cuda and cfg.fp16:
                        batch_feats = batch_feats.half()

                    try:
                        generated_ids = model.generate(input_features=batch_feats, **gen_kwargs)
                    except ValueError as e:
                        msg = str(e)
                        if any(k in msg for k in ("forced_decoder_ids", "return_timestamps", "language", "task")):
                            generated_ids = model.generate(input_features=batch_feats)
                        else:
                            raise

                    texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                    for c, text_out in zip(chunk, texts):
                        pred = (text_out or "").strip()
                        ref = (c.get("ref") or "").strip()

                        preds_raw.append(pred)
                        refs_raw.append(ref)

                        if cfg.normalize_mode in {"whisper_basic", "basic"}:
                            pred_n = _basic_whisperish_normalize(pred)
                            ref_n = _basic_whisperish_normalize(ref)
                        else:
                            pred_n = pred
                            ref_n = ref

                        per_utt_wer.append(jiwer.wer(ref_n, pred_n))

                        per_item.append(
                            {
                                "audio_path": c["audio_path"],
                                "duration_sec_raw": c["duration_sec_raw"],
                                "duration_sec_eval": c["duration_sec_eval"],
                                "ref": ref,
                                "pred": pred,
                                "ref_norm": ref_n,
                                "pred_norm": pred_n,
                                "vad": c.get("vad", {"vad_applied": False}),
                            }
                        )

                    pending = pending[len(chunk):]

                    if use_cuda:
                        mem_ratio = _cuda_mem_ratio()
                        avg_d = float(np.mean([x["duration_sec_eval"] for x in chunk])) if chunk else 0.0
                        cur_bs, oom_cooldown = _adjust_batch_size(cur_bs, avg_d, mem_ratio, cfg, oom_cooldown)
                    
                    # Periodic cleanup to prevent memory fragmentation
                    batch_count += 1
                    if batch_count % cfg.cleanup_interval == 0:
                        if use_cuda:
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        gc.collect()
                        
                        # Incremental save predictions during cleanup
                        save_incremental_results(results, all_predictions, out_json)
                        
                        pbar.set_postfix({"bs": cur_bs, "vram": f"{mem_ratio:.0%}" if use_cuda else "cpu", "cleanup": f"batch_{batch_count}"})
                    
                    # Save every 10 batches to prevent data loss
                    elif batch_count % 10 == 0 and out_json is not None:
                        save_incremental_results(results, all_predictions, out_json)

                except torch.cuda.OutOfMemoryError:
                    if use_cuda:
                        torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Skip the problematic batch and continue
                    old_bs = cur_bs
                    cur_bs = max(cfg.batch_min, max(1, int(cur_bs // 2)))
                    oom_cooldown = 5
                    
                    # Output problematic files to console
                    print(f"\n⚠ OOM Error: Skipping batch of {len(chunk)} items due to insufficient GPU memory")
                    print("📁 Problematic files in this batch:")
                    for i, bad_item in enumerate(chunk, 1):
                        audio_path = bad_item.get("audio_path", "unknown")
                        duration = bad_item.get("duration_sec_eval", bad_item.get("duration_sec_raw", "unknown"))
                        print(f"   {i}. {audio_path} (duration: {duration}s)")
                        skipped.append({"audio_path": audio_path, "reason": "oom_batch_skipped"})
                    
                    print(f"🔧 Reducing batch size from {old_bs} to {cur_bs} and continuing...")
                    print(f"💾 Current progress: {len(preds_raw)}/{len(test_rows)} samples processed\n")
                    
                    # Consume the problematic chunk entirely
                    pending = pending[len(chunk):]
                    continue

                finally:
                    try:
                        del batch_feats
                    except Exception:
                        pass
                    try:
                        del generated_ids
                    except Exception:
                        pass

    # Final metrics (micro)
    if cfg.normalize_mode in {"whisper_basic", "basic"}:
        refs = [_basic_whisperish_normalize(r) for r in refs_raw]
        preds = [_basic_whisperish_normalize(p) for p in preds_raw]
    else:
        refs = refs_raw
        preds = preds_raw

    if len(refs) == 0:
        metrics = {
            "samples": 0,
            "skipped": len(skipped),
            "wer_micro": None,
            "cer_micro": None,
            "wer_macro": None,
            "wer_by_duration": {"0-1s": None, "1-2s": None, "2-5s": None, "5-10s": None, "10-30s": None},
            "normalize_mode": cfg.normalize_mode,
            "dynamic_audio_ctx": bool(cfg.dynamic_audio_ctx),
            "vad": cfg.vad.__dict__,
            "decode": {
                "num_beams": int(cfg.num_beams),
                "temperature": float(cfg.temperature),
                "max_new_tokens": int(cfg.max_new_tokens),
                "language": cfg.language,
                "task": cfg.task,
            },
        }
        if skipped:
            metrics["skipped_examples"] = skipped[:20]

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return metrics, per_item

    wer_micro = float(jiwer.wer(refs, preds))
    cer_micro = float(jiwer.cer(refs, preds))
    wer_macro = float(np.mean(per_utt_wer)) if per_utt_wer else None

    # Duration buckets based on *evaluation duration* (post-VAD if enabled)
    buckets: Dict[str, List[float]] = {"0-1s": [], "1-2s": [], "2-5s": [], "5-10s": [], "10-30s": []}
    for row in per_item:
        d = float(row.get("duration_sec_eval", row.get("duration_sec_raw", 0.0)))
        if cfg.normalize_mode in {"whisper_basic", "basic"}:
            w = jiwer.wer(row["ref_norm"], row["pred_norm"])
        else:
            w = jiwer.wer(row["ref"], row["pred"])

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

    bucket_means = {k: (float(np.mean(v)) if v else None) for k, v in buckets.items()}

    metrics = {
        "samples": len(refs),
        "skipped": len(skipped),
        "wer_micro": wer_micro,
        "cer_micro": cer_micro,
        "wer_macro": float(wer_macro) if wer_macro is not None else None,
        "wer_by_duration": bucket_means,
        "normalize_mode": cfg.normalize_mode,
        "dynamic_audio_ctx": bool(cfg.dynamic_audio_ctx),
        "vad": cfg.vad.__dict__,
        "decode": {
            "num_beams": int(cfg.num_beams),
            "temperature": float(cfg.temperature),
            "max_new_tokens": int(cfg.max_new_tokens),
            "language": cfg.language,
            "task": cfg.task,
        },
    }

    if skipped:
        metrics["skipped_examples"] = skipped[:20]

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return metrics, per_item


def load_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def recalculate_metrics_from_predictions(all_predictions: dict, model_name: str, cfg: EvalConfig) -> Tuple[Dict, List[dict]]:
    """Recalculate metrics for a model using existing predictions."""
    preds_raw = []
    refs_raw = []
    per_utt_wer = []
    per_item = []
    skipped = 0
    
    for audio_path, data in all_predictions.items():
        if "predictions" not in data or model_name not in data["predictions"]:
            skipped += 1
            continue
            
        pred_data = data["predictions"][model_name]
        pred = pred_data.get("pred", "").strip()
        ref = data.get("reference", "").strip()
        
        preds_raw.append(pred)
        refs_raw.append(ref)
        
        if cfg.normalize_mode in {"whisper_basic", "basic"}:
            pred_n = _basic_whisperish_normalize(pred)
            ref_n = _basic_whisperish_normalize(ref)
        else:
            pred_n = pred
            ref_n = ref
            
        per_utt_wer.append(jiwer.wer(ref_n, pred_n))
        
        per_item.append({
            "audio_path": audio_path,
            "duration_sec_raw": pred_data.get("duration_sec_raw", 0.0),
            "duration_sec_eval": pred_data.get("duration_sec_eval", 0.0),
            "ref": ref,
            "pred": pred,
            "ref_norm": ref_n,
            "pred_norm": pred_n,
            "vad": pred_data.get("vad", {"vad_applied": False}),
        })
    
    if len(refs_raw) == 0:
        return {
            "samples": 0,
            "skipped": skipped,
            "wer_micro": None,
            "cer_micro": None,
            "wer_macro": None,
            "wer_by_duration": {"0-1s": None, "1-2s": None, "2-5s": None, "5-10s": None, "10-30s": None},
            "normalize_mode": cfg.normalize_mode,
            "dynamic_audio_ctx": bool(cfg.dynamic_audio_ctx),
            "vad": cfg.vad.__dict__,
            "decode": {
                "num_beams": int(cfg.num_beams),
                "temperature": float(cfg.temperature),
                "max_new_tokens": int(cfg.max_new_tokens),
                "language": cfg.language,
                "task": cfg.task,
            },
        }, per_item

    # Calculate metrics
    if cfg.normalize_mode in {"whisper_basic", "basic"}:
        refs = [_basic_whisperish_normalize(r) for r in refs_raw]
        preds = [_basic_whisperish_normalize(p) for p in preds_raw]
    else:
        refs = refs_raw
        preds = preds_raw

    wer_micro = float(jiwer.wer(refs, preds))
    cer_micro = float(jiwer.cer(refs, preds))
    wer_macro = float(np.mean(per_utt_wer)) if per_utt_wer else None

    # Duration buckets
    buckets: Dict[str, List[float]] = {"0-1s": [], "1-2s": [], "2-5s": [], "5-10s": [], "10-30s": []}
    for row in per_item:
        d = float(row.get("duration_sec_eval", row.get("duration_sec_raw", 0.0)))
        if cfg.normalize_mode in {"whisper_basic", "basic"}:
            w = jiwer.wer(row["ref_norm"], row["pred_norm"])
        else:
            w = jiwer.wer(row["ref"], row["pred"])

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

    bucket_means = {k: (float(np.mean(v)) if v else None) for k, v in buckets.items()}

    metrics = {
        "samples": len(refs),
        "skipped": skipped,
        "wer_micro": wer_micro,
        "cer_micro": cer_micro,
        "wer_macro": float(wer_macro) if wer_macro is not None else None,
        "wer_by_duration": bucket_means,
        "normalize_mode": cfg.normalize_mode,
        "dynamic_audio_ctx": bool(cfg.dynamic_audio_ctx),
        "vad": cfg.vad.__dict__,
        "decode": {
            "num_beams": int(cfg.num_beams),
            "temperature": float(cfg.temperature),
            "max_new_tokens": int(cfg.max_new_tokens),
            "language": cfg.language,
            "task": cfg.task,
        },
    }

    return metrics, per_item


def save_incremental_results(results: dict, all_predictions: dict, out_json: Path) -> None:
    """Save intermediate results after each model evaluation."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    per_sample_json = out_json.parent / "evaluation_per_sample_predictions.json"
    with per_sample_json.open("w", encoding="utf-8") as f:
        json.dump(list(all_predictions.values()), f, indent=2, ensure_ascii=False)

    print(f"✓ Incremental results saved to: {out_json}")
    print(f"✓ Per-sample predictions saved to: {per_sample_json}")


def load_existing_results(out_json: Path) -> Tuple[dict, dict]:
    """Load existing results for resume capability."""
    if not out_json.exists():
        return {}, {}

    try:
        with out_json.open("r", encoding="utf-8") as f:
            results = json.load(f)

        per_sample_json = out_json.parent / "evaluation_per_sample_predictions.json"
        all_predictions = {}
        if per_sample_json.exists():
            with per_sample_json.open("r", encoding="utf-8") as f:
                per_sample_data = json.load(f)
                for item in per_sample_data:
                    all_predictions[item["audio_path"]] = item

        print(f"✓ Loaded existing results from: {out_json}")
        print(f"✓ Found {len(results.get('models', []))} already evaluated models")
        return results, all_predictions
    except Exception as e:
        print(f"⚠ Could not load existing results: {e}")
        return {}, {}


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_manifest", required=True, type=Path)
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

    # Inference batching (evaluation speed)
    ap.add_argument("--batch_size", type=int, default=0, help="0=auto default (GPU:8, CPU:1)")
    ap.add_argument("--auto_batch", type=int, default=1, help="1=auto-adjust batch size to use more VRAM safely; 0=static")
    ap.add_argument("--batch_min", type=int, default=1, help="Minimum batch size when auto_batch is enabled")
    ap.add_argument("--batch_max", type=int, default=64, help="Maximum batch size when auto_batch is enabled")

    # Half precision inference (GPU only)
    ap.add_argument("--fp16", type=int, default=1, help="1=use float16 model+inputs on CUDA to reduce VRAM; 0=float32")

    # Memory thresholds for auto_batch (allocated VRAM / total VRAM)
    ap.add_argument("--mem_low", type=float, default=0.60, help="If below this VRAM ratio, try increasing batch size")
    ap.add_argument("--mem_high", type=float, default=0.88, help="If above this VRAM ratio, try decreasing batch size")
    
    # Periodic cleanup settings
    ap.add_argument("--cleanup_interval", type=int, default=50, help="Cleanup memory every N batches to prevent fragmentation")

    ap.add_argument("--dynamic_audio_ctx", type=int, default=1, help="1=enable mel cropping by duration; 0=disable")

    ap.add_argument("--normalize", default="whisper_basic", choices=["whisper_basic", "none"])

    # VAD flags (keyboard-like)
    ap.add_argument("--vad_filter", type=int, default=1, help="1=enable Silero VAD trim; 0=disable")
    ap.add_argument("--vad_policy", default="skip", choices=["skip", "keep", "empty"], help="What to do if VAD finds no speech")
    ap.add_argument("--vad_threshold", type=float, default=0.5)
    ap.add_argument("--vad_min_speech_ms", type=int, default=250)
    ap.add_argument("--vad_min_silence_ms", type=int, default=100)
    ap.add_argument("--vad_speech_pad_ms", type=int, default=200)

    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--resume", action="store_true", help="Resume from existing results")
    ap.add_argument("--force_resume", action="store_true", help="Force resume: use existing predictions to skip evaluated audio files")

    args = ap.parse_args()

    if args.batch_size <= 0:
        # Conservative defaults: CPU=1, CUDA=8 (auto_batch can still back off on OOM)
        args.batch_size = 8 if str(args.device).startswith("cuda") and torch.cuda.is_available() else 1

    if not args.test_manifest.exists():
        raise FileNotFoundError(args.test_manifest)
    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(args.checkpoint_dir)
    if not (0.0 <= args.percentage <= 100.0):
        raise ValueError("--percentage must be 0..100")

    rows = load_jsonl(args.test_manifest)

    # Subset
    if args.percentage < 100.0:
        import random

        random.seed(42)
        k = max(1, int(len(rows) * (args.percentage / 100.0)))
        rows = random.sample(rows, k)

    # Checkpoints
    checkpoints = list(args.checkpoint_dir.glob("model_epoch_*"))
    checkpoints.sort(key=lambda p: int(p.name.split("_")[2]) if p.name.split("_")[2].isdigit() else 0)

    models: List[str] = []
    if args.compare_openai_tiny:
        models.append("openai/whisper-tiny.en")
    models.append(str(args.base_model))
    models.extend([str(p) for p in checkpoints])

    print(f"Device: {args.device}")
    print(f"Test samples: {len(rows)} (subset)")
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
    )

    out_json = args.out_json
    if out_json is None:
        out_json = args.checkpoint_dir / "evaluation_results_futo_like.json"

    # Resume logic: load existing results if requested
    results = {
        "test_manifest": str(args.test_manifest),
        "checkpoint_dir": str(args.checkpoint_dir),
        "percentage": float(args.percentage),
        "models": [],
        "cfg": {
            **cfg.__dict__,
            "vad": vad_cfg.__dict__,
        },
    }
    all_predictions = {}

    if args.resume or args.force_resume:
        existing_results, existing_predictions = load_existing_results(out_json)
        if existing_results:
            results = existing_results
            all_predictions = existing_predictions

    # Get list of already evaluated models to skip them
    evaluated_models = {model_info["model"] for model_info in results.get("models", [])}
    print(f"Already evaluated models: {len(evaluated_models)}")

    # Force resume: filter out already evaluated audio files
    if args.force_resume and all_predictions:
        original_count = len(rows)
        # Filter rows to only include audio files that haven't been fully evaluated for all models
        unevaluated_audio = []
        models_to_eval = [m for m in models if m not in evaluated_models]
        
        for row in rows:
            audio_path = str(row.get("audio_path", ""))
            if audio_path in all_predictions:
                # Check if this audio file has predictions for all already evaluated models
                existing_preds = all_predictions[audio_path].get("predictions", {})
                # If any model in models_to_eval is missing, keep this row for evaluation
                if any(model_name not in existing_preds for model_name in models_to_eval):
                    unevaluated_audio.append(row)
            else:
                unevaluated_audio.append(row)
        
        rows = unevaluated_audio
        print(f"Force resume: filtered from {original_count} to {len(rows)} audio files")
        if len(rows) == 0:
            print("All audio files already evaluated. Nothing to do.")
            return

    best_micro = results.get("summary", {}).get("best_wer_micro")
    best_model = results.get("summary", {}).get("best_wer_micro_model")

    # Per-sample prediction comparison - only for unevaluated models
    for row in rows:
        apath = str(row.get("audio_path", ""))
        if apath not in all_predictions:
            all_predictions[apath] = {
                "audio_path": apath,
                "reference": (row.get("raw_transcription") or "").strip(),
                "predictions": {},
            }

    vad_trimmer = SileroVADTrimmer() if cfg.vad.enabled else None

    for m in models:
        # Skip already evaluated models
        if m in evaluated_models:
            print(f"\n⏭ Skipping already evaluated model: {m}")
            continue

        print("\n" + "=" * 70)
        print(f"Evaluating: {m}")
        print("=" * 70)

        # Force resume: check if we can recalculate from existing predictions
        if args.force_resume and all_predictions:
            model_name = Path(m).name if Path(m).exists() else m
            has_predictions = any(
                model_name in data.get("predictions", {}) 
                for data in all_predictions.values()
            )
            
            if has_predictions:
                print(f"📊 Recalculating metrics from existing predictions for {model_name}")
                metrics, per_item = recalculate_metrics_from_predictions(all_predictions, model_name, cfg)
                results["models"].append({"model": m, "metrics": metrics})
                
                # Update per-sample predictions structure (already exists)
                for item in per_item:
                    apath = item["audio_path"]
                    if apath in all_predictions and model_name in all_predictions[apath].get("predictions", {}):
                        # Ensure the prediction data has the expected structure
                        pred_data = all_predictions[apath]["predictions"][model_name]
                        pred_data.update({
                            "pred_norm": item["pred_norm"],
                            "duration_sec_raw": item.get("duration_sec_raw"),
                            "duration_sec_eval": item.get("duration_sec_eval"),
                            "vad": item.get("vad"),
                        })
            else:
                print(f"⚠ No existing predictions found for {model_name}, running full evaluation")
                metrics, per_item = eval_one_model(m, rows, processor, cfg, vad_trimmer=vad_trimmer, results=results, all_predictions=all_predictions, out_json=out_json)
                results["models"].append({"model": m, "metrics": metrics})

                model_name = Path(m).name if Path(m).exists() else m
                for item in per_item:
                    apath = item["audio_path"]
                    if apath not in all_predictions:
                        all_predictions[apath] = {
                            "audio_path": apath,
                            "reference": (item.get("ref") or "").strip(),
                            "predictions": {},
                        }
                    all_predictions[apath]["predictions"][model_name] = {
                        "pred": item["pred"],
                        "pred_norm": item["pred_norm"],
                        "duration_sec_raw": item.get("duration_sec_raw"),
                        "duration_sec_eval": item.get("duration_sec_eval"),
                        "vad": item.get("vad"),
                    }
        else:
            # Normal evaluation
            metrics, per_item = eval_one_model(m, rows, processor, cfg, vad_trimmer=vad_trimmer, results=results, all_predictions=all_predictions, out_json=out_json)
            results["models"].append({"model": m, "metrics": metrics})

            model_name = Path(m).name if Path(m).exists() else m
            for item in per_item:
                apath = item["audio_path"]
                if apath not in all_predictions:
                    all_predictions[apath] = {
                        "audio_path": apath,
                        "reference": (item.get("ref") or "").strip(),
                        "predictions": {},
                    }
                all_predictions[apath]["predictions"][model_name] = {
                    "pred": item["pred"],
                    "pred_norm": item["pred_norm"],
                    "duration_sec_raw": item.get("duration_sec_raw"),
                    "duration_sec_eval": item.get("duration_sec_eval"),
                    "vad": item.get("vad"),
                }

        print(f"samples={metrics['samples']} skipped={metrics['skipped']}")
        print(f"WER micro: {metrics['wer_micro']} | WER macro: {metrics['wer_macro']} | CER: {metrics['cer_micro']}")
        print(f"WER by duration: {metrics.get('wer_by_duration')}")

        if metrics.get("wer_micro") is not None:
            if best_micro is None or metrics["wer_micro"] < best_micro:
                best_micro = metrics["wer_micro"]
                best_model = m

        # Update summary and save incrementally
        results["summary"] = {"best_wer_micro_model": best_model, "best_wer_micro": best_micro}
        save_incremental_results(results, all_predictions, out_json)
        
        # Aggressive memory cleanup between models
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    print("\n" + "#" * 70)
    print("FINAL")
    print("#" * 70)
    print(json.dumps(results["summary"], indent=2))
    print(f"Final results saved to: {out_json}")
    print(f"Per-sample predictions saved to: {out_json.parent / 'evaluation_per_sample_predictions.json'}")


if __name__ == "__main__":
    main()
