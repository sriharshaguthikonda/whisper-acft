r"""stage_14_eval_futo_like.py

Goal
----
Evaluate Whisper checkpoints in a way that's closer to how FUTO Keyboard / FUTO Voice Input behaves
(whisper.cpp style: short dictations, dynamic audio context, beam search, symbol suppression-ish).

This is still Hugging Face inference (not whisper.cpp), but it fixes the biggest eval pitfalls:
- consistent decoding config across checkpoints
- optional dynamic audio-context emulation (crop mel frames by utterance length)
- robust audio loading (stereo->mono, float32)
- strict accounting of skipped samples
- both micro-WER (corpus) and macro-WER (per-utterance mean)
- duration bucket breakdown (keyboard dictations are short)

Run
---
python stage_14a_futo_like_evaluate_test_manifest.py \
  --test_manifest "I:\Record_chunks\pairs_manifest_filtered_with_noises_and_others_voices_mixed_aug_gain_aug_rir_real_silent_test_manifest.jsonl" \
  --checkpoint_dir "I:\Dynamic_n_ctx_checkpoints_partialctx" \
  --base_model "futo-org/acft-whisper-tiny.en" \
  --percentage 100 \
  --device cuda \
  --num_beams 5 \
  --temperature 0.0 \
  --normalize whisper_basic \
  --dynamic_audio_ctx 1

Notes
-----
If you want true apples-to-apples with FUTO keyboard, the gold standard is:
- convert checkpoints to whisper.cpp format (gguf)
- evaluate via whisper.cpp using the same --audio-context schedule + beam settings

But as a *checkpoint ranker before ACFT stage*, this script is usually good enough.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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
    """A pragmatic normaliser for WER that matches typical Whisper eval better than raw strings.

    - lowercase
    - collapse whitespace
    - remove most punctuation (keeps apostrophes inside words)

    This is not identical to OpenAI whisper.normalizers, but it's stable + dependency-free.
    """
    s = s.strip().lower()
    # Replace fancy apostrophes
    s = s.replace("’", "'")
    # Remove punctuation except apostrophe inside words
    s = re.sub(r"(?!\B'\b)[^a-z0-9\s']+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if s == "<|nospeech|>":
        return ""
    return s


def make_jiwer_transform(mode: str):
    mode = (mode or "none").lower()
    if mode == "none":
        return None
    if mode in {"whisper_basic", "basic"}:
        # We'll normalise ourselves (faster + predictable)
        return "custom_basic"

    raise ValueError(f"Unknown normalize mode: {mode}")


# ----------------------------
# Audio helpers
# ----------------------------

def load_audio_mono_16k(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=False)

    # stereo -> mono
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)

    # force float32
    audio = np.asarray(audio, dtype=np.float32)

    # handle NaNs
    if not np.isfinite(audio).all():
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if sr != 16000:
        # Use resample_poly if available for quality; fall back to linear.
        try:
            import scipy.signal
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
# Dynamic audio context emulation
# ----------------------------

FULL_MEL_FRAMES = 3000  # whisper feature extractor pads/truncates to 30s => 3000 frames


def mel_frames_for_duration(duration_sec: float) -> int:
    """Approx mel frame count for a given duration (cap at 30s)."""
    d = max(0.0, min(30.0, float(duration_sec)))
    frames = int(round((FULL_MEL_FRAMES / 30.0) * d))
    return max(1, min(FULL_MEL_FRAMES, frames))


def crop_input_features_for_duration(input_features: torch.Tensor, duration_sec: float) -> torch.Tensor:
    """Crop time dimension of Whisper mel features to approximate whisper.cpp --audio-context speedups.

    input_features expected shape either:
    - (B, 80, T)
    - (B, T, 80)

    We crop T.
    """
    frames = mel_frames_for_duration(duration_sec)

    if input_features.ndim != 3:
        return input_features

    b, d1, d2 = input_features.shape
    if d1 == 80:
        # (B, 80, T)
        return input_features[:, :, :frames]
    if d2 == 80:
        # (B, T, 80)
        return input_features[:, :frames, :]

    # Unknown layout; don't crop
    return input_features


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
    normalize_mode: str
    device: str


def build_generate_kwargs(processor: WhisperProcessor, model: WhisperForConditionalGeneration, cfg: EvalConfig) -> Dict:
    """Return ONLY generic generation kwargs.

    Important:
    - Some checkpoints/configs (or Transformers versions) will reject Whisper-specific kwargs
      (e.g., forced_decoder_ids) and treat them as unused model_kwargs.
    - To stay robust across your checkpoint zoo, we keep kwargs minimal and instead
      set whisper prompts via model.generation_config in eval_one_model().
    """

    gen_kwargs = {
        "num_beams": int(cfg.num_beams),
        "temperature": float(cfg.temperature),
        "do_sample": False,
        "max_new_tokens": int(cfg.max_new_tokens),
    }

    return gen_kwargs


def eval_one_model(
    model_id_or_path: str,
    test_rows: List[dict],
    processor: WhisperProcessor,
    cfg: EvalConfig,
) -> Tuple[Dict, List[dict]]:

    model = WhisperForConditionalGeneration.from_pretrained(model_id_or_path)
    model.to(cfg.device)
    model.eval()

    # Set Whisper prompt via generation_config (robust: doesn't go through generate(**kwargs) parsing)
    # This avoids failures like: "model_kwargs not used: ['forced_decoder_ids']".
    try:
        fids = processor.get_decoder_prompt_ids(language=cfg.language, task=cfg.task)
        if hasattr(model, "generation_config") and hasattr(model.generation_config, "forced_decoder_ids"):
            model.generation_config.forced_decoder_ids = fids
    except Exception:
        pass

    gen_kwargs = build_generate_kwargs(processor, model, cfg)

    preds_raw: List[str] = []
    refs_raw: List[str] = []
    per_item: List[dict] = []
    skipped: List[dict] = []

    # For macro-WER we compute per-utterance WER and then average
    per_utt_wer: List[float] = []

    with torch.inference_mode():
        for item in tqdm(test_rows, desc=f"eval {Path(model_id_or_path).name}"):
            ap = Path(item.get("audio_path", ""))
            if not ap.exists():
                skipped.append({"audio_path": str(ap), "reason": "missing_file"})
                continue

            try:
                audio, sr = load_audio_mono_16k(ap)
                dur = seconds_from_audio(audio, sr)

                # Feature extraction
                inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
                input_features = inputs["input_features"].to(cfg.device)

                if cfg.dynamic_audio_ctx:
                    input_features = crop_input_features_for_duration(input_features, dur)

                # Generate
                try:
                    generated_ids = model.generate(input_features=input_features, **gen_kwargs)
                except ValueError as e:
                    # Some configs may still be strict; retry with the bare minimum.
                    msg = str(e)
                    if "forced_decoder_ids" in msg or "return_timestamps" in msg or "language" in msg or "task" in msg:
                        generated_ids = model.generate(input_features=input_features)
                    else:
                        raise
                pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

                ref = (item.get("raw_transcription") or "").strip()

                preds_raw.append(pred)
                refs_raw.append(ref)

                # Normalised versions for WER
                if cfg.normalize_mode in {"whisper_basic", "basic"}:
                    pred_n = _basic_whisperish_normalize(pred)
                    ref_n = _basic_whisperish_normalize(ref)
                else:
                    pred_n = pred
                    ref_n = ref

                # per-utt WER (macro)
                per_utt_wer.append(jiwer.wer(ref_n, pred_n))

                per_item.append(
                    {
                        "audio_path": str(ap),
                        "duration_sec": dur,
                        "ref": ref,
                        "pred": pred,
                        "ref_norm": ref_n,
                        "pred_norm": pred_n,
                    }
                )

            except Exception as e:
                skipped.append({"audio_path": str(ap), "reason": f"exception: {type(e).__name__}: {e}"})
                continue

    # Final metrics (micro)
    if cfg.normalize_mode in {"whisper_basic", "basic"}:
        refs = [_basic_whisperish_normalize(r) for r in refs_raw]
        preds = [_basic_whisperish_normalize(p) for p in preds_raw]
    else:
        refs = refs_raw
        preds = preds_raw

    if len(refs) == 0:
        # Keep the schema stable so the caller can always print keys.
        metrics = {
            "samples": 0,
            "skipped": len(skipped),
            "wer_micro": None,
            "cer_micro": None,
            "wer_macro": None,
            "wer_by_duration": {
                "0-1s": None,
                "1-2s": None,
                "2-5s": None,
                "5-10s": None,
                "10-30s": None,
            },
            "normalize_mode": cfg.normalize_mode,
            "dynamic_audio_ctx": bool(cfg.dynamic_audio_ctx),
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

        # Cleanup
        del model
        torch.cuda.empty_cache()
        gc.collect()

        return metrics, per_item

    wer_micro = jiwer.wer(refs, preds)
    cer_micro = jiwer.cer(refs, preds)
    wer_macro = float(np.mean(per_utt_wer)) if per_utt_wer else None

    # Duration buckets: keyboard dictations are often very short
    buckets = {
        "0-1s": [],
        "1-2s": [],
        "2-5s": [],
        "5-10s": [],
        "10-30s": [],
    }
    for row in per_item:
        d = row["duration_sec"]
        w = jiwer.wer(row["ref_norm"], row["pred_norm"]) if cfg.normalize_mode in {"whisper_basic", "basic"} else jiwer.wer(row["ref"], row["pred"])
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
        "wer_micro": float(wer_micro),
        "cer_micro": float(cer_micro),
        "wer_macro": float(wer_macro) if wer_macro is not None else None,
        "wer_by_duration": bucket_means,
        "normalize_mode": cfg.normalize_mode,
        "dynamic_audio_ctx": bool(cfg.dynamic_audio_ctx),
        "decode": {
            "num_beams": int(cfg.num_beams),
            "temperature": float(cfg.temperature),
            "max_new_tokens": int(cfg.max_new_tokens),
            "language": cfg.language,
            "task": cfg.task,
        },
    }

    # Attach skipped in metrics (kept small)
    if skipped:
        metrics["skipped_examples"] = skipped[:20]

    # Cleanup
    del model
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_manifest", required=True, type=Path)
    ap.add_argument("--checkpoint_dir", required=True, type=Path)
    ap.add_argument("--percentage", type=float, default=100.0)

    ap.add_argument("--base_model", default="futo-org/acft-whisper-tiny.en")
    ap.add_argument("--compare_openai_tiny", action="store_true", help="Also evaluate openai/whisper-tiny.en for reference")

    ap.add_argument("--base_processor_id", default="openai/whisper-tiny.en", help="Processor/tokenizer ID")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="transcribe")

    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)

    ap.add_argument("--dynamic_audio_ctx", type=int, default=1, help="1=enable mel cropping by duration; 0=disable")

    ap.add_argument("--normalize", default="whisper_basic", choices=["whisper_basic", "none"])

    ap.add_argument("--out_json", type=Path, default=None)

    args = ap.parse_args()

    if not args.test_manifest.exists():
        raise FileNotFoundError(args.test_manifest)
    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(args.checkpoint_dir)
    if not (0.0 <= args.percentage <= 100.0):
        raise ValueError("--percentage must be 0..100")

    rows = load_jsonl(args.test_manifest)

    # subset
    if args.percentage < 100.0:
        import random

        random.seed(42)
        k = max(1, int(len(rows) * (args.percentage / 100.0)))
        rows = random.sample(rows, k)

    # checkpoints
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

    # processor (keep fixed across models)
    processor = WhisperProcessor.from_pretrained(args.base_processor_id)

    cfg = EvalConfig(
        base_processor_id=args.base_processor_id,
        language=args.language,
        task=args.task,
        num_beams=args.num_beams,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        dynamic_audio_ctx=bool(args.dynamic_audio_ctx),
        normalize_mode=args.normalize,
        device=args.device,
    )

    results = {
        "test_manifest": str(args.test_manifest),
        "checkpoint_dir": str(args.checkpoint_dir),
        "percentage": float(args.percentage),
        "models": [],
        "cfg": cfg.__dict__,
    }

    best_micro = None
    best_model = None

    # Store per-item predictions for all models, keyed by audio_path
    all_predictions: Dict[str, dict] = {}
    for row in rows:
        ap = str(row.get("audio_path", ""))
        all_predictions[ap] = {
            "audio_path": ap,
            "reference": (row.get("raw_transcription") or "").strip(),
            "predictions": {},
        }

    for m in models:
        print("\n" + "=" * 70)
        print(f"Evaluating: {m}")
        print("=" * 70)

        metrics, per_item = eval_one_model(m, rows, processor, cfg)
        results["models"].append({"model": m, "metrics": metrics})

        # Store predictions for this model
        model_name = Path(m).name if Path(m).exists() else m
        for item in per_item:
            ap = item["audio_path"]
            if ap in all_predictions:
                all_predictions[ap]["predictions"][model_name] = {
                    "pred": item["pred"],
                    "pred_norm": item["pred_norm"],
                }

        print(f"samples={metrics['samples']} skipped={metrics['skipped']}")
        print(f"WER micro: {metrics['wer_micro']} | WER macro: {metrics['wer_macro']} | CER: {metrics['cer_micro']}")
        print(f"WER by duration: {metrics.get('wer_by_duration')}")

        # If everything got skipped, show the reason(s)
        if metrics.get("samples", 0) == 0 and metrics.get("skipped_examples"):
            print("Skipped examples (first few):")
            for ex in metrics["skipped_examples"][:5]:
                print(f"  - {ex.get('audio_path')}: {ex.get('reason')}")

        if metrics.get("wer_micro") is not None:
            if best_micro is None or metrics["wer_micro"] < best_micro:
                best_micro = metrics["wer_micro"]
                best_model = m

    results["summary"] = {"best_wer_micro_model": best_model, "best_wer_micro": best_micro}

    out_json = args.out_json
    if out_json is None:
        out_json = args.checkpoint_dir / "evaluation_results_futo_like.json"

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save per-sample comparison file
    per_sample_json = out_json.parent / "evaluation_per_sample_predictions.json"
    per_sample_data = list(all_predictions.values())
    with per_sample_json.open("w", encoding="utf-8") as f:
        json.dump(per_sample_data, f, indent=2, ensure_ascii=False)

    print("\n" + "#" * 70)
    print("FINAL")
    print("#" * 70)
    print(json.dumps(results["summary"], indent=2))
    print(f"Saved: {out_json}")
    print(f"Saved per-sample predictions: {per_sample_json}")


if __name__ == "__main__":
    main()
