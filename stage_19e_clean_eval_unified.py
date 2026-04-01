#!/usr/bin/env python3
"""Clean (no-mix) evaluation for HF Transformers + NeMo models.

Outputs are Stage-19E-compatible JSON structures, with a clean condition:
- evaluation_results_clean.json
- evaluation_per_sample_predictions_clean.json
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import jiwer
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor

import stage_19e_edge_and_moonshine_targetmix_sweep_with_cer as s19e
from cloud_eval_common import ModelSpec, load_models_csv


def _force_utf8_stdio() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _run_args_blob(args: argparse.Namespace, out_json: Path) -> dict:
    return {
        "version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": [str(a) for a in sys.argv],
        "python": sys.executable,
        "cwd": os.getcwd(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} | {"out_json": str(out_json)},
    }


def _save_outputs(
    results: dict,
    all_predictions: dict,
    out_json: Path,
    out_per_sample_json: Path,
    run_args: Optional[dict],
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    payload = s19e._pack_per_sample_payload(all_predictions, run_args=run_args)
    out_per_sample_json.parent.mkdir(parents=True, exist_ok=True)
    out_per_sample_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved clean outputs: {out_json} | {out_per_sample_json}")


def _load_existing(out_json: Path, out_per_sample_json: Path) -> tuple[dict, dict, Optional[dict]]:
    results: dict = {}
    all_predictions: dict = {}
    meta: Optional[dict] = None

    if out_json.exists():
        try:
            blob = json.loads(out_json.read_text(encoding="utf-8"))
            if isinstance(blob, dict):
                results = blob
        except Exception as exc:
            print(f"Warning: could not load {out_json}: {exc}")

    if out_per_sample_json.exists():
        all_predictions, meta = s19e.load_per_sample_predictions(out_per_sample_json)

    return results, all_predictions, meta


def _extract_text(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        txt = item.get("text")
        return txt if isinstance(txt, str) else ""
    txt = getattr(item, "text", None)
    return txt if isinstance(txt, str) else ""


def _load_nemo_model(model_ref: str, device: str) -> Any:
    try:
        import nemo.collections.asr as nemo_asr  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing NeMo dependencies. Install with: pip install \"nemo_toolkit[asr]>=2.4.0\""
        ) from exc

    if Path(model_ref).is_file() and Path(model_ref).suffix.lower() == ".nemo":
        model = nemo_asr.models.ASRModel.restore_from(restore_path=str(model_ref), map_location=device)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_ref)
        model = model.to(device)
    model.eval()
    return model


def _nemo_transcribe_batch(model: Any, audios: list[np.ndarray], batch_size: int, num_workers: int) -> list[str]:
    kwargs: dict[str, Any] = {
        "batch_size": max(1, int(batch_size)),
        "num_workers": max(0, int(num_workers)),
        "return_hypotheses": False,
        "verbose": False,
    }
    try:
        out = model.transcribe(audio=audios, **kwargs)
    except TypeError:
        kwargs.pop("verbose", None)
        out = model.transcribe(audio=audios, **kwargs)

    if isinstance(out, tuple) and out:
        out = out[0]
    if isinstance(out, list):
        return [_extract_text(x) for x in out]
    return [_extract_text(out)]


def _compute_clean_metrics(items: list[dict], normalize_mode: str) -> dict:
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
    refs_raw = [str(x.get("target_ref", "") or "") for x in items]
    if normalize_mode in {"whisper_basic", "basic"}:
        preds = [s19e._basic_whisperish_normalize(p) for p in preds_raw]
        refs = [s19e._basic_whisperish_normalize(r) for r in refs_raw]
    else:
        preds = preds_raw
        refs = refs_raw

    wer_micro = float(jiwer.wer(refs, preds))
    cer_micro = float(jiwer.cer(refs, preds))
    wer_utt = [float(x["wer_target"]) for x in items if x.get("wer_target") is not None]
    cer_utt = [float(x["cer_target"]) for x in items if x.get("cer_target") is not None]

    return {
        "samples": len(items),
        "wer_micro_target": wer_micro,
        "cer_micro_target": cer_micro,
        "wer_macro_target": float(np.mean(wer_utt)) if wer_utt else None,
        "cer_macro_target": float(np.mean(cer_utt)) if cer_utt else None,
        "wer_micro_other": None,
        "cer_micro_other": None,
        "wer_macro_other": None,
        "cer_macro_other": None,
        "win_rate_target_closer": None,
        "avg_margin_other_minus_target": None,
        "avg_margin_cer_other_minus_target": None,
        "likely_hit_max_token_cap_rate": None,
        "wer_by_duration_target": s19e._wer_by_duration(items, "wer_target"),
        "wer_by_duration_other": {"0-1s": None, "1-2s": None, "2-5s": None, "5-10s": None, "10-30s": None},
    }


def _build_target_rows(args: argparse.Namespace) -> tuple[list[dict], dict]:
    rows = s19e.load_jsonl(args.test_manifest)
    scores = s19e.load_speaker_scores_csv(args.speaker_scores_csv)
    targets, target_info = s19e.select_targets_sorted(
        rows=rows,
        scores=scores,
        target_take=int(args.target_max),
        target_percent=float(args.target_percentage),
    )
    if args.percentage < 100.0:
        targets = s19e._deterministic_subsample(
            targets,
            key_fn=lambda r: s19e._norm_windows_key(str(r.get("audio_path", ""))),
            percentage=float(args.percentage),
            seed=int(args.seed),
        )
    return targets, target_info


def _make_clean_samples(target_rows: list[dict]) -> list[dict]:
    samples: list[dict] = []
    for idx, row in enumerate(target_rows):
        audio_path = str(row.get("audio_path", "") or "").strip()
        target_ref = (s19e._row_transcript(row) or "").strip()
        if not audio_path or not target_ref:
            continue
        key = f"clean::{s19e._norm_windows_key(audio_path)}::i{idx}"
        samples.append(
            {
                "sample_key": key,
                "target_audio_path": audio_path,
                "target_ref": target_ref,
            }
        )
    return samples


def _ensure_prediction_stubs(samples: list[dict], all_predictions: dict) -> None:
    for sample in samples:
        key = sample["sample_key"]
        if key not in all_predictions or not isinstance(all_predictions.get(key), dict):
            all_predictions[key] = {
                "mix_key": key,
                "cond_id": "clean",
                "snr_db": None,
                "overlap": None,
                "target_audio_path": sample["target_audio_path"],
                "other_audio_path": None,
                "target_reference": sample["target_ref"],
                "other_reference": "",
                "predictions": {},
            }
            continue
        item = all_predictions[key]
        if not isinstance(item.get("predictions"), dict):
            item["predictions"] = {}
        if not item.get("cond_id"):
            item["cond_id"] = "clean"
        if not item.get("target_audio_path"):
            item["target_audio_path"] = sample["target_audio_path"]
        if not item.get("target_reference"):
            item["target_reference"] = sample["target_ref"]
        if "other_reference" not in item:
            item["other_reference"] = ""


def _device_from_arg(device_arg: str) -> str:
    if str(device_arg) == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device_arg)


def _eval_one_hf_model(
    model_ref: str,
    samples: list[dict],
    args: argparse.Namespace,
    all_predictions: dict,
    active_keys: set[str],
    out_json: Path,
    out_per_sample_json: Path,
    results: dict,
    run_args: dict,
) -> tuple[dict, dict]:
    processor_id = s19e._resolve_processor_id_for_model(model_ref, args.base_processor_id)
    processor = AutoProcessor.from_pretrained(processor_id)
    model = s19e._load_model_for_eval(model_ref, lora_merge=False, lora_base_model=None)

    device = _device_from_arg(args.device)
    model = model.to(device)
    model.eval()
    model_type = s19e._safe_model_type(model_ref)
    is_moonshine = s19e._is_moonshine_family(model_type, model_ref)
    is_whisper = s19e._is_whisper_family(model_type, model_ref)

    forced_decoder_ids = None
    if is_whisper:
        try:
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
        except Exception:
            forced_decoder_ids = None

    audio_cache = s19e.AudioCacheLRU(int(max(0.0, float(args.audio_cache_gb)) * (1024**3)))

    def get_audio_cached(path: Path) -> tuple[np.ndarray, int]:
        cached = audio_cache.get(str(path))
        if cached is not None:
            return cached
        audio, sr = s19e.load_audio_mono_16k(path)
        audio_cache.put(str(path), audio, sr)
        return audio, sr

    current_batch = max(1, int(args.batch_size))
    skipped: list[dict] = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0
    started = time.perf_counter()
    eval_count = 0

    pending: list[dict] = []
    pbar = tqdm(samples, desc=f"[clean-hf] {model_ref}", unit="utt")

    def flush_pending(chunk: list[dict]) -> None:
        nonlocal current_batch, total_infer_sec, total_audio_sec, eval_count
        if not chunk:
            return
        audios = [x["audio"] for x in chunk]
        model_inputs = processor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        if "attention_mask" not in model_inputs:
            feats = model_inputs.get("input_features")
            if feats is not None and hasattr(feats, "shape"):
                bsz = int(feats.shape[0])
                t = int(feats.shape[-1])
                model_inputs["attention_mask"] = torch.ones((bsz, t), dtype=torch.long)

        for k in list(model_inputs.keys()):
            model_inputs[k] = model_inputs[k].to(device)

        max_length = None
        if is_moonshine:
            seq_len = int(model_inputs["input_features"].shape[-1])
            max_length = int(math.ceil(float(seq_len) * (float(args.moonshine_token_limit_tps) / 16000.0)))
            max_length = max(8, min(max_length, int(args.max_new_tokens)))

        gen_kwargs = {
            "num_beams": int(args.num_beams),
            "temperature": float(args.temperature),
            "max_new_tokens": int(args.max_new_tokens),
        }
        if max_length is not None:
            gen_kwargs["max_length"] = int(max_length)
        if forced_decoder_ids is not None:
            gen_kwargs["forced_decoder_ids"] = forced_decoder_ids

        t0 = time.perf_counter()
        generated = model.generate(**model_inputs, **gen_kwargs)
        infer_sec = time.perf_counter() - t0
        total_infer_sec += infer_sec
        decoded = processor.batch_decode(generated.detach().cpu(), skip_special_tokens=True)

        for meta, pred in zip(chunk, decoded):
            pred_s = (pred or "").strip()
            ref = (meta["target_ref"] or "").strip()
            if args.normalize in {"whisper_basic", "basic"}:
                pred_n = s19e._basic_whisperish_normalize(pred_s)
                ref_n = s19e._basic_whisperish_normalize(ref)
            else:
                pred_n, ref_n = pred_s, ref

            wer_t = float(jiwer.wer(ref_n, pred_n))
            cer_t = float(jiwer.cer(ref_n, pred_n))
            mix_key = meta["sample_key"]
            all_predictions.setdefault(
                mix_key,
                {
                    "mix_key": mix_key,
                    "cond_id": "clean",
                    "snr_db": None,
                    "overlap": None,
                    "target_audio_path": meta["target_audio_path"],
                    "other_audio_path": None,
                    "target_reference": ref,
                    "other_reference": "",
                    "predictions": {},
                },
            )
            all_predictions[mix_key]["predictions"][model_ref] = {
                "pred": pred_s,
                "wer_target": wer_t,
                "wer_other": None,
                "cer_target": cer_t,
                "cer_other": None,
                "win_target_closer": None,
                "duration_sec_eval": float(meta["duration_sec_eval"]),
                "duration_sec_infer": float(infer_sec / max(1, len(decoded))),
                "likely_hit_max_token_cap": None,
            }
            total_audio_sec += float(meta["duration_sec_eval"])
            eval_count += 1

    idx = 0
    while idx < len(samples):
        sample = samples[idx]
        key = sample["sample_key"]
        pbar.update(1)
        if key not in active_keys:
            idx += 1
            continue
        if model_ref in all_predictions.get(key, {}).get("predictions", {}):
            idx += 1
            continue

        try:
            audio, sr = get_audio_cached(Path(sample["target_audio_path"]))
            if sr != 16000:
                raise RuntimeError(f"Expected 16kHz audio, got {sr}")
            pending.append(
                {
                    "sample_key": key,
                    "target_audio_path": sample["target_audio_path"],
                    "target_ref": sample["target_ref"],
                    "duration_sec_eval": s19e.seconds_from_audio(audio, sr),
                    "audio": audio.astype(np.float32),
                }
            )
            idx += 1
            if len(pending) < current_batch:
                continue

            try:
                flush_pending(pending)
                pending = []
            except torch.cuda.OutOfMemoryError:
                if current_batch > 1:
                    current_batch = max(1, current_batch // 2)
                    print(f"[clean-hf] CUDA OOM, reducing batch to {current_batch}")
                else:
                    failed = pending.pop(0)
                    skipped.append({"sample_key": failed["sample_key"], "reason": "cuda_oom_batch1"})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                for failed in pending:
                    skipped.append(
                        {
                            "sample_key": failed.get("sample_key"),
                            "reason": f"hf_inference_exception: {type(exc).__name__}: {exc}",
                        }
                    )
                pending = []
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if args.save_every > 0 and (eval_count % int(args.save_every) == 0):
                _save_outputs(results, all_predictions, out_json, out_per_sample_json, run_args)

        except Exception as exc:
            skipped.append({"sample_key": key, "reason": f"{type(exc).__name__}: {exc}"})
            idx += 1
            continue

    pbar.close()
    if pending:
        try:
            flush_pending(pending)
        except Exception as exc:
            for failed in pending:
                skipped.append(
                    {
                        "sample_key": failed.get("sample_key"),
                        "reason": f"hf_inference_exception_final: {type(exc).__name__}: {exc}",
                    }
                )

    total_wall = max(0.0, time.perf_counter() - started)
    items: list[dict] = []
    for key, blob in all_predictions.items():
        if key not in active_keys:
            continue
        preds = blob.get("predictions", {})
        if model_ref not in preds:
            continue
        p = preds[model_ref]
        items.append(
            {
                "mix_key": key,
                "cond_id": "clean",
                "duration_sec_eval": p.get("duration_sec_eval"),
                "target_audio_path": blob.get("target_audio_path"),
                "target_ref": blob.get("target_reference", ""),
                "pred": p.get("pred", ""),
                "wer_target": p.get("wer_target"),
                "cer_target": p.get("cer_target"),
            }
        )

    overall = _compute_clean_metrics(items, args.normalize)
    overall["model_type"] = model_type or "hf_transformers"
    overall["processor_id"] = processor_id
    overall["model_num_params"] = int(model.num_parameters()) if hasattr(model, "num_parameters") else None
    overall["skipped"] = len(skipped)
    overall["eval_audio_sec"] = float(total_audio_sec)
    overall["eval_wall_time_sec"] = float(total_wall)
    overall["eval_infer_only_sec"] = float(total_infer_sec)
    if total_wall > 0.0 and total_audio_sec > 0.0:
        overall["eval_throughput_xrt"] = float(total_audio_sec / total_wall)
        overall["eval_rtf"] = float(total_wall / total_audio_sec)
    else:
        overall["eval_throughput_xrt"] = None
        overall["eval_rtf"] = None

    by_cond = {"clean": dict(overall)}

    del model
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return overall, by_cond


def _eval_one_nemo_model(
    model_ref: str,
    samples: list[dict],
    args: argparse.Namespace,
    all_predictions: dict,
    active_keys: set[str],
    out_json: Path,
    out_per_sample_json: Path,
    results: dict,
    run_args: dict,
) -> tuple[dict, dict]:
    device = _device_from_arg(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    model = _load_nemo_model(model_ref, device=device)
    audio_cache = s19e.AudioCacheLRU(int(max(0.0, float(args.audio_cache_gb)) * (1024**3)))

    def get_audio_cached(path: Path) -> tuple[np.ndarray, int]:
        cached = audio_cache.get(str(path))
        if cached is not None:
            return cached
        audio, sr = s19e.load_audio_mono_16k(path)
        audio_cache.put(str(path), audio, sr)
        return audio, sr

    skipped: list[dict] = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0
    eval_count = 0
    started = time.perf_counter()

    batch_audio: list[np.ndarray] = []
    batch_meta: list[dict] = []

    for sample in tqdm(samples, desc=f"[clean-nemo] {model_ref}", unit="utt"):
        key = sample["sample_key"]
        if key not in active_keys:
            continue
        if model_ref in all_predictions.get(key, {}).get("predictions", {}):
            continue
        try:
            audio, sr = get_audio_cached(Path(sample["target_audio_path"]))
            if sr != 16000:
                raise RuntimeError(f"Expected 16kHz audio, got {sr}")
            batch_audio.append(audio.astype(np.float32))
            batch_meta.append(
                {
                    "sample_key": key,
                    "target_audio_path": sample["target_audio_path"],
                    "target_ref": sample["target_ref"],
                    "duration_sec_eval": s19e.seconds_from_audio(audio, sr),
                }
            )
            if len(batch_audio) < int(args.batch_size):
                continue

            start = time.perf_counter()
            decoded = _nemo_transcribe_batch(
                model=model,
                audios=batch_audio,
                batch_size=int(args.batch_size),
                num_workers=int(args.infer_num_workers),
            )
            infer_sec = time.perf_counter() - start
            total_infer_sec += infer_sec

            for meta, pred in zip(batch_meta, decoded):
                pred_s = (pred or "").strip()
                ref = (meta["target_ref"] or "").strip()
                if args.normalize in {"whisper_basic", "basic"}:
                    pred_n = s19e._basic_whisperish_normalize(pred_s)
                    ref_n = s19e._basic_whisperish_normalize(ref)
                else:
                    pred_n, ref_n = pred_s, ref

                wer_t = float(jiwer.wer(ref_n, pred_n))
                cer_t = float(jiwer.cer(ref_n, pred_n))
                mix_key = meta["sample_key"]
                all_predictions.setdefault(
                    mix_key,
                    {
                        "mix_key": mix_key,
                        "cond_id": "clean",
                        "snr_db": None,
                        "overlap": None,
                        "target_audio_path": meta["target_audio_path"],
                        "other_audio_path": None,
                        "target_reference": ref,
                        "other_reference": "",
                        "predictions": {},
                    },
                )
                all_predictions[mix_key]["predictions"][model_ref] = {
                    "pred": pred_s,
                    "wer_target": wer_t,
                    "wer_other": None,
                    "cer_target": cer_t,
                    "cer_other": None,
                    "win_target_closer": None,
                    "duration_sec_eval": float(meta["duration_sec_eval"]),
                    "duration_sec_infer": float(infer_sec / max(1, len(decoded))),
                    "likely_hit_max_token_cap": None,
                }
                total_audio_sec += float(meta["duration_sec_eval"])
                eval_count += 1

            batch_audio = []
            batch_meta = []

            if args.save_every > 0 and (eval_count % int(args.save_every) == 0):
                _save_outputs(results, all_predictions, out_json, out_per_sample_json, run_args)

        except Exception as exc:
            skipped.append({"sample_key": key, "reason": f"{type(exc).__name__}: {exc}"})
            continue

    if batch_audio:
        start = time.perf_counter()
        decoded = _nemo_transcribe_batch(
            model=model,
            audios=batch_audio,
            batch_size=int(args.batch_size),
            num_workers=int(args.infer_num_workers),
        )
        infer_sec = time.perf_counter() - start
        total_infer_sec += infer_sec
        for meta, pred in zip(batch_meta, decoded):
            pred_s = (pred or "").strip()
            ref = (meta["target_ref"] or "").strip()
            if args.normalize in {"whisper_basic", "basic"}:
                pred_n = s19e._basic_whisperish_normalize(pred_s)
                ref_n = s19e._basic_whisperish_normalize(ref)
            else:
                pred_n, ref_n = pred_s, ref
            wer_t = float(jiwer.wer(ref_n, pred_n))
            cer_t = float(jiwer.cer(ref_n, pred_n))
            mix_key = meta["sample_key"]
            all_predictions.setdefault(
                mix_key,
                {
                    "mix_key": mix_key,
                    "cond_id": "clean",
                    "snr_db": None,
                    "overlap": None,
                    "target_audio_path": meta["target_audio_path"],
                    "other_audio_path": None,
                    "target_reference": ref,
                    "other_reference": "",
                    "predictions": {},
                },
            )
            all_predictions[mix_key]["predictions"][model_ref] = {
                "pred": pred_s,
                "wer_target": wer_t,
                "wer_other": None,
                "cer_target": cer_t,
                "cer_other": None,
                "win_target_closer": None,
                "duration_sec_eval": float(meta["duration_sec_eval"]),
                "duration_sec_infer": float(infer_sec / max(1, len(decoded))),
                "likely_hit_max_token_cap": None,
            }
            total_audio_sec += float(meta["duration_sec_eval"])
            eval_count += 1

    total_wall = max(0.0, time.perf_counter() - started)
    items: list[dict] = []
    for key, blob in all_predictions.items():
        if key not in active_keys:
            continue
        preds = blob.get("predictions", {})
        if model_ref not in preds:
            continue
        p = preds[model_ref]
        items.append(
            {
                "mix_key": key,
                "cond_id": "clean",
                "duration_sec_eval": p.get("duration_sec_eval"),
                "target_audio_path": blob.get("target_audio_path"),
                "target_ref": blob.get("target_reference", ""),
                "pred": p.get("pred", ""),
                "wer_target": p.get("wer_target"),
                "cer_target": p.get("cer_target"),
            }
        )

    overall = _compute_clean_metrics(items, args.normalize)
    overall["model_type"] = "nemo_parakeet"
    overall["processor_id"] = None
    overall["model_num_params"] = None
    overall["skipped"] = len(skipped)
    overall["eval_audio_sec"] = float(total_audio_sec)
    overall["eval_wall_time_sec"] = float(total_wall)
    overall["eval_infer_only_sec"] = float(total_infer_sec)
    if total_wall > 0.0 and total_audio_sec > 0.0:
        overall["eval_throughput_xrt"] = float(total_audio_sec / total_wall)
        overall["eval_rtf"] = float(total_wall / total_audio_sec)
    else:
        overall["eval_throughput_xrt"] = None
        overall["eval_rtf"] = None

    by_cond = {"clean": dict(overall)}

    del model
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return overall, by_cond


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Clean (no-targetmix) evaluation for HF + NeMo backends")
    ap.add_argument("--models_file", required=True, type=Path)
    ap.add_argument("--test_manifest", required=True, type=Path)
    ap.add_argument("--speaker_scores_csv", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--out_per_sample_json", type=Path, default=None)

    ap.add_argument("--language_mode_filter", default="en")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--percentage", type=float, default=100.0)
    ap.add_argument("--target_percentage", type=float, default=100.0)
    ap.add_argument("--target_max", type=int, default=0)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--infer_num_workers", type=int, default=0)
    ap.add_argument("--audio_cache_gb", type=float, default=1.0)
    ap.add_argument("--ffmpeg_path", default="ffmpeg")

    ap.add_argument("--base_processor_id", default="openai/whisper-small.en")
    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="transcribe")
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--moonshine_token_limit_tps", type=float, default=6.5)
    ap.add_argument("--normalize", default="whisper_basic", choices=["whisper_basic", "none"])

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force_resume", action="store_true")
    ap.add_argument("--recalc_metrics", action="store_true")
    ap.add_argument("--save_every", type=int, default=100)
    ap.add_argument("--skip_model_failures", type=int, default=1)
    args = ap.parse_args()

    if args.batch_size <= 0:
        args.batch_size = 1
    if args.target_max < 0:
        raise ValueError("--target_max must be >= 0")
    if not (0.0 <= args.percentage <= 100.0):
        raise ValueError("--percentage must be 0..100")
    if not (0.0 <= args.target_percentage <= 100.0):
        raise ValueError("--target_percentage must be 0..100")
    if args.moonshine_token_limit_tps <= 0:
        raise ValueError("--moonshine_token_limit_tps must be > 0")
    return args


def _select_models(models: list[ModelSpec], language_mode_filter: str) -> list[ModelSpec]:
    mode = (language_mode_filter or "").strip().lower()
    out: list[ModelSpec] = []
    for m in models:
        if not m.enabled:
            continue
        lm = (m.language_mode or "").strip().lower()
        if mode and lm != mode:
            continue
        out.append(m)
    return out


def main() -> None:
    _force_utf8_stdio()
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_json or (args.output_dir / "evaluation_results_clean.json")
    out_per_sample_json = args.out_per_sample_json or (args.output_dir / "evaluation_per_sample_predictions_clean.json")

    core_load = s19e.load_audio_mono_16k
    s19e.load_audio_mono_16k = s19e.make_loader_with_ffmpeg(str(args.ffmpeg_path), core_load)

    run_args = _run_args_blob(args, out_json)
    models = _select_models(load_models_csv(args.models_file), args.language_mode_filter)
    if not models:
        raise RuntimeError("No enabled models after applying language_mode_filter")

    target_rows, target_info = _build_target_rows(args)
    samples = _make_clean_samples(target_rows)
    if not samples:
        raise RuntimeError("No target samples found for clean evaluation")
    active_keys = {s["sample_key"] for s in samples}

    results = {
        "mode": "clean_target_only",
        "test_manifest": str(args.test_manifest),
        "speaker_scores_csv": str(args.speaker_scores_csv),
        "output_dir": str(args.output_dir),
        "pair_info": {
            "target_info": target_info,
            "samples_total": len(samples),
            "percentage": float(args.percentage),
            "target_percentage": float(args.target_percentage),
            "target_max": int(args.target_max),
        },
        "base_pairs": len(samples),
        "conditions": [{"snr_db": None, "overlap": None, "cond_id": "clean"}],
        "pairs_total": len(samples),
        "cfg": {
            "device": args.device,
            "language": args.language,
            "task": args.task,
            "num_beams": args.num_beams,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "normalize_mode": args.normalize,
            "batch": {
                "batch_size": int(args.batch_size),
                "infer_num_workers": int(args.infer_num_workers),
            },
            "audio_cache_gb": float(args.audio_cache_gb),
            "ffmpeg_path": str(args.ffmpeg_path),
        },
        "models_requested": [m.model_ref for m in models],
        "model_failures": [],
        "models": [],
    }
    all_predictions: dict = {}

    if args.resume or args.force_resume:
        existing_results, existing_predictions, _ = _load_existing(out_json, out_per_sample_json)
        if existing_results:
            results = existing_results
            all_predictions = existing_predictions
    if not isinstance(results.get("model_failures"), list):
        results["model_failures"] = []

    _ensure_prediction_stubs(samples, all_predictions)
    existing_models = {m.get("model") for m in results.get("models", []) if isinstance(m, dict)}

    print(f"Clean eval samples: {len(samples)}")
    print(f"Models to evaluate: {len(models)}")

    for spec in models:
        model_ref = spec.model_ref
        model_results_key = model_ref
        model_already_done = model_results_key in existing_models
        wants_recalc = bool(args.recalc_metrics or args.force_resume)

        if model_already_done and not wants_recalc:
            missing = 0
            for sample in samples:
                preds = all_predictions.get(sample["sample_key"], {}).get("predictions", {})
                if model_ref not in preds:
                    missing += 1
            if missing == 0:
                print(f"Skipping already evaluated model: {model_ref}")
                continue
            print(f"Evaluating missing clean samples for {model_ref}: {missing}")

        print("=" * 80)
        print(f"Clean evaluation: {model_ref} ({spec.backend})")
        print("=" * 80)

        try:
            if model_already_done and wants_recalc:
                items: list[dict] = []
                for key, blob in all_predictions.items():
                    if key not in active_keys:
                        continue
                    preds = blob.get("predictions", {})
                    if model_ref not in preds:
                        continue
                    p = preds[model_ref]
                    items.append(
                        {
                            "mix_key": key,
                            "cond_id": "clean",
                            "duration_sec_eval": p.get("duration_sec_eval"),
                            "target_audio_path": blob.get("target_audio_path"),
                            "target_ref": blob.get("target_reference", ""),
                            "pred": p.get("pred", ""),
                            "wer_target": p.get("wer_target"),
                            "cer_target": p.get("cer_target"),
                        }
                    )
                overall = _compute_clean_metrics(items, args.normalize)
                overall["model_type"] = spec.backend
                overall["skipped"] = int(overall.get("skipped") or 0)
                by_cond = {"clean": dict(overall)}
            elif spec.backend == "hf_transformers":
                overall, by_cond = _eval_one_hf_model(
                    model_ref=model_ref,
                    samples=samples,
                    args=args,
                    all_predictions=all_predictions,
                    active_keys=active_keys,
                    out_json=out_json,
                    out_per_sample_json=out_per_sample_json,
                    results=results,
                    run_args=run_args,
                )
            elif spec.backend == "nemo_parakeet":
                overall, by_cond = _eval_one_nemo_model(
                    model_ref=model_ref,
                    samples=samples,
                    args=args,
                    all_predictions=all_predictions,
                    active_keys=active_keys,
                    out_json=out_json,
                    out_per_sample_json=out_per_sample_json,
                    results=results,
                    run_args=run_args,
                )
            else:
                raise RuntimeError(f"Unsupported backend: {spec.backend}")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            if bool(args.skip_model_failures):
                print(f"Model failed and skipped: {model_ref} | {err}")
                results["model_failures"].append({"model": model_ref, "error": err})
                _save_outputs(results, all_predictions, out_json, out_per_sample_json, run_args)
                continue
            raise

        results["models"] = [m for m in results.get("models", []) if m.get("model") != model_results_key]
        results["models"].append(
            {
                "model": model_results_key,
                "metrics_overall": overall,
                "metrics_by_condition": by_cond,
            }
        )
        results["model_failures"] = [
            f for f in results["model_failures"] if str(f.get("model")) != model_results_key
        ]
        _save_outputs(results, all_predictions, out_json, out_per_sample_json, run_args)

        print(
            f"samples={overall.get('samples')} "
            f"WER_target={overall.get('wer_micro_target')} "
            f"CER_target={overall.get('cer_micro_target')} "
            f"speed_xrt={overall.get('eval_throughput_xrt')}"
        )

    print("Clean evaluation complete.")
    s19e.beep()


if __name__ == "__main__":
    main()
