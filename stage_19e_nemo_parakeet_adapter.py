#!/usr/bin/env python3
"""Append NeMo/Parakeet model results to Stage 19E targetmix outputs.

This script mirrors Stage 19E pair construction and output schema, then runs
NeMo ASR inference for the requested models. It writes into the same files:

- evaluation_results_futo_like_targetmix_sweep.json
- evaluation_per_sample_predictions_targetmix_sweep.json

so charting/reporting tools can continue to work unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import jiwer
import numpy as np
import torch
from tqdm import tqdm

import stage_19e_edge_and_moonshine_targetmix_sweep_with_cer as s19e


def _force_utf8_stdio() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


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


def _parse_models_csv(value: Optional[str]) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[\n,;]+", str(value).strip())
    return [p.strip() for p in parts if p.strip()]


def _load_nemo_model(model_ref: str, device: str) -> Any:
    try:
        import nemo.collections.asr as nemo_asr  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing NeMo dependencies. Install with: pip install \"nemo_toolkit[asr]>=2.4.0\""
        ) from exc

    # Hub id path: use from_pretrained. Local .nemo: restore.
    if Path(model_ref).is_file() and Path(model_ref).suffix.lower() == ".nemo":
        model = nemo_asr.models.ASRModel.restore_from(restore_path=str(model_ref), map_location=device)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_ref)
        model = model.to(device)
    model.eval()
    return model


def _transcribe_batch(model: Any, audios: list[np.ndarray], batch_size: int, num_workers: int) -> list[str]:
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


def _build_conditions(args: argparse.Namespace) -> list[s19e.MixCondition]:
    snr_list = s19e.parse_float_list(args.sweep_snr_db) or [10.0]
    if args.disable_overlap_sweep:
        overlap_list = None
    else:
        overlap_list = s19e.parse_overlap_list(args.sweep_overlap)

    conditions: list[s19e.MixCondition] = []
    if overlap_list is None:
        for snr in snr_list:
            conditions.append(s19e.MixCondition(snr_db=float(snr), overlap=None))
    else:
        for snr in snr_list:
            for overlap in overlap_list:
                conditions.append(s19e.MixCondition(snr_db=float(snr), overlap=float(overlap)))
    return conditions


def _build_pairs(args: argparse.Namespace, conditions: list[s19e.MixCondition]) -> tuple[list[dict], list[dict], dict]:
    rows = s19e.load_jsonl(args.test_manifest)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Optional ffmpeg fallback for mixed codecs.
    core_load = s19e.load_audio_mono_16k
    s19e.load_audio_mono_16k = s19e.make_loader_with_ffmpeg(str(args.ffmpeg_path), core_load)

    if args.others_dir is not None:
        score_db = s19e.load_speaker_scores_csv(args.speaker_scores_csv)
        target_rows, target_info = s19e.select_targets_sorted(
            rows=rows,
            scores=score_db,
            target_take=int(args.target_max),
            target_percent=float(args.target_percentage),
        )

        other_paths = s19e.scan_audio_files(args.others_dir)
        other_tx: Optional[s19e.TranscriptDB] = None
        if args.others_manifest is not None:
            other_tx = s19e.build_transcript_db_from_manifest(Path(args.others_manifest))

        if args.pairs_manifest is None:
            args.pairs_manifest = args.checkpoint_dir / "pairs_manifest_targetmix_sweep_othersdir.jsonl"
        args.pairs_manifest.parent.mkdir(parents=True, exist_ok=True)

        if (args.resume or args.force_resume) and args.pairs_manifest.exists() and not args.rebuild_pairs:
            base_pairs = s19e.load_jsonl(args.pairs_manifest)
            pair_info = {
                "loaded_from": str(args.pairs_manifest),
                "base_pairs": len(base_pairs),
                "target_info": target_info,
            }
            print(f"Loaded base pairs: {args.pairs_manifest} ({len(base_pairs)})")
        else:
            base_pairs, base_info = s19e.build_base_pairs_targets_vs_others(
                target_rows=target_rows,
                other_paths=other_paths,
                other_tx=other_tx,
                mix_per_target=int(args.mix_per_target),
                pairing_mode=str(args.pairing_mode),
                seed=int(args.seed),
                allow_missing_other_ref=bool(args.allow_missing_other_ref),
            )
            pair_info = {"target_info": target_info, "base_info": base_info}
            with args.pairs_manifest.open("w", encoding="utf-8") as f:
                for row in base_pairs:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"Saved base pairs: {args.pairs_manifest}")
    else:
        labels = s19e.load_speaker_sort_scores(args.speaker_scores_csv)
        base_pairs, pair_info = s19e.build_target_other_base_pairs(
            rows=rows,
            labels=labels,
            mix_per_target=int(args.mix_per_target),
            pairing_mode=str(args.pairing_mode),
            seed=int(args.seed),
            target_percentage=float(args.target_percentage),
            target_max=int(args.target_max),
        )

    if args.percentage < 100.0:
        base_pairs = s19e._deterministic_subsample(
            base_pairs,
            key_fn=lambda bp: str(bp.get("base_key", "")),
            percentage=float(args.percentage),
            seed=int(args.seed),
        )

    pair_rows = s19e.expand_pairs_with_conditions(base_pairs, conditions)
    return rows, pair_rows, pair_info


def _ensure_prediction_stubs(pair_rows: list[dict], all_predictions: dict) -> None:
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


def _run_args_blob(args: argparse.Namespace, out_json: Path) -> dict:
    return {
        "version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": [str(a) for a in sys.argv],
        "python": sys.executable,
        "cwd": os.getcwd(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()} | {"out_json": str(out_json)},
    }


def _save_incremental_results_safe(results: dict, all_predictions: dict, out_json: Path, run_args: Optional[dict]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    per_sample_json = out_json.parent / "evaluation_per_sample_predictions_targetmix_sweep.json"
    payload = s19e._pack_per_sample_payload(all_predictions, run_args=run_args)
    per_sample_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        print(f"Saved incremental outputs: {out_json} | {per_sample_json}")
    except Exception:
        pass


def _evaluate_one_nemo_model(
    model_ref: str,
    pair_rows: list[dict],
    args: argparse.Namespace,
    all_predictions: dict,
    active_keys: set[str],
    out_json: Path,
    results: dict,
    run_args: dict,
) -> tuple[dict, dict]:
    model_name = str(model_ref)
    model_results_key = str(model_ref)

    audio_cache = s19e.AudioCacheLRU(int(max(0.0, float(args.audio_cache_gb)) * (1024**3)))

    def get_audio_cached(path: Path) -> tuple[np.ndarray, int]:
        cached = audio_cache.get(str(path))
        if cached is not None:
            return cached
        audio, sr = s19e.load_audio_mono_16k(path)
        audio_cache.put(str(path), audio, sr)
        return audio, sr

    vad_cfg = s19e.VADConfig(
        enabled=bool(args.vad_filter),
        policy=str(args.vad_policy),
        threshold=float(args.vad_threshold),
        min_speech_duration_ms=int(args.vad_min_speech_ms),
        min_silence_duration_ms=int(args.vad_min_silence_ms),
        speech_pad_ms=int(args.vad_speech_pad_ms),
    )
    vad_trimmer = s19e.SileroVADTrimmer() if vad_cfg.enabled else None

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    print(f"[nemo] Loading model: {model_ref}")
    model = _load_nemo_model(model_ref, device=device)

    # Keep deterministic mixing per-pair.
    np_seed = int(args.seed)

    skipped: list[dict] = []
    batch_audio: list[np.ndarray] = []
    batch_meta: list[dict] = []
    total_audio_sec = 0.0
    total_infer_sec = 0.0
    first_infer_wall = time.perf_counter()
    eval_count = 0

    pbar = tqdm(pair_rows, desc=f"[nemo] {model_ref}", unit="mix")
    for item in pbar:
        key = item["mix_key"]
        if key not in active_keys:
            continue
        if model_name in all_predictions.get(key, {}).get("predictions", {}):
            continue

        try:
            t_audio, t_sr = get_audio_cached(Path(item["target_audio_path"]))
            o_audio, o_sr = get_audio_cached(Path(item["other_audio_path"]))
            if t_sr != 16000 or o_sr != 16000:
                raise RuntimeError(f"expected 16kHz audio, got target={t_sr} other={o_sr}")

            rng = np.random.default_rng(
                s19e._stable_uint64_from_str(f"{key}||seed{np_seed}", seed=np_seed)
            )
            mix, _ = s19e.mix_target_with_other(
                target=t_audio,
                other=o_audio,
                snr_db_target_over_other=float(item["snr_db"]),
                other_offset_mode=str(args.other_offset_mode),
                rng=rng,
                other_peak_ratio=float(args.other_peak_ratio),
                overlap_ratio=item.get("overlap", None),
            )

            dur_eval = s19e.seconds_from_audio(mix, 16000)
            if vad_trimmer is not None:
                mix_vad, vad_meta = vad_trimmer.trim(mix, 16000, vad_cfg)
                if mix_vad.size == 0:
                    policy = str(vad_cfg.policy).lower()
                    if policy == "skip":
                        skipped.append(
                            {
                                "mix_key": key,
                                "reason": "vad_empty_skip",
                                "cond_id": item.get("cond_id", ""),
                            }
                        )
                        continue
                    if policy == "empty":
                        mix = np.zeros((0,), dtype=np.float32)
                    else:
                        mix = mix
                else:
                    mix = mix_vad.astype(np.float32)
                _ = vad_meta

            batch_audio.append(mix.astype(np.float32))
            batch_meta.append(
                {
                    "mix_key": key,
                    "cond_id": item.get("cond_id", ""),
                    "snr_db": float(item.get("snr_db")),
                    "overlap": item.get("overlap", None),
                    "target_audio_path": item["target_audio_path"],
                    "other_audio_path": item["other_audio_path"],
                    "target_ref": item["target_ref"],
                    "other_ref": item["other_ref"],
                    "duration_sec_eval": float(dur_eval),
                }
            )

            if len(batch_audio) < int(args.batch_size):
                continue

            start = time.perf_counter()
            texts = _transcribe_batch(
                model=model,
                audios=batch_audio,
                batch_size=int(args.batch_size),
                num_workers=int(args.infer_num_workers),
            )
            infer_sec = time.perf_counter() - start
            total_infer_sec += infer_sec

            for meta, pred in zip(batch_meta, texts):
                pred_s = (pred or "").strip()
                target_ref = (meta.get("target_ref") or "").strip()
                other_ref = (meta.get("other_ref") or "").strip()

                if args.normalize in {"whisper_basic", "basic"}:
                    pred_n = s19e._basic_whisperish_normalize(pred_s)
                    target_n = s19e._basic_whisperish_normalize(target_ref)
                    other_n = s19e._basic_whisperish_normalize(other_ref)
                else:
                    pred_n, target_n, other_n = pred_s, target_ref, other_ref

                wer_t = float(jiwer.wer(target_n, pred_n))
                cer_t = float(jiwer.cer(target_n, pred_n))
                wer_o = float(jiwer.wer(other_n, pred_n))
                cer_o = float(jiwer.cer(other_n, pred_n))

                mix_key = meta["mix_key"]
                all_predictions.setdefault(
                    mix_key,
                    {
                        "mix_key": mix_key,
                        "cond_id": meta["cond_id"],
                        "snr_db": float(meta["snr_db"]),
                        "overlap": meta["overlap"],
                        "target_audio_path": meta["target_audio_path"],
                        "other_audio_path": meta["other_audio_path"],
                        "target_reference": target_ref,
                        "other_reference": other_ref,
                        "predictions": {},
                    },
                )
                all_predictions[mix_key]["predictions"][model_name] = {
                    "pred": pred_s,
                    "wer_target": wer_t,
                    "wer_other": wer_o,
                    "cer_target": cer_t,
                    "cer_other": cer_o,
                    "win_target_closer": bool(wer_t < wer_o),
                    "duration_sec_eval": meta["duration_sec_eval"],
                    "duration_sec_infer": float(infer_sec / max(1, len(texts))),
                    "likely_hit_max_token_cap": None,
                }
                total_audio_sec += float(meta["duration_sec_eval"])
                eval_count += 1

            batch_audio = []
            batch_meta = []

            if args.save_every > 0 and (eval_count % int(args.save_every) == 0):
                _save_incremental_results_safe(results, all_predictions, out_json, run_args=run_args)

        except Exception as exc:
            skipped.append(
                {
                    "mix_key": key,
                    "reason": f"nemo_exception: {type(exc).__name__}: {exc}",
                    "cond_id": item.get("cond_id", ""),
                }
            )
            continue

    if batch_audio:
        start = time.perf_counter()
        texts = _transcribe_batch(
            model=model,
            audios=batch_audio,
            batch_size=int(args.batch_size),
            num_workers=int(args.infer_num_workers),
        )
        infer_sec = time.perf_counter() - start
        total_infer_sec += infer_sec
        for meta, pred in zip(batch_meta, texts):
            pred_s = (pred or "").strip()
            target_ref = (meta.get("target_ref") or "").strip()
            other_ref = (meta.get("other_ref") or "").strip()

            if args.normalize in {"whisper_basic", "basic"}:
                pred_n = s19e._basic_whisperish_normalize(pred_s)
                target_n = s19e._basic_whisperish_normalize(target_ref)
                other_n = s19e._basic_whisperish_normalize(other_ref)
            else:
                pred_n, target_n, other_n = pred_s, target_ref, other_ref

            wer_t = float(jiwer.wer(target_n, pred_n))
            cer_t = float(jiwer.cer(target_n, pred_n))
            wer_o = float(jiwer.wer(other_n, pred_n))
            cer_o = float(jiwer.cer(other_n, pred_n))
            mix_key = meta["mix_key"]
            all_predictions.setdefault(
                mix_key,
                {
                    "mix_key": mix_key,
                    "cond_id": meta["cond_id"],
                    "snr_db": float(meta["snr_db"]),
                    "overlap": meta["overlap"],
                    "target_audio_path": meta["target_audio_path"],
                    "other_audio_path": meta["other_audio_path"],
                    "target_reference": target_ref,
                    "other_reference": other_ref,
                    "predictions": {},
                },
            )
            all_predictions[mix_key]["predictions"][model_name] = {
                "pred": pred_s,
                "wer_target": wer_t,
                "wer_other": wer_o,
                "cer_target": cer_t,
                "cer_other": cer_o,
                "win_target_closer": bool(wer_t < wer_o),
                "duration_sec_eval": meta["duration_sec_eval"],
                "duration_sec_infer": float(infer_sec / max(1, len(texts))),
                "likely_hit_max_token_cap": None,
            }
            total_audio_sec += float(meta["duration_sec_eval"])
            eval_count += 1

    total_wall = max(0.0, time.perf_counter() - first_infer_wall)
    active_model_keys = {pr["mix_key"] for pr in pair_rows}
    overall, by_cond = s19e.recompute_metrics_from_saved_predictions(
        all_predictions=all_predictions,
        model_name=model_name,
        normalize_mode=str(args.normalize),
        active_keys=active_model_keys,
    )

    if int(overall.get("samples") or 0) == 0 and pair_rows:
        first_reason = skipped[0]["reason"] if skipped else "no_predictions_saved"
        raise RuntimeError(
            f"No predictions were saved for NeMo model '{model_ref}'. "
            f"Skipped={len(skipped)} first_reason={first_reason}"
        )

    overall["model_type"] = "nemo_parakeet"
    overall["processor_id"] = None
    overall["model_num_params"] = None
    overall["skipped"] = len(skipped)
    overall["eval_audio_sec"] = float(total_audio_sec)
    overall["eval_wall_time_sec"] = float(total_wall)
    if total_wall > 0.0 and total_audio_sec > 0.0:
        overall["eval_throughput_xrt"] = float(total_audio_sec / total_wall)
        overall["eval_rtf"] = float(total_wall / total_audio_sec)
    else:
        overall["eval_throughput_xrt"] = None
        overall["eval_rtf"] = None
    overall["eval_infer_only_sec"] = float(total_infer_sec)

    # Keep by-condition schema unchanged but add speed hints.
    for cond_metrics in by_cond.values():
        cond_metrics.setdefault("eval_throughput_xrt", overall.get("eval_throughput_xrt"))
        cond_metrics.setdefault("eval_rtf", overall.get("eval_rtf"))

    # Cleanup model memory before next one.
    del model
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return overall, by_cond


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage 19E NeMo adapter (targetmix)")

    ap.add_argument("--test_manifest", required=True, type=Path)
    ap.add_argument("--speaker_scores_csv", required=True, type=Path)
    ap.add_argument("--checkpoint_dir", required=True, type=Path)
    ap.add_argument("--out_json", type=Path, default=None)

    ap.add_argument("--models_csv", required=True, help="Comma/semicolon/newline separated NeMo model refs")
    ap.add_argument("--percentage", type=float, default=100.0)
    ap.add_argument("--target_percentage", type=float, default=100.0)
    ap.add_argument("--target_max", type=int, default=0)
    ap.add_argument("--mix_per_target", type=int, default=1)
    ap.add_argument("--pairing_mode", default="round_robin", choices=["round_robin", "random", "hash"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--other_offset_mode", default="start", choices=["start", "random"])
    ap.add_argument("--other_peak_ratio", type=float, default=1.0)
    ap.add_argument("--sweep_snr_db", type=str, default="20,10,5,0,-5")
    ap.add_argument("--sweep_overlap", type=str, default="0,0.25,0.5,0.75,1")
    ap.add_argument("--disable_overlap_sweep", action="store_true")

    ap.add_argument("--others_dir", type=Path, default=None)
    ap.add_argument("--others_manifest", type=Path, default=None)
    ap.add_argument("--allow_missing_other_ref", type=int, default=0)
    ap.add_argument("--pairs_manifest", type=Path, default=None)
    ap.add_argument("--rebuild_pairs", action="store_true")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--infer_num_workers", type=int, default=0)
    ap.add_argument("--audio_cache_gb", type=float, default=1.0)
    ap.add_argument("--ffmpeg_path", default="ffmpeg")
    ap.add_argument("--normalize", default="whisper_basic", choices=["whisper_basic", "none"])

    # Optional VAD settings mirror Stage 19E.
    ap.add_argument("--vad_filter", type=int, default=0)
    ap.add_argument("--vad_policy", default="skip", choices=["skip", "keep", "empty"])
    ap.add_argument("--vad_threshold", type=float, default=0.5)
    ap.add_argument("--vad_min_speech_ms", type=int, default=250)
    ap.add_argument("--vad_min_silence_ms", type=int, default=100)
    ap.add_argument("--vad_speech_pad_ms", type=int, default=200)

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
    return args


def main() -> None:
    _force_utf8_stdio()
    args = _parse_args()
    conditions = _build_conditions(args)
    rows, pair_rows, pair_info = _build_pairs(args, conditions)

    if not pair_rows:
        raise RuntimeError("No evaluation pairs generated.")

    out_json = args.out_json
    if out_json is None:
        out_json = args.checkpoint_dir / "evaluation_results_futo_like_targetmix_sweep.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)

    run_args = _run_args_blob(args, out_json)

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
        "base_pairs": int(len({row["base_key"] for row in pair_rows})),
        "conditions": [{"snr_db": c.snr_db, "overlap": c.overlap, "cond_id": c.cond_id()} for c in conditions],
        "pairs_total": len(pair_rows),
        "cfg": {
            "device": args.device,
            "normalize_mode": args.normalize,
            "batch": {"batch_size": int(args.batch_size), "infer_num_workers": int(args.infer_num_workers)},
            "pairing": {
                "mix_per_target": int(args.mix_per_target),
                "pairing_mode": str(args.pairing_mode),
                "seed": int(args.seed),
                "target_percentage": float(args.target_percentage),
                "target_max": int(args.target_max),
                "percentage_pairs": float(args.percentage),
            },
            "mixing": {
                "other_offset_mode": str(args.other_offset_mode),
                "other_peak_ratio": float(args.other_peak_ratio),
            },
            "audio_cache_gb": float(args.audio_cache_gb),
            "ffmpeg_path": str(args.ffmpeg_path),
            "backend": "nemo_parakeet",
        },
        "models_requested": [],
        "model_failures": [],
        "models": [],
    }
    all_predictions: dict = {}

    if args.resume or args.force_resume:
        existing_results, existing_predictions, _ = s19e.load_existing_results(out_json)
        if existing_results:
            results = existing_results
            all_predictions = existing_predictions

    models = s19e._dedupe_keep_order(_parse_models_csv(args.models_csv))
    if not models:
        raise RuntimeError("No NeMo models provided via --models_csv")
    active_keys = {pr["mix_key"] for pr in pair_rows}

    _ensure_prediction_stubs(pair_rows, all_predictions)

    existing_models = {m.get("model") for m in results.get("models", []) if isinstance(m, dict)}
    requested = list(results.get("models_requested", []))
    for m in models:
        if m not in requested:
            requested.append(m)
    results["models_requested"] = requested
    if not isinstance(results.get("model_failures"), list):
        results["model_failures"] = []

    print(f"Device: {args.device}")
    print(f"Test manifest rows: {len(rows)}")
    print(f"Total eval pairs: {len(pair_rows)}")
    print(f"NeMo models requested: {len(models)}")

    for model_ref in models:
        model_name = str(model_ref)
        model_results_key = str(model_ref)
        model_already_done = model_results_key in existing_models
        wants_recalc = bool(args.recalc_metrics or args.force_resume)

        if model_already_done and not wants_recalc:
            missing = 0
            for pr in pair_rows:
                pred_map = all_predictions.get(pr["mix_key"], {}).get("predictions", {})
                if model_name not in pred_map:
                    missing += 1
            if missing == 0:
                print(f"Skipping already evaluated NeMo model: {model_ref}")
                continue
            print(f"Evaluating missing pairs for {model_ref}: {missing} remaining")

        print("=" * 80)
        print(f"Evaluating NeMo model: {model_ref}")
        print("=" * 80)
        try:
            if model_already_done and wants_recalc:
                overall, by_cond = s19e.recompute_metrics_from_saved_predictions(
                    all_predictions=all_predictions,
                    model_name=model_name,
                    normalize_mode=args.normalize,
                    active_keys=active_keys,
                )
                overall["model_type"] = "nemo_parakeet"
                overall["skipped"] = int(overall.get("skipped") or 0)
            else:
                overall, by_cond = _evaluate_one_nemo_model(
                    model_ref=model_ref,
                    pair_rows=pair_rows,
                    args=args,
                    all_predictions=all_predictions,
                    active_keys=active_keys,
                    out_json=out_json,
                    results=results,
                    run_args=run_args,
                )
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            if bool(args.skip_model_failures):
                print(f"Model failed and skipped: {model_ref} | {err}")
                results["model_failures"].append({"model": model_ref, "error": err})
                _save_incremental_results_safe(results, all_predictions, out_json, run_args=run_args)
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
            f for f in results.get("model_failures", []) if str(f.get("model")) != model_results_key
        ]
        _save_incremental_results_safe(results, all_predictions, out_json, run_args=run_args)

        print(
            f"samples={overall.get('samples')} "
            f"WER_target={overall.get('wer_micro_target')} "
            f"CER_target={overall.get('cer_micro_target')} "
            f"speed_xrt={overall.get('eval_throughput_xrt')}"
        )

    print("NeMo adapter run complete.")
    s19e.beep()


if __name__ == "__main__":
    main()
