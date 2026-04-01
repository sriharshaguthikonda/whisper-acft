#!/usr/bin/env python3
"""Cloud-first orchestrator for Indian-accent ASR evaluation.

Flow:
1) Resolve private eval-pack from Hugging Face (or local paths from config).
2) Run targetmix protocol:
   - HF models via stage_19e_edge_and_moonshine_targetmix_sweep_with_cer.py
   - NeMo models via stage_19e_nemo_parakeet_adapter.py
3) Run clean protocol via stage_19e_clean_eval_unified.py
4) Build consolidated leaderboard (accuracy + speed composite)
5) Optionally upload result bundle to a private HF dataset repo.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from cloud_eval_common import (
    ModelSpec,
    beep_done,
    ensure_dir,
    load_json,
    load_models_csv,
    resolve_hf_token,
    run_cmd,
    split_models_by_backend,
    write_json,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _cfg_get(cfg: dict, path: list[str], default: Any = None) -> Any:
    cur: Any = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _resolve_path(path_value: Optional[str], base: Optional[Path] = None) -> Optional[Path]:
    if not path_value:
        return None
    p = Path(path_value)
    if p.is_absolute():
        return p
    if base is None:
        return p.resolve()
    return (base / p).resolve()


def _snapshot_download_private_pack(
    cfg: dict,
    token: str,
    repo_root: Path,
    run_root: Path,
) -> Optional[Path]:
    enabled = bool(_cfg_get(cfg, ["hf_eval_pack", "enabled"], False))
    if not enabled:
        return None

    repo_id = str(_cfg_get(cfg, ["hf_eval_pack", "repo_id"], "")).strip()
    if not repo_id:
        raise RuntimeError("hf_eval_pack.enabled=true but hf_eval_pack.repo_id is empty")

    revision = str(_cfg_get(cfg, ["hf_eval_pack", "revision"], "main"))
    repo_type = str(_cfg_get(cfg, ["hf_eval_pack", "repo_type"], "dataset"))
    local_dir_raw = _cfg_get(cfg, ["hf_eval_pack", "local_dir"], "")
    if local_dir_raw:
        local_dir = _resolve_path(str(local_dir_raw), base=repo_root)
    else:
        local_dir = run_root / "hf_eval_pack"
    ensure_dir(local_dir)

    allow_patterns = _cfg_get(cfg, ["hf_eval_pack", "allow_patterns"], None)
    ignore_patterns = _cfg_get(cfg, ["hf_eval_pack", "ignore_patterns"], None)

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for eval-pack download") from exc

    print(f"[hf] Downloading eval-pack repo: {repo_id} (revision={revision})")
    resolved = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        token=token,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
    resolved_path = Path(resolved).resolve()
    print(f"[hf] Eval-pack ready at: {resolved_path}")
    return resolved_path


def _resolve_data_paths(cfg: dict, eval_pack_dir: Optional[Path], repo_root: Path) -> dict[str, Optional[Path]]:
    paths_cfg = _cfg_get(cfg, ["paths"], {}) or {}

    latest_pack_prefix = None
    if eval_pack_dir is not None:
        latest_ptr = eval_pack_dir / "eval_packs" / "LATEST_PACK.json"
        if latest_ptr.exists():
            try:
                blob = json.loads(latest_ptr.read_text(encoding="utf-8"))
                pip = str(blob.get("path_in_repo", "")).strip()
                if pip:
                    latest_pack_prefix = pip.replace("\\", "/").strip("/")
            except Exception:
                latest_pack_prefix = None

    def _expand_latest(rel: str) -> str:
        if not rel:
            return rel
        rr = rel.replace("\\", "/")
        if "LATEST_PACK" in rr and latest_pack_prefix:
            rr = rr.replace("eval_packs/LATEST_PACK", latest_pack_prefix)
            rr = rr.replace("LATEST_PACK", latest_pack_prefix.split("/")[-1])
        return rr

    def from_cfg_or_pack(path_key: str, pack_key: str) -> Optional[Path]:
        direct = _resolve_path(str(paths_cfg.get(path_key, "")).strip(), base=repo_root)
        if direct:
            return direct
        rel = _expand_latest(str(_cfg_get(cfg, ["hf_eval_pack", "files", pack_key], "")).strip())
        if rel and eval_pack_dir is not None:
            return (eval_pack_dir / rel).resolve()
        return None

    resolved = {
        "test_manifest": from_cfg_or_pack("test_manifest", "test_manifest"),
        "speaker_scores_csv": from_cfg_or_pack("speaker_scores_csv", "speaker_scores_csv"),
        "others_dir": from_cfg_or_pack("others_dir", "others_dir"),
        "others_manifest": from_cfg_or_pack("others_manifest", "others_manifest"),
        "pairs_manifest": from_cfg_or_pack("pairs_manifest", "pairs_manifest"),
    }

    required = ["test_manifest", "speaker_scores_csv"]
    for key in required:
        if resolved[key] is None:
            raise RuntimeError(f"Missing required path: {key}. Set config.paths.{key} or hf_eval_pack.files.{key}")
    return resolved


def _filter_models_by_language(models: list[ModelSpec], language_mode: str) -> list[ModelSpec]:
    mode = (language_mode or "").strip().lower()
    if not mode:
        return [m for m in models if m.enabled]
    out: list[ModelSpec] = []
    for m in models:
        if not m.enabled:
            continue
        if (m.language_mode or "").strip().lower() == mode:
            out.append(m)
    return out


def _state_default() -> dict:
    return {
        "started_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "targetmix": {},
        "clean": {},
    }


def _save_state(state_path: Path, state: dict) -> None:
    state["updated_at_utc"] = _utc_now()
    write_json(state_path, state)


def _run_targetmix_hf_one(
    model: ModelSpec,
    cfg: dict,
    data_paths: dict[str, Optional[Path]],
    targetmix_dir: Path,
    python_exe: str,
) -> float:
    script = Path(__file__).resolve().parent / "stage_19e_edge_and_moonshine_targetmix_sweep_with_cer.py"
    out_json = targetmix_dir / "evaluation_results_futo_like_targetmix_sweep.json"

    device = str(_cfg_get(cfg, ["runtime", "device"], "cuda"))
    batch_size = int(_cfg_get(cfg, ["runtime", "batch_size"], 1))
    auto_batch = int(_cfg_get(cfg, ["runtime", "auto_batch"], 0))
    batch_min = int(_cfg_get(cfg, ["runtime", "batch_min"], 1))
    batch_max = int(_cfg_get(cfg, ["runtime", "batch_max"], 8))
    fp16 = int(_cfg_get(cfg, ["runtime", "fp16"], 1))
    mem_low = float(_cfg_get(cfg, ["runtime", "mem_low"], 0.60))
    mem_high = float(_cfg_get(cfg, ["runtime", "mem_high"], 0.88))
    cleanup_interval = int(_cfg_get(cfg, ["runtime", "cleanup_interval"], 50))
    audio_cache_gb = float(_cfg_get(cfg, ["runtime", "audio_cache_gb"], 0.5))
    ffmpeg_path = str(_cfg_get(cfg, ["runtime", "ffmpeg_path"], "ffmpeg"))
    normalize = str(_cfg_get(cfg, ["runtime", "normalize"], "whisper_basic"))
    vad_filter = int(_cfg_get(cfg, ["runtime", "vad_filter"], 0))

    tcfg = _cfg_get(cfg, ["protocols", "targetmix"], {}) or {}

    cmd = [
        python_exe,
        str(script),
        "--test_manifest",
        str(data_paths["test_manifest"]),
        "--speaker_scores_csv",
        str(data_paths["speaker_scores_csv"]),
        "--checkpoint_dir",
        str(targetmix_dir),
        "--out_json",
        str(out_json),
        "--base_model",
        "",
        "--models_csv",
        model.model_ref,
        "--append_checkpoints",
        "0",
        "--percentage",
        str(float(tcfg.get("percentage", 100.0))),
        "--target_percentage",
        str(float(tcfg.get("target_percentage", 100.0))),
        "--target_max",
        str(int(tcfg.get("target_max", 0))),
        "--mix_per_target",
        str(int(tcfg.get("mix_per_target", 5))),
        "--pairing_mode",
        str(tcfg.get("pairing_mode", "round_robin")),
        "--seed",
        str(int(tcfg.get("seed", 42))),
        "--other_offset_mode",
        str(tcfg.get("other_offset_mode", "start")),
        "--other_peak_ratio",
        str(float(tcfg.get("other_peak_ratio", 1.0))),
        "--sweep_snr_db",
        str(tcfg.get("sweep_snr_db", "15,5,0")),
        "--sweep_overlap",
        str(tcfg.get("sweep_overlap", "0.25,0.75,1")),
        "--device",
        device,
        "--batch_size",
        str(batch_size),
        "--auto_batch",
        str(auto_batch),
        "--batch_min",
        str(batch_min),
        "--batch_max",
        str(batch_max),
        "--fp16",
        str(fp16),
        "--mem_low",
        str(mem_low),
        "--mem_high",
        str(mem_high),
        "--cleanup_interval",
        str(cleanup_interval),
        "--audio_cache_gb",
        str(audio_cache_gb),
        "--ffmpeg_path",
        ffmpeg_path,
        "--normalize",
        normalize,
        "--vad_filter",
        str(vad_filter),
        "--resume",
        "--skip_model_failures",
        "1",
    ]

    if bool(tcfg.get("disable_overlap_sweep", False)):
        cmd.append("--disable_overlap_sweep")
    if bool(tcfg.get("force_resume", False)):
        cmd.append("--force_resume")
    if bool(tcfg.get("recalc_metrics", False)):
        cmd.append("--recalc_metrics")

    if data_paths.get("others_dir") is not None:
        cmd.extend(["--others_dir", str(data_paths["others_dir"])])
    if data_paths.get("others_manifest") is not None:
        cmd.extend(["--others_manifest", str(data_paths["others_manifest"])])
    if data_paths.get("pairs_manifest") is not None:
        cmd.extend(["--pairs_manifest", str(data_paths["pairs_manifest"])])

    t0 = time.perf_counter()
    run_cmd(cmd, cwd=Path(__file__).resolve().parent)
    return float(time.perf_counter() - t0)


def _run_targetmix_nemo_one(
    model: ModelSpec,
    cfg: dict,
    data_paths: dict[str, Optional[Path]],
    targetmix_dir: Path,
    python_exe: str,
) -> float:
    script = Path(__file__).resolve().parent / "stage_19e_nemo_parakeet_adapter.py"
    out_json = targetmix_dir / "evaluation_results_futo_like_targetmix_sweep.json"
    tcfg = _cfg_get(cfg, ["protocols", "targetmix"], {}) or {}

    cmd = [
        python_exe,
        str(script),
        "--test_manifest",
        str(data_paths["test_manifest"]),
        "--speaker_scores_csv",
        str(data_paths["speaker_scores_csv"]),
        "--checkpoint_dir",
        str(targetmix_dir),
        "--out_json",
        str(out_json),
        "--models_csv",
        model.model_ref,
        "--percentage",
        str(float(tcfg.get("percentage", 100.0))),
        "--target_percentage",
        str(float(tcfg.get("target_percentage", 100.0))),
        "--target_max",
        str(int(tcfg.get("target_max", 0))),
        "--mix_per_target",
        str(int(tcfg.get("mix_per_target", 5))),
        "--pairing_mode",
        str(tcfg.get("pairing_mode", "round_robin")),
        "--seed",
        str(int(tcfg.get("seed", 42))),
        "--other_offset_mode",
        str(tcfg.get("other_offset_mode", "start")),
        "--other_peak_ratio",
        str(float(tcfg.get("other_peak_ratio", 1.0))),
        "--sweep_snr_db",
        str(tcfg.get("sweep_snr_db", "15,5,0")),
        "--sweep_overlap",
        str(tcfg.get("sweep_overlap", "0.25,0.75,1")),
        "--device",
        str(_cfg_get(cfg, ["runtime", "device"], "cuda")),
        "--batch_size",
        str(int(_cfg_get(cfg, ["runtime", "nemo_batch_size"], _cfg_get(cfg, ["runtime", "batch_size"], 1)))),
        "--infer_num_workers",
        str(int(_cfg_get(cfg, ["runtime", "nemo_infer_num_workers"], 0))),
        "--audio_cache_gb",
        str(float(_cfg_get(cfg, ["runtime", "audio_cache_gb"], 0.5))),
        "--ffmpeg_path",
        str(_cfg_get(cfg, ["runtime", "ffmpeg_path"], "ffmpeg")),
        "--normalize",
        str(_cfg_get(cfg, ["runtime", "normalize"], "whisper_basic")),
        "--vad_filter",
        str(int(_cfg_get(cfg, ["runtime", "vad_filter"], 0))),
        "--resume",
        "--skip_model_failures",
        "1",
    ]

    if bool(tcfg.get("disable_overlap_sweep", False)):
        cmd.append("--disable_overlap_sweep")
    if bool(tcfg.get("force_resume", False)):
        cmd.append("--force_resume")
    if bool(tcfg.get("recalc_metrics", False)):
        cmd.append("--recalc_metrics")

    if data_paths.get("others_dir") is not None:
        cmd.extend(["--others_dir", str(data_paths["others_dir"])])
    if data_paths.get("others_manifest") is not None:
        cmd.extend(["--others_manifest", str(data_paths["others_manifest"])])
    if data_paths.get("pairs_manifest") is not None:
        cmd.extend(["--pairs_manifest", str(data_paths["pairs_manifest"])])

    t0 = time.perf_counter()
    run_cmd(cmd, cwd=Path(__file__).resolve().parent)
    return float(time.perf_counter() - t0)


def _run_clean_protocol(
    cfg: dict,
    models_file: Path,
    data_paths: dict[str, Optional[Path]],
    clean_dir: Path,
    python_exe: str,
) -> None:
    script = Path(__file__).resolve().parent / "stage_19e_clean_eval_unified.py"
    c_cfg = _cfg_get(cfg, ["protocols", "clean"], {}) or {}
    cmd = [
        python_exe,
        str(script),
        "--models_file",
        str(models_file),
        "--test_manifest",
        str(data_paths["test_manifest"]),
        "--speaker_scores_csv",
        str(data_paths["speaker_scores_csv"]),
        "--output_dir",
        str(clean_dir),
        "--language_mode_filter",
        str(_cfg_get(cfg, ["runtime", "language_mode_filter"], "en")),
        "--seed",
        str(int(c_cfg.get("seed", 42))),
        "--percentage",
        str(float(c_cfg.get("percentage", 100.0))),
        "--target_percentage",
        str(float(c_cfg.get("target_percentage", 100.0))),
        "--target_max",
        str(int(c_cfg.get("target_max", 0))),
        "--device",
        str(_cfg_get(cfg, ["runtime", "device"], "cuda")),
        "--batch_size",
        str(int(_cfg_get(cfg, ["runtime", "batch_size"], 1))),
        "--infer_num_workers",
        str(int(_cfg_get(cfg, ["runtime", "nemo_infer_num_workers"], 0))),
        "--audio_cache_gb",
        str(float(_cfg_get(cfg, ["runtime", "audio_cache_gb"], 0.5))),
        "--ffmpeg_path",
        str(_cfg_get(cfg, ["runtime", "ffmpeg_path"], "ffmpeg")),
        "--base_processor_id",
        str(_cfg_get(cfg, ["runtime", "base_processor_id"], "openai/whisper-small.en")),
        "--language",
        str(_cfg_get(cfg, ["runtime", "language"], "en")),
        "--task",
        str(_cfg_get(cfg, ["runtime", "task"], "transcribe")),
        "--num_beams",
        str(int(_cfg_get(cfg, ["runtime", "num_beams"], 5))),
        "--temperature",
        str(float(_cfg_get(cfg, ["runtime", "temperature"], 0.0))),
        "--max_new_tokens",
        str(int(_cfg_get(cfg, ["runtime", "max_new_tokens"], 128))),
        "--moonshine_token_limit_tps",
        str(float(_cfg_get(cfg, ["runtime", "moonshine_token_limit_tps"], 6.5))),
        "--normalize",
        str(_cfg_get(cfg, ["runtime", "normalize"], "whisper_basic")),
        "--resume",
        "--skip_model_failures",
        "1",
    ]
    if bool(c_cfg.get("force_resume", False)):
        cmd.append("--force_resume")
    if bool(c_cfg.get("recalc_metrics", False)):
        cmd.append("--recalc_metrics")

    run_cmd(cmd, cwd=Path(__file__).resolve().parent)


def _load_results_map(results_json: Path) -> dict[str, dict]:
    if not results_json.exists():
        return {}
    try:
        data = json.loads(results_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[str, dict] = {}
    for entry in data.get("models", []) if isinstance(data, dict) else []:
        if not isinstance(entry, dict):
            continue
        model_key = str(entry.get("model", "")).strip()
        if not model_key:
            continue
        out[model_key] = entry.get("metrics_overall", {}) if isinstance(entry.get("metrics_overall"), dict) else {}
    return out


def _sum_audio_sec_from_per_sample(per_sample_json: Path, model_ref: str) -> Optional[float]:
    if not per_sample_json.exists():
        return None
    try:
        payload = json.loads(per_sample_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    candidate_keys = [model_ref]
    try:
        base = Path(model_ref).name
        if base and base not in candidate_keys:
            candidate_keys.append(base)
    except Exception:
        pass
    total = 0.0
    hit = False
    for item in payload:
        if not isinstance(item, dict):
            continue
        preds = item.get("predictions", {})
        if not isinstance(preds, dict):
            continue
        pred = None
        for key in candidate_keys:
            if key in preds:
                pred = preds[key]
                break
        if pred is None:
            continue
        if not isinstance(pred, dict):
            continue
        dur = pred.get("duration_sec_eval")
        if dur is None:
            continue
        total += float(dur)
        hit = True
    return total if hit else None


def _model_lookup(models: list[ModelSpec]) -> dict[str, ModelSpec]:
    return {m.model_ref: m for m in models}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _build_leaderboard(
    cfg: dict,
    models: list[ModelSpec],
    run_root: Path,
    state: dict,
) -> tuple[Path, Path]:
    targetmix_results = run_root / "targetmix" / "evaluation_results_futo_like_targetmix_sweep.json"
    targetmix_per_sample = run_root / "targetmix" / "evaluation_per_sample_predictions_targetmix_sweep.json"
    clean_results = run_root / "clean" / "evaluation_results_clean.json"

    tm_map = _load_results_map(targetmix_results)
    cl_map = _load_results_map(clean_results)
    lookup = _model_lookup(models)

    acc_weight = float(_cfg_get(cfg, ["leaderboard", "accuracy_weight"], 0.8))
    speed_weight = float(_cfg_get(cfg, ["leaderboard", "speed_weight"], 0.2))
    clean_acc_blend = float(_cfg_get(cfg, ["leaderboard", "clean_accuracy_blend"], 0.4))
    targetmix_acc_blend = float(_cfg_get(cfg, ["leaderboard", "targetmix_accuracy_blend"], 0.6))

    rows: list[dict[str, Any]] = []
    for model in models:
        if not model.enabled:
            continue
        ref = model.model_ref
        tm = tm_map.get(ref, {})
        cl = cl_map.get(ref, {})

        tm_wer = _safe_float(tm.get("wer_micro_target"))
        cl_wer = _safe_float(cl.get("wer_micro_target"))
        tm_acc = (1.0 - tm_wer) if tm_wer is not None else None
        cl_acc = (1.0 - cl_wer) if cl_wer is not None else None

        acc_num = 0.0
        acc_den = 0.0
        if tm_acc is not None:
            acc_num += float(targetmix_acc_blend) * tm_acc
            acc_den += float(targetmix_acc_blend)
        if cl_acc is not None:
            acc_num += float(clean_acc_blend) * cl_acc
            acc_den += float(clean_acc_blend)
        accuracy = (acc_num / acc_den) if acc_den > 0.0 else None

        speed = _safe_float(tm.get("eval_throughput_xrt"))
        if speed is None:
            # derive targetmix speed from state timing + per-sample audio seconds
            timing = _cfg_get(state, ["targetmix", ref], {})
            elapsed = _safe_float(timing.get("elapsed_sec")) if isinstance(timing, dict) else None
            audio_sec = _sum_audio_sec_from_per_sample(targetmix_per_sample, ref)
            if elapsed and elapsed > 0.0 and audio_sec and audio_sec > 0.0:
                speed = float(audio_sec / elapsed)
        if speed is None:
            speed = _safe_float(cl.get("eval_throughput_xrt"))

        rows.append(
            {
                "model_ref": ref,
                "model_name": model.model_name,
                "backend": model.backend,
                "language_mode": model.language_mode,
                "wer_targetmix": tm_wer,
                "wer_clean": cl_wer,
                "accuracy_unscaled": accuracy,
                "speed_xrt_unscaled": speed,
            }
        )

    acc_vals = [r["accuracy_unscaled"] for r in rows if r["accuracy_unscaled"] is not None]
    speed_vals = [r["speed_xrt_unscaled"] for r in rows if r["speed_xrt_unscaled"] is not None]
    acc_min = min(acc_vals) if acc_vals else None
    acc_max = max(acc_vals) if acc_vals else None
    speed_min = min(speed_vals) if speed_vals else None
    speed_max = max(speed_vals) if speed_vals else None

    for row in rows:
        acc = row["accuracy_unscaled"]
        spd = row["speed_xrt_unscaled"]
        if acc is None or acc_min is None or acc_max is None or acc_max == acc_min:
            row["accuracy_norm"] = 1.0 if acc is not None else None
        else:
            row["accuracy_norm"] = float((acc - acc_min) / (acc_max - acc_min))

        if spd is None or speed_min is None or speed_max is None or speed_max == speed_min:
            row["speed_norm"] = 1.0 if spd is not None else None
        else:
            row["speed_norm"] = float((spd - speed_min) / (speed_max - speed_min))

        if row["accuracy_norm"] is None and row["speed_norm"] is None:
            row["composite_score"] = None
        elif row["accuracy_norm"] is None:
            row["composite_score"] = float(row["speed_norm"])
        elif row["speed_norm"] is None:
            row["composite_score"] = float(row["accuracy_norm"])
        else:
            row["composite_score"] = (
                float(acc_weight) * float(row["accuracy_norm"]) + float(speed_weight) * float(row["speed_norm"])
            )

    rows.sort(
        key=lambda r: (r["composite_score"] is None, -(r["composite_score"] or -1e9), r["model_ref"])
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    leaderboard_json = run_root / "leaderboard_consolidated.json"
    leaderboard_csv = run_root / "leaderboard_consolidated.csv"
    write_json(leaderboard_json, {"generated_at_utc": _utc_now(), "rows": rows})

    with leaderboard_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "rank",
                "model_ref",
                "model_name",
                "backend",
                "language_mode",
                "wer_targetmix",
                "wer_clean",
                "accuracy_unscaled",
                "speed_xrt_unscaled",
                "accuracy_norm",
                "speed_norm",
                "composite_score",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Leaderboard written: {leaderboard_json} | {leaderboard_csv}")
    return leaderboard_json, leaderboard_csv


def _maybe_upload_results_to_hf(cfg: dict, run_root: Path, token: str) -> None:
    upload_cfg = _cfg_get(cfg, ["results", "upload_to_hf"], {}) or {}
    if not bool(upload_cfg.get("enabled", False)):
        return

    repo_id = str(upload_cfg.get("repo_id", "")).strip()
    if not repo_id:
        raise RuntimeError("results.upload_to_hf.enabled=true but repo_id is empty")
    repo_type = str(upload_cfg.get("repo_type", "dataset")).strip()
    private = bool(upload_cfg.get("private", True))
    path_in_repo = str(upload_cfg.get("path_in_repo", "")).strip()
    commit_message = str(upload_cfg.get("commit_message", "Upload cloud eval results")).strip()

    try:
        from huggingface_hub import HfApi
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for result uploads") from exc

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True, token=token)
    print(f"[hf] Uploading result bundle to {repo_type}:{repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(run_root),
        path_in_repo=path_in_repo or None,
        token=token,
        commit_message=commit_message,
    )
    print("[hf] Result upload complete.")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Cloud eval orchestrator (Kaggle + Colab compatible)")
    ap.add_argument("--config", required=True, type=Path, help="JSON config for cloud eval stack")
    ap.add_argument("--models_file", required=True, type=Path, help="CSV model list contract")
    ap.add_argument("--repo_root", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--run_root", type=Path, default=None, help="Output root for this run")
    ap.add_argument("--hf_token_env", default="HF_TOKEN", help="Preferred env var name for HF token")
    ap.add_argument("--skip_targetmix", action="store_true")
    ap.add_argument("--skip_clean", action="store_true")
    ap.add_argument("--skip_leaderboard", action="store_true")
    ap.add_argument("--skip_upload", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")
    cfg = load_json(args.config)
    repo_root = args.repo_root.resolve()

    run_root = args.run_root.resolve() if args.run_root else None
    if run_root is None:
        default_run_root = _cfg_get(cfg, ["results", "root_dir"], "")
        if default_run_root:
            run_root = _resolve_path(str(default_run_root), base=repo_root)
        else:
            run_root = repo_root / "cloud_eval_runs" / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    ensure_dir(run_root)
    targetmix_dir = ensure_dir(run_root / "targetmix")
    clean_dir = ensure_dir(run_root / "clean")

    print(f"Run root: {run_root}")
    state_path = run_root / "run_state.json"
    state = load_json(state_path, default=_state_default())
    _save_state(state_path, state)

    token = resolve_hf_token(repo_root, env_var=args.hf_token_env)
    if not token:
        raise RuntimeError(
            f"HF token missing. Set env var {args.hf_token_env} (or HF_TOKEN), or place it in {repo_root / '.env'}"
        )
    os.environ["HF_TOKEN"] = token

    eval_pack_dir = _snapshot_download_private_pack(cfg, token, repo_root, run_root)
    data_paths = _resolve_data_paths(cfg, eval_pack_dir, repo_root)

    models = _filter_models_by_language(
        load_models_csv(args.models_file),
        language_mode=str(_cfg_get(cfg, ["runtime", "language_mode_filter"], "en")),
    )
    if not models:
        raise RuntimeError("No enabled models after language filtering")
    hf_models, nemo_models = split_models_by_backend(models)
    python_exe = sys.executable

    write_json(
        run_root / "run_manifest.json",
        {
            "generated_at_utc": _utc_now(),
            "config_path": str(args.config.resolve()),
            "models_file": str(args.models_file.resolve()),
            "data_paths": {k: str(v) if v else None for k, v in data_paths.items()},
            "run_root": str(run_root),
            "targetmix_dir": str(targetmix_dir),
            "clean_dir": str(clean_dir),
            "models_enabled": [m.__dict__ for m in models],
        },
    )

    if not args.skip_targetmix and bool(_cfg_get(cfg, ["protocols", "targetmix", "enabled"], True)):
        print("=== Targetmix protocol ===")
        for spec in hf_models:
            print(f"[targetmix][hf] {spec.model_ref}")
            elapsed = _run_targetmix_hf_one(
                model=spec,
                cfg=cfg,
                data_paths=data_paths,
                targetmix_dir=targetmix_dir,
                python_exe=python_exe,
            )
            state.setdefault("targetmix", {})[spec.model_ref] = {"elapsed_sec": elapsed, "completed_at_utc": _utc_now()}
            _save_state(state_path, state)

        for spec in nemo_models:
            print(f"[targetmix][nemo] {spec.model_ref}")
            elapsed = _run_targetmix_nemo_one(
                model=spec,
                cfg=cfg,
                data_paths=data_paths,
                targetmix_dir=targetmix_dir,
                python_exe=python_exe,
            )
            state.setdefault("targetmix", {})[spec.model_ref] = {"elapsed_sec": elapsed, "completed_at_utc": _utc_now()}
            _save_state(state_path, state)
    else:
        print("Skipping targetmix protocol.")

    if not args.skip_clean and bool(_cfg_get(cfg, ["protocols", "clean", "enabled"], True)):
        print("=== Clean protocol ===")
        _run_clean_protocol(
            cfg=cfg,
            models_file=args.models_file.resolve(),
            data_paths=data_paths,
            clean_dir=clean_dir,
            python_exe=python_exe,
        )
        state["clean"]["completed_at_utc"] = _utc_now()
        _save_state(state_path, state)
    else:
        print("Skipping clean protocol.")

    if not args.skip_leaderboard:
        _build_leaderboard(cfg, models, run_root, state)
    else:
        print("Skipping leaderboard build.")

    if not args.skip_upload:
        _maybe_upload_results_to_hf(cfg, run_root, token)
    else:
        print("Skipping HF result upload.")

    print("Cloud evaluation orchestration complete.")
    beep_done()


if __name__ == "__main__":
    main()
