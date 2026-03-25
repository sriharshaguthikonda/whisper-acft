#!/usr/bin/env python3
"""Benchmark Stage-19E model efficiency: accuracy / (resources * time).

Inputs:
- Stage-19E results JSON (for accuracy + parameter counts)
- Stage-19E base pairs JSONL (for audio files to transcribe)

Outputs:
- CSV with timing and efficiency metrics
- Printed ranking tables
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_pairs_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def unique_audio_paths(pairs: List[dict]) -> List[Path]:
    s = set()
    for p in pairs:
        t = p.get("target_audio_path")
        o = p.get("other_audio_path")
        if t:
            s.add(str(t))
        if o:
            s.add(str(o))
    return [Path(x) for x in sorted(s)]


def simple_resample(x: np.ndarray, sr_in: int, sr_out: int = 16000) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    dur = float(len(x)) / float(sr_in)
    n_out = max(1, int(round(dur * sr_out)))
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False, dtype=np.float64)
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float64)
    y = np.interp(xq, xp, x.astype(np.float64, copy=False))
    return y.astype(np.float32)


def load_audio_16k(path: Path) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = np.asarray(wav, dtype=np.float32)
    if sr != 16000:
        wav = simple_resample(wav, sr_in=int(sr), sr_out=16000)
    return wav


def maybe_set_forced_decoder_ids(model, processor, language: str, task: str) -> None:
    try:
        prompt_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        if hasattr(model, "generation_config") and hasattr(model.generation_config, "forced_decoder_ids"):
            model.generation_config.forced_decoder_ids = prompt_ids
    except Exception:
        pass


def maybe_pad_whisper_inputs(model_id: str, model_type: str, model_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    is_whisper = ("whisper" in (model_type or "").lower()) or ("whisper" in model_id.lower())
    if not is_whisper:
        return model_inputs
    feats = model_inputs.get("input_features")
    if feats is None or feats.ndim != 3:
        return model_inputs
    cur = int(feats.shape[-1])
    if cur < 3000:
        pad = torch.zeros((feats.shape[0], feats.shape[1], 3000 - cur), dtype=feats.dtype, device=feats.device)
        model_inputs["input_features"] = torch.cat([feats, pad], dim=-1)
    elif cur > 3000:
        model_inputs["input_features"] = feats[..., :3000]
    return model_inputs


def time_model(
    model_id: str,
    audios: List[np.ndarray],
    device: str,
    language: str,
    task: str,
    num_beams: int,
    max_new_tokens: int,
) -> Dict[str, float]:
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
    model.eval()
    if str(device).startswith("cuda"):
        try:
            model.half()
        except Exception:
            pass
    model_type = str(getattr(getattr(model, "config", None), "model_type", "") or "").lower()
    maybe_set_forced_decoder_ids(model, processor, language=language, task=task)
    try:
        if hasattr(model, "generation_config") and hasattr(model.generation_config, "max_length"):
            model.generation_config.max_length = None
    except Exception:
        pass

    gen_kwargs = {"num_beams": int(num_beams), "max_new_tokens": int(max_new_tokens), "temperature": 0.0}
    model_dtype = None
    try:
        model_dtype = next(model.parameters()).dtype
    except Exception:
        model_dtype = None

    # Warmup
    with torch.inference_mode():
        a0 = audios[0]
        proc_kwargs = {"sampling_rate": 16000, "return_tensors": "pt"}
        if "whisper" in model_type or "whisper" in model_id.lower():
            proc_kwargs.update({"padding": "max_length", "max_length": 3000, "truncation": True})
        else:
            proc_kwargs.update({"padding": True})
        try:
            inputs = processor([a0], return_attention_mask=True, **proc_kwargs)
        except TypeError:
            inputs = processor([a0], **proc_kwargs)
        model_inputs = {}
        for k, v in inputs.items():
            if not torch.is_tensor(v):
                continue
            t = v.to(device)
            if model_dtype is not None and t.is_floating_point():
                t = t.to(dtype=model_dtype)
            model_inputs[k] = t
        model_inputs = maybe_pad_whisper_inputs(model_id, model_type, model_inputs)
        _ = model.generate(**model_inputs, **gen_kwargs)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    times: List[float] = []
    total_audio_sec = 0.0

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for wav in audios:
            total_audio_sec += float(len(wav)) / 16000.0
            proc_kwargs = {"sampling_rate": 16000, "return_tensors": "pt"}
            if "whisper" in model_type or "whisper" in model_id.lower():
                proc_kwargs.update({"padding": "max_length", "max_length": 3000, "truncation": True})
            else:
                proc_kwargs.update({"padding": True})
            try:
                inputs = processor([wav], return_attention_mask=True, **proc_kwargs)
            except TypeError:
                inputs = processor([wav], **proc_kwargs)
            model_inputs = {}
            for k, v in inputs.items():
                if not torch.is_tensor(v):
                    continue
                t = v.to(device)
                if model_dtype is not None and t.is_floating_point():
                    t = t.to(dtype=model_dtype)
                model_inputs[k] = t
            model_inputs = maybe_pad_whisper_inputs(model_id, model_type, model_inputs)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(**model_inputs, **gen_kwargs)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            _ = processor.batch_decode(out.detach().cpu(), skip_special_tokens=True)
            times.append(float(t1 - t0))

    peak_vram_gb = float(torch.cuda.max_memory_allocated() / (1024**3)) if device.startswith("cuda") else math.nan
    t = np.asarray(times, dtype=np.float64)
    result = {
        "n_samples_speed": int(len(times)),
        "latency_mean_s": float(np.mean(t)),
        "latency_p50_s": float(np.quantile(t, 0.50)),
        "latency_p90_s": float(np.quantile(t, 0.90)),
        "audio_total_s": float(total_audio_sec),
        "wall_total_s": float(np.sum(t)),
        "rtf_mean": float(np.sum(t) / max(1e-9, total_audio_sec)),
        "peak_vram_gb": peak_vram_gb,
    }

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def main() -> None:
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", type=Path, required=True)
    ap.add_argument("--pairs_jsonl", type=Path, required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--sample_count", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--language", type=str, default="en")
    ap.add_argument("--task", type=str, default="transcribe")
    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    res = read_json(args.results_json)
    models = [m for m in res.get("models", []) if isinstance(m, dict)]

    pairs = load_pairs_jsonl(args.pairs_jsonl)
    paths = [p for p in unique_audio_paths(pairs) if p.exists()]
    if not paths:
        raise RuntimeError("No audio files found from pairs JSONL.")

    rnd = random.Random(int(args.seed))
    rnd.shuffle(paths)
    paths = paths[: max(1, int(args.sample_count))]
    audios = [load_audio_16k(p) for p in paths]

    rows: List[dict] = []
    for m in models:
        model_id = str(m.get("model"))
        overall = m.get("metrics_overall", {}) or {}
        print(f"[bench] {model_id}")
        try:
            timing = time_model(
                model_id=model_id,
                audios=audios,
                device=str(args.device),
                language=str(args.language),
                task=str(args.task),
                num_beams=int(args.num_beams),
                max_new_tokens=int(args.max_new_tokens),
            )
        except Exception as e:
            print(f"[bench] skip {model_id}: {type(e).__name__}: {e}")
            continue

        wer_t = float(overall.get("wer_micro_target"))
        win = float(overall.get("win_rate_target_closer"))
        params = float(overall.get("model_num_params"))
        params_m = params / 1_000_000.0
        lat = float(timing["latency_mean_s"])

        acc_invwer = 1.0 / (1.0 + max(0.0, wer_t))
        eff_invwer = acc_invwer / max(1e-12, params_m * lat)
        eff_win = win / max(1e-12, params_m * lat)

        rows.append(
            {
                "model": model_id,
                "wer_target": wer_t,
                "win_rate": win,
                "params_m": params_m,
                **timing,
                "acc_invwer": acc_invwer,
                "eff_invwer_per_param_time": eff_invwer,
                "eff_winrate_per_param_time": eff_win,
            }
        )

    import pandas as pd  # local import to keep startup minimal

    df = pd.DataFrame(rows)
    df = df.sort_values("eff_winrate_per_param_time", ascending=False).reset_index(drop=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False, encoding="utf-8")

    print("\nTop by eff_winrate_per_param_time:")
    print(df[["model", "eff_winrate_per_param_time", "win_rate", "params_m", "latency_mean_s"]].head(12).to_string(index=False))
    print("\nTop by eff_invwer_per_param_time:")
    print(
        df.sort_values("eff_invwer_per_param_time", ascending=False)[
            ["model", "eff_invwer_per_param_time", "acc_invwer", "wer_target", "params_m", "latency_mean_s"]
        ]
        .head(12)
        .to_string(index=False)
    )
    print(f"\nWrote: {args.out_csv}")


if __name__ == "__main__":
    main()
