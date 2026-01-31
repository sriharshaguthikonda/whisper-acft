#!/usr/bin/env python3
"""stage_19d_plot_eval_charts.py

Make charts + a compact leaderboard for Stage 19c sweep outputs.

Input
-----
A JSON produced by:
  stage_19c_specific_target_advanced_futo_like_evaluate_targetmix_sweep_with_cer.py

Outputs
-------
- PNG charts (bar charts, pareto, per-model heatmaps, SNR curves)
- model_summary.csv
- evaluation_charts_bundle.zip

Usage (Windows PowerShell)
-------------------------
I:\Whisper-training-env\Scripts\python.exe i:\whisper-acft\stage_19d_plot_eval_charts.py `
  --in_json "I:\Stage_2_shuffle_Dynamic_n_ctx_checkpoints_partialctx6\evaluation_results_futo_like_targetmix_sweep.json" `
  --out_dir "I:\Stage_2_shuffle_Dynamic_n_ctx_checkpoints_partialctx6\eval_charts"

Notes
-----
- No seaborn.
- No explicit colour choices.
- One plot per figure (no subplots).
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _beep() -> None:
    try:
        import winsound

        winsound.Beep(1000, 250)
        winsound.Beep(1400, 200)
    except Exception:
        print("\a", end="", flush=True)


def short_model_name(s: str) -> str:
    s = str(s).replace("\\", "/")
    return s.split("/")[-1]


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return s
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def build_summary_df(models: list[dict]) -> pd.DataFrame:
    rows = []
    for m in models:
        mo = m.get("metrics_overall", {}) or {}
        name = m.get("model", "unknown")

        mbc = m.get("metrics_by_condition", {}) or {}
        win_rates, margins = [], []
        for _cid, met in mbc.items():
            if not isinstance(met, dict):
                continue
            w = met.get("win_rate_target_closer")
            mar = met.get("avg_margin_other_minus_target")
            if w is not None:
                win_rates.append(float(w))
            if mar is not None:
                margins.append(float(mar))

        row = {
            "model": name,
            "model_short": short_model_name(name),
            "samples": mo.get("samples"),
            "wer_micro_target": mo.get("wer_micro_target"),
            "cer_micro_target": mo.get("cer_micro_target"),
            "wer_micro_other": mo.get("wer_micro_other"),
            "cer_micro_other": mo.get("cer_micro_other"),
            "win_rate_target_closer": mo.get("win_rate_target_closer"),
            "avg_margin_other_minus_target": mo.get("avg_margin_other_minus_target"),
            "avg_margin_cer_other_minus_target": mo.get("avg_margin_cer_other_minus_target"),
            "win_rate_worst_case": min(win_rates) if win_rates else None,
            "margin_worst_case": min(margins) if margins else None,
            "n_conditions": len(win_rates),
            "skipped": mo.get("skipped"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Composite ranking: separation + robustness, penalise target WER
    df["separation_score"] = (
        zscore(df["win_rate_target_closer"]).fillna(0)
        + zscore(df["avg_margin_other_minus_target"]).fillna(0)
        - zscore(df["wer_micro_target"]).fillna(0)
    )
    df = df.sort_values("separation_score", ascending=False).reset_index(drop=True)
    return df


def build_long_df(data: dict) -> pd.DataFrame:
    cond_meta = {
        c["cond_id"]: (float(c["snr_db"]), float(c["overlap"]))
        for c in (data.get("conditions") or [])
        if isinstance(c, dict) and "cond_id" in c
    }

    long_rows = []
    for m in data.get("models") or []:
        model = m.get("model", "unknown")
        mbc = m.get("metrics_by_condition", {}) or {}
        for cid, met in mbc.items():
            if cid not in cond_meta or not isinstance(met, dict):
                continue
            snr, ov = cond_meta[cid]
            long_rows.append(
                {
                    "model": model,
                    "model_short": short_model_name(model),
                    "cond_id": cid,
                    "snr_db": snr,
                    "overlap": ov,
                    "samples": float(met.get("samples") or 0),
                    "win_rate": met.get("win_rate_target_closer"),
                    "margin": met.get("avg_margin_other_minus_target"),
                    "wer_t": met.get("wer_micro_target"),
                    "wer_o": met.get("wer_micro_other"),
                }
            )

    return pd.DataFrame(long_rows)


def heatmap_for_model(
    data: dict,
    model_entry: dict,
    out_dir: Path,
    metric_key: str,
    title_suffix: str,
    cbar_label: str,
) -> None:
    conds = data.get("conditions") or []
    snr_vals = sorted({float(c["snr_db"]) for c in conds})
    ov_vals = sorted({float(c["overlap"]) for c in conds})

    mbc = model_entry.get("metrics_by_condition", {}) or {}
    grid = np.full((len(snr_vals), len(ov_vals)), np.nan, dtype="float64")

    for c in conds:
        cid = c["cond_id"]
        snr = float(c["snr_db"])
        ov = float(c["overlap"])
        met = mbc.get(cid, {})
        if isinstance(met, dict) and met.get(metric_key) is not None:
            i = snr_vals.index(snr)
            j = ov_vals.index(ov)
            grid[i, j] = float(met[metric_key])

    ms = short_model_name(model_entry.get("model", "unknown"))
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, aspect="auto", interpolation="nearest")
    plt.colorbar(label=cbar_label)
    plt.yticks(range(len(snr_vals)), [f"{v:+g} dB" for v in snr_vals])
    plt.xticks(range(len(ov_vals)), [f"{v:.2f}" for v in ov_vals])
    plt.xlabel("Overlap placement ratio (0=start, 1=end)")
    plt.ylabel("SNR (TARGET over OTHER)")
    plt.title(f"{ms}: {title_suffix}")
    save_fig(out_dir / f"heatmap_{metric_key}__{ms}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    in_path = Path(args.in_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(in_path.read_text(encoding="utf-8"))

    df = build_summary_df(data.get("models") or [])
    summary_csv = out_dir / "model_summary.csv"
    df[
        [
            "model",
            "samples",
            "win_rate_target_closer",
            "avg_margin_other_minus_target",
            "wer_micro_target",
            "wer_micro_other",
            "win_rate_worst_case",
            "margin_worst_case",
            "separation_score",
        ]
    ].to_csv(summary_csv, index=False)

    # Bar charts
    plt.figure(figsize=(10, 5))
    plt.bar(df["model_short"], df["win_rate_target_closer"])
    plt.ylabel("Win rate: P(WER_target < WER_other)")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.title("Overall 'target is closer' win rate (higher is better)")
    save_fig(out_dir / "01_overall_win_rate.png")

    plt.figure(figsize=(10, 5))
    plt.bar(df["model_short"], df["avg_margin_other_minus_target"])
    plt.ylabel("Avg margin (WER_other - WER_target)")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.title("Overall separation margin (higher is better)")
    save_fig(out_dir / "02_overall_margin.png")

    plt.figure(figsize=(10, 5))
    plt.bar(df["model_short"], df["wer_micro_target"])
    plt.ylabel("WER vs TARGET transcript (micro)")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.title("Target WER (lower is better)")
    save_fig(out_dir / "03_target_wer.png")

    # Pareto scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(df["wer_micro_target"], df["avg_margin_other_minus_target"])
    plt.xlabel("WER_target (lower is better)")
    plt.ylabel("Avg margin (higher is better)")
    plt.title("Pareto view: accuracy vs separation")
    for _, r in df.iterrows():
        plt.annotate(
            r["model_short"],
            (float(r["wer_micro_target"]), float(r["avg_margin_other_minus_target"])),
            fontsize=8,
        )
    save_fig(out_dir / "04_pareto_targetwer_vs_margin.png")

    # SNR curves
    df_long = build_long_df(data)
    if not df_long.empty:
        df_snr = (
            df_long.groupby(["model_short", "snr_db"], as_index=False)
            .agg(win_rate=("win_rate", "mean"), margin=("margin", "mean"))
            .sort_values(["model_short", "snr_db"])
        )

        plt.figure(figsize=(8, 6))
        for ms in df_snr["model_short"].unique():
            s = df_snr[df_snr["model_short"] == ms]
            plt.plot(s["snr_db"], s["win_rate"], marker="o", label=ms)
        plt.xlabel("SNR (TARGET over OTHER), dB")
        plt.ylabel("Mean win rate (over overlaps)")
        plt.title("Win rate vs SNR (mean across overlaps)")
        plt.legend()
        save_fig(out_dir / "05_winrate_vs_snr.png")

        plt.figure(figsize=(8, 6))
        for ms in df_snr["model_short"].unique():
            s = df_snr[df_snr["model_short"] == ms]
            plt.plot(s["snr_db"], s["margin"], marker="o", label=ms)
        plt.xlabel("SNR (TARGET over OTHER), dB")
        plt.ylabel("Mean margin (WER_other - WER_target)")
        plt.title("Separation margin vs SNR (mean across overlaps)")
        plt.legend()
        save_fig(out_dir / "06_margin_vs_snr.png")

    # Per-model heatmaps
    for m in data.get("models") or []:
        heatmap_for_model(
            data,
            m,
            out_dir,
            metric_key="win_rate_target_closer",
            title_suffix="win rate by SNR × overlap",
            cbar_label="Win rate",
        )
        heatmap_for_model(
            data,
            m,
            out_dir,
            metric_key="avg_margin_other_minus_target",
            title_suffix="margin by SNR × overlap",
            cbar_label="Avg margin (WER_other - WER_target)",
        )

    # Zip bundle
    zip_out = out_dir.parent / "evaluation_charts_bundle.zip"
    with zipfile.ZipFile(zip_out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fp in sorted(out_dir.glob("*.png")):
            z.write(fp, arcname=f"eval_charts/{fp.name}")
        z.write(summary_csv, arcname="eval_charts/model_summary.csv")

    print("Wrote:", out_dir)
    print("Bundle:", zip_out)
    _beep()


if __name__ == "__main__":
    main()
