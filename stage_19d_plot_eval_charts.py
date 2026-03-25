#!/usr/bin/env python3
r"""stage_19d_plot_eval_charts.py

Make charts + a compact leaderboard for Stage 19c sweep outputs.

Input
-----
A JSON produced by:
  stage_19c_specific_target_advanced_futo_like_evaluate_targetmix_sweep_with_cer.py

Outputs
-------
- Image charts (format selectable via --img_format)
- model_summary.csv

Usage (Windows PowerShell)
-------------------------
I:\Whisper-training-env\Scripts\python.exe i:\whisper-acft\stage_19d_plot_eval_charts.py `
  --in_json "I:\Stage_17_aug_futo_wer_rank64_dora_dyn_ctx_chkpts_small_en_26\evaluation_per_sample_predictions_targetmix_sweep.json" `
  --out_dir "I:\Stage_17_aug_futo_wer_rank64_dora_dyn_ctx_chkpts_small_en_26"

Notes
-----
- No seaborn.
- Dark theme plots (explicit styling applied).
- Heatmaps use VIBGYOR; red = worse, violet = better.
- One plot per figure (no subplots).
- Use --img_format webp/jpg for smaller files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HEATMAP_CMAP = "rainbow_r"  # VIBGYOR with red = low/bad, violet = high/good

IMG_FORMAT = "png"
IMG_QUALITY = 90
IMG_DPI = 200

def _normalize_format(fmt: str) -> str:
    fmt = (fmt or "png").strip().lower()
    if fmt == "jpeg":
        return "jpg"
    return fmt


def _with_format(path: Path, fmt: str) -> Path:
    fmt = _normalize_format(fmt)
    ext = f".{fmt}"
    if path.suffix.lower() != ext:
        return path.with_suffix(ext)
    return path

def _extract_per_sample_meta(data: object) -> dict | None:
    if isinstance(data, dict):
        if "__meta__" in data and isinstance(data.get("__meta__"), dict):
            meta = data.get("__meta__")
            if isinstance(meta.get("run_args"), dict):
                return meta.get("run_args")
            return meta
        if isinstance(data.get("run_args"), dict):
            return data.get("run_args")
        items = data.get("items") or data.get("samples")
        if isinstance(items, list):
            return _extract_per_sample_meta(items)
        return None

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            if "__meta__" in item and isinstance(item.get("__meta__"), dict):
                meta = item.get("__meta__")
                if isinstance(meta.get("run_args"), dict):
                    return meta.get("run_args")
                return meta
            if "run_args" in item and "mix_key" not in item and isinstance(item.get("run_args"), dict):
                return item.get("run_args")
    return None


def _resolve_results_json(in_path: Path, data: object) -> Path | None:
    meta = _extract_per_sample_meta(data)
    if isinstance(meta, dict) and isinstance(meta.get("args"), dict):
        out_json = meta.get("args", {}).get("out_json")
        if out_json:
            p = Path(str(out_json))
            if p.exists():
                return p
    fallback = in_path.parent / "evaluation_results_futo_like_targetmix_sweep.json"
    if fallback.exists():
        return fallback
    return None


def _beep() -> None:
    try:
        import winsound

        winsound.Beep(1000, 250)
        winsound.Beep(1400, 200)
    except Exception:
        print("\a", end="", flush=True)


def short_model_name(s: str) -> str:
    s = str(s).replace("\\", "/")
    name = s.split("/")[-1]
    # Replace invalid Windows filename characters
    for char in ['<', '>', ':', '"', '|', '?', '*']:
        name = name.replace(char, '_')
    return name


def display_model_name(s: str, max_len: int = 24) -> str:
    """Short, human-friendly label for plots."""
    import re

    base = short_model_name(s)
    match = re.search(r"model_epoch_(\d+)", base)
    if match:
        return f"ep{int(match.group(1)):06d}"
    if len(base) <= max_len:
        return base
    keep = max(4, (max_len - 3) // 2)
    return f"{base[:keep]}...{base[-keep:]}"


def make_unique_labels(labels: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for lab in labels:
        if lab not in seen:
            seen[lab] = 1
            out.append(lab)
        else:
            seen[lab] += 1
            out.append(f"{lab}-{seen[lab]}")
    return out


def get_epoch_number(model_name: str) -> int:
    """Extract epoch number from model name for sorting."""
    import re
    # Look for pattern like model_epoch_000000
    match = re.search(r'model_epoch_(\d+)', model_name)
    if match:
        return int(match.group(1))
    # For base model, return -1 so it comes first
    return -1 if 'acft-whisper' in model_name else 999999


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return s
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fmt = _normalize_format(IMG_FORMAT)
    out_path = _with_format(path, fmt)
    face = plt.rcParams.get("figure.facecolor", "black")
    save_kwargs = {"dpi": IMG_DPI, "facecolor": face, "format": "jpeg" if fmt == "jpg" else fmt}
    if fmt in {"jpg", "webp"}:
        save_kwargs["pil_kwargs"] = {"quality": int(IMG_QUALITY), "optimize": True}
    try:
        plt.savefig(out_path, **save_kwargs)
    except Exception as e:
        if fmt != "png":
            print(f"[warn] savefig failed for format '{fmt}': {e}. Falling back to png.")
            out_path = _with_format(path, "png")
            plt.savefig(out_path, dpi=IMG_DPI, facecolor=face, format="png")
        else:
            raise
    plt.close()


def normalize_score(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().all():
        return s
    lo = s.min()
    hi = s.max()
    if pd.isna(lo) or pd.isna(hi) or (hi - lo) == 0:
        return pd.Series(0.5, index=s.index)
    if higher_is_better:
        return (s - lo) / (hi - lo)
    return (hi - s) / (hi - lo)


def pick_top_models(df: pd.DataFrame, top_k: int = 8) -> list[str]:
    df_sorted = df.sort_values("separation_score", ascending=False)
    top = df_sorted["model_short"].head(top_k).tolist()
    base = df_sorted[df_sorted["epoch_number"] == -1]["model_short"].tolist()
    for b in base:
        if b not in top:
            top.insert(0, b)
    return top


def ranked_barh(
    df: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    out_path: Path,
    higher_is_better: bool = True,
    top_k: int | None = None,
    label_col: str = "model_short",
) -> None:
    s = df[[label_col, metric]].copy()
    s[metric] = pd.to_numeric(s[metric], errors="coerce")
    s = s.dropna()
    if s.empty:
        return
    s = s.sort_values(metric, ascending=not higher_is_better)
    if top_k:
        s = s.head(top_k)
    plt.figure(figsize=(9, max(4, 0.35 * len(s))))
    plt.barh(s[label_col], s[metric])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.gca().invert_yaxis()
    save_fig(out_path)


def boxplot_by_model(
    df_long: pd.DataFrame,
    metric: str,
    model_order: list[str],
    label_map: dict[str, str],
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    if df_long.empty:
        return
    data = []
    labels = []
    for m in model_order:
        vals = pd.to_numeric(
            df_long.loc[df_long["model_short"] == m, metric], errors="coerce"
        ).dropna()
        if len(vals) == 0:
            continue
        data.append(vals.to_numpy())
        labels.append(label_map.get(m, m))
    if not data:
        return
    plt.figure(figsize=(9, max(4, 0.3 * len(labels))))
    plt.boxplot(data, tick_labels=labels, vert=False, showfliers=False)
    plt.xlabel(xlabel)
    plt.title(title)
    save_fig(out_path)


def bump_rank_chart(
    df_long: pd.DataFrame,
    data: dict,
    metric: str,
    title: str,
    out_path: Path,
    higher_is_better: bool = True,
    top_models: list[str] | None = None,
    label_map: dict[str, str] | None = None,
) -> None:
    if df_long.empty:
        return
    conds = data.get("conditions") or []
    if not conds:
        return
    conds_sorted = sorted(conds, key=lambda c: (float(c["snr_db"]), float(c["overlap"])))
    cond_ids = [c["cond_id"] for c in conds_sorted]
    cond_labels = [
        f"{float(c['snr_db']):+g}dB ov{float(c['overlap']):.2f}" for c in conds_sorted
    ]

    pivot = df_long.pivot_table(
        index="model_short", columns="cond_id", values=metric, aggfunc="mean"
    )
    pivot = pivot.reindex(columns=cond_ids)
    ranks = pivot.rank(axis=0, ascending=not higher_is_better, method="min")
    if top_models:
        ordered = [m for m in top_models if m in ranks.index]
        ranks = ranks.loc[ordered]
    if ranks.empty:
        return

    x = np.arange(len(cond_ids))
    plt.figure(figsize=(10, 6))
    for model in ranks.index:
        label = label_map.get(model, model) if label_map else model
        plt.plot(x, ranks.loc[model].values, marker="o", label=label)
    plt.gca().invert_yaxis()
    plt.xticks(x, cond_labels, rotation=30, ha="right")
    plt.ylabel("Rank (1 = best)")
    plt.title(title)
    plt.legend()
    save_fig(out_path)


def parallel_scorecard(
    df: pd.DataFrame,
    metrics: list[tuple[str, str, bool]],
    top_models: list[str],
    out_path: Path,
    label_map: dict[str, str] | None = None,
) -> None:
    if df.empty:
        return
    norm_cols = {}
    for col, label, higher_is_better in metrics:
        norm_cols[label] = normalize_score(df[col], higher_is_better)
    norm_df = pd.DataFrame(norm_cols, index=df["model_short"])
    ordered = [m for m in top_models if m in norm_df.index]
    norm_df = norm_df.loc[ordered]
    if norm_df.empty:
        return
    labels = list(norm_df.columns)
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 6))
    for model in norm_df.index:
        label = label_map.get(model, model) if label_map else model
        plt.plot(x, norm_df.loc[model].values, marker="o", label=label)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Normalized score (1 = best)")
    plt.title("Scorecard (normalized across models)")
    plt.legend()
    save_fig(out_path)


def grouped_bar_two_metrics(
    df: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    s = df[["model_label", metric_a, metric_b]].copy()
    s[metric_a] = pd.to_numeric(s[metric_a], errors="coerce")
    s[metric_b] = pd.to_numeric(s[metric_b], errors="coerce")
    s = s.dropna()
    if s.empty:
        return
    x = np.arange(len(s))
    width = 0.38
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, s[metric_a], width, label=label_a)
    plt.bar(x + width / 2, s[metric_b], width, label=label_b)
    plt.ylabel(ylabel)
    plt.xlabel("Model")
    plt.xticks(x, s["model_label"], rotation=30, ha="right")
    plt.title(title)
    plt.legend()
    save_fig(out_path)


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

    # Add epoch number for sorting
    df["epoch_number"] = df["model"].apply(get_epoch_number)
    df["model_label"] = make_unique_labels(df["model"].apply(display_model_name).tolist())
    
    # Composite ranking: separation + robustness, penalise target WER
    df["separation_score"] = (
        zscore(df["win_rate_target_closer"]).fillna(0)
        + zscore(df["avg_margin_other_minus_target"]).fillna(0)
        - zscore(df["wer_micro_target"]).fillna(0)
    )
    
    # Sort by epoch number (ascending) to show training progression
    df = df.sort_values("epoch_number", ascending=True).reset_index(drop=True)
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
                    "cer_t": met.get("cer_micro_target"),
                    "cer_o": met.get("cer_micro_other"),
                    "margin_cer": met.get("avg_margin_cer_other_minus_target"),
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
    display_label: str | None = None,
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

    if np.isnan(grid).all():
        return

    ms = short_model_name(model_entry.get("model", "unknown"))
    label = display_label or ms
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, aspect="auto", interpolation="nearest", cmap=HEATMAP_CMAP)
    plt.colorbar(label=cbar_label)
    plt.yticks(range(len(snr_vals)), [f"{v:+g} dB" for v in snr_vals])
    plt.xticks(range(len(ov_vals)), [f"{v:.2f}" for v in ov_vals])
    plt.xlabel("Overlap placement ratio (0=start, 1=end)")
    plt.ylabel("SNR (TARGET over OTHER)")
    plt.title(f"{label}: {title_suffix}")
    save_fig(out_dir / f"heatmap_{metric_key}__{ms}.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--no_cer", action="store_true",
                    help="Disable CER charts (CER charts are enabled by default).")
    ap.add_argument("--img_format", default="webp", choices=["png", "jpg", "jpeg", "webp"],
                    help="Image format for plots (default: webp).")
    ap.add_argument("--img_quality", type=int, default=90,
                    help="Lossy image quality for jpg/webp (1-100).")
    ap.add_argument("--img_dpi", type=int, default=200,
                    help="DPI for saved figures.")
    args = ap.parse_args()

    in_path = Path(args.in_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    global IMG_FORMAT, IMG_QUALITY, IMG_DPI
    IMG_FORMAT = args.img_format
    IMG_QUALITY = max(1, min(100, int(args.img_quality)))
    IMG_DPI = max(50, int(args.img_dpi))

    # Dark theme for all plots
    plt.style.use("dark_background")
    plt.rcParams.update(
        {
            "figure.facecolor": "#111316",
            "axes.facecolor": "#111316",
            "savefig.facecolor": "#111316",
            "savefig.edgecolor": "#111316",
            "axes.edgecolor": "#cccccc",
            "axes.labelcolor": "#e6e6e6",
            "xtick.color": "#e6e6e6",
            "ytick.color": "#e6e6e6",
            "text.color": "#e6e6e6",
            "grid.color": "#444444",
            "grid.alpha": 0.3,
            "legend.facecolor": "#111316",
            "legend.edgecolor": "#444444",
        }
    )

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if isinstance(data, list) or (isinstance(data, dict) and "models" not in data):
        results_path = _resolve_results_json(in_path, data)
        if results_path is None:
            raise SystemExit(
                "Input JSON looks like per-sample predictions. "
                "Could not find the results JSON. "
                "Pass --in_json pointing at evaluation_results_futo_like_targetmix_sweep.json."
            )
        print(f"Detected per-sample JSON; loading results from: {results_path}")
        data = json.loads(results_path.read_text(encoding="utf-8"))
    include_cer = not bool(args.no_cer)

    df = build_summary_df(data.get("models") or [])
    label_map = dict(zip(df["model_short"], df["model_label"]))
    summary_csv = out_dir / "model_summary.csv"
    df[
        [
            "model",
            "model_label",
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

    top_models = pick_top_models(df, top_k=8)

    # Bar charts (WER)
    plt.figure(figsize=(10, 5))
    plt.bar(df["model_label"], df["win_rate_target_closer"])
    plt.ylabel("Win rate: P(WER_target < WER_other)")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.title("Overall 'target is closer' win rate (higher is better)")
    save_fig(out_dir / "01_overall_win_rate.png")

    plt.figure(figsize=(10, 5))
    plt.bar(df["model_label"], df["avg_margin_other_minus_target"])
    plt.ylabel("Avg margin (WER_other - WER_target)")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.title("Overall separation margin (higher is better)")
    save_fig(out_dir / "02_overall_margin.png")

    plt.figure(figsize=(10, 5))
    plt.bar(df["model_label"], df["wer_micro_target"])
    plt.ylabel("WER vs TARGET transcript (micro)")
    plt.xlabel("Model")
    plt.xticks(rotation=30, ha="right")
    plt.title("Target WER (lower is better)")
    save_fig(out_dir / "03_target_wer.png")

    # Pareto scatter (WER)
    plt.figure(figsize=(8, 6))
    plt.scatter(df["wer_micro_target"], df["avg_margin_other_minus_target"])
    plt.xlabel("WER_target (lower is better)")
    plt.ylabel("Avg margin (higher is better)")
    plt.title("Pareto view: accuracy vs separation")
    for _, r in df.iterrows():
        plt.annotate(
            r["model_label"],
            (float(r["wer_micro_target"]), float(r["avg_margin_other_minus_target"])),
            fontsize=8,
        )
    save_fig(out_dir / "04_pareto_targetwer_vs_margin.png")

    # Leaderboard-style ranked charts
    ranked_barh(
        df,
        metric="separation_score",
        title="Composite separation score (higher is better)",
        xlabel="Separation score",
        out_path=out_dir / "07_leaderboard_separation_score.png",
        higher_is_better=True,
        label_col="model_label",
    )
    ranked_barh(
        df,
        metric="win_rate_worst_case",
        title="Worst-case win rate across conditions (higher is better)",
        xlabel="Worst-case win rate",
        out_path=out_dir / "08_worst_case_win_rate.png",
        higher_is_better=True,
        label_col="model_label",
    )
    ranked_barh(
        df,
        metric="margin_worst_case",
        title="Worst-case margin across conditions (higher is better)",
        xlabel="Worst-case margin (WER_other - WER_target)",
        out_path=out_dir / "09_worst_case_margin.png",
        higher_is_better=True,
        label_col="model_label",
    )

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
            plt.plot(s["snr_db"], s["win_rate"], marker="o", label=label_map.get(ms, ms))
        plt.xlabel("SNR (TARGET over OTHER), dB")
        plt.ylabel("Mean win rate (over overlaps)")
        plt.title("Win rate vs SNR (mean across overlaps)")
        plt.legend()
        save_fig(out_dir / "05_winrate_vs_snr.png")

        plt.figure(figsize=(8, 6))
        for ms in df_snr["model_short"].unique():
            s = df_snr[df_snr["model_short"] == ms]
            plt.plot(s["snr_db"], s["margin"], marker="o", label=label_map.get(ms, ms))
        plt.xlabel("SNR (TARGET over OTHER), dB")
        plt.ylabel("Mean margin (WER_other - WER_target)")
        plt.title("Separation margin vs SNR (mean across overlaps)")
        plt.legend()
        save_fig(out_dir / "06_margin_vs_snr.png")

        df_ov = (
            df_long.groupby(["model_short", "overlap"], as_index=False)
            .agg(win_rate=("win_rate", "mean"), margin=("margin", "mean"))
            .sort_values(["model_short", "overlap"])
        )

        plt.figure(figsize=(8, 6))
        for ms in df_ov["model_short"].unique():
            s = df_ov[df_ov["model_short"] == ms]
            plt.plot(s["overlap"], s["win_rate"], marker="o", label=label_map.get(ms, ms))
        plt.xlabel("Overlap placement ratio (0=start, 1=end)")
        plt.ylabel("Mean win rate (over SNRs)")
        plt.title("Win rate vs overlap (mean across SNRs)")
        plt.legend()
        save_fig(out_dir / "10_winrate_vs_overlap.png")

        plt.figure(figsize=(8, 6))
        for ms in df_ov["model_short"].unique():
            s = df_ov[df_ov["model_short"] == ms]
            plt.plot(s["overlap"], s["margin"], marker="o", label=label_map.get(ms, ms))
        plt.xlabel("Overlap placement ratio (0=start, 1=end)")
        plt.ylabel("Mean margin (WER_other - WER_target)")
        plt.title("Separation margin vs overlap (mean across SNRs)")
        plt.legend()
        save_fig(out_dir / "11_margin_vs_overlap.png")

        boxplot_by_model(
            df_long,
            metric="win_rate",
            model_order=df["model_short"].tolist(),
            label_map=label_map,
            title="Win-rate distribution across conditions",
            xlabel="Win rate (higher is better)",
            out_path=out_dir / "12_boxplot_winrate_by_condition.png",
        )
        boxplot_by_model(
            df_long,
            metric="margin",
            model_order=df["model_short"].tolist(),
            label_map=label_map,
            title="Margin distribution across conditions",
            xlabel="Margin (WER_other - WER_target)",
            out_path=out_dir / "13_boxplot_margin_by_condition.png",
        )
        bump_rank_chart(
            df_long,
            data,
            metric="win_rate",
            title="Rank by condition (win rate)",
            out_path=out_dir / "14_bump_rank_by_condition.png",
            higher_is_better=True,
            top_models=top_models,
            label_map=label_map,
        )
        parallel_scorecard(
            df,
            metrics=[
                ("win_rate_target_closer", "Win rate", True),
                ("avg_margin_other_minus_target", "Margin", True),
                ("wer_micro_target", "WER_target", False),
                ("wer_micro_other", "WER_other", False),
                ("win_rate_worst_case", "Worst win", True),
                ("margin_worst_case", "Worst margin", True),
            ],
            top_models=top_models,
            out_path=out_dir / "15_scorecard_parallel_coords.png",
            label_map=label_map,
        )

    # CER charts
    if include_cer:
        plt.figure(figsize=(10, 5))
        plt.bar(df["model_label"], df["avg_margin_cer_other_minus_target"])
        plt.ylabel("Avg margin (CER_other - CER_target)")
        plt.xlabel("Model")
        plt.xticks(rotation=30, ha="right")
        plt.title("Overall CER separation margin (higher is better)")
        save_fig(out_dir / "16_overall_cer_margin.png")

        plt.figure(figsize=(10, 5))
        plt.bar(df["model_label"], df["cer_micro_target"])
        plt.ylabel("CER vs TARGET transcript (micro)")
        plt.xlabel("Model")
        plt.xticks(rotation=30, ha="right")
        plt.title("Target CER (lower is better)")
        save_fig(out_dir / "17_target_cer.png")

        plt.figure(figsize=(10, 5))
        plt.bar(df["model_label"], df["cer_micro_other"])
        plt.ylabel("CER vs OTHER transcript (micro)")
        plt.xlabel("Model")
        plt.xticks(rotation=30, ha="right")
        plt.title("Other CER (higher is better)")
        save_fig(out_dir / "18_other_cer.png")

        plt.figure(figsize=(8, 6))
        plt.scatter(df["cer_micro_target"], df["avg_margin_cer_other_minus_target"])
        plt.xlabel("CER_target (lower is better)")
        plt.ylabel("Avg CER margin (higher is better)")
        plt.title("Pareto view: CER accuracy vs separation")
        for _, r in df.iterrows():
            plt.annotate(
                r["model_label"],
                (float(r["cer_micro_target"]), float(r["avg_margin_cer_other_minus_target"])),
                fontsize=8,
            )
        save_fig(out_dir / "19_pareto_targetcer_vs_margin_cer.png")

        if not df_long.empty:
            df_snr_cer = (
                df_long.groupby(["model_short", "snr_db"], as_index=False)
                .agg(cer_t=("cer_t", "mean"), cer_o=("cer_o", "mean"), margin_cer=("margin_cer", "mean"))
                .sort_values(["model_short", "snr_db"])
            )

            plt.figure(figsize=(8, 6))
            for ms in df_snr_cer["model_short"].unique():
                s = df_snr_cer[df_snr_cer["model_short"] == ms]
                plt.plot(s["snr_db"], s["cer_t"], marker="o", label=label_map.get(ms, ms))
            plt.xlabel("SNR (TARGET over OTHER), dB")
            plt.ylabel("Mean CER_target (over overlaps)")
            plt.title("Target CER vs SNR (mean across overlaps)")
            plt.legend()
            save_fig(out_dir / "20_cer_target_vs_snr.png")

            plt.figure(figsize=(8, 6))
            for ms in df_snr_cer["model_short"].unique():
                s = df_snr_cer[df_snr_cer["model_short"] == ms]
                plt.plot(s["snr_db"], s["margin_cer"], marker="o", label=label_map.get(ms, ms))
            plt.xlabel("SNR (TARGET over OTHER), dB")
            plt.ylabel("Mean CER margin (CER_other - CER_target)")
            plt.title("CER separation margin vs SNR (mean across overlaps)")
            plt.legend()
            save_fig(out_dir / "21_cer_margin_vs_snr.png")

            df_ov_cer = (
                df_long.groupby(["model_short", "overlap"], as_index=False)
                .agg(cer_t=("cer_t", "mean"), margin_cer=("margin_cer", "mean"))
                .sort_values(["model_short", "overlap"])
            )

            plt.figure(figsize=(8, 6))
            for ms in df_ov_cer["model_short"].unique():
                s = df_ov_cer[df_ov_cer["model_short"] == ms]
                plt.plot(s["overlap"], s["cer_t"], marker="o", label=label_map.get(ms, ms))
            plt.xlabel("Overlap placement ratio (0=start, 1=end)")
            plt.ylabel("Mean CER_target (over SNRs)")
            plt.title("Target CER vs overlap (mean across SNRs)")
            plt.legend()
            save_fig(out_dir / "22_cer_target_vs_overlap.png")

            plt.figure(figsize=(8, 6))
            for ms in df_ov_cer["model_short"].unique():
                s = df_ov_cer[df_ov_cer["model_short"] == ms]
                plt.plot(s["overlap"], s["margin_cer"], marker="o", label=label_map.get(ms, ms))
            plt.xlabel("Overlap placement ratio (0=start, 1=end)")
            plt.ylabel("Mean CER margin (CER_other - CER_target)")
            plt.title("CER separation margin vs overlap (mean across SNRs)")
            plt.legend()
            save_fig(out_dir / "23_cer_margin_vs_overlap.png")

            boxplot_by_model(
                df_long,
                metric="cer_t",
                model_order=df["model_short"].tolist(),
                label_map=label_map,
                title="Target CER distribution across conditions",
                xlabel="CER_target (lower is better)",
                out_path=out_dir / "24_boxplot_cer_target_by_condition.png",
            )
            boxplot_by_model(
                df_long,
                metric="margin_cer",
                model_order=df["model_short"].tolist(),
                label_map=label_map,
                title="CER margin distribution across conditions",
                xlabel="CER margin (CER_other - CER_target)",
                out_path=out_dir / "25_boxplot_cer_margin_by_condition.png",
            )

        # Combined WER+CER views (single images)
        grouped_bar_two_metrics(
            df,
            metric_a="wer_micro_target",
            metric_b="cer_micro_target",
            label_a="WER_target",
            label_b="CER_target",
            title="Target error: WER vs CER (lower is better)",
            ylabel="Error rate",
            out_path=out_dir / "26_target_wer_vs_cer.png",
        )
        grouped_bar_two_metrics(
            df,
            metric_a="wer_micro_other",
            metric_b="cer_micro_other",
            label_a="WER_other",
            label_b="CER_other",
            title="Other error: WER vs CER (higher is better)",
            ylabel="Error rate",
            out_path=out_dir / "27_other_wer_vs_cer.png",
        )
        grouped_bar_two_metrics(
            df,
            metric_a="avg_margin_other_minus_target",
            metric_b="avg_margin_cer_other_minus_target",
            label_a="WER margin",
            label_b="CER margin",
            title="Separation margin: WER vs CER (higher is better)",
            ylabel="Margin",
            out_path=out_dir / "28_margin_wer_vs_cer.png",
        )

    # Per-model heatmaps
    models = data.get("models") or []
    # Sort models by epoch number for consistent ordering
    models = sorted(models, key=lambda m: get_epoch_number(m.get("model", "")))
    for m in models:
        ms = short_model_name(m.get("model", "unknown"))
        label = label_map.get(ms, ms)
        heatmap_for_model(
            data,
            m,
            out_dir,
            metric_key="win_rate_target_closer",
            title_suffix="win rate by SNR × overlap",
            cbar_label="Win rate",
            display_label=label,
        )
        heatmap_for_model(
            data,
            m,
            out_dir,
            metric_key="avg_margin_other_minus_target",
            title_suffix="margin by SNR × overlap",
            cbar_label="Avg margin (WER_other - WER_target)",
            display_label=label,
        )
        if include_cer:
            heatmap_for_model(
                data,
                m,
                out_dir,
                metric_key="cer_micro_target",
                title_suffix="target CER by SNR × overlap",
                cbar_label="CER_target",
                display_label=label,
            )
            heatmap_for_model(
                data,
                m,
                out_dir,
                metric_key="avg_margin_cer_other_minus_target",
                title_suffix="CER margin by SNR × overlap",
                cbar_label="Avg CER margin (other - target)",
                display_label=label,
            )

    print("Wrote:", out_dir)
    _beep()


if __name__ == "__main__":
    main()
