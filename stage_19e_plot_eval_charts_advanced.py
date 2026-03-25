#!/usr/bin/env python3
r"""stage_19e_plot_eval_charts_advanced.py

Advanced visualization pack for Stage-19 target-vs-other sweep evaluations.

This script complements stage_19d by adding:
- Pareto/frontier views (quality vs separation vs model size)
- ECDF / survival distribution views for per-sample behavior
- Head-to-head dominance matrices between models
- Condition-rank and stability charts

Input
-----
Pass either:
- evaluation_results_futo_like_targetmix_sweep.json
or
- evaluation_per_sample_predictions_targetmix_sweep.json

Usage (Windows PowerShell)
-------------------------
C:\whisper-edge-eval-venv\Scripts\python.exe i:\whisper-acft\stage_19e_plot_eval_charts_advanced.py `
  --in_json "I:\Stage_19e_edge_eval_20260315\evaluation_per_sample_predictions_targetmix_sweep.json" `
  --out_dir "I:\Stage_19e_edge_eval_20260315\advanced_charts"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


IMG_FORMAT = "png"
IMG_DPI = 220
IMG_QUALITY = 92


def _normalize_format(fmt: str) -> str:
    fmt = (fmt or "png").strip().lower()
    if fmt == "jpeg":
        return "jpg"
    return fmt


def _with_format(path: Path, fmt: str) -> Path:
    fmt = _normalize_format(fmt)
    ext = f".{fmt}"
    if path.suffix.lower() == ext:
        return path
    return path.with_suffix(ext)


def _save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fmt = _normalize_format(IMG_FORMAT)
    out = _with_format(path, fmt)
    kwargs = {"dpi": IMG_DPI, "format": "jpeg" if fmt == "jpg" else fmt}
    if fmt in {"jpg", "webp"}:
        kwargs["pil_kwargs"] = {"quality": int(IMG_QUALITY), "optimize": True}
    plt.savefig(out, **kwargs)
    plt.close()


def _beep() -> None:
    try:
        import winsound  # type: ignore

        winsound.Beep(900, 180)
        winsound.Beep(1200, 220)
    except Exception:
        print("\a", end="", flush=True)


def _short_model_name(name: str) -> str:
    base = str(name).replace("\\", "/").split("/")[-1]
    for char in ['<', '>', ':', '"', '|', '?', '*']:
        base = base.replace(char, "_")
    return base


def _display_model_name(name: str, max_len: int = 26) -> str:
    s = _short_model_name(name)
    if len(s) <= max_len:
        return s
    keep = max(4, (max_len - 3) // 2)
    return f"{s[:keep]}...{s[-keep:]}"


def _json_load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_per_sample_meta(data: object) -> Optional[dict]:
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


def _resolve_results_json(in_path: Path, in_data: object) -> Optional[Path]:
    meta = _extract_per_sample_meta(in_data)
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


def _as_per_sample_rows(per_sample_payload: object) -> List[dict]:
    rows: List[dict] = []
    if isinstance(per_sample_payload, dict):
        rows = per_sample_payload.get("items") or per_sample_payload.get("samples") or []
        if not isinstance(rows, list):
            rows = []
        return [r for r in rows if isinstance(r, dict)]
    if isinstance(per_sample_payload, list):
        out = []
        for r in per_sample_payload:
            if not isinstance(r, dict):
                continue
            if "__meta__" in r and "mix_key" not in r:
                continue
            out.append(r)
        return out
    return rows


def _build_summary_df(results_data: dict) -> pd.DataFrame:
    rows = []
    for m in results_data.get("models", []) or []:
        if not isinstance(m, dict):
            continue
        overall = m.get("metrics_overall") or {}
        if not isinstance(overall, dict):
            overall = {}
        model = str(m.get("model", "unknown"))
        rows.append(
            {
                "model": model,
                "model_short": _short_model_name(model),
                "label": _display_model_name(model),
                "samples": overall.get("samples"),
                "wer_target": overall.get("wer_micro_target"),
                "wer_other": overall.get("wer_micro_other"),
                "cer_target": overall.get("cer_micro_target"),
                "cer_other": overall.get("cer_micro_other"),
                "win_rate": overall.get("win_rate_target_closer"),
                "margin": overall.get("avg_margin_other_minus_target"),
                "margin_cer": overall.get("avg_margin_cer_other_minus_target"),
                "param_count": overall.get("model_num_params"),
                "model_type": overall.get("model_type"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for c in ["samples", "wer_target", "wer_other", "cer_target", "cer_other", "win_rate", "margin", "margin_cer", "param_count"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["param_m"] = df["param_count"] / 1_000_000.0
    return df


def _build_long_df(per_sample_rows: List[dict]) -> pd.DataFrame:
    long_rows: List[dict] = []
    for item in per_sample_rows:
        if not isinstance(item, dict):
            continue
        mix_key = item.get("mix_key")
        if not mix_key:
            continue
        cond_id = item.get("cond_id")
        snr_db = item.get("snr_db")
        overlap = item.get("overlap")
        preds = item.get("predictions") or {}
        if not isinstance(preds, dict):
            continue

        for model, pred in preds.items():
            if not isinstance(pred, dict):
                continue
            wt = pred.get("wer_target")
            wo = pred.get("wer_other")
            ct = pred.get("cer_target")
            co = pred.get("cer_other")
            margin = None
            margin_cer = None
            if wt is not None and wo is not None:
                try:
                    margin = float(wo) - float(wt)
                except Exception:
                    margin = None
            if ct is not None and co is not None:
                try:
                    margin_cer = float(co) - float(ct)
                except Exception:
                    margin_cer = None

            long_rows.append(
                {
                    "mix_key": str(mix_key),
                    "cond_id": str(cond_id),
                    "snr_db": snr_db,
                    "overlap": overlap,
                    "model": str(model),
                    "model_short": _short_model_name(str(model)),
                    "wer_target": wt,
                    "wer_other": wo,
                    "cer_target": ct,
                    "cer_other": co,
                    "margin": margin,
                    "margin_cer": margin_cer,
                    "win": pred.get("win_target_closer"),
                    "duration_sec_eval": pred.get("duration_sec_eval"),
                }
            )
    df = pd.DataFrame(long_rows)
    if df.empty:
        return df
    for c in ["snr_db", "overlap", "wer_target", "wer_other", "cer_target", "cer_other", "margin", "margin_cer", "duration_sec_eval"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["win"] = df["win"].astype("boolean")
    return df


def _norm01(series: pd.Series, higher_is_better: bool) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    lo = s.min(skipna=True)
    hi = s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    if higher_is_better:
        return (s - lo) / (hi - lo)
    return (hi - s) / (hi - lo)


def _ecdf(values: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray([v for v in values if pd.notna(v)], dtype=np.float64))
    if x.size == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    y = np.arange(1, x.size + 1, dtype=np.float64) / float(x.size)
    return x, y


def _survival(values: Iterable[float]) -> Tuple[np.ndarray, np.ndarray]:
    x, y = _ecdf(values)
    if x.size == 0:
        return x, y
    return x, 1.0 - y + (1.0 / float(x.size))


def _pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    d = df[[x_col, y_col, "label"]].dropna().sort_values(x_col)
    keep_idx: List[int] = []
    best_y = -np.inf
    for idx, row in d.iterrows():
        y = float(row[y_col])
        if y > best_y:
            keep_idx.append(idx)
            best_y = y
    return d.loc[keep_idx].sort_values(x_col)


def _plot_pareto(df: pd.DataFrame, out_dir: Path) -> None:
    d = df.dropna(subset=["wer_target", "win_rate"]).copy()
    if d.empty:
        return
    plt.figure(figsize=(9, 6))
    size = np.where(pd.notna(d["param_m"]), np.clip(d["param_m"], 8, 400) * 2.2, 140.0)
    sc = plt.scatter(d["wer_target"], d["win_rate"], s=size, c=d["margin"], cmap="viridis", alpha=0.85, edgecolors="k", linewidths=0.4)
    frontier = _pareto_frontier(d, "wer_target", "win_rate")
    if len(frontier) >= 2:
        plt.plot(frontier["wer_target"], frontier["win_rate"], color="black", linewidth=1.6, linestyle="--", label="Pareto frontier")
        plt.legend(loc="best")
    for _, r in d.iterrows():
        plt.annotate(str(r["label"]), (float(r["wer_target"]), float(r["win_rate"])), fontsize=8)
    plt.xlabel("Target WER (lower is better)")
    plt.ylabel("Win rate target-closer (higher is better)")
    plt.title("Pareto view: target quality vs separation behavior")
    cb = plt.colorbar(sc)
    cb.set_label("Avg WER margin (other - target)")
    _save_fig(out_dir / "adv_01_pareto_wer_vs_winrate.png")

    if d["param_m"].notna().any():
        d2 = d.dropna(subset=["param_m"]).copy()
        if not d2.empty:
            plt.figure(figsize=(9, 6))
            plt.scatter(d2["param_m"], d2["wer_target"], c=d2["win_rate"], cmap="plasma", s=130, edgecolors="k", linewidths=0.4)
            for _, r in d2.iterrows():
                plt.annotate(str(r["label"]), (float(r["param_m"]), float(r["wer_target"])), fontsize=8)
            plt.xlabel("Model size (million parameters)")
            plt.ylabel("Target WER (lower is better)")
            plt.title("Size vs target WER (color = win rate)")
            plt.xscale("log")
            cb = plt.colorbar()
            cb.set_label("Win rate")
            _save_fig(out_dir / "adv_02_size_vs_target_wer.png")


def _plot_distribution_curves(df_long: pd.DataFrame, top_models: List[str], label_map: Dict[str, str], out_dir: Path) -> None:
    if df_long.empty:
        return

    plt.figure(figsize=(9, 6))
    for m in top_models:
        vals = df_long.loc[df_long["model"] == m, "wer_target"].dropna().values
        x, y = _ecdf(vals)
        if x.size == 0:
            continue
        plt.plot(x, y, linewidth=1.8, label=label_map.get(m, _display_model_name(m)))
    plt.xlabel("Per-sample target WER")
    plt.ylabel("ECDF")
    plt.title("ECDF of target WER (left/down is better)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    _save_fig(out_dir / "adv_03_ecdf_target_wer.png")

    plt.figure(figsize=(9, 6))
    for m in top_models:
        vals = df_long.loc[df_long["model"] == m, "margin"].dropna().values
        x, y = _survival(vals)
        if x.size == 0:
            continue
        plt.plot(x, y, linewidth=1.8, label=label_map.get(m, _display_model_name(m)))
    plt.xlabel("Per-sample WER margin (other - target)")
    plt.ylabel("Survival P(margin >= x)")
    plt.title("Tail behavior of separation margin (higher curve is better)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.2)
    _save_fig(out_dir / "adv_04_survival_margin.png")


def _pairwise_prob_matrix(pivot_df: pd.DataFrame, better: str = "lower") -> pd.DataFrame:
    models = list(pivot_df.columns)
    n = len(models)
    mat = np.full((n, n), np.nan, dtype=np.float64)
    for i, mi in enumerate(models):
        for j, mj in enumerate(models):
            if i == j:
                mat[i, j] = 0.5
                continue
            a = pd.to_numeric(pivot_df[mi], errors="coerce")
            b = pd.to_numeric(pivot_df[mj], errors="coerce")
            mask = a.notna() & b.notna()
            if mask.sum() == 0:
                continue
            if better == "lower":
                mat[i, j] = float((a[mask] < b[mask]).mean())
            else:
                mat[i, j] = float((a[mask] > b[mask]).mean())
    return pd.DataFrame(mat, index=models, columns=models)


def _plot_head_to_head(df_long: pd.DataFrame, model_order: List[str], label_map: Dict[str, str], out_dir: Path) -> None:
    if df_long.empty:
        return

    base = df_long[df_long["model"].isin(model_order)].copy()
    if base.empty:
        return

    piv_wt = base.pivot_table(index="mix_key", columns="model", values="wer_target", aggfunc="mean")
    piv_wt = piv_wt.reindex(columns=model_order)
    m_wt = _pairwise_prob_matrix(piv_wt, better="lower")

    piv_margin = base.pivot_table(index="mix_key", columns="model", values="margin", aggfunc="mean")
    piv_margin = piv_margin.reindex(columns=model_order)
    m_margin = _pairwise_prob_matrix(piv_margin, better="higher")

    def _draw(mat: pd.DataFrame, title: str, out_name: str) -> None:
        d = mat.values
        fig_w = max(8.5, 0.55 * len(mat.columns))
        fig_h = max(7.5, 0.55 * len(mat.index))
        plt.figure(figsize=(fig_w, fig_h))
        im = plt.imshow(d, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        plt.colorbar(im, label="Probability")
        xlabels = [label_map.get(m, _display_model_name(m, max_len=18)) for m in mat.columns]
        ylabels = [label_map.get(m, _display_model_name(m, max_len=18)) for m in mat.index]
        plt.xticks(np.arange(len(xlabels)), xlabels, rotation=45, ha="right")
        plt.yticks(np.arange(len(ylabels)), ylabels)
        plt.xlabel("Compared model (column)")
        plt.ylabel("Reference model (row)")
        plt.title(title)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                if np.isnan(d[i, j]):
                    continue
                txt = f"{d[i, j]:.2f}"
                plt.text(j, i, txt, ha="center", va="center", fontsize=7, color="white")
        _save_fig(out_dir / out_name)

    _draw(
        m_wt,
        "Head-to-head dominance: P(row has lower target WER than column)",
        "adv_05_head_to_head_target_wer.png",
    )
    _draw(
        m_margin,
        "Head-to-head dominance: P(row has higher WER margin than column)",
        "adv_06_head_to_head_margin.png",
    )


def _plot_condition_rank_and_stability(
    df_long: pd.DataFrame,
    model_order: List[str],
    label_map: Dict[str, str],
    out_dir: Path,
) -> None:
    if df_long.empty:
        return

    cond = (
        df_long.groupby(["cond_id", "snr_db", "overlap", "model"], as_index=False)
        .agg(
            win_rate=("win", "mean"),
            wer_target=("wer_target", "mean"),
            margin=("margin", "mean"),
        )
    )
    if cond.empty:
        return

    cond = cond[cond["model"].isin(model_order)].copy()
    if cond.empty:
        return

    cond["rank_win"] = cond.groupby("cond_id")["win_rate"].rank(method="min", ascending=False)
    cond["rank_wer"] = cond.groupby("cond_id")["wer_target"].rank(method="min", ascending=True)
    cond["rank_blend"] = 0.6 * cond["rank_win"] + 0.4 * cond["rank_wer"]

    order_conds = (
        cond[["cond_id", "snr_db", "overlap"]]
        .drop_duplicates()
        .sort_values(["snr_db", "overlap", "cond_id"])
    )
    cond_order = order_conds["cond_id"].tolist()

    heat = (
        cond.pivot_table(index="cond_id", columns="model", values="rank_blend", aggfunc="mean")
        .reindex(index=cond_order, columns=model_order)
    )
    heat = heat.apply(pd.to_numeric, errors="coerce")

    heat_values = heat.to_numpy(dtype=float, na_value=np.nan)

    plt.figure(figsize=(max(9, 0.55 * len(model_order)), max(7, 0.34 * len(cond_order))))
    im = plt.imshow(heat_values, aspect="auto", cmap="magma_r", vmin=1.0, vmax=float(len(model_order)))
    plt.colorbar(im, label="Blended rank (1 is best)")
    plt.xticks(np.arange(len(model_order)), [label_map.get(m, _display_model_name(m, 16)) for m in model_order], rotation=45, ha="right")
    plt.yticks(np.arange(len(cond_order)), cond_order, fontsize=8)
    plt.title("Condition-wise blended rank heatmap")
    plt.xlabel("Model")
    plt.ylabel("Condition")
    _save_fig(out_dir / "adv_07_condition_rank_heatmap.png")

    stab = (
        cond.groupby("model", as_index=False)
        .agg(
            win_mean=("win_rate", "mean"),
            win_std=("win_rate", "std"),
            margin_mean=("margin", "mean"),
            margin_std=("margin", "std"),
            worst_win=("win_rate", "min"),
            worst_margin=("margin", "min"),
        )
    )
    stab = stab.set_index("model").reindex(model_order).reset_index()

    plt.figure(figsize=(9, 6))
    plt.scatter(stab["win_mean"], stab["win_std"], s=120, c=stab["margin_mean"], cmap="coolwarm", edgecolors="k", linewidths=0.5)
    for _, r in stab.iterrows():
        plt.annotate(label_map.get(str(r["model"]), _display_model_name(str(r["model"]), 18)), (float(r["win_mean"]), float(r["win_std"])), fontsize=8)
    plt.xlabel("Mean win rate across conditions (higher is better)")
    plt.ylabel("Std(win rate) across conditions (lower is better)")
    plt.title("Stability map: average strength vs volatility")
    cb = plt.colorbar()
    cb.set_label("Mean margin")
    plt.grid(alpha=0.2)
    _save_fig(out_dir / "adv_08_stability_mean_vs_std.png")

    plt.figure(figsize=(9, 6))
    plt.scatter(stab["margin_mean"], stab["worst_margin"], s=120, c=stab["win_mean"], cmap="viridis", edgecolors="k", linewidths=0.5)
    for _, r in stab.iterrows():
        plt.annotate(label_map.get(str(r["model"]), _display_model_name(str(r["model"]), 18)), (float(r["margin_mean"]), float(r["worst_margin"])), fontsize=8)
    plt.xlabel("Mean WER margin across conditions")
    plt.ylabel("Worst-condition WER margin")
    plt.title("Worst-case robustness vs average separation")
    cb = plt.colorbar()
    cb.set_label("Mean win rate")
    plt.grid(alpha=0.2)
    _save_fig(out_dir / "adv_09_robustness_worst_vs_mean_margin.png")


def _plot_performance_profile(df_long: pd.DataFrame, model_order: List[str], label_map: Dict[str, str], out_dir: Path) -> None:
    if df_long.empty:
        return

    cond_model = (
        df_long[df_long["model"].isin(model_order)]
        .groupby(["mix_key", "model"], as_index=False)
        .agg(wer_target=("wer_target", "mean"))
    )
    if cond_model.empty:
        return

    pivot = cond_model.pivot_table(index="mix_key", columns="model", values="wer_target", aggfunc="mean")
    pivot = pivot.reindex(columns=model_order)
    if pivot.empty:
        return

    best = pivot.min(axis=1, skipna=True)
    valid_rows = best.notna() & (best > 0)
    if not valid_rows.any():
        return
    pivot = pivot.loc[valid_rows]
    best = best.loc[valid_rows]

    ratio = pivot.divide(best, axis=0)
    finite_vals = ratio.to_numpy(dtype=float)
    finite_vals = finite_vals[np.isfinite(finite_vals)]
    if finite_vals.size == 0:
        return

    tau_max = float(np.nanpercentile(finite_vals, 95))
    tau_max = max(1.2, min(6.0, tau_max))
    taus = np.linspace(1.0, tau_max, 120)

    plt.figure(figsize=(9, 6))
    for m in model_order:
        col = pd.to_numeric(ratio[m], errors="coerce").dropna()
        if col.empty:
            continue
        y = [float((col <= t).mean()) for t in taus]
        plt.plot(taus, y, linewidth=1.8, label=label_map.get(m, _display_model_name(m)))

    plt.xlabel("Performance ratio τ (model WER / best WER per sample)")
    plt.ylabel("P(ratio ≤ τ)")
    plt.title("Performance profile on target WER (higher curve is better)")
    plt.ylim(0.0, 1.02)
    plt.xlim(1.0, tau_max)
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8, loc="lower right")
    _save_fig(out_dir / "adv_10_performance_profile_target_wer.png")


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1200, seed: int = 42) -> Tuple[float, float, float]:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(vals))
    if vals.size == 1:
        return mean, mean, mean
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    boots = vals[idx].mean(axis=1)
    lo = float(np.nanpercentile(boots, 2.5))
    hi = float(np.nanpercentile(boots, 97.5))
    return mean, lo, hi


def _plot_bootstrap_ci(df_long: pd.DataFrame, model_order: List[str], label_map: Dict[str, str], out_dir: Path) -> None:
    if df_long.empty:
        return

    rows: List[dict] = []
    for m in model_order:
        d = df_long.loc[df_long["model"] == m, "win"].dropna()
        if d.empty:
            continue
        vals = d.astype(float).to_numpy(dtype=np.float64)
        mean, lo, hi = _bootstrap_mean_ci(vals)
        rows.append({"model": m, "mean": mean, "lo": lo, "hi": hi})
    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("mean", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))
    x = df["mean"].to_numpy(dtype=np.float64)
    xerr = np.vstack([x - df["lo"].to_numpy(dtype=np.float64), df["hi"].to_numpy(dtype=np.float64) - x])

    plt.figure(figsize=(9, max(5, 0.45 * len(df))))
    plt.errorbar(x, y, xerr=xerr, fmt="o", capsize=3, color="#1f77b4")
    plt.yticks(y, [label_map.get(m, _display_model_name(m, 18)) for m in df["model"]])
    plt.xlabel("Win rate target-closer")
    plt.title("Bootstrap 95% CI of win rate")
    plt.grid(axis="x", alpha=0.25)
    plt.xlim(0.0, 1.0)
    _save_fig(out_dir / "adv_11_bootstrap_ci_winrate.png")


def _write_summary_tables(df_summary: pd.DataFrame, df_long: pd.DataFrame, out_dir: Path) -> None:
    if df_summary.empty:
        return

    df = df_summary.copy()
    df["score"] = (
        0.35 * _norm01(df["win_rate"], True)
        + 0.35 * _norm01(df["margin"], True)
        + 0.20 * _norm01(df["wer_target"], False)
        + 0.10 * _norm01(df["param_m"], False)
    )
    df = df.sort_values(["score", "win_rate", "margin"], ascending=[False, False, False])

    if not df_long.empty:
        per_model_tail = (
            df_long.groupby("model", as_index=False)
            .agg(
                wer_target_p50=("wer_target", "median"),
                wer_target_p90=("wer_target", lambda s: float(np.nanpercentile(pd.to_numeric(s, errors="coerce").dropna(), 90)) if pd.to_numeric(s, errors="coerce").dropna().size > 0 else np.nan),
                margin_p10=("margin", lambda s: float(np.nanpercentile(pd.to_numeric(s, errors="coerce").dropna(), 10)) if pd.to_numeric(s, errors="coerce").dropna().size > 0 else np.nan),
            )
        )
        df = df.merge(per_model_tail, how="left", on="model")

    out_cols = [
        "model",
        "model_type",
        "samples",
        "param_m",
        "wer_target",
        "wer_other",
        "cer_target",
        "cer_other",
        "win_rate",
        "margin",
        "margin_cer",
        "wer_target_p50",
        "wer_target_p90",
        "margin_p10",
        "score",
    ]
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    out_csv = out_dir / "advanced_model_leaderboard.csv"
    df[out_cols].to_csv(out_csv, index=False, encoding="utf-8")

    best = df.iloc[0]
    md = [
        "# Advanced Eval Summary",
        "",
        f"- Models: {len(df)}",
        f"- Best by composite score: `{best['model']}`",
        f"- Best win rate: `{df.loc[df['win_rate'].idxmax(), 'model']}` ({df['win_rate'].max():.4f})" if df["win_rate"].notna().any() else "- Best win rate: n/a",
        f"- Lowest target WER: `{df.loc[df['wer_target'].idxmin(), 'model']}` ({df['wer_target'].min():.4f})" if df["wer_target"].notna().any() else "- Lowest target WER: n/a",
        "",
        "See `advanced_model_leaderboard.csv` for full ranking fields.",
    ]
    (out_dir / "advanced_summary.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True, type=Path)
    ap.add_argument("--out_dir", type=Path, default=None)
    ap.add_argument("--img_format", type=str, default="png", choices=["png", "jpg", "jpeg", "webp"])
    ap.add_argument("--img_quality", type=int, default=92)
    ap.add_argument("--img_dpi", type=int, default=220)
    ap.add_argument("--top_k", type=int, default=10, help="Top-K models to emphasize in distribution/pairwise plots.")
    args = ap.parse_args()

    global IMG_FORMAT, IMG_DPI, IMG_QUALITY
    IMG_FORMAT = args.img_format
    IMG_DPI = int(args.img_dpi)
    IMG_QUALITY = int(args.img_quality)

    in_json = args.in_json
    in_data = _json_load(in_json)

    if isinstance(in_data, dict) and isinstance(in_data.get("models"), list):
        results_json = in_json
        results_data = in_data
        per_sample_json = in_json.parent / "evaluation_per_sample_predictions_targetmix_sweep.json"
        if not per_sample_json.exists():
            raise FileNotFoundError(f"Per-sample file not found next to results JSON: {per_sample_json}")
        per_sample_data = _json_load(per_sample_json)
    else:
        per_sample_json = in_json
        per_sample_data = in_data
        results_json = _resolve_results_json(in_json, in_data)
        if results_json is None or not results_json.exists():
            raise FileNotFoundError("Could not locate evaluation_results_futo_like_targetmix_sweep.json")
        results_data = _json_load(results_json)

    out_dir = args.out_dir if args.out_dir is not None else in_json.parent / "advanced_charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sample_rows = _as_per_sample_rows(per_sample_data)
    df_summary = _build_summary_df(results_data)
    df_long = _build_long_df(per_sample_rows)

    if df_summary.empty:
        raise RuntimeError("No model summary rows found in results JSON.")

    df_rank = df_summary.copy()
    df_rank["score"] = (
        0.35 * _norm01(df_rank["win_rate"], True)
        + 0.35 * _norm01(df_rank["margin"], True)
        + 0.20 * _norm01(df_rank["wer_target"], False)
        + 0.10 * _norm01(df_rank["param_m"], False)
    )
    df_rank = df_rank.sort_values(["score", "win_rate", "margin"], ascending=[False, False, False])
    model_order = df_rank["model"].tolist()

    top_k = max(3, int(args.top_k))
    top_models = model_order[: min(top_k, len(model_order))]
    label_map = {m: _display_model_name(m) for m in model_order}

    _plot_pareto(df_rank, out_dir)
    _plot_distribution_curves(df_long, top_models, label_map, out_dir)
    _plot_head_to_head(df_long, top_models, label_map, out_dir)
    _plot_condition_rank_and_stability(df_long, top_models, label_map, out_dir)
    _plot_performance_profile(df_long, top_models, label_map, out_dir)
    _plot_bootstrap_ci(df_long, top_models, label_map, out_dir)
    _write_summary_tables(df_rank, df_long, out_dir)

    manifest = {
        "results_json": str(results_json),
        "per_sample_json": str(per_sample_json),
        "rows_summary": int(len(df_summary)),
        "rows_long": int(len(df_long)),
        "top_models": top_models,
    }
    (out_dir / "advanced_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote advanced charts to: {out_dir}")
    _beep()


if __name__ == "__main__":
    main()
