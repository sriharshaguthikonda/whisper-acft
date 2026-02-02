#!/usr/bin/env python3
"""Stage 19c evaluation wrapper.

Targets are picked from CSV (sorted by score desc).
Others are picked from --others_dir (transcripts via --others_manifest).
Place this next to the original stage_19c core script.





 i:\Whisper-training-env\Scripts\python.exe "i:\whisper-acft\stage_19c_specific_target_advanced_futo_like_evaluate_targetmix_sweep_with_cer__others_dir.py" --test_manifest "I:\Record_chunks\pairs_manifest_combined_all_datasets_randomized_test.jsonl" --speaker_scores_csv "I:\whisper-acft\speaker_sort_scores.csv" --others_dir "I:\Record_others_chunks" --others_manifest "I:\Record_others_chunks\pairs_pending_stereo.jsonl" --checkpoint_dir "I:\Stage_2_shuffle_n_ctx_stage_7_checkpoints_partialctx_tiny_en_11\20260202_033631" --mix_per_target 1 --pairing_mode hash --target_take 0 --target_percent 1 --sweep_snr_db "20,10,5,0,-5" --sweep_overlap "0,0.25,0.5,0.75,1" --audio_cache_gb 1 --resume --recalc_metrics



"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# Ensure the core Stage 19c script is importable when both scripts sit in the same folder.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    import stage_19c_specific_target_advanced_futo_like_evaluate_targetmix_sweep_with_cer as core
except Exception as e:
    raise SystemExit(
        "Could not import the core Stage 19c script.\n"
        "Put this wrapper in the SAME folder as:\n"
        "  stage_19c_specific_target_advanced_futo_like_evaluate_targetmix_sweep_with_cer.py\n"
        f"Import error: {type(e).__name__}: {e}"
    )

# ----------------------------
# Speaker CSV (file, score, decision, ...)
# ----------------------------

@dataclass
class SpeakerScoreDB:
    decision_by_path: Dict[str, str]
    decision_by_basename: Dict[str, str]
    score_by_path: Dict[str, float]
    score_by_basename: Dict[str, float]

    def decision_for(self, audio_path: str) -> Optional[str]:
        k = core._norm_windows_key(audio_path)
        if k in self.decision_by_path:
            return self.decision_by_path[k]
        b = core._basename(audio_path)
        return self.decision_by_basename.get(b)

    def score_for(self, audio_path: str) -> Optional[float]:
        k = core._norm_windows_key(audio_path)
        if k in self.score_by_path:
            return self.score_by_path[k]
        b = core._basename(audio_path)
        return self.score_by_basename.get(b)


def load_speaker_scores_csv(csv_path: Path) -> SpeakerScoreDB:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    decision_by_path: Dict[str, str] = {}
    score_by_path: Dict[str, float] = {}
    basename_decisions: Dict[str, Dict[str, int]] = {}
    basename_scores: Dict[str, List[float]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")

        if "file" not in reader.fieldnames or "decision" not in reader.fieldnames:
            raise ValueError(f"CSV must contain 'file' and 'decision' columns. Found: {reader.fieldnames}")

        for row in reader:
            fp = (row.get("file") or "").strip()
            if not fp:
                continue

            decision = (row.get("decision") or "").strip().upper()
            if decision not in {"TARGET", "OTHER"}:
                continue

            score_raw = (row.get("score") or "").strip()
            score: Optional[float] = None
            if score_raw:
                try:
                    score = float(score_raw)
                except Exception:
                    score = None

            nk = core._norm_windows_key(fp)
            decision_by_path[nk] = decision
            if score is not None:
                score_by_path[nk] = score

            b = core._basename(fp)
            basename_decisions.setdefault(b, {})
            basename_decisions[b][decision] = basename_decisions[b].get(decision, 0) + 1
            if score is not None:
                basename_scores.setdefault(b, []).append(score)

    # Use basename matching only if basename maps to exactly one decision type.
    decision_by_basename: Dict[str, str] = {}
    score_by_basename: Dict[str, float] = {}

    for b, counts in basename_decisions.items():
        if len(counts) == 1:
            decision_by_basename[b] = next(iter(counts.keys()))
            if b in basename_scores and basename_scores[b]:
                # If multiple rows share the same basename, take max score.
                score_by_basename[b] = float(max(basename_scores[b]))

    return SpeakerScoreDB(
        decision_by_path=decision_by_path,
        decision_by_basename=decision_by_basename,
        score_by_path=score_by_path,
        score_by_basename=score_by_basename,
    )

# ----------------------------
# Transcript lookup for OTHER files (optional)
# ----------------------------

@dataclass
class TranscriptDB:
    by_path: Dict[str, str]
    by_basename: Dict[str, str]

    def text_for(self, audio_path: str) -> Optional[str]:
        k = core._norm_windows_key(audio_path)
        if k in self.by_path:
            return self.by_path[k]
        b = core._basename(audio_path)
        return self.by_basename.get(b)


def load_jsonl_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _row_transcript(row: dict) -> str:
    # keep compatible with your manifests
    for k in ("raw_transcription", "transcript", "text"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def build_transcript_db_from_manifest(manifest_path: Path) -> TranscriptDB:
    rows = load_jsonl_rows(manifest_path)
    by_path: Dict[str, str] = {}
    basename_candidates: Dict[str, List[str]] = {}

    for r in rows:
        ap = str(r.get("audio_path", "") or "").strip()
        if not ap:
            continue
        tx = _row_transcript(r)
        if not tx:
            continue
        nk = core._norm_windows_key(ap)
        by_path[nk] = tx
        b = core._basename(ap)
        basename_candidates.setdefault(b, []).append(tx)

    by_basename: Dict[str, str] = {}
    for b, txs in basename_candidates.items():
        if len(txs) == 1:
            by_basename[b] = txs[0]
        else:
            # choose the longest non-empty as a best-effort fallback
            by_basename[b] = max(txs, key=len)

    return TranscriptDB(by_path=by_path, by_basename=by_basename)


def scan_audio_files(root: Path) -> List[Path]:
    exts = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aac", ".opus"}
    out: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    out.sort(key=lambda p: core._norm_windows_key(str(p)))
    return out

# ----------------------------
# Audio loading (adds ffmpeg fallback for mp3/m4a)
# ----------------------------

def make_loader_with_ffmpeg(ffmpeg_path: str, core_load_orig):
    def _load_audio_mono_16k(p: Path) -> Tuple[np.ndarray, int]:
        # Try soundfile first (fast for wav/flac/ogg)
        try:
            audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
            if isinstance(audio, np.ndarray) and audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio = np.asarray(audio, dtype=np.float32)
            if not np.isfinite(audio).all():
                audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

            if sr != 16000:
                # reuse core's resampler path by calling its original helper
                return core_load_orig(p)

            return audio, int(sr)
        except Exception:
            pass

        # Fallback: ffmpeg decode -> float32 PCM @ 16k mono
        cmd = [
            ffmpeg_path,
            "-v",
            "error",
            "-i",
            str(p),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-",
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", errors="ignore")
            raise RuntimeError(f"ffmpeg failed for {p}: {err}")

        audio = np.frombuffer(proc.stdout, dtype=np.float32)
        if audio.size == 0:
            raise RuntimeError(f"ffmpeg produced empty audio for {p}")
        if not np.isfinite(audio).all():
            audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        return audio.astype(np.float32), 16000

    return _load_audio_mono_16k

# ----------------------------
# Pair building (targets sorted by score, others from folder)
# ----------------------------

def select_targets_sorted(
    rows: List[dict],
    scores: SpeakerScoreDB,
    target_take: int,
    target_percent: float,
) -> Tuple[List[dict], Dict]:
    labelled = 0
    unknown = 0
    targets: List[Tuple[float, dict]] = []

    for r in rows:
        ap = str(r.get("audio_path", "") or "").strip()
        if not ap:
            continue
        dec = scores.decision_for(ap)
        if dec is None:
            unknown += 1
            continue
        labelled += 1
        if dec != "TARGET":
            continue

        tx = _row_transcript(r)
        if not tx:
            # still allow, but this makes WER meaningless; skip
            continue

        sc = scores.score_for(ap)
        scv = float(sc) if sc is not None else float("-inf")
        targets.append((scv, r))

    targets.sort(key=lambda t: (-(t[0]), core._norm_windows_key(str(t[1].get("audio_path", "")))))

    if not (0.0 <= float(target_percent) <= 100.0):
        raise ValueError("--target_percent must be 0..100")

    if float(target_percent) < 100.0:
        k = max(1, int(len(targets) * (float(target_percent) / 100.0)))
        targets = targets[:k]

    if int(target_take) > 0:
        targets = targets[: int(target_take)]

    info = {
        "rows_total": len(rows),
        "rows_labelled": labelled,
        "rows_unknown": unknown,
        "targets_selected": len(targets),
    }

    return [r for _, r in targets], info


def build_base_pairs_targets_vs_others(
    target_rows: List[dict],
    other_paths: List[Path],
    other_tx: Optional[TranscriptDB],
    mix_per_target: int,
    pairing_mode: str,
    seed: int,
    allow_missing_other_ref: bool,
) -> Tuple[List[dict], Dict]:

    if not target_rows:
        raise RuntimeError("No TARGET rows selected. Check CSV matching and transcripts in --test_manifest.")
    if not other_paths:
        raise RuntimeError("No OTHER audio files found in --others_dir.")

    # Filter others to only those with transcripts if you require other_ref
    usable_others: List[Path] = []
    missing_tx = 0
    for p in other_paths:
        if other_tx is None:
            usable_others.append(p)
            continue
        tx = other_tx.text_for(str(p))
        if tx and tx.strip():
            usable_others.append(p)
        else:
            missing_tx += 1

    if other_tx is not None and not allow_missing_other_ref:
        if not usable_others:
            raise RuntimeError(
                "You provided --others_manifest but none of the files in --others_dir matched a transcript.\n"
                "Fix matching (same basenames or full paths) OR set --allow_missing_other_ref=1."
            )

    others = usable_others if (other_tx is not None and not allow_missing_other_ref) else other_paths

    rng = random.Random(int(seed))

    pairs: List[dict] = []
    for i, trow in enumerate(target_rows):
        t_ap = str(trow.get("audio_path", "") or "")
        t_ref = _row_transcript(trow)
        t_key = core._norm_windows_key(t_ap)

        for k in range(int(mix_per_target)):
            if pairing_mode == "round_robin":
                oi = (i * int(mix_per_target) + k) % len(others)
            elif pairing_mode == "random":
                oi = rng.randrange(0, len(others))
            else:  # hash (default)
                oi = int(core._stable_uint64_from_str(f"{t_key}||k{k}", int(seed))) % len(others)

            o_path = others[oi]
            o_ap = str(o_path)
            o_ref = ""
            if other_tx is not None:
                o_ref = (other_tx.text_for(o_ap) or "").strip()

            if not o_ref and not allow_missing_other_ref:
                # try next few in a deterministic way
                found = False
                for j in range(1, min(25, len(others))):
                    oi2 = (oi + j) % len(others)
                    o_ap2 = str(others[oi2])
                    o_ref2 = (other_tx.text_for(o_ap2) or "").strip() if other_tx is not None else ""
                    if o_ref2:
                        o_ap = o_ap2
                        o_ref = o_ref2
                        found = True
                        break
                if not found:
                    continue

            if core._norm_windows_key(o_ap) == t_key:
                continue

            base_key = f"{t_key}||{core._norm_windows_key(o_ap)}||k{k}"

            pairs.append(
                {
                    "base_key": base_key,
                    "target_audio_path": t_ap,
                    "other_audio_path": o_ap,
                    "target_ref": t_ref,
                    "other_ref": o_ref,
                }
            )

    info = {
        "targets": len(target_rows),
        "others": len(other_paths),
        "others_missing_transcript": int(missing_tx),
        "pairs": len(pairs),
        "pairing_mode": pairing_mode,
        "mix_per_target": int(mix_per_target),
    }

    if not pairs:
        raise RuntimeError("No pairs were generated. Likely because other transcripts were missing.")

    return pairs, info


def load_pairs_jsonl(path: Path) -> List[dict]:
    return load_jsonl_rows(path)


def save_pairs_jsonl(pairs: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in pairs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--test_manifest", required=True, type=Path,
                    help="JSONL manifest of candidate TARGET rows (must include audio_path and raw_transcription).")
    ap.add_argument("--speaker_scores_csv", required=True, type=Path,
                    help="CSV from Stage 3 (columns: file,score,decision,reason...).")
    ap.add_argument("--checkpoint_dir", required=True, type=Path)

    # NEW: where OTHER files come from
    ap.add_argument("--others_dir", required=True, type=Path,
                    help="Folder containing OTHER-speaker audio files to mix with your TARGETs.")
    ap.add_argument("--others_manifest", default=None, type=Path,
                    help="Optional JSONL manifest for OTHER files (audio_path + raw_transcription).")

    # target selection
    ap.add_argument("--target_take", type=int, default=0,
                    help="0=all. Otherwise take top N TARGET rows by score.")
    ap.add_argument("--target_percent", type=float, default=100.0,
                    help="Take top X%% of TARGET rows by score (after sorting).")

    # evaluation subsample (applied to base pairs *after* construction)
    ap.add_argument("--percentage", type=float, default=100.0,
                    help="Optional subsample of base pairs for speed. Uses random sampling.")

    ap.add_argument("--allow_missing_other_ref", type=int, default=0,
                    help="If 0, OTHER files without transcripts are skipped/blocked when --others_manifest is given.")

    ap.add_argument("--pairs_manifest", type=Path, default=None,
                    help="Where to save/load the constructed base pairs JSONL (for stable resume).")
    ap.add_argument("--rebuild_pairs", action="store_true")

    # Models
    ap.add_argument("--base_model", default="futo-org/acft-whisper-tiny.en")
    ap.add_argument("--compare_openai_tiny", action="store_true")
    ap.add_argument("--base_processor_id", default="openai/whisper-tiny.en")

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--language", default="en")
    ap.add_argument("--task", default="transcribe")

    ap.add_argument("--num_beams", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)

    # batching
    ap.add_argument("--batch_size", type=int, default=0, help="0=auto default (GPU:8, CPU:1)")
    ap.add_argument("--auto_batch", type=int, default=0)
    ap.add_argument("--batch_min", type=int, default=1)
    ap.add_argument("--batch_max", type=int, default=64)

    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--mem_low", type=float, default=0.60)
    ap.add_argument("--mem_high", type=float, default=0.88)
    ap.add_argument("--cleanup_interval", type=int, default=50)

    ap.add_argument("--dynamic_audio_ctx", type=int, default=1)
    ap.add_argument("--normalize", default="whisper_basic", choices=["whisper_basic", "none"])

    # VAD
    ap.add_argument("--vad_filter", type=int, default=1)
    ap.add_argument("--vad_policy", default="skip", choices=["skip", "keep", "empty"])
    ap.add_argument("--vad_threshold", type=float, default=0.5)
    ap.add_argument("--vad_min_speech_ms", type=int, default=250)
    ap.add_argument("--vad_min_silence_ms", type=int, default=100)
    ap.add_argument("--vad_speech_pad_ms", type=int, default=200)

    # pairing
    ap.add_argument("--mix_per_target", type=int, default=1)
    ap.add_argument("--pairing_mode", default="hash", choices=["hash", "round_robin", "random"])
    ap.add_argument("--seed", type=int, default=42)

    # mixing rules
    ap.add_argument("--other_offset_mode", default="start", choices=["start", "random"],
                    help="Used only when overlap sweep is disabled.")
    ap.add_argument("--other_peak_ratio", type=float, default=1.0)

    # sweep grid
    ap.add_argument("--sweep_snr_db", type=str, default="20,10,5,0,-5", help="Comma/space separated list")
    ap.add_argument("--sweep_overlap", type=str, default="0,0.25,0.5,0.75,1", help="Comma/space separated overlap ratios in [0,1]")
    ap.add_argument("--disable_overlap_sweep", action="store_true")

    # caching
    ap.add_argument("--audio_cache_gb", type=float, default=1.0, help="0 disables")

    # misc
    ap.add_argument("--ffmpeg_path", default="ffmpeg", help="ffmpeg executable (needed for mp3/m4a).")

    # output/resume
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--force_resume", action="store_true")
    ap.add_argument("--recalc_metrics", action="store_true",
                    help="Recompute metrics from saved predictions for models already present in results (no inference).")

    args = ap.parse_args()

    if args.batch_size <= 0:
        args.batch_size = 8 if str(args.device).startswith("cuda") and torch.cuda.is_available() else 1

    if not (0.0 <= float(args.percentage) <= 100.0):
        raise ValueError("--percentage must be 0..100")

    # Monkeypatch core loader for broader format support
    core_load_orig = core.load_audio_mono_16k
    core.load_audio_mono_16k = make_loader_with_ffmpeg(str(args.ffmpeg_path), core_load_orig)

    # Load inputs
    rows = load_jsonl_rows(args.test_manifest)
    score_db = load_speaker_scores_csv(args.speaker_scores_csv)

    target_rows, target_info = select_targets_sorted(
        rows=rows,
        scores=score_db,
        target_take=int(args.target_take),
        target_percent=float(args.target_percent),
    )

    other_paths = scan_audio_files(args.others_dir)

    other_tx: Optional[TranscriptDB] = None
    if args.others_manifest is not None:
        other_tx = build_transcript_db_from_manifest(Path(args.others_manifest))

    # Build or load base pairs
    if args.pairs_manifest is None:
        args.pairs_manifest = args.checkpoint_dir / "pairs_manifest_targetmix_sweep_othersdir.jsonl"

    base_pairs: List[dict]
    pair_info: Dict

    if (args.resume or args.force_resume) and args.pairs_manifest.exists() and not args.rebuild_pairs:
        base_pairs = load_pairs_jsonl(args.pairs_manifest)
        pair_info = {
            "loaded_from": str(args.pairs_manifest),
            "base_pairs": len(base_pairs),
            "target_info": target_info,
        }
        print(f"✓ Loaded base pairs from {args.pairs_manifest} ({len(base_pairs)} pairs)")
    else:
        base_pairs, base_info = build_base_pairs_targets_vs_others(
            target_rows=target_rows,
            other_paths=other_paths,
            other_tx=other_tx,
            mix_per_target=int(args.mix_per_target),
            pairing_mode=str(args.pairing_mode),
            seed=int(args.seed),
            allow_missing_other_ref=bool(args.allow_missing_other_ref),
        )
        pair_info = {"target_info": target_info, "base_info": base_info}
        save_pairs_jsonl(base_pairs, args.pairs_manifest)
        print(f"✓ Saved base pairs to {args.pairs_manifest}")

    # Optional subsample base pairs BEFORE sweep
    if float(args.percentage) < 100.0:
        rng = random.Random(int(args.seed))
        k = max(1, int(len(base_pairs) * (float(args.percentage) / 100.0)))
        base_pairs = rng.sample(base_pairs, k)

    snr_list = core.parse_float_list(args.sweep_snr_db) or [10.0]
    if args.disable_overlap_sweep:
        overlap_list = None
    else:
        overlap_list = core.parse_overlap_list(args.sweep_overlap)

    conditions: List[core.MixCondition] = []
    if overlap_list is None:
        for snr in snr_list:
            conditions.append(core.MixCondition(snr_db=float(snr), overlap=None))
    else:
        for snr in snr_list:
            for ov in overlap_list:
                conditions.append(core.MixCondition(snr_db=float(snr), overlap=float(ov)))

    pair_rows = core.expand_pairs_with_conditions(base_pairs, conditions)

    # checkpoints - look for both model_epoch_* and s20_model_epoch_* patterns
    checkpoints = list(args.checkpoint_dir.glob("model_epoch_*"))
    checkpoints.extend(args.checkpoint_dir.glob("s20_model_epoch_*"))

    def _ckpt_key(p: Path) -> int:
        try:
            # Handle both "model_epoch_XXXXXX" and "s20_model_epoch_XXXXXX" patterns
            parts = p.name.split("_")
            if parts[0] == "s20" and parts[1] == "model" and parts[2] == "epoch":
                return int(parts[3])
            elif parts[0] == "model" and parts[1] == "epoch":
                return int(parts[2])
            else:
                return 0
        except Exception:
            return 0

    checkpoints.sort(key=_ckpt_key)

    models: List[str] = []
    if args.compare_openai_tiny:
        models.append("openai/whisper-tiny.en")
    models.append(str(args.base_model))
    models.extend([str(p) for p in checkpoints])

    processor = core.WhisperProcessor.from_pretrained(args.base_processor_id)

    vad_cfg = core.VADConfig(
        enabled=bool(args.vad_filter),
        policy=str(args.vad_policy),
        threshold=float(args.vad_threshold),
        min_speech_duration_ms=int(args.vad_min_speech_ms),
        min_silence_duration_ms=int(args.vad_min_silence_ms),
        speech_pad_ms=int(args.vad_speech_pad_ms),
    )

    audio_cache_bytes = int(max(0, float(args.audio_cache_gb)) * (1024**3))

    cfg = core.EvalConfig(
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
        pairing_mode=str(args.pairing_mode),
        seed=int(args.seed),
        other_offset_mode=str(args.other_offset_mode),
        other_peak_ratio=float(args.other_peak_ratio),
        conditions=conditions,
        audio_cache_bytes=audio_cache_bytes,
    )

    out_json = args.out_json
    if out_json is None:
        out_json = args.checkpoint_dir / "evaluation_results_futo_like_targetmix_sweep_othersdir.json"

    results = {
        "mode": "others_dir",
        "test_manifest": str(args.test_manifest),
        "speaker_scores_csv": str(args.speaker_scores_csv),
        "checkpoint_dir": str(args.checkpoint_dir),
        "others_dir": str(args.others_dir),
        "others_manifest": str(args.others_manifest) if args.others_manifest else None,
        "pairs_manifest": str(args.pairs_manifest),
        "target_take": int(args.target_take),
        "target_percent": float(args.target_percent),
        "percentage": float(args.percentage),
        "pair_info": pair_info,
        "base_pairs": len(base_pairs),
        "conditions": [{"snr_db": c.snr_db, "overlap": c.overlap, "cond_id": c.cond_id()} for c in conditions],
        "pairs_total": len(pair_rows),
        "cfg": {
            "device": cfg.device,
            "language": cfg.language,
            "task": cfg.task,
            "num_beams": cfg.num_beams,
            "temperature": cfg.temperature,
            "max_new_tokens": cfg.max_new_tokens,
            "dynamic_audio_ctx": cfg.dynamic_audio_ctx,
            "normalize_mode": cfg.normalize_mode,
            "vad": vad_cfg.__dict__,
            "batch": {
                "batch_size": cfg.batch_size,
                "auto_batch": cfg.auto_batch,
                "batch_min": cfg.batch_min,
                "batch_max": cfg.batch_max,
                "fp16": cfg.fp16,
                "mem_low": cfg.mem_low,
                "mem_high": cfg.mem_high,
                "cleanup_interval": cfg.cleanup_interval,
            },
            "pairing": {
                "mix_per_target": cfg.mix_per_target,
                "pairing_mode": cfg.pairing_mode,
                "seed": cfg.seed,
            },
            "mixing": {
                "other_offset_mode": cfg.other_offset_mode,
                "other_peak_ratio": cfg.other_peak_ratio,
            },
            "audio_cache_gb": float(args.audio_cache_gb),
            "ffmpeg_path": str(args.ffmpeg_path),
        },
        "models": [],
    }

    all_predictions: Dict[str, dict] = {}

    if args.resume or args.force_resume:
        existing_results, existing_predictions = core.load_existing_results(out_json)
        if existing_results:
            results = existing_results
            all_predictions = existing_predictions

    evaluated_models = {m["model"] for m in results.get("models", [])}

    print(f"Device: {args.device}")
    print(f"Targets selected: {target_info.get('targets_selected')}")
    print(f"OTHER files in folder: {len(other_paths)}")
    print(f"Base pairs: {len(base_pairs)}")
    print(f"Conditions: {len(conditions)}")
    print(f"Total eval pairs: {len(pair_rows)}")
    print(f"Models to eval: {len(models)}")
    print(f"Already evaluated models: {len(evaluated_models)}")

    # Ensure prediction stubs exist for every pair (helps force_resume)
    for pr in pair_rows:
        key = pr["mix_key"]
        if key not in all_predictions:
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

    vad_trimmer = core.SileroVADTrimmer() if cfg.vad.enabled else None

    for m in models:
        model_already_done = m in evaluated_models
        model_name = Path(m).name if Path(m).exists() else m
        wants_recalc = args.recalc_metrics or args.force_resume

        if model_already_done and not wants_recalc:
            print(f"\n⏭ Skipping already evaluated model: {m}")
            continue

        print("\n" + "=" * 80)
        if model_already_done and wants_recalc:
            print(f"Re-evaluating metrics from saved predictions: {m}")
        else:
            print(f"Evaluating: {m}")
        print("=" * 80)

        if model_already_done and wants_recalc:
            has_predictions = any(model_name in v.get("predictions", {}) for v in all_predictions.values())
            if has_predictions:
                print(f"📊 Recalculating metrics from existing predictions for {model_name}")
                overall, by_cond = core.recompute_metrics_from_saved_predictions(
                    all_predictions, model_name, cfg.normalize_mode
                )
                # Replace old metrics entry (if any) to avoid duplicates/stale data.
                results["models"] = [mm for mm in results.get("models", []) if mm.get("model") != m]
                results["models"].append({"model": m, "metrics_overall": overall, "metrics_by_condition": by_cond})
                core.save_incremental_results(results, all_predictions, out_json)
                continue
            else:
                print("⚠ Recalc requested but no saved predictions found; running full evaluation.")

        overall, by_cond = core.eval_one_model(
            model_id_or_path=m,
            pair_rows=pair_rows,
            processor=processor,
            cfg=cfg,
            vad_trimmer=vad_trimmer,
            all_predictions=all_predictions,
            out_json=out_json,
        )

        results["models"].append({"model": m, "metrics_overall": overall, "metrics_by_condition": by_cond})
        core.save_incremental_results(results, all_predictions, out_json)

        print(f"samples={overall.get('samples')} skipped={overall.get('skipped')}")
        print(f"WER target micro={overall.get('wer_micro_target')} | WER other micro={overall.get('wer_micro_other')}")
        print(f"CER target micro={overall.get('cer_micro_target')} | CER other micro={overall.get('cer_micro_other')}")
        print(f"win_rate(target closer)={overall.get('win_rate_target_closer')} avg_margin(other-target)={overall.get('avg_margin_other_minus_target')}")

    print("\nDone.")
    core.beep()


if __name__ == "__main__":
    main()
