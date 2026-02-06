#!/usr/bin/env python3
"""stage_10b_add_speech_tempo_pause_aware_idempotent.py

Stage 10b (Mach1-lite): pause-aware, non-uniform time compression.

Goal
- Sound closer to a *natural fast speaker* than simple uniform tempo.
- Core idea from Mach1-style approaches: compress pauses/silences much more than speech.

Approach (practical + robust)
1) Canonicalise input to PCM WAV (sample_rate/channels/bit_depth).
2) Detect silences via ffmpeg silencedetect.
3) Split into speech + silence segments.
4) Apply SoX `tempo -s` to speech segments (mild, e.g. 1.05–1.20).
5) For silence segments:
   - --pause_policy compress: apply SoX `tempo -s <silence_factor>` (keeps any room tone, just shorter)
   - --pause_policy truncate: keep only (duration / silence_factor) seconds (hard cut; often sounds more natural)
6) Concatenate processed segments.

Idempotency
- Same manifest lineage pattern as Stage 10.
- Resume-safe via SQLite --seen_db + output WAV validation.

References
- Mach1: Nonuniform time-scale modification emphasises heavier pause compression vs speech. (Covell & Withgott, 1998)
- FFmpeg silencedetect filter provides silence_start/silence_end ranges.
- SoX supports `tempo -s` and `--combine concatenate`.

Example

& "I:\\Whisper-training-env\\Scripts\\python.exe" "I:\\whisper-acft\\stage_10b_add_speech_tempo_pause_aware_idempotent.py" `
  --in_manifest  "I:\Record_chunks\pairs_manifest_stereo_english_only_filtered_with_uids_score_bottom_filtered.jsonl" `
  --out_manifest "I:\\Record_chunks\\pairs_manifest_stage10b_tempo_pause.jsonl" `
  --out_dir      "I:\\Record_chunks_tempo_pause" `
  --ratio 0.30 `
  --copies 1 `
  --tempo_min 1.05 --tempo_max 1.20 `
  --mode choice --tempo_factors "1.05,1.07,1.09,1.10,1.12,1.14,1.16,1.18,1.20" `
  --pause_policy truncate --silence_factor 2.8 `
  --silence_noise_db -35 --silence_min_dur 0.15 `
  --workers 4 `
  --stage_name tempo_speech_pause `
  --seen_db "I:\\Record_chunks\\seen_stage10b_tempo_pause.sqlite"

"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from pipeline_uid_utils import (
    SQLiteSeenSet,
    default_seen_db,
    is_valid_wav,
    safe_unlink,
    make_aug_uid,
    rng_for,
    safe_beep,
    should_select,
)


# -------------------------
# Basic row eligibility
# -------------------------

def is_original_row(row: Dict[str, Any]) -> bool:
    # Mirror Stage 8/9/10 style: only process un-augmented base rows by default.
    if row.get("aug_stage"):
        return False
    base_uid = row.get("base_uid")
    uid = row.get("uid") or base_uid
    if base_uid and uid and uid != base_uid:
        return False
    return True


# -------------------------
# Factor selection
# -------------------------

def _parse_factors(s: str) -> List[float]:
    out: List[float] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except Exception:
            continue
    out = [x for x in out if x > 0]
    return out


def _quantise(x: float) -> float:
    return float(round(float(x), 3))


def choose_tempo(base_uid: str, stage_name: str, copy_idx: int, args) -> float:
    rng = rng_for(base_uid, stage_name, copy_idx)
    if args.mode == "choice":
        factors = _parse_factors(args.tempo_factors)
        if factors:
            return _quantise(rng.choice(factors))
    lo = float(args.tempo_min)
    hi = float(args.tempo_max)
    if hi <= 0 or lo <= 0 or hi < lo:
        lo, hi = 1.05, 1.20
    return _quantise(rng.uniform(lo, hi))


# -------------------------
# External tool availability
# -------------------------

def _ensure_exe_available(exe: str, hint: str) -> str:
    p = Path(exe)
    if p.exists():
        return str(p)
    found = shutil.which(exe)
    if found:
        return found
    raise SystemExit(f"Required executable not found: {exe}. {hint}")


def _ensure_tools(args) -> None:
    args.sox = _ensure_exe_available(
        args.sox,
        "Install SoX or pass --sox <path-to-sox.exe>. (Chocolatey: choco install sox.portable)",
    )
    args.ffmpeg = _ensure_exe_available(
        args.ffmpeg,
        "Install ffmpeg or pass --ffmpeg <path-to-ffmpeg.exe>.",
    )


# -------------------------
# Audio prep: canonical WAV
# -------------------------

def sox_convert_to_canonical(
    sox_exe: str,
    in_audio: Path,
    out_wav: Path,
    sample_rate: int,
    channels: int,
    bit_depth: int,
) -> Tuple[bool, str]:
    # sox IN -r SR -c CH -b BIT OUT
    cmd = [
        sox_exe,
        str(in_audio),
        "-r",
        str(int(sample_rate)),
        "-c",
        str(int(channels)),
        "-b",
        str(int(bit_depth)),
        str(out_wav),
    ]
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if cp.returncode != 0:
            err = (cp.stderr or cp.stdout or "").strip()
            if len(err) > 300:
                err = err[:300] + " …"
            return False, f"sox-convert-failed({cp.returncode}): {err}"
        return True, "ok"
    except Exception as e:
        return False, f"sox-convert-exception:{type(e).__name__}:{e}"


# -------------------------
# Silence detection via ffmpeg
# -------------------------

_SIL_START_RE = re.compile(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)")
_SIL_END_RE = re.compile(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)")


def get_wav_duration_sec(wav_path: Path) -> Optional[float]:
    try:
        import soundfile as sf

        info = sf.info(str(wav_path))
        if not info.frames or not info.samplerate:
            return None
        return float(info.frames) / float(info.samplerate)
    except Exception:
        return None


def detect_silences_ffmpeg(
    ffmpeg_exe: str,
    wav_path: Path,
    noise_db: float,
    min_dur: float,
) -> Tuple[bool, List[Tuple[float, float]], str]:
    """Return list of (sil_start, sil_end) in seconds."""

    # Example:
    # ffmpeg -i in.wav -af silencedetect=n=-35dB:d=0.15 -f null -
    filt = f"silencedetect=n={noise_db}dB:d={min_dur}"
    cmd = [
        ffmpeg_exe,
        "-hide_banner",
        "-nostats",
        "-i",
        str(wav_path),
        "-af",
        filt,
        "-f",
        "null",
        "-",
    ]

    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        # silencedetect logs to stderr
        log = cp.stderr or ""
        if cp.returncode != 0 and not log:
            return False, [], f"ffmpeg-silencedetect-failed({cp.returncode})"

        starts: List[float] = []
        ends: List[float] = []
        for line in log.splitlines():
            m = _SIL_START_RE.search(line)
            if m:
                starts.append(float(m.group(1)))
            m = _SIL_END_RE.search(line)
            if m:
                ends.append(float(m.group(1)))

        # Pair up in order (ffmpeg emits start then end; trailing silence may have start without end)
        silences: List[Tuple[float, float]] = []
        j = 0
        for s in starts:
            # Find the next end >= s
            while j < len(ends) and ends[j] < s:
                j += 1
            if j < len(ends):
                silences.append((s, ends[j]))
                j += 1
            else:
                # Trailing silence: end = duration
                dur = get_wav_duration_sec(wav_path)
                if dur is None:
                    # fallback: ignore incomplete trailing silence
                    continue
                silences.append((s, float(dur)))

        # Sort + merge overlaps
        silences.sort(key=lambda x: x[0])
        return True, _merge_intervals(silences, gap=0.0), "ok"

    except Exception as e:
        return False, [], f"ffmpeg-silencedetect-exception:{type(e).__name__}:{e}"


def _merge_intervals(iv: List[Tuple[float, float]], gap: float = 0.02) -> List[Tuple[float, float]]:
    if not iv:
        return []
    out = []
    cur_s, cur_e = iv[0]
    for s, e in iv[1:]:
        if s <= cur_e + gap:
            cur_e = max(cur_e, e)
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    out.append((cur_s, cur_e))
    return out


def build_segments(
    total_dur: float,
    silences: List[Tuple[float, float]],
    edge_pad: float,
    min_seg: float,
) -> List[Tuple[str, float, float]]:
    """Build labeled segments: ('speech'|'silence', start, end)."""

    # Apply edge padding: treat small slices near speech as speech by shrinking silence.
    adj: List[Tuple[float, float]] = []
    for s, e in silences:
        s2 = max(0.0, s + edge_pad)
        e2 = min(total_dur, e - edge_pad)
        if e2 > s2:
            adj.append((s2, e2))

    adj = _merge_intervals(adj, gap=0.0)

    segs: List[Tuple[str, float, float]] = []
    t = 0.0
    for s, e in adj:
        if s > t:
            segs.append(("speech", t, s))
        segs.append(("silence", s, e))
        t = e
    if t < total_dur:
        segs.append(("speech", t, total_dur))

    # Drop tiny segments by merging into neighbours
    if min_seg > 0:
        segs = _merge_tiny_segments(segs, min_seg)

    return segs


def _merge_tiny_segments(segs: List[Tuple[str, float, float]], min_seg: float) -> List[Tuple[str, float, float]]:
    if not segs:
        return segs
    out: List[Tuple[str, float, float]] = []
    for kind, s, e in segs:
        if e - s < min_seg:
            # merge with previous if possible
            if out and out[-1][0] == kind:
                pk, ps, pe = out[-1]
                out[-1] = (pk, ps, e)
            elif out:
                # merge into previous regardless of kind (prefer no micro-gaps)
                pk, ps, pe = out[-1]
                out[-1] = (pk, ps, e)
            else:
                out.append((kind, s, e))
        else:
            if out and out[-1][0] == kind:
                pk, ps, pe = out[-1]
                out[-1] = (pk, ps, e)
            else:
                out.append((kind, s, e))
    return out


# -------------------------
# SoX segment processing
# -------------------------

def is_canonical_wav(
    wav_path: Path,
    sample_rate: int,
    channels: int,
    bit_depth: int,
) -> bool:
    try:
        import soundfile as sf

        info = sf.info(str(wav_path))
        if info.format != "WAV":
            return False
        if info.samplerate != int(sample_rate) or info.channels != int(channels):
            return False

        bits = getattr(info, "bits_per_sample", None)
        if bits is not None:
            if int(bits) != int(bit_depth):
                return False
        else:
            subtype = (info.subtype or "").upper()
            if str(int(bit_depth)) not in subtype:
                return False
        return True
    except Exception:
        return False


def sox_trim(
    sox_exe: str,
    in_wav: Path,
    out_wav: Path,
    start: float,
    dur: float,
) -> Tuple[bool, str]:
    cmd = [sox_exe, str(in_wav), str(out_wav), "trim", f"{start:.6f}", f"{dur:.6f}"]
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if cp.returncode != 0:
            err = (cp.stderr or cp.stdout or "").strip()
            if len(err) > 300:
                err = err[:300] + " …"
            return False, f"sox-trim-failed({cp.returncode}): {err}"
        return True, "ok"
    except Exception as e:
        return False, f"sox-trim-exception:{type(e).__name__}:{e}"


def sox_tempo_speech(
    sox_exe: str,
    in_wav: Path,
    out_wav: Path,
    factor: float,
) -> Tuple[bool, str]:
    cmd = [sox_exe, str(in_wav), str(out_wav), "tempo", "-s", f"{float(factor):.3f}"]
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if cp.returncode != 0:
            err = (cp.stderr or cp.stdout or "").strip()
            if len(err) > 300:
                err = err[:300] + " …"
            return False, f"sox-tempo-failed({cp.returncode}): {err}"
        return True, "ok"
    except Exception as e:
        return False, f"sox-tempo-exception:{type(e).__name__}:{e}"


def sox_concat(
    sox_exe: str,
    in_wavs: List[Path],
    out_wav: Path,
) -> Tuple[bool, str]:
    if not in_wavs:
        return False, "concat-no-inputs"

    # SoX: sox --combine concatenate in1.wav in2.wav ... out.wav
    cmd = [sox_exe, "--combine", "concatenate"] + [str(p) for p in in_wavs] + [str(out_wav)]
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if cp.returncode != 0:
            err = (cp.stderr or cp.stdout or "").strip()
            if len(err) > 300:
                err = err[:300] + " …"
            return False, f"sox-concat-failed({cp.returncode}): {err}"
        return True, "ok"
    except Exception as e:
        return False, f"sox-concat-exception:{type(e).__name__}:{e}"


# -------------------------
# Output naming
# -------------------------

def build_out_wav_name(
    row: Dict[str, Any],
    stage_name: str,
    new_uid: str,
    copy_idx: int,
    tempo_factor: float,
    silence_factor: float,
    pause_policy: str,
    out_dir: Path,
) -> str:
    base_uid = (row.get("base_uid") or row.get("uid") or "")[:12]
    aug_uid = (new_uid or "")[:12]

    ttag = f"{tempo_factor:.2f}".replace(".", "p")
    stag = f"{silence_factor:.2f}".replace(".", "p")
    ptag = "tr" if pause_policy == "truncate" else "cp"

    fname = f"{base_uid}_{aug_uid}__{stage_name}__t{ttag}__s{stag}__{ptag}__c{copy_idx:02d}.wav"
    return str(out_dir / fname)


# -------------------------
# Worker: process one row+copy
# -------------------------

def process_one(
    row: Dict[str, Any],
    stage_name: str,
    copy_idx: int,
    out_dir: Path,
    args,
    seen: SQLiteSeenSet,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    base_uid = row.get("base_uid") or row.get("uid")
    if not base_uid:
        return False, None, "missing base_uid"

    tempo = choose_tempo(base_uid, stage_name, copy_idx, args)
    silence_factor = float(args.silence_factor)
    pause_policy = args.pause_policy

    extra = f"tempo{tempo:.3f}|sil{silence_factor:.3f}|{pause_policy}|n{args.silence_noise_db}|d{args.silence_min_dur}"
    aug_key = f"{base_uid}:{stage_name}:{copy_idx}:{extra}"

    if seen.contains(aug_key):
        return True, None, "already-seen"

    new_uid = make_aug_uid(base_uid, stage_name, copy_idx, extra=extra)
    out_wav = build_out_wav_name(row, stage_name, new_uid, copy_idx, tempo, silence_factor, pause_policy, out_dir)
    out_wav_p = Path(out_wav)

    # Already exists + valid => mark seen
    if out_wav_p.exists() and out_wav_p.stat().st_size > 0:
        if is_valid_wav(out_wav_p, min_frames=16):
            seen.add(aug_key)
            return True, None, "already-exists"
        safe_unlink(out_wav_p)

    in_ap = Path(row.get("audio_path", ""))
    if not in_ap.exists():
        return False, None, f"missing-audio:{in_ap}"

    if args.dry_run:
        out_row = dict(row)
        out_row["parent_uid"] = row.get("uid") or base_uid
        out_row["base_uid"] = base_uid
        out_row["uid"] = new_uid
        out_row["aug_stage"] = stage_name
        out_row["aug_copy_idx"] = copy_idx
        out_row["out_wav"] = out_wav
        out_row["audio_path"] = out_wav
        out_row.setdefault("aug_meta", {})
        out_row["aug_meta"] = {
            **out_row["aug_meta"],
            "tempo_factor": float(tempo),
            "pause_policy": pause_policy,
            "silence_factor": float(silence_factor),
            "dry_run": True,
        }
        seen.add(aug_key)
        return True, out_row, "dry-run"

    # Temp workspace per job
    job_dir = out_dir / "_tmp" / f"{base_uid[:12]}_{new_uid[:12]}_{os.getpid()}_{threading.get_ident()}"
    job_dir.mkdir(parents=True, exist_ok=True)

    use_original = (
        args.skip_canonical_if_ok
        and in_ap.suffix.lower() == ".wav"
        and is_canonical_wav(in_ap, int(args.sample_rate), int(args.channels), int(args.bit_depth))
        and is_valid_wav(in_ap, min_frames=16)
    )

    if use_original:
        tmp_in = in_ap
    else:
        tmp_in = job_dir / "in_canon.wav"
        ok, status = sox_convert_to_canonical(
            args.sox,
            in_ap,
            tmp_in,
            sample_rate=int(args.sample_rate),
            channels=int(args.channels),
            bit_depth=int(args.bit_depth),
        )
        if not ok:
            _cleanup_dir(job_dir)
            return False, None, status

        if not is_valid_wav(tmp_in, min_frames=16):
            _cleanup_dir(job_dir)
            return False, None, "canonical-invalid-wav"

    total_dur = get_wav_duration_sec(tmp_in)
    if total_dur is None or total_dur <= 0:
        _cleanup_dir(job_dir)
        return False, None, "duration-unavailable"

    ok, sils, status = detect_silences_ffmpeg(
        args.ffmpeg,
        tmp_in,
        noise_db=float(args.silence_noise_db),
        min_dur=float(args.silence_min_dur),
    )
    if not ok:
        # Fallback: treat whole file as speech
        sils = []

    segs = build_segments(
        total_dur=float(total_dur),
        silences=sils,
        edge_pad=float(args.edge_pad_sec),
        min_seg=float(args.min_segment_sec),
    )

    processed: List[Path] = []

    min_effect_sec = float(args.min_effect_sec)

    for i, (kind, s, e) in enumerate(segs):
        dur = max(0.0, float(e) - float(s))
        if dur <= 0:
            continue

        # optionally drop ultra-short silences entirely
        if kind == "silence" and dur < float(args.drop_silence_below_sec):
            continue

        seg_wav = job_dir / f"seg_{i:04d}_{kind}.wav"
        ok, st = sox_trim(args.sox, tmp_in, seg_wav, start=float(s), dur=float(dur))
        if not ok:
            _cleanup_dir(job_dir)
            return False, None, st

        # For extremely short segments, avoid tempo (can break WSOLA-style effects)
        if dur < min_effect_sec:
            proc_wav = job_dir / f"proc_{i:04d}_{kind}.wav"
            # copy via sox (keeps wav header consistent)
            ok, st = sox_trim(args.sox, seg_wav, proc_wav, start=0.0, dur=float(dur))
            if not ok:
                _cleanup_dir(job_dir)
                return False, None, f"short-copy-failed:{st}"
            processed.append(proc_wav)
            continue

        if kind == "speech":
            proc_wav = job_dir / f"proc_{i:04d}_speech.wav"
            ok, st = sox_tempo_speech(args.sox, seg_wav, proc_wav, factor=float(tempo))
            if not ok:
                _cleanup_dir(job_dir)
                return False, None, st
            processed.append(proc_wav)
        else:
            if pause_policy == "compress":
                proc_wav = job_dir / f"proc_{i:04d}_silence_cp.wav"
                ok, st = sox_tempo_speech(args.sox, seg_wav, proc_wav, factor=float(silence_factor))
                if not ok:
                    _cleanup_dir(job_dir)
                    return False, None, st
                processed.append(proc_wav)
            else:
                # truncate: keep only dur/silence_factor seconds (hard cut)
                keep = float(dur) / max(1e-6, float(silence_factor))
                keep = max(0.0, keep)
                if keep <= 0:
                    continue
                proc_wav = job_dir / f"proc_{i:04d}_silence_tr.wav"
                ok, st = sox_trim(args.sox, seg_wav, proc_wav, start=0.0, dur=float(keep))
                if not ok:
                    _cleanup_dir(job_dir)
                    return False, None, st
                processed.append(proc_wav)

    if not processed:
        _cleanup_dir(job_dir)
        return False, None, "no-processed-segments"

    tmp_out = out_wav_p.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}.wav")
    safe_unlink(tmp_out)

    ok, st = sox_concat(args.sox, processed, tmp_out)
    if not ok:
        safe_unlink(tmp_out)
        _cleanup_dir(job_dir)
        return False, None, st

    if not is_valid_wav(tmp_out, min_frames=16):
        safe_unlink(tmp_out)
        _cleanup_dir(job_dir)
        return False, None, "concat-invalid-wav"

    try:
        os.replace(str(tmp_out), str(out_wav_p))
    except Exception as e:
        safe_unlink(tmp_out)
        _cleanup_dir(job_dir)
        return False, None, f"replace-failed:{type(e).__name__}:{e}"

    if not is_valid_wav(out_wav_p, min_frames=16):
        safe_unlink(out_wav_p)
        _cleanup_dir(job_dir)
        return False, None, "final-invalid-wav"

    out_row = dict(row)
    out_row["parent_uid"] = row.get("uid") or base_uid
    out_row["base_uid"] = base_uid
    out_row["uid"] = new_uid
    out_row["aug_stage"] = stage_name
    out_row["aug_copy_idx"] = copy_idx
    out_row["out_wav"] = str(out_wav_p)
    out_row["audio_path"] = str(out_wav_p)

    out_row.setdefault("aug_meta", {})
    out_row["aug_meta"] = {
        **out_row["aug_meta"],
        "tempo_factor": float(tempo),
        "pause_policy": pause_policy,
        "silence_factor": float(silence_factor),
        "silence_noise_db": float(args.silence_noise_db),
        "silence_min_dur": float(args.silence_min_dur),
        "edge_pad_sec": float(args.edge_pad_sec),
        "min_segment_sec": float(args.min_segment_sec),
        "engine": "ffmpeg_silencedetect + sox tempo -s",
    }

    seen.add(aug_key)
    _cleanup_dir(job_dir)
    return True, out_row, "ok"


def _cleanup_dir(p: Path) -> None:
    # Best-effort cleanup; keep failures silent (Windows file locks happen)
    try:
        if p.exists():
            for child in sorted(p.glob("**/*"), reverse=True):
                try:
                    if child.is_file():
                        child.unlink()
                    elif child.is_dir():
                        child.rmdir()
                except Exception:
                    pass
            try:
                p.rmdir()
            except Exception:
                pass
    except Exception:
        pass


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_manifest", required=True)
    ap.add_argument("--out_manifest", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--stage_name", default="tempo_speech_pause")
    ap.add_argument("--ratio", type=float, required=True)
    ap.add_argument("--copies", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)

    ap.add_argument("--sox", default="sox", help="Path to sox.exe or 'sox' if on PATH")
    ap.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg.exe or 'ffmpeg' if on PATH")

    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--bit_depth", type=int, default=16)

    ap.add_argument("--tempo_min", type=float, default=1.05)
    ap.add_argument("--tempo_max", type=float, default=1.20)
    ap.add_argument("--tempo_factors", default="1.05,1.10,1.15,1.20")
    ap.add_argument("--mode", choices=["random_uniform", "choice"], default="choice")

    ap.add_argument("--pause_policy", choices=["compress", "truncate"], default="truncate")
    ap.add_argument("--silence_factor", type=float, default=2.5, help=">1 shrinks silences more")

    ap.add_argument("--silence_noise_db", type=float, default=-35.0)
    ap.add_argument("--silence_min_dur", type=float, default=0.15)

    ap.add_argument(
        "--edge_pad_sec",
        type=float,
        default=0.03,
        help="Shrink detected silences by this pad on both sides (treat near-voiced silence as speech).",
    )

    ap.add_argument(
        "--min_segment_sec",
        type=float,
        default=0.04,
        help="Merge tiny segments into neighbours to avoid micro-chops.",
    )

    ap.add_argument(
        "--min_effect_sec",
        type=float,
        default=0.12,
        help="Below this duration, skip tempo effect (just keep segment).",
    )

    ap.add_argument(
        "--drop_silence_below_sec",
        type=float,
        default=0.02,
        help="Drop ultra-short silences completely (often just boundary jitter).",
    )

    ap.add_argument(
        "--skip_canonical_if_ok",
        action="store_true",
        help="If input is already mono/16k/16b WAV, reuse it instead of re-encoding.",
    )

    ap.add_argument("--seen_db", default="")
    ap.add_argument("--allow_augmented_input", action="store_true")
    ap.add_argument("--dry_run", action="store_true")

    args = ap.parse_args()

    _ensure_tools(args)

    in_path = Path(args.in_manifest)
    out_path = Path(args.out_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen_db = args.seen_db or default_seen_db(out_path, args.stage_name)
    seen = SQLiteSeenSet(seen_db)

    max_pending = max(8, int(args.workers) * 4)
    pending: List[Future] = []

    n_total = 0
    n_selected = 0
    n_submitted = 0

    def flush_one(fut: Future, f_out) -> None:
        ok, out_row, status = fut.result()
        if out_row:
            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("a", encoding="utf-8") as f_out:
        with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            pbar = tqdm(total=0, desc=f"{args.stage_name} augment", unit="job")

            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                n_total += 1

                if not args.allow_augmented_input and not is_original_row(row):
                    continue

                base_uid = row.get("base_uid") or row.get("uid")
                if not base_uid:
                    continue

                if not should_select(base_uid, args.stage_name, float(args.ratio)):
                    continue

                n_selected += 1

                for copy_idx in range(1, int(args.copies) + 1):
                    fut = ex.submit(process_one, row, args.stage_name, copy_idx, out_dir, args, seen)
                    pending.append(fut)
                    n_submitted += 1

                    if len(pending) >= max_pending:
                        done = pending.pop(0)
                        flush_one(done, f_out)
                        pbar.total = n_submitted
                        pbar.update(1)

            for fut in pending:
                flush_one(fut, f_out)
                pbar.total = n_submitted
                pbar.update(1)
            pbar.close()

    seen.commit()
    seen.close()

    print(
        f"Stage {args.stage_name}: scanned {n_total} rows; selected {n_selected} base rows; "
        f"submitted {n_submitted} job(s) (copies={int(args.copies)})."
    )
    safe_beep()


if __name__ == "__main__":
    main()
