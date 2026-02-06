"""manifest_sanity_check_and_poison_guard.py



https://chatgpt.com/g/g-p-6969433d33d4819187ec3158a8f3745f-whisper-training/c/6969613e-5b3c-8323-a964-bd0c0d9e9736





Goal
----
Find (and optionally filter out) manifest rows that can poison Whisper training:
- missing files
- wrong sample rate
- 0/negative duration
- too-long duration
- NaN/Inf samples
- extreme amplitude / silence mismatched with non-empty transcript

Usage
-----
python manifest_sanity_check_and_poison_guard.py \
  --manifest_in I:\\Record_chunks\pairs_manifest_filtered_with_noises_with_mix_gain_rir_silent.jsonl \
  --manifest_out I:\\Record_chunks\\pairs_manifest.CLEAN.jsonl \
  --report_out I:\\Record_chunks\\manifest_bad_rows.csv



  python manifest_sanity_check_and_poison_guard.py --manifest_in "I:\Record_chunks\pairs_manifest_filtered_with_noises_with_mix_gain_rir_silent.jsonl" --manifest_out "I:\Record_chunks\pairs_manifest.CLEAN.jsonl" --report_out "I:\Record_chunks\manifest_bad_rows.csv"

Optional:
  --target_sr 16000
  --max_sec 30
  --min_sec 0.05
  --decode_check_sec 2.0
  --drop_silent_with_text

Notes
-----
This is intentionally conservative: it only *drops* rows when there's a clear problem.
For anything borderline, it keeps the row but reports it.
"""

from __future__ import annotations

import argparse
import csv
import json

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception as e:
    raise SystemExit(f"soundfile is required. Install with: pip install soundfile. Error: {e}")


@dataclass
class RowIssue:
    line_no: int
    audio_path: str
    reason: str
    detail: str


def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except Exception as e:
                yield i, {"__parse_error__": True, "__raw__": line, "__error__": repr(e)}


def quick_header(path: str) -> Tuple[Optional[int], Optional[int], Optional[float], Optional[str]]:
    """Return (sr, frames, dur_sec, subtype) or (None, None, None, None) on failure."""
    try:
        info = sf.info(path)
        sr = int(info.samplerate)
        frames = int(info.frames)
        dur = float(frames) / float(sr) if sr > 0 else None
        return sr, frames, dur, str(info.subtype)
    except Exception:
        return None, None, None, None


def decode_prefix(path: str, seconds: float, sr_expected: int) -> Optional[np.ndarray]:
    """Decode up to `seconds` from file. Returns float32 mono or None on failure."""
    try:
        # Read whole file if it is short; otherwise only prefix frames.
        info = sf.info(path)
        sr = int(info.samplerate)
        if sr != sr_expected:
            return None
        max_frames = int(seconds * sr)
        frames_to_read = min(int(info.frames), max_frames)
        wav, sr2 = sf.read(path, dtype="float32", always_2d=False, frames=frames_to_read)
        if int(sr2) != sr_expected:
            return None
        if wav is None:
            return None
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=-1)
        return wav
    except Exception:
        return None


def analyse_row(
    line_no: int,
    obj: Dict,
    target_sr: int,
    max_sec: float,
    min_sec: float,
    decode_check_sec: float,
    drop_silent_with_text: bool,
) -> Tuple[bool, List[RowIssue]]:
    """Return (keep_row, issues)."""
    issues: List[RowIssue] = []

    if obj.get("__parse_error__"):
        issues.append(RowIssue(line_no, "<jsonl>", "json_parse_error", obj.get("__error__", "")))
        return False, issues

    ap = obj.get("audio_path") or obj.get("audio")
    txt = (obj.get("raw_transcription") or obj.get("text") or "").strip()

    if not ap or not isinstance(ap, str):
        issues.append(RowIssue(line_no, str(ap), "missing_audio_path", "audio_path is missing or not a string"))
        return False, issues

    if not os.path.exists(ap):
        issues.append(RowIssue(line_no, ap, "missing_file", "file does not exist"))
        return False, issues

    sr, frames, dur, subtype = quick_header(ap)
    if sr is None or dur is None:
        issues.append(RowIssue(line_no, ap, "bad_header", "soundfile.info failed"))
        return False, issues

    if sr != target_sr:
        issues.append(RowIssue(line_no, ap, "wrong_sample_rate", f"sr={sr} expected={target_sr}"))
        return False, issues

    if frames <= 0 or dur <= 0:
        issues.append(RowIssue(line_no, ap, "zero_or_negative_duration", f"frames={frames} dur={dur}"))
        return False, issues

    if dur > max_sec:
        issues.append(RowIssue(line_no, ap, "too_long", f"dur={dur:.3f}s max={max_sec}"))
        return False, issues

    if dur < min_sec:
        # Too short is not always wrong, but it is commonly junk.
        issues.append(RowIssue(line_no, ap, "very_short", f"dur={dur:.3f}s min={min_sec}"))
        # Keep, but report.

    # Decode prefix and check for NaN/Inf and amplitude
    wav = decode_prefix(ap, seconds=decode_check_sec, sr_expected=target_sr)
    if wav is None:
        issues.append(RowIssue(line_no, ap, "decode_failed", f"could not decode prefix or sr mismatch"))
        return False, issues

    if not np.isfinite(wav).all():
        issues.append(RowIssue(line_no, ap, "non_finite_audio", "wav contains NaN/Inf"))
        return False, issues

    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(wav)))) if wav.size else 0.0

    if peak == 0.0 or rms == 0.0:
        # Pure silence (or all zeros) is fine *only if* the label indicates no speech.
        # You decide how strict you want this.
        if txt:
            msg = f"silent_audio_with_text peak={peak:.6g} rms={rms:.6g} text_len={len(txt)}"
            issues.append(RowIssue(line_no, ap, "silent_with_text", msg))
            if drop_silent_with_text:
                return False, issues

    # Extreme amplitude is suspicious (usually indicates clipping/mis-scaling)
    # float32 WAV decoded by soundfile should be roughly within [-1,1] for PCM.
    if peak > 1.2:
        issues.append(RowIssue(line_no, ap, "amplitude_too_high", f"peak={peak:.3f} (possible scaling bug)"))
        # Keep but report (you may choose to drop)

    return True, issues


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_in", required=True)
    ap.add_argument("--manifest_out", required=True)
    ap.add_argument("--report_out", required=True)

    ap.add_argument("--target_sr", type=int, default=16000)
    ap.add_argument("--max_sec", type=float, default=30.0)
    ap.add_argument("--min_sec", type=float, default=0.05)
    ap.add_argument("--decode_check_sec", type=float, default=2.0)
    ap.add_argument("--drop_silent_with_text", action="store_true")

    args = ap.parse_args()

    kept = 0
    dropped = 0
    issues_all: List[RowIssue] = []

    os.makedirs(os.path.dirname(os.path.abspath(args.manifest_out)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.report_out)), exist_ok=True)

    with open(args.manifest_out, "w", encoding="utf-8") as fout:
        for line_no, obj in iter_jsonl(args.manifest_in):
            keep, issues = analyse_row(
                line_no=line_no,
                obj=obj,
                target_sr=args.target_sr,
                max_sec=args.max_sec,
                min_sec=args.min_sec,
                decode_check_sec=args.decode_check_sec,
                drop_silent_with_text=args.drop_silent_with_text,
            )
            issues_all.extend(issues)
            if keep:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1

    # Write report
    with open(args.report_out, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["line_no", "audio_path", "reason", "detail"])
        for it in issues_all:
            w.writerow([it.line_no, it.audio_path, it.reason, it.detail])

    # Summary
    print(f"Input manifest: {args.manifest_in}")
    print(f"Output manifest: {args.manifest_out}")
    print(f"Report: {args.report_out}")
    print(f"Kept: {kept}  Dropped: {dropped}")
    print(f"Issues logged: {len(issues_all)}")


if __name__ == "__main__":
    main()



"""

https://chatgpt.com/g/g-p-6969433d33d4819187ec3158a8f3745f-whisper-training/c/6969613e-5b3c-8323-a964-bd0c0d9e9736


"""