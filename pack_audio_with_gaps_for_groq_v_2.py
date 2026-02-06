#!/usr/bin/env python3
"""Pack many small audio files into Groq-friendly batch files with a fixed silence gap.

Why you need this
- Uploading ~8000 files to Groq will very likely hit request-per-day limits on the free tier.
- Takeout audio often includes a few corrupted files that ffprobe can read, but ffmpeg fails to decode.

What this script does
1) Recursively finds audio files under --input_dir
2) Probes duration (ffprobe)
3) Strictly decode-checks each file (ffmpeg -> null sink)
   - undecodable files are skipped and logged to bad_files_decode_fail.txt
4) Packs clips into batches under --max_mb (size estimated from target bitrate)
5) Concatenates each batch with --gap_sec seconds of silence between clips
6) Encodes to a small Groq-friendly format (default: OGG/Opus, 16 kHz mono, 24 kbps)
7) Writes mapping files:
   - batch_map.csv
   - batch_map.jsonl

Requirements
- ffmpeg and ffprobe must be on PATH
- Python 3.9+

Windows run example (use your env python):
  I:/Whisper-training-env/Scripts/python.exe pack_audio_with_gaps_for_groq_v2.py \
    --input_dir "I:/Record_harsha/Google_takeout" \
    --out_dir   "I:\Record_harsha\Google_takeout_groq_batches\" \
    --gap_sec 5 \
    --max_mb 24 \
    --container ogg --codec opus --bitrate_kbps 24

Faster but less strict decode check (may miss corruption later in the file):
  --decode_check_seconds 20

Skip decode check entirely (not recommended with Takeout):
  --no_decode_check
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm


AUDIO_EXTS = {
    ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".mpeg", ".mpga", ".aac"
}


@dataclass(frozen=True)
class ClipInfo:
    path: Path
    duration_sec: float


def which_or_die(exe: str) -> str:
    p = shutil.which(exe)
    if not p:
        raise SystemExit(
            f"ERROR: '{exe}' not found on PATH. Install ffmpeg and ensure ffmpeg/ffprobe are on PATH."
        )
    return p


def list_audio_files(input_dir: Path, recursive: bool = True) -> List[Path]:
    if recursive:
        files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    else:
        files = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]

    files.sort(key=lambda p: str(p).lower())
    return files


def file_fingerprint(p: Path) -> Tuple[int, float]:
    st = p.stat()
    return int(st.st_size), float(st.st_mtime)


def load_cache(cache_path: Path) -> Dict[str, dict]:
    if not cache_path.exists():
        return {}
    out: Dict[str, dict] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "path" in obj:
                    out[obj["path"]] = obj
            except Exception:
                continue
    return out


def append_cache_rows(cache_path: Path, rows: List[dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def ffprobe_duration_seconds(ffprobe_bin: str, audio_path: Path) -> Optional[float]:
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
        if not out:
            return None
        dur = float(out)
        if math.isnan(dur) or dur <= 0:
            return None
        return dur
    except Exception:
        return None


def ffmpeg_decode_check(ffmpeg_bin: str, audio_path: Path, seconds: float = 0.0) -> Tuple[bool, str]:
    cmd = [ffmpeg_bin, "-hide_banner", "-v", "error", "-i", str(audio_path)]
    if seconds and seconds > 0:
        cmd += ["-t", f"{seconds}"]
    cmd += ["-f", "null", "-"]

    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
        if p.returncode == 0:
            return True, ""
        err = (p.stderr or "").strip()
        return False, err[:2500]
    except subprocess.TimeoutExpired:
        return False, "Decode timeout (30s)"
    except Exception as e:
        return False, f"Exception: {str(e)[:2000]}"


def probe_durations_cached(
    ffprobe_bin: str,
    files: List[Path],
    workers: int,
    cache_path: Path,
) -> List[ClipInfo]:
    cache = load_cache(cache_path)

    to_probe: List[Path] = []
    duration_map: Dict[Path, float] = {}

    for p in files:
        key = str(p)
        fp = file_fingerprint(p)
        cached = cache.get(key)
        if cached and tuple(cached.get("fp", [])) == fp and cached.get("duration_sec"):
            duration_map[p] = float(cached["duration_sec"])
        else:
            to_probe.append(p)

    new_rows: List[dict] = []

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = {ex.submit(ffprobe_duration_seconds, ffprobe_bin, p): p for p in to_probe}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Probing durations", unit="file"):
            p = futures[fut]
            dur = fut.result()
            if dur is None:
                new_rows.append({"path": str(p), "fp": list(file_fingerprint(p)), "duration_sec": None})
                continue
            duration_map[p] = dur
            new_rows.append({"path": str(p), "fp": list(file_fingerprint(p)), "duration_sec": dur})

            if len(new_rows) >= 300:
                append_cache_rows(cache_path, new_rows)
                new_rows = []

    if new_rows:
        append_cache_rows(cache_path, new_rows)

    return [ClipInfo(path=p, duration_sec=duration_map[p]) for p in files if p in duration_map]


def estimate_mb(duration_sec: float, bitrate_kbps: int) -> float:
    return duration_sec * (bitrate_kbps * 1000.0) / 8.0 / (1024.0 * 1024.0)


def pack_into_batches(clips: List[ClipInfo], gap_sec: float, max_mb: float, bitrate_kbps: int) -> List[List[ClipInfo]]:
    if max_mb <= 0:
        return [clips]

    batches: List[List[ClipInfo]] = []
    current: List[ClipInfo] = []
    current_total = 0.0

    for clip in clips:
        prospective = clip.duration_sec if not current else current_total + gap_sec + clip.duration_sec
        prospective_mb = estimate_mb(prospective, bitrate_kbps)

        if not current and prospective_mb > max_mb:
            # huge single clip; still force a batch
            batches.append([clip])
            current = []
            current_total = 0.0
            continue

        if current and prospective_mb > max_mb:
            batches.append(current)
            current = [clip]
            current_total = clip.duration_sec
        else:
            if current:
                current_total += gap_sec + clip.duration_sec
            else:
                current_total = clip.duration_sec
            current.append(clip)

    if current:
        batches.append(current)

    return batches


def posix_path(p: Path) -> str:
    # concat demuxer is happiest with forward slashes
    s = p.resolve().as_posix().replace("'", "'\\''")
    return s


def create_silence_wav(ffmpeg_bin: str, silence_path: Path, sec: float, sr: int) -> None:
    cmd = [
        ffmpeg_bin, "-y",
        "-f", "lavfi",
        "-i", f"anullsrc=r={sr}:cl=mono",
        "-t", f"{sec}",
        str(silence_path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def export_batch(
    ffmpeg_bin: str,
    batch_idx: int,
    clips: List[ClipInfo],
    out_dir: Path,
    gap_sec: float,
    silence_path: Path,
    container: str,
    codec: str,
    bitrate_kbps: int,
    sample_rate: int,
    resume: bool,
) -> Tuple[Path, List[Tuple[Path, float, float]]]:

    out_name = f"batch_{batch_idx:05d}.{container}"
    out_path = out_dir / out_name

    mapping: List[Tuple[Path, float, float]] = []
    t = 0.0

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        list_path = Path(f.name)
        for i, clip in enumerate(clips):
            f.write(f"file '{posix_path(clip.path)}'\n")

            start = t
            end = start + clip.duration_sec
            mapping.append((clip.path, start, end))
            t = end

            if i != len(clips) - 1:
                f.write(f"file '{posix_path(silence_path)}'\n")
                t += gap_sec

    try:
        if resume and out_path.exists() and out_path.stat().st_size > 0:
            return out_path, mapping

        cmd = [
            ffmpeg_bin, "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-ac", "1",
            "-ar", str(sample_rate),
        ]

        if codec == "opus":
            cmd += [
                "-c:a", "libopus",
                "-b:a", f"{bitrate_kbps}k",
                "-vbr", "off",
                "-application", "voip",
                "-compression_level", "10",
                "-strict", "-2",
            ]
        elif codec == "mp3":
            cmd += ["-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k"]
        elif codec == "aac":
            cmd += ["-c:a", "aac", "-b:a", f"{bitrate_kbps}k"]
        else:
            raise SystemExit("Unsupported codec. Use opus|mp3|aac")

        cmd += [str(out_path)]

        try:
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return out_path, mapping
        except subprocess.CalledProcessError as e:
            print(f"Error encoding batch {batch_idx}: {e}")
            print(f"Command: {' '.join(cmd)}")
            # Try to continue with next batch instead of failing completely
            raise e
    finally:
        try:
            list_path.unlink(missing_ok=True)
        except Exception:
            pass


def beep_done() -> None:
    try:
        import winsound  # type: ignore

        winsound.Beep(1000, 350)
        winsound.Beep(1200, 350)
    except Exception:
        sys.stdout.write(chr(7))
        sys.stdout.flush()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)

    ap.add_argument("--gap_sec", type=float, default=5.0)
    ap.add_argument("--max_mb", type=float, default=24.0)

    ap.add_argument("--container", choices=["ogg", "webm", "mp3", "m4a"], default="ogg")
    ap.add_argument("--codec", choices=["opus", "mp3", "aac"], default="opus")
    ap.add_argument("--bitrate_kbps", type=int, default=24)
    ap.add_argument("--sample_rate", type=int, default=16000)

    ap.add_argument("--recursive", action="store_true", default=True)
    ap.add_argument("--no_recursive", action="store_false", dest="recursive")

    ap.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 8) // 2))

    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no_resume", action="store_false", dest="resume")

    ap.add_argument("--decode_check", action="store_true", default=True)
    ap.add_argument("--no_decode_check", action="store_false", dest="decode_check")
    ap.add_argument("--decode_check_seconds", type=float, default=0.0)

    args = ap.parse_args()

    ffmpeg_bin = which_or_die("ffmpeg")
    ffprobe_bin = which_or_die("ffprobe")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Basic sanity for codec/container combos
    if args.codec == "opus" and args.container not in {"ogg", "webm"}:
        raise SystemExit("Opus should use --container ogg or webm")
    if args.codec == "aac" and args.container != "m4a":
        raise SystemExit("AAC should use --container m4a")
    if args.codec == "mp3" and args.container != "mp3":
        raise SystemExit("MP3 should use --container mp3")

    files = list_audio_files(args.input_dir, recursive=args.recursive)
    if not files:
        raise SystemExit(f"No audio files found in: {args.input_dir}")

    cache_path = args.out_dir / "_cache_probe.jsonl"
    clips = probe_durations_cached(ffprobe_bin, files, workers=args.workers, cache_path=cache_path)
    if not clips:
        raise SystemExit("No readable audio files after probing durations.")

    # Decode check to catch corrupt Takeout files
    if args.decode_check:
        bad_log = args.out_dir / "bad_files_decode_fail.txt"
        good: List[ClipInfo] = []
        bad_rows: List[str] = []

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(ffmpeg_decode_check, ffmpeg_bin, ci.path, args.decode_check_seconds): ci for ci in clips}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="Decode-check", unit="file"):
                ci = futs[fut]
                ok, err = fut.result()
                if ok:
                    good.append(ci)
                else:
                    bad_rows.append(f"{ci.path}\n{err}\n---\n")

        good_set = {c.path for c in good}
        clips = [ci for ci in clips if ci.path in good_set]

        if bad_rows:
            bad_log.write_text("".join(bad_rows), encoding="utf-8")
            print(f"\nSkipped {len(bad_rows)} undecodable file(s). See: {bad_log}")

        if not clips:
            raise SystemExit("All files failed decode-check. Input set looks broken.")

    batches = pack_into_batches(clips, gap_sec=args.gap_sec, max_mb=args.max_mb, bitrate_kbps=args.bitrate_kbps)

    # One reusable silence wav
    tmp_dir = args.out_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    silence_path = tmp_dir / f"silence_{int(args.gap_sec * 1000)}ms.wav"
    if not silence_path.exists():
        create_silence_wav(ffmpeg_bin, silence_path, args.gap_sec, sr=args.sample_rate)

    all_rows: List[dict] = []

    for idx, batch in enumerate(tqdm(batches, desc="Encoding batches", unit="batch"), start=1):
        try:
            out_path, mapping = export_batch(
                ffmpeg_bin=ffmpeg_bin,
                batch_idx=idx,
                clips=batch,
                out_dir=args.out_dir,
                gap_sec=args.gap_sec,
                silence_path=silence_path,
                container=args.container,
                codec=args.codec,
                bitrate_kbps=args.bitrate_kbps,
                sample_rate=args.sample_rate,
                resume=args.resume,
            )

            for (orig, start, end) in mapping:
                all_rows.append(
                    {
                        "original_file": str(orig),
                        "batch_file": str(out_path),
                        "start_sec": round(start, 3),
                        "end_sec": round(end, 3),
                        "gap_sec": args.gap_sec,
                    }
                )
        except Exception as e:
            print(f"Failed to encode batch {idx}, skipping: {e}")
            # Log the failed batch files for debugging
            failed_log = args.out_dir / "failed_batches.txt"
            with failed_log.open("a", encoding="utf-8") as f:
                f.write(f"Batch {idx} failed: {e}\n")
                for clip in batch:
                    f.write(f"  {clip.path}\n")
                f.write("---\n")
            continue

    map_csv = args.out_dir / "batch_map.csv"
    with map_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["original_file", "batch_file", "start_sec", "end_sec", "gap_sec"])
        w.writeheader()
        w.writerows(all_rows)

    map_jsonl = args.out_dir / "batch_map.jsonl"
    with map_jsonl.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone. Created {len(batches)} batch file(s) in: {args.out_dir}")
    print(f"Mapping: {map_csv}")
    print(f"Mapping: {map_jsonl}")

    beep_done()


if __name__ == "__main__":
    main()
