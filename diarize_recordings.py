import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from pyannote.audio import Audio, Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run speaker diarization over a folder of audio files and write a JSON summary."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Folder containing .wav files to diarize.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write diarization summary JSON.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN"),
        help="Hugging Face token with access to pyannote models. "
        "If omitted, falls back to HUGGINGFACE_TOKEN / HF_TOKEN env vars.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device string understood by torch (e.g., 'cuda', 'cuda:0', 'cpu'). "
        "Defaults to torch's choice.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files to process (for quick tests).",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=1024,
        help="Skip files smaller than this many bytes (helps avoid empty placeholders).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from existing output JSON if present (skip already processed files).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Disable resume; reprocess all files.",
    )
    return parser.parse_args()


def build_pipeline(hf_token: str, device: Optional[str]) -> Pipeline:
    if not hf_token:
        raise ValueError(
            "Hugging Face token is required. Set --hf-token or HUGGINGFACE_TOKEN / HF_TOKEN."
        )
    torch_device = torch.device(device) if device else None
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )
    if torch_device:
        pipeline.to(torch_device)
    return pipeline


def list_audio_files(input_dir: Path, min_bytes: int, limit: int | None) -> list[Path]:
    candidates = [p for p in input_dir.glob("*.wav") if p.is_file() and p.stat().st_size >= min_bytes]
    candidates.sort()
    if limit:
        candidates = candidates[:limit]
    return candidates


def diarize_file(path: Path, pipeline: Pipeline, audio: Audio) -> dict:
    diarization = pipeline(str(path))
    segments = []
    per_speaker: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"total_speech": 0.0, "segments": 0}
    )

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        duration = max(0.0, end - start)
        segments.append(
            {
                "speaker": speaker,
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
            }
        )
        per_speaker[speaker]["total_speech"] += duration
        per_speaker[speaker]["segments"] += 1

    segments.sort(key=lambda s: s["start"])

    # Use Audio helper to get full file duration (covers trailing silence).
    file_duration = float(audio.get_duration(file=str(path)))

    # Aggregate totals by speaker
    speaker_summaries = {
        speaker: {
            "total_speech": round(values["total_speech"], 3),
            "segments": int(values["segments"]),
        }
        for speaker, values in per_speaker.items()
    }

    return {
        "file_name": path.name,
        "path": str(path),
        "duration": round(file_duration, 3),
        "num_speakers": len(per_speaker),
        "speakers": speaker_summaries,
        "segments": segments,
    }


def main() -> None:
    args = parse_args()
    audio_dir: Path = args.input_dir

    if not audio_dir.exists() or not audio_dir.is_dir():
        raise SystemExit(f"Input dir does not exist: {audio_dir}")

    files = list_audio_files(audio_dir, min_bytes=args.min_bytes, limit=args.limit)
    if not files:
        raise SystemExit(f"No .wav files found in {audio_dir} meeting size >= {args.min_bytes} bytes.")

    # Resume support: load existing summaries and skip processed files
    processed: dict[str, dict] = {}
    if args.resume and args.output_json.exists():
        try:
            with args.output_json.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            for entry in existing.get("files", []):
                processed[entry.get("file_name")] = entry
            dataset_aggregate = defaultdict(float, existing.get("overall", {}).get("total_speech_seconds_by_file_local_speaker", {}))
            print(f"Resume enabled: found {len(processed)} completed files in {args.output_json}", flush=True)
        except Exception as exc:
            print(f"Resume ignored (failed to read existing JSON): {exc}", flush=True)
            dataset_aggregate = defaultdict(float)
    else:
        dataset_aggregate = defaultdict(float)

    pipeline = build_pipeline(args.hf_token, device=args.device)
    audio = Audio(sample_rate=16000, mono=True)
    if args.device:
        device_str = args.device
    elif hasattr(pipeline, "device"):
        device_str = str(pipeline.device)
    elif torch.cuda.is_available():
        device_str = torch.cuda.get_device_name(0)
    else:
        device_str = "cpu"
    print(
        f"Using device: {device_str} | total files: {len(files)} | already done: {len(processed)} | to process: {len(files) - len(processed)}",
        flush=True,
    )

    file_summaries = list(processed.values()) if processed else []

    start_time = time.time()
    remaining_files = [p for p in files if p.name not in processed]
    print(f"Total files: {len(files)} | Already done: {len(processed)} | To process: {len(remaining_files)}", flush=True)

    for idx, wav_path in enumerate(remaining_files, start=1):
        tick = time.time()
        global_idx = len(processed) + idx
        print(
            f"[{global_idx}/{len(files)} | {idx}/{len(remaining_files)}] Diarizing {wav_path.name} ...",
            flush=True,
        )
        summary = diarize_file(wav_path, pipeline, audio)
        file_summaries.append(summary)
        for speaker, details in summary["speakers"].items():
            dataset_aggregate[speaker] += details["total_speech"]
        took = time.time() - tick
        avg = (time.time() - start_time) / (len(processed) + idx)
        eta = avg * (len(files) - (len(processed) + idx)) / 60
        print(
            f"    done in {took/60:.2f} min | avg {avg/60:.2f} min/file | ETA {eta:.1f} min",
            flush=True,
        )

        overall = {
            "total_files": len(files),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_speech_seconds_by_file_local_speaker": {
                speaker: round(seconds, 3) for speaker, seconds in dataset_aggregate.items()
            },
            "notes": (
                "Speaker labels are local to each file; they do not guarantee the same real person across files. "
                "To obtain global speakers across the dataset, add a cross-file clustering/enrollment step."
            ),
        }

        output = {
            "overall": overall,
            "files": file_summaries,
        }

        # Incremental write: write after each file via temp then replace for safety.
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = args.output_json.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        tmp_path.replace(args.output_json)

    elapsed = time.time() - start_time
    print(f"Diarization finished in {elapsed/60:.1f} minutes. Wrote {args.output_json}")


if __name__ == "__main__":
    main()
