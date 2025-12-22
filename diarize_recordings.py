import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

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
    return parser.parse_args()


def build_pipeline(hf_token: str) -> Pipeline:
    if not hf_token:
        raise ValueError(
            "Hugging Face token is required. Set --hf-token or HUGGINGFACE_TOKEN / HF_TOKEN."
        )
    return Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
    )


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

    pipeline = build_pipeline(args.hf_token)
    audio = Audio(sample_rate=16000, mono=True)

    dataset_aggregate: dict[str, float] = defaultdict(float)
    file_summaries = []

    start_time = time.time()
    for idx, wav_path in enumerate(files, start=1):
        print(f"[{idx}/{len(files)}] Diarizing {wav_path.name} ...", flush=True)
        summary = diarize_file(wav_path, pipeline, audio)
        file_summaries.append(summary)
        for speaker, details in summary["speakers"].items():
            dataset_aggregate[speaker] += details["total_speech"]

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

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Diarization finished in {elapsed/60:.1f} minutes. Wrote {args.output_json}")


if __name__ == "__main__":
    main()
