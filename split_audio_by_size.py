import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

from tqdm import tqdm


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"{name} is required but not found in PATH.")


def probe_media(path: Path) -> Tuple[float, float]:
    """Return (duration_sec, bitrate_bps) using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,bit_rate",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    fmt = data.get("format") or {}
    duration = float(fmt.get("duration") or 0.0)
    bitrate = float(fmt.get("bit_rate") or 0.0)
    if duration <= 0:
        raise ValueError(f"ffprobe could not read duration for {path}")
    # Fall back to size-based bitrate if ffprobe omitted it
    if bitrate <= 0:
        size_bytes = path.stat().st_size
        bitrate = (size_bytes * 8) / max(duration, 1e-6)
    return duration, bitrate


def split_audio(
    input_path: Path,
    output_dir: Path,
    limit_mb: float = 25.0,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """
    Split input audio into chunks each under limit_mb (approx) by duration slicing.

    Returns (created, skipped).
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    limit_bytes = limit_mb * 1024 * 1024
    size_bytes = input_path.stat().st_size
    if size_bytes <= limit_bytes:
        return 0, 0  # Nothing to do

    duration, bitrate = probe_media(input_path)

    # Use bitrate to propose a safe chunk duration that lands under the size limit.
    # Apply a small safety margin to counter mux overhead.
    target_bytes = limit_bytes * 0.98
    proposed_chunk_sec = max(1.0, (target_bytes * 8) / bitrate)

    chunk_count = max(1, math.ceil(duration / proposed_chunk_sec))
    chunk_sec = duration / chunk_count

    output_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    for idx in tqdm(range(chunk_count), desc="Splitting", unit="chunk"):
        start = idx * chunk_sec
        end = duration if idx == chunk_count - 1 else min(duration, (idx + 1) * chunk_sec)
        if end - start <= 0:
            continue
        out_name = f"{input_path.stem}_part{idx+1:04d}{input_path.suffix}"
        out_path = output_dir / out_name

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(input_path),
            "-c",
            "copy",
            str(out_path),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        created += 1

    return created, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split an audio file into < limit MB chunks using ffmpeg and ffprobe."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the source audio file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to place chunks (defaults to input file's directory).",
    )
    parser.add_argument(
        "--limit-mb",
        type=float,
        default=25.0,
        help="Maximum size per chunk in megabytes (default: 25).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate chunks even if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    require_tool("ffprobe")
    require_tool("ffmpeg")

    args = parse_args()
    input_path: Path = args.input
    output_dir: Path = args.output_dir or input_path.parent

    try:
        created, skipped = split_audio(
            input_path=input_path,
            output_dir=output_dir,
            limit_mb=args.limit_mb,
            overwrite=args.overwrite,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Error: {exc}") from exc

    if created == 0 and skipped == 0:
        print("File is already under the size limit; no chunks created.")
    else:
        print(f"Done. Chunks created: {created}, skipped existing: {skipped}.")


if __name__ == "__main__":
    main()
