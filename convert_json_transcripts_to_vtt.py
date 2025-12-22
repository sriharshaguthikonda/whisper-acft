import argparse
import json
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence

from tqdm import tqdm


def format_timestamp(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp hh:mm:ss.mmm."""
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def extract_segments(payload: Mapping) -> Sequence[Mapping]:
    """
    Find Whisper/Groq verbose_json segments.

    Expected shape:
    { "groq_response": { "segments": [ { "start": ..., "end": ..., "text": ... }, ... ] } }
    Falls back to top-level "segments" if groq_response is absent.
    """
    response: MutableMapping = payload.get("groq_response") or {}
    segments = response.get("segments") or payload.get("segments")
    if not segments:
        raise ValueError("No segments found in JSON (expected groq_response.segments).")
    return segments  # type: ignore[return-value]


def segments_to_vtt_lines(segments: Iterable[Mapping]) -> List[str]:
    lines = ["WEBVTT", ""]
    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        text = (segment.get("text") or "").strip()
        if start is None or end is None or not text:
            # Skip malformed segments while keeping resumability.
            continue
        start_ts = format_timestamp(float(start))
        end_ts = format_timestamp(float(end))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")  # blank line between cues
    if len(lines) == 2:  # only header plus blank
        raise ValueError("No valid segments with start/end/text to write.")
    return lines


def process_file(json_path: Path, output_dir: Path, overwrite: bool) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vtt_path = output_dir / (json_path.stem + ".vtt")
    if vtt_path.exists() and not overwrite:
        return None  # resumable: skip already-created file

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    segments = extract_segments(payload)
    vtt_lines = segments_to_vtt_lines(segments)
    vtt_path.write_text("\n".join(vtt_lines) + "\n", encoding="utf-8")
    return vtt_path


def convert_all(input_dir: Path, output_dir: Path, overwrite: bool) -> None:
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    successes, skipped, failures = 0, 0, 0
    for json_file in tqdm(json_files, desc="Converting JSON to VTT", unit="file"):
        try:
            result = process_file(json_file, output_dir, overwrite)
            if result is None:
                skipped += 1
            else:
                successes += 1
        except Exception as exc:  # noqa: BLE001
            failures += 1
            tqdm.write(f"[ERROR] {json_file.name}: {exc}")

    tqdm.write(
        f"Done. Created: {successes}, skipped: {skipped} (already present), failures: {failures}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Groq/Whisper verbose_json transcripts to WebVTT."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"i:\P2GPT_google_drive\My Drive\Transcriptions"),
        help="Directory containing transcript JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write VTT files (defaults to input directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate VTT even if it already exists (otherwise skipped for resumability).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    convert_all(args.input_dir, output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
