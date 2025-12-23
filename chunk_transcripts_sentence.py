import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Sequence

from tqdm import tqdm

PAD_SECONDS = 0.15
MAX_SENTENCE_SECONDS = 25.0


@dataclass
class SentenceChunk:
    start: float
    end: float
    text: str


def split_text_to_sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter to avoid external model downloads.
    Splits on punctuation followed by whitespace.
    """
    cleaned = text.strip()
    if not cleaned:
        return []
    # Preserve punctuation at end of sentence
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if p.strip()]


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


def segment_to_sentence_chunks(segment: Mapping) -> List[SentenceChunk]:
    """
    Split a single Whisper segment into sentence-aligned chunks.
    Timing is distributed proportionally by character span.
    """
    text = (segment.get("text") or "").strip()
    if not text:
        return []
    start = float(segment["start"])
    end = float(segment["end"])
    duration = max(0.0, end - start)
    sentences = split_text_to_sentences(text)
    if not sentences:
        return []

    # If only one sentence, keep original timing
    if len(sentences) == 1 or duration == 0:
        return [SentenceChunk(start=start, end=end, text=sentences[0])]

    total_chars = sum(len(s) for s in sentences)
    chunks: List[SentenceChunk] = []
    cursor = start
    for idx, sent in enumerate(sentences):
        proportion = len(sent) / total_chars if total_chars else 1 / len(sentences)
        seg_duration = duration * proportion
        seg_end = start + duration if idx == len(sentences) - 1 else cursor + seg_duration
        chunks.append(SentenceChunk(start=cursor, end=seg_end, text=sent))
        cursor = seg_end
    return chunks


def build_sentence_chunks(segments: Iterable[Mapping]) -> List[SentenceChunk]:
    chunks: List[SentenceChunk] = []
    for seg in segments:
        chunks.extend(segment_to_sentence_chunks(seg))
    # Filter out extremely long sentences to keep under model-friendly limit
    clipped: List[SentenceChunk] = []
    for ch in chunks:
        if ch.end - ch.start > MAX_SENTENCE_SECONDS:
            mid = ch.start + MAX_SENTENCE_SECONDS
            clipped.append(SentenceChunk(ch.start, mid, ch.text))
        else:
            clipped.append(ch)
    return clipped


def run_ffmpeg_cut(in_path: str, out_path: str, start_s: float, end_s: float, sr: int = 16000) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        in_path,
        "-ac",
        "1",
        "-ar",
        str(sr),
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def process_single_json(json_path: Path, chunks_dir: Path, manifest_dir: Path | None, overwrite: bool) -> int:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    segments = extract_segments(payload)
    chunks = build_sentence_chunks(segments)

    audio_path = payload["input_file"]["path"]
    base = Path(audio_path).stem

    manifest_records: List[str] = []
    created = 0

    for idx, ch in enumerate(chunks):
        start = max(0.0, ch.start - PAD_SECONDS)
        end = ch.end + PAD_SECONDS
        out_wav = chunks_dir / f"{base}_sent{idx:04d}.wav"
        if out_wav.exists() and not overwrite:
            continue
        run_ffmpeg_cut(audio_path, str(out_wav), start, end)
        created += 1
        record = {
            "audio_path": str(out_wav),
            "raw_transcription": ch.text,
            "source_audio": audio_path,
            "chunk_index": idx,
            "chunk_start": start,
            "chunk_end": end,
            "transcript_json": str(json_path),
        }
        manifest_records.append(json.dumps(record, ensure_ascii=False))

    if manifest_dir and manifest_records:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{json_path.stem}.jsonl"
        with manifest_path.open("w", encoding="utf-8") as mf:
            mf.write("\n".join(manifest_records) + "\n")

    return created


def convert_all(
    input_dir: Path,
    output_dir: Path,
    manifest_dir: Path | None,
    overwrite: bool,
    workers: int,
) -> None:
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_created = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_single_json, jf, output_dir, manifest_dir, overwrite): jf
            for jf in json_files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Chunking transcripts", unit="file"):
            jf = futures[fut]
            try:
                created = fut.result()
                total_created += created
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"[ERROR] {jf.name}: {exc}")

    tqdm.write(f"Done. Created {total_created} chunks across {len(json_files)} files.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk transcript JSON files into sentence-aligned audio clips."
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
        default=Path(r"i:\P2GPT_google_drive\My Drive\Record_chunks_sentence"),
        help="Where to write chunked WAV files.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path(r"i:\P2GPT_google_drive\My Drive\Record_chunks_sentence\manifests"),
        help="Directory to write per-JSON manifest .jsonl files (one line per chunk).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate WAVs even if they already exist (otherwise skipped for resumability).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(4, (os.cpu_count() or 4)),
        help="Number of worker processes for chunking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        manifest_dir=args.manifest_dir,
        overwrite=args.overwrite,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
