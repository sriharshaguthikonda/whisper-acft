import argparse
import json
import os
import re
import subprocess
import sys
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


def is_wav_mono_16k(audio_path: Path) -> bool:
    if audio_path.suffix.lower() != ".wav":
        return False
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=channels,sample_rate",
                "-of",
                "csv=p=0",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        parts = result.stdout.strip().split(",")
        if len(parts) != 2:
            return False
        channels = int(parts[0])
        sample_rate = int(parts[1])
        return channels == 1 and sample_rate == 16000
    except Exception:
        return False


def run_ffmpeg_cut(
    in_path: str, out_path: str, start_s: float, end_s: float, sr: int = 16000, copy_if_wav: bool = False
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if copy_if_wav:
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start_s:.3f}",
            "-to",
            f"{end_s:.3f}",
            "-i",
            in_path,
            "-c",
            "copy",
            out_path,
        ]
    else:
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


def resolve_audio_path(audio_path: str, audio_root: Path) -> Path:
    original = Path(audio_path)
    if original.exists():
        return original

    # Handle Colab-style prefix
    colab_prefix = Path("/content/drive/My Drive")
    if str(original).startswith(str(colab_prefix)):
        candidate = audio_root / original.name
        if candidate.exists():
            return candidate
        # Try .wav instead of original suffix
        candidate_wav = candidate.with_suffix(".wav")
        if candidate_wav.exists():
            return candidate_wav

    # Fallback: try audio_root + filename with wav
    fallback = audio_root / original.name
    if fallback.exists():
        return fallback
    fallback_wav = fallback.with_suffix(".wav")
    if fallback_wav.exists():
        return fallback_wav

    # Last resort return original (will fail downstream and get logged)
    return original


def process_single_json(
    json_path: Path, chunks_dir: Path, audio_root: Path, overwrite: bool
) -> tuple[int, int, List[str]]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    segments = extract_segments(payload)
    chunks = build_sentence_chunks(segments)

    audio_path = payload["input_file"]["path"]
    resolved_audio = resolve_audio_path(audio_path, audio_root)
    base = Path(resolved_audio).stem

    manifest_records: List[str] = []
    created = 0
    skipped = 0
    copy_ok = is_wav_mono_16k(resolved_audio)

    # Pre-compute outputs to allow a fast skip when everything already exists
    outputs = [chunks_dir / f"{base}_sent{idx:04d}.wav" for idx in range(len(chunks))]
    if not overwrite and all(p.exists() for p in outputs):
        # All chunks already present; return quickly without rebuilding manifest records
        return 0, len(outputs), []

    for idx, (ch, out_wav) in enumerate(zip(chunks, outputs)):
        start = max(0.0, ch.start - PAD_SECONDS)
        end = ch.end + PAD_SECONDS
        if out_wav.exists() and not overwrite:
            skipped += 1
        else:
            run_ffmpeg_cut(str(resolved_audio), str(out_wav), start, end, copy_if_wav=copy_ok)
            created += 1
        record = {
            "audio_path": str(out_wav),
            "raw_transcription": ch.text,
            "source_audio": str(resolved_audio),
            "chunk_index": idx,
            "chunk_start": start,
            "chunk_end": end,
            "transcript_json": str(json_path),
        }
        manifest_records.append(json.dumps(record, ensure_ascii=False))

    return created, skipped, manifest_records


def convert_all(
    input_dir: Path,
    output_dir: Path,
    manifest_path: Path,
    audio_root: Path,
    overwrite: bool,
    workers: int,
) -> None:
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_created = 0
    total_skipped = 0
    existing_audio: set[str] = set()
    if manifest_path.exists() and not overwrite:
        with manifest_path.open("r", encoding="utf-8") as mf:
            for line in mf:
                try:
                    rec = json.loads(line)
                    ap = rec.get("audio_path")
                    if ap:
                        existing_audio.add(ap)
                except Exception:
                    continue
        tqdm.write(f"Loaded {len(existing_audio)} existing manifest entries from {manifest_path}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "w" if overwrite else "a"
    manifest_file = manifest_path.open(write_mode, encoding="utf-8")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        submit_pbar = tqdm(total=len(json_files), desc="Queueing files", unit="file")
        futures = {}
        for jf in json_files:
            fut = executor.submit(process_single_json, jf, output_dir, audio_root, overwrite)
            futures[fut] = jf
            submit_pbar.update(1)
        submit_pbar.close()

        process_pbar = tqdm(total=len(futures), desc="Processing files", unit="file")
        for fut in as_completed(futures):
            jf = futures[fut]
            try:
                created, skipped, manifest_records = fut.result()
                total_created += created
                total_skipped += skipped
                new_lines = 0
                for line in manifest_records:
                    try:
                        rec = json.loads(line)
                        ap = rec.get("audio_path")
                    except Exception:
                        ap = None
                    if ap and ap in existing_audio:
                        continue
                    if ap:
                        existing_audio.add(ap)
                    manifest_file.write(line + "\n")
                    new_lines += 1
                tqdm.write(f"[{jf.name}] created={created} skipped={skipped}")
            except Exception as exc:  # noqa: BLE001
                tqdm.write(f"[ERROR] {jf.name}: {exc}")
            finally:
                process_pbar.update(1)
        process_pbar.close()

    manifest_file.close()
    tqdm.write(
        f"Done. Created {total_created} chunks, skipped {total_skipped} existing files across {len(json_files)} JSONs."
    )


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
        default=Path(r"i:\P2GPT_google_drive\My Drive\Record_chunks"),
        help="Where to write chunked WAV files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path(r"i:\P2GPT_google_drive\My Drive\Record_chunks\pairs_manifest_sentence.jsonl"),
        help="Path to aggregated manifest JSONL (appends unless --overwrite).",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path(r"I:\Record_wav"),
        help="Directory containing full-length source audio files (used to remap Colab paths).",
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
        manifest_path=args.manifest_path,
        audio_root=args.audio_root,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    # Beep to signal completion (cross-platform best effort)
    try:
        if sys.platform.startswith("win"):
            import winsound

            winsound.Beep(1000, 600)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


if __name__ == "__main__":
    main()
