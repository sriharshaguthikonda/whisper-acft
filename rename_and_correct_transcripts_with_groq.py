from __future__ import annotations

import argparse
import json
import re
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import cycle, product

import requests


SYSTEM_PROMPT = (
    "You are renaming audio files that contain FAKE medical consultations. "
    "Generate a concise, human-readable filename stem for the conversation AND provide a corrected transcript. "
    "Rules for filename: return only the filename stem (no extension); use underscores instead of spaces; "
    "keep filename descriptive but specific to the main topic.\n\n"
    "Transcript correction rule: Correct only and only the punctuation! do not change anything else!"
    "Respond ONLY as compact JSON with two keys: "
    '{"filename_stem": "<stem_with_underscores>", "corrected_transcript": "<corrected text>"}'
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "filename_stem": {"type": "string"},
        "corrected_transcript": {"type": "string"},
    },
    "required": ["filename_stem", "corrected_transcript"],
    "additionalProperties": False,
}


"""
use :

python rename_and_correct_transcripts_with_groq.py --transcripts-dir "i:\P2GPT_google_drive\My Drive\Transcriptions" --audio-dir "i:\Record" --report rename_and_corrected_transcript_report.json --retry-backoff-base 2
"""

def load_env(env_path: Path) -> Dict[str, str]:
    env_vars: Dict[str, str] = {}
    if not env_path.exists():
        return env_vars
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env_vars[key.strip()] = value.strip()
    return env_vars


def parse_args() -> argparse.Namespace:
    env = load_env(Path(".env"))
    default_keys = env.get("GROQ_API_KEYS", "")
    default_models = env.get("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
    default_transcripts = env.get("TRANSCRIPTS_DIR", "")
    default_audio = env.get("AUDIO_DIR", "")

    parser = argparse.ArgumentParser(
        description="Rename audio files and correct transcripts using Groq chat model suggestions based on transcript JSON files."
    )
    parser.add_argument(
        "--transcripts-dir",
        type=Path,
        default=Path(default_transcripts) if default_transcripts else None,
        help="Directory containing transcript JSON files.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path(default_audio) if default_audio else None,
        help="Directory containing audio files to rename.",
    )
    parser.add_argument(
        "--api-keys",
        type=str,
        default=default_keys,
        help="Comma-separated Groq API keys (rotated on errors).",
    )
    parser.add_argument(
        "--model",
        "--models",
        dest="models",
        type=str,
        default=default_models,
        help="Groq chat models (comma-separated) to rotate on errors.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply renames. Without this flag, only a dry-run report is produced.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("rename_and_corrected_transcript_report.json"),
        help="Path to write the dry-run/apply report (defaults to rename_and_corrected_transcript_report.json).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per transcript across rotated API keys.",
    )
    parser.add_argument(
        "--flush-interval",
        type=int,
        default=5,
        help="Flush report to disk after this many items to reduce data loss on interruption.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume using existing report file to skip already processed transcripts (default: on).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Disable resume; process all transcripts regardless of existing report.",
    )
    parser.add_argument(
        "--retry-backoff-base",
        type=float,
        default=1.5,
        help="Base seconds for exponential backoff on API errors (attempt^backoff_base).",
    )
    parser.add_argument(
        "--per-request-delay",
        type=float,
        default=0.0,
        help="Optional fixed sleep seconds between API calls to reduce rate limits.",
    )
    parser.add_argument(
        "--strict-output",
        action="store_true",
        help="Force strict structured output (only works on supported models like openai/gpt-oss-20b, openai/gpt-oss-120b).",
    )
    return parser.parse_args()


def sanitize_filename(stem: str, max_len: int = 80) -> str:
    # Replace spaces with underscores
    stem = stem.replace(" ", "_")
    # Remove punctuation except underscores and hyphens
    stem = re.sub(r"[^\w\-]", "_", stem)
    # Collapse multiple underscores
    stem = re.sub(r"_+", "_", stem)
    # Trim underscores/hyphens at ends
    stem = stem.strip("._-")
    # Enforce length
    if len(stem) > max_len:
        stem = stem[:max_len].rstrip("._-")
    # Fallback if empty
    if not stem:
        stem = "untitled"
    return stem


def clean_text(text: str) -> str:
    """Normalize punctuation, remove hard newlines, and collapse whitespace."""
    replacements = {
        "\u2011": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_speaker_labels(text: str) -> str:
    """Remove simple speaker tags like 'Doctor:' or 'Patient:'."""
    pattern = re.compile(
        r"\b(?:doctor|dr|patient|nurse|speaker|interviewer|interviewee|caller|agent|user|customer|client|mr|mrs|ms|miss|sir|madam)\s*:\s*",
        re.IGNORECASE,
    )
    return pattern.sub("", text).strip()


def rotate(values: List[str]):
    while True:
        for v in values:
            yield v


def rotate_pairs(keys: List[str], models: List[str]):
    pairs: List[Tuple[str, str]] = [(k, m) for k in keys for m in models]
    while True:
        for pair in pairs:
            yield pair


def extract_response(content: str) -> Tuple[str, str]:
    """Parse model JSON output safely."""
    content = content.strip()
    # strip code fences if present
    if content.startswith("```"):
        content = content.strip("`")
        if "\n" in content:
            content = content.split("\n", 1)[1]
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # try to salvage JSON if model adds prose
        if "{" in content and "}" in content:
            snippet = content[content.find("{") : content.rfind("}") + 1]
            data = json.loads(snippet)
        else:
            raise ValueError("Model did not return JSON.")
    filename_stem = data.get("filename_stem") or data.get("new_filename_stem")
    corrected_transcript = data.get("corrected_transcript") or data.get("transcript") or ""
    if not filename_stem or not corrected_transcript:
        raise ValueError("Missing filename_stem or corrected_transcript in model response.")
    return str(filename_stem).strip(), str(corrected_transcript).strip()


def call_groq_chat(
    api_key: str,
    model: str,
    transcript_text: str,
    original_name: str,
    per_request_delay: float = 0.0,
    strict_output: bool = False,
) -> Tuple[str, str]:
    if per_request_delay > 0:
        time.sleep(per_request_delay)
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original filename: {original_name}\n"
                f"Transcript:\n{transcript_text}\n\n"
                "Respond with only the new filename stem (no extension)."
            ),
        },
    ]
    resp = requests.post(
        url,
        headers=headers,
        json={
            "model": model,
            "messages": messages,
            "temperature": 0.2,
            # Allow ample room for corrected transcript and filename JSON
            "max_tokens": 4096,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "rename_and_correct_transcript",
                    # strict=True only on supported models; otherwise best-effort
                    "strict": strict_output,
                    "schema": RESPONSE_SCHEMA,
                },
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices or not choices[0].get("message", {}).get("content"):
        raise ValueError("Empty response from model")
    choice = choices[0]["message"]["content"].strip()
    return extract_response(choice)


def propose_name_and_correction(
    keys: List[str],
    models: List[str],
    transcript_text: str,
    original_name: str,
    max_retries: int,
    backoff_base: float,
    per_request_delay: float,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    pairs: List[Tuple[str, str]] = [(k, m) for k, m in product(keys, models)]
    if not pairs:
        return None, None, "No key/model pairs to use."
    pair_rotator = cycle(pairs)
    total_attempts = max_retries * len(pairs)
    last_error: Optional[str] = None
    for attempt in range(total_attempts):
        api_key, model = next(pair_rotator)
        try:
            stem, corrected_transcript = call_groq_chat(
                api_key=api_key,
                model=model,
                transcript_text=transcript_text,
                original_name=original_name,
                per_request_delay=per_request_delay,
                strict_output=False,
            )
            corrected_transcript = strip_speaker_labels(clean_text(corrected_transcript))
            return stem, corrected_transcript, None
        except requests.exceptions.HTTPError as exc:
            body = ""
            if exc.response is not None:
                try:
                    body = exc.response.text
                except Exception:
                    body = ""
            last_error = (
                f"HTTPError {exc.response.status_code if exc.response else ''}: "
                f"{body or exc}"
            )
            # Backoff on rate limits and rotate
            retry_after = 0.0
            if exc.response is not None:
                try:
                    retry_after = float(exc.response.headers.get("Retry-After", "0"))
                except Exception:
                    retry_after = 0.0
            base = max(1.0, backoff_base ** (attempt + 1))
            jitter = random.uniform(0.0, 0.5 * base)
            sleep_s = base + retry_after + jitter
            time.sleep(sleep_s)
            continue
        except Exception as exc:  # broad to log any API issue
            last_error = f"{type(exc).__name__}: {exc}"
            sleep_s = max(1.0, backoff_base ** (attempt + 1))
            time.sleep(sleep_s)
            continue
    return None, None, last_error


def flush_report(report: List[Dict[str, str]], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def load_existing_report(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def normalize_path(p: Path) -> str:
    try:
        return str(p.resolve()).lower()
    except Exception:
        return str(p).lower()


def find_unique_path(target: Path) -> Path:
    if not target.exists():
        return target
    counter = 1
    while True:
        candidate = target.with_name(f"{target.stem}_{counter}{target.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1
        if counter > 9999:
            raise RuntimeError("Too many collisions for file: {}".format(target))


def load_transcript(json_path: Path) -> Tuple[str, str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    input_name = data.get("input_file", {}).get("name")
    transcript_text = data.get("groq_response", {}).get("text", "")
    if not input_name:
        raise ValueError(f"Missing input_file.name in {json_path}")
    return input_name, transcript_text


def main() -> None:
    args = parse_args()

    if not args.transcripts_dir or not args.transcripts_dir.exists():
        raise SystemExit("Transcripts directory missing or not provided.")
    if not args.audio_dir or not args.audio_dir.exists():
        raise SystemExit("Audio directory missing or not provided.")

    keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    if not keys:
        raise SystemExit("No Groq API keys provided. Set GROQ_API_KEYS in .env or pass --api-keys.")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        raise SystemExit("No Groq models provided. Set GROQ_MODEL in .env or pass --models/--model.")

    report: List[Dict[str, str]] = []
    if args.resume:
        existing = load_existing_report(args.report)
        if existing:
            report = existing
            print(f"Loaded existing report with {len(report)} entries (resume enabled).")

    transcripts = sorted(args.transcripts_dir.glob("*.json"))
    total = len(transcripts)
    print(f"Found {total} transcript files. Starting {'apply' if args.apply else 'dry run'}...")
    if total > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    processed_set = {
        normalize_path(Path(item.get("transcript"))) for item in report if item.get("transcript")
    }

    bar_width = 30
    def render_progress(current: int, total_items: int) -> None:
        if total_items == 0:
            return
        fraction = min(1.0, current / total_items)
        filled = int(bar_width * fraction)
        bar = "#" * filled + "-" * (bar_width - filled)
        sys.stdout.write(f"\r[{bar}] {current}/{total_items}")
        sys.stdout.flush()

    render_progress(0, total)

    for idx, json_file in enumerate(transcripts, start=1):
        transcript_key = normalize_path(json_file)
        if args.resume and transcript_key in processed_set:
            print(f"[{idx}/{total}] Skipping {json_file.name} (already in report).")
            render_progress(idx, total)
            continue
        print(f"[{idx}/{total}] Processing {json_file.name}...")
        try:
            input_name, transcript_text = load_transcript(json_file)
        except Exception as exc:
            report.append(
                {
                    "transcript": str(json_file.resolve()),
                    "original": "",
                    "proposed": "",
                    "corrected_transcript": "",
                    "original_transcript": "",
                    "action": "error",
                    "detail": f"failed to load transcript: {exc}",
                }
            )
            flush_report(report, args.report)
            render_progress(idx, total)
            continue

        original_audio = args.audio_dir / input_name
        if not original_audio.exists():
            print(f"  ⚠️  Audio not found for {input_name}")
            report.append(
                {
                    "transcript": str(json_file.resolve()),
                    "original": input_name,
                    "proposed": "",
                    "original_transcript": transcript_text,
                    "corrected_transcript": "",
                    "action": "missing_audio",
                    "detail": "audio file not found",
                }
            )
            flush_report(report, args.report)
            render_progress(idx, total)
            continue

        stem, corrected_transcript, error = propose_name_and_correction(
            keys=keys,
            models=models,
            transcript_text=transcript_text,
            original_name=input_name,
            max_retries=args.max_retries,
            backoff_base=args.retry_backoff_base,
            per_request_delay=args.per_request_delay,
        )
        if stem is None:
            print(f"  ❌ Model error: {error}")
            report.append(
                {
                    "transcript": str(json_file.resolve()),
                    "original": input_name,
                    "proposed": "",
                    "original_transcript": transcript_text,
                    "corrected_transcript": "",
                    "action": "error",
                    "detail": error or "unknown error",
                }
            )
            flush_report(report, args.report)
            render_progress(idx, total)
            continue

        sanitized_stem = sanitize_filename(stem)
        target_path = args.audio_dir / f"{sanitized_stem}{original_audio.suffix}"
        if target_path.exists():
            target_path = find_unique_path(target_path)

        action = "would_rename" if not args.apply else "renamed"
        detail = ""

        if args.apply:
            try:
                original_audio.rename(target_path)
                print(f"  ✅ Renamed to {target_path.name}")
            except Exception as exc:
                action = "error"
                detail = f"rename failed: {exc}"
                print(f"  ❌ Rename failed: {exc}")
        else:
            print(f"  ➡️  Would rename to {target_path.name}")

        report.append(
            {
                "transcript": str(json_file.resolve()),
                "original": input_name,
                "proposed": target_path.name,
                "original_transcript": transcript_text,
                "corrected_transcript": corrected_transcript or "",
                "action": action,
                "detail": detail,
            }
        )
        flush_report(report, args.report)
        print(f"  💾 Report checkpoint saved after {idx} items.")
        render_progress(idx, total)

    flush_report(report, args.report)
    if total > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()
    print(f"Report written to {args.report} ({len(report)} items).")
    if not args.apply:
        print("Dry run complete. Re-run with --apply to perform renames.")
    else:
        print("Apply complete.")


if __name__ == "__main__":
    main()
