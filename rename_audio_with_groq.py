from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


SYSTEM_PROMPT = (
    "You are renaming audio files that contain FAKE medical consultations. "
    "Generate a concise, human-readable title for the conversation. "
    "Return only the filename stem (no extension). Use underscores instead of spaces. "
    "Avoid unsafe or identifying details. Keep it short and specific to the main topic."
)


"""
use : 

python rename_audio_with_groq.py --transcripts-dir "i:\P2GPT_google_drive\My Drive\Transcriptions" --audio-dir "i:\Record" --report rename_report.json --retry-backoff-base 2
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
    default_models = env.get("GROQ_MODEL", "llama-3.1-8b-instant")
    default_transcripts = env.get("TRANSCRIPTS_DIR", "")
    default_audio = env.get("AUDIO_DIR", "")

    parser = argparse.ArgumentParser(
        description="Rename audio files using Groq chat model suggestions based on transcript JSON files."
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
        default=Path("rename_report.json"),
        help="Path to write the dry-run/apply report (defaults to rename_report.json).",
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


def rotate(values: List[str]):
    while True:
        for v in values:
            yield v


def rotate_pairs(keys: List[str], models: List[str]):
    pairs: List[Tuple[str, str]] = [(k, m) for k in keys for m in models]
    while True:
        for pair in pairs:
            yield pair


def call_groq_chat(
    api_key: str,
    model: str,
    transcript_text: str,
    original_name: str,
) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
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
            "max_tokens": 64,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices or not choices[0].get("message", {}).get("content"):
        raise ValueError("Empty response from model")
    choice = choices[0]["message"]["content"].strip()
    # Ensure single line
    return choice.splitlines()[0].strip()


def propose_name(
    keys: List[str],
    models: List[str],
    transcript_text: str,
    original_name: str,
    max_retries: int,
    backoff_base: float,
) -> Tuple[Optional[str], Optional[str]]:
    pair_rotator = rotate_pairs(keys, models)
    last_error: Optional[str] = None
    for attempt in range(max_retries):
        api_key, model = next(pair_rotator)
        try:
            stem = call_groq_chat(api_key, model, transcript_text, original_name)
            print(transcript_text)
            return stem, None
        except requests.exceptions.HTTPError as exc:
            last_error = f"HTTPError: {exc}"
            # Backoff on rate limits and rotate
            sleep_s = max(1.0, backoff_base ** (attempt + 1))
            time.sleep(sleep_s)
            continue
        except Exception as exc:  # broad to log any API issue
            last_error = f"{type(exc).__name__}: {exc}"
            sleep_s = max(1.0, backoff_base ** (attempt + 1))
            time.sleep(sleep_s)
            continue
    return None, last_error


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

    processed_set = {
        normalize_path(Path(item.get("transcript"))) for item in report if item.get("transcript")
    }

    for idx, json_file in enumerate(transcripts, start=1):
        transcript_key = normalize_path(json_file)
        if args.resume and transcript_key in processed_set:
            print(f"[{idx}/{total}] Skipping {json_file.name} (already in report).")
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
                    "action": "error",
                    "detail": f"failed to load transcript: {exc}",
                }
            )
            continue

        original_audio = args.audio_dir / input_name
        if not original_audio.exists():
            print(f"  ⚠️  Audio not found for {input_name}")
            report.append(
                {
                    "transcript": str(json_file.resolve()),
                    "original": input_name,
                    "proposed": "",
                    "action": "missing_audio",
                    "detail": "audio file not found",
                }
            )
            continue

        stem, error = propose_name(
            keys=keys,
            models=models,
            transcript_text=transcript_text,
            original_name=input_name,
            max_retries=args.max_retries,
            backoff_base=args.retry_backoff_base,
        )
        if stem is None:
            print(f"  ❌ Model error: {error}")
            report.append(
                {
                    "transcript": str(json_file.resolve()),
                    "original": input_name,
                    "proposed": "",
                    "action": "error",
                    "detail": error or "unknown error",
                }
            )
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
                "action": action,
                "detail": detail,
            }
        )

        if idx % args.flush_interval == 0:
            flush_report(report, args.report)
            print(f"  💾 Report checkpoint saved after {idx} items.")

    flush_report(report, args.report)
    print(f"Report written to {args.report} ({len(report)} items).")
    if not args.apply:
        print("Dry run complete. Re-run with --apply to perform renames.")
    else:
        print("Apply complete.")


if __name__ == "__main__":
    main()
