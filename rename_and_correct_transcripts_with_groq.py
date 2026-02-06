from __future__ import annotations

"""rename_and_correct_transcripts_with_groq.py

What this does
- Reads transcript JSON files (same input format as your Groq script).
- For each transcript:
  1) asks an LLM to propose a concise filename_stem
  2) asks the LLM to correct ONLY punctuation in the transcript (no word changes)
- Optionally renames matching audio files (when --apply is set)
- Writes/updates a report JSON file for resume + audit.

Supported providers (rotated automatically)
- Groq (OpenAI-compatible)
- OpenRouter (OpenAI-compatible)
- GitHub Models (OpenAI-compatible)
- Cloudflare Workers AI (OpenAI-compatible)
- Together (OpenAI-compatible)
- Fireworks (OpenAI-compatible)
- Mistral (OpenAI-compatible)
- Gemini (native REST generateContent)

You control providers via --providers and environment variables in .env.

Example:
  python rename_and_correct_transcripts_with_groq.py \
    --transcripts-dir "I:\Transcriptions" \
    --audio-dir "I:\Record_harsha" \
    --providers "groq,github,openrouter,cloudflare,gemini" \
    --report rename_and_corrected_transcript_report.json \
    --retry-backoff-base 2



Notes
- This script will NOT strip speaker labels and will NOT normalise whitespace.
  (It only sanitises the filename stem.)
"""

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


# ----------------------------
# PROMPTING
# ----------------------------
SYSTEM_PROMPT = (
    "You are renaming audio files that contain FAKE medical consultations. "
    "Generate a concise, human-readable filename stem for the conversation AND provide a corrected transcript.\n"
    "Rules for filename:\n"
    "- Return ONLY the filename stem (no extension)\n"
    "- Use underscores instead of spaces\n"
    "- Keep it descriptive and specific to the main topic\n\n"
    "Transcript correction rule:\n"
    "- Correct ONLY punctuation.\n"
    "- Do NOT add/remove/change any words, spelling, casing, numbers, or speaker labels.\n"
    "- Keep whitespace and line breaks as close as possible to the input.\n\n"
    "Respond ONLY as compact JSON with exactly these two keys: "
    '{"filename_stem":"<stem_with_underscores>","corrected_transcript":"<text>"}'
)

RESPONSE_SCHEMA: Dict[str, object] = {
    "type": "object",
    "properties": {
        "filename_stem": {"type": "string"},
        "corrected_transcript": {"type": "string"},
    },
    "required": ["filename_stem", "corrected_transcript"],
    "additionalProperties": False,
}


# ----------------------------
# PROVIDER CONFIG
# ----------------------------
@dataclass
class Target:
    name: str
    kind: str  # openai_compat | gemini
    chat_url: str
    api_keys: List[str]
    models: List[str]
    extra_headers: Dict[str, str]
    supports_response_format: bool = True


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


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def build_targets_from_env_and_args(args: argparse.Namespace, env: Dict[str, str]) -> List[Target]:
    """Build enabled targets from env vars and CLI.

    You enable providers via --providers (comma-separated).
    Each provider reads keys+models from env.
    """

    want = {p.strip().lower() for p in _split_csv(args.providers)}
    targets: List[Target] = []

    # --- Groq (OpenAI-compatible) ---
    if "groq" in want:
        keys = _split_csv(args.api_keys) or _split_csv(env.get("GROQ_API_KEYS", ""))
        models = _split_csv(args.models) or _split_csv(env.get("GROQ_MODELS", "")) or [
            env.get("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
        ]
        if keys:
            targets.append(
                Target(
                    name="groq",
                    kind="openai_compat",
                    chat_url="https://api.groq.com/openai/v1/chat/completions",
                    api_keys=keys,
                    models=models,
                    extra_headers={},
                    supports_response_format=True,
                )
            )
        else:
            print("⚠️  No Groq API keys found, skipping Groq provider")

    # --- OpenRouter (OpenAI-compatible) ---
    if "openrouter" in want:
        keys = _split_csv(env.get("OPENROUTER_API_KEYS", ""))
        models = _split_csv(env.get("OPENROUTER_MODELS", ""))
        # Optional but recommended by OpenRouter
        referer = env.get("OPENROUTER_HTTP_REFERER", "")
        title = env.get("OPENROUTER_X_TITLE", "")
        headers: Dict[str, str] = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        if keys and models:
            targets.append(
                Target(
                    name="openrouter",
                    kind="openai_compat",
                    chat_url="https://openrouter.ai/api/v1/chat/completions",
                    api_keys=keys,
                    models=models,
                    extra_headers=headers,
                    supports_response_format=True,  # may pass through; script will auto-fallback if rejected
                )
            )
        else:
            print("⚠️  No OpenRouter API keys or models found, skipping OpenRouter provider")

    # --- GitHub Models (OpenAI-compatible) ---
    if "github" in want or "github_models" in want:
        keys = _split_csv(env.get("GITHUB_MODELS_TOKENS", "")) or _split_csv(env.get("GITHUB_MODELS_TOKEN", ""))
        models = _split_csv(env.get("GITHUB_MODELS", ""))
        # GitHub recommends these headers for their REST APIs
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if keys and models:
            targets.append(
                Target(
                    name="github_models",
                    kind="openai_compat",
                    chat_url="https://models.github.ai/inference/chat/completions",
                    api_keys=keys,
                    models=models,
                    extra_headers=headers,
                    supports_response_format=True,
                )
            )
        else:
            print("⚠️  No GitHub Models tokens or models found, skipping GitHub Models provider")

    # --- Cloudflare Workers AI (OpenAI-compatible) ---
    if "cloudflare" in want:
        account_id = env.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
        keys = _split_csv(env.get("CLOUDFLARE_API_TOKENS", "")) or _split_csv(env.get("CLOUDFLARE_API_TOKEN", ""))
        models = _split_csv(env.get("CLOUDFLARE_MODELS", ""))
        if account_id and keys and models:
            targets.append(
                Target(
                    name="cloudflare",
                    kind="openai_compat",
                    chat_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1/chat/completions",
                    api_keys=keys,
                    models=models,
                    extra_headers={},
                    supports_response_format=True,
                )
            )
        else:
            print("⚠️  No Cloudflare account ID, API tokens, or models found, skipping Cloudflare provider")

    # --- Together (OpenAI-compatible) ---
    if "together" in want:
        keys = _split_csv(env.get("TOGETHER_API_KEYS", "")) or _split_csv(env.get("TOGETHER_API_KEY", ""))
        models = _split_csv(env.get("TOGETHER_MODELS", ""))
        if keys and models:
            targets.append(
                Target(
                    name="together",
                    kind="openai_compat",
                    chat_url="https://api.together.xyz/v1/chat/completions",
                    api_keys=keys,
                    models=models,
                    extra_headers={},
                    supports_response_format=True,
                )
            )
        else:
            print("⚠️  No Together API keys or models found, skipping Together provider")

    # --- Fireworks (OpenAI-compatible) ---
    if "fireworks" in want:
        keys = _split_csv(env.get("FIREWORKS_API_KEYS", "")) or _split_csv(env.get("FIREWORKS_API_KEY", ""))
        models = _split_csv(env.get("FIREWORKS_MODELS", ""))
        if keys and models:
            targets.append(
                Target(
                    name="fireworks",
                    kind="openai_compat",
                    chat_url="https://api.fireworks.ai/inference/v1/chat/completions",
                    api_keys=keys,
                    models=models,
                    extra_headers={},
                    supports_response_format=True,
                )
            )
        else:
            print("⚠️  No Fireworks API keys or models found, skipping Fireworks provider")

    # --- Mistral (OpenAI-compatible endpoint) ---
    if "mistral" in want:
        keys = _split_csv(env.get("MISTRAL_API_KEYS", "")) or _split_csv(env.get("MISTRAL_API_KEY", ""))
        models = _split_csv(env.get("MISTRAL_MODELS", ""))
        if keys and models:
            targets.append(
                Target(
                    name="mistral",
                    kind="openai_compat",
                    chat_url="https://api.mistral.ai/v1/chat/completions",
                    api_keys=keys,
                    models=models,
                    extra_headers={},
                    supports_response_format=True,
                )
            )
        else:
            print("⚠️  No Mistral API keys or models found, skipping Mistral provider")

    # --- Gemini (native REST generateContent) ---
    if "gemini" in want:
        keys = _split_csv(env.get("GEMINI_API_KEYS", "")) or _split_csv(env.get("GEMINI_API_KEY", ""))
        models = _split_csv(env.get("GEMINI_MODELS", ""))
        if keys and models:
            # model is inserted into URL at call time
            targets.append(
                Target(
                    name="gemini",
                    kind="gemini",
                    chat_url="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                    api_keys=keys,
                    models=models,
                    extra_headers={},
                    supports_response_format=False,
                )
            )
        else:
            print("⚠️  No Gemini API keys or models found, skipping Gemini provider")

    return targets


# ----------------------------
# CLI
# ----------------------------
def parse_args() -> argparse.Namespace:
    env = load_env(Path(".env"))

    default_keys = env.get("GROQ_API_KEYS", "")
    default_models = env.get("GROQ_MODELS", env.get("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct"))
    default_transcripts = env.get("TRANSCRIPTS_DIR", "")
    default_audio = env.get("AUDIO_DIR", "")

    parser = argparse.ArgumentParser(
        description="Rename audio files and correct punctuation-only transcripts using multiple LLM APIs (rotated)."
    )
    parser.add_argument("--transcripts-dir", type=Path, default=Path(default_transcripts) if default_transcripts else None)
    parser.add_argument("--audio-dir", type=Path, default=Path(default_audio) if default_audio else None)

    # Backwards-compatible Groq args (also used if you only want Groq)
    parser.add_argument("--api-keys", type=str, default=default_keys, help="Comma-separated Groq API keys (or leave empty and use .env).")
    parser.add_argument("--model", "--models", dest="models", type=str, default=default_models, help="Groq models (comma-separated) for Groq provider.")

    # New: enable/disable providers
    parser.add_argument(
        "--providers",
        type=str,
        default=env.get("PROVIDERS", "groq"),
        help="Comma-separated providers to use: groq,openrouter,github,cloudflare,together,fireworks,mistral,gemini",
    )

    parser.add_argument("--apply", action="store_true", help="Apply renames. Without this flag, only a dry-run report is produced.")
    parser.add_argument("--report", type=Path, default=Path("rename_and_corrected_transcript_report.json"))

    parser.add_argument("--max-retries", type=int, default=3, help="Retries per transcript across rotated (provider,key,model) pairs.")
    parser.add_argument("--flush-interval", type=int, default=5, help="Flush report to disk after this many items.")

    parser.add_argument("--resume", action="store_true", default=True, help="Resume using existing report file to skip already processed transcripts (default: on).")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Disable resume; process all transcripts regardless of existing report.")

    parser.add_argument("--retry-backoff-base", type=float, default=1.5, help="Base seconds for exponential backoff on API errors.")
    parser.add_argument("--per-request-delay", type=float, default=0.0, help="Optional fixed sleep seconds between API calls.")

    parser.add_argument(
        "--strict-output",
        action="store_true",
        help="Request strict structured output when the provider supports it (OpenAI-style response_format json_schema).",
    )

    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds.")

    return parser.parse_args()


# ----------------------------
# UTIL
# ----------------------------
def sanitize_filename(stem: str, max_len: int = 80) -> str:
    stem = stem.replace(" ", "_")
    stem = re.sub(r"[^\w\-]", "_", stem)
    stem = re.sub(r"_+", "_", stem)
    stem = stem.strip("._-")
    if len(stem) > max_len:
        stem = stem[:max_len].rstrip("._-")
    return stem or "untitled"


def extract_response(content: str) -> Tuple[str, str]:
    content = content.strip()
    # strip code fences if present
    if content.startswith("```"):
        content = content.strip("`")
        if "\n" in content:
            content = content.split("\n", 1)[1]
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # salvage JSON if model adds prose
        if "{" in content and "}" in content:
            snippet = content[content.find("{") : content.rfind("}") + 1]
            data = json.loads(snippet)
        else:
            raise ValueError("Model did not return JSON.")

    filename_stem = data.get("filename_stem") or data.get("new_filename_stem")
    corrected_transcript = data.get("corrected_transcript") or data.get("transcript")
    if not filename_stem or corrected_transcript is None:
        raise ValueError("Missing filename_stem or corrected_transcript in model response.")
    return str(filename_stem).strip(), str(corrected_transcript)


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
            raise RuntimeError(f"Too many collisions for file: {target}")


def flush_report(report: List[Dict[str, object]], path: Path) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def load_existing_report(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []


def load_transcript(json_path: Path) -> Tuple[str, str]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    input_name = data.get("input_file", {}).get("name")
    transcript_text = data.get("groq_response", {}).get("text", "")
    if not input_name:
        raise ValueError(f"Missing input_file.name in {json_path}")
    return input_name, transcript_text


# ----------------------------
# PROVIDER CALLS
# ----------------------------
def _messages_for_openai(transcript_text: str, original_name: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Original filename: {original_name}\n"
                f"Transcript:\n{transcript_text}\n\n"
                "Return ONLY the JSON object."
            ),
        },
    ]


def call_openai_compat_chat(
    *,
    target: Target,
    api_key: str,
    model: str,
    transcript_text: str,
    original_name: str,
    timeout_s: float,
    per_request_delay: float,
    strict_output: bool,
) -> Tuple[str, str]:
    if per_request_delay > 0:
        time.sleep(per_request_delay)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    headers.update(target.extra_headers or {})

    payload: Dict[str, object] = {
        "model": model,
        "messages": _messages_for_openai(transcript_text, original_name),
        "temperature": 0.2,
        "max_tokens": 4096,
    }

    # Try strict structured output when supported
    if target.supports_response_format:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "rename_and_correct_transcript",
                "strict": bool(strict_output),
                "schema": RESPONSE_SCHEMA,
            },
        }

    def _post(p: Dict[str, object]) -> requests.Response:
        return requests.post(target.chat_url, headers=headers, json=p, timeout=timeout_s)

    resp = _post(payload)

    # Auto-fallback: some providers reject response_format
    if resp.status_code in (400, 404) and target.supports_response_format:
        text = ""
        try:
            text = resp.text or ""
        except Exception:
            text = ""
        hint = (text or "").lower()
        if "response_format" in hint or "json_schema" in hint or "unknown" in hint or "unrecognized" in hint:
            payload.pop("response_format", None)
            resp = _post(payload)

    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices or not choices[0].get("message", {}).get("content"):
        raise ValueError("Empty response from model")
    content = str(choices[0]["message"]["content"]).strip()
    return extract_response(content)


def call_gemini_generate_content(
    *,
    api_key: str,
    model: str,
    transcript_text: str,
    original_name: str,
    timeout_s: float,
    per_request_delay: float,
) -> Tuple[str, str]:
    if per_request_delay > 0:
        time.sleep(per_request_delay)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    params = {"key": api_key}
    headers = {"Content-Type": "application/json"}

    user_text = (
        f"Original filename: {original_name}\n"
        f"Transcript:\n{transcript_text}\n\n"
        "Return ONLY the JSON object."
    )

    # Use responseSchema (OpenAPI subset) with responseMimeType=application/json
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }

    resp = requests.post(url, params=params, headers=headers, json=body, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()

    candidates = data.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini: empty candidates")
    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    if not parts or not parts[0].get("text"):
        raise ValueError("Gemini: missing content.parts[0].text")

    text = str(parts[0]["text"]).strip()
    return extract_response(text)


# ----------------------------
# ROTATION + RETRIES
# ----------------------------
def propose_name_and_correction(
    *,
    targets: List[Target],
    transcript_text: str,
    original_name: str,
    max_retries: int,
    backoff_base: float,
    per_request_delay: float,
    strict_output: bool,
    timeout_s: float,
) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, str]], Optional[str]]:
    """Returns (stem, corrected_transcript, meta, error)."""

    # Build list of (provider, key, model) tuples - try each provider once before retrying
    expanded: List[Tuple[Target, str, str]] = []
    for t in targets:
        for k in t.api_keys:
            for m in t.models:
                expanded.append((t, k, m))

    if not expanded:
        return None, None, None, "No provider/key/model pairs configured."

    last_error: Optional[str] = None
    
    # Try each provider once before retrying any
    for provider_attempt in range(max(1, max_retries + 1)):
        for t, api_key, model in expanded:
            try:
                if t.kind == "gemini":
                    stem, corrected = call_gemini_generate_content(
                        api_key=api_key,
                        model=model,
                        transcript_text=transcript_text,
                        original_name=original_name,
                        timeout_s=timeout_s,
                        per_request_delay=per_request_delay,
                    )
                else:
                    stem, corrected = call_openai_compat_chat(
                        target=t,
                        api_key=api_key,
                        model=model,
                        transcript_text=transcript_text,
                        original_name=original_name,
                        timeout_s=timeout_s,
                        per_request_delay=per_request_delay,
                        strict_output=strict_output,
                    )
                meta = {"provider": t.name, "model": model}
                return stem, corrected, meta, None

            except requests.exceptions.HTTPError as exc:
                body = ""
                status = ""
                if exc.response is not None:
                    status = str(exc.response.status_code)
                    try:
                        body = exc.response.text
                    except Exception:
                        body = ""
                last_error = f"HTTPError {status}: {body or exc}"
                
                # Only backoff if this is the last provider, otherwise try next provider immediately
                is_last_provider = (t == expanded[-1][0] and api_key == expanded[-1][1] and model == expanded[-1][2])
                if is_last_provider:
                    retry_after = 0.0
                    if exc.response is not None:
                        try:
                            retry_after = float(exc.response.headers.get("Retry-After", "0"))
                        except Exception:
                            retry_after = 0.0
                    base = max(1.0, backoff_base ** provider_attempt)
                    jitter = random.uniform(0.0, 0.5 * base)
                    time.sleep(base + retry_after + jitter)
                # If not last provider, continue immediately to next provider (no backoff)
                continue

            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                # Only backoff if this is the last provider
                is_last_provider = (t == expanded[-1][0] and api_key == expanded[-1][1] and model == expanded[-1][2])
                if is_last_provider:
                    base = max(1.0, backoff_base ** provider_attempt)
                    jitter = random.uniform(0.0, 0.3 * base)
                    time.sleep(base + jitter)
                continue

    return None, None, None, last_error


# ----------------------------
# MAIN
# ----------------------------
def main() -> None:
    args = parse_args()
    env = load_env(Path(".env"))

    if not args.transcripts_dir or not args.transcripts_dir.exists():
        raise SystemExit("Transcripts directory missing or not provided.")
    if not args.audio_dir or not args.audio_dir.exists():
        raise SystemExit("Audio directory missing or not provided.")

    targets = build_targets_from_env_and_args(args, env)
    if not targets:
        raise SystemExit(
            "No providers configured.\n"
            "Fix: set --providers and add matching API keys/models in .env."
        )

    report: List[Dict[str, object]] = []
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
        normalize_path(Path(item.get("transcript")))
        for item in report
        if item.get("transcript") and item.get("corrected_transcript")
    }

    script_start_time = time.time()
    last_successful_time: Optional[float] = None
    successful_requests_count = 0

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
            print(f"[{idx}/{total}] Skipping {json_file.name} (already processed).")
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
                    "timestamp": datetime.now().isoformat(),
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
                    "timestamp": datetime.now().isoformat(),
                }
            )
            flush_report(report, args.report)
            render_progress(idx, total)
            continue

        request_start_time = time.time()
        stem, corrected_transcript, meta, error = propose_name_and_correction(
            targets=targets,
            transcript_text=transcript_text,
            original_name=input_name,
            max_retries=args.max_retries,
            backoff_base=args.retry_backoff_base,
            per_request_delay=args.per_request_delay,
            strict_output=args.strict_output,
            timeout_s=args.timeout,
        )
        request_end_time = time.time()
        request_duration = request_end_time - request_start_time

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
                    "provider": (meta or {}).get("provider", ""),
                    "model": (meta or {}).get("model", ""),
                    "timestamp": datetime.now().isoformat(),
                    "request_duration_seconds": "",
                    "time_since_last_success_seconds": "",
                }
            )
            flush_report(report, args.report)
            render_progress(idx, total)
            continue

        successful_requests_count += 1
        time_since_last_success = ""
        if last_successful_time is not None:
            time_since_last_success = str(request_start_time - last_successful_time)
        last_successful_time = request_end_time

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

        prov = (meta or {}).get("provider", "")
        mod = (meta or {}).get("model", "")

        print(f"  🔌 Provider: {prov} | Model: {mod}")
        print(f"  ⏱️  Request duration: {request_duration:.2f}s")
        if time_since_last_success:
            try:
                print(f"  ⏱️  Time since last success: {float(time_since_last_success):.2f}s")
            except Exception:
                print(f"  ⏱️  Time since last success: {time_since_last_success}")
        print(f"  📊 Success count: {successful_requests_count}")

        report.append(
            {
                "transcript": str(json_file.resolve()),
                "original": input_name,
                "proposed": target_path.name,
                "original_transcript": transcript_text,
                "corrected_transcript": corrected_transcript,
                "action": action,
                "detail": detail,
                "provider": prov,
                "model": mod,
                "timestamp": datetime.now().isoformat(),
                "request_duration_seconds": f"{request_duration:.2f}",
                "time_since_last_success_seconds": time_since_last_success,
            }
        )

        flush_report(report, args.report)
        print(f"  💾 Report checkpoint saved after {idx} items.")
        render_progress(idx, total)

    flush_report(report, args.report)
    if total > 0:
        sys.stdout.write("\n")
        sys.stdout.flush()

    total_script_time = time.time() - script_start_time
    print(f"Report written to {args.report} ({len(report)} items).")
    print(f"Total script runtime: {total_script_time:.2f}s")
    print(f"Successful requests: {successful_requests_count}")
    if successful_requests_count > 0:
        print(f"Average time per success: {total_script_time / successful_requests_count:.2f}s")

    if not args.apply:
        print("Dry run complete. Re-run with --apply to perform renames.")
    else:
        print("Apply complete.")


if __name__ == "__main__":
    main()
