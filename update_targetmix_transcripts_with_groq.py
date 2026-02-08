#!/usr/bin/env python3
"""Transcribe target/other audio in target-mix eval JSON via Groq and update references.

This script:
  - Loads evaluation_per_sample_predictions_targetmix_sweep.json (array)
  - Finds unique target_audio_path + other_audio_path entries
  - Transcribes each unique audio with Groq STT
  - Updates target_reference / other_reference (or adds new fields)

It reuses groq_transcribe_local_utils.py for the actual Groq request plumbing.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dotenv import load_dotenv

from groq_transcribe_local_utils import (  # type: ignore
    GROQ_TRANSCRIBE_URL,
    Key,
    KeyPool,
    groq_transcribe_requests,
)


def canonical_audio_key(p: str) -> str:
    if not p:
        return ""
    return os.path.normpath(p).replace("\\", "/").lower()


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i].isspace():
        i += 1
    return i


def _best_effort_parse_array(text: str) -> List[Any]:
    items: List[Any] = []
    dec = json.JSONDecoder()
    i = _skip_ws(text, 0)
    if i >= len(text) or text[i] != "[":
        return items
    i += 1
    while True:
        i = _skip_ws(text, i)
        if i >= len(text):
            break
        if text[i] == "]":
            break
        try:
            obj, j = dec.raw_decode(text, i)
            items.append(obj)
            i = _skip_ws(text, j)
            if i < len(text) and text[i] == ",":
                i += 1
                continue
            if i < len(text) and text[i] == "]":
                break
        except json.JSONDecodeError:
            break
    return items


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[warn] JSONDecodeError: {e.msg} at line {e.lineno} col {e.colno}")
        text = path.read_text(encoding="utf-8", errors="replace")
        items = _best_effort_parse_array(text)
        if items:
            print(f"[warn] Proceeding with best-effort parse: {len(items)} items recovered.")
            return items
        raise


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def load_cache_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get("canonical") or canonical_audio_key(obj.get("path", ""))
            if key:
                cache[key] = obj
    return cache


def append_cache_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def _load_keys_from_env_names(env_names: Sequence[str]) -> List[Key]:
    keys: List[Key] = []
    for name in env_names:
        v = os.environ.get(name)
        if v:
            # Support comma-separated keys
            if "," in v:
                for i, key_value in enumerate(v.split(",")):
                    key_value = key_value.strip()
                    if key_value:
                        keys.append(Key(name=f"{name}_{i+1}", value=key_value))
            else:
                keys.append(Key(name=name, value=v))
    return keys


def _split_csv(value: str) -> List[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def _load_keys_from_env_file(env_path: Path, env_names: Sequence[str]) -> List[Key]:
    keys: List[Key] = []
    if not env_path.exists():
        return keys

    continuation_key: Optional[str] = None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continuation_key = None
            continue

        if "=" in raw:
            name, val = raw.split("=", 1)
            name = name.strip()
            val = val.strip()
            if val and len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
                val = val[1:-1]
            if name in env_names:
                for v in _split_csv(val):
                    keys.append(Key(name=f"{name}_file_{len(keys)+1}", value=v))
                if name == "GROQ_API_KEYS" and val.rstrip().endswith(","):
                    continuation_key = name
                else:
                    continuation_key = None
            else:
                continuation_key = None
            continue

        if continuation_key == "GROQ_API_KEYS":
            for v in _split_csv(raw):
                keys.append(Key(name=f"{continuation_key}_file_{len(keys)+1}", value=v))
        else:
            continuation_key = None

    return keys


def _load_keys_from_file(keys_file: Path) -> List[Key]:
    keys: List[Key] = []
    for line in keys_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, val = line.split("=", 1)
            name = name.strip()
            val = val.strip()
        else:
            name = f"key{len(keys)+1}"
            val = line
        if val:
            keys.append(Key(name=name, value=val))
    return keys


def extract_text(resp_json: Dict[str, Any]) -> str:
    if not isinstance(resp_json, dict):
        return ""
    text = resp_json.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    segs = resp_json.get("segments")
    if isinstance(segs, list):
        parts = []
        for s in segs:
            if isinstance(s, dict):
                t = s.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
        if parts:
            return " ".join(parts).strip()
    return ""


def transcribe_one(
    audio_path: Path,
    pool: KeyPool,
    *,
    model: str,
    language: Optional[str],
    prompt: Optional[str],
    temperature: float,
    response_format: str,
    timestamp_granularities: Sequence[str],
    timeout_s: float,
    max_attempts: int,
    min_interval_s: float,
    jitter_s: float,
    last_request_by_key: Dict[str, float],
) -> Tuple[str, Dict[str, Any], str]:
    attempts = 0
    while True:
        attempts += 1
        key = pool.get()
        if min_interval_s > 0:
            now = time.time()
            last_t = last_request_by_key.get(key.name, 0.0)
            to_wait = (last_t + min_interval_s) - now
            if to_wait > 0:
                time.sleep(to_wait + random.uniform(0, jitter_s))
        try:
            r = groq_transcribe_requests(
                audio_path=audio_path,
                api_key=key.value,
                model=model,
                language=language,
                prompt=prompt,
                temperature=temperature,
                response_format=response_format,
                timestamp_granularities=timestamp_granularities,
                timeout_s=timeout_s,
            )
            last_request_by_key[key.name] = time.time()

            if r.status_code == 429:
                retry_after = r.headers.get("retry-after")
                ra_s = float(retry_after) if retry_after and retry_after.replace(".", "", 1).isdigit() else None
                pool.cooldown(key.name, retry_after=ra_s)
                if attempts < max_attempts:
                    continue
                r.raise_for_status()

            if 500 <= r.status_code < 600:
                pool.cooldown(key.name, retry_after=10)
                if attempts < max_attempts:
                    continue
                r.raise_for_status()

            r.raise_for_status()

            if response_format == "text":
                text = (r.text or "").strip()
                resp_json = {"text": text}
            else:
                resp_json = r.json()
                text = extract_text(resp_json)

            return key.name, resp_json, text
        except Exception:
            pool.cooldown(key.name, retry_after=15)
            if attempts < max_attempts:
                continue
            raise


def build_audio_list(data: List[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for obj in data:
        if not isinstance(obj, dict):
            continue
        p_t = obj.get("target_audio_path")
        p_o = obj.get("other_audio_path")
        if isinstance(p_t, str) and p_t.strip():
            paths.append(p_t.strip())
        if isinstance(p_o, str) and p_o.strip():
            paths.append(p_o.strip())
    return paths


def build_reference_map(data: List[Dict[str, Any]]) -> Dict[str, str]:
    ref_map: Dict[str, str] = {}
    for obj in data:
        if not isinstance(obj, dict):
            continue
        for path_key, ref_key in (
            ("target_audio_path", "target_reference"),
            ("other_audio_path", "other_reference"),
        ):
            p = obj.get(path_key)
            ref = obj.get(ref_key)
            if isinstance(p, str) and isinstance(ref, str):
                p = p.strip()
                ref = ref.strip()
                if not p or not ref:
                    continue
                k = canonical_audio_key(p)
                if not k:
                    continue
                existing = ref_map.get(k)
                if not existing or len(ref) > len(existing):
                    ref_map[k] = ref
    return ref_map


def build_prompt(reference: Optional[str], user_prompt: Optional[str], max_chars: int) -> str:
    lines = [
        "You are a transcription engine. Return the best possible transcript with proper punctuation and casing.",
    ]
    if user_prompt:
        lines.append(f"Additional instructions: {user_prompt.strip()}")
    if reference:
        ref = reference.strip()
        if max_chars and max_chars > 0 and len(ref) > max_chars:
            ref = ref[:max_chars].rstrip()
        lines.append("Reference transcript (may contain errors):")
        lines.append(ref)
    return "\n\n".join(lines).strip()


def apply_updates(
    data: List[Dict[str, Any]],
    cache: Dict[str, Dict[str, Any]],
    *,
    mode: str,
    target_field: str,
    other_field: str,
) -> Tuple[int, int]:
    updated_t = 0
    updated_o = 0
    for obj in data:
        if not isinstance(obj, dict):
            continue
        
        # Skip metadata entries
        if obj.get("__meta__"):
            continue
            
        t_path = obj.get("target_audio_path")
        if isinstance(t_path, str):
            key = canonical_audio_key(t_path)
            entry = cache.get(key)
            text = entry.get("text") if entry else None
            if isinstance(text, str) and text.strip():
                if mode == "replace":
                    obj["target_reference"] = text
                else:
                    obj[target_field] = text
                updated_t += 1

        o_path = obj.get("other_audio_path")
        if isinstance(o_path, str):
            key = canonical_audio_key(o_path)
            entry = cache.get(key)
            text = entry.get("text") if entry else None
            if isinstance(text, str) and text.strip():
                if mode == "replace":
                    obj["other_reference"] = text
                else:
                    obj[other_field] = text
                updated_o += 1
    return updated_t, updated_o


def _unique_env_paths(paths: Sequence[Path]) -> List[Path]:
    seen = set()
    out: List[Path] = []
    for p in paths:
        try:
            rp = p.resolve()
        except Exception:
            rp = p
        key = str(rp).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_json", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, default=None)
    ap.add_argument("--inplace", action="store_true", default=True)
    ap.add_argument("--cache_jsonl", type=Path, default=None)

    ap.add_argument("--mode", choices=["add", "replace"], default="replace",
                    help="add: write *_reference_groq fields, replace: overwrite *_reference fields")
    ap.add_argument("--target_field", default="target_reference_groq")
    ap.add_argument("--other_field", default="other_reference_groq")

    ap.add_argument("--model", default="whisper-large-v3-turbo")
    ap.add_argument("--language", default="en")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--use_reference_prompt", action="store_true", default=True)
    ap.add_argument("--no_reference_prompt", action="store_false", dest="use_reference_prompt")
    ap.add_argument("--prompt_max_chars", type=int, default=600,
                    help="Max chars of reference transcript to include in prompt (0 = no limit)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--response_format", default="verbose_json", choices=["json", "verbose_json", "text"])
    ap.add_argument("--timestamp_granularities", nargs="*", default=["segment"],
                    help="Use with verbose_json. Options: word, segment")
    ap.add_argument("--timeout_s", type=float, default=float(os.getenv("GROQ_TIMEOUT_S", "300.0")))
    ap.add_argument("--max_attempts", type=int, default=5)

    ap.add_argument("--max_rpm", type=float, default=float(os.getenv("GROQ_MAX_RPM", "18.0")))
    ap.add_argument("--jitter_s", type=float, default=float(os.getenv("GROQ_JITTER_S", "0.25")))

    ap.add_argument("--api_key", default=None)
    ap.add_argument("--key_env_names", nargs="*", default=["GROQ_API_KEY", "GROQ_API_KEYS"])
    ap.add_argument("--keys_file", type=Path, default=None)
    ap.add_argument("--env_file", type=Path, default=None,
                    help="Optional .env file to load Groq keys from (in addition to CWD/script .env).")

    args = ap.parse_args()

    if not args.input_json.exists():
        raise SystemExit(f"Missing input_json: {args.input_json}")

    if args.out_json is None:
        args.out_json = args.input_json.with_name(args.input_json.stem + "_groq" + args.input_json.suffix)

    if args.inplace:
        args.out_json = args.input_json

    if args.cache_jsonl is None:
        args.cache_jsonl = args.input_json.with_name(args.input_json.stem + "_groq_cache.jsonl")

    # Load .envs (explicit, CWD, script dir) without overriding existing env vars
    env_candidates: List[Path] = []
    if args.env_file:
        env_candidates.append(args.env_file)
    env_candidates.append(Path(".env"))
    env_candidates.append(Path(__file__).with_name(".env"))
    env_files = _unique_env_paths(env_candidates)
    for p in env_files:
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)

    # Keys
    keys: List[Key] = []
    if args.api_key:
        keys.append(Key("api_key", args.api_key))
    if args.keys_file:
        keys.extend(_load_keys_from_file(args.keys_file))
    keys.extend(_load_keys_from_env_names(args.key_env_names))
    for env_path in env_files:
        keys.extend(_load_keys_from_env_file(env_path, args.key_env_names))

    # De-dup by value
    seen = set()
    uniq: List[Key] = []
    for k in keys:
        if k.value in seen:
            continue
        seen.add(k.value)
        uniq.append(k)
    keys = uniq

    if not keys:
        raise SystemExit("No Groq API keys found. Set GROQ_API_KEY or pass --keys_file / --api_key.")

    pool = KeyPool(keys)

    if args.timestamp_granularities and args.response_format != "verbose_json":
        print("NOTE: timestamp_granularities requires response_format=verbose_json on Groq; forcing.")
        args.response_format = "verbose_json"

    # Load data
    data = load_json(args.input_json)
    if not isinstance(data, list):
        raise SystemExit("Expected input_json to be a JSON array.")

    # Build unique audio list
    dict_rows = [x for x in data if isinstance(x, dict)]
    all_paths = build_audio_list(dict_rows)
    uniq_map: Dict[str, str] = {}
    for p in all_paths:
        k = canonical_audio_key(p)
        if k and k not in uniq_map:
            uniq_map[k] = p

    print(f"Found {len(all_paths)} audio references ({len(uniq_map)} unique).")

    ref_map = build_reference_map(dict_rows)

    # Load cache
    cache = load_cache_jsonl(args.cache_jsonl)
    print(f"Loaded cache: {len(cache)} entries from {args.cache_jsonl}")

    # Transcribe missing
    min_interval = 60.0 / args.max_rpm if args.max_rpm and args.max_rpm > 0 else 0.0
    last_request_by_key: Dict[str, float] = {}
    ok = 0
    skipped = 0
    failed = 0

    for canonical_key, path_str in uniq_map.items():
        ref_text = ref_map.get(canonical_key)
        prompt_used = args.prompt
        if args.use_reference_prompt:
            prompt_used = build_prompt(ref_text, args.prompt, int(args.prompt_max_chars))

        cached = cache.get(canonical_key) or {}
        if cached.get("text") and cached.get("prompt_effective") == prompt_used:
            skipped += 1
            continue

        audio_path = Path(path_str)
        if not audio_path.exists():
            failed += 1
            print(f"[missing] {path_str}")
            continue

        try:
            key_name, resp_json, text = transcribe_one(
                audio_path,
                pool,
                model=args.model,
                language=args.language,
                prompt=prompt_used,
                temperature=args.temperature,
                response_format=args.response_format,
                timestamp_granularities=args.timestamp_granularities,
                timeout_s=args.timeout_s,
                max_attempts=args.max_attempts,
                min_interval_s=min_interval,
                jitter_s=args.jitter_s,
                last_request_by_key=last_request_by_key,
            )
            ok += 1
            entry = {
                "canonical": canonical_key,
                "path": path_str,
                "text": text,
                "request": {
                    "endpoint": GROQ_TRANSCRIBE_URL,
                    "model": args.model,
                    "language": args.language,
                    "prompt": prompt_used,
                    "temperature": args.temperature,
                    "response_format": args.response_format,
                    "timestamp_granularities": list(args.timestamp_granularities),
                    "api_key_name": key_name,
                },
                "groq_response": resp_json,
                "prompt_effective": prompt_used,
                "ts": time.time(),
            }
            append_cache_jsonl(args.cache_jsonl, entry)
            cache[canonical_key] = entry
            print(f"[ok] {audio_path.name} -> {len(text)} chars")
        except Exception as e:
            failed += 1
            print(f"[fail] {path_str} :: {repr(e)}")

    # Apply updates
    updated_t, updated_o = apply_updates(
        [x for x in data if isinstance(x, dict)],
        cache,
        mode=args.mode,
        target_field=args.target_field,
        other_field=args.other_field,
    )

    # Write output
    if args.inplace:
        backup = args.input_json.with_suffix(args.input_json.suffix + ".bak")
        if not backup.exists():
            backup.write_text(args.input_json.read_text(encoding="utf-8"), encoding="utf-8")
        write_json(args.out_json, data)
    else:
        write_json(args.out_json, data)

    print("\nDone")
    print(f"Transcribed OK: {ok} | Skipped (cache): {skipped} | Failed: {failed}")
    print(f"Updated target refs: {updated_t} | Updated other refs: {updated_o}")
    print(f"Output: {args.out_json}")
    print(f"Cache: {args.cache_jsonl}")


if __name__ == "__main__":
    main()
