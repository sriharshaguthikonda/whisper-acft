from __future__ import annotations

import re
from typing import Optional


RUN_PREFIX = "RUN"
TOKEN_ORDER = ("k", "s", "b", "m", "a", "q", "c", "r", "id")


def slug_token(value: object, default: str = "unk") -> str:
    """Convert any token-like value to lowercase slug form."""
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text:
        return default
    text = text.replace("_", "-")
    text = re.sub(r"[^a-z0-9-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or default


def _alias(value: object, mapping: dict[str, str], default: str = "unk") -> str:
    token = slug_token(value, default=default)
    return mapping.get(token, token)


def normalize_kind(value: object) -> str:
    return _alias(
        value,
        {
            "train-only": "train-only",
            "trainonly": "train-only",
            "train-eval": "train-eval",
            "traineval": "train-eval",
            "eval-only": "eval-only",
            "evalonly": "eval-only",
            "partial": "partial",
            "misc": "misc",
            "unk": "unk",
        },
        default="unk",
    )


def normalize_quant(value: object) -> str:
    return _alias(
        value,
        {
            "qat": "qat",
            "noqat": "noqat",
            "fp16": "noqat",
            "int8": "qat",
            "unk": "unk",
        },
        default="unk",
    )


def normalize_ctx(value: object) -> str:
    return _alias(
        value,
        {
            "dyn": "dyn",
            "dynamic": "dyn",
            "static": "static",
            "full": "static",
            "full-ctx": "static",
            "unk": "unk",
        },
        default="unk",
    )


def build_run_folder_name(
    *,
    kind: object,
    stage: object,
    base: object,
    method: object,
    adapter: object,
    quant: object,
    ctx: object,
    rows: object,
    run_id: object,
) -> str:
    """Build canonical run folder name."""
    fields = {
        "k": normalize_kind(kind),
        "s": slug_token(stage),
        "b": slug_token(base),
        "m": slug_token(method),
        "a": slug_token(adapter),
        "q": normalize_quant(quant),
        "c": normalize_ctx(ctx),
        "r": slug_token(rows),
        "id": slug_token(run_id),
    }
    parts = [f"{key}-{fields[key]}" for key in TOKEN_ORDER]
    return f"{RUN_PREFIX}__" + "__".join(parts)


def parse_run_folder_name(folder_name: str) -> Optional[dict[str, str]]:
    """Parse canonical run folder name to token dict."""
    if not folder_name or not folder_name.startswith(f"{RUN_PREFIX}__"):
        return None
    tokens = folder_name.split("__")[1:]
    if not tokens:
        return None
    parsed: dict[str, str] = {}
    for token in tokens:
        if "-" not in token:
            continue
        key, value = token.split("-", 1)
        if not key:
            continue
        parsed[key] = slug_token(value)
    if not all(key in parsed for key in TOKEN_ORDER):
        return None
    return {key: parsed[key] for key in TOKEN_ORDER}


def is_canonical_run_folder_name(folder_name: str) -> bool:
    return parse_run_folder_name(folder_name) is not None
