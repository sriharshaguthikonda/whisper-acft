#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


REQUIRED_MODEL_COLUMNS = {
    "model_name",
    "backend",
    "model_ref",
    "language_mode",
    "enabled",
}


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    backend: str
    model_ref: str
    language_mode: str
    enabled: bool
    decoder_preset: str = ""
    batch_hint: str = ""
    notes: str = ""
    expected_sr: str = ""


def parse_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not env_path.exists():
        return values
    for raw in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            values[k] = v
    return values


def resolve_hf_token(repo_root: Path, env_var: str = "HF_TOKEN") -> str:
    direct = os.environ.get(env_var, "").strip()
    if direct:
        return direct

    # Keep fallback aliases for compatibility.
    for alias in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(alias, "").strip()
        if value:
            return value

    env_values = parse_env_file(repo_root / ".env")
    direct_file = env_values.get(env_var, "").strip()
    if direct_file:
        return direct_file
    for alias in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
        value = env_values.get(alias, "").strip()
        if value:
            return value
    return ""


def normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def load_json(path: Path, default: Optional[dict] = None) -> dict:
    if default is None:
        default = {}
    if not path.exists():
        return dict(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return dict(default)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_cmd(cmd: list[str], *, cwd: Optional[Path] = None, env: Optional[dict[str, str]] = None) -> None:
    pretty = " ".join(cmd)
    print(f"$ {pretty}")
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


def load_models_csv(csv_path: Path) -> list[ModelSpec]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Model list CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        missing = REQUIRED_MODEL_COLUMNS - headers
        if missing:
            raise ValueError(
                f"Missing required model list columns: {sorted(missing)}. Found: {sorted(headers)}"
            )

        out: list[ModelSpec] = []
        for idx, row in enumerate(reader, start=2):
            model_name = (row.get("model_name") or "").strip()
            backend = (row.get("backend") or "").strip().lower()
            model_ref = (row.get("model_ref") or "").strip()
            language_mode = (row.get("language_mode") or "").strip().lower()
            enabled = normalize_bool(row.get("enabled"), default=True)

            if not model_name:
                raise ValueError(f"Row {idx}: model_name is empty")
            if backend not in {"hf_transformers", "nemo_parakeet"}:
                raise ValueError(
                    f"Row {idx}: backend must be hf_transformers or nemo_parakeet, got '{backend}'"
                )
            if not model_ref:
                raise ValueError(f"Row {idx}: model_ref is empty")
            if not language_mode:
                raise ValueError(f"Row {idx}: language_mode is empty")

            out.append(
                ModelSpec(
                    model_name=model_name,
                    backend=backend,
                    model_ref=model_ref,
                    language_mode=language_mode,
                    enabled=enabled,
                    decoder_preset=(row.get("decoder_preset") or "").strip(),
                    batch_hint=(row.get("batch_hint") or "").strip(),
                    notes=(row.get("notes") or "").strip(),
                    expected_sr=(row.get("expected_sr") or "").strip(),
                )
            )

    return out


def split_models_by_backend(models: list[ModelSpec]) -> tuple[list[ModelSpec], list[ModelSpec]]:
    active = [m for m in models if m.enabled]
    hf_models = [m for m in active if m.backend == "hf_transformers"]
    nemo_models = [m for m in active if m.backend == "nemo_parakeet"]
    return hf_models, nemo_models


def beep_done() -> None:
    if os.name != "nt":
        return
    try:
        import winsound  # type: ignore

        winsound.Beep(1000, 300)
        winsound.Beep(1200, 300)
        winsound.Beep(1500, 500)
    except Exception:
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass
