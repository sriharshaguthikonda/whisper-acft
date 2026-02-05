"""stage17_main_utils_peft.py

PEFT/LoRA helper utilities for Stage 17 training.

Goal:
- Keep your existing Stage 17 script mostly unchanged.
- Enable LoRA (parameter-efficient fine-tuning) via env vars.
- Make resume logic work when the checkpoint folder contains only PEFT adapter weights.

Why LoRA here?
- Full fine-tuning can drift fast (and your WER can go up even when per-sample text looks OK).
- LoRA freezes the base weights and trains small low-rank matrices instead.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Any

_PEFT_OVERRIDES: dict[str, Optional[Any]] = {
    "WHISPER_USE_PEFT": None,
    "WHISPER_LORA_R": None,
    "WHISPER_LORA_ALPHA": None,
    "WHISPER_LORA_DROPOUT": None,
    "WHISPER_LORA_TARGET_MODULES": None,
    "WHISPER_LORA_USE_DORA": None,
    "WHISPER_LORA_USE_RSLORA": None,
    "WHISPER_LORA_INIT": None,
}


def set_peft_overrides(**kwargs) -> None:
    """Set PEFT/LoRA defaults without using environment variables.

    Accepted keys:
    - WHISPER_USE_PEFT
    - WHISPER_LORA_R
    - WHISPER_LORA_ALPHA
    - WHISPER_LORA_DROPOUT
    - WHISPER_LORA_TARGET_MODULES
    - WHISPER_LORA_USE_DORA
    - WHISPER_LORA_USE_RSLORA
    - WHISPER_LORA_INIT
    """
    for key, value in kwargs.items():
        if key not in _PEFT_OVERRIDES:
            raise KeyError(f"Unknown PEFT override: {key}")
        _PEFT_OVERRIDES[key] = value


def _override(name: str):
    if name in _PEFT_OVERRIDES:
        return _PEFT_OVERRIDES[name]
    return None


def peft_is_enabled() -> bool:
    val = _override("WHISPER_USE_PEFT")
    if val is None:
        val = os.environ.get("WHISPER_USE_PEFT", "0")
    return str(val).strip() in ("1", "true", "True", "YES", "yes")


def _env_int(name: str, default: int) -> int:
    ov = _override(name)
    if ov is not None:
        try:
            return int(str(ov).strip())
        except Exception:
            return int(default)
    try:
        return int(str(os.environ.get(name, default)).strip())
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    ov = _override(name)
    if ov is not None:
        try:
            return float(str(ov).strip())
        except Exception:
            return float(default)
    try:
        return float(str(os.environ.get(name, default)).strip())
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    ov = _override(name)
    if ov is not None:
        val = ov
    else:
        val = os.environ.get(name, None)
    if val is None:
        return bool(default)
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    try:
        return bool(int(s))
    except Exception:
        return bool(default)


def _env_csv(name: str, default_csv: str) -> list[str]:
    ov = _override(name)
    if ov is not None:
        if isinstance(ov, (list, tuple)):
            return [str(p).strip() for p in ov if str(p).strip()]
        s = str(ov or "").strip()
    else:
        s = str(os.environ.get(name, default_csv) or "").strip()
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]


def _env_lora_init(name: str, default: Any):
    ov = _override(name)
    if ov is not None:
        val = ov
    else:
        val = os.environ.get(name, None)
    if val is None or val == "":
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if int(val) == 0:
            return False
        if int(val) == 1:
            return True
    s = str(val).strip()
    if not s:
        return default
    low = s.lower()
    if low in ("1", "true", "yes", "y", "on"):
        return True
    if low in ("0", "false", "no", "n", "off"):
        return False
    return s


def is_peft_checkpoint_dir(path: str) -> bool:
    """Heuristic: PEFT adapter checkpoints contain adapter_config.json."""
    if not path or not os.path.isdir(path):
        return False
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


def load_whisper_student_with_optional_peft(
    *,
    model_dir: Optional[str],
    base_model_id: str,
    generation_config=None,
    want_trainable_adapter: bool = False,
):
    """Load:
    - normal checkpoint dir (full weights) => WhisperForConditionalGeneration.from_pretrained(dir)
    - PEFT checkpoint dir (adapter weights only) => base model + PeftModel.from_pretrained
    - no dir => base model
    """
    from transformers import WhisperForConditionalGeneration

    if model_dir and os.path.isdir(model_dir):
        if is_peft_checkpoint_dir(model_dir):
            # Adapter-only checkpoint
            try:
                from peft import PeftModel
            except Exception as e:
                raise RuntimeError(
                    "Found a PEFT adapter checkpoint but 'peft' is not installed. "
                    "Install it in your env: pip install -U peft"
                ) from e

            print("Loading student from PEFT adapter checkpoint:", model_dir)
            base = WhisperForConditionalGeneration.from_pretrained(base_model_id, generation_config=generation_config)
            model = PeftModel.from_pretrained(base, model_dir, is_trainable=bool(want_trainable_adapter))
            return model

        print("Loading student from full checkpoint:", model_dir)
        return WhisperForConditionalGeneration.from_pretrained(model_dir, generation_config=generation_config)

    print("Loading student from base:", base_model_id)
    return WhisperForConditionalGeneration.from_pretrained(base_model_id, generation_config=generation_config)


def maybe_wrap_with_lora_peft(model):
    """If WHISPER_USE_PEFT=1, wrap model with LoRA using PEFT.

    Env knobs:
    - WHISPER_LORA_R (default 16)
    - WHISPER_LORA_ALPHA (default 32)
    - WHISPER_LORA_DROPOUT (default 0.05)
    - WHISPER_LORA_TARGET_MODULES (default "q_proj,v_proj")
    - WHISPER_LORA_USE_DORA (default 0)
    - WHISPER_LORA_USE_RSLORA (default 0)
    - WHISPER_LORA_INIT (default True; or "pissa"/"olora"/"loftq")
    """
    if not peft_is_enabled():
        return model

    try:
        from peft import LoraConfig, get_peft_model, TaskType
        from peft.peft_model import PeftModel
    except Exception as e:
        raise RuntimeError(
            "WHISPER_USE_PEFT=1 but 'peft' is not installed. "
            "Install it in your env: pip install -U peft"
        ) from e

    if isinstance(model, PeftModel):
        # Already wrapped (e.g., resumed from adapter checkpoint)
        return model

    r = _env_int("WHISPER_LORA_R", 16)
    alpha = _env_int("WHISPER_LORA_ALPHA", 32)
    dropout = _env_float("WHISPER_LORA_DROPOUT", 0.05)
    target_modules = _env_csv("WHISPER_LORA_TARGET_MODULES", "q_proj,v_proj")
    use_dora = _env_bool("WHISPER_LORA_USE_DORA", False)
    use_rslora = _env_bool("WHISPER_LORA_USE_RSLORA", False)
    init_lora_weights = _env_lora_init("WHISPER_LORA_INIT", True)

    cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=target_modules,
        bias="none",
        use_dora=bool(use_dora),
        use_rslora=bool(use_rslora),
        init_lora_weights=init_lora_weights,
    )
    model = get_peft_model(model, cfg)

    # Helpful printout
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def iter_trainable_params(model) -> Iterable:
    """Only params with requires_grad=True."""
    for p in model.parameters():
        if getattr(p, "requires_grad", False):
            yield p
