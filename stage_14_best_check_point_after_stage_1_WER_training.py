r"""evaluate_stage1_best_futo_style_transformers.py

Purpose
- Pick the best Stage-1 (supervised WER) checkpoint to start Stage-2 (ACFT robustness) training.
- Uses Transformers inference but pins decoding to be *deterministic* and “keyboard-like” (greedy, no sampling).

Important reality check
- Transformers DOES NOT expose whisper.cpp's `audio_ctx` partial-encoder behaviour.
  So this script selects the best Stage-1 checkpoint by normal (full 30s) encoder eval.
- To test `audio_ctx` robustness (the *real* FUTO target), evaluate with whisper.cpp `-ac/--audio-context`.

Inputs
- validation_files_list.json (list of {audio_path/raw_transcription} or {path/ref})
- CHECKPOINT_DIR containing model_epoch_XXXXXX folders

Outputs
- checkpoint_eval_stage1_best.json
- best_stage1_checkpoint.json (+ .txt) with the best checkpoint path
"""

import os
import json
import time
import re
import inspect
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import librosa
from tqdm.auto import tqdm

from transformers import WhisperProcessor, WhisperModel, WhisperForConditionalGeneration

try:
    from jiwer import wer as jiwer_wer
except Exception:
    jiwer_wer = None

# -----------------
# CONFIG
# -----------------
TEST_AUDIO_DIR = r"i:\\Record_chunks\\testing_audio_data"
TEST_LIST_JSON = os.path.join(TEST_AUDIO_DIR, "validation_files_list.json")
CHECKPOINT_DIR = r"I:\\P2GPT_google_drive\\My Drive\\checkpoints_partialctx"

# Vocab-aware defaults:
#  - 51864 → English-only tokenizer/model
#  - 51865 → Multilingual tokenizer/model
BASE_MODEL_ID_EN = "futo-org/acft-whisper-tiny.en"
BASE_MODEL_ID_MULTI = "openai/whisper-tiny"
PROCESSOR_ID_EN = "openai/whisper-tiny.en"
PROCESSOR_ID_MULTI = "openai/whisper-tiny"

TARGET_SR = 16000
MAX_NEW_TOKENS = 128
EVAL_LANGUAGE = "en"
EVAL_TASK = "transcribe"

# Keep deterministic decoding (closest to whisper.cpp greedy + no-fallback)
DECODE_KWARGS: Dict[str, Any] = {
    "max_new_tokens": MAX_NEW_TOKENS,
    "num_beams": 1,
    "do_sample": False,
    "temperature": 0.0,
    # If supported by your Transformers version:
    "condition_on_prev_tokens": False,
    # Do NOT pass a temperature tuple (that enables temperature fallback)
}

# Optional: limit number of validation items for quick iteration
EVAL_LIMIT: Optional[int] = None  # e.g. 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------
# Helpers
# -----------------

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9'\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def wer_fallback(refs: List[str], hyps: List[str]) -> float:
    def _edit(a, b):
        n, m = len(a), len(b)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[n][m]

    edits = 0
    words = 0
    for r, h in zip(refs, hyps):
        rw = normalize_text(r).split()
        hw = normalize_text(h).split()
        edits += _edit(rw, hw)
        words += len(rw)
    return float(edits) / float(max(1, words))


def compute_wer(refs: List[str], hyps: List[str]) -> float:
    refs_n = [normalize_text(x) for x in refs]
    hyps_n = [normalize_text(x) for x in hyps]
    if jiwer_wer is not None:
        return float(jiwer_wer(refs_n, hyps_n))
    return float(wer_fallback(refs_n, hyps_n))


def load_audio_strict(audio_path: str) -> np.ndarray:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(audio_path)

    waveform, _sr = librosa.load(audio_path, sr=TARGET_SR, mono=True, dtype=np.float32)

    if waveform is None or len(waveform) < TARGET_SR // 10:
        raise RuntimeError(f"Audio too short/empty after load: {audio_path}")

    rms = float(np.sqrt(np.mean(np.square(waveform))))
    peak = float(np.max(np.abs(waveform)))
    if rms < 1e-4 or peak < 1e-3:
        print(f"⚠️  Near-silent audio? rms={rms:.2e}, peak={peak:.2e} :: {audio_path}")

    return waveform


def resolve_audio_path(manifest_path: str) -> str:
    if not manifest_path:
        raise FileNotFoundError("Empty audio_path")

    p = manifest_path.replace("/", "\\")
    if os.path.exists(p):
        return p

    base = os.path.basename(p)
    direct = os.path.join(TEST_AUDIO_DIR, base)
    if os.path.exists(direct):
        return direct

    stem, ext = os.path.splitext(base)
    dup_hits = sorted(Path(TEST_AUDIO_DIR).glob(stem + "__dup*" + ext))
    if dup_hits:
        return str(dup_hits[0])

    raise FileNotFoundError(
        f"Could not resolve audio. Manifest path: {manifest_path}\n"
        f"Missing basename: {base}\n"
        f"Looked in: {TEST_AUDIO_DIR} (no recursion)"
    )


def find_checkpoints() -> List[Tuple[int, str]]:
    checkpoints: List[Tuple[int, str]] = []
    d = Path(CHECKPOINT_DIR)
    if not d.exists():
        return checkpoints

    for item in d.iterdir():
        if item.is_dir() and item.name.startswith("model_epoch_"):
            try:
                ep = int(item.name.split("_")[-1])
                checkpoints.append((ep, str(item)))
            except Exception:
                pass

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def _choose_ids(vocab_size: int, processors_cache: Dict[str, WhisperProcessor]) -> Tuple[str, WhisperProcessor]:
    if vocab_size == 51864:
        return BASE_MODEL_ID_EN, processors_cache["en"]
    if vocab_size == 51865:
        return BASE_MODEL_ID_MULTI, processors_cache["multi"]
    raise RuntimeError(f"Unsupported vocab_size {vocab_size}; add a mapping for this size.")


def _filter_generate_kwargs(gen: WhisperForConditionalGeneration, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(gen.generate)
    supported = set(sig.parameters.keys())
    out = {}
    for k, v in kwargs.items():
        if k in supported:
            out[k] = v
    return out


def load_checkpoint_as_gen(checkpoint_dir: str, base_id: str) -> WhisperForConditionalGeneration:
    """Robust loader:
    - If the checkpoint directory is a WhisperForConditionalGeneration save, load it directly.
    - Else load WhisperModel and copy weights into a base WhisperForConditionalGeneration.
    """
    try:
        gen = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir)
        return gen
    except Exception:
        # Fall back: checkpoint is WhisperModel
        train = WhisperModel.from_pretrained(checkpoint_dir)
        gen = WhisperForConditionalGeneration.from_pretrained(base_id)
        gen.model.load_state_dict(train.state_dict(), strict=True)
        return gen


@torch.no_grad()
def eval_one_model(
    model_name: str,
    model_dir: str,
    base_id_hint: str,
    processor_hint: WhisperProcessor,
    items: List[dict],
) -> Dict[str, Any]:

    # Load once just to read vocab_size (cheap) so we pick correct tokenizer/base
    model_train = WhisperModel.from_pretrained(model_dir)
    base_id, processor = _choose_ids(model_train.config.vocab_size, processors_cache={
        "en": processors_cache_global["en"],
        "multi": processors_cache_global["multi"],
    })
    del model_train

    gen = load_checkpoint_as_gen(model_dir, base_id)
    gen.to(DEVICE)
    gen.eval()

    forced_decoder_ids = processor.get_decoder_prompt_ids(language=EVAL_LANGUAGE, task=EVAL_TASK)

    # Deterministic, greedy decode
    gen_kwargs = dict(DECODE_KWARGS)
    gen_kwargs["forced_decoder_ids"] = forced_decoder_ids
    gen_kwargs = _filter_generate_kwargs(gen, gen_kwargs)

    refs: List[str] = []
    hyps: List[str] = []
    rows_out: List[dict] = []
    missing: List[dict] = []
    total_time = 0.0

    for it in tqdm(items, desc=f"Eval {model_name}"):
        src_path = it.get("audio_path") or it.get("path") or ""
        ref = (it.get("raw_transcription") or it.get("ref") or "").strip()

        try:
            real_path = resolve_audio_path(src_path)
            audio = load_audio_strict(real_path)
        except Exception as e:
            missing.append({"audio_path": src_path, "error": str(e)})
            continue

        t0 = time.time()
        input_features = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_features.to(DEVICE)

        pred_ids = gen.generate(
            input_features=input_features,
            **gen_kwargs,
        )

        hyp = processor.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()
        dt = time.time() - t0
        total_time += dt

        refs.append(ref)
        hyps.append(hyp)

        rows_out.append({
            "file": os.path.basename(real_path),
            "path": real_path,
            "ref": ref,
            "hyp": hyp,
            "time_s": dt,
        })

    if missing:
        print(f"\n⚠️  {model_name}: skipped {len(missing)} items (missing/unloadable audio). Showing first 15:")
        for m in missing[:15]:
            print(f"  - {m['audio_path']}  |  {m['error']}")
        print()

    if not rows_out:
        return {
            "model": model_name,
            "model_dir": model_dir,
            "wer": None,
            "avg_time_s": None,
            "n": 0,
            "missing": missing,
            "samples": [],
        }

    wer = compute_wer(refs, hyps)

    return {
        "model": model_name,
        "model_dir": model_dir,
        "wer": float(wer),
        "avg_time_s": float(total_time / max(1, len(rows_out))),
        "n": len(rows_out),
        "missing": missing,
        "samples": rows_out,
    }


processors_cache_global: Dict[str, WhisperProcessor] = {}


def main():
    print(f"Device: {DEVICE}")
    print(f"TEST_AUDIO_DIR: {TEST_AUDIO_DIR}")
    print(f"TEST_LIST_JSON: {TEST_LIST_JSON}")
    print(f"CHECKPOINT_DIR: {CHECKPOINT_DIR}")

    if not os.path.isfile(TEST_LIST_JSON):
        raise FileNotFoundError(f"Missing {TEST_LIST_JSON}")

    with open(TEST_LIST_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not items:
        raise RuntimeError("validation_files_list.json is empty")

    if EVAL_LIMIT is not None:
        items = items[: int(EVAL_LIMIT)]
        print(f"EVAL_LIMIT active: {len(items)} items")

    processors_cache_global["en"] = WhisperProcessor.from_pretrained(PROCESSOR_ID_EN)
    processors_cache_global["multi"] = WhisperProcessor.from_pretrained(PROCESSOR_ID_MULTI)

    checkpoints = find_checkpoints()
    if not checkpoints:
        raise RuntimeError(f"No checkpoints found in {CHECKPOINT_DIR}")

    results: List[Dict[str, Any]] = []

    # Evaluate BASE (optional but useful)
    base_dir = BASE_MODEL_ID_EN
    results.append(eval_one_model("BASE", base_dir, BASE_MODEL_ID_EN, processors_cache_global["en"], items))

    # Checkpoints
    for ep, ckpt_dir in checkpoints:
        r = eval_one_model(f"epoch_{ep:06d}", ckpt_dir, BASE_MODEL_ID_EN, processors_cache_global["en"], items)
        r["epoch"] = int(ep)
        results.append(r)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    out_path = os.path.join(os.getcwd(), "checkpoint_eval_stage1_best.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Pick best epoch (lowest WER, excluding BASE)
    epoch_results = [r for r in results if r.get("model", "").startswith("epoch_") and r.get("wer") is not None]
    best = min(epoch_results, key=lambda x: x["wer"]) if epoch_results else None

    print("\nSummary (WER lower is better):")
    for r in results:
        print(f"  {r['model']:<14}  WER={r['wer']}  avg_time={r['avg_time_s']}  n={r['n']}")

    if best is None:
        print("\nNo valid checkpoint results to select best from.")
        print(f"Saved: {out_path}")
        return

    best_out = {
        "best_model": best["model"],
        "best_epoch": best.get("epoch"),
        "best_wer": best["wer"],
        "best_model_dir": best["model_dir"],
        "decode_kwargs": DECODE_KWARGS,
        "note": "Use best_model_dir as starting weights for Stage-2 ACFT training (model_train + model_ref anchor).",
    }

    best_json = os.path.join(os.getcwd(), "best_stage1_checkpoint.json")
    best_txt = os.path.join(os.getcwd(), "best_stage1_checkpoint.txt")

    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(best_out, f, indent=2, ensure_ascii=False)

    with open(best_txt, "w", encoding="utf-8") as f:
        f.write(best_out["best_model_dir"] + "\n")

    print("\n=== BEST Stage-1 checkpoint ===")
    print(f"  {best_out['best_model']}  WER={best_out['best_wer']}")
    print(f"  dir: {best_out['best_model_dir']}")
    print(f"\nSaved:\n  {out_path}\n  {best_json}\n  {best_txt}")


if __name__ == "__main__":
    main()
