r"""
Local runnable training script (Windows spawn-safe) for FUTO ACFT Whisper.

Fixes your crash:
- Removes collate_fn closures (not picklable on Windows spawn)
- Uses top-level collate_fn + worker_init_fn so DataLoader workers can start

Run (PowerShell)
--------------
I:\Whisper-training-env\Scripts\python.exe I:\whisper-acft\Whisper_Futo_finetuned_model_training_only_local_WIN_SAFE.py

"""

from __future__ import annotations

import os
import json
import time
import shutil
import hashlib
import gc
import tempfile
import csv
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# -----------------------------
# CONFIG
# -----------------------------

# If True, force HF/Transformers to run offline (local cache only)
OFFLINE_MODE = False

MANIFEST_PATH = r"i:\P2GPT_google_drive\My Drive\Record_chunks\pairs_manifest_sentence.jsonl"
TRAINED_JSONL_PATH = r"i:\Record_chunks\trained_files.jsonl"
CHECKPOINT_DIR = r"i:\checkpoints_partialctx"
LOCAL_EPOCH_CACHE_ROOT = r"i:\epoch_cache"  # fast local disk
SCORE_CSV_PATH = r"i:\whisper-acft\speaker_sort_scores.csv"  # optional: speaker score CSV

# Hugging Face model + processor
MODEL_ID = "futo-org/acft-whisper-tiny.en"
# Best default: use the matching OpenAI processor for tiny.en
PROCESSOR_ID = "openai/whisper-tiny.en"

# Optional: set a HF cache dir (recommended if your system drive is tight)
HF_CACHE_DIR = r"i:\hf_cache"  # set to None to use default

# Cleanup options (careful)
DELETE_TRAINED_FROM_DRIVE = False
DRIVE_CLEANUP_MODE = "archive"  # "archive" | "delete"
DRIVE_ALLOWED_PREFIX = r"i:\Record_chunks\\"
DRIVE_ARCHIVE_DIR = r"i:\Record_chunks\_trained_archive"

# Validation/testing files management
MOVE_VALIDATION_FILES_TO_TESTING_FOLDER = True
TESTING_FOLDER_PATH = r"i:\Record_chunks\testing_audio_data"

# Audio / training
TARGET_SR = 16000
N_SAMPLES_PER_EPOCH = 5000
VAL_SIZE = 200
VAL_BATCH_SIZE = 16
VAL_MAX_BATCHES = 50  # set to None to run full val
EVAL_EVERY_EPOCH = True
EVAL_LANGUAGE = "en"
EVAL_TASK = "transcribe"
MAX_NEW_TOKENS = 128

BATCH_SIZE = 25
GRAD_ACCUM_STEPS = 1
NUM_WORKERS = 2
PERSISTENT_WORKERS = True

EVAL_NUM_WORKERS = 2
EVAL_PERSISTENT_WORKERS = True

MAX_AUDIO_SECONDS = 29.0
LR = 1e-6
MAX_EPOCHS = 999999

# Distillation context
FULL_ENCODER_CONTEXT_LENGTH = 1500

# Threading
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["DATASETS_AUDIO_USE_TORCHCODEC"] = "false"

# -----------------------------
# Environment switches
# -----------------------------

def _set_env_flag(name: str, value: Optional[str]):
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value

# Keep this BEFORE importing transformers
if OFFLINE_MODE:
    _set_env_flag("TRANSFORMERS_OFFLINE", "1")
    _set_env_flag("HF_HUB_OFFLINE", "1")
else:
    _set_env_flag("TRANSFORMERS_OFFLINE", None)
    _set_env_flag("HF_HUB_OFFLINE", None)

# Sometimes Windows + MKL throws duplicate lib warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if HF_CACHE_DIR:
    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    # These are respected by HF/Transformers
    os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_CACHE_DIR, "datasets"))
# Transformers v5+ treats caching as a huggingface_hub concern; HF_HOME is the supported knob.

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import librosa

try:
    from transformers import WhisperProcessor, WhisperModel, WhisperForConditionalGeneration
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print(f"❌ ERROR: transformers import failed: {e}")
    TRANSFORMERS_AVAILABLE = False

try:
    from jiwer import wer as jiwer_wer
except Exception:
    jiwer_wer = None

# -----------------------------
# Globals used by DataLoader workers
# (must be top-level for Windows spawn)
# -----------------------------

global_processor: Optional[WhisperProcessor] = None


def cleanup_memory(aggressive: bool = False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.synchronize()
    if aggressive:
        gc.collect()
        gc.collect()


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _norm_path(p: str) -> str:
    try:
        return os.path.normcase(os.path.abspath(p))
    except Exception:
        return p


def load_scores_csv(path: str) -> Dict[str, float]:
    """Load speaker scores CSV (file,score,decision,reason)."""
    scores: Dict[str, float] = {}
    if not path or not os.path.isfile(path):
        return scores
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fp = row.get("file")
                if not fp:
                    continue
                fp_norm = _norm_path(fp)
                try:
                    score_val = float(row.get("score"))
                except Exception:
                    continue
                prev = scores.get(fp_norm)
                if prev is None or score_val > prev:
                    scores[fp_norm] = score_val
    except Exception as e:
        print(f"WARNING: failed to read scores CSV {path}: {e}")
    return scores


def reorder_manifest_by_score(manifest_rows: List[dict], score_map: Dict[str, float]) -> Tuple[List[dict], int]:
    """Sort manifest rows descending by score; unknown scores go last."""
    enriched: List[Tuple[Optional[float], int, dict]] = []
    matched = 0
    for idx, r in enumerate(manifest_rows):
        ap = r.get("audio_path") or r.get("audio_path_original")
        score = score_map.get(_norm_path(ap)) if ap else None
        if score is not None:
            matched += 1
        enriched.append((score, idx, r))

    def sort_key(x: Tuple[Optional[float], int, dict]):
        score, idx, _ = x
        # Known scores first (descending). Unknown (None) last.
        if score is None:
            return (1, idx)
        return (0, -score, idx)

    enriched.sort(key=sort_key)
    return [r for _, _, r in enriched], matched


def map_colab_path_to_local(colab_path: str) -> str:
    r"""Convert Colab paths or P2GPT drive paths into your local i:\Record_chunks\... paths."""
    
    if not colab_path:
        return colab_path

    p = colab_path
    if "P2GPT_google_drive" in p:
        p = p.replace("i:\\P2GPT_google_drive\\My Drive\\Record_chunks\\", "i:\\Record_chunks\\")
    elif p.startswith("/content/drive/"):
        p = p.replace("/content/drive/MyDrive/", "i:\\")
        p = p.replace("/", "\\")
        p = re.sub(r"__(\w+)_chunk(\d+)", r"_sent\2", p)
    else:
        p = p.replace("/", "\\")
    return p


def stable_local_name(src_path: str) -> str:
    h = hashlib.sha1(src_path.encode("utf-8")).hexdigest()[:10]
    base = os.path.basename(src_path)
    return f"{h}_{base}"


def cleanup_trained_drive_audio(drive_paths: List[str], mode: str, allowed_prefix: str, archive_dir: str):
    if not drive_paths:
        return
    if mode not in ("archive", "delete"):
        raise ValueError(f"Unknown DRIVE_CLEANUP_MODE: {mode}")
    os.makedirs(archive_dir, exist_ok=True)
    for p in drive_paths:
        if not isinstance(p, str) or not p.startswith(allowed_prefix):
            continue
        if not os.path.exists(p):
            continue
        try:
            if mode == "archive":
                dst = os.path.join(archive_dir, stable_local_name(p))
                if not os.path.exists(dst):
                    shutil.move(p, dst)
            else:
                os.remove(p)
        except Exception:
            pass


def find_latest_checkpoint_epoch(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None, None, None
    model_epochs, state_epochs = {}, {}
    for name in os.listdir(checkpoint_dir):
        p = os.path.join(checkpoint_dir, name)
        if os.path.isdir(p) and name.startswith("model_epoch_"):
            try:
                ep = int(name[len("model_epoch_"):])
                model_epochs[ep] = p
            except Exception:
                pass
        if os.path.isfile(p) and name.startswith("training_state_epoch_") and name.endswith(".pt"):
            try:
                ep_str = name[len("training_state_epoch_"):-len(".pt")]
                ep = int(ep_str)
                state_epochs[ep] = p
            except Exception:
                pass
    if not model_epochs and not state_epochs:
        return None, None, None
    latest_epoch = max(list(model_epochs.keys()) + list(state_epochs.keys()))
    return latest_epoch, model_epochs.get(latest_epoch), state_epochs.get(latest_epoch)


def ensure_processor_files(target_dir: str, processor: WhisperProcessor) -> List[str]:
    """Copy needed processor/tokenizer files into checkpoint dir if missing."""
    needed = [
        "preprocessor_config.json",
        "feature_extractor.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
        "generation_config.json",
    ]
    created: List[str] = []
    os.makedirs(target_dir, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        processor.save_pretrained(tmp)
        for name in needed:
            src = os.path.join(tmp, name)
            dst = os.path.join(target_dir, name)
            if not os.path.isfile(src) or os.path.isfile(dst):
                continue
            try:
                shutil.copy2(src, dst)
                created.append(dst)
            except Exception:
                pass
    return created


def remove_files(paths: List[str] | None):
    for p in paths or []:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass


def _hf_from_pretrained(cls, model_id: str, token: Optional[str], local_files_only: bool, cache_dir: Optional[str]):
    """Compatibility wrapper: token vs use_auth_token across transformers versions."""
    kwargs = {"local_files_only": local_files_only}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    try:
        return cls.from_pretrained(model_id, token=token, **kwargs)
    except TypeError:
        # older versions
        return cls.from_pretrained(model_id, use_auth_token=token, **kwargs)


def load_audio_with_librosa(audio_path: str) -> np.ndarray:
    try:
        waveform, _sr = librosa.load(audio_path, sr=TARGET_SR, mono=True, dtype=np.float32)
        if MAX_AUDIO_SECONDS is not None and MAX_AUDIO_SECONDS > 0:
            max_len = int(MAX_AUDIO_SECONDS * TARGET_SR)
            if waveform.shape[0] > max_len:
                waveform = waveform[:max_len]
        return waveform.astype(np.float32, copy=False)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return np.zeros(TARGET_SR, dtype=np.float32)


class AudioTextDataset(Dataset):
    def __init__(self, rows: List[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r = self.rows[idx]
        audio_path = map_colab_path_to_local(r["audio_path"])
        text = r.get("raw_transcription", "") or ""
        wave = load_audio_with_librosa(audio_path)
        length_sec = float(wave.shape[0]) / float(TARGET_SR)
        return {
            "waveform": wave,
            "text": text,
            "length_sec": length_sec,
        }


# -----------------------------
# Windows spawn-safe collate + worker init
# -----------------------------

def worker_init_fn(worker_id: int):
    """Runs in each DataLoader worker process (Windows spawn)."""
    global global_processor
    if global_processor is None:
        hf_token = os.getenv("HF_TOKEN")
        global_processor = _hf_from_pretrained(
            WhisperProcessor,
            PROCESSOR_ID,
            token=hf_token,
            local_files_only=OFFLINE_MODE,
            cache_dir=HF_CACHE_DIR,
        )


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Top-level collate_fn so it is picklable on Windows spawn."""
    global global_processor
    if global_processor is None:
        # num_workers=0 path
        hf_token = os.getenv("HF_TOKEN")
        global_processor = _hf_from_pretrained(
            WhisperProcessor,
            PROCESSOR_ID,
            token=hf_token,
            local_files_only=OFFLINE_MODE,
            cache_dir=HF_CACHE_DIR,
        )

    waveforms = [item["waveform"] for item in batch]
    texts = [item.get("text", "") for item in batch]
    lengths = [float(item.get("length_sec", len(w) / float(TARGET_SR))) for item, w in zip(batch, waveforms)]

    feats = global_processor.feature_extractor(
        waveforms,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=TARGET_SR * 30,
    )
    tok = global_processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    return {
        "lengths": torch.tensor(lengths, dtype=torch.float32),
        "input_features": feats.input_features,
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "texts": texts,
    }


def build_loader_from_rows(
    rows_for_epoch: List[dict],
    batch_size: int,
    num_workers: int,
    persistent_workers: bool,
) -> DataLoader:
    ds = AudioTextDataset(rows_for_epoch)
    pin_memory = torch.cuda.is_available()

    kwargs: Dict[str, Any] = dict(
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    if num_workers > 0:
        kwargs["worker_init_fn"] = worker_init_fn
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = persistent_workers
        # Explicitly set spawn for clarity (Windows default)
        kwargs["multiprocessing_context"] = "spawn"

    return DataLoader(ds, **kwargs)


# -----------------------------
# Training core
# -----------------------------

def pick_n_ctx_from_batch(lengths_sec: torch.Tensor, max_embed_positions: int) -> int:
    max_len = float(lengths_sec.max().item())
    n_ctx = int(round((FULL_ENCODER_CONTEXT_LENGTH / 30.0) * max_len))
    jitter = max(1, min(64, n_ctx // 3))
    n_ctx = n_ctx + int(torch.randint(-jitter, jitter + 1, (1,), device=lengths_sec.device).item())
    n_ctx = max(1, min(n_ctx, max_embed_positions))
    return n_ctx


def masked_hidden_mse(hs_pred: torch.Tensor, hs_tgt: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    mask = attn_mask.unsqueeze(0).unsqueeze(-1).to(dtype=hs_pred.dtype)
    diff2 = (hs_pred - hs_tgt).pow(2) * mask
    return diff2.sum() / mask.sum().clamp_min(1.0)


def compute_partially_encoder(model: WhisperModel, data: torch.Tensor, n_audio_ctx: int) -> torch.Tensor:
    target_mel_seq_len = 2 * n_audio_ctx
    diffy = target_mel_seq_len - data.shape[2]
    if diffy > 0:
        data = nn.functional.pad(data, [0, diffy, 0, 0, 0, 0], "constant", 0.0)
    elif diffy < 0:
        data = data[:, :, :target_mel_seq_len]

    if n_audio_ctx == FULL_ENCODER_CONTEXT_LENGTH:
        return model.encoder(data).last_hidden_state

    input_embeds = nn.functional.gelu(model.encoder.conv1(data))
    input_embeds = nn.functional.gelu(model.encoder.conv2(input_embeds))
    input_embeds = input_embeds.permute(0, 2, 1)
    embed_pos = model.encoder.embed_positions.weight[: input_embeds.shape[1]]
    hidden_states = input_embeds + embed_pos
    hidden_states = nn.functional.dropout(hidden_states, p=model.encoder.dropout, training=model.encoder.training)

    for encoder_layer in model.encoder.layers:
        to_drop = False
        if model.encoder.training and torch.rand([]) < model.encoder.layerdrop:
            to_drop = True
        if not to_drop:
            if model.encoder.gradient_checkpointing and model.encoder.training:
                layer_outputs = model.encoder._gradient_checkpointing_func(
                    encoder_layer.__call__, hidden_states, None, None, False
                )
            else:
                layer_outputs = encoder_layer(hidden_states, None, layer_head_mask=None, output_attentions=False)
            hidden_states = layer_outputs[0]

    hidden_states = model.encoder.layer_norm(hidden_states)
    return hidden_states


def _word_edit_distance(ref_words: List[str], hyp_words: List[str]) -> int:
    n, m = len(ref_words), len(hyp_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m]


def word_error_rate(refs: List[str], hyps: List[str]) -> float:
    total_edits = 0
    total_words = 0
    for r, h in zip(refs, hyps):
        r = (r or "").strip().lower()
        h = (h or "").strip().lower()
        r_words = [w for w in r.split() if w]
        h_words = [w for w in h.split() if w]
        total_edits += _word_edit_distance(r_words, h_words)
        total_words += len(r_words)
    return float(total_edits) / float(max(1, total_words))


@torch.no_grad()
def evaluate_distill_loss(
    loader: DataLoader,
    model_train: WhisperModel,
    model_base: WhisperModel,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    model_train.eval()
    total = 0.0
    steps = 0
    for i, batch in enumerate(tqdm(loader, desc="Eval distill", leave=False)):
        if VAL_MAX_BATCHES is not None and i >= VAL_MAX_BATCHES:
            break
        input_features = batch["input_features"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        max_embed_positions = model_train.encoder.embed_positions.weight.shape[0]
        n_ctx = pick_n_ctx_from_batch(lengths, max_embed_positions)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            enc_partial = compute_partially_encoder(model_train, input_features, n_ctx)
            out_partial = model_train.decoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                encoder_hidden_states=enc_partial,
                output_hidden_states=True,
            )
            enc_full = compute_partially_encoder(model_base, input_features, FULL_ENCODER_CONTEXT_LENGTH)
            out_full = model_base.decoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                encoder_hidden_states=enc_full,
                output_hidden_states=True,
            )
            hs_p = torch.stack(out_partial.hidden_states, dim=0)
            hs_f = torch.stack(out_full.hidden_states, dim=0)
            loss = masked_hidden_mse(hs_p, hs_f, attn_mask)

        total += float(loss.item())
        steps += 1

    model_train.train()
    cleanup_memory()
    return total / max(1, steps)


@torch.no_grad()
def evaluate_wer(
    loader: DataLoader,
    model_train: WhisperModel,
    gen_model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    forced_decoder_ids,
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> float:
    gen_model.model.load_state_dict(model_train.state_dict(), strict=True)
    gen_model.eval()

    refs: List[str] = []
    hyps: List[str] = []

    for i, batch in enumerate(tqdm(loader, desc="Eval WER", leave=False)):
        if VAL_MAX_BATCHES is not None and i >= VAL_MAX_BATCHES:
            break
        input_features = batch["input_features"].to(device, non_blocking=True)
        ref_texts = batch.get("texts") or processor.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            pred_ids = gen_model.generate(
                inputs=input_features,
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=MAX_NEW_TOKENS,
            )
        pred_texts = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        for r, p in zip(ref_texts, pred_texts):
            refs.append((r or "").strip().lower())
            hyps.append((p or "").strip().lower())

    cleanup_memory()
    if jiwer_wer is not None:
        return float(jiwer_wer(refs, hyps))
    return float(word_error_rate(refs, hyps))


def save_checkpoint(
    epoch_num: int,
    model_train: WhisperModel,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    subset_start_idx: int,
    subset_count: int,
    device: str,
) -> Tuple[str, str, bool]:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    model_dir = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch_num:06d}")
    os.makedirs(model_dir, exist_ok=True)

    model_train.to("cpu").save_pretrained(model_dir)
    model_train.to(device)

    state = {
        "epoch": epoch_num,
        "subset_start_idx": subset_start_idx,
        "subset_count": subset_count,
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "timestamp": time.time(),
        "model_id": MODEL_ID,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "n_samples_per_epoch": N_SAMPLES_PER_EPOCH,
    }

    state_path = os.path.join(CHECKPOINT_DIR, f"training_state_epoch_{epoch_num:06d}.pt")
    tmp_state_path = state_path + ".tmp"
    torch.save(state, tmp_state_path)
    os.replace(tmp_state_path, state_path)

    # Optionally verify checkpoint before deleting data
    verified_ok = True
    if DELETE_TRAINED_FROM_DRIVE:
        try:
            _ = WhisperModel.from_pretrained(model_dir)
            _ = torch.load(state_path, map_location="cpu")
        except Exception:
            verified_ok = False
            print("WARNING: checkpoint verification failed; will NOT delete trained audio for this epoch.")

    cleanup_memory(aggressive=True)
    print("Saved checkpoint:", model_dir)
    return model_dir, state_path, verified_ok


def train_one_epoch_on_loader(
    epoch_num: int,
    loader: DataLoader,
    model_train: WhisperModel,
    model_base: WhisperModel,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: str,
    use_amp: bool,
    amp_dtype: torch.dtype,
):
    model_train.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    steps = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_num} (subset)")
    for batch_idx, batch in enumerate(pbar):
        input_features = batch["input_features"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)

        max_embed_positions = model_train.encoder.embed_positions.weight.shape[0]
        n_ctx = pick_n_ctx_from_batch(lengths, max_embed_positions)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            enc_partial = compute_partially_encoder(model_train, input_features, n_ctx)
            out_partial = model_train.decoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                encoder_hidden_states=enc_partial,
                output_hidden_states=True,
            )
            with torch.no_grad():
                enc_full = compute_partially_encoder(model_base, input_features, FULL_ENCODER_CONTEXT_LENGTH)
                out_full = model_base.decoder(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    encoder_hidden_states=enc_full,
                    output_hidden_states=True,
                )
            hs_p = torch.stack(out_partial.hidden_states, dim=0)
            hs_f = torch.stack(out_full.hidden_states, dim=0)
            loss = masked_hidden_mse(hs_p, hs_f, attn_mask)
            loss = loss / float(GRAD_ACCUM_STEPS)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running += float(loss.item())
        steps += 1
        pbar.set_postfix({"loss": f"{(running / max(1, steps)):.6f}", "n_ctx": int(n_ctx), "bs": int(input_features.shape[0])})

    cleanup_memory()
    return running / max(1, steps)


def select_next_untrained(
    manifest_rows: List[dict],
    trained_set: set,
    pointer: int,
    n: int,
    holdout_set: Optional[set] = None,
) -> Tuple[List[dict], int, int]:
    selected: List[dict] = []
    start_idx = pointer
    while pointer < len(manifest_rows) and len(selected) < n:
        r = manifest_rows[pointer]
        ap = r["audio_path_original"]
        if ap not in trained_set and (holdout_set is None or ap not in holdout_set):
            selected.append({"audio_path": ap, "raw_transcription": r.get("raw_transcription", ""), "audio_path_original": ap})
        pointer += 1
    return selected, start_idx, pointer


def move_validation_files_to_testing_folder(val_rows: List[dict]):
    os.makedirs(TESTING_FOLDER_PATH, exist_ok=True)

    marker = os.path.join(TESTING_FOLDER_PATH, "validation_files_list.json")
    if os.path.exists(marker):
        print("✅ Validation files already moved (found validation_files_list.json). Skipping.")
        return

    moved_count = 0
    failed_count = 0

    print(f"\n🔄 Moving {len(val_rows)} validation files to testing folder...")
    print(f"📁 Destination: {TESTING_FOLDER_PATH}")

    for row in tqdm(val_rows, desc="Moving validation files", unit="file"):
        audio_path = row.get("audio_path", "")
        if not audio_path:
            continue
        local_path = map_colab_path_to_local(audio_path)

        if os.path.exists(local_path):
            try:
                filename = os.path.basename(local_path)
                dest_path = os.path.join(TESTING_FOLDER_PATH, filename)
                shutil.move(local_path, dest_path)
                moved_count += 1

                transcript_json = row.get("transcript_json", "")
                if transcript_json:
                    local_transcript = map_colab_path_to_local(transcript_json)
                    if os.path.exists(local_transcript):
                        transcript_filename = os.path.basename(local_transcript)
                        dest_transcript = os.path.join(TESTING_FOLDER_PATH, transcript_filename)
                        shutil.move(local_transcript, dest_transcript)

            except Exception as e:
                print(f"⚠️  Failed to move {local_path}: {e}")
                failed_count += 1
        else:
            print(f"⚠️  File not found: {local_path}")
            failed_count += 1

    print(f"✅ Successfully moved: {moved_count} files")
    if failed_count > 0:
        print(f"❌ Failed to move: {failed_count} files")

    with open(marker, "w", encoding="utf-8") as f:
        json.dump(val_rows, f, indent=2, ensure_ascii=False)

    print(f"📋 Saved file list to: {marker}")
    print("🎯 Validation files are now isolated from training data!")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    if not TRANSFORMERS_AVAILABLE:
        raise SystemExit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    amp_dtype = torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("Device:", device)
    print("OFFLINE_MODE:", OFFLINE_MODE)
    print("MODEL_ID:", MODEL_ID)
    print("PROCESSOR_ID:", PROCESSOR_ID)

    hf_token = os.getenv("HF_TOKEN")

    # Load processor in main process (also used when num_workers=0)
    processor: WhisperProcessor = _hf_from_pretrained(
        WhisperProcessor,
        PROCESSOR_ID,
        token=hf_token,
        local_files_only=OFFLINE_MODE,
        cache_dir=HF_CACHE_DIR,
    )

    # Set global for single-worker path
    global_processor = processor

    manifest_rows = read_jsonl(MANIFEST_PATH)
    if not manifest_rows:
        raise RuntimeError("Manifest is empty or not found")

    score_map = load_scores_csv(SCORE_CSV_PATH)
    if score_map:
        manifest_rows, matched_scores = reorder_manifest_by_score(manifest_rows, score_map)
        print(f"Reordered manifest by speaker scores: matched {matched_scores} of {len(manifest_rows)} rows")

    trained_rows = read_jsonl(TRAINED_JSONL_PATH)
    trained_set = {r.get("audio_path") for r in trained_rows if r.get("audio_path")}
    print("Manifest rows:", len(manifest_rows))
    print("Already trained:", len(trained_set))

    # Normalise manifest
    for r in manifest_rows:
        r["audio_path_original"] = r.get("audio_path")

    latest_ckpt_epoch, latest_model_dir, latest_state_path = find_latest_checkpoint_epoch(CHECKPOINT_DIR)
    latest_state = None
    if latest_ckpt_epoch is not None and latest_state_path:
        try:
            latest_state = torch.load(latest_state_path, map_location="cpu")
        except Exception:
            latest_state = None

    trained_based_epoch = len(trained_set) // max(1, int(N_SAMPLES_PER_EPOCH))
    ckpt_based_epoch = (latest_ckpt_epoch + 1) if latest_ckpt_epoch is not None else 0
    epoch_num_start = max(trained_based_epoch, ckpt_based_epoch)

    # Resume / start model
    if latest_ckpt_epoch is not None and latest_model_dir and os.path.isdir(latest_model_dir):
        print(f"Resuming model_train from: {latest_model_dir}")
        model_train = _hf_from_pretrained(
            WhisperModel,
            latest_model_dir,
            token=hf_token,
            local_files_only=True,
            cache_dir=HF_CACHE_DIR,
        )
    else:
        model_train = _hf_from_pretrained(
            WhisperModel,
            MODEL_ID,
            token=hf_token,
            local_files_only=OFFLINE_MODE,
            cache_dir=HF_CACHE_DIR,
        )

    model_train.to(device)
    model_train.train()

    # Teacher/reference model stays fixed
    model_base = _hf_from_pretrained(
        WhisperModel,
        MODEL_ID,
        token=hf_token,
        local_files_only=OFFLINE_MODE,
        cache_dir=HF_CACHE_DIR,
    )
    model_base.to(device)
    model_base.eval()

    optimizer = torch.optim.Adam(model_train.parameters(), lr=LR)

    if latest_state is not None:
        try:
            if latest_state.get("optimizer") is not None:
                optimizer.load_state_dict(latest_state["optimizer"])
            if use_amp and latest_state.get("scaler") is not None:
                scaler.load_state_dict(latest_state["scaler"])
        except Exception:
            pass

    # Validation holdout (first VAL_SIZE untrained rows)
    holdout_set = set()
    val_loader: Optional[DataLoader] = None
    gen_model: Optional[WhisperForConditionalGeneration] = None
    forced_decoder_ids = None

    start_pointer = 0
    while start_pointer < len(manifest_rows) and manifest_rows[start_pointer]["audio_path_original"] in trained_set:
        start_pointer += 1

    val_rows: List[dict] = []
    if VAL_SIZE and VAL_SIZE > 0:
        for r in manifest_rows:
            ap = r.get("audio_path")
            if not ap or ap in trained_set or ap in holdout_set:
                continue
            val_rows.append({"audio_path": ap, "raw_transcription": r.get("raw_transcription", ""), "audio_path_original": ap})
            holdout_set.add(ap)
            if len(val_rows) >= int(VAL_SIZE):
                break

        # Report missing validation files
        missing = 0
        for vr in val_rows:
            if not os.path.exists(map_colab_path_to_local(vr["audio_path"])):
                missing += 1
        if missing:
            print(f"⚠️  Validation missing locally: {missing} files")

        if val_rows:
            val_loader = build_loader_from_rows(
                val_rows,
                batch_size=VAL_BATCH_SIZE,
                num_workers=EVAL_NUM_WORKERS,
                persistent_workers=EVAL_PERSISTENT_WORKERS,
            )
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=EVAL_LANGUAGE, task=EVAL_TASK)
            gen_model = _hf_from_pretrained(
                WhisperForConditionalGeneration,
                MODEL_ID,
                token=hf_token,
                local_files_only=OFFLINE_MODE,
                cache_dir=HF_CACHE_DIR,
            )
            gen_model.to(device)
            gen_model.eval()
            print("Validation set ready.")

    if start_pointer >= len(manifest_rows):
        print("All files already trained.")
        raise SystemExit(0)

    epoch_num = epoch_num_start
    pointer = start_pointer

    os.makedirs(LOCAL_EPOCH_CACHE_ROOT, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    while epoch_num < MAX_EPOCHS:
        selected, subset_start_idx, pointer_after_current = select_next_untrained(
            manifest_rows, trained_set, pointer, N_SAMPLES_PER_EPOCH, holdout_set=holdout_set
        )
        pointer = pointer_after_current

        if not selected:
            print("No more untrained samples.")
            break

        print(f"\n=== Epoch {epoch_num} | {len(selected)} samples | manifest slice [{subset_start_idx}:{pointer_after_current}] ===")

        # Build training loader
        train_loader = build_loader_from_rows(
            selected,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            persistent_workers=PERSISTENT_WORKERS,
        )

        avg_loss = train_one_epoch_on_loader(
            epoch_num,
            train_loader,
            model_train,
            model_base,
            optimizer,
            scaler if use_amp else None,
            device,
            use_amp,
            amp_dtype,
        )

        print(f"Epoch {epoch_num} avg loss: {avg_loss:.6f}")

        # Mark trained
        now = time.time()
        trained_append = []
        for r in selected:
            ap = r["audio_path_original"]
            if ap not in trained_set:
                trained_set.add(ap)
                trained_append.append({"audio_path": ap, "epoch": epoch_num, "trained_at": now})
        if trained_append:
            append_jsonl(TRAINED_JSONL_PATH, trained_append)
            print("Appended trained records:", len(trained_append))

        # Save checkpoint
        model_dir, state_path, ckpt_ok = save_checkpoint(
            epoch_num,
            model_train,
            optimizer,
            scaler if use_amp else None,
            subset_start_idx=subset_start_idx,
            subset_count=len(selected),
            device=device,
        )

        # Make checkpoint self-contained
        created_proc_files = ensure_processor_files(model_dir, processor)

        # Evaluate
        if EVAL_EVERY_EPOCH and val_loader is not None and gen_model is not None:
            distill_val = None
            wer_val = None
            try:
                distill_val = evaluate_distill_loss(val_loader, model_train, model_base, device, use_amp, amp_dtype)
            except Exception as e:
                print("WARNING: distill eval failed", repr(e))
            try:
                wer_val = evaluate_wer(val_loader, model_train, gen_model, processor, forced_decoder_ids, device, use_amp, amp_dtype)
            except Exception as e:
                print("WARNING: WER eval failed", repr(e))

            if distill_val is not None and wer_val is not None:
                print(f"Eval (val) distill_loss: {distill_val:.6f} | WER: {wer_val:.4f}")
            elif distill_val is not None:
                print(f"Eval (val) distill_loss: {distill_val:.6f}")
            elif wer_val is not None:
                print(f"Eval (val) WER: {wer_val:.4f}")

            # Move validation files AFTER the first evaluation in this run
            if MOVE_VALIDATION_FILES_TO_TESTING_FOLDER and val_rows:
                # only do it once (file presence check inside function)
                if epoch_num == epoch_num_start:
                    move_validation_files_to_testing_folder(val_rows)

        # Cleanup temporary processor files we added to the checkpoint dir
        remove_files(created_proc_files)

        # Optional: delete/archive audio once checkpoint is verified
        if DELETE_TRAINED_FROM_DRIVE and ckpt_ok:
            drive_paths = [r.get("audio_path_original") for r in selected if r.get("audio_path_original")]
            cleanup_trained_drive_audio(
                drive_paths,
                mode=DRIVE_CLEANUP_MODE,
                allowed_prefix=DRIVE_ALLOWED_PREFIX,
                archive_dir=DRIVE_ARCHIVE_DIR,
            )

        epoch_num += 1

    print("Training run finished.")
    print("Total trained marked:", len(trained_set))
