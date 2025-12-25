"""
Local runnable training script converted from non_collab_Local_Whisper_training_only.ipynb.
Adjust paths below to your local files (no Colab mounts).
"""

import os
import json
import time
import shutil
import hashlib
import gc
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets import Dataset, Audio
from transformers import WhisperProcessor, WhisperModel, WhisperForConditionalGeneration

try:
    from jiwer import wer as jiwer_wer
except Exception:
    jiwer_wer = None

# ----------------------------------
# CONFIG: update these for local paths
# ----------------------------------
MANIFEST_PATH = r"i:\P2GPT_google_drive\My Drive\Record_chunks\pairs_manifest.jsonl"
TRAINED_JSONL_PATH = r"i:\P2GPT_google_drive\My Drive\Record_chunks\trained_files.jsonl"
CHECKPOINT_DIR = r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx"
LOCAL_EPOCH_CACHE_ROOT = r"c:\temp\epoch_cache"  # fast local disk

DELETE_TRAINED_FROM_DRIVE = False  # set True to delete/archive originals after checkpoint
DRIVE_CLEANUP_MODE = "archive"    # "archive" | "delete"
DRIVE_ALLOWED_PREFIX = r"i:\P2GPT_google_drive\My Drive\Record_chunks\\"
DRIVE_ARCHIVE_DIR = r"i:\P2GPT_google_drive\My Drive\Record_chunks\_trained_archive"

TARGET_SR = 16000
MODEL_SIZE = "tiny"
N_SAMPLES_PER_EPOCH = 500

VAL_SIZE = 200
VAL_BATCH_SIZE = 4
VAL_MAX_BATCHES = None
EVAL_EVERY_EPOCH = True
EVAL_LANGUAGE = "en"
EVAL_TASK = "transcribe"
MAX_NEW_TOKENS = 128

BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
NUM_WORKERS = 0  # safer on Windows
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = False

EVAL_NUM_WORKERS = 0
EVAL_PERSISTENT_WORKERS = False
EVAL_PREFETCH_FACTOR = 2

MAX_AUDIO_SECONDS = 29.0
LR = 1e-6
MAX_EPOCHS = 999999

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = device == "cuda"
amp_dtype = torch.float16
use_grad_scaler = use_amp
scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

FULL_ENCODER_CONTEXT_LENGTH = 1500

# ----------------------------------
# helpers
# ----------------------------------

def cleanup_memory(aggressive: bool = False):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            torch.cuda.synchronize()
    if aggressive:
        gc.collect()
        gc.collect()


def ensure_processor_files(target_dir: str, processor: WhisperProcessor):
    """Copy needed processor/tokenizer files into checkpoint dir if missing. Returns list of created file paths."""
    needed = [
        "preprocessor_config.json",
        "feature_extractor.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    created = []
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


def remove_files(paths):
    for p in paths or []:
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass


def read_jsonl(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stable_local_name(src_path: str) -> str:
    h = hashlib.sha1(src_path.encode("utf-8")).hexdigest()[:10]
    base = os.path.basename(src_path)
    return f"{h}_{base}"


def copy_epoch_subset_to_local(selected_rows, epoch_dir: str, show_progress: bool = True):
    os.makedirs(epoch_dir, exist_ok=True)
    it = tqdm(selected_rows, desc=f"Copying -> {epoch_dir}", unit="file") if show_progress else selected_rows
    kept = []
    for row in it:
        src = row["audio_path"]
        if not os.path.exists(src):
            continue
        dst = os.path.join(epoch_dir, stable_local_name(src))
        try:
            if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
                row["audio_path_local"] = dst
                kept.append(row)
                continue
        except OSError:
            pass
        shutil.copy2(src, dst)
        row["audio_path_local"] = dst
        kept.append(row)
    return kept


def cleanup_old_epoch_dirs(root: str, keep_last_k: int = 2):
    if not os.path.exists(root):
        return
    dirs = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.startswith("epoch_"):
            try:
                epoch_num = int(name.split("_")[-1])
                dirs.append((epoch_num, p))
            except Exception:
                continue
    dirs.sort()
    if len(dirs) <= keep_last_k:
        return
    for _, p in dirs[:-keep_last_k]:
        shutil.rmtree(p, ignore_errors=True)


def cleanup_trained_drive_audio(drive_paths, mode: str, allowed_prefix: str, archive_dir: str):
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


def to_mono_float32(audio_obj):
    if hasattr(audio_obj, "get_all_samples"):
        samples = audio_obj.get_all_samples()
        wave = samples.data
        sr = int(samples.sample_rate)
        if wave.ndim == 2:
            wave = wave.mean(dim=0)
        return wave.cpu().numpy().astype(np.float32, copy=False), sr
    if isinstance(audio_obj, dict) and "array" in audio_obj:
        sr = int(audio_obj["sampling_rate"])
        wave = np.asarray(audio_obj["array"])
        if wave.ndim == 2:
            wave = wave.mean(axis=-1)
        return wave.astype(np.float32, copy=False), sr
    raise TypeError(f"Unsupported audio type: {type(audio_obj)}")


def collate_batch(examples):
    waveforms, texts, lengths = [], [], []
    for ex in examples:
        txt = ex.get("raw_transcription")
        if not txt:
            continue
        wave_np, sr = to_mono_float32(ex["audio"])
        if sr != TARGET_SR:
            continue
        dur = wave_np.shape[0] / float(sr)
        if dur > MAX_AUDIO_SECONDS:
            continue
        waveforms.append(wave_np)
        texts.append(txt.lower())
        lengths.append(dur)
    if not waveforms:
        return None
    feats = processor.feature_extractor(
        waveforms,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=TARGET_SR * 30,
    )
    tok = processor.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    return {
        "lengths": torch.tensor(lengths, dtype=torch.float32),
        "input_features": feats.input_features,
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "texts": texts,
    }


def pick_n_ctx_from_batch(lengths_sec: torch.Tensor, max_embed_positions: int):
    max_len = float(lengths_sec.max().item())
    n_ctx = int(round((FULL_ENCODER_CONTEXT_LENGTH / 30.0) * max_len))
    jitter = max(1, min(64, n_ctx // 3))
    n_ctx = n_ctx + int(torch.randint(-jitter, jitter + 1, (1,), device=lengths_sec.device).item())
    n_ctx = max(1, min(n_ctx, max_embed_positions))
    return n_ctx


def masked_hidden_mse(hs_pred: torch.Tensor, hs_tgt: torch.Tensor, attn_mask: torch.Tensor):
    mask = attn_mask.unsqueeze(0).unsqueeze(-1).to(dtype=hs_pred.dtype)
    diff2 = (hs_pred - hs_tgt).pow(2) * mask
    return diff2.sum() / mask.sum().clamp_min(1.0)


def _word_edit_distance(ref_words, hyp_words):
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


def word_error_rate(refs, hyps):
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
def evaluate_distill_loss(loader: DataLoader):
    model_train.eval()
    total = 0.0
    steps = 0
    for i, batch in enumerate(tqdm(loader, desc="Eval distill", leave=False)):
        if batch is None:
            continue
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
def evaluate_wer(loader: DataLoader, gen_model: WhisperForConditionalGeneration, forced_decoder_ids):
    gen_model.model.load_state_dict(model_train.state_dict(), strict=True)
    gen_model.eval()
    refs, hyps = [], []
    for i, batch in enumerate(tqdm(loader, desc="Eval WER", leave=False)):
        if batch is None:
            continue
        if VAL_MAX_BATCHES is not None and i >= VAL_MAX_BATCHES:
            break
        input_features = batch["input_features"].to(device, non_blocking=True)
        ref_texts = batch.get("texts") or processor.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            pred_ids = gen_model.generate(inputs=input_features, forced_decoder_ids=forced_decoder_ids, max_new_tokens=MAX_NEW_TOKENS)
        pred_texts = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        for r, p in zip(ref_texts, pred_texts):
            refs.append((r or "").strip().lower())
            hyps.append((p or "").strip().lower())
    cleanup_memory()
    if jiwer_wer is not None:
        return float(jiwer_wer(refs, hyps))
    return float(word_error_rate(refs, hyps))


def save_checkpoint(epoch_num: int, subset_start_idx: int, subset_count: int):
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
        "scaler": scaler.state_dict() if use_grad_scaler else None,
        "timestamp": time.time(),
        "model_size": MODEL_SIZE,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "n_samples_per_epoch": N_SAMPLES_PER_EPOCH,
    }
    state_path = os.path.join(CHECKPOINT_DIR, f"training_state_epoch_{epoch_num:06d}.pt")
    tmp_state_path = state_path + ".tmp"
    torch.save(state, tmp_state_path)
    os.replace(tmp_state_path, state_path)
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


def build_loader_from_rows(rows_for_epoch, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, persistent_workers=PERSISTENT_WORKERS, prefetch_factor=PREFETCH_FACTOR):
    slim = []
    for r in rows_for_epoch:
        slim.append({
            "audio_path": r["audio_path_local"],
            "raw_transcription": r.get("raw_transcription", ""),
            "audio_path_original": r.get("audio_path_original", r.get("audio_path")),
        })
    ds = Dataset.from_list(slim)
    ds = ds.rename_column("audio_path", "audio")
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    pin_memory = device == "cuda"
    kwargs = dict(batch_size=batch_size, shuffle=False, collate_fn=collate_batch, num_workers=num_workers, pin_memory=pin_memory)
    if num_workers > 0:
        kwargs.update(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor)
    return DataLoader(ds, **kwargs)


def compute_partially_encoder(model: WhisperModel, data: torch.Tensor, n_audio_ctx: int):
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
                layer_outputs = model.encoder._gradient_checkpointing_func(encoder_layer.__call__, hidden_states, None, None, False)
            else:
                layer_outputs = encoder_layer(hidden_states, None, layer_head_mask=None, output_attentions=False)
            hidden_states = layer_outputs[0]
    hidden_states = model.encoder.layer_norm(hidden_states)
    return hidden_states


def train_one_epoch_on_loader(epoch_num: int, loader: DataLoader):
    model_train.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    steps = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch_num} (subset)")
    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue
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
        if use_grad_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            if use_grad_scaler:
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


def select_next_untrained(manifest_rows, trained_set, pointer: int, n: int, holdout_set=None):
    selected = []
    start_idx = pointer
    while pointer < len(manifest_rows) and len(selected) < n:
        r = manifest_rows[pointer]
        ap = r["audio_path_original"]
        if ap not in trained_set and (holdout_set is None or ap not in holdout_set):
            selected.append({"audio_path": ap, "raw_transcription": r.get("raw_transcription", ""), "audio_path_original": ap})
        pointer += 1
    return selected, start_idx, pointer


if __name__ == "__main__":
    print("Device:", device)
    manifest_rows = read_jsonl(MANIFEST_PATH)
    if not manifest_rows:
        raise RuntimeError("Manifest is empty or not found")
    trained_rows = read_jsonl(TRAINED_JSONL_PATH)
    trained_set = {r.get("audio_path") for r in trained_rows if r.get("audio_path")}
    print("Manifest rows:", len(manifest_rows))
    print("Already trained:", len(trained_set))

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

    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{MODEL_SIZE}")
    if latest_ckpt_epoch is not None and latest_model_dir and os.path.isdir(latest_model_dir):
        print(f"Resuming model_train from: {latest_model_dir}")
        model_train = WhisperModel.from_pretrained(latest_model_dir)
    else:
        model_train = WhisperModel.from_pretrained(f"openai/whisper-{MODEL_SIZE}")
    model_train.to(device)
    model_train.train()
    model_base = WhisperModel.from_pretrained(f"openai/whisper-{MODEL_SIZE}")
    model_base.to(device)
    model_base.eval()
    optimizer = torch.optim.Adam(model_train.parameters(), lr=LR)
    if latest_state is not None:
        try:
            if latest_state.get("optimizer") is not None:
                optimizer.load_state_dict(latest_state["optimizer"])
            if use_grad_scaler and latest_state.get("scaler") is not None:
                scaler.load_state_dict(latest_state["scaler"])
        except Exception:
            pass

    holdout_set = set()
    val_loader = None
    gen_model = None
    forced_decoder_ids = None
    for r in manifest_rows:
        r["audio_path_original"] = r.get("audio_path")
    start_pointer = 0
    while start_pointer < len(manifest_rows) and manifest_rows[start_pointer]["audio_path_original"] in trained_set:
        start_pointer += 1
    if VAL_SIZE and VAL_SIZE > 0:
        val_drive_rows = []
        for r in manifest_rows:
            ap = r.get("audio_path")
            if not ap or ap in trained_set or ap in holdout_set:
                continue
            val_drive_rows.append({"audio_path": ap, "raw_transcription": r.get("raw_transcription", ""), "audio_path_original": ap})
            holdout_set.add(ap)
            if len(val_drive_rows) >= int(VAL_SIZE):
                break
        if val_drive_rows:
            local_val_dir = os.path.join(LOCAL_EPOCH_CACHE_ROOT, "_val_cache")
            val_cached = copy_epoch_subset_to_local(val_drive_rows, local_val_dir, show_progress=True)
            if val_cached:
                val_loader = build_loader_from_rows(
                    val_cached,
                    VAL_BATCH_SIZE,
                    num_workers=EVAL_NUM_WORKERS,
                    persistent_workers=EVAL_PERSISTENT_WORKERS,
                    prefetch_factor=EVAL_PREFETCH_FACTOR,
                )
                forced_decoder_ids = processor.get_decoder_prompt_ids(language=EVAL_LANGUAGE, task=EVAL_TASK)
                gen_model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{MODEL_SIZE}")
                gen_model.to(device)
                gen_model.eval()
                print("Validation set ready.")
            else:
                print("Validation cache is empty; skipping validation.")
                val_loader = None
    if start_pointer >= len(manifest_rows):
        print("All files already trained.")
        raise SystemExit(0)

    epoch_num = epoch_num_start
    pointer = start_pointer
    executor = ThreadPoolExecutor(max_workers=1)
    prefetch_future = None
    prefetch_rows = None
    prefetch_pointer_after = None
    prefetch_slice = None

    os.makedirs(LOCAL_EPOCH_CACHE_ROOT, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    while epoch_num < MAX_EPOCHS:
        if prefetch_rows is not None:
            selected = prefetch_rows
            subset_start_idx, pointer_after_current = prefetch_slice
            pointer = prefetch_pointer_after
            prefetch_rows = prefetch_pointer_after = prefetch_slice = prefetch_future = None
        else:
            selected, subset_start_idx, pointer_after_current = select_next_untrained(manifest_rows, trained_set, pointer, N_SAMPLES_PER_EPOCH, holdout_set=holdout_set)
            pointer = pointer_after_current
            if not selected:
                print("No more untrained samples.")
                break
            print(f"\n=== Epoch {epoch_num} | {len(selected)} samples | manifest slice [{subset_start_idx}:{pointer_after_current}] ===")
            epoch_dir = os.path.join(LOCAL_EPOCH_CACHE_ROOT, f"epoch_{epoch_num:06d}")
            selected = copy_epoch_subset_to_local(selected, epoch_dir, show_progress=True)
            if not selected:
                print("All selected samples failed to copy. Stopping.")
                break
        next_rows, next_start_idx, pointer_after_next = select_next_untrained(manifest_rows, trained_set, pointer, N_SAMPLES_PER_EPOCH, holdout_set=holdout_set)
        if next_rows:
            next_epoch_dir = os.path.join(LOCAL_EPOCH_CACHE_ROOT, f"epoch_{(epoch_num + 1):06d}")
            print(f"Prefetching next epoch: epoch_{epoch_num+1:06d} slice [{next_start_idx}:{pointer_after_next}] -> {next_epoch_dir}")
            prefetch_future = executor.submit(copy_epoch_subset_to_local, next_rows, next_epoch_dir, False)
            prefetch_pointer_after = pointer_after_next
            prefetch_slice = (next_start_idx, pointer_after_next)
        else:
            prefetch_future = prefetch_pointer_after = prefetch_slice = None
        loader = build_loader_from_rows(selected)
        avg_loss = train_one_epoch_on_loader(epoch_num, loader)
        print(f"Epoch {epoch_num} avg loss: {avg_loss:.6f}")
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
        model_dir, state_path, ckpt_ok = save_checkpoint(epoch_num, subset_start_idx=subset_start_idx, subset_count=len(selected))
        created_proc_files = ensure_processor_files(model_dir, processor)
        if EVAL_EVERY_EPOCH and val_loader is not None:
            distill_val = wer_val = None
            try:
                distill_val = evaluate_distill_loss(val_loader)
            except Exception as e:
                print("WARNING: distill eval failed", repr(e))
            try:
                wer_val = evaluate_wer(val_loader, gen_model, forced_decoder_ids)
            except Exception as e:
                print("WARNING: WER eval failed", repr(e))
            if distill_val is not None and wer_val is not None:
                print(f"Eval (val) distill_loss: {distill_val:.6f} | WER: {wer_val:.4f}")
            elif distill_val is not None:
                print(f"Eval (val) distill_loss: {distill_val:.6f}")
            elif wer_val is not None:
                print(f"Eval (val) WER: {wer_val:.4f}")
        remove_files(created_proc_files)
        if DELETE_TRAINED_FROM_DRIVE and ckpt_ok:
            drive_paths = [r.get("audio_path_original") for r in selected if r.get("audio_path_original")]
            cleanup_trained_drive_audio(drive_paths, mode=DRIVE_CLEANUP_MODE, allowed_prefix=DRIVE_ALLOWED_PREFIX, archive_dir=DRIVE_ARCHIVE_DIR)
        if prefetch_future is not None:
            try:
                prefetch_rows = prefetch_future.result()
                if not prefetch_rows:
                    print("WARNING: prefetch produced no usable rows.")
                    prefetch_rows = None
            except Exception as e:
                print("WARNING: background prefetch failed:", repr(e))
                prefetch_rows = None
        cleanup_old_epoch_dirs(LOCAL_EPOCH_CACHE_ROOT, keep_last_k=2)
        epoch_num += 1
    print("Training run finished.")
    print("Total trained marked:", len(trained_set))
