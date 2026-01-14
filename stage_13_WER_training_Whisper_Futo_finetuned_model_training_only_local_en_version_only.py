# ============================================
# Whisper partial-context training — BETTER no-cache mode
# Improvements vs your current script:
# 1) Works the same way (scan manifest forward, skip trained/missing/empty, fill N_SAMPLES_PER_EPOCH)
# 2) Prefetch works EVEN when USE_LOCAL_CACHE=False (prefetches next epoch's selection)
# 3) Optional fast pre-validation using soundfile.info (sample-rate + duration) so you don't “lose” samples later
# 4) If LAMBDA_ACFT == 0, we DO NOT run the reference model and DO NOT request hidden_states (big speed win)
# 5) pick_n_ctx can be set to a *bucketed minimum* that still covers the audio in the batch
# ============================================

# ============================================
# CELL 1/9 — Install + imports
# ============================================

#!pip -q install -U "transformers>=4.38" datasets accelerate soundfile tqdm

import os, json, time, shutil, hashlib, gc, math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import soundfile as sf
from datasets import Dataset

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    GenerationConfig,
)

# ============================================
# CELL 2/9 — Settings
# ============================================

# --- Data ---
MANIFEST_PATH = "i:/Record_chunks/pairs_manifest_local.jsonl"
TRAINED_JSONL_PATH = "i:/Record_chunks/trained_stage1.jsonl"

CHECKPOINT_DIR = "i:/checkpoints_partialctx"

# If you're running on the same machine as the audio (local disk), caching copies is unnecessary.
USE_LOCAL_CACHE = False
LOCAL_EPOCH_CACHE_ROOT = "i:/epoch_cache"  # only used when USE_LOCAL_CACHE=True

# Prefetch works in BOTH modes (cache and no-cache)
PREFETCH_THREADS = 1

# Optional: pre-validate candidate audio using sf.info (fast header read)
# This prevents selecting files that will later be dropped by collate_batch (wrong sr, too long, etc.)
VALIDATE_AUDIO_IN_SELECTOR = True
VALIDATE_THREADS = 8  # lower if you hit "too many open files" on your OS
VALIDATE_BLOCK = 256  # validate candidates in blocks

# --- Model ---
FUTO_MODEL_ID = "futo-org/acft-whisper-tiny.en"
PROCESSOR_ID = "openai/whisper-tiny.en"  # must match the base family

TARGET_SR = 16000
N_SAMPLES_PER_EPOCH = 5000
MAX_EPOCHS = 999999

# --- Training knobs ---
BATCH_SIZE = 32
GRAD_ACCUM_STEPS = 2
LR = 5e-6
MAX_AUDIO_SECONDS = 29.0  # we pad features to 30s; this filters absurdly long chunks

# Loss weights
LAMBDA_ACFT = 0.00        # robustness term
LAMBDA_CE = 1.00          # ASR term

# DataLoader knobs
NUM_WORKERS = 0           # Windows: keep 0. Colab/Linux: 2–4 can speed up.

# Cleanup aggressiveness
CLEANUP_EVERY_N_STEPS = 20
KEEP_LAST_LOCAL_EPOCH_DIRS = 2

# Copying (only used when USE_LOCAL_CACHE=True)
COPY_THREADS = 2

# --- Optional: delete/archive trained Drive audio ---
DELETE_TRAINED_FROM_DRIVE = False
DRIVE_CLEANUP_MODE = "delete"  # "archive" | "delete"
DRIVE_ALLOWED_PREFIX = "/content/drive/MyDrive/Record_chunks/"
DRIVE_ARCHIVE_DIR = "/content/drive/MyDrive/Record_chunks/_trained_archive"

# --- Speed/stability knobs ---
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    torch.backends.cudnn.benchmark = True


device = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = (device == "cuda")
amp_dtype = torch.float16
use_grad_scaler = (device == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("WARNING: CPU training will be very slow.")


def cleanup_memory(tag: str = ""):
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
    if tag and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserv = torch.cuda.memory_reserved() / (1024**3)
        print(f"[mem] {tag} | cuda alloc={alloc:.2f}GB reserved={reserv:.2f}GB")


# ============================================
# CELL 3/9 — Manifest + trained tracking + selection
# ============================================

if USE_LOCAL_CACHE:
    os.makedirs(LOCAL_EPOCH_CACHE_ROOT, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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


def cleanup_old_epoch_dirs(root: str, keep_last_k: int = 2):
    if not os.path.exists(root):
        return
    dirs = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.startswith("epoch_"):
            try:
                ep = int(name.split("_")[-1])
                dirs.append((ep, p))
            except Exception:
                continue
    dirs.sort()
    if len(dirs) <= keep_last_k:
        return
    for _, p in dirs[:-keep_last_k]:
        shutil.rmtree(p, ignore_errors=True)


def load_trained_set(path: str) -> set:
    rows = read_jsonl(path)
    return {r.get("audio_path") for r in rows if r.get("audio_path")}


# Load manifest (slim)
manifest_rows = []
with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        ap = obj.get("audio_path")
        if not ap:
            continue
        manifest_rows.append({
            "audio_path": ap,
            "raw_transcription": obj.get("raw_transcription", ""),
        })

trained_set = load_trained_set(TRAINED_JSONL_PATH)
print("Manifest rows:", len(manifest_rows))
print("Already trained:", len(trained_set))


def audio_header_ok(path: str) -> bool:
    """Fast validation: checks sample rate + duration via header.
    Returns True if path looks usable. (Does NOT decode the whole file.)
    """
    try:
        info = sf.info(path)
        if info.samplerate != TARGET_SR:
            return False
        dur = float(info.frames) / float(info.samplerate)
        if dur <= 0.0 or dur > MAX_AUDIO_SECONDS:
            return False
        return True
    except Exception:
        return False


def _copy_one(src: str, dst: str):
    try:
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            return True, "skipped"
        shutil.copy2(src, dst)
        return True, "copied"
    except Exception:
        return False, "failed"


def select_epoch_rows(pointer: int, trained_set: set, need: int, manifest_rows: list):
    """No-cache selection.

    Critical improvement: (optional) validate sr+duration in selector so you really get ~need usable samples.
    """

    kept = []
    scanned = 0
    stats = {
        "kept": 0,
        "scanned": 0,
        "trained_skipped": 0,
        "empty_transcription": 0,
        "missing": 0,
        "bad_header": 0,
    }

    p = pointer

    # We validate in blocks to keep threading simple.
    while p < len(manifest_rows) and len(kept) < need:
        block = []

        while p < len(manifest_rows) and len(block) < VALIDATE_BLOCK and len(kept) + len(block) < need:
            r = manifest_rows[p]
            p += 1
            scanned += 1

            ap = r.get("audio_path")
            txt = (r.get("raw_transcription") or "").strip()

            if not ap:
                continue
            if ap in trained_set:
                stats["trained_skipped"] += 1
                continue
            if not txt:
                stats["empty_transcription"] += 1
                continue
            if not os.path.exists(ap):
                stats["missing"] += 1
                continue

            block.append({"audio_path": ap, "raw_transcription": txt})

        if not block:
            continue

        if not VALIDATE_AUDIO_IN_SELECTOR:
            kept.extend(block)
            continue

        # Validate headers in parallel
        def vtask(row):
            return row, audio_header_ok(row["audio_path"])

        with ThreadPoolExecutor(max_workers=max(1, int(VALIDATE_THREADS))) as ex:
            for row, ok in ex.map(vtask, block):
                if ok:
                    kept.append(row)
                    if len(kept) >= need:
                        break
                else:
                    stats["bad_header"] += 1

    stats["kept"] = len(kept)
    stats["scanned"] = scanned
    return kept, p, stats


def prepare_epoch_cache(pointer: int, trained_set: set, need: int, manifest_rows: list, epoch_dir: str):
    """Cache mode: copy selected audio into epoch_dir."""

    os.makedirs(epoch_dir, exist_ok=True)

    kept = []
    scanned = 0
    stats = {"copied": 0, "skipped": 0, "missing": 0, "failed": 0, "kept": 0}

    candidates = []
    p = pointer

    while p < len(manifest_rows) and len(candidates) < (need * 2) and len(kept) + len(candidates) < need:
        r = manifest_rows[p]
        scanned += 1
        ap = r.get("audio_path")
        txt = (r.get("raw_transcription") or "").strip()
        p += 1

        if not ap:
            continue
        if ap in trained_set:
            continue
        if not txt:
            continue
        if not os.path.exists(ap):
            stats["missing"] += 1
            continue
        if VALIDATE_AUDIO_IN_SELECTOR and not audio_header_ok(ap):
            continue

        candidates.append({"audio_path": ap, "raw_transcription": txt})

    def task(row):
        src = row["audio_path"]
        dst = os.path.join(epoch_dir, stable_local_name(src))
        ok, st = _copy_one(src, dst)
        if ok:
            row2 = dict(row)
            row2["audio_path_local"] = dst
            return True, st, row2
        return False, st, None

    if candidates:
        with ThreadPoolExecutor(max_workers=COPY_THREADS) as ex:
            for ok, st, row2 in ex.map(task, candidates):
                if st in stats:
                    stats[st] += 1
                else:
                    stats["skipped"] += 1
                if ok and row2 is not None and os.path.exists(row2["audio_path_local"]):
                    kept.append(row2)
                    if len(kept) >= need:
                        break

    stats["kept"] = len(kept)
    meta = {"copy_stats": stats, "scanned_lines": scanned}
    return kept, p, meta


# ============================================
# CELL 4/9 — Processor + audio decode + collate
# ============================================

processor = WhisperProcessor.from_pretrained(PROCESSOR_ID)

N_SAMPLES_30S = processor.feature_extractor.n_samples
NB_MAX_FRAMES = processor.feature_extractor.nb_max_frames

PAD_ID = processor.tokenizer.pad_token_id
DECODER_START_ID = None


def decode_mono_16k(path: str):
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if sr != TARGET_SR:
            return None, None
        if wav.ndim == 2:
            wav = wav.mean(axis=-1)
        return wav, sr
    except Exception:
        return None, None


def pad_or_trim_to_30s(wav: np.ndarray):
    n = wav.shape[0]
    if n == N_SAMPLES_30S:
        return wav
    if n > N_SAMPLES_30S:
        return wav[:N_SAMPLES_30S]
    out = np.zeros((N_SAMPLES_30S,), dtype=np.float32)
    out[:n] = wav
    return out


def shift_tokens_right(labels: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    shifted = shifted.masked_fill(shifted == -100, pad_token_id)
    return shifted


def collate_batch(examples):
    waveforms_30s = []
    lengths_sec = []
    texts = []

    for ex in examples:
        txt = (ex.get("raw_transcription") or "").strip()
        if not txt:
            continue

        ap = ex.get("audio")
        if not ap:
            continue

        wav, sr = decode_mono_16k(ap)
        if wav is None:
            continue

        dur = float(wav.shape[0]) / float(sr)
        if dur > MAX_AUDIO_SECONDS:
            continue

        lengths_sec.append(dur)
        waveforms_30s.append(pad_or_trim_to_30s(wav))
        texts.append(txt)

    if not waveforms_30s:
        return None

    feats = processor.feature_extractor(
        waveforms_30s,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    )

    if feats.input_features.shape[-1] != NB_MAX_FRAMES:
        raise RuntimeError(f"Bad mel length: {feats.input_features.shape} (expected last dim {NB_MAX_FRAMES})")

    tok = processor.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=448,
    )

    labels = tok["input_ids"].clone()
    labels[labels == PAD_ID] = -100

    return {
        "lengths": torch.tensor(lengths_sec, dtype=torch.float32),
        "input_features": feats.input_features,
        "labels": labels,
        "attention_mask": tok["attention_mask"],
    }


# ============================================
# CELL 5/9 — Models + optimizer
# ============================================

try:
    gen_config = GenerationConfig.from_pretrained(FUTO_MODEL_ID)
except Exception:
    gen_config = None


def find_latest_checkpoint_epoch(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None, None, None

    model_epochs = {}
    state_epochs = {}

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
                ep = int(name[len("training_state_epoch_"):-3])
                state_epochs[ep] = p
            except Exception:
                pass

    if not model_epochs and not state_epochs:
        return None, None, None

    latest_epoch = max(list(model_epochs.keys()) + list(state_epochs.keys()))
    return latest_epoch, model_epochs.get(latest_epoch), state_epochs.get(latest_epoch)


def load_student_from_checkpoint_or_base(model_dir: str | None):
    if model_dir and os.path.isdir(model_dir):
        print("Loading student from checkpoint:", model_dir)
        return WhisperForConditionalGeneration.from_pretrained(model_dir, generation_config=gen_config)
    print("Loading student from base:", FUTO_MODEL_ID)
    return WhisperForConditionalGeneration.from_pretrained(FUTO_MODEL_ID, generation_config=gen_config)


latest_ckpt_epoch, latest_model_dir, latest_state_path = find_latest_checkpoint_epoch(CHECKPOINT_DIR)
latest_state = None
if latest_state_path:
    try:
        latest_state = torch.load(latest_state_path, map_location="cpu")
        print("Found training state:", latest_state_path)
    except Exception as e:
        print("WARNING: cannot read state file, weights-only resume.", repr(e))
        latest_state = None

trained_based_epoch = len(trained_set) // max(1, int(N_SAMPLES_PER_EPOCH))
ckpt_based_epoch = (latest_ckpt_epoch + 1) if latest_ckpt_epoch is not None else 0
epoch_num_start = max(trained_based_epoch, ckpt_based_epoch)
print(f"Auto epoch start: {epoch_num_start} (trained_based={trained_based_epoch}, ckpt_based={ckpt_based_epoch})")

model_train = load_student_from_checkpoint_or_base(latest_model_dir)
model_train.to(device)
model_train.train()

DECODER_START_ID = int(model_train.config.decoder_start_token_id)

# Only load reference model if ACFT is enabled
want_acft = (LAMBDA_ACFT > 0.0)
if want_acft:
    model_ref = WhisperForConditionalGeneration.from_pretrained(FUTO_MODEL_ID, generation_config=gen_config)
    model_ref.to(device)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)
else:
    model_ref = None

optimizer = torch.optim.AdamW(model_train.parameters(), lr=LR)

if latest_state is not None:
    try:
        if latest_state.get("optimizer") is not None:
            optimizer.load_state_dict(latest_state["optimizer"])
            print("Resumed optimizer state.")
        if use_grad_scaler and latest_state.get("scaler") is not None:
            scaler.load_state_dict(latest_state["scaler"])
            print("Resumed GradScaler state.")
    except Exception as e:
        print("WARNING: failed to resume optimizer/scaler:", repr(e))

cleanup_memory("after model load")


# ============================================
# CELL 6/9 — Partial encoder + losses
# ============================================

FULL_ENCODER_CONTEXT_LENGTH = int(model_train.config.max_source_positions)  # 1500


def compute_partially_encoder(whisper_model, input_features: torch.Tensor, n_audio_ctx: int):
    target_mel_seq_len = 2 * int(n_audio_ctx)
    diff = target_mel_seq_len - input_features.shape[2]

    if diff > 0:
        input_features = F.pad(input_features, (0, diff, 0, 0, 0, 0), "constant", 0.0)
    elif diff < 0:
        input_features = input_features[:, :, :target_mel_seq_len]

    if n_audio_ctx == FULL_ENCODER_CONTEXT_LENGTH:
        return whisper_model.encoder(input_features).last_hidden_state

    enc = whisper_model.encoder

    x = F.gelu(enc.conv1(input_features))
    x = F.gelu(enc.conv2(x))
    x = x.permute(0, 2, 1)

    pos = enc.embed_positions.weight[: x.shape[1]]
    hs = x + pos
    hs = F.dropout(hs, p=enc.dropout, training=enc.training)

    for layer in enc.layers:
        drop = False
        if enc.training and torch.rand(()) < enc.layerdrop:
            drop = True
        if not drop:
            if enc.gradient_checkpointing and enc.training:
                out = enc._gradient_checkpointing_func(layer.__call__, hs, None, None, False)
            else:
                out = layer(hs, None, layer_head_mask=None, output_attentions=False)
            hs = out[0]

    hs = enc.layer_norm(hs)
    return hs


# IMPORTANT: This version NEVER truncates real audio information.
# It picks the smallest context bucket that can cover the longest audio in the batch.
CTX_BUCKETS = [256, 384, 512, 768, 1024, 1500]


def pick_n_ctx_from_batch(lengths_sec: torch.Tensor, max_embed_positions: int):
    # Whisper encoder has 1500 time steps for 30s -> 50 steps/sec
    max_dur = float(lengths_sec.max().item()) if lengths_sec.numel() else 30.0
    required = int(math.ceil(max_dur * 50.0))
    required = max(1, min(required, max_embed_positions))

    for b in CTX_BUCKETS:
        if b >= required:
            return int(min(b, max_embed_positions))
    return int(min(max_embed_positions, FULL_ENCODER_CONTEXT_LENGTH))


def masked_hidden_mse(hs_pred: torch.Tensor, hs_tgt: torch.Tensor, attn_mask: torch.Tensor):
    mask = attn_mask.unsqueeze(0).unsqueeze(-1).to(dtype=hs_pred.dtype)
    diff2 = (hs_pred - hs_tgt).pow(2) * mask
    return diff2.sum() / mask.sum().clamp_min(1.0)


def get_lm_head(model: WhisperForConditionalGeneration):
    if hasattr(model, "proj_out"):
        return model.proj_out
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise AttributeError("No lm head found (expected proj_out or lm_head)")


# ============================================
# CELL 7/9 — Checkpointing
# ============================================


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
        "model_id": FUTO_MODEL_ID,
        "processor_id": PROCESSOR_ID,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM_STEPS,
        "n_samples_per_epoch": N_SAMPLES_PER_EPOCH,
        "lambda_acft": LAMBDA_ACFT,
        "lambda_ce": LAMBDA_CE,
        "use_local_cache": USE_LOCAL_CACHE,
        "validate_audio_in_selector": VALIDATE_AUDIO_IN_SELECTOR,
    }

    state_path = os.path.join(CHECKPOINT_DIR, f"training_state_epoch_{epoch_num:06d}.pt")
    tmp_state_path = state_path + ".tmp"
    torch.save(state, tmp_state_path)
    os.replace(tmp_state_path, state_path)

    print("Saved checkpoint:", model_dir)
    cleanup_memory(f"after save checkpoint {epoch_num}")
    return model_dir, state_path


# ============================================
# CELL 8/9 — DataLoader + training loop
# ============================================


def build_loader_from_rows(rows_for_epoch):
    if USE_LOCAL_CACHE:
        audio_paths = [r["audio_path_local"] for r in rows_for_epoch]
    else:
        audio_paths = [r["audio_path"] for r in rows_for_epoch]

    slim = [{"audio": ap, "raw_transcription": r.get("raw_transcription", "")} for ap, r in zip(audio_paths, rows_for_epoch)]

    ds = Dataset.from_list(slim)

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,  # better training (within-epoch)
        collate_fn=collate_batch,
        num_workers=NUM_WORKERS,
        pin_memory=(device == "cuda"),
    )

    return loader


def train_one_epoch(epoch_num: int, loader: DataLoader):
    model_train.train()
    optimizer.zero_grad(set_to_none=True)

    lm_head = get_lm_head(model_train)

    running = 0.0
    steps = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch_num}")

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue

        input_features = batch["input_features"].to(device, non_blocking=True)
        lengths = batch["lengths"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)

        decoder_input_ids = shift_tokens_right(labels, pad_token_id=PAD_ID, decoder_start_token_id=DECODER_START_ID)

        max_embed_positions = model_train.model.encoder.embed_positions.weight.shape[0]
        n_ctx = pick_n_ctx_from_batch(lengths, max_embed_positions)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            # Student forward
            enc_partial = compute_partially_encoder(model_train.model, input_features, n_ctx)

            dec_partial = model_train.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=attn_mask,
                encoder_hidden_states=enc_partial,
                output_hidden_states=want_acft,  # ONLY compute hidden_states if ACFT enabled
                use_cache=False,
            )

            logits = lm_head(dec_partial.last_hidden_state)
            loss_ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )

            if want_acft:
                with torch.no_grad():
                    enc_full = compute_partially_encoder(model_ref.model, input_features, FULL_ENCODER_CONTEXT_LENGTH)
                    dec_full = model_ref.model.decoder(
                        input_ids=decoder_input_ids,
                        attention_mask=attn_mask,
                        encoder_hidden_states=enc_full,
                        output_hidden_states=True,
                        use_cache=False,
                    )

                hs_p = torch.stack(dec_partial.hidden_states, dim=0)
                hs_f = torch.stack(dec_full.hidden_states, dim=0)
                loss_acft = masked_hidden_mse(hs_p, hs_f, attn_mask)
            else:
                loss_acft = torch.zeros((), device=device)

            loss = (LAMBDA_CE * loss_ce) + (LAMBDA_ACFT * loss_acft)
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

        pbar.set_postfix({
            "loss": f"{(running / max(1, steps)):.6f}",
            "ce": f"{loss_ce.item():.4f}",
            "acft": f"{loss_acft.item():.4f}" if want_acft else "off",
            "n_ctx": int(n_ctx),
            "bs": int(input_features.shape[0]),
        })

        # Cleanup
        del batch
        del input_features, lengths, labels, attn_mask, decoder_input_ids
        del enc_partial, dec_partial, logits, loss_ce, loss_acft, loss
        if want_acft:
            del enc_full, dec_full, hs_p, hs_f

        if (batch_idx + 1) % CLEANUP_EVERY_N_STEPS == 0:
            cleanup_memory(f"epoch {epoch_num} step {batch_idx+1}")

    cleanup_memory(f"end epoch {epoch_num}")
    return running / max(1, steps)


# ============================================
# CELL 9/9 — Main loop with async prefetch (works in BOTH modes)
# ============================================


def cleanup_trained_drive_audio(drive_paths, mode: str, allowed_prefix: str, archive_dir: str):
    if not drive_paths:
        return
    if mode not in ("archive", "delete"):
        raise ValueError(f"Unknown DRIVE_CLEANUP_MODE: {mode}")
    os.makedirs(archive_dir, exist_ok=True)

    moved = deleted = missing = skipped = 0
    for p in drive_paths:
        if not isinstance(p, str) or not p.startswith(allowed_prefix):
            skipped += 1
            continue
        if not os.path.exists(p):
            missing += 1
            continue
        try:
            if mode == "archive":
                dst = os.path.join(archive_dir, stable_local_name(p))
                if os.path.exists(dst):
                    skipped += 1
                    continue
                shutil.move(p, dst)
                moved += 1
            else:
                os.remove(p)
                deleted += 1
        except Exception:
            skipped += 1

    print(f"Drive cleanup done (mode={mode}). moved={moved}, deleted={deleted}, missing={missing}, skipped={skipped}")


# Start pointer: first untrained in manifest (best effort)
pointer = 0
while pointer < len(manifest_rows) and manifest_rows[pointer]["audio_path"] in trained_set:
    pointer += 1

if pointer >= len(manifest_rows):
    print("All files are trained.")
    raise SystemExit(0)

print("Starting pointer:", pointer)

epoch = epoch_num_start

prefetch_exec = ThreadPoolExecutor(max_workers=max(1, int(PREFETCH_THREADS)))
prefetch_future = None

try:
    while epoch < MAX_EPOCHS:
        if USE_LOCAL_CACHE:
            cleanup_old_epoch_dirs(LOCAL_EPOCH_CACHE_ROOT, keep_last_k=KEEP_LAST_LOCAL_EPOCH_DIRS)

        # Consume prefetched epoch if ready
        if prefetch_future is not None and prefetch_future.done():
            current_rows, new_pointer, meta = prefetch_future.result()
            pointer = new_pointer
            prefetch_future = None
            print(f"\nEpoch {epoch} ready: kept={len(current_rows)} | {meta}")
        else:
            # Build synchronously
            if USE_LOCAL_CACHE:
                epoch_dir = os.path.join(LOCAL_EPOCH_CACHE_ROOT, f"epoch_{epoch:06d}")
                print(f"\nPreparing epoch {epoch} cache -> {epoch_dir}")
                current_rows, pointer, meta = prepare_epoch_cache(
                    pointer=pointer,
                    trained_set=trained_set,
                    need=N_SAMPLES_PER_EPOCH,
                    manifest_rows=manifest_rows,
                    epoch_dir=epoch_dir,
                )
                print(f"Epoch {epoch} cache ready: kept={len(current_rows)} | {meta}")
            else:
                print(f"\nSelecting epoch {epoch} rows (no cache) ...")
                current_rows, pointer, meta = select_epoch_rows(
                    pointer=pointer,
                    trained_set=trained_set,
                    need=N_SAMPLES_PER_EPOCH,
                    manifest_rows=manifest_rows,
                )
                print(f"Epoch {epoch} rows ready: kept={len(current_rows)} | {meta}")

        if not current_rows:
            print("No more usable samples. Stopping.")
            break

        # Kick off next prefetch (BOTH modes)
        if prefetch_future is None:
            if USE_LOCAL_CACHE:
                next_epoch_dir = os.path.join(LOCAL_EPOCH_CACHE_ROOT, f"epoch_{(epoch+1):06d}")
                print(f"Prefetching epoch {epoch+1} (copy-cache) -> {next_epoch_dir}")
                prefetch_future = prefetch_exec.submit(
                    prepare_epoch_cache,
                    pointer,
                    trained_set,
                    N_SAMPLES_PER_EPOCH,
                    manifest_rows,
                    next_epoch_dir,
                )
            else:
                print(f"Prefetching epoch {epoch+1} (no cache) ...")
                prefetch_future = prefetch_exec.submit(
                    select_epoch_rows,
                    pointer,
                    trained_set,
                    N_SAMPLES_PER_EPOCH,
                    manifest_rows,
                )

        # Train
        loader = build_loader_from_rows(current_rows)
        avg_loss = train_one_epoch(epoch, loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.6f}")

        # Mark trained
        trained_to_append = [{"audio_path": r["audio_path"]} for r in current_rows if r.get("audio_path")]
        append_jsonl(TRAINED_JSONL_PATH, trained_to_append)
        for r in trained_to_append:
            trained_set.add(r["audio_path"])

        # Save
        save_checkpoint(epoch, subset_start_idx=-1, subset_count=len(current_rows))

        # Optional cleanup
        if DELETE_TRAINED_FROM_DRIVE:
            drive_paths = [r.get("audio_path") for r in current_rows if r.get("audio_path")]
            cleanup_trained_drive_audio(drive_paths, DRIVE_CLEANUP_MODE, DRIVE_ALLOWED_PREFIX, DRIVE_ARCHIVE_DIR)

        del loader
        del current_rows
        cleanup_memory(f"after epoch {epoch}")

        epoch += 1

finally:
    try:
        prefetch_exec.shutdown(wait=False, cancel_futures=True)
    except Exception:
        try:
            prefetch_exec.shutdown(wait=False)
        except Exception:
            pass

print("Done.")
