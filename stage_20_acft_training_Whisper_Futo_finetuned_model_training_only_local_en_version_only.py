##        #######   ######     ###    ##       
##       ##     ## ##    ##   ## ##   ##       
##       ##     ## ##        ##   ##  ##       
##       ##     ## ##       ##     ## ##       
##       ##     ## ##       ######### ##       
##       ##     ## ##    ## ##     ## ##       
########  #######   ######  ##     ## ######## 

# Stage 20 ACFT Training Script

#!pip -q install -U "transformers>=4.38" datasets accelerate soundfile tqdm




"""

usage
---------------------------------------------------
i:\Whisper-training-env\Scripts\python.exe i:\whisper-acft\stage_20_acft_training_Whisper_Futo_finetuned_model_training_only_local_en_version_only.py --manifest "i:/Record_chunks/pairs_manifest_combined_all_datasets_randomized_train_no_reverb.jsonl" --checkpoint_dir "i:/Stage_2_shuffle_Dynamic_n_ctx_checkpoints_partialctx_tiny_en_8/" --reset_trained

"""

import os, json, time, shutil, hashlib, gc, argparse
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
MANIFEST_PATH = os.environ.get(
    "WHISPER_MANIFEST",
    "i:/Record_chunks/pairs_manifest_combined_all_datasets_randomized_train_no_reverb.jsonl",
)

DEFAULT_CKPT_ROOT = os.environ.get(
    "WHISPER_CKPT_ROOT",
    "i:/Stage_2_shuffle_Dynamic_n_ctx_stage_7_checkpoints_partialctx_tiny_en_11",
)

# Checkpoints (single fixed directory; no run tags).
CHECKPOINT_DIR = os.environ.get("WHISPER_CHECKPOINT_DIR") or DEFAULT_CKPT_ROOT

# Track trained list inside the run dir by default; reuse Stage 18 progress unless overridden.
TRAINED_JSONL_PATH = os.environ.get("WHISPER_TRAINED_JSONL") or os.path.join(CHECKPOINT_DIR, "trained_stage1.jsonl")

# ----------------------------
# CLI overrides (no env vars needed)
# ----------------------------
argp = argparse.ArgumentParser(description="Stage 20 ACFT training (Whisper)")
argp.add_argument("--manifest", dest="manifest_path", help="Manifest JSONL to train from")
argp.add_argument("--checkpoint_dir", dest="checkpoint_dir", help="Checkpoint directory")
argp.add_argument("--trained_jsonl", dest="trained_jsonl", help="Path to trained list jsonl")
argp.add_argument("--reset_trained", action="store_true", help="Ignore trained list and start from manifest head")
args, _unknown = argp.parse_known_args()

if args.manifest_path:
    MANIFEST_PATH = args.manifest_path

if args.checkpoint_dir:
    CHECKPOINT_DIR = args.checkpoint_dir

if args.trained_jsonl:
    TRAINED_JSONL_PATH = args.trained_jsonl

RESET_TRAINED = args.reset_trained or (str(os.environ.get("WHISPER_RESET_TRAINED", "0")).lower() in {"1", "true", "yes"})

# If you're running on the same machine as the audio (local disk), caching copies is unnecessary.
# Set to False to stream directly from manifest paths without staging per-epoch copies.
USE_LOCAL_CACHE = False
LOCAL_EPOCH_CACHE_ROOT = "i:/epoch_cache"  # only used when USE_LOCAL_CACHE=True

# --- Model ---
# IMPORTANT:
# - FUTO repos often do NOT include preprocessor_config.json.
# - So load processor from the matching OpenAI whisper base.
FUTO_MODEL_ID = "futo-org/acft-whisper-tiny.en"
PROCESSOR_ID = "openai/whisper-tiny.en"  # must match the base family

TARGET_SR = 16000
N_SAMPLES_PER_EPOCH = 5000
MAX_EPOCHS = 999999

# --- Training knobs ---
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2
LR = 5e-6
MAX_AUDIO_SECONDS = 29.0  # we still pad features to 30s; this just filters absurdly long chunks
BUCKET_SECS = 2.5          # duration bin width (light bucketing)
BUCKET_TOKENS = 32         # transcript word-count bin
BUCKET_BLOCK_SHUFFLE = 128 # shuffle within blocks after bucket flatten to avoid over-clustering

# Loss weights
LAMBDA_ACFT = 1.00       # robustness term
LAMBDA_CE = 0.15         # ASR term (WER) — higher to curb repetition

# Regularisation / stability
CE_LABEL_SMOOTH = 0.05
GRAD_CLIP_NORM = 1.0
FULL_CONTEXT_PROB = 0.15  # force full-context batches sometimes to anchor decoder

# Checkpoint naming (set env to avoid collisions with other stages/runs)
MODEL_PREFIX = os.environ.get("WHISPER_MODEL_PREFIX", "s20_model_epoch_")
STATE_PREFIX = os.environ.get("WHISPER_STATE_PREFIX", "s20_training_state_epoch_")

# DataLoader knobs
NUM_WORKERS = 0
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = False

# Cleanup aggressiveness
CLEANUP_EVERY_N_STEPS = 20
KEEP_LAST_LOCAL_EPOCH_DIRS = 2

# Copying
COPY_THREADS = 2
PREFETCH_THREADS = 2  # background prefetch thread

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
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    except AttributeError:
        pass  # Older PyTorch versions may not have this attribute
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except AttributeError:
        pass  # Older PyTorch versions may not have this attribute
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
    if tag:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserv = torch.cuda.memory_reserved() / (1024**3)
            print(f"[mem] {tag} | cuda alloc={alloc:.2f}GB reserved={reserv:.2f}GB")


# ============================================
# CELL 3/9 — Manifest + trained tracking + selection that SKIPS missing until it fills 5000
# ============================================

if USE_LOCAL_CACHE:
    os.makedirs(LOCAL_EPOCH_CACHE_ROOT, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Bad batch debugging
BAD_BATCH_DIR = os.path.join(CHECKPOINT_DIR, "bad_batches")
os.makedirs(BAD_BATCH_DIR, exist_ok=True)


def _dump_bad_batch(tag: str, batch: dict, extra: dict | None = None):
    """Persist enough info to reproduce a NaN/Inf batch."""
    try:
        payload = {
            "tag": tag,
            "audio_paths": batch.get("audio_paths"),
            "texts": batch.get("texts"),
            "lengths": batch.get("lengths").detach().cpu().tolist() if isinstance(batch.get("lengths"), torch.Tensor) else batch.get("lengths"),
        }
        if extra:
            payload.update(extra)
        out = os.path.join(BAD_BATCH_DIR, f"{tag}.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _isfinite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())
print(f"[paths] manifest={MANIFEST_PATH}")
print(f"[paths] checkpoint_dir={CHECKPOINT_DIR}")
print(f"[paths] trained_list={TRAINED_JSONL_PATH}")


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
if RESET_TRAINED:
    print("RESET_TRAINED is set: ignoring existing trained list and starting from manifest head.")
    trained_set = set()

print("Manifest rows:", len(manifest_rows))
print("Already trained:", len(trained_set))


def _copy_one(src: str, dst: str):
    try:
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            return True, "skipped"
        shutil.copy2(src, dst)
        return True, "copied"
    except Exception:
        return False, "failed"


def prepare_epoch_cache_async(
    epoch_num: int,
    pointer: int,
    trained_set: set,
    need: int,
    manifest_rows: list,
    epoch_dir: str | None,
    copy_threads: int = COPY_THREADS,
    copy_to_cache: bool = True,
):
    """Scan forward from pointer until we KEEP need copied rows.

    Skips:
      - trained paths
      - missing paths
      - empty transcription

    Returns:
      kept_rows, new_pointer, stats
    """

    if copy_to_cache and epoch_dir:
        os.makedirs(epoch_dir, exist_ok=True)

    kept = []
    scanned = 0
    stats = {"copied": 0, "skipped": 0, "missing": 0, "failed": 0, "kept": 0}

    # Pre-collect candidates (bounded by scan-ahead)
    # We scan until we either keep need or hit end.
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

        candidates.append({"audio_path": ap, "raw_transcription": txt})

    # Parallel copy
    if copy_to_cache:
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
            with ThreadPoolExecutor(max_workers=copy_threads) as ex:
                for ok, st, row2 in ex.map(task, candidates):
                    if st in stats:
                        stats[st] += 1
                    else:
                        stats["skipped"] += 1
                    if ok and row2 is not None and os.path.exists(row2["audio_path_local"]):
                        kept.append(row2)
                        if len(kept) >= need:
                            break
    else:
        # No copy: use original paths directly
        for row in candidates:
            row2 = dict(row)
            row2["audio_path_local"] = row["audio_path"]
            stats["kept"] += 1
            kept.append(row2)
            if len(kept) >= need:
                break

    if not copy_to_cache:
        stats["copied"] = 0  # clarify for logging

    stats["kept"] = len(kept)

    return kept, p, {"copy_stats": stats, "scanned_lines": scanned}


# ============================================
# CELL 4/9 — Processor + audio decode + collate (pads to 30s so mel length is ALWAYS 3000)
# ============================================

processor = WhisperProcessor.from_pretrained(PROCESSOR_ID)

N_SAMPLES_30S = processor.feature_extractor.n_samples  # ~480000 at 16k
NB_MAX_FRAMES = processor.feature_extractor.nb_max_frames  # 3000

PAD_ID = processor.tokenizer.pad_token_id
DECODER_START_ID = None  # filled from model config later


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
    # returns padded waveform (float32) of exactly N_SAMPLES_30S
    n = wav.shape[0]
    if n == N_SAMPLES_30S:
        return wav
    if n > N_SAMPLES_30S:
        return wav[:N_SAMPLES_30S]
    out = np.zeros((N_SAMPLES_30S,), dtype=np.float32)
    out[:n] = wav
    return out


_duration_cache = {}


def audio_duration_sec(path: str) -> float:
    """Fastish duration lookup with a tiny in-memory cache."""
    key = os.path.abspath(path)
    if key in _duration_cache:
        return _duration_cache[key]
    try:
        info = sf.info(path)
        dur = float(info.frames) / float(info.samplerate)
    except Exception:
        wav, sr = decode_mono_16k(path)
        dur = float(wav.shape[0]) / float(sr) if wav is not None and sr else 0.0
    _duration_cache[key] = dur
    return dur


def shift_tokens_right(labels: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    # labels: [B, T]
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0] = decoder_start_token_id
    # replace -100 with pad for decoder inputs
    shifted = shifted.masked_fill(shifted == -100, pad_token_id)
    return shifted


def collate_batch(examples):
    waveforms_30s = []
    lengths_sec = []
    texts = []
    audio_paths = []

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
        audio_paths.append(ap)

    if not waveforms_30s:
        return None

    feats = processor.feature_extractor(
        waveforms_30s,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
    )

    # Hard guarantee
    if feats.input_features.shape[-1] != NB_MAX_FRAMES:
        raise RuntimeError(f"Bad mel length: {feats.input_features.shape} (expected last dim {NB_MAX_FRAMES})")

    # Tokenise labels
    tok = processor.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=448,
    )

    labels = tok["input_ids"].clone()
    labels[labels == PAD_ID] = -100

    # NOTE: decoder_input_ids computed later once DECODER_START_ID is known

    batch = {
        "lengths": torch.tensor(lengths_sec, dtype=torch.float32),
        "input_features": feats.input_features,
        "labels": labels,
        "labels_raw": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "audio_paths": audio_paths,
        "texts": texts,
    }

    return batch


# ============================================
# CELL 5/9 — Models (student: WhisperForConditionalGeneration) + frozen reference + optimizer
# ============================================

# Generation config from model repo (optional but keeps warnings down)
# You can also omit; training doesn't use generation.
try:
    gen_config = GenerationConfig.from_pretrained(FUTO_MODEL_ID)
except Exception:
    gen_config = None


def load_student_from_checkpoint_or_base(model_dir: str | None):
    if model_dir and os.path.isdir(model_dir):
        print("Loading student from checkpoint:", model_dir)
        return WhisperForConditionalGeneration.from_pretrained(model_dir, generation_config=gen_config)
    print("Loading student from base:", FUTO_MODEL_ID)
    return WhisperForConditionalGeneration.from_pretrained(FUTO_MODEL_ID, generation_config=gen_config)


def _collect_epochs(checkpoint_dir: str, prefixes: list[str], is_dir: bool):
    out = {}
    for name in os.listdir(checkpoint_dir):
        p = os.path.join(checkpoint_dir, name)
        if is_dir and not os.path.isdir(p):
            continue
        if (not is_dir) and not os.path.isfile(p):
            continue
        for prefix in prefixes:
            if name.startswith(prefix):
                try:
                    ep = int(name[len(prefix):].split(".")[0])
                    out[ep] = p
                except Exception:
                    pass
    return out


def find_latest_checkpoint_epoch(checkpoint_dir: str):
    if not os.path.isdir(checkpoint_dir):
        return None, None, None

    model_epochs = _collect_epochs(checkpoint_dir, [MODEL_PREFIX, "model_epoch_"], is_dir=True)
    state_epochs = _collect_epochs(checkpoint_dir, [STATE_PREFIX, "training_state_epoch_"], is_dir=False)

    if not model_epochs and not state_epochs:
        return None, None, None

    latest_epoch = max(list(model_epochs.keys()) + list(state_epochs.keys()))
    return latest_epoch, model_epochs.get(latest_epoch), state_epochs.get(latest_epoch)


latest_ckpt_epoch, latest_model_dir, latest_state_path = find_latest_checkpoint_epoch(CHECKPOINT_DIR)
latest_state = None
if latest_state_path:
    try:
        latest_state = torch.load(latest_state_path, map_location="cpu")
        print("Found training state:", latest_state_path)
    except Exception as e:
        print("WARNING: cannot read state file, weights-only resume.", repr(e))
        latest_state = None

# Epoch start (best effort)
trained_based_epoch = len(trained_set) // max(1, int(N_SAMPLES_PER_EPOCH))
ckpt_based_epoch = (latest_ckpt_epoch + 1) if latest_ckpt_epoch is not None else 0
epoch_num_start = max(trained_based_epoch, ckpt_based_epoch)
print(f"Auto epoch start: {epoch_num_start} (trained_based={trained_based_epoch}, ckpt_based={ckpt_based_epoch})")

# Student model
model_train = load_student_from_checkpoint_or_base(latest_model_dir)
model_train.to(device)
model_train.train()

# Fill decoder start id for shifting
DECODER_START_ID = int(model_train.config.decoder_start_token_id)

# Frozen reference: full-context behaviour anchor (ACFT)
# IMPORTANT: use the SAME family (FUTO base) so CE can adapt without being pulled back to OpenAI.
model_ref = WhisperForConditionalGeneration.from_pretrained(FUTO_MODEL_ID, generation_config=gen_config)
model_ref.to(device)
model_ref.eval()
for p in model_ref.parameters():
    p.requires_grad_(False)

# Optimiser
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
# CELL 6/9 — Partial encoder + losses (ACFT + CE)
# ============================================

FULL_ENCODER_CONTEXT_LENGTH = int(model_train.config.max_source_positions)  # 1500


def compute_partially_encoder(whisper_model, input_features: torch.Tensor, n_audio_ctx: int):
    """whisper_model is model_train.model or model_ref.model (WhisperModel)."""

    target_mel_seq_len = 2 * int(n_audio_ctx)
    diff = target_mel_seq_len - input_features.shape[2]

    if diff > 0:
        input_features = F.pad(input_features, (0, diff, 0, 0, 0, 0), "constant", 0.0)
    elif diff < 0:
        input_features = input_features[:, :, :target_mel_seq_len]

    if n_audio_ctx == FULL_ENCODER_CONTEXT_LENGTH:
        return whisper_model.encoder(input_features).last_hidden_state

    # Manual forward (matches the original idea)
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


def pick_n_ctx_from_batch(lengths_sec: torch.Tensor, max_embed_positions: int):
    return FULL_ENCODER_CONTEXT_LENGTH


import math

# FULL_ENCODER_CONTEXT_LENGTH is typically 1500 for Whisper (30s)
# max_embed_positions is usually the same (1500) for tiny/base/small


def pick_n_ctx_from_batch(
    lengths_sec: torch.Tensor,
    max_embed_positions: int,
    *,
    full_ctx: int = 1500,
    safety_sec: float = 0.20,
    round_to: int = 16,
    jitter_max: int = 64,
):
    """Pick a *batch-level* encoder context (audio_ctx) based on the longest sample.

    IMPORTANT SAFETY PROPERTY:
    - We NEVER pick a context shorter than needed to cover the longest audio in the batch.
      That means we won't truncate real speech and then force the decoder to emit labels
      for audio it never heard.

    Whisper mapping:
    - 30s -> audio_ctx=1500
    - so 1s ~ 50 ctx tokens
    - mel frames are 2*audio_ctx because of the encoder's stride-2 conv.
    """

    max_len = float(lengths_sec.max().item())

    # ctx needed to cover the *audio* (plus a tiny safety margin for trailing speech)
    base = (full_ctx / 30.0) * (max_len + safety_sec)
    base = int(math.ceil(base))

    # round up for nicer shapes (optional)
    if round_to and round_to > 1:
        base = int(math.ceil(base / round_to) * round_to)

    base = max(1, min(base, max_embed_positions))

    # jitter ONLY upward (or clamp downward to base) to avoid truncating speech
    # If you want symmetric jitter, do it but clamp: n_ctx = max(base, base + rand)
    jitter = min(jitter_max, max(0, base // 10))  # mild jitter
    if jitter > 0:
        add = int(torch.randint(0, jitter + 1, (1,), device=lengths_sec.device).item())
        n_ctx = base + add
    else:
        n_ctx = base

    return int(max(1, min(n_ctx, max_embed_positions)))


# ----- Stage 2 training tip (ACFT) -----
# If your goal is: keep your Stage-1 WER improvements *and* become robust to dynamic audio_ctx,
# set your reference model to the Stage-1 checkpoint (frozen), and train the student with
# dynamic n_ctx using mostly/only the hidden-state matching loss (ACFT).
#
# Typical:
#   LAMBDA_CE   = 0.0  (or very small like 0.05)
#   LAMBDA_ACFT = 1.0
#
# This mirrors the method described in futo-org/whisper-acft.


def masked_hidden_mse(hs_pred: torch.Tensor, hs_tgt: torch.Tensor, attn_mask: torch.Tensor):
    # hs_* : [L, B, T, D]
    # IMPORTANT: do this in float32 and normalise across ALL elements.
    hs_pred = hs_pred.float()
    hs_tgt = hs_tgt.float()
    mask = attn_mask.unsqueeze(0).unsqueeze(-1).to(dtype=torch.float32)  # [1,B,T,1]
    diff2 = (hs_pred - hs_tgt).pow(2) * mask
    # mask.sum() counts B*T, but diff2.sum() also sums over L and D. Normalise by L*D as well.
    denom = mask.sum() * float(hs_pred.shape[0]) * float(hs_pred.shape[-1])
    return diff2.sum(dtype=torch.float32) / denom.clamp_min(1.0)


def get_lm_head(model: WhisperForConditionalGeneration):
    # transformers has used proj_out in many seq2seq models
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

    model_dir = os.path.join(CHECKPOINT_DIR, f"{MODEL_PREFIX}{epoch_num:06d}")
    os.makedirs(model_dir, exist_ok=True)

    # Save student
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
    }

    state_path = os.path.join(CHECKPOINT_DIR, f"{STATE_PREFIX}{epoch_num:06d}.pt")
    tmp_state_path = state_path + ".tmp"
    torch.save(state, tmp_state_path)
    os.replace(tmp_state_path, state_path)

    # Quick verify load
    verified_ok = True
    try:
        _ = WhisperForConditionalGeneration.from_pretrained(model_dir)
        _ = torch.load(state_path, map_location="cpu")
    except Exception as e:
        verified_ok = False
        print("WARNING: checkpoint verification failed.", repr(e))

    print("Saved checkpoint:", model_dir)
    cleanup_memory(f"after save checkpoint {epoch_num}")
    return model_dir, state_path, verified_ok


# ============================================
# CELL 8/9 — DataLoader + training loop (CE + ACFT)
# ============================================


def build_loader_from_rows(rows_for_epoch):
    # Light bucketing by duration and transcript length to avoid "one long clip kills the batch"
    buckets = {}
    rng = random.Random()

    for r in rows_for_epoch:
        dur = audio_duration_sec(r["audio_path_local"])
        txt = (r.get("raw_transcription") or "").strip()
        tok_len = len(txt.split())
        bkey = (int(dur // BUCKET_SECS), int(tok_len // BUCKET_TOKENS))
        buckets.setdefault(bkey, []).append({
            "audio": r["audio_path_local"],
            "raw_transcription": txt,
        })

    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)

    ordered = []
    for k in bucket_keys:
        rng.shuffle(buckets[k])
        ordered.extend(buckets[k])

    if BUCKET_BLOCK_SHUFFLE > 1:
        shuffled = []
        for i in range(0, len(ordered), BUCKET_BLOCK_SHUFFLE):
            block = ordered[i:i+BUCKET_BLOCK_SHUFFLE]
            rng.shuffle(block)
            shuffled.extend(block)
        ordered = shuffled

    ds = Dataset.from_list(ordered)

    pin_memory = (device == "cuda")
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
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
        labels = batch["labels"].to(device, non_blocking=True)           # -100 for pad
        labels_raw = batch["labels_raw"].to(device, non_blocking=True)   # pad ids intact
        attn_mask = batch["attention_mask"].to(device, non_blocking=True)

        # Build decoder_input_ids ourselves (avoid prepare_decoder_input_ids_from_labels)
        decoder_input_ids = shift_tokens_right(labels, pad_token_id=PAD_ID, decoder_start_token_id=DECODER_START_ID)

        max_embed_positions = model_train.model.encoder.embed_positions.weight.shape[0]
        n_ctx = pick_n_ctx_from_batch(lengths, max_embed_positions)
        if torch.rand(()) < FULL_CONTEXT_PROB:
            n_ctx = FULL_ENCODER_CONTEXT_LENGTH

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            # -------- Student: partial encoder --------
            enc_partial = compute_partially_encoder(model_train.model, input_features, n_ctx)
            dec_partial = model_train.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=attn_mask,
                encoder_hidden_states=enc_partial,
                output_hidden_states=True,
                use_cache=False,
            )

            # CE loss (WER): logits from last hidden
            logits = lm_head(dec_partial.last_hidden_state)
            loss_ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=CE_LABEL_SMOOTH,
            )

        # -------- Reference: full encoder (ACFT target) --------
        # Do reference pass in full precision to avoid fp16 SDPA/overflow quirks.
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
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

        # Fail fast BEFORE backward()
        if not _isfinite(loss_ce) or not _isfinite(hs_p) or not _isfinite(hs_f):
            _dump_bad_batch(f"ep{epoch_num}_b{batch_idx:06d}", batch, {
                "n_ctx": int(n_ctx),
                "loss_ce": float(loss_ce.detach().cpu().item()) if torch.isfinite(loss_ce) else None,
            })
            optimizer.zero_grad(set_to_none=True)
            continue

        loss_acft = masked_hidden_mse(hs_p, hs_f, attn_mask)
        if not _isfinite(loss_acft):
            _dump_bad_batch(f"ep{epoch_num}_b{batch_idx:06d}", batch, {
                "n_ctx": int(n_ctx),
                "loss_ce": float(loss_ce.detach().cpu().item()),
            })
            optimizer.zero_grad(set_to_none=True)
            continue

        loss = (LAMBDA_CE * loss_ce) + (LAMBDA_ACFT * loss_acft)
        loss = loss / float(GRAD_ACCUM_STEPS)

        if use_grad_scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            if use_grad_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_train.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model_train.parameters(), GRAD_CLIP_NORM)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running += float(loss.item())
        steps += 1

        pbar.set_postfix({
            "loss": f"{(running / max(1, steps)):.6f}",
            "ce": f"{loss_ce.item():.4f}",
            "acft": f"{loss_acft.item():.4f}",
            "n_ctx": int(n_ctx),
            "bs": int(input_features.shape[0]),
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        })

        # Cleanup
        del batch
        del input_features, lengths, labels, labels_raw, attn_mask, decoder_input_ids
        del enc_partial, dec_partial, logits, loss_ce
        del enc_full, dec_full, hs_p, hs_f, loss_acft, loss

        if (batch_idx + 1) % CLEANUP_EVERY_N_STEPS == 0:
            cleanup_memory(f"epoch {epoch_num} step {batch_idx+1}")

    cleanup_memory(f"end epoch {epoch_num}")
    return running / max(1, steps)


# ============================================
# CELL 9/9 — Main loop with async prefetch cache
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

prefetch_exec = ThreadPoolExecutor(max_workers=PREFETCH_THREADS) if USE_LOCAL_CACHE else None
prefetch_future = None
prefetch_result = None

try:
    while epoch < MAX_EPOCHS:
        if USE_LOCAL_CACHE:
            cleanup_old_epoch_dirs(LOCAL_EPOCH_CACHE_ROOT, keep_last_k=KEEP_LAST_LOCAL_EPOCH_DIRS)

            # Consume prefetched epoch if ready
            if prefetch_future is not None and prefetch_future.done():
                current_rows, new_pointer, meta = prefetch_future.result()
                pointer = new_pointer
                prefetch_future = None
                print(f"\nEpoch {epoch} cache ready: kept={len(current_rows)} | {meta}")
            else:
                # Build cache synchronously
                epoch_dir = os.path.join(LOCAL_EPOCH_CACHE_ROOT, f"epoch_{epoch:06d}")
                print(f"\nPreparing epoch {epoch} cache -> {epoch_dir}")
                current_rows, pointer, meta = prepare_epoch_cache_async(
                    epoch_num=epoch,
                    pointer=pointer,
                    trained_set=trained_set,
                    need=N_SAMPLES_PER_EPOCH,
                    manifest_rows=manifest_rows,
                    epoch_dir=epoch_dir,
                    copy_to_cache=True,
                )
                print(f"Epoch {epoch} cache ready: kept={len(current_rows)} | {meta}")
        else:
            # No local cache: use manifest paths directly
            print(f"\nPreparing epoch {epoch} directly from manifest (no cache)")
            current_rows, pointer, meta = prepare_epoch_cache_async(
                epoch_num=epoch,
                pointer=pointer,
                trained_set=trained_set,
                need=N_SAMPLES_PER_EPOCH,
                manifest_rows=manifest_rows,
                epoch_dir=None,
                copy_threads=0,
                copy_to_cache=False,
            )
            print(f"Epoch {epoch} ready (no cache): kept={len(current_rows)} | {meta}")

        if not current_rows:
            print("No more usable samples. Stopping.")
            break

        # Kick off next prefetch when using cache
        if USE_LOCAL_CACHE:
            next_epoch_dir = os.path.join(LOCAL_EPOCH_CACHE_ROOT, f"epoch_{(epoch+1):06d}")
            if prefetch_future is None:
                print(f"Prefetching epoch {epoch+1} in background -> {next_epoch_dir}")
                prefetch_future = prefetch_exec.submit(
                    prepare_epoch_cache_async,
                    epoch+1,
                    pointer,
                    trained_set,
                    N_SAMPLES_PER_EPOCH,
                    manifest_rows,
                    next_epoch_dir,
                    COPY_THREADS,
                    True,
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
    if prefetch_exec is not None:
        try:
            prefetch_exec.shutdown(wait=False, cancel_futures=True)
        except Exception:
            try:
                prefetch_exec.shutdown(wait=False)
            except Exception:
                pass

print("Done.")
