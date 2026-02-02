# ============================================
# Whisper partial-context training — BETTER no-cache mode
# + LR warmup + decay + proper resume LR override (FULL SCRIPT)
#
# This is your Stage 18 script with the learning-rate fixes applied:
# - Explicit LR_START / LR_FLOOR / WARMUP_STEPS / DECAY_GAMMA
# - Optional scheduler (warmup + epoch-based decay)
# - Proper resume: restores global_step + scheduler and (optionally) forces LR_START
# - Shows LR in progress bar
# - Default: DYNAMIC_BATCH_SIZE=False while you tune LR (you can turn it back on later)
# ============================================

# ============================================
# CELL 1/9 — Install + imports
# ============================================

#!pip -q install -U "transformers>=4.38" datasets accelerate soundfile tqdm

import os, json, time, shutil, hashlib, gc, math, winsound, sys, atexit, tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional, Dict

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

# =============================
# PATCH 1) Durable IO + keys
# =============================

def canonical_audio_key(p: str) -> str:
    """Stable key across Windows path casing and slash variants."""
    if not p:
        return ""
    p = os.path.normpath(p)
    p = p.replace("\\", "/").lower()
    return p


def read_jsonl(path: str):
    """Tolerant JSONL reader (skips partial/corrupt lines instead of blowing up)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] bad jsonl line {ln} in {path}; skipping")
    return rows


def append_jsonl(path: str, rows):
    """Append JSONL with flush+fsync so crashes don't lose buffered data."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8", errors="replace", buffering=1) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            # Some filesystems may not support fsync well; flush still helps.
            pass


def atomic_write_json(path: str, obj):
    """Atomic JSON write (write temp in same dir, then os.replace)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=d)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

# ============================================
# CELL 2/9 — Settings
# ============================================

# --- Data ---
MANIFEST_PATH = "I:/Record_chunks/pairs_manifest_combined_all_datasets_randomized_train_no_reverb_filtered.jsonl"

# Make each run isolated. If you want fixed naming, set RUN_TAG manually.
RUN_TAG = os.environ.get('WHISPER_RUN_TAG') or datetime.now().strftime('%Y%m%d_%H%M%S')
CHECKPOINT_DIR = f"i:/Stage_2_shuffle_checkpoints_partialctx_tiny_en_13/{RUN_TAG}"

# Put run-state files INSIDE the checkpoint directory so runs do not poison each other.
TRAINED_JSONL_PATH = os.path.join(CHECKPOINT_DIR, "trained_stage1.jsonl")

# =============================
# PATCH 2) Run state + pending epoch plan
# =============================
RUN_STATE_PATH = os.path.join(CHECKPOINT_DIR, "run_state.json")
PENDING_PLAN_PATH = os.path.join(CHECKPOINT_DIR, "pending_epoch_plan.json")

# --- Add a 'start fresh' option ---
START_FRESH = bool(int(os.environ.get('WHISPER_START_FRESH', '0')))


def manifest_signature(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"


def load_run_state() -> dict:
    if not os.path.exists(RUN_STATE_PATH):
        return {}
    try:
        with open(RUN_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as e:
        print("[warn] could not read run_state.json; ignoring:", repr(e))
        return {}


def save_run_state(state: dict):
    state = dict(state)
    state["updated_ts"] = time.time()
    atomic_write_json(RUN_STATE_PATH, state)


def load_pending_plan() -> Optional[Dict]:
    if not os.path.exists(PENDING_PLAN_PATH):
        return None
    try:
        with open(PENDING_PLAN_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("[warn] could not read pending plan; deleting:", repr(e))
        try:
            os.remove(PENDING_PLAN_PATH)
        except Exception:
            pass
        return None


def save_pending_plan(plan: dict):
    plan = dict(plan)
    plan["created_ts"] = time.time()
    atomic_write_json(PENDING_PLAN_PATH, plan)


def clear_pending_plan():
    try:
        if os.path.exists(PENDING_PLAN_PATH):
            os.remove(PENDING_PLAN_PATH)
    except Exception:
        pass

# Console Logging Setup
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# In main(), right after ensuring CHECKPOINT_DIR exists:
if START_FRESH and os.path.isdir(CHECKPOINT_DIR) and any(d.startswith('model_epoch_') for d in os.listdir(CHECKPOINT_DIR)):
    raise SystemExit(
        f"Refusing to start in a non-empty checkpoint dir: {CHECKPOINT_DIR}\n"
        "Either delete/rename it, or set WHISPER_START_FRESH=0 to resume deliberately."
    )

LOG_PATH = os.path.join(CHECKPOINT_DIR, "console.log")

# ===== Debug: pinpoint which file causes non-finite / bad CE loss =====
DEBUG_NONFINITE_CE = True
NONFINITE_CE_LOG_JSONL = os.path.join(CHECKPOINT_DIR, "debug_nonfinite_ce.jsonl")
NONFINITE_CE_TOPK = 5              # how many worst samples to log per bad batch
NONFINITE_CE_SAVE_TENSORS = False  # set True to also dump a .pt with tensors (can get big)
NONFINITE_CE_TENSOR_DIR = os.path.join(CHECKPOINT_DIR, "debug_tensors")
CE_SPIKE_THRESHOLD = None          # e.g. 50.0 to log extreme-but-finite CE spikes


class _Tee:
    """Duplicate writes to console + log file (works for tqdm which expects write/flush/isatty)."""

    def __init__(self, console_stream, log_file):
        self._console = console_stream
        self._log = log_file

    def write(self, data):
        # tqdm and print both call write() a lot; keep it robust.
        try:
            self._console.write(data)
        except Exception:
            pass
        try:
            self._log.write(data)
            self._log.flush()  # so you don't lose anything if it crashes
        except Exception:
            pass

    def flush(self):
        try:
            self._console.flush()
        except Exception:
            pass
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return bool(self._console.isatty())
        except Exception:
            return False

    def fileno(self):
        # Some libs ask for this.
        try:
            return self._console.fileno()
        except Exception:
            raise OSError("fileno() not supported")


_log_fh = open(LOG_PATH, "a", encoding="utf-8", errors="replace", buffering=1)
_log_fh.write("\n" + "=" * 80 + "\n")
_log_fh.write(f"Run started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
_log_fh.write("Command: " + " ".join(map(str, sys.argv)) + "\n")
_log_fh.write("=" * 80 + "\n")
_log_fh.flush()

_orig_stdout = sys.stdout
_orig_stderr = sys.stderr
sys.stdout = _Tee(_orig_stdout, _log_fh)
sys.stderr = _Tee(_orig_stderr, _log_fh)


def _close_console_log():
    # Restore streams first (helps avoid odd behaviour during interpreter shutdown)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    try:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
    except Exception:
        pass

    try:
        _log_fh.flush()
        _log_fh.close()
    except Exception:
        pass


atexit.register(_close_console_log)
# ===== End Console Logging Setup =====

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

# ===== Combined WER + ACFT knobs =====
# Goal: keep WER going down, while ACFT prevents repetition / collapse when audio_ctx is dynamic.
# Direct script parameters - modify these values as needed
ACFT_REFERENCE_MODEL_ID = FUTO_MODEL_ID  # Reference model for ACFT (default: same as training model)
CE_LABEL_SMOOTH = 0.05  # Label smoothing for CE loss (0.0 = disabled, 0.05 = recommended)
LAMBDA_CE = 1.0  # Weight for cross-entropy loss
LAMBDA_ACFT = 0.30  # Weight for ACFT robustness loss
# Optional ramp: start ACFT small then ramp up over first N optimizer steps
ACFT_RAMP_STEPS = 200  # 0 disables ramp, 200 = ramp over first 200 steps

# Dynamic audio_ctx (partial context) to teach robustness
FORCE_FULL_AUDIO_CTX = False  # True = always use full context, False = use dynamic context
AUDIO_CTX_SAFETY_SEC = 0.20  # Safety margin in seconds for dynamic context
AUDIO_CTX_ROUND_TO = 16  # Round context to multiples of this value
AUDIO_CTX_JITTER_MAX = 64  # Maximum upward jitter for dynamic context

TARGET_SR = 16000
N_SAMPLES_PER_EPOCH = 5016
MAX_EPOCHS = 999999

# --- Training knobs ---
BATCH_SIZE = 8  # Reduced from 24 to prevent CUDA OOM
GRAD_ACCUM_STEPS = 4  # Increased from 2 to maintain effective batch size
MIN_BATCH_SIZE = 4  # Minimum batch size for very large files
MAX_BATCH_SIZE = 40  # Maximum batch size for small files
MAX_AUDIO_SECONDS = 30.0  # we pad features to 30s; this filters absurdly long chunks

# --- Gradient clipping to prevent exploding gradients ---
MAX_GRAD_NORM = 0.1  # Very aggressive gradient clipping to prevent NaN

# ----------------------------
# Learning rate + scheduler (NEW)
# ----------------------------
# Start here:
# - If training is unstable or WER collapses after epoch_000001: reduce LR_START (e.g., 2e-6 -> 1e-6 -> 5e-7)
# - If training is too slow / no learning: increase LR_START (e.g., 2e-6 -> 5e-6)
LR_START = 2e-6          # Reduce LR further to prevent NaN
LR_FLOOR = 2e-7           # Higher floor to prevent LR from getting too small
WARMUP_STEPS = 50         # Slightly longer warmup
DECAY_GAMMA = 0.8         # Gentler decay per epoch
USE_SCHEDULER = True        # Enable scheduler for better stability
RESUME_OVERRIDE_LR = True   # IMPORTANT: force LR_START even when resuming optimizer state

# --- Fix the dangerous behaviour: NEVER bump LR UP on resume ---
# Replace it with a clamp-only policy:
RESUME_CLAMP_LR = True

# Keep old name so existing code that logs LR still works
LR = LR_START

# Dynamic batch size settings
# NOTE: while tuning LR, keep this OFF for stability.
# After LR is stable, you can set True again.
DYNAMIC_BATCH_SIZE = False
MEMORY_THRESHOLD_HIGH = 0.95  # Reduce batch size if memory usage > 85%
MEMORY_THRESHOLD_LOW = 0.60  # Increase batch size if memory usage < 60%
DURATION_THRESHOLD_LARGE = 20.0  # Files longer than this get smaller batch size
DURATION_THRESHOLD_SMALL = 10.0  # Files shorter than this can use larger batch size

# Loss weights are now defined above (env overridable)

# DataLoader knobs
NUM_WORKERS = 0           # Windows: keep 0. Colab/Linux: 2–4 can speed up.

# Cleanup aggressiveness
CLEANUP_EVERY_N_STEPS = 10  # More frequent cleanup
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
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
    except AttributeError:
        pass
    try:
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except AttributeError:
        pass
    torch.backends.cudnn.benchmark = True


def clamp_optimizer_lr(optimizer, max_lr: float):
    """Clamp LR DOWN to max_lr; never increases it."""
    for pg in optimizer.param_groups:
        pg['lr'] = min(float(pg.get('lr', max_lr)), float(max_lr))


def estimate_opt_steps_per_epoch(n_samples: int, batch_size: int, grad_accum: int) -> int:
    bs = max(1, int(batch_size))
    ga = max(1, int(grad_accum))
    micro_batches = int(math.ceil(float(n_samples) / float(bs)))
    return max(1, int(math.ceil(float(micro_batches) / float(ga))))


OPT_STEPS_PER_EPOCH = estimate_opt_steps_per_epoch(N_SAMPLES_PER_EPOCH, BATCH_SIZE, GRAD_ACCUM_STEPS)


device = "cuda" if torch.cuda.is_available() else "cpu"
DISABLE_AMP = False  # Set to True to disable mixed precision (helps with NaN issues)
use_amp = (device == "cuda") and (not DISABLE_AMP)
amp_dtype = torch.float16
use_grad_scaler = (device == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_grad_scaler)

print("Device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"[lr] OPT_STEPS_PER_EPOCH≈{OPT_STEPS_PER_EPOCH} | WARMUP_STEPS={WARMUP_STEPS} | LR_START={LR_START} | LR_FLOOR={LR_FLOOR} | DECAY_GAMMA={DECAY_GAMMA}")
else:
    print("WARNING: CPU training will be very slow.")


# Global variables for dynamic batch size
current_batch_size = BATCH_SIZE
batch_size_history = []
om_adjustments = 0


def check_memory_availability(required_gb: float = 2.0) -> bool:
    """Check if enough GPU memory is available."""
    if not torch.cuda.is_available():
        return True
    try:
        alloc = torch.cuda.memory_allocated() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        available = total - alloc
        return available > required_gb
    except Exception:
        return True


def beep_notification(frequency: int = 1000, duration: int = 500):
    """Play a beep notification when script completes."""
    try:
        if os.name == 'nt':
            winsound.Beep(frequency, duration)
        else:
            print('\a')
    except Exception:
        pass


def cleanup_memory(tag: str = ""):
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        except Exception:
            pass
    if tag and torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024**3)
        reserv = torch.cuda.memory_reserved() / (1024**3)
        max_alloc = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[mem] {tag} | cuda alloc={alloc:.2f}GB reserved={reserv:.2f}GB max={max_alloc:.2f}GB")
        if alloc > max_alloc * 0.9:
            print(f"[mem] WARNING: High memory usage ({alloc:.2f}GB / {max_alloc:.2f}GB)")


def get_memory_usage_ratio() -> float:
    """Get current GPU memory usage as ratio (0.0 to 1.0)."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        alloc = torch.cuda.memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        return alloc / total
    except Exception:
        return 0.0


def calculate_optimal_batch_size(avg_duration: float, memory_ratio: float) -> int:
    """Calculate optimal batch size based on audio duration and memory usage."""
    global current_batch_size, om_adjustments

    if not DYNAMIC_BATCH_SIZE:
        return BATCH_SIZE

    suggested_size = current_batch_size

    if avg_duration > DURATION_THRESHOLD_LARGE:
        duration_factor = DURATION_THRESHOLD_LARGE / avg_duration
        suggested_size = max(MIN_BATCH_SIZE, int(current_batch_size * duration_factor))

    elif avg_duration < DURATION_THRESHOLD_SMALL:
        duration_factor = avg_duration / DURATION_THRESHOLD_SMALL
        suggested_size = min(MAX_BATCH_SIZE, int(current_batch_size * (1 + duration_factor * 0.5)))

    if memory_ratio > MEMORY_THRESHOLD_HIGH:
        memory_factor = MEMORY_THRESHOLD_HIGH / memory_ratio
        suggested_size = max(MIN_BATCH_SIZE, int(suggested_size * memory_factor))

    elif memory_ratio < MEMORY_THRESHOLD_LOW:
        memory_factor = (memory_ratio / MEMORY_THRESHOLD_LOW) * 0.3 + 0.7
        suggested_size = min(MAX_BATCH_SIZE, int(suggested_size * (1 / memory_factor)))

    if abs(suggested_size - current_batch_size) >= 2:
        old_size = current_batch_size
        current_batch_size = suggested_size
        om_adjustments += 1
        batch_size_history.append({
            'step': om_adjustments,
            'old_size': old_size,
            'new_size': current_batch_size,
            'avg_duration': avg_duration,
            'memory_ratio': memory_ratio,
            'timestamp': time.time()
        })
        print(f"[batch] Adjusted batch size: {old_size} -> {current_batch_size} (avg_dur={avg_duration:.1f}s, mem={memory_ratio:.1%})")

    return current_batch_size


# ============================================
# CELL 3/9 — Manifest + trained tracking + selection
# ============================================

if USE_LOCAL_CACHE:
    os.makedirs(LOCAL_EPOCH_CACHE_ROOT, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


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
    out = set()
    for r in rows:
        key = (r.get("key") or r.get("uid") or "").strip()
        if not key:
            key = canonical_audio_key(r.get("audio_path"))
        if key:
            out.add(key)
    return out


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
        uid = (obj.get("uid") or obj.get("id") or "").strip()
        key = uid if uid else canonical_audio_key(ap)
        manifest_rows.append({
            "audio_path": ap,
            "raw_transcription": obj.get("raw_transcription", ""),
            "uid": uid,
            "key": key,
        })

trained_set = load_trained_set(TRAINED_JSONL_PATH)
print("Manifest rows:", len(manifest_rows))
print("Already trained:", len(trained_set))


def audio_header_ok(path: str) -> bool:
    """Fast validation: checks sample rate + duration via header."""
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
    """No-cache selection."""

    kept = []
    scanned = 0
    stats = {
        "kept": 0,
        "scanned": 0,
        "trained_skipped": 0,
        "missing": 0,
        "bad_header": 0,
    }

    p = pointer

    while p < len(manifest_rows) and len(kept) < need:
        block = []

        while p < len(manifest_rows) and len(block) < VALIDATE_BLOCK and len(kept) + len(block) < need:
            r = manifest_rows[p]
            p += 1
            scanned += 1

            ap = r.get("audio_path")
            txt = (r.get("raw_transcription") or "").strip()
            
            key = r.get("key") or canonical_audio_key(ap)

            if not ap:
                continue
            if key in trained_set:
                stats["trained_skipped"] += 1
                continue
            if not txt:
                continue
            if not os.path.exists(ap):
                stats["missing"] += 1
                continue

            block.append({"audio_path": ap, "raw_transcription": txt, "key": key})

        if not block:
            continue

        if not VALIDATE_AUDIO_IN_SELECTOR:
            kept.extend(block)
            continue

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
        key = r.get("key") or canonical_audio_key(ap)
        if key in trained_set:
            continue
        if not txt:
            continue
        if not os.path.exists(ap):
            stats["missing"] += 1
            continue
        if VALIDATE_AUDIO_IN_SELECTOR and not audio_header_ok(ap):
            continue

        candidates.append({"audio_path": ap, "raw_transcription": txt, "key": key})

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


# ============================
# Helper functions to fix token issues
# ============================

def _tok_id(tokenizer, tok: str, fallback: int | None = None) -> int:
    tid = tokenizer.convert_tokens_to_ids(tok)
    if tid is None or int(tid) < 0:
        if fallback is None:
            raise RuntimeError(f"Token not found in tokenizer vocab: {tok}")
        return int(fallback)
    return int(tid)


def fix_whisper_special_tokens(processor, model):
    """Hard-sets the correct decoder start token and prints token sanity."""
    tok = processor.tokenizer

    sot_id = _tok_id(tok, "<|startoftranscript|>")
    nospeech_id = _tok_id(tok, "<|nospeech|>", fallback=50362)  # fallback only for printing

    print("[tokens] pad:", tok.pad_token_id, "eos:", tok.eos_token_id, "bos:", tok.bos_token_id)
    print("[tokens] <|startoftranscript|> id:", sot_id, "->", tok.decode([sot_id]))
    print("[tokens] <|nospeech|> id:", nospeech_id, "->", tok.decode([nospeech_id]))

    # CRITICAL: decoder_start_token_id should be SOT, not nospeech.
    before = getattr(model.config, "decoder_start_token_id", None)
    model.config.decoder_start_token_id = sot_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.decoder_start_token_id = sot_id
        # also make sure generation knows where to stop
        model.generation_config.eos_token_id = tok.eos_token_id
        model.generation_config.pad_token_id = tok.pad_token_id

    print("[tokens] decoder_start_token_id was:", before, "now:", model.config.decoder_start_token_id,
          "->", tok.decode([model.config.decoder_start_token_id]))

    # If you see that your previous decoder_start_token_id decodes to <|nospeech|>,
    # that is a serious config bug and can cause pathological decoding.

    return sot_id


def strip_leading_token(labels, token_id: int, pad_id: int):
    """If every sequence begins with token_id (and it's not padding), drop that column."""
    # Safety: if token_id is not initialised yet, do nothing.
    if token_id is None:
        return labels
    # labels are int64 tensor shaped [B, T]
    if labels.numel() == 0:
        return labels
    first_col = labels[:, 0]
    # Only strip if it's consistently present (and not already masked)
    if torch.all(first_col == token_id):
        return labels[:, 1:]
    return labels

N_SAMPLES_30S = processor.feature_extractor.n_samples
NB_MAX_FRAMES = processor.feature_extractor.nb_max_frames

PAD_ID = processor.tokenizer.pad_token_id
DECODER_START_ID = None
SOT_ID = None  # set later by fix_whisper_special_tokens()


def decode_mono_16k(path: str):
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=False)
        if sr != TARGET_SR:
            return None, None
        if wav.ndim == 2:
            wav = wav.mean(axis=-1)

        wav = np.asarray(wav, dtype=np.float32)
        if not np.isfinite(wav).all():
            wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

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

    # Keep meta so we can pinpoint the exact file when CE loss blows up.
    meta_audio = []
    meta_audio_orig = []
    meta_uid = []
    meta_row_index = []
    meta_text = []

    # PATCH 6: Track keys and audio used
    keys_used = []
    audio_used = []

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

        w30 = pad_or_trim_to_30s(wav)
        if not np.isfinite(w30).all():
            # decode_mono_16k already nan_to_num()s, so this should be rare.
            print(f"BAD AUDIO (non-finite) at: {ap}")
            print(f"  - Audio shape: {w30.shape}")
            print(f"  - Contains NaN: {np.isnan(w30).any()}")
            print(f"  - Contains Inf: {np.isinf(w30).any()}")
            print(f"  - Min/Max values: [{np.nanmin(w30):.6f}, {np.nanmax(w30):.6f}]")
            print(f"  - Text: '{txt[:100]}...'")
            continue

        lengths_sec.append(dur)
        waveforms_30s.append(w30)
        texts.append(txt)

        meta_audio.append(ap)
        meta_audio_orig.append(ex.get("audio_path_orig") or ap)
        meta_uid.append(ex.get("uid"))
        meta_row_index.append(ex.get("_epoch_row_index"))
        meta_text.append(txt)

        # PATCH 6: Track keys and audio used for each accepted sample
        keys_used.append(ex.get("key") or canonical_audio_key(ex.get("audio_path_orig") or ap))
        audio_used.append(ex.get("audio_path_orig") or ap)

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
    
    # CRITICAL: Strip leading SOT token from labels to prevent pathological training
    labels = strip_leading_token(labels, token_id=SOT_ID, pad_id=PAD_ID)
    
    # Also adjust attention_mask if we stripped a token
    attn_mask = tok["attention_mask"]
    if attn_mask is not None and attn_mask.shape[1] == labels.shape[1] + 1:
        attn_mask = attn_mask[:, 1:]

    # Check the potentially sliced attention mask, not the original
    if attn_mask is None or attn_mask.sum().item() == 0:
        return None

    return {
        "lengths": torch.tensor(lengths_sec, dtype=torch.float32),
        "input_features": feats.input_features,
        "labels": labels,
        "attention_mask": attn_mask,  # Use adjusted attention mask
        "meta_audio": meta_audio,
        "meta_audio_orig": meta_audio_orig,
        "meta_uid": meta_uid,
        "meta_row_index": meta_row_index,
        "meta_text": meta_text,
        "keys": keys_used,  # PATCH 6: Return keys used
        "audio_used": audio_used,  # PATCH 6: Return audio paths used
    }
# ============================================
# CELL 9/9 — Main loop with async prefetch (works in BOTH modes)
# ============================================

# Fail fast with a helpful error if the Stage-18 middle section is missing.
_required = ["build_loader_from_rows", "train_one_epoch", "save_checkpoint"]
_missing = [name for name in _required if name not in globals()]
if _missing:
    raise SystemExit(
        "This file is missing core training functions: " + ", ".join(_missing) + "\n"
        "You likely copied only parts of your Stage 18 script.\n"
        "Fix: re-add the missing middle section (models/optimizer/scheduler + train loop + checkpoint code) "
        "from your working Stage 18 script, then re-run."
    )


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


# PATCH 8: Pointer persistence with run state and manifest signature checking
sig = manifest_signature(MANIFEST_PATH)
run_state = load_run_state()

if run_state.get("manifest_sig") and run_state.get("manifest_sig") != sig:
    print("[warn] manifest changed since last run; resetting pointer/epoch and clearing pending plan")
    run_state = {}
    clear_pending_plan()

pointer = int(run_state.get("pointer", 0) or 0)
epoch = int(run_state.get("epoch", epoch_num_start) or epoch_num_start)

# bump pointer forward if it lands on already-trained keys
while pointer < len(manifest_rows):
    k = manifest_rows[pointer].get("key") or canonical_audio_key(manifest_rows[pointer].get("audio_path"))
    if k and k in trained_set:
        pointer += 1
        continue
    break

save_run_state({"manifest_sig": sig, "pointer": pointer, "epoch": epoch})

if pointer >= len(manifest_rows):
    print("All files are trained.")
    raise SystemExit(0)

print("Starting pointer:", pointer)

prefetch_exec = ThreadPoolExecutor(max_workers=max(1, int(PREFETCH_THREADS)))
prefetch_future = None

try:
    while epoch < MAX_EPOCHS:
        if USE_LOCAL_CACHE:
            cleanup_old_epoch_dirs(LOCAL_EPOCH_CACHE_ROOT, keep_last_k=KEEP_LAST_LOCAL_EPOCH_DIRS)

        # PATCH 8: Pending plan resume logic
        pending = load_pending_plan()
        if pending and pending.get("manifest_sig") == sig and int(pending.get("epoch", -1)) == int(epoch):
            print(f"\n[resume] Using pending epoch plan for epoch {epoch} (pointer {pending.get('pointer_start')} -> {pending.get('pointer_end')})")
            current_rows = pending.get("rows") or []
            pointer_end = int(pending.get("pointer_end", pointer))
            meta = pending.get("meta") or {}
        else:
            pointer_start = int(pointer)
            # your existing selection/caching code, BUT store the returned pointer separately:
            if USE_LOCAL_CACHE:
                epoch_dir = os.path.join(LOCAL_EPOCH_CACHE_ROOT, f"epoch_{epoch:06d}")
                print(f"\nPreparing epoch {epoch} cache -> {epoch_dir}")
                current_rows, pointer_end, meta = prepare_epoch_cache(
                    pointer=pointer_start,
                    trained_set=trained_set,
                    need=N_SAMPLES_PER_EPOCH,
                    manifest_rows=manifest_rows,
                    epoch_dir=epoch_dir,
                )
                print(f"Epoch {epoch} cache ready: kept={len(current_rows)} | {meta}")
            else:
                print(f"\nSelecting epoch {epoch} rows (no cache) ...")
                current_rows, pointer_end, meta = select_epoch_rows(
                    pointer=pointer_start,
                    trained_set=trained_set,
                    need=N_SAMPLES_PER_EPOCH,
                    manifest_rows=manifest_rows,
                )
                print(f"Epoch {epoch} rows ready: kept={len(current_rows)} | {meta}")
            save_pending_plan({
                "manifest_sig": sig,
                "epoch": int(epoch),
                "pointer_start": int(pointer_start),
                "pointer_end": int(pointer_end),
                "rows": current_rows,
                "meta": meta,
            })

        if prefetch_future is not None and prefetch_future.done():
            prefetch_future.result()  # consume but ignore, we're using pending plan
            prefetch_future = None
        else:
            if prefetch_future is None:
                if USE_LOCAL_CACHE:
                    next_epoch_dir = os.path.join(LOCAL_EPOCH_CACHE_ROOT, f"epoch_{(epoch+1):06d}")
                    print(f"Prefetching epoch {epoch+1} (copy-cache) -> {next_epoch_dir}")
                    prefetch_future = prefetch_exec.submit(
                        prepare_epoch_cache,
                        pointer_end,  # Use pointer_end for next epoch
                        trained_set,
                        N_SAMPLES_PER_EPOCH,
                        manifest_rows,
                        next_epoch_dir,
                    )
                else:
                    print(f"Prefetching epoch {epoch+1} (no cache) ...")
                    prefetch_future = prefetch_exec.submit(
                        select_epoch_rows,
                        pointer_end,  # Use pointer_end for next epoch
                        trained_set,
                        N_SAMPLES_PER_EPOCH,
                        manifest_rows,
                    )

        if not current_rows:
            print("No more usable samples. Stopping.")
            break

        loader = build_loader_from_rows(current_rows)
        avg_loss, trained_keys_epoch = train_one_epoch(epoch, loader)
        print(f"Epoch {epoch} avg loss: {avg_loss:.6f}")

        # PATCH 8: append only keys that actually contributed to optimiser steps
        trained_to_append = [{"key": k, "epoch": int(epoch), "ts": time.time()} for k in sorted(trained_keys_epoch)]
        append_jsonl(TRAINED_JSONL_PATH, trained_to_append)
        trained_set.update(trained_keys_epoch)

        save_checkpoint(epoch, subset_start_idx=-1, subset_count=len(trained_keys_epoch))

        # PATCH 8: COMMIT pointer + advance epoch only after successful checkpoint + trained write
        pointer = int(pointer_end)
        clear_pending_plan()
        epoch += 1
        save_run_state({"manifest_sig": sig, "pointer": pointer, "epoch": epoch})

        if DELETE_TRAINED_FROM_DRIVE:
            drive_paths = [r.get("audio_path") for r in current_rows if r.get("audio_path")]
            cleanup_trained_drive_audio(drive_paths, DRIVE_CLEANUP_MODE, DRIVE_ALLOWED_PREFIX, DRIVE_ARCHIVE_DIR)

        del loader
        del current_rows
        cleanup_memory(f"after epoch {epoch}")

finally:
    try:
        prefetch_exec.shutdown(wait=False, cancel_futures=True)
    except Exception:
        try:
            prefetch_exec.shutdown(wait=False)
        except Exception:
            pass

print("Done.")

# =============================
# OPTIONAL: one-off repair of trained_stage1.jsonl
# =============================

def repair_trained_jsonl(in_path: str, out_path: str):
    """Deduplicate + convert older {audio_path: ...} lines into {key: ...} lines."""
    rows = read_jsonl(in_path)
    seen = set()
    fixed = []
    for r in rows:
        key = (r.get("key") or r.get("uid") or "").strip()
        if not key:
            key = canonical_audio_key(r.get("audio_path"))
        if not key or key in seen:
            continue
        seen.add(key)
        fixed.append({"key": key, "epoch": r.get("epoch"), "ts": r.get("ts")})

    append_jsonl(out_path, fixed)
    print(f"Wrote {len(fixed)} unique trained keys -> {out_path}")


# Usage (Windows paths example):
# repair_trained_jsonl(
#   in_path="i:/Record_chunks/trained_stage1.jsonl",
#   out_path="i:/Record_chunks/trained_stage1_repaired.jsonl",
# )
beep_notification(1000, 500)
time.sleep(0.1)
beep_notification(1200, 500)
