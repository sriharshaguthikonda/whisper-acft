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


epoch_num_start = 0  # default; will be overridden if checkpoints exist

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
MANIFEST_PATH = "I:/Record_chunks/pairs_manifest_combined_train_with_tempo_pause_randomized_updated.jsonl"

# Checkpoints (single fixed directory; no run tags).
CHECKPOINT_DIR = "i:/Stage_17_shuffle_wer_acft_checkpoints_partialctx_tiny_en_14"

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


def _append_jsonl(path: str, obj: dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        print("[warn] failed writing JSONL:", path, repr(e))


def log_nonfinite_ce(keys, n_ctx: int, lr_now: float, note: str = ""):
    rec = {
        "ts": time.time(),
        "note": note,
        "n_ctx": int(n_ctx),
        "lr": float(lr_now),
        "keys": list(keys) if keys is not None else None,
    }
    _append_jsonl(NONFINITE_CE_LOG_JSONL, rec)


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
FUTO_MODEL_ID = "futo-org/acft-whisper-small.en"
PROCESSOR_ID = "openai/whisper-small.en"  # must match the base family

# ===== Combined WER + ACFT knobs =====
# Goal: keep WER going down, while ACFT prevents repetition / collapse when audio_ctx is dynamic.
# Direct script parameters - modify these values as needed
ACFT_REFERENCE_MODEL_ID = FUTO_MODEL_ID  # Reference model for ACFT (default: same as training model)
CE_LABEL_SMOOTH = 0.05  # Label smoothing for CE loss (0.0 = disabled, 0.05 = recommended)
LAMBDA_CE = 1.0  # Weight for cross-entropy loss
LAMBDA_ACFT = 0.10  # Weight for ACFT robustness loss
# Optional: temporarily disable ACFT entirely while debugging instability.
# LAMBDA_ACFT = 0.0
# Optional ramp: start ACFT small then ramp up over first N optimizer steps
ACFT_RAMP_STEPS = 800  # 0 disables ramp; higher values ramp more slowly

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
MAX_GRAD_NORM = 0.5  # Conservative clipping while debugging instability

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
# Runtime toggle (recommended on GTX 1660 if you see NaNs): set WHISPER_DISABLE_AMP=1
DISABLE_AMP = bool(int(os.environ.get("WHISPER_DISABLE_AMP", "1")))
use_amp = (device == "cuda") and (not DISABLE_AMP)
amp_dtype = torch.float16
# Enable GradScaler only when autocast is enabled
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
    
    # NOTE:
    # We DO NOT return a decoder attention mask here, because the decoder inputs are shifted
    # (decoder_input_ids = shift_tokens_right(...)) inside the training loop.
    # Using tokenizer's attention_mask directly for the decoder is WRONG when right-padding:
    # it masks out the last real token after shifting.
    # We'll build decoder_attention_mask from decoder_input_ids in the train loop.
    labels_attention_mask = tok.get("attention_mask")
    if labels_attention_mask is None or labels_attention_mask.sum().item() == 0:
        return None

    return {
        "lengths": torch.tensor(lengths_sec, dtype=torch.float32),
        "input_features": feats.input_features,
        "labels": labels,
        "labels_attention_mask": labels_attention_mask,
        "meta_audio": meta_audio,
        "meta_audio_orig": meta_audio_orig,
        "meta_uid": meta_uid,
        "meta_row_index": meta_row_index,
        "meta_text": meta_text,
        "keys": keys_used,  # PATCH 6: Return keys used
        "audio_used": audio_used,  # PATCH 6: Return audio paths used
    }


# ============================================
# CELL 5/9 — Models + optimizer + scheduler (RESTORED)
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
    print("Loading student from base:", PROCESSOR_ID)
    return WhisperForConditionalGeneration.from_pretrained(PROCESSOR_ID, generation_config=gen_config)


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
epoch_num_start = max(int(trained_based_epoch), int(ckpt_based_epoch))
print(f"Auto epoch start: {epoch_num_start} (trained_based={trained_based_epoch}, ckpt_based={ckpt_based_epoch})")

model_train = load_student_from_checkpoint_or_base(latest_model_dir)
model_train.to(device)
model_train.train()

# Fix critical token configuration issues
SOT_ID = fix_whisper_special_tokens(processor, model_train)
DECODER_START_ID = SOT_ID

want_acft = (float(LAMBDA_ACFT) > 0.0)
if want_acft:
    model_ref = WhisperForConditionalGeneration.from_pretrained(ACFT_REFERENCE_MODEL_ID, generation_config=gen_config)
    model_ref.to(device)
    model_ref.eval()
    for p in model_ref.parameters():
        p.requires_grad_(False)
else:
    model_ref = None

optimizer = torch.optim.AdamW(model_train.parameters(), lr=float(LR_START))

# Track optimizer steps globally
global_step = 0

# Scheduler (warmup + epoch decay)
scheduler = None
if USE_SCHEDULER:
    def lr_mult(step: int) -> float:
        if step < WARMUP_STEPS:
            return float(step) / max(1.0, float(WARMUP_STEPS))
        post = step - WARMUP_STEPS
        epoch_i = int(post // max(1, OPT_STEPS_PER_EPOCH))
        mult = (DECAY_GAMMA ** epoch_i)
        floor_mult = float(LR_FLOOR) / float(LR_START)
        if mult < floor_mult:
            mult = floor_mult
        return float(mult)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_mult)
    print("[lr] Scheduler enabled: warmup + epoch decay")
else:
    print("[lr] Scheduler disabled: constant LR")

# Resume optimizer/scaler/scheduler/global_step
if latest_state is not None:
    try:
        if latest_state.get("optimizer") is not None:
            optimizer.load_state_dict(latest_state["optimizer"])
            print("Resumed optimizer state.")
        if use_amp and latest_state.get("scaler") is not None:
            scaler.load_state_dict(latest_state["scaler"])
            print("Resumed GradScaler state.")
        global_step = int(latest_state.get("global_step", 0) or 0)
        if scheduler is not None and latest_state.get("scheduler") is not None:
            try:
                scheduler.load_state_dict(latest_state["scheduler"])
                print("Resumed scheduler state.")
            except Exception as e:
                print("WARNING: failed to resume scheduler:", repr(e))
    except Exception as e:
        print("WARNING: failed to resume optimizer/scaler:", repr(e))

if RESUME_CLAMP_LR:
    clamp_optimizer_lr(optimizer, float(LR_START))
    if scheduler is not None and hasattr(scheduler, 'base_lrs'):
        scheduler.base_lrs = [min(float(b), float(LR_START)) for b in scheduler.base_lrs]
    print(f"[lr] Clamped LR to LR_START on resume: {LR_START} (global_step={global_step})")

cleanup_memory("after model load")


# ============================================
# CELL 6/9 — Partial encoder + losses (RESTORED)
# ============================================

FULL_ENCODER_CONTEXT_LENGTH = int(getattr(model_train.config, "max_source_positions", 1500))


def compute_partially_encoder(whisper_model, input_features: torch.Tensor, n_audio_ctx: int):
    target_mel_seq_len = 2 * int(n_audio_ctx)
    diff = target_mel_seq_len - input_features.shape[2]

    if diff > 0:
        input_features = F.pad(input_features, (0, diff, 0, 0, 0, 0), "constant", 0.0)
    elif diff < 0:
        input_features = input_features[:, :, :target_mel_seq_len]

    if int(n_audio_ctx) == int(FULL_ENCODER_CONTEXT_LENGTH):
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


def pick_n_ctx_from_batch(
    lengths_sec: torch.Tensor,
    max_embed_positions: int,
    *,
    full_ctx: int,
    safety_sec: float = 0.20,
    round_to: int = 16,
    jitter_max: int = 64,
) -> int:
    if not lengths_sec.numel():
        return int(min(full_ctx, max_embed_positions))

    max_len = float(lengths_sec.max().item())
    base = (float(full_ctx) / 30.0) * (max_len + float(safety_sec))
    base = int(math.ceil(base))

    if round_to and round_to > 1:
        base = int(math.ceil(base / round_to) * round_to)

    base = max(1, min(base, max_embed_positions))

    jitter = min(int(jitter_max), max(0, base // 10))
    if jitter > 0:
        add = int(torch.randint(0, jitter + 1, (1,), device=lengths_sec.device).item())
        n_ctx = base + add
    else:
        n_ctx = base

    return int(max(1, min(n_ctx, max_embed_positions)))


def masked_hidden_mse(hs_pred: torch.Tensor, hs_tgt: torch.Tensor, attn_mask: torch.Tensor):
    hs_pred = hs_pred.float()
    hs_tgt = hs_tgt.float()
    mask = attn_mask.unsqueeze(0).unsqueeze(-1).to(dtype=torch.float32)  # [1,B,T,1]
    diff2 = (hs_pred - hs_tgt).pow(2) * mask
    denom = mask.sum() * float(hs_pred.shape[0]) * float(hs_pred.shape[-1])
    return diff2.sum(dtype=torch.float32) / denom.clamp_min(1.0)


def get_lm_head(model: WhisperForConditionalGeneration):
    if hasattr(model, "proj_out"):
        return model.proj_out
    if hasattr(model, "lm_head"):
        return model.lm_head
    raise AttributeError("No lm head found (expected proj_out or lm_head)")


# ============================================
# CELL 7/9 — Checkpointing (RESTORED)
# ============================================


def save_checkpoint(epoch_num: int, subset_start_idx: int, subset_count: int):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    model_dir = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch_num:06d}")
    os.makedirs(model_dir, exist_ok=True)

    # Save model weights/config
    model_train.to("cpu").save_pretrained(model_dir)
    model_train.to(device)

    state = {
        "epoch": int(epoch_num),
        "subset_start_idx": int(subset_start_idx),
        "subset_count": int(subset_count),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if use_amp else None,
        "global_step": int(global_step),
        "ts": time.time(),
    }

    state_path = os.path.join(CHECKPOINT_DIR, f"training_state_epoch_{epoch_num:06d}.pt")
    torch.save(state, state_path)
    print("✓ Saved checkpoint:", model_dir)


# ============================================
# CELL 8/9 — Training loop (RESTORED)
# ============================================


def build_loader_from_rows(rows_for_epoch):
    if USE_LOCAL_CACHE:
        audio_paths = [r["audio_path_local"] for r in rows_for_epoch]
    else:
        audio_paths = [r["audio_path"] for r in rows_for_epoch]

    slim = []
    for ap_use, r in zip(audio_paths, rows_for_epoch):
        slim.append({
            "audio": ap_use,
            "raw_transcription": r.get("raw_transcription", ""),
            "audio_path_orig": r.get("audio_path"),
            "key": r.get("key") or canonical_audio_key(r.get("audio_path")),
            "uid": r.get("uid"),
        })

    ds = Dataset.from_list(slim)

    return DataLoader(
        ds,
        batch_size=int(BATCH_SIZE),
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=int(NUM_WORKERS),
        pin_memory=(device == "cuda"),
        drop_last=False,
    )


def _isfinite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def train_one_epoch(epoch_num: int, loader):
    global global_step

    model_train.train()
    optimizer.zero_grad(set_to_none=True)
    lm_head = get_lm_head(model_train)

    # Track unscaled loss so avg_loss is comparable across different GRAD_ACCUM_STEPS.
    running = 0.0
    n_batches = 0

    accum = 0
    trained_keys_epoch = set()
    accum_keys = []

    pbar = tqdm(loader, desc=f"Epoch {epoch_num}", leave=False)

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue

        input_features = batch["input_features"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        # Build decoder attention mask from decoder_input_ids (correct for right-padding + shifting)
        # NOTE: we keep labels_attention_mask only for debugging/inspection.
        labels_attention_mask = batch.get("labels_attention_mask")
        lengths = batch["lengths"].to(device, non_blocking=True)
        keys = batch.get("keys") or []

        max_embed_positions = int(model_train.model.encoder.embed_positions.weight.shape[0])
        if FORCE_FULL_AUDIO_CTX:
            n_ctx = int(FULL_ENCODER_CONTEXT_LENGTH)
        else:
            n_ctx = pick_n_ctx_from_batch(
                lengths,
                max_embed_positions,
                full_ctx=int(FULL_ENCODER_CONTEXT_LENGTH),
                safety_sec=float(AUDIO_CTX_SAFETY_SEC),
                round_to=int(AUDIO_CTX_ROUND_TO),
                jitter_max=int(AUDIO_CTX_JITTER_MAX),
            )

        # Prepare decoder inputs
        labels_for_shift = labels.clone()
        labels_for_shift[labels_for_shift == -100] = int(PAD_ID)
        decoder_input_ids = shift_tokens_right(labels_for_shift, int(PAD_ID), int(DECODER_START_ID))
        decoder_attention_mask = (decoder_input_ids != int(PAD_ID)).to(dtype=torch.long, device=device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            enc_partial = compute_partially_encoder(model_train.model, input_features, int(n_ctx))
            dec_partial = model_train.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=enc_partial,
                output_hidden_states=True,
                use_cache=False,
            )
            logits = lm_head(dec_partial.last_hidden_state)
            loss_ce = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
                label_smoothing=float(CE_LABEL_SMOOTH),
            )

        lr_now = float(optimizer.param_groups[0].get("lr", 0.0))
        if not _isfinite(loss_ce):
            print("[warn] non-finite CE loss; skipping batch")
            if DEBUG_NONFINITE_CE:
                log_nonfinite_ce(keys, n_ctx=int(n_ctx), lr_now=lr_now, note="non-finite CE")
            optimizer.zero_grad(set_to_none=True)
            accum = 0
            accum_keys = []
            continue

        if want_acft:
            # Reference forward pass in full precision for stability
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                enc_full = compute_partially_encoder(model_ref.model, input_features, int(FULL_ENCODER_CONTEXT_LENGTH))
                dec_full = model_ref.model.decoder(
                    input_ids=decoder_input_ids,
                    attention_mask=decoder_attention_mask,
                    encoder_hidden_states=enc_full,
                    output_hidden_states=True,
                    use_cache=False,
                )

            hs_p = torch.stack(dec_partial.hidden_states, dim=0)
            hs_f = torch.stack(dec_full.hidden_states, dim=0)

            if (not _isfinite(hs_p)) or (not _isfinite(hs_f)):
                print("[warn] non-finite hidden states in ACFT; skipping batch")
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                accum_keys = []
                continue

            loss_acft = masked_hidden_mse(hs_p, hs_f, decoder_attention_mask)
            if not _isfinite(loss_acft):
                print("[warn] non-finite ACFT loss; skipping batch")
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                accum_keys = []
                continue
        else:
            loss_acft = torch.zeros((), device=device)

        if want_acft and int(ACFT_RAMP_STEPS) > 0:
            ramp = min(1.0, float(global_step) / float(max(1, int(ACFT_RAMP_STEPS))))
            lambda_acft_eff = float(LAMBDA_ACFT) * ramp
        else:
            lambda_acft_eff = float(LAMBDA_ACFT)

        loss = (float(LAMBDA_CE) * loss_ce) + (lambda_acft_eff * loss_acft)
        loss = loss / float(GRAD_ACCUM_STEPS)

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        running += float(loss.detach().cpu().item()) * float(GRAD_ACCUM_STEPS)
        n_batches += 1
        accum += 1
        accum_keys.extend(list(keys))

        # step
        if accum >= int(GRAD_ACCUM_STEPS):
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), float(MAX_GRAD_NORM))

            did_optim_step = True
            if use_amp:
                scale_before = float(scaler.get_scale())
                scaler.step(optimizer)
                scaler.update()
                scale_after = float(scaler.get_scale())
                did_optim_step = (scale_after >= scale_before)
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and did_optim_step:
                scheduler.step()

            global_step += 1
            trained_keys_epoch.update(accum_keys)
            accum_keys = []
            accum = 0

        lr_now = float(optimizer.param_groups[0].get("lr", 0.0))
        pbar.set_postfix({
            "loss": float(loss.detach().cpu().item()) * float(GRAD_ACCUM_STEPS),
            "ce": float(loss_ce.detach().cpu().item()),
            "acft": float(loss_acft.detach().cpu().item()) if want_acft else 0.0,
            "n_ctx": int(n_ctx),
            "bs": int(input_features.shape[0]),
            "lr": lr_now,
        })

    # Final flush: don't throw away the last partial accumulation.
    # Without this, you permanently skip a tail of samples every epoch (pointer advances).
    if accum > 0:
        scale_fix = float(GRAD_ACCUM_STEPS) / float(accum)

        if use_amp:
            scaler.unscale_(optimizer)
        # Rescale grads so the effective divisor is `accum` not `GRAD_ACCUM_STEPS`.
        for p in model_train.parameters():
            if p.grad is not None:
                p.grad.mul_(scale_fix)

        torch.nn.utils.clip_grad_norm_(model_train.parameters(), float(MAX_GRAD_NORM))

        did_optim_step = True
        if use_amp:
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            did_optim_step = (scale_after >= scale_before)
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None and did_optim_step:
            scheduler.step()

        global_step += 1
        trained_keys_epoch.update(accum_keys)

    avg_loss = (running / max(1, n_batches))
    return avg_loss, trained_keys_epoch


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
