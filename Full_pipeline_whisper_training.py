# %% [markdown]
# # Colab: run stages 1 -> 17 (full script)
# Paste into Colab: lines starting with `# %%` are cell separators.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# ---------- CONFIG (EDIT THESE) ----------
REPO_URL = "https://github.com/sriharshaguthikonda/whisper-acft.git"
REPO_DIR_DRIVE = "/content/drive/MyDrive/whisper-acft"
REPO_DIR_LOCAL = "/content/whisper-acft"
USE_LOCAL_REPO = True  # keep repo on local disk (avoid Drive writes)

REPO_DIR = REPO_DIR_LOCAL if USE_LOCAL_REPO else REPO_DIR_DRIVE

DATA_ROOT_DRIVE = "/content/drive/MyDrive"  # <-- change to your data root on Drive
LOCAL_DATA_ROOT = "/content/whisper-data"   # local (faster) copy
USE_LOCAL_DATA = True                       # copy data locally and run from there

DATA_ROOT = LOCAL_DATA_ROOT if USE_LOCAL_DATA else DATA_ROOT_DRIVE

# Save key manifests back to Drive to survive crashes/disconnects.
SYNC_IMPORTANT_TO_DRIVE = True
DRIVE_SYNC_ROOT = f"{DATA_ROOT_DRIVE}/pipeline_checkpoints"
PIPELINE_STATE_FILE = f"{DRIVE_SYNC_ROOT}/pipeline_state.json"

# Logging
PIPELINE_LOG = f"{DATA_ROOT}/pipeline.log"
DRIVE_SUMMARY_LOG = f"{DRIVE_SYNC_ROOT}/pipeline_summary.log"

# Caches (avoid re-downloading HF/transformers/torch assets)
CACHE_ROOT = "/content/cache"
HF_HOME = f"{CACHE_ROOT}/hf"
TRANSFORMERS_CACHE = f"{CACHE_ROOT}/transformers"
TORCH_HOME = f"{CACHE_ROOT}/torch"
SYNC_CACHE_TO_DRIVE = True
CACHE_DRIVE_ROOT = f"{DATA_ROOT_DRIVE}/cache"

# Copy strategy
COPY_TRANSCRIPTS_FOR_STAGE1 = True
COPY_AUDIO_FOR_STAGE1 = False  # set True if stage 1 should use local audio
BACKGROUND_COPY_LATER = False  # set True to copy non-stage1 deps in background

# Output verification
VERIFY_OUTPUTS = True
VERIFY_JSONL_SAMPLE = 5

# Optional: rclone (faster on huge transfers; requires rclone config)
USE_RCLONE = False
RCLONE_REMOTE = ""  # e.g. "gdrive:MyDrive"
RCLONE_TRANSFERS = 8

TRANSCRIPT_DIR = f"{DATA_ROOT}/Transcriptions_corrected"
AUDIO_SOURCE_DIR = f"{DATA_ROOT}/Record_harsha"

TRANSCRIPT_DIR_DRIVE = f"{DATA_ROOT_DRIVE}/Transcriptions_corrected"
TRANSCRIPT_DIR_LOCAL = f"{LOCAL_DATA_ROOT}/Transcriptions_corrected"
AUDIO_SOURCE_DIR_DRIVE = f"{DATA_ROOT_DRIVE}/Record_harsha"
AUDIO_SOURCE_DIR_LOCAL = f"{LOCAL_DATA_ROOT}/Record_harsha"

if USE_LOCAL_DATA:
    TRANSCRIPT_DIR_STAGE1 = TRANSCRIPT_DIR_LOCAL if COPY_TRANSCRIPTS_FOR_STAGE1 else TRANSCRIPT_DIR_DRIVE
    AUDIO_SOURCE_DIR_STAGE1 = AUDIO_SOURCE_DIR_LOCAL if COPY_AUDIO_FOR_STAGE1 else AUDIO_SOURCE_DIR_DRIVE
else:
    TRANSCRIPT_DIR_STAGE1 = TRANSCRIPT_DIR
    AUDIO_SOURCE_DIR_STAGE1 = AUDIO_SOURCE_DIR
CHUNKS_DIR = f"{DATA_ROOT}/Record_chunks"

TARGET_REF_DIR = f"{DATA_ROOT}/Record_only_by_harsha"
OTHER_REF_DIR = f"{DATA_ROOT}/Record_others_compacted"  # set "" to disable
OTHER_VOICES_DIR = f"{DATA_ROOT}/Record_others_compacted"

NOISE_DIR = f"{DATA_ROOT}/noise/RIRS_NOISES/pointsource_noises"
RIR_DIR = f"{DATA_ROOT}/noise/RIRS_NOISES/real_rirs_isotropic_noises"

TEST_CHUNKS_DIR = f"{DATA_ROOT}/Record_test_chunks"

CHECKPOINT_DIR = f"{DATA_ROOT}/stage17_checkpoints"
STAGE17_SCRIPT = "stage_17_WER_acft_Whisper_Futo_finetuned_model_training_only_local_en_version_only_qat_dora.py"
START_FRESH = 0  # 1 = refuse to resume if checkpoints exist

HF_TOKEN = ""  # optional; set in env or enter when prompted
DEVICE = "cuda"  # "cpu" if no GPU
SPEAKER_SORT_DRY_RUN = True  # True = only produce CSV, no moves/copies
USE_ADVANCED_RANDOMIZE = True  # True -> stage_15_b

RANDOM_SEED = 1337

# Augmentation ratios / copies
NOISE_RATIO, NOISE_COPIES = 0.5, 1
VOICE_RATIO, VOICE_COPIES = 0.8, 1
GAIN_RATIO,  GAIN_COPIES  = 0.1, 1
REVERB_RATIO, REVERB_COPIES = 0.3, 1
TEMPO_RATIO, TEMPO_COPIES = 0.3, 1

BOTTOM_PERCENT = 30.0
TEST_RATIO = 0.1

SPEAKER_THRESHOLD = 0.10
SPEAKER_WORKERS = 4
SPEAKER_BATCH = 16

AUTO_WORKERS = True  # set False to keep the fixed values above

# Stage toggles (set True to skip)
SKIP_STAGE_1 = False
SKIP_STAGE_2 = False
SKIP_STAGE_3 = False
SKIP_STAGE_3B = False
SKIP_STAGE_4 = False
SKIP_STAGE_6 = False
SKIP_STAGE_7 = False
SKIP_STAGE_8 = False
SKIP_STAGE_9 = False
SKIP_STAGE_10B = False
SKIP_STAGE_12 = False
SKIP_STAGE_13 = False
SKIP_STAGE_14 = False
SKIP_STAGE_15 = False
SKIP_STAGE_16 = False
SKIP_STAGE_17 = False

# %%
# ---------- DERIVED PATHS ----------
TASKS_PENDING = f"{CHUNKS_DIR}/tasks_pending.jsonl"
PAIRS_PENDING = f"{CHUNKS_DIR}/pairs_pending.jsonl"

STAGE2_MANIFEST  = f"{CHUNKS_DIR}/pairs_manifest_stereo.jsonl"
STAGE3B_MANIFEST = f"{CHUNKS_DIR}/pairs_manifest_stereo_english_only.jsonl"
STAGE4_MANIFEST  = f"{CHUNKS_DIR}/pairs_manifest_stereo_english_only_filtered.jsonl"

STAGE6_MANIFEST  = f"{CHUNKS_DIR}/pairs_manifest_stage6_noise.jsonl"
STAGE7_MANIFEST  = f"{CHUNKS_DIR}/pairs_manifest_stage7_voice.jsonl"
STAGE8_MANIFEST  = f"{CHUNKS_DIR}/pairs_manifest_stage8_gain.jsonl"
STAGE9_MANIFEST  = f"{CHUNKS_DIR}/pairs_manifest_stage9_reverb.jsonl"
STAGE10B_MANIFEST = f"{CHUNKS_DIR}/pairs_manifest_stage10b_tempo_pause.jsonl"

STAGE12_MANIFEST = f"{CHUNKS_DIR}/pairs_manifest_stage12_bottom_filtered.jsonl"
STAGE13_TRAIN    = f"{CHUNKS_DIR}/pairs_manifest_stage13_train.jsonl"
STAGE13_TEST     = f"{CHUNKS_DIR}/pairs_manifest_stage13_test.jsonl"
STAGE14_TRAIN    = f"{CHUNKS_DIR}/pairs_manifest_stage14_train_no_targets.jsonl"
STAGE15_TRAIN    = f"{CHUNKS_DIR}/pairs_manifest_stage15_train_no_targets_randomized.jsonl"

NOISE_OUT_DIR  = f"{CHUNKS_DIR}/noise_augmented"
VOICE_OUT_DIR  = f"{CHUNKS_DIR}/voice_augmented"
GAIN_OUT_DIR   = f"{CHUNKS_DIR}/gain_augmented"
REVERB_OUT_DIR = f"{CHUNKS_DIR}/reverb_augmented"
TEMPO_OUT_DIR  = f"{CHUNKS_DIR}/tempo_pause_augmented"
SEEN_DIR       = f"{CHUNKS_DIR}/_seen"

TARGET_OUT_DIR = f"{CHUNKS_DIR}/speaker_sorted/target"
OTHER_OUT_DIR  = f"{CHUNKS_DIR}/speaker_sorted/other"

SCORES_CSV = f"{REPO_DIR}/speaker_sort_scores.csv"
STATE_FILE = f"{CHUNKS_DIR}/speaker_sort_state.json"

# If you have a common-segments state file, point here.
# If it doesn't exist, stage 4 will just skip filtering.
COMMON_SEGMENTS_STATE = f"{REPO_DIR}/most_commonly_spoken_segments_state.json"

# %%
# ---------- SETUP ----------
import os, sys, subprocess, shlex, re, shutil, json, time, threading, hashlib
from pathlib import Path

os.environ["DEBIAN_FRONTEND"] = "noninteractive"
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ["TORCH_HOME"] = TORCH_HOME
Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN and not SKIP_STAGE_3:
    try:
        from getpass import getpass
        HF_TOKEN = getpass("HF_TOKEN (optional, for pyannote). Leave blank to skip: ").strip()
    except Exception:
        HF_TOKEN = ""
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

try:
    import orjson as _orjson  # type: ignore
    HAS_ORJSON = True
    def json_loads(s: str):
        return _orjson.loads(s)
except Exception:
    HAS_ORJSON = False
    def json_loads(s: str):
        return json.loads(s)

def log(msg: str) -> None:
    print(msg)
    try:
        Path(PIPELINE_LOG).parent.mkdir(parents=True, exist_ok=True)
        with open(PIPELINE_LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def log_drive_summary(msg: str) -> None:
    if not SYNC_IMPORTANT_TO_DRIVE:
        return
    try:
        Path(DRIVE_SUMMARY_LOG).parent.mkdir(parents=True, exist_ok=True)
        path = Path(DRIVE_SUMMARY_LOG)
        lines = []
        if path.exists():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception:
                lines = []
        lines.append(msg)
        # keep the tail small to reduce Drive churn
        lines = lines[-200:]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass

def run(cmd):
    if isinstance(cmd, str):
        log(f"$ {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    else:
        log("$ " + " ".join(shlex.quote(str(c)) for c in cmd))
        subprocess.run([str(c) for c in cmd], check=True)

def seed_manifest(src, dst):
    src_p, dst_p = Path(src), Path(dst)
    if not dst_p.exists():
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        dst_p.write_text(src_p.read_text(encoding="utf-8"), encoding="utf-8")

def replace_line(path, key, value):
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    new, n = re.subn(rf"^{re.escape(key)}\s*=.*$", f"{key} = {value!r}", txt, flags=re.M)
    if n == 0:
        raise RuntimeError(f"Could not find '{key} =' in {path}")
    p.write_text(new, encoding="utf-8")

def patch_winsound_stage14(path):
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if "_WinSoundShim" in txt:
        return
    new = txt.replace(
        "import winsound  # For beep notification",
        "try:\n    import winsound  # type: ignore\n"
        "except Exception:\n"
        "    class _WinSoundShim:\n"
        "        def Beep(self, *a, **k):\n            pass\n"
        "    winsound = _WinSoundShim()"
    )
    if new != txt:
        p.write_text(new, encoding="utf-8")
    else:
        print(f"warn: winsound import not found in {path}")

def patch_winsound_stage17(path):
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if "_WinSoundShim" in txt:
        return
    needle = "import os, json, time, shutil, hashlib, gc, math, winsound, sys, atexit, tempfile"
    repl = (
        "import os, json, time, shutil, hashlib, gc, math, sys, atexit, tempfile\n\n"
        "try:\n    import winsound  # type: ignore\n"
        "except Exception:\n"
        "    class _WinSoundShim:\n"
        "        def Beep(self, *a, **k):\n            pass\n"
        "    winsound = _WinSoundShim()"
    )
    if needle in txt:
        p.write_text(txt.replace(needle, repl), encoding="utf-8")
    else:
        print(f"warn: winsound import line not found in {path}")

def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def sync_files_to_drive(paths, label=""):
    if not (SYNC_IMPORTANT_TO_DRIVE and USE_LOCAL_DATA):
        return
    dst_root = Path(DRIVE_SYNC_ROOT)
    dst_root.mkdir(parents=True, exist_ok=True)
    src_root = Path(DATA_ROOT)
    copied = 0
    for p in paths:
        src = Path(p)
        if not src.exists():
            continue
        try:
            rel = src.relative_to(src_root)
            dst = dst_root / rel
        except ValueError:
            dst = dst_root / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            try:
                if dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
                    continue
            except Exception:
                pass
            try:
                if dst.stat().st_size == src.stat().st_size and file_hash(dst) == file_hash(src):
                    continue
            except Exception:
                pass
        shutil.copy2(src, dst)
        copied += 1
    if copied:
        log(f"[sync] {label} -> {dst_root} ({copied} file(s))")

def restore_files_from_drive(paths, label=""):
    if not (SYNC_IMPORTANT_TO_DRIVE and USE_LOCAL_DATA):
        return
    src_root = Path(DRIVE_SYNC_ROOT)
    dst_root = Path(DATA_ROOT)
    restored = 0
    for p in paths:
        dst = Path(p)
        try:
            rel = dst.relative_to(dst_root)
            src = src_root / rel
        except ValueError:
            src = src_root / dst.name
        if not src.exists():
            continue
        if dst.exists() and dst.stat().st_size > 0:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        restored += 1
    if restored:
        log(f"[restore] {label} <- {src_root} ({restored} file(s))")

def restore_cache_from_drive():
    if not SYNC_CACHE_TO_DRIVE:
        return
    src = Path(CACHE_DRIVE_ROOT)
    if not src.exists():
        return
    Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", "--info=progress2", f"{src}/", f"{CACHE_ROOT}/"])
    log(f"[cache] restored from {src}")

def sync_cache_to_drive():
    if not SYNC_CACHE_TO_DRIVE:
        return
    src = Path(CACHE_ROOT)
    if not src.exists():
        return
    dst = Path(CACHE_DRIVE_ROOT)
    dst.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", "--info=progress2", f"{src}/", f"{dst}/"])
    log(f"[cache] synced to {dst}")

COPY_EVENTS = {}
BACKGROUND_COPY_THREAD = None

def _drive_path_for_local(local_path: str) -> Path | None:
    try:
        rel = Path(local_path).relative_to(Path(DATA_ROOT))
    except Exception:
        return None
    return Path(DATA_ROOT_DRIVE) / rel

def copy_path_from_drive(local_path: str, label=""):
    src = _drive_path_for_local(local_path)
    if src is None:
        return
    dst = Path(local_path)
    if dst.exists():
        if dst.is_dir():
            try:
                if any(dst.iterdir()):
                    return
            except Exception:
                pass
        else:
            if dst.stat().st_size > 0:
                return
    if not src.exists():
        log(f"[copy] missing on Drive: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if USE_RCLONE and RCLONE_REMOTE:
            rel = src.relative_to(Path(DATA_ROOT_DRIVE))
            remote_src = f"{RCLONE_REMOTE}/{rel.as_posix()}"
            run(["rclone", "copy", remote_src, str(dst), "--transfers", str(RCLONE_TRANSFERS)])
        else:
            run(["rsync", "-a", "--info=progress2", f"{src}/", f"{dst}/"])
    else:
        shutil.copy2(src, dst)
    log(f"[copy] {label} {src} -> {dst}")

def start_background_copy(paths, label="bg"):
    global BACKGROUND_COPY_THREAD
    todo = [p for p in paths if p]
    for p in todo:
        COPY_EVENTS[str(p)] = threading.Event()

    def worker():
        for p in todo:
            copy_path_from_drive(p, label=label)
            ev = COPY_EVENTS.get(str(p))
            if ev:
                ev.set()

    BACKGROUND_COPY_THREAD = threading.Thread(target=worker, daemon=True)
    BACKGROUND_COPY_THREAD.start()

def wait_for_paths(paths):
    for p in paths:
        ev = COPY_EVENTS.get(str(p))
        if ev:
            ev.wait()

def ensure_local_paths(paths, label=""):
    wanted = [p for p in paths if p]
    if not wanted:
        return
    wait_for_paths(wanted)
    missing = [p for p in wanted if not Path(p).exists()]
    for p in missing:
        copy_path_from_drive(p, label=f"on-demand:{label}")
    missing = [p for p in wanted if not Path(p).exists()]
    if missing:
        raise RuntimeError(f"[{label}] required paths missing: {missing}")

def collect_jsonl_paths(jsonl_path: str, key: str) -> list[str]:
    out = set()
    p = Path(jsonl_path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json_loads(line)
            except Exception:
                continue
            val = obj.get(key)
            if isinstance(val, str) and val:
                out.add(val)
    return sorted(out)

def rel_paths_to_drive(paths: list[str]) -> list[str]:
    rels = []
    for p in paths:
        try:
            rel = Path(p).relative_to(Path(DATA_ROOT_DRIVE))
            rels.append(rel.as_posix())
            continue
        except Exception:
            pass
        try:
            rel = Path(p).relative_to(Path(DATA_ROOT))
            rels.append(rel.as_posix())
            continue
        except Exception:
            pass
    return rels

def rsync_files_from(rel_paths: list[str], src_root: str, dst_root: str, label=""):
    if not rel_paths:
        return
    tmp = Path("/tmp/rsync_files.txt")
    tmp.write_text("\n".join(rel_paths) + "\n", encoding="utf-8")
    run(["rsync", "-a", "--info=progress2", f"--files-from={tmp}", f"{src_root}/", f"{dst_root}/"])
    log(f"[copy] files-from {label}: {len(rel_paths)} files")

def rewrite_paths_in_jsonl(jsonl_path: str, old_root: str, new_root: str, keys: list[str]) -> None:
    p = Path(jsonl_path)
    if not p.exists():
        return
    tmp = p.with_suffix(p.suffix + ".tmp")
    changed = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f_in, tmp.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json_loads(line)
            except Exception:
                continue
            for k in keys:
                v = obj.get(k)
                if isinstance(v, str) and v.startswith(old_root):
                    obj[k] = new_root + v[len(old_root):]
                    changed += 1
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if changed:
        os.replace(tmp, p)
        log(f"[rewrite] {jsonl_path}: {changed} path(s) updated")
    else:
        tmp.unlink(missing_ok=True)

def load_pipeline_state() -> dict:
    if not SYNC_IMPORTANT_TO_DRIVE:
        return {"completed": []}
    p = Path(PIPELINE_STATE_FILE)
    if not p.exists():
        return {"completed": []}
    try:
        state = json_loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {"completed": []}
    if "completed" not in state:
        state["completed"] = []
        if state.get("status") == "done" and state.get("stage"):
            state["completed"].append(state["stage"])
    return state

PIPELINE_STATE = load_pipeline_state()

def save_pipeline_state(state: dict) -> None:
    if not SYNC_IMPORTANT_TO_DRIVE:
        return
    dst = Path(PIPELINE_STATE_FILE)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

def update_stage_state(stage, status):
    if not SYNC_IMPORTANT_TO_DRIVE:
        return
    state = PIPELINE_STATE
    completed = set(state.get("completed", []))
    if status == "done":
        completed.add(stage)
    state.update(
        {
            "stage": stage,
            "status": status,
            "timestamp": time.time(),
            "completed": sorted(completed),
        }
    )
    save_pipeline_state(state)
    log_drive_summary(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {stage} {status}")

EXPECTED_JSONL_KEYS = {
    "tasks_pending.jsonl": ["audio_path", "out_wav"],
    "pairs_pending.jsonl": ["audio_path"],
}

def expected_keys_for(path: Path) -> list[str] | None:
    name = path.name
    if name in EXPECTED_JSONL_KEYS:
        return EXPECTED_JSONL_KEYS[name]
    if name.startswith("pairs_manifest_"):
        return ["audio_path"]
    return None

def verify_jsonl(path: Path) -> bool:
    try:
        required_keys = expected_keys_for(path)
        parsed = 0
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json_loads(line)
                except Exception:
                    return False
                if required_keys:
                    for k in required_keys:
                        if k not in obj:
                            return False
                parsed += 1
                if parsed >= max(1, VERIFY_JSONL_SAMPLE):
                    break
        return parsed > 0
    except Exception:
        return False

def outputs_ok(paths) -> bool:
    for p in paths:
        path = Path(p)
        if path.is_dir():
            try:
                if not any(path.iterdir()):
                    return False
            except Exception:
                return False
        else:
            if not path.exists() or path.stat().st_size <= 0:
                return False
            if VERIFY_OUTPUTS and path.suffix.lower() == ".jsonl":
                if not verify_jsonl(path):
                    return False
    return True

def assert_outputs(paths, stage):
    if outputs_ok(paths):
        return
    missing = [str(p) for p in paths if not Path(p).exists()]
    raise RuntimeError(f"[{stage}] expected outputs missing/empty: {missing}")

SKIP_FLAGS = {
    "stage_1": SKIP_STAGE_1,
    "stage_2": SKIP_STAGE_2,
    "stage_3": SKIP_STAGE_3,
    "stage_3b": SKIP_STAGE_3B,
    "stage_4": SKIP_STAGE_4,
    "stage_6": SKIP_STAGE_6,
    "stage_7": SKIP_STAGE_7,
    "stage_8": SKIP_STAGE_8,
    "stage_9": SKIP_STAGE_9,
    "stage_10b": SKIP_STAGE_10B,
    "stage_12": SKIP_STAGE_12,
    "stage_13": SKIP_STAGE_13,
    "stage_14": SKIP_STAGE_14,
    "stage_15": SKIP_STAGE_15,
    "stage_16": SKIP_STAGE_16,
    "stage_17": SKIP_STAGE_17,
}

def should_run(stage, outputs):
    if SKIP_FLAGS.get(stage, False):
        log(f"[skip] {stage} (flag set)")
        update_stage_state(stage, "skipped")
        return False
    completed = set(PIPELINE_STATE.get("completed", []))
    if stage in completed and outputs_ok(outputs):
        log(f"[skip] {stage} already completed and outputs present")
        update_stage_state(stage, "skipped_existing")
        return False
    return True

def check_required_paths():
    missing = []
    def req(path, label):
        if not Path(path).exists():
            missing.append(f"{label}: {path}")

    if not SKIP_STAGE_1:
        req(TRANSCRIPT_DIR_STAGE1, "TRANSCRIPT_DIR_STAGE1")
        req(AUDIO_SOURCE_DIR_STAGE1, "AUDIO_SOURCE_DIR_STAGE1")

    if missing:
        raise RuntimeError("Missing required paths:\n" + "\n".join(missing))

# %%
# ---------- WORKER AUTOTUNE ----------
CPU_COUNT = os.cpu_count() or 2
MAX_WORKERS = max(1, CPU_COUNT)

# Use max CPU cores without oversubscription (safer on Colab)
FFMPEG_WORKERS = MAX_WORKERS
TASK_WORKERS = MAX_WORKERS
AUG_WORKERS = MAX_WORKERS
FFMPEG_THREADS = max(1, min(4, max(1, MAX_WORKERS // max(1, FFMPEG_WORKERS))))

if AUTO_WORKERS:
    SPEAKER_WORKERS = MAX_WORKERS

# %%
# ---------- CLONE / UPDATE ----------
if not Path(REPO_DIR).exists():
    run(["git", "clone", REPO_URL, REPO_DIR])
else:
    run(["git", "-C", REPO_DIR, "pull"])
os.chdir(REPO_DIR)

# %%
# ---------- SYSTEM DEPS ----------
run(["apt-get", "update", "-y"])
pkgs = ["ffmpeg", "sox", "rsync"]
if USE_RCLONE:
    pkgs.append("rclone")
run(["apt-get", "install", "-y"] + pkgs)

# %%
# ---------- RESTORE CACHE ----------
restore_cache_from_drive()

# %%
# ---------- COPY DATA TO LOCAL ----------
if USE_LOCAL_DATA:
    Path(LOCAL_DATA_ROOT).mkdir(parents=True, exist_ok=True)

    # Copy only what's needed for Stage 1 first (fast startup)
    stage1_paths = []
    if COPY_TRANSCRIPTS_FOR_STAGE1:
        stage1_paths.append(TRANSCRIPT_DIR_LOCAL)
    if COPY_AUDIO_FOR_STAGE1:
        stage1_paths.append(AUDIO_SOURCE_DIR_LOCAL)
    for p in stage1_paths:
        copy_path_from_drive(p, label="stage1")

    # Optional background copy (disabled by default)
    if BACKGROUND_COPY_LATER:
        later_paths = [
            TARGET_REF_DIR,
            OTHER_REF_DIR if OTHER_REF_DIR else "",
            OTHER_VOICES_DIR,
            NOISE_DIR,
            RIR_DIR,
        ]
        start_background_copy(later_paths, label="bg")

# %%
# ---------- RESTORE SYNCED FILES ----------
restore_files_from_drive([TASKS_PENDING, PAIRS_PENDING, STAGE2_MANIFEST], label="stage1/2")

# %%
# ---------- PYTHON DEPS ----------
PY = sys.executable
run([PY, "-m", "pip", "install", "-q", "-U", "pip"])
run([PY, "-m", "pip", "install", "-q",
     "transformers", "datasets", "accelerate",
     "soundfile", "librosa", "numpy", "pandas", "scipy", "tqdm", "orjson",
     "jiwer", "edlib", "pyannote.audio", "huggingface_hub"])

# %%
# ---------- PATCH WINDOWS-SPECIFIC PATHS ----------
replace_line("stage_1_Manifest_creation_local_only.py", "TRANSCRIPT_DIR", TRANSCRIPT_DIR_STAGE1)
replace_line("stage_1_Manifest_creation_local_only.py", "CHUNKS_DIR", CHUNKS_DIR)
replace_line("stage_1_Manifest_creation_local_only.py", "AUDIO_SOURCE_DIR", AUDIO_SOURCE_DIR_STAGE1)

replace_line("Stage_16_move_test_chunks_update_test_manifest.py", "manifest_path", STAGE13_TEST)
replace_line("Stage_16_move_test_chunks_update_test_manifest.py", "target_dir", TEST_CHUNKS_DIR)

replace_line(STAGE17_SCRIPT, "MANIFEST_PATH", STAGE15_TRAIN)
replace_line(STAGE17_SCRIPT, "CHECKPOINT_DIR", CHECKPOINT_DIR)

patch_winsound_stage14("stage_14_remove_target_files_from_manifest.py")
patch_winsound_stage17(STAGE17_SCRIPT)

# %%
# ---------- CONFIG SANITY CHECKS ----------
check_required_paths()





# %%
# ---------- STAGE 1 ----------
if should_run("stage_1", [TASKS_PENDING, PAIRS_PENDING]):
    update_stage_state("stage_1", "running")
    run([PY, "stage_1_Manifest_creation_local_only.py"])
    assert_outputs([TASKS_PENDING, PAIRS_PENDING], "stage_1")
    # If stage1 ran on Drive paths, rewrite to local root for downstream stages.
    if USE_LOCAL_DATA:
        if not COPY_AUDIO_FOR_STAGE1:
            rewrite_paths_in_jsonl(TASKS_PENDING, DATA_ROOT_DRIVE, DATA_ROOT, ["audio_path"])
            rewrite_paths_in_jsonl(PAIRS_PENDING, DATA_ROOT_DRIVE, DATA_ROOT, ["audio_path", "source_audio"])
        if not COPY_TRANSCRIPTS_FOR_STAGE1:
            rewrite_paths_in_jsonl(PAIRS_PENDING, DATA_ROOT_DRIVE, DATA_ROOT, ["transcript_json"])
    sync_files_to_drive([TASKS_PENDING, PAIRS_PENDING], label="stage1")
    update_stage_state("stage_1", "done")




# %%
# ---------- STAGE 2 ----------
if should_run("stage_2", [STAGE2_MANIFEST]):
    update_stage_state("stage_2", "running")
    # Copy only audio referenced by tasks_pending.jsonl (and transcripts from pairs_pending.jsonl)
    if USE_LOCAL_DATA:
        audio_paths = collect_jsonl_paths(TASKS_PENDING, "audio_path")
        rel_audio = rel_paths_to_drive(audio_paths)
        rsync_files_from(rel_audio, DATA_ROOT_DRIVE, DATA_ROOT, label="audio_for_stage2")

        transcript_paths = collect_jsonl_paths(PAIRS_PENDING, "transcript_json")
        rel_transcripts = rel_paths_to_drive(transcript_paths)
        rsync_files_from(rel_transcripts, DATA_ROOT_DRIVE, DATA_ROOT, label="transcripts_for_stage2")

    run([PY, "stage_2_chunk_transcripts_sentence_parallel.py",
         "--tasks_pending_path", TASKS_PENDING,
         "--pairs_pending_path", PAIRS_PENDING,
         "--out_pairs_path", STAGE2_MANIFEST,
         "--ffmpeg_workers", str(FFMPEG_WORKERS),
         "--task_workers", str(TASK_WORKERS),
         "--ffmpeg_threads", str(FFMPEG_THREADS),
         "--stereo_policy", "split_drop_dupes"])
    assert_outputs([STAGE2_MANIFEST], "stage_2")
    sync_files_to_drive([STAGE2_MANIFEST], label="stage2")
    update_stage_state("stage_2", "done")

# %%
# ---------- STAGE 3 ----------
if should_run("stage_3", [SCORES_CSV]):
    ensure_local_paths([TARGET_REF_DIR] + ([OTHER_REF_DIR] if OTHER_REF_DIR else []), label="stage_3")
    update_stage_state("stage_3", "running")
    cmd = [
        PY, "stage_3_sort_audio_files_by_speaker_target_other.py",
        "--in", CHUNKS_DIR,
        "--target_ref_dir", TARGET_REF_DIR,
        "--target_out", TARGET_OUT_DIR,
        "--other_out", OTHER_OUT_DIR,
        "--threshold", str(SPEAKER_THRESHOLD),
        "--device", DEVICE,
        "--batch_size", str(SPEAKER_BATCH),
        "--workers", str(SPEAKER_WORKERS),
        "--state_file", STATE_FILE,
    ]
    if OTHER_REF_DIR:
        cmd += ["--other_ref_dir", OTHER_REF_DIR]
    if SPEAKER_SORT_DRY_RUN:
        cmd += ["--dry_run", "--copy"]
    else:
        cmd += ["--copy"]
    run(cmd)
    assert_outputs([SCORES_CSV], "stage_3")
    update_stage_state("stage_3", "done")

# %%
# ---------- STAGE 3b ----------
if should_run("stage_3b", [STAGE3B_MANIFEST]):
    update_stage_state("stage_3b", "running")
    run([PY, "stage_3b_filter_english_only.py",
         "--input", STAGE2_MANIFEST,
         "--output", STAGE3B_MANIFEST,
         "--min-english-ratio", "0.7"])
    assert_outputs([STAGE3B_MANIFEST], "stage_3b")
    update_stage_state("stage_3b", "done")

# %%
# ---------- STAGE 4 ----------
if should_run("stage_4", [STAGE4_MANIFEST]):
    update_stage_state("stage_4", "running")
    run([PY, "stage_4_Delete_common_fillers_words_from_manifest.py",
         "--input", STAGE3B_MANIFEST,
         "--output", STAGE4_MANIFEST,
         "--state-file", COMMON_SEGMENTS_STATE,
         "--min-frequency", "3"])
    assert_outputs([STAGE4_MANIFEST], "stage_4")
    update_stage_state("stage_4", "done")

# %%
# ---------- STAGE 6 ----------
if should_run("stage_6", [STAGE6_MANIFEST]):
    ensure_local_paths([NOISE_DIR], label="stage_6")
    update_stage_state("stage_6", "running")
    Path(SEEN_DIR).mkdir(parents=True, exist_ok=True)
    seed_manifest(STAGE4_MANIFEST, STAGE6_MANIFEST)
    run([PY, "stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise.py",
         "--in_manifest", STAGE4_MANIFEST,
         "--out_manifest", STAGE6_MANIFEST,
         "--noises_dir", NOISE_DIR,
         "--scores_csv", SCORES_CSV,
         "--out_dir", NOISE_OUT_DIR,
         "--stage_name", "noise_mix",
         "--seen_db", f"{SEEN_DIR}/stage6_noise_mix.sqlite",
         "--ratio", str(NOISE_RATIO),
         "--copies", str(NOISE_COPIES),
         "--snr_db_min", "5",
         "--snr_db_max", "20",
         "--max_bad_to_good_ratio", "1.0",
         "--good_floor_db", "-125",
         "--workers", str(AUG_WORKERS)])
    assert_outputs([STAGE6_MANIFEST], "stage_6")
    update_stage_state("stage_6", "done")

# %%
# ---------- STAGE 7 ----------
if should_run("stage_7", [STAGE7_MANIFEST]):
    ensure_local_paths([OTHER_VOICES_DIR], label="stage_7")
    update_stage_state("stage_7", "running")
    seed_manifest(STAGE6_MANIFEST, STAGE7_MANIFEST)
    run([PY, "stage_7_add_others_voices_to_my_audio_fast_idempotent.py",
         "--in_manifest", STAGE6_MANIFEST,
         "--out_manifest", STAGE7_MANIFEST,
         "--other_voices_dir", OTHER_VOICES_DIR,
         "--out_dir", VOICE_OUT_DIR,
         "--ratio", str(VOICE_RATIO),
         "--copies", str(VOICE_COPIES),
         "--snr_db_min", "5",
         "--snr_db_max", "10",
         "--max_bad_to_good_ratio", "0.15",
         "--scores_csv", SCORES_CSV,
         "--workers", str(AUG_WORKERS),
         "--stage_name", "voice_mix",
         "--seen_db", f"{SEEN_DIR}/stage7_voice_mix.sqlite"])
    assert_outputs([STAGE7_MANIFEST], "stage_7")
    update_stage_state("stage_7", "done")

# %%
# ---------- STAGE 8 ----------
if should_run("stage_8", [STAGE8_MANIFEST]):
    update_stage_state("stage_8", "running")
    seed_manifest(STAGE7_MANIFEST, STAGE8_MANIFEST)
    run([PY, "stage_8_add_random_gain_to_high_score_voices_parallel_idempotent.py",
         "--in_manifest", STAGE7_MANIFEST,
         "--out_manifest", STAGE8_MANIFEST,
         "--out_dir", GAIN_OUT_DIR,
         "--ratio", str(GAIN_RATIO),
         "--copies", str(GAIN_COPIES),
         "--min_db", "-12",
         "--max_db", "12",
         "--workers", str(AUG_WORKERS),
         "--stage_name", "random_gain",
         "--seen_db", f"{SEEN_DIR}/stage8_random_gain.sqlite"])
    assert_outputs([STAGE8_MANIFEST], "stage_8")
    update_stage_state("stage_8", "done")

# %%
# ---------- STAGE 9 ----------
if should_run("stage_9", [STAGE9_MANIFEST]):
    ensure_local_paths([RIR_DIR], label="stage_9")
    update_stage_state("stage_9", "running")
    seed_manifest(STAGE8_MANIFEST, STAGE9_MANIFEST)
    run([PY, "stage_9_add_reverb_idempotent.py",
         "--in_manifest", STAGE8_MANIFEST,
         "--out_manifest", STAGE9_MANIFEST,
         "--rir_dir", RIR_DIR,
         "--out_dir", REVERB_OUT_DIR,
         "--ratio", str(REVERB_RATIO),
         "--copies", str(REVERB_COPIES),
         "--workers", str(AUG_WORKERS),
         "--stage_name", "reverb",
         "--seen_db", f"{SEEN_DIR}/stage9_reverb.sqlite"])
    assert_outputs([STAGE9_MANIFEST], "stage_9")
    update_stage_state("stage_9", "done")

# %%
# ---------- STAGE 10b ----------
if should_run("stage_10b", [STAGE10B_MANIFEST]):
    update_stage_state("stage_10b", "running")
    seed_manifest(STAGE9_MANIFEST, STAGE10B_MANIFEST)
    run([PY, "stage_10_b_add_speech_tempo_pause_aware_idempotent.py",
         "--in_manifest", STAGE9_MANIFEST,
         "--out_manifest", STAGE10B_MANIFEST,
         "--out_dir", TEMPO_OUT_DIR,
         "--ratio", str(TEMPO_RATIO),
         "--copies", str(TEMPO_COPIES),
         "--workers", str(AUG_WORKERS),
         "--stage_name", "tempo_speech_pause",
         "--seen_db", f"{SEEN_DIR}/stage10b_tempo_pause.sqlite",
         "--tempo_min", "1.05",
         "--tempo_max", "1.20",
         "--mode", "choice",
         "--tempo_factors", "1.05,1.07,1.09,1.10,1.12,1.14,1.16,1.18,1.20",
         "--pause_policy", "truncate",
         "--silence_factor", "2.8",
         "--silence_noise_db", "-35",
         "--silence_min_dur", "0.15"])
    assert_outputs([STAGE10B_MANIFEST], "stage_10b")
    update_stage_state("stage_10b", "done")

# %%
# ---------- STAGE 12 ----------
if should_run("stage_12", [STAGE12_MANIFEST]):
    update_stage_state("stage_12", "running")
    run([PY, "stage_12_remove_bottom_percent_by_speaker_scores.py",
         "--input_manifest", STAGE10B_MANIFEST,
         "--output_manifest", STAGE12_MANIFEST,
         "--speaker_scores_csv", SCORES_CSV,
         "--bottom_percent", str(BOTTOM_PERCENT)])
    assert_outputs([STAGE12_MANIFEST], "stage_12")
    update_stage_state("stage_12", "done")

# %%
# ---------- STAGE 13 ----------
if should_run("stage_13", [STAGE13_TRAIN, STAGE13_TEST]):
    update_stage_state("stage_13", "running")
    run([PY, "stage_13_group_split_train_test.py",
         "--input_manifest", STAGE12_MANIFEST,
         "--test_manifest", STAGE13_TEST,
         "--train_manifest", STAGE13_TRAIN,
         "--test_ratio", str(TEST_RATIO),
         "--seed", str(RANDOM_SEED)])
    assert_outputs([STAGE13_TRAIN, STAGE13_TEST], "stage_13")
    update_stage_state("stage_13", "done")

# %%
# ---------- STAGE 14 ----------
if should_run("stage_14", [STAGE14_TRAIN]):
    update_stage_state("stage_14", "running")
    run([PY, "stage_14_remove_target_files_from_manifest.py",
         "--input_manifest", STAGE13_TRAIN,
         "--output_manifest", STAGE14_TRAIN,
         "--speaker_scores_csv", SCORES_CSV])
    assert_outputs([STAGE14_TRAIN], "stage_14")
    update_stage_state("stage_14", "done")

# %%
# ---------- STAGE 15 ----------
if should_run("stage_15", [STAGE15_TRAIN]):
    update_stage_state("stage_15", "running")
    if USE_ADVANCED_RANDOMIZE:
        run([PY, "stage_15_b_advanced_randomize_manifest.py",
             "--input_manifest", STAGE14_TRAIN,
             "--output_manifest", STAGE15_TRAIN,
             "--seed", str(RANDOM_SEED)])
    else:
        run([PY, "stage_15_a_randomize_manifest.py",
             "--input_manifest", STAGE14_TRAIN,
             "--output_manifest", STAGE15_TRAIN,
             "--seed", str(RANDOM_SEED)])
    assert_outputs([STAGE15_TRAIN], "stage_15")
    update_stage_state("stage_15", "done")

# %%
# ---------- STAGE 16 ----------
if should_run("stage_16", [STAGE13_TEST]):
    update_stage_state("stage_16", "running")
    run([PY, "Stage_16_move_test_chunks_update_test_manifest.py"])
    assert_outputs([STAGE13_TEST], "stage_16")
    update_stage_state("stage_16", "done")

# %%
# ---------- STAGE 17 ----------
if should_run("stage_17", [CHECKPOINT_DIR]):
    update_stage_state("stage_17", "running")
    os.environ["WHISPER_START_FRESH"] = str(START_FRESH)
    run([PY, STAGE17_SCRIPT])
    assert_outputs([CHECKPOINT_DIR], "stage_17")
    update_stage_state("stage_17", "done")

# %%
# ---------- SYNC CACHE ----------
sync_cache_to_drive()

print("✅ Pipeline complete.")
