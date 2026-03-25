# %% [markdown]
# # Colab: eval-only (stage 19c / 19d)
# Paste into Colab. Cells are split by `# %%`.

# %%
from google.colab import drive
drive.mount("/content/drive")

# %%
# ---------- CONFIG (EDIT THESE) ----------
REPO_URL = "https://github.com/sriharshaguthikonda/whisper-acft.git"
REPO_DIR_DRIVE = "/content/drive/MyDrive/whisper-acft"
REPO_DIR_LOCAL = "/content/whisper-acft"
USE_LOCAL_REPO = True  # True = clone to local disk (faster). False = run from Drive.

REPO_DIR = REPO_DIR_LOCAL if USE_LOCAL_REPO else REPO_DIR_DRIVE

# Root on Drive where you copied data
DATA_ROOT_DRIVE = "/content/drive/MyDrive"  # <-- change if you copied elsewhere
DATA_ROOT_ACTIVE = DATA_ROOT_DRIVE  # drive stays canonical; cache used opportunistically

# Optional: prefetch required data to local disk (faster eval I/O)
PREFETCH_TO_LOCAL = True
PREFETCH_IN_BACKGROUND = True
PREFETCH_IGNORE_EXISTING = True
LOCAL_DATA_ROOT = "/content/whisper-data"

# Required inputs for eval
TEST_MANIFEST = f"{DATA_ROOT_DRIVE}/Record_chunks/pairs_manifest_stage13_test.jsonl"
SPEAKER_SCORES_CSV = f"{DATA_ROOT_DRIVE}/speaker_sort_scores.csv"
CHECKPOINT_DIR = f"{DATA_ROOT_DRIVE}/Stage_17_aug_futo_wer_rank32_dora_dyn_ctx_chkpts_small_en_25"  # folder with model_epoch_*

# Optional: use external OTHER voices for mixing
USE_OTHERS_DIR = False
OTHERS_DIR = f"{DATA_ROOT_DRIVE}/Record_others_compacted"
OTHERS_MANIFEST = f"{DATA_ROOT_DRIVE}/Record_others_compacted/pairs_pending_stereo.jsonl"

# Eval sweep config (edit as needed)
MIX_PER_TARGET = 5
SWEEP_SNR_DB = "15,5,0"
SWEEP_OVERLAP = "0.25,0.75,1"
PERCENTAGE = 100  # overall subsample after pairing (0-100)
TARGET_PERCENTAGE = 100  # target subsample before pairing (0-100)
TARGET_MAX = 0  # 0 = no cap
RESUME = True
SKIP_BASE_MODEL = False
VAD_FILTER = 0
AUTO_BATCH = 1
BATCH_SIZE = 1
BATCH_MIN = 1
BATCH_MAX = 8
MEM_LOW = 0.55
MEM_HIGH = 0.85
CLEANUP_INTERVAL = 10
AUDIO_CACHE_GB = 0.25
FP16 = 1
LORA_MERGE = True
LORA_BASE_MODEL = "futo-org/acft-whisper-small.en"

# Path rewrite (handles I:\ or other roots inside jsonl/csv you copied)
RUN_REWRITE_PATHS = True
JSONL_REWRITE_FILES = [TEST_MANIFEST]
CSV_REWRITE_FILES = [SPEAKER_SCORES_CSV]

# Output
EVAL_19C_OUT_JSON = f"{CHECKPOINT_DIR}/evaluation_results_futo_like_targetmix_sweep.json"

# %%
# ---------- SETUP ----------
import os
import sys
import json
import csv
import shutil
import subprocess
import threading
from pathlib import Path

def run(cmd, env=None):
    print("$ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    assert p.stdout is not None
    for out_line in p.stdout:
        print(out_line, end="")
    rc = p.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)

# Clone repo if needed
if not Path(REPO_DIR).exists():
    run(["git", "clone", REPO_URL, REPO_DIR])
os.chdir(REPO_DIR)

# Install deps
run(["pip", "-q", "install", "transformers", "torch", "soundfile", "jiwer", "tqdm", "numpy", "pandas", "matplotlib"])
run(["apt-get", "-qq", "update"])
run(["apt-get", "-qq", "install", "-y", "ffmpeg", "rsync"])

# %%
# ---------- OPTIONAL PREFETCH TO LOCAL ----------
def _to_local_path(p: str) -> str:
    if p.startswith(DATA_ROOT_DRIVE):
        return LOCAL_DATA_ROOT + p[len(DATA_ROOT_DRIVE):]
    return p

def _prefetch_one(src: str) -> None:
    src_p = Path(src)
    dst = _to_local_path(src)
    dst_p = Path(dst)
    rsync_args = ["rsync", "-a", "--info=progress2"]
    if PREFETCH_IGNORE_EXISTING:
        rsync_args.append("--ignore-existing")
    if src_p.is_dir():
        dst_p.mkdir(parents=True, exist_ok=True)
        subprocess.run(rsync_args + [f"{src_p}/", f"{dst_p}/"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else:
        dst_p.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(rsync_args + [str(src_p), str(dst_p)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def _prefetch_worker(items: list[str]) -> None:
    for item in items:
        _prefetch_one(item)

if PREFETCH_TO_LOCAL:
    Path(LOCAL_DATA_ROOT).mkdir(parents=True, exist_ok=True)
    prefetch_items = [
        f"{DATA_ROOT_DRIVE}/Record_test_chunks",
        CHECKPOINT_DIR,
    ]
    if USE_OTHERS_DIR:
        prefetch_items += [
            OTHERS_DIR,
        ]
    if PREFETCH_IN_BACKGROUND:
        threading.Thread(target=_prefetch_worker, args=(prefetch_items,), daemon=True).start()
    else:
        _prefetch_worker(prefetch_items)

# %%
# ---------- PATH REWRITE HELPERS ----------
ANCHOR_DIRS = [
    "Record_test_chunks",
    "Record_chunks",
    "Record_others_compacted",
    "Record_others_chunks",
    "Record_harsha",
    "Transcriptions_corrected",
]

def _norm_slashes(p: str) -> str:
    return (p or "").replace("\\", "/")

def _remap_by_anchor(p: str) -> str:
    """Remap Windows/old roots by locating a known anchor dir and re-rooting to Drive."""
    s = _norm_slashes(p)
    if s.startswith(DATA_ROOT_ACTIVE):
        return s
    s_low = s.lower()
    for anchor in ANCHOR_DIRS:
        a = anchor.lower()
        marker = f"/{a}/"
        idx = s_low.find(marker)
        if idx != -1:
            suffix = s[idx + 1:]  # drop leading slash
            return f"{DATA_ROOT_ACTIVE}/{suffix}"
        marker_tail = f"/{a}"
        idx_tail = s_low.rfind(marker_tail)
        if idx_tail != -1 and idx_tail + len(marker_tail) == len(s_low):
            suffix = s[idx_tail + 1:]
            return f"{DATA_ROOT_ACTIVE}/{suffix}"
    return s

# ---------- PATCH EVAL SCRIPT FOR CACHE FALLBACK ----------
PATCH_EVAL_SCRIPT = True
EVAL_19C_SCRIPT = "stage_19c_specific_target_advanced_futo_like_evaluate_targetmix_sweep_with_cer.py"
EVAL_19C_PATCHED = "stage_19c_eval_cached.py"

def patch_eval_script(src: str, dst: str) -> str:
    src_p = Path(src)
    dst_p = Path(dst)
    if not src_p.exists():
        raise FileNotFoundError(src)
    txt = src_p.read_text(encoding="utf-8")
    if "# -- cache-fallback patch" in txt:
        dst_p.write_text(txt, encoding="utf-8")
        return str(dst_p)
    if "# -- cache-fallback patch" in txt:
        return str(src_p)
    insert_after = "import soundfile as sf"
    if insert_after not in txt:
        dst_p.write_text(txt, encoding="utf-8")
        return str(dst_p)
    patch = """
# -- cache-fallback patch (eval_only_colab)
CACHE_ROOT = os.environ.get("EVAL_CACHE_ROOT", "")
DRIVE_ROOT = os.environ.get("EVAL_DRIVE_ROOT", "")
def _resolve_cache_path(p: Path) -> Path:
    try:
        s = str(p)
    except Exception:
        return p
    if CACHE_ROOT and DRIVE_ROOT and s.startswith(DRIVE_ROOT):
        alt = CACHE_ROOT + s[len(DRIVE_ROOT):]
        if os.path.exists(alt):
            return Path(alt)
    return p
# -- end cache-fallback patch
"""
    txt = txt.replace(insert_after, insert_after + patch)
    txt = txt.replace(
        "def load_audio_mono_16k(path: Path) -> Tuple[np.ndarray, int]:",
        "def load_audio_mono_16k(path: Path) -> Tuple[np.ndarray, int]:\n    path = _resolve_cache_path(path)",
    )
    txt = txt.replace(
        "def _load_audio_mono_16k(p: Path) -> Tuple[np.ndarray, int]:",
        "def _load_audio_mono_16k(p: Path) -> Tuple[np.ndarray, int]:\n        p = _resolve_cache_path(p)",
    )
    txt = txt.replace(
        "models.extend([str(p) for p in checkpoints])",
        "models.extend([str(p) for p in checkpoints])\n"
        "    if os.environ.get(\"EVAL_SKIP_BASE_MODEL\") == \"1\":\n"
        "        models = [m for m in models if str(m) != str(args.base_model)]",
    )
    dst_p.write_text(txt, encoding="utf-8")
    return str(dst_p)

def rewrite_jsonl_paths(jsonl_path: str, keys: list[str]) -> None:
    p = Path(jsonl_path)
    if not p.exists():
        print(f"[rewrite] skip (missing): {p}")
        return
    tmp = p.with_suffix(p.suffix + ".tmp")
    changed = 0
    with p.open("r", encoding="utf-8", errors="ignore") as f_in, tmp.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                f_out.write(line + "\n")
                continue
            for k in keys:
                v = obj.get(k)
                if isinstance(v, str):
                    nv = _remap_by_anchor(v)
                    if nv != v:
                        obj[k] = nv
                        changed += 1
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    if changed:
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(p, bak)
        os.replace(tmp, p)
        print(f"[rewrite] {p}: {changed} path(s) updated")
    else:
        tmp.unlink(missing_ok=True)
        print(f"[rewrite] {p}: no changes")

def rewrite_csv_paths(csv_path: str, column: str = "file") -> None:
    p = Path(csv_path)
    if not p.exists():
        print(f"[rewrite] skip (missing): {p}")
        return
    tmp = p.with_suffix(p.suffix + ".tmp")
    changed = 0
    with p.open("r", encoding="utf-8", newline="") as f_in, tmp.open("w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames or column not in reader.fieldnames:
            print(f"[rewrite] {p}: column '{column}' not found; skipping")
            tmp.unlink(missing_ok=True)
            return
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in reader:
            # Some CSVs have ragged rows; csv.DictReader stores extras under key None.
            if None in row:
                row.pop(None, None)
            v = row.get(column, "")
            if v:
                nv = _remap_by_anchor(v)
                if nv != v:
                    row[column] = nv
                    changed += 1
            writer.writerow(row)
    if changed:
        bak = p.with_suffix(p.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(p, bak)
        os.replace(tmp, p)
        print(f"[rewrite] {p}: {changed} path(s) updated")
    else:
        tmp.unlink(missing_ok=True)
        print(f"[rewrite] {p}: no changes")

# %%
# ---------- OPTIONAL PATH REWRITE ----------
if RUN_REWRITE_PATHS:
    for p in JSONL_REWRITE_FILES:
        rewrite_jsonl_paths(p, ["audio_path", "transcript_json", "source_audio"])
    for p in CSV_REWRITE_FILES:
        rewrite_csv_paths(p, column="file")

# %%
# ---------- STAGE 19C: EVALUATION ----------
eval_script = EVAL_19C_SCRIPT
if PATCH_EVAL_SCRIPT:
    eval_script = patch_eval_script(EVAL_19C_SCRIPT, EVAL_19C_PATCHED)

eval_cmd = [
    sys.executable,
    eval_script,
    "--test_manifest", TEST_MANIFEST,
    "--speaker_scores_csv", SPEAKER_SCORES_CSV,
    "--checkpoint_dir", CHECKPOINT_DIR,
    "--mix_per_target", str(MIX_PER_TARGET),
    "--sweep_snr_db", str(SWEEP_SNR_DB),
    "--sweep_overlap", str(SWEEP_OVERLAP),
    "--percentage", str(PERCENTAGE),
    "--target_percentage", str(TARGET_PERCENTAGE),
    "--target_max", str(TARGET_MAX),
    "--vad_filter", str(VAD_FILTER),
    "--auto_batch", str(AUTO_BATCH),
    "--batch_size", str(BATCH_SIZE),
    "--batch_min", str(BATCH_MIN),
    "--batch_max", str(BATCH_MAX),
    "--mem_low", str(MEM_LOW),
    "--mem_high", str(MEM_HIGH),
    "--cleanup_interval", str(CLEANUP_INTERVAL),
    "--audio_cache_gb", str(AUDIO_CACHE_GB),
    "--fp16", str(FP16),
    "--ffmpeg_path", "ffmpeg",
]
if USE_OTHERS_DIR:
    eval_cmd += ["--others_dir", OTHERS_DIR, "--others_manifest", OTHERS_MANIFEST]
if LORA_MERGE:
    eval_cmd += ["--lora_merge", "--lora_base_model", LORA_BASE_MODEL]
if RESUME:
    eval_cmd += ["--resume"]
eval_env = None
if PREFETCH_TO_LOCAL:
    eval_env = os.environ.copy()
    eval_env["EVAL_CACHE_ROOT"] = LOCAL_DATA_ROOT
    eval_env["EVAL_DRIVE_ROOT"] = DATA_ROOT_DRIVE
    if SKIP_BASE_MODEL:
        eval_env["EVAL_SKIP_BASE_MODEL"] = "1"
elif SKIP_BASE_MODEL:
    eval_env = os.environ.copy()
    eval_env["EVAL_SKIP_BASE_MODEL"] = "1"
run(eval_cmd, env=eval_env)

# %%
# ---------- STAGE 19D: CHARTS ----------
run([
    sys.executable,
    "stage_19d_plot_eval_charts.py",
    "--in_json", EVAL_19C_OUT_JSON,
    "--out_dir", CHECKPOINT_DIR,
])

print("✅ Eval complete.")
