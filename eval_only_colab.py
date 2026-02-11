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

# Required inputs for eval
TEST_MANIFEST = f"{DATA_ROOT_DRIVE}/Record_chunks/pairs_manifest_stage13_test.jsonl"
SPEAKER_SCORES_CSV = f"{DATA_ROOT_DRIVE}/speaker_sort_scores.csv"
CHECKPOINT_DIR = f"{DATA_ROOT_DRIVE}/stage17_checkpoints"  # folder with model_epoch_*

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
from pathlib import Path

def run(cmd):
    print("$ " + (" ".join(cmd) if isinstance(cmd, list) else cmd))
    subprocess.run(cmd, check=True)

# Clone repo if needed
if not Path(REPO_DIR).exists():
    run(["git", "clone", REPO_URL, REPO_DIR])
os.chdir(REPO_DIR)

# Install deps
run(["pip", "-q", "install", "transformers", "torch", "soundfile", "jiwer", "tqdm", "numpy", "pandas", "matplotlib"])
run(["apt-get", "-qq", "update"])
run(["apt-get", "-qq", "install", "-y", "ffmpeg"])

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
    if s.startswith(DATA_ROOT_DRIVE):
        return s
    s_low = s.lower()
    for anchor in ANCHOR_DIRS:
        a = anchor.lower()
        marker = f"/{a}/"
        idx = s_low.find(marker)
        if idx != -1:
            suffix = s[idx + 1:]  # drop leading slash
            return f"{DATA_ROOT_DRIVE}/{suffix}"
        marker_tail = f"/{a}"
        idx_tail = s_low.rfind(marker_tail)
        if idx_tail != -1 and idx_tail + len(marker_tail) == len(s_low):
            suffix = s[idx_tail + 1:]
            return f"{DATA_ROOT_DRIVE}/{suffix}"
    return s

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
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
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
eval_cmd = [
    sys.executable,
    "stage_19c_specific_target_advanced_futo_like_evaluate_targetmix_sweep_with_cer.py",
    "--test_manifest", TEST_MANIFEST,
    "--speaker_scores_csv", SPEAKER_SCORES_CSV,
    "--checkpoint_dir", CHECKPOINT_DIR,
    "--mix_per_target", str(MIX_PER_TARGET),
    "--sweep_snr_db", str(SWEEP_SNR_DB),
    "--sweep_overlap", str(SWEEP_OVERLAP),
    "--percentage", str(PERCENTAGE),
    "--target_percentage", str(TARGET_PERCENTAGE),
    "--target_max", str(TARGET_MAX),
    "--ffmpeg_path", "ffmpeg",
]
if USE_OTHERS_DIR:
    eval_cmd += ["--others_dir", OTHERS_DIR, "--others_manifest", OTHERS_MANIFEST]
run(eval_cmd)

# %%
# ---------- STAGE 19D: CHARTS ----------
run([
    sys.executable,
    "stage_19d_plot_eval_charts.py",
    "--in_json", EVAL_19C_OUT_JSON,
    "--out_dir", CHECKPOINT_DIR,
])

print("✅ Eval complete.")
