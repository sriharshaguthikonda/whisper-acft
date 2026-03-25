# %% [markdown]
# # Local: run stages 1 -> 17 (full script)
# Windows/local version of Full_pipeline_whisper_training.py.

# %%
import os
import argparse
import json
from pathlib import Path
from run_folder_naming import build_run_folder_name, slug_token as slug_run_token

# ---------- CONFIG (EDIT THESE) ----------
REPO_URL = "https://github.com/sriharshaguthikonda/whisper-acft.git"
REPO_DIR = str(Path(__file__).resolve().parent)  # set to your repo root if different
UPDATE_REPO = False  # set True to git fetch/reset; use with care
RESET_REPO_HARD = False  # only applies when UPDATE_REPO is True

# Data root selection (scan I:\ for expected folders)
DATA_ROOT_OVERRIDE = ""  # e.g. r"I:\"
DATA_ROOT_SCAN_ROOT = r"I:\\"
PROMPT_FOR_DATA_ROOT = True

# Python to run all stages (set "" to use current interpreter)
PYTHON_EXE = r"I:\Whisper-training-env\Scripts\python.exe"

# Optional RIRS overrides
RIRS_ROOT_OVERRIDE = ""  # e.g. r"I:\noise\RIRS_NOISES"
NOISE_DIR_OVERRIDE = ""  # e.g. r"I:\noise\RIRS_NOISES\pointsource_noises"
RIR_DIR_OVERRIDE = ""    # e.g. r"I:\noise\RIRS_NOISES\real_rirs_isotropic_noises"

# Disable Drive sync on local runs
USE_LOCAL_DATA = False  # local data is already on disk; no rsync/rclone
SYNC_IMPORTANT_TO_DRIVE = False
SYNC_RECORD_CHUNKS_TO_DRIVE = False
SYNC_CACHE_TO_DRIVE = False
RESTORE_RECORD_CHUNKS = False

# Copy strategy (unused when USE_LOCAL_DATA is False)
COPY_TRANSCRIPTS_FOR_STAGE1 = False
COPY_AUDIO_FOR_STAGE1 = False
BACKGROUND_COPY_LATER = False

# Optional: rclone (unused on local)
USE_RCLONE = False
RCLONE_REMOTE = ""
RCLONE_TRANSFERS = 8

# Output verification
VERIFY_OUTPUTS = True
VERIFY_JSONL_SAMPLE = 5
ENABLE_OUTPUT_SIGNATURES = True  # manifest diff / hash skip
CAPTURE_STAGE_OUTPUT = True  # set False to stream tqdm/progress live
AUTO_SKIP_IF_OUTPUTS_PRESENT = True  # skip stages when expected outputs already exist
AUTO_SKIP_EXCLUDE_STAGES = {"stage_17"}  # stages that should resume even if outputs exist
AUTO_SKIP_UPSTREAM_FROM_OUTPUTS = True  # skip earlier stages if a later stage output exists

# Torch/torchvision compatibility fix (for pyannote/lightning)
FIX_TORCHVISION = True
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu121"
TORCH_FORCE_REINSTALL = True
TORCH_FIX_ONCE = False

def _inspect_root(p: Path, required: list[str], optional: list[str]) -> tuple[bool, int]:
    if not p.exists() or not p.is_dir():
        return False, 0
    if any(not (p / d).exists() for d in required):
        return False, 0
    score = sum(1 for d in optional if (p / d).exists())
    return True, score

def pick_data_root() -> Path:
    if DATA_ROOT_OVERRIDE:
        return Path(DATA_ROOT_OVERRIDE)
    scan_root = Path(DATA_ROOT_SCAN_ROOT)
    required = ["Transcriptions_corrected", "Record_harsha"]
    optional = [
        "Record_only_by_harsha",
        "Record_others_compacted",
        "Record_chunks",
        "Record_test_chunks",
        "RIRS_NOISES",
    ]
    candidates: list[tuple[int, Path]] = []
    ok, score = _inspect_root(scan_root, required, optional)
    if ok:
        candidates.append((score, scan_root))
    try:
        for child in scan_root.iterdir():
            ok, score = _inspect_root(child, required, optional)
            if ok:
                candidates.append((score, child))
    except Exception:
        pass
    if not candidates:
        print(f"[data] no candidates under {scan_root}; using scan root")
        return scan_root
    candidates.sort(key=lambda item: (-item[0], str(item[1]).lower()))
    if len(candidates) == 1 or not PROMPT_FOR_DATA_ROOT:
        return candidates[0][1]
    print("[data] candidates:")
    for idx, (score, path) in enumerate(candidates, 1):
        present = [d for d in optional if (path / d).exists()]
        present_txt = ", ".join(present) if present else "none"
        print(f"  {idx}) {path} (score {score}; optional: {present_txt})")
    choice = input(f"Select data root [1-{len(candidates)}] (blank=1): ").strip()
    if choice.isdigit():
        pick = max(1, min(len(candidates), int(choice)))
        return candidates[pick - 1][1]
    return candidates[0][1]

def _parse_override_value(raw: str):
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _parse_overrides_from_set(items: list[str]) -> dict:
    overrides: dict = {}
    for item in items or []:
        if "=" not in item:
            print(f"[warn] ignoring invalid --set (expected KEY=VALUE): {item}")
            continue
        key, raw = item.split("=", 1)
        key = key.strip()
        if not key:
            continue
        overrides[key] = _parse_override_value(raw.strip())
    return overrides


def _apply_skip_list(skip_list: str, overrides: dict) -> None:
    if not skip_list:
        return
    mapping = {
        "1": "SKIP_STAGE_1_MANIFEST_CREATION",
        "2": "SKIP_STAGE_2_CHUNK_TRANSCRIPTS",
        "3": "SKIP_STAGE_3_SPEAKER_SORT",
        "3b": "SKIP_STAGE_3B_FILTER_ENGLISH",
        "4": "SKIP_STAGE_4_DELETE_COMMON_FILLERS",
        "5": "SKIP_STAGE_5_CONVERT_NUMBER_WORDS",
        "6": "SKIP_STAGE_6_ADD_NOISE",
        "7": "SKIP_STAGE_7_ADD_OTHER_VOICES",
        "8": "SKIP_STAGE_8_ADD_RANDOM_GAIN",
        "9": "SKIP_STAGE_9_ADD_REVERB",
        "10b": "SKIP_STAGE_10B_ADD_TEMPO_PAUSE",
        "11": "SKIP_STAGE_11_ADD_FREQUENCY",
        "12": "SKIP_STAGE_12_REMOVE_BOTTOM_PERCENT",
        "13": "SKIP_STAGE_13_SPLIT_TRAIN_TEST",
        "14": "SKIP_STAGE_14_REMOVE_TARGET_FILES",
        "15": "SKIP_STAGE_15_RANDOMIZE_MANIFEST",
        "16": "SKIP_STAGE_16_MOVE_TEST_CHUNKS",
        "17": "SKIP_STAGE_17_TRAIN",
    }
    tokens = [t.strip() for t in skip_list.replace(";", ",").split(",") if t.strip()]
    for token in tokens:
        norm = token.lower().replace("stage", "").replace("_", "")
        key = mapping.get(norm)
        if key:
            overrides[key] = True
        else:
            print(f"[warn] unknown skip token: {token}")


def _parse_cli_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Full pipeline (local)")
    ap.add_argument("--config", help="JSON config overrides (keys match variable names)")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config KEY=JSON_VALUE (repeatable), e.g. --set DATA_ROOT_OVERRIDE=\"I:\\\"",
    )
    ap.add_argument("--data-root", help="Override data root (sets DATA_ROOT_OVERRIDE)")
    ap.add_argument("--data-root-scan-root", help="Override data root scan base (DATA_ROOT_SCAN_ROOT)")
    ap.add_argument("--prompt-for-data-root", dest="prompt_for_data_root", action="store_true")
    ap.add_argument("--no-prompt-for-data-root", dest="prompt_for_data_root", action="store_false")
    ap.set_defaults(prompt_for_data_root=None)
    ap.add_argument("--python-exe", help="Python interpreter to run all stages")
    ap.add_argument("--rirs-root", help="Override RIRS root (RIRS_NOISES)")
    ap.add_argument("--noise-dir", help="Override noises dir (pointsource_noises)")
    ap.add_argument("--rir-dir", help="Override RIR dir (real_rirs_isotropic_noises)")
    ap.add_argument("--base-model-id", help="Base model id for training")
    ap.add_argument("--processor-id", help="Processor id for training")
    ap.add_argument("--stage17-script", help="Stage 17 training script filename")
    ap.add_argument("--start-fresh", type=int, choices=[0, 1], help="Refuse to resume if checkpoints exist")
    ap.add_argument("--skip", help="Comma-separated stages to skip (e.g., 1,3b,10b,17)")
    ap.add_argument("--run-eval-19c", action="store_true")
    ap.add_argument("--run-eval-19d", action="store_true")
    ap.add_argument("--auto-skip-existing-outputs", dest="auto_skip_outputs", action="store_true")
    ap.add_argument("--no-auto-skip-existing-outputs", dest="auto_skip_outputs", action="store_false")
    ap.set_defaults(auto_skip_outputs=None)
    ap.add_argument("--auto-skip-upstream", dest="auto_skip_upstream", action="store_true")
    ap.add_argument("--no-auto-skip-upstream", dest="auto_skip_upstream", action="store_false")
    ap.set_defaults(auto_skip_upstream=None)
    ap.add_argument("--no-gguf-conversion", dest="run_gguf_conversion", action="store_false")
    ap.set_defaults(run_gguf_conversion=None)
    ap.add_argument("--print-config", action="store_true", help="Print resolved config and exit")
    return ap.parse_args()


def _load_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = json.load(f) or {}
            if isinstance(cfg, dict):
                overrides.update(cfg)
            else:
                print("[warn] config file did not contain a JSON object; ignoring")
        except Exception as e:
            raise SystemExit(f"Failed to read config: {args.config} ({e})")
    overrides.update(_parse_overrides_from_set(args.set))
    if args.data_root:
        overrides["DATA_ROOT_OVERRIDE"] = args.data_root
    if args.data_root_scan_root:
        overrides["DATA_ROOT_SCAN_ROOT"] = args.data_root_scan_root
    if args.prompt_for_data_root is not None:
        overrides["PROMPT_FOR_DATA_ROOT"] = bool(args.prompt_for_data_root)
    if args.python_exe:
        overrides["PYTHON_EXE"] = args.python_exe
    if args.rirs_root:
        overrides["RIRS_ROOT_OVERRIDE"] = args.rirs_root
    if args.noise_dir:
        overrides["NOISE_DIR_OVERRIDE"] = args.noise_dir
    if args.rir_dir:
        overrides["RIR_DIR_OVERRIDE"] = args.rir_dir
    if args.base_model_id:
        overrides["BASE_MODEL_ID"] = args.base_model_id
    if args.processor_id:
        overrides["PROCESSOR_ID"] = args.processor_id
    if args.stage17_script:
        overrides["STAGE17_SCRIPT"] = args.stage17_script
    if args.start_fresh is not None:
        overrides["START_FRESH"] = int(args.start_fresh)
    if args.run_eval_19c:
        overrides["RUN_EVAL_19C"] = True
    if args.run_eval_19d:
        overrides["RUN_EVAL_19D"] = True
    if args.auto_skip_outputs is not None:
        overrides["AUTO_SKIP_IF_OUTPUTS_PRESENT"] = bool(args.auto_skip_outputs)
    if args.auto_skip_upstream is not None:
        overrides["AUTO_SKIP_UPSTREAM_FROM_OUTPUTS"] = bool(args.auto_skip_upstream)
    if args.run_gguf_conversion is not None:
        overrides["RUN_GGUF_CONVERSION"] = bool(args.run_gguf_conversion)
    _apply_skip_list(args.skip, overrides)
    return overrides


def _apply_overrides(overrides: dict, *, warn_unknown: bool = True) -> None:
    if not overrides:
        return
    for key, value in overrides.items():
        if not key.isupper():
            print(f"[warn] ignoring non-uppercase key '{key}'")
            continue
        if key not in globals():
            if warn_unknown:
                print(f"[warn] unknown key '{key}' ignored")
            continue
        globals()[key] = value


_CLI_ARGS = _parse_cli_args()
_CLI_OVERRIDES = _load_overrides(_CLI_ARGS)
_apply_overrides(_CLI_OVERRIDES, warn_unknown=False)
if _CLI_ARGS.print_config:
    _cfg = {k: globals()[k] for k in sorted(globals()) if k.isupper()}
    print(json.dumps(_cfg, indent=2, default=str))
    raise SystemExit(0)

DATA_ROOT_PATH = pick_data_root()
DATA_ROOT = str(DATA_ROOT_PATH)
DATA_ROOT_DRIVE = DATA_ROOT  # no Drive on local runs
LOCAL_DATA_ROOT = DATA_ROOT
print(f"[data] DATA_ROOT = {DATA_ROOT}")

DRIVE_SYNC_ROOT = str(Path(DATA_ROOT) / "pipeline_checkpoints")
PIPELINE_STATE_FILE = str(Path(DRIVE_SYNC_ROOT) / "pipeline_state.json")
DRIVE_RECORD_CHUNKS = str(Path(DRIVE_SYNC_ROOT) / "Record_chunks")

PIPELINE_LOG = str(Path(DATA_ROOT) / "pipeline.log")
DRIVE_SUMMARY_LOG = str(Path(DRIVE_SYNC_ROOT) / "pipeline_summary.log")

# Caches (avoid re-downloading HF/transformers/torch assets)
CACHE_ROOT = str(Path(DATA_ROOT) / "cache")
HF_HOME = str(Path(CACHE_ROOT) / "hf")
TRANSFORMERS_CACHE = str(Path(CACHE_ROOT) / "transformers")
TORCH_HOME = str(Path(CACHE_ROOT) / "torch")
TORCH_FIX_MARKER = str(Path(CACHE_ROOT) / "torch_fix_ok.txt")
CACHE_DRIVE_ROOT = str(Path(DATA_ROOT) / "cache_drive_unused")
CACHE_RSYNC_SIZE_ONLY = True

# Optional: pre-stage1 audio copy (unused on local)
PRESTAGE1_AUDIO_COPY = False
PRESTAGE1_TASKS_PENDING = str(Path(DATA_ROOT) / "Record_chunks" / "tasks_pending.jsonl")

TRANSCRIPT_DIR = str(Path(DATA_ROOT) / "Transcriptions_corrected")
AUDIO_SOURCE_DIR = str(Path(DATA_ROOT) / "Record_harsha")

TRANSCRIPT_DIR_DRIVE = str(Path(DATA_ROOT_DRIVE) / "Transcriptions_corrected")
TRANSCRIPT_DIR_LOCAL = str(Path(LOCAL_DATA_ROOT) / "Transcriptions_corrected")
AUDIO_SOURCE_DIR_DRIVE = str(Path(DATA_ROOT_DRIVE) / "Record_harsha")
AUDIO_SOURCE_DIR_LOCAL = str(Path(LOCAL_DATA_ROOT) / "Record_harsha")

if USE_LOCAL_DATA:
    TRANSCRIPT_DIR_STAGE1 = TRANSCRIPT_DIR_LOCAL if COPY_TRANSCRIPTS_FOR_STAGE1 else TRANSCRIPT_DIR_DRIVE
    AUDIO_SOURCE_DIR_STAGE1 = AUDIO_SOURCE_DIR_LOCAL if COPY_AUDIO_FOR_STAGE1 else AUDIO_SOURCE_DIR_DRIVE
else:
    TRANSCRIPT_DIR_STAGE1 = TRANSCRIPT_DIR
    AUDIO_SOURCE_DIR_STAGE1 = AUDIO_SOURCE_DIR
CHUNKS_DIR = str(Path(DATA_ROOT) / "Record_chunks")

TARGET_REF_DIR = str(Path(DATA_ROOT) / "Record_only_by_harsha")
OTHER_REF_DIR = str(Path(DATA_ROOT) / "Record_others_compacted")  # set "" to disable
OTHER_VOICES_DIR = str(Path(DATA_ROOT) / "Record_others_compacted")

def pick_rirs_root() -> Path:
    candidates = [
        Path(DATA_ROOT) / "RIRS_NOISES",
        Path(DATA_ROOT) / "noise" / "RIRS_NOISES",
        Path(DATA_ROOT_SCAN_ROOT) / "noise" / "RIRS_NOISES",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]

if RIRS_ROOT_OVERRIDE:
    RIRS_ROOT = Path(RIRS_ROOT_OVERRIDE)
else:
    RIRS_ROOT = pick_rirs_root()
if NOISE_DIR_OVERRIDE:
    NOISE_DIR = NOISE_DIR_OVERRIDE
else:
    NOISE_DIR = str(RIRS_ROOT / "pointsource_noises")
if RIR_DIR_OVERRIDE:
    RIR_DIR = RIR_DIR_OVERRIDE
else:
    RIR_DIR = str(RIRS_ROOT / "real_rirs_isotropic_noises")
print(f"[data] RIRS_ROOT = {RIRS_ROOT}")

TEST_CHUNKS_DIR = str(Path(DATA_ROOT) / "Record_test_chunks")

# Stage 17 checkpoint directory (auto when empty)
CHECKPOINT_DIR = ""  # set to a full path to force a specific dir
CHECKPOINTS_ROOT = ""  # base folder for auto-named checkpoints (default: DATA_ROOT)
STAGE17_CHECKPOINT_PREFIX = ""  # override auto prefix (no trailing index)
STAGE17_CHECKPOINT_TAG = ""  # optional experiment tag (legacy suffix or canonical run-id suffix)
USE_CANONICAL_RUN_FOLDER_NAMING = True
STAGE17_REUSE_EXISTING_PREFIX = False  # set True only if you want to keep resuming old legacy prefixes
RUN_NAME_KIND_OVERRIDE = ""  # e.g. "train-eval"
RUN_NAME_METHOD_OVERRIDE = ""  # e.g. "s17-qat-dora"
RUN_NAME_ADAPTER_OVERRIDE = ""  # e.g. "dora-r64-a16"
RUN_NAME_ROWS_OVERRIDE = ""  # integer row count or "unk"
RUN_NAME_ID_HINT = "auto"  # stable hint; resolver appends _N for repeated runs
STAGE17_SCRIPT = "stage_17_WER_acft_Whisper_Futo_finetuned_model_training_only_local_en_version_only_qat_dora.py"
START_FRESH = 0  # 1 = refuse to resume if checkpoints exist
BASE_MODEL_ID = "futo-org/acft-whisper-small.en"
PROCESSOR_ID = "openai/whisper-small.en"

HF_TOKEN = ""  # optional; set in env or enter when prompted
DEVICE = "cuda"  # "cpu" if no GPU
SPEAKER_SORT_DRY_RUN = True  # True = only produce CSV, no moves/copies
USE_ADVANCED_RANDOMIZE = True  # True -> stage_15_b

RANDOM_SEED = 1337

# Augmentation ratios / copies
NOISE_RATIO, NOISE_COPIES = 0.5, 1
VOICE_RATIO, VOICE_COPIES = 0.3, 1
GAIN_RATIO,  GAIN_COPIES  = 0.1, 1
REVERB_RATIO, REVERB_COPIES = 0.3, 1
TEMPO_RATIO, TEMPO_COPIES = 0.3, 1
FREQ_RATIO, FREQ_COPIES = 0.1, 1

BOTTOM_PERCENT = 30.0
TEST_RATIO = 0.1

SPEAKER_THRESHOLD = 0.10
SPEAKER_WORKERS = 4
SPEAKER_BATCH = 16

AUTO_WORKERS = True  # set False to keep the fixed values above
FFMPEG_WORKERS_MULT = 2.0  # oversubscribe ffmpeg workers (e.g., 2.0 = 2x CPU cores)
TASK_WORKERS_MULT = 1.0
STAGE2_CHECK_CHUNKS = True  # verify Stage-2 manifest rows exist on disk after rsync
STAGE2_RESET_ON_MISSING = True  # reset stage2 state if manifest rows are missing

# Stage toggles (set True to skip)
SKIP_STAGE_1_MANIFEST_CREATION = False
SKIP_STAGE_2_CHUNK_TRANSCRIPTS = False
SKIP_STAGE_3_SPEAKER_SORT = False
SKIP_STAGE_3B_FILTER_ENGLISH = False
SKIP_STAGE_4_DELETE_COMMON_FILLERS = False
SKIP_STAGE_5_CONVERT_NUMBER_WORDS = False
SKIP_STAGE_6_ADD_NOISE = False
SKIP_STAGE_7_ADD_OTHER_VOICES = False
SKIP_STAGE_8_ADD_RANDOM_GAIN = False
SKIP_STAGE_9_ADD_REVERB = False
SKIP_STAGE_10B_ADD_TEMPO_PAUSE = False
SKIP_STAGE_11_ADD_FREQUENCY = False
SKIP_STAGE_12_REMOVE_BOTTOM_PERCENT = True
SKIP_STAGE_13_SPLIT_TRAIN_TEST = False
SKIP_STAGE_14_REMOVE_TARGET_FILES = False
SKIP_STAGE_15_RANDOMIZE_MANIFEST = False
SKIP_STAGE_16_MOVE_TEST_CHUNKS = False
SKIP_STAGE_17_TRAIN = False

RUN_EVAL_19C = False
RUN_EVAL_19D = False




EVAL_19C_SCRIPT = "stage_19c_specific_target_advanced_futo_like_evaluate_targetmix_sweep_with_cer.py"
EVAL_19D_SCRIPT = "stage_19d_plot_eval_charts.py"
EVAL_19C_ARGS = []  # e.g. ["--model", "path", "--data", "path"]
EVAL_19D_ARGS = []
EVAL_ADD_BATCH_ARGS = False  # set True if eval scripts accept --batch_size/--max_samples
EVAL_BATCH_SIZE = 4
EVAL_MAX_SAMPLES = 0  # 0 = no limit
EVAL_19C_OUT_JSON = ""
EVAL_19D_OUT_JSON = ""

CHARTS_DIR = f"{DATA_ROOT}/charts"
CHARTS_DRIVE_DIR = f"{DRIVE_SYNC_ROOT}/charts"
SAVE_CHARTS_TO_DRIVE = True

# LoRA/DoRA merge + export
PEFT_ENABLED = False  # set True if using LoRA/DoRA
MERGE_PEFT_AFTER_STAGE17 = True
MERGED_MODEL_DIR = f"{DATA_ROOT}/stage17_merged_best"

# GGUF/GGML conversion (uses convert_to_gguf.py)
RUN_GGUF_CONVERSION = True
GGUF_QTYPE = "Q8_0"  # Q4_0/Q5_0/Q5_1/Q8_0 or "" for FP16
WHISPER_CPP_ROOT = f"{REPO_DIR}/whisper.cpp"
WHISPER_OPENAI_REPO = f"{REPO_DIR}/whisper"
GGUF_EXPORT_DIR = f"{DATA_ROOT}/gguf_export"
GGUF_OUT_DIR = f"{DATA_ROOT}/gguf_output"

# %%
# ---------- DERIVED PATHS ----------
_apply_overrides(_CLI_OVERRIDES)
TASKS_PENDING = f"{CHUNKS_DIR}/tasks_pending.jsonl"
PAIRS_PENDING = f"{CHUNKS_DIR}/pairs_pending.jsonl"

STAGE2_MANIFEST  = f"{CHUNKS_DIR}/pairs_manifest_stereo.jsonl"
STAGE3B_MANIFEST = f"{CHUNKS_DIR}/pairs_manifest_stereo_english_only.jsonl"
STAGE4_MANIFEST  = f"{CHUNKS_DIR}/pairs_manifest_stereo_english_only_filtered.jsonl"
STAGE5_MANIFEST  = STAGE4_MANIFEST if SKIP_STAGE_5_CONVERT_NUMBER_WORDS else f"{CHUNKS_DIR}/pairs_manifest_stage5_numbers.jsonl"

STAGE6_MANIFEST  = f"{CHUNKS_DIR}/pairs_manifest_stage6_noise.jsonl"
STAGE7_MANIFEST  = f"{CHUNKS_DIR}/pairs_manifest_stage7_voice.jsonl"
STAGE8_MANIFEST  = f"{CHUNKS_DIR}/pairs_manifest_stage8_gain.jsonl"
STAGE9_MANIFEST  = f"{CHUNKS_DIR}/pairs_manifest_stage9_reverb.jsonl"
STAGE10B_MANIFEST = f"{CHUNKS_DIR}/pairs_manifest_stage10b_tempo_pause.jsonl"

STAGE11_MANIFEST = STAGE10B_MANIFEST if SKIP_STAGE_11_ADD_FREQUENCY else f"{CHUNKS_DIR}/pairs_manifest_stage11_frequency.jsonl"
STAGE12_MANIFEST = STAGE11_MANIFEST if SKIP_STAGE_12_REMOVE_BOTTOM_PERCENT else f"{CHUNKS_DIR}/pairs_manifest_stage12_bottom_filtered.jsonl"
STAGE13_TRAIN    = f"{CHUNKS_DIR}/pairs_manifest_stage13_train.jsonl"
STAGE13_TEST     = f"{CHUNKS_DIR}/pairs_manifest_stage13_test.jsonl"
STAGE14_TRAIN    = f"{CHUNKS_DIR}/pairs_manifest_stage14_train_no_targets.jsonl"
STAGE15_TRAIN    = f"{CHUNKS_DIR}/pairs_manifest_stage15_train_no_targets_randomized.jsonl"
STAGE16_BACKUP   = f"{STAGE13_TEST}.backup"

NOISE_OUT_DIR  = f"{CHUNKS_DIR}/noise_augmented"
VOICE_OUT_DIR  = f"{CHUNKS_DIR}/voice_augmented"
GAIN_OUT_DIR   = f"{CHUNKS_DIR}/gain_augmented"
REVERB_OUT_DIR = f"{CHUNKS_DIR}/reverb_augmented"
TEMPO_OUT_DIR  = f"{CHUNKS_DIR}/tempo_pause_augmented"
FREQ_OUT_DIR   = f"{CHUNKS_DIR}/frequency_augmented"
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

def _load_hf_token_from_dotenv() -> None:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"):
        return
    for p in (Path(".env"), Path(__file__).with_name(".env")):
        if not p.exists():
            continue
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for raw in lines:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key not in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
                continue
            value = value.strip()
            if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
                value = value[1:-1]
            if " #" in value:
                value = value.split(" #", 1)[0].rstrip()
            if value:
                os.environ.setdefault(key, value)
                return

os.environ["DEBIAN_FRONTEND"] = "noninteractive"
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE
os.environ["TORCH_HOME"] = TORCH_HOME
# Avoid Triton/Dynamo import crashes on Colab GPUs.
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["PYTHONFAULTHANDLER"] = "1"
Path(CACHE_ROOT).mkdir(parents=True, exist_ok=True)
_load_hf_token_from_dotenv()
if not HF_TOKEN:
    # Try Colab secrets first, then env var.
    try:
        from google.colab import userdata  # type: ignore
        HF_TOKEN = (userdata.get("HF_TOKEN") or "").strip()
    except Exception:
        HF_TOKEN = ""
if not HF_TOKEN:
    HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")
if not HF_TOKEN and not SKIP_STAGE_3_SPEAKER_SORT:
    try:
        from getpass import getpass
        HF_TOKEN = getpass("HF_TOKEN (optional, for pyannote). Leave blank to skip: ").strip()
    except Exception:
        HF_TOKEN = ""
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ.setdefault("HUGGINGFACE_TOKEN", HF_TOKEN)
if PEFT_ENABLED:
    os.environ["WHISPER_USE_PEFT"] = "1"

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

def _sanitize_checkpoint_tag(value: str) -> str:
    if not value:
        return ""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value))
    return cleaned.strip("_")

def _model_tag_from_id(model_id: str) -> str:
    name = (model_id or "").split("/")[-1].lower()
    size = ""
    for cand in ("tiny", "small", "base", "medium", "large"):
        if cand in name:
            size = cand
            break
    if not size:
        size = _sanitize_checkpoint_tag(name) or "model"
    lang = "en" if (".en" in name or name.endswith("en") or "_en" in name) else ""
    return f"{size}_{lang}" if lang else size

def _provider_tag_from_id(model_id: str) -> str:
    low = (model_id or "").lower()
    if "futo" in low:
        return "futo"
    if "openai" in low:
        return "openai"
    org = low.split("/", 1)[0] if "/" in low else low
    return _sanitize_checkpoint_tag(org) or "model"

def _stage17_aug_tag() -> str:
    aug_stages = [
        SKIP_STAGE_6_ADD_NOISE,
        SKIP_STAGE_7_ADD_OTHER_VOICES,
        SKIP_STAGE_8_ADD_RANDOM_GAIN,
        SKIP_STAGE_9_ADD_REVERB,
        SKIP_STAGE_10B_ADD_TEMPO_PAUSE,
        SKIP_STAGE_11_ADD_FREQUENCY,
    ]
    any_aug = any(not flag for flag in aug_stages)
    return "aug" if any_aug else "no_aug"

def _stage17_script_path() -> Path:
    p = Path(STAGE17_SCRIPT)
    if p.is_absolute():
        return p
    return Path(REPO_DIR) / p

def _detect_stage17_script_flags(script_path: str) -> dict:
    flags = {"acft": False, "dora": False, "lora": False, "dyn_ctx": None, "qat": False}
    p = Path(script_path)
    name = p.name.lower()
    if "acft" in name:
        flags["acft"] = True
    if "dora" in name:
        flags["dora"] = True
    if "lora" in name:
        flags["lora"] = True
    if "qat" in name:
        flags["qat"] = True
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"^\s*FORCE_FULL_AUDIO_CTX\s*=\s*(True|False)", text, flags=re.M)
        if m:
            flags["dyn_ctx"] = (m.group(1) == "False")
        if re.search(r"WHISPER_LORA_USE_DORA\s*=\s*['\"]?1", text):
            flags["dora"] = True
        if re.search(r"WHISPER_USE_PEFT\s*=\s*['\"]?1", text) or "WHISPER_LORA_" in text:
            flags["lora"] = True
        m = re.search(r"^\s*QAT_ENABLE\s*=\s*(True|False)", text, flags=re.M)
        if m and m.group(1) == "True":
            flags["qat"] = True
    except Exception:
        pass
    return flags

def _qat_bits_from_env_or_script(script_path: str) -> str:
    env_bits = os.environ.get("WHISPER_QAT_BITS", "").strip()
    if env_bits.isdigit():
        return env_bits
    try:
        text = Path(script_path).read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"QAT_BITS\s*=\s*int\([^,]+,\s*['\"](\d+)['\"]", text)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "6"

def _count_manifest_rows(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "unk"
    try:
        total = 0
        with p.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                total += chunk.count(b"\n")
        if total > 0:
            return str(total)
    except Exception:
        pass
    return "unk"

def _canonical_base_tag_from_id(model_id: str) -> str:
    low = (model_id or "").lower()
    if "futo" in low:
        provider = "futo"
    elif "openai" in low:
        provider = "openai"
    else:
        provider = slug_run_token(low.split("/", 1)[0] if "/" in low else low, default="unk")
    size = "unk"
    for cand in ("tiny", "small", "base", "medium", "large"):
        if cand in low:
            size = cand
            break
    lang = "en" if (".en" in low or low.endswith("en") or "_en" in low) else "multi"
    if size == "unk":
        return "unk"
    return slug_run_token(f"{provider}-{size}-{lang}", default="unk")

def _canonical_method_tag(flags: dict) -> str:
    if RUN_NAME_METHOD_OVERRIDE:
        return slug_run_token(RUN_NAME_METHOD_OVERRIDE, default="unk")
    if flags["dora"]:
        return "s17-qat-dora" if flags["qat"] else "s17-dora"
    if flags["lora"]:
        return "s17-qat-lora" if flags["qat"] else "s17-lora"
    if flags["acft"]:
        return "s17-qat-full" if flags["qat"] else "s17-full"
    return "s17-unk"

def _canonical_adapter_tag(flags: dict) -> str:
    if RUN_NAME_ADAPTER_OVERRIDE:
        return slug_run_token(RUN_NAME_ADAPTER_OVERRIDE, default="unk")
    rank = os.environ.get("WHISPER_LORA_R", "").strip()
    alpha = os.environ.get("WHISPER_LORA_ALPHA", "").strip()
    if flags["dora"]:
        if rank.isdigit() and alpha.isdigit():
            return f"dora-r{rank}-a{alpha}"
        if rank.isdigit():
            return f"dora-r{rank}"
        return "dora"
    if flags["lora"]:
        if rank.isdigit() and alpha.isdigit():
            return f"lora-r{rank}-a{alpha}"
        if rank.isdigit():
            return f"lora-r{rank}"
        return "lora"
    if flags["acft"]:
        return "full"
    return "unk"

def _canonical_rows_tag() -> str:
    if RUN_NAME_ROWS_OVERRIDE:
        return slug_run_token(RUN_NAME_ROWS_OVERRIDE, default="unk")
    return _count_manifest_rows(STAGE15_TRAIN)

def _canonical_kind_tag() -> str:
    if RUN_NAME_KIND_OVERRIDE:
        return slug_run_token(RUN_NAME_KIND_OVERRIDE, default="unk")
    return "train-eval" if (RUN_EVAL_19C or RUN_EVAL_19D) else "train-only"

def _canonical_ctx_tag(flags: dict) -> str:
    if flags["dyn_ctx"] is True:
        return "dyn"
    if flags["dyn_ctx"] is False:
        return "static"
    return "unk"

def _canonical_run_id_tag() -> str:
    base = RUN_NAME_ID_HINT or "auto"
    if STAGE17_CHECKPOINT_TAG:
        base = f"{base}-{STAGE17_CHECKPOINT_TAG}"
    return slug_run_token(base, default="auto")

def _stage17_canonical_prefix(flags: dict) -> str:
    return build_run_folder_name(
        kind=_canonical_kind_tag(),
        stage="17",
        base=_canonical_base_tag_from_id(BASE_MODEL_ID),
        method=_canonical_method_tag(flags),
        adapter=_canonical_adapter_tag(flags),
        quant="qat" if flags["qat"] else "noqat",
        ctx=_canonical_ctx_tag(flags),
        rows=_canonical_rows_tag(),
        run_id=_canonical_run_id_tag(),
    )

def _stage17_checkpoint_prefix() -> str:
    if STAGE17_CHECKPOINT_PREFIX:
        return STAGE17_CHECKPOINT_PREFIX
    script_path = _stage17_script_path()
    flags = _detect_stage17_script_flags(str(script_path))
    if USE_CANONICAL_RUN_FOLDER_NAMING:
        return _stage17_canonical_prefix(flags)
    parts = ["Stage_17", _stage17_aug_tag(), _provider_tag_from_id(BASE_MODEL_ID), "wer"]
    if flags["acft"]:
        parts.append("acft")
    if flags["dora"]:
        parts.append("dora")
    elif flags["lora"]:
        parts.append("lora")
    if flags["dyn_ctx"] is True:
        parts.append("dyn_ctx")
    elif flags["dyn_ctx"] is False:
        parts.append("full_ctx")
    if flags["qat"]:
        parts.append(f"qat{_qat_bits_from_env_or_script(str(script_path))}_0")
    if STAGE17_CHECKPOINT_TAG:
        parts.append(_sanitize_checkpoint_tag(STAGE17_CHECKPOINT_TAG))
    parts.append("chkpts")
    parts.append(_model_tag_from_id(BASE_MODEL_ID))
    return "_".join(p for p in parts if p)

def _split_tokens(value: str) -> list[str]:
    return [t for t in value.lower().split("_") if t]

def _strip_checkpoint_index(name: str) -> tuple[str, int | None]:
    base = name[:-7] if name.endswith("_merged") else name
    m = re.match(r"^(.*?)(?:_(\d+))$", base)
    if m:
        return m.group(1), int(m.group(2))
    return base, None

def _desired_stage17_tokens() -> tuple[list[str], list[str], list[str]]:
    base_tokens: list[str] = []
    def add(token: str) -> None:
        base_tokens.extend(_split_tokens(token))

    add("Stage_17")
    add(_stage17_aug_tag())
    add(_provider_tag_from_id(BASE_MODEL_ID))
    add("wer")
    flags = _detect_stage17_script_flags(str(_stage17_script_path()))
    if flags["dora"]:
        add("dora")
    elif flags["lora"]:
        add("lora")
    if flags["dyn_ctx"] is True:
        add("dyn_ctx")
    elif flags["dyn_ctx"] is False:
        add("full_ctx")
    add("chkpts")
    model_tag = _model_tag_from_id(BASE_MODEL_ID)
    add(model_tag)

    optional_tokens: list[str] = []
    if flags["acft"]:
        optional_tokens.extend(_split_tokens("acft"))
    if flags["qat"]:
        optional_tokens.extend(_split_tokens(f"qat{_qat_bits_from_env_or_script(str(_stage17_script_path()))}_0"))

    model_tokens = _split_tokens(model_tag)
    size_token = model_tokens[0] if model_tokens else "model"
    core_tokens = ["stage", "17", _provider_tag_from_id(BASE_MODEL_ID), "wer", size_token]
    if "en" in model_tokens:
        core_tokens.append("en")
    return base_tokens, optional_tokens, core_tokens

def _pick_existing_stage17_prefix(root: Path) -> str | None:
    base_tokens, optional_tokens, core_tokens = _desired_stage17_tokens()
    best_prefix = None
    best_score = -1
    best_idx = -1
    try:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            base, idx = _strip_checkpoint_index(child.name)
            tokens = _split_tokens(base)
            if not all(t in tokens for t in core_tokens):
                continue
            if "chkpts" not in tokens and "checkpoints" not in tokens:
                continue
            score = sum(1 for t in base_tokens + optional_tokens if t in tokens)
            idx_val = idx if idx is not None else 0
            if score > best_score or (score == best_score and idx_val > best_idx):
                best_prefix = base
                best_score = score
                best_idx = idx_val
    except Exception:
        return None
    return best_prefix

def _checkpoint_has_state(path: Path) -> bool:
    try:
        if (path / "run_state.json").exists():
            return True
        for child in path.iterdir():
            if child.is_dir() and child.name.startswith("model_epoch_"):
                return True
            if child.is_file() and child.name.startswith("training_state_epoch_"):
                return True
    except Exception:
        return False
    return False

def _parse_checkpoint_index(name: str, prefix: str) -> int | None:
    if name.endswith("_merged"):
        return None
    if name == prefix:
        return 0
    if not name.startswith(prefix + "_"):
        return None
    tail = name[len(prefix) + 1 :]
    m = re.match(r"^(\d+)$", tail)
    if not m:
        return None
    return int(m.group(1))

def resolve_stage17_checkpoint_dir() -> str:
    if CHECKPOINT_DIR:
        return CHECKPOINT_DIR
    root = Path(CHECKPOINTS_ROOT) if CHECKPOINTS_ROOT else Path(DATA_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    prefix = _stage17_checkpoint_prefix() or "Stage_17_chkpts"
    if not STAGE17_CHECKPOINT_PREFIX and STAGE17_REUSE_EXISTING_PREFIX:
        existing_prefix = _pick_existing_stage17_prefix(root)
        if existing_prefix:
            prefix = existing_prefix
    candidates = []
    try:
        for child in root.iterdir():
            if not child.is_dir():
                continue
            idx = _parse_checkpoint_index(child.name, prefix)
            if idx is None:
                continue
            candidates.append((idx, child.stat().st_mtime, _checkpoint_has_state(child), child))
    except Exception:
        candidates = []
    if candidates and not START_FRESH:
        with_state = [c for c in candidates if c[2]]
        pick = max(with_state or candidates, key=lambda c: (c[0], c[1]))
        return str(pick[3])
    next_idx = max((c[0] for c in candidates), default=0) + 1
    return str(root / f"{prefix}_{next_idx}")

CHECKPOINT_DIR = resolve_stage17_checkpoint_dir()
log(f"[stage17] CHECKPOINT_DIR = {CHECKPOINT_DIR}")

def run(cmd):
    if isinstance(cmd, str):
        log(f"$ {cmd}")
        subprocess.run(cmd, shell=True, check=True)
    else:
        log("$ " + " ".join(shlex.quote(str(c)) for c in cmd))
        subprocess.run([str(c) for c in cmd], check=True)

def run_stage(cmd):
    if isinstance(cmd, str):
        log(f"$ {cmd}")
        if CAPTURE_STAGE_OUTPUT:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            p = subprocess.run(cmd, shell=True)
    else:
        log("$ " + " ".join(shlex.quote(str(c)) for c in cmd))
        if CAPTURE_STAGE_OUTPUT:
            p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
        else:
            p = subprocess.run([str(c) for c in cmd])
    if CAPTURE_STAGE_OUTPUT:
        if p.stdout:
            print("STDOUT:\n", p.stdout)
        if p.stderr:
            print("STDERR:\n", p.stderr)
    if p.returncode != 0:
        raise subprocess.CalledProcessError(p.returncode, p.args, output=getattr(p, "stdout", None), stderr=getattr(p, "stderr", None))

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

def patch_winsound_simple(path):
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if "_WinSoundShim" in txt or "import winsound" not in txt:
        return
    new = txt.replace(
        "import winsound",
        "try:\n    import winsound  # type: ignore\n"
        "except Exception:\n"
        "    class _WinSoundShim:\n"
        "        def Beep(self, *a, **k):\n            pass\n"
        "    winsound = _WinSoundShim()"
    )
    if new != txt:
        p.write_text(new, encoding="utf-8")

def patch_stage1_tokenizer(path):
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if "WhisperProcessor" not in txt:
        return
    txt = txt.replace("from transformers import WhisperProcessor", "from transformers import AutoTokenizer")
    old = "processor = WhisperProcessor.from_pretrained(BASE_PROCESSOR_ID)"
    new = (
        "def _load_tokenizer():\n"
        "    try:\n"
        "        from transformers import WhisperTokenizerFast\n"
        "        return WhisperTokenizerFast.from_pretrained(BASE_PROCESSOR_ID)\n"
        "    except Exception:\n"
        "        pass\n"
        "    try:\n"
        "        from transformers import WhisperTokenizer\n"
        "        return WhisperTokenizer.from_pretrained(BASE_PROCESSOR_ID)\n"
        "    except Exception:\n"
        "        pass\n"
        "    return AutoTokenizer.from_pretrained(BASE_PROCESSOR_ID)\n\n"
        "tokenizer = _load_tokenizer()"
    )
    if old in txt:
        txt = txt.replace(old, new)
    txt = txt.replace("processor.tokenizer(", "tokenizer(")
    p.write_text(txt, encoding="utf-8")

def patch_stage9_reverb_mono(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    txt = p.read_text(encoding="utf-8")
    if "_ensure_mono" not in txt:
        return
    # Add mono guards in convolution and after RIR load.
    if "def _fft_convolve_same" in txt and "x = _ensure_mono(x)" not in txt:
        txt = txt.replace(
            "def _fft_convolve_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:\n"
            "    \"\"\"Fast-ish convolution returning same length as x.\"\"\"\n",
            "def _fft_convolve_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:\n"
            "    \"\"\"Fast-ish convolution returning same length as x.\"\"\"\n"
            "    if x.ndim > 1:\n"
            "        x = _ensure_mono(x)\n"
            "    if h.ndim > 1:\n"
            "        h = _ensure_mono(h)\n",
        )
    if "rir, sr_r = sf.read" in txt and "rir = _ensure_mono(rir)" not in txt:
        txt = txt.replace(
            "    rir, sr_r = sf.read(rir_path, dtype=\"float32\", always_2d=False)\n",
            "    rir, sr_r = sf.read(rir_path, dtype=\"float32\", always_2d=False)\n"
            "    rir = _ensure_mono(rir).astype(np.float32)\n",
        )
    p.write_text(txt, encoding="utf-8")

def patch_stage12_scores_fallback(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    txt = p.read_text(encoding="utf-8")
    insert = (
        "def candidate_keys(entry: dict) -> list[str]:\n"
        "    \"\"\"Return possible canonical keys to match scores (audio_path first).\"\"\"\n"
        "    keys = []\n"
        "    for k in (\n"
        "        \"audio_path\",\n"
        "        \"source_audio\",\n"
        "        \"source_audio_path\",\n"
        "        \"original_audio\",\n"
        "        \"orig_audio\",\n"
        "        \"base_audio\",\n"
        "    ):\n"
        "        v = entry.get(k)\n"
        "        if isinstance(v, str) and v:\n"
        "            keys.append(canonical_key(v))\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for k in keys:\n"
        "        if k and k not in seen:\n"
        "            out.append(k)\n"
        "            seen.add(k)\n"
        "    return out\n\n\n"
    )
    if "def candidate_keys" not in txt and "def canonical_key" in txt:
        txt = txt.replace(
            "    p = re.sub(r\"/+\", \"/\", p)\n    return p.casefold()\n\n\n",
            "    p = re.sub(r\"/+\", \"/\", p)\n    return p.casefold()\n\n\n"
            "def canonical_rel_key(p: str) -> str:\n"
            "    \"\"\"Canonical key using relative tail for common roots.\"\"\"\n"
            "    if not p:\n"
            "        return \"\"\n"
            "    p = canonical_key(p)\n"
            "    for marker in (\"/record_chunks/\", \"/record_harsha/\"):\n"
            "        idx = p.find(marker)\n"
            "        if idx != -1:\n"
            "            return p[idx:]\n"
            "    return p\n\n\n" + insert,
        )
    txt = txt.replace(
        "        'kept_with_score': 0,\n        'kept_no_score': 0\n    }\n",
        "        'kept_with_score': 0,\n        'kept_no_score': 0,\n        'matched_audio_path': 0,\n        'matched_fallback': 0\n    }\n",
    )
    txt = txt.replace(
        "    keys = []\n"
        "    for k in (\n"
        "        \"audio_path\",\n"
        "        \"source_audio\",\n"
        "        \"source_audio_path\",\n"
        "        \"original_audio\",\n"
        "        \"orig_audio\",\n"
        "        \"base_audio\",\n"
        "    ):\n"
        "        v = entry.get(k)\n"
        "        if isinstance(v, str) and v:\n"
        "            keys.append(canonical_key(v))\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for k in keys:\n"
        "        if k and k not in seen:\n"
        "            out.append(k)\n"
        "            seen.add(k)\n"
        "    return out\n\n\n",
        "    keys = []\n"
        "    for k in (\n"
        "        \"audio_path\",\n"
        "        \"source_audio\",\n"
        "        \"source_audio_path\",\n"
        "        \"original_audio\",\n"
        "        \"orig_audio\",\n"
        "        \"base_audio\",\n"
        "    ):\n"
        "        v = entry.get(k)\n"
        "        if isinstance(v, str) and v:\n"
        "            keys.append(canonical_key(v))\n"
        "    seen = set()\n"
        "    out = []\n"
        "    for k in keys:\n"
        "        if k and k not in seen:\n"
        "            out.append(k)\n"
        "            seen.add(k)\n"
        "    rels = []\n"
        "    for k in out:\n"
        "        rk = canonical_rel_key(k)\n"
        "        if rk and rk not in seen:\n"
        "            rels.append(rk)\n"
        "            seen.add(rk)\n"
        "    out.extend(rels)\n"
        "    return out\n\n\n",
    )
    txt = txt.replace(
        "    normalized_score_dict = {canonical_key(k): v for k, v in score_dict.items()}\n",
        "    normalized_score_dict = {}\n"
        "    for k, v in score_dict.items():\n"
        "        ck = canonical_key(k)\n"
        "        normalized_score_dict[ck] = v\n"
        "        rk = canonical_rel_key(k)\n"
        "        if rk:\n"
        "            normalized_score_dict[rk] = v\n",
    )
    txt = txt.replace(
        "        audio_path_key = canonical_key(entry.get('audio_path',''))\n        \n        if audio_path_key in normalized_score_dict:\n            score = normalized_score_dict[audio_path_key]\n",
        "        keys = candidate_keys(entry)\n"
        "        score = None\n"
        "        if keys:\n"
        "            if keys[0] in normalized_score_dict:\n"
        "                score = normalized_score_dict[keys[0]]\n"
        "                stats['matched_audio_path'] += 1\n"
        "            else:\n"
        "                for k in keys[1:]:\n"
        "                    if k in normalized_score_dict:\n"
        "                        score = normalized_score_dict[k]\n"
        "                        stats['matched_fallback'] += 1\n"
        "                        break\n"
        "\n"
        "        if score is not None:\n",
    )
    txt = txt.replace(
        "    print(f\"  - No scores found: {stats['kept_no_score']:,}\")\n",
        "    print(f\"  - No scores found: {stats['kept_no_score']:,}\")\n"
        "    print(f\"Matched by audio_path: {stats['matched_audio_path']:,}\")\n"
        "    print(f\"Matched by fallback keys: {stats['matched_fallback']:,}\")\n",
    )
    p.write_text(txt, encoding="utf-8")

def patch_stage14_scores_robust(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    txt = p.read_text(encoding="utf-8")
    if "def canonical_rel_key" not in txt:
        if "import re" not in txt:
            txt = txt.replace("import pandas as pd\n", "import pandas as pd\nimport re\n")
        insert = (
            "\n\ndef canonical_key(p: str) -> str:\n"
            "    if not p:\n"
            "        return \"\"\n"
            "    p = str(p).strip().strip('\"').strip(\"'\")\n"
            "    p = p.replace('\\\\', '/')\n"
            "    p = re.sub(r\"/+\", \"/\", p)\n"
            "    return p.casefold()\n\n"
            "def canonical_rel_key(p: str) -> str:\n"
            "    if not p:\n"
            "        return \"\"\n"
            "    p = canonical_key(p)\n"
            "    for marker in (\"/record_chunks/\", \"/record_harsha/\"):\n"
            "        idx = p.find(marker)\n"
            "        if idx != -1:\n"
            "            return p[idx:]\n"
            "    return p\n"
        )
        if "import winsound" in txt:
            txt = txt.replace("import winsound  # For beep notification\n", "import winsound  # For beep notification\n" + insert + "\n")
        else:
            txt = txt.replace("from tqdm import tqdm\n", "from tqdm import tqdm\n" + insert + "\n")
    if "try:" not in txt or "on_bad_lines='skip'" not in txt:
        txt = re.sub(
            r"(?m)^(\\s*)df = pd\\.read_csv\\(csv_path\\)\\s*$",
            r"\\1try:\n"
            r"\\1    df = pd.read_csv(csv_path)\n"
            r"\\1except pd.errors.ParserError as e:\n"
            r"\\1    print(f\"CSV parsing error: {e}\")\n"
            r"\\1    print(\"Attempting to read with more robust settings...\")\n"
            r"\\1    df = pd.read_csv(csv_path, on_bad_lines='skip', quoting=3)",
            txt,
        )
    txt = txt.replace(
        "            target_files.add(file_path.lower())\n",
        "            ck = canonical_key(file_path)\n"
        "            target_files.add(ck)\n"
        "            rk = canonical_rel_key(file_path)\n"
        "            if rk:\n"
        "                target_files.add(rk)\n",
    )
    txt = txt.replace(
        "    # Normalize target files set to lowercase for case-insensitive matching\n"
        "    normalized_target_files = {f.lower() for f in target_files}\n"
        "    \n",
        "",
    )
    txt = txt.replace(
        "        audio_path = entry.get('audio_path', '').lower()\n"
        "        \n"
        "        if audio_path in normalized_target_files:\n",
        "        audio_path = entry.get('audio_path', '')\n"
        "        source_audio = entry.get('source_audio', '')\n"
        "        keys = []\n"
        "        if audio_path:\n"
        "            keys.extend([canonical_key(audio_path), canonical_rel_key(audio_path)])\n"
        "        if source_audio:\n"
        "            keys.extend([canonical_key(source_audio), canonical_rel_key(source_audio)])\n"
        "        \n"
        "        if any(k in target_files for k in keys if k):\n",
    )
    txt = txt.replace(
        "            # Check if this file was in the CSV at all\n"
        "            if audio_path and not any(audio_path == target_file for target_file in normalized_target_files):\n"
        "                # We don't have info about this file from the CSV\n"
        "                pass  # This is normal, many files won't be in the scores CSV\n",
        "",
    )
    p.write_text(txt, encoding="utf-8")

def stage_banner(name: str) -> None:
    log("\n\n" + "=" * 20 + f" {name} " + "=" * 20)

def is_peft_checkpoint_dir(path: str) -> bool:
    return Path(path, "adapter_config.json").exists()

def find_latest_checkpoint_dir(checkpoint_dir: str) -> str | None:
    p = Path(checkpoint_dir)
    if not p.exists():
        return None
    best = None
    best_epoch = -1
    for child in p.iterdir():
        if child.is_dir() and child.name.startswith("model_epoch_"):
            try:
                epoch = int(child.name.split("_")[-1])
            except Exception:
                continue
            if epoch > best_epoch:
                best_epoch = epoch
                best = str(child)
    return best

def best_checkpoint_dir() -> str | None:
    return find_latest_checkpoint_dir(CHECKPOINT_DIR)

def sync_charts_to_drive(paths):
    if not SAVE_CHARTS_TO_DRIVE:
        return
    dst_root = Path(CHARTS_DRIVE_DIR)
    dst_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in paths:
        src = Path(p)
        if not src.exists():
            continue
        dst = dst_root / src.name
        shutil.copy2(src, dst)
        copied += 1
    if copied:
        log(f"[sync] charts -> {dst_root} ({copied} file(s))")

def save_chart_from_metrics(json_path: str, out_png: str, title: str) -> None:
    import matplotlib.pyplot as plt
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Metrics JSON not found: {json_path}")
    data = json_loads(p.read_text(encoding="utf-8"))

    plt.figure(figsize=(8, 4))
    if isinstance(data, list) and data and isinstance(data[0], dict):
        # Try line plot: epoch vs wer
        xs, ys = [], []
        for row in data:
            if "epoch" in row and ("wer" in row or "avg_wer_target" in row):
                xs.append(row.get("epoch"))
                ys.append(row.get("wer", row.get("avg_wer_target")))
        if xs and ys:
            plt.plot(xs, ys, marker="o")
            plt.xlabel("epoch")
            plt.ylabel("WER")
        else:
            # Fallback: histogram of WER-like values
            vals = [row.get("wer") for row in data if isinstance(row.get("wer"), (int, float))]
            if vals:
                plt.hist(vals, bins=20)
                plt.xlabel("WER")
                plt.ylabel("count")
    elif isinstance(data, dict):
        keys = ["avg_wer_target", "avg_wer_other", "wer"]
        vals = {k: data[k] for k in keys if k in data and isinstance(data[k], (int, float))}
        if vals:
            plt.bar(list(vals.keys()), list(vals.values()))
            plt.ylabel("WER")

    plt.title(title)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

def _torch_index_url() -> str:
    if TORCH_INDEX_URL:
        return TORCH_INDEX_URL
    try:
        import torch
        cuda = (torch.version.cuda or "").strip()
    except Exception:
        return "https://download.pytorch.org/whl/cpu"
    if not cuda:
        try:
            # If GPU exists but torch is CPU-only, prefer CUDA wheels.
            r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if r.returncode == 0:
                return "https://download.pytorch.org/whl/cu121"
        except Exception:
            pass
    if cuda.startswith("12.4"):
        return "https://download.pytorch.org/whl/cu124"
    if cuda.startswith("12.1"):
        return "https://download.pytorch.org/whl/cu121"
    if cuda.startswith("11.8"):
        return "https://download.pytorch.org/whl/cu118"
    return "https://download.pytorch.org/whl/cpu"

def _torch_smoke_check() -> tuple[bool, str]:
    code = (
        "import torch, torchvision\n"
        "import sys\n"
        "ok = True\n"
        "if " + ("True" if DEVICE == "cuda" else "False") + " and not torch.cuda.is_available():\n"
        "    ok = False\n"
        "ver = f\"torch={torch.__version__} torchvision={torchvision.__version__} cuda={torch.version.cuda or 'cpu'}\"\n"
        "print(ver)\n"
        "sys.exit(0 if ok else 2)\n"
    )
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    msg = (r.stdout or r.stderr or "").strip()
    return (r.returncode == 0, msg)

def ensure_torchvision_compat():
    if not FIX_TORCHVISION:
        return
    if TORCH_FIX_ONCE and Path(TORCH_FIX_MARKER).exists():
        return
    ok, msg = _torch_smoke_check()
    if ok:
        log(f"[fix] torch/torchvision OK: {msg}")
        if TORCH_FIX_ONCE:
            Path(TORCH_FIX_MARKER).parent.mkdir(parents=True, exist_ok=True)
            Path(TORCH_FIX_MARKER).write_text("ok\n", encoding="utf-8")
        return
    log(f"[fix] torch/torchvision check failed: {msg}")
    url = _torch_index_url()
    log(f"[fix] reinstall torch/torchvision/torchaudio from {url}")
    pip_args = [PY, "-m", "pip", "install", "-q", "--upgrade", "--index-url", url]
    if TORCH_FORCE_REINSTALL:
        pip_args.append("--force-reinstall")
    run(pip_args + ["torch", "torchvision", "torchaudio"])
    ok, msg = _torch_smoke_check()
    if ok:
        log(f"[fix] torch/torchvision OK after install: {msg}")
        if TORCH_FIX_ONCE:
            Path(TORCH_FIX_MARKER).parent.mkdir(parents=True, exist_ok=True)
            Path(TORCH_FIX_MARKER).write_text("ok\n", encoding="utf-8")
    else:
        log(f"[fix] torch/torchvision still failing after install: {msg}")
        log("[fix] if this persists, restart the runtime to clear cached imports")

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
    rsync_args = ["rsync", "-a", "--info=progress2"]
    if CACHE_RSYNC_SIZE_ONLY:
        rsync_args.append("--size-only")
    run(rsync_args + [f"{src}/", f"{CACHE_ROOT}/"])
    log(f"[cache] restored from {src}")

def sync_cache_to_drive():
    if not SYNC_CACHE_TO_DRIVE:
        return
    src = Path(CACHE_ROOT)
    if not src.exists():
        return
    dst = Path(CACHE_DRIVE_ROOT)
    dst.mkdir(parents=True, exist_ok=True)
    rsync_args = ["rsync", "-a", "--info=progress2"]
    if CACHE_RSYNC_SIZE_ONLY:
        rsync_args.append("--size-only")
    run(rsync_args + [f"{src}/", f"{dst}/"])
    log(f"[cache] synced to {dst}")

def sync_record_chunks_to_drive():
    if not SYNC_RECORD_CHUNKS_TO_DRIVE:
        return
    src = Path(CHUNKS_DIR)
    if not src.exists():
        return
    dst = Path(DRIVE_RECORD_CHUNKS)
    dst.mkdir(parents=True, exist_ok=True)
    if USE_RCLONE and RCLONE_REMOTE:
        rel = dst.relative_to(Path(DATA_ROOT_DRIVE))
        remote_dst = f"{RCLONE_REMOTE}/{rel.as_posix()}"
        run(["rclone", "copy", str(src), remote_dst, "--transfers", str(RCLONE_TRANSFERS)])
        log(f"[sync] record chunks -> {remote_dst}")
    else:
        run(["rsync", "-a", "--info=progress2", f"{src}/", f"{dst}/"])
        log(f"[sync] record chunks -> {dst}")

def restore_record_chunks_from_drive():
    if not (SYNC_RECORD_CHUNKS_TO_DRIVE and USE_LOCAL_DATA and RESTORE_RECORD_CHUNKS):
        return
    src = Path(DRIVE_RECORD_CHUNKS)
    if not src.exists():
        return
    dst = Path(CHUNKS_DIR)
    dst.mkdir(parents=True, exist_ok=True)
    if USE_RCLONE and RCLONE_REMOTE:
        rel = src.relative_to(Path(DATA_ROOT_DRIVE))
        remote_src = f"{RCLONE_REMOTE}/{rel.as_posix()}"
        run(["rclone", "copy", remote_src, str(dst), "--transfers", str(RCLONE_TRANSFERS)])
        log(f"[restore] record chunks <- {remote_src}")
    else:
        run(["rsync", "-a", "--info=progress2", f"{src}/", f"{dst}/"])
        log(f"[restore] record chunks <- {src}")

COPY_EVENTS = {}
BACKGROUND_COPY_THREAD = None

def _drive_path_for_local(local_path: str) -> Path | None:
    p = Path(local_path)
    for root in (Path(DATA_ROOT), Path(LOCAL_DATA_ROOT), Path(DATA_ROOT_DRIVE)):
        try:
            rel = p.relative_to(root)
            return Path(DATA_ROOT_DRIVE) / rel
        except Exception:
            continue
    return None

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

def normalize_to_drive_path(path_str: str) -> Path | None:
    p = Path(path_str)
    for root in (Path(DATA_ROOT_DRIVE), Path(DATA_ROOT), Path(LOCAL_DATA_ROOT)):
        try:
            rel = p.relative_to(root)
            return Path(DATA_ROOT_DRIVE) / rel
        except Exception:
            continue
    return None

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
        drive_p = normalize_to_drive_path(p)
        if drive_p is None:
            continue
        try:
            rel = drive_p.relative_to(Path(DATA_ROOT_DRIVE))
            rels.append(rel.as_posix())
        except Exception:
            continue
    return rels

def rsync_files_from(rel_paths: list[str], src_root: str, dst_root: str, label=""):
    if not rel_paths:
        return
    tmp = Path(os.environ.get("TEMP", "/tmp")) / "rsync_files.txt"
    tmp.write_text("\n".join(rel_paths) + "\n", encoding="utf-8")
    run(["rsync", "-a", "--info=progress2", f"--files-from={tmp}", f"{src_root}/", f"{dst_root}/"])
    log(f"[copy] files-from {label}: {len(rel_paths)} files")

def _remap_path(val: str, new_root: str) -> str:
    for root in (DATA_ROOT_DRIVE, DATA_ROOT, LOCAL_DATA_ROOT):
        if val.startswith(root):
            return new_root + val[len(root):]
    return val

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
                if isinstance(v, str):
                    nv = _remap_path(v, new_root)
                    if nv != v:
                        obj[k] = nv
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

def _file_signature(path: Path) -> dict:
    if path.is_dir():
        total = 0
        count = 0
        for p in path.rglob("*"):
            if p.is_file():
                count += 1
                try:
                    total += p.stat().st_size
                except Exception:
                    pass
        return {"type": "dir", "files": count, "bytes": total}
    try:
        size = path.stat().st_size
    except Exception:
        size = -1
    sig = {"type": "file", "bytes": size}
    if path.suffix.lower() == ".jsonl":
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                rows = sum(1 for _ in f)
            sig["rows"] = rows
        except Exception:
            pass
    try:
        sig["sha256"] = file_hash(path)
    except Exception:
        pass
    return sig

def outputs_signature(paths) -> dict:
    if not ENABLE_OUTPUT_SIGNATURES:
        return {}
    out = {}
    for p in paths:
        path = Path(p)
        if path.exists():
            out[str(path)] = _file_signature(path)
    return out

def record_outputs_signature(stage: str, paths) -> None:
    if not ENABLE_OUTPUT_SIGNATURES:
        return
    state = PIPELINE_STATE
    sigs = state.get("output_sigs", {})
    sigs[stage] = outputs_signature(paths)
    state["output_sigs"] = sigs
    save_pipeline_state(state)

def outputs_signature_matches(stage: str, paths) -> bool:
    if not ENABLE_OUTPUT_SIGNATURES:
        return False
    sigs = PIPELINE_STATE.get("output_sigs", {})
    if stage not in sigs:
        return False
    return sigs.get(stage) == outputs_signature(paths)

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
        record_outputs_signature(stage, paths)
        return
    missing = [str(p) for p in paths if not Path(p).exists()]
    raise RuntimeError(f"[{stage}] expected outputs missing/empty: {missing}")

STAGE_ORDER_FOR_AUTO_SKIP = [
    "stage_1",
    "stage_2",
    "stage_3",
    "stage_3b",
    "stage_4",
    "stage_5",
    "stage_6",
    "stage_7",
    "stage_8",
    "stage_9",
    "stage_10b",
    "stage_11",
    "stage_12",
    "stage_13",
    "stage_14",
    "stage_15",
    "stage_16",
]

STAGE_OUTPUTS_FOR_AUTO_SKIP = {
    "stage_1": [TASKS_PENDING, PAIRS_PENDING],
    "stage_2": [STAGE2_MANIFEST],
    "stage_3": [SCORES_CSV],
    "stage_3b": [STAGE3B_MANIFEST],
    "stage_4": [STAGE4_MANIFEST],
    "stage_5": [STAGE5_MANIFEST],
    "stage_6": [STAGE6_MANIFEST],
    "stage_7": [STAGE7_MANIFEST],
    "stage_8": [STAGE8_MANIFEST],
    "stage_9": [STAGE9_MANIFEST],
    "stage_10b": [STAGE10B_MANIFEST],
    "stage_11": [STAGE11_MANIFEST],
    "stage_12": [STAGE12_MANIFEST],
    "stage_13": [STAGE13_TRAIN, STAGE13_TEST],
    "stage_14": [STAGE14_TRAIN],
    "stage_15": [STAGE15_TRAIN],
    "stage_16": [STAGE13_TEST, STAGE16_BACKUP],
}

STAGE_SKIP_VAR_MAP = {
    "stage_1": "SKIP_STAGE_1_MANIFEST_CREATION",
    "stage_2": "SKIP_STAGE_2_CHUNK_TRANSCRIPTS",
    "stage_3": "SKIP_STAGE_3_SPEAKER_SORT",
    "stage_3b": "SKIP_STAGE_3B_FILTER_ENGLISH",
    "stage_4": "SKIP_STAGE_4_DELETE_COMMON_FILLERS",
    "stage_5": "SKIP_STAGE_5_CONVERT_NUMBER_WORDS",
    "stage_6": "SKIP_STAGE_6_ADD_NOISE",
    "stage_7": "SKIP_STAGE_7_ADD_OTHER_VOICES",
    "stage_8": "SKIP_STAGE_8_ADD_RANDOM_GAIN",
    "stage_9": "SKIP_STAGE_9_ADD_REVERB",
    "stage_10b": "SKIP_STAGE_10B_ADD_TEMPO_PAUSE",
    "stage_11": "SKIP_STAGE_11_ADD_FREQUENCY",
    "stage_12": "SKIP_STAGE_12_REMOVE_BOTTOM_PERCENT",
    "stage_13": "SKIP_STAGE_13_SPLIT_TRAIN_TEST",
    "stage_14": "SKIP_STAGE_14_REMOVE_TARGET_FILES",
    "stage_15": "SKIP_STAGE_15_RANDOMIZE_MANIFEST",
    "stage_16": "SKIP_STAGE_16_MOVE_TEST_CHUNKS",
}

def _infer_latest_completed_stage() -> str | None:
    for stage in reversed(STAGE_ORDER_FOR_AUTO_SKIP):
        outputs = STAGE_OUTPUTS_FOR_AUTO_SKIP.get(stage)
        if outputs and outputs_ok(outputs):
            return stage
    return None

def _apply_auto_skip_upstream_from_outputs() -> None:
    if not AUTO_SKIP_UPSTREAM_FROM_OUTPUTS:
        return
    latest = _infer_latest_completed_stage()
    if not latest:
        return
    log(f"[auto] latest completed stage from outputs: {latest}; skipping upstream stages")
    for stage in STAGE_ORDER_FOR_AUTO_SKIP:
        if stage == latest:
            break
        var = STAGE_SKIP_VAR_MAP.get(stage)
        if var:
            globals()[var] = True

_apply_auto_skip_upstream_from_outputs()

SKIP_FLAGS = {
    "stage_1": SKIP_STAGE_1_MANIFEST_CREATION,
    "stage_2": SKIP_STAGE_2_CHUNK_TRANSCRIPTS,
    "stage_3": SKIP_STAGE_3_SPEAKER_SORT,
    "stage_3b": SKIP_STAGE_3B_FILTER_ENGLISH,
    "stage_4": SKIP_STAGE_4_DELETE_COMMON_FILLERS,
    "stage_5": SKIP_STAGE_5_CONVERT_NUMBER_WORDS,
    "stage_6": SKIP_STAGE_6_ADD_NOISE,
    "stage_7": SKIP_STAGE_7_ADD_OTHER_VOICES,
    "stage_8": SKIP_STAGE_8_ADD_RANDOM_GAIN,
    "stage_9": SKIP_STAGE_9_ADD_REVERB,
    "stage_10b": SKIP_STAGE_10B_ADD_TEMPO_PAUSE,
    "stage_11": SKIP_STAGE_11_ADD_FREQUENCY,
    "stage_12": SKIP_STAGE_12_REMOVE_BOTTOM_PERCENT,
    "stage_13": SKIP_STAGE_13_SPLIT_TRAIN_TEST,
    "stage_14": SKIP_STAGE_14_REMOVE_TARGET_FILES,
    "stage_15": SKIP_STAGE_15_RANDOMIZE_MANIFEST,
    "stage_16": SKIP_STAGE_16_MOVE_TEST_CHUNKS,
    "stage_17": SKIP_STAGE_17_TRAIN,
}

def should_run(stage, outputs, force=False):
    if SKIP_FLAGS.get(stage, False):
        log(f"[skip] {stage} (flag set)")
        update_stage_state(stage, "skipped")
        return False
    if force:
        return True
    outputs_ready = outputs_ok(outputs)
    if outputs_ready and AUTO_SKIP_IF_OUTPUTS_PRESENT and stage not in AUTO_SKIP_EXCLUDE_STAGES:
        log(f"[skip] {stage} outputs already exist")
        update_stage_state(stage, "skipped_existing")
        return False
    completed = set(PIPELINE_STATE.get("completed", []))
    if outputs_ready and stage in completed:
        log(f"[skip] {stage} already completed and outputs present")
        update_stage_state(stage, "skipped_existing")
        return False
    if outputs_ready and outputs_signature_matches(stage, outputs):
        log(f"[skip] {stage} outputs match previous signature")
        update_stage_state(stage, "skipped_existing")
        return False
    return True

def _count_jsonl_rows(path: Path) -> int:
    try:
        with path.open("rb") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0

def _stage2_done_count(state_path: Path) -> int:
    if not state_path.exists():
        return 0
    try:
        obj = json_loads(state_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return 0
    done = obj.get("done")
    if isinstance(done, list):
        return len(done)
    if isinstance(done, int):
        return int(done)
    return 0

def stage2_needs_resume() -> bool:
    tasks_path = Path(TASKS_PENDING)
    if not tasks_path.exists():
        return False
    if STAGE2_CHECK_CHUNKS:
        total, missing = stage2_manifest_missing_chunks(Path(STAGE2_MANIFEST))
        if total > 0 and missing > 0:
            log(f"[stage2] manifest missing chunks: {missing}/{total}")
            if STAGE2_RESET_ON_MISSING:
                reset_stage2_outputs()
            return True
    remaining_path = Path(CHUNKS_DIR) / "tasks_remaining.jsonl"
    if remaining_path.exists() and _count_jsonl_rows(remaining_path) > 0:
        return True
    state_path = Path(CHUNKS_DIR) / "stage2_cut_state.json"
    done_count = _stage2_done_count(state_path)
    if done_count <= 0:
        return False
    total = _count_jsonl_rows(tasks_path)
    return total > 0 and done_count < total

def stage2_manifest_missing_chunks(manifest_path: Path) -> tuple[int, int]:
    if not manifest_path.exists():
        return 0, 0
    total = 0
    missing = 0
    try:
        with manifest_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json_loads(line)
                except Exception:
                    continue
                audio_path = obj.get("audio_path") or obj.get("out_wav")
                if not isinstance(audio_path, str) or not audio_path:
                    continue
                total += 1
                if not Path(audio_path).exists():
                    missing += 1
    except Exception:
        return 0, 0
    return total, missing

def reset_stage2_outputs() -> None:
    paths = [
        Path(STAGE2_MANIFEST),
        Path(CHUNKS_DIR) / "stage2_cut_state.json",
        Path(CHUNKS_DIR) / "stage2_stereo_cache.json",
        Path(CHUNKS_DIR) / "tasks_remaining.jsonl",
    ]
    for p in paths:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

def check_required_paths():
    missing = []
    def req(path, label):
        if not Path(path).exists():
            missing.append(f"{label}: {path}")

    if not SKIP_STAGE_1_MANIFEST_CREATION:
        req(TRANSCRIPT_DIR_STAGE1, "TRANSCRIPT_DIR_STAGE1")
        req(AUDIO_SOURCE_DIR_STAGE1, "AUDIO_SOURCE_DIR_STAGE1")

    if missing:
        raise RuntimeError("Missing required paths:\n" + "\n".join(missing))

# %%
# ---------- WORKER AUTOTUNE ----------
CPU_COUNT = os.cpu_count() or 2
MAX_WORKERS = max(1, CPU_COUNT)

# Tune workers (optionally oversubscribe ffmpeg)
FFMPEG_WORKERS = max(1, min(int(MAX_WORKERS * float(FFMPEG_WORKERS_MULT)), 64))
TASK_WORKERS = max(1, min(int(MAX_WORKERS * float(TASK_WORKERS_MULT)), 64))
AUG_WORKERS = MAX_WORKERS
# Heuristic: keep total threads <= CPU count (cap per-worker threads to 4)
if FFMPEG_WORKERS <= 0:
    FFMPEG_THREADS = 1
else:
    FFMPEG_THREADS = max(1, min(4, (CPU_COUNT // FFMPEG_WORKERS) or 1))

if AUTO_WORKERS:
    SPEAKER_WORKERS = MAX_WORKERS

# %%
# ---------- CLONE / UPDATE ----------
if not Path(REPO_DIR).exists():
    run(["git", "clone", REPO_URL, REPO_DIR])
elif UPDATE_REPO:
    run(["git", "-C", REPO_DIR, "fetch", "--all", "--prune"])
    if RESET_REPO_HARD:
        run(["git", "-C", REPO_DIR, "reset", "--hard", "origin/HEAD"])
os.chdir(REPO_DIR)

# %%
# ---------- SYSTEM DEPS ----------
import zipfile

def _add_to_path(p: Path) -> None:
    if not p.exists():
        return
    os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")

def _ensure_tool(name: str, extra_paths: list[Path] | None = None) -> bool:
    if shutil.which(name):
        return True
    if extra_paths:
        for p in extra_paths:
            _add_to_path(p)
            if shutil.which(name):
                return True
    return False

def _ensure_sox_from_zip() -> None:
    if shutil.which("sox"):
        return
    sox_dir = Path(REPO_DIR) / "sox_bin"
    sox_exe = sox_dir / "sox.exe"
    sox_zip = sox_dir / "sox.zip"
    if sox_exe.exists():
        _add_to_path(sox_dir)
        return
    if sox_zip.exists():
        try:
            with zipfile.ZipFile(sox_zip, "r") as zf:
                zf.extractall(sox_dir)
            _add_to_path(sox_dir)
        except Exception as exc:
            log(f"[deps] failed to extract sox.zip: {exc}")

_ensure_sox_from_zip()
missing_tools = []
if not _ensure_tool("ffmpeg"):
    missing_tools.append("ffmpeg")
if not _ensure_tool("sox", extra_paths=[Path(REPO_DIR) / "sox_bin"]):
    missing_tools.append("sox")
if USE_LOCAL_DATA or SYNC_CACHE_TO_DRIVE or SYNC_RECORD_CHUNKS_TO_DRIVE:
    if not _ensure_tool("rsync"):
        missing_tools.append("rsync")
if USE_RCLONE and not _ensure_tool("rclone"):
    missing_tools.append("rclone")
if missing_tools:
    log(f"[deps] missing system tools: {', '.join(sorted(set(missing_tools)))}")

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

    if RESTORE_RECORD_CHUNKS:
        restore_record_chunks_from_drive()

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
PY = PYTHON_EXE if PYTHON_EXE else sys.executable
if PYTHON_EXE and not Path(PYTHON_EXE).exists():
    log(f"[warn] PYTHON_EXE not found: {PYTHON_EXE}; falling back to {sys.executable}")
    PY = sys.executable
run([PY, "-m", "pip", "install", "-q", "-U", "pip"])

def get_installed_version(pkg: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(pkg)
    except Exception:
        return None

def install_if_needed(pkgs: list[str]) -> None:
    missing = []
    for p in pkgs:
        name = p.split("==", 1)[0]
        have = get_installed_version(name)
        want = p.split("==", 1)[1] if "==" in p else None
        if have is None:
            missing.append(p)
        elif want and have != want:
            missing.append(p)
    if missing:
        run([PY, "-m", "pip", "install", "-q"] + missing)
    else:
        log("[deps] all required packages already installed")

def log_installed_versions(pkgs: list[str]) -> None:
    for p in pkgs:
        name = p.split("==", 1)[0]
        ver = get_installed_version(name)
        log(f"[deps] {name}=={ver if ver else 'not-installed'}")

deps = [
    "transformers",
    "datasets",
    "accelerate",
    "soundfile",
    "librosa",
    "numpy",
    "pandas",
    "scipy",
    "tqdm",
    "orjson",
    "matplotlib",
    "jiwer",
    "edlib",
    "pyannote.audio",
    "huggingface_hub",
]

install_if_needed(deps)
log_installed_versions(deps)

# Fix torchvision mismatch if needed (pyannote/lightning uses torchmetrics->torchvision)
ensure_torchvision_compat()

# %%
# ---------- PATCH WINDOWS-SPECIFIC PATHS ----------
# %%
# ---------- CONFIG SANITY CHECKS ----------
check_required_paths()





# %%
# ---------- STAGE 1 ----------
if should_run("stage_1", [TASKS_PENDING, PAIRS_PENDING]):
    stage_banner("STAGE 1")
    update_stage_state("stage_1", "running")
    run_stage([
        PY, "stage_1_Manifest_creation_local_only.py",
        "--transcript-dir", TRANSCRIPT_DIR_STAGE1,
        "--chunks-dir", CHUNKS_DIR,
        "--audio-source-dir", AUDIO_SOURCE_DIR_STAGE1,
        "--acft-model-id", BASE_MODEL_ID,
        "--base-processor-id", PROCESSOR_ID,
    ])
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
force_stage2 = stage2_needs_resume()
if should_run("stage_2", [STAGE2_MANIFEST], force=force_stage2):
    stage_banner("STAGE 2")
    update_stage_state("stage_2", "running")
    if force_stage2:
        log("[resume] stage_2 has remaining tasks; resuming chunking")
    # Copy only audio referenced by tasks_pending.jsonl (and transcripts from pairs_pending.jsonl)
    if USE_LOCAL_DATA:
        audio_paths = collect_jsonl_paths(TASKS_PENDING, "audio_path")
        rel_audio = rel_paths_to_drive(audio_paths)
        rsync_files_from(rel_audio, DATA_ROOT_DRIVE, DATA_ROOT, label="audio_for_stage2")

        transcript_paths = collect_jsonl_paths(PAIRS_PENDING, "transcript_json")
        rel_transcripts = rel_paths_to_drive(transcript_paths)
        rsync_files_from(rel_transcripts, DATA_ROOT_DRIVE, DATA_ROOT, label="transcripts_for_stage2")

    run_stage([PY, "stage_2_chunk_transcripts_sentence_parallel.py",
         "--tasks_pending_path", TASKS_PENDING,
         "--pairs_pending_path", PAIRS_PENDING,
         "--out_pairs_path", STAGE2_MANIFEST,
         "--ffmpeg_workers", str(FFMPEG_WORKERS),
         "--task_workers", str(TASK_WORKERS),
         "--ffmpeg_threads", str(FFMPEG_THREADS),
         "--stereo_policy", "split_drop_dupes"])
    assert_outputs([STAGE2_MANIFEST], "stage_2")
    sync_files_to_drive([STAGE2_MANIFEST], label="stage2")
    sync_record_chunks_to_drive()
    update_stage_state("stage_2", "done")
# %%
# ---------- STAGE 3 ----------
if should_run("stage_3", [SCORES_CSV]):
    stage_banner("STAGE 3")
    ensure_local_paths([TARGET_REF_DIR] + ([OTHER_REF_DIR] if OTHER_REF_DIR else []), label="stage_3")
    update_stage_state("stage_3", "running")
    if DEVICE == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                log("[warn] CUDA not available; switching Stage 3 to CPU")
                DEVICE = "cpu"
                SPEAKER_WORKERS = max(1, min(SPEAKER_WORKERS, 2))
        except Exception:
            DEVICE = "cpu"
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
    try:
        run_stage(cmd)
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").lower()
        if "out of memory" in err or "cuda out of memory" in err:
            log("[oom] stage_3 hit OOM; retrying with lower batch/workers and CPU")
            SPEAKER_BATCH = max(1, SPEAKER_BATCH // 2)
            SPEAKER_WORKERS = max(1, min(SPEAKER_WORKERS, 2))
            DEVICE = "cpu"
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
            run_stage(cmd)
        else:
            raise
    assert_outputs([SCORES_CSV], "stage_3")
    update_stage_state("stage_3", "done")
# %%
# ---------- STAGE 3b ----------
if should_run("stage_3b", [STAGE3B_MANIFEST]):
    stage_banner("STAGE 3B")
    update_stage_state("stage_3b", "running")
    run_stage([PY, "stage_3b_filter_english_only.py",
         "--input", STAGE2_MANIFEST,
         "--output", STAGE3B_MANIFEST,
         "--min-english-ratio", "0.7"])
    assert_outputs([STAGE3B_MANIFEST], "stage_3b")
    update_stage_state("stage_3b", "done")
# %%
# ---------- STAGE 4 ----------
if should_run("stage_4", [STAGE4_MANIFEST]):
    stage_banner("STAGE 4")
    update_stage_state("stage_4", "running")
    run_stage([PY, "stage_4_Delete_common_fillers_words_from_manifest.py",
         "--input", STAGE3B_MANIFEST,
         "--output", STAGE4_MANIFEST,
         "--state-file", COMMON_SEGMENTS_STATE,
         "--min-frequency", "3"])
    assert_outputs([STAGE4_MANIFEST], "stage_4")
    update_stage_state("stage_4", "done")
# %%
# ---------- STAGE 5 ----------
if should_run("stage_5", [STAGE5_MANIFEST]):
    stage_banner("STAGE 5")
    update_stage_state("stage_5", "running")
    run_stage([PY, "stage_5_convert_number_words_to_digits.py",
         "--input", STAGE4_MANIFEST,
         "--output", STAGE5_MANIFEST,
         "--field", "raw_transcription"])
    assert_outputs([STAGE5_MANIFEST], "stage_5")
    update_stage_state("stage_5", "done")
# %%
# ---------- STAGE 6 ----------
if should_run("stage_6", [STAGE6_MANIFEST]):
    stage_banner("STAGE 6")
    ensure_local_paths([NOISE_DIR], label="stage_6")
    update_stage_state("stage_6", "running")
    Path(SEEN_DIR).mkdir(parents=True, exist_ok=True)
    seed_manifest(STAGE5_MANIFEST, STAGE6_MANIFEST)
    run_stage([PY, "stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise.py",
         "--in_manifest", STAGE5_MANIFEST,
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
    stage_banner("STAGE 7")
    ensure_local_paths([OTHER_VOICES_DIR], label="stage_7")
    update_stage_state("stage_7", "running")
    seed_manifest(STAGE6_MANIFEST, STAGE7_MANIFEST)
    run_stage([PY, "stage_7_add_others_voices_to_my_audio_fast_idempotent.py",
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
    stage_banner("STAGE 8")
    update_stage_state("stage_8", "running")
    seed_manifest(STAGE7_MANIFEST, STAGE8_MANIFEST)
    run_stage([PY, "stage_8_add_random_gain_to_high_score_voices_parallel_idempotent.py",
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
    stage_banner("STAGE 9")
    ensure_local_paths([RIR_DIR], label="stage_9")
    update_stage_state("stage_9", "running")
    seed_manifest(STAGE8_MANIFEST, STAGE9_MANIFEST)
    run_stage([PY, "stage_9_add_reverb_idempotent.py",
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
    stage_banner("STAGE 10B")
    update_stage_state("stage_10b", "running")
    seed_manifest(STAGE9_MANIFEST, STAGE10B_MANIFEST)
    run_stage([PY, "stage_10_b_add_speech_tempo_pause_aware_idempotent.py",
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
    sync_record_chunks_to_drive()
    update_stage_state("stage_10b", "done")
# %%
# ---------- STAGE 11 ----------
if should_run("stage_11", [STAGE11_MANIFEST]):
    stage_banner("STAGE 11")
    update_stage_state("stage_11", "running")
    seed_manifest(STAGE10B_MANIFEST, STAGE11_MANIFEST)
    run_stage([PY, "stage_11_add_frequency_manipulation_idempotent.py",
         "--in_manifest", STAGE10B_MANIFEST,
         "--out_manifest", STAGE11_MANIFEST,
         "--out_dir", FREQ_OUT_DIR,
         "--ratio", str(FREQ_RATIO),
         "--copies", str(FREQ_COPIES),
         "--workers", str(AUG_WORKERS),
         "--stage_name", "frequency_shift",
         "--seen_db", f"{SEEN_DIR}/stage11_frequency_shift.sqlite",
         "--semitones_min", "-1.0",
         "--semitones_max", "1.0",
         "--mode", "choice",
         "--semitones_choices=-1.5,-1.0,-0.5,0.5,1.0,1.5"])
    assert_outputs([STAGE11_MANIFEST], "stage_11")
    sync_record_chunks_to_drive()
    update_stage_state("stage_11", "done")
# %%
# ---------- STAGE 12 ----------
if should_run("stage_12", [STAGE12_MANIFEST]):
    stage_banner("STAGE 12")
    update_stage_state("stage_12", "running")
    run_stage([PY, "stage_12_remove_bottom_percent_by_speaker_scores.py",
         "--input_manifest", STAGE11_MANIFEST,
         "--output_manifest", STAGE12_MANIFEST,
         "--speaker_scores_csv", SCORES_CSV,
         "--bottom_percent", str(BOTTOM_PERCENT)])
    assert_outputs([STAGE12_MANIFEST], "stage_12")
    update_stage_state("stage_12", "done")
# %%
# ---------- STAGE 13 ----------
if should_run("stage_13", [STAGE13_TRAIN, STAGE13_TEST]):
    stage_banner("STAGE 13")
    update_stage_state("stage_13", "running")
    run_stage([PY, "stage_13_group_split_train_test.py",
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
    stage_banner("STAGE 14")
    update_stage_state("stage_14", "running")
    run_stage([PY, "stage_14_remove_target_files_from_manifest.py",
         "--input_manifest", STAGE13_TRAIN,
         "--output_manifest", STAGE14_TRAIN,
         "--speaker_scores_csv", SCORES_CSV])
    assert_outputs([STAGE14_TRAIN], "stage_14")
    update_stage_state("stage_14", "done")
# %%
# ---------- STAGE 15 ----------
if should_run("stage_15", [STAGE15_TRAIN]):
    stage_banner("STAGE 15")
    update_stage_state("stage_15", "running")
    if USE_ADVANCED_RANDOMIZE:
        run_stage([PY, "stage_15_b_advanced_randomize_manifest.py",
             "--input_manifest", STAGE14_TRAIN,
             "--output_manifest", STAGE15_TRAIN,
             "--seed", str(RANDOM_SEED)])
    else:
        run_stage([PY, "stage_15_a_randomize_manifest.py",
             "--input_manifest", STAGE14_TRAIN,
             "--output_manifest", STAGE15_TRAIN,
             "--seed", str(RANDOM_SEED)])
    assert_outputs([STAGE15_TRAIN], "stage_15")
    sync_record_chunks_to_drive()
    update_stage_state("stage_15", "done")
# %%
# ---------- STAGE 16 ----------
if should_run("stage_16", [STAGE13_TEST, STAGE16_BACKUP]):
    stage_banner("STAGE 16")
    update_stage_state("stage_16", "running")
    run_stage([
        PY, "Stage_16_move_test_chunks_update_test_manifest.py",
        "--manifest_path", STAGE13_TEST,
        "--target_dir", TEST_CHUNKS_DIR,
        "--mode", "move",
        "--backup_suffix", ".backup",
    ])
    assert_outputs([STAGE13_TEST, STAGE16_BACKUP], "stage_16")
    sync_record_chunks_to_drive()
    update_stage_state("stage_16", "done")
# %%
# ---------- STAGE 17 ----------
if should_run("stage_17", [CHECKPOINT_DIR]):
    stage_banner("STAGE 17")
    update_stage_state("stage_17", "running")
    os.environ["WHISPER_START_FRESH"] = str(START_FRESH)
    run_stage([
        PY, STAGE17_SCRIPT,
        "--manifest-path", STAGE15_TRAIN,
        "--checkpoint-dir", CHECKPOINT_DIR,
        "--futo-model-id", BASE_MODEL_ID,
        "--processor-id", PROCESSOR_ID,
        "--start-fresh", str(START_FRESH),
    ])
    assert_outputs([CHECKPOINT_DIR], "stage_17")
    update_stage_state("stage_17", "done")

# %%
# ---------- STAGE 18: MERGE PEFT (if enabled) ----------
MERGED_MODEL_READY = False
BEST_CHECKPOINT_DIR = best_checkpoint_dir()
if MERGE_PEFT_AFTER_STAGE17 and PEFT_ENABLED:
    stage_banner("STAGE 18: MERGE PEFT")
    if not BEST_CHECKPOINT_DIR:
        raise RuntimeError("No checkpoint found to merge.")
    run_stage([
        PY, "stage17_merge_peft_checkpoint_to_full_model.py",
        "--peft_dir", BEST_CHECKPOINT_DIR,
        "--out_dir", MERGED_MODEL_DIR,
        "--base_model_id", BASE_MODEL_ID,
    ])
    assert_outputs([MERGED_MODEL_DIR], "stage_18_merge_peft")
    MERGED_MODEL_READY = True

# %%
# ---------- STAGE 19C: EVALUATION ----------
if RUN_EVAL_19C:
    stage_banner("STAGE 19C: EVALUATION")
    if not Path(EVAL_19C_SCRIPT).exists():
        raise RuntimeError(f"Eval script not found: {EVAL_19C_SCRIPT}")
    if not EVAL_19C_ARGS:
        log("[warn] EVAL_19C_ARGS is empty; set args if your script requires them.")
    eval_args = [str(x) for x in EVAL_19C_ARGS]
    if EVAL_ADD_BATCH_ARGS:
        eval_args += ["--batch_size", str(EVAL_BATCH_SIZE)]
        if EVAL_MAX_SAMPLES:
            eval_args += ["--max_samples", str(EVAL_MAX_SAMPLES)]
    run_stage([PY, EVAL_19C_SCRIPT] + eval_args)
    if EVAL_19C_OUT_JSON:
        assert_outputs([EVAL_19C_OUT_JSON], "stage_19c_eval")

# %%
# ---------- STAGE 19D: EVALUATION ----------
if RUN_EVAL_19D:
    stage_banner("STAGE 19D: EVALUATION")
    if not Path(EVAL_19D_SCRIPT).exists():
        raise RuntimeError(f"Eval script not found: {EVAL_19D_SCRIPT}")
    if not EVAL_19D_ARGS:
        log("[warn] EVAL_19D_ARGS is empty; set args if your script requires them.")
    eval_args = [str(x) for x in EVAL_19D_ARGS]
    if EVAL_ADD_BATCH_ARGS:
        eval_args += ["--batch_size", str(EVAL_BATCH_SIZE)]
        if EVAL_MAX_SAMPLES:
            eval_args += ["--max_samples", str(EVAL_MAX_SAMPLES)]
    run_stage([PY, EVAL_19D_SCRIPT] + eval_args)
    if EVAL_19D_OUT_JSON:
        assert_outputs([EVAL_19D_OUT_JSON], "stage_19d_eval")

# %%
# ---------- STAGE 19E: CHARTS ----------
if RUN_EVAL_19C or RUN_EVAL_19D:
    stage_banner("STAGE 19E: CHARTS")
    charts = []
    if EVAL_19C_OUT_JSON and Path(EVAL_19C_OUT_JSON).exists():
        out_png = str(Path(CHARTS_DIR) / "eval_19c.png")
        save_chart_from_metrics(EVAL_19C_OUT_JSON, out_png, "Evaluation 19C")
        charts.append(out_png)
    if EVAL_19D_OUT_JSON and Path(EVAL_19D_OUT_JSON).exists():
        out_png = str(Path(CHARTS_DIR) / "eval_19d.png")
        save_chart_from_metrics(EVAL_19D_OUT_JSON, out_png, "Evaluation 19D")
        charts.append(out_png)
    if charts:
        sync_charts_to_drive(charts)

# %%
# ---------- STAGE 20: GGUF/GGML CONVERSION ----------
if RUN_GGUF_CONVERSION:
    stage_banner("STAGE 20: GGUF/GGML CONVERSION")
    src_ckpt = MERGED_MODEL_DIR if MERGED_MODEL_READY else (BEST_CHECKPOINT_DIR or "")
    if not src_ckpt:
        raise RuntimeError("No checkpoint found for conversion.")
    run_stage([
        PY, "convert_to_gguf.py",
        "--checkpoint_dir", src_ckpt,
        "--export_dir", GGUF_EXPORT_DIR,
        "--out_dir", GGUF_OUT_DIR,
        "--whisper_cpp", WHISPER_CPP_ROOT,
        "--whisper_repo", WHISPER_OPENAI_REPO,
        "--base_model", BASE_MODEL_ID,
        "--processor", PROCESSOR_ID,
        "--qtype", GGUF_QTYPE,
    ])

# %%
# ---------- SYNC CACHE ----------
sync_cache_to_drive()
sync_record_chunks_to_drive()

print("✅ Pipeline complete.")
