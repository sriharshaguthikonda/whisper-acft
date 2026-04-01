# %% [markdown]
# # Colab: Cloud Eval Stack (targetmix + clean)
# Cells are split by `# %%`. Run top-to-bottom.

# %%
import os
import subprocess
from pathlib import Path

try:
    from google.colab import userdata  # type: ignore
except Exception:
    userdata = None


def run(cmd: list[str], cwd: str | None = None) -> None:
    print("$ " + " ".join(cmd))
    p = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


# %%
# ---------- CONFIG ----------
REPO_URL = "https://github.com/sriharshaguthikonda/whisper-acft.git"
REPO_DIR = "/content/whisper-acft"
CONFIG_PATH = "/content/whisper-acft/cloud_eval_config.json"
MODELS_FILE = "/content/whisper-acft/cloud_eval_models.csv"
RUN_ROOT = "/content/cloud_eval_runs/run_01"

# Option A: set HF_TOKEN in Colab secrets (recommended)
if userdata is not None:
    try:
        token = userdata.get("HF_TOKEN")
        if token:
            os.environ["HF_TOKEN"] = token
    except Exception:
        pass

# Option B: uncomment and paste token once (avoid storing in notebook)
# os.environ["HF_TOKEN"] = "hf_xxx"


# %%
# ---------- SETUP ----------
if not Path(REPO_DIR).exists():
    run(["git", "clone", REPO_URL, REPO_DIR])

run(
    [
        "pip",
        "-q",
        "install",
        "huggingface_hub",
        "transformers",
        "soundfile",
        "jiwer",
        "numpy",
        "pandas",
        "tqdm",
        "torch",
        "nemo_toolkit[asr]>=2.4.0",
    ]
)

run(["apt-get", "-qq", "update"])
run(["apt-get", "-qq", "install", "-y", "ffmpeg"])


# %%
# ---------- CREATE LOCAL COPIES OF CONFIG/MODEL LIST ----------
run(["cp", "/content/whisper-acft/cloud_eval_config.example.json", CONFIG_PATH], cwd=REPO_DIR)
run(["cp", "/content/whisper-acft/cloud_eval_models.example.csv", MODELS_FILE], cwd=REPO_DIR)
print("Edit these files before run if needed:")
print(CONFIG_PATH)
print(MODELS_FILE)


# %%
# ---------- RUN ORCHESTRATOR ----------
run(
    [
        "python",
        "/content/whisper-acft/cloud_eval_orchestrator.py",
        "--config",
        CONFIG_PATH,
        "--models_file",
        MODELS_FILE,
        "--repo_root",
        "/content/whisper-acft",
        "--run_root",
        RUN_ROOT,
    ],
    cwd=REPO_DIR,
)
