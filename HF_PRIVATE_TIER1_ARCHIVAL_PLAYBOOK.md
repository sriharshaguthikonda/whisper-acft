# HF Private Tier-1 Model Archival Playbook

Last updated: 2026-03-25

## Goal
Archive active model artifacts from `I:\RUN__*` folders to private Hugging Face repos, then remove large local model binaries while keeping reproducible restore paths.

This project uses a hybrid access strategy:

- Central cache: `I:\hf_model_cache`
- Per-run local pointers: `MODEL_POINTER.json`, `MODEL_POINTER_README.md`, `HF_MODEL_REPO.url`
- Execution history: `HF_PRIVATE_TIER1_EXECUTION_LOG.md`

## Naming and Scope

- Repo prefix: `Whisper-acft` (as requested)
- Repo layout: one private HF model repo per run folder
- Upload scope: model artifact files with extensions:
  - `.safetensors`, `.bin`, `.pt`, `.pth`, `.ckpt`, `.gguf`, `.ggml`, `.onnx`
- Default size filter: upload and cleanup files `>= 50 MB`

## Prerequisites

1. `HF_TOKEN` is set in `.env` (or environment).
2. Use training venv Python with user site packages disabled:
   - `set PYTHONNOUSERSITE=1`
3. Ensure `huggingface_hub` is available in that interpreter.

## Tools Added

- `tools/hf_tier1_archive_private_models.py`
  - Discovers run folders
  - Creates/uses private HF repos
  - Uploads selected model files
  - Verifies remote files before deletion
  - Deletes local archived files when cleanup is enabled
  - Writes per-run pointer/link files
  - Writes reports to `hf_tier1_reports/`
  - Updates registry `HF_PRIVATE_TIER1_REGISTRY.json`

- `tools/hf_tier1_restore_from_pointer.py`
  - Reads `MODEL_POINTER.json`
  - Restores from pinned revision using central cache and optional local restore dir

- `tools/hf_tier1_archive_repo_by_repo.ps1`
  - Operational uploader/cleanup runner that processes runs one-by-one
  - Uses repo mapping from a dry-run CSV
  - Uploads each file with `hf upload`
  - Verifies remote file list before local deletion
  - Writes run pointers, session reports, and updates registry

## Standard Commands

### 1) Dry-run inventory (no upload/delete)

```powershell
Set-Location I:\whisper-acft
$env:PYTHONNOUSERSITE = "1"
I:\Whisper-training-env\Scripts\python.exe I:\whisper-acft\tools\hf_tier1_archive_private_models.py --dry-run --repo-prefix Whisper-acft --root I:\ --repo-root I:\whisper-acft --cache-dir I:\hf_model_cache
```

### 2) Upload + verify + cleanup local model binaries

```powershell
Set-Location I:\whisper-acft
& I:\whisper-acft\tools\hf_tier1_archive_repo_by_repo.ps1 -RepoRoot I:\whisper-acft -RunsRoot I:\ -CacheDir I:\hf_model_cache -CleanupLocal
```

### 3) Restore one run from pointer (when needed)

```powershell
Set-Location I:\whisper-acft
$env:PYTHONNOUSERSITE = "1"
I:\Whisper-training-env\Scripts\python.exe I:\whisper-acft\tools\hf_tier1_restore_from_pointer.py --pointer "I:\RUN__<your-run>\MODEL_POINTER.json" --cache-dir I:\hf_model_cache --local-dir "I:\RUN__<your-run>\restored_model"
```

## Outputs and Tracking

- Per-run folder files:
  - `MODEL_POINTER.json`
  - `MODEL_POINTER_README.md`
  - `HF_MODEL_REPO.url`

- Project-level tracking:
  - `HF_PRIVATE_TIER1_REGISTRY.json`
  - `hf_tier1_reports/hf_tier1_archive_<timestamp>.json`
  - `hf_tier1_reports/hf_tier1_archive_<timestamp>.csv`

## Current Large-File Inventory Snapshot

Scan snapshot before archival run:

- RUN folders found: `34`
- RUN folders with model files >= 50MB: `15`
- Large model files (>= 50MB): `94`
- Total size: `13.737 GB`

Top run folders by large model file size:

- `RUN__k-train-eval__s-17__b-futo-small-en__m-s17-qat-dora__a-dora-r8-a16__q-qat__c-dyn__r-148296__id-22` (~2.23 GB)
- `RUN__k-train-eval__s-17__b-futo-small-en__m-s17-qat-dora__a-dora-r64-a32__q-qat__c-dyn__r-161297__id-26` (~1.68 GB)
- `RUN__k-train-eval__s-18__b-futo-tiny-en__m-s18-full__a-full__q-noqat__c-dyn__r-112987__id-8` (~1.55 GB)
- `RUN__k-train-eval__s-17__b-openai-tiny-en__m-s17-full__a-full__q-noqat__c-dyn__r-46760__id-14` (~1.48 GB)
- `RUN__k-train-eval__s-17__b-openai-tiny-en__m-s17-qat-full__a-full__q-qat__c-dyn__r-38560__id-15` (~1.12 GB)

## Safety Rules

- Never delete local files unless upload verification passes.
- Keep logs, eval JSONs, charts, and metadata local.
- Only model artifact binaries in selected extension set are cleaned up.
- Pointer files are mandatory for each archived run.

## Troubleshooting

- If SSL/import issues appear in venv Python, ensure:
  - `PYTHONNOUSERSITE=1`
- If auth fails:
  - validate `HF_TOKEN` in `.env`
  - test with `hf auth whoami`
- If one run fails upload:
  - rerun with `--include-run "<run-folder-name>"`
  - script is idempotent with `exist_ok=True`
