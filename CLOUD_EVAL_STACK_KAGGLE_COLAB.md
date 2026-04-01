# Cloud Eval Stack (Kaggle + Colab)

This stack runs two protocols with one orchestrator:

- **Targetmix**: Stage 19E logic + NeMo adapter
- **Clean**: no-mix baseline

Outputs include a consolidated leaderboard using:

- `0.8 * normalized_accuracy + 0.2 * normalized_speed`

## Core files

- `I:\whisper-acft\cloud_eval_orchestrator.py`
- `I:\whisper-acft\stage_19e_nemo_parakeet_adapter.py`
- `I:\whisper-acft\stage_19e_clean_eval_unified.py`
- `I:\whisper-acft\cloud_eval_config.example.json`
- `I:\whisper-acft\cloud_eval_models.example.csv`
- `I:\whisper-acft\eval_cloud_colab.py`
- `I:\whisper-acft\eval_cloud_kaggle.py`

## Model-list contract (CSV)

Required columns:

- `model_name`
- `backend` (`hf_transformers` or `nemo_parakeet`)
- `model_ref`
- `language_mode`
- `enabled`

Optional:

- `decoder_preset`
- `batch_hint`
- `notes`
- `expected_sr`

Use `I:\whisper-acft\cloud_eval_models.example.csv` as template.

`I:\whisper-acft\cloud_eval_config.example.json` is already wired to `eval_packs/LATEST_PACK/...`.
After running the eval-pack uploader, the orchestrator resolves `LATEST_PACK` via `eval_packs/LATEST_PACK.json` automatically.
The manifest `audio_path` values are the source of truth for audio lookup.
Nested folder depth under `audio/` (for example sharded layouts like `audio/Record_test_chunks/shard_0000/...`) is supported.

## Local Windows run

1) Copy templates and edit:

```powershell
Copy-Item "I:\whisper-acft\cloud_eval_config.example.json" "I:\whisper-acft\cloud_eval_config.json"
Copy-Item "I:\whisper-acft\cloud_eval_models.example.csv" "I:\whisper-acft\cloud_eval_models.csv"
```

2) Run orchestrator:

```powershell
I:\Whisper-training-env\Scripts\python.exe I:\whisper-acft\cloud_eval_orchestrator.py `
  --config "I:\whisper-acft\cloud_eval_config.json" `
  --models_file "I:\whisper-acft\cloud_eval_models.csv" `
  --repo_root "I:\whisper-acft" `
  --run_root "I:\whisper-acft\cloud_eval_runs\run_01"
```

3) Outputs:

- `I:\whisper-acft\cloud_eval_runs\run_01\targetmix\evaluation_results_futo_like_targetmix_sweep.json`
- `I:\whisper-acft\cloud_eval_runs\run_01\clean\evaluation_results_clean.json`
- `I:\whisper-acft\cloud_eval_runs\run_01\leaderboard_consolidated.csv`
- `I:\whisper-acft\cloud_eval_runs\run_01\leaderboard_consolidated.json`

## Colab / Kaggle

- Colab notebook-script: `I:\whisper-acft\eval_cloud_colab.py`
- Kaggle notebook-script: `I:\whisper-acft\eval_cloud_kaggle.py`

Both scripts:

- install dependencies,
- copy config/model templates,
- run the same orchestrator command.

## Resume behavior

- Re-running with same `--run_root` resumes from existing result JSON/per-sample files.
- `run_state.json` stores per-model targetmix timing for speed normalization fallback.
