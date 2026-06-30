# Resumable Kaggle Notebooks

Last updated: 2026-06-30

## Files

- `notebooks/kaggle/01_generate_chunks_publish.ipynb`
- `notebooks/kaggle/02_train_smoke_resume.ipynb`
- `tools/kaggle_acft_helpers.py`

The notebooks are orchestration wrappers. Stage logic stays in the existing repo scripts.

## Notebook 01: Generate Chunks

Default profile:

- `PROFILE=smoke`
- `RUN_TAG=<UTC timestamp>`
- `PUBLISH_AFTER_EACH_STAGE=1`
- `DRY_RUN_PUBLISH=0`

Behavior:

1. Reads the canonical Kaggle source dataset list from `docs/KAGGLE_PRIMARY_TRAINING_DATA_EXPORT.md`.
2. Reconstructs old top-level paths under `/kaggle/working/acft_data`:
   - `Record_harsha`
   - `Transcriptions_corrected`
   - `Record_only_by_harsha`
   - `Record_others_compacted`
   - `noise/RIRS_NOISES`
   - `whisper-acft`
3. Runs existing stages:
   - Stage 1: `stage_1_Manifest_creation_local_only.py`
   - Stage 2: `stage_2_chunk_transcripts_sentence_parallel.py`
   - Stage 13: `stage_13_group_split_train_test.py`
   - Stage 15: `stage_15_b_advanced_randomize_manifest.py`
4. Writes resume state under `/kaggle/working/acft_runs/<RUN_TAG>/state`.
5. Writes published resume state under `/kaggle/working/acft_data/Record_chunks/_kaggle_state/<RUN_TAG>`.
6. Versions `/kaggle/working/acft_data/Record_chunks` as a private Kaggle Dataset.

Chunk dataset handle pattern:

```text
drsriharshaguthik/acft-kaggle-chunks-<run-tag>-<profile>
```

## Notebook 02: Train Smoke Resume

Default profile:

- `PROFILE=smoke`
- `LR_START=1e-6`
- `MAX_EPOCHS=1`
- `N_SAMPLES_PER_EPOCH=32`
- `START_FRESH=0`
- `PUBLIC_RATIO=0.30`

Behavior:

1. Finds the attached chunks dataset under `/kaggle/input`.
2. Uses Stage 15 randomized train manifest if present, then Stage 13 train, then raw Stage 2 manifest.
3. Builds a capped public-ASR mix from clean English LibriSpeech/FLEURS-style sources when Hugging Face streaming is available.
4. Keeps Common Voice disabled unless explicitly set with `ENABLE_COMMON_VOICE=1`.
5. Tags public rows with `dataset_scope=public_asr` and `exclude_from_private_eval=true`.
6. Writes private eval manifests without public rows.
7. Runs existing Stage 17:

```text
stage_17_WER_acft_Whisper_Futo_finetuned_model_training_only_local_en_version_only_qat_dora.py
```

Training dataset handle pattern:

```text
drsriharshaguthik/acft-kaggle-train-<run-tag>-<profile>
```

## Resume Contract

Each stage has:

- declared input paths,
- declared output paths,
- JSON config,
- SHA-256 signature,
- state file in `<run-root>/state/<stage>.resume.json`.

A stage is skipped only when:

- all declared outputs exist, and
- the stored signature matches the current input/config signature.

If Kaggle stops mid-run, rerun the notebook with the same `RUN_TAG` and attach the last published dataset. Stage 2 and Stage 17 also keep their own internal resume files/checkpoints.

Notebook 01 restores an attached prior chunks dataset into `/kaggle/working/acft_data/Record_chunks` before running stages. Notebook 02 restores an attached prior train dataset into `/kaggle/working/acft_train_runs/<RUN_TAG>` before Stage 17.

## Publishing

`tools/kaggle_acft_helpers.py` writes `dataset-metadata.json` and `_publish_plan.json`.

Publish backend order:

1. KaggleHub `dataset_upload` when available.
2. Kaggle CLI `kaggle datasets version`.
3. Kaggle CLI `kaggle datasets create --private`.

Set `DRY_RUN_PUBLISH=1` to validate metadata without uploading.

## Trial Run

1. In Kaggle, create a new notebook and upload/copy `notebooks/kaggle/01_generate_chunks_publish.ipynb`.
2. Attach all canonical private source datasets listed in `docs/KAGGLE_PRIMARY_TRAINING_DATA_EXPORT.md`.
3. Attach repo files, or set `ACFT_GIT_URL` to a cloneable repo URL.
4. Keep defaults for first trial: `PROFILE=smoke`, `PUBLISH_AFTER_EACH_STAGE=1`, `DRY_RUN_PUBLISH=0`.
5. Run all cells. If Kaggle API auth fails, add Kaggle API credentials through notebook secrets or `/root/.kaggle/kaggle.json`, then rerun the publish cell.
6. Confirm a private chunks dataset appears under the chunks handle pattern.
7. For a resume trial, restart with the same `RUN_TAG` and attach the chunks dataset from step 6; notebook 01 will restore `_kaggle_state` and skip matching completed stages.
8. Create a Kaggle notebook from `02_train_smoke_resume.ipynb`.
9. Attach the chunks dataset from step 6.
10. Use a GPU runtime for Stage 17.
11. Run with defaults: `LR_START=1e-6`, `MAX_EPOCHS=1`, `N_SAMPLES_PER_EPOCH=32`, `START_FRESH=0`.
12. Confirm a private train dataset appears under the train handle pattern.
13. For a training resume trial, restart with the same `RUN_TAG`, attach both the chunks dataset and the prior train dataset; notebook 02 will restore checkpoints/state before Stage 17.

## Local Validation

Focused local checks:

```powershell
python -m pytest test_kaggle_acft_helpers.py -q
python -m json.tool notebooks\kaggle\01_generate_chunks_publish.ipynb > $null
python -m json.tool notebooks\kaggle\02_train_smoke_resume.ipynb > $null
python -m compileall -q tools\kaggle_acft_helpers.py
```

Local Stage 1/2 fixture helper:

```python
from tools import kaggle_acft_helpers as kh
fixture = kh.write_stage12_smoke_fixture(".kaggle_smoke/source")
```

This creates a synthetic `Record_harsha`, `Transcriptions_corrected`, and `Record_chunks` tree without private audio/transcripts.

## References

- Kaggle Notebooks: https://www.kaggle.com/docs/notebooks
- Kaggle API: https://github.com/Kaggle/kaggle-api
- KaggleHub: https://github.com/Kaggle/kagglehub
- Hugging Face Datasets streaming: https://huggingface.co/docs/datasets/stream
