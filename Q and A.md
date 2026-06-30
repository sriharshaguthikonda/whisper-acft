# Q and A

Last updated: 2026-06-30

## Current Status

- Baseline Kaggle/export docs/tools/evidence committed in `71f2dba`.
- Resumable Kaggle helper, tests, and chunk notebook committed in `dd9a93f`.
- Training smoke/resume notebook committed in `4b73387`.
- Docs update in progress for final workflow commit.

## Open Questions For User

None blocking.

## Decisions Applied

- Use two Kaggle notebooks:
  - `notebooks/kaggle/01_generate_chunks_publish.ipynb`
  - `notebooks/kaggle/02_train_smoke_resume.ipynb`
- Use Kaggle private datasets as persistent storage between sessions.
- Reconstruct canonical source top-level paths under `/kaggle/working/acft_data`.
- Publish generated chunks/manifests under `drsriharshaguthik/acft-kaggle-chunks-<run-tag>-<profile>`.
- Publish training checkpoints/logs under `drsriharshaguthik/acft-kaggle-train-<run-tag>-<profile>`.
- Keep Common Voice optional until access/licensing input is explicit.
- Cap public-ASR rows so public rows are at most about 30 percent of the mixed train manifest.

## Verification Log

- `python -m pytest test_kaggle_acft_helpers.py test_kaggle_publish_record_chunks_chunked.py -q` passed: 13 tests.
- Notebook JSON parsed for both Kaggle notebooks.
- Notebook code cells parsed with `ast.parse`.
- `python -m compileall -q tools\kaggle_acft_helpers.py` passed.



## USER COMMENTS

1. tell me how to run these on kaggle as well?
2. dont think you have used the memory system?!

## Agent Answers

1. Kaggle run steps:
   - Upload/copy `notebooks/kaggle/01_generate_chunks_publish.ipynb` into a Kaggle notebook.
   - Attach all canonical private source datasets listed in `docs/KAGGLE_PRIMARY_TRAINING_DATA_EXPORT.md`.
   - Attach repo files or set `ACFT_GIT_URL` to a cloneable repo URL.
   - Run notebook 01 with defaults: `PROFILE=smoke`, `PUBLISH_AFTER_EACH_STAGE=1`, `DRY_RUN_PUBLISH=0`.
   - Confirm the private chunks dataset appears as `drsriharshaguthik/acft-kaggle-chunks-<run-tag>-smoke`.
   - Upload/copy `notebooks/kaggle/02_train_smoke_resume.ipynb`.
   - Attach the chunks dataset from notebook 01, use a GPU runtime, and run with defaults: `LR_START=1e-6`, `MAX_EPOCHS=1`, `N_SAMPLES_PER_EPOCH=32`, `START_FRESH=0`.
   - Confirm the private train dataset appears as `drsriharshaguthik/acft-kaggle-train-<run-tag>-smoke`.
   - For resume tests, rerun with the same `RUN_TAG` and attach the previous chunks/train dataset. Notebook 01 restores `_kaggle_state`; notebook 02 restores `state/` and `checkpoints/`.
2. Memory system was used:
   - Queried project memory for canonical Kaggle source/export context and supersession.
   - Used the active memory result that says canonical source datasets must come from `docs/KAGGLE_PRIMARY_TRAINING_DATA_EXPORT.md`, avoid failed attempt datasets, and keep generated chunks out of source mirror.
   - Will write a closeout memory after push with the new notebook/helper/docs facts.
