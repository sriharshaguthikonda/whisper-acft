# Roadmap

Last updated: 2026-06-30

## Current Milestone: Resumable Kaggle Chunk And Training Notebooks

Status: implemented locally; pending Kaggle smoke trial.

Completed:

- Identified primary training inputs from the local pipeline.
- Uploaded required primary audio and transcript sources to private Kaggle datasets.
- Uploaded optional regeneration inputs: reference audio, compacted other-speaker audio, RIRS noise, and small state files.
- Excluded generated chunks, checkpoints, run outputs, caches, and envs from the source mirror.
- Preserved top-level folder names and relative paths in Kaggle datasets.
- Verified canonical `Record_harsha` coverage: 18,748 uploaded files, 2 confirmed rejected files, 0 missing, 0 extra, 0 duplicates.
- Documented transcript rejects and generated conflict artifacts.
- Deleted approved local generated chunk audio: `I:\Record_chunks` `.wav` files and all files in `I:\Record_test_chunks`.
- Deleted generated Record_chunks/Record_test_chunks Kaggle staging/probe folders.
- Preserved source data plus Record_chunks manifest/state files.
- Added `tools/kaggle_acft_helpers.py` for canonical source reconstruction, stage signatures, resume decisions, Kaggle Dataset metadata/publish, public-ASR capping, and local smoke fixtures.
- Added `notebooks/kaggle/01_generate_chunks_publish.ipynb` to regenerate chunks on Kaggle and publish chunk/manifests as a private dataset.
- Added `notebooks/kaggle/02_train_smoke_resume.ipynb` to attach generated chunks, mix capped clean public ASR data, run Stage 17 smoke training, and publish checkpoints/logs as a private dataset.
- Added `docs/KAGGLE_RESUMABLE_NOTEBOOKS.md`.

Details:

- `docs/KAGGLE_PRIMARY_TRAINING_DATA_EXPORT.md`
- `docs/GENERATED_CHUNK_CLEANUP_2026-06-29.md`
- `docs/KAGGLE_RESUMABLE_NOTEBOOKS.md`

## Next Work

- Decide whether Kaggle-rejected source files should remain local-only, be rewritten/redacted upstream, or be excluded permanently.
- Run Kaggle trial notebook 01 with `PROFILE=smoke` and verify published chunks dataset visibility.
- Attach the chunks dataset to notebook 02, run one small Stage 17 epoch, stop/restart once, and verify checkpoint resume.
- Decide whether Common Voice can be enabled after access/licensing input.
- Keep `I:\Record_chunks` non-audio manifests/state files unless separately approved for deletion.
