# Roadmap

Last updated: 2026-06-29

## Current Milestone: Primary Source Data On Kaggle

Status: complete with documented Kaggle rejects and local generated chunk cleanup.

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

Details:

- `docs/KAGGLE_PRIMARY_TRAINING_DATA_EXPORT.md`
- `docs/GENERATED_CHUNK_CLEANUP_2026-06-29.md`

## Next Work

- Decide whether Kaggle-rejected source files should remain local-only, be rewritten/redacted upstream, or be excluded permanently.
- Update Kaggle/Colab notebooks to use the canonical source datasets instead of local `I:\` paths and regenerate chunks on Kaggle.
- Build a small Kaggle training notebook that reconstructs `Record_chunks`/`Record_test_chunks` under `/kaggle/working/acft_chunks/...` before training.
- Add a restore script that downloads canonical datasets and overlays them by relative path.
- Keep `I:\Record_chunks` non-audio manifests/state files unless separately approved for deletion.
