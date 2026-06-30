# Architecture

Last updated: 2026-06-29

## Data Lineage

The training pipeline is source-first:

1. Primary audio is stored under `I:\Record_harsha`.
2. Corrected transcript JSON is stored under `I:\Transcriptions_corrected`.
3. Stage 1 creates manifests from primary audio and corrected transcripts.
4. Stage 2 chunks audio into `Record_chunks`.
5. Later stages sort speakers, add augmentation, split train/test, and train/evaluate models.

`Record_chunks`, `Stage_*`, and checkpoint folders are reproducible outputs, not primary source data.

As of 2026-06-29, generated chunk audio has been cleaned locally:

- `I:\Record_chunks` exists and preserves 38 non-audio manifest/state/provenance files, with 0 `.wav` files.
- `I:\Record_test_chunks` exists and is empty.
- `I:\Record_harsha` and `I:\Transcriptions_corrected` remain the source-of-truth inputs.
- Cleanup details are in `docs/GENERATED_CHUNK_CLEANUP_2026-06-29.md`.

## Kaggle Source Mirror

The Kaggle mirror for source recovery is documented in:

- `docs/KAGGLE_PRIMARY_TRAINING_DATA_EXPORT.md`

Use the canonical dataset list there. It preserves top-level source folder names and excludes generated chunks.

Generated chunk upload retries were stopped. Kaggle CLI tar mode strips the top folder from archive members, so any future chunk-training notebook must reconstruct old paths under `/kaggle/working/acft_chunks/<top-folder>/...` before consuming manifests that expect preserved `Record_chunks` or `Record_test_chunks` paths.

## Repo Intel

`.repo-intel/manifest.json` exists for this repo. In this run, repo-map status was orientation-only because the worktree was dirty and full map generation timed out. Use live files for edits and repo-map slices only as orientation unless a fresh scan succeeds.

## Publisher Boundary

`tools/kaggle_publish_primary_training_data.py` is the publisher used for primary source data. It stages manifests, copies selected files, creates Kaggle metadata, uploads chunked datasets, and verifies visible file paths.

For mutable source folders, prefer manifest-driven selection with `--extra_manifest_jsonl` over row slicing.
