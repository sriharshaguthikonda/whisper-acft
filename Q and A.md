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

1. tell me how to run these as well!