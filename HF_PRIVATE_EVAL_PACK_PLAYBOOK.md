# HF Private Eval-Pack Playbook

This workflow creates a private Hugging Face **dataset** repo snapshot of your evaluation set, with rewritten manifests and speaker-score paths so cloud runs work without manual file uploads.

## Script

- `I:\whisper-acft\tools\hf_private_eval_pack_uploader.py`

## What the uploader does

- Reads your source test manifest and optional OTHER manifest.
- Copies referenced audio files into a portable `audio/` tree.
- Rewrites manifest `audio_path` entries to portable paths.
- Rewrites `speaker_sort_scores.csv` `file` values when a path mapping exists.
- Writes `PACK_METADATA.json` with checksums and counts.
- Uploads to a private HF dataset repo under `eval_packs/<pack_id>`.
- Updates `eval_packs/LATEST_PACK.json` pointer.

## Run (local Windows)

```powershell
I:\Whisper-training-env\Scripts\python.exe I:\whisper-acft\tools\hf_private_eval_pack_uploader.py `
  --repo-id "Sri-Harsha-Guthikonda/whisper-acft-indian-accent-eval-private" `
  --repo-root "I:\whisper-acft" `
  --source-root "I:\" `
  --pack-tag "stage13-indian-accent-en" `
  --manifest "I:\Record_chunks\pairs_manifest_stage13_test.jsonl" `
  --speaker-scores-csv "I:\whisper-acft\speaker_sort_scores.csv" `
  --others-manifest "I:\Record_others_compacted\pairs_pending_stereo.jsonl" `
  --private 1
```

## Dry-run

```powershell
I:\Whisper-training-env\Scripts\python.exe I:\whisper-acft\tools\hf_private_eval_pack_uploader.py `
  --repo-id "Sri-Harsha-Guthikonda/whisper-acft-indian-accent-eval-private" `
  --repo-root "I:\whisper-acft" `
  --source-root "I:\" `
  --pack-tag "stage13-indian-accent-en" `
  --manifest "I:\Record_chunks\pairs_manifest_stage13_test.jsonl" `
  --speaker-scores-csv "I:\whisper-acft\speaker_sort_scores.csv" `
  --dry-run `
  --keep-staging
```

## Required secret

- `HF_TOKEN` in:
  - environment variable, or
  - `I:\whisper-acft\.env`
