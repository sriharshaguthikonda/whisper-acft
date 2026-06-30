# Kaggle Primary Training Data Export

Last updated: 2026-06-29

## Outcome

The training-needed primary data from `I:\` was uploaded to private Kaggle Datasets under `drsriharshaguthik/*`.

Generated chunks and run artifacts were not treated as source of truth. The canonical Kaggle source set preserves the old top-level folder names and relative paths:

- `Record_harsha/...`
- `Transcriptions_corrected/...`
- `Record_only_by_harsha/...`
- `Record_others_compacted/...`
- `noise/RIRS_NOISES/...`
- `whisper-acft/speaker_sort_scores.csv`
- `whisper-acft/most_commonly_spoken_segments_state.json`

No local source data was deleted. Generated chunk audio was later deleted after explicit approval; see `docs/GENERATED_CHUNK_CLEANUP_2026-06-29.md`.

## Source Selection

Evidence came from `Full_pipeline_whisper_training_local.py` and `pipeline.log`.

Required primary inputs:

- `I:\Record_harsha`
- `I:\Transcriptions_corrected`

Optional regeneration inputs uploaded:

- `I:\Record_only_by_harsha`
- `I:\Record_others_compacted`
- `I:\noise\RIRS_NOISES`
- `I:\whisper-acft\speaker_sort_scores.csv`
- `I:\whisper-acft\most_commonly_spoken_segments_state.json`

Excluded generated outputs:

- `Record_chunks`
- `Record_chunks_*`
- `Record_test_chunks`
- `Stage_*`
- `RUN__*`
- model checkpoints, caches, envs, and run output folders

The two `Transcriptions_corrected/stage_0_apply_corrections_from_report_conflicts_*` files are generated correction-conflict artifacts and are not part of canonical transcript coverage.

## Canonical Kaggle Datasets

### Refs / Noise / State

- `drsriharshaguthik/acft-moonshine-primary-refs-noise`

### Record_harsha Audio

Canonical audio coverage is exactly:

- `drsriharshaguthik/acft-moonshine-src-record-harsha-001`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-002`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-003`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-004`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-005`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-006`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-007`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-008`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-009`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-010`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-011`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-012`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-014`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-015`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-016`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-018`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-hidden-clean-001`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-hidden-clean-002`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-hidden-clean-003`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-hidden-clean-004`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-hidden-clean-005`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-hidden-clean-006`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-hidden-clean-007`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-hidden-clean-008`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-hidden-clean-009`

Do not use `acft-moonshine-src-record-harsha-013` or `acft-moonshine-src-record-harsha-017`; Kaggle hid them with 403. The `hidden-clean-*` datasets replace both original hidden shards from saved manifests, excluding only confirmed rejected audio files.

Audio verification:

- Original manifest files: 18,750
- Canonical uploaded files: 18,748
- Confirmed Kaggle rejects: 2
- Missing expected paths: 0
- Extra paths: 0
- Duplicate paths: 0

Confirmed Kaggle-rejected audio files:

- `Record_harsha/Michael Lewis's Gender Dysphoria Consultation.m4a`
- `Record_harsha/Robbie's Post-Traumatic Stress Disorder_ A Doctor's Consultation.m4a`

### Corrected Transcripts

Canonical transcript coverage is sharded. Use the verified datasets below.

Base/rest shards:

- `drsriharshaguthik/acft-moonshine-src-transcripts-shard-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-rest-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-rest-002`
- `drsriharshaguthik/acft-moonshine-src-transcripts-rest-003`
- `drsriharshaguthik/acft-moonshine-src-transcripts-rest-004`
- `drsriharshaguthik/acft-moonshine-src-transcripts-rest-005`
- `drsriharshaguthik/acft-moonshine-src-transcripts-rest-006`

Tail 7 shards:

- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-002`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-004`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-005`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003a-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003-part-002`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003-part-003`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003b-file-002`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003b-file-003`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003b-file-004`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003b-file-005`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003b-file-006`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003b-file-007`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003b-file-008`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003d-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-suspect-a-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-suspect-a-002`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003e-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-suspect-b-002`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail7-003f-001`

Tail 8 shards:

- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-002`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-003`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-a-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-r8808-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-b1-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-b2-002`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-b2-003`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-b2-004`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-c-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-c-002`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-c-003`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-c-004`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-c-005`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-c-006`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-d-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-r8847-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-e-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-f-001`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-f-002`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-f-003`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-f-004`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-f-005`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-f-006`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-f-007`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-f-008`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-f-009`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-f-010`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-f-011`
- `drsriharshaguthik/acft-moonshine-src-transcripts-tail8-f-012`

Transcript verification:

- `Transcriptions_corrected` total files observed: 8,864
- Generated conflict artifacts excluded: 2
- Confirmed Kaggle rejects: 3
- Expected canonical transcript files on Kaggle: 8,859

Confirmed Kaggle-rejected transcript files:

- `Transcriptions_corrected/A Teenager's Struggle_ Navigating Sexuality and Mental Health.json`
- `Transcriptions_corrected/Michael Lewis's Gender Dysphoria Consultation.json`
- `Transcriptions_corrected/Robbie's Post-Traumatic Stress Disorder_ A Doctor's Consultation.json`

## Non-Canonical Kaggle Attempts

Do not use these as source-of-truth datasets:

- `drsriharshaguthik/acft-moonshine-primary-training-data`
- `drsriharshaguthik/acft-moonshine-primary-transcriptions-corrected`
- `drsriharshaguthik/acft-moonshine-primary-record-harsha`
- `drsriharshaguthik/acft-moonshine-src-transcripts`
- `drsriharshaguthik/acft-moonshine-archive-transcripts`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-013`
- `drsriharshaguthik/acft-moonshine-src-record-harsha-017`
- Any `record-harsha-013*`, `record-harsha-017*`, or `record-harsha-recovery-*` attempt dataset not listed as canonical above

Those were partial, seed-only, hidden, duplicate-producing, or diagnostic isolation uploads.

## Generated Chunk Cleanup

Generated chunk audio was removed locally on 2026-06-29 after explicit approval.

Deleted:

- `I:\Record_chunks`: `.wav` chunk files only.
- `I:\Record_test_chunks`: all files; the folder contained only generated `.wav` chunks.
- Generated Record_chunks/Record_test_chunks staging and probe folders under `J:\kaggle_publish`.

Preserved:

- `I:\Record_chunks` root folder and 38 non-audio manifest/state/provenance files.
- `I:\Record_test_chunks` root folder.
- `I:\Record_harsha`.
- `I:\Transcriptions_corrected`.
- Canonical primary-source Kaggle staging folders.

Evidence and exact counts:

- `docs/GENERATED_CHUNK_CLEANUP_2026-06-29.md`
- `artifacts/generated_chunk_delete_20260629/delete_plan.json`
- `artifacts/generated_chunk_delete_20260629/delete_after.json`
- `artifacts/generated_chunk_delete_20260629/delete_kaggle_publish_plan.json`
- `artifacts/generated_chunk_delete_20260629/delete_kaggle_publish_after.json`

## Publisher Script

Publisher:

```powershell
python I:\whisper-acft\tools\kaggle_publish_primary_training_data.py --help
```

Important options added during this export:

- `--chunk_file_limit`
- `--chunk_byte_limit_gb`
- `--row_start_index`
- `--row_end_index`
- `--extra_file_list`
- `--extra_manifest_jsonl`
- `--exclude_rel_path_list`
- `--no_default_includes`

Use `--extra_manifest_jsonl` for recovery from saved manifests. Do not use row slices against a mutable live source directory unless the source inventory is known frozen.

Clean audio hidden-shard command shape:

```powershell
python I:\whisper-acft\tools\kaggle_publish_primary_training_data.py `
  --source_root I:\ `
  --stage_root J:\kaggle_publish\acft-moonshine-src-record-harsha-hidden-clean-publish `
  --no_default_includes `
  --no_default_extra_files `
  --extra_manifest_jsonl J:\kaggle_publish\acft-moonshine-src-record-harsha-publish\acft-moonshine-src-record-harsha-013\manifests\primary_source_inventory.jsonl `
  --extra_manifest_jsonl J:\kaggle_publish\acft-moonshine-src-record-harsha-publish\acft-moonshine-src-record-harsha-017\manifests\primary_source_inventory.jsonl `
  --exclude_rel_path_list I:\whisper-acft\docs\kaggle_record_harsha_hidden_clean_excludes.txt `
  --owner drsriharshaguthik `
  --chunk_slug_prefix acft-moonshine-src-record-harsha-hidden-clean `
  --chunk_file_limit 50 `
  --upload
```

## Local Evidence

Primary stage/report roots:

- `J:\kaggle_publish\acft-moonshine-primary-refs-noise-publish`
- `J:\kaggle_publish\acft-moonshine-src-record-harsha-publish`
- `J:\kaggle_publish\acft-moonshine-src-record-harsha-hidden-clean-publish`
- `J:\kaggle_publish\acft-moonshine-src-transcripts-*`

Useful logs:

- `J:\kaggle_publish\logs\record_harsha_upload_20260615_083651.out.log`
- `J:\kaggle_publish\logs\record_harsha_hidden_clean_20260615_110703.out.log`

Repo-intel status:

- `.repo-intel/manifest.json` exists.
- `repo-map scan --repo I:\whisper-acft --json` identified `sriharshaguthikonda/whisper-acft`, branch `main`.
- Repo-map artifact was orientation-only because the worktree was dirty and full artifact generation timed out on this large repo.

## Restore Guidance

In Kaggle, attach the canonical datasets and read files by their preserved relative paths under `/kaggle/input/<dataset-slug>/...`.

For a local restore, download canonical datasets and overlay by relative path into one root. If duplicate attempts are downloaded accidentally, dedupe by `rel_path` and prefer the canonical list in this document.

Local generated chunk audio has already been deleted as approved on 2026-06-29. Do not delete the preserved non-audio manifests/state files under `I:\Record_chunks` unless there is a separate explicit approval.
