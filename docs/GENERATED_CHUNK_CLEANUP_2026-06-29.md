# Generated Chunk Cleanup 2026-06-29

## Outcome

Approved generated chunk cleanup was completed on 2026-06-29.

Deleted:

- `I:\Record_chunks`: generated `.wav` chunk files only.
- `I:\Record_test_chunks`: all files; the folder contained only generated `.wav` chunks.
- `J:\kaggle_publish`: generated Record_chunks/Record_test_chunks staging and probe folders only.

Preserved:

- `I:\Record_chunks` root folder.
- `I:\Record_test_chunks` root folder.
- `I:\Record_chunks` non-audio manifest/state/provenance files.
- Primary source folders `I:\Record_harsha` and `I:\Transcriptions_corrected`.
- Canonical primary-source Kaggle staging folders under `J:\kaggle_publish`.

## Why This Was Safe

The deleted audio files are generated pipeline outputs, not source data.

Evidence:

- `Full_pipeline_whisper_training_local.py` defines `CHUNKS_DIR = I:\Record_chunks` and `TEST_CHUNKS_DIR = I:\Record_test_chunks`.
- Stage 2 runs `stage_2_chunk_transcripts_sentence_parallel.py`, which cuts chunks from source audio and writes `pairs_manifest_stereo.jsonl`.
- Stages 6-11 generate augmentation outputs under `Record_chunks`.
- Stage 13 creates train/test manifests.
- Stage 16 runs `Stage_16_move_test_chunks_update_test_manifest.py` to move test audio into `Record_test_chunks`.
- Source folders still exist: `I:\Record_harsha` and `I:\Transcriptions_corrected`.

## Local Deletion Details

`I:\Record_chunks` before deletion:

- Total files: 192,261
- Total size: 26.89 GiB
- Generated `.wav` files deleted: 192,223
- Generated `.wav` bytes deleted: 26,257,168,390
- Non-audio files preserved: 38

`I:\Record_test_chunks` before deletion:

- Total files deleted: 17,898
- Total bytes deleted: 2,782,251,192
- File type: `.wav` only

Combined local generated audio deleted:

- Files: 210,121
- Bytes: 29,039,419,582

Post-delete verification:

- `I:\Record_chunks`: exists, 38 files, 0 `.wav`
- `I:\Record_test_chunks`: exists, 0 files, 0 `.wav`
- `I:\Record_harsha`: exists
- `I:\Transcriptions_corrected`: exists

Evidence files:

- `I:\whisper-acft\artifacts\generated_chunk_delete_20260629\delete_plan.json`
- `I:\whisper-acft\artifacts\generated_chunk_delete_20260629\delete_after.json`

## Kaggle Staging Cleanup

Deleted generated staging/probe folders:

- `J:\kaggle_publish\acft-moonshine-record-chunks-chunked-publish`
- `J:\kaggle_publish\acft-moonshine-record-test-chunks-chunked-publish`
- `J:\kaggle_publish\acft-moonshine-record-test-chunks-byte450-publish`
- `J:\kaggle_publish\acft-moonshine-Record_chunks_publish`
- `J:\kaggle_publish\kaggle-tar-smoke`
- `J:\kaggle_publish\probe_tar_smoke_download`
- `J:\kaggle_publish\probe_audio_001_download`
- `J:\kaggle_publish\probe_audio_001_meta`
- `J:\kaggle_publish\probe_test_manifest_meta`

Combined staging/probe deletion:

- Files: 429,760
- Bytes: 64,589,650,913

Evidence files:

- `I:\whisper-acft\artifacts\generated_chunk_delete_20260629\delete_kaggle_publish_plan.json`
- `I:\whisper-acft\artifacts\generated_chunk_delete_20260629\delete_kaggle_publish_after.json`

## Kaggle Chunk Upload Note

Chunk dataset upload work was stopped after failed large tar/version attempts.

Findings before cleanup:

- The first invalid manifest slug was `acft-moonshine-record-test-chunks-manifests-chunked`, length 51, above Kaggle's 50-character dataset slug limit.
- The publisher now validates Kaggle dataset id, slug length, slug character policy, title length, and subtitle length before upload.
- Current b450 test-chunk names passed local name preflight, but audio dataset version processing still failed/hid uploaded versions.
- Kaggle CLI tar mode strips the top folder from archive members. A top-level folder such as `Record_test_chunks` must be reconstructed in Kaggle working storage if old paths are needed.

No generated chunk audio was kept locally solely for Kaggle upload retry.
