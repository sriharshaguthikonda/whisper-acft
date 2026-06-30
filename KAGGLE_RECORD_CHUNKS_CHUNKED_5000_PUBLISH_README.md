# Kaggle Chunked Publish Runbook (5000 Files per Dataset)

This runbook is for low-risk publishing of full `I:\Record_chunks` to many smaller private Kaggle datasets.

## Why This Variant

- Instead of ~13 GB shard uploads, this splits by file-count (default `5000` files per dataset).
- Result on this machine: `39` audio datasets, each roughly `~0.687 GB` tar payload.
- Lower blast radius per failure; reruns only repeat one chunk upload at a time.

## Script

- `I:\whisper-acft\tools\kaggle_publish_record_chunks_chunked.py`

## Current Defaults

- Owner: `drsriharshaguthik`
- Audio slug prefix: `acft-moonshine-record-chunks-audio-chunk`
- Chunk dataset IDs:
  - `drsriharshaguthik/acft-moonshine-record-chunks-audio-chunk-001`
  - ... through `-039`
- Manifests dataset:
  - `drsriharshaguthik/acft-moonshine-record-chunks-manifests-chunked`
- License used in metadata: `CC0-1.0`

## What The Script Does

1. Enumerates all files under `I:\Record_chunks`.
2. Deterministically assigns files into chunk bins (`<= 5000` files/chunk), roughly byte-balanced.
3. Materializes dataset folders as:
   - `<stage_root>\acft-moonshine-record-chunks-audio-chunk-XYZ\Record_chunks\...`
4. Rewrites manifests to `/kaggle/input/<chunk-slug>/Record_chunks/...` and preserves `audio_path_original`.
5. Writes index and summaries:
   - `chunk_assignment.jsonl`
   - `chunk_summary.json`
   - `audio_path_index.jsonl`
   - `manifest_rewrite_stats.json`
6. Creates/versions chunk datasets + manifests dataset.
7. Supports strict two-phase publishing:
   - `--publish_phase precreate`: create all datasets using tiny seed payloads.
   - `--publish_phase upload`: upload real content with `datasets version` only.
8. Protects `--skip_copy` with `chunk_plan_signature.json` so stale staged folders are not reused with a changed chunk plan.

## Kaggle Async Behavior Note

- Kaggle `datasets create` can return success while metadata/files endpoints still return temporary `403`.
- Script behavior: accepted create is treated as success; immediate files verification is skipped when dataset is not yet visible.
- On a later rerun, those datasets typically go through `version` path and verify cleanly.

## Validated On This Machine

- Local full `--skip_upload` run: passed.
- Live smoke upload: passed for
  - `acft-moonshine-record-chunks-audio-chunk-001` (create accepted)
  - `acft-moonshine-record-chunks-manifests-chunked` (create + files listing visible)
- Stage root:
  - `J:\kaggle_publish\acft-moonshine-record-chunks-chunked-publish`

## Run Commands (PowerShell)

### 1) Auth in current shell

```powershell
$env:KAGGLE_API_TOKEN = [System.Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN','User')
& "C:\Users\deletable\AppData\Roaming\Python\Python311\Scripts\kaggle.exe" config view
```

### 2) One-time local materialization + manifest rewrite (no upload)

```powershell
python "I:\whisper-acft\tools\kaggle_publish_record_chunks_chunked.py" ^
  --chunk_file_limit 5000 ^
  --workers 2 ^
  --skip_upload
```

### 3) Precreate all datasets first (no chunk data upload)

```powershell
python "I:\whisper-acft\tools\kaggle_publish_record_chunks_chunked.py" ^
  --publish_phase precreate ^
  --chunk_file_limit 5000 ^
  --upload_retry_max 12 ^
  --upload_retry_backoff_seconds 120 ^
  --sleep_between_uploads_seconds 30
```

### 4) Full upload of all chunks + manifests (version only; reuses staged folders)

```powershell
python "I:\whisper-acft\tools\kaggle_publish_record_chunks_chunked.py" ^
  --publish_phase upload ^
  --chunk_file_limit 5000 ^
  --workers 2 ^
  --skip_copy ^
  --upload_retry_max 12 ^
  --upload_retry_backoff_seconds 120 ^
  --sleep_between_uploads_seconds 45
```

### 5) Resume upload from chunk N

```powershell
python "I:\whisper-acft\tools\kaggle_publish_record_chunks_chunked.py" ^
  --publish_phase upload ^
  --chunk_file_limit 5000 ^
  --workers 2 ^
  --skip_copy ^
  --start_chunk_index 22 ^
  --end_chunk_index 39 ^
  --upload_retry_max 12 ^
  --upload_retry_backoff_seconds 120 ^
  --sleep_between_uploads_seconds 45
```

### 6) Optional smoke upload of only first chunk + manifests

```powershell
python "I:\whisper-acft\tools\kaggle_publish_record_chunks_chunked.py" ^
  --publish_phase upload ^
  --chunk_file_limit 5000 ^
  --workers 2 ^
  --skip_copy ^
  --max_chunks_upload 1
```

## Verify Datasets Later (when visibility catches up)

```powershell
$env:KAGGLE_API_TOKEN = [System.Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN','User')
& "C:\Users\deletable\AppData\Roaming\Python\Python311\Scripts\kaggle.exe" datasets files "drsriharshaguthik/acft-moonshine-record-chunks-audio-chunk-001" --page-size 200
& "C:\Users\deletable\AppData\Roaming\Python\Python311\Scripts\kaggle.exe" datasets files "drsriharshaguthik/acft-moonshine-record-chunks-manifests-chunked" --page-size 200
```

## Kaggle Notebook Paths

- Audio chunks:
  - `/kaggle/input/acft-moonshine-record-chunks-audio-chunk-001/Record_chunks/...`
  - ... `-039`
- Chunked manifests:
  - `/kaggle/input/acft-moonshine-record-chunks-manifests-chunked/pairs_manifest_stage15_train_no_targets_randomized_kaggle.jsonl`
  - `/kaggle/input/acft-moonshine-record-chunks-manifests-chunked/pairs_manifest_stage13_test_randomized_kaggle.jsonl`

## Artifacts

- Main report:
  - `J:\kaggle_publish\acft-moonshine-record-chunks-chunked-publish\publish_report.json`
- Human summary:
  - `J:\kaggle_publish\acft-moonshine-record-chunks-chunked-publish\publish_report.md`
