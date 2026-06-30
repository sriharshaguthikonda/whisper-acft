# Kaggle Private Publish Runbook: `Record_chunks`

This document records what was implemented, what was fixed, and how to run the publish manually.

## Scope

- Source folder: `I:\Record_chunks`
- Approx size discovered: ~26.708 GB
- File count discovered: 192,256 files
- Publish target: Kaggle private datasets (2 audio shards + 1 manifests/index dataset)

## What Was Implemented

- Script added/updated: `I:\whisper-acft\tools\kaggle_publish_record_chunks.py`
- Script behavior:
  - Enumerates full `Record_chunks` tree.
  - Deterministically splits files into two balanced shards (`a`/`b`) using largest-first greedy assignment.
  - Materializes shard trees as:
    - `<stage_root>\<shard-slug>\Record_chunks\...`
  - Rewrites manifests to Kaggle input paths and preserves `audio_path_original`.
  - Writes index and summary artifacts:
    - `shard_assignment.jsonl`
    - `shard_summary.json`
    - `audio_path_index.jsonl`
    - `manifest_rewrite_stats.json`
  - Creates/versions Kaggle datasets and verifies file listings.

## Root Cause of Previous Kaggle Failure

- Kaggle rejected dataset IDs containing underscores.
- Error seen: `Slug can only contain alphanumeric or "-" characters`.

## Fixes Applied

1. Updated default dataset IDs to slug-safe format (hyphenated, lowercase):
   - `drsriharshaguthik/acft-moonshine-record-chunks-audio-shard-a`
   - `drsriharshaguthik/acft-moonshine-record-chunks-audio-shard-b`
   - `drsriharshaguthik/acft-moonshine-record-chunks-manifests`
2. Switched metadata license default to `CC0-1.0` because Kaggle returned `Please select a valid license` for the previous license value in this flow.
3. Hardened upload logic:
   - Treats CLI textual failure lines as errors even when return code is 0.
   - Adds post-upload existence check via `kaggle datasets metadata`.
4. Improved copy robustness:
   - Windows long-path conversion (`\\?\` support).
   - Retry logic for transient `PermissionError`/`FileNotFoundError`.
5. Renamed existing staging folders to slug-safe names (same staged content retained):
   - `J:\kaggle_publish\acft-moonshine-Record_chunks_publish\acft-moonshine-record-chunks-audio-shard-a`
   - `J:\kaggle_publish\acft-moonshine-Record_chunks_publish\acft-moonshine-record-chunks-audio-shard-b`
   - `J:\kaggle_publish\acft-moonshine-Record_chunks_publish\acft-moonshine-record-chunks-manifests`

## Current Staging/Log Paths

- Stage root: `J:\kaggle_publish\acft-moonshine-Record_chunks_publish`
- Previous logs:
  - `publish_run.log`
  - `publish_run_retry.log`
- Prior copy failure ledger (from first attempt):
  - `copy_failures.jsonl`

## Validation Status (This Machine)

- Local full `--skip_upload` run: passed (`scan`, `shard`, `copy`, metadata rewrite, report write).
- Kaggle manifests dataset create: passed with slug-safe ID and `CC0-1.0` license.
- Verified dataset listing:
  - `drsriharshaguthik/acft-moonshine-record-chunks-manifests`

## Manual Run Commands (PowerShell)

Use these exactly from your machine.

### 1) Set token in current shell and verify auth

```powershell
$env:KAGGLE_API_TOKEN = [System.Environment]::GetEnvironmentVariable('KAGGLE_API_TOKEN','User')
& "C:\Users\deletable\AppData\Roaming\Python\Python311\Scripts\kaggle.exe" config view
```

Expected: username includes `drsriharshaguthik` and auth method `ACCESS_TOKEN`.

### 2) Run full publish (create or version as needed)

```powershell
python "I:\whisper-acft\tools\kaggle_publish_record_chunks.py" ^
  --source_root "I:\Record_chunks" ^
  --stage_root "J:\kaggle_publish\acft-moonshine-Record_chunks_publish" ^
  --train_manifest "I:\Record_chunks\pairs_manifest_stage15_train_no_targets_randomized.jsonl" ^
  --test_manifest "I:\Record_chunks\pairs_manifest_stage13_test_randomized.jsonl" ^
  --kaggle_exe "C:\Users\deletable\AppData\Roaming\Python\Python311\Scripts\kaggle.exe" ^
  --dataset_a "drsriharshaguthik/acft-moonshine-record-chunks-audio-shard-a" ^
  --dataset_b "drsriharshaguthik/acft-moonshine-record-chunks-audio-shard-b" ^
  --dataset_m "drsriharshaguthik/acft-moonshine-record-chunks-manifests" ^
  --license_name "CC0-1.0" ^
  --workers 2
```

Note: Existing staged files will be skipped if sizes match; this supports reruns/resume-like behavior.

### 3) Verify datasets on Kaggle

```powershell
& "C:\Users\deletable\AppData\Roaming\Python\Python311\Scripts\kaggle.exe" datasets files "drsriharshaguthik/acft-moonshine-record-chunks-audio-shard-a" --page-size 200
& "C:\Users\deletable\AppData\Roaming\Python\Python311\Scripts\kaggle.exe" datasets files "drsriharshaguthik/acft-moonshine-record-chunks-audio-shard-b" --page-size 200
& "C:\Users\deletable\AppData\Roaming\Python\Python311\Scripts\kaggle.exe" datasets files "drsriharshaguthik/acft-moonshine-record-chunks-manifests" --page-size 200
```

### 4) Check publish report artifacts

```powershell
Get-Content "J:\kaggle_publish\acft-moonshine-Record_chunks_publish\publish_report.json"
Get-Content "J:\kaggle_publish\acft-moonshine-Record_chunks_publish\publish_report.md"
```

## Kaggle Notebook Mount Paths

Use these in training/eval notebooks after publish:

- `/kaggle/input/acft-moonshine-record-chunks-audio-shard-a/Record_chunks/...`
- `/kaggle/input/acft-moonshine-record-chunks-audio-shard-b/Record_chunks/...`
- Rewritten manifests are in:
  - `/kaggle/input/acft-moonshine-record-chunks-manifests/pairs_manifest_stage15_train_no_targets_randomized_kaggle.jsonl`
  - `/kaggle/input/acft-moonshine-record-chunks-manifests/pairs_manifest_stage13_test_randomized_kaggle.jsonl`

## Troubleshooting Quick Checks

1. If upload fails with slug errors, confirm IDs contain only lowercase letters, numbers, and `-`.
2. If auth fails, refresh user env token and open a new shell.
3. If copy errors reappear, rerun with `--workers 1`.
4. If command stops mid-upload, rerun the same command; it will version existing datasets and skip already copied files.
