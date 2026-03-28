# HF Private Tier-1 Execution Log

Last updated: 2026-03-26

## Objective

Archive large model binaries from local `I:\RUN__*` folders to private Hugging Face repos (one repo per run), then keep local pointer/link files for restore.

HF profile / namespace:

- `https://huggingface.co/Sri-Harsha-Guthikonda`

## What Was Executed

1. Verified token-based auth from `.env` (`HF_TOKEN`) and validated private repo access.
2. Ran trial upload (private repo + upload + download verification).
3. Ran repo-by-repo archival using:
   - `tools/hf_tier1_archive_repo_by_repo.ps1`
   - input mapping from: `hf_tier1_reports/hf_tier1_archive_20260325_215859.csv`
4. Generated/updated per-run pointers:
   - `MODEL_POINTER.json`
   - `MODEL_POINTER_README.md`
   - `HF_MODEL_REPO.url`
5. Verified hybrid restore flow with:
   - `tools/hf_tier1_restore_from_pointer.py`

## Final Result Snapshot

- Hugging Face private repos created/available for archival set: `15`
- Local large model files remaining (`>= 50MB`, model extensions): `0`
- Run folders with pointer files: `15`
- Registry updated:
  - `HF_PRIVATE_TIER1_REGISTRY.json`
- Session reports:
  - `hf_tier1_reports/hf_tier1_repo_by_repo_20260326_032138.csv/.json`
  - `hf_tier1_reports/hf_tier1_repo_by_repo_20260326_035628.csv/.json`
  - `hf_tier1_reports/hf_tier1_repo_by_repo_20260326_035745.csv/.json`

## Canonical Operational Commands

Repo-by-repo archival:

```powershell
Set-Location I:\whisper-acft
& I:\whisper-acft\tools\hf_tier1_archive_repo_by_repo.ps1 -RepoRoot I:\whisper-acft -RunsRoot I:\ -CacheDir I:\hf_model_cache -InputCsv I:\whisper-acft\hf_tier1_reports\hf_tier1_archive_20260325_215859.csv -CleanupLocal
```

Restore from pointer:

```powershell
Set-Location I:\whisper-acft
$env:PYTHONNOUSERSITE = "1"
I:\Whisper-training-env\Scripts\python.exe I:\whisper-acft\tools\hf_tier1_restore_from_pointer.py --pointer "I:\RUN__<run>\MODEL_POINTER.json" --cache-dir I:\hf_model_cache --local-dir "I:\RUN__<run>\restored_model"
```

## Notes

- All archival repos are private under your account.
- For future runs, keep using one repo per run and update the CSV mapping first.
- If token exposure is suspected in terminal history, rotate `HF_TOKEN`.
