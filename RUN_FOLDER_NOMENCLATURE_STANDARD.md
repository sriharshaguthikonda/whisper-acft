# Run Folder Nomenclature Standard

Last updated: 2026-03-25

## Goal
Use one universal naming system that works for current and future training methods.

Folder names must be derived from artifacts/logs (not only from legacy names).
Original names are preserved via junction aliases for backward compatibility.

## Canonical Format

`RUN__k-<kind>__s-<stage>__b-<base>__m-<method>__a-<adapter>__q-<quant>__c-<ctx>__r-<rows>__id-<id>`

All values are lowercase slug tokens (`a-z`, `0-9`, `-`).

## Token Definitions

- `k` run kind:
  - `train-only`, `train-eval`, `eval-only`, `partial`, `misc`, `unk`
- `s` pipeline stage or phase:
  - examples: `17`, `18`, `19e`, `legacy`
- `b` base model family:
  - examples: `futo-small-en`, `futo-tiny-en`, `openai-tiny-en`, `unk`
- `m` method token:
  - stage-prefixed and recipe-oriented
  - examples: `s17-qat-dora`, `s17-qat-lora`, `s17-full`, `eval-only`, `unk`
- `a` adapter token:
  - examples: `dora-r64-a16`, `lora-r32-a16`, `full`, `unk`
- `q` quantization mode:
  - `qat`, `noqat`, `unk`
- `c` context mode:
  - `dyn`, `static`, `unk`
- `r` row count:
  - integer string when known, else `unk`
- `id` run identity hint:
  - examples: `auto`, `20260325`, `exp-b`

## Method Token Guidance (Future-Proof)

Use `m` as `s<stage>-<recipe>-<adapterfamily>[-<extra>]`.

Examples for new methods:
- `s17-qlora`
- `s17-ia3`
- `s20-distill-lora`
- `s22-multitask-full`

If a method does not fit existing values, add a new slug in `m` and keep all other tokens unchanged.

## Repeated Runs With Same Metadata

If the exact canonical prefix repeats, append numeric suffix outside the canonical part:

- `RUN__...__id-auto_1`
- `RUN__...__id-auto_2`

This keeps metadata stable while allowing resumable iteration.

## Extension Rules

- Unknown values must be explicit as `unk` (never omitted).
- Keep token order fixed for stable sorting/search.
- If a new dimension is required, append a new token at the end:
  - example: `__x-loss-ctc`

## Alias Policy (Original Names)

When renaming existing folders:

1. Rename the real folder to canonical format.
2. Recreate original folder name as a junction alias to canonical folder.
3. Write `RENAMED_FOLDER_NOTE.md` in canonical folder.

## Required Rename Note

Each renamed canonical folder must include `RENAMED_FOLDER_NOTE.md` with:
- old and new paths
- reason for rename
- evidence source (logs/artifacts/scripts)
- timestamp

## Implementation In This Repo

- Canonical name helpers: `run_folder_naming.py`
- Historical rename tooling:
  - `tools/build_run_rename_plan.ps1`
  - `tools/apply_run_rename_plan.ps1`
- Stage-17 local pipeline can auto-generate canonical run names via:
  - `USE_CANONICAL_RUN_FOLDER_NAMING = True` in `Full_pipeline_whisper_training_local.py`
