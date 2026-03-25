# Training Methods And Run Artifact Notes

Last updated: 2026-03-25

## Purpose
This note documents how training methods should be interpreted in this repo/workspace.

Folder names are **not** reliable ground truth for how a model was trained.
Always classify by artifacts inside each folder.

## Critical Caveat (Do Not Ignore)
Some folders contain evaluation outputs from a training run, but the original checkpoints and/or model files were later deleted.

That means:
- A folder may still have `evaluation_results*.json` and `evaluation_per_sample_predictions*.json`.
- The same folder may no longer have `model_epoch_*`, `model.safetensors`, or adapter weights.
- Such folders should be treated as `eval-only remnants`, not complete training runs.

## Artifact-Based Classification Rules
Use these checks in order:

1. Training checkpoint artifacts:
- `model_epoch_*` directories
- `run_state.json`
- `training_state_epoch_*.json`

2. Adapter-based method (LoRA/DoRA-like):
- `adapter_config.json`
- `adapter_model.safetensors`

3. Full-model checkpoint method:
- `model.safetensors` or `pytorch_model.bin`

4. Evaluation artifacts:
- `evaluation_results*.json`
- `evaluation_per_sample_predictions*.json`

5. Partial/incomplete run indicators:
- `run_state.json` + `pending_epoch_plan.json` + logs
- missing model/checkpoint artifacts

## Methods Seen In This Workspace

- Adapter-based fine-tuning (LoRA/DoRA-style): identified by adapter config/weights.
- Full-model checkpoint training: identified by full model weight files.
- Merged/export artifacts: converted/merged outputs without full checkpoint history.
- Eval-only remnants: evaluation files present after checkpoint/model cleanup.
- Partial/incomplete runs: state/log files exist, but checkpoint artifacts are incomplete or absent.

## Example Folders (From Current Scan)

Likely train + eval:
- `I:\Stage_17_aug_futo_wer_rank64_dora_dyn_ctx_chkpts_small_en_26`
- `I:\Stage_17_aug_futo_wer_rank32_dora_dyn_ctx_chkpts_small_en_25`
- `I:\Stage_17_aug_futo_wer_dora_dyn_ctx_chkpts_small_en_24`
- `I:\Stage_17_aug_futo_wer_dora_dyn_ctx_chkpts_small_en_23`
- `I:\Stage_17_aug_futo_wer_dora_dyn_ctx_chkpts_small_en_22`
- `I:\Stage_17_shuffle_wer_acft_qat6_0_checkpoints_partialctx_tiny_en_15`
- `I:\Stage_17_no_aug_openai_wer_acft_qat6_0_checkpoints_partialctx_tiny_en_16`
- `I:\Stage_17_no_aug_openai_wer_acft_lora_qat6_0_chkpts_tiny_en_17`
- `I:\Stage_17_aug_openai_wer_acft_lora_qat6_0_chkpts_tiny_en_18`
- `I:\Stage_17_aug_openai_wer_lora_dyn_ctx_qat6_0_chkpts_tiny_en_19`
- `I:\Stage_17_aug_futo_wer_dora_dyn_ctx_qat6_0_chkpts_tiny_en_20`

Likely eval-only remnants (checkpoints/models may have been removed):
- `I:\Stage_19e_edge_eval_20260315_full_pct100`
- `I:\Stage_19e_edge_eval_full100_20260315`
- `I:\Dynamic_n_ctx_checkpoints_partialctx`
- `I:\Dynamic_n_ctx_checkpoints_partialctx2`
- `I:\Dynamic_n_ctx_checkpoints_partialctx3`
- `I:\Stage_18_shuffle_Dynamic_n_ctx_checkpoints_partialctx4`
- `I:\Stage_18_shuffle_Dynamic_n_ctx_checkpoints_partialctx5`
- `I:\Stage_18_shuffle_Dynamic_n_ctx_checkpoints_partialctx6`

Likely partial/incomplete or mixed:
- `I:\Stage_17_aug_futo_wer_dora_dyn_ctx_chkpts_tiny_en_21`
- `I:\Stage_18_shuffle_Dynamic_n_ctx_stage_7_9_checkpoints_partialctx_tiny_en_9`
- `I:\Stage_18_shuffle_Dynamic_n_ctx_stage_7_checkpoints_partialctx_tiny_en_10`
- `I:\Stage_18_shuffle_Dynamic_n_ctx_stage_7_checkpoints_partialctx_tiny_en_11`
- `I:\Stage_2_shuffle_Dynamic_n_ctx_stage_7_checkpoints_partialctx_tiny_en_11`

Non-training/data or unrelated:
- `I:\Record_test_chunks`
- `I:\Record_chunks_tempo_pause`
- `I:\delete_as_soon_as_you_see_this_ai_prompt_queue`

## Decision Policy For Future Work
- Do not assume method from directory name alone.
- Require artifact verification before comparing methods.
- Mark each run as one of:
  - `train_checkpoint_dir`
  - `train_plus_eval_dir`
  - `eval_results_dir`
  - `partial_or_incomplete`
  - `data_or_non_training`

