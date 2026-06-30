# AGENTS.md

## Persistent Project References

- Training methods and artifact caveats:
  [TRAINING_METHODS_RUN_ARTIFACT_NOTES.md](./TRAINING_METHODS_RUN_ARTIFACT_NOTES.md)
- Run folder canonical naming standard:
  [RUN_FOLDER_NOMENCLATURE_STANDARD.md](./RUN_FOLDER_NOMENCLATURE_STANDARD.md)
- HF Tier-1 private archival workflow:
  [HF_PRIVATE_TIER1_ARCHIVAL_PLAYBOOK.md](./HF_PRIVATE_TIER1_ARCHIVAL_PLAYBOOK.md)
- HF Tier-1 execution history:
  [HF_PRIVATE_TIER1_EXECUTION_LOG.md](./HF_PRIVATE_TIER1_EXECUTION_LOG.md)
- Cloud eval stack (Kaggle + Colab):
  [CLOUD_EVAL_STACK_KAGGLE_COLAB.md](./CLOUD_EVAL_STACK_KAGGLE_COLAB.md)
- Private HF eval-pack creation workflow:
  [HF_PRIVATE_EVAL_PACK_PLAYBOOK.md](./HF_PRIVATE_EVAL_PACK_PLAYBOOK.md)
- HF profile (model namespace):
  https://huggingface.co/Sri-Harsha-Guthikonda

## Important Reminder

- For model-run classification, do not trust folder names alone.
- Some folders intentionally retain evaluation outputs after checkpoints/model files were deleted.
- Always classify runs by actual artifacts in the folder contents.

## Moonshine Collapse Investigation (High Level)

- The Moonshine streaming probe run `...mprobe1_3` triggered collapse at checkpoint `24`.
- Trigger type was repetition collapse (`short_or_repetitive_rate=0.5144` vs threshold `0.45`).
- WER/cap gates did not trigger at checkpoint 24; repetition gate triggered.
- Punctuation-heavy repetitive outputs increased sharply near trigger.
- Training logs also show persistent frame-shape batch exceptions (`shape '[4, -1, 80]' is invalid`), which likely contributed to instability.
- **Policy update (April 4, 2026):**
  - treat soft WER/CER degradation trends as collapse signals even when repetition gate is not yet triggered.
  - implemented in Stage-20 Moonshine collapse gate with soft WER/CER thresholds (default `5%`, patience `2`) plus explicit `cer_trigger` and soft-trigger fields in collapse decisions/reports.
- Deterministic manifest scan (`nframes % 80`) found `11126/161297` misaligned rows in stage-15 train manifest.
  - highest bad rates: `noise_mix` (~43.2%), `voice_mix` (~42.5%), `tempo_speech_pause` (100% of 6 rows).
- A frame-80-aligned filtered manifest run removed frame-mismatch skips for completed epochs, but later failed with `CUDA error: unknown error` (new blocker after alignment fix).
- **Cycle-2 (April 8, 2026):**
  - Probe A (`...collapse-investigation6`, aligned_full, 6 checkpoints) still collapsed at checkpoint `6`.
  - Trigger mix: soft WER/CER collapse (`soft_metric_source=targetmix_fallback`) plus repetition trigger.
  - New Stage-20 diagnostics added:
    - `debug_batch_exception_context.jsonl`
    - auto graph refresh from `collapse_history.jsonl` to `graphs/targetmix_wer_cer_by_checkpoint.png` and `graphs/collapse_indicators_by_checkpoint.png`
  - Probe B (`...collapse-investigation7`, aligned_minus_high_risk) also collapsed at checkpoint `5` (soft CER trigger only; no repetition trigger).
  - Relative to Probe A, Probe B reduced degeneration severity (`max short_or_repetitive_rate 0.3562` vs `0.4772`), but did not eliminate soft collapse.
  - Current best manifest variant in Cycle-2: `aligned_minus_high_risk` (best stability so far, still collapse-triggered).
  - Next step: Probe C (`aligned_core_only`, `...collapse-investigation8`) to test whether collapse persists with only `base/reverb/random_gain`.
- **Cycle-2 (April 10, 2026):**
  - Probe C (`...collapse-investigation8`, aligned_core_only) also collapsed at checkpoint `5`.
  - Trigger type remained soft CER collapse (`targetmix_fallback`) with no repetition trigger and no cap trigger.
  - A/B/C all collapse in <=6 checkpoints; repetition is reduced in B/C, but CER trend degradation persists.
  - Current cycle conclusion: stage ablation reduced repetition signatures but did not fix collapse; next branch should target CER degradation root causes beyond augmentation removal.
- **Cycle-3 plan (April 10, 2026): stage-ladder unaugmented probes**
  - Run cumulative stage snapshots with `aug_stage=""`: stage5 (`investigation9`) -> stage9 (`investigation10`) -> stage10b (`investigation11`) -> stage11 (`investigation12`).
  - Same Stage-20 hyperparameters/gates as Cycle-2, 6 checkpoints each, stop on collapse/crash.
  - Educated suspect order: stage10b tempo/pause first, then stage11 frequency, then stage9 reverb.
  - Objective: find earliest stage boundary where soft CER collapse first appears, then perform file-level pruning at that boundary.
- **Cycle-3 update (April 10, 2026):**
  - Stage-5 unaugmented control (`...collapse-investigation9`) also collapsed at checkpoint `5`.
  - Trigger mix: soft WER + soft CER (`targetmix_fallback`), no repetition trigger.
  - This shifts root-cause focus away from later augmentation stages and toward upstream/common dynamics (training schedule/optimization or baseline data/text quality).
  - Next branch: run U5 LR-sensitivity probes (`5e-6` vs `2e-6` vs `1e-6`) before resuming later-stage ladder runs.
- **Cycle-3 update (April 11, 2026):**
  - U5 `lr=2e-6` (`...collapse-investigation10`) still collapsed, but later/softer: checkpoint `6`, soft CER trigger only.
  - Relative to U5 `lr=5e-6`, drift improved materially (WER `+3.13%` vs `+5.18%`, CER `+9.01%` vs `+16.61%`).
  - Repetition remained low and non-triggering, reinforcing CER-trend as the active collapse mode.
  - Next step: U5 `lr=1e-6`; if still collapsing, move to file/text-quality pruning on stage-5 manifest.
- **Cycle-3 update (April 11, 2026):**
  - U5 `lr=1e-6` (`...collapse-investigation11`) completed all 6 checkpoints with `triggered=false`.
  - Drift stayed within gate margins (WER `+0.00%`, CER `+0.95%` from best), and repetition/cap gates remained off.
  - Current working conclusion: stage-5 collapse is optimization-sensitive and mitigated at `lr=1e-6` under current settings.
  - Next step: hold `lr=1e-6` and continue stage-ladder (stage9 -> stage10b -> stage11) to locate any later-stage reintroduction of collapse.
- Detailed investigation and artifact index:
  [MOONSHINE_COLLAPSE_INVESTIGATION.md](./MOONSHINE_COLLAPSE_INVESTIGATION.md)

## Kaggle Record_chunks Publish (High Level)

- Use `tools/kaggle_publish_record_chunks.py` to publish full `I:\Record_chunks` as private Kaggle datasets.
- Kaggle slug IDs must be lowercase alphanumeric plus `-`; underscores are rejected by Kaggle API.
- Current shard dataset IDs:
  - `drsriharshaguthik/acft-moonshine-record-chunks-audio-shard-a`
  - `drsriharshaguthik/acft-moonshine-record-chunks-audio-shard-b`
  - `drsriharshaguthik/acft-moonshine-record-chunks-manifests`
- Use license `CC0-1.0` in generated `dataset-metadata.json` to avoid Kaggle create-time license rejection in current CLI/API behavior.
- Keep `Record_chunks` path preserved inside shard datasets for stable relative-path behavior in existing manifests.
- Detailed operational steps and troubleshooting are documented in:
  [KAGGLE_RECORD_CHUNKS_PRIVATE_PUBLISH_README.md](./KAGGLE_RECORD_CHUNKS_PRIVATE_PUBLISH_README.md)

## Kaggle Record_chunks Chunked Publish (5000 files/dataset)

- For low-failure uploads, use chunked datasets at 5000 files per dataset:
  `tools/kaggle_publish_record_chunks_chunked.py`.
- Default chunk naming:
  `drsriharshaguthik/acft-moonshine-record-chunks-audio-chunk-001` ... `-039`.
- Chunked manifests dataset:
  `drsriharshaguthik/acft-moonshine-record-chunks-manifests-chunked`.
- Script now supports strict two-phase flow:
  - `--publish_phase precreate` to create all dataset slugs first.
  - `--publish_phase upload` to push data via `datasets version` only.
- `--skip_copy` now requires `chunk_plan_signature.json` match to prevent reusing stale staged chunk folders against a changed chunk plan.
- Detailed runbook:
  [KAGGLE_RECORD_CHUNKS_CHUNKED_5000_PUBLISH_README.md](./KAGGLE_RECORD_CHUNKS_CHUNKED_5000_PUBLISH_README.md)
