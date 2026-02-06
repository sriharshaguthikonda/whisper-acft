# Idempotent Augmentation Pipeline

This directory contains a complete **idempotent** augmentation pipeline for Whisper training data. All augmentation stages are now deterministic and resume-safe.

## 🎯 Problem Solved

**Before**: Every re-run would create new augmented files because:
- Random selection of rows
- Random filenames with embedded parameters (`snr8.3dB`, `gain+2.1dB`)
- No duplicate detection
- Already augmented files treated as fresh candidates

**After**: Fully idempotent pipeline:
- ✅ **Stable base_uid** created once at chunking stage
- ✅ **Deterministic selection** per base_uid using hash-based bucketing
- ✅ **Deterministic RNG** seeded per (base_uid, stage, copy_idx)
- ✅ **SQLite seen index** prevents duplicates on resume/re-run
- ✅ **Stable filenames** with embedded UIDs (no random parameters)
- ✅ **Original row protection** (only augments uid==base_uid)

## 📁 Files Created

### Core Utilities
- `pipeline_uid_utils.py` - Stable hashing, UID generation, SQLite seen set
- `backfill_uids_in_manifest.py` - Add missing base_uid to existing manifests
- `stage_aug_idempotent_template.py` - Template for new augmentation stages

### Idempotent Stages
- `stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise_idempotent.py` - Noise augmentation
- `stage_7_add_others_voices_to_my_audio_fast_idempotent.py` - Voice mixing
- `stage_8_add_random_gain_to_high_score_voices_parallel_idempotent.py` - Random gain
- `stage_9_add_reverb_idempotent.py` - RIR convolution (reverb)

### Pipeline Runner
- `run_idempotent_pipeline.py` - Complete pipeline orchestrator
- `IDEMPOTENT_PIPELINE_README.md` - This documentation

## 🚀 Quick Start

### 1. Backfill base_uid (if needed)

If your existing manifest doesn't have `base_uid` fields:

```bash
python backfill_uids_in_manifest.py \
  --in_jsonl "I:/Record_chunks/base_manifest.jsonl" \
  --out_jsonl "I:/Record_chunks/base_manifest_with_uid.jsonl"
```

### 2. Run Complete Pipeline

```bash
python run_idempotent_pipeline.py \
  --base_manifest "I:/Record_chunks/base_manifest_with_uid.jsonl" \
  --output_dir "I:/Record_chunks/augmented" \
  --other_voices_dir "I:/Record_others_16k_wav" \
  --noise_dir "I:/noise/RIRS_NOISES/pointsource_noises" \
  --rir_dir "I:/noise/RIRS_NOISES/real_rirs_isotropic_noises" \
  --scores_csv "I:/whisper-acft/speaker_sort_scores.csv"
```

### 3. Individual Stages

You can also run stages individually:

```bash
# Noise augmentation
python stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise_idempotent.py \
  --in_manifest "I:/Record_chunks/base_manifest.jsonl" \
  --out_manifest "I:/Record_chunks/with_noise.jsonl" \
  --noises_dir "I:/noise/RIRS_NOISES/pointsource_noises" \
  --out_dir "I:/Record_chunks/noise_augmented" \
  --stage_name "noise_mix" \
  --ratio 0.5 \
  --copies 1 \
  --snr_db_min 5 --snr_db_max 20 \
  --workers 8

# Voice mixing
python stage_7_add_others_voices_to_my_audio_fast_idempotent.py \
  --in_manifest "I:/Record_chunks/with_noise.jsonl" \
  --out_manifest "I:/Record_chunks/with_voice.jsonl" \
  --other_voices_dir "I:/Record_others_16k_wav" \
  --out_dir "I:/Record_chunks/voice_augmented" \
  --stage_name "voice_mix" \
  --ratio 0.8 \
  --copies 1 \
  --snr_db_min 5 --snr_db_max 10 \
  --workers 8

# Random gain
python stage_8_add_random_gain_to_high_score_voices_parallel_idempotent.py \
  --in_manifest "I:/Record_chunks/with_voice.jsonl" \
  --out_manifest "I:/Record_chunks/with_gain.jsonl" \
  --out_dir "I:/Record_chunks/gain_augmented" \
  --stage_name "random_gain" \
  --ratio 0.1 \
  --copies 1 \
  --min_db -12 --max_db 12 \
  --workers 8

# Reverb
python stage_9_add_reverb_idempotent.py \
  --in_manifest "I:/Record_chunks/with_gain.jsonl" \
  --out_manifest "I:/Record_chunks/with_reverb.jsonl" \
  --rir_dir "I:/noise/RIRS_NOISES/real_rirs_isotropic_noises" \
  --out_dir "I:/Record_chunks/reverb_augmented" \
  --stage_name "reverb" \
  --ratio 0.3 \
  --copies 1 \
  --wet_min 0.2 --wet_max 0.8 \
  --workers 8
```

## 🔧 Key Features

### Stable UIDs
- `base_uid`: Stable identifier for original chunk (never changes)
- `uid`: Unique identifier for each augmented copy
- `parent_uid`: References the source row
- `aug_stage`: Stage name (e.g., "noise_mix", "voice_mix")
- `aug_copy_idx`: Copy number (1, 2, 3...)

### Deterministic Selection
```python
# Instead of: random.sample(rows, int(len(rows) * ratio))
# Now uses: hash-based bucketing
selected = should_select(base_uid, stage_name, ratio)
```

### Deterministic RNG
```python
# Instead of: random.Random(seed)
# Now uses: deterministic seed per output
rng = rng_for(base_uid, stage_name, copy_idx)
```

### Stable Filenames
```python
# Before: "chunk__row123__noisy__snr8.3dB__noise_abc.wav"
# After:  "chunk__uida1b2c3d4e5__noise_mix__copy01.wav"
```

### SQLite Seen Index
- Tracks `base_uid:stage_name:copy_idx` combinations
- Prevents duplicates across runs/resumes
- Handles manifest order changes
- Automatic cleanup and recovery

## 📊 Manifest Structure

Each augmented row contains:

```json
{
  "base_uid": "a1b2c3d4e5f6g7h8",           // Stable original ID
  "uid": "i9j0k1l2m3n4o5p6",               // This augmented copy's ID
  "parent_uid": "a1b2c3d4e5f6g7h8",       // Source row ID
  "aug_stage": "noise_mix",                // Stage that created this
  "aug_copy_idx": 1,                       // Copy number
  "out_wav": "chunk__uidi9j0k1l2m3n4o5p6__noise_mix__copy01.wav",
  "audio_path": "chunk__uidi9j0k1l2m3n4o5p6__noise_mix__copy01.wav",
  "aug_meta": {
    "snr_db": 12.3,
    "noise_source": "/path/to/noise.wav",
    "max_bad_to_good_ratio": 1.0,
    "good_floor_db": -45.0
  }
}
```

## 🔄 Resume & Re-run Behavior

### Scenario 1: Interrupted Run
- Pipeline stops at 60% completion
- Restart with same command
- Automatically skips completed rows (via SQLite index)
- Continues from where it left off

### Scenario 2: Parameter Change
- Change `--ratio 0.5` to `--ratio 0.7`
- Different rows selected (deterministic)
- Previously created files remain valid
- Only new augmentations created

### Scenario 3: Manifest Order Change
- Shuffle or filter input manifest
- Same base_uids get same augmentations
- No duplicate files created
- Output manifest order may differ, but content is stable

## 🛠️ Adding New Augmentation Stages

Use the template:

```python
# 1. Copy stage_aug_idempotent_template.py to your_new_stage.py
# 2. Implement do_augmentation() function
# 3. Update stage_name and parameters
# 4. Add to run_idempotent_pipeline.py if desired
```

Example `do_augmentation()`:
```python
def do_augmentation(row: Dict[str, Any], rng: random.Random, out_wav: str, args) -> Dict[str, Any]:
    # Load audio
    audio, sr = load_audio(row["audio_path"])
    
    # Apply your augmentation (use rng for randomness)
    augmented_audio = your_augmentation(audio, rng, args)
    
    # Save output
    save_audio(out_wav, augmented_audio, sr)
    
    # Return metadata
    return {"your_param": value, "other_param": value}
```

## 🎛️ Configuration

### Stage Ratios
- `--noise_ratio 0.5`: 50% of rows get noise augmentation
- `--voice_ratio 0.8`: 80% of rows get voice mixing
- `--gain_ratio 0.1`: 10% of rows get random gain
- `--reverb_ratio 0.3`: 30% of rows get reverb

### Copy Counts
- `--copies 1`: Create 1 augmented copy per selected row
- `--copies 3`: Create 3 different augmented copies per selected row

### Audio Parameters
- `--snr_db_min 5 --snr_db_max 20`: SNR range for noise/voice
- `--min_db -12 --max_db 12`: Gain range in dB
- `--wet_min 0.2 --wet_max 0.8`: Wet/dry mix for reverb

## 🐛 Troubleshooting

### Missing base_uid
```
❌ Missing base_uid
```
Solution: Run backfill script first.

### No RIR files found
```
❌ No valid RIR files found
```
Solution: Check RIR directory path and file formats.

### SQLite database locked
```
sqlite3.OperationalError: database is locked
```
Solution: Wait for other processes to finish, or delete `.seen.sqlite` files.

### Memory issues
Reduce `--workers` count or process smaller manifest chunks.

## 📈 Performance

- **Parallel processing**: All stages support multithreading
- **Memory efficient**: Streaming manifest processing
- **Resume safe**: SQLite index prevents re-work
- **Disk efficient**: No duplicate files created

## 🎉 Benefits

1. **Deterministic**: Same input → Same output (always)
2. **Resume-safe**: Stop and restart anytime
3. **No duplicates**: Never creates the same augmentation twice
4. **Stable filenames**: Easy to track and manage
5. **Metadata-rich**: Full augmentation parameters stored
6. **Pipeline-friendly**: Easy to chain stages
7. **Debuggable**: Clear error messages and progress tracking

## 📞 Support

The idempotent pipeline is designed to be robust and self-healing. If you encounter issues:

1. Check that all input directories exist and contain valid audio files
2. Ensure base_uid is present in your manifest
3. Delete `.seen.sqlite` files to reset the seen index
4. Use `--dry_run` to preview commands without executing

Happy augmenting! 🎵
