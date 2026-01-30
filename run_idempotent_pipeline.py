#!/usr/bin/env python3
"""run_idempotent_pipeline.py

Complete idempotent augmentation pipeline using stable UIDs.
This script demonstrates how to run all augmentation stages in order
with guaranteed idempotency and resume capability.

Usage:
-------
python run_idempotent_pipeline.py \
  --base_manifest "i:/Record_chunks/base_manifest.jsonl" \
  --output_dir "i:/Record_chunks/augmented" \
  --other_voices_dir "i:/Record_others_16k_wav" \
  --noise_dir "i:/noise/RIRS_NOISES/pointsource_noises" \
  --rir_dir "i:/noise/RIRS_NOISES/real_rirs_isotropic_noises" \
  --scores_csv "i:/whisper-acft/speaker_sort_scores.csv"

Stage Configuration:
--------------------
Each stage can be configured independently:
- --noise_ratio: Fraction of rows to augment with noise (default: 0.5)
- --voice_ratio: Fraction of rows to augment with other voices (default: 0.8)
- --gain_ratio: Fraction of rows to augment with random gain (default: 0.1)
- --reverb_ratio: Fraction of rows to augment with reverb (default: 0.3)

The pipeline is fully idempotent - you can stop and resume at any point,
and re-running will not create duplicate files or manifest entries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from pipeline_uid_utils import safe_beep


def seed_output_manifest(in_manifest: Path, out_manifest: Path) -> None:
    """Seed output manifest with base rows if it doesn't exist."""
    if out_manifest.exists():
        return  # Already seeded
    
    print(f"Seeding output manifest: {out_manifest}")
    with in_manifest.open("r", encoding="utf-8") as f_in, \
         out_manifest.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if line:
                f_out.write(line + "\n")
    
    print(f"Seeded {out_manifest} with base rows")


def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError as e:
        print(f"❌ {description} failed: {e}")
        return False


def check_file_exists(path: Path, description: str) -> bool:
    """Check if a required file exists."""
    if path.exists():
        print(f"✅ Found {description}: {path}")
        return True
    else:
        print(f"❌ Missing {description}: {path}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Run complete idempotent augmentation pipeline")
    
    # Required paths
    ap.add_argument("--base_manifest", required=True, help="Base manifest (should have base_uid)")
    ap.add_argument("--output_dir", required=True, help="Output directory for all augmented files")
    ap.add_argument("--other_voices_dir", required=True, help="Directory with other voice files")
    ap.add_argument("--noise_dir", required=True, help="Directory with noise files")
    ap.add_argument("--rir_dir", required=True, help="Directory with RIR files")
    ap.add_argument("--scores_csv", help="Speaker scores CSV for target selection")
    
    # Stage ratios
    ap.add_argument("--noise_ratio", type=float, default=0.5, help="Fraction for noise augmentation")
    ap.add_argument("--voice_ratio", type=float, default=0.8, help="Fraction for voice mixing")
    ap.add_argument("--gain_ratio", type=float, default=0.1, help="Fraction for random gain")
    ap.add_argument("--reverb_ratio", type=float, default=0.3, help="Fraction for reverb")
    
    # Stage copies
    ap.add_argument("--noise_copies", type=int, default=1, help="Noise copies per selected row")
    ap.add_argument("--voice_copies", type=int, default=1, help="Voice copies per selected row")
    ap.add_argument("--gain_copies", type=int, default=1, help="Gain copies per selected row")
    ap.add_argument("--reverb_copies", type=int, default=1, help="Reverb copies per selected row")
    
    # Audio parameters
    ap.add_argument("--snr_db_min", type=float, default=5.0, help="Minimum SNR for noise/voice mixing")
    ap.add_argument("--snr_db_max", type=float, default=20.0, help="Maximum SNR for noise/voice mixing")
    ap.add_argument("--gain_min_db", type=float, default=-12.0, help="Minimum gain in dB")
    ap.add_argument("--gain_max_db", type=float, default=12.0, help="Maximum gain in dB")
    ap.add_argument("--wet_min", type=float, default=0.2, help="Minimum wet mix for reverb")
    ap.add_argument("--wet_max", type=float, default=0.8, help="Maximum wet mix for reverb")
    
    # Processing parameters
    ap.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    ap.add_argument("--target_sr", type=int, default=16000, help="Target sample rate")
    
    # Control options
    ap.add_argument("--skip_noise", action="store_true", help="Skip noise augmentation stage")
    ap.add_argument("--skip_voice", action="store_true", help="Skip voice mixing stage")
    ap.add_argument("--skip_gain", action="store_true", help="Skip random gain stage")
    ap.add_argument("--skip_reverb", action="store_true", help="Skip reverb stage")
    ap.add_argument("--dry_run", action="store_true", help="Only show commands, don't execute")
    
    args = ap.parse_args()
    
    # Setup paths
    base_manifest = Path(args.base_manifest)
    output_dir = Path(args.output_dir)
    other_voices_dir = Path(args.other_voices_dir)
    noise_dir = Path(args.noise_dir)
    rir_dir = Path(args.rir_dir)
    scores_csv = Path(args.scores_csv) if args.scores_csv else None
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stable location for all idempotency databases (do NOT tie this to manifest filenames).
    seen_dir = output_dir / "_seen"
    seen_dir.mkdir(parents=True, exist_ok=True)
    
    # Check required files
    print("Checking required files...")
    all_files_exist = True
    
    all_files_exist &= check_file_exists(base_manifest, "Base manifest")
    all_files_exist &= check_file_exists(other_voices_dir, "Other voices directory")
    all_files_exist &= check_file_exists(noise_dir, "Noise directory")
    all_files_exist &= check_file_exists(rir_dir, "RIR directory")
    if scores_csv:
        all_files_exist &= check_file_exists(scores_csv, "Scores CSV")
    
    if not all_files_exist:
        print("\n❌ Some required files are missing. Please check the paths above.")
        sys.exit(1)
    
    # Check if base manifest has base_uid (run backfill if needed)
    print("\nChecking base manifest for base_uid...")
    has_base_uid = False
    try:
        with base_manifest.open("r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                row = json.loads(first_line)
                if row.get("base_uid"):
                    has_base_uid = True
    except Exception as e:
        print(f"Error checking base manifest: {e}")
        sys.exit(1)
    
    if not has_base_uid:
        print("⚠️  Base manifest missing base_uid. Running backfill...")
        backfilled_manifest = base_manifest.parent / f"{base_manifest.stem}_with_uid{base_manifest.suffix}"
        backfill_cmd = [
            sys.executable, "backfill_uids_in_manifest.py",
            "--in_jsonl", str(base_manifest),
            "--out_jsonl", str(backfilled_manifest)
        ]
        
        if not args.dry_run:
            if not run_command(backfill_cmd, "Backfill base_uid"):
                print("❌ Failed to backfill base_uid")
                sys.exit(1)
        else:
            print(f"Would run: {' '.join(backfill_cmd)}")
        
        base_manifest = backfilled_manifest
    
    # Stage 1: Noise augmentation
    current_manifest = base_manifest
    if not args.skip_noise:
        noise_manifest = output_dir / "with_noise.jsonl"
        noise_dir = output_dir / "noise_augmented"
        
        # Seed output manifest with base rows
        seed_output_manifest(current_manifest, noise_manifest)
        
        python_exe = sys.executable
        noise_cmd = [
            python_exe, "stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise.py",
            "--in_manifest", str(current_manifest),
            "--out_manifest", str(noise_manifest),
            "--noises_dir", str(args.noise_dir),
            "--out_dir", str(noise_dir),
            "--ratio", str(args.noise_ratio),
            "--copies", str(args.noise_copies),
            "--snr_db_min", str(args.snr_db_min),
            "--snr_db_max", str(args.snr_db_max),
            "--workers", str(args.workers),
            "--stage_name", "noise_mix",
            "--seen_db", str(seen_dir / "noise_mix.seen.sqlite"),
        ]
        
        if scores_csv:
            noise_cmd.extend(["--scores_csv", str(scores_csv)])
        
        if not args.dry_run:
            if not run_command(noise_cmd, "Noise augmentation"):
                print("❌ Noise augmentation failed")
                sys.exit(1)
        else:
            print(f"Would run: {' '.join(noise_cmd)}")
        
        current_manifest = noise_manifest
    
    # Stage 2: Voice mixing
    if not args.skip_voice:
        voice_manifest = output_dir / "with_voice.jsonl"
        voice_dir = output_dir / "voice_augmented"
        
        # Seed output manifest with base rows
        seed_output_manifest(current_manifest, voice_manifest)
        
        # IMPORTANT: use the *idempotent* stage script
        voice_cmd = [
            sys.executable, "stage_7_add_others_voices_to_my_audio_fast_idempotent.py",
            "--in_manifest", str(current_manifest),
            "--out_manifest", str(voice_manifest),
            "--other_voices_dir", str(other_voices_dir),
            "--out_dir", str(voice_dir),
            "--stage_name", "voice_mix",
            "--ratio", str(args.voice_ratio),
            "--copies", str(args.voice_copies),
            "--snr_db_min", str(args.snr_db_min),
            "--snr_db_max", str(args.snr_db_max),
            "--target_sr", str(args.target_sr),
            "--workers", str(args.workers),
            "--seen_db", str(seen_dir / "voice_mix.seen.sqlite"),
        ]
        
        if not args.dry_run:
            if not run_command(voice_cmd, "Voice mixing"):
                print("❌ Voice mixing failed")
                sys.exit(1)
        else:
            print(f"Would run: {' '.join(voice_cmd)}")
        
        current_manifest = voice_manifest
    
    # Stage 3: Random gain
    if not args.skip_gain:
        gain_manifest = output_dir / "with_gain.jsonl"
        gain_dir = output_dir / "gain_augmented"
        
        # Seed output manifest with base rows
        seed_output_manifest(current_manifest, gain_manifest)
        
        # IMPORTANT: use the *idempotent* stage script
        gain_cmd = [
            sys.executable, "stage_8_add_random_gain_to_high_score_voices_parallel_idempotent.py",
            "--in_manifest", str(current_manifest),
            "--out_manifest", str(gain_manifest),
            "--out_dir", str(gain_dir),
            "--stage_name", "random_gain",
            "--ratio", str(args.gain_ratio),
            "--copies", str(args.gain_copies),
            "--min_db", str(args.gain_min_db),
            "--max_db", str(args.gain_max_db),
            "--workers", str(args.workers),
            "--seen_db", str(seen_dir / "random_gain.seen.sqlite"),
        ]
        
        if not args.dry_run:
            if not run_command(gain_cmd, "Random gain augmentation"):
                print("❌ Random gain augmentation failed")
                sys.exit(1)
        else:
            print(f"Would run: {' '.join(gain_cmd)}")
        
        current_manifest = gain_manifest
    
    # Stage 4: Reverb
    if not args.skip_reverb:
        reverb_manifest = output_dir / "with_reverb.jsonl"
        reverb_dir = output_dir / "reverb_augmented"
        
        # Seed output manifest with base rows
        seed_output_manifest(current_manifest, reverb_manifest)
        
        # IMPORTANT: use the *idempotent* stage script
        reverb_cmd = [
            sys.executable, "stage_9_add_reverb_idempotent.py",
            "--in_manifest", str(current_manifest),
            "--out_manifest", str(reverb_manifest),
            "--rir_dir", str(rir_dir),
            "--out_dir", str(reverb_dir),
            "--stage_name", "reverb",
            "--ratio", str(args.reverb_ratio),
            "--copies", str(args.reverb_copies),
            "--wet_min", str(args.wet_min),
            "--wet_max", str(args.wet_max),
            "--workers", str(args.workers),
            "--seen_db", str(seen_dir / "reverb.seen.sqlite"),
        ]
        
        if not args.dry_run:
            if not run_command(reverb_cmd, "Reverb augmentation"):
                print("❌ Reverb augmentation failed")
                sys.exit(1)
        else:
            print(f"Would run: {' '.join(reverb_cmd)}")
        
        current_manifest = reverb_manifest
    
    # Final summary
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"Final manifest: {current_manifest}")
    print(f"Output directory: {output_dir}")
    
    if not args.dry_run:
        # Count final rows
        try:
            row_count = 0
            with current_manifest.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        row_count += 1
            print(f"Total rows in final manifest: {row_count}")
        except Exception:
            pass
        
        safe_beep()
    
    print("\nAll augmentation stages are now idempotent!")
    print("You can re-run this pipeline at any time - it will not create duplicates.")


if __name__ == "__main__":
    main()
