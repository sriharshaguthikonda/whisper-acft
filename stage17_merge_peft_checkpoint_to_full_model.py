#!/usr/bin/env python3
"""stage17_merge_peft_checkpoint_to_full_model.py

Merge a PEFT/LoRA adapter checkpoint into a full Whisper model directory.

Why you need this:
- Your Stage 17 LoRA checkpoints will be *adapter-only* (adapter_config.json + adapter weights).
- Some of your evaluation / export scripts expect a normal Transformers model folder.

Usage (PowerShell):
  $env:HF_HUB_DISABLE_TELEMETRY="1"
  i:\Whisper-training-env\Scripts\python.exe i:\whisper-acft\stage17_merge_peft_checkpoint_to_full_model.py `
    --peft_dir "I:\...\checkpoints\model_epoch_000010" `
    --out_dir  "I:\...\checkpoints\model_epoch_000010_merged" `
    --base_model_id "futo-org/acft-whisper-tiny.en"
"""

import argparse
import os
import winsound


def _beep():
    try:
        winsound.Beep(880, 200)
        winsound.Beep(988, 200)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--peft_dir", required=True, help="Adapter checkpoint dir (contains adapter_config.json)")
    ap.add_argument("--out_dir", required=True, help="Output dir for merged full model")
    ap.add_argument("--base_model_id", default="openai/whisper-tiny.en", help="Base Whisper model id")
    args = ap.parse_args()

    from transformers import WhisperForConditionalGeneration, GenerationConfig
    from peft import PeftModel

    if not os.path.isfile(os.path.join(args.peft_dir, "adapter_config.json")):
        raise SystemExit(f"Not a PEFT checkpoint dir: {args.peft_dir}")

    try:
        gen_config = GenerationConfig.from_pretrained(args.base_model_id)
    except Exception:
        gen_config = None

    base = WhisperForConditionalGeneration.from_pretrained(args.base_model_id, generation_config=gen_config)
    model = PeftModel.from_pretrained(base, args.peft_dir, is_trainable=False)

    # Merge LoRA weights into the base model and drop adapter modules.
    merged = model.merge_and_unload(progressbar=True, safe_merge=True)

    os.makedirs(args.out_dir, exist_ok=True)
    merged.save_pretrained(args.out_dir)

    print("✓ Saved merged full model to:", args.out_dir)
    _beep()


if __name__ == "__main__":
    main()

