#!/usr/bin/env python3
r"""export_merge_and_convert_whispercpp.py

What you have:
- Your training checkpoints are saved as Hugging Face `WhisperModel` (encoder/decoder only).
- whisper.cpp's `models/convert-h5-to-ggml.py` loads `WhisperForConditionalGeneration`.

So if you point the converter at a `WhisperModel` checkpoint dir, it will either:
- fail, or
- silently create a model with a randomly initialised output head (junk transcription).

This script fixes that by creating an EXPORT DIR that contains a *real* WhisperForConditionalGeneration:
- Start from BASE_MODEL_ID (brings the correct output head)
- Copy your fine-tuned encoder/decoder weights into it
- Save the merged model + matching tokenizer/feature-extractor
Then it calls whisper.cpp conversion + (optional) quantize.

NOTE: whisper.cpp (as of early 2026) primarily uses GGML .bin models, not GGUF.

Usage (Windows):
  i:\Whisper-training-env\Scripts\python.exe i:\whisper-acft\export_merge_and_convert_whispercpp.py \
    --checkpoint_dir i:\checkpoints_partialctx\model_epoch_000003 \
    --export_dir     i:\checkpoints_partialctx\export_epoch_000003_hfseq2seq \
    --whisper_cpp    i:\whisper-acft\whisper.cpp \
    --whisper_repo   i:\whisper-acft\whisper \
    --qtype Q8_0

qtype for whisper.cpp quantize: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 (or omit for FP16).
"""

import os
import sys
import json
import argparse
import subprocess
import re
from pathlib import Path

import torch

from transformers import WhisperProcessor, WhisperModel, WhisperForConditionalGeneration


def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def export_merged_seq2seq(
    checkpoint_dir: str,
    export_dir: str,
    base_model_id: str,
    processor_id: str,
    hf_token: str | None = None,
):
    """Create a proper HF WhisperForConditionalGeneration directory from a WhisperModel checkpoint."""
    ensure_dir(export_dir)

    # Load your fine-tuned encoder/decoder
    try:
        ckpt = WhisperModel.from_pretrained(checkpoint_dir, token=hf_token)
    except Exception:
        # Fall back to a full seq2seq checkpoint (e.g., merged PEFT)
        seq2seq = WhisperForConditionalGeneration.from_pretrained(checkpoint_dir, token=hf_token)
        ckpt = seq2seq.model

    # Load base seq2seq model (brings proj_out / output head)
    gen = WhisperForConditionalGeneration.from_pretrained(base_model_id, token=hf_token)

    # Copy encoder/decoder weights
    gen.model.load_state_dict(ckpt.state_dict(), strict=True)

    # Make config explicitly seq2seq
    cfg = gen.config.to_dict()
    cfg["architectures"] = ["WhisperForConditionalGeneration"]
    with open(os.path.join(export_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Save weights
    # Use safe_serialization=True (safetensors) if available; transformers handles it.
    gen.save_pretrained(export_dir)

    # Save tokenizer + feature extractor + processor
    # IMPORTANT: processor_id must match your model family (use whisper-small.en for English-only)
    proc = WhisperProcessor.from_pretrained(processor_id, token=hf_token)
    proc.save_pretrained(export_dir)

    print(f"✅ Exported merged HF seq2seq model to: {export_dir}")


def find_converter(whisper_cpp_root: str) -> str:
    p = os.path.join(whisper_cpp_root, "models", "convert-h5-to-ggml.py")
    if not os.path.exists(p):
        raise FileNotFoundError(f"convert-h5-to-ggml.py not found at: {p}")
    return p


def find_quantize(whisper_cpp_root: str) -> str:
    candidates = [
        os.path.join(whisper_cpp_root, "build", "bin", "Release", "quantize.exe"),
        os.path.join(whisper_cpp_root, "build", "bin", "quantize.exe"),
        os.path.join(whisper_cpp_root, "quantize.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "quantize.exe not found. Build whisper.cpp first, e.g.:\n"
        "  cmake -B build -DCMAKE_BUILD_TYPE=Release\n"
        "  cmake --build build --config Release"
    )


def extract_checkpoint_info(checkpoint_dir: str) -> dict:
    """Extract meaningful information from checkpoint directory name."""
    checkpoint_name = os.path.basename(os.path.normpath(checkpoint_dir))
    
    # Extract epoch number from names like "model_epoch_000001"
    epoch_match = re.search(r'epoch_(\d+)', checkpoint_name)
    epoch_num = epoch_match.group(1) if epoch_match else "unknown"
    
    # Extract other info if present
    info = {
        'checkpoint_name': checkpoint_name,
        'epoch': epoch_num,
        'base_name': checkpoint_name
    }
    
    return info


def generate_model_name(checkpoint_info: dict, qtype: str = "", use_f32: bool = False) -> str:
    """Generate a meaningful model filename."""
    base_name = "whisper"
    
    # Add epoch info
    if checkpoint_info['epoch'] != "unknown":
        base_name += f"-epoch{checkpoint_info['epoch']}"
    
    # Add precision info
    if use_f32:
        base_name += "-f32"
    elif qtype:
        base_name += f"-{qtype.lower()}"
    else:
        base_name += "-fp16"
    
    return base_name


def generate_base_name(checkpoint_info: dict) -> str:
    """Generate base name without quantization info."""
    base_name = "whisper"
    
    # Add epoch info
    if checkpoint_info['epoch'] != "unknown":
        base_name += f"-epoch{checkpoint_info['epoch']}"
    
    return base_name


def convert_to_ggml(export_dir: str, out_dir: str, whisper_cpp_root: str, whisper_repo_root: str, use_f32: bool = False, model_name: str = "ggml-model"):
    ensure_dir(out_dir)
    converter = find_converter(whisper_cpp_root)

    # convert-h5-to-ggml.py expects: dir_model, path-to-openai-whisper-repo, dir-output, [use-f32]
    cmd = [sys.executable, converter, export_dir, whisper_repo_root, out_dir]
    if use_f32:
        cmd.append("use-f32")

    print("\nRunning conversion:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Keep the original filename for quantization input
    default_filename = "ggml-model-f32.bin" if use_f32 else "ggml-model.bin"
    ggml_path = os.path.join(out_dir, default_filename)
    
    if not os.path.exists(ggml_path):
        raise FileNotFoundError(f"Expected output not found: {ggml_path}")

    print(f"✅ GGML model: {ggml_path}")
    return ggml_path


def quantize_ggml(ggml_fp16_path: str, qtype: str, whisper_cpp_root: str, model_name: str) -> str:
    qexe = find_quantize(whisper_cpp_root)

    base_dir = os.path.dirname(ggml_fp16_path)
    qtype_norm = qtype.strip().lower()
    
    # Create a temporary output path for quantization
    temp_out_path = os.path.join(base_dir, f"temp_quantized_{qtype_norm}.bin")
    
    # The final output path with the meaningful name
    final_out_path = os.path.join(base_dir, f"{model_name}.bin")

    cmd = [qexe, ggml_fp16_path, temp_out_path, qtype_norm]
    print("\nRunning quantize:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    
    # Rename the temp file to the final meaningful name
    if os.path.exists(temp_out_path):
        os.rename(temp_out_path, final_out_path)

    print(f"✅ Quantized model: {final_out_path}")
    return final_out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", required=True, help="HF checkpoint dir saved as WhisperModel (e.g., model_epoch_000003)")
    ap.add_argument("--export_dir", required=True, help="Output HF seq2seq dir to create")
    ap.add_argument("--out_dir", default=None, help="Where to put ggml-model.bin (default: <export_dir>\\ggml)")
    ap.add_argument("--whisper_cpp", required=True, help="Path to whisper.cpp root")
    ap.add_argument("--whisper_repo", required=True, help="Path to OpenAI whisper repo root (must contain whisper/assets/mel_filters.npz)")

    ap.add_argument("--base_model", default="futo-org/acft-whisper-small.en", help="Base seq2seq model id for LM head")
    ap.add_argument("--processor", default="openai/whisper-small.en", help="Processor id (tokenizer/feature extractor). Use .en if English-only")

    ap.add_argument("--qtype", default="", help="Quantize type (Q4_0,Q4_1,Q5_0,Q5_1,Q8_0). Empty = no quantize")
    ap.add_argument("--use_f32", action="store_true", help="Export F32 GGML instead of default FP16")

    args = ap.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    
    # Extract checkpoint information for meaningful naming
    checkpoint_info = extract_checkpoint_info(args.checkpoint_dir)
    base_name = generate_base_name(checkpoint_info)
    fp16_name = generate_model_name(checkpoint_info, "", args.use_f32)
    quantized_name = generate_model_name(checkpoint_info, args.qtype.strip(), args.use_f32)
    
    print(f"📁 Converting checkpoint: {checkpoint_info['checkpoint_name']}")
    print(f"🏷️  Base name: {base_name}")
    print(f"📄 FP16 model: {fp16_name}")
    if args.qtype.strip():
        print(f"🔢 Quantized model: {quantized_name}")

    # Sanity: whisper repo must contain mel_filters.npz
    mel = os.path.join(args.whisper_repo, "whisper", "assets", "mel_filters.npz")
    if not os.path.exists(mel):
        raise FileNotFoundError(f"OpenAI whisper repo not found / wrong root. Missing: {mel}")

    export_merged_seq2seq(
        checkpoint_dir=os.path.abspath(args.checkpoint_dir),
        export_dir=os.path.abspath(args.export_dir),
        base_model_id=args.base_model,
        processor_id=args.processor,
        hf_token=hf_token,
    )

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(os.path.abspath(args.export_dir), "ggml")
    ggml_fp = convert_to_ggml(
        export_dir=os.path.abspath(args.export_dir),
        out_dir=out_dir,
        whisper_cpp_root=os.path.abspath(args.whisper_cpp),
        whisper_repo_root=os.path.abspath(args.whisper_repo),
        use_f32=args.use_f32,
        model_name=fp16_name,
    )
    
    # Rename the FP16/F32 file to have a meaningful name
    if ggml_fp:
        fp16_path = os.path.join(out_dir, f"{fp16_name}.bin")
        if os.path.exists(ggml_fp) and ggml_fp != fp16_path:
            os.rename(ggml_fp, fp16_path)
            print(f"✅ Renamed FP16 model: {fp16_path}")
            ggml_fp = fp16_path

    if args.qtype.strip():
        quantize_ggml(ggml_fp, args.qtype.strip(), os.path.abspath(args.whisper_cpp), quantized_name)


if __name__ == "__main__":
    main()
