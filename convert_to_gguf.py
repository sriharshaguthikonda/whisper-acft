#!/usr/bin/env python3
"""
Convert a fine-tuned Whisper HF model to GGUF format for whisper.cpp

Usage:
    python convert_to_gguf.py --model_dir ./model_train-tiny3_extracted --output_dir ./gguf_output
"""

import os
import sys
import argparse
import subprocess

def download_tokenizer_files(model_dir: str, base_model: str = "openai/whisper-tiny"):
    """Download missing tokenizer files from base model."""
    from transformers import WhisperTokenizer, WhisperProcessor, WhisperFeatureExtractor
    
    print(f"Downloading tokenizer files from {base_model}...")
    
    # Download and save tokenizer
    tokenizer = WhisperTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(model_dir)
    
    # Download and save feature extractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model)
    feature_extractor.save_pretrained(model_dir)
    
    # Download and save processor
    processor = WhisperProcessor.from_pretrained(base_model)
    processor.save_pretrained(model_dir)
    
    print(f"Tokenizer files saved to {model_dir}")

def fix_config_for_conversion(model_dir: str):
    """Fix config.json to have proper architecture for Seq2Seq."""
    import json
    
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Ensure architecture is set correctly for speech-to-text
    if config.get("architectures") == ["WhisperModel"]:
        config["architectures"] = ["WhisperForConditionalGeneration"]
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print("Fixed config.json: Changed architecture to WhisperForConditionalGeneration")

def convert_hf_to_gguf(model_dir: str, output_dir: str, whisper_cpp_path: str = None):
    """Convert HF model to GGUF using whisper.cpp converter."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Try to find whisper.cpp converter
    if whisper_cpp_path:
        converter_candidates = [
            os.path.join(whisper_cpp_path, "models", "convert-hf-to-gguf.py"),
            os.path.join(whisper_cpp_path, "scripts", "convert-hf-to-gguf.py"),
        ]
    else:
        # Common locations
        converter_candidates = [
            r"C:\whisper.cpp\models\convert-hf-to-gguf.py",
            r"C:\Users\deletable\whisper.cpp\models\convert-hf-to-gguf.py",
            os.path.expanduser("~/whisper.cpp/models/convert-hf-to-gguf.py"),
        ]
    
    converter_path = None
    for c in converter_candidates:
        if os.path.exists(c):
            converter_path = c
            break
    
    if converter_path:
        print(f"Found converter at: {converter_path}")
        cmd = [
            sys.executable,
            converter_path,
            "--hf-repo", model_dir,
            "--outdir", output_dir,
        ]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"GGUF model saved to {output_dir}")
    else:
        # Alternative: use transformers' built-in GGUF export if available
        print("whisper.cpp converter not found. Trying alternative method...")
        try:
            convert_with_ctranslate2(model_dir, output_dir)
        except Exception as e:
            print(f"Alternative conversion failed: {e}")
            print("\nManual steps to convert:")
            print("1. Clone whisper.cpp: git clone https://github.com/ggerganov/whisper.cpp")
            print("2. Run: python whisper.cpp/models/convert-hf-to-gguf.py --hf-repo", model_dir, "--outdir", output_dir)

def convert_with_ctranslate2(model_dir: str, output_dir: str):
    """Convert to CTranslate2 format (alternative to GGUF)."""
    try:
        import ctranslate2
        print("Converting to CTranslate2 format...")
        converter = ctranslate2.converters.TransformersConverter(model_dir)
        converter.convert(output_dir, quantization="int8")
        print(f"CTranslate2 model saved to {output_dir}")
    except ImportError:
        print("ctranslate2 not installed. Install with: pip install ctranslate2")
        raise

def main():
    parser = argparse.ArgumentParser(description="Convert fine-tuned Whisper model to GGUF")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to HF model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for GGUF")
    parser.add_argument("--whisper_cpp", type=str, default=None, help="Path to whisper.cpp root")
    parser.add_argument("--base_model", type=str, default="openai/whisper-tiny", help="Base model for tokenizer")
    args = parser.parse_args()
    
    model_dir = os.path.abspath(args.model_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    print(f"Model directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    
    # Step 1: Download missing tokenizer files
    download_tokenizer_files(model_dir, args.base_model)
    
    # Step 2: Fix config if needed
    fix_config_for_conversion(model_dir)
    
    # Step 3: Convert to GGUF
    convert_hf_to_gguf(model_dir, output_dir, args.whisper_cpp)
    
    print("\nConversion complete!")

if __name__ == "__main__":
    main()
