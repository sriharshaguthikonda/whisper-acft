#!/usr/bin/env python3
r"""
Convert a fine-tuned Whisper HF model to GGUF format for whisper.cpp

Usage:
    # Quantized GGUF (default q8_0; change with --qtype q5_1, q4_0, f16, etc.)
    python convert_to_gguf.py --model_dir ./model_train-tiny3_extracted --output_dir ./gguf_output --qtype q8_0

    # Legacy GGML path (if you must):
    mkdir ggml_output; python whisper.cpp/models/convert-pt-to-ggml.py model_train-tiny3.pt r"C:\Users\deletable\AppData\Roaming\Python\Python312\site-packages" ./ggml_output
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

def convert_hf_to_gguf(model_dir: str, output_dir: str, whisper_cpp_path: str = None, qtype: str | None = "q8_0"):
    """Convert HF model to GGML using whisper.cpp converter (with optional quantization)."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find whisper.cpp converter (convert-h5-to-ggml.py)
    if whisper_cpp_path:
        converter_candidates = [
            os.path.join(whisper_cpp_path, "models", "convert-h5-to-ggml.py"),
        ]
        whisper_repo_candidates = [
            os.path.join(os.path.dirname(whisper_cpp_path), "whisper"),
        ]
    else:
        # Common locations
        converter_candidates = [
            r"i:\whisper-acft\whisper.cpp\models\convert-h5-to-ggml.py",
            r"whisper.cpp\models\convert-h5-to-ggml.py",
            os.path.expanduser("~/whisper.cpp/models/convert-h5-to-ggml.py"),
        ]
        whisper_repo_candidates = [
            r"i:\whisper-acft\whisper",
            r"whisper",
            os.path.expanduser("~/whisper"),
        ]
    
    converter_path = None
    for c in converter_candidates:
        if os.path.exists(c):
            converter_path = c
            break
    
    whisper_repo_path = None
    for w in whisper_repo_candidates:
        if os.path.exists(os.path.join(w, "whisper", "assets", "mel_filters.npz")):
            whisper_repo_path = w
            break
    
    if converter_path and whisper_repo_path:
        print(f"Found converter at: {converter_path}")
        print(f"Found whisper repo at: {whisper_repo_path}")
        
        # Run convert-h5-to-ggml.py
        # Usage: convert-h5-to-ggml.py dir_model path-to-whisper-repo dir-output [use-f32]
        use_f32 = "use-f32" if qtype == "f32" or qtype == "f16" else None
        cmd = [
            sys.executable,
            converter_path,
            model_dir,
            whisper_repo_path,
            output_dir,
        ]
        if use_f32:
            cmd.append(use_f32)
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        # Handle quantization if needed
        ggml_file = os.path.join(output_dir, "ggml-model.bin")
        if os.path.exists(ggml_file) and qtype and qtype not in ["f16", "f32"]:
            print(f"\nQuantizing to {qtype}...")
            quantize_ggml(ggml_file, qtype, whisper_cpp_path)
        
        print(f"GGML model saved to {output_dir}")
    else:
        # Alternative: use transformers' built-in export if available
        print("whisper.cpp converter not found. Trying alternative method...")
        print(f"Converter path: {converter_path}")
        print(f"Whisper repo path: {whisper_repo_path}")
        try:
            convert_with_ctranslate2(model_dir, output_dir)
        except Exception as e:
            print(f"Alternative conversion failed: {e}")
            print("\nManual steps to convert:")
            print("1. Clone whisper.cpp: git clone https://github.com/ggerganov/whisper.cpp")
            print("2. Clone whisper: git clone https://github.com/openai/whisper")
            print("3. Run: python whisper.cpp/models/convert-h5-to-ggml.py", model_dir, "./whisper", output_dir)

def quantize_ggml(ggml_file: str, qtype: str, whisper_cpp_path: str = None):
    """Quantize GGML model using whisper.cpp quantize tool."""
    
    # Find quantize executable
    if whisper_cpp_path:
        quantize_candidates = [
            os.path.join(whisper_cpp_path, "build", "bin", "Release", "quantize.exe"),
            os.path.join(whisper_cpp_path, "build", "bin", "quantize.exe"),
            os.path.join(whisper_cpp_path, "quantize.exe"),
        ]
    else:
        quantize_candidates = [
            r"i:\whisper-acft\whisper.cpp\build\bin\Release\quantize.exe",
            r"whisper.cpp\build\bin\Release\quantize.exe",
            r"whisper.cpp\build\bin\quantize.exe",
        ]
    
    quantize_path = None
    for q in quantize_candidates:
        if os.path.exists(q):
            quantize_path = q
            break
    
    if not quantize_path:
        print(f"Warning: quantize tool not found. Skipping quantization.")
        print("To build quantize tool, run: cmake --build whisper.cpp/build --config Release")
        return
    
    # Create quantized output filename
    base_dir = os.path.dirname(ggml_file)
    quantized_file = os.path.join(base_dir, f"ggml-model-{qtype}.bin")
    
    # Run quantization
    cmd = [quantize_path, ggml_file, quantized_file, qtype]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Quantized model saved to {quantized_file}")

def convert_with_ctranslate2(model_dir: str, output_dir: str):
    """Convert to CTranslate2 format (alternative to GGML)."""
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
    parser.add_argument(
        "--qtype",
        type=str,
        default="q8_0",
        help="Quantization type for GGUF (e.g., q8_0, q5_1, q4_0, f16). Set to '' to keep full precision.",
    )
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
    convert_hf_to_gguf(model_dir, output_dir, args.whisper_cpp, args.qtype)
    
    print("\nConversion complete!")

if __name__ == "__main__":
    main()
