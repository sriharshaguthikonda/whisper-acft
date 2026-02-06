# Whisper Model Conversion to GGML Format

This guide explains how to convert fine-tuned Whisper models from HuggingFace format to GGML format for use with whisper.cpp, including quantization options.

## Overview

The conversion process takes a fine-tuned Whisper model (in HuggingFace/safetensors format) and converts it to GGML format, which can be used with the whisper.cpp inference engine. The process also supports quantization to reduce model size while maintaining accuracy.

## Prerequisites

### Required Repositories

1. **whisper.cpp** - C++ implementation of Whisper
   ```bash
   git clone https://github.com/ggerganov/whisper.cpp
   cd whisper.cpp
   cmake -B build
   cmake --build build --config Release
   ```

2. **OpenAI Whisper** - Original Python implementation (needed for mel_filters.npz)
   ```bash
   git clone https://github.com/openai/whisper
   ```

### Required Python Packages

```bash
pip install transformers torch numpy
```

### Directory Structure

Your workspace should look like this:
```
whisper-acft/
├── convert_to_gguf.py          # Conversion script
├── model_train-tiny3_extracted/ # Your fine-tuned model
│   ├── config.json
│   ├── model.safetensors
│   ├── vocab.json
│   └── ... (other tokenizer files)
├── whisper.cpp/                 # whisper.cpp repository
│   ├── models/
│   │   └── convert-h5-to-ggml.py
│   └── build/
│       └── bin/
│           └── Release/
│               └── quantize.exe
└── whisper/                     # OpenAI whisper repository
    └── whisper/
        └── assets/
            └── mel_filters.npz
```

## Conversion Process

### Step 1: Prepare Your Model

Ensure your fine-tuned model directory contains:
- `config.json` - Model configuration
- `model.safetensors` or `pytorch_model.bin` - Model weights
- `vocab.json` - Vocabulary
- `added_tokens.json` - Additional tokens
- `tokenizer_config.json` - Tokenizer configuration
- Other tokenizer files (merges.txt, etc.)

### Step 2: Run the Conversion Script

#### Basic Conversion (with q8_0 quantization)

```bash
python convert_to_gguf.py \
    --model_dir ./model_train-tiny3_extracted \
    --output_dir ./gguf_output \
    --qtype q8_0
```

#### Full Precision (f16)

```bash
python convert_to_gguf.py \
    --model_dir ./model_train-tiny3_extracted \
    --output_dir ./gguf_output \
    --qtype f16
```

#### Different Quantization Levels

```bash
# q5_1 quantization (smaller, slightly lower quality)
python convert_to_gguf.py \
    --model_dir ./model_train-tiny3_extracted \
    --output_dir ./gguf_output \
    --qtype q5_1

# q4_0 quantization (smallest, lower quality)
python convert_to_gguf.py \
    --model_dir ./model_train-tiny3_extracted \
    --output_dir ./gguf_output \
    --qtype q4_0
```

### Step 3: Output Files

After successful conversion, you'll find in the output directory:

- `ggml-model.bin` - Full precision f16 model (~144 MB for tiny model)
- `ggml-model-q8_0.bin` - Quantized model (~41 MB for tiny model)

## Quantization Options

| Quantization Type | Size Reduction | Quality | Use Case |
|-------------------|----------------|---------|----------|
| f16 | None (baseline) | Best | Maximum accuracy needed |
| f32 | Larger | Best | Full precision |
| q8_0 | ~70% | Excellent | Recommended for most uses |
| q5_1 | ~75% | Very Good | Good balance |
| q4_0 | ~80% | Good | Smallest size, acceptable quality |

## What the Script Does

1. **Downloads tokenizer files** - Ensures all required tokenizer files are present from the base model
2. **Fixes config.json** - Updates architecture to `WhisperForConditionalGeneration` if needed
3. **Converts to GGML** - Uses `convert-h5-to-ggml.py` from whisper.cpp
4. **Quantizes model** - Applies quantization using whisper.cpp's quantize tool

## Advanced Usage

### Specify Custom Paths

```bash
python convert_to_gguf.py \
    --model_dir ./model_train-tiny3_extracted \
    --output_dir ./gguf_output \
    --whisper_cpp ./path/to/whisper.cpp \
    --base_model openai/whisper-tiny \
    --qtype q8_0
```

### Command-Line Arguments

- `--model_dir` - Path to your fine-tuned HuggingFace model directory (required)
- `--output_dir` - Output directory for GGML files (required)
- `--whisper_cpp` - Path to whisper.cpp repository (optional, auto-detected)
- `--base_model` - Base model for tokenizer files (default: openai/whisper-tiny)
- `--qtype` - Quantization type: q8_0, q5_1, q4_0, f16, f32 (default: q8_0)

## Using the Converted Model

### With whisper.cpp CLI

```bash
# Using the quantized model
./whisper.cpp/build/bin/Release/whisper-cli \
    -m ./gguf_output/ggml-model-q8_0.bin \
    -f ./audio.wav

# Using the full precision model
./whisper.cpp/build/bin/Release/whisper-cli \
    -m ./gguf_output/ggml-model.bin \
    -f ./audio.wav
```

### With whisper.cpp Server

```bash
./whisper.cpp/build/bin/Release/server \
    -m ./gguf_output/ggml-model-q8_0.bin
```

## Troubleshooting

### Issue: "whisper.cpp converter not found"

**Solution:** Ensure whisper.cpp is cloned and the path is correct:
```bash
git clone https://github.com/ggerganov/whisper.cpp
```

### Issue: "mel_filters.npz not found"

**Solution:** Clone the OpenAI whisper repository:
```bash
git clone https://github.com/openai/whisper
```

### Issue: "quantize tool not found"

**Solution:** Build whisper.cpp with CMake:
```bash
cd whisper.cpp
cmake -B build
cmake --build build --config Release
```

### Issue: "output directory already exists"

**Solution:** Remove the existing output directory:
```bash
Remove-Item -Path .\gguf_output -Recurse -Force  # PowerShell
# or
rm -rf ./gguf_output  # Bash
```

### Issue: Model architecture error

**Solution:** The script automatically fixes this, but if issues persist, manually edit `config.json`:
```json
{
  "architectures": ["WhisperForConditionalGeneration"]
}
```

## Performance Comparison

Based on whisper-tiny model:

| Model Type | Size | Inference Speed | Quality |
|------------|------|-----------------|---------|
| Original PyTorch | 151 MB | Baseline | 100% |
| GGML f16 | 77.7 MB | ~2x faster | 100% |
| GGML q8_0 | 43.5 MB | ~2.5x faster | ~99% |
| GGML q5_1 | ~35 MB | ~3x faster | ~97% |
| GGML q4_0 | ~30 MB | ~3.5x faster | ~95% |

## Notes

- **GGML vs GGUF**: This script uses GGML format (the format supported by whisper.cpp). Despite the script name mentioning GGUF, whisper.cpp currently uses GGML format.
- **Quantization**: q8_0 is recommended for most use cases as it provides excellent quality with significant size reduction.
- **Model Size**: The size reduction percentages are approximate and vary based on the model architecture.
- **Compatibility**: The converted models work with whisper.cpp and other GGML-compatible inference engines.

## Example Workflow

Complete workflow from fine-tuning to deployment:

```bash
# 1. Fine-tune your model (already done)
# Result: model_train-tiny3_extracted/

# 2. Convert to GGML with quantization
python convert_to_gguf.py \
    --model_dir ./model_train-tiny3_extracted \
    --output_dir ./ggml_output \
    --qtype q8_0

# 3. Test the converted model
./whisper.cpp/build/bin/Release/whisper-cli \
    -m ./ggml_output/ggml-model-q8_0.bin \
    -f ./test_audio.wav

# 4. Deploy to production
# Copy ggml-model-q8_0.bin to your deployment server
```

## Google Colab: End-to-End Notebook Guide

This section shows how to run the full conversion + quantization pipeline on Google Colab from a clean runtime.

### 1) Set up runtime
```python
import os, subprocess, textwrap, json, sys

!nvidia-smi  # optional: check GPU
```

### 2) Install dependencies
```python
!pip install -q transformers torch numpy ctranslate2  # torch uses Colab prebuilt CUDA
```

### 3) Clone required repos
```python
!git clone https://github.com/ggerganov/whisper.cpp
!git clone https://github.com/openai/whisper
```

### 4) Bring your fine-tuned model
Choose one:
- **Option A: Hugging Face repo** (public or with token):
```python
from huggingface_hub import snapshot_download
model_dir = "model_train-tiny3_extracted"
snapshot_download(repo_id="your-hf-username/your-whisper-model", local_dir=model_dir, token=None)
```
- **Option B: Upload ZIP then unzip**:
```python
from google.colab import files
uploaded = files.upload()  # upload your model_train-tiny3_extracted.zip
!unzip -o model_train-tiny3_extracted.zip -d model_train-tiny3_extracted
```

Ensure `model_dir` now contains `config.json`, `model.safetensors` (or `pytorch_model.bin`), tokenizer files, etc.

### 5) Download the conversion script
```python
!wget -O convert_to_gguf.py https://raw.githubusercontent.com/sriharshaguthikonda/whisper-acft/save-bin/convert_to_gguf.py
```

### 6) Run conversion (q8_0 quantized)
```python
!python convert_to_gguf.py \
  --model_dir ./model_train-tiny3_extracted \
  --output_dir ./gguf_output \
  --qtype q8_0 \
  --whisper_cpp ./whisper.cpp
```

Outputs (in `./gguf_output`):
- `ggml-model.bin` (f16)
- `ggml-model-q8_0.bin` (quantized)

### 7) Optional: download results from Colab
```python
from google.colab import files
files.download("gguf_output/ggml-model-q8_0.bin")
# or download the whole folder as a zip
!zip -r gguf_output.zip gguf_output
files.download("gguf_output.zip")
```

### 8) Optional: run quick inference with whisper.cpp (CPU)
```python
%cd whisper.cpp
!cmake -B build
!cmake --build build --config Release
!./build/bin/Release/whisper-cli -m ../gguf_output/ggml-model-q8_0.bin -f ./samples/jfk.wav
```

## Additional Resources

- [whisper.cpp GitHub](https://github.com/ggerganov/whisper.cpp)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [HuggingFace Whisper Fine-tuning Guide](https://huggingface.co/blog/fine-tune-whisper)
- [GGML Format Documentation](https://github.com/ggerganov/ggml)

## License

This conversion script follows the same license as the whisper.cpp and OpenAI Whisper projects.
