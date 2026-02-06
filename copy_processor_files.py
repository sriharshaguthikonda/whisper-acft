import os
import shutil
from transformers import WhisperProcessor

BASE_DIR = r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx"
TMP_DIR = os.path.join(BASE_DIR, "processor_tmp")
TARGETS = [
    os.path.join(BASE_DIR, d)
    for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d)) and d.startswith("model_epoch_")
]
FILES = [
    "preprocessor_config.json",
    "feature_extractor.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def main():
    os.makedirs(TMP_DIR, exist_ok=True)
    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    proc.save_pretrained(TMP_DIR)

    for tgt in TARGETS:
        os.makedirs(tgt, exist_ok=True)
        for fname in FILES:
            src = os.path.join(TMP_DIR, fname)
            dst = os.path.join(tgt, fname)

            # skip if target already has the file
            if os.path.exists(dst):
                print(f"Skip (exists): {fname} in {tgt}")
                continue

            if os.path.exists(src):
                shutil.copy2(src, dst)
                print(f"Copied {fname} -> {tgt}")
            else:
                print(f"Missing in source: {fname}")


if __name__ == "__main__":
    main()
