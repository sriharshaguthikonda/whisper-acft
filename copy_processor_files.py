import os
import shutil
from transformers import WhisperProcessor

TMP_DIR = r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx\processor_tmp"
TARGETS = [
    r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx\model_epoch_000001",
    r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx\model_epoch_000006",
    r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx\model_epoch_000007",
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
            if os.path.exists(src):
                shutil.copy(src, dst)
                print(f"Copied {fname} -> {tgt}")
            else:
                print(f"Missing in source: {fname}")

if __name__ == "__main__":
    main()
