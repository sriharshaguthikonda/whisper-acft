import os

BASE_DIR = r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx"
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
    for tgt in TARGETS:
        for fname in FILES:
            path = os.path.join(tgt, fname)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Deleted {fname} from {tgt}")
                except OSError as e:
                    print(f"Failed to delete {fname} from {tgt}: {e}")
            else:
                print(f"Skip (missing): {fname} in {tgt}")


if __name__ == "__main__":
    main()
