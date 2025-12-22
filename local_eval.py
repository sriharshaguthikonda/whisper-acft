import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

try:
    import evaluate
except ImportError:  # pragma: no cover - helper hint
    raise SystemExit("Please install the 'evaluate' package: pip install evaluate")


@dataclass
class Sample:
    audio_path: str
    text: str


def normalize_path(p: str | None) -> str | None:
    if not p:
        return p
    # map Colab paths to local drive
    mapped = p.replace("/content/drive/MyDrive", r"i:\P2GPT_google_drive\My Drive")
    mapped = mapped.replace("/content/drive/My Drive", r"i:\P2GPT_google_drive\My Drive")
    return os.path.normpath(mapped)


def load_manifest(path: str) -> List[Sample]:
    samples: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = normalize_path(row.get("audio_path") or row.get("audio"))
            text = row.get("raw_transcription") or row.get("text") or row.get("transcription")
            if not audio or not text:
                continue
            samples.append(Sample(audio_path=audio, text=text))
    return samples


def load_exclude_set(path: str | None) -> set[str]:
    if not path:
        return set()
    exclude = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            audio = normalize_path(row.get("audio_path") or row.get("audio"))
            if audio:
                exclude.add(audio)
                exclude.add(os.path.basename(audio))
    return exclude


def pick_subset(samples: Sequence[Sample], k: int, seed: int) -> List[Sample]:
    rng = random.Random(seed)
    if k <= 0 or k >= len(samples):
        return list(samples)
    return rng.sample(samples, k)


def normalizer_from_processor(processor: WhisperProcessor):
    norm = getattr(processor.tokenizer, "_normalize", None)
    if norm is None:
        # fallback no-op
        return lambda x: x
    return norm


_NORM_FAILS = 0


def safe_normalize(norm_fn, text: str, verbose: bool = False):
    if text is None:
        return ""
    try:
        return norm_fn(text) if norm_fn else text
    except Exception as e:
        global _NORM_FAILS
        _NORM_FAILS += 1
        if verbose or _NORM_FAILS <= 3:
            print(f"Normalize failed; using raw text. Err: {e}")
        return text


def load_audio(path: str, target_sr: int = 16000):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != target_sr:
        import librosa  # lazy import to avoid mandatory dep at import time

        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return data, sr


def evaluate_checkpoint(
    ckpt_path: str,
    samples: Sequence[Sample],
    device: str,
    do_cer: bool,
):
    processor = WhisperProcessor.from_pretrained(ckpt_path)
    model = WhisperForConditionalGeneration.from_pretrained(ckpt_path).to(device)
    model.eval()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer") if do_cer else None
    normalize = normalizer_from_processor(processor)

    preds, refs = [], []
    t_audio = 0.0
    t_wall = 0.0
    skipped = 0

    for sample in tqdm(samples, desc=f"Evaluating {os.path.basename(ckpt_path)}", unit="utt"):
        try:
            wav, sr = load_audio(sample.audio_path, target_sr=processor.feature_extractor.sampling_rate)
        except Exception as e:
            print(f"Skip {sample.audio_path}: {e}")
            skipped += 1
            continue

        t0 = time.time()
        inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = model.generate(**inputs)
        t1 = time.time()
        text = processor.batch_decode(generated, skip_special_tokens=True)[0]

        preds.append(safe_normalize(normalize, text))
        refs.append(safe_normalize(normalize, sample.text))

        t_audio += len(wav) / sr
        t_wall += t1 - t0

    wer = wer_metric.compute(predictions=preds, references=refs) if preds else float("nan")
    cer = cer_metric.compute(predictions=preds, references=refs) if cer_metric and preds else None
    rtf = t_wall / t_audio if t_audio else float("nan")

    return {
        "wer": wer,
        "cer": cer,
        "rtf": rtf,
        "samples": len(samples),
        "wall_time_sec": t_wall,
        "audio_time_sec": t_audio,
        "skipped": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Whisper checkpoints on a manifest subset")
    parser.add_argument("--manifest", required=True, help="Path to manifest .jsonl (audio_path, raw_transcription)")
    parser.add_argument("--exclude-manifest", help="Path to manifest .jsonl whose audio paths will be excluded from eval")
    parser.add_argument("--checkpoints", nargs="+", required=True, help="One or more checkpoint directories")
    parser.add_argument("--sample-size", type=int, default=64, help="Number of samples to evaluate (random subset)")
    parser.add_argument("--seed", type=int, default=17, help="RNG seed for sampling")
    parser.add_argument("--device", default="cuda", help="torch device, e.g. cuda or cpu")
    parser.add_argument("--cer", action="store_true", help="Also compute Character Error Rate")
    args = parser.parse_args()

    all_samples = load_manifest(args.manifest)
    if not all_samples:
        raise SystemExit("No usable rows found in manifest")

    exclude = load_exclude_set(args.exclude_manifest)
    if exclude:
        before = len(all_samples)
        all_samples = [s for s in all_samples if s.audio_path not in exclude and os.path.basename(s.audio_path) not in exclude]
        print(f"Excluded {before - len(all_samples)} rows present in exclude manifest (path or basename match); remaining {len(all_samples)}")

    subset = pick_subset(all_samples, args.sample_size, args.seed)
    print(f"Loaded {len(all_samples)} total rows; evaluating {len(subset)} samples")

    for ckpt in tqdm(args.checkpoints, desc="Checkpoints", unit="ckpt"):
        if not os.path.isdir(ckpt):
            print(f"Skip (not a directory): {ckpt}")
            continue
        print(f"\n>>> Evaluating: {ckpt}")
        stats = evaluate_checkpoint(ckpt, subset, args.device, args.cer)
        print(
            "WER: {wer:.4f} | CER: {cer} | RTF: {rtf:.3f} | samples: {n} | skipped: {sk} | wall: {wall:.1f}s".format(
                wer=stats["wer"],
                cer=f"{stats['cer']:.4f}" if stats["cer"] is not None else "-",
                rtf=stats["rtf"],
                n=stats["samples"],
                sk=stats["skipped"],
                wall=stats["wall_time_sec"],
            )
        )


if __name__ == "__main__":
    main()
