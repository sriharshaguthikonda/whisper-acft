import json
import os
import random
import time
from dataclasses import dataclass
from typing import List, Sequence

import tkinter as tk
from tkinter import filedialog
import numpy as np
import torch
import soundfile as sf
import shutil
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
    start: float | None = None
    end: float | None = None


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


def ensure_processor_files(target_dir: str, source_dir: str) -> List[str]:
    """Copy common processor/tokenizer files from source into target if missing. Return list of created file paths."""
    if not os.path.isdir(source_dir):
        return []
    needed = [
        "preprocessor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    ]
    created: List[str] = []
    for name in needed:
        src = os.path.join(source_dir, name)
        dst = os.path.join(target_dir, name)
        if os.path.isfile(dst):
            continue
        if not os.path.isfile(src):
            continue
        try:
            shutil.copy2(src, dst)
            created.append(dst)
        except Exception as e:
            print(f"Warn: failed to copy {name} to {target_dir}: {e}")
    return created


def discover_checkpoints(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    dirs = []
    for name in os.listdir(root_dir):
        full = os.path.join(root_dir, name)
        if os.path.isdir(full) and name.startswith("model_epoch_"):
            dirs.append(full)
    return sorted(dirs)


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


def load_audio(path: str, target_sr: int = 16000, start: float | None = None, end: float | None = None):
    data = None
    sr = None
    try:
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
    except Exception:
        # Fallback for m4a/mp3/etc. using librosa/audioread
        import librosa  # lazy import

        data, sr = librosa.load(path, sr=None, mono=True)

    if sr != target_sr:
        import librosa  # lazy import to avoid mandatory dep at import time

        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if start is not None or end is not None:
        s_idx = int(sr * start) if start is not None else 0
        e_idx = int(sr * end) if end is not None else len(data)
        s_idx = max(0, s_idx)
        e_idx = min(len(data), e_idx)
        data = data[s_idx:e_idx]

    return data, sr


def load_pairs_exclude(path: str | None) -> set[str]:
    if not path or not os.path.isfile(path):
        return set()
    ex = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("audio_path", "source_audio"):
                val = normalize_path(row.get(key))
                if val:
                    ex.add(val)
                    ex.add(os.path.basename(val))
    return ex


def load_transcript_segments(json_path: str) -> List[Sample]:
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    segments = []
    groq = obj.get("groq_response") or {}
    segs = groq.get("segments") or []
    if segs:
        for seg in segs:
            text = seg.get("text")
            start = seg.get("start")
            end = seg.get("end")
            if text is None or start is None or end is None:
                continue
            segments.append(Sample(audio_path="", text=text, start=float(start), end=float(end)))
    else:
        # fallback to full text
        text = groq.get("text") or obj.get("text")
        if text:
            segments.append(Sample(audio_path="", text=text, start=None, end=None))
    return segments


def pick_audio_files_and_transcript_dir() -> tuple[List[str], str]:
    root = tk.Tk()
    root.withdraw()
    audio_files = filedialog.askopenfilenames(
        title="Select audio files for evaluation",
        filetypes=[("Audio", "*.wav *.mp3 *.m4a *.flac *.ogg *.opus"), ("All files", "*.*")],
        initialdir=DEFAULT_AUDIO_DIR,
    )
    transcript_dir = filedialog.askdirectory(title="Select folder containing transcription JSONs", initialdir=DEFAULT_TRANSCRIPT_DIR)
    root.destroy()
    return list(audio_files), transcript_dir


def build_samples_from_selection(audio_files: List[str], transcript_dir: str, exclude: set[str]) -> List[Sample]:
    samples: List[Sample] = []
    for audio in audio_files:
        base = os.path.splitext(os.path.basename(audio))[0]
        if audio in exclude or base in exclude:
            print(f"Skip (excluded training/pairs): {audio}")
            continue
        json_path = os.path.join(transcript_dir, base + ".json")
        if not os.path.isfile(json_path):
            print(f"No transcript for {audio}; skipping")
            continue
        segs = load_transcript_segments(json_path)
        if not segs:
            print(f"No segments/text in transcript {json_path}; skipping")
            continue
        for seg in segs:
            samples.append(Sample(audio_path=audio, text=seg.text, start=seg.start, end=seg.end))
    return samples


def evaluate_checkpoint(
    ckpt_path: str,
    samples: Sequence[Sample],
    device: str,
    do_cer: bool,
):
    processor = WhisperProcessor.from_pretrained(ckpt_path)
    model = WhisperForConditionalGeneration.from_pretrained(ckpt_path).to(device)
    if device.startswith("cuda"):
        model = model.half()
    model.eval()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer") if do_cer else None
    normalize = normalizer_from_processor(processor)

    preds, refs = [], []
    t_audio = 0.0
    t_wall = 0.0
    skipped = 0
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    for sample in tqdm(samples, desc=f"Evaluating {os.path.basename(ckpt_path)}", unit="utt"):
        try:
            if sample.audio_path in audio_cache:
                full_wav, sr = audio_cache[sample.audio_path]
            else:
                full_wav, sr = load_audio(
                    sample.audio_path,
                    target_sr=processor.feature_extractor.sampling_rate,
                    start=None,
                    end=None,
                )
                audio_cache[sample.audio_path] = (full_wav, sr)
            wav = full_wav
            if sample.start is not None or sample.end is not None:
                s_idx = int(sr * sample.start) if sample.start is not None else 0
                e_idx = int(sr * sample.end) if sample.end is not None else len(wav)
                s_idx = max(0, s_idx)
                e_idx = min(len(wav), e_idx)
                wav = wav[s_idx:e_idx]
        except Exception as e:
            print(f"Skip {sample.audio_path}: {e}")
            skipped += 1
            continue

        t0 = time.time()
        inputs = processor(wav, sampling_rate=sr, return_tensors="pt").to(device)
        autocast_ctx = torch.cuda.amp.autocast if device.startswith("cuda") else torch.cpu.amp.autocast
        with torch.no_grad(), autocast_ctx():
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


DEFAULT_MANIFEST = r"i:\P2GPT_google_drive\My Drive\Record_chunks\pairs_manifest.jsonl"
DEFAULT_EXCLUDE = r"i:\P2GPT_google_drive\My Drive\Record_chunks\trained_files.jsonl"
DEFAULT_CHECKPOINTS = [
    r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx\model_epoch_000001",
    r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx\model_epoch_000006",
    r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx\model_epoch_000007",
]
DEFAULT_SAMPLE_SIZE = 64
DEFAULT_SEED = 17
DEFAULT_DEVICE = "cuda"
DEFAULT_CER = True
USE_SELECTION_GUI = True  # if True, choose audio files and transcription folder at runtime
PAIRS_MANIFEST = DEFAULT_MANIFEST  # exclude anything listed in pairs_manifest.jsonl
DEFAULT_AUDIO_DIR = r"I:\Record"
DEFAULT_TRANSCRIPT_DIR = r"I:\P2GPT_google_drive\My Drive\Transcriptions"
PROCESSOR_SOURCE = r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx\model_epoch_000001"


def main():
    ckpt_dirs: List[str] = discover_checkpoints(r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx")
    if not ckpt_dirs:
        ckpt_dirs = list(DEFAULT_CHECKPOINTS)
    device = DEFAULT_DEVICE
    do_cer = DEFAULT_CER

    if USE_SELECTION_GUI:
        print("Select audio files and transcription folder...")
        audio_files, transcript_dir = pick_audio_files_and_transcript_dir()
        if not audio_files or not transcript_dir:
            raise SystemExit("Selection cancelled.")
        exclude_pairs = load_pairs_exclude(PAIRS_MANIFEST)
        samples_all = build_samples_from_selection(audio_files, transcript_dir, exclude_pairs)
        if not samples_all:
            raise SystemExit("No usable samples after exclusions.")
        # random subset capped by DEFAULT_SAMPLE_SIZE if set (>0)
        if DEFAULT_SAMPLE_SIZE > 0 and len(samples_all) > DEFAULT_SAMPLE_SIZE:
            rng = random.Random(DEFAULT_SEED)
            subset = rng.sample(samples_all, DEFAULT_SAMPLE_SIZE)
        else:
            subset = samples_all
        print(f"Selected {len(subset)} segment samples from {len(audio_files)} audio files (after exclusions and sampling)")
    else:
        manifest_path = DEFAULT_MANIFEST
        exclude_path = DEFAULT_EXCLUDE
        sample_size = DEFAULT_SAMPLE_SIZE
        seed = DEFAULT_SEED

        all_samples = load_manifest(manifest_path)
        if not all_samples:
            raise SystemExit("No usable rows found in manifest")

        exclude = load_exclude_set(exclude_path)
        if exclude:
            before = len(all_samples)
            all_samples = [s for s in all_samples if s.audio_path not in exclude and os.path.basename(s.audio_path) not in exclude]
            print(f"Excluded {before - len(all_samples)} rows present in exclude manifest (path or basename match); remaining {len(all_samples)}")

        subset = pick_subset(all_samples, sample_size, seed)
        print(f"Loaded {len(all_samples)} total rows; evaluating {len(subset)} samples")

    for ckpt in tqdm(ckpt_dirs, desc="Checkpoints", unit="ckpt"):
        if not os.path.isdir(ckpt):
            print(f"Skip (not a directory): {ckpt}")
            continue
        created_files = ensure_processor_files(ckpt, PROCESSOR_SOURCE)
        preproc = os.path.join(ckpt, "preprocessor_config.json")
        if not os.path.isfile(preproc):
            print(f"Skip (no preprocessor_config.json): {ckpt}")
            for f in created_files:
                try:
                    os.remove(f)
                except Exception:
                    pass
            continue
        print(f"\n>>> Evaluating: {ckpt}")
        try:
            stats = evaluate_checkpoint(ckpt, subset, device, do_cer)
        except OSError as e:
            print(f"Skip (failed to load): {ckpt} | err: {e}")
            for f in created_files:
                try:
                    os.remove(f)
                except Exception:
                    pass
            continue
        for f in created_files:
            try:
                os.remove(f)
            except Exception:
                pass
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
