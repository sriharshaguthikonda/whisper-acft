import json
import os
import random
import time
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Sequence
from pathlib import Path

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











DEFAULT_MANIFEST = r"i:\P2GPT_google_drive\My Drive\Record_chunks\pairs_manifest.jsonl"
DEFAULT_EXCLUDE = r"i:\P2GPT_google_drive\My Drive\Record_chunks\trained_files.jsonl"
DEFAULT_CHECKPOINTS = [
    os.path.join(
        r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx", d
    )
    for d in os.listdir(r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx")
    if os.path.isdir(os.path.join(r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx", d))
    and d.startswith("model_epoch_")
]
DEFAULT_SAMPLE_SIZE = 64
DEFAULT_SEED = 17
DEFAULT_DEVICE = "cuda"
DEFAULT_CER = True
DEFAULT_BATCH_SIZE = 16
DEFAULT_DEVICES = ["cuda"]  # e.g., ["cuda:0", "cuda:1"] to spread checkpoints across GPUs
USE_MULTIPROCESS = False  # set True to parallelize checkpoints across devices
USE_TORCH_COMPILE = False  # set True to use torch.compile for faster inference (requires PyTorch 2.x)
USE_SDPA = True  # use scaled dot product attention (flash attention) if available
USE_SELECTION_GUI = False  # if True, choose audio files and transcription folder at runtime
EVAL_BASELINE = True  # also evaluate a baseline hub model
BASELINE_MODEL = "openai/whisper-tiny"
PAIRS_MANIFEST = DEFAULT_MANIFEST  # exclude anything listed in pairs_manifest.jsonl
DEFAULT_AUDIO_DIR = r"I:\Record"
DEFAULT_TRANSCRIPT_DIR = r"I:\P2GPT_google_drive\My Drive\Transcriptions"
PROCESSOR_SOURCE = r"i:\P2GPT_google_drive\My Drive\checkpoints_partialctx\model_epoch_000001"



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
                base = os.path.basename(audio)
                exclude.add(base)
                exclude.add(Path(base).stem.lower())
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
                    base = os.path.basename(val)
                    ex.add(base)
                    ex.add(Path(base).stem.lower())
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
        base_full = os.path.basename(audio)
        base_stem = Path(base_full).stem
        base_stem_lower = base_stem.lower()
        if (
            audio in exclude
            or base_full in exclude
            or base_stem in exclude
            or base_stem_lower in exclude
        ):
            print(f"Skip (excluded training/pairs): {audio}")
            continue
        json_path = os.path.join(transcript_dir, base_stem + ".json")
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
    batch_size: int = 4,
):
    processor = WhisperProcessor.from_pretrained(ckpt_path)
    attn_impl = "sdpa" if USE_SDPA else None
    model = WhisperForConditionalGeneration.from_pretrained(
        ckpt_path,
        attn_implementation=attn_impl,
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
    ).to(device)
    model.eval()
    if USE_TORCH_COMPILE:
        model = torch.compile(model, mode="reduce-overhead")

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer") if do_cer else None
    normalize = normalizer_from_processor(processor)

    preds, refs = [], []
    t_audio = 0.0
    t_wall = 0.0
    skipped = 0
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}

    for b_start in tqdm(range(0, len(samples), max(1, batch_size)), desc=f"Evaluating {os.path.basename(ckpt_path)}", unit="batch"):
        b_end = min(len(samples), b_start + max(1, batch_size))
        batch = samples[b_start:b_end]
        wavs = []
        batch_refs = []
        batch_lengths = []
        sr = processor.feature_extractor.sampling_rate
        target_len = int(sr * 30)  # pad/extend short clips to 30s to reach expected mel length

        for sample in batch:
            try:
                if sample.audio_path in audio_cache:
                    full_wav, sr_loaded = audio_cache[sample.audio_path]
                else:
                    full_wav, sr_loaded = load_audio(
                        sample.audio_path,
                        target_sr=sr,
                        start=None,
                        end=None,
                    )
                    audio_cache[sample.audio_path] = (full_wav, sr_loaded)
                wav = full_wav
                if sample.start is not None or sample.end is not None:
                    s_idx = int(sr_loaded * sample.start) if sample.start is not None else 0
                    e_idx = int(sr_loaded * sample.end) if sample.end is not None else len(wav)
                    s_idx = max(0, s_idx)
                    e_idx = min(len(wav), e_idx)
                    wav = wav[s_idx:e_idx]
                if len(wav) < target_len:
                    wav = np.pad(wav, (0, target_len - len(wav)), mode="constant")
                wavs.append(wav)
                batch_refs.append(sample.text)
                batch_lengths.append(len(wav) / sr_loaded)
            except Exception as e:
                print(f"Skip {sample.audio_path}: {e}")
                skipped += 1

        if not wavs:
            continue

        t0 = time.time()
        inputs = processor(
            wavs,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
            return_attention_mask=True,
        ).to(device)
        with torch.no_grad(), torch.amp.autocast(device_type="cuda" if device.startswith("cuda") else "cpu"):
            generated = model.generate(
                **inputs,
                num_beams=1,
                do_sample=False,
                max_new_tokens=256,
            )
        t1 = time.time()
        texts = processor.batch_decode(generated, skip_special_tokens=True)

        for text, ref, dur in zip(texts, batch_refs, batch_lengths):
            preds.append(safe_normalize(normalize, text))
            refs.append(safe_normalize(normalize, ref))
            t_audio += dur
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



def run_checkpoint_loop(ckpt_dirs: Sequence[str], subset: Sequence[Sample], device: str, do_cer: bool, batch_size: int):
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
        print(f"\n>>> Evaluating: {ckpt} on {device}")
        try:
            stats = evaluate_checkpoint(ckpt, subset, device, do_cer, batch_size=batch_size)
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

    if EVAL_BASELINE:
        print(f"\n>>> Evaluating baseline: {BASELINE_MODEL} on {DEFAULT_DEVICE}")
        try:
            stats = evaluate_checkpoint(BASELINE_MODEL, subset, DEFAULT_DEVICE, do_cer, batch_size=DEFAULT_BATCH_SIZE)
            print(
                "Baseline WER: {wer:.4f} | CER: {cer} | RTF: {rtf:.3f} | samples: {n} | skipped: {sk} | wall: {wall:.1f}s".format(
                    wer=stats["wer"],
                    cer=f"{stats['cer']:.4f}" if stats["cer"] is not None else "-",
                    rtf=stats["rtf"],
                    n=stats["samples"],
                    sk=stats["skipped"],
                    wall=stats["wall_time_sec"],
                )
            )
        except Exception as e:
            print(f"Baseline eval failed: {e}")

    if USE_MULTIPROCESS and len(DEFAULT_DEVICES) > 1:
        # chunk checkpoints across devices and spawn separate processes
        print(f"Launching multiprocess evaluation across devices: {DEFAULT_DEVICES}")

        def _chunks(seq, n):
            k, m = divmod(len(seq), n)
            for i in range(n):
                start = i * k + min(i, m)
                end = (i + 1) * k + min(i + 1, m)
                yield seq[start:end]

        procs = []
        for dev, ckpt_subset in zip(DEFAULT_DEVICES, _chunks(ckpt_dirs, len(DEFAULT_DEVICES))):
            if not ckpt_subset:
                continue
            p = mp.Process(
                target=run_checkpoint_loop,
                args=(ckpt_subset, subset, dev, do_cer, DEFAULT_BATCH_SIZE),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    else:
        run_checkpoint_loop(ckpt_dirs, subset, device, do_cer, DEFAULT_BATCH_SIZE)


if __name__ == "__main__":
    main()
