#!/usr/bin/env python3
"""
Add random noise to all audio files in a folder.

- If any of audio_dir, noise_dir, or out_dir is missing, a folder picker dialog is shown.
- Otherwise CLI args are used as-is.
"""

import argparse
import os
import random
import numpy as np
import soundfile as sf
import tkinter as tk
from tkinter import filedialog

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".opus", ".m4a", ".mp3"}


def list_audio_files(root: str) -> list[str]:
    files = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in AUDIO_EXTS:
            files.append(path)
    return sorted(files)


def load_audio(path: str, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_sr and sr != target_sr:
        import librosa  # lazy

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio.astype(np.float32), sr


def save_audio(path: str, audio: np.ndarray, sr: int) -> None:
    sf.write(path, audio, sr)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x) + 1e-12)))


def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(clean):
        reps = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, reps)
    noise = noise[: len(clean)]

    clean_rms = rms(clean)
    noise_rms = rms(noise)
    if noise_rms < 1e-6 or clean_rms < 1e-6:
        return clean

    desired_noise_rms = clean_rms / (10 ** (snr_db / 20))
    noise = noise * (desired_noise_rms / noise_rms)

    mixed = clean + noise
    peak = float(np.max(np.abs(mixed) + 1e-9))
    if peak > 1.0:
        mixed = mixed / peak
    return mixed


def pick_directory(title: str, initialdir: str | None = None) -> str | None:
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askdirectory(title=title, initialdir=initialdir)
    root.destroy()
    return path


def main(args: argparse.Namespace) -> None:
    audio_dir = args.audio_dir or pick_directory("Select CLEAN audio folder")
    noise_dir = args.noise_dir or pick_directory("Select NOISE folder")
    out_dir = args.out_dir or pick_directory("Select OUTPUT folder for noisy audio")
    if not audio_dir or not noise_dir or not out_dir:
        raise SystemExit("Selection cancelled.")

    os.makedirs(out_dir, exist_ok=True)
    clean_files = list_audio_files(audio_dir)
    noise_files = list_audio_files(noise_dir)
    if not clean_files:
        raise SystemExit("No audio files found in audio_dir")
    if not noise_files:
        raise SystemExit("No noise files found in noise_dir")

    rng = random.Random(args.seed)

    for clean_path in clean_files:
        clean, sr = load_audio(clean_path, target_sr=args.sr)
        noise_path = rng.choice(noise_files)
        noise, _ = load_audio(noise_path, target_sr=sr)
        snr_db = rng.uniform(args.min_snr, args.max_snr)

        mixed = mix_at_snr(clean, noise, snr_db)

        base = os.path.splitext(os.path.basename(clean_path))[0]
        out_path = os.path.join(out_dir, f"{base}_noisy_snr{snr_db:.1f}.wav")
        save_audio(out_path, mixed, sr)
        print(f"Saved {out_path} (noise={os.path.basename(noise_path)}, SNR={snr_db:.1f} dB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", help="Folder with clean audio")
    parser.add_argument("--noise_dir", help="Folder with noise clips")
    parser.add_argument("--out_dir", help="Where to write noisy audio")
    parser.add_argument("--min_snr", type=float, default=0.0, help="Minimum SNR (dB)")
    parser.add_argument("--max_snr", type=float, default=20.0, help="Maximum SNR (dB)")
    parser.add_argument("--sr", type=int, default=16000, help="Target sample rate; resamples if needed")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parsed = parser.parse_args()
    if parsed.max_snr < parsed.min_snr:
        parser.error("max_snr must be >= min_snr")
    main(parsed)
