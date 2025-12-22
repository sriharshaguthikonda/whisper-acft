import multiprocessing as mp
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf
from tqdm import tqdm


DEFAULT_SRC_DIR = Path(r"I:\Record")
DEFAULT_DST_DIR = Path(r"I:\Record_wav")
TARGET_SR = 16_000
DEFAULT_WORKERS = max(mp.cpu_count() - 1, 1)
USE_FFMPEG = True  # if ffmpeg is available, use it for faster/robust decoding

# Supported audio extensions we will attempt to convert
EXTS = {".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a"}


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio with soundfile, fallback to librosa/audioread."""
    try:
        data, sr = sf.read(path)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data, sr
    except Exception:
        import librosa  # lazy

        data, sr = librosa.load(path, sr=None, mono=True)
        return data, sr


def convert_file(src: Path, dst: Path) -> bool:
    """Convert one file to mono 16k wav. Returns True on success."""
    try:
        data, sr = load_audio(src)
        if sr != TARGET_SR:
            import librosa  # lazy

            data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        # ensure mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, data, sr)
        return True
    except Exception as e:
        print(f"Failed to convert {src}: {e}")
        return False


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def convert_with_ffmpeg(src: Path, dst: Path) -> bool:
    """Use ffmpeg to resample to 16k mono wav. Returns True on success."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        str(dst),
    ]
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        return result.returncode == 0 and dst.exists()
    except Exception as e:
        print(f"ffmpeg failed for {src}: {e}")
        return False


def iter_audio_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p


def _convert_task(args: tuple[Path, Path]) -> tuple[bool, Path]:
    src, dst = args
    if dst.exists():
        return False, src  # skipped
    ok = convert_file(src, dst)
    return ok, src


def dispatch_task(args: tuple[Path, Path, bool]) -> tuple[bool, Path]:
    src, dst, use_ffmpeg = args
    if use_ffmpeg:
        ok = convert_with_ffmpeg(src, dst)
        if ok or dst.exists():
            return ok, src
        # fall back if ffmpeg failed
    return _convert_task((src, dst))


def main(src_dir: Path = DEFAULT_SRC_DIR, dst_dir: Path = DEFAULT_DST_DIR, workers: int = DEFAULT_WORKERS):
    if not src_dir.exists():
        raise SystemExit(f"Source directory not found: {src_dir}")

    files = list(iter_audio_files(src_dir))
    if not files:
        raise SystemExit(f"No audio files found in {src_dir}")

    tasks = []
    for src in files:
        rel = src.relative_to(src_dir)
        dst = dst_dir / rel.with_suffix(".wav")
        tasks.append((src, dst))

    converted = 0
    skipped = 0
    failed = 0
    use_ffmpeg = USE_FFMPEG and has_ffmpeg()
    with mp.Pool(processes=max(1, workers)) as pool:
        tasks_with_flag = [(src, dst, use_ffmpeg) for (src, dst) in tasks]
        for ok, src in tqdm(pool.imap_unordered(dispatch_task, tasks_with_flag), total=len(tasks_with_flag), desc="Converting to wav", unit="file"):
            if ok:
                converted += 1
            else:
                # determine if it was skipped or failed by checking output existence
                rel = src.relative_to(src_dir)
                dst = dst_dir / rel.with_suffix(".wav")
                if dst.exists():
                    skipped += 1
                else:
                    failed += 1

    print(f"Done. Converted {converted}, skipped existing {skipped}, failed {failed}, total {len(tasks)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert all audio under a folder to 16k mono wav.")
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC_DIR, help="Source folder (default: I:\\Record)")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST_DIR, help="Destination folder for wavs (default: I:\\Record_wav)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel workers")
    args = parser.parse_args()

    main(src_dir=args.src, dst_dir=args.dst, workers=args.workers)
