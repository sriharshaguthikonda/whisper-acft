"""
Filter audio files by similarity to a reference speaker using SpeechBrain ECAPA-TDNN.

Features:
- GPU if available, otherwise CPU.
- Multiprocessing for throughput.
- Resumable via state JSON (tracks processed files and decisions).
- Progress bars with tqdm.
- Optional deletion of non-matching files; otherwise writes report.
"""



"""
usage:
python i:\whisper-acft\filter_by_speaker.py `
  --reference "c:\path\to\reference_voice.wav" `
  --input-dir "i:\P2GPT_google_drive\My Drive\Transcriptions" `
  --state-file "i:\whisper-acft\speaker_filter_state.json" `
  --report-json "i:\whisper-acft\speaker_filter_report.json" `
  --threshold 0.80 `
  --workers 0
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
DEFAULT_THRESHOLD = 0.80


@dataclass
class Decision:
    similarity: float
    accepted: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter audio files by speaker match.")
    parser.add_argument("--reference", required=True, help="Path to reference audio of the target speaker.")
    parser.add_argument("--input-dir", required=True, help="Directory containing audio files to check.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Cosine similarity threshold for acceptance.")
    parser.add_argument("--state-file", required=True, help="Path to JSON state for resumable runs.")
    parser.add_argument("--report-json", help="Where to write a JSON report of decisions (dry-run or after delete).")
    parser.add_argument("--delete", action="store_true", help="Delete files that do not match the speaker.")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0 => CPU count).")
    return parser.parse_args()


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_embedder(device: str) -> EncoderClassifier:
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
        savedir="pretrained_models/spkrec-ecapa-voxceleb",
    )


def load_audio_embedding(encoder: EncoderClassifier, audio_path: pathlib.Path) -> torch.Tensor:
    signal, sr = torchaudio.load(audio_path)
    if sr != 16000:
        signal = torchaudio.functional.resample(signal, sr, 16000)
    with torch.no_grad():
        emb = encoder.encode_batch(signal.to(encoder.device)).squeeze(0).mean(dim=0).cpu()
    return emb


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(vec1, vec2, dim=0).item()


def discover_audio_files(root: pathlib.Path) -> List[pathlib.Path]:
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS]


def load_state(state_path: pathlib.Path) -> Dict[str, Decision]:
    if not state_path.exists():
        return {}
    with state_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: Decision(**v) for k, v in data.get("decisions", {}).items()}


def save_state(state_path: pathlib.Path, decisions: Dict[str, Decision]) -> None:
    serializable = {k: {"similarity": v.similarity, "accepted": v.accepted} for k, v in decisions.items()}
    tmp_path = state_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump({"decisions": serializable}, f, indent=2)
    tmp_path.replace(state_path)


def worker_task(audio_path: str, ref_vec: torch.Tensor) -> Tuple[str, float]:
    device = pick_device()
    encoder = load_embedder(device)
    emb = load_audio_embedding(encoder, pathlib.Path(audio_path))
    sim = cosine_similarity(ref_vec, emb)
    return audio_path, sim


def run_filter(args: argparse.Namespace) -> Dict[str, Decision]:
    input_dir = pathlib.Path(args.input_dir)
    state_path = pathlib.Path(args.state_file)
    report_path = pathlib.Path(args.report_json) if args.report_json else None

    device = pick_device()
    encoder = load_embedder(device)
    ref_vec = load_audio_embedding(encoder, pathlib.Path(args.reference))

    state = load_state(state_path)
    seen = set(state.keys())
    files = [p for p in discover_audio_files(input_dir) if str(p) not in seen]

    if files:
        workers = args.workers or multiprocessing.cpu_count()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(worker_task, str(p), ref_vec): p for p in files}
            for fut in tqdm(as_completed(futures), total=len(files), desc="Scoring", unit="file"):
                path_str, sim = fut.result()
                state[path_str] = Decision(similarity=sim, accepted=sim >= args.threshold)
                save_state(state_path, state)
    else:
        print("No new files to process.")

    if args.delete:
        rejected = [pathlib.Path(p) for p, d in state.items() if not d.accepted]
        for p in tqdm(rejected, desc="Deleting", unit="file"):
            try:
                p.unlink()
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to delete {p}: {exc}", file=sys.stderr)

    if report_path:
        report_data = {
            "threshold": args.threshold,
            "decisions": {
                p: {"similarity": d.similarity, "accepted": d.accepted} for p, d in state.items()
            },
        }
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

    return state


def main() -> None:
    args = parse_args()
    run_filter(args)


if __name__ == "__main__":
    main()
