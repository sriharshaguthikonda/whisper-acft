import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from pyannote.audio import Inference, Model
from sklearn.cluster import AgglomerativeClustering


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster speakers across files using diarization summary and speaker embeddings. "
            "Produces a JSON mapping local speakers to global speaker IDs."
        )
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Path to diarization summary JSON (output of diarize_recordings.py).",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        required=True,
        help="Root folder containing the referenced wav files.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path to write clustered speaker JSON.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN"),
        help="Hugging Face token with access to pyannote/embedding.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.25,
        help="Cosine distance threshold for clustering (lower merges fewer). Ignored if --n-speakers is set.",
    )
    parser.add_argument(
        "--n-speakers",
        type=int,
        default=None,
        help="Optional fixed number of global speakers. If set, overrides distance threshold.",
    )
    parser.add_argument(
        "--max-segments-per-speaker",
        type=int,
        default=10,
        help="Cap segments used per local speaker to limit runtime; sampled earliest segments.",
    )
    parser.add_argument(
        "--min-segment-duration",
        type=float,
        default=0.5,
        help="Skip very short segments (seconds) when computing embeddings.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_local_embeddings(
    files: List[Dict],
    audio_root: Path,
    inference: Inference,
    max_segments: int,
    min_duration: float,
) -> Tuple[List[np.ndarray], List[Tuple[str, str]], Dict[Tuple[str, str], float]]:
    """
    Returns:
        embeddings: list of vectors
        keys: list of (file_name, local_speaker)
        speech_durations: total speech seconds per key
    """
    embeddings: List[np.ndarray] = []
    keys: List[Tuple[str, str]] = []
    speech_durations: Dict[Tuple[str, str], float] = {}

    for file_entry in files:
        file_path = audio_root / file_entry["file_name"]
        segments_by_spk: Dict[str, List[Dict]] = defaultdict(list)
        for seg in file_entry["segments"]:
            duration = seg["duration"]
            if duration < min_duration:
                continue
            segments_by_spk[seg["speaker"]].append(seg)

        for local_spk, segs in segments_by_spk.items():
            segs = sorted(segs, key=lambda s: s["start"])[:max_segments]
            vecs = []
            total_speech = 0.0
            for seg in segs:
                total_speech += seg["duration"]
                emb = inference.crop(str(file_path), (seg["start"], seg["end"]))
                vecs.append(emb)
            if not vecs:
                continue
            mean_vec = np.mean(np.stack(vecs), axis=0)
            embeddings.append(mean_vec)
            key = (file_entry["file_name"], local_spk)
            keys.append(key)
            speech_durations[key] = total_speech

    return embeddings, keys, speech_durations


def cluster_embeddings(
    embeddings: List[np.ndarray],
    distance_threshold: float,
    n_speakers: int | None,
) -> np.ndarray:
    if not embeddings:
        raise ValueError("No embeddings to cluster.")
    X = np.stack(embeddings)
    if n_speakers is not None:
        clustering = AgglomerativeClustering(
            n_clusters=n_speakers,
            metric="cosine",
            linkage="average",
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
    return clustering.fit_predict(X)


def build_output(
    labels: np.ndarray,
    keys: List[Tuple[str, str]],
    speech_durations: Dict[Tuple[str, str], float],
) -> Dict:
    # Map global cluster id to members
    global_map: Dict[int, Dict] = defaultdict(lambda: {"members": [], "total_speech": 0.0})

    for label, key in zip(labels, keys):
        file_name, local_spk = key
        duration = speech_durations.get(key, 0.0)
        global_map[label]["members"].append(
            {
                "file_name": file_name,
                "local_speaker": local_spk,
                "speech_seconds": round(duration, 3),
            }
        )
        global_map[label]["total_speech"] += duration

    global_speakers = []
    for idx, (label, data) in enumerate(sorted(global_map.items(), key=lambda x: -x[1]["total_speech"])):
        global_id = f"G{idx:02d}"
        global_speakers.append(
            {
                "global_speaker": global_id,
                "total_speech_seconds": round(data["total_speech"], 3),
                "members": data["members"],
            }
        )

    # Per-file view
    per_file: Dict[str, List[Dict]] = defaultdict(list)
    for label, key in zip(labels, keys):
        file_name, local_spk = key
        global_idx = np.where(np.array(labels) == label)[0][0]  # first occurrence rank
        global_id = f"G{global_idx:02d}"
        per_file[file_name].append({"local_speaker": local_spk, "global_speaker": global_id})

    return {
        "global_speakers": global_speakers,
        "files": per_file,
        "notes": "Global IDs are derived from clustering pyannote/embedding vectors averaged per local speaker.",
    }


def main() -> None:
    args = parse_args()
    if not args.hf_token:
        raise SystemExit("Hugging Face token required for pyannote/embedding. Set --hf-token or env var.")

    summary = load_summary(args.summary_json)
    files = summary.get("files", [])
    if not files:
        raise SystemExit("No files found in summary JSON.")

    model = Model.from_pretrained("pyannote/embedding", use_auth_token=args.hf_token)
    inference = Inference(model, window="whole")

    embeddings, keys, speech_durations = compute_local_embeddings(
        files,
        audio_root=args.audio_root,
        inference=inference,
        max_segments=args.max_segments_per_speaker,
        min_duration=args.min_segment_duration,
    )

    labels = cluster_embeddings(
        embeddings,
        distance_threshold=args.distance_threshold,
        n_speakers=args.n_speakers,
    )

    output = build_output(labels, keys, speech_durations)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote clustered speaker mapping to {args.output_json}")


if __name__ == "__main__":
    main()
