import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

from tqdm import tqdm


def extract_segments(payload: Mapping) -> Sequence[Mapping]:
    """
    Find Whisper/Groq verbose_json segments.

    Supported shapes (best-effort):
      {"groq_response": {"segments": [...]}}
      {"segments": [...]}
      {"response": {"segments": [...]}}
      {"result": {"segments": [...]}}
      {"data": {"segments": [...]}}

    Returns:
      sequence of segment mappings with at least start/end/text when valid.
    """
    response: MutableMapping = payload.get("groq_response") or {}
    segments = response.get("segments")
    if segments:
        return segments  # type: ignore[return-value]

    segments = payload.get("segments")
    if segments:
        return segments  # type: ignore[return-value]

    for key in ("response", "result", "data"):
        nested = payload.get(key)
        if isinstance(nested, Mapping) and nested.get("segments"):
            return nested.get("segments")  # type: ignore[return-value]

    raise ValueError("No segments found in JSON (expected segments / groq_response.segments).")


def normalize_text(text: str) -> str:
    """Trim and collapse whitespace to make comparisons consistent."""
    return " ".join((text or "").strip().split())


def process_file(path: Path) -> Tuple[str, Counter, Optional[str]]:
    """
    Read a single transcript JSON and return a Counter of normalized segment texts.
    Returns (path, counter, error_message_if_any).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - want to capture any decode issue
        return (str(path), Counter(), f"Failed to parse JSON: {exc}")

    try:
        segments = extract_segments(data)
    except Exception as exc:  # noqa: BLE001 - allow per-file errors without stopping
        return (str(path), Counter(), f"Failed to extract segments: {exc}")

    counter: Counter = Counter()
    for seg in segments:
        text = normalize_text(seg.get("text", ""))
        if text:
            counter[text] += 1
    return (str(path), counter, None)


def load_state(state_path: Path) -> Tuple[Counter, set]:
    if not state_path.exists():
        return Counter(), set()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    counts = Counter(payload.get("counts", {}))
    processed = set(payload.get("processed_files", []))
    return counts, processed


def save_state(state_path: Path, counts: Counter, processed: Iterable[str]) -> None:
    state_path.write_text(
        json.dumps(
            {
                "counts": counts,
                "processed_files": list(processed),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def run(
    input_dir: Path,
    state_file: Optional[Path],
    top_k: int,
    min_count: int,
    workers: Optional[int],
    output_json: Optional[Path],
) -> None:
    counts, processed = load_state(state_file) if state_file else (Counter(), set())

    files = sorted(input_dir.glob("*.json"))
    remaining = [p for p in files if str(p) not in processed]
    if not remaining:
        print("No new JSON files to process.")
        return

    progress = tqdm(total=len(remaining), desc="Scanning transcripts", unit="file")
    warn_count = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_file, path): path for path in remaining}
        for future in as_completed(futures):
            path_str, counter, err = future.result()
            if err:
                tqdm.write(f"[warn] {path_str}: {err}")
                warn_count += 1
            else:
                counts.update(counter)
                processed.add(path_str)
                if state_file:
                    save_state(state_file, counts, processed)
            progress.update(1)
    progress.close()

    sorted_items = counts.most_common()
    if output_json:
        output_json.write_text(
            json.dumps(sorted_items, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"\nTop segments (warnings: {warn_count}):")
    for idx, (text, cnt) in enumerate(sorted_items[:top_k], start=1):
        if cnt < min_count:
            continue
        print(f"{idx:>3}. [{cnt}]: {text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Find most common transcript segments across JSON files."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing transcript JSON files.",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Optional path to save progress for resuming.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of most common segments to display.",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=1,
        help="Only show segments appearing at least this many times.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of processes (0 = auto).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write full frequency list as JSON.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    workers = args.workers or None
    run(
        input_dir=args.input_dir,
        state_file=args.state_file,
        top_k=args.top_k,
        min_count=args.min_count,
        workers=workers or None,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
