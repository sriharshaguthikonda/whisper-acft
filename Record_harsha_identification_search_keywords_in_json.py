






"""
usage :

python c:\\Windows_software\\whisper-acft\\Record_harsha_identification_search_keywords_in_json.py --input-dir "i:\\P2GPT_google_drive\\My Drive\\Transcriptions" --state-file "i:\\P2GPT_google_drive\\My Drive\\Transcriptions\\keyword_search_state.json"

"""









import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

from tqdm import tqdm







def extract_segments(payload: Mapping) -> Sequence[Mapping]:
    """
    Best-effort extraction of Whisper/Groq verbose_json segments.
    Supports shapes:
      {"groq_response": {"segments": [...]}},
      {"segments": [...]},
      {"response": {"segments": [...]}},
      {"result": {"segments": [...]}},
      {"data": {"segments": [...]}},
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

    return []


def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def tokenize(text: str) -> List[str]:
    return [tok for tok in re.split(r"[^A-Za-z0-9-]+", text.lower()) if tok]


def levenshtein_distance(a: str, b: str, max_distance: int) -> int:
    """
    Compute Levenshtein distance with early exit when the running minimum
    exceeds max_distance (for speed).
    """
    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1

    # Ensure a is shorter
    if len(a) > len(b):
        a, b = b, a

    previous_row = list(range(len(a) + 1))
    for i, cb in enumerate(b, start=1):
        current_row = [i]
        min_row = current_row[0]
        for j, ca in enumerate(a, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (ca != cb)
            current = min(insertions, deletions, substitutions)
            current_row.append(current)
            if current < min_row:
                min_row = current
        if min_row > max_distance:
            return max_distance + 1
        previous_row = current_row

    return previous_row[-1]


def collect_text_fields(payload: Mapping | list) -> List[str]:
    """
    Collect candidate text fields from various JSON shapes.
    Handles dict- and list-shaped payloads without crashing.
    """
    texts: List[str] = []

    # If payload is a list, scan list items that are dicts for string values.
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, Mapping):
                for v in item.values():
                    if isinstance(v, str):
                        texts.append(v)
            elif isinstance(item, str):
                texts.append(item)
        return [normalize_text(t) for t in texts if t]

    # Whole text if present
    groq_resp: MutableMapping = payload.get("groq_response") or {}
    main_text = groq_resp.get("text")
    if isinstance(main_text, str):
        texts.append(main_text)

    # Segment texts
    segments = extract_segments(payload)
    for seg in segments:
        text = seg.get("text")
        if isinstance(text, str):
            texts.append(text)

    # Fallback: all string fields at top level
    for value in payload.values():
        if isinstance(value, str):
            texts.append(value)
    return [normalize_text(t) for t in texts if t]


@dataclass
class MatchResult:
    path: str
    matched_keywords: Set[str]
    sample_tokens: Dict[str, Set[str]]
    error: Optional[str] = None


def process_file(path: Path, keywords: Sequence[str], max_distance: int) -> MatchResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return MatchResult(path=str(path), matched_keywords=set(), sample_tokens={}, error=str(exc))

    texts = collect_text_fields(payload)
    if not texts:
        return MatchResult(path=str(path), matched_keywords=set(), sample_tokens={}, error="No text found")

    matched: Set[str] = set()
    samples: Dict[str, Set[str]] = {kw: set() for kw in keywords}
    for text in texts:
        for token in tokenize(text):
            for kw in keywords:
                dist = levenshtein_distance(token, kw, max_distance)
                if dist <= max_distance:
                    matched.add(kw)
                    samples[kw].add(token)
        # quick exit if all matched
        if len(matched) == len(keywords):
            break

    # Drop empty samples for cleaner output
    samples = {k: v for k, v in samples.items() if v}
    return MatchResult(path=str(path), matched_keywords=matched, sample_tokens=samples, error=None)


def save_state(state_path: Path, processed: Iterable[str], results: List[Dict]) -> None:
    state_path.write_text(
        json.dumps({"processed_files": list(processed), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_state(state_path: Path) -> Tuple[Set[str], List[Dict]]:
    if not state_path.exists():
        return set(), []
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    return set(payload.get("processed_files", [])), payload.get("results", [])


def run(
    input_dir: Path,
    keywords: Sequence[str],
    max_distance: int,
    workers: Optional[int],
    state_file: Optional[Path],
    output_json: Optional[Path],
) -> None:
    processed, results = load_state(state_file) if state_file else (set(), [])

    files = sorted(input_dir.glob("*.json"))
    remaining = [p for p in files if str(p) not in processed]
    if not remaining:
        print("No new JSON files to process.")
        return

    progress = tqdm(total=len(remaining), desc="Searching transcripts", unit="file")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_file, path, keywords, max_distance): path for path in remaining
        }
        for future in as_completed(futures):
            result = future.result()
            if result.error:
                tqdm.write(f"[warn] {result.path}: {result.error}")
            elif result.matched_keywords:
                results.append(
                    {
                        "path": result.path,
                        "matched_keywords": sorted(result.matched_keywords),
                        "sample_tokens": {k: sorted(v) for k, v in result.sample_tokens.items()},
                    }
                )
            processed.add(result.path)
            if state_file:
                save_state(state_file, processed, results)
            progress.update(1)
    progress.close()

    matched_paths = {item["path"] for item in results}
    print(f"\nMatched {len(matched_paths)} files containing any of the keywords.")
    for idx, path in enumerate(sorted(matched_paths), start=1):
        print(f"{idx:>3}. {path}")

    if output_json:
        output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Results written to {output_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuzzy search transcript JSON files for keywords (Levenshtein)."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Directory containing transcript JSON files.",
    )
    parser.add_argument(
        "--keywords",
        nargs="+",
        default=["gutikonda", "8028267"],
        help="Keywords to search for (space-separated).",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=2,
        help="Maximum Levenshtein distance allowed for a match.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of processes to use (0 = auto).",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=None,
        help="Optional path to resume progress.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write matched files and tokens.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    workers = args.workers or None
    run(
        input_dir=args.input_dir,
        keywords=[kw.lower() for kw in args.keywords],
        max_distance=args.max_distance,
        workers=workers,
        state_file=args.state_file,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
