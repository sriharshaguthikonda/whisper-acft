import os, json, pathlib, subprocess, hashlib, wave, threading
import argparse
import numpy as np
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Try to use orjson for faster JSON parsing if available
try:
    import orjson
    HAS_ORJSON = True
except ImportError:
    HAS_ORJSON = False
    orjson = None

# ----------------------------
# USER SETTINGS (CLI overrides)
# ----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: build pending manifest/tasks from transcripts")
    p.add_argument("--acft-model-id", default="futo-org/acft-whisper-small.en", help="Model id for metadata")
    p.add_argument("--base-processor-id", default="openai/whisper-small.en", help="Tokenizer/processor id")
    p.add_argument("--load-model-for-debug", action="store_true", help="Load model weights for debugging")

    p.add_argument("--transcript-dir", default=r"I:\Transcriptions_patched_corrected", help="Directory of transcript JSON files")
    p.add_argument("--chunks-dir", default=r"I:\Record_chunks", help="Output chunks directory")
    p.add_argument("--audio-source-dir", default=r"I:\Record_harsha", help="Directory containing source audio files")

    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--max-out-seconds", type=float, default=30.0)

    # Training-label safety
    p.add_argument("--max-label-tokens", type=int, default=420)

    # Segment-level behaviour
    p.add_argument("--context-pad", type=float, default=0.20)
    p.add_argument("--min-seg-sec", type=float, default=2.00)
    p.add_argument("--merge-gap-for-short", type=float, default=0.5)
    p.add_argument("--merge-short-segments", action="store_true", help="Merge short segments forward")
    p.add_argument("--no-keep-tiny-segments", dest="keep_tiny_segments", action="store_false")
    p.set_defaults(keep_tiny_segments=True)

    # Segment-quality filtering (Groq / Whisper verbose metrics)
    p.add_argument("--no-apply-segment-quality-filters", dest="apply_segment_quality_filters", action="store_false")
    p.set_defaults(apply_segment_quality_filters=True)
    p.add_argument("--no-include-review-segments", dest="include_review_segments", action="store_false")
    p.set_defaults(include_review_segments=True)

    p.add_argument("--no-speech-prob-safe-max", type=float, default=0.10)
    p.add_argument("--no-speech-prob-drop-min", type=float, default=0.60)
    p.add_argument("--avg-logprob-safe-min", type=float, default=-0.30)
    p.add_argument("--avg-logprob-drop-max", type=float, default=-0.60)
    p.add_argument("--compression-ratio-safe-max", type=float, default=2.00)
    p.add_argument("--compression-ratio-drop-min", type=float, default=2.50)
    return p.parse_args()


args = parse_args()

ACFT_MODEL_ID = args.acft_model_id
BASE_PROCESSOR_ID = args.base_processor_id
LOAD_MODEL_FOR_DEBUG = args.load_model_for_debug

TRANSCRIPT_DIR = args.transcript_dir
CHUNKS_DIR = args.chunks_dir
AUDIO_SOURCE_DIR = args.audio_source_dir

TARGET_SR = int(args.target_sr)
MAX_OUT_SECONDS = float(args.max_out_seconds)
MAX_OUT_FRAMES = int(MAX_OUT_SECONDS * TARGET_SR)
DUR_CAP_SEC = (MAX_OUT_FRAMES - 1) / float(TARGET_SR)  # ~29.9999s at 16k

# Training-label safety
MAX_LABEL_TOKENS = int(args.max_label_tokens)

# Segment-level behaviour
CONTEXT_PAD = float(args.context_pad)            # small padding (0.05–0.20 recommended)
MIN_SEG_SEC = float(args.min_seg_sec)            # if a segment is shorter than this, merge forward
MERGE_GAP_FOR_SHORT = float(args.merge_gap_for_short)    # only merge short segments if the gap is <= this
MERGE_SHORT_SEGMENTS = bool(args.merge_short_segments)  # set True to merge short segments; False keeps every segment standalone
KEEP_TINY_SEGMENTS = bool(args.keep_tiny_segments)     # keep very short segments even if under MIN_SEG_SEC

# Segment-quality filtering (Groq / Whisper verbose metrics)
APPLY_SEGMENT_QUALITY_FILTERS = bool(args.apply_segment_quality_filters)
INCLUDE_REVIEW_SEGMENTS = bool(args.include_review_segments)

NO_SPEECH_PROB_SAFE_MAX = float(args.no_speech_prob_safe_max)
NO_SPEECH_PROB_DROP_MIN = float(args.no_speech_prob_drop_min)

AVG_LOGPROB_SAFE_MIN = float(args.avg_logprob_safe_min)
AVG_LOGPROB_DROP_MAX = float(args.avg_logprob_drop_max)

COMPRESSION_RATIO_SAFE_MAX = float(args.compression_ratio_safe_max)
COMPRESSION_RATIO_DROP_MIN = float(args.compression_ratio_drop_min)

# Output files
CHUNKS_DIR_P = pathlib.Path(CHUNKS_DIR)
CHUNKS_DIR_P.mkdir(parents=True, exist_ok=True)

AUDIO_SOURCE_DIR_P = pathlib.Path(AUDIO_SOURCE_DIR)
if not AUDIO_SOURCE_DIR_P.exists():
    raise FileNotFoundError(f"AUDIO_SOURCE_DIR does not exist: {AUDIO_SOURCE_DIR_P}")

MANIFEST_PATH = str(CHUNKS_DIR_P / "pairs_manifest.jsonl")
PENDING_PAIRS_PATH = str(CHUNKS_DIR_P / "pairs_pending.jsonl")
PENDING_TASKS_PATH = str(CHUNKS_DIR_P / "tasks_pending.jsonl")

# Use tokenizer only to estimate label token lengths (avoid WhisperProcessor -> torchvision dependency).
def _load_tokenizer():
    try:
        from transformers import WhisperTokenizerFast
        return WhisperTokenizerFast.from_pretrained(BASE_PROCESSOR_ID)
    except Exception:
        pass
    try:
        from transformers import WhisperTokenizer
        return WhisperTokenizer.from_pretrained(BASE_PROCESSOR_ID)
    except Exception:
        pass
    return AutoTokenizer.from_pretrained(BASE_PROCESSOR_ID)

tokenizer = _load_tokenizer()

# Stage 1 is manifest planning only. Loading model weights is wasted time.
model = None
if LOAD_MODEL_FOR_DEBUG:
    from transformers import WhisperForConditionalGeneration
    model = WhisperForConditionalGeneration.from_pretrained(ACFT_MODEL_ID)






# ----------------------------
# MANIFEST HELPERS
# ----------------------------

def load_processed_jsons_from_manifest(path: str) -> set:
    processed = set()
    if not os.path.exists(path):
        return processed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if HAS_ORJSON:
                    obj = orjson.loads(line)
                else:
                    obj = json.loads(line)
                tj = obj.get("transcript_json")
                if tj:
                    processed.add(tj)
            except (json.JSONDecodeError, ValueError if not HAS_ORJSON else orjson.JSONDecodeError):
                continue
    return processed


def load_processed_jsons_cached(manifest_path: str, cache_path: str) -> set:
    """Load processed JSONs from manifest with caching based on manifest mtime/size."""
    import json
    from pathlib import Path

    manifest_path = Path(manifest_path)
    cache_path = Path(cache_path)

    if not manifest_path.exists():
        return set()

    st = manifest_path.stat()
    if cache_path.exists():
        try:
            if HAS_ORJSON:
                cache_data = orjson.loads(cache_path.read_bytes())
            else:
                cache_data = json.loads(cache_path.read_text(encoding="utf-8"))
            if cache_data.get("mtime") == st.st_mtime and cache_data.get("size") == st.st_size:
                return set(cache_data.get("processed", []))
        except Exception:
            pass  # rebuild cache

    processed = load_processed_jsons_from_manifest(str(manifest_path))
    cache_content = {"mtime": st.st_mtime, "size": st.st_size, "processed": sorted(processed)}
    if HAS_ORJSON:
        cache_path.write_bytes(orjson.dumps(cache_content))
    else:
        cache_path.write_text(
            json.dumps(cache_content),
            encoding="utf-8"
        )
    return processed


def write_jsonl_overwrite(path: str, rows: list) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    # JSONL/NDJSON MUST be 1 JSON object per line (no pretty-print / multi-line objects).
    # Also: OPT_APPEND_NEWLINE already adds a newline, so don't add another one.
    if HAS_ORJSON:
        with open(path, "wb") as f:
            for obj in rows:
                f.write(orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE))
    else:
        with open(path, "w", encoding="utf-8") as f:
            for obj in rows:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_manifest_jsonl(path: str, rows: list) -> None:
    if not rows:
        return
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    if HAS_ORJSON:
        with open(path, "ab") as f:
            for obj in rows:
                f.write(orjson.dumps(obj, option=orjson.OPT_APPEND_NEWLINE))
    else:
        with open(path, "a", encoding="utf-8") as f:
            for obj in rows:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                if HAS_ORJSON:
                    rows.append(orjson.loads(line))
                else:
                    rows.append(json.loads(line))
            except (json.JSONDecodeError, ValueError if not HAS_ORJSON else orjson.JSONDecodeError):
                continue
    return rows






# ----------------------------
# FAST HELPERS
# ----------------------------

_ffprobe_cache = {}
_ffprobe_lock = threading.Lock()

def ffprobe_duration_sec(path: str) -> float:
    with _ffprobe_lock:
        if path in _ffprobe_cache:
            return _ffprobe_cache[path]
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd).decode("utf-8").strip()
    dur = float(out)
    with _ffprobe_lock:
        _ffprobe_cache[path] = dur
    return dur


def safe_id_from_path(p: str, n: int = 10) -> str:
    return hashlib.sha1(p.encode("utf-8")).hexdigest()[:n]


def chunk_filename(audio_path: str, transcript_json_path: str, chunk_index: int) -> str:
    base = pathlib.Path(audio_path).stem
    jid = safe_id_from_path(transcript_json_path, n=10)
    return f"{base}__{jid}_chunk{chunk_index:04d}.wav"


def is_useless_segment(seg: dict) -> bool:
    txt = (seg.get("text") or "").strip()
    if txt in {"", ".", "…"}:
        return True

    if APPLY_SEGMENT_QUALITY_FILTERS:
        tier, _reasons, _metrics = assess_quality_from_segment(seg)
        if tier == "drop":
            return True
        if tier == "review" and not INCLUDE_REVIEW_SEGMENTS:
            return True

    s = seg.get("start", None)
    e = seg.get("end", None)
    if s is None or e is None:
        return True
    if float(e) <= float(s):
        return True

    return False


def assess_quality_from_metrics(
    no_speech_prob: float | None,
    avg_logprob: float | None,
    compression_ratio: float | None,
) -> tuple[str, list[str], dict]:
    """Return (tier, reasons, metrics).

    tier: "safe" | "review" | "drop"
    Conservative: any hard-drop => drop.
    """

    metrics = {
        "no_speech_prob": None if no_speech_prob is None else float(no_speech_prob),
        "avg_logprob": None if avg_logprob is None else float(avg_logprob),
        "compression_ratio": None if compression_ratio is None else float(compression_ratio),
    }

    reasons: list[str] = []

    # Hard-drop checks
    nsp = metrics["no_speech_prob"]
    if nsp is not None and nsp >= NO_SPEECH_PROB_DROP_MIN:
        reasons.append(f"no_speech_prob>={NO_SPEECH_PROB_DROP_MIN}")

    alp = metrics["avg_logprob"]
    if alp is not None and alp < AVG_LOGPROB_DROP_MAX:
        reasons.append(f"avg_logprob<{AVG_LOGPROB_DROP_MAX}")

    cr = metrics["compression_ratio"]
    if cr is not None and cr > COMPRESSION_RATIO_DROP_MIN:
        reasons.append(f"compression_ratio>{COMPRESSION_RATIO_DROP_MIN}")

    if reasons:
        return "drop", reasons, metrics

    # Review checks
    review_reasons: list[str] = []
    if nsp is not None and nsp >= NO_SPEECH_PROB_SAFE_MAX:
        review_reasons.append(f"no_speech_prob>={NO_SPEECH_PROB_SAFE_MAX}")
    if alp is not None and alp <= AVG_LOGPROB_SAFE_MIN:
        review_reasons.append(f"avg_logprob<={AVG_LOGPROB_SAFE_MIN}")
    if cr is not None and cr >= COMPRESSION_RATIO_SAFE_MAX:
        review_reasons.append(f"compression_ratio>={COMPRESSION_RATIO_SAFE_MAX}")

    if review_reasons:
        return "review", review_reasons, metrics
    return "safe", [], metrics


def assess_quality_from_segment(seg: dict) -> tuple[str, list[str], dict]:
    """Assess quality using segment fields when present."""
    return assess_quality_from_metrics(
        seg.get("no_speech_prob", None),
        seg.get("avg_logprob", None),
        seg.get("compression_ratio", None),
    )


POSSIBLE_EXTENSIONS = ['.m4a', '.wav', '.mp3', '.flac', '.ogg']


# NEW FUNCTION: resolve_audio_path
def resolve_audio_path(audio_path_from_json: str, fallback_dir: pathlib.Path | None = None) -> str | None:
    # Try the path exactly as it is given
    if os.path.exists(audio_path_from_json):
        return audio_path_from_json

    dir_name = os.path.dirname(audio_path_from_json)
    base_name_stem = os.path.splitext(os.path.basename(audio_path_from_json))[0]

    # Fallback: look in provided fallback_dir by basename
    if fallback_dir is not None and fallback_dir.exists():
        original_ext = os.path.splitext(audio_path_from_json)[1].lower()
        candidate_exts = [original_ext] + [ext for ext in POSSIBLE_EXTENSIONS if ext != original_ext]
        for ext in candidate_exts:
            candidate = fallback_dir / f"{base_name_stem}{ext}"
            if candidate.exists():
                return str(candidate)

    # If the original directory exists, try to match case-insensitively there.
    if os.path.exists(dir_name):
        for fname in os.listdir(dir_name):
            fname_stem, fname_ext = os.path.splitext(fname)
            if fname_stem.lower() == base_name_stem.lower() and fname_ext.lower() in POSSIBLE_EXTENSIONS:
                full_resolved_path = os.path.join(dir_name, fname)
                if os.path.exists(full_resolved_path):
                    return full_resolved_path
    return None


def build_audio_stem_set(audio_dir: pathlib.Path) -> set:
    stems = set()
    audio_exts = {".m4a", ".wav", ".mp3", ".flac", ".ogg"}
    with os.scandir(str(audio_dir)) as entries:
        for entry in entries:
            if entry.is_file():
                suffix = pathlib.Path(entry.name).suffix.lower()
                if suffix in audio_exts:
                    stems.add(pathlib.Path(entry.name).stem.lower())
    return stems


# Core must fit into a <=30s window after padding.
MAX_CORE_ALLOWED = MAX_OUT_SECONDS - 2.0 * CONTEXT_PAD


def _estimate_segment_tokens(segments: list) -> list:
    """Token length estimate using WhisperProcessor tokenizer.

    IMPORTANT: This uses seg['text'] only.
    We do NOT use seg['tokens'] from Groq.
    """
    n = len(segments)
    seg_tok = [0] * n

    useful_idx, useful_txt = [], []
    for i, seg in enumerate(segments):
        if is_useless_segment(seg):
            continue
        t = (seg.get("text") or "").strip().lower()
        if t:
            useful_idx.append(i)
            useful_txt.append(t)

    if useful_txt:
        enc = tokenizer(useful_txt, add_special_tokens=False)
        lens = [len(ids) for ids in enc["input_ids"]]
        for i, L in zip(useful_idx, lens):
            seg_tok[i] = int(L)

    return seg_tok


def build_segment_level_chunks(segments: list) -> list:
    """Segment-level chunks.

    Rule:
    - Each chunk starts from one segment.
    - If that segment is shorter than MIN_SEG_SEC, merge forward while:
        * next segment is not useless
        * gap <= MERGE_GAP_FOR_SHORT
        * label tokens stay <= MAX_LABEL_TOKENS
        * core length stays <= MAX_CORE_ALLOWED
    - target_out_sec = core_span + 2*CONTEXT_PAD (capped to MAX_OUT_SECONDS)
    """

    n = len(segments)
    seg_tok = _estimate_segment_tokens(segments)

    chunks = []
    i = 0

    while i < n:
        seg = segments[i]
        if is_useless_segment(seg):
            i += 1
            continue

        s = float(seg["start"])
        e = float(seg["end"])
        txt = (seg.get("text") or "").strip()
        if not txt:
            i += 1
            continue

        cur_s, cur_e = s, e
        texts = [txt]
        tok_sum = seg_tok[i]

        # Track (worst) quality metrics across merged segments (conservative).
        used_idxs = [i]
        nsp_max = None
        alp_min = None
        cr_max = None

        def _accumulate_metrics(_seg: dict):
            nonlocal nsp_max, alp_min, cr_max
            _nsp = _seg.get("no_speech_prob", None)
            _alp = _seg.get("avg_logprob", None)
            _cr = _seg.get("compression_ratio", None)
            if _nsp is not None:
                _nsp = float(_nsp)
                nsp_max = _nsp if nsp_max is None else max(nsp_max, _nsp)
            if _alp is not None:
                _alp = float(_alp)
                alp_min = _alp if alp_min is None else min(alp_min, _alp)
            if _cr is not None:
                _cr = float(_cr)
                cr_max = _cr if cr_max is None else max(cr_max, _cr)

        _accumulate_metrics(seg)

        # If too short, merge forward cautiously—only when allowed.
        if MERGE_SHORT_SEGMENTS:
            while (cur_e - cur_s) < MIN_SEG_SEC and (i + 1) < n:
                nxt = segments[i + 1]
                if is_useless_segment(nxt):
                    i += 1
                    continue

                ns = float(nxt["start"])
                ne = float(nxt["end"])
                gap = ns - cur_e
                if gap > MERGE_GAP_FOR_SHORT:
                    break

                ntext = (nxt.get("text") or "").strip()
                if not ntext:
                    i += 1
                    continue

                # Safety: token cap
                ntoks = seg_tok[i + 1]
                if (tok_sum + ntoks) > MAX_LABEL_TOKENS:
                    break

                # Safety: core must fit
                if (ne - cur_s) > MAX_CORE_ALLOWED:
                    break

                # Accept merge
                texts.append(ntext)
                tok_sum += ntoks
                cur_e = ne
                used_idxs.append(i + 1)
                _accumulate_metrics(nxt)
                i += 1

        core_span = cur_e - cur_s
        if MERGE_SHORT_SEGMENTS and core_span < MIN_SEG_SEC and not KEEP_TINY_SEGMENTS:
            i += 1
            continue

        target_out = float(min(MAX_OUT_SECONDS, core_span + 2.0 * CONTEXT_PAD))

        # Compute final chunk quality
        if APPLY_SEGMENT_QUALITY_FILTERS:
            tier, reasons, _metrics = assess_quality_from_metrics(nsp_max, alp_min, cr_max)
            if tier == "drop":
                i += 1
                continue
            if tier == "review" and not INCLUDE_REVIEW_SEGMENTS:
                i += 1
                continue
        else:
            tier, reasons = None, []

        chunks.append({
            "start": float(cur_s),
            "end": float(cur_e),
            "text": " ".join([t.strip() for t in texts]).strip(),
            "target_out_sec": target_out,
            "quality_tier": tier,
            "quality_reasons": reasons,
            "no_speech_prob_max": nsp_max,
            "avg_logprob_min": alp_min,
            "compression_ratio_max": cr_max,
            "segments_used": used_idxs,
        })

        i += 1

    return chunks


def compute_centered_window(audio_total: float, core_start: float, core_end: float, target_out: float):
    core_start = float(max(0.0, core_start))
    core_end = float(min(audio_total, core_end))
    if core_end < core_start:
        core_end = core_start

    core_span = core_end - core_start
    target_out = float(min(target_out, MAX_OUT_SECONDS))

    # Must contain core + context padding if possible
    need = min(MAX_OUT_SECONDS, max(target_out, core_span + 2.0 * CONTEXT_PAD))

    mid = 0.5 * (core_start + core_end)
    start = mid - need / 2.0
    start = max(0.0, min(start, audio_total - need))
    return start, need






# ----------------------------
# MAIN: PLAN + SAVE PENDING JSONL
# ----------------------------

TRANSCRIPT_DIR_P = pathlib.Path(TRANSCRIPT_DIR)
json_files = []
with os.scandir(str(TRANSCRIPT_DIR_P)) as entries:
    for entry in entries:
        if entry.is_file() and entry.name.endswith(".json"):
            json_files.append(pathlib.Path(entry.path))
json_files.sort()

processed_jsons = load_processed_jsons_cached(MANIFEST_PATH, str(CHUNKS_DIR_P / "processed_cache.json"))
audio_stems = build_audio_stem_set(AUDIO_SOURCE_DIR_P)

# Load transcripts that were part of a previous pending plan
pending_processed_jsons_from_file = set()
if os.path.exists(PENDING_PAIRS_PATH):
    for obj in read_jsonl(PENDING_PAIRS_PATH):
        if "transcript_json" in obj:
            pending_processed_jsons_from_file.add(obj["transcript_json"])

# Combine all processed/pending transcripts
all_already_processed_jsons = processed_jsons.union(pending_processed_jsons_from_file)

new_json_files = [p for p in json_files if str(p) not in all_already_processed_jsons]

print("Found JSONs:", len(json_files))
print("Already processed (final manifest):", len(processed_jsons))
print("Already processed (pending file):", len(pending_processed_jsons_from_file))
print("Planning NEW JSONs now:", len(new_json_files))

pending_pairs = []
pending_tasks = []
bad_json = 0
bad_audio = 0
skipped_no_matching_audio = 0

for jf in tqdm(new_json_files, desc="Planning segment-level chunks"):
    jf_str = str(jf)

    try:
        if HAS_ORJSON:
            obj = orjson.loads(jf.read_bytes())
        else:
            obj = json.loads(jf.read_text(encoding="utf-8"))
        audio_path_from_json = obj["input_file"]["path"]
        segments = obj["groq_response"]["segments"]
    except Exception:
        bad_json += 1
        continue

    base_stem = pathlib.Path(audio_path_from_json).stem.lower()
    if base_stem not in audio_stems:
        skipped_no_matching_audio += 1
        continue

    # Resolve the actual audio path, considering different extensions
    resolved_audio_path = resolve_audio_path(audio_path_from_json, fallback_dir=AUDIO_SOURCE_DIR_P)

    if resolved_audio_path is None:
        bad_audio += 1
        continue

    chunks = build_segment_level_chunks(segments)
    if not chunks:
        continue

    for idx, ch in enumerate(chunks):
        out_name = chunk_filename(resolved_audio_path, jf_str, idx)
        out_wav = str(CHUNKS_DIR_P / out_name)

        target_out = float(min(ch["target_out_sec"], MAX_OUT_SECONDS))

        pending_pairs.append({
            "audio_path": out_wav,
            "raw_transcription": ch["text"],     # <--- ALWAYS text, never Groq tokens
            "source_audio": resolved_audio_path,
            "chunk_index": idx,
            "transcript_json": jf_str,
            "duration_sec_target": target_out,
            "sr": TARGET_SR,
            "model": ACFT_MODEL_ID, # Changed from MODEL to ACFT_MODEL_ID

            # Optional extra metadata (won't break downstream JSONL readers)
            "quality_tier": ch.get("quality_tier", None),
            "quality_reasons": ch.get("quality_reasons", []),
            "no_speech_prob_max": ch.get("no_speech_prob_max", None),
            "avg_logprob_min": ch.get("avg_logprob_min", None),
            "compression_ratio_max": ch.get("compression_ratio_max", None),
        })

        # Only cut if the wav doesn't already exist
        if not os.path.exists(out_wav):
            pending_tasks.append({
                "audio_path": resolved_audio_path,
                "out_wav": out_wav,
                "core_start": float(ch["start"]),
                "core_end": float(ch["end"]),
                "target_out_sec": target_out,
            })

write_jsonl_overwrite(PENDING_PAIRS_PATH, pending_pairs)
write_jsonl_overwrite(PENDING_TASKS_PATH, pending_tasks)

print("\nSaved pending files:")
print("  pending pairs:", PENDING_PAIRS_PATH, "| rows:", len(pending_pairs))
print("  pending tasks:", PENDING_TASKS_PATH, "| rows:", len(pending_tasks))
print("Bad JSON:", bad_json, "| Missing audio:", bad_audio)
