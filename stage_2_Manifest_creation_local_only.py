import os, json, pathlib, subprocess, hashlib, wave, threading
import numpy as np
from tqdm.auto import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ----------------------------
# USER SETTINGS
# ----------------------------
ACFT_MODEL_ID = "futo-org/acft-whisper-tiny.en"     # weights
BASE_PROCESSOR_ID = "openai/whisper-tiny.en"        # processor (feature extractor + tokenizer)

TRANSCRIPT_DIR = "i:\\Transcriptions"
CHUNKS_DIR = "i:\\Record_chunks_bad_quality"
AUDIO_SOURCE_DIR = "i:\\Record_bad_quality"  # directory containing full-length audios to include

TARGET_SR = 16000
MAX_OUT_SECONDS = 30.0
MAX_OUT_FRAMES = int(MAX_OUT_SECONDS * TARGET_SR)
DUR_CAP_SEC = (MAX_OUT_FRAMES - 1) / float(TARGET_SR)  # ~29.9999s at 16k

# Training-label safety
MAX_LABEL_TOKENS = 420

# Segment-level behaviour
CONTEXT_PAD = 0.10            # small padding (0.05–0.20 recommended)
MIN_SEG_SEC = 0.80            # if a segment is shorter than this, merge forward
MERGE_GAP_FOR_SHORT = 0.5    # only merge short segments if the gap is <= this
MERGE_SHORT_SEGMENTS = True  # set True to merge short segments; False keeps every segment standalone
KEEP_TINY_SEGMENTS = True     # keep very short segments even if under MIN_SEG_SEC

# Groq no_speech_prob is often not reliable for filtering. Leave disabled.
# If YOUR metadata is reliable, set something like 0.95.
NO_SPEECH_PROB_DROP = None

# Output files
CHUNKS_DIR_P = pathlib.Path(CHUNKS_DIR)
CHUNKS_DIR_P.mkdir(parents=True, exist_ok=True)

AUDIO_SOURCE_DIR_P = pathlib.Path(AUDIO_SOURCE_DIR)
if not AUDIO_SOURCE_DIR_P.exists():
    raise FileNotFoundError(f"AUDIO_SOURCE_DIR does not exist: {AUDIO_SOURCE_DIR_P}")

MANIFEST_PATH = str(CHUNKS_DIR_P / "pairs_manifest_bad_quality.jsonl")
PENDING_PAIRS_PATH = str(CHUNKS_DIR_P / "pairs_pending_bad_quality.jsonl")
PENDING_TASKS_PATH = str(CHUNKS_DIR_P / "tasks_pending_bad_quality.jsonl")

# Use the *base* processor (has preprocessor_config.json etc.)
processor = WhisperProcessor.from_pretrained(BASE_PROCESSOR_ID)

# If you actually want to run inference with the ACFT weights:
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
                obj = json.loads(line)
                tj = obj.get("transcript_json")
                if tj:
                    processed.add(tj)
            except json.JSONDecodeError:
                continue
    return processed


def write_jsonl_overwrite(path: str, rows: list) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def append_manifest_jsonl(path: str, rows: list) -> None:
    if not rows:
        return
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
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
            rows.append(json.loads(line))
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

    if NO_SPEECH_PROB_DROP is not None:
        nsp = seg.get("no_speech_prob", None)
        if nsp is not None and float(nsp) >= float(NO_SPEECH_PROB_DROP):
            return True

    s = seg.get("start", None)
    e = seg.get("end", None)
    if s is None or e is None:
        return True
    if float(e) <= float(s):
        return True

    return False


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
    for entry in audio_dir.glob("*"):
        if entry.is_file() and entry.suffix.lower() in audio_exts:
            stems.add(entry.stem.lower())
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
        enc = processor.tokenizer(useful_txt, add_special_tokens=False)
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
                i += 1

        core_span = cur_e - cur_s
        if MERGE_SHORT_SEGMENTS and core_span < MIN_SEG_SEC and not KEEP_TINY_SEGMENTS:
            i += 1
            continue

        target_out = float(min(MAX_OUT_SECONDS, core_span + 2.0 * CONTEXT_PAD))

        chunks.append({
            "start": float(cur_s),
            "end": float(cur_e),
            "text": " ".join([t.strip() for t in texts]).strip(),
            "target_out_sec": target_out,
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
json_files = sorted(TRANSCRIPT_DIR_P.glob("*.json"))

processed_jsons = load_processed_jsons_from_manifest(MANIFEST_PATH)
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
