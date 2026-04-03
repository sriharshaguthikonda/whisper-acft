from __future__ import annotations

import hashlib
import os
import os.path
import random
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# --- Path normalisation (Windows-friendly) ---

def canonicalise_path(p: str) -> str:
    """Return a stable, comparable absolute path string.

    - normalises slashes
    - makes absolute (best-effort, does NOT require existence)
    - collapses .. and .
    - final slash normalisation
    """
    # normalise slashes first
    p = p.replace("/", "\\")

    # make absolute (best-effort, does NOT require existence)
    try:
        p_abs = str(Path(os.path.expanduser(p)).resolve(strict=False))
    except Exception:
        try:
            p_abs = str(Path(os.path.expanduser(p)).absolute())
        except Exception:
            p_abs = p

    # collapse .. and .
    try:
        p_abs = os.path.normpath(p_abs)
    except Exception:
        pass

    # final slash normalisation
    p_abs = p_abs.replace("/", "\\")
    return p_abs.lower()


# --- Stable hashing ---

def _blake2b_hex(data: bytes, digest_bytes: int = 10, person: bytes = b"acft") -> str:
    # digest_bytes 10 => 20 hex chars; tweak if you want shorter/longer
    h = hashlib.blake2b(data, digest_size=digest_bytes, person=person)
    return h.hexdigest()


def stable_hash_hex(s: str, digest_bytes: int = 10, person: str = "acft") -> str:
    return _blake2b_hex(s.encode("utf-8", errors="ignore"), digest_bytes=digest_bytes, person=person.encode("ascii", errors="ignore"))


def stable_hash_int(s: str, person: str = "acft") -> int:
    # 8 bytes -> 64-bit integer
    hx = _blake2b_hex(s.encode("utf-8", errors="ignore"), digest_bytes=8, person=person.encode("ascii", errors="ignore"))
    return int(hx, 16)


# --- Kaggle-safe filename helpers ---

_KAGGLE_UNSAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")
_KAGGLE_MULTI_UNDERSCORE_RE = re.compile(r"_+")
_KAGGLE_MULTI_DASH_RE = re.compile(r"-+")
_KAGGLE_MULTI_DOT_RE = re.compile(r"\.+")


def _truncate_with_hash(text: str, max_len: int, person: str) -> str:
    """Deterministically truncate long strings while preserving uniqueness."""
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    suffix = stable_hash_hex(text, digest_bytes=4, person=person)
    if max_len <= len(suffix):
        return suffix[:max_len]
    head_len = max_len - len(suffix) - 1
    head = text[:head_len].rstrip("._-")
    if not head:
        return suffix[:max_len]
    return f"{head}_{suffix}"


def sanitize_kaggle_token(value: object, default: str = "x", max_len: int = 64) -> str:
    """Normalize text to a Kaggle-safe token: [A-Za-z0-9._-]."""
    default_text = "" if default is None else str(default).strip()
    allow_empty_default = default_text == ""

    text = "" if value is None else str(value).strip()
    text = _KAGGLE_UNSAFE_CHARS_RE.sub("_", text)
    text = _KAGGLE_MULTI_UNDERSCORE_RE.sub("_", text)
    text = _KAGGLE_MULTI_DASH_RE.sub("-", text)
    text = _KAGGLE_MULTI_DOT_RE.sub(".", text)
    text = text.strip("._-")

    if not text:
        text = default_text

    if max_len > 0 and text and len(text) > max_len:
        text = _truncate_with_hash(text, max_len=max_len, person="kgtok")

    text = text.strip("._-")
    if text:
        return text
    if allow_empty_default:
        return ""

    fallback = default_text.strip("._-")
    return fallback or "x"


def sanitize_kaggle_filename(stem: object, ext: str = ".wav", max_name_len: int = 120) -> str:
    """Build a Kaggle-safe filename with deterministic truncation."""
    max_name_len = max(8, int(max_name_len))
    raw_ext = (ext or ".wav").strip()
    if not raw_ext.startswith("."):
        raw_ext = "." + raw_ext
    safe_ext = "." + sanitize_kaggle_token(raw_ext[1:], default="wav", max_len=16).lower()

    stem_budget = max(1, max_name_len - len(safe_ext))
    safe_stem = sanitize_kaggle_token(stem, default="audio", max_len=stem_budget)
    safe_stem = safe_stem or "audio"

    filename = f"{safe_stem}{safe_ext}"
    if len(filename) <= max_name_len:
        return filename

    safe_stem = _truncate_with_hash(safe_stem, max_len=stem_budget, person="kgfile")
    safe_stem = safe_stem.strip("._-") or "audio"
    return f"{safe_stem}{safe_ext}"


def kaggle_safe_wav_name(parts: Iterable[object], max_name_len: int = 120) -> str:
    """Join parts into a normalized WAV filename."""
    cleaned_parts = []
    for part in parts:
        token = sanitize_kaggle_token(part, default="", max_len=48)
        if token:
            cleaned_parts.append(token)
    stem = "_".join(cleaned_parts) if cleaned_parts else "audio"
    return sanitize_kaggle_filename(stem=stem, ext=".wav", max_name_len=max_name_len)


# --- UID creation ---

def make_base_uid(
    orig_audio_path: str,
    chunk_index: int,
    core_start: float,
    core_end: float,
    extra: str = "",
    digest_bytes: int = 10,
) -> str:
    """Stable UID for the *original chunk* (never changes).

    Ingredients are chosen to be stable across re-runs and independent of any augmentation.
    """
    p = canonicalise_path(orig_audio_path)
    # float formatting: fixed precision to avoid tiny drift
    key = f"{p}|chunk{chunk_index:06d}|{core_start:.3f}-{core_end:.3f}|{extra}"
    return stable_hash_hex(key, digest_bytes=digest_bytes)


def make_aug_uid(base_uid: str, stage_name: str, copy_idx: int, extra: str = "", digest_bytes: int = 8) -> str:
    """UID for an augmented output row (unique per stage + copy)."""
    key = f"{base_uid}|{stage_name}|copy{copy_idx:03d}|{extra}"
    return stable_hash_hex(key, digest_bytes=digest_bytes)


# --- Deterministic selection + RNG ---

def should_select(base_uid: str, stage_name: str, ratio: float, buckets: int = 10_000) -> bool:
    """Deterministically select approx ratio of base_uids for this stage."""
    if ratio <= 0:
        return False
    if ratio >= 1:
        return True
    v = stable_hash_int(f"{base_uid}|{stage_name}") % buckets
    return v < int(ratio * buckets)


def rng_for(base_uid: str, stage_name: str, copy_idx: int, extra: str = "") -> random.Random:
    seed = stable_hash_int(f"{base_uid}|{stage_name}|{copy_idx}|{extra}")
    return random.Random(seed)


# --- Idempotency: persistent 'seen' index ---

@dataclass
class SQLiteSeenSet:
    """Persistent set of string keys backed by SQLite.

    Thread-safe (single connection + lock) and resume-safe.

    Schema:
      CREATE TABLE IF NOT EXISTS seen (k TEXT PRIMARY KEY);

    Notes:
    - We keep a small in-process cache of keys added in this run to avoid
      extra SELECT round-trips for common patterns.
    - For correctness we still consult SQLite when key is not in cache.
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # One shared connection is fine *if* we guard all access with a lock.
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=60,  # Increased from 30 to 60 seconds
            check_same_thread=False,  # <- critical for ThreadPoolExecutor
        )

        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=NORMAL;")
            cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA busy_timeout=30000;")  # Increased from 5000 to 30000
            cur.execute("CREATE TABLE IF NOT EXISTS seen (k TEXT PRIMARY KEY);")
            cur.close()
            self._conn.commit()

        # Only keys added during *this* run (not all historic keys).
        self._added_cache: set[str] = set()

    def contains(self, key: str) -> bool:
        if key in self._added_cache:
            return True
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT 1 FROM seen WHERE k=? LIMIT 1;", (key,))
            row = cur.fetchone()
            cur.close()
        return row is not None

    def add(self, key: str) -> bool:
        """Add key; return True if it was newly inserted, False if already present."""
        if key in self._added_cache:
            return False
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT OR IGNORE INTO seen(k) VALUES (?);", (key,))
            inserted = cur.rowcount == 1
            cur.close()
            # Keep commits batched unless you need crash-consistency per insert.
            # (Stages usually call commit() at the end.)
        if inserted:
            self._added_cache.add(key)
        return inserted

    def commit(self) -> None:
        with self._lock:
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.commit()
            finally:
                self._conn.close()


# --- Robust idempotency helpers: stable seen-db + atomic audio writes ---

def default_seen_db(out_manifest: Path, stage_name: str) -> str:
    """Choose a stable seen-db path.

    Backwards compatible:
    - If legacy "<out_manifest>.seen.sqlite" exists, keep using it.
    - Otherwise use "<out_manifest.parent>/_seen/<stage_name>.seen.sqlite".
    """
    out_manifest = Path(out_manifest)
    legacy = Path(str(out_manifest) + ".seen.sqlite")
    new = out_manifest.parent / "_seen" / f"{stage_name}.seen.sqlite"
    if legacy.exists() and not new.exists():
        return str(legacy)
    new.parent.mkdir(parents=True, exist_ok=True)
    return str(new)


def is_valid_wav(path: str | Path, min_frames: int = 16) -> bool:
    """Fast header-level validation.

    This prevents the common failure mode where a crashed write leaves a non-zero
    file that is *not* a readable WAV, which your pipeline would otherwise treat
    as already done.
    """
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return (info.frames is not None) and (int(info.frames) >= int(min_frames)) and (int(info.samplerate) > 0)
    except Exception:
        return False


def safe_unlink(path: str | Path) -> None:
    try:
        Path(path).unlink()
    except FileNotFoundError:
        return
    except Exception:
        return


def atomic_write_wav_pcm16(path: Path, audio, sr: int, subtype: str = "PCM_16", min_frames: int = 16) -> None:
    """Atomic WAV write: tmp -> validate -> os.replace.

    - tmp file is in the same directory as the final file, so replacement is atomic
      (same filesystem).
    - validates tmp via soundfile header before replacing.
    """
    import soundfile as sf
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}.wav")
    safe_unlink(tmp)
    try:
        sf.write(str(tmp), np.asarray(audio, dtype=np.float32), int(sr), subtype=subtype)
        if not is_valid_wav(tmp, min_frames=min_frames):
            raise RuntimeError(f"atomic_write_wav_pcm16 produced invalid wav: {tmp}")
        os.replace(str(tmp), str(path))
    finally:
        # If anything failed before replace, clean up tmp.
        safe_unlink(tmp)


def safe_beep() -> None:
    """Notify completion. Works on Windows; harmless elsewhere."""
    try:
        import winsound  # type: ignore

        winsound.Beep(880, 250)
        winsound.Beep(988, 250)
    except Exception:
        print("\a", end="", flush=True)
