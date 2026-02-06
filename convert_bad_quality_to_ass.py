#!/usr/bin/env python3
"""Convert specific transcript JSON files to ASS subtitles for bad quality recordings."""

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from tqdm import tqdm


# ---------------------------
# Timestamp formatting
# ---------------------------

def format_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp hh:mm:ss.mmm."""
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_timestamp_ass(seconds: float) -> str:
    """Convert seconds to ASS timestamp h:mm:ss.cc (centiseconds)."""
    cs = int(round(seconds * 100))
    hours, rem = divmod(cs, 360_000)
    minutes, rem = divmod(rem, 6_000)
    secs, cs = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cs:02d}"


# ---------------------------
# Segment extraction
# ---------------------------

def extract_segments(payload: Mapping) -> Sequence[Mapping]:
    """Find Whisper/Groq verbose_json segments (best-effort)."""
    response: MutableMapping = payload.get("groq_response") or {}
    segments = response.get("segments")
    if segments:
        return segments  # type: ignore[return-value]

    segments = payload.get("segments")
    if segments:
        return segments  # type: ignore[return-value]

    for k in ("response", "result", "data"):
        v = payload.get(k)
        if isinstance(v, Mapping) and v.get("segments"):
            return v.get("segments")  # type: ignore[return-value]

    raise ValueError("No segments found in JSON (expected segments / groq_response.segments).")


# ---------------------------
# Robust normalisation (percentiles)
# ---------------------------

def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def percentile(sorted_vals: List[float], p: float) -> float:
    """p in [0,100]. Linear interpolation percentile."""
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]

    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if c == f:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def lp_norm_bounds(segments: List[Mapping]) -> Tuple[float, float]:
    """Robust (lo, hi) for avg_logprob normalisation using P5..P95."""
    vals: List[float] = []
    for s in segments:
        v = s.get("avg_logprob")
        if v is None:
            continue
        try:
            vals.append(float(v))
        except Exception:
            continue

    vals.sort()

    if len(vals) < 5:
        return (-2.0, 0.0)

    lo = percentile(vals, 5)
    hi = percentile(vals, 95)
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return lo, hi


def score_1_to_100(lp: float, lo: float, hi: float) -> Tuple[int, float]:
    """Return (score 1..100, lp_norm 0..1) from avg_logprob and robust bounds."""
    lp_n = clamp((lp - lo) / (hi - lo))
    score = max(1, min(100, int(round(lp_n * 99)) + 1))
    return score, lp_n


def style_from_lp_norm(lp_n: float) -> str:
    """Map lp_norm [0..1] to VIBGYOR, where Violet is best and Red is worst."""
    styles = ["Red", "Orange", "Yellow", "Green", "Blue", "Indigo", "Violet"]
    idx = int(clamp(lp_n) * 7)
    if idx >= 7:
        idx = 6
    return styles[idx]


# ---------------------------
# Trivial/Junk detection (extendable)
# ---------------------------

def normalise_text_for_exact_match(text: str) -> str:
    """Normalise for safe matching: lowercase, remove punctuation, collapse spaces."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]+", " ", t)  # drop punctuation
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Built-in exact matches (normalised forms)
DEFAULT_TRIVIAL_PHRASES = {
    # Core acknowledgement junk
    "ok",
    "okay",
    "ok doctor",
    "okay doctor",
    "ok doc",
    "okay doc",
    "ok dr",
    "okay dr",

    "yes",
    "yeah",
    "yep",
    "ya",
    "yes doctor",
    "yeah doctor",
    "yep doctor",
    "yes doc",
    "yeah doc",
    "yes dr",
    "yeah dr",

    "no",
    "nope",
    "nah",
    "no doctor",
    "no doc",
    "no dr",

    "right",
    "alright",
    "all right",
    "sure",
    "fine",

    # From your frequency list
    "hello",
    "hello doctor",
    "hello there",
    "hello there doctor",

    "thank you",
    "thank you doctor",
    "thank you very much",

    "okay all right",
    "okay alright",
    "okay thank you",

    "yes yes",
    "yeah yeah",
    "yeah yeah yeah",
    "no no",
    "no no no",

    "nothing",
    "nothing else",
    "no nothing",
    "normal",

    "i dont know",
    "i dont know doctor",

    "i dont think so",
    "i dont think so doctor",

    "i see",
    "i understand",

    "oh",

    # single-word junk that often becomes its own cue
    "and",
    "so",
    "you",

    # station/meta cues
    "enter the room",
    "two minutes remaining",
    "move on to the next station",
    "begin",

    # repeated prompts you listed (treat as junk by same policy)
    "could you please confirm your age for me",
    "could you please confirm your name and age for me",
    "what would you like me to call you",

    "any allergies",
    "any allergies by any chance",
    "any fever",

    "do you smoke",
    "do you drink alcohol",

    "does that make sense",
    "is that okay",

    "could you tell me a little bit more",
    "what do you do for a living",
    "what do you want to know",

    "thats it",
    "thats good",
    "not exactly",

    "like what",
    "what is that",
    "what is that doctor",
    "like what doctor",
}

# Built-in regex patterns for common variants and repeats.
# These operate on NORMALISED text (punctuation removed).
DEFAULT_TRIVIAL_REGEX = [
    # mm/hmm/um/uh (optionally with doctor/doc/dr)
    r"^(m+|h+m+|um+|uh+|erm+)( (doctor|doc|dr))?$",
    r"^(mm hmm|mm hm|uh huh|uh uh)( (doctor|doc|dr))?$",

    # repeated ok/okay: "okay", "okay okay", "ok ok ok" (with optional doctor)
    r"^(dr )?(ok(ay)?)( (ok(ay)?))*$",
    r"^(dr )?(ok(ay)?)( (ok(ay)?))* (doctor|doc|dr)$",

    # repeated yes/yeah and repeated no
    r"^(yes|yeah|yep)( (yes|yeah|yep))*$",
    r"^(yes|yeah|yep)( (yes|yeah|yep))* (doctor|doc|dr)$",
    r"^no( no)*$",
    r"^no( no)* (doctor|doc|dr)$",

    # pure digits (station numbers like 8028267)
    r"^\d+$",
]


def load_trivial_patterns(file_path: Optional[Path], replace_default: bool) -> Tuple[set[str], List[re.Pattern]]:
    exact = set() if replace_default else set(DEFAULT_TRIVIAL_PHRASES)
    regex_list: List[re.Pattern] = [] if replace_default else [re.compile(p, flags=re.IGNORECASE) for p in DEFAULT_TRIVIAL_REGEX]

    if not file_path:
        return exact, regex_list

    raw = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in raw:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.lower().startswith("re:"):
            pat = s[3:].strip()
            if pat:
                regex_list.append(re.compile(pat, flags=re.IGNORECASE))
        else:
            exact.add(normalise_text_for_exact_match(s))

    return exact, regex_list


def is_trivial(text: str, exact: set[str], regex_list: List[re.Pattern]) -> bool:
    norm = normalise_text_for_exact_match(text)
    if not norm:
        return False
    if norm in exact:
        return True
    for rx in regex_list:
        if rx.fullmatch(norm):
            return True
    return False


def deterministic_keep(key: str, fraction: float, seed: str) -> bool:
    f = clamp(float(fraction), 0.0, 1.0)
    if f <= 0.0:
        return False
    if f >= 1.0:
        return True

    digest = hashlib.sha1((seed + "|" + key).encode("utf-8")).digest()
    n = int.from_bytes(digest[:8], "big")
    return (n / 2**64) < f


# ---------------------------
# Writers
# ---------------------------

def escape_ass_text(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("\n", r"\N")
    )


def ass_header_vibgyor(title: str = "Whisper quality (VIBGYOR by avg_logprob)") -> str:
    return f"""[Script Info]
Title: {title}
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Violet,Arial,20,&H00EE82EE,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: Indigo,Arial,20,&H0082004B,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: Blue,Arial,20,&H00FF0000,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: Green,Arial,20,&H0000FF00,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: Yellow,Arial,20,&H0000FFFF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: Orange,Arial,20,&H0000A5FF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1
Style: Red,Arial,20,&H000000FF,&H000000FF,&H00000000,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""".rstrip("\n")


DialogueEntry = Tuple[float, float, str, str, str]  # start, end, style, actor, text


def segments_to_ass_text(
    segments: List[Mapping],
    lo: float,
    hi: float,
    title: str,
    extra_dialogues: Optional[List[DialogueEntry]] = None,
) -> str:
    lines: List[str] = [ass_header_vibgyor(title)]
    entries: List[DialogueEntry] = []

    for s in segments:
        start = s.get("start")
        end = s.get("end")
        text = (s.get("text") or "").strip()

        if not normalise_text_for_exact_match(text):
            continue

        if start is None or end is None or not text:
            continue

        lp = float(s.get("avg_logprob") or 0.0)
        score, lp_n = score_1_to_100(lp, lo, hi)

        style = style_from_lp_norm(lp_n)
        actor = f"{score:03d}"  # 001..100

        entries.append((float(start), float(end), style, actor, escape_ass_text(text)))

    if extra_dialogues:
        entries.extend(extra_dialogues)

    if not entries:
        raise ValueError("No valid segments with start/end/text to write ASS.")

    entries.sort(key=lambda x: (x[0], x[1]))

    for start, end, style, actor, text in entries:
        start_ts = format_timestamp_ass(start)
        end_ts = format_timestamp_ass(end)
        lines.append(f"Dialogue: 0,{start_ts},{end_ts},{style},{actor},0,0,0,,{text}")

    return "\n".join(lines) + "\n"


# ---------------------------
# Filtering policy
# ---------------------------

def should_keep_segment(
    s: Mapping,
    lo: float,
    hi: float,
    file_key: str,
    exact_trivial: set[str],
    regex_trivial: List[re.Pattern],
    trivial_bad_score_max: int,
    trivial_good_score_min: int,
    keep_good_trivial_fraction: float,
    keep_seed: str,
) -> bool:
    text = (s.get("text") or "").strip()

    # Drop punctuation-only / empty-after-normalisation cues (like ".")
    norm = normalise_text_for_exact_match(text)
    if not norm:
        return False

    # Keep everything non-trivial.
    if not is_trivial(text, exact_trivial, regex_trivial):
        return True

    # Trivial: score it and apply your rule.
    lp = float(s.get("avg_logprob") or 0.0)
    score, _lp_n = score_1_to_100(lp, lo, hi)

    # Drop all bad-score trivial cues.
    if score <= int(trivial_bad_score_max):
        return False

    # Keep only a fraction of good-score trivial cues.
    if score >= int(trivial_good_score_min):
        sid = s.get("id")
        start = s.get("start")
        end = s.get("end")
        key = f"{file_key}|{sid}|{start}|{end}|{norm}"
        return deterministic_keep(key, keep_good_trivial_fraction, keep_seed)

    # Mid-score trivial cues: drop.
    return False


# ---------------------------
# Processing
# ---------------------------

def process_file(
    json_path: Path,
    output_dir: Path,
    overwrite: bool,
    exact_trivial: set[str],
    regex_trivial: List[re.Pattern],
    trivial_bad_score_max: int,
    trivial_good_score_min: int,
    keep_good_trivial_fraction: float,
    keep_seed: str,
    ass_title: str,
) -> bool:
    """Process a single JSON file and create ASS subtitle. Returns True if ASS was written."""
    
    # Determine the corresponding audio file name and output ASS path
    audio_filename = json_path.stem + ".m4a"  # Assuming .m4a extension
    ass_path = output_dir / (json_path.stem + ".ass")
    
    print(f"Processing {json_path.name} -> {ass_path.name}")

    if not overwrite and ass_path.exists():
        print(f"  ASS file already exists, skipping: {ass_path}")
        return False

    try:
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"  Error reading JSON file {json_path}: {e}")
        return False

    try:
        segments_all = list(extract_segments(payload))
        lo, hi = lp_norm_bounds(segments_all)
        file_key = json_path.stem

        segments = [
            s
            for s in segments_all
            if should_keep_segment(
                s,
                lo,
                hi,
                file_key,
                exact_trivial,
                regex_trivial,
                trivial_bad_score_max,
                trivial_good_score_min,
                keep_good_trivial_fraction,
                keep_seed,
            )
        ]

        if not segments:
            print(f"  No valid segments found in {json_path.name}")
            return False

        ass_text = segments_to_ass_text(
            segments,
            lo=lo,
            hi=hi,
            title=ass_title,
            extra_dialogues=None,
        )
        
        ass_path.write_text(ass_text, encoding="utf-8")
        print(f"  Created ASS subtitle: {ass_path}")
        return True

    except Exception as e:
        print(f"  Error processing {json_path.name}: {e}")
        return False


def main() -> None:
    """Main function to process specific bad quality recordings."""
    
    # Define the specific files to process
    transcript_dir = Path(r"i:\Transcriptions")
    output_dir = Path(r"I:\Record_bad_quality")
    
    # Specific transcript files for the bad quality recordings
    transcript_files = [
        transcript_dir / "New recording 23.json",
        transcript_dir / "New recording 73.json",
    ]
    
    # Parameters
    overwrite = False
    exact_trivial, regex_trivial = load_trivial_patterns(None, False)
    trivial_bad_score_max = 35
    trivial_good_score_min = 80
    keep_good_trivial_fraction = 0.10
    keep_seed = "v1"
    ass_title = "Whisper quality (VIBGYOR by avg_logprob)"
    
    print(f"Converting transcripts to ASS subtitles...")
    print(f"Output directory: {output_dir}")
    print(f"Files to process: {len(transcript_files)}")
    
    created_ass = 0
    failures = 0
    
    for json_path in transcript_files:
        if not json_path.exists():
            print(f"Transcript file not found: {json_path}")
            failures += 1
            continue
            
        if process_file(
            json_path=json_path,
            output_dir=output_dir,
            overwrite=overwrite,
            exact_trivial=exact_trivial,
            regex_trivial=regex_trivial,
            trivial_bad_score_max=trivial_bad_score_max,
            trivial_good_score_min=trivial_good_score_min,
            keep_good_trivial_fraction=keep_good_trivial_fraction,
            keep_seed=keep_seed,
            ass_title=ass_title,
        ):
            created_ass += 1
        else:
            failures += 1
    
    print(f"\nDone. Created ASS: {created_ass}, Failures: {failures}")
    
    # Add beep sound when done
    print('\a')


if __name__ == "__main__":
    main()
