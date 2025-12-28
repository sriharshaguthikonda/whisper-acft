#!/usr/bin/env python3
"""Convert Groq/OpenAI Whisper verbose_json transcripts to WebVTT and ASS.

Outputs
- .vtt: timings + text
- .ass: timings + text, with the Actor/Name field set to a 1..100 quality score
        derived from robust-normalised avg_logprob. Colour style uses VIBGYOR.

Filtering goal
- Identify standalone "junk/trivial" cues (okay/yes doctor/no doctor/hello/thank you/etc).
- Drop ALL junk cues with a bad score.
- Keep ONLY a small fraction (default 10%) of *good* junk cues, deterministically.
- Keep everything else.

Extendability
- Add more junk phrases later using:
    --trivial-phrases-file extra_trivial.txt
  By default it EXTENDS the built-in list.
  Use --replace-trivial-phrases to replace the built-in list.

Trivial phrases file format
- One entry per line
- Blank lines and lines starting with # are ignored
- Plain lines are exact matches after normalisation (lowercase, punctuation removed)
- Lines starting with "re:" are regex patterns (we use fullmatch)

Usage examples
  python json_to_subs.py --input-dir "I:\\P2GPT_google_drive\\My Drive\\Transcriptions" --write-ass
  python json_to_subs.py --input-dir . --output-dir out --write-vtt --write-ass --overwrite
  python json_to_subs.py --input-dir . --write-ass --trivial-phrases-file extra_trivial.txt
"""

from __future__ import annotations

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

def segments_to_vtt_lines(segments: Iterable[Mapping]) -> List[str]:
    lines = ["WEBVTT", ""]
    wrote_any = False

    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        text = (segment.get("text") or "").strip()

        # Drop punctuation-only cues (e.g. ".")
        if not normalise_text_for_exact_match(text):
            continue

        if start is None or end is None or not text:
            continue

        start_ts = format_timestamp_vtt(float(start))
        end_ts = format_timestamp_vtt(float(end))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
        wrote_any = True

    if not wrote_any:
        raise ValueError("No valid segments with start/end/text to write VTT.")

    return lines


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


def detect_text_differences(
    corrected_segments: Sequence[Mapping],
    original_segments: Sequence[Mapping],
    time_tolerance: float = 0.25,
) -> List[Tuple[float, float, str, str]]:
    """Return list of (start, end, original_text, corrected_text) where text differs ignoring punctuation."""
    diffs: List[Tuple[float, float, str, str]] = []
    pairs = zip(corrected_segments, original_segments)
    for corr_seg, orig_seg in pairs:
        try:
            corr_start = float(corr_seg.get("start"))
            corr_end = float(corr_seg.get("end"))
            orig_start = float(orig_seg.get("start"))
            orig_end = float(orig_seg.get("end"))
        except Exception:
            continue

        if (
            abs(corr_start - orig_start) > time_tolerance
            or abs(corr_end - orig_end) > time_tolerance
        ):
            continue

        corr_text = (corr_seg.get("text") or "").strip()
        orig_text = (orig_seg.get("text") or "").strip()

        if normalise_text_for_exact_match(corr_text) == normalise_text_for_exact_match(orig_text):
            continue

        diffs.append((corr_start, corr_end, orig_text, corr_text))

    return diffs


def diff_entries_to_dialogues(diffs: List[Tuple[float, float, str, str]]) -> List[DialogueEntry]:
    """Create ASS dialogue entries for differing segments using Red for original and Green for corrected."""
    entries: List[DialogueEntry] = []
    for start, end, orig_text, corr_text in diffs:
        entries.append((start, end, "Red", "001", escape_ass_text(orig_text)))
        entries.append((start, end, "Green", "001", escape_ass_text(corr_text)))
    return entries


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
    write_vtt: bool,
    write_ass: bool,
    compare_dir: Optional[Path],
    diff_time_tolerance: float,
    exact_trivial: set[str],
    regex_trivial: List[re.Pattern],
    trivial_bad_score_max: int,
    trivial_good_score_min: int,
    keep_good_trivial_fraction: float,
    keep_seed: str,
    ass_title: str,
) -> Optional[Tuple[bool, bool]]:
    vtt_path = output_dir / (json_path.stem + ".vtt")
    ass_path = output_dir / (json_path.stem + ".ass")

    if not overwrite:
        vtt_ok = (not write_vtt) or vtt_path.exists()
        ass_ok = (not write_ass) or ass_path.exists()
        if vtt_ok and ass_ok:
            return None

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

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

    vtt_written = False
    ass_written = False

    diff_matches: List[Tuple[float, float, str, str]] = []
    extra_dialogues: Optional[List[DialogueEntry]] = None
    segments_for_ass = segments

    if write_vtt and (overwrite or not vtt_path.exists()):
        vtt_lines = segments_to_vtt_lines(segments)
        vtt_path.write_text("\n".join(vtt_lines) + "\n", encoding="utf-8")
        vtt_written = True

    if write_ass and compare_dir:
        orig_path = compare_dir / json_path.name
        if orig_path.exists():
            try:
                orig_payload = json.loads(orig_path.read_text(encoding="utf-8"))
                orig_segments = list(extract_segments(orig_payload))
                diff_matches = detect_text_differences(
                    segments_all, orig_segments, time_tolerance=diff_time_tolerance
                )
                if diff_matches:
                    extra_dialogues = diff_entries_to_dialogues(diff_matches)
                    def _is_diff(seg: Mapping) -> bool:
                        try:
                            s_start = float(seg.get("start"))
                            s_end = float(seg.get("end"))
                        except Exception:
                            return False
                        for d_start, d_end, _o, _c in diff_matches:
                            if (
                                abs(s_start - d_start) <= diff_time_tolerance
                                and abs(s_end - d_end) <= diff_time_tolerance
                            ):
                                return True
                        return False

                    segments_for_ass = [s for s in segments if not _is_diff(s)]
            except Exception:
                extra_dialogues = extra_dialogues  # keep None/previous

    if write_ass and (overwrite or not ass_path.exists()):
        ass_text = segments_to_ass_text(
            segments_for_ass,
            lo=lo,
            hi=hi,
            title=ass_title,
            extra_dialogues=extra_dialogues,
        )
        ass_path.write_text(ass_text, encoding="utf-8")
        ass_written = True

    return vtt_written, ass_written


def convert_all(
    input_dir: Path,
    output_dir: Path,
    overwrite: bool,
    workers: int,
    write_vtt: bool,
    write_ass: bool,
    compare_dir: Optional[Path],
    diff_time_tolerance: float,
    exact_trivial: set[str],
    regex_trivial: List[re.Pattern],
    trivial_bad_score_max: int,
    trivial_good_score_min: int,
    keep_good_trivial_fraction: float,
    keep_seed: str,
    ass_title: str,
) -> None:
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    if output_dir.exists():
        if not output_dir.is_dir():
            raise ValueError(f"Output path exists and is not a directory: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    if compare_dir and not compare_dir.exists():
        raise FileNotFoundError(f"Compare directory not found: {compare_dir}")

    if not write_vtt and not write_ass:
        raise ValueError("Nothing to do: enable --write-vtt and/or --write-ass")

    created_vtt = 0
    created_ass = 0
    skipped = 0
    failures = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(
                process_file,
                p,
                output_dir,
                overwrite,
                write_vtt,
                write_ass,
                compare_dir,
                diff_time_tolerance,
                exact_trivial,
                regex_trivial,
                trivial_bad_score_max,
                trivial_good_score_min,
                keep_good_trivial_fraction,
                keep_seed,
                ass_title,
            ): p
            for p in json_files
        }

        with tqdm(total=len(json_files), desc="Converting JSON", unit="file") as bar:
            for fut in as_completed(future_map):
                json_file = future_map[fut]
                try:
                    result = fut.result()
                    if result is None:
                        skipped += 1
                    else:
                        vtt_written, ass_written = result
                        if vtt_written:
                            created_vtt += 1
                        if ass_written:
                            created_ass += 1
                except Exception as exc:
                    failures += 1
                    bar.write(f"[ERROR] {json_file.name}: {exc}")
                bar.update(1)

    tqdm.write(
        "Done. "
        f"Created VTT: {created_vtt}, Created ASS: {created_ass}, "
        f"Skipped: {skipped} (already present), Failures: {failures}"
    )


# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Groq/OpenAI Whisper verbose_json transcripts to WebVTT and/or ASS (with junk filtering)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(r"i:\\P2GPT_google_drive\\My Drive\\Transcriptions_corrected"),
        help="Directory containing transcript JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"I:\Record"),
        help="Where to write subtitle files (defaults to input directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Regenerate outputs even if they already exist (default: off).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(4, (os.cpu_count() or 4)),
        help="Number of worker threads (I/O bound).",
    )
    parser.add_argument(
        "--write-vtt",
        action="store_true",
        help="Also write .vtt files (default: off).",
    )
    parser.add_argument(
        "--no-write-ass",
        dest="write_ass",
        action="store_false",
        help="Do not write .ass files (default: on).",
    )
    parser.set_defaults(write_ass=True)

    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        help="Optional directory containing original transcripts to compare against input-dir for diff overlay.",
    )
    parser.add_argument(
        "--diff-time-tolerance",
        type=float,
        default=0.25,
        help="Max seconds difference to treat segment start/end as matching when diffing.",
    )

    parser.add_argument(
        "--ass-title",
        type=str,
        default="Whisper quality (VIBGYOR by avg_logprob)",
        help="ASS script title (shows in subtitle editors).",
    )

    # Junk filtering controls
    parser.add_argument(
        "--trivial-bad-score-max",
        type=int,
        default=35,
        help="Drop ALL trivial/junk cues with score <= this value.",
    )
    parser.add_argument(
        "--trivial-good-score-min",
        type=int,
        default=80,
        help="Treat trivial/junk cues with score >= this as 'good'.",
    )
    parser.add_argument(
        "--keep-good-trivial-fraction",
        type=float,
        default=0.10,
        help="Keep only this fraction of good trivial/junk cues (deterministic).",
    )
    parser.add_argument(
        "--keep-seed",
        type=str,
        default="v1",
        help="Seed for deterministic keep/drop of good trivial/junk cues.",
    )
    parser.add_argument(
        "--trivial-phrases-file",
        type=Path,
        default=None,
        help="Optional text file containing extra trivial phrases and/or regex (prefix regex lines with 're:').",
    )
    parser.add_argument(
        "--replace-trivial-phrases",
        action="store_true",
        help="Replace the built-in trivial list with the file instead of extending it.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir

    exact_trivial, regex_trivial = load_trivial_patterns(
        file_path=args.trivial_phrases_file,
        replace_default=args.replace_trivial_phrases,
    )

    convert_all(
        input_dir=args.input_dir,
        output_dir=output_dir,
        overwrite=args.overwrite,
        workers=args.workers,
        write_vtt=args.write_vtt,
        write_ass=args.write_ass,
        compare_dir=args.compare_dir,
        diff_time_tolerance=args.diff_time_tolerance,
        exact_trivial=exact_trivial,
        regex_trivial=regex_trivial,
        trivial_bad_score_max=args.trivial_bad_score_max,
        trivial_good_score_min=args.trivial_good_score_min,
        keep_good_trivial_fraction=args.keep_good_trivial_fraction,
        keep_seed=args.keep_seed,
        ass_title=args.ass_title,
    )


if __name__ == "__main__":
    main()

