#!/usr/bin/env python3
"""Live mic speed test for Parakeet-TDT 0.6B v3 from Hugging Face (via NeMo).


I:\Whisper-training-env\Scripts\python.exe I:\whisper-acft\live_parakeet_tdt_stream_speed_test.py --device cuda --mic_device 1 --chunk_secs 1.5 --left_context_secs 4 --right_context_secs 0.5 --block_secs 0.1 --start_active 


Controls (focused terminal window):
- SPACE: toggle recognition active/inactive
- S: print speed stats snapshot
- Q or ESC: quit

By default, this script prints only recognized text lines.
Use --debug for detailed runtime logs and speed metrics.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(text: str) -> str:
    return _color(text, "32")


def _yellow(text: str) -> str:
    return _color(text, "33")


def _red(text: str) -> str:
    return _color(text, "31")


def _cyan(text: str) -> str:
    return _color(text, "36")


def _beep_done() -> None:
    if os.name != "nt":
        return
    try:
        import winsound  # type: ignore

        winsound.Beep(1000, 300)
        winsound.Beep(1200, 300)
        winsound.Beep(1500, 500)
    except Exception:
        pass


def _beep_ready() -> None:
    if os.name != "nt":
        return
    try:
        import winsound  # type: ignore

        winsound.Beep(900, 180)
        winsound.Beep(1200, 220)
    except Exception:
        pass


def _beep_toggle(is_active: bool) -> None:
    if os.name != "nt":
        return
    try:
        import winsound  # type: ignore

        winsound.Beep(1200 if is_active else 650, 120)
    except Exception:
        pass


@contextlib.contextmanager
def _suppress_stream_output(enabled: bool) -> Any:
    if not enabled:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _extract_text(result_item: Any) -> str:
    if result_item is None:
        return ""
    if isinstance(result_item, str):
        return result_item
    if isinstance(result_item, dict):
        txt = result_item.get("text")
        return txt if isinstance(txt, str) else ""
    txt = getattr(result_item, "text", None)
    return txt if isinstance(txt, str) else ""


def _shorten(text: str, max_len: int = 160) -> str:
    text = " ".join(text.strip().split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _word_tokens(text: str) -> list[str]:
    return [tok for tok in text.strip().split() if tok]


def _token_key(token: str) -> str:
    key = re.sub(r"[^A-Za-z0-9]+", "", token).lower()
    return key if key else token.lower()


def _common_prefix_words(left: list[str], right: list[str]) -> list[str]:
    limit = min(len(left), len(right))
    idx = 0
    while idx < limit and _token_key(left[idx]) == _token_key(right[idx]):
        idx += 1
    return right[:idx]


@dataclass
class SpeedStats:
    chunks: int = 0
    total_audio_sec: float = 0.0
    total_infer_sec: float = 0.0
    min_infer_sec: float = float("inf")
    max_infer_sec: float = 0.0

    def add(self, audio_sec: float, infer_sec: float) -> None:
        self.chunks += 1
        self.total_audio_sec += audio_sec
        self.total_infer_sec += infer_sec
        if infer_sec < self.min_infer_sec:
            self.min_infer_sec = infer_sec
        if infer_sec > self.max_infer_sec:
            self.max_infer_sec = infer_sec

    def summary_line(self) -> str:
        if self.chunks == 0:
            return "chunks=0"
        avg_infer = self.total_infer_sec / self.chunks
        avg_audio = self.total_audio_sec / self.chunks
        avg_rtf = self.total_infer_sec / self.total_audio_sec if self.total_audio_sec > 0 else float("inf")
        avg_xrt = (1.0 / avg_rtf) if avg_rtf > 0 else float("inf")
        min_infer = self.min_infer_sec if self.min_infer_sec != float("inf") else 0.0
        return (
            f"chunks={self.chunks} | avg_audio={avg_audio:.2f}s | avg_infer={avg_infer:.2f}s | "
            f"avg_RTF={avg_rtf:.3f} | avg_xRealtime={avg_xrt:.2f}x | "
            f"min_infer={min_infer:.2f}s | max_infer={self.max_infer_sec:.2f}s"
        )


def _keyboard_listener(cmd_q: "queue.Queue[str]", stop_event: threading.Event) -> None:
    if os.name != "nt":
        return

    import msvcrt  # type: ignore

    while not stop_event.is_set():
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                # Skip extended key second byte.
                if msvcrt.kbhit():
                    _ = msvcrt.getwch()
                continue
            if ch == " ":
                cmd_q.put("toggle")
            elif ch.lower() == "s":
                cmd_q.put("stats")
            elif ch.lower() == "q" or ch == "\x1b":
                cmd_q.put("quit")
                stop_event.set()
                return
        time.sleep(0.01)


def _require_deps(debug: bool) -> tuple[Any, Any, Any]:
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: numpy. Install with:\n"
            "python -m pip install numpy"
        ) from exc

    try:
        import sounddevice as sd  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: sounddevice. Install with:\n"
            "python -m pip install sounddevice"
        ) from exc

    try:
        if not debug:
            os.environ.setdefault("NEMO_LOG_LEVEL", "ERROR")
        import torch  # type: ignore
        import nemo.collections.asr as nemo_asr  # type: ignore
        if not debug:
            try:
                from nemo.utils import logging as nemo_logging  # type: ignore

                nemo_logging.set_verbosity(nemo_logging.ERROR)
            except Exception:
                pass
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: NeMo ASR stack. Install with:\n"
            "python -m pip install \"nemo_toolkit[asr]>=2.4.0\" torch"
        ) from exc

    return np, sd, (torch, nemo_asr)


def _load_model(model_id: str, device_req: str, torch: Any, nemo_asr: Any, debug: bool) -> tuple[Any, str]:
    if device_req == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_req
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA, but torch.cuda.is_available() is False.")

    if debug:
        print(_cyan(f"Loading model: {model_id}"))
    with _suppress_stream_output(enabled=not debug):
        asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    asr_model = asr_model.to(device)
    asr_model.eval()
    return asr_model, device


def _transcribe_once(
    asr_model: Any,
    audio_input: Any,
    timestamps: bool,
    debug: bool,
    batch_size: int,
    num_workers: int,
) -> list[str]:
    kwargs: dict[str, Any] = {}
    if timestamps:
        kwargs["timestamps"] = True

    kwargs["batch_size"] = max(1, int(batch_size))
    kwargs["num_workers"] = max(0, int(num_workers))
    kwargs["verbose"] = bool(debug)
    kwargs["return_hypotheses"] = False

    with _suppress_stream_output(enabled=not debug):
        try:
            out = asr_model.transcribe(audio=audio_input, **kwargs)
        except TypeError:
            kwargs.pop("verbose", None)
            out = asr_model.transcribe(audio=audio_input, **kwargs)

    if isinstance(out, tuple) and out:
        out = out[0]
    if isinstance(out, list):
        return [_extract_text(item) for item in out]
    return [_extract_text(out)]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Live speed test for nvidia/parakeet-tdt-0.6b-v3 with SPACE toggle."
    )
    ap.add_argument("--model_id", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--chunk_secs", type=float, default=1.5)
    ap.add_argument("--left_context_secs", type=float, default=8.0)
    ap.add_argument("--right_context_secs", type=float, default=0.6)
    ap.add_argument("--block_secs", type=float, default=0.2)
    ap.add_argument("--infer_batch_size", type=int, default=1, help="NeMo transcribe batch_size")
    ap.add_argument("--infer_num_workers", type=int, default=0, help="NeMo transcribe num_workers")
    ap.add_argument("--mic_device", default=None, help="sounddevice input device id or name")
    ap.add_argument("--timestamps", action="store_true", help="Ask NeMo for timestamps (extra overhead)")
    ap.add_argument("--start_active", action="store_true", help="Start recognition active immediately")
    ap.add_argument("--warmup", action="store_true", default=True, help="Warm up one chunk before live loop")
    ap.add_argument("--no_warmup", dest="warmup", action="store_false")
    ap.add_argument("--debug", action="store_true", help="Show detailed logs and speed metrics")
    ap.add_argument("--list_devices", action="store_true")
    ap.add_argument("--max_chunks", type=int, default=0, help="Stop after N chunks (0 = run until quit)")
    ap.add_argument("--out_jsonl", default="", help="Optional JSONL file for per-chunk speed logs")
    return ap.parse_args()


def _normalize_device_arg(raw_value: Optional[str]) -> Any:
    if raw_value is None:
        return None
    txt = str(raw_value).strip()
    if not txt:
        return None
    if txt.lstrip("+-").isdigit():
        return int(txt)
    return raw_value


def main() -> int:
    args = _parse_args()

    if args.list_devices:
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:
            raise SystemExit(
                "Missing dependency: sounddevice. Install with:\n"
                "python -m pip install sounddevice"
            ) from exc
        print(sd.query_devices())
        return 0

    np, sd, nemo_stack = _require_deps(debug=args.debug)
    torch, nemo_asr = nemo_stack

    if args.chunk_secs <= 0:
        raise SystemExit("--chunk_secs must be > 0.")
    if args.left_context_secs < 0:
        raise SystemExit("--left_context_secs must be >= 0.")
    if args.right_context_secs < 0:
        raise SystemExit("--right_context_secs must be >= 0.")
    if args.block_secs <= 0:
        raise SystemExit("--block_secs must be > 0.")
    if args.infer_batch_size <= 0:
        raise SystemExit("--infer_batch_size must be > 0.")
    if args.infer_num_workers < 0:
        raise SystemExit("--infer_num_workers must be >= 0.")

    # Ensure ANSI colors work in modern Windows terminals.
    if os.name == "nt":
        os.system("")

    asr_model, resolved_device = _load_model(args.model_id, args.device, torch, nemo_asr, debug=args.debug)
    if args.debug:
        print(_green(f"Model ready on device={resolved_device}"))
    mic_device = _normalize_device_arg(args.mic_device)

    sample_rate = int(args.sample_rate)
    chunk_samples = int(round(args.chunk_secs * sample_rate))
    left_context_samples = int(round(args.left_context_secs * sample_rate))
    right_context_samples = int(round(args.right_context_secs * sample_rate))
    block_samples = max(1, int(round(args.block_secs * sample_rate)))

    if chunk_samples < 1:
        raise SystemExit("Computed chunk_samples < 1. Check --chunk_secs and --sample_rate.")

    if args.warmup:
        if args.debug:
            print(_cyan("Running warmup pass..."))
        _ = _transcribe_once(
            asr_model=asr_model,
            audio_input=np.zeros(chunk_samples, dtype=np.float32),
            timestamps=False,
            debug=args.debug,
            batch_size=args.infer_batch_size,
            num_workers=args.infer_num_workers,
        )
        if args.debug:
            print(_green("Warmup done."))

    _beep_ready()

    out_jsonl_path: Optional[Path] = Path(args.out_jsonl).resolve() if args.out_jsonl else None
    if out_jsonl_path is not None:
        out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    if args.debug:
        print("")
        print(_cyan("Controls: SPACE=toggle, S=stats, Q/ESC=quit"))
        print(_yellow(f"Initial state: {'ACTIVE' if args.start_active else 'INACTIVE'}"))
        print(
            _cyan(
                f"Streaming window: chunk={args.chunk_secs:.2f}s, "
                f"left_ctx={args.left_context_secs:.2f}s, right_ctx={args.right_context_secs:.2f}s"
            )
        )
        print("")
    else:
        print(
            f"Ready. SPACE=toggle, Q/ESC=quit, S=stats. "
            f"(chunk={args.chunk_secs:.1f}s, left={args.left_context_secs:.1f}s, right={args.right_context_secs:.1f}s)"
        )

    stop_event = threading.Event()
    cmd_q: "queue.Queue[str]" = queue.Queue()
    audio_q: "queue.Queue[tuple[Any, float]]" = queue.Queue(maxsize=256)

    dropped_blocks = {"count": 0}

    def audio_callback(indata: Any, frames: int, time_info: Any, status: Any) -> None:
        if status:
            # Keep callback lightweight and avoid noisy logs.
            pass
        try:
            audio_q.put_nowait((indata[:, 0].copy(), time.perf_counter()))
        except queue.Full:
            dropped_blocks["count"] += 1

    kb_thread = threading.Thread(target=_keyboard_listener, args=(cmd_q, stop_event), daemon=True)
    kb_thread.start()

    stats = SpeedStats()
    active = bool(args.start_active)
    chunk_idx = 0
    stream_buffer = np.zeros(0, dtype=np.float32)
    stream_base_sample = 0
    next_chunk_start_sample = 0
    committed_words: list[str] = []
    prev_window_words: Optional[list[str]] = None

    log_fh = None
    try:
        if out_jsonl_path is not None:
            log_fh = out_jsonl_path.open("a", encoding="utf-8")

        with sd.InputStream(
            samplerate=sample_rate,
            blocksize=block_samples,
            channels=1,
            dtype="float32",
            callback=audio_callback,
            device=mic_device,
        ):
            while not stop_event.is_set():
                while True:
                    try:
                        cmd = cmd_q.get_nowait()
                    except queue.Empty:
                        break

                    if cmd == "toggle":
                        active = not active
                        stream_buffer = np.zeros(0, dtype=np.float32)
                        stream_base_sample = 0
                        next_chunk_start_sample = 0
                        committed_words = []
                        prev_window_words = None
                        if args.debug:
                            state = _green("ACTIVE") if active else _yellow("INACTIVE")
                            print(f"{state} at {time.strftime('%H:%M:%S')}")
                        else:
                            _beep_toggle(active)
                    elif cmd == "stats":
                        print(_cyan(stats.summary_line()))
                    elif cmd == "quit":
                        stop_event.set()
                        break

                if stop_event.is_set():
                    break

                try:
                    block, block_end_ts = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                if not active:
                    continue

                stream_buffer = np.concatenate((stream_buffer, block), axis=0)
                available_end_sample = stream_base_sample + int(stream_buffer.shape[0])

                # Process chunks only when we have enough lookahead (right context).
                while available_end_sample >= (next_chunk_start_sample + chunk_samples + right_context_samples):
                    chunk_idx += 1
                    chunk_start_sample = next_chunk_start_sample
                    chunk_end_sample = chunk_start_sample + chunk_samples

                    window_start_sample = max(0, chunk_start_sample - left_context_samples)
                    window_end_sample = chunk_end_sample + right_context_samples

                    local_start = max(0, window_start_sample - stream_base_sample)
                    local_end = max(local_start, window_end_sample - stream_base_sample)
                    window_audio = stream_buffer[local_start:local_end]

                    samples_after_chunk = max(0, available_end_sample - chunk_end_sample)
                    chunk_end_ts = block_end_ts - (float(samples_after_chunk) / float(sample_rate))

                    infer_start = time.perf_counter()
                    texts = _transcribe_once(
                        asr_model=asr_model,
                        audio_input=window_audio,
                        timestamps=args.timestamps,
                        debug=args.debug,
                        batch_size=args.infer_batch_size,
                        num_workers=args.infer_num_workers,
                    )
                    infer_end = time.perf_counter()

                    text = texts[0] if texts else ""
                    infer_sec = infer_end - infer_start
                    chunk_audio_sec = float(chunk_samples) / float(sample_rate)
                    queue_lag = max(0.0, infer_start - chunk_end_ts)
                    rtf = infer_sec / chunk_audio_sec if chunk_audio_sec > 0 else float("inf")
                    xrt = (1.0 / rtf) if rtf > 0 else float("inf")

                    current_words = _word_tokens(text)
                    emitted_words: list[str] = []
                    if prev_window_words is not None:
                        stable_words = _common_prefix_words(prev_window_words, current_words)
                        if len(stable_words) > len(committed_words):
                            emitted_words = stable_words[len(committed_words) :]
                            committed_words.extend(emitted_words)
                    prev_window_words = current_words
                    emitted_text = " ".join(emitted_words).strip()

                    stats.add(chunk_audio_sec, infer_sec)

                    if args.debug:
                        print(
                            f"[{chunk_idx:05d}] "
                            f"audio={chunk_audio_sec:4.2f}s | infer={infer_sec:4.2f}s | "
                            f"RTF={rtf:5.3f} | xRealtime={xrt:4.2f}x | lag={queue_lag:4.2f}s | "
                            f"window={len(window_audio)/sample_rate:4.2f}s | "
                            f"emit={_shorten(emitted_text, 80)} | raw={_shorten(text, 120)}"
                        )
                    else:
                        if emitted_text:
                            print(emitted_text, flush=True)

                    if log_fh is not None:
                        payload = {
                            "chunk_index": chunk_idx,
                            "audio_sec": chunk_audio_sec,
                            "window_audio_sec": float(len(window_audio)) / float(sample_rate),
                            "infer_sec": infer_sec,
                            "rtf": rtf,
                            "x_realtime": xrt,
                            "queue_lag_sec": queue_lag,
                            "left_context_secs": float(args.left_context_secs),
                            "right_context_secs": float(args.right_context_secs),
                            "emitted_text": emitted_text,
                            "raw_text": text,
                            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                        log_fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
                        log_fh.flush()

                    next_chunk_start_sample += chunk_samples

                    # Keep only what we need for left-context on the next step.
                    keep_from_sample = max(0, next_chunk_start_sample - left_context_samples - block_samples)
                    if keep_from_sample > stream_base_sample:
                        drop_count = keep_from_sample - stream_base_sample
                        if drop_count >= int(stream_buffer.shape[0]):
                            stream_buffer = np.zeros(0, dtype=np.float32)
                        else:
                            stream_buffer = stream_buffer[drop_count:]
                        stream_base_sample = keep_from_sample
                        available_end_sample = stream_base_sample + int(stream_buffer.shape[0])

                    if args.max_chunks > 0 and chunk_idx >= args.max_chunks:
                        stop_event.set()
                        break

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        # Flush tail text that may never have become stable before shutdown.
        if prev_window_words is not None and len(prev_window_words) > len(committed_words):
            tail_words = prev_window_words[len(committed_words) :]
            tail_text = " ".join(tail_words).strip()
            if tail_text:
                if args.debug:
                    print(_cyan(f"Final tail emit: {_shorten(tail_text, 120)}"))
                else:
                    print(tail_text, flush=True)

        if log_fh is not None:
            try:
                log_fh.close()
            except Exception:
                pass

    if args.debug:
        print("")
        print(_cyan("Final stats: " + stats.summary_line()))
        if dropped_blocks["count"] > 0:
            print(_yellow(f"Dropped audio callback blocks: {dropped_blocks['count']}"))
    _beep_done()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
