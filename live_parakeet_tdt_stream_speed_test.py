#!/usr/bin/env python3
"""Live mic speed test for Parakeet-TDT 0.6B v3 from Hugging Face (via NeMo).

Controls (focused terminal window):
- SPACE: toggle recognition active/inactive
- S: print speed stats snapshot
- Q or ESC: quit
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import tempfile
import threading
import time
import wave
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


def _write_wav_pcm16(path: Path, samples: Any, sample_rate: int) -> None:
    # Keep WAV writing stdlib-only to avoid extra dependencies.
    import numpy as np  # type: ignore

    clipped = np.clip(samples, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16, copy=False)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


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


def _require_deps() -> tuple[Any, Any, Any]:
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
        import torch  # type: ignore
        import nemo.collections.asr as nemo_asr  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Missing dependency: NeMo ASR stack. Install with:\n"
            "python -m pip install \"nemo_toolkit[asr]>=2.4.0\" torch"
        ) from exc

    return np, sd, (torch, nemo_asr)


def _load_model(model_id: str, device_req: str, torch: Any, nemo_asr: Any) -> tuple[Any, str]:
    if device_req == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_req
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested CUDA, but torch.cuda.is_available() is False.")

    print(_cyan(f"Loading model: {model_id}"))
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
    asr_model = asr_model.to(device)
    asr_model.eval()
    return asr_model, device


def _transcribe_once(asr_model: Any, wav_path: Path, timestamps: bool) -> str:
    if timestamps:
        try:
            out = asr_model.transcribe([str(wav_path)], timestamps=True)
        except TypeError:
            out = asr_model.transcribe([str(wav_path)])
    else:
        out = asr_model.transcribe([str(wav_path)])

    if isinstance(out, tuple) and out:
        out = out[0]
    if isinstance(out, list) and out:
        return _extract_text(out[0])
    return _extract_text(out)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Live speed test for nvidia/parakeet-tdt-0.6b-v3 with SPACE toggle."
    )
    ap.add_argument("--model_id", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--sample_rate", type=int, default=16000)
    ap.add_argument("--chunk_secs", type=float, default=2.0)
    ap.add_argument("--block_secs", type=float, default=0.2)
    ap.add_argument("--mic_device", default=None, help="sounddevice input device id or name")
    ap.add_argument("--timestamps", action="store_true", help="Ask NeMo for timestamps (extra overhead)")
    ap.add_argument("--start_active", action="store_true", help="Start recognition active immediately")
    ap.add_argument("--warmup", action="store_true", default=True, help="Warm up one chunk before live loop")
    ap.add_argument("--no_warmup", dest="warmup", action="store_false")
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

    np, sd, nemo_stack = _require_deps()
    torch, nemo_asr = nemo_stack

    if args.chunk_secs <= 0:
        raise SystemExit("--chunk_secs must be > 0.")
    if args.block_secs <= 0:
        raise SystemExit("--block_secs must be > 0.")

    # Ensure ANSI colors work in modern Windows terminals.
    if os.name == "nt":
        os.system("")

    asr_model, resolved_device = _load_model(args.model_id, args.device, torch, nemo_asr)
    print(_green(f"Model ready on device={resolved_device}"))
    mic_device = _normalize_device_arg(args.mic_device)

    sample_rate = int(args.sample_rate)
    chunk_samples = int(round(args.chunk_secs * sample_rate))
    block_samples = max(1, int(round(args.block_secs * sample_rate)))

    if chunk_samples < 1:
        raise SystemExit("Computed chunk_samples < 1. Check --chunk_secs and --sample_rate.")

    if args.warmup:
        print(_cyan("Running warmup pass..."))
        with tempfile.TemporaryDirectory(prefix="parakeet_live_warmup_") as warm_dir:
            warm_path = Path(warm_dir) / "warmup.wav"
            _write_wav_pcm16(warm_path, np.zeros(chunk_samples, dtype=np.float32), sample_rate)
            _ = _transcribe_once(asr_model, warm_path, timestamps=False)
        print(_green("Warmup done."))

    out_jsonl_path: Optional[Path] = Path(args.out_jsonl).resolve() if args.out_jsonl else None
    if out_jsonl_path is not None:
        out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    print("")
    print(_cyan("Controls: SPACE=toggle, S=stats, Q/ESC=quit"))
    print(_yellow(f"Initial state: {'ACTIVE' if args.start_active else 'INACTIVE'}"))
    print("")

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
    buffers: list[Any] = []
    buffered_samples = 0
    buffered_end_ts = time.perf_counter()

    tmp_dir = Path(tempfile.mkdtemp(prefix="parakeet_live_chunks_"))
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
                        buffers.clear()
                        buffered_samples = 0
                        state = _green("ACTIVE") if active else _yellow("INACTIVE")
                        print(f"{state} at {time.strftime('%H:%M:%S')}")
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

                buffers.append(block)
                buffered_samples += int(block.shape[0])
                buffered_end_ts = block_end_ts

                while buffered_samples >= chunk_samples:
                    merged = np.concatenate(buffers, axis=0)
                    chunk = merged[:chunk_samples]
                    remainder = merged[chunk_samples:]
                    remainder_sec = float(remainder.shape[0]) / float(sample_rate)
                    chunk_end_ts = buffered_end_ts - remainder_sec

                    buffers = [remainder] if remainder.size else []
                    buffered_samples = int(remainder.shape[0])

                    chunk_idx += 1
                    chunk_path = tmp_dir / f"chunk_{chunk_idx:06d}.wav"
                    _write_wav_pcm16(chunk_path, chunk, sample_rate)

                    infer_start = time.perf_counter()
                    text = _transcribe_once(asr_model, chunk_path, timestamps=args.timestamps)
                    infer_end = time.perf_counter()

                    infer_sec = infer_end - infer_start
                    audio_sec = float(chunk.shape[0]) / float(sample_rate)
                    queue_lag = max(0.0, infer_start - chunk_end_ts)
                    rtf = infer_sec / audio_sec if audio_sec > 0 else float("inf")
                    xrt = (1.0 / rtf) if rtf > 0 else float("inf")

                    stats.add(audio_sec, infer_sec)

                    print(
                        f"[{chunk_idx:05d}] "
                        f"audio={audio_sec:4.2f}s | infer={infer_sec:4.2f}s | "
                        f"RTF={rtf:5.3f} | xRealtime={xrt:4.2f}x | lag={queue_lag:4.2f}s | "
                        f"text={_shorten(text)}"
                    )

                    if log_fh is not None:
                        payload = {
                            "chunk_index": chunk_idx,
                            "audio_sec": audio_sec,
                            "infer_sec": infer_sec,
                            "rtf": rtf,
                            "x_realtime": xrt,
                            "queue_lag_sec": queue_lag,
                            "text": text,
                            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                        log_fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
                        log_fh.flush()

                    try:
                        chunk_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                    if args.max_chunks > 0 and chunk_idx >= args.max_chunks:
                        stop_event.set()
                        break

    except KeyboardInterrupt:
        stop_event.set()
    finally:
        if log_fh is not None:
            try:
                log_fh.close()
            except Exception:
                pass
        try:
            for wav_file in tmp_dir.glob("*.wav"):
                wav_file.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except Exception:
            pass

    print("")
    print(_cyan("Final stats: " + stats.summary_line()))
    if dropped_blocks["count"] > 0:
        print(_yellow(f"Dropped audio callback blocks: {dropped_blocks['count']}"))
    _beep_done()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
