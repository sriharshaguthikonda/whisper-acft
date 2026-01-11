"""

What this adds vs v5
--------------------
1) Confidence + debugging so you can verify channel selection:
   - --print-sims              prints per-channel similarity scores per file
   - --write-raw-channels      writes ch0.wav/ch1.wav (16 kHz mono) so you can listen
   - decision JSON always stores: channel_index, similarity, second_similarity, margin, accepted

2) More aggressive variants (without you needing to tweak flags each time):
   - 10_ctc_fixedH            (your chosen params)
   - 15_ctc_fixedH_aggressive (beta/h_max/top_db pushed more)
   - 18_ctc_fixedH_plus_specsub (fixedH then magnitude spectral subtraction)
   - 19_ctc_fixedH_aggr_plus_specsub

3) Optional deep denoisers (often less robotic):
   - deepFilter (DeepFilterNet) if installed
   - facebookresearch denoiser if installed

Why you might feel the wrong channel is being selected
------------------------------------------------------
- Your reference audio does NOT sound like your target voice in these recordings (different mic, distance, room, or too much background).
- Both channels contain your voice similarly (strong bleed), so similarities become close (low margin).
- The file has more than 2 channels or channels are swapped / silent.

Fixes:
- Use a clean 10–30s reference snippet recorded in a similar setting.
- Increase --margin (e.g. 0.10) to force clearer wins.
- Use --write-raw-channels + --print-sims to confirm.
- If you KNOW the correct channel, you can override with --force-channel 0 or 1.

Usage (PowerShell)
------------------

Usage (PowerShell)
------------------
# Run this in PowerShell (note the line-continuation ` at the end of each line):

python "i:\whisper-acft\isolate_channel_by_speaker.py" `
  --reference "I:\My_voice\CRISPR Data analysis final (slow paced) - Dr Sri Harsha Guthikonda.mp3" `
  --input-dir "I:\Record" `
  --output-dir "I:\Record_Channel_selected" `
  --state-file "i:\whisper-acft\channel_isolation_state.json" `
  --report-json "c:\Windows_software\whisper-acft\channel_isolation_report.json" `
  --threshold 0.80 `
  --margin 0.05 `
  --write-all `
  --print-sims `
  --write-raw-channels `
  --ctc-beta 1.3 --ctc-h-max 4.0 `
  --specsub-beta 1.3 --specsub-floor 0.05 `
  --write-raw-channels`
  --enable-deepfilter `
  --enable-denoiser `
  --denoiser-preset dns64



"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torchaudio
from huggingface_hub import snapshot_download
from speechbrain.pretrained import EncoderClassifier
from tqdm import tqdm


AUDIO_EXTS_DEFAULT = {".m4a", ".wav", ".mp3", ".flac"}
DEFAULT_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
DEFAULT_THRESHOLD = 0.80
DEFAULT_MARGIN = 0.05

TARGET_SR = 16_000

CTC_N_FFT = 1024
CTC_HOP = 256


@dataclass
class Decision:
    channel_index: int
    similarity: float
    second_similarity: float
    margin: float
    accepted: bool
    output_paths: Dict[str, str]
    num_channels: int
    input_mtime: float
    sims_by_channel: List[float]
    error: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pick speaker-matching channel, then write multiple enhanced variants.")

    p.add_argument("--reference", required=True, help="Reference audio for target speaker (any format).")

    g_in = p.add_mutually_exclusive_group(required=True)
    g_in.add_argument("--input-dir", help="Directory containing recordings.")
    g_in.add_argument("--input-file", help="Single input file (for quick testing).")

    p.add_argument("--output-dir", required=True, help="Directory to write outputs (mirrors input tree).")

    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Cosine similarity threshold.")
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN, help="Require best-second_best >= margin (if >=2 channels).")

    p.add_argument("--state-file", help="JSON state for resumable runs.")
    p.add_argument("--report-json", help="Where to write final JSON report.")
    p.add_argument("--reprocess-changed", action="store_true", help="Reprocess if input mtime changed.")
    p.add_argument("--force", action="store_true", help="Force reprocess everything (ignores state).")

    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device for speaker embedder.")
    p.add_argument("--cache-dir", default=r"I:\\pretrained_models", help="Cache directory for SpeechBrain model.")
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="SpeechBrain speaker model id.")

    p.add_argument("--audio-exts", default=",".join(sorted(AUDIO_EXTS_DEFAULT)), help="Comma-separated extensions to scan.")

    # Embedding sampling
    p.add_argument("--segment-seconds", type=float, default=10.0, help="Seconds per segment for embedding.")
    p.add_argument("--segments", type=int, default=3, help="Number of segments sampled across the file.")

    # FFmpeg fallback
    p.add_argument("--ffmpeg", default="ffmpeg", help="Path to ffmpeg for decode fallback.")
    p.add_argument("--no-ffmpeg-fallback", action="store_true", help="Disable ffmpeg fallback.")

    # Output variants
    p.add_argument("--write-all", action="store_true", help="Write all variants (recommended).")
    p.add_argument("--write-selected-only", action="store_true", help="Only write selected channel (no enhancement).")

    # Channel override
    p.add_argument("--force-channel", type=int, default=-1, help="Override selection with channel index (e.g. 0 or 1). -1 = auto.")

    # Debug outputs
    p.add_argument("--print-sims", action="store_true", help="Print similarity scores per channel for each file.")
    p.add_argument("--write-raw-channels", action="store_true", help="Write raw mono channels (16k) for listening.")

    # Crosstalk cancellation knobs
    p.add_argument("--ctc-h-max", type=float, default=2.0, help="Clamp |H(f)| for fixed crosstalk canceller.")
    p.add_argument("--ctc-beta", type=float, default=1.0, help="Subtract beta*H(f)*Y (beta <1.0 gentler, >1.0 more aggressive).")
    p.add_argument("--ctc-estimation-top-db", type=float, default=6.0, help="Estimate H(f) mostly when interferer is strong: frames where ref energy is within top X dB.")

    # Spectral subtraction knobs (post-CTC)
    p.add_argument("--specsub-beta", type=float, default=1.0, help="Magnitude subtraction amount (higher = more aggressive).")
    p.add_argument("--specsub-floor", type=float, default=0.08, help="Spectral floor fraction to reduce musical noise (lower = more aggressive, more artifacts).")

    # Optional external denoisers
    p.add_argument("--enable-deepfilter", action="store_true", help="Enable DeepFilterNet stage if installed.")
    p.add_argument("--enable-denoiser", action="store_true", help="Enable facebookresearch denoiser stage if installed.")
    p.add_argument("--denoiser-preset", choices=["dns48", "dns64", "master64"], default="dns64", help="Which pretrained denoiser preset.")

    return p.parse_args()


def pick_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_model_local_dir(model_id: str, cache_dir: str) -> str:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("SPEECHBRAIN_CACHE", cache_dir)
    os.environ.setdefault("SPEECHBRAIN_LOCAL_STRATEGY", "copy")

    cache_p = pathlib.Path(cache_dir)
    cache_p.mkdir(parents=True, exist_ok=True)

    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_dir=str(cache_p / model_id.replace("/", "_")),
        local_dir_use_symlinks=False,
        allow_patterns="*",
    )

    label_txt = pathlib.Path(local_dir) / "label_encoder.txt"
    label_ckpt = pathlib.Path(local_dir) / "label_encoder.ckpt"
    if label_txt.exists() and not label_ckpt.exists():
        shutil.copyfile(label_txt, label_ckpt)

    return str(local_dir)


def load_embedder(device: str, local_dir: str) -> EncoderClassifier:
    return EncoderClassifier.from_hparams(
        source=local_dir,
        run_opts={"device": device},
        savedir=local_dir,
    )


def _load_audio_torchaudio(path: pathlib.Path) -> Tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return wav, int(sr)


def _load_audio_ffmpeg(path: pathlib.Path, ffmpeg_exe: str) -> Tuple[torch.Tensor, int]:
    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    try:
        cmd = [
            ffmpeg_exe,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(path),
            "-c:a",
            "pcm_s16le",
            tmp_wav,
        ]
        subprocess.run(cmd, check=True)
        wav, sr = _load_audio_torchaudio(pathlib.Path(tmp_wav))
        return wav, sr
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass


def load_audio_any(path: pathlib.Path, ffmpeg_exe: str, allow_ffmpeg_fallback: bool) -> Tuple[torch.Tensor, int]:
    try:
        return _load_audio_torchaudio(path)
    except Exception:
        if not allow_ffmpeg_fallback:
            raise
        return _load_audio_ffmpeg(path, ffmpeg_exe)


def resample_if_needed(wavs: torch.Tensor, sr: int, target_sr: int) -> Tuple[torch.Tensor, int]:
    if sr == target_sr:
        return wavs, sr
    resampler = torchaudio.transforms.Resample(int(sr), int(target_sr))
    return resampler(wavs), int(target_sr)


def save_mono_wav(wav_mono_t: torch.Tensor, sr: int, out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if wav_mono_t.ndim == 1:
        wav_mono_t = wav_mono_t.unsqueeze(0)
    torchaudio.save(str(out_path), wav_mono_t.cpu(), int(sr))


def normalize_audio(wavs: torch.Tensor) -> torch.Tensor:
    peak = wavs.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    return wavs / peak


def make_segments(wavs_ct: torch.Tensor, segment_samples: int, segments: int) -> List[torch.Tensor]:
    c, t = wavs_ct.shape
    if t <= segment_samples or segments <= 1:
        return [wavs_ct[:, :segment_samples] if t > segment_samples else wavs_ct]

    max_start = t - segment_samples
    starts = torch.linspace(0, max_start, steps=segments).round().to(torch.int64)
    out: List[torch.Tensor] = []
    for s in starts.tolist():
        s = int(s)
        out.append(wavs_ct[:, s : s + segment_samples])
    return out


def encode_channel_embeddings(
    encoder: EncoderClassifier,
    wavs_ct_16k: torch.Tensor,
    segment_seconds: float,
    segments: int,
) -> torch.Tensor:
    wavs_ct_16k = normalize_audio(wavs_ct_16k)

    seg_samples = int(max(1, round(segment_seconds * TARGET_SR)))
    seg_list = make_segments(wavs_ct_16k, seg_samples, segments)

    emb_sum: Optional[torch.Tensor] = None
    for seg in seg_list:
        seg = seg.to(encoder.device)
        with torch.no_grad():
            embs = encoder.encode_batch(seg)
        if embs.ndim == 3:
            embs = embs.squeeze(1)
        emb_sum = embs if emb_sum is None else (emb_sum + embs)

    assert emb_sum is not None
    return emb_sum / float(len(seg_list))


def cosine_sims(embs_cd: torch.Tensor, ref_d: torch.Tensor) -> torch.Tensor:
    embs = torch.nn.functional.normalize(embs_cd, p=2, dim=1)
    ref = torch.nn.functional.normalize(ref_d, p=2, dim=0)
    return torch.matmul(embs, ref.unsqueeze(1)).squeeze(1)


def compute_reference_vector(
    encoder: EncoderClassifier,
    reference_path: pathlib.Path,
    ffmpeg_exe: str,
    allow_ffmpeg_fallback: bool,
    segment_seconds: float,
    segments: int,
) -> torch.Tensor:
    wavs, sr = load_audio_any(reference_path, ffmpeg_exe=ffmpeg_exe, allow_ffmpeg_fallback=allow_ffmpeg_fallback)
    if wavs.ndim == 1:
        wavs = wavs.unsqueeze(0)
    wavs, _ = resample_if_needed(wavs, sr, TARGET_SR)
    if wavs.shape[0] > 1:
        wavs = wavs.mean(dim=0, keepdim=True)

    emb = encode_channel_embeddings(encoder, wavs, segment_seconds=segment_seconds, segments=max(1, min(segments, 3)))[0]
    return emb.detach()


def _stft(x_t: torch.Tensor, n_fft: int, hop: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=x_t.device)
    return torch.stft(x_t, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)


def _istft(X_ft: torch.Tensor, n_fft: int, hop: int, length: int) -> torch.Tensor:
    window = torch.hann_window(n_fft, device=X_ft.device)
    return torch.istft(X_ft, n_fft=n_fft, hop_length=hop, window=window, length=length)


def crosstalk_cancel_fixedH(
    target_t: torch.Tensor,
    ref_t: torch.Tensor,
    n_fft: int,
    hop: int,
    h_max: float,
    beta: float,
    estimation_top_db: float,
) -> torch.Tensor:
    Xt = _stft(target_t, n_fft=n_fft, hop=hop)
    Yt = _stft(ref_t, n_fft=n_fft, hop=hop)

    ref_pow = (Yt.abs() ** 2).mean(dim=0)
    maxp = ref_pow.max().clamp_min(1e-12)
    thr = maxp / (10 ** (estimation_top_db / 10.0))
    w = (ref_pow >= thr).to(torch.float32)
    if w.sum() < 2:
        w = torch.ones_like(w)
    w = w.unsqueeze(0)

    Sxy = (Xt * torch.conj(Yt) * w).sum(dim=1)
    Syy = ((Yt.abs() ** 2) * w).sum(dim=1).clamp_min(1e-12)

    Hf = Sxy / Syy
    Hmag = Hf.abs().clamp(max=float(h_max))
    Hf = Hmag * torch.exp(1j * torch.angle(Hf))

    Xclean = Xt - float(beta) * (Hf.unsqueeze(1) * Yt)
    return _istft(Xclean, n_fft=n_fft, hop=hop, length=target_t.numel())


def specsub_with_ref(
    target_t: torch.Tensor,
    ref_t: torch.Tensor,
    n_fft: int,
    hop: int,
    beta: float,
    floor: float,
) -> torch.Tensor:
    """Magnitude spectral subtraction using ref magnitude as noise estimate.

    More aggressive:
      - higher beta
      - lower floor

    This can improve transcription but may add artifacts if pushed too far.
    """
    X = _stft(target_t, n_fft=n_fft, hop=hop)
    R = _stft(ref_t, n_fft=n_fft, hop=hop)

    magX = torch.abs(X)
    magR = torch.abs(R)

    magY = magX - float(beta) * magR
    floor_mag = float(floor) * magX
    magY = torch.maximum(magY, floor_mag)

    Y = magY * torch.exp(1j * torch.angle(X))
    return _istft(Y, n_fft=n_fft, hop=hop, length=target_t.numel())


def _which(cmd: str) -> Optional[str]:
    from shutil import which

    return which(cmd)


def deepfilter_available() -> bool:
    return _which("deepFilter") is not None


def denoiser_available() -> bool:
    try:
        import denoiser  # noqa: F401

        return True
    except Exception:
        return False


def run_deepfilter(in_wav: pathlib.Path, out_wav: pathlib.Path) -> bool:
    exe = _which("deepFilter")
    if not exe:
        return False

    with tempfile.TemporaryDirectory() as td:
        td_p = pathlib.Path(td)
        cmd = [exe, "--output-dir", str(td_p), str(in_wav)]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return False

        cand = td_p / in_wav.name
        if not cand.exists():
            wavs = list(td_p.glob("*.wav"))
            if not wavs:
                return False
            cand = wavs[0]

        out_wav.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cand, out_wav)
        return True


def run_denoiser(in_wav_dir: pathlib.Path, out_dir: pathlib.Path, preset: str, device: str) -> bool:
    if not denoiser_available():
        return False

    preset_flag = f"--{preset}"

    cmd = [
        sys.executable,
        "-m",
        "denoiser.enhance",
        preset_flag,
        "--device",
        device,
        "--noisy_dir",
        str(in_wav_dir),
        "--out_dir",
        str(out_dir),
        "--batch_size",
        "1",
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


def load_state(state_path: pathlib.Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {"meta": {}, "decisions": {}}
    with state_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state_path: pathlib.Path, state: Dict[str, Any]) -> None:
    tmp = state_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_path)


def discover_audio_files(root: pathlib.Path, exts: Iterable[str]) -> List[pathlib.Path]:
    exts_l = {e.lower().strip() for e in exts if e.strip()}
    out: List[pathlib.Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts_l:
            out.append(p)
    return out


def process_one(
    encoder: EncoderClassifier,
    ref_vec: torch.Tensor,
    path: pathlib.Path,
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    ffmpeg_exe: str,
    allow_ffmpeg_fallback: bool,
    segment_seconds: float,
    segments: int,
    threshold: float,
    margin: float,
    write_all: bool,
    write_selected_only: bool,
    force_channel: int,
    print_sims: bool,
    write_raw_channels: bool,
    ctc_h_max: float,
    ctc_beta: float,
    ctc_estimation_top_db: float,
    specsub_beta: float,
    specsub_floor: float,
    enable_deepfilter: bool,
    enable_denoiser: bool,
    denoiser_preset: str,
) -> Dict[str, Any]:
    mtime = path.stat().st_mtime
    try:
        wavs, sr = load_audio_any(path, ffmpeg_exe=ffmpeg_exe, allow_ffmpeg_fallback=allow_ffmpeg_fallback)
        if wavs.ndim == 1:
            wavs = wavs.unsqueeze(0)

        wavs_16k, _ = resample_if_needed(wavs, sr, TARGET_SR)
        num_channels = int(wavs_16k.shape[0])

        embs = encode_channel_embeddings(encoder, wavs_16k, segment_seconds=segment_seconds, segments=segments)
        sims_t = cosine_sims(embs, ref_vec.to(encoder.device)).detach().cpu()
        sims = [float(x) for x in sims_t.tolist()]

        # Pick channel
        if force_channel >= 0 and force_channel < num_channels:
            best_idx = int(force_channel)
            best_sim = sims[best_idx]
            # second best for reporting
            second_sim = max([s for i, s in enumerate(sims) if i != best_idx], default=-1.0)
        else:
            if num_channels == 1:
                best_idx = 0
                best_sim = sims[0]
                second_sim = -1.0
            else:
                # top2
                top2 = torch.topk(sims_t, k=2)
                best_idx = int(top2.indices[0].item())
                best_sim = float(top2.values[0].item())
                second_sim = float(top2.values[1].item())

        m = float(best_sim - second_sim) if num_channels > 1 else 0.0
        accepted = (best_sim >= float(threshold)) and (m >= float(margin) if num_channels > 1 else True)

        if print_sims:
            print(f"\n[file] {path}")
            print(f"  sims={['%.3f'%s for s in sims]}  selected={best_idx}  best={best_sim:.3f}  second={second_sim:.3f}  margin={m:.3f}  accepted={accepted}")

        # Output base path
        if path.is_file() and args.input_file:
            # single file mode: keep flat
            rel = pathlib.Path(path.name)
        else:
            rel = path.relative_to(input_dir)

        base_out = (output_dir / rel).with_suffix(".wav")

        outputs: Dict[str, str] = {}

        # Optional raw channel dumps for listening
        if write_raw_channels:
            raw_dir = base_out.parent / "__raw_channels"
            for ch in range(min(num_channels, 8)):
                out_raw = raw_dir / f"{base_out.stem}_ch{ch}.wav"
                save_mono_wav(wavs_16k[ch], TARGET_SR, out_raw)
            outputs["raw_channels_dir"] = str(raw_dir)

        selected = wavs_16k[best_idx]
        other = None
        if num_channels >= 2:
            other = wavs_16k[1 - best_idx] if num_channels == 2 else wavs_16k[int((best_idx + 1) % num_channels)]

        # Always write selected-only if requested or write_all
        if write_selected_only or write_all:
            out_sel = base_out.parent / "00_selected" / base_out.name
            save_mono_wav(selected, TARGET_SR, out_sel)
            outputs["selected"] = str(out_sel)

        if not write_all:
            d = Decision(
                channel_index=best_idx,
                similarity=float(best_sim),
                second_similarity=float(second_sim),
                margin=float(m),
                accepted=bool(accepted),
                output_paths=outputs,
                num_channels=num_channels,
                input_mtime=float(mtime),
                sims_by_channel=sims,
                error="",
            )
            return dataclasses.asdict(d)

        # If no other channel, stop at selected
        if other is None:
            d = Decision(
                channel_index=best_idx,
                similarity=float(best_sim),
                second_similarity=float(second_sim),
                margin=float(m),
                accepted=bool(accepted),
                output_paths=outputs,
                num_channels=num_channels,
                input_mtime=float(mtime),
                sims_by_channel=sims,
                error="",
            )
            return dataclasses.asdict(d)

        # Method 10: fixedH with your params
        clean_10 = crosstalk_cancel_fixedH(
            selected,
            other,
            n_fft=CTC_N_FFT,
            hop=CTC_HOP,
            h_max=float(ctc_h_max),
            beta=float(ctc_beta),
            estimation_top_db=float(ctc_estimation_top_db),
        )
        out_10 = base_out.parent / "10_ctc_fixedH" / base_out.name
        save_mono_wav(clean_10, TARGET_SR, out_10)
        outputs["ctc_fixedH"] = str(out_10)

        # Method 15: more aggressive fixedH (auto-pushed)
        ag_beta = float(ctc_beta) * 1.25
        ag_hmax = float(ctc_h_max) * 1.5
        ag_top = float(ctc_estimation_top_db) + 6.0
        clean_15 = crosstalk_cancel_fixedH(
            selected,
            other,
            n_fft=CTC_N_FFT,
            hop=CTC_HOP,
            h_max=float(ag_hmax),
            beta=float(ag_beta),
            estimation_top_db=float(ag_top),
        )
        out_15 = base_out.parent / "15_ctc_fixedH_aggressive" / base_out.name
        save_mono_wav(clean_15, TARGET_SR, out_15)
        outputs["ctc_fixedH_aggressive"] = str(out_15)

        # Method 18: fixedH + spectral subtraction
        clean_18 = specsub_with_ref(
            target_t=clean_10,
            ref_t=other,
            n_fft=CTC_N_FFT,
            hop=CTC_HOP,
            beta=float(specsub_beta),
            floor=float(specsub_floor),
        )
        out_18 = base_out.parent / "18_ctc_fixedH_plus_specsub" / base_out.name
        save_mono_wav(clean_18, TARGET_SR, out_18)
        outputs["ctc_plus_specsub"] = str(out_18)

        # Method 19: aggressive fixedH + spectral subtraction
        clean_19 = specsub_with_ref(
            target_t=clean_15,
            ref_t=other,
            n_fft=CTC_N_FFT,
            hop=CTC_HOP,
            beta=float(specsub_beta) * 1.15,
            floor=max(0.02, float(specsub_floor) * 0.75),
        )
        out_19 = base_out.parent / "19_ctc_fixedH_aggr_plus_specsub" / base_out.name
        save_mono_wav(clean_19, TARGET_SR, out_19)
        outputs["ctc_aggr_plus_specsub"] = str(out_19)

        # Optional: DeepFilterNet (usually needs 48k)
        if enable_deepfilter and deepfilter_available():
            try:
                with tempfile.TemporaryDirectory() as td:
                    td_p = pathlib.Path(td)
                    in48 = td_p / "in48.wav"
                    out48 = td_p / "out48.wav"
                    sel48, _ = resample_if_needed(selected.unsqueeze(0), TARGET_SR, 48_000)
                    save_mono_wav(sel48.squeeze(0), 48_000, in48)
                    if run_deepfilter(in48, out48) and out48.exists():
                        df_wavs, df_sr = _load_audio_torchaudio(out48)
                        df_wavs, _ = resample_if_needed(df_wavs, df_sr, TARGET_SR)
                        out_df = base_out.parent / "20_df_deepfilter" / base_out.name
                        save_mono_wav(df_wavs[0], TARGET_SR, out_df)
                        outputs["deepfilter"] = str(out_df)
            except Exception:
                pass

        # Optional: denoiser
        if enable_denoiser and denoiser_available():
            try:
                with tempfile.TemporaryDirectory() as td:
                    td_p = pathlib.Path(td)
                    noisy_dir = td_p / "noisy"
                    out_dir = td_p / "out"
                    noisy_dir.mkdir(parents=True, exist_ok=True)
                    out_dir.mkdir(parents=True, exist_ok=True)

                    in_w = noisy_dir / base_out.name
                    save_mono_wav(selected, TARGET_SR, in_w)
                    device_flag = "cuda" if torch.cuda.is_available() else "cpu"
                    if run_denoiser(noisy_dir, out_dir, preset=denoiser_preset, device=device_flag):
                        cand = out_dir / base_out.name
                        if cand.exists():
                            w, sr2 = _load_audio_torchaudio(cand)
                            w, _ = resample_if_needed(w, sr2, TARGET_SR)
                            out_dn = base_out.parent / "40_denoiser" / base_out.name
                            save_mono_wav(w[0], TARGET_SR, out_dn)
                            outputs["denoiser"] = str(out_dn)
            except Exception:
                pass

        d = Decision(
            channel_index=best_idx,
            similarity=float(best_sim),
            second_similarity=float(second_sim),
            margin=float(m),
            accepted=bool(accepted),
            output_paths=outputs,
            num_channels=num_channels,
            input_mtime=float(mtime),
            sims_by_channel=sims,
            error="",
        )
        return dataclasses.asdict(d)

    except Exception as e:
        fail = Decision(
            channel_index=0,
            similarity=-1.0,
            second_similarity=-1.0,
            margin=0.0,
            accepted=False,
            output_paths={},
            num_channels=0,
            input_mtime=float(mtime),
            sims_by_channel=[],
            error=f"{type(e).__name__}: {e}",
        )
        return dataclasses.asdict(fail)


def main() -> None:
    global args
    args = parse_args()

    output_dir = pathlib.Path(args.output_dir)
    state_path = pathlib.Path(args.state_file) if args.state_file else None
    report_path = pathlib.Path(args.report_json) if args.report_json else None

    device = pick_device(args.device)
    allow_ffmpeg_fallback = not args.no_ffmpeg_fallback

    model_dir = prepare_model_local_dir(args.model_id, args.cache_dir)
    encoder = load_embedder(device, model_dir)

    ref_vec = compute_reference_vector(
        encoder,
        reference_path=pathlib.Path(args.reference),
        ffmpeg_exe=args.ffmpeg,
        allow_ffmpeg_fallback=allow_ffmpeg_fallback,
        segment_seconds=float(args.segment_seconds),
        segments=int(args.segments),
    )

    state = load_state(state_path) if state_path else {"meta": {}, "decisions": {}}
    state.setdefault("meta", {})
    state.setdefault("decisions", {})
    state["meta"].update(
        {
            "updated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "threshold": float(args.threshold),
            "margin": float(args.margin),
            "target_sr": int(TARGET_SR),
            "model_id": str(args.model_id),
            "device": str(device),
            "segment_seconds": float(args.segment_seconds),
            "segments": int(args.segments),
            "allow_ffmpeg_fallback": bool(allow_ffmpeg_fallback),
            "write_all": bool(args.write_all),
            "write_selected_only": bool(args.write_selected_only),
            "force_channel": int(args.force_channel),
            "print_sims": bool(args.print_sims),
            "write_raw_channels": bool(args.write_raw_channels),
            "ctc_h_max": float(args.ctc_h_max),
            "ctc_beta": float(args.ctc_beta),
            "ctc_estimation_top_db": float(args.ctc_estimation_top_db),
            "specsub_beta": float(args.specsub_beta),
            "specsub_floor": float(args.specsub_floor),
            "enable_deepfilter": bool(args.enable_deepfilter),
            "enable_denoiser": bool(args.enable_denoiser),
            "denoiser_preset": str(args.denoiser_preset),
        }
    )

    # Build file list
    files: List[pathlib.Path] = []
    if args.input_file:
        p = pathlib.Path(args.input_file)
        if not p.exists():
            raise FileNotFoundError(str(p))
        files = [p]
        input_dir = p.parent
    else:
        input_dir = pathlib.Path(args.input_dir)
        exts = [e.strip() for e in args.audio_exts.split(",") if e.strip()]
        files = discover_audio_files(input_dir, exts)

    to_process: List[pathlib.Path] = []
    for p in files:
        k = str(p.resolve())
        if args.force or not state_path:
            to_process.append(p)
            continue
        prev = state["decisions"].get(k)
        if prev is None:
            to_process.append(p)
            continue
        if args.reprocess_changed:
            try:
                if float(prev.get("input_mtime", -1)) != float(p.stat().st_mtime):
                    to_process.append(p)
            except FileNotFoundError:
                pass

    print(f"[device] {device}")
    print(f"[deepFilter] available={deepfilter_available()} enabled={bool(args.enable_deepfilter)}")
    print(f"[denoiser]  available={denoiser_available()} enabled={bool(args.enable_denoiser)} preset={args.denoiser_preset}")
    print(f"[discovery] total={len(files)} to_process={len(to_process)} already_done={len(files) - len(to_process)}")

    try:
        for p in tqdm(to_process, desc="Processing", unit="file"):
            k = str(p.resolve())
            if state_path and (not args.force) and (not args.reprocess_changed) and k in state["decisions"]:
                continue

            decision = process_one(
                encoder=encoder,
                ref_vec=ref_vec,
                path=p,
                input_dir=input_dir,
                output_dir=output_dir,
                ffmpeg_exe=args.ffmpeg,
                allow_ffmpeg_fallback=allow_ffmpeg_fallback,
                segment_seconds=float(args.segment_seconds),
                segments=int(args.segments),
                threshold=float(args.threshold),
                margin=float(args.margin),
                write_all=bool(args.write_all),
                write_selected_only=bool(args.write_selected_only),
                force_channel=int(args.force_channel),
                print_sims=bool(args.print_sims),
                write_raw_channels=bool(args.write_raw_channels),
                ctc_h_max=float(args.ctc_h_max),
                ctc_beta=float(args.ctc_beta),
                ctc_estimation_top_db=float(args.ctc_estimation_top_db),
                specsub_beta=float(args.specsub_beta),
                specsub_floor=float(args.specsub_floor),
                enable_deepfilter=bool(args.enable_deepfilter),
                enable_denoiser=bool(args.enable_denoiser),
                denoiser_preset=str(args.denoiser_preset),
            )

            if state_path:
                state["decisions"][k] = decision
                save_state(state_path, state)

    except KeyboardInterrupt:
        print("\n[stop] Ctrl+C received. Stopping cleanly.")

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)


if __name__ == "__main__":
    main()
