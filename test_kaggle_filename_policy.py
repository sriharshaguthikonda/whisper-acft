#!/usr/bin/env python3
"""Deterministic checks for Kaggle-safe filename policy."""

from __future__ import annotations

import re
from pathlib import Path

import stage_10_b_add_speech_tempo_pause_aware_idempotent as stage10b
import stage_11_add_frequency_manipulation_idempotent as stage11
import stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise as stage6
import stage_7_add_others_voices_to_my_audio_fast_idempotent as stage7
import stage_8_add_random_gain_to_high_score_voices_parallel_idempotent as stage8
import stage_9_add_reverb_idempotent as stage9
from pipeline_uid_utils import kaggle_safe_wav_name, sanitize_kaggle_filename, sanitize_kaggle_token


SAFE_WAV_RE = re.compile(r"^[A-Za-z0-9._-]+\.wav$")
MAX_WAV_NAME_LEN = 120


def _assert_safe_wav_name(name: str) -> None:
    assert SAFE_WAV_RE.match(name), f"invalid wav name: {name}"
    assert len(name) <= MAX_WAV_NAME_LEN, f"wav name too long ({len(name)}): {name}"


def test_sanitize_token() -> None:
    assert sanitize_kaggle_token("voice_mix", default="x") == "voice_mix"
    assert sanitize_kaggle_token("voice mix, (v2) & extra", default="x") == "voice_mix_v2_extra"
    assert sanitize_kaggle_token(",,,,", default="fallback") == "fallback"


def test_sanitize_filename() -> None:
    raw = "Abnormal liver function tests Gilbert syndrome.__3491dead5d_chunk0000__L"
    name = sanitize_kaggle_filename(raw, ext=".wav", max_name_len=MAX_WAV_NAME_LEN)
    _assert_safe_wav_name(name)

    very_long = "A" * 400 + " !!! "
    n1 = sanitize_kaggle_filename(very_long, ext=".wav", max_name_len=64)
    n2 = sanitize_kaggle_filename(very_long, ext=".wav", max_name_len=64)
    assert n1 == n2, "truncation must be deterministic"
    assert len(n1) <= 64
    _assert_safe_wav_name(n1)


def test_kaggle_safe_wav_name() -> None:
    name = kaggle_safe_wav_name(
        ["base uid", "aug uid", "tempo speech/pause&(x)", "copy01"],
        max_name_len=MAX_WAV_NAME_LEN,
    )
    _assert_safe_wav_name(name)


def test_stage_builders() -> dict[str, str]:
    row = {"base_uid": "0123456789abcdef0123456789abcdef"}
    new_uid = "fedcba9876543210fedcba9876543210"
    dirty_stage = "tempo speech/pause&(x)"
    out_dir = Path(r"I:\tmp")

    outputs = {
        "stage6": Path(stage6.build_out_wav_name(row, dirty_stage, new_uid, 1, out_dir)).name,
        "stage7": Path(stage7.build_out_wav_name(row, dirty_stage, new_uid, 1, out_dir)).name,
        "stage8": Path(stage8.build_out_wav_name(row, dirty_stage, new_uid, 1, out_dir)).name,
        "stage9": Path(stage9.build_out_wav_name(row, dirty_stage, new_uid, 1, out_dir)).name,
        "stage10b": Path(
            stage10b.build_out_wav_name(
                row,
                dirty_stage,
                new_uid,
                1,
                tempo_factor=1.17,
                silence_factor=2.80,
                pause_policy="truncate",
                out_dir=out_dir,
            )
        ).name,
        "stage11": Path(stage11.build_out_wav_name(row, dirty_stage, new_uid, 1, out_dir)).name,
    }

    for stage_name, name in outputs.items():
        _assert_safe_wav_name(name)
        assert "__" not in name, f"unexpected double separator left in {stage_name}: {name}"
    return outputs


def main() -> None:
    test_sanitize_token()
    test_sanitize_filename()
    test_kaggle_safe_wav_name()

    outputs = test_stage_builders()
    smoke_dir = Path("test_sample/kaggle_filename_policy_smoke")
    smoke_dir.mkdir(parents=True, exist_ok=True)
    for p in smoke_dir.glob("*"):
        if p.is_file():
            p.unlink()
    for name in outputs.values():
        (smoke_dir / name).write_bytes(b"wav-smoke")
    print("PASS: Kaggle filename policy tests")


if __name__ == "__main__":
    main()
