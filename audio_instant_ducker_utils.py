"""audio_instant_ducker_utils.py

Goal
----
When mixing a "bad" signal (noise / other speakers) into a "good" signal (your clean/high-score audio),
ENFORCE this hard constraint at *every sample*:

    |bad(t)| <= max_ratio * |good(t)|

This is basically "ducking" / sidechain-style attenuation, except enforced deterministically.

Why this matters in your pipeline
--------------------------------
Your current SNR-based scaling controls *average* level (RMS), but peaks can still make the bad audio
momentarily louder than the good audio. This utility clamps/ducks the bad signal per-sample.

Drop-in usage
-------------
In stage_6 and stage_7, after you compute `bad_scaled = bad * gain` (SNR scaling), do:

    from audio_instant_ducker_utils import duck_bad_under_good

    bad_scaled = duck_bad_under_good(
        bad_scaled,
        good=target_speech,
        max_ratio=args.max_bad_to_good_ratio,
        good_floor_db=args.good_floor_db,
    )

    mixed = target_speech + bad_scaled

Notes
-----
- If `good` is silent (near 0), strict enforcement forces `bad` to 0 too.
  That is the *only* way to guarantee "bad never louder".
- If you want room-tone/noise during silence, you must relax the rule:
  set `good_floor_db` to something like -45 dB (NOT strict).

"""

from __future__ import annotations

import numpy as np


def _to_magnitude(x: np.ndarray) -> np.ndarray:
    """Return per-sample magnitude for mono or multi-channel audio.

    For shape (T,), returns (T,)
    For shape (T, C), returns (T,) using max(abs) across channels.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        return np.abs(x)
    if x.ndim == 2:
        return np.max(np.abs(x), axis=1)
    raise ValueError(f"Audio array must be 1D or 2D (got shape={x.shape})")


def _db_to_amp(db: float) -> float:
    return float(10.0 ** (db / 20.0))


def duck_bad_under_good(
    bad: np.ndarray,
    *,
    good: np.ndarray,
    max_ratio: float = 1.0,
    good_floor_db: float = -120.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """Scale `bad` per-sample so it never exceeds `good` (times `max_ratio`).

    Enforces: |bad_out(t)| <= max_ratio * max(|good(t)|, floor)

    Parameters
    ----------
    bad:
        The noise / other-speaker segment you will add.
    good:
        The clean / target segment you want to dominate.
    max_ratio:
        1.0 means bad is never louder than good (strict reading of your requirement).
        0.7 is stricter (bad always at least ~3 dB lower than good).
    good_floor_db:
        Default -120 dB is effectively 0 => strict. Increase (e.g., -45) if you
        intentionally want some bad audio to remain during near-silence.
    eps:
        Numerical stability.

    Returns
    -------
    bad_out:
        Same shape as `bad`.
    """
    if max_ratio <= 0:
        raise ValueError("max_ratio must be > 0")

    bad = np.asarray(bad)
    good = np.asarray(good)

    if bad.shape[0] != good.shape[0]:
        raise ValueError(f"Length mismatch: bad={bad.shape[0]} good={good.shape[0]}")

    good_mag = _to_magnitude(good)
    bad_mag = _to_magnitude(bad)

    floor_amp = _db_to_amp(good_floor_db)
    good_mag = np.maximum(good_mag, floor_amp)

    # gain(t) <= 1 and chosen so that bad_mag(t)*gain(t) <= max_ratio*good_mag(t)
    gain = np.minimum(1.0, (max_ratio * good_mag) / (bad_mag + eps))

    if bad.ndim == 2:
        gain = gain[:, None]

    return bad * gain


def worst_ratio(bad: np.ndarray, good: np.ndarray, eps: float = 1e-12) -> float:
    """Diagnostic: max_t |bad(t)| / |good(t)| (ignoring exact zeros via eps)."""
    good_mag = _to_magnitude(good)
    bad_mag = _to_magnitude(bad)
    return float(np.max(bad_mag / (good_mag + eps)))


# ---------------------------
# PATCH SNIPPETS (copy/paste)
# ---------------------------

# stage_6_add_noise_to_high_score_audio_chunks_manifest_with_noise.py
# ---------------------------------------------------------------
# Add args:
#   parser.add_argument('--max_bad_to_good_ratio', type=float, default=1.0)
#   parser.add_argument('--good_floor_db', type=float, default=-120.0)
#
# In the mixing section, replace:
#   mixed = target_speech + (noise_seg * gain)
# with:
#   from audio_instant_ducker_utils import duck_bad_under_good
#   noise_scaled = noise_seg * gain
#   noise_scaled = duck_bad_under_good(
#       noise_scaled,
#       good=target_speech,
#       max_ratio=args.max_bad_to_good_ratio,
#       good_floor_db=args.good_floor_db,
#   )
#   mixed = target_speech + noise_scaled
#

# stage_7_add_others_voices_to_my_audio.py
# ----------------------------------------
# Add the same args.
#
# In the mixing section, replace:
#   mixed = target_speech + (other_voice_seg * gain)
# with:
#   from audio_instant_ducker_utils import duck_bad_under_good
#   other_scaled = other_voice_seg * gain
#   other_scaled = duck_bad_under_good(
#       other_scaled,
#       good=target_speech,
#       max_ratio=args.max_bad_to_good_ratio,
#       good_floor_db=args.good_floor_db,
#   )
#   mixed = target_speech + other_scaled
