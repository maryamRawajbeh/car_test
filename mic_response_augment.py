# -*- coding: utf-8 -*-
r"""
Simulates generic phone-microphone frequency-response variation.

Different phones' mics differ in frequency response (typically band-limited
somewhere in the ~60-250Hz to ~6-9kHz range, varies by device/model) and
self-noise floor -- we don't know Reddit's or any specific user's exact
device (see SESSION_HANDOFF.md section 1: the original training data's
recording device is unknown), so instead of targeting one specific
fingerprint we can't identify, this generates a WIDE randomized range of
plausible mic frequency responses. Same underlying strategy as
compression_augment.py (train on the transformation itself, don't just
hope the model generalizes to it) applied to a second, independent
dimension of "audio fingerprint" -- device/mic hardware, not codec
compression.

Pure in-process scipy filtering (no ffmpeg subprocess) -- much cheaper
per call than compress_roundtrip, so more variants per file is affordable.
"""

import random

import numpy as np
from scipy.signal import butter, sosfilt

HIGHPASS_RANGE_HZ = (60, 250)
LOWPASS_RANGE_HZ = (6000, 9000)


def simulate_mic_response(y, sr, rng=None):
    """y: float32 mono waveform at sr. Returns a same-length float32 waveform
    passed through a randomized highpass+lowpass pair (2nd-order Butterworth,
    applied via sosfilt for numerical stability), bracketing the range of
    frequency responses real phone mics plausibly have. Peak-renormalized
    afterward since filtering changes the peak amplitude."""
    rng = rng or random
    hp_cutoff = rng.uniform(*HIGHPASS_RANGE_HZ)
    lp_cutoff = min(rng.uniform(*LOWPASS_RANGE_HZ), sr / 2 - 100)

    sos_hp = butter(2, hp_cutoff, btype="highpass", fs=sr, output="sos")
    out = sosfilt(sos_hp, y)

    sos_lp = butter(2, lp_cutoff, btype="lowpass", fs=sr, output="sos")
    out = sosfilt(sos_lp, out)

    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak
    return out.astype(np.float32)
