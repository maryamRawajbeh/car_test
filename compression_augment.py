# -*- coding: utf-8 -*-
r"""
Round-trips a waveform through WebM/Opus encode+decode, using the EXACT
SAME ffmpeg flags as garageai-audio-analysis/app/services/audio_converter.py's
convert_to_wav() (mono, target sample rate) -- simulating what a browser's
MediaRecorder(stream) capture + the backend's ffmpeg conversion actually
produces, instead of the clean lab-recorded audio the models have always
been trained on.

Built after mic_pipeline_test/test_mic_pipeline_domain_shift.py measured a
REAL, severe accuracy collapse from this exact pipeline on the currently
deployed model (brake: 100% clean -> 20% at 32kbps webm) -- this module is
the training-side fix: instead of generic waveform augmentation (time-shift/
volume/noise, already established to HURT the traditional ML model, see
SESSION_HANDOFF.md section 3), generate TRAIN-only copies that have actually
been through the real compression pipeline, so the model can learn to
ignore compression artifacts instead of (apparently) partly relying on them.

Uses imageio_ffmpeg's bundled static binary (installed into this venv
specifically for this) -- self-contained, no dependency on
garageai-audio-analysis's separate venv/path.
"""

import os
import tempfile
import subprocess

import numpy as np
import soundfile as sf
from imageio_ffmpeg import get_ffmpeg_exe

FFMPEG_EXE = get_ffmpeg_exe()

DEFAULT_BITRATES_KBPS = [16, 32, 64]


def compress_roundtrip(y, sr, bitrate_kbps):
    """y: float32 mono waveform at sr. Returns a same-length float32 waveform
    that has been through webm/opus encode at bitrate_kbps then decoded back,
    via ffmpeg -- exactly what the real app's upload pipeline produces."""
    with tempfile.TemporaryDirectory() as td:
        in_wav = os.path.join(td, "in.wav")
        webm_path = os.path.join(td, "a.webm")
        out_wav = os.path.join(td, "out.wav")

        sf.write(in_wav, y, sr)

        # timeout=30: a hung ffmpeg subprocess (rare, but seen with malformed audio in
        # other projects) would otherwise stall an entire multi-hour preprocessing run
        # with no indication of which file caused it
        subprocess.run(
            [FFMPEG_EXE, "-y", "-loglevel", "error", "-i", in_wav,
             "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", webm_path],
            capture_output=True, check=True, timeout=30,
        )
        subprocess.run(
            [FFMPEG_EXE, "-y", "-loglevel", "error", "-i", webm_path,
             "-ac", "1", "-ar", str(sr), out_wav],
            capture_output=True, check=True, timeout=30,
        )

        y_out, _ = sf.read(out_wav, dtype="float32")

    # webm/opus roundtrip shifts length by a handful of samples (codec framing) --
    # pad/trim back to the exact original length so downstream fit_to_length()
    # behaves identically to any other variant
    if len(y_out) > len(y):
        y_out = y_out[:len(y)]
    elif len(y_out) < len(y):
        y_out = np.pad(y_out, (0, len(y) - len(y_out)))
    return y_out
