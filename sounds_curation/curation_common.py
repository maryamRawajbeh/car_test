# -*- coding: utf-8 -*-
r"""
Shared energy-based 5-second window selection, used by both
select_candidates.py (the sounds\ Freesound pool) and
select_brake_dataset_candidates.py (the brake_dataset_freesound\accepted
pool) so the two stay identical by construction instead of hand-copied.

See select_candidates.py's module docstring for the full rationale
(streaming block-RMS energy, why not librosa.resample in this venv, etc).
"""

import os
from fractions import Fraction

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

WINDOW_SEC = 5.0
HOP_SEC = 0.1                       # energy-profile resolution
TOP_K = 3                           # up to this many alternate windows per file
MIN_CANDIDATE_SEPARATION_SEC = 2.5  # non-overlap requirement between ranked candidates
CANDIDATE_SR = 22050                 # matches TARGET_SR in preprocessing.py
LOW_ENERGY_DBFS = -40.0              # informational flag only, not a filter


def resample(y, orig_sr, target_sr):
    """Plain scipy resampler. Not librosa.resample: librosa.core.audio
    imports the `soxr` package unconditionally at module level, and this
    venv's soxr_ext native DLL fails to load -- res_type= doesn't help
    since the broken import happens before res_type is even looked at."""
    if orig_sr == target_sr:
        return y
    frac = Fraction(target_sr, orig_sr).limit_denominator(1000)
    return resample_poly(y, frac.numerator, frac.denominator).astype(np.float32)


def compute_block_rms(path, sr, hop_frames):
    """Streams the file in hop_frames-sized blocks (mono-mixed) and returns
    an array of per-block RMS values, without ever holding the full file
    in memory."""
    rms_values = []
    for block in sf.blocks(path, blocksize=hop_frames, dtype="float32", always_2d=True):
        mono = block.mean(axis=1)
        rms_values.append(float(np.sqrt(np.mean(mono ** 2)) + 1e-12))
    return np.array(rms_values, dtype=np.float64)


def find_candidate_windows(block_rms, hop_frames, window_frames, top_k):
    """Picks up to top_k non-overlapping windows ranked by mean energy,
    using a prefix-sum so every possible window's energy is O(1) to look up."""
    n_blocks = len(block_rms)
    blocks_per_window = max(1, round(window_frames / hop_frames))

    if n_blocks <= blocks_per_window:
        # file shorter than (or ~equal to) one window: nothing to choose between
        return [{"start_block": 0, "score": float(np.mean(block_rms)) if n_blocks else 0.0}]

    prefix = np.concatenate([[0.0], np.cumsum(block_rms)])
    n_starts = n_blocks - blocks_per_window + 1
    window_sums = prefix[blocks_per_window:blocks_per_window + n_starts] - prefix[0:n_starts]
    window_means = window_sums / blocks_per_window

    order = np.argsort(window_means)[::-1]
    min_sep_blocks = max(1, round(MIN_CANDIDATE_SEPARATION_SEC / HOP_SEC))

    chosen = []
    for start_block in order:
        if any(abs(int(start_block) - c["start_block"]) < min_sep_blocks for c in chosen):
            continue
        chosen.append({"start_block": int(start_block), "score": float(window_means[start_block])})
        if len(chosen) >= top_k:
            break
    return chosen


def extract_candidates_for_file(path, out_prefix, candidates_dir, force=False, top_k=TOP_K):
    """Runs the full pipeline for one source file: energy profile -> ranked
    windows -> trimmed+resampled candidate wavs written to candidates_dir as
    "<out_prefix>_c<rank>.wav". Returns (source_meta, candidates_list)."""
    info = sf.info(path)
    sr = info.samplerate
    total_frames = info.frames
    duration_sec = total_frames / sr if sr else 0.0

    hop_frames = max(1, int(round(sr * HOP_SEC)))
    window_frames = int(round(sr * WINDOW_SEC))
    short_file = duration_sec <= WINDOW_SEC

    block_rms = compute_block_rms(path, sr, hop_frames)
    windows = find_candidate_windows(block_rms, hop_frames, window_frames,
                                      top_k=1 if short_file else top_k)

    candidates = []
    for rank, w in enumerate(windows):
        start_frame = w["start_block"] * hop_frames
        if short_file:
            start_frame = 0
            stop_frame = total_frames
        else:
            stop_frame = min(total_frames, start_frame + window_frames)
            start_frame = max(0, stop_frame - window_frames)  # keep full 5s if near EOF

        out_name = f"{out_prefix}_c{rank}.wav"
        out_path = os.path.join(candidates_dir, out_name)

        if force or not os.path.exists(out_path):
            y, read_sr = sf.read(path, start=start_frame, stop=stop_frame, dtype="float32",
                                  always_2d=True)
            y = y.mean(axis=1)  # mono mixdown
            if read_sr != CANDIDATE_SR:
                y = resample(y, read_sr, CANDIDATE_SR)
            sf.write(out_path, y, CANDIDATE_SR, subtype="PCM_16")
        else:
            y = None

        if y is None:
            y_existing, _ = sf.read(out_path, dtype="float32")
            peak = float(np.max(np.abs(y_existing))) if len(y_existing) else 0.0
            rms = float(np.sqrt(np.mean(y_existing ** 2))) if len(y_existing) else 0.0
        else:
            peak = float(np.max(np.abs(y))) if len(y) else 0.0
            rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0

        rms_dbfs = 20 * np.log10(rms + 1e-12)

        candidates.append({
            "rank": rank,
            "filename": out_name,
            "start_sec": round(start_frame / sr, 2),
            "end_sec": round(stop_frame / sr, 2),
            "energy_score": round(w["score"], 6),
            "peak": round(peak, 4),
            "rms_dbfs": round(rms_dbfs, 1),
            "low_energy": bool(rms_dbfs < LOW_ENERGY_DBFS),
        })

    source_meta = {
        "orig_duration_sec": round(duration_sec, 2),
        "orig_samplerate": sr,
        "orig_channels": info.channels,
        "short_file": short_file,
    }
    return source_meta, candidates
