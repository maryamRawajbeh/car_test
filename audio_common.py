# -*- coding: utf-8 -*-
r"""
===================================================================
Shared audio loading + feature extraction
===================================================================
Used by preprocessing.py, predict.py, train_transfer_learning.py and
train_panns.py so all of them stay numerically identical by
construction, instead of each keeping its own hand-copied version of
these functions (which previously had a comment warning "must stay
IDENTICAL to preprocessing.py's" -- a manual promise that's easy to
break silently).

Also computes the STFT ONCE per clip and reuses it for every
downstream feature (MFCC, deltas, spectral stats, log-mel), instead
of each librosa.feature.* call recomputing its own STFT/mel
spectrogram independently. Verified numerically identical (bit-exact
on synthetic audio) to the original independent-calls version.
"""

import random

import numpy as np
import librosa


def aug_time_shift(y, max_shift_frac=0.1):
    """Copied verbatim from train_transfer_learning.py (YAMNet) for exact
    numerical consistency across every script that uses it. Zero-pads the
    vacated side instead of np.roll, which would otherwise wrap the clip's
    tail/head around and create an unrealistic click at the seam."""
    shift = int(len(y) * random.uniform(-max_shift_frac, max_shift_frac))
    if shift == 0:
        return y.copy()
    shifted = np.zeros_like(y)
    if shift > 0:
        shifted[shift:] = y[:len(y) - shift]
    else:
        shifted[:len(y) + shift] = y[-shift:]
    return shifted


def aug_volume_change(y, low=0.7, high=1.3):
    """Copied verbatim from train_transfer_learning.py (YAMNet)."""
    return y * random.uniform(low, high)


def aug_add_noise(y, noise_factor=0.001):
    """Copied verbatim from train_transfer_learning.py (YAMNet). 0.001 (not
    0.005) -- see that file's comment: 0.005 was empirically ~5.5x more
    disruptive to the feature vector than the other two augmentation
    techniques, and measurably hurt the traditional ML model's accuracy."""
    noise = np.random.randn(len(y)).astype(np.float32)
    return y + noise_factor * noise


def make_augmented_version(y):
    """Copied verbatim from train_transfer_learning.py (YAMNet): randomly
    combine 1-3 light augmentations (no pitch shifting -- that could change
    the diagnostic character of the sound)."""
    out = y.copy()
    techniques = random.sample([aug_time_shift, aug_volume_change, aug_add_noise], k=random.randint(1, 3))
    for t in techniques:
        out = t(out)
    if np.max(np.abs(out)) > 0:
        out = out / np.max(np.abs(out))
    return out.astype(np.float32)


def load_clean_audio(path, target_sr, target_duration=None, top_db=25,
                      min_valid_duration=None, fallback_to_untrimmed=False):
    """Load mono audio at target_sr, trim silence, peak-normalize to [-1, 1].

    target_duration : if given, pad (zeros at the end) or trim to exactly
        this many seconds.
    min_valid_duration : if given and the TRIMMED audio is shorter than this
        many seconds, returns (None, sr) -- caller treats this as a
        corrupted/empty recording (used by preprocessing.py to exclude
        near-silent files).
    fallback_to_untrimmed : if True and trimming removes the entire clip
        (e.g. an extremely quiet recording), falls back to the untrimmed
        audio instead of continuing with an empty array. Used by callers
        that must always return *something* for a single real-world file
        (predict.py, transfer-learning scripts) rather than dropping it.
    """
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db)

    if min_valid_duration is not None and len(y_trimmed) / sr < min_valid_duration:
        return None, sr

    if len(y_trimmed) == 0 and fallback_to_untrimmed:
        y_trimmed = y

    peak = np.max(np.abs(y_trimmed)) if len(y_trimmed) else 0.0
    if peak > 0:
        y_trimmed = y_trimmed / peak

    if target_duration is not None:
        target_len = int(sr * target_duration)
        if len(y_trimmed) > target_len:
            y_trimmed = y_trimmed[:target_len]
        else:
            y_trimmed = np.pad(y_trimmed, (0, target_len - len(y_trimmed)))

    return y_trimmed, sr


def compute_shared_spectra(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Computes the magnitude STFT and the (linear-power) mel spectrogram
    ONCE, for reuse by both extract_mfcc_vector and extract_log_mel below.

    Returns (S_mag, mel_power):
      S_mag     -- |STFT(y)|, the magnitude spectrogram (power=1). This is
                   exactly what spectral_centroid/bandwidth/rolloff/contrast
                   expect when given an S= argument.
      mel_power -- linear-power mel spectrogram (mel_basis . S_mag**2),
                   exactly what melspectrogram(y=y, ...) would have produced.
    """
    S_mag = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    mel_power = librosa.feature.melspectrogram(S=S_mag ** 2, sr=sr, n_mels=n_mels)
    return S_mag, mel_power


def extract_mfcc_vector(y, sr, n_mfcc=40, n_mels=128, n_fft=2048, hop_length=512, shared=None):
    """Feature vector for the traditional ML models (SVM/RF/XGBoost):
      - MFCC mean+std, MFCC delta mean+std, MFCC delta-delta mean+std
      - spectral contrast mean+std
      - zero crossing rate, spectral centroid/bandwidth/rolloff, RMS (mean+std each)
      - HPSS (harmonic-percussive source separation) energy ratio -- belt squeal is a
        tonal/harmonic sound, sway clunks are impulsive/percussive; splitting the clip
        into its harmonic and percussive components and comparing their energy gives a
        direct, physically-motivated signal for exactly this distinction. This is the
        currently-deployed 269-dim feature vector (scaler.pkl / best_traditional_model.pkl
        were fit on exactly this, HPSS included) -- do not drop it without retraining.

    `shared`, if given, is the (S_mag, mel_power) tuple from
    compute_shared_spectra(y, sr, ...) for this same y -- avoids
    recomputing the STFT/mel spectrogram from scratch. HPSS is NOT part of
    `shared`: librosa.effects.hpss needs its own internal STFT->mask->ISTFT
    round trip back to the time domain, so it can't reuse S_mag directly
    without changing the actual numbers produced (and thus silently breaking
    compatibility with the already-fitted scaler/model).
    """
    if shared is None:
        shared = compute_shared_spectra(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_mag, mel_power = shared

    # NOTE: ref=1.0 (power_to_db's default) here on purpose -- this must match
    # what librosa.feature.mfcc(y=y, ...) computes internally
    # (power_to_db(melspectrogram(...)), no ref=np.max). extract_log_mel below
    # uses ref=np.max instead, since that's a DIFFERENT feature (CNN input),
    # not interchangeable with this one despite sharing the same mel_power.
    log_mel_for_mfcc = librosa.power_to_db(mel_power)
    mfcc = librosa.feature.mfcc(S=log_mel_for_mfcc, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(sr=sr, S=S_mag)
    bandwidth = librosa.feature.spectral_bandwidth(sr=sr, S=S_mag)
    rolloff = librosa.feature.spectral_rolloff(sr=sr, S=S_mag)
    contrast = librosa.feature.spectral_contrast(sr=sr, S=S_mag)
    rms = librosa.feature.rms(y=y)  # kept on y: rms(S=...) uses a different (unwindowed) power convention

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_rms = librosa.feature.rms(y=y_harmonic)
    percussive_rms = librosa.feature.rms(y=y_percussive)
    harmonic_energy = np.mean(harmonic_rms)
    percussive_energy = np.mean(percussive_rms)
    harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-8)

    def mean_std(feat, axis=1):
        return np.mean(feat, axis=axis), np.std(feat, axis=axis)

    mfcc_mean, mfcc_std = mean_std(mfcc)
    d1_mean, d1_std = mean_std(mfcc_delta)
    d2_mean, d2_std = mean_std(mfcc_delta2)
    contrast_mean, contrast_std = mean_std(contrast)

    scalars = []
    for feat in (zcr, centroid, bandwidth, rolloff, rms):
        scalars.append(np.mean(feat))
        scalars.append(np.std(feat))

    hpss_scalars = [
        harmonic_energy, np.std(harmonic_rms),
        percussive_energy, np.std(percussive_rms),
        harmonic_ratio,
    ]

    return np.concatenate([
        mfcc_mean, mfcc_std,
        d1_mean, d1_std,
        d2_mean, d2_std,
        contrast_mean, contrast_std,
        scalars,
        hpss_scalars,
    ])


def extract_log_mel(y, sr, n_mels=128, n_fft=2048, hop_length=512, shared=None):
    """Log-mel spectrogram (image) for the CNN, normalized relative to its
    own peak (ref=np.max) -- NOT the same dB reference as the MFCC path."""
    if shared is None:
        shared = compute_shared_spectra(y, sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    _, mel_power = shared
    return librosa.power_to_db(mel_power, ref=np.max)


def build_mfcc_feature_names(n_mfcc=40, n_contrast_bands=7):
    names = [f"mfcc_mean_{i+1}" for i in range(n_mfcc)]
    names += [f"mfcc_std_{i+1}" for i in range(n_mfcc)]
    names += [f"mfcc_delta_mean_{i+1}" for i in range(n_mfcc)]
    names += [f"mfcc_delta_std_{i+1}" for i in range(n_mfcc)]
    names += [f"mfcc_delta2_mean_{i+1}" for i in range(n_mfcc)]
    names += [f"mfcc_delta2_std_{i+1}" for i in range(n_mfcc)]
    names += [f"spectral_contrast_mean_{i+1}" for i in range(n_contrast_bands)]
    names += [f"spectral_contrast_std_{i+1}" for i in range(n_contrast_bands)]
    for base in ["zcr", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "rms"]:
        names += [f"{base}_mean", f"{base}_std"]
    names += ["hpss_harmonic_energy_mean", "hpss_harmonic_energy_std",
              "hpss_percussive_energy_mean", "hpss_percussive_energy_std",
              "hpss_harmonic_ratio"]
    return names
