# -*- coding: utf-8 -*-
r"""
===================================================================
Preprocessing v3 - Belt / Brake / Sway audio classification
Matches the official project proposal (sections 3-8), except Future Work.
===================================================================

WHAT THIS SCRIPT DOES:
  1) Loads metadata for all 3 classes (belt, brake, sway) from Excel files
  2) Matches each Excel row to a real .wav file (number-based matching,
     robust to naming differences / typos)
  3) Standardizes every recording: mono, 22.05kHz, trims silence,
     normalizes volume, pads/trims to EXACTLY 5 seconds (per proposal)
  4) Flags and removes obviously corrupted recordings
  5) Splits files into Train / Validation / Test (stratified, 70/15/15)
     BEFORE any augmentation, so no augmented copy of a test/val file
     ever leaks into training.
  6) Applies data augmentation (time shift, volume change, light noise)
     ONLY to the training set, tripling its size.
  7) Extracts TWO feature representations for every clip (original +
     augmented):
       - Method 1: MFCC-based feature vector -> for traditional ML
       - Method 2: Log-Mel spectrogram -> for the CNN
  8) Saves everything ready to train both model types.

OUTPUT FILES (inside processed_data/):
  features.csv              -> full MFCC feature table (train+val+test, incl. augmented)
  X_train_mfcc.npy / X_val_mfcc.npy / X_test_mfcc.npy   -> scaled MFCC features
  X_train_mel.npy / X_val_mel.npy / X_test_mel.npy       -> log-mel spectrograms (for CNN)
  y_train.npy / y_val.npy / y_test.npy                   -> encoded labels (shared by both methods)
  scaler.pkl                -> StandardScaler fit on training MFCC features
  mel_stats.pkl             -> mean/std used to normalize mel spectrograms
  label_encoder.pkl         -> class name <-> number mapping
  config.pkl                -> all settings, needed later by predict.py
  corrupted_files.csv       -> files that were excluded because they could not be read properly
  unmatched_files.csv       -> excel rows that had no matching real audio file
  dataset_documentation.csv -> FULL metadata (brand, model, year, fuel type, body type,
                                description, risk level, recording condition, source, duration...)
                                for every matched file + which split it landed in. Documentation
                                only -- the models never read this file.

===================================================================
HOW TO RUN (Windows):
===================================================================
1) Install libraries (once):
     pip install pandas numpy librosa soundfile scikit-learn openpyxl matplotlib tqdm

2) Put this file inside C:\Users\hp\Desktop\car_test\preprocessing.py, with this layout:
     car_test\belt\belt-cut\   (belt_*.xlsx metadata + belt_*_cut.wav audio, together)
     car_test\brake\           (brake_*.xlsx metadata + brake audio files, together)
     car_test\sway\sway-cut\   (sway_*.xlsx metadata + sway_*_cut.wav audio, together)

3) Run:
     cd C:\Users\hp\Desktop\car_test
     python preprocessing.py

If your folders are somewhere else, edit TEST_BASE_DIR and/or the "audio_dir" paths
inside CLASSES below -- the excel file for each class is auto-detected from whichever
folder is listed as that class's audio_dir, so it just needs to sit in the same folder.
===================================================================
"""

import os
import re
import glob
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from audio_common import (
    load_clean_audio as _shared_load_clean_audio,
    compute_shared_spectra,
    extract_mfcc_vector as _shared_extract_mfcc_vector,
    extract_log_mel as _shared_extract_log_mel,
    build_mfcc_feature_names,
)
from compression_augment import compress_roundtrip, DEFAULT_BITRATES_KBPS
from mic_response_augment import simulate_mic_response

# ===================== SETTINGS - edit here if needed =====================

# Both overridable via env vars (unset = exact same hardcoded defaults as always) so a
# driver script (e.g. run_augmentation_ablation.py) can point a subprocess at an isolated
# output folder while still reading the real audio from its real location, without ever
# editing this file. CAR_TEST_RAW_DATA_DIR = where belt/brake/sway audio actually live;
# CAR_TEST_OUTPUT_DIR = where processed_data/ gets written.
TEST_BASE_DIR = os.environ.get("CAR_TEST_RAW_DATA_DIR", r"C:\Users\hp\Desktop\car_test")

CLASSES = [
    {
        "name": "belt",
        "audio_dir": os.path.join(TEST_BASE_DIR, "belt", "belt-cut"),
        "excel_path": None,  # auto-detected below (any xlsx inside that same folder)
    },
    {
        "name": "brake",
        "audio_dir": os.path.join(TEST_BASE_DIR, "brake"),
        "excel_path": None,  # auto-detected below (any xlsx inside that same folder)
    },
    {
        "name": "sway",
        "audio_dir": os.path.join(TEST_BASE_DIR, "sway", "sway-cut"),
        "excel_path": None,  # auto-detected below (any xlsx inside that same folder)
    },
]

OUTPUT_DIR = os.environ.get("CAR_TEST_OUTPUT_DIR", os.path.join(TEST_BASE_DIR, "processed_data"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 22050          # unified sample rate (matches proposal's suggested 22.05kHz)
TARGET_DURATION = 5.0       # EXACTLY 5 seconds, as required by the proposal
N_MFCC = 40
N_MELS = 128
HOP_LENGTH = 512

TEST_SIZE = 0.15
VAL_SIZE = 0.15             # fraction of the ORIGINAL data (not of train)
RANDOM_STATE = 42

# how many augmented copies to create per TRAINING file (val/test are never augmented).
# Overridable via the PREPROCESSING_N_AUGMENTATIONS env var (rather than just editing this
# constant directly) so a driver script (e.g. run_augmentation_ablation.py) can control it
# per run even though the actual per-file work happens in separate worker processes --
# each worker re-imports this module fresh on spawn, so a monkeypatched module attribute
# set only in the parent process would silently be ignored by the workers, but an
# inherited environment variable is seen identically by parent and workers alike.
N_AUGMENTATIONS_PER_TRAIN_FILE = int(os.environ.get("PREPROCESSING_N_AUGMENTATIONS", 2))

# FAST MODE: set to True when you only want to re-tune/re-run the TRADITIONAL ML
# models (SVM/RF/XGBoost) and don't need the CNN's log-mel spectrograms this time.
# Skips ~half the per-file work (no log-mel extraction, no SpecAugment), so
# preprocessing.py finishes noticeably faster. Leave False for a normal full run
# (needed before train_cnn.py, since it reads X_train_mel.npy etc.).
FAST_MODE_SKIP_MEL = os.environ.get("PREPROCESSING_FAST_MODE_SKIP_MEL", "0") == "1"

# When a recording (after silence trimming) is longer than TARGET_DURATION, augmented
# TRAIN copies take a random 5s window instead of always the same start-to-5s window
# (free extra variety, since it's essentially zero extra computation). The ORIGINAL
# unaugmented sample, and ALL of val/test, always keep the deterministic start-crop --
# so evaluation stays exactly as reproducible as before.
RANDOM_CROP_ENABLED = True

MIN_VALID_DURATION_SEC = 0.3  # anything shorter than this after trimming = considered corrupted

# Loading + feature extraction (librosa) is entirely CPU-bound and was previously done
# one file at a time on a single core. This runs it across a process pool instead --
# the single biggest speedup available for this script on a multi-core machine.
# Set to 1 to force the old strictly-sequential behavior (e.g. for easier debugging).
N_WORKERS = int(os.environ.get("PREPROCESSING_N_WORKERS", max(1, (os.cpu_count() or 2) - 1)))

# SpecAugment (applied ONLY to the log-mel spectrograms of the TRAIN split, on top of the
# waveform-level augmentation above -- masks random time/frequency bands so the CNN doesn't
# overfit to any single narrow band or moment in the clip). Never applied to val/test.
SPEC_AUGMENT_ENABLED = True
SPEC_AUGMENT_FREQ_MASKS = 1        # how many frequency bands to mask
SPEC_AUGMENT_TIME_MASKS = 1        # how many time bands to mask
SPEC_AUGMENT_FREQ_MASK_PARAM = 16  # max width (in mel bins) of each frequency mask
SPEC_AUGMENT_TIME_MASK_PARAM = 24  # max width (in time frames) of each time mask

# =============================================================================


def resolve_excel_path(folder, class_name):
    """Finds the metadata .xlsx file that lives in the SAME folder as the audio files
    for this class (works for belt/brake/sway alike -- no need to hard-code the exact
    excel filename, since naming varies a bit: 'belt_info.xlsx', 'brake_info', etc.)."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Audio folder not found for class '{class_name}': {folder}")
    candidates = glob.glob(os.path.join(folder, "*.xlsx")) + glob.glob(os.path.join(folder, "*.xls"))
    # prefer a file whose name actually mentions the class or 'info', if more than one xlsx exists
    preferred = [c for c in candidates if class_name in os.path.basename(c).lower()
                 or "info" in os.path.basename(c).lower()]
    candidates = preferred or candidates
    if not candidates:
        raise FileNotFoundError(
            f"No .xlsx/.xls metadata file found inside {folder} for class '{class_name}'. "
            f"Make sure the excel file is in the same folder as the audio for this class."
        )
    chosen = candidates[0]
    print(f"   -> auto-detected '{class_name}' metadata file: {chosen}")
    return chosen


def build_vehicle_groups(df):
    """
    Build a 'group id' per row for group-aware splitting.
    Rows with real, specific brand+model+year get grouped together
    (so all their files stay in ONE partition, preventing leakage).
    Rows with missing/placeholder values ('unknown', 'nan', empty) do
    NOT get merged with each other - each becomes its own singleton
    group, since sharing the word 'Unknown' does not mean they are
    actually the same vehicle.
    """
    def clean(value):
        s = str(value).strip().lower()
        if s in ["", "nan", "none", "unknown", "unkown", "n/a"]:
            return None
        return s

    groups = []
    for idx, row in df.reset_index(drop=True).iterrows():
        brand = clean(row.get("brand"))
        model = clean(row.get("model"))
        year = clean(row.get("year"))
        if brand and model and year:
            groups.append(f"{row['class']}_{brand}_{model}_{year}")
        else:
            groups.append(f"__unique_{row['class']}_{idx}")  # no grouping constraint
    return groups


def greedy_balanced_group_split(df, group_col="vehicle_group", class_col="class",
                                 ratios=(0.70, 0.15, 0.15), split_names=("train", "val", "test"),
                                 seed=42):
    """
    Assigns whole vehicle-groups to train/val/test such that:
      - every file belonging to the same real vehicle stays in ONE partition (no leakage)
      - the proportion of each class in each partition stays close to the target ratios
    This is done per-class independently (since every group belongs to exactly one class),
    using a greedy 'fill the most underfilled bucket' strategy.
    """
    rng = random.Random(seed)
    assignment = {}

    for cls in df[class_col].unique():
        cls_df = df[df[class_col] == cls]
        group_sizes = list(cls_df.groupby(group_col).size().items())
        rng.shuffle(group_sizes)
        group_sizes.sort(key=lambda x: -x[1])  # largest groups placed first for better balance

        total = sum(size for _, size in group_sizes)
        targets = {name: total * ratio for name, ratio in zip(split_names, ratios)}
        current = {name: 0 for name in split_names}

        for group_id, size in group_sizes:
            # send this group to whichever split is currently furthest below its target
            deficits = {name: targets[name] - current[name] for name in split_names}
            chosen = max(deficits, key=deficits.get)
            assignment[group_id] = chosen
            current[chosen] += size

    return assignment


def extract_number(filename):
    match = re.search(r"(\d+)", str(filename))
    return int(match.group(1)) if match else None


def build_file_index(folder):
    index, duplicates = {}, []
    for fname in os.listdir(folder):
        if fname.lower().endswith(".wav"):
            num = extract_number(fname)
            if num is None:
                continue
            if num in index:
                duplicates.append((num, index[num], fname))
            else:
                index[num] = os.path.join(folder, fname)
    if duplicates:
        print(f"   WARNING: {len(duplicates)} duplicate numbers found in {folder}")
    return index


def standardize_columns(df):
    rename_map = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "_")
        if key in ["gas/diesel", "gas_diesel"]:
            rename_map[col] = "fuel_type"
        elif key in ["file_name", "filename"]:
            rename_map[col] = "file_name"
        elif key in ["discribtion", "description"]:
            rename_map[col] = "description"
        elif key in ["reference", "refrence"]:
            rename_map[col] = "reference"
        else:
            rename_map[col] = key
    df = df.rename(columns=rename_map)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
    return df


def load_all_metadata():
    print("\n[1] Loading Excel metadata for all classes...")
    all_rows = []
    for cfg in CLASSES:
        excel_path = cfg["excel_path"] or resolve_excel_path(cfg["audio_dir"], cfg["name"])
        df = pd.read_excel(excel_path)
        df = standardize_columns(df)
        df = df.drop(columns=[c for c in ["reference", "source", "note"] if c in df.columns], errors="ignore")
        df["class"] = cfg["name"]
        df["audio_dir"] = cfg["audio_dir"]
        print(f"   - {cfg['name']}: {len(df)} rows loaded from {excel_path}")
        all_rows.append(df)
    combined = pd.concat(all_rows, ignore_index=True, sort=False)
    return combined


def match_all_files(df):
    print("\n[2] Matching Excel rows to real audio files (number-based matching)...")
    matched_paths, matched_mask = [], []

    for class_name in df["class"].unique():
        sub_idx = df["class"] == class_name
        folder = df.loc[sub_idx, "audio_dir"].iloc[0]
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Audio folder not found for class '{class_name}': {folder}")
        file_index = build_file_index(folder)
        print(f"   - {class_name}: {len(file_index)} real .wav files found in {folder}")

        n_matched = 0
        for fname in df.loc[sub_idx, "file_name"]:
            num = extract_number(fname)
            if num is not None and num in file_index:
                matched_paths.append(file_index[num])
                matched_mask.append(True)
                n_matched += 1
            else:
                matched_paths.append(None)
                matched_mask.append(False)
        print(f"   - {class_name}: matched {n_matched} / {sub_idx.sum()} rows")

    df = df.copy()
    df["matched_path"] = matched_paths
    matched_df = df[[m for m in matched_mask]].reset_index(drop=True)
    unmatched_df = df[[not m for m in matched_mask]].reset_index(drop=True)

    if len(unmatched_df) > 0:
        unmatched_df.to_csv(os.path.join(OUTPUT_DIR, "unmatched_files.csv"), index=False, encoding="utf-8-sig")
        print(f"   -> {len(unmatched_df)} unmatched rows saved to unmatched_files.csv")

    return matched_df


def load_variable_length_audio(path, target_sr=TARGET_SR):
    """Load, mono, resample, trim silence, normalize -- but do NOT yet fix the
    duration to 5 seconds. Kept separate from fit_to_length() so that training
    can optionally take a different random 5s window per augmented copy (see
    RANDOM_CROP_ENABLED) instead of always the same fixed start-to-5s window.
    Thin wrapper around audio_common.load_clean_audio (shared with predict.py etc.):
    returns None (instead of falling back to untrimmed audio) if the trimmed clip
    is too short -- this script's original corrupted-file behavior."""
    y, _ = _shared_load_clean_audio(
        path, target_sr=target_sr, target_duration=None,
        min_valid_duration=MIN_VALID_DURATION_SEC, fallback_to_untrimmed=False,
    )
    return y  # None if considered corrupted / empty recording


def fit_to_length(y, target_sr=TARGET_SR, target_duration=TARGET_DURATION, random_crop=False):
    """Pads with zeros if shorter than target duration. If longer: crops from the
    start (random_crop=False, the deterministic behavior -- always used for val/test
    and for the original/unaugmented train sample) or from a random offset
    (random_crop=True -- used only for augmented train copies, as a free extra
    source of variety: the network then sees a different 5s slice of a long
    recording across its augmented copies instead of always the exact same slice)."""
    target_len = int(target_sr * target_duration)
    if len(y) > target_len:
        if random_crop:
            max_start = len(y) - target_len
            start = random.randint(0, max_start)
            y = y[start:start + target_len]
        else:
            y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)))
    return y


def load_clean_audio(path, target_sr=TARGET_SR, target_duration=TARGET_DURATION):
    """Convenience wrapper kept for anything that still wants load+fix-length in one
    call -- always uses the deterministic start-crop. The train/val/test processing
    below calls load_variable_length_audio() + fit_to_length() directly instead, so
    it can control random_crop per variant."""
    y = load_variable_length_audio(path, target_sr=target_sr)
    if y is None:
        return None
    return fit_to_length(y, target_sr=target_sr, target_duration=target_duration, random_crop=False)


# ---------------------- data augmentation (train only) ----------------------

def aug_time_shift(y, max_ratio=0.2):
    """Shifts the clip forward/backward in time. Uses zero-padding on the side that
    empties out, NOT np.roll -- a circular roll would wrap the tail of the clip around
    to the front (or vice versa), creating an abrupt, unrealistic click/discontinuity
    right at the seam. Padding with silence instead keeps every local moment of the
    original waveform physically plausible."""
    shift = int(random.uniform(-max_ratio, max_ratio) * len(y))
    if shift == 0:
        return y.copy()
    shifted = np.zeros_like(y)
    if shift > 0:
        shifted[shift:] = y[:len(y) - shift]
    else:
        shifted[:len(y) + shift] = y[-shift:]
    return shifted


def aug_volume_change(y, factor_range=(0.7, 1.3)):
    return y * random.uniform(*factor_range)


def aug_add_noise(y, noise_level=0.001):
    """noise_level was 0.005 originally -- measured empirically (mean L2 shift in the
    MFCC-mean vector across real belt/brake/sway files) to perturb the feature vector
    ~5.5x more than aug_time_shift or aug_volume_change do, which was making augmented
    training samples look like noise-dominated garbage rather than a mild realistic
    variation -- confirmed as the cause of a real accuracy regression when augmentation
    was enabled (93.2%->89.5% test accuracy on the traditional ML model, on otherwise
    identical data/split). 0.001 brings its perturbation magnitude in line with the
    other two techniques instead of dominating them."""
    return y + np.random.normal(0, noise_level, len(y))


# Which augmentation techniques make_augmented_version is allowed to combine.
# Overridable via the PREPROCESSING_AUG_TECHNIQUES env var (comma-separated subset of
# time_shift,volume,noise) -- lets a diagnostic driver compare e.g. "no noise at all"
# vs "all three" without editing this file, using the same worker-process-safe
# mechanism as PREPROCESSING_N_AUGMENTATIONS (see the comment on that constant).
_ALL_AUG_TECHNIQUES = {"time_shift": aug_time_shift, "volume": aug_volume_change, "noise": aug_add_noise}
_enabled_technique_names = os.environ.get("PREPROCESSING_AUG_TECHNIQUES", "time_shift,volume,noise").split(",")
AUGMENTATION_TECHNIQUES = [_ALL_AUG_TECHNIQUES[n] for n in _enabled_technique_names if n in _ALL_AUG_TECHNIQUES]

# Separate from the waveform augmentation above ON PURPOSE -- that one (time-shift/
# volume/noise) is a GENERIC perturbation with no specific real-world justification,
# and was measured to actively HURT the traditional ML model (93.2% -> 89.5%, see
# SESSION_HANDOFF.md section 3). This one is different in kind: it round-trips each
# TRAIN copy through the REAL app's webm/opus + ffmpeg pipeline (compression_augment.py),
# after mic_pipeline_test/test_mic_pipeline_domain_shift.py measured that pipeline
# alone collapsing the deployed model's brake accuracy from 100% to ~20-40%. The goal
# is narrow and evidence-driven: teach the model to ignore compression artifacts it
# currently seems to partly rely on, not "more variety" in general. Independently
# toggleable so it can be tested in isolation from the (already rejected) generic
# augmentation above -- run with PREPROCESSING_N_AUGMENTATIONS=0 to test this alone.
COMPRESSION_AUGMENT_ENABLED = os.environ.get("PREPROCESSING_COMPRESSION_AUGMENT", "0") == "1"
COMPRESSION_BITRATES_KBPS = [int(b) for b in os.environ.get(
    "PREPROCESSING_COMPRESSION_BITRATES", ",".join(str(b) for b in DEFAULT_BITRATES_KBPS)).split(",")]

# Second, independent "audio fingerprint" dimension: mic/device frequency-response
# variation (mic_response_augment.py), as opposed to codec compression above. We don't
# know the recording device behind the original training data (see SESSION_HANDOFF.md
# section 1) or any given end user's phone, so instead of targeting one fingerprint,
# this generates TRAIN-only copies across a wide randomized range of plausible phone-mic
# frequency responses -- each call to simulate_mic_response() picks new random cutoffs,
# so N_MIC_RESPONSE_VARIANTS copies of the same file each get a different simulated
# "device". Independently toggleable from compression augmentation (both can be on at
# once for the final model, or tested in isolation).
MIC_RESPONSE_AUGMENT_ENABLED = os.environ.get("PREPROCESSING_MIC_RESPONSE_AUGMENT", "0") == "1"
N_MIC_RESPONSE_VARIANTS = int(os.environ.get("PREPROCESSING_N_MIC_RESPONSE_VARIANTS", 2))

# Compound variant of the two above: mic-response filter THEN compression, on the SAME
# clip, matching the REAL signal path (a phone mic colors the audio, then the app
# compresses whatever the mic produced -- these two transformations never happen
# independently in reality). Built after compression-only vs compression+mic-as-separate-
# dimensions (compression_augmented vs compression_and_mic_augmented ablation runs) showed
# a real trade-off: training on the two kinds of distortion as SEPARATE variants made the
# model moderately better at each but measurably worse at compression specifically than
# training on compression alone. Hypothesis: training on the actual COMPOUND transformation
# (not two separate ones) should do better at both simultaneously instead of splitting the
# difference. Mutually exclusive in practice with the two flags above -- use this AND set
# PREPROCESSING_COMPRESSION_AUGMENT=0 / PREPROCESSING_MIC_RESPONSE_AUGMENT=0 to test it in
# isolation the same way the other two were tested.
COMPOUND_AUGMENT_ENABLED = os.environ.get("PREPROCESSING_COMPOUND_AUGMENT", "0") == "1"
N_COMPOUND_VARIANTS = int(os.environ.get("PREPROCESSING_N_COMPOUND_VARIANTS", 5))


def make_augmented_version(y):
    """Randomly combine 1 to len(AUGMENTATION_TECHNIQUES) augmentation techniques (per
    proposal: small shifts, small volume changes, light background noise - no pitch
    shifting)."""
    out = y.copy()
    techniques = random.sample(AUGMENTATION_TECHNIQUES, k=random.randint(1, len(AUGMENTATION_TECHNIQUES)))
    for t in techniques:
        out = t(out)
    # re-normalize after augmentation so amplitude stays in a sane range
    if np.max(np.abs(out)) > 0:
        out = out / np.max(np.abs(out))
    return out


def spec_augment(log_mel, n_freq_masks=SPEC_AUGMENT_FREQ_MASKS, n_time_masks=SPEC_AUGMENT_TIME_MASKS,
                  freq_mask_param=SPEC_AUGMENT_FREQ_MASK_PARAM, time_mask_param=SPEC_AUGMENT_TIME_MASK_PARAM):
    """SpecAugment: masks random frequency bands and time bands directly on the
    log-mel image (sets them to the image's own mean, i.e. 'silence' in that band/frame).
    Applied AFTER the waveform-level augmentation, ONLY on the train split, so the CNN
    sees extra variety it cannot get from raw-audio augmentation alone. Never used for
    the MFCC features (those stay based purely on the waveform-level augmentation)."""
    out = log_mel.copy()
    n_mels, n_frames = out.shape
    fill_value = out.mean()

    for _ in range(n_freq_masks):
        f = random.randint(0, min(freq_mask_param, n_mels - 1))
        if f == 0:
            continue
        f0 = random.randint(0, n_mels - f)
        out[f0:f0 + f, :] = fill_value

    for _ in range(n_time_masks):
        t = random.randint(0, min(time_mask_param, n_frames - 1))
        if t == 0:
            continue
        t0 = random.randint(0, n_frames - t)
        out[:, t0:t0 + t] = fill_value

    return out


# ---------------------- feature extraction (both methods, shared STFT) ----------------------
# extract_mfcc_vector, extract_log_mel and build_mfcc_feature_names now live in
# audio_common.py (imported above) so preprocessing.py, predict.py and the
# transfer-learning scripts can never drift out of sync with each other. They
# also now compute the STFT/mel spectrogram ONCE per clip and reuse it for
# every downstream feature, instead of each librosa.feature.* call redoing its
# own STFT independently (verified numerically identical to the old version).


def _process_one_row(args):
    """Runs in a worker process (see N_WORKERS / ProcessPoolExecutor in
    process_split below). Loads + cleans ONE file, generates its augmented
    variants if requested, and extracts MFCC (+ log-mel, unless
    FAST_MODE_SKIP_MEL) features for each variant.

    Pure function of `args` (no shared mutable state) so it is safe to run
    concurrently across processes. Module-level constants (TARGET_SR, N_MFCC,
    ...) are available automatically -- each worker re-imports this module.
    """
    row_dict, augment = args
    matched_path, class_name = row_dict["matched_path"], row_dict["class"]

    y_var = load_variable_length_audio(matched_path)
    if y_var is None:
        return {"corrupted_row": row_dict}

    # original sample: always the deterministic start-crop (identical behavior for
    # both train's own original copy and all of val/test)
    y = fit_to_length(y_var, random_crop=False)

    variants = [(y, False)]
    if augment:
        for _ in range(N_AUGMENTATIONS_PER_TRAIN_FILE):
            # augmented train copies: random crop window (if the source recording is
            # longer than 5s -- else this is a no-op, same as before) followed by the
            # usual waveform augmentations
            cropped = fit_to_length(y_var, random_crop=RANDOM_CROP_ENABLED)
            variants.append((make_augmented_version(cropped), True))
        if COMPRESSION_AUGMENT_ENABLED:
            # one extra TRAIN-only copy per configured bitrate, built from the SAME
            # clean deterministic crop as the original (not the random-cropped/
            # waveform-augmented copies above) -- isolates "does compression-roundtrip
            # help" from the already-separately-tested generic waveform augmentation
            for kbps in COMPRESSION_BITRATES_KBPS:
                variants.append((compress_roundtrip(y, TARGET_SR, kbps), True))
        if MIC_RESPONSE_AUGMENT_ENABLED:
            # same principle, second dimension: randomized simulated mic frequency
            # response instead of codec compression -- also built from the clean
            # deterministic crop, independent of the compression variants above
            for _ in range(N_MIC_RESPONSE_VARIANTS):
                variants.append((simulate_mic_response(y, TARGET_SR), True))
        if COMPOUND_AUGMENT_ENABLED:
            # mic-response filter THEN compression, on the SAME variant -- the real
            # signal path, not two independent perturbations (see comment on
            # COMPOUND_AUGMENT_ENABLED above)
            for _ in range(N_COMPOUND_VARIANTS):
                device_colored = simulate_mic_response(y, TARGET_SR)
                bitrate = random.choice(COMPRESSION_BITRATES_KBPS)
                variants.append((compress_roundtrip(device_colored, TARGET_SR, bitrate), True))

    mfcc_rows, mel_rows, aug_flags = [], [], []
    for variant_y, is_aug in variants:
        shared = compute_shared_spectra(variant_y, TARGET_SR, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mfcc_rows.append(_shared_extract_mfcc_vector(variant_y, TARGET_SR, n_mfcc=N_MFCC, shared=shared))
        if FAST_MODE_SKIP_MEL:
            mel_rows.append(None)
        else:
            mel = _shared_extract_log_mel(variant_y, TARGET_SR, shared=shared)
            if augment and SPEC_AUGMENT_ENABLED:
                mel = spec_augment(mel)
            mel_rows.append(mel)
        aug_flags.append(is_aug)

    return {
        "corrupted_row": None,
        "mfcc_rows": mfcc_rows,
        "mel_rows": mel_rows,
        "label": class_name,
        "file_name": os.path.basename(matched_path),
        "aug_flags": aug_flags,
    }


# =============================================================================


def main():
    print("=" * 70)
    print("Preprocessing v3 - Belt / Brake / Sway classification")
    print(f"Target duration: {TARGET_DURATION}s | Sample rate: {TARGET_SR}Hz")
    print("=" * 70)

    meta = load_all_metadata()
    matched = match_all_files(meta)

    if len(matched) == 0:
        print("\n!! No files matched at all. Check your folder paths.")
        return

    print("\n   Class balance (matched files, before removing corrupted ones):")
    print(matched["class"].value_counts().to_string())

    # ---------- split BEFORE augmentation, GROUP-AWARE + CLASS-BALANCED ----------
    print("\n[3] Splitting into Train / Validation / Test (group-aware AND class-balanced)...")
    matched = matched.reset_index(drop=True)
    matched["vehicle_group"] = build_vehicle_groups(matched)

    n_real_groups = (matched["vehicle_group"].value_counts() > 1).sum()
    print(f"   - detected {n_real_groups} real vehicles with more than one recording;"
          f" all their files will be kept together in the same partition")

    ratios = (1 - TEST_SIZE - VAL_SIZE, VAL_SIZE, TEST_SIZE)
    assignment = greedy_balanced_group_split(
        matched, group_col="vehicle_group", class_col="class",
        ratios=ratios, split_names=("train", "val", "test"), seed=RANDOM_STATE
    )
    matched["split_assigned"] = matched["vehicle_group"].map(assignment)

    train_df = matched[matched["split_assigned"] == "train"].reset_index(drop=True)
    val_df = matched[matched["split_assigned"] == "val"].reset_index(drop=True)
    test_df = matched[matched["split_assigned"] == "test"].reset_index(drop=True)

    print(f"   - train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)} (original files, before augmentation)")
    print("   - class balance per split (should now be close to the target ratios):")
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        counts = split_df["class"].value_counts().to_dict()
        print(f"       {split_name}: {counts}")

    # ---------- documentation export (proposal section 3: metadata stored for documentation) ----------
    # This is the FULL metadata table (brand, model, year, fuel type, body type, description,
    # risk level, recording condition, recording source, duration, ...) for every matched file,
    # plus which split it was assigned to. The classifier itself never reads this file -- it
    # exists purely so the dataset is documented and auditable, as required by the proposal.
    doc_cols = [c for c in matched.columns if c not in ("audio_dir",)]
    matched[doc_cols].to_csv(os.path.join(OUTPUT_DIR, "dataset_documentation.csv"),
                             index=False, encoding="utf-8-sig")
    print(f"   - saved full metadata documentation table -> dataset_documentation.csv ({len(matched)} rows)")

    # ---------- process every split ----------
    mfcc_names = build_mfcc_feature_names(n_mfcc=N_MFCC)
    corrupted_rows = []

    def process_split(df, split_name, augment):
        mfcc_rows, mel_rows, labels, file_names, aug_flags = [], [], [], [], []
        desc = f"   processing {split_name}"
        tasks = [(row.to_dict(), augment) for _, row in df.iterrows()]

        if N_WORKERS <= 1 or len(tasks) < 2:
            # sequential fallback (also used for tiny splits, where process-pool
            # startup overhead would outweigh any benefit)
            results = (_process_one_row(t) for t in tasks)
            results = tqdm(results, total=len(tasks), desc=desc)
            results = list(results)
        else:
            results = [None] * len(tasks)
            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = {executor.submit(_process_one_row, t): i for i, t in enumerate(tasks)}
                for future in tqdm(as_completed(futures), total=len(tasks), desc=desc):
                    results[futures[future]] = future.result()

        for res in results:
            if res["corrupted_row"] is not None:
                corrupted_rows.append(res["corrupted_row"])
                continue
            mfcc_rows.extend(res["mfcc_rows"])
            mel_rows.extend(res["mel_rows"])
            aug_flags.extend(res["aug_flags"])
            labels.extend([res["label"]] * len(res["mfcc_rows"]))
            file_names.extend([res["file_name"]] * len(res["mfcc_rows"]))

        return mfcc_rows, mel_rows, labels, file_names, aug_flags

    print(f"\n[4-5] Preprocessing audio + extracting MFCC and Log-Mel features "
          f"(using {N_WORKERS} worker process{'es' if N_WORKERS != 1 else ''})...")
    train_mfcc, train_mel, train_labels, train_files, train_aug = process_split(train_df, "train", augment=True)
    val_mfcc, val_mel, val_labels, val_files, val_aug = process_split(val_df, "validation", augment=False)
    test_mfcc, test_mel, test_labels, test_files, test_aug = process_split(test_df, "test", augment=False)

    if corrupted_rows:
        pd.DataFrame(corrupted_rows).to_csv(os.path.join(OUTPUT_DIR, "corrupted_files.csv"), index=False, encoding="utf-8-sig")
        print(f"\n   WARNING: {len(corrupted_rows)} files were excluded as corrupted/empty (see corrupted_files.csv)")

    print(f"\n   Final sample counts -> train: {len(train_labels)} (incl. augmented) | val: {len(val_labels)} | test: {len(test_labels)}")

    # ---------- save the MFCC feature table (for documentation / inspection) ----------
    all_mfcc = train_mfcc + val_mfcc + test_mfcc
    all_labels = train_labels + val_labels + test_labels
    all_files = train_files + val_files + test_files
    all_aug = train_aug + val_aug + test_aug
    all_split = ["train"] * len(train_labels) + ["val"] * len(val_labels) + ["test"] * len(test_labels)

    features_df = pd.DataFrame(all_mfcc, columns=mfcc_names)
    features_df["label"] = all_labels
    features_df["file_name"] = all_files
    features_df["augmented"] = all_aug
    features_df["split"] = all_split
    features_df.to_csv(os.path.join(OUTPUT_DIR, "features.csv"), index=False, encoding="utf-8-sig")
    print(f"   saved features.csv with shape {features_df.shape}")

    # ---------- encode labels ----------
    print("\n[6] Encoding labels...")
    le = LabelEncoder()
    le.fit(all_labels)
    print(f"   - classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    y_train = le.transform(train_labels)
    y_val = le.transform(val_labels)
    y_test = le.transform(test_labels)

    # ---------- MFCC: scale using train statistics only ----------
    print("\n[7] Scaling MFCC features (StandardScaler fit on train only)...")
    scaler = StandardScaler()
    X_train_mfcc = scaler.fit_transform(np.array(train_mfcc))
    X_val_mfcc = scaler.transform(np.array(val_mfcc))
    X_test_mfcc = scaler.transform(np.array(test_mfcc))

    # ---------- Mel spectrograms: normalize using train statistics only ----------
    if FAST_MODE_SKIP_MEL:
        print("[8] FAST_MODE_SKIP_MEL is ON -- skipping log-mel spectrograms entirely.")
        print("    (X_*_mel.npy, mel_stats.pkl NOT written this run -- train_cnn.py needs a")
        print("     normal run with FAST_MODE_SKIP_MEL=False before/after this to work.)")
    else:
        print("[8] Normalizing log-mel spectrograms (train statistics only)...")
        X_train_mel = np.stack(train_mel)
        X_val_mel = np.stack(val_mel)
        X_test_mel = np.stack(test_mel)

        mel_mean = X_train_mel.mean()
        mel_std = X_train_mel.std()
        X_train_mel = (X_train_mel - mel_mean) / (mel_std + 1e-8)
        X_val_mel = (X_val_mel - mel_mean) / (mel_std + 1e-8)
        X_test_mel = (X_test_mel - mel_mean) / (mel_std + 1e-8)

        # add channel dimension for CNN: (N, n_mels, time_frames, 1)
        X_train_mel = X_train_mel[..., np.newaxis]
        X_val_mel = X_val_mel[..., np.newaxis]
        X_test_mel = X_test_mel[..., np.newaxis]

        print(f"   - mel spectrogram shape per sample: {X_train_mel.shape[1:]}")

    # ---------- save everything ----------
    print("\n[9] Saving all outputs to processed_data/ ...")
    np.save(os.path.join(OUTPUT_DIR, "X_train_mfcc.npy"), X_train_mfcc)
    np.save(os.path.join(OUTPUT_DIR, "X_val_mfcc.npy"), X_val_mfcc)
    np.save(os.path.join(OUTPUT_DIR, "X_test_mfcc.npy"), X_test_mfcc)

    if not FAST_MODE_SKIP_MEL:
        np.save(os.path.join(OUTPUT_DIR, "X_train_mel.npy"), X_train_mel)
        np.save(os.path.join(OUTPUT_DIR, "X_val_mel.npy"), X_val_mel)
        np.save(os.path.join(OUTPUT_DIR, "X_test_mel.npy"), X_test_mel)
    else:
        print("   (skipped writing X_*_mel.npy -- FAST_MODE_SKIP_MEL is on. Any existing "
              "mel files from a previous full run are left untouched, NOT refreshed.)")

    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_val.npy"), y_val)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUTPUT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    if not FAST_MODE_SKIP_MEL:
        with open(os.path.join(OUTPUT_DIR, "mel_stats.pkl"), "wb") as f:
            pickle.dump({"mean": mel_mean, "std": mel_std}, f)
        mel_shape = X_train_mel.shape[1:]
    else:
        mel_shape = None  # not computed this run -- config.pkl keeps whatever was last valid, if any

    config = {
        "target_sr": TARGET_SR,
        "target_duration": TARGET_DURATION,
        "n_mfcc": N_MFCC,
        "n_mels": N_MELS,
        "hop_length": HOP_LENGTH,
        "mel_shape": mel_shape,
        "classes": list(le.classes_),
        # dim of the traditional-ML feature vector this scaler/model pair was fit on --
        # predict.py checks a freshly-extracted vector against this before scaler.transform,
        # so a future feature_extraction.py change that isn't matched by retraining fails
        # loudly instead of silently mis-scaling every prediction.
        "mfcc_feature_dim": int(X_train_mfcc.shape[1]),
    }
    config_path = os.path.join(OUTPUT_DIR, "config.pkl")
    if FAST_MODE_SKIP_MEL and os.path.exists(config_path):
        # don't clobber a previously-saved, correct mel_shape with None
        with open(config_path, "rb") as f:
            old_config = pickle.load(f)
        config["mel_shape"] = old_config.get("mel_shape", None)
    with open(config_path, "wb") as f:
        pickle.dump(config, f)

    print("\n" + "=" * 70)
    print("DONE! Everything is saved in:")
    print(f"   {OUTPUT_DIR}")
    print("=" * 70)
    if FAST_MODE_SKIP_MEL:
        print("""
FAST_MODE_SKIP_MEL was ON: only MFCC-based features were refreshed (fast).
Next step: run train_traditional_ml.py.
(train_cnn.py / train_transfer_learning.py need a normal run first -- set
FAST_MODE_SKIP_MEL=False and re-run preprocessing.py before using them.)
""")
    else:
        print("""
Next step: run train_traditional_ml.py and train_cnn.py to train both model types.
""")


if __name__ == "__main__":
    main()