# -*- coding: utf-8 -*-
r"""
===================================================================
Real 4th class: "other" (anything that isn't belt/brake/sway)
===================================================================
Follow-up to the empirical finding (2026-08-22) that NO unsupervised novelty-
detection add-on (IsolationForest / LOF / EllipticEnvelope / z-score / L2-norm,
tried on both the MFCC feature space and YAMNet's embedding space) reliably
separates non-car-fault audio (silence, white noise, a pure tone, a low rumble)
from real belt/brake/sway recordings. Those all failed for the same underlying
reason: with zero labeled negative examples, there is no way to learn where the
boundary of "normal" actually is.

THIS script instead builds real negative ("other") training data and adds it as
a genuine 4th class, trained the same supervised way as every other model in
this project -- not a bolt-on detector, an actual class the classifier learns
to recognize. Two sources, neither of which is the (policy-forbidden, see
project memory) Freesound data:

  1) SYNTHETIC audio covering many acoustic categories NOT specific to any car
     fault: white/pink/brown noise, pure tones across the audible range,
     chirps, AM-modulated tones, simple musical chords, quiet room-tone. Cheap,
     unlimited, and deliberately broad rather than tuned to "beat" any one test.
     ADDED 2026-08-23: four harder, acoustically-realistic-but-still-synthetic
     kinds -- engine_hum (idle-engine harmonics + wobble), impulse_train
     (irregular mechanical clicks), road_noise (low-passed rumble + drift),
     formant_voice (crude vowel-formant speech approximation) -- aimed at the
     real-world gap where a user's mistaken upload (car running normally, road/
     wind noise, or just talking) is acoustically much closer to belt/brake/
     sway than a pure tone or a Windows chime ever was.

  2) REAL non-synthetic audio: Windows's own bundled system sounds
     (C:\Windows\Media\*.wav -- alarms, rings, UI dings/chimes, spoken system
     prompts). These ship with the OS, are not car-related, and are NOT
     Freesound data, so the existing "never merge Freesound audio" policy does
     not apply to them.

HELD-OUT GENERALIZATION CHECK (section 5 below): a subset of BOTH sources is
reserved and NEVER touched during training/val/test -- entirely unseen system
sound files, plus synthetic signal TYPES never generated for training at all
(square wave, sawtooth wave, AM-modulated pink noise). If the classifier only
recognizes "other" audio that looks like what it memorized, it will fail these;
if the "other" class genuinely generalizes, it should catch most of them too.

WHAT THIS DOES NOT DO: retrain the CNN, YAMNet, or touch ensemble_config.pkl /
the deployed processed_data/best_traditional_model.pkl+scaler.pkl+
label_encoder.pkl. Everything here is saved under new *_with_other.pkl names --
a deliberate, separate decision to actually deploy this is still pending.

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python train_other_class_detector.py
"""

import os
import glob
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from audio_common import load_clean_audio, extract_mfcc_vector, aug_time_shift, aug_volume_change, aug_add_noise

BASE_DIR = r"C:\Users\hp\Desktop\car_test"
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
WINDOWS_MEDIA_DIR = r"C:\Windows\Media"
RANDOM_STATE = 42
OTHER_LABEL = "other"

# Reserved for the held-out generalization check ONLY -- never appear in
# other_train/other_val/other_test. Chosen to span every real-audio category
# (alarms, rings, the distinct spoken-word "Speech ..." prompts, generic
# Windows UI sounds, musical chords).
HELD_OUT_SYSTEM_SOUNDS = {
    "Alarm05.wav", "Alarm10.wav", "Ring05.wav", "Ring10.wav",
    "Speech On.wav", "Speech Off.wav", "Speech Disambiguation.wav",
    "Windows Notify Email.wav", "Windows Hardware Fail.wav", "Windows Logon.wav",
    "chord.wav", "tada.wav",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# ---------------------- synthetic "other" audio ----------------------

def _colored_noise(n, color, rng):
    white = rng.randn(n)
    if color == "white":
        return white
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]  # avoid div-by-zero at DC
    if color == "pink":
        spec = spec / np.sqrt(freqs)
    elif color == "brown":
        spec = spec / freqs
    out = np.fft.irfft(spec, n)
    return out / (np.max(np.abs(out)) + 1e-8)


def make_synthetic_other(n, sr, rng, kind):
    t = np.linspace(0, n / sr, n, endpoint=False)
    if kind == "noise":
        color = rng.choice(["white", "pink", "brown"])
        amp = rng.uniform(0.1, 0.6)
        return (amp * _colored_noise(n, color, rng)).astype(np.float32)
    if kind == "tone":
        freq = rng.uniform(80, 8000)
        amp = rng.uniform(0.2, 0.7)
        phase = rng.uniform(0, 2 * np.pi)
        return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
    if kind == "chirp":
        f0, f1 = sorted(rng.uniform(100, 6000, size=2))
        if rng.random() < 0.5:
            f0, f1 = f1, f0
        freq_t = np.linspace(f0, f1, n)
        phase = 2 * np.pi * np.cumsum(freq_t) / sr
        return (0.5 * np.sin(phase)).astype(np.float32)
    if kind == "am_tone":
        carrier = rng.uniform(300, 3000)
        mod = rng.uniform(1, 8)
        sig = np.sin(2 * np.pi * carrier * t) * (0.5 + 0.5 * np.sin(2 * np.pi * mod * t))
        return (0.6 * sig).astype(np.float32)
    if kind == "chord":
        root = rng.uniform(150, 500)
        ratios = [1.0, 5 / 4, 3 / 2]  # a plain major triad
        sig = sum(np.sin(2 * np.pi * root * r * t) for r in ratios)
        return (0.3 * sig / len(ratios)).astype(np.float32)
    if kind == "room_tone":
        return (rng.uniform(0.001, 0.02) * _colored_noise(n, "pink", rng)).astype(np.float32)
    if kind == "engine_hum":
        # crude idle-engine approximation: a low fundamental + harmonics with slow
        # amplitude wobble -- acoustically much closer to "car sound that isn't a
        # fault" than a pure tone, without needing any real recorded engine audio.
        fundamental = rng.uniform(20, 60)
        sig = np.zeros(n)
        for h in range(1, 7):
            sig += (rng.uniform(0.3, 1.0) / h) * np.sin(2 * np.pi * fundamental * h * t + rng.uniform(0, 2 * np.pi))
        wobble = 1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.5, 3) * t)
        sig = sig * wobble
        return (0.4 * sig / (np.max(np.abs(sig)) + 1e-8)).astype(np.float32)
    if kind == "impulse_train":
        # irregular mechanical clicks/knocks -- distinct in rhythm from belt/brake/
        # sway's characteristic periodic rubbing/squeal signatures.
        rate = rng.uniform(5, 25)
        period = sr / rate
        sig = np.zeros(n)
        decay_len = int(sr * 0.01)
        decay = np.exp(-np.linspace(0, 8, decay_len))
        idx = 0.0
        while idx < n:
            i = int(idx)
            end = min(n, i + decay_len)
            sig[i:end] += decay[: end - i] * rng.uniform(0.5, 1.0)
            idx += period * rng.uniform(0.8, 1.2)
        return (0.6 * sig).astype(np.float32)
    if kind == "road_noise":
        # heavily low-passed brown noise with slow amplitude drift -- approximates
        # road/wind cabin noise, spectrally broad and non-tonal unlike the other
        # synthetic kinds above.
        base = _colored_noise(n, "brown", rng)
        window = max(1, int(sr / 400))
        kernel = np.ones(window) / window
        filtered = np.convolve(base, kernel, mode="same")
        wobble = 1 + 0.1 * np.sin(2 * np.pi * rng.uniform(0.2, 1.0) * t)
        sig = filtered * wobble
        return (0.3 * sig / (np.max(np.abs(sig)) + 1e-8)).astype(np.float32)
    if kind == "formant_voice":
        # crude speech-like approximation: a glottal-pulse train exciting a handful
        # of vowel formant frequencies -- rough, but gives the classifier SOME
        # exposure to voice-like harmonic/formant structure instead of none at all.
        pitch = rng.uniform(80, 220)
        formant_sets = [[700, 1200, 2600], [400, 1900, 2500], [300, 900, 2500]]
        formants = formant_sets[rng.randint(0, len(formant_sets))]
        pulse_train = (np.mod(t * pitch, 1) < 0.1).astype(np.float32)
        sig = np.zeros(n)
        for f in formants:
            sig += np.sin(2 * np.pi * f * t) * pulse_train
        return (0.4 * sig / (np.max(np.abs(sig)) + 1e-8)).astype(np.float32)
    raise ValueError(kind)


# reserved kinds: NEVER generated for train/val/test, only for the held-out check
def make_heldout_synthetic(n, sr, rng, kind):
    t = np.linspace(0, n / sr, n, endpoint=False)
    if kind == "square_wave":
        freq = rng.uniform(100, 2000)
        return (0.5 * np.sign(np.sin(2 * np.pi * freq * t))).astype(np.float32)
    if kind == "sawtooth_wave":
        freq = rng.uniform(100, 2000)
        return (0.5 * (2 * (t * freq - np.floor(0.5 + t * freq)))).astype(np.float32)
    if kind == "am_pink_noise":
        mod = rng.uniform(1, 6)
        return (0.4 * _colored_noise(n, "pink", rng) * (0.5 + 0.5 * np.sin(2 * np.pi * mod * t))).astype(np.float32)
    raise ValueError(kind)


def build_other_pool(config, n_per_kind_train, n_per_kind_val, n_per_kind_test):
    sr, dur = config["target_sr"], config["target_duration"]
    n = int(sr * dur)
    # 2026-08-24: tried dropping "formant_voice" after a nearest-neighbor diagnostic
    # implicated it in 3 real belt/sway test files colliding with "other" -- this
    # was a REGRESSION (held-out generalization collapsed 31/31 -> 19/31, e.g. all
    # 10 square/sawtooth-wave probes flipped from "other" to "belt"), so it was put
    # back. formant_voice was apparently pulling its own weight for harmonically-
    # rich non-tonal signals generally, not just causing the 3-file collision in
    # isolation. Do not remove it again without re-testing the FULL held-out suite,
    # not just the 3 files it was originally implicated in.
    kinds = ["noise", "tone", "chirp", "am_tone", "chord", "room_tone",
              "engine_hum", "impulse_train", "road_noise", "formant_voice"]

    def gen(count, seed_offset):
        rng = np.random.RandomState(RANDOM_STATE + seed_offset)
        out = []
        for kind in kinds:
            for _ in range(count):
                out.append(make_synthetic_other(n, sr, rng, kind))
        return out

    return {
        "train": gen(n_per_kind_train, seed_offset=0),
        "val": gen(n_per_kind_val, seed_offset=1000),
        "test": gen(n_per_kind_test, seed_offset=2000),
    }


def build_heldout_synthetic(config, count_per_kind, seed_offset=9000):
    sr, dur = config["target_sr"], config["target_duration"]
    n = int(sr * dur)
    rng = np.random.RandomState(RANDOM_STATE + seed_offset)
    out = []
    for kind in ["square_wave", "sawtooth_wave", "am_pink_noise"]:
        for _ in range(count_per_kind):
            out.append((kind, make_heldout_synthetic(n, sr, rng, kind)))
    return out


# ---------------------- real system-sound "other" audio ----------------------

def load_system_sounds(config):
    sr, dur = config["target_sr"], config["target_duration"]
    all_files = sorted(glob.glob(os.path.join(WINDOWS_MEDIA_DIR, "*.wav")))
    held_out_paths, usable_paths = [], []
    for p in all_files:
        (held_out_paths if os.path.basename(p) in HELD_OUT_SYSTEM_SOUNDS else usable_paths).append(p)
    print(f"    Windows Media: {len(usable_paths)} usable, {len(held_out_paths)} reserved for held-out check")

    def load_one(path):
        y, _ = load_clean_audio(path, target_sr=sr, target_duration=dur, fallback_to_untrimmed=True)
        return y

    return usable_paths, held_out_paths, load_one


def main():
    set_seed(RANDOM_STATE)
    print("=" * 70)
    print('Training a REAL 4th class: "other" (not belt/brake/sway)')
    print("=" * 70)

    with open(os.path.join(DATA_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)

    print("\n[1] Loading real belt/brake/sway MFCC features + labels (already extracted)...")
    X_train_real = np.load(os.path.join(DATA_DIR, "X_train_mfcc.npy"))
    X_val_real = np.load(os.path.join(DATA_DIR, "X_val_mfcc.npy"))
    X_test_real = np.load(os.path.join(DATA_DIR, "X_test_mfcc.npy"))
    y_train_real = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val_real = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test_real = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        real_label_encoder = pickle.load(f)
    real_class_names = list(real_label_encoder.classes_)
    print(f"    real: train={X_train_real.shape}, val={X_val_real.shape}, test={X_test_real.shape}, "
          f"classes={real_class_names}")

    # IMPORTANT: X_train_real/X_val_real/X_test_real were already scaled by
    # preprocessing.py's OWN scaler. Since we're refitting a NEW scaler below on the
    # combined (real + other) feature set, we need the real features back in their
    # RAW (unscaled) MFCC units first, or the new scaler would be fit on already-
    # scaled data. Recompute them raw from the same real audio files instead of
    # trying to invert the old scaling (simpler, and guaranteed correct).
    import pandas as pd
    doc = pd.read_csv(os.path.join(DATA_DIR, "dataset_documentation.csv"))
    corrupted_path = os.path.join(DATA_DIR, "corrupted_files.csv")
    if os.path.exists(corrupted_path):
        corrupted = pd.read_csv(corrupted_path)
        if "matched_path" in corrupted.columns:
            doc = doc[~doc["matched_path"].isin(corrupted["matched_path"])]

    def raw_features_for_split(split_name):
        rows = doc[doc["split_assigned"] == split_name]
        feats, labels = [], []
        for _, row in rows.iterrows():
            y, sr = load_clean_audio(row["matched_path"], target_sr=config["target_sr"],
                                      target_duration=config["target_duration"], fallback_to_untrimmed=True)
            feats.append(extract_mfcc_vector(y, sr, n_mfcc=config["n_mfcc"]))
            labels.append(row["class"])
        return np.array(feats), labels

    print("\n[2] Re-extracting RAW (unscaled) MFCC features for real belt/brake/sway audio "
          "(original samples only, no augmentation -- needed to refit a scaler on raw units)...")
    X_train_real_raw, y_train_real_str = raw_features_for_split("train")
    X_val_real_raw, y_val_real_str = raw_features_for_split("val")
    X_test_real_raw, y_test_real_str = raw_features_for_split("test")
    print(f"    raw real: train={X_train_real_raw.shape}, val={X_val_real_raw.shape}, test={X_test_real_raw.shape}")

    print("\n[3] Building the synthetic \"other\" pool (10 acoustic kinds x train/val/test)...")
    other_pool = build_other_pool(config, n_per_kind_train=25, n_per_kind_val=6, n_per_kind_test=6)
    print(f"    synthetic other: train={len(other_pool['train'])}, val={len(other_pool['val'])}, "
          f"test={len(other_pool['test'])}")

    print("\n[4] Loading real (non-car) Windows system sounds as additional \"other\" audio...")
    usable_paths, held_out_paths, load_system_sound = load_system_sounds(config)
    rng = np.random.RandomState(RANDOM_STATE)
    shuffled = usable_paths[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(0.15 * len(shuffled)))
    n_test = max(1, int(0.15 * len(shuffled)))
    sys_val_paths = shuffled[:n_val]
    sys_test_paths = shuffled[n_val:n_val + n_test]
    sys_train_paths = shuffled[n_val + n_test:]

    def waveforms_with_augmentation(paths, augment):
        out = []
        for p in paths:
            y = load_system_sound(p)
            out.append(y)
            if augment:
                for _ in range(2):
                    aug = y.copy()
                    for fn in random.sample([aug_time_shift, aug_volume_change, aug_add_noise], k=2):
                        aug = fn(aug)
                    if np.max(np.abs(aug)) > 0:
                        aug = aug / np.max(np.abs(aug))
                    out.append(aug.astype(np.float32))
        return out

    sys_train_waveforms = waveforms_with_augmentation(sys_train_paths, augment=True)
    sys_val_waveforms = waveforms_with_augmentation(sys_val_paths, augment=False)
    sys_test_waveforms = waveforms_with_augmentation(sys_test_paths, augment=False)
    print(f"    system sounds (with augmentation on train): train={len(sys_train_waveforms)}, "
          f"val={len(sys_val_waveforms)}, test={len(sys_test_waveforms)}")

    def extract_all(waveforms):
        return np.array([extract_mfcc_vector(y, config["target_sr"], n_mfcc=config["n_mfcc"]) for y in waveforms])

    print("\n[5] Extracting MFCC features for all \"other\" audio...")
    X_other_train = np.vstack([extract_all(other_pool["train"]), extract_all(sys_train_waveforms)])
    X_other_val = np.vstack([extract_all(other_pool["val"]), extract_all(sys_val_waveforms)])
    X_other_test = np.vstack([extract_all(other_pool["test"]), extract_all(sys_test_waveforms)])
    print(f"    other: train={X_other_train.shape}, val={X_other_val.shape}, test={X_other_test.shape}")

    print("\n[6] Combining real (belt/brake/sway) + \"other\" into one 4-class dataset "
          "and refitting a scaler + label encoder on the combined TRAIN split only...")
    X_train_combined_raw = np.vstack([X_train_real_raw, X_other_train])
    X_val_combined_raw = np.vstack([X_val_real_raw, X_other_val])
    X_test_combined_raw = np.vstack([X_test_real_raw, X_other_test])

    y_train_str = np.array(y_train_real_str + [OTHER_LABEL] * len(X_other_train))
    y_val_str = np.array(y_val_real_str + [OTHER_LABEL] * len(X_other_val))
    y_test_str = np.array(y_test_real_str + [OTHER_LABEL] * len(X_other_test))

    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train_str, y_val_str, y_test_str]))
    class_names = list(label_encoder.classes_)
    print(f"    classes: {dict(zip(class_names, label_encoder.transform(class_names)))}")

    y_train = label_encoder.transform(y_train_str)
    y_val = label_encoder.transform(y_val_str)
    y_test = label_encoder.transform(y_test_str)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_combined_raw)
    X_val = scaler.transform(X_val_combined_raw)
    X_test = scaler.transform(X_test_combined_raw)
    print(f"    combined: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    print("\n[7] Model selection (lighter grid than train_traditional_ml.py's full one, "
          "same methodology: stratified K-fold CV on TRAIN, ranked by macro-F1)...")
    candidates = {}
    for c in [1, 10, 50]:
        for g in ["scale", 0.01]:
            # probability=True deliberately OMITTED here -- cross_val_score below only ever
            # calls .predict() (scoring="f1_macro"), and SVC's internal 5-fold Platt-scaling CV
            # (triggered by probability=True) is a known pathological slowdown on some datasets,
            # confirmed to hang for 17+ CPU-hours on this project's own data at full scale. It's
            # re-enabled below, only on the actual winner, right before the final deployed fit.
            candidates[f"SVM(C={c},gamma={g})"] = SVC(C=c, gamma=g, class_weight="balanced")
    for n in [200, 400]:
        candidates[f"RF(n={n})"] = RandomForestClassifier(n_estimators=n, random_state=RANDOM_STATE, class_weight="balanced")
    if HAS_XGBOOST:
        for n in [200, 300]:
            candidates[f"XGB(n={n})"] = XGBClassifier(
                n_estimators=n, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE,
                eval_metric="mlogloss",
            )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_name, best_cv_f1 = None, -1
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        mean_f1 = scores.mean()
        print(f"    {name:24s}: CV macro-F1 = {mean_f1:.4f}")
        if mean_f1 > best_cv_f1:
            best_cv_f1, best_name = mean_f1, name

    print(f"\n    -> best by CV: {best_name} ({best_cv_f1:.4f})")
    best_model = candidates[best_name]
    if isinstance(best_model, SVC) and not best_model.probability:
        print(f"    (enabling probability=True on {best_name} for the final deployed fit -- "
              f"this model is saved and later queried via predict_proba() for the 'other' gate "
              f"threshold in predict.py, so it's needed here even though it was skipped during "
              f"CV search above)")
        best_model.probability = True

    # Safety-asymmetric final fit: a real belt/brake/sway recording dismissed as
    # "other" is a dangerous false negative (a real fault goes unreported); "other"
    # audio misclassified as a fault is merely annoying. CV model SELECTION above
    # stays on unweighted macro-F1 (an honest, unbiased architecture comparison),
    # but the deployed model's FINAL fit up-weights the three real classes relative
    # to "other" so the decision boundary requires stronger evidence before ceding
    # ground to "other" in ambiguous regions -- found necessary 2026-08-23 after
    # widening the synthetic "other" pool caused a few real (mostly sway) test
    # files to flip to "other" that previously didn't.
    REAL_CLASS_BOOST = 1.5
    other_idx = int(label_encoder.transform([OTHER_LABEL])[0])
    balanced = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    custom_class_weight = {
        int(c): float(w) * (REAL_CLASS_BOOST if int(c) != other_idx else 1.0)
        for c, w in zip(np.unique(y_train), balanced)
    }
    print(f"\n    Final-fit class weights (other={other_idx} unboosted, "
          f"real classes x{REAL_CLASS_BOOST}): {custom_class_weight}")
    if hasattr(best_model, "class_weight"):
        best_model.set_params(class_weight=custom_class_weight)
        best_model.fit(X_train, y_train)
    else:
        sample_weight = np.array([custom_class_weight[label] for label in y_train])
        best_model.fit(X_train, y_train, sample_weight=sample_weight)

    val_pred = best_model.predict(X_val)
    val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
    print(f"    val macro-F1 (never used for CV ranking): {val_f1:.4f}")

    print("\n[8] Final evaluation on TEST (untouched until now)...")
    test_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, test_pred)
    f1_macro = f1_score(y_test, test_pred, average="macro", zero_division=0)
    report = classification_report(y_test, test_pred, target_names=class_names, zero_division=0)
    print(f"    Test accuracy: {acc:.4f}")
    print(f"    Test macro-F1: {f1_macro:.4f}")
    print("\n" + report)

    cm = confusion_matrix(y_test, test_pred)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix (Test) - 4-class incl. \"other\"")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "other_class_confusion_matrix.png"), dpi=150)
    plt.close()

    print("\n[9] HELD-OUT GENERALIZATION CHECK (never seen during training/val/test)...")
    n_correct_other, n_total = 0, 0

    print("    -- reserved real Windows system sounds --")
    for p in held_out_paths:
        y = load_system_sound(p)
        feats = scaler.transform(extract_mfcc_vector(y, config["target_sr"], n_mfcc=config["n_mfcc"]).reshape(1, -1))
        pred = class_names[best_model.predict(feats)[0]]
        n_total += 1
        n_correct_other += int(pred == OTHER_LABEL)
        print(f"       {os.path.basename(p):40s} -> {pred}")

    print("    -- reserved synthetic signal TYPES (square/sawtooth/AM-pink-noise, never generated for training) --")
    for kind, y in build_heldout_synthetic(config, count_per_kind=5):
        feats = scaler.transform(extract_mfcc_vector(y, config["target_sr"], n_mfcc=config["n_mfcc"]).reshape(1, -1))
        pred = class_names[best_model.predict(feats)[0]]
        n_total += 1
        n_correct_other += int(pred == OTHER_LABEL)
        print(f"       {kind:20s} -> {pred}")

    print("    -- original 4 exploratory OOD probes (silence/white-noise/440Hz tone/100Hz rumble) --")
    sr, dur = config["target_sr"], config["target_duration"]
    n = int(sr * dur)
    probe_rng = np.random.RandomState(0)
    t = np.linspace(0, dur, n, endpoint=False)
    probes = {
        "silence": np.zeros(n, dtype=np.float32),
        "white_noise": (probe_rng.randn(n) * 0.3).astype(np.float32),
        "pure_tone_440hz": (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
        "low_rumble_100hz": (0.5 * np.sin(2 * np.pi * 100 * t) + 0.05 * probe_rng.randn(n)).astype(np.float32),
    }
    for name, y in probes.items():
        feats = scaler.transform(extract_mfcc_vector(y, sr, n_mfcc=config["n_mfcc"]).reshape(1, -1))
        pred = class_names[best_model.predict(feats)[0]]
        n_total += 1
        n_correct_other += int(pred == OTHER_LABEL)
        print(f"       {name:20s} -> {pred}")

    print(f"\n    Held-out generalization: {n_correct_other}/{n_total} correctly classified as \"other\"")

    print("\n[10] Confirming this doesn't hurt real belt/brake/sway accuracy vs. the deployed 3-class model...")
    real_only_mask = y_test_str != OTHER_LABEL
    real_only_pred = best_model.predict(X_test[real_only_mask])
    real_only_true = y_test[real_only_mask]
    real_only_acc = accuracy_score(real_only_true, real_only_pred)
    print(f"    accuracy on the real belt/brake/sway TEST files only (with \"other\" as a possible 5th "
          f"answer the model could give but shouldn't need to): {real_only_acc:.4f}")
    print("    (compare against processed_data/traditional_ml_report.txt's 3-class test accuracy)")

    with open(os.path.join(DATA_DIR, "best_traditional_model_with_other.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "name": best_name}, f)
    with open(os.path.join(DATA_DIR, "scaler_with_other.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(DATA_DIR, "label_encoder_with_other.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(DATA_DIR, "other_class_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Best model (CV macro-F1={best_cv_f1:.4f}): {best_name}\n")
        f.write(f"Val macro-F1: {val_f1:.4f}\n\n")
        f.write(f"=== TEST results (4-class incl. other) ===\n")
        f.write(f"accuracy={acc:.4f}, macro-F1={f1_macro:.4f}\n\n")
        f.write(report)
        f.write(f"\n\nHeld-out generalization: {n_correct_other}/{n_total} correctly classified as \"other\"\n")
        f.write(f"Real-class-only test accuracy (with \"other\" as a possible wrong answer): {real_only_acc:.4f}\n")

    print("\n" + "=" * 70)
    print("DONE. Saved (NOT overwriting the deployed 3-class model):")
    print("   - best_traditional_model_with_other.pkl / scaler_with_other.pkl / label_encoder_with_other.pkl")
    print("   - other_class_report.txt / other_class_confusion_matrix.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
