# -*- coding: utf-8 -*-
r"""
===================================================================
"Other" / out-of-distribution detector for belt/brake/sway
===================================================================
Every classifier in this project (traditional ML, CNN, YAMNet, ...) is a forced
3-way softmax: it MUST output belt/brake/sway probabilities that sum to 1, even
for audio that isn't a car fault sound at all (silence, white noise, a music
note, human speech, ...). Empirically confirmed (2026-08-22): silence, white
noise, a pure 440Hz tone, and a 100Hz rumble all get confidently classified as
"belt" (58-100% confidence depending on the model) by the deployed pipeline --
there is no "none of these" option built into the classifiers themselves.

THIS SCRIPT does not touch those classifiers. Instead it fits an unsupervised
novelty detector (IsolationForest) on the SAME scaled MFCC feature vectors the
traditional ML model already uses (X_train_mfcc.npy) -- no negative/"other"
training data needed, since novelty detection only needs to learn what
belt/brake/sway sound LIKE, not what everything else sounds like. At inference
time (predict.py / the FastAPI service), this runs ALONGSIDE the normal
classifier: if the input's feature vector looks nothing like the training
distribution, the final prediction is overridden to "other" regardless of what
the classifier itself says -- the classifier's own softmax can never do this on
its own, no matter how the confidence threshold is tuned (a confidently-wrong
58-100% score is still "confident" by any reasonable threshold).

THRESHOLD CALIBRATION: the cutoff on IsolationForest's decision_function is
picked from the TRAIN set only (the 1st percentile of train scores -- i.e. by
construction, at most ~1% of the training data itself would be flagged). The
VAL and TEST splits are then used purely to REPORT the real held-out false-
positive rate (genuine belt/brake/sway samples wrongly flagged as "other") --
they are never used to pick the threshold itself, so this number is honest.

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python train_novelty_detector.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.ensemble import IsolationForest

from audio_common import extract_mfcc_vector

BASE_DIR = r"C:\Users\hp\Desktop\car_test"
DATA_DIR = os.path.join(BASE_DIR, "processed_data")

# Percentile of TRAIN decision_function scores used as the "other" cutoff --
# i.e. by construction, ~this fraction of genuine training samples would be
# (falsely) flagged if run against themselves. 1.0 = keep it conservative
# (rarely reject a real belt/brake/sway sample); raise it (e.g. 5.0) if the
# real-world false-positive rate reported below turns out too low to be useful,
# lower it (e.g. 0.1) if too many genuine test-set samples get flagged.
TRAIN_PERCENTILE_CUTOFF = 1.0


def make_synthetic_ood_waveforms(target_sr, target_duration):
    """A handful of audio signals that are clearly NOT a car fault sound, used
    only to sanity-check the detector below -- not used to fit it (fitting is
    unsupervised, train-distribution-only; these have no labels and are not
    real recordings)."""
    n = int(target_sr * target_duration)
    rng = np.random.RandomState(0)
    t = np.linspace(0, target_duration, n, endpoint=False)
    return {
        "silence": np.zeros(n, dtype=np.float32),
        "white_noise": (rng.randn(n) * 0.3).astype(np.float32),
        "pure_tone_440hz": (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32),
        "low_rumble_100hz": (0.5 * np.sin(2 * np.pi * 100 * t) + 0.05 * rng.randn(n)).astype(np.float32),
    }


def main():
    print("=" * 70)
    print('Training the "other" / out-of-distribution novelty detector')
    print("=" * 70)

    with open(os.path.join(DATA_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    with open(os.path.join(DATA_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    X_train = np.load(os.path.join(DATA_DIR, "X_train_mfcc.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val_mfcc.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test_mfcc.npy"))
    print(f"\n[1] Loaded features: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

    print("\n[2] Fitting IsolationForest on TRAIN features only (unsupervised, no labels used)...")
    detector = IsolationForest(n_estimators=300, random_state=42, n_jobs=-1)
    detector.fit(X_train)

    train_scores = detector.decision_function(X_train)
    threshold = float(np.percentile(train_scores, TRAIN_PERCENTILE_CUTOFF))
    print(f"    -> cutoff = {TRAIN_PERCENTILE_CUTOFF}th percentile of TRAIN scores = {threshold:.4f}")
    print("       (score below this on a real prediction -> override final_prediction to \"other\")")

    print("\n[3] Held-out false-positive rate (genuine belt/brake/sway wrongly flagged as \"other\"):")
    for name, X in [("val", X_val), ("test", X_test)]:
        scores = detector.decision_function(X)
        fpr = float(np.mean(scores < threshold))
        print(f"    {name:5s}: {fpr*100:.2f}% of {len(X)} genuine samples would be misflagged as \"other\"")

    print("\n[4] Sanity check on synthetic non-car-fault audio (NOT used for fitting):")
    ood_waveforms = make_synthetic_ood_waveforms(config["target_sr"], config["target_duration"])
    n_caught = 0
    for name, y in ood_waveforms.items():
        feats = extract_mfcc_vector(y, config["target_sr"], n_mfcc=config["n_mfcc"]).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        score = detector.decision_function(feats_scaled)[0]
        flagged = score < threshold
        n_caught += int(flagged)
        print(f"    {name:18s}: score={score:+.4f} -> {'FLAGGED as \"other\" (correct)' if flagged else 'NOT flagged (missed)'}")
    print(f"\n    Caught {n_caught}/{len(ood_waveforms)} synthetic out-of-distribution samples.")

    with open(os.path.join(DATA_DIR, "novelty_detector.pkl"), "wb") as f:
        pickle.dump({"model": detector, "threshold": threshold, "train_percentile_cutoff": TRAIN_PERCENTILE_CUTOFF}, f)

    print("\n" + "=" * 70)
    print("DONE. Saved processed_data/novelty_detector.pkl")
    print('predict.py / the FastAPI service load this and override final_prediction to')
    print('"other" whenever a real request\'s feature vector scores below the threshold,')
    print("regardless of what the belt/brake/sway classifier itself says.")
    print("=" * 70)


if __name__ == "__main__":
    main()
