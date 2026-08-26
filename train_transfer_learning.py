# -*- coding: utf-8 -*-
r"""
===================================================================
Transfer Learning - YAMNet embeddings + a simple classifier head
===================================================================
This is the model most likely to beat BOTH the SVM/RF baseline and the
from-scratch CNN, because it reuses a network (YAMNet, Google's audio
event classifier) already pretrained on ~2 million labeled YouTube
clips. With our dataset (roughly a thousand original recordings) it is
usually more data-efficient than training a CNN from zero.

WHAT THIS SCRIPT DOES:
  1) Reads dataset_documentation.csv (produced by preprocessing.py) to
     get the file path, class, and TRAIN/VAL/TEST split for every
     matched recording -- the SAME leakage-safe split used everywhere
     else in this project.
  2) Loads each recording at 16 kHz mono (YAMNet's required input
     rate), trims silence, normalizes, fixes to 5 seconds (same
     convention as the rest of the project).
  3) Runs YAMNet (downloaded once from TensorFlow Hub, then cached
     locally) to get one embedding per ~0.96s audio frame, and
     average-pools those frames into a single 1024-dim vector per clip.
  4) Trains a Logistic Regression and an SVM on top of these
     embeddings (grid search a few settings), picks the best on the
     VALIDATION set, reports final metrics on the TEST set -- same
     methodology as train_traditional_ml.py.
  5) TRAIN split only: generates the same waveform augmentation as
     preprocessing.py (time shift / volume change / light noise, no
     pitch shifting) so YAMNet gets a comparable amount of training
     data to the CNN and traditional ML models, instead of far less.
     Val/test are NEVER augmented, to keep evaluation honest.

REQUIREMENTS (extra, on top of the rest of the project):
    pip install tensorflow tensorflow-hub
    Needs an internet connection the FIRST time it runs (to download
    YAMNet, ~15MB). It is cached afterwards, so later runs work offline.

    NOTE: if you get "ModuleNotFoundError: No module named 'pkg_resources'"
    when this script tries to import tensorflow_hub, that's because recent
    setuptools versions removed pkg_resources (which the older
    tensorflow_hub package still needs). Quick fix:
        pip install "setuptools<80"
    This script ALSO tries a second loading method automatically
    (kagglehub, Google's newer replacement for tfhub.dev) if
    tensorflow_hub can't be imported at all, so it should work either way.

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python train_transfer_learning.py
"""

import os
import pickle
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from audio_common import load_clean_audio as _shared_load_clean_audio

BASE_DIR = r"C:\Users\hp\Desktop\car_test"
DATA_DIR = os.path.join(BASE_DIR, "processed_data")

YAMNET_SR = 16000          # YAMNet requires exactly 16kHz mono input
TARGET_DURATION = 5.0      # keep consistent with the rest of the project
YAMNET_TFHUB_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_KAGGLE_HANDLE = "google/yamnet/tensorFlow2/yamnet/1"

# same augmentation idea as preprocessing.py, applied here too (on the 16kHz waveform)
# so YAMNet gets the same amount of training data as the CNN/traditional ML models
# instead of being trained on far fewer (unaugmented) examples.
N_AUGMENTATIONS_PER_TRAIN_FILE = 3


def load_yamnet():
    """Tries tensorflow_hub first (the original way). If that fails for ANY
    reason (missing pkg_resources, tfhub.dev being unreachable, etc.), falls
    back to kagglehub -- Google migrated all TF Hub models to Kaggle Models,
    so this is the current recommended path and doesn't need pkg_resources."""
    try:
        import tensorflow_hub as hub
        print("   (loading via tensorflow_hub)")
        return hub.load(YAMNET_TFHUB_URL)
    except Exception as e:
        print(f"   tensorflow_hub failed ({type(e).__name__}: {e}) -- trying kagglehub instead...")
        import tensorflow as tf
        import kagglehub
        model_path = kagglehub.model_download(YAMNET_KAGGLE_HANDLE)
        print(f"   (loaded via kagglehub, cached at {model_path})")
        return tf.saved_model.load(model_path)

CLASSIFIER_GRID = {
    "Logistic Regression (C=1)": lambda: LogisticRegression(C=1, max_iter=2000, class_weight="balanced"),
    "Logistic Regression (C=10)": lambda: LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
    # probability=True deliberately OMITTED here: CalibratedClassifierCV below already wraps
    # every candidate in its OWN external 5-fold calibration and does not need the base
    # estimator to support predict_proba at all -- so probability=True here would just stack
    # a SECOND, redundant internal 5-fold Platt-scaling CV underneath (up to ~25x the fit cost
    # for nothing), which is a known pathological slowdown on some datasets (confirmed to hang
    # for 17+ CPU-hours on this project's own data at full scale, in a sibling script that
    # didn't even have the extra CalibratedClassifierCV wrapper).
    "SVM (linear, C=1)": lambda: SVC(kernel="linear", C=1, class_weight="balanced"),
    "SVM (rbf, C=10)": lambda: SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
}

# A linear classifier on top of 1024-dim YAMNet embeddings, fit on a few thousand
# samples, tends to see the training data as almost perfectly separable -- so its
# raw predict_proba comes out massively overconfident (e.g. >99.99% on two thirds
# of test predictions while true test accuracy is only ~78%). CalibratedClassifierCV
# wraps each candidate in 5-fold cross-validated Platt scaling so predict_proba
# reflects genuine confidence instead -- this does NOT change which class wins
# (argmax), so accuracy/macro-F1 below are unaffected; only the confidence NUMBERS
# shown to users (and fed into the ensemble/stacking fusion) become trustworthy.
CALIBRATE_PROBABILITIES = True
CALIBRATION_CV_FOLDS = 5


def load_clean_audio_16k(path):
    """Shared with preprocessing.py/predict.py via audio_common (same trim/
    normalize/pad + empty-clip fallback logic), just at YAMNet's 16kHz rate."""
    y, _ = _shared_load_clean_audio(
        path, target_sr=YAMNET_SR, target_duration=TARGET_DURATION, fallback_to_untrimmed=True,
    )
    return y.astype(np.float32)


def aug_time_shift(y, max_shift_frac=0.1):
    """Same fix as preprocessing.py's aug_time_shift: zero-pad the vacated side
    instead of np.roll, which would otherwise wrap the clip's tail/head around
    and create an unrealistic click at the seam."""
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
    return y * random.uniform(low, high)


def aug_add_noise(y, noise_factor=0.001):
    """Lowered from 0.005 to match preprocessing.py's aug_add_noise -- see the comment
    there: 0.005 was empirically ~5.5x more disruptive to the feature vector than the
    other two augmentation techniques, and measurably hurt the traditional ML model's
    accuracy when enabled."""
    noise = np.random.randn(len(y)).astype(np.float32)
    return y + noise_factor * noise


def make_augmented_version(y):
    """Same idea as preprocessing.py: randomly combine 1-3 light augmentations
    (no pitch shifting -- that could change the diagnostic character of the sound)."""
    out = y.copy()
    techniques = random.sample([aug_time_shift, aug_volume_change, aug_add_noise], k=random.randint(1, 3))
    for t in techniques:
        out = t(out)
    if np.max(np.abs(out)) > 0:
        out = out / np.max(np.abs(out))
    return out.astype(np.float32)


def extract_embeddings(paths, yamnet_model, desc="embedding extraction"):
    from tqdm import tqdm
    embeddings = []
    for path in tqdm(paths, desc=desc):
        y = load_clean_audio_16k(path)
        _, frame_embeddings, _ = yamnet_model(y)
        # average-pool over the ~5 frames covering the 5-second clip -> one 1024-dim vector
        embeddings.append(np.mean(frame_embeddings.numpy(), axis=0))
    return np.array(embeddings)


def extract_embeddings_train_augmented(paths, labels, yamnet_model, n_augment):
    """Like extract_embeddings, but for EACH train file also generates n_augment
    augmented copies (same technique as preprocessing.py) and gets a YAMNet
    embedding for each variant. Returns an EXPANDED (X, y) pair -- e.g. 626 train
    files with n_augment=3 becomes ~2504 (embedding, label) pairs, matching the
    same data volume the CNN/traditional ML models already train on."""
    from tqdm import tqdm
    embeddings, expanded_labels = [], []
    for path, label in tqdm(list(zip(paths, labels)), desc="train embeddings (original + augmented)"):
        y = load_clean_audio_16k(path)
        variants = [y] + [make_augmented_version(y) for _ in range(n_augment)]
        for variant in variants:
            _, frame_embeddings, _ = yamnet_model(variant)
            embeddings.append(np.mean(frame_embeddings.numpy(), axis=0))
            expanded_labels.append(label)
    return np.array(embeddings), np.array(expanded_labels)


def evaluate(model, X, y, class_names):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "report": classification_report(y, y_pred, target_names=class_names, zero_division=0),
        "y_pred": y_pred,
    }


def main():
    print("=" * 70)
    print("Transfer Learning - YAMNet embeddings + classifier head")
    print("=" * 70)

    doc_path = os.path.join(DATA_DIR, "dataset_documentation.csv")
    if not os.path.exists(doc_path):
        print(f"\n!! {doc_path} not found. Run preprocessing.py first.")
        return
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)

    doc_df = pd.read_csv(doc_path)

    # exclude the same files preprocessing.py already flagged as corrupted/empty, so this
    # script trains/evaluates on the exact same usable files as the rest of the project
    corrupted_path = os.path.join(DATA_DIR, "corrupted_files.csv")
    if os.path.exists(corrupted_path):
        corrupted_df = pd.read_csv(corrupted_path)
        if "matched_path" in corrupted_df.columns:
            before = len(doc_df)
            doc_df = doc_df[~doc_df["matched_path"].isin(corrupted_df["matched_path"])].reset_index(drop=True)
            skipped = before - len(doc_df)
            if skipped:
                print(f"   (skipping {skipped} file(s) already flagged as corrupted by preprocessing.py)")
    train_df = doc_df[doc_df["split_assigned"] == "train"]
    val_df = doc_df[doc_df["split_assigned"] == "val"]
    test_df = doc_df[doc_df["split_assigned"] == "test"]
    print(f"Using the SAME leakage-safe split as the rest of the project: "
          f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"(train will be expanded with {N_AUGMENTATIONS_PER_TRAIN_FILE} augmented copies per "
          f"file -- same technique as preprocessing.py -- so YAMNet gets a comparable amount "
          f"of training data to the CNN/traditional ML models, instead of far less.)")

    print("\n[1] Loading YAMNet (tensorflow_hub, falling back to kagglehub if needed)...")
    yamnet_model = load_yamnet()

    print("\n[2] Extracting embeddings...")
    y_train_raw = label_encoder.transform(train_df["class"])
    X_train, y_train = extract_embeddings_train_augmented(
        list(train_df["matched_path"]), y_train_raw, yamnet_model, N_AUGMENTATIONS_PER_TRAIN_FILE
    )
    X_val = extract_embeddings(list(val_df["matched_path"]), yamnet_model, "val embeddings")
    X_test = extract_embeddings(list(test_df["matched_path"]), yamnet_model, "test embeddings")

    y_val = label_encoder.transform(val_df["class"])
    y_test = label_encoder.transform(test_df["class"])
    print(f"   -> train expanded from {len(train_df)} to {len(y_train)} samples "
          f"(val/test stay unaugmented, as always, for an honest evaluation)")

    print("\n[3] Scaling embeddings (StandardScaler fit on train only)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\n[4] Training candidate classifier heads, evaluating on VALIDATION set...\n")
    if CALIBRATE_PROBABILITIES:
        print(f"    (each candidate wrapped in {CALIBRATION_CV_FOLDS}-fold CalibratedClassifierCV so "
              f"predict_proba isn't overconfident -- see comment above CLASSIFIER_GRID)")
    val_results, trained = {}, {}
    for name, builder in CLASSIFIER_GRID.items():
        if CALIBRATE_PROBABILITIES:
            model = CalibratedClassifierCV(builder(), method="sigmoid", cv=CALIBRATION_CV_FOLDS)
        else:
            model = builder()
        model.fit(X_train, y_train)
        trained[name] = model
        res = evaluate(model, X_val, y_val, class_names)
        val_results[name] = res
        print(f"   -> {name}: Val accuracy={res['accuracy']:.4f} | Val macro-F1={res['f1_macro']:.4f}")

    best_name = max(val_results, key=lambda n: val_results[n]["f1_macro"])
    best_model = trained[best_name]
    print(f"\n[5] Best classifier head on validation set: {best_name}")

    print(f"\n[6] Final evaluation of '{best_name}' on the TEST set (never seen before)...")
    test_res = evaluate(best_model, X_test, y_test, class_names)
    print(f"   Test accuracy: {test_res['accuracy']:.4f}")
    print(f"   Test macro-F1: {test_res['f1_macro']:.4f}")
    print(f"   Test macro-precision: {test_res['precision_macro']:.4f}")
    print(f"   Test macro-recall: {test_res['recall_macro']:.4f}")
    print("\n" + test_res["report"])

    with open(os.path.join(DATA_DIR, "transfer_learning_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"BEST CLASSIFIER HEAD: {best_name}\n\n")
        f.write("=== Validation results (all candidates) ===\n")
        for name, res in val_results.items():
            f.write(f"\n{name}: accuracy={res['accuracy']:.4f}, macro-F1={res['f1_macro']:.4f}\n")
        f.write("\n=== FINAL TEST results (best model only) ===\n")
        f.write(f"accuracy={test_res['accuracy']:.4f}, macro-F1={test_res['f1_macro']:.4f}, "
                f"macro-precision={test_res['precision_macro']:.4f}, macro-recall={test_res['recall_macro']:.4f}\n\n")
        f.write(test_res["report"])

    cm = confusion_matrix(y_test, test_res["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax, cmap="Oranges", values_format="d")
    ax.set_title(f"Confusion Matrix (Test) - YAMNet + {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "transfer_learning_confusion_matrix.png"), dpi=150)
    plt.close()

    with open(os.path.join(DATA_DIR, "best_transfer_model.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "name": best_name, "scaler": scaler}, f)

    # save val/test probabilities so evaluate_ensemble.py can optionally fuse this model too
    np.save(os.path.join(DATA_DIR, "transfer_val_probs.npy"), best_model.predict_proba(X_val))
    np.save(os.path.join(DATA_DIR, "transfer_test_probs.npy"), best_model.predict_proba(X_test))

    print("\n" + "=" * 70)
    print("DONE. Saved:")
    print("   - best_transfer_model.pkl")
    print("   - transfer_learning_report.txt")
    print("   - transfer_learning_confusion_matrix.png")
    print("   - transfer_val_probs.npy / transfer_test_probs.npy (for evaluate_ensemble.py)")
    print("=" * 70)
    print("\nCompare this test macro-F1 against traditional_ml_report.txt and cnn_report.txt. "
          "If it wins, evaluate_ensemble.py will automatically include it in the fusion too.")


if __name__ == "__main__":
    main()