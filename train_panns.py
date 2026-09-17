# -*- coding: utf-8 -*-
r"""
===================================================================
PANNs (CNN14) Transfer Learning - alternative to YAMNet, for comparison
===================================================================
Same idea as train_transfer_learning.py (YAMNet), but using PANNs
(Pretrained Audio Neural Networks, CNN14 architecture) instead --
also pretrained on AudioSet (~2 million YouTube clips), but a bigger,
newer network than YAMNet that usually scores higher on published
audio-tagging benchmarks. Built purely as an EXTRA comparison point
for the report/discussion ("we also tried a newer/bigger pretrained
audio model and confirmed traditional ML is still the best choice
for this dataset size") -- NOT expected to beat the traditional ML
result (92.46% macro-F1), based on the same reasoning that already
applied to YAMNet.

WHAT THIS SCRIPT DOES:
  1) Reads dataset_documentation.csv (produced by preprocessing.py) --
     SAME leakage-safe train/val/test split as everywhere else.
  2) Loads each recording at 32 kHz mono (PANNs' required input rate --
     different from YAMNet's 16kHz and the rest of the project's
     22050Hz), trims silence, normalizes, fixes to 5 seconds.
  3) Runs PANNs (CNN14, downloaded once via the `panns_inference`
     package, then cached locally) to get a single 2048-dim embedding
     per clip.
  4) Trains a Logistic Regression and an SVM on top of these
     embeddings (same small grid search as train_transfer_learning.py),
     picks the best on the VALIDATION set, reports final metrics on
     the TEST set.
  Train is NOT augmented (same lesson learned from YAMNet: augmenting
  a frozen pretrained model's input tends to hurt more than help here).

REQUIREMENTS (extra, on top of the rest of the project):
    pip install panns-inference torch
    Needs an internet connection the FIRST time it runs (downloads the
    CNN14 checkpoint, ~300MB, to ~/panns_data/). Cached afterwards.

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python train_panns.py
"""

import os
import pickle
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from audio_common import load_clean_audio as _shared_load_clean_audio

BASE_DIR = os.environ.get("CAR_TEST_RAW_DATA_DIR", r"C:\Users\hp\Desktop\car_test")
DATA_DIR = os.environ.get("CAR_TEST_OUTPUT_DIR", os.path.join(BASE_DIR, "processed_data"))

PANNS_SR = 32000           # PANNs (CNN14) requires exactly 32kHz mono input
TARGET_DURATION = 5.0      # keep consistent with the rest of the project
BATCH_SIZE = 16            # how many clips to run through PANNs at once (speed)

CLASSIFIER_GRID = {
    "Logistic Regression (C=1)": lambda: LogisticRegression(C=1, max_iter=2000, class_weight="balanced"),
    "Logistic Regression (C=10)": lambda: LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
    # probability=True deliberately OMITTED here -- see the refit-the-winner block near
    # best_model selection below for why (it's a known pathological SVC slowdown, confirmed
    # to hang for 17+ CPU-hours on this project's own data at full scale).
    "SVM (linear, C=1)": lambda: SVC(kernel="linear", C=1, class_weight="balanced"),
    "SVM (rbf, C=10)": lambda: SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
}


def load_clean_audio_32k(path):
    """Shared with preprocessing.py/predict.py via audio_common (same trim/
    normalize/pad + empty-clip fallback logic), just at PANNs' 32kHz rate."""
    y, _ = _shared_load_clean_audio(
        path, target_sr=PANNS_SR, target_duration=TARGET_DURATION, fallback_to_untrimmed=True,
    )
    return y.astype(np.float32)


def load_panns_model():
    """Downloads the CNN14 checkpoint on first run (cached under ~/panns_data/
    afterwards). Uses CPU by default -- same as the rest of this project, since
    training here is just a small classifier head, not the big network itself."""
    from panns_inference import AudioTagging
    print("   (loading PANNs CNN14 via panns_inference -- first run downloads ~300MB checkpoint)")
    return AudioTagging(checkpoint_path=None, device="cpu")


def extract_embeddings(paths, panns_model, desc="embedding extraction"):
    from tqdm import tqdm
    embeddings = []
    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=desc):
        batch_paths = paths[i:i + BATCH_SIZE]
        batch_audio = np.stack([load_clean_audio_32k(p) for p in batch_paths])
        _, embedding = panns_model.inference(batch_audio)  # embedding shape: (batch, 2048)
        embeddings.append(embedding)
    return np.concatenate(embeddings, axis=0)


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
    print("PANNs (CNN14) Transfer Learning - alternative to YAMNet")
    print("=" * 70)

    doc_path = os.path.join(DATA_DIR, "dataset_documentation.csv")
    if not os.path.exists(doc_path):
        print(f"\n!! {doc_path} not found. Run preprocessing.py first.")
        return
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)

    doc_df = pd.read_csv(doc_path)

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
    print("(No augmentation on train -- same lesson learned from YAMNet: augmenting a frozen "
          "pretrained model's input tends to confuse its small classifier head more than help.)")

    print("\n[1] Loading PANNs (CNN14)...")
    panns_model = load_panns_model()

    print("\n[2] Extracting embeddings (2048-dim per clip)...")
    X_train = extract_embeddings(list(train_df["matched_path"]), panns_model, "train embeddings")
    X_val = extract_embeddings(list(val_df["matched_path"]), panns_model, "val embeddings")
    X_test = extract_embeddings(list(test_df["matched_path"]), panns_model, "test embeddings")

    y_train = label_encoder.transform(train_df["class"])
    y_val = label_encoder.transform(val_df["class"])
    y_test = label_encoder.transform(test_df["class"])

    print("\n[3] Scaling embeddings (StandardScaler fit on train only)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("\n[4] Training candidate classifier heads, evaluating on VALIDATION set...\n")
    val_results, trained = {}, {}
    for name, builder in CLASSIFIER_GRID.items():
        model = builder()
        model.fit(X_train, y_train)
        trained[name] = model
        res = evaluate(model, X_val, y_val, class_names)
        val_results[name] = res
        print(f"   -> {name}: Val accuracy={res['accuracy']:.4f} | Val macro-F1={res['f1_macro']:.4f}")

    best_name = max(val_results, key=lambda n: val_results[n]["f1_macro"])
    best_model = trained[best_name]
    print(f"\n[5] Best classifier head on validation set: {best_name}")

    if isinstance(best_model, SVC) and not best_model.probability:
        print(f"    (refitting {best_name} with probability=True for calibrated predict_proba -- "
              f"skipped during search to avoid SVC's internal 5-fold Platt-scaling CV, a known "
              f"pathological slowdown on some datasets)")
        best_model = CLASSIFIER_GRID[best_name]()
        best_model.probability = True
        best_model.fit(X_train, y_train)
        trained[best_name] = best_model

    print(f"\n[6] Final evaluation of '{best_name}' on the TEST set (never seen before)...")
    test_res = evaluate(best_model, X_test, y_test, class_names)
    print(f"   Test accuracy: {test_res['accuracy']:.4f}")
    print(f"   Test macro-F1: {test_res['f1_macro']:.4f}")
    print(f"   Test macro-precision: {test_res['precision_macro']:.4f}")
    print(f"   Test macro-recall: {test_res['recall_macro']:.4f}")
    print("\n" + test_res["report"])

    with open(os.path.join(DATA_DIR, "panns_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"BEST CLASSIFIER HEAD: {best_name}\n\n")
        f.write("=== Validation results (all candidates) ===\n")
        for name, res in val_results.items():
            f.write(f"\n{name}: accuracy={res['accuracy']:.4f}, macro-F1={res['f1_macro']:.4f}\n")
        f.write("\n=== FINAL TEST results (best model only) ===\n")
        f.write(f"accuracy={test_res['accuracy']:.4f}, macro-F1={test_res['f1_macro']:.4f}, "
                f"macro-precision={test_res['precision_macro']:.4f}, macro-recall={test_res['recall_macro']:.4f}\n\n")
        f.write(test_res["report"])
        f.write("\n\n(Comparison point only -- see README.md for why traditional ML is still "
                 "the recommended production model regardless of this result.)")

    cm = confusion_matrix(y_test, test_res["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax, cmap="Greens", values_format="d")
    ax.set_title(f"Confusion Matrix (Test) - PANNs + {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "panns_confusion_matrix.png"), dpi=150)
    plt.close()

    with open(os.path.join(DATA_DIR, "best_panns_model.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "name": best_name, "scaler": scaler}, f)

    # save val/test probabilities so evaluate_ensemble.py can optionally fuse this model too
    np.save(os.path.join(DATA_DIR, "panns_val_probs.npy"), best_model.predict_proba(X_val))
    np.save(os.path.join(DATA_DIR, "panns_test_probs.npy"), best_model.predict_proba(X_test))

    print("\n" + "=" * 70)
    print("DONE. Saved:")
    print("   - best_panns_model.pkl")
    print("   - panns_report.txt")
    print("   - panns_confusion_matrix.png")
    print("   - panns_val_probs.npy / panns_test_probs.npy (for evaluate_ensemble.py)")
    print("=" * 70)
    print("\nCompare this test macro-F1 against traditional_ml_report.txt, cnn_report.txt, and "
          "transfer_learning_report.txt (YAMNet) -- this is a comparison data point for the "
          "report/discussion, not expected to beat the traditional ML result.")


if __name__ == "__main__":
    main()