# -*- coding: utf-8 -*-
r"""
===================================================================
BEATs (Bidirectional Encoder representation from Audio Transformers)
===================================================================
Same recipe as train_panns.py / train_ast.py / train_clap.py /
train_passt.py: freeze a pretrained AudioSet model, extract one
embedding per clip, train a small classifier head on top. BEATs
(Microsoft) is a self-supervised transformer that is, at the time of
writing, one of the strongest general-purpose AudioSet models
published (stronger than AST/PANNs on several published benchmarks).
Built as the LAST and strongest extra comparison point for the
report/discussion -- NOT expected to beat the traditional ML result
(86.47% test accuracy), same domain-shift reasoning as every other
transfer-learning model tried in this project: even the strongest
general AudioSet model still doesn't know what a belt squeal or a
sway clunk specifically sounds like, only what "general everyday/
YouTube sound" looks like in embedding space.

WHY THIS SCRIPT IS DIFFERENT FROM THE OTHERS:
  BEATs has no clean pip/Hugging Face package (unlike AST/CLAP via
  `transformers`, or PaSST via `hear21passt`). Microsoft's official
  code (github.com/microsoft/unilm/tree/master/beats) is NOT a pip
  package -- this project vendors the 3 source files it actually
  needs for inference (BEATs.py, backbone.py, modules.py -- copied
  as-is under `beats_vendor/`, MIT-licensed, headers preserved;
  Tokenizers.py/quantizer.py were NOT vendored because they're only
  needed for BEATs' OWN pretraining pipeline, not for loading a
  finished checkpoint and extracting features).
  The official checkpoint is only distributed via OneDrive links
  (fragile to script), so this uses a verified Hugging Face mirror
  instead (`Bencr/beats-checkpoints`, same file, MIT-licensed
  weights) -- downloaded once and cached locally under
  `beats_vendor/` (722MB, NOT committed to git -- see .gitignore).

WHAT THIS SCRIPT DOES:
  1) Reads dataset_documentation.csv (produced by preprocessing.py) --
     SAME leakage-safe train/val/test split as everywhere else.
  2) Loads each recording at 16 kHz mono (BEATs' required input rate,
     same as AST), trims silence, normalizes, fixes to 5 seconds (via
     the shared audio_common.load_clean_audio, same as every other
     script here).
  3) Runs it through the pretrained BEATs checkpoint
     (BEATs_iter3+ AS2M) and mean-pools its per-timestep output
     (encoder_embed_dim=768) over time -> one embedding per clip,
     same pooling approach as train_ast.py.
  4) Trains a Logistic Regression and an SVM on top of these
     embeddings (same small grid search as the other transfer-
     learning scripts), picks the best on the VALIDATION set, reports
     final metrics on the TEST set.
  Train is NOT augmented (same lesson learned from every other
  frozen-embedding model in this project). Like train_clap.py/
  train_passt.py, the classifier head is fit on a class-balanced
  SUBSET of train (BEATs is a heavy CPU-only transformer here) --
  val/test are always full and untouched.

REQUIREMENTS (extra, on top of the rest of the project):
    torch/torchaudio (already installed for PaSST)
    Needs an internet connection the FIRST time it runs (downloads the
    ~722MB checkpoint from the Hugging Face mirror, cached afterwards
    under beats_vendor/).

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python train_beats.py
"""

import os
import sys
import pickle
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
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
BEATS_VENDOR_DIR = os.path.join(BASE_DIR, "beats_vendor")
sys.path.insert(0, BEATS_VENDOR_DIR)  # BEATs.py does `from backbone import ...` (flat, non-relative import)

# Hugging Face mirror of Microsoft's official BEATs_iter3+ (AS2M) checkpoint (MIT-licensed
# weights, see module docstring for why this mirror is used instead of the official OneDrive link)
BEATS_CHECKPOINT_URL = (
    "https://huggingface.co/datasets/Bencr/beats-checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt"
)
BEATS_CHECKPOINT_LOCAL = os.path.join(BEATS_VENDOR_DIR, "BEATs_iter3_plus_AS2M.pt")

BEATS_SR = 16000           # BEATs requires 16kHz mono input (same as AST)
TARGET_DURATION = 5.0      # keep consistent with the rest of the project
BATCH_SIZE = 8             # BEATs is a heavy transformer -- smaller batches on CPU

# see the module docstring -- same subset-for-speed rationale as train_clap.py/train_passt.py
TRAIN_SUBSET_PER_CLASS = 67  # ~200 total across 3 classes (vs 626 full) -- fixed seed below
TRAIN_SUBSET_SEED = 42

CLASSIFIER_GRID = {
    "Logistic Regression (C=1)": lambda: LogisticRegression(C=1, max_iter=2000, class_weight="balanced"),
    "Logistic Regression (C=10)": lambda: LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
    # probability=True deliberately OMITTED here -- see the refit-the-winner block near
    # best_model selection below for why (it's a known pathological SVC slowdown, confirmed
    # to hang for 17+ CPU-hours on this project's own data at full scale).
    "SVM (linear, C=1)": lambda: SVC(kernel="linear", C=1, class_weight="balanced"),
    "SVM (rbf, C=10)": lambda: SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"),
}


def load_clean_audio_16k(path):
    """Shared with preprocessing.py/predict.py via audio_common (same trim/
    normalize/pad + empty-clip fallback logic), just at BEATs' 16kHz rate."""
    y, _ = _shared_load_clean_audio(
        path, target_sr=BEATS_SR, target_duration=TARGET_DURATION, fallback_to_untrimmed=True,
    )
    return y.astype(np.float32)


def ensure_checkpoint_downloaded():
    if os.path.exists(BEATS_CHECKPOINT_LOCAL):
        return
    os.makedirs(BEATS_VENDOR_DIR, exist_ok=True)
    print(f"   (downloading BEATs checkpoint, ~722MB, first run only -- from {BEATS_CHECKPOINT_URL})")

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            print(f"\r   {pct}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)", end="", flush=True)

    urllib.request.urlretrieve(BEATS_CHECKPOINT_URL, BEATS_CHECKPOINT_LOCAL, reporthook=_progress)
    print()


def load_beats_model():
    """Loads the vendored BEATs.py/backbone.py/modules.py (see beats_vendor/) with the
    downloaded checkpoint. CPU by default -- same as the rest of this project, since
    training here is just a small classifier head, not the transformer itself."""
    ensure_checkpoint_downloaded()
    from BEATs import BEATs, BEATsConfig  # from beats_vendor/, added to sys.path above

    checkpoint = torch.load(BEATS_CHECKPOINT_LOCAL, map_location="cpu", weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def extract_embeddings(paths, model, desc="embedding extraction"):
    from tqdm import tqdm
    embeddings = []
    with torch.no_grad():
        for i in tqdm(range(0, len(paths), BATCH_SIZE), desc=desc):
            batch_paths = paths[i:i + BATCH_SIZE]
            batch_audio = torch.tensor(np.stack([load_clean_audio_16k(p) for p in batch_paths]))
            features, _ = model.extract_features(batch_audio, padding_mask=None)
            # mean-pool the per-timestep output over time -> one fixed-size embedding per
            # clip, same idea as train_ast.py's mean-pooled last_hidden_state
            pooled = features.mean(dim=1)
            embeddings.append(pooled.numpy())
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
    print("BEATs (Bidirectional Encoder representation from Audio Transformers)")
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

    train_df_full = doc_df[doc_df["split_assigned"] == "train"]
    val_df = doc_df[doc_df["split_assigned"] == "val"]
    test_df = doc_df[doc_df["split_assigned"] == "test"]

    train_df = pd.concat([
        g.sample(n=min(len(g), TRAIN_SUBSET_PER_CLASS), random_state=TRAIN_SUBSET_SEED)
        for _, g in train_df_full.groupby("class")
    ])
    print(f"Using the SAME leakage-safe split as the rest of the project: "
          f"train={len(train_df_full)} (subsampled to {len(train_df)}, class-balanced, seed={TRAIN_SUBSET_SEED}), "
          f"val={len(val_df)}, test={len(test_df)}")
    print("(No augmentation on train -- same lesson learned from every other frozen-embedding "
          "model in this project.)")

    print("\n[1] Loading BEATs (downloading checkpoint on first run, ~722MB)...")
    beats_model = load_beats_model()

    print("\n[2] Extracting embeddings (mean-pooled, per clip)...")
    X_train = extract_embeddings(list(train_df["matched_path"]), beats_model, "train embeddings")
    X_val = extract_embeddings(list(val_df["matched_path"]), beats_model, "val embeddings")
    X_test = extract_embeddings(list(test_df["matched_path"]), beats_model, "test embeddings")

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

    with open(os.path.join(DATA_DIR, "beats_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"NOTE: classifier head trained on a class-balanced SUBSET of train "
                f"({len(train_df)}/{len(train_df_full)} files, seed={TRAIN_SUBSET_SEED}) for "
                f"CPU-time practicality -- val/test are FULL and untouched, so these numbers are "
                f"directly comparable to every other model's report in this project.\n\n")
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
    ax.set_title(f"Confusion Matrix (Test) - BEATs + {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "beats_confusion_matrix.png"), dpi=150)
    plt.close()

    with open(os.path.join(DATA_DIR, "best_beats_model.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "name": best_name, "scaler": scaler}, f)

    # save val/test probabilities so evaluate_ensemble.py can optionally fuse this model too
    np.save(os.path.join(DATA_DIR, "beats_val_probs.npy"), best_model.predict_proba(X_val))
    np.save(os.path.join(DATA_DIR, "beats_test_probs.npy"), best_model.predict_proba(X_test))

    print("\n" + "=" * 70)
    print("DONE. Saved:")
    print("   - best_beats_model.pkl")
    print("   - beats_report.txt")
    print("   - beats_confusion_matrix.png")
    print("   - beats_val_probs.npy / beats_test_probs.npy (for evaluate_ensemble.py)")
    print("=" * 70)
    print("\nCompare this test macro-F1 against every other model's report -- this is the "
          "strongest single AudioSet comparison point in the report, not expected to beat the "
          "traditional ML result.")


if __name__ == "__main__":
    main()
