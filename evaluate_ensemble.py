# -*- coding: utf-8 -*-
r"""
===================================================================
Ensemble evaluation - fuse traditional ML + CNN (+ transfer learning if available)
===================================================================
train_traditional_ml.py, train_cnn.py, and train_transfer_learning.py
each save class PROBABILITIES for the validation and test sets, in the
SAME row order (all built from the same val_df/test_df split from
preprocessing.py). That makes late-fusion almost free: average the
models' probability vectors and take the argmax.

This script:
  1) Auto-detects which of the two/three/four models have saved probability
     files (traditional ML is required; CNN, transfer-learning and PANNs
     are each used automatically if present).
  2) Tries TWO fusion strategies and picks whichever scores higher on the
     VALIDATION set:
       a) Weighted averaging -- coarse grid search over fusion weights
          (summing to 1), including the edge cases of using just one or
          two of the models, in case fusion doesn't actually help.
       b) Stacking -- a Logistic Regression meta-classifier trained on the
          concatenated per-class probability vectors from every available
          model. Unlike a single fixed weight per model, stacking can
          learn e.g. "trust the CNN more specifically when Sway is
          involved" instead of one global trade-off -- often stronger once
          3+ models are available, at the cost of needing enough
          validation samples to fit it without overfitting (which is
          exactly why both strategies are tried and compared honestly on
          validation instead of assuming stacking always wins).
  3) Applies whichever strategy won ONCE on the untouched TEST set and
     reports final metrics + confusion matrix.

Run this AFTER train_traditional_ml.py and train_cnn.py (and, if you
made them, train_transfer_learning.py / train_panns.py).

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python evaluate_ensemble.py
"""

import os
import pickle
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

BASE_DIR = r"C:\Users\hp\Desktop\car_test"
DATA_DIR = os.path.join(BASE_DIR, "processed_data")

WEIGHT_STEP = 0.1  # grid resolution for the weight search (0.1 = 11 values per model)

# each entry: (readable name, val-probs filename, test-probs filename) -- traditional ML
# is always required; the others are included automatically only if their files exist.
MODEL_SOURCES = [
    ("Traditional ML", "traditional_val_probs.npy", "traditional_test_probs.npy"),
    ("CNN", "cnn_val_probs.npy", "cnn_test_probs.npy"),
    ("Transfer Learning (YAMNet)", "transfer_val_probs.npy", "transfer_test_probs.npy"),
    ("PANNs (CNN14)", "panns_val_probs.npy", "panns_test_probs.npy"),
    ("AST", "ast_val_probs.npy", "ast_test_probs.npy"),
    ("CLAP", "clap_val_probs.npy", "clap_test_probs.npy"),
    ("PaSST", "passt_val_probs.npy", "passt_test_probs.npy"),
    ("BEATs", "beats_val_probs.npy", "beats_test_probs.npy"),
    ("EfficientAT (fine-tuned)", "efficientat_ft_val_probs.npy", "efficientat_ft_test_probs.npy"),
]

# NOTE (2026-08-15): the currently-DEPLOYED ensemble_config.pkl was hand-tuned via a separate
# nested-CV analysis (see car_test_ensemble_fix_20260811 in project memory / SESSION_HANDOFF.md),
# NOT by a plain run of this script's own single-split weight/stacking comparison -- that
# comparison is known to be noisy on a 134-sample validation set, and gets noisier the more
# models are added to the search (more free parameters, same tiny val set). Do NOT casually
# re-run this script's main() expecting it to reproduce or safely refresh the deployed config
# now that AST/CLAP/PaSST/BEATs/EfficientAT are also available here -- it WILL auto-include all
# of them in the search and overwrite ensemble_config.pkl with an untested result. Fusing the
# new comparison models into production is a deliberate future decision, not a side effect of
# running this file.


def score_macro_f1(probs, y_true):
    y_pred = np.argmax(probs, axis=1)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def weight_grid(n_models, step=WEIGHT_STEP):
    """All combinations of n_models non-negative weights (multiples of `step`) that sum to 1."""
    steps = int(round(1.0 / step))
    for combo in itertools.product(range(steps + 1), repeat=n_models - 1):
        used = sum(combo)
        if used > steps:
            continue
        last = steps - used
        weights = tuple(c * step for c in combo) + (last * step,)
        yield weights


def main():
    print("=" * 70)
    print("Ensemble evaluation - fusing every available trained model")
    print("=" * 70)

    if not os.path.exists(os.path.join(DATA_DIR, "traditional_val_probs.npy")):
        print("\n!! traditional_val_probs.npy not found. Run train_traditional_ml.py first "
              "(it must finish successfully -- this script requires at least that model).")
        return

    available = []
    for name, val_file, test_file in MODEL_SOURCES:
        val_path = os.path.join(DATA_DIR, val_file)
        test_path = os.path.join(DATA_DIR, test_file)
        if os.path.exists(val_path) and os.path.exists(test_path):
            available.append((name, np.load(val_path), np.load(test_path)))
        else:
            print(f"   (skipping {name}: not trained yet -- run its script first if you want it included)")

    if len(available) < 2:
        print("\n!! Need at least 2 trained models with saved probabilities to build an ensemble. "
              "Train the CNN (and/or transfer learning model) too, then re-run this script.")
        return

    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)

    names = [a[0] for a in available]
    val_probs_list = [a[1] for a in available]
    test_probs_list = [a[2] for a in available]

    for name, vp in zip(names, val_probs_list):
        print(f"   standalone validation macro-F1 -- {name}: {score_macro_f1(vp, y_val):.4f}")

    print(f"\n[1a] Searching fusion WEIGHTS on the VALIDATION set for: {', '.join(names)} ...")
    best_weights, best_weights_val_f1 = None, -1
    for weights in weight_grid(len(available)):
        fused_val = sum(w * vp for w, vp in zip(weights, val_probs_list))
        f1 = score_macro_f1(fused_val, y_val)
        if f1 > best_weights_val_f1:
            best_weights_val_f1, best_weights = f1, weights

    print("   -> best weights: " + ", ".join(f"{n}={w:.2f}" for n, w in zip(names, best_weights)))
    print(f"      (validation macro-F1 = {best_weights_val_f1:.4f})")

    print(f"\n[1b] Training a STACKING meta-classifier (Logistic Regression) on the "
          f"concatenated probabilities of: {', '.join(names)} ...")
    X_val_stack = np.concatenate(val_probs_list, axis=1)
    X_test_stack = np.concatenate(test_probs_list, axis=1)
    meta_model = LogisticRegression(max_iter=2000, class_weight="balanced")
    meta_model.fit(X_val_stack, y_val)
    stacking_val_f1 = score_macro_f1(meta_model.predict_proba(X_val_stack), y_val)
    print(f"      (validation macro-F1 = {stacking_val_f1:.4f})")

    if stacking_val_f1 > best_weights_val_f1:
        fusion_mode = "stacking"
        best_val_f1 = stacking_val_f1
        print(f"\n[2] STACKING wins on validation ({stacking_val_f1:.4f} > {best_weights_val_f1:.4f}) "
              f"-- applying it on the TEST set (never used for tuning)...")
        fused_test = meta_model.predict_proba(X_test_stack)
    else:
        fusion_mode = "weights"
        best_val_f1 = best_weights_val_f1
        print(f"\n[2] Weighted averaging wins (or ties) on validation "
              f"({best_weights_val_f1:.4f} >= {stacking_val_f1:.4f}) -- applying those weights "
              f"on the TEST set (never used for tuning)...")
        fused_test = sum(w * tp for w, tp in zip(best_weights, test_probs_list))

    y_pred = np.argmax(fused_test, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    print(f"   Ensemble test accuracy: {acc:.4f}")
    print(f"   Ensemble test macro-F1: {f1_macro:.4f}")
    print(f"   Ensemble test macro-precision: {precision_macro:.4f}")
    print(f"   Ensemble test macro-recall: {recall_macro:.4f}")
    print("\n" + report)

    with open(os.path.join(DATA_DIR, "ensemble_report.txt"), "w", encoding="utf-8") as f:
        f.write("Models fused: " + ", ".join(names) + "\n")
        f.write(f"Fusion strategy chosen (by validation macro-F1): {fusion_mode}\n")
        f.write(f"  - weighted averaging: val macro-F1={best_weights_val_f1:.4f}, "
                + ", ".join(f"{n}={w:.2f}" for n, w in zip(names, best_weights)) + "\n")
        f.write(f"  - stacking (Logistic Regression): val macro-F1={stacking_val_f1:.4f}\n")
        f.write(f"Validation macro-F1 of the chosen strategy: {best_val_f1:.4f}\n\n")
        f.write(f"Test accuracy: {acc:.4f}\n")
        f.write(f"Test macro-F1: {f1_macro:.4f}\n")
        f.write(f"Test macro-precision: {precision_macro:.4f}\n")
        f.write(f"Test macro-recall: {recall_macro:.4f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax, cmap="Purples", values_format="d")
    ax.set_title("Confusion Matrix (Test) - Ensemble")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "ensemble_confusion_matrix.png"), dpi=150)
    plt.close()

    with open(os.path.join(DATA_DIR, "ensemble_config.pkl"), "wb") as f:
        pickle.dump({
            "mode": fusion_mode,           # "weights" or "stacking"
            "names": names,                # order used for both weights and the stacking features
            "weights": best_weights,       # used when mode == "weights"
            "meta_model": meta_model,      # used when mode == "stacking"
        }, f)

    print("\n" + "=" * 70)
    print("DONE. Saved:")
    print("   - ensemble_report.txt")
    print("   - ensemble_confusion_matrix.png")
    print(f"   - ensemble_config.pkl (fusion strategy: {fusion_mode}, used by predict.py --model ensemble)")
    print("=" * 70)
    print("\nCompare this macro-F1 against each standalone model's report -- fusion usually "
          "helps a bit, but report whatever actually happened on your data, honestly.")


if __name__ == "__main__":
    main()