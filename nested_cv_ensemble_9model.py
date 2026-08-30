# -*- coding: utf-8 -*-
r"""
===================================================================
Nested-CV validation: does expanding the ensemble from 4 to 9 models
genuinely help, or is the naive single-split search overfitting a
134-sample validation set?
===================================================================
Direct continuation of the 2026-08-11 ensemble fix (see project memory
car_test_ensemble_fix_20260811 / SESSION_HANDOFF.md): that investigation
found a single val-split weight/stacking comparison unreliable on this
project's small validation set, and that the problem gets WORSE as more
models (more free weight parameters) are added to the search -- exactly
evaluate_ensemble.py's own warning comment about naively re-running it
now that AST/CLAP/PaSST/BEATs/EfficientAT are all available.

This script splits the 134 validation samples into 5 folds. For each
fold, it re-runs the weight search on the OTHER 4 folds (never the held-
out one) for three candidate configurations, then scores each on the
held-out fold:
  (a) CURRENT DEPLOYED: fixed weights (0.30 Traditional / 0.60 CNN /
      0.10 YAMNet), no search at all -- the honest baseline.
  (b) 4-MODEL SEARCH: weight search restricted to the same four models
      already in production (Traditional/CNN/YAMNet/PANNs), to isolate
      whether just re-optimizing among the ALREADY-DEPLOYED models helps.
  (c) 9-MODEL SEARCH: weight search across all nine models.
Comparing the out-of-fold mean/std across these three tells us whether
the naive single-split "CLAP=0.60, EfficientAT=0.40" result found by
evaluate_ensemble.py is a real, generalizable improvement or noise.

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python nested_cv_ensemble_9model.py
"""
import os
import pickle
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
WEIGHT_STEP = 0.1
N_FOLDS = 5
RANDOM_STATE = 42

MODEL_SOURCES = [
    ("Traditional ML", "traditional_val_probs.npy"),
    ("CNN", "cnn_val_probs.npy"),
    ("Transfer Learning (YAMNet)", "transfer_val_probs.npy"),
    ("PANNs (CNN14)", "panns_val_probs.npy"),
    ("AST", "ast_val_probs.npy"),
    ("CLAP", "clap_val_probs.npy"),
    ("PaSST", "passt_val_probs.npy"),
    ("BEATs", "beats_val_probs.npy"),
    ("EfficientAT (fine-tuned)", "efficientat_ft_val_probs.npy"),
]

DEPLOYED_NAMES = ["Traditional ML", "CNN", "Transfer Learning (YAMNet)"]
DEPLOYED_WEIGHTS = (0.30, 0.60, 0.10)

FOUR_MODEL_NAMES = ["Traditional ML", "CNN", "Transfer Learning (YAMNet)", "PANNs (CNN14)"]


def score_macro_f1(probs, y_true):
    y_pred = np.argmax(probs, axis=1)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def weight_grid(n_models, step=WEIGHT_STEP):
    steps = int(round(1.0 / step))
    for combo in itertools.product(range(steps + 1), repeat=n_models - 1):
        used = sum(combo)
        if used > steps:
            continue
        last = steps - used
        yield tuple(c * step for c in combo) + (last * step,)


def best_weights_on(probs_list, y):
    best_w, best_f1 = None, -1
    for w in weight_grid(len(probs_list)):
        fused = sum(wi * p for wi, p in zip(w, probs_list))
        f1 = score_macro_f1(fused, y)
        if f1 > best_f1:
            best_f1, best_w = f1, w
    return best_w, best_f1


def main():
    all_probs = {}
    for name, val_file in MODEL_SOURCES:
        path = os.path.join(DATA_DIR, val_file)
        if os.path.exists(path):
            all_probs[name] = np.load(path)
    print("Models available:", ", ".join(all_probs.keys()))

    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    n = len(y_val)
    print(f"n validation samples: {n}\n")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    idx = np.arange(n)

    results = {"deployed_fixed": [], "4model_search": [], "9model_search": []}
    fold_weights_9model = []
    fold_weights_4model = []

    for fold_i, (train_idx, test_idx) in enumerate(kf.split(idx)):
        y_tr, y_te = y_val[train_idx], y_val[test_idx]

        # (a) deployed fixed weights -- no search, just score on held-out fold
        deployed_probs_tr = [all_probs[n_] for n_ in DEPLOYED_NAMES]
        fused_te = sum(w * p[test_idx] for w, p in zip(DEPLOYED_WEIGHTS, deployed_probs_tr))
        results["deployed_fixed"].append(score_macro_f1(fused_te, y_te))

        # (b) 4-model search (restricted to already-deployed model family + PANNs)
        four_probs_tr = [all_probs[n_][train_idx] for n_ in FOUR_MODEL_NAMES]
        w4, _ = best_weights_on(four_probs_tr, y_tr)
        four_probs_te = [all_probs[n_][test_idx] for n_ in FOUR_MODEL_NAMES]
        fused_te = sum(w * p for w, p in zip(w4, four_probs_te))
        results["4model_search"].append(score_macro_f1(fused_te, y_te))
        fold_weights_4model.append(w4)

        # (c) 9-model search
        names9 = list(all_probs.keys())
        nine_probs_tr = [all_probs[n_][train_idx] for n_ in names9]
        w9, _ = best_weights_on(nine_probs_tr, y_tr)
        nine_probs_te = [all_probs[n_][test_idx] for n_ in names9]
        fused_te = sum(w * p for w, p in zip(w9, nine_probs_te))
        results["9model_search"].append(score_macro_f1(fused_te, y_te))
        fold_weights_9model.append(w9)

        print(f"Fold {fold_i+1}/{N_FOLDS}: deployed={results['deployed_fixed'][-1]:.4f}  "
              f"4model={results['4model_search'][-1]:.4f} (w={dict(zip(FOUR_MODEL_NAMES, w4))})  "
              f"9model={results['9model_search'][-1]:.4f} (w={dict((k,v) for k,v in zip(names9, w9) if v>0)})")

    print("\n" + "=" * 70)
    print("SUMMARY (out-of-fold macro-F1, mean +/- std across 5 folds)")
    print("=" * 70)
    for key, label in [("deployed_fixed", "Deployed fixed weights (0.30/0.60/0.10, 3 models)"),
                        ("4model_search", "4-model weight search (Traditional/CNN/YAMNet/PANNs)"),
                        ("9model_search", "9-model weight search (all available models)")]:
        vals = np.array(results[key])
        print(f"{label:60s} mean={vals.mean():.4f}  std={vals.std():.4f}  min={vals.min():.4f}  max={vals.max():.4f}")

    print("\nPer-fold 9-model winning weight sets:")
    names9 = list(all_probs.keys())
    for i, w in enumerate(fold_weights_9model):
        nonzero = {n_: v for n_, v in zip(names9, w) if v > 0}
        print(f"  fold {i+1}: {nonzero}")


if __name__ == "__main__":
    main()
