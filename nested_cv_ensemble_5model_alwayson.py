# -*- coding: utf-8 -*-
r"""
===================================================================
Nested-CV validation, restricted to the FIVE always-loaded models
===================================================================
Follow-up to nested_cv_ensemble_9model.py. That script showed a
9-model weight search (which includes CLAP) genuinely beats the
deployed 3-model fixed weights under proper out-of-fold validation.
But CLAP is opt-in-only in the live service (RAM/latency), so this
script repeats the exact same nested-CV procedure using ONLY the
models that are already computed on every prediction today:

    Traditional ML, CNN, YAMNet, PANNs (CNN14), EfficientAT (fine-tuned)

Procedure (identical to the 9-model script):
  * 5-fold split of the 134-sample validation set.
  * For each fold: grid weight search (0.1 step) on the OTHER 4 folds,
    score the winning vector on the held-out fold -> out-of-fold mean/std.
  * Compare against the deployed fixed weights (0.30/0.60/0.10, 3 models).
  * Final candidate = element-wise MEAN of the 5 folds' winning weight
    vectors (a regulariser against winner-take-all extremity), then
    evaluated EXACTLY ONCE on the untouched test set. No further
    test-set iteration after that number is printed.

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python nested_cv_ensemble_5model_alwayson.py
"""
import os
import itertools
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
WEIGHT_STEP = 0.1
N_FOLDS = 5
RANDOM_STATE = 42

ALWAYSON_SOURCES = [
    ("Traditional ML", "traditional_val_probs.npy", "traditional_test_probs.npy"),
    ("CNN", "cnn_val_probs.npy", "cnn_test_probs.npy"),
    ("Transfer Learning (YAMNet)", "transfer_val_probs.npy", "transfer_test_probs.npy"),
    ("PANNs (CNN14)", "panns_val_probs.npy", "panns_test_probs.npy"),
    ("EfficientAT (fine-tuned)", "efficientat_ft_val_probs.npy", "efficientat_ft_test_probs.npy"),
]

DEPLOYED_NAMES = ["Traditional ML", "CNN", "Transfer Learning (YAMNet)"]
DEPLOYED_WEIGHTS = (0.30, 0.60, 0.10)


def score_macro_f1(probs, y_true):
    return f1_score(y_true, np.argmax(probs, axis=1), average="macro", zero_division=0)


def weight_grid(n_models, step=WEIGHT_STEP):
    steps = int(round(1.0 / step))
    for combo in itertools.product(range(steps + 1), repeat=n_models - 1):
        used = sum(combo)
        if used > steps:
            continue
        last = steps - used
        yield tuple(c * step for c in combo) + (last * step,)


def best_weights_on(probs_list, y):
    best_w, best_f1 = None, -1.0
    for w in weight_grid(len(probs_list)):
        fused = sum(wi * p for wi, p in zip(w, probs_list))
        f1 = score_macro_f1(fused, y)
        if f1 > best_f1:
            best_f1, best_w = f1, w
    return best_w, best_f1


def main():
    names = [n for n, _, _ in ALWAYSON_SOURCES]
    val_probs = {n: np.load(os.path.join(DATA_DIR, vf)) for n, vf, _ in ALWAYSON_SOURCES}
    test_probs = {n: np.load(os.path.join(DATA_DIR, tf)) for n, _, tf in ALWAYSON_SOURCES}
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    print("Always-on models:", ", ".join(names))
    print(f"n val = {len(y_val)}   n test = {len(y_test)}\n")

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    idx = np.arange(len(y_val))

    oof_deployed, oof_5model = [], []
    fold_weights = []

    for fold_i, (tr, te) in enumerate(kf.split(idx)):
        y_tr, y_te = y_val[tr], y_val[te]

        dep_te = sum(w * val_probs[n][te] for w, n in zip(DEPLOYED_WEIGHTS, DEPLOYED_NAMES))
        oof_deployed.append(score_macro_f1(dep_te, y_te))

        probs_tr = [val_probs[n][tr] for n in names]
        w5, _ = best_weights_on(probs_tr, y_tr)
        probs_te = [val_probs[n][te] for n in names]
        fused_te = sum(w * p for w, p in zip(w5, probs_te))
        oof_5model.append(score_macro_f1(fused_te, y_te))
        fold_weights.append(w5)

        nz = {n: round(v, 2) for n, v in zip(names, w5) if v > 0}
        print(f"Fold {fold_i+1}/{N_FOLDS}: deployed={oof_deployed[-1]:.4f}  "
              f"5model={oof_5model[-1]:.4f}  w={nz}")

    dep = np.array(oof_deployed)
    m5 = np.array(oof_5model)
    print("\n" + "=" * 70)
    print("OUT-OF-FOLD macro-F1 (mean +/- std, 5 folds)")
    print("=" * 70)
    print(f"Deployed fixed (0.30/0.60/0.10, 3 models)     mean={dep.mean():.4f}  std={dep.std():.4f}")
    print(f"5-model always-on weight search               mean={m5.mean():.4f}  std={m5.std():.4f}")

    avg_w = np.mean(np.array(fold_weights), axis=0)
    avg_w = avg_w / avg_w.sum()
    avg_w_round = np.round(avg_w, 2)
    avg_w_round = avg_w_round / avg_w_round.sum()  # renormalise after rounding
    print("\nElement-wise mean of the 5 folds' winning weight vectors:")
    for n, v, vr in zip(names, avg_w, avg_w_round):
        print(f"  {n:32s} {v:.4f}  (rounded -> {vr:.2f})")

    # ----- ONE-SHOT test-set evaluation of the averaged weight vector -----
    fused_test = sum(w * test_probs[n] for w, n in zip(avg_w_round, names))
    y_pred = np.argmax(fused_test, axis=1)
    acc = accuracy_score(y_test, y_pred)
    mf1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print("\n" + "=" * 70)
    print("ONE-SHOT TEST-SET RESULT (averaged fold weights, rounded)")
    print("=" * 70)
    print(f"weights (rounded) = {dict((n, round(float(v), 2)) for n, v in zip(names, avg_w_round))}")
    print(f"test accuracy  = {acc*100:.2f}%")
    print(f"test macro-F1  = {mf1*100:.2f}%")

    # deployed config's own test number, for reference
    dep_test = sum(w * test_probs[n] for w, n in zip(DEPLOYED_WEIGHTS, DEPLOYED_NAMES))
    dep_pred = np.argmax(dep_test, axis=1)
    print(f"\n(reference) deployed 3-model config on same test set: "
          f"acc={accuracy_score(y_test, dep_pred)*100:.2f}%  "
          f"macroF1={f1_score(y_test, dep_pred, average='macro', zero_division=0)*100:.2f}%")


if __name__ == "__main__":
    main()
