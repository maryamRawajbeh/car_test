# -*- coding: utf-8 -*-
r"""
===================================================================
Does augmentation actually help/hurt the CNN, once NN training noise
is accounted for?
===================================================================
run_augmentation_ablation.py already compared no_augmentation vs
with_augmentation ONCE each (see ablation_results/*/cnn_report.txt):
  no_augmentation:   77.44%... wait, see file -- 0.8045 test accuracy
  with_augmentation: 0.7744 test accuracy
A ~3-point gap. But the CNN A/B tests done this session (run_ab_test_cnn.py)
showed a single CNN training run can land 87.22% one time and ~80.5% the
"documented" time -- run-to-run noise from weight init / batch order /
early-stopping timing is real, even with a fixed seed (TF isn't perfectly
deterministic on CPU with multi-threaded ops). So a single-run 3-point gap
between conditions could easily just be noise, not a real effect.

This script settles it: retrains the CNN N_REPEATS times per condition,
using a DIFFERENT random seed each time (so each repeat is a genuinely
independent draw, not just TF's residual nondeterminism around one seed),
and reports the full distribution (mean +/- std) per condition instead of
one number each.

Does NOT touch preprocessing or Freesound data at all -- reuses the
already-extracted, already-normalized mel spectrograms sitting in
ablation_results/no_augmentation/ and ablation_results/with_augmentation/
exactly as train_cnn.py would load them.

Run:
  cd C:\Users\hp\Desktop\car_test
  python -u run_cnn_augmentation_variance.py [--repeats 5]
"""

import os
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATION_DIR = os.path.join(SCRIPT_DIR, "ablation_results")
# "compression_augmented" added after mic_pipeline_test/test_mic_pipeline_domain_shift.py
# found the deployed model collapsing under the real webm/opus pipeline -- this condition
# is the fix (TRAIN-only clips round-tripped through that same real pipeline, see
# compression_augment.py). Verifying it holds up over 5 independent CNN training runs,
# not just the one lucky/unlucky run already done, before it goes anywhere near production.
# Default kept as-is for backward compat; override with --conditions for other studies
# (e.g. --conditions no_augmentation,with_augmentation for the original waveform-aug axis).
DEFAULT_CONDITIONS = ["compression_augmented", "compression_and_mic_augmented"]

EPOCHS = 40
BATCH_SIZE = 16
USE_TEMPORAL_HEAD = True
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0


def load_condition(name):
    d = os.path.join(ABLATION_DIR, name)
    X_train = np.load(os.path.join(d, "X_train_mel.npy"))
    X_val = np.load(os.path.join(d, "X_val_mel.npy"))
    X_test = np.load(os.path.join(d, "X_test_mel.npy"))
    y_train = np.load(os.path.join(d, "y_train.npy"))
    y_val = np.load(os.path.join(d, "y_val.npy"))
    y_test = np.load(os.path.join(d, "y_test.npy"))
    with open(os.path.join(d, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    return X_train, X_val, X_test, y_train, y_val, y_test, list(label_encoder.classes_)


def train_once(X_train, y_train, X_val, y_val, X_test, y_test, n_classes, seed):
    import tensorflow as tf
    from tensorflow.keras import callbacks
    from cnn_model import build_cnn

    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_cnn(input_shape=X_train.shape[1:], n_classes=n_classes,
                       use_temporal_head=USE_TEMPORAL_HEAD, use_focal_loss=USE_FOCAL_LOSS,
                       focal_gamma=FOCAL_GAMMA)

    class_weight_values = compute_class_weight(class_weight="balanced", classes=np.arange(n_classes), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weight_values)}

    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS,
                         batch_size=BATCH_SIZE, callbacks=[early_stop, reduce_lr],
                         class_weight=class_weight_dict, verbose=0)

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    n_epochs_ran = len(history.history["loss"])
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "epochs_ran": n_epochs_ran,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS),
                         help="Comma-separated ablation_results/<name> subfolders to compare")
    args = parser.parse_args()
    CONDITIONS = args.conditions.split(",")

    all_results = {}
    for cond in CONDITIONS:
        print(f"\n{'#'*70}\nCONDITION: {cond}\n{'#'*70}")
        X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_condition(cond)
        print(f"  train={X_train.shape[0]}  val={X_val.shape[0]}  test={X_test.shape[0]}")

        runs = []
        for i in range(args.repeats):
            seed = args.base_seed + i
            res = train_once(X_train, y_train, X_val, y_val, X_test, y_test, len(class_names), seed)
            runs.append(res)
            print(f"  run {i+1}/{args.repeats} (seed={seed}, {res['epochs_ran']} epochs): "
                  f"accuracy={res['accuracy']:.4f}  f1_macro={res['f1_macro']:.4f}")
        all_results[cond] = runs

    print("\n\n" + "=" * 70)
    print(f"SUMMARY -- {args.repeats} independent CNN training runs per condition")
    print("=" * 70)
    print(f"{'condition':20s} {'mean_acc':>10s} {'std_acc':>9s} {'mean_f1':>10s} {'std_f1':>9s} {'min_acc':>9s} {'max_acc':>9s}")
    stats = {}
    for cond in CONDITIONS:
        accs = [r["accuracy"] for r in all_results[cond]]
        f1s = [r["f1_macro"] for r in all_results[cond]]
        stats[cond] = (np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s), np.min(accs), np.max(accs))
        print(f"{cond:20s} {stats[cond][0]:10.4f} {stats[cond][1]:9.4f} {stats[cond][2]:10.4f} "
              f"{stats[cond][3]:9.4f} {stats[cond][4]:9.4f} {stats[cond][5]:9.4f}")

    mean_a, std_a = stats[CONDITIONS[0]][0], stats[CONDITIONS[0]][1]
    mean_b, std_b = stats[CONDITIONS[1]][0], stats[CONDITIONS[1]][1]
    gap = mean_a - mean_b
    pooled_std = np.sqrt((std_a ** 2 + std_b ** 2) / 2)
    print(f"\n{CONDITIONS[0]} - {CONDITIONS[1]} mean accuracy gap: {gap:+.4f}")
    print(f"per-condition std (run-to-run noise): {std_a:.4f} / {std_b:.4f} (pooled ~{pooled_std:.4f})")
    if pooled_std > 0 and abs(gap) < 1.5 * pooled_std:
        print("-> the gap is SMALLER than run-to-run noise -- likely NOT a real effect, just NN training variance.")
    else:
        print("-> the gap is LARGER than run-to-run noise -- likely a real effect, not just variance.")


if __name__ == "__main__":
    main()
