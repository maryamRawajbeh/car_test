# -*- coding: utf-8 -*-
r"""
===================================================================
Augmentation x seed variance study - Traditional ML (SVM/RF/XGBoost)
===================================================================
train_traditional_ml.py's grid search + CV-ranking is otherwise fully
DETERMINISTIC: same MFCC features in, same RANDOM_STATE=42 everywhere
(SVM's probability-calibration folds, RF's bootstrap, XGBoost's tree
construction, the CV fold assignment itself) -- running it twice on
the same data gives byte-identical results. So unlike the neural-net
models, "5 runs" only means something here if the RANDOM_STATE itself
varies per run; this script does exactly that (mirrors
train_traditional_ml.py's model-selection logic but with RANDOM_STATE
swapped for each of 5 seeds), on top of the SAME two full-data feature
caches the CNN variance study also reuses:
  ablation_results/no_augmentation/    (624 train files, no aug)
  ablation_results/with_augmentation/  (1872 train files = 624 x (1 + 2 aug copies))

Results appended to processed_data/traditional_ml_augmentation_variance.csv
(one row per (condition, seed), flushed immediately).

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python run_traditional_ml_augmentation_variance.py [--seeds 1,2,3,4,5]
"""

import os
import csv
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score

import train_traditional_ml as tml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATION_DIR = os.path.join(SCRIPT_DIR, "ablation_results")
DATA_DIR = os.path.join(SCRIPT_DIR, "processed_data")
RESULTS_CSV = os.path.join(DATA_DIR, "traditional_ml_augmentation_variance.csv")

CONDITIONS = ["no_augmentation", "with_augmentation"]
CV_FOLDS = tml.CV_FOLDS


def load_condition(name):
    d = os.path.join(ABLATION_DIR, name)
    X_train = np.load(os.path.join(d, "X_train_mfcc.npy"))
    X_val = np.load(os.path.join(d, "X_val_mfcc.npy"))
    X_test = np.load(os.path.join(d, "X_test_mfcc.npy"))
    y_train = np.load(os.path.join(d, "y_train.npy"))
    y_val = np.load(os.path.join(d, "y_val.npy"))
    y_test = np.load(os.path.join(d, "y_test.npy"))
    with open(os.path.join(d, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    return X_train, X_val, X_test, y_train, y_val, y_test, list(label_encoder.classes_)


def run_once(X_train, y_train, X_val, y_val, X_test, y_test, seed):
    """Same CV-rank -> refit-winner -> evaluate pipeline as train_traditional_ml.py's
    main(), but with RANDOM_STATE swapped to `seed` everywhere (grid contents/hyperparams
    are untouched -- only the source of randomness moves)."""
    tml.RANDOM_STATE = seed  # build_candidate_specs() reads this global at call time
    specs = tml.build_candidate_specs()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)

    cv_results = []
    for name, cls, base_kwargs in specs:
        est = tml.build_estimator(cls, base_kwargs, want_probability=False)
        cv_n_jobs = -1 if cls.__name__ == "SVC" else 1
        scores = cross_val_score(est, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=cv_n_jobs)
        cv_results.append({"name": name, "cls": cls, "kwargs": base_kwargs, "cv_mean": scores.mean()})

    best = max(cv_results, key=lambda r: r["cv_mean"])
    best_model = tml.build_estimator(best["cls"], best["kwargs"], want_probability=True)
    best_model.fit(X_train, y_train)

    y_test_pred = best_model.predict(X_test)
    return {
        "best_name": best["name"],
        "cv_macro_f1": best["cv_mean"],
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_macro_f1": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3,4,5")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    is_new_file = not os.path.exists(RESULTS_CSV)
    csv_file = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if is_new_file:
        writer.writerow(["condition", "seed", "best_model", "cv_macro_f1", "test_accuracy", "test_macro_f1"])

    for cond in CONDITIONS:
        print("\n" + "=" * 70)
        print(f"CONDITION: {cond}")
        print("=" * 70)
        X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_condition(cond)
        print(f"  train={X_train.shape[0]}  val={X_val.shape[0]}  test={X_test.shape[0]}")

        for seed in seeds:
            res = run_once(X_train, y_train, X_val, y_val, X_test, y_test, seed)
            print(f"  seed={seed}: best={res['best_name']}  "
                  f"cv_f1={res['cv_macro_f1']:.4f}  test_acc={res['test_accuracy']:.4f}  "
                  f"test_f1={res['test_macro_f1']:.4f}")
            writer.writerow([cond, seed, res["best_name"], f"{res['cv_macro_f1']:.4f}",
                              f"{res['test_accuracy']:.4f}", f"{res['test_macro_f1']:.4f}"])
            csv_file.flush()

    csv_file.close()
    print(f"\nDONE. Results appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
