# -*- coding: utf-8 -*-
r"""
===================================================================
Traditional ML training - Belt / Brake / Sway (using MFCC features)
Matches proposal section 5, "Model 1: Traditional Machine Learning"
===================================================================
Trains SVM and Random Forest (the two classifiers suggested in the
proposal), plus XGBoost as an extra stronger baseline if installed,
each over a small hyperparameter grid, on the MFCC features produced
by preprocessing.py.

MODEL SELECTION: each candidate is ranked with stratified K-fold cross-
validation on the TRAIN split only (macro-F1). A single val-set score
(only ~130 files, ~15% of the data) is noisy enough that a few points of
difference between two configs can be pure sample variance rather than a
real improvement (see README limitations). K-fold CV uses many more
train/holdout combinations, giving a much more stable ranking. The val
and test sets stay completely untouched by model selection, and are only
used afterwards to report honest, final metrics for the winning config.

HOW TO RUN:
    pip install xgboost   (optional, but recommended -- usually beats plain Random Forest)
    cd C:\Users\hp\Desktop\car_test
    python train_traditional_ml.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
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
    print("NOTE: xgboost is not installed (pip install xgboost) -- skipping the XGBoost model, "
          "will only compare SVM and Random Forest.")

BASE_DIR = os.environ.get("CAR_TEST_RAW_DATA_DIR", r"C:\Users\hp\Desktop\car_test")
DATA_DIR = os.environ.get("CAR_TEST_OUTPUT_DIR", os.path.join(BASE_DIR, "processed_data"))

RANDOM_STATE = 42
CV_FOLDS = 5  # stratified K-fold CV on the TRAIN split, used only for model selection

# Small hand-rolled hyperparameter grids (ranked via cross-validation below) instead of
# hard-coded single settings -- a lightweight stand-in for GridSearchCV that keeps the
# rest of the pipeline (report generation, saved artifacts) unchanged.
SVM_GRID = [
    {"C": c, "gamma": g}
    for c in [1, 10, 50]
    for g in ["scale", 0.01, 0.001]
]

RF_GRID = [
    {"n_estimators": n, "max_depth": d}
    for n in [200, 300, 500]
    for d in [None, 20, 40]
]

XGB_GRID = [
    {"n_estimators": n, "max_depth": d, "learning_rate": lr}
    for n in [200, 300]
    for d in [3, 6]
    for lr in [0.05, 0.1]
]


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train_mfcc.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val_mfcc.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test_mfcc.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder


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


def build_candidate_specs():
    """(name, estimator_class, base_kwargs) per grid point -- NOT fitted yet.
    base_kwargs never sets 'probability' for SVM: during the CV search every
    fold would otherwise ALSO run its own internal 5-fold Platt-scaling CV
    just to expose predict_proba, which is never used for ranking (the
    f1_macro scorer only calls .predict()) -- that made grid search far
    slower than necessary. probability=True is only turned on once, for the
    single winning config, after search is done.
    RandomForest/XGBoost get n_jobs=-1 (and XGBoost tree_method='hist') so
    each fit uses every CPU core instead of just one.
    """
    specs = []

    for params in SVM_GRID:
        name = f"SVM (C={params['C']}, gamma={params['gamma']})"
        specs.append((name, SVC, dict(kernel="rbf", random_state=RANDOM_STATE,
                                       class_weight="balanced", **params)))

    for params in RF_GRID:
        name = f"Random Forest (n_estimators={params['n_estimators']}, max_depth={params['max_depth']})"
        specs.append((name, RandomForestClassifier, dict(random_state=RANDOM_STATE,
                                                           class_weight="balanced", n_jobs=-1, **params)))

    if HAS_XGBOOST:
        for params in XGB_GRID:
            name = (f"XGBoost (n_estimators={params['n_estimators']}, "
                     f"max_depth={params['max_depth']}, lr={params['learning_rate']})")
            specs.append((name, XGBClassifier, dict(random_state=RANDOM_STATE, eval_metric="mlogloss",
                                                      n_jobs=-1, tree_method="hist", **params)))

    return specs


def build_estimator(cls, base_kwargs, want_probability):
    kwargs = dict(base_kwargs)
    if cls is SVC:
        kwargs["probability"] = want_probability
    return cls(**kwargs)


def main():
    print("=" * 70)
    print("Training Traditional ML models: SVM vs Random Forest vs XGBoost (CV-ranked grid search)")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_data()
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    specs = build_candidate_specs()
    print(f"\nTrying {len(specs)} candidate configurations "
          f"({len(SVM_GRID)} SVM + {len(RF_GRID)} Random Forest"
          + (f" + {len(XGB_GRID)} XGBoost" if HAS_XGBOOST else "") + ")")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n[1] Ranking every candidate with {CV_FOLDS}-fold cross-validation on the "
          f"TRAIN split only (macro-F1) -- val/test stay untouched until a winner is picked...\n")
    cv_results = []
    for name, cls, base_kwargs in specs:
        est = build_estimator(cls, base_kwargs, want_probability=False)
        # SVM has no internal parallelism of its own -- let cross_val_score parallelize
        # across folds instead. RF/XGB already parallelize internally (n_jobs=-1 above),
        # so keep the CV loop itself sequential for them to avoid oversubscribing cores.
        cv_n_jobs = -1 if cls is SVC else 1
        scores = cross_val_score(est, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=cv_n_jobs)
        cv_results.append({"name": name, "cls": cls, "kwargs": base_kwargs,
                            "cv_mean": scores.mean(), "cv_std": scores.std()})
        print(f"   -> {name}: CV macro-F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best = max(cv_results, key=lambda r: r["cv_mean"])
    print(f"\n[2] Best configuration by {CV_FOLDS}-fold CV macro-F1: {best['name']} "
          f"({best['cv_mean']:.4f} +/- {best['cv_std']:.4f})")

    print(f"\n[3] Refitting '{best['name']}' on the FULL train split"
          + (" (probability=True this time, for calibrated confidence scores)" if best["cls"] is SVC else "")
          + "...")
    best_model = build_estimator(best["cls"], best["kwargs"], want_probability=True)
    best_model.fit(X_train, y_train)
    best_name = best["name"]

    print("\n[4] Evaluating on the VALIDATION set (held out from model selection entirely)...")
    val_res = evaluate(best_model, X_val, y_val, class_names)
    print(f"   Val accuracy: {val_res['accuracy']:.4f} | Val macro-F1: {val_res['f1_macro']:.4f}")

    print(f"\n[5] Final evaluation of '{best_name}' on the TEST set (never seen before)...")
    test_res = evaluate(best_model, X_test, y_test, class_names)
    print(f"   Test accuracy: {test_res['accuracy']:.4f}")
    print(f"   Test macro-F1: {test_res['f1_macro']:.4f}")
    print(f"   Test macro-precision: {test_res['precision_macro']:.4f}")
    print(f"   Test macro-recall: {test_res['recall_macro']:.4f}")
    print("\n" + test_res["report"])

    report_lines = [f"BEST MODEL: {best_name}\n"]
    report_lines.append(f"=== {CV_FOLDS}-fold CV ranking (TRAIN split only, macro-F1) ===")
    for r in sorted(cv_results, key=lambda r: -r["cv_mean"]):
        report_lines.append(f"{r['name']}: cv_macro_F1={r['cv_mean']:.4f} (+/- {r['cv_std']:.4f})")
    report_lines.append(f"\n=== VALIDATION results (winner only, not used for model selection) ===")
    report_lines.append(f"accuracy={val_res['accuracy']:.4f}, macro-F1={val_res['f1_macro']:.4f}")
    report_lines.append(val_res["report"])
    report_lines.append("\n=== FINAL TEST results (winner only) ===")
    report_lines.append(f"accuracy={test_res['accuracy']:.4f}, macro-F1={test_res['f1_macro']:.4f}, "
                         f"macro-precision={test_res['precision_macro']:.4f}, macro-recall={test_res['recall_macro']:.4f}")
    report_lines.append(test_res["report"])

    with open(os.path.join(DATA_DIR, "traditional_ml_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    with open(os.path.join(DATA_DIR, "best_traditional_model.pkl"), "wb") as f:
        pickle.dump({"model": best_model, "name": best_name}, f)

    # save probabilities on val + test for the best model -- reused by evaluate_ensemble.py
    np.save(os.path.join(DATA_DIR, "traditional_test_probs.npy"), best_model.predict_proba(X_test))
    np.save(os.path.join(DATA_DIR, "traditional_val_probs.npy"), best_model.predict_proba(X_val))

    # confusion matrix (test set)
    cm = confusion_matrix(y_test, test_res["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix (Test) - {best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "traditional_confusion_matrix.png"), dpi=150)
    plt.close()

    if hasattr(best_model, "feature_importances_"):
        feature_names = pd.read_csv(os.path.join(DATA_DIR, "features.csv")).drop(
            columns=["label", "file_name", "augmented", "split"]
        ).columns
        importances = best_model.feature_importances_
        idx = np.argsort(importances)[::-1][:20]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh(range(len(idx)), importances[idx][::-1])
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([feature_names[i] for i in idx][::-1])
        ax.set_title(f"Top 20 Feature Importances - {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(DATA_DIR, "traditional_feature_importance.png"), dpi=150)
        plt.close()

    print("\n" + "=" * 70)
    print("DONE. Saved:")
    print("   - best_traditional_model.pkl")
    print("   - traditional_ml_report.txt (CV ranking + val + test metrics)")
    print("   - traditional_confusion_matrix.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
