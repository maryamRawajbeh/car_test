# -*- coding: utf-8 -*-
r"""
===================================================================
Independent verification of best_traditional_model_with_other.pkl
===================================================================
Re-loads the SAVED artifacts fresh (doesn't trust train_other_class_detector.py's
own printed numbers) and checks the one failure mode that actually matters in
production: does adding the "other" gate ever cause a REAL belt/brake/sway
recording to be dismissed as "other" (a dangerous false negative -- a genuine
fault going unreported) -- as opposed to "other" audio being misclassified as a
real fault (a false positive -- annoying, but not dangerous).

Also cross-checks against the CURRENTLY DEPLOYED 3-class model on the exact same
real test files, so the comparison is apples-to-apples (same files, same split),
not two different classifiers' own self-reported numbers.

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python verify_other_class_detector.py
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from audio_common import load_clean_audio, extract_mfcc_vector
from predict import OTHER_GATE_THRESHOLD

BASE_DIR = r"C:\Users\hp\Desktop\car_test"
DATA_DIR = os.path.join(BASE_DIR, "processed_data")


def main():
    print("=" * 70)
    print("Independent verification of the \"other\"-class traditional model")
    print("=" * 70)

    with open(os.path.join(DATA_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)

    print("\n[1] Loading SAVED with_other artifacts fresh (not trusting the training run's own numbers)...")
    with open(os.path.join(DATA_DIR, "best_traditional_model_with_other.pkl"), "rb") as f:
        saved = pickle.load(f)
        other_model, other_model_name = saved["model"], saved["name"]
    with open(os.path.join(DATA_DIR, "scaler_with_other.pkl"), "rb") as f:
        other_scaler = pickle.load(f)
    with open(os.path.join(DATA_DIR, "label_encoder_with_other.pkl"), "rb") as f:
        other_encoder = pickle.load(f)
    other_classes = list(other_encoder.classes_)
    print(f"    model: {other_model_name}, classes: {other_classes}")

    print("\n[2] Loading the CURRENTLY DEPLOYED 3-class model for a controlled comparison...")
    with open(os.path.join(DATA_DIR, "best_traditional_model.pkl"), "rb") as f:
        deployed = pickle.load(f)
        deployed_model, deployed_model_name = deployed["model"], deployed["name"]
    with open(os.path.join(DATA_DIR, "scaler.pkl"), "rb") as f:
        deployed_scaler = pickle.load(f)
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        deployed_encoder = pickle.load(f)
    print(f"    model: {deployed_model_name}")

    print("\n[3] Re-extracting features for the real val+test belt/brake/sway files "
          "(same leakage-safe split as everywhere else in this project)...")
    doc = pd.read_csv(os.path.join(DATA_DIR, "dataset_documentation.csv"))
    corrupted_path = os.path.join(DATA_DIR, "corrupted_files.csv")
    if os.path.exists(corrupted_path):
        corrupted = pd.read_csv(corrupted_path)
        if "matched_path" in corrupted.columns:
            doc = doc[~doc["matched_path"].isin(corrupted["matched_path"])]

    def load_split(split_name):
        rows = doc[doc["split_assigned"] == split_name]
        feats, labels, paths = [], [], []
        for _, row in rows.iterrows():
            y, sr = load_clean_audio(row["matched_path"], target_sr=config["target_sr"],
                                      target_duration=config["target_duration"], fallback_to_untrimmed=True)
            feats.append(extract_mfcc_vector(y, sr, n_mfcc=config["n_mfcc"]))
            labels.append(row["class"])
            paths.append(row["matched_path"])
        return np.array(feats), np.array(labels), paths

    X_val_raw, y_val_str, val_paths = load_split("val")
    X_test_raw, y_test_str, test_paths = load_split("test")
    print(f"    val={len(y_val_str)}, test={len(y_test_str)} real files")

    for split_name, X_raw, y_str, paths in [("VAL", X_val_raw, y_val_str, val_paths), ("TEST", X_test_raw, y_test_str, test_paths)]:
        print(f"\n[4] {split_name} split -- with_other model's predictions on REAL belt/brake/sway files only...")
        X_scaled = other_scaler.transform(X_raw)
        pred_idx = other_model.predict(X_scaled)
        pred_str = np.array([other_classes[i] for i in pred_idx])

        acc = accuracy_score(y_str, pred_str)
        print(f"    accuracy (treating \"other\" as an available-but-wrong answer): {acc:.4f}")

        cm = confusion_matrix(y_str, pred_str, labels=other_classes)
        print(f"    confusion matrix (rows=true, cols={other_classes}):")
        for true_cls, row in zip(other_classes, cm):
            print(f"      {true_cls:8s}: {dict(zip(other_classes, row))}")

        # THE critical safety check: real fault recordings dismissed as "other"
        false_negatives = [(p, t) for p, t, pr in zip(paths, y_str, pred_str) if pr == "other"]
        print(f"    !! Real fault recordings misclassified as \"other\" (dangerous false negatives): "
              f"{len(false_negatives)} / {len(y_str)}")
        for p, t in false_negatives:
            print(f"       true={t}: {p}")

        print(f"\n[5] {split_name} split -- CONTROLLED comparison: currently-deployed 3-class model "
              f"on the EXACT SAME files...")
        y_deployed_true = deployed_encoder.transform(y_str)
        X_deployed_scaled = deployed_scaler.transform(X_raw)
        deployed_pred_idx = deployed_model.predict(X_deployed_scaled)
        deployed_pred_str = np.array([deployed_encoder.classes_[i] for i in deployed_pred_idx])
        deployed_acc = accuracy_score(y_deployed_true, deployed_pred_idx)
        print(f"    deployed 3-class model accuracy on these same {len(y_str)} files: {deployed_acc:.4f}")
        print(f"    with_other model accuracy on these same files:                    {acc:.4f}")
        print(f"    delta: {acc - deployed_acc:+.4f}")

        print(f"\n[6] {split_name} split -- AS ACTUALLY DEPLOYED: gate uses P(other) >= "
              f"{OTHER_GATE_THRESHOLD} (not argmax); when it doesn't fire, the real prediction "
              f"comes from the deployed 3-class model above (not from this with_other model's own "
              f"belt/brake/sway guess) -- this is exactly what predict.py does end to end...")
        probs = other_model.predict_proba(X_scaled)
        other_idx = other_classes.index("other")
        gate_fires = probs[:, other_idx] >= OTHER_GATE_THRESHOLD
        final_pred_str = np.where(gate_fires, "other", deployed_pred_str)
        final_acc = accuracy_score(y_str, final_pred_str)
        print(f"    end-to-end accuracy (treating \"other\" as an available-but-wrong answer): {final_acc:.4f}")
        final_fn = [(p, t) for p, t, pr in zip(paths, y_str, final_pred_str) if pr == "other"]
        print(f"    !! Real fault recordings dismissed as \"other\" end-to-end (dangerous false negatives): "
              f"{len(final_fn)} / {len(y_str)}")
        for p, t in final_fn:
            print(f"       true={t}: {p}")

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)


if __name__ == "__main__":
    main()
