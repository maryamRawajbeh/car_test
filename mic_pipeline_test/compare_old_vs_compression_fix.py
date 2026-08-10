# -*- coding: utf-8 -*-
r"""
===================================================================
Did the compression-augmentation fix actually work?
===================================================================
test_mic_pipeline_domain_shift.py measured the OLD deployed model (from
C:\Users\hp\Desktop\car_test\processed_data) collapsing under webm/opus
compression on 15 test files. This script is the real verdict: evaluate
BOTH the OLD model and the NEWLY TRAINED compression-augmented model
(ablation_results\compression_augmented -- trained on TRAIN-only clips
round-tripped through the exact same real webm/opus pipeline, see
compression_augment.py / preprocessing.py's COMPRESSION_AUGMENT_ENABLED)
on the SAME compressed audio, across the FULL 133-file test set (not
just 15) -- and compare accuracy at each compression level.

Both models are evaluated on EXACTLY the same file list (the deployed
model's own test split) and exactly the same compressed audio bytes per
file -- the only thing that differs is which model/scaler processes them.

Run:
  cd C:\Users\hp\Desktop\car_test\mic_pipeline_test
  python compare_old_vs_compression_fix.py
"""

import os
import sys
import pickle
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CAR_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAR_TEST_DIR)
from audio_common import load_clean_audio, extract_mfcc_vector  # noqa: E402
from compression_augment import compress_roundtrip  # noqa: E402

OLD_MODEL_DIR = r"C:\Users\hp\Desktop\car_test\processed_data"
NEW_MODEL_DIR = r"C:\Users\hp\Desktop\car_test\ablation_results\compression_augmented"
WEBM_BITRATES_KBPS = [16, 32, 64]
TMP_DIR = tempfile.mkdtemp(prefix="compare_old_vs_fix_")


def load_model(model_dir):
    with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, "best_traditional_model.pkl"), "rb") as f:
        model = pickle.load(f)["model"]
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    return {"config": config, "scaler": scaler, "model": model, "label_encoder": label_encoder}


def predict_array(y, sr, bundle):
    feats = extract_mfcc_vector(y, sr, n_mfcc=bundle["config"]["n_mfcc"]).reshape(1, -1)
    feats_scaled = bundle["scaler"].transform(feats)
    probs = bundle["model"].predict_proba(feats_scaled)[0]
    classes = list(bundle["label_encoder"].classes_)
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])


def get_test_files():
    doc_path = os.path.join(OLD_MODEL_DIR, "dataset_documentation.csv")
    df = pd.read_csv(doc_path)
    test_df = df[df["split_assigned"] == "test"]
    files = []
    for _, row in test_df.iterrows():
        path = row["matched_path"].replace(r"C:\Users\hp\Downloads\car_test", r"C:\Users\hp\Desktop\car_test")
        if os.path.exists(path):
            files.append((row["class"], path))
    return files


def main():
    print(f"OLD model: {OLD_MODEL_DIR}")
    print(f"NEW model: {NEW_MODEL_DIR}")
    old_bundle = load_model(OLD_MODEL_DIR)
    new_bundle = load_model(NEW_MODEL_DIR)
    sr = old_bundle["config"]["target_sr"]
    duration = old_bundle["config"]["target_duration"]
    assert sr == new_bundle["config"]["target_sr"] and duration == new_bundle["config"]["target_duration"]

    files = get_test_files()
    print(f"Evaluating {len(files)} test files (full test split) at {WEBM_BITRATES_KBPS} kbps webm + clean.\n")

    rows = []
    for i, (true_label, path) in enumerate(files, 1):
        y, _ = load_clean_audio(path, target_sr=sr, target_duration=duration, fallback_to_untrimmed=True)

        row = {"file": os.path.basename(path), "true_label": true_label}
        for tag, y_variant in [("clean", y)] + [(f"webm{k}", compress_roundtrip(y, sr, k)) for k in WEBM_BITRATES_KBPS]:
            old_pred, old_conf = predict_array(y_variant, sr, old_bundle)
            new_pred, new_conf = predict_array(y_variant, sr, new_bundle)
            row[f"{tag}_old_pred"] = old_pred
            row[f"{tag}_old_correct"] = old_pred == true_label
            row[f"{tag}_new_pred"] = new_pred
            row[f"{tag}_new_correct"] = new_pred == true_label
        rows.append(row)
        if i % 20 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] processed")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_old_vs_fix_results.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY -- accuracy on the FULL test set, OLD (deployed) vs NEW (compression-augmented)")
    print("=" * 70)
    print(f"{'condition':12s} {'OLD acc':>10s} {'NEW acc':>10s} {'delta':>10s}")
    tags = ["clean"] + [f"webm{k}" for k in WEBM_BITRATES_KBPS]
    for tag in tags:
        old_acc = df[f"{tag}_old_correct"].mean()
        new_acc = df[f"{tag}_new_correct"].mean()
        print(f"{tag:12s} {old_acc:10.3f} {new_acc:10.3f} {new_acc - old_acc:+10.3f}")

    print("\n=== per-class, webm32 (the worst condition for the OLD model) ===")
    print(df.groupby("true_label")[["webm32_old_correct", "webm32_new_correct"]].mean())

    print(f"\nFull per-file results saved to: {out_csv}")


if __name__ == "__main__":
    main()
