# -*- coding: utf-8 -*-
"""Confusion matrix for the currently-deployed SVM traditional model on the real
test split, extracted fresh from audio (same reasoning as check_overfitting.py --
avoids any stale-cache risk). Needed for the paper's Table 5, marked "pending
re-verification" there."""
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from audio_common import load_clean_audio, extract_mfcc_vector

DATA_DIR = "processed_data"

with open(os.path.join(DATA_DIR, "config.pkl"), "rb") as f:
    config = pickle.load(f)
with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)
with open(os.path.join(DATA_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)
with open(os.path.join(DATA_DIR, "best_traditional_model.pkl"), "rb") as f:
    saved = pickle.load(f)
model = saved["model"]

doc_df = pd.read_csv(os.path.join(DATA_DIR, "dataset_documentation.csv"))
corrupted_path = os.path.join(DATA_DIR, "corrupted_files.csv")
if os.path.exists(corrupted_path):
    corrupted_df = pd.read_csv(corrupted_path)
    if "matched_path" in corrupted_df.columns:
        doc_df = doc_df[~doc_df["matched_path"].isin(corrupted_df["matched_path"])].reset_index(drop=True)
test_df = doc_df[doc_df["split_assigned"] == "test"].reset_index(drop=True)

y_test = label_encoder.transform(test_df["class"])
feats = []
for p in test_df["matched_path"]:
    y, sr = load_clean_audio(p, config["target_sr"], config["target_duration"], fallback_to_untrimmed=True)
    feats.append(extract_mfcc_vector(y, sr, config["n_mfcc"]))
X_test = np.array(feats)
X_test_scaled = scaler.transform(X_test)
y_pred = model.predict(X_test_scaled)

class_names = list(label_encoder.classes_)
cm = confusion_matrix(y_test, y_pred)
print(f"Model: {saved['name']}")
print(f"Classes (row/col order): {class_names}")
print("Confusion matrix (rows=actual, cols=predicted):")
print(cm)
print()
print(classification_report(y_test, y_pred, target_names=class_names, digits=4))

# per-100-samples normalized version, matching the paper's existing table style
cm_norm = (cm / cm.sum(axis=1, keepdims=True) * 100).round().astype(int)
print("Per-100-samples version:")
print(cm_norm)
