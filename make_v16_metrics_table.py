# -*- coding: utf-8 -*-
"""Build the full held-out-test metrics table for the v16 presentation:
Accuracy, Macro-Precision, Macro-Recall, Macro-F1, Macro-AUC (OVR macro), and
per-class F1 (Belt / Brake / Sway) for the 10 configurations + the superseded
3-model ensemble.

Single held-out run (133 recordings) -- NOT the 5-seed means shown in the
bar chart.

  * Deployed XGBoost row  -> features extracted FRESH from the test audio
    (audio_common.extract_mfcc_vector, exactly like compute_confusion_matrix.py)
    because processed_data/X_test_mfcc.npy is a stale cache that no longer
    matches the current 269-feature scaler/model (it yields a broken 44% --
    the classic staleness trap). Fresh extraction reproduces the deck's 93.23%.
  * Every backbone / CNN row -> the probability vectors that model's own
    train_*.py wrote to processed_data/<key>_test_probs.npy.
  * Ensemble row -> deployed historical weighted average
    SVM 0.30 / CNN 0.60 / YAMNet 0.10 (reproduces ensemble_report.txt's 89.47%).
"""
import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)

from audio_common import load_clean_audio, extract_mfcc_vector

warnings.filterwarnings("ignore")
DATA = "processed_data"

y_test = np.load(os.path.join(DATA, "y_test.npy"))
le = pickle.load(open(os.path.join(DATA, "label_encoder.pkl"), "rb"))
classes = list(le.classes_)  # ['belt','brake','sway']
print("classes:", classes, "| y_test bincount:", np.bincount(y_test))


def metrics_row(name, probs):
    probs = np.asarray(probs, dtype=float)
    preds = probs.argmax(1)
    f1c = f1_score(y_test, preds, average=None, labels=[0, 1, 2])
    try:
        auc = roc_auc_score(y_test, probs, multi_class="ovr", average="macro")
    except Exception as e:
        auc = float("nan")
        print("  AUC failed for", name, e)
    return {
        "Configuration": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Macro-Precision": precision_score(y_test, preds, average="macro"),
        "Macro-Recall": recall_score(y_test, preds, average="macro"),
        "Macro-F1": f1_score(y_test, preds, average="macro"),
        "Macro-AUC": auc,
        "Belt-F1": f1c[0], "Brake-F1": f1c[1], "Sway-F1": f1c[2],
    }


rows = []

# ---------- 1. Deployed XGBoost : FRESH feature extraction from test audio ----------
config = pickle.load(open(os.path.join(DATA, "config.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(DATA, "scaler.pkl"), "rb"))
saved = pickle.load(open(os.path.join(DATA, "best_traditional_model.pkl"), "rb"))
xgb = saved["model"]
print("\nDeployed traditional model:", saved["name"])

doc_df = pd.read_csv(os.path.join(DATA, "dataset_documentation.csv"))
corr = os.path.join(DATA, "corrupted_files.csv")
if os.path.exists(corr):
    cdf = pd.read_csv(corr)
    if "matched_path" in cdf.columns:
        doc_df = doc_df[~doc_df["matched_path"].isin(cdf["matched_path"])].reset_index(drop=True)
test_df = doc_df[doc_df["split_assigned"] == "test"].reset_index(drop=True)
y_test_fresh = le.transform(test_df["class"])
assert np.array_equal(y_test_fresh, y_test), "fresh test order != y_test.npy order"

feats = []
for p in test_df["matched_path"]:
    y, sr = load_clean_audio(p, config["target_sr"], config["target_duration"], fallback_to_untrimmed=True)
    feats.append(extract_mfcc_vector(y, sr, config["n_mfcc"]))
X = scaler.transform(np.array(feats))
xgb_probs = xgb.predict_proba(X)
np.save(os.path.join(DATA, "xgboost_deployed_test_probs.npy"), xgb_probs)
rows.append(metrics_row("XGBoost  (Traditional ML, deployed)", xgb_probs))
preds = xgb_probs.argmax(1)
print("\n=== Deployed XGBoost held-out test (fresh extraction) ===")
print(classification_report(y_test, preds, target_names=classes, digits=4))
print("confusion (rows=actual belt/brake/sway):\n", confusion_matrix(y_test, preds))

# ---------- 2..9  backbones / CNN  from saved probability vectors ----------
prob_files = [
    ("CLAP", "clap_test_probs.npy"),
    ("PaSST", "passt_test_probs.npy"),
    ("EfficientAT  (fine-tuned)", "efficientat_ft_test_probs.npy"),
    ("AST", "ast_test_probs.npy"),
    ("BEATs", "beats_test_probs.npy"),
    ("YAMNet", "transfer_test_probs.npy"),
    ("PANNs / CNN14", "panns_test_probs.npy"),
    ("CNN  (from scratch)", "cnn_test_probs.npy"),
]
for name, fn in prob_files:
    rows.append(metrics_row(name, np.load(os.path.join(DATA, fn))))

# ---------- 10. Superseded 3-model weighted ensemble ----------
svm_probs = np.load(os.path.join(DATA, "traditional_test_probs.npy"))
cnn_probs = np.load(os.path.join(DATA, "cnn_test_probs.npy"))
yam_probs = np.load(os.path.join(DATA, "transfer_test_probs.npy"))
ens = 0.30 * svm_probs + 0.60 * cnn_probs + 0.10 * yam_probs
ens = ens / ens.sum(1, keepdims=True)
rows.append(metrics_row("Ensemble  (SVM+CNN+YAMNet, superseded)", ens))

df = pd.DataFrame(rows)
for c in df.columns:
    if c != "Configuration":
        df[c] = (df[c] * 100).round(2)
df.to_csv(os.path.join(DATA, "v16_full_metrics_table.csv"), index=False)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print("\n\n============ v16 FULL METRICS TABLE (held-out test, single run, %) ============")
print(df.to_string(index=False))
print("\nsaved -> processed_data/v16_full_metrics_table.csv")
