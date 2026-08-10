# -*- coding: utf-8 -*-
r"""
===================================================================
Same "does the Freesound data actually help" question, for PANNs
===================================================================
See run_ab_test.py (traditional ML) and run_ab_test_cnn.py (CNN) for the
same methodology applied to the other two models. This one targets
PANNs (CNN14 embeddings + a small sklearn classifier head), per
train_panns.py.

Methodology:
  1. Read the SAME leakage-safe split train_panns.py itself uses
     (dataset_documentation.csv's split_assigned column, from
     ablation_results\no_augmentation\ -- the current best real-data run).
  2. Extract 2048-dim PANNs embeddings (32kHz) for every original
     train/val/test file AND every curated Freesound candidate's rank-0
     window. Cached (new_data_panns_embeddings_cache.pkl for the new
     clips; original_panns_embeddings_cache.pkl for original files) since
     a CNN14 forward pass per clip isn't free.
  3. Baseline: same small classifier grid as train_panns.py (Logistic
     Regression / SVM), pick the winner by VALIDATION macro-F1 once, on
     the original data only. That winning classifier TYPE is then locked
     in for every arm below -- so a delta in the A/B table reflects the
     data, not a different arm quietly picking a different classifier.
  4. Per arm: add extra clips' embeddings to train, refit scaler+classifier
     on train+extra (same locked classifier type), evaluate on the SAME
     original test split.

Run:
  cd C:\Users\hp\Desktop\car_test\sounds_curation
  python run_ab_test_panns.py [--refresh-cache]
"""

import os
import sys
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAR_TEST_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CAR_TEST_DIR)
from audio_common import load_clean_audio  # noqa: E402

MODEL_DIR = os.path.join(CAR_TEST_DIR, "ablation_results", "no_augmentation")
CANDIDATES_DIR = os.path.join(BASE_DIR, "candidates")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")
ORIG_CACHE_PATH = os.path.join(BASE_DIR, "original_panns_embeddings_cache.pkl")
NEW_CACHE_PATH = os.path.join(BASE_DIR, "new_data_panns_embeddings_cache.pkl")

PANNS_SR = 32000
TARGET_DURATION = 5.0
BATCH_SIZE = 16

CLASSIFIER_GRID = {
    "Logistic Regression (C=1)": lambda: LogisticRegression(C=1, max_iter=2000, class_weight="balanced"),
    "Logistic Regression (C=10)": lambda: LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
    "SVM (linear, C=1)": lambda: SVC(kernel="linear", C=1, probability=True, class_weight="balanced"),
    "SVM (rbf, C=10)": lambda: SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced"),
}


def load_clean_audio_32k(path):
    y, _ = load_clean_audio(path, target_sr=PANNS_SR, target_duration=TARGET_DURATION, fallback_to_untrimmed=True)
    return y.astype(np.float32)


def extract_embeddings(paths, panns_model, desc):
    embeddings = []
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i + BATCH_SIZE]
        batch_audio = np.stack([load_clean_audio_32k(p) for p in batch_paths])
        _, embedding = panns_model.inference(batch_audio)
        embeddings.append(embedding)
        done = min(i + BATCH_SIZE, len(paths))
        if done % (BATCH_SIZE * 5) == 0 or done == len(paths):
            print(f"  [{desc}] {done}/{len(paths)}")
    return np.concatenate(embeddings, axis=0)


def evaluate(model, X, y, class_names):
    y_pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y, y_pred, average="macro", zero_division=0),
        "report": classification_report(y, y_pred, target_names=class_names, zero_division=0),
    }


def fit_and_eval(X_train_raw, y_train, X_test_raw, y_test, class_names, classifier_builder):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    model = classifier_builder()
    model.fit(X_train, y_train)
    res = evaluate(model, X_test, y_test, class_names)
    res["n_train"] = len(y_train)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)

    doc_df = pd.read_csv(os.path.join(MODEL_DIR, "dataset_documentation.csv"))
    train_df = doc_df[doc_df["split_assigned"] == "train"]
    val_df = doc_df[doc_df["split_assigned"] == "val"]
    test_df = doc_df[doc_df["split_assigned"] == "test"]
    print(f"Original split: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    print("\n[1] Loading PANNs (CNN14) via panns_inference (cached checkpoint, no download needed)...")
    from panns_inference import AudioTagging
    panns_model = AudioTagging(checkpoint_path=None, device="cpu")

    print("\n[2] Extracting/loading cached embeddings for the ORIGINAL train/val/test files...")
    if os.path.exists(ORIG_CACHE_PATH) and not args.refresh_cache:
        with open(ORIG_CACHE_PATH, "rb") as f:
            orig = pickle.load(f)
        X_train_raw, y_train = orig["X_train"], orig["y_train"]
        X_val_raw, y_val = orig["X_val"], orig["y_val"]
        X_test_raw, y_test = orig["X_test"], orig["y_test"]
        print("  loaded from cache.")
    else:
        X_train_raw = extract_embeddings(list(train_df["matched_path"]), panns_model, "train")
        X_val_raw = extract_embeddings(list(val_df["matched_path"]), panns_model, "val")
        X_test_raw = extract_embeddings(list(test_df["matched_path"]), panns_model, "test")
        y_train = train_df["class"].to_numpy()
        y_val = val_df["class"].to_numpy()
        y_test = test_df["class"].to_numpy()
        with open(ORIG_CACHE_PATH, "wb") as f:
            pickle.dump({"X_train": X_train_raw, "y_train": y_train, "X_val": X_val_raw, "y_val": y_val,
                         "X_test": X_test_raw, "y_test": y_test}, f)

    print("\n[3] Baseline: trying classifier grid, picking winner by VALIDATION macro-F1...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_raw)
    X_val_s = scaler.transform(X_val_raw)
    val_results = {}
    for name, builder in CLASSIFIER_GRID.items():
        model = builder()
        model.fit(X_train_s, y_train)
        res = evaluate(model, X_val_s, y_val, class_names)
        val_results[name] = res
        print(f"   -> {name}: val accuracy={res['accuracy']:.4f}  val macro-F1={res['f1_macro']:.4f}")
    best_name = max(val_results, key=lambda n: val_results[n]["f1_macro"])
    best_builder = CLASSIFIER_GRID[best_name]
    print(f"   Locked in classifier for all arms below: {best_name}")

    baseline = fit_and_eval(X_train_raw, y_train, X_test_raw, y_test, class_names, best_builder)
    print(f"\n=== [0] Baseline sanity check ({best_name}) on ORIGINAL train, eval on ORIGINAL test ===")
    print(f"  accuracy={baseline['accuracy']:.4f}  f1_macro={baseline['f1_macro']:.4f}  "
          f"(compare to the model on file: ~0.83 / ~0.83 -- different split source so not identical)")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = []
    meta = {}
    for it in manifest["items"]:
        cand = it["candidates"][0]
        path = os.path.join(CANDIDATES_DIR, cand["filename"])
        records.append((it["id"], path, it["label"]))
        model_agree = (cand.get("model_predicted_label") == it["label"]) if cand.get("model_predicted_label") else None
        meta[it["id"]] = (it.get("pool", "sounds"), model_agree)

    print(f"\n[4] Extracting/loading cached PANNs embeddings for {len(records)} curated Freesound clips...")
    new_cache = {}
    if os.path.exists(NEW_CACHE_PATH) and not args.refresh_cache:
        with open(NEW_CACHE_PATH, "rb") as f:
            new_cache = pickle.load(f)
    todo = [r for r in records if r[0] not in new_cache]
    if todo:
        paths = [r[1] for r in todo]
        embs = extract_embeddings(paths, panns_model, "new clips")
        for (item_id, _, label), emb in zip(todo, embs):
            new_cache[item_id] = (emb, label)
        with open(NEW_CACHE_PATH, "wb") as f:
            pickle.dump(new_cache, f)
    else:
        print("  all cached already.")

    def arm_rows(pools=None, require_model_agree=False):
        X, y = [], []
        for item_id, (emb, label) in new_cache.items():
            pool, model_agree = meta[item_id]
            if pools is not None and pool not in pools:
                continue
            if require_model_agree and not model_agree:
                continue
            X.append(emb)
            y.append(label)
        return np.array(X), np.array(y)

    arms = [
        ("sounds_all", dict(pools=["sounds"], require_model_agree=False)),
        ("sounds_model_agree_only", dict(pools=["sounds"], require_model_agree=True)),
        ("brake_dataset_freesound_all", dict(pools=["brake_dataset_freesound"], require_model_agree=False)),
        ("brake_dataset_freesound_model_agree_only", dict(pools=["brake_dataset_freesound"], require_model_agree=True)),
        ("both_pools_all", dict(pools=["sounds", "brake_dataset_freesound"], require_model_agree=False)),
        ("both_pools_model_agree_only", dict(pools=["sounds", "brake_dataset_freesound"], require_model_agree=True)),
    ]

    print(f"\n=== A/B arms ({best_name}): baseline train + extra Freesound data, eval on the SAME original test ===")
    print(f"{'arm':42s} {'+rows':>7s} {'accuracy':>10s} {'f1_macro':>10s} {'delta_acc':>10s}")
    results = {}
    for name, filt in arms:
        X_extra, y_extra = arm_rows(**filt)
        if len(X_extra) == 0:
            print(f"{name:42s} {'(no rows, skipped)':>7s}")
            continue
        X_train_combined = np.vstack([X_train_raw, X_extra])
        y_train_combined = np.concatenate([y_train, y_extra])
        res = fit_and_eval(X_train_combined, y_train_combined, X_test_raw, y_test, class_names, best_builder)
        results[name] = res
        delta = res["accuracy"] - baseline["accuracy"]
        print(f"{name:42s} {len(X_extra):7d} {res['accuracy']:10.4f} {res['f1_macro']:10.4f} {delta:+10.4f}")

    print("\n=== Per-class detail for each arm ===")
    print(f"\n[baseline]\n{baseline['report']}")
    for name, _ in arms:
        if name in results:
            print(f"[{name}]\n{results[name]['report']}")


if __name__ == "__main__":
    main()
