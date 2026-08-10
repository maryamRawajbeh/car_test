# -*- coding: utf-8 -*-
r"""
===================================================================
The actual "is this Freesound data good enough to merge" test
===================================================================
Empirically answers the question the review app's model-agreement badges
can only hint at: does adding the curated Freesound candidates (belt/sway
from sounds\, brake from sounds\break_* AND brake_dataset_freesound\accepted)
to TRAIN actually improve real held-out TEST accuracy, or hurt it?

Methodology (matches SESSION_HANDOFF.md section 4 item 4's plan exactly):
  1. Load the ORIGINAL raw (pre-scaling) train/val/test features from
     ablation_results\no_augmentation\features.csv -- this is the exact
     91.2%-accuracy XGBoost run's data, before StandardScaler was applied.
  2. Reproduce the baseline from scratch (fresh StandardScaler + fresh
     XGBoost with the exact winning hyperparameters) as a sanity check --
     if this doesn't land close to the real 93.2%/0.932 test result
     already on file, something in this script's reimplementation is
     wrong and the A/B deltas below can't be trusted.
  3. For each candidate arm: take the listed extra clips (rank-0 window
     only, one per source file -- never rank 1/2 alongside rank 0 from the
     SAME source file, since that would just be near-duplicate training
     rows), extract RAW features with the identical audio_common pipeline,
     add them to TRAIN ONLY, refit a NEW scaler+model on train+extra,
     evaluate on the SAME untouched original test split.
  4. TEST is never modified in any arm -- that's the whole point.

Feature extraction is cached to new_data_features_cache.pkl (655 clips,
each an audio decode + MFCC/HPSS extraction -- not free) so re-running
this script to try different arm combinations doesn't redo it.

Run:
  cd C:\Users\hp\Desktop\car_test\sounds_curation
  python run_ab_test.py [--workers N] [--refresh-cache]
"""

import os
import sys
import json
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAR_TEST_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CAR_TEST_DIR)
from audio_common import load_clean_audio, extract_mfcc_vector  # noqa: E402

MODEL_DIR = os.path.join(CAR_TEST_DIR, "ablation_results", "no_augmentation")
CANDIDATES_DIR = os.path.join(BASE_DIR, "candidates")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")
CACHE_PATH = os.path.join(BASE_DIR, "new_data_features_cache.pkl")

RANDOM_STATE = 42
XGB_KWARGS = dict(n_estimators=300, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE,
                   eval_metric="mlogloss", n_jobs=-1, tree_method="hist")

N_WORKERS = int(os.environ.get("SOUNDS_CURATION_N_WORKERS", max(1, (os.cpu_count() or 2) - 1)))

_config = None


def _init_worker(config):
    global _config
    _config = config


def extract_one(args):
    item_id, path, label = args
    try:
        y, sr = load_clean_audio(path, target_sr=_config["target_sr"],
                                  target_duration=_config["target_duration"],
                                  fallback_to_untrimmed=True)
        feats = extract_mfcc_vector(y, sr, n_mfcc=_config["n_mfcc"])
        return item_id, label, feats, None
    except Exception as e:
        return item_id, label, None, f"{item_id}: {e}"


def build_or_load_feature_cache(records, config, workers, refresh):
    """records: list of (item_id, path, label). Returns dict item_id -> (feats, label)."""
    cache = {}
    if os.path.exists(CACHE_PATH) and not refresh:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)

    todo = [r for r in records if r[0] not in cache]
    if todo:
        print(f"Extracting raw features for {len(todo)} new clips ({len(records) - len(todo)} already cached)...")
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(config,)) as ex:
            futures = [ex.submit(extract_one, r) for r in todo]
            done = 0
            for fut in as_completed(futures):
                item_id, label, feats, err = fut.result()
                done += 1
                if err:
                    print(f"  [{done}/{len(todo)}] ERROR: {err}")
                else:
                    cache[item_id] = (feats, label)
                if done % 100 == 0 or done == len(todo):
                    print(f"  [{done}/{len(todo)}] extracted")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    else:
        print(f"Using cached features for all {len(records)} clips ({CACHE_PATH}).")

    return cache


def load_original_splits():
    df = pd.read_csv(os.path.join(MODEL_DIR, "features.csv"))
    feature_cols = [c for c in df.columns if c not in ("label", "file_name", "augmented", "split")]
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    return (
        train[feature_cols].to_numpy(), train["label"].to_numpy(),
        val[feature_cols].to_numpy(), val["label"].to_numpy(),
        test[feature_cols].to_numpy(), test["label"].to_numpy(),
        feature_cols,
    )


def fit_and_eval(X_train_raw, y_train, X_test_raw, y_test, class_names):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y_train_enc = np.array([label_to_idx[y] for y in y_train])
    y_test_enc = np.array([label_to_idx[y] for y in y_test])

    model = XGBClassifier(**XGB_KWARGS)
    model.fit(X_train, y_train_enc)
    y_pred = model.predict(X_test)

    return {
        "n_train": len(y_train),
        "accuracy": accuracy_score(y_test_enc, y_pred),
        "f1_macro": f1_score(y_test_enc, y_pred, average="macro"),
        "precision_macro": precision_score(y_test_enc, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test_enc, y_pred, average="macro", zero_division=0),
        "report": classification_report(y_test_enc, y_pred, target_names=class_names, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    class_names = list(config["classes"])

    print("Loading original train/val/test raw features from features.csv...")
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, feature_cols = load_original_splits()
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}  ({len(feature_cols)} features)")

    print("\n=== [0] Baseline sanity check: fresh scaler+model on ORIGINAL train, eval on ORIGINAL test ===")
    baseline = fit_and_eval(X_train_raw, y_train, X_test_raw, y_test, class_names)
    print(f"  accuracy={baseline['accuracy']:.4f}  f1_macro={baseline['f1_macro']:.4f}  "
          f"(compare to the model on file: 0.9323 / 0.9323 -- should be close)")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    records = []
    meta = {}  # item_id -> (pool, model_agree)
    for it in manifest["items"]:
        cand = it["candidates"][0]  # rank-0 only -- see module docstring
        path = os.path.join(CANDIDATES_DIR, cand["filename"])
        records.append((it["id"], path, it["label"]))
        model_agree = (cand.get("model_predicted_label") == it["label"]) if cand.get("model_predicted_label") else None
        meta[it["id"]] = (it.get("pool", "sounds"), model_agree)

    cache = build_or_load_feature_cache(records, config, args.workers, args.refresh_cache)

    def arm_rows(pools=None, require_model_agree=False):
        X, y = [], []
        for item_id, (feats, label) in cache.items():
            pool, model_agree = meta[item_id]
            if pools is not None and pool not in pools:
                continue
            if require_model_agree and not model_agree:
                continue
            X.append(feats)
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

    print("\n=== A/B arms: baseline train + extra Freesound data, eval on the SAME original test ===")
    print(f"{'arm':42s} {'+rows':>7s} {'accuracy':>10s} {'f1_macro':>10s} {'delta_acc':>10s}")
    results = {}
    for name, filt in arms:
        X_extra, y_extra = arm_rows(**filt)
        if len(X_extra) == 0:
            print(f"{name:42s} {'(no rows, skipped)':>7s}")
            continue
        X_train_combined = np.vstack([X_train_raw, X_extra])
        y_train_combined = np.concatenate([y_train, y_extra])
        res = fit_and_eval(X_train_combined, y_train_combined, X_test_raw, y_test, class_names)
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
