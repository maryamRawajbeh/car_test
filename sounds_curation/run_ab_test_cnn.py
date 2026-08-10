# -*- coding: utf-8 -*-
r"""
===================================================================
Same "does the Freesound data actually help" question, for the CNN
===================================================================
run_ab_test.py answered this for the hand-crafted-feature (traditional
ML / XGBoost) pipeline: NO across every arm tried (SESSION_HANDOFF.md
section 4). But the user's actual goal is improving CNN/PANNs/YAMNet --
per section 3 finding #1, the CNN was "roughly unaffected" by synthetic
waveform augmentation (unlike traditional ML, which augmentation hurt),
which is some prior evidence a spectrogram-based model might tolerate
(or even benefit from) more diverse REAL audio better than the
statistical-feature model did. This script tests that empirically
instead of assuming it either way.

Methodology mirrors run_ab_test.py exactly:
  1. Un-normalize the ORIGINAL X_train_mel.npy/X_test_mel.npy back to raw
     log-mel using the saved mel_stats.pkl (mean/std) -- they were saved
     POST-normalization, and normalization must be refit on train+extra
     for a fair comparison, so this recovers the pre-normalization values.
  2. Baseline sanity check: refit normalization + a fresh CNN on the
     original train, evaluate on the original test -- should land close
     to the 80.5%/0.800 on file (NN training has some inherent run-to-run
     variance even with seeds set, unlike the deterministic XGBoost test).
  3. Per arm: add extra clips' log-mel (rank-0 window only) to train,
     refit normalization on train+extra, retrain the SAME CNN architecture
     (cnn_model.build_cnn, same temporal head + focal loss config as
     train_cnn.py) from scratch, evaluate on the SAME original test.

Log-mel extraction is cached (new_data_mel_cache.pkl) since it's the
expensive part; re-running to see different arms doesn't redo it.

Run:
  cd C:\Users\hp\Desktop\car_test\sounds_curation
  python run_ab_test_cnn.py [--refresh-cache]
"""

import os
import sys
import json
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAR_TEST_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CAR_TEST_DIR)
from audio_common import load_clean_audio, extract_log_mel  # noqa: E402
from cnn_model import build_cnn  # noqa: E402

MODEL_DIR = os.path.join(CAR_TEST_DIR, "ablation_results", "no_augmentation")
CANDIDATES_DIR = os.path.join(BASE_DIR, "candidates")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")
CACHE_PATH = os.path.join(BASE_DIR, "new_data_mel_cache.pkl")

RANDOM_STATE = 42
EPOCHS = 40
BATCH_SIZE = 16
USE_TEMPORAL_HEAD = True
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0


def extract_one(args):
    item_id, path, label, target_sr, target_duration = args
    try:
        y, sr = load_clean_audio(path, target_sr=target_sr, target_duration=target_duration,
                                  fallback_to_untrimmed=True)
        mel = extract_log_mel(y, sr)
        return item_id, label, mel, None
    except Exception as e:
        return item_id, label, None, f"{item_id}: {e}"


def build_or_load_mel_cache(records, target_sr, target_duration, refresh):
    cache = {}
    if os.path.exists(CACHE_PATH) and not refresh:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)

    todo = [r for r in records if r[0] not in cache]
    if todo:
        print(f"Extracting log-mel for {len(todo)} new clips ({len(records) - len(todo)} already cached)...")
        # sequential, not ProcessPoolExecutor: TF/torch import in this process makes forking
        # workers on Windows (spawn) re-trigger their own TF import per worker, wasteful for
        # ~655 clips that already took the parallel hit once in run_ab_test.py's feature cache
        for i, (item_id, path, label) in enumerate(todo, 1):
            _, _, mel, err = extract_one((item_id, path, label, target_sr, target_duration))
            if err:
                print(f"  [{i}/{len(todo)}] ERROR: {err}")
            else:
                cache[item_id] = (mel, label)
            if i % 100 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] extracted")
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    else:
        print(f"Using cached log-mel for all {len(records)} clips ({CACHE_PATH}).")
    return cache


def train_and_eval(X_train_raw, y_train_labels, X_test_raw, y_test_labels, class_names, n_classes):
    import tensorflow as tf
    from tensorflow.keras import callbacks
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    mean = X_train_raw.mean()
    std = X_train_raw.std()
    X_train = ((X_train_raw - mean) / (std + 1e-8))[..., np.newaxis]
    X_test = ((X_test_raw - mean) / (std + 1e-8))[..., np.newaxis]

    label_to_idx = {c: i for i, c in enumerate(class_names)}
    y_train = np.array([label_to_idx[y] for y in y_train_labels])
    y_test = np.array([label_to_idx[y] for y in y_test_labels])

    model = build_cnn(input_shape=X_train.shape[1:], n_classes=n_classes,
                       use_temporal_head=USE_TEMPORAL_HEAD, use_focal_loss=USE_FOCAL_LOSS,
                       focal_gamma=FOCAL_GAMMA)

    class_weight_values = compute_class_weight(class_weight="balanced", classes=np.arange(n_classes), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weight_values)}

    early_stop = callbacks.EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=4, min_lr=1e-6)

    # NOTE: monitors "loss" (train), not "val_loss" -- the new-data arms don't get a matching
    # extra validation split (val stays the original, untouched X_val_mel.npy would need the
    # same un-normalize/renormalize treatment for a fair comparison across arms, which adds
    # complexity for no real benefit here since we only report the TEST metric below).
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
              callbacks=[early_stop, reduce_lr], class_weight=class_weight_dict, verbose=0)

    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    return {
        "n_train": len(y_train),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "report": classification_report(y_test, y_pred, target_names=class_names, zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh-cache", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    class_names = list(config["classes"])
    n_classes = len(class_names)

    print("Loading original train/test log-mel spectrograms + un-normalizing via mel_stats.pkl...")
    X_train_norm = np.load(os.path.join(MODEL_DIR, "X_train_mel.npy"))[..., 0]
    X_test_norm = np.load(os.path.join(MODEL_DIR, "X_test_mel.npy"))[..., 0]
    y_train_idx = np.load(os.path.join(MODEL_DIR, "y_train.npy"))
    y_test_idx = np.load(os.path.join(MODEL_DIR, "y_test.npy"))
    with open(os.path.join(MODEL_DIR, "mel_stats.pkl"), "rb") as f:
        mel_stats = pickle.load(f)

    X_train_raw = X_train_norm * (mel_stats["std"] + 1e-8) + mel_stats["mean"]
    X_test_raw = X_test_norm * (mel_stats["std"] + 1e-8) + mel_stats["mean"]
    y_train = np.array([class_names[i] for i in y_train_idx])
    y_test = np.array([class_names[i] for i in y_test_idx])
    print(f"  train={len(y_train)}  test={len(y_test)}  mel shape={X_train_raw.shape[1:]}")

    print("\n=== [0] Baseline sanity check: fresh CNN on ORIGINAL train, eval on ORIGINAL test ===")
    baseline = train_and_eval(X_train_raw, y_train, X_test_raw, y_test, class_names, n_classes)
    print(f"  accuracy={baseline['accuracy']:.4f}  f1_macro={baseline['f1_macro']:.4f}  "
          f"(compare to the model on file: ~0.805 / ~0.800 -- NN training has run-to-run "
          f"variance even with seeds, so 'close' not exact)")

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

    cache = build_or_load_mel_cache(records, config["target_sr"], config["target_duration"], args.refresh_cache)

    def arm_rows(pools=None, require_model_agree=False):
        X, y = [], []
        for item_id, (mel, label) in cache.items():
            pool, model_agree = meta[item_id]
            if pools is not None and pool not in pools:
                continue
            if require_model_agree and not model_agree:
                continue
            X.append(mel)
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
        X_train_combined = np.concatenate([X_train_raw, X_extra], axis=0)
        y_train_combined = np.concatenate([y_train, y_extra])
        res = train_and_eval(X_train_combined, y_train_combined, X_test_raw, y_test, class_names, n_classes)
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
