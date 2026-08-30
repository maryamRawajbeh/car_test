# -*- coding: utf-8 -*-
"""
Direct train-vs-val-vs-test accuracy check for the actually-deployed models in
processed_data/ -- answers "is this model overfitting" with real numbers instead
of assumption. A model that scores much higher on train than on held-out val/test
has memorized training examples rather than learned generalizable patterns.

IMPORTANT: extracts features FRESH from the raw audio for every file (same
load_clean_audio/extract_mfcc_vector/extract_log_mel functions predict.py itself
uses for a real single-file prediction) instead of using the cached
X_train_mfcc.npy/X_train_mel.npy arrays in processed_data/. Those caches turned out
to be STALE relative to the currently-deployed cnn_model.keras/mel_stats.pkl (a
5-day-older preprocessing.py run -- the exact "processed_data staleness trap"
already documented elsewhere in this project's history), which silently produced
~33% "accuracy" (i.e. compared the deployed model against features/labels that
didn't correspond to it) on a first attempt at this check. Re-deriving everything
fresh from dataset_documentation.csv's file paths + the current audio_common.py
code sidesteps that risk entirely, at the cost of being slower (re-extracts
~893 files' worth of features instead of loading a cache).
"""
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from audio_common import load_clean_audio, extract_mfcc_vector, extract_log_mel

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")


def load_splits():
    doc_df = pd.read_csv(os.path.join(DATA_DIR, "dataset_documentation.csv"))
    corrupted_path = os.path.join(DATA_DIR, "corrupted_files.csv")
    if os.path.exists(corrupted_path):
        corrupted_df = pd.read_csv(corrupted_path)
        if "matched_path" in corrupted_df.columns:
            doc_df = doc_df[~doc_df["matched_path"].isin(corrupted_df["matched_path"])].reset_index(drop=True)
    splits = {}
    for name in ("train", "val", "test"):
        splits[name] = doc_df[doc_df["split_assigned"] == name].reset_index(drop=True)
    return splits


def extract_all(paths, config):
    """Returns (mfcc_matrix, mel_tensor) for every path, loading audio ONCE per
    file and deriving both feature types from it (same underlying clean audio
    predict.py would use for a real prediction on that file)."""
    mfcc_rows, mel_rows = [], []
    for p in tqdm(paths, desc="extracting features"):
        y, sr = load_clean_audio(p, config["target_sr"], config["target_duration"], fallback_to_untrimmed=True)
        mfcc_rows.append(extract_mfcc_vector(y, sr, config["n_mfcc"]))
        mel_rows.append(extract_log_mel(y, sr, config["n_mels"], config["hop_length"]))
    return np.array(mfcc_rows), np.array(mel_rows)[..., np.newaxis]


def report(name, y_train, p_train, y_val, p_val, y_test, p_test):
    acc_train = accuracy_score(y_train, p_train)
    acc_val = accuracy_score(y_val, p_val)
    acc_test = accuracy_score(y_test, p_test)
    print(f"\n{name}")
    print(f"  train accuracy: {acc_train:.4f}  (n={len(y_train)})")
    print(f"  val   accuracy: {acc_val:.4f}  (n={len(y_val)})   (gap train-val: {acc_train - acc_val:+.4f})")
    print(f"  test  accuracy: {acc_test:.4f}  (n={len(y_test)})   (gap train-test: {acc_train - acc_test:+.4f})")


def main():
    with open(os.path.join(DATA_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    splits = load_splits()
    print(f"train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")

    y_train = label_encoder.transform(splits["train"]["class"])
    y_val = label_encoder.transform(splits["val"]["class"])
    y_test = label_encoder.transform(splits["test"]["class"])

    print("\nExtracting TRAIN features fresh from audio (this is the slow part)...")
    X_train_mfcc, X_train_mel = extract_all(list(splits["train"]["matched_path"]), config)
    print("Extracting VAL features fresh from audio...")
    X_val_mfcc, X_val_mel = extract_all(list(splits["val"]["matched_path"]), config)
    print("Extracting TEST features fresh from audio...")
    X_test_mfcc, X_test_mel = extract_all(list(splits["test"]["matched_path"]), config)

    # --- Traditional ML ---
    with open(os.path.join(DATA_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(DATA_DIR, "best_traditional_model.pkl"), "rb") as f:
        saved = pickle.load(f)
    tml_model = saved["model"]
    report(
        f"Traditional ML -- deployed ({saved['name']})",
        y_train, tml_model.predict(scaler.transform(X_train_mfcc)),
        y_val, tml_model.predict(scaler.transform(X_val_mfcc)),
        y_test, tml_model.predict(scaler.transform(X_test_mfcc)),
    )

    # --- CNN ---
    import tensorflow as tf
    import cnn_model  # noqa: F401 -- registers FrequencyAveragePooling/SparseCategoricalFocalLoss
    with open(os.path.join(DATA_DIR, "mel_stats.pkl"), "rb") as f:
        mel_stats = pickle.load(f)
    cnn = tf.keras.models.load_model(os.path.join(DATA_DIR, "cnn_model.keras"))

    def norm(X):
        return (X - mel_stats["mean"]) / (mel_stats["std"] + 1e-8)

    p_train = np.argmax(cnn.predict(norm(X_train_mel), verbose=0), axis=1)
    p_val = np.argmax(cnn.predict(norm(X_val_mel), verbose=0), axis=1)
    p_test = np.argmax(cnn.predict(norm(X_test_mel), verbose=0), axis=1)
    report("CNN -- deployed (cnn_model.keras)", y_train, p_train, y_val, p_val, y_test, p_test)

    # --- YAMNet (third core ensemble member) ---
    from train_transfer_learning import load_yamnet, load_clean_audio_16k
    with open(os.path.join(DATA_DIR, "best_transfer_model.pkl"), "rb") as f:
        yamnet_saved = pickle.load(f)
    yamnet_model_clf = yamnet_saved["model"]
    yamnet_scaler = yamnet_saved["scaler"]
    yamnet = load_yamnet()

    def yamnet_embed(paths):
        out = []
        for p in tqdm(paths, desc="yamnet embeddings"):
            y = load_clean_audio_16k(p)
            _, frame_embeddings, _ = yamnet(y)
            out.append(np.mean(frame_embeddings.numpy(), axis=0))
        return np.array(out)

    X_train_yam = yamnet_embed(list(splits["train"]["matched_path"]))
    X_val_yam = yamnet_embed(list(splits["val"]["matched_path"]))
    X_test_yam = yamnet_embed(list(splits["test"]["matched_path"]))
    report(
        "YAMNet -- deployed (best_transfer_model.pkl)",
        y_train, yamnet_model_clf.predict(yamnet_scaler.transform(X_train_yam)),
        y_val, yamnet_model_clf.predict(yamnet_scaler.transform(X_val_yam)),
        y_test, yamnet_model_clf.predict(yamnet_scaler.transform(X_test_yam)),
    )


if __name__ == "__main__":
    main()
