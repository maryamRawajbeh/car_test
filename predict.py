# -*- coding: utf-8 -*-
r"""
===================================================================
Predict script v2 - classify a NEW audio file as belt / brake / sway
Works with either the Traditional ML model or the CNN.
===================================================================
HOW TO RUN:
    python predict.py "C:\path\to\sound.wav"                     (traditional ML, default)
    python predict.py "C:\path\to\sound.wav" --model cnn         (CNN only)
    python predict.py "C:\path\to\sound.wav" --model ensemble    (fused traditional ML + CNN,
                                                                    usually the most accurate --
                                                                    needs evaluate_ensemble.py to
                                                                    have run first)
"""

import os
import sys
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa

from audio_common import (
    load_clean_audio as _shared_load_clean_audio,
    extract_mfcc_vector as _extract_mfcc_vector,
    extract_log_mel as _extract_log_mel,
)

BASE_DIR = r"C:\Users\hp\Desktop\car_test"
DATA_DIR = os.path.join(BASE_DIR, "processed_data")


def load_config():
    with open(os.path.join(DATA_DIR, "config.pkl"), "rb") as f:
        return pickle.load(f)


def _check_feature_dim(feats, config):
    """Fail loudly instead of silently mis-scaling if feature_extraction.py's output
    dimension ever drifts from what scaler.pkl/best_traditional_model.pkl were fit on."""
    expected = config.get("mfcc_feature_dim")
    actual = feats.shape[1]
    if expected is not None and actual != expected:
        raise RuntimeError(
            f"Traditional-ML feature dim mismatch: extractor produced {actual}-dim "
            f"features but scaler.pkl/best_traditional_model.pkl were fit on "
            f"{expected}-dim features. The feature extraction code and the deployed "
            f"model are out of sync -- retrain before serving predictions."
        )


def load_clean_audio(path, target_sr, target_duration):
    """Same cleaning as preprocessing.py (shared via audio_common), with
    fallback_to_untrimmed=True so a near-silent input file returns silence
    instead of crashing (preprocessing.py instead just excludes such files
    from training, which isn't an option here for a single live prediction)."""
    y, sr = _shared_load_clean_audio(
        path, target_sr=target_sr, target_duration=target_duration,
        fallback_to_untrimmed=True,
    )
    return y, sr


def extract_mfcc_vector(y, sr, n_mfcc):
    """Thin wrapper around audio_common's shared implementation -- guaranteed
    IDENTICAL to preprocessing.py's, since both import the same function."""
    return _extract_mfcc_vector(y, sr, n_mfcc=n_mfcc)


def extract_log_mel(y, sr, n_mels, hop_length):
    return _extract_log_mel(y, sr, n_mels=n_mels, hop_length=hop_length)


def predict_traditional(file_path, config):
    with open(os.path.join(DATA_DIR, "best_traditional_model.pkl"), "rb") as f:
        saved = pickle.load(f)
    model, model_name = saved["model"], saved["name"]
    with open(os.path.join(DATA_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    y, sr = load_clean_audio(file_path, config["target_sr"], config["target_duration"])
    feats = extract_mfcc_vector(y, sr, config["n_mfcc"]).reshape(1, -1)
    _check_feature_dim(feats, config)
    feats_scaled = scaler.transform(feats)

    pred = model.predict(feats_scaled)[0]
    predicted_label = label_encoder.inverse_transform([pred])[0]

    print(f"\nModel used: {model_name} (Traditional ML)")
    print("=" * 50)
    print(f"PREDICTION: {predicted_label.upper()}")
    print("=" * 50)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feats_scaled)[0]
        print("\nConfidence per class:")
        for cls, p in zip(label_encoder.classes_, probs):
            print(f"   {cls:10s}: {p*100:.2f}%")


def get_traditional_probs(file_path, config):
    with open(os.path.join(DATA_DIR, "best_traditional_model.pkl"), "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    with open(os.path.join(DATA_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    y, sr = load_clean_audio(file_path, config["target_sr"], config["target_duration"])
    feats = extract_mfcc_vector(y, sr, config["n_mfcc"]).reshape(1, -1)
    _check_feature_dim(feats, config)
    feats_scaled = scaler.transform(feats)
    return model.predict_proba(feats_scaled)[0]


def get_cnn_probs(file_path, config):
    import tensorflow as tf
    import cnn_model  # noqa: F401 -- registers the custom layer/loss cnn_model.keras needs to deserialize

    model = tf.keras.models.load_model(os.path.join(DATA_DIR, "cnn_model.keras"))
    with open(os.path.join(DATA_DIR, "mel_stats.pkl"), "rb") as f:
        mel_stats = pickle.load(f)

    y, sr = load_clean_audio(file_path, config["target_sr"], config["target_duration"])
    log_mel = extract_log_mel(y, sr, config["n_mels"], config["hop_length"])
    log_mel = (log_mel - mel_stats["mean"]) / (mel_stats["std"] + 1e-8)
    log_mel = log_mel[np.newaxis, ..., np.newaxis]
    return model.predict(log_mel, verbose=0)[0]


def get_transfer_probs(file_path):
    """Uses YAMNet + the saved classifier head from train_transfer_learning.py.
    Note this loads audio at 16kHz (YAMNet's required rate), independent of `config`."""
    try:
        import tensorflow_hub as hub
        yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    except Exception:
        import tensorflow as tf
        import kagglehub
        model_path = kagglehub.model_download("google/yamnet/tensorFlow2/yamnet/1")
        yamnet = tf.saved_model.load(model_path)

    with open(os.path.join(DATA_DIR, "best_transfer_model.pkl"), "rb") as f:
        saved = pickle.load(f)
    model, scaler = saved["model"], saved["scaler"]

    y, _ = _shared_load_clean_audio(file_path, target_sr=16000, target_duration=5.0, fallback_to_untrimmed=True)

    _, frame_embeddings, _ = yamnet(y.astype(np.float32))
    embedding = np.mean(frame_embeddings.numpy(), axis=0).reshape(1, -1)
    embedding_scaled = scaler.transform(embedding)
    return model.predict_proba(embedding_scaled)[0]


def get_panns_probs(file_path):
    """Uses PANNs (CNN14) + the saved classifier head from train_panns.py.
    Note this loads audio at 32kHz (PANNs' required rate), independent of `config`."""
    from panns_inference import AudioTagging

    with open(os.path.join(DATA_DIR, "best_panns_model.pkl"), "rb") as f:
        saved = pickle.load(f)
    model, scaler = saved["model"], saved["scaler"]

    y, _ = _shared_load_clean_audio(file_path, target_sr=32000, target_duration=5.0, fallback_to_untrimmed=True)

    panns_model = AudioTagging(checkpoint_path=None, device="cpu")
    _, embedding = panns_model.inference(y.astype(np.float32)[np.newaxis, :])
    embedding_scaled = scaler.transform(embedding)
    return model.predict_proba(embedding_scaled)[0]


PROB_SOURCES = {
    "Traditional ML": get_traditional_probs,          # needs (file_path, config)
    "CNN": get_cnn_probs,                              # needs (file_path, config)
    "Transfer Learning (YAMNet)": get_transfer_probs,  # needs (file_path,) only
    "PANNs (CNN14)": get_panns_probs,                  # needs (file_path,) only
}

# names that take only (file_path,) instead of (file_path, config)
_SINGLE_ARG_SOURCES = {"Transfer Learning (YAMNet)", "PANNs (CNN14)"}


def predict_ensemble(file_path, config):
    ensemble_config_path = os.path.join(DATA_DIR, "ensemble_config.pkl")
    if not os.path.exists(ensemble_config_path):
        print("\n!! ensemble_config.pkl not found. Run evaluate_ensemble.py first "
              "(after training at least two of: traditional ML, CNN, transfer learning, PANNs).")
        return
    with open(ensemble_config_path, "rb") as f:
        ensemble_cfg = pickle.load(f)
    mode = ensemble_cfg["mode"]
    names = ensemble_cfg["names"]

    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    # each model's probability vector for this ONE file, in the same order evaluate_ensemble.py used
    probs_list = []
    for name in names:
        if name in _SINGLE_ARG_SOURCES:
            probs_list.append(PROB_SOURCES[name](file_path))
        else:
            probs_list.append(PROB_SOURCES[name](file_path, config))

    if mode == "stacking":
        meta_model = ensemble_cfg["meta_model"]
        stacked_features = np.concatenate(probs_list).reshape(1, -1)
        fused = meta_model.predict_proba(stacked_features)[0]
        mode_desc = "Stacking (Logistic Regression meta-classifier)"
    else:
        weights = ensemble_cfg["weights"]
        fused = sum(w * p for w, p in zip(weights, probs_list))
        mode_desc = "Weighted average (" + ", ".join(f"{n}={w:.2f}" for n, w in zip(names, weights)) + ")"

    pred = np.argmax(fused)
    predicted_label = label_encoder.inverse_transform([pred])[0]

    print(f"\nModel used: Ensemble [{mode_desc}]")
    print("=" * 50)
    print(f"PREDICTION: {predicted_label.upper()}")
    print("=" * 50)
    print("\nConfidence per class:")
    for cls, p in zip(label_encoder.classes_, fused):
        print(f"   {cls:10s}: {p*100:.2f}%")


def predict_cnn(file_path, config):
    import tensorflow as tf  # imported here so traditional-only users don't need tensorflow installed
    import cnn_model  # noqa: F401 -- registers the custom layer/loss cnn_model.keras needs to deserialize

    model = tf.keras.models.load_model(os.path.join(DATA_DIR, "cnn_model.keras"))
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    with open(os.path.join(DATA_DIR, "mel_stats.pkl"), "rb") as f:
        mel_stats = pickle.load(f)

    y, sr = load_clean_audio(file_path, config["target_sr"], config["target_duration"])
    log_mel = extract_log_mel(y, sr, config["n_mels"], config["hop_length"])
    log_mel = (log_mel - mel_stats["mean"]) / (mel_stats["std"] + 1e-8)
    log_mel = log_mel[np.newaxis, ..., np.newaxis]  # shape (1, n_mels, time, 1)

    probs = model.predict(log_mel, verbose=0)[0]
    pred = np.argmax(probs)
    predicted_label = label_encoder.inverse_transform([pred])[0]

    print("\nModel used: CNN (Deep Learning)")
    print("=" * 50)
    print(f"PREDICTION: {predicted_label.upper()}")
    print("=" * 50)
    print("\nConfidence per class:")
    for cls, p in zip(label_encoder.classes_, probs):
        print(f"   {cls:10s}: {p*100:.2f}%")


def predict_transfer(file_path, config):
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    probs = get_transfer_probs(file_path)
    pred = np.argmax(probs)
    predicted_label = label_encoder.inverse_transform([pred])[0]

    print("\nModel used: Transfer Learning (YAMNet)")
    print("=" * 50)
    print(f"PREDICTION: {predicted_label.upper()}")
    print("=" * 50)
    print("\nConfidence per class:")
    for cls, p in zip(label_encoder.classes_, probs):
        print(f"   {cls:10s}: {p*100:.2f}%")


def predict_panns(file_path, config):
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    probs = get_panns_probs(file_path)
    pred = np.argmax(probs)
    predicted_label = label_encoder.inverse_transform([pred])[0]

    print("\nModel used: PANNs (CNN14)")
    print("=" * 50)
    print(f"PREDICTION: {predicted_label.upper()}")
    print("=" * 50)
    print("\nConfidence per class:")
    for cls, p in zip(label_encoder.classes_, probs):
        print(f"   {cls:10s}: {p*100:.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", nargs="?", help="Path to the .wav file to classify")
    parser.add_argument("--model", choices=["traditional", "cnn", "transfer", "panns", "ensemble"],
                         default="traditional",
                         help="Which trained model to use (default: traditional). 'transfer' uses "
                              "YAMNet, 'panns' uses PANNs CNN14. 'ensemble' fuses whichever models "
                              "were trained (needs evaluate_ensemble.py to have run).")
    args = parser.parse_args()

    file_path = args.file_path or input("Enter the full path of the .wav file: ").strip().strip('"')

    if not os.path.exists(file_path):
        print(f"\nERROR: file not found: {file_path}")
        return

    config = load_config()
    print(f"Processing: {file_path}")

    if args.model == "cnn":
        predict_cnn(file_path, config)
    elif args.model == "transfer":
        predict_transfer(file_path, config)
    elif args.model == "panns":
        predict_panns(file_path, config)
    elif args.model == "ensemble":
        predict_ensemble(file_path, config)
    else:
        predict_traditional(file_path, config)


if __name__ == "__main__":
    main()