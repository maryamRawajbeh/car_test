# -*- coding: utf-8 -*-
r"""
CNN-model variant of test_full_133_compression_robustness.py -- same 133-file held-out
test set, same current-production decode path (no forced ffmpeg resample), same
16/32/64 kbps WebM/Opus bitrates + lossless-decode control. Uses predict.py's
get_cnn_probs() pattern (log-mel features + cnn_model.keras + mel_stats.pkl) instead of
the traditional-ML MFCC path.

Run:
    cd C:\Users\hp\Desktop\car_test
    COMPRESSION_TEST_MODEL_DIR=<dir with cnn_model.keras/mel_stats.pkl/config.pkl> \
        python mic_pipeline_test/test_cnn_compression_robustness.py
"""
import os
import sys
import pickle
import tempfile
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

CAR_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAR_TEST_DIR)
from audio_common import load_clean_audio, extract_log_mel  # noqa: E402
from imageio_ffmpeg import get_ffmpeg_exe  # noqa: E402

MODEL_DIR = os.environ.get("COMPRESSION_TEST_MODEL_DIR",
                            os.path.join(CAR_TEST_DIR, "processed_data"))
FFMPEG_EXE = get_ffmpeg_exe()
BITRATES_KBPS = [16, 32, 64]
TMP_DIR = tempfile.mkdtemp(prefix="cnn_compression_")


def encode_to_webm(input_wav, output_webm, bitrate_kbps):
    subprocess.run([FFMPEG_EXE, "-y", "-loglevel", "error", "-i", input_wav,
                    "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", output_webm],
                   capture_output=True, check=True, timeout=30)


def convert_to_wav_current_prod(input_path, output_path):
    subprocess.run([FFMPEG_EXE, "-y", "-loglevel", "error", "-i", input_path, output_path],
                   capture_output=True, check=True, timeout=30)


def load_artifacts():
    import tensorflow as tf
    import cnn_model  # noqa: F401 -- registers custom layer/loss for deserialization
    with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "mel_stats.pkl"), "rb") as f:
        mel_stats = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "cnn_model.keras"))
    return config, mel_stats, model, label_encoder


def predict_from_wav(path, config, mel_stats, model, label_encoder):
    y, sr = load_clean_audio(path, target_sr=config["target_sr"],
                              target_duration=config["target_duration"],
                              fallback_to_untrimmed=True)
    log_mel = extract_log_mel(y, sr, n_mels=config["n_mels"], hop_length=config["hop_length"])
    log_mel = (log_mel - mel_stats["mean"]) / (mel_stats["std"] + 1e-8)
    log_mel = log_mel[np.newaxis, ..., np.newaxis]
    probs = model.predict(log_mel, verbose=0)[0]
    classes = list(label_encoder.classes_)
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])


def get_full_test_set():
    doc_path = os.path.join(MODEL_DIR, "dataset_documentation.csv")
    df = pd.read_csv(doc_path)
    test_df = df[df["split_assigned"] == "test"].copy()
    files = []
    for _, row in test_df.iterrows():
        path = row["matched_path"]
        if os.path.exists(path):
            files.append((row["class"], path))
        else:
            print(f"  [skip] not found: {path}")
    return files


def main():
    print(f"Loading the CNN model from: {MODEL_DIR}")
    config, mel_stats, model, label_encoder = load_artifacts()

    files = get_full_test_set()
    print(f"Testing all {len(files)} held-out TEST-split recordings (CNN, current "
          f"production decode path) at {BITRATES_KBPS} kbps WebM/Opus + a lossless-decode control.\n")

    rows = []
    for i, (true_label, path) in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(path))[0]
        row = {"file": base, "true_label": true_label}

        clean_pred, clean_conf = predict_from_wav(path, config, mel_stats, model, label_encoder)
        row["clean_pred"] = clean_pred
        row["clean_conf"] = clean_conf
        row["clean_correct"] = clean_pred == true_label

        control_wav = os.path.join(TMP_DIR, f"{base}_control.wav")
        convert_to_wav_current_prod(path, control_wav)
        c_pred, c_conf = predict_from_wav(control_wav, config, mel_stats, model, label_encoder)
        row["control_pred"] = c_pred
        row["control_conf"] = c_conf
        row["control_correct"] = c_pred == true_label
        os.remove(control_wav)

        for kbps in BITRATES_KBPS:
            webm_path = os.path.join(TMP_DIR, f"{base}_{kbps}k.webm")
            rt_wav = os.path.join(TMP_DIR, f"{base}_{kbps}k_rt.wav")
            encode_to_webm(path, webm_path, kbps)
            convert_to_wav_current_prod(webm_path, rt_wav)
            pred, conf = predict_from_wav(rt_wav, config, mel_stats, model, label_encoder)
            row[f"webm{kbps}_pred"] = pred
            row[f"webm{kbps}_conf"] = conf
            row[f"webm{kbps}_correct"] = pred == true_label
            os.remove(webm_path)
            os.remove(rt_wav)

        if i % 20 == 0 or i == len(files):
            print(f"  ...{i}/{len(files)} done")
        rows.append(row)

    df = pd.DataFrame(rows)
    tag = os.path.basename(os.path.normpath(MODEL_DIR))
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"cnn_compression_results_{tag}.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print("OVERALL SUMMARY  (n=%d held-out test recordings, CNN)" % len(df))
    print("=" * 78)
    conditions = ["clean", "control"] + [f"webm{k}" for k in BITRATES_KBPS]
    print(f"{'condition':16s} {'accuracy':>10s} {'macro-F1':>10s} {'mean_conf':>10s} {'n_flipped':>10s}")
    y_true = df["true_label"].tolist()
    for cond in conditions:
        pred_col = f"{cond}_pred"
        conf_col = f"{cond}_conf"
        y_pred = df[pred_col].tolist()
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro")
        conf = df[conf_col].mean()
        flipped = (df[pred_col] != df["clean_pred"]).sum() if cond != "clean" else 0
        print(f"{cond:16s} {acc:10.4f} {f1m:10.4f} {conf:10.4f} {flipped:10d}")

    print(f"\nFull per-file results saved to: {out_csv}")


if __name__ == "__main__":
    main()
