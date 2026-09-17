# -*- coding: utf-8 -*-
r"""
===================================================================
Full held-out-test-set compression-robustness check for the deployed
XGBoost model (Section 9.4's "single most important open item").
===================================================================
The paper's own Section 9.4 states that the strongest configuration
(XGBoost, 93.23% clean-audio accuracy) has never been evaluated through the
real WebM/Opus compression pipeline the deployed system's browser
microphone input actually passes through -- unlike the earlier production
model, which WAS checked and collapsed on Brake specifically. This script
closes that gap on the full 133-recording held-out test partition (not a
5-per-class subsample).

Mirrors the CURRENT (2026-09-08-fixed) production decode path exactly:
garageai-audio-analysis/app/services/audio_converter.py's convert_to_wav()
now does NOT force "-ac"/"-ar" in ffmpeg -- forcing those was a prior bug
that put a ~0.2% distortion into every clip and flipped this same brittle
(max_depth=3) XGBoost model's predictions. Only container/codec decoding is
done in ffmpeg here; the single mono-downmix + resample step is left to
librosa via load_clean_audio, exactly like the real deployed inference path.
This is deliberately NOT the same decode step as compression_augment.py's
compress_roundtrip() (which still forces "-ac 1 -ar {sr}" on decode --
correct for that module's own offline-training-augmentation purpose, but
not a faithful mirror of the current deployed inference path for this check).

Tests: clean (direct load) vs a lossless ffmpeg-decode control (isolates
"does the decode step alone change anything" from "does lossy compression
hurt") vs WebM/Opus at 16/32/64 kbps.

Run:
    cd C:\Users\hp\Desktop\car_test
    python mic_pipeline_test\test_full_133_compression_robustness.py
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
from audio_common import load_clean_audio, extract_mfcc_vector  # noqa: E402
from imageio_ffmpeg import get_ffmpeg_exe  # noqa: E402

MODEL_DIR = os.environ.get("COMPRESSION_TEST_MODEL_DIR", os.path.join(CAR_TEST_DIR, "processed_data"))
FFMPEG_EXE = get_ffmpeg_exe()
BITRATES_KBPS = [16, 32, 64]
TMP_DIR = tempfile.mkdtemp(prefix="full133_compression_")


def encode_to_webm(input_wav, output_webm, bitrate_kbps):
    subprocess.run([FFMPEG_EXE, "-y", "-loglevel", "error", "-i", input_wav,
                    "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", output_webm],
                   capture_output=True, check=True, timeout=30)


def convert_to_wav_current_prod(input_path, output_path):
    """Exact mirror of the CURRENT (post-2026-09-08-fix) audio_converter.py:
    container/codec decode only -- no -ac/-ar. Resample+mono is left to
    load_clean_audio (librosa), same as the real deployed inference path."""
    subprocess.run([FFMPEG_EXE, "-y", "-loglevel", "error", "-i", input_path, output_path],
                   capture_output=True, check=True, timeout=30)


def load_artifacts():
    with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "best_traditional_model.pkl"), "rb") as f:
        saved = pickle.load(f)
        model, model_name = saved["model"], saved["name"]
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    return config, scaler, model, model_name, label_encoder


def predict_from_wav(path, config, scaler, model, label_encoder):
    y, sr = load_clean_audio(path, target_sr=config["target_sr"],
                              target_duration=config["target_duration"],
                              fallback_to_untrimmed=True)
    feats = extract_mfcc_vector(y, sr, n_mfcc=config["n_mfcc"]).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    probs = model.predict_proba(feats_scaled)[0]
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
    print(f"Loading the deployed model from: {MODEL_DIR}")
    config, scaler, model, model_name, label_encoder = load_artifacts()
    print(f"Model: {model_name}")

    files = get_full_test_set()
    print(f"Testing all {len(files)} held-out TEST-split recordings "
          f"(current production decode path, no forced ffmpeg resample) "
          f"at {BITRATES_KBPS} kbps WebM/Opus + a lossless-decode control.\n")

    rows = []
    for i, (true_label, path) in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(path))[0]
        row = {"file": base, "true_label": true_label}

        clean_pred, clean_conf = predict_from_wav(path, config, scaler, model, label_encoder)
        row["clean_pred"] = clean_pred
        row["clean_conf"] = clean_conf
        row["clean_correct"] = clean_pred == true_label

        control_wav = os.path.join(TMP_DIR, f"{base}_control.wav")
        convert_to_wav_current_prod(path, control_wav)
        c_pred, c_conf = predict_from_wav(control_wav, config, scaler, model, label_encoder)
        row["control_pred"] = c_pred
        row["control_conf"] = c_conf
        row["control_correct"] = c_pred == true_label
        os.remove(control_wav)

        for kbps in BITRATES_KBPS:
            webm_path = os.path.join(TMP_DIR, f"{base}_{kbps}k.webm")
            rt_wav = os.path.join(TMP_DIR, f"{base}_{kbps}k_rt.wav")
            encode_to_webm(path, webm_path, kbps)
            convert_to_wav_current_prod(webm_path, rt_wav)
            pred, conf = predict_from_wav(rt_wav, config, scaler, model, label_encoder)
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
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"full133_compression_results_{tag}.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print("OVERALL SUMMARY  (n=%d held-out test recordings)" % len(df))
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

    print("\nPER-CLASS report by condition:")
    for cond in conditions:
        pred_col = f"{cond}_pred"
        print(f"\n-- {cond} --")
        print(classification_report(y_true, df[pred_col], digits=4, zero_division=0))

    print("\nConfusion matrices (rows=true, cols=pred, order=belt,brake,sway):")
    labels_order = ["belt", "brake", "sway"]
    for cond in conditions:
        cm = confusion_matrix(y_true, df[f"{cond}_pred"], labels=labels_order)
        print(f"\n-- {cond} --")
        print(pd.DataFrame(cm, index=labels_order, columns=labels_order))

    print(f"\nFull per-file results saved to: {out_csv}")


if __name__ == "__main__":
    main()
