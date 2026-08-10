# -*- coding: utf-8 -*-
r"""
===================================================================
Does the REAL app's mic -> webm -> ffmpeg -> wav pipeline break the
REAL deployed model?
===================================================================
SESSION_HANDOFF.md section 5 flagged this as unresolved: does real-world
domain shift (recording device/compression, not just "different label")
explain the "accuracy feels bad in the app" complaint. This session's
Freesound investigation found the deployed model can be extremely
sensitive to recording-condition artifacts (e.g. 100% brake recall on
the original dataset but ~0% agreement on Freesound-sourced brake audio)
-- raising the question of whether it's similarly fragile against the
ACTUAL production audio path, not just a different internet source.

Methodology:
  1. Take known TEST-split files (held-out, labels certain) from the
     REAL deployed model's own split (dataset_documentation.csv in
     C:\Users\hp\Desktop\car_test\processed_data -- confirmed via
     garageai-audio-analysis\.env's MODELS_DIR to be the ACTUAL path the
     live microservice loads from; SESSION_HANDOFF.md's claim that this
     is Downloads\car_test\processed_data is now stale --
     Downloads\car_test doesn't even exist anymore).
  2. For each file, encode it to WebM/Opus at a few bitrates
     representative of what a browser's MediaRecorder(stream) (no
     explicit options -- see garageai-frontend/src/app/pages/Home.tsx)
     would plausibly produce, using the EXACT SAME bundled ffmpeg binary
     the microservice uses (imageio_ffmpeg, from
     garageai-audio-analysis\venv).
  3. Decode back to mono 22050Hz wav using the EXACT SAME ffmpeg command
     as app/services/audio_converter.py's convert_to_wav().
  4. Run the REAL deployed traditional-ML model (config.pkl/scaler.pkl/
     best_traditional_model.pkl/label_encoder.pkl from
     car_test\processed_data) on both the original clean file and every
     compressed-roundtrip version, and compare predictions/confidence.

A lossless WAV roundtrip (no lossy codec, just the same ffmpeg resample
step) is included as a control -- isolates "does resampling alone hurt"
from "does the lossy webm/opus compression hurt".

Run:
  cd C:\Users\hp\Desktop\car_test\mic_pipeline_test
  python test_mic_pipeline_domain_shift.py
"""

import os
import sys
import glob
import pickle
import tempfile
import subprocess
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CAR_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAR_TEST_DIR)
from audio_common import load_clean_audio, extract_mfcc_vector  # noqa: E402

MODEL_DIR = r"C:\Users\hp\Desktop\car_test\processed_data"  # the ACTUAL deployed model (see module docstring)
FFMPEG_EXE = r"C:\Users\hp\Desktop\garageai-audio-analysis\venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"

N_FILES_PER_CLASS = 5
WEBM_BITRATES_KBPS = [16, 32, 64]  # brackets typical browser MediaRecorder audio bitrates
TMP_DIR = tempfile.mkdtemp(prefix="mic_pipeline_test_")


def encode_to_webm(input_wav, output_webm, bitrate_kbps):
    cmd = [FFMPEG_EXE, "-y", "-loglevel", "error", "-i", input_wav,
           "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", output_webm]
    subprocess.run(cmd, capture_output=True, check=True)


def convert_to_wav(input_path, output_path, sample_rate=22050):
    """Exact mirror of garageai-audio-analysis/app/services/audio_converter.py's
    convert_to_wav() -- same ffmpeg binary, same flags."""
    cmd = [FFMPEG_EXE, "-y", "-loglevel", "error", "-i", input_path,
           "-ac", "1", "-ar", str(sample_rate), output_path]
    subprocess.run(cmd, capture_output=True, check=True)


def load_real_model():
    with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "best_traditional_model.pkl"), "rb") as f:
        model = pickle.load(f)["model"]
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    return config, scaler, model, label_encoder


def predict_file(path, config, scaler, model, label_encoder):
    y, sr = load_clean_audio(path, target_sr=config["target_sr"], target_duration=config["target_duration"],
                              fallback_to_untrimmed=True)
    feats = extract_mfcc_vector(y, sr, n_mfcc=config["n_mfcc"]).reshape(1, -1)
    feats_scaled = scaler.transform(feats)
    probs = model.predict_proba(feats_scaled)[0]
    classes = list(label_encoder.classes_)
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx]), dict(zip(classes, probs.round(4).tolist()))


def get_test_files():
    doc_path = os.path.join(MODEL_DIR, "dataset_documentation.csv")
    df = pd.read_csv(doc_path)
    test_df = df[df["split_assigned"] == "test"]
    files = []
    for cls in ["belt", "brake", "sway"]:
        rows = test_df[test_df["class"] == cls].head(N_FILES_PER_CLASS)
        for _, row in rows.iterrows():
            # SESSION_HANDOFF.md's paths point at Downloads\car_test, which no longer
            # exists -- the raw audio is actually still under Desktop\car_test (this
            # session confirmed it there)
            path = row["matched_path"].replace(r"C:\Users\hp\Downloads\car_test", r"C:\Users\hp\Desktop\car_test")
            if os.path.exists(path):
                files.append((cls, path))
            else:
                print(f"  [skip] not found: {path}")
    return files


def main():
    print(f"Loading the REAL deployed model from: {MODEL_DIR}")
    config, scaler, model, label_encoder = load_real_model()

    files = get_test_files()
    print(f"Testing {len(files)} known TEST-split files (never used in training) "
          f"across {WEBM_BITRATES_KBPS} kbps WebM/Opus + a lossless-wav control.\n")

    rows = []
    for true_label, path in files:
        base = os.path.splitext(os.path.basename(path))[0]

        orig_pred, orig_conf, orig_probs = predict_file(path, config, scaler, model, label_encoder)
        row = {"file": base, "true_label": true_label,
               "clean_pred": orig_pred, "clean_conf": orig_conf, "clean_correct": orig_pred == true_label}

        # control: lossless wav roundtrip through the exact same ffmpeg resample step
        control_wav = os.path.join(TMP_DIR, f"{base}_control.wav")
        convert_to_wav(path, control_wav, config["target_sr"])
        c_pred, c_conf, _ = predict_file(control_wav, config, scaler, model, label_encoder)
        row["control_pred"] = c_pred
        row["control_conf"] = c_conf
        row["control_correct"] = c_pred == true_label

        for kbps in WEBM_BITRATES_KBPS:
            webm_path = os.path.join(TMP_DIR, f"{base}_{kbps}k.webm")
            rt_wav = os.path.join(TMP_DIR, f"{base}_{kbps}k_rt.wav")
            encode_to_webm(path, webm_path, kbps)
            convert_to_wav(webm_path, rt_wav, config["target_sr"])
            pred, conf, _ = predict_file(rt_wav, config, scaler, model, label_encoder)
            row[f"webm{kbps}_pred"] = pred
            row[f"webm{kbps}_conf"] = conf
            row[f"webm{kbps}_correct"] = pred == true_label

        rows.append(row)
        flips = [f"{kbps}k->{row[f'webm{kbps}_pred']}" for kbps in WEBM_BITRATES_KBPS
                 if row[f"webm{kbps}_pred"] != orig_pred]
        flip_note = f"  FLIPPED: {', '.join(flips)}" if flips else ""
        print(f"  {base:30s} true={true_label:6s} clean={orig_pred}({orig_conf:.2f}) "
              f"control={c_pred}({c_conf:.2f}){flip_note}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mic_pipeline_results.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'condition':15s} {'accuracy':>10s} {'mean_conf':>10s} {'n_flipped_from_clean':>22s}")
    n = len(df)
    print(f"{'clean':15s} {df['clean_correct'].mean():10.3f} {df['clean_conf'].mean():10.3f} {'-':>22s}")
    print(f"{'control(wav rt)':15s} {df['control_correct'].mean():10.3f} {df['control_conf'].mean():10.3f} "
          f"{(df['control_pred'] != df['clean_pred']).sum():22d}")
    for kbps in WEBM_BITRATES_KBPS:
        acc = df[f"webm{kbps}_correct"].mean()
        conf = df[f"webm{kbps}_conf"].mean()
        flipped = (df[f"webm{kbps}_pred"] != df["clean_pred"]).sum()
        print(f"{'webm ' + str(kbps) + 'kbps':15s} {acc:10.3f} {conf:10.3f} {flipped:22d}")

    print(f"\nFull per-file results saved to: {out_csv}")


if __name__ == "__main__":
    main()
