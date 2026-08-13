# -*- coding: utf-8 -*-
r"""
===================================================================
Does the REAL fused ensemble (Traditional ML + CNN + YAMNet, weighted
0.30/0.60/0.10) hold up under the real mic -> webm -> ffmpeg pipeline?
===================================================================
Every prior compression-robustness test in this folder (test_mic_pipeline_
domain_shift.py, compare_old_vs_full_fix.py, compare_old_vs_compression_
fix.py) only exercised the standalone traditional-ML model. The CNN was
separately shown to be compression-fragile pre-augmentation, and now
holds the LARGEST fusion weight (0.60) in the new robustified ensemble
config -- so this is the first test of whether the ACTUAL final fused
decision (what a user sees in the app) survives compression, not just
one component of it.

Run:
  cd C:\Users\hp\Desktop\car_test\mic_pipeline_test
  python test_full_ensemble_compression.py
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
import librosa

CAR_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAR_TEST_DIR)
from audio_common import load_clean_audio, extract_mfcc_vector, extract_log_mel  # noqa: E402

PROC_DIR = r"C:\Users\hp\Desktop\car_test\processed_data"
COMP_DIR = r"C:\Users\hp\Desktop\car_test\ablation_results\compression_augmented"
FFMPEG_EXE = r"C:\Users\hp\Desktop\garageai-audio-analysis\venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
if not os.path.exists(FFMPEG_EXE):
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

WEIGHTS = {"traditional": 0.30, "cnn": 0.60, "yamnet": 0.10}
N_FILES_PER_CLASS = 5
WEBM_BITRATES_KBPS = [16, 32, 64]
TMP_DIR = tempfile.mkdtemp(prefix="full_ensemble_test_")


def encode_to_webm(input_wav, output_webm, bitrate_kbps):
    cmd = [FFMPEG_EXE, "-y", "-loglevel", "error", "-i", input_wav,
           "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", output_webm]
    subprocess.run(cmd, capture_output=True, check=True)


def convert_to_wav(input_path, output_path, sample_rate=22050):
    cmd = [FFMPEG_EXE, "-y", "-loglevel", "error", "-i", input_path,
           "-ac", "1", "-ar", str(sample_rate), output_path]
    subprocess.run(cmd, capture_output=True, check=True)


def load_everything():
    config = pickle.load(open(os.path.join(COMP_DIR, "config.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(COMP_DIR, "scaler.pkl"), "rb"))
    trad_model = pickle.load(open(os.path.join(COMP_DIR, "best_traditional_model.pkl"), "rb"))["model"]

    import tensorflow as tf
    import cnn_model as cnn_model_module  # noqa: F401 -- registers FrequencyAveragePooling/SparseCategoricalFocalLoss
    cnn_model = tf.keras.models.load_model(os.path.join(COMP_DIR, "cnn_model.keras"), compile=False)
    mel_stats = pickle.load(open(os.path.join(COMP_DIR, "mel_stats.pkl"), "rb"))

    transfer_saved = pickle.load(open(os.path.join(PROC_DIR, "best_transfer_model.pkl"), "rb"))
    transfer_model = transfer_saved["model"]
    transfer_scaler = transfer_saved["scaler"]
    import tensorflow_hub as hub
    print("Loading YAMNet (may take a moment)...")
    yamnet = hub.load("https://tfhub.dev/google/yamnet/1")

    label_encoder = pickle.load(open(os.path.join(PROC_DIR, "label_encoder.pkl"), "rb"))
    return {
        "config": config, "scaler": scaler, "trad_model": trad_model,
        "cnn_model": cnn_model, "mel_stats": mel_stats,
        "transfer_model": transfer_model, "transfer_scaler": transfer_scaler, "yamnet": yamnet,
        "label_encoder": label_encoder,
    }


def predict_fused(path, m):
    config = m["config"]
    classes = list(m["label_encoder"].classes_)

    y, sr = load_clean_audio(path, target_sr=config["target_sr"], target_duration=config["target_duration"],
                              fallback_to_untrimmed=True)
    feats = extract_mfcc_vector(y, sr, n_mfcc=config["n_mfcc"]).reshape(1, -1)
    feats_scaled = m["scaler"].transform(feats)
    trad_probs = m["trad_model"].predict_proba(feats_scaled)[0]

    log_mel = extract_log_mel(y, sr, n_mels=config["n_mels"], hop_length=config["hop_length"])
    log_mel = (log_mel - m["mel_stats"]["mean"]) / (m["mel_stats"]["std"] + 1e-8)
    log_mel = log_mel[np.newaxis, ..., np.newaxis]
    cnn_probs = m["cnn_model"].predict(log_mel, verbose=0)[0]

    y16, sr16 = librosa.load(path, sr=16000, mono=True)
    y16, _ = librosa.effects.trim(y16, top_db=25)
    if np.max(np.abs(y16)) > 0:
        y16 = y16 / np.max(np.abs(y16))
    target_len = int(16000 * 5.0)
    y16 = y16[:target_len] if len(y16) > target_len else np.pad(y16, (0, target_len - len(y16)))
    _, frame_embeddings, _ = m["yamnet"](y16.astype(np.float32))
    embedding = np.mean(frame_embeddings.numpy(), axis=0).reshape(1, -1)
    embedding_scaled = m["transfer_scaler"].transform(embedding)
    yamnet_probs = m["transfer_model"].predict_proba(embedding_scaled)[0]

    fused = WEIGHTS["traditional"] * trad_probs + WEIGHTS["cnn"] * cnn_probs + WEIGHTS["yamnet"] * yamnet_probs
    idx = int(np.argmax(fused))
    return classes[idx], float(fused[idx]), {
        "trad": (classes[int(np.argmax(trad_probs))], float(np.max(trad_probs))),
        "cnn": (classes[int(np.argmax(cnn_probs))], float(np.max(cnn_probs))),
        "yamnet": (classes[int(np.argmax(yamnet_probs))], float(np.max(yamnet_probs))),
    }


def get_test_files():
    doc_path = os.path.join(PROC_DIR, "dataset_documentation.csv")
    df = pd.read_csv(doc_path)
    test_df = df[df["split_assigned"] == "test"]
    files = []
    for cls in ["belt", "brake", "sway"]:
        rows = test_df[test_df["class"] == cls].head(N_FILES_PER_CLASS)
        for _, row in rows.iterrows():
            path = row["matched_path"].replace(r"C:\Users\hp\Downloads\car_test", r"C:\Users\hp\Desktop\car_test")
            if os.path.exists(path):
                files.append((cls, path))
    return files


def main():
    m = load_everything()
    files = get_test_files()
    print(f"Testing {len(files)} test-split files across clean + control(wav rt) + {WEBM_BITRATES_KBPS} kbps webm\n")

    rows = []
    for true_label, path in files:
        base = os.path.splitext(os.path.basename(path))[0]
        row = {"file": base, "true_label": true_label}

        pred, conf, detail = predict_fused(path, m)
        row["clean_pred"], row["clean_conf"], row["clean_correct"] = pred, conf, pred == true_label

        control_wav = os.path.join(TMP_DIR, f"{base}_control.wav")
        convert_to_wav(path, control_wav, m["config"]["target_sr"])
        pred, conf, _ = predict_fused(control_wav, m)
        row["control_pred"], row["control_conf"], row["control_correct"] = pred, conf, pred == true_label

        for kbps in WEBM_BITRATES_KBPS:
            webm_path = os.path.join(TMP_DIR, f"{base}_{kbps}k.webm")
            rt_wav = os.path.join(TMP_DIR, f"{base}_{kbps}k_rt.wav")
            encode_to_webm(path, webm_path, kbps)
            convert_to_wav(webm_path, rt_wav, m["config"]["target_sr"])
            pred, conf, detail = predict_fused(rt_wav, m)
            row[f"webm{kbps}_pred"] = pred
            row[f"webm{kbps}_conf"] = conf
            row[f"webm{kbps}_correct"] = pred == true_label
            row[f"webm{kbps}_trad"] = f"{detail['trad'][0]}({detail['trad'][1]:.2f})"
            row[f"webm{kbps}_cnn"] = f"{detail['cnn'][0]}({detail['cnn'][1]:.2f})"
            row[f"webm{kbps}_yamnet"] = f"{detail['yamnet'][0]}({detail['yamnet'][1]:.2f})"

        rows.append(row)
        flips = [f"{kbps}k->{row[f'webm{kbps}_pred']}" for kbps in WEBM_BITRATES_KBPS
                 if row[f"webm{kbps}_pred"] != row["clean_pred"]]
        flip_note = f"  FLIPPED: {', '.join(flips)}" if flips else ""
        print(f"  {base:25s} true={true_label:6s} clean={row['clean_pred']}({row['clean_conf']:.2f}) "
              f"control={row['control_pred']}({row['control_conf']:.2f}){flip_note}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_ensemble_compression_results.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY -- FULL FUSED ENSEMBLE (Traditional=0.30, CNN=0.60, YAMNet=0.10)")
    print("=" * 70)
    print(f"{'condition':15s} {'accuracy':>10s} {'mean_conf':>10s} {'n_flipped_from_clean':>22s}")
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
