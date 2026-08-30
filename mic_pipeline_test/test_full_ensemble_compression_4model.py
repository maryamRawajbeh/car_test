# -*- coding: utf-8 -*-
r"""
===================================================================
Compression robustness of the NEW 4-model deployed ensemble
  {Traditional ML 0.14, CNN 0.18, PANNs/CNN14 0.04, EfficientAT 0.64}
vs the OLD 3-model one {Traditional 0.30, CNN 0.60, YAMNet 0.10}
===================================================================
Follow-up to test_full_ensemble_compression.py. That script tested the
old Traditional/CNN/YAMNet fusion end-to-end through the real
mic -> webm/opus -> ffmpeg pipeline and got 86.7% clean -> 93.3% under
WebM compression (15 files, 5 per class).

On 2026-08-30 the deployed ensemble_config.pkl was re-tuned via nested
5-fold CV to {Traditional 0.14, CNN 0.18, PANNs 0.04, EfficientAT 0.64}
-- dominated by the fine-tuned EfficientAT model, which (unlike
Traditional ML and CNN) was NOT retrained on compression-augmented data.
So the new fusion's real-pipeline robustness needs its own end-to-end
check before the docs can claim it.

This script:
  * loads all five models from the deployed processed_data/ (its
    Traditional ML + CNN are the compression-augmented ones -- verified
    byte-identical to ablation_results/compression_augmented/);
  * runs the FULL 133-file test split through clean / control(wav
    round-trip) / webm 16-32-64 kbps;
  * for every file+condition, computes BOTH the new 4-model fused
    decision and the old 3-model one from the same per-model
    probabilities, so the before/after comparison is on identical audio.

Run:
  cd C:\Users\hp\Desktop\car_test\mic_pipeline_test
  python test_full_ensemble_compression_4model.py
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
FFMPEG_EXE = r"C:\Users\hp\Desktop\garageai-audio-analysis\venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
if not os.path.exists(FFMPEG_EXE):
    import imageio_ffmpeg
    FFMPEG_EXE = imageio_ffmpeg.get_ffmpeg_exe()

NEW_WEIGHTS = {"traditional": 0.14, "cnn": 0.18, "panns": 0.04, "efficientat": 0.64}
OLD_WEIGHTS = {"traditional": 0.30, "cnn": 0.60, "yamnet": 0.10}
N_FILES_PER_CLASS = int(os.environ.get("N_FILES_PER_CLASS", "999"))  # default: all of them
WEBM_BITRATES_KBPS = [16, 32, 64]
TMP_DIR = tempfile.mkdtemp(prefix="ensemble4_comp_")


def encode_to_webm(input_wav, output_webm, bitrate_kbps):
    cmd = [FFMPEG_EXE, "-y", "-loglevel", "error", "-i", input_wav,
           "-c:a", "libopus", "-b:a", f"{bitrate_kbps}k", output_webm]
    subprocess.run(cmd, capture_output=True, check=True)


def convert_to_wav(input_path, output_path, sample_rate=22050):
    cmd = [FFMPEG_EXE, "-y", "-loglevel", "error", "-i", input_path,
           "-ac", "1", "-ar", str(sample_rate), output_path]
    subprocess.run(cmd, capture_output=True, check=True)


def load_everything():
    config = pickle.load(open(os.path.join(PROC_DIR, "config.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(PROC_DIR, "scaler.pkl"), "rb"))
    trad_model = pickle.load(open(os.path.join(PROC_DIR, "best_traditional_model.pkl"), "rb"))["model"]

    import tensorflow as tf
    import cnn_model as _cnn  # noqa: F401 -- registers custom layers/loss for load_model
    cnn_model = tf.keras.models.load_model(os.path.join(PROC_DIR, "cnn_model.keras"), compile=False)
    mel_stats = pickle.load(open(os.path.join(PROC_DIR, "mel_stats.pkl"), "rb"))

    transfer_saved = pickle.load(open(os.path.join(PROC_DIR, "best_transfer_model.pkl"), "rb"))
    transfer_model = transfer_saved["model"]
    transfer_scaler = transfer_saved["scaler"]
    print("Loading YAMNet ...")
    try:
        import tensorflow_hub as hub
        yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
    except Exception:  # tfhub.dev cache incomplete / deprecated -- same fallback the live service uses
        import kagglehub
        yamnet = tf.saved_model.load(kagglehub.model_download("google/yamnet/tensorFlow2/yamnet/1"))

    panns_saved = pickle.load(open(os.path.join(PROC_DIR, "best_panns_model.pkl"), "rb"))
    panns_head, panns_scaler = panns_saved["model"], panns_saved["scaler"]
    from panns_inference import AudioTagging
    print("Loading PANNs (CNN14) ...")
    panns_tagger = AudioTagging(checkpoint_path=None, device="cpu")

    import torch
    sys.path.insert(0, os.path.join(CAR_TEST_DIR, "efficientat_vendor"))
    from models.mn.model import get_model as get_mn
    from models.preprocess import AugmentMelSTFT
    from helpers.utils import NAME_TO_WIDTH
    print("Loading EfficientAT (fine-tuned) ...")
    eat_mel = AugmentMelSTFT(n_mels=128, sr=32000)
    eat_mel.eval()
    eat_model = get_mn(num_classes=3, pretrained_name=None, width_mult=NAME_TO_WIDTH("mn10_as"), head_type="mlp")
    eat_model.load_state_dict(torch.load(os.path.join(PROC_DIR, "best_efficientat_ft_model.pt"), map_location="cpu"))
    eat_model.eval()

    label_encoder = pickle.load(open(os.path.join(PROC_DIR, "label_encoder.pkl"), "rb"))
    return {
        "config": config, "scaler": scaler, "trad_model": trad_model,
        "cnn_model": cnn_model, "mel_stats": mel_stats,
        "transfer_model": transfer_model, "transfer_scaler": transfer_scaler, "yamnet": yamnet,
        "panns_head": panns_head, "panns_scaler": panns_scaler, "panns_tagger": panns_tagger,
        "eat_mel": eat_mel, "eat_model": eat_model,
        "label_encoder": label_encoder,
    }


def _resample_5s(path, sr):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    n = int(sr * 5.0)
    return (y[:n] if len(y) > n else np.pad(y, (0, n - len(y)))).astype(np.float32)


def per_model_probs(path, m):
    config = m["config"]

    y, sr = load_clean_audio(path, target_sr=config["target_sr"], target_duration=config["target_duration"],
                             fallback_to_untrimmed=True)
    feats = extract_mfcc_vector(y, sr, n_mfcc=config["n_mfcc"]).reshape(1, -1)
    trad = m["trad_model"].predict_proba(m["scaler"].transform(feats))[0]

    log_mel = extract_log_mel(y, sr, n_mels=config["n_mels"], hop_length=config["hop_length"])
    log_mel = (log_mel - m["mel_stats"]["mean"]) / (m["mel_stats"]["std"] + 1e-8)
    cnn = m["cnn_model"].predict(log_mel[np.newaxis, ..., np.newaxis], verbose=0)[0]

    y16 = _resample_5s(path, 16000)
    _, frame_emb, _ = m["yamnet"](y16)
    emb = np.mean(frame_emb.numpy(), axis=0).reshape(1, -1)
    yamnet = m["transfer_model"].predict_proba(m["transfer_scaler"].transform(emb))[0]

    y32 = _resample_5s(path, 32000)
    _, p_emb = m["panns_tagger"].inference(y32[np.newaxis, :])
    panns = m["panns_head"].predict_proba(m["panns_scaler"].transform(p_emb))[0]

    import torch
    with torch.no_grad():
        spec = m["eat_mel"](torch.tensor(y32[np.newaxis, :])).unsqueeze(1)
        logits, _ = m["eat_model"](spec)
        eat = torch.softmax(logits, dim=1).numpy()[0]

    return {"traditional": trad, "cnn": cnn, "yamnet": yamnet, "panns": panns, "efficientat": eat}


def fuse(probs, weights):
    fused = sum(weights[k] * probs[k] for k in weights)
    return fused


def get_test_files():
    df = pd.read_csv(os.path.join(PROC_DIR, "dataset_documentation.csv"))
    test_df = df[df["split_assigned"] == "test"]
    files = []
    for cls in ["belt", "brake", "sway"]:
        rows = test_df[test_df["class"] == cls].head(N_FILES_PER_CLASS)
        for _, row in rows.iterrows():
            path = row["matched_path"].replace(r"C:\Users\hp\Downloads\car_test", r"C:\Users\hp\Desktop\car_test")
            if os.path.exists(path):
                files.append((cls, path))
    return files


CONDITIONS = ["clean", "control", "webm16", "webm32", "webm64"]


def main():
    m = load_everything()
    classes = list(m["label_encoder"].classes_)
    files = get_test_files()
    print(f"\nTesting {len(files)} test-split files x {CONDITIONS}\n")

    rows = []
    for i, (true_label, path) in enumerate(files, 1):
        base = os.path.splitext(os.path.basename(path))[0]
        row = {"file": base, "true_label": true_label}

        variants = {}
        variants["clean"] = path
        ctrl = os.path.join(TMP_DIR, f"{base}_ctrl.wav")
        convert_to_wav(path, ctrl, m["config"]["target_sr"])
        variants["control"] = ctrl
        for kbps in WEBM_BITRATES_KBPS:
            wp = os.path.join(TMP_DIR, f"{base}_{kbps}.webm")
            rt = os.path.join(TMP_DIR, f"{base}_{kbps}.wav")
            encode_to_webm(path, wp, kbps)
            convert_to_wav(wp, rt, m["config"]["target_sr"])
            variants[f"webm{kbps}"] = rt

        for cond in CONDITIONS:
            probs = per_model_probs(variants[cond], m)
            new_fused = fuse(probs, NEW_WEIGHTS)
            old_fused = fuse(probs, OLD_WEIGHTS)
            new_pred = classes[int(np.argmax(new_fused))]
            old_pred = classes[int(np.argmax(old_fused))]
            row[f"{cond}_new_pred"] = new_pred
            row[f"{cond}_new_conf"] = float(np.max(new_fused))
            row[f"{cond}_new_correct"] = new_pred == true_label
            row[f"{cond}_old_pred"] = old_pred
            row[f"{cond}_old_correct"] = old_pred == true_label
            for k, v in probs.items():
                row[f"{cond}_{k}"] = f"{classes[int(np.argmax(v))]}({np.max(v):.2f})"

        rows.append(row)
        print(f"  [{i}/{len(files)}] {base:24s} true={true_label:6s} "
              f"clean:new={row['clean_new_pred']} old={row['clean_old_pred']}  "
              f"webm32:new={row['webm32_new_pred']} old={row['webm32_old_pred']}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_ensemble_compression_4model_results.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 78)
    print("SUMMARY  (n=%d)   NEW = %s" % (len(df), NEW_WEIGHTS))
    print("                 OLD = %s" % (OLD_WEIGHTS,))
    print("=" * 78)
    print(f"{'condition':14s} {'NEW acc':>9s} {'NEW conf':>9s} {'OLD acc':>9s}   {'NEW flips':>9s} {'OLD flips':>9s}")
    for cond in CONDITIONS:
        nacc = df[f"{cond}_new_correct"].mean()
        nconf = df[f"{cond}_new_conf"].mean()
        oacc = df[f"{cond}_old_correct"].mean()
        nflip = (df[f"{cond}_new_pred"] != df["clean_new_pred"]).sum() if cond != "clean" else 0
        oflip = (df[f"{cond}_old_pred"] != df["clean_old_pred"]).sum() if cond != "clean" else 0
        print(f"{cond:14s} {nacc:9.3f} {nconf:9.3f} {oacc:9.3f}   {nflip:9d} {oflip:9d}")

    webm_cols_new = [f"webm{k}_new_correct" for k in WEBM_BITRATES_KBPS]
    webm_cols_old = [f"webm{k}_old_correct" for k in WEBM_BITRATES_KBPS]
    print("\navg over all 3 webm bitrates:  NEW=%.3f  OLD=%.3f"
          % (df[webm_cols_new].values.mean(), df[webm_cols_old].values.mean()))
    print(f"\nper-file results -> {out_csv}")

    # per-class clean vs webm for the NEW ensemble
    print("\nNEW ensemble, per-class accuracy:")
    print(f"{'class':8s} {'clean':>8s} {'webm16':>8s} {'webm32':>8s} {'webm64':>8s}")
    for cls in ["belt", "brake", "sway"]:
        sub = df[df["true_label"] == cls]
        vals = [sub["clean_new_correct"].mean()] + [sub[f"webm{k}_new_correct"].mean() for k in WEBM_BITRATES_KBPS]
        print(f"{cls:8s} " + " ".join(f"{v:8.3f}" for v in vals))


if __name__ == "__main__":
    main()
