# -*- coding: utf-8 -*-
r"""
===================================================================
Did adding mic-response simulation on top of compression augmentation
help further -- against BOTH real-pipeline compression AND simulated
device/mic variation?
===================================================================
Extends compare_old_vs_compression_fix.py with a second stress dimension:
instead of just webm/opus compression, also test a few FIXED (not
randomized, for reproducibility) representative "device profiles" built
from mic_response_augment.py's highpass/lowpass ranges -- a stand-in for
"what if this came from a phone mic we've never seen".

Compares three models on EXACTLY the same file list and same audio
variants:
  OLD   = the currently deployed model (car_test\processed_data)
  COMP  = compression-only fix (ablation_results\compression_augmented)
  FULL  = compression + mic-response fix (ablation_results\compression_and_mic_augmented)

Run:
  cd C:\Users\hp\Desktop\car_test\mic_pipeline_test
  python compare_old_vs_full_fix.py
"""

import os
import sys
import pickle
import tempfile
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

CAR_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CAR_TEST_DIR)
from audio_common import load_clean_audio, extract_mfcc_vector  # noqa: E402
from compression_augment import compress_roundtrip  # noqa: E402

OLD_MODEL_DIR = r"C:\Users\hp\Desktop\car_test\processed_data"
COMP_MODEL_DIR = r"C:\Users\hp\Desktop\car_test\ablation_results\compression_augmented"
FULL_MODEL_DIR = r"C:\Users\hp\Desktop\car_test\ablation_results\compression_and_mic_augmented"
COMPOUND_MODEL_DIR = r"C:\Users\hp\Desktop\car_test\ablation_results\compound_augmented"

WEBM_BITRATES_KBPS = [16, 32, 64]

# fixed (not randomized) representative device profiles, spanning
# mic_response_augment.py's HIGHPASS_RANGE_HZ/LOWPASS_RANGE_HZ -- reproducible,
# unlike the randomized training-time augmentation
DEVICE_PROFILES = {
    "device_narrow": (200, 6500),   # aggressively band-limited mic
    "device_medium": (120, 7500),   # typical mid-range phone mic
    "device_wide": (70, 9000),      # a more full-range mic
}


def simulate_device(y, sr, hp_cutoff, lp_cutoff):
    sos_hp = butter(2, hp_cutoff, btype="highpass", fs=sr, output="sos")
    out = sosfilt(sos_hp, y)
    sos_lp = butter(2, min(lp_cutoff, sr / 2 - 100), btype="lowpass", fs=sr, output="sos")
    out = sosfilt(sos_lp, out)
    peak = np.max(np.abs(out))
    return (out / peak).astype(np.float32) if peak > 0 else out.astype(np.float32)


def load_model(model_dir):
    with open(os.path.join(model_dir, "config.pkl"), "rb") as f:
        config = pickle.load(f)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(model_dir, "best_traditional_model.pkl"), "rb") as f:
        model = pickle.load(f)["model"]
    with open(os.path.join(model_dir, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    return {"config": config, "scaler": scaler, "model": model, "label_encoder": label_encoder}


def predict_array(y, sr, bundle):
    feats = extract_mfcc_vector(y, sr, n_mfcc=bundle["config"]["n_mfcc"]).reshape(1, -1)
    feats_scaled = bundle["scaler"].transform(feats)
    probs = bundle["model"].predict_proba(feats_scaled)[0]
    classes = list(bundle["label_encoder"].classes_)
    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx])


def get_test_files():
    doc_path = os.path.join(OLD_MODEL_DIR, "dataset_documentation.csv")
    df = pd.read_csv(doc_path)
    test_df = df[df["split_assigned"] == "test"]
    files = []
    for _, row in test_df.iterrows():
        path = row["matched_path"].replace(r"C:\Users\hp\Downloads\car_test", r"C:\Users\hp\Desktop\car_test")
        if os.path.exists(path):
            files.append((row["class"], path))
    return files


def main():
    models = {
        "OLD": load_model(OLD_MODEL_DIR),
        "COMP": load_model(COMP_MODEL_DIR),
        "FULL": load_model(FULL_MODEL_DIR),
        "COMPOUND": load_model(COMPOUND_MODEL_DIR),
    }
    sr = models["OLD"]["config"]["target_sr"]
    duration = models["OLD"]["config"]["target_duration"]

    files = get_test_files()
    print(f"Evaluating {len(files)} test files across clean + {WEBM_BITRATES_KBPS} kbps webm + "
          f"{list(DEVICE_PROFILES)} device profiles + compound (device-filter THEN webm32) variants, "
          f"on OLD vs COMP vs FULL vs COMPOUND models.\n")

    rows = []
    for i, (true_label, path) in enumerate(files, 1):
        y, _ = load_clean_audio(path, target_sr=sr, target_duration=duration, fallback_to_untrimmed=True)

        variants = [("clean", y)]
        variants += [(f"webm{k}", compress_roundtrip(y, sr, k)) for k in WEBM_BITRATES_KBPS]
        variants += [(name, simulate_device(y, sr, hp, lp)) for name, (hp, lp) in DEVICE_PROFILES.items()]
        # the realistic JOINT condition: device coloring THEN webm compression, on the
        # SAME clip -- this is what a real user with a real (unknown) phone actually
        # produces, not either transformation in isolation
        variants += [(f"compound_{name}", compress_roundtrip(simulate_device(y, sr, hp, lp), sr, 32))
                     for name, (hp, lp) in DEVICE_PROFILES.items()]

        row = {"file": os.path.basename(path), "true_label": true_label}
        for tag, y_variant in variants:
            for model_name, bundle in models.items():
                pred, conf = predict_array(y_variant, sr, bundle)
                row[f"{tag}_{model_name}_correct"] = pred == true_label
        rows.append(row)
        if i % 20 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] processed")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_old_vs_full_fix_results.csv")
    df.to_csv(out_csv, index=False)

    model_names = ["OLD", "COMP", "FULL", "COMPOUND"]
    print("\n" + "=" * 70)
    print("SUMMARY -- accuracy on the FULL test set")
    print("  OLD=deployed  COMP=compression-only  FULL=compression+mic as separate dims  "
          "COMPOUND=device-filter THEN compression, same clip")
    print("=" * 70)
    tags = (["clean"] + [f"webm{k}" for k in WEBM_BITRATES_KBPS] + list(DEVICE_PROFILES)
            + [f"compound_{n}" for n in DEVICE_PROFILES])
    header = f"{'condition':16s}" + "".join(f"{m:>10s}" for m in model_names)
    print(header)
    for tag in tags:
        vals = [df[f"{tag}_{m}_correct"].mean() for m in model_names]
        print(f"{tag:16s}" + "".join(f"{v:10.3f}" for v in vals))

    print(f"\nFull per-file results saved to: {out_csv}")


if __name__ == "__main__":
    main()
