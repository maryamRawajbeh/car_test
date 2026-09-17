# -*- coding: utf-8 -*-
r"""
===================================================================
Deploy the compound-augmented SVM to the REAL production directory
===================================================================
Follows the exact same safe pattern as deploy_compression_augmented_model.py
(back up first, copy only the matched best_traditional_model.pkl + scaler.pkl
pair), but points at this session's validated fix: SVM (C=10, RBF kernel)
trained on the 626-recording training partition augmented with 5 "compound"
copies per file (simulated mic-response filter, then WebM/Opus compression
at a random one of 16/32/64 kbps) -- see the paper's Section 9.4.

Full 133-file held-out test set results (real WebM/Opus round-trip, current
production decode path, no forced ffmpeg resample):
  XGBoost (currently deployed): 93.23% clean -> 39.10% at 32kbps (collapse)
  This SVM:                     86.47% clean -> 80.45% / 84.96% / 84.96%
                                 at 16 / 32 / 64 kbps

config.pkl and label_encoder.pkl are unchanged from production (verified
byte-identical feature-extraction config and class order), so only the
model+scaler pair is touched -- same as the precedent script.

Does NOT touch cnn_model.keras, mel_stats.pkl, best_transfer_model.pkl,
best_panns_model.pkl, or ensemble_config.pkl. The weighted ensemble's
Traditional-ML input (80% of its blend) now comes from this more
compression-robust model automatically, without needing its own retraining
or reweighting -- but its end-to-end compression behavior has not been
separately re-verified here, only the standalone traditional-ML model.

Run:
  cd C:\Users\hp\Desktop\car_test
  python deploy_compound_augmented_svm.py [--yes]   (--yes skips the confirmation prompt)
"""

import os
import sys
import shutil
import argparse
from datetime import datetime

SOURCE_DIR = r"C:\Users\hp\Desktop\car_test\ablation_results\compound_augmented_v1"
DEST_DIR = r"C:\Users\hp\Desktop\car_test\processed_data"

FILES = ["best_traditional_model.pkl", "scaler.pkl"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    print(f"Source (new, compound-augmented SVM): {SOURCE_DIR}")
    print(f"Destination (REAL PRODUCTION):         {DEST_DIR}")
    print()
    for f in FILES:
        src = os.path.join(SOURCE_DIR, f)
        if not os.path.exists(src):
            print(f"!! Missing source file: {src} -- aborting, nothing was copied.")
            sys.exit(1)
    print("All source files present:", ", ".join(FILES))

    if not args.yes:
        answer = input("\nThis will overwrite live production model files (after backing up the "
                        "current ones). Type 'deploy' to continue: ")
        if answer.strip() != "deploy":
            print("Aborted -- nothing was changed.")
            sys.exit(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(DEST_DIR, f"_backup_before_compound_svm_deploy_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    print(f"\nBacking up current files to: {backup_dir}")
    for f in FILES:
        dest_path = os.path.join(DEST_DIR, f)
        if os.path.exists(dest_path):
            shutil.copy2(dest_path, os.path.join(backup_dir, f))
            print(f"  backed up: {f}")
        else:
            print(f"  (no existing {f} to back up)")

    print("\nCopying new compound-augmented SVM files into production...")
    for f in FILES:
        shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(DEST_DIR, f))
        print(f"  deployed: {f}")

    print(f"\nDone. Old files backed up at: {backup_dir}")
    print("Restart the garageai-audio-analysis microservice for it to pick up the new files "
          "(model_loader.py loads everything once at startup).")
    print("\nRecommended: sanity-check with a real recording through the actual app before "
          "considering this fully verified in production.")


if __name__ == "__main__":
    main()
