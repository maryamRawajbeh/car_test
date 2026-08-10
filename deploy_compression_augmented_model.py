# -*- coding: utf-8 -*-
r"""
===================================================================
Deploy the compression-augmented model to the REAL production directory
===================================================================
See SESSION_HANDOFF.md section 4. Only run this AFTER
run_cnn_augmentation_variance.py has confirmed the compression-augmented
CNN's improvement holds up across multiple training seeds (not just one
lucky run) -- this script does not check that itself, it trusts the
person running it to have verified it first.

Copies ONLY the two file-pairs that the domain-shift fix actually
touched, as matched pairs (never one half without the other):
  - best_traditional_model.pkl + scaler.pkl
  - cnn_model.keras + mel_stats.pkl (mel normalization stats MUST match
    the exact model they were computed alongside)

Deliberately does NOT touch best_transfer_model.pkl (YAMNet),
best_panns_model.pkl, or ensemble_config.pkl -- untouched by this fix,
and don't affect the real final prediction anyway (ensemble weights are
0.8 traditional / 0.2 CNN / 0.0 YAMNet, PANNs not in the weighted list
at all -- see SESSION_HANDOFF.md section 1).

Backs up whatever is currently in the destination first (own timestamped
folder, never overwritten) -- this is live production, treat it that way.

Run:
  cd C:\Users\hp\Desktop\car_test
  python deploy_compression_augmented_model.py [--yes]   (--yes skips the confirmation prompt)
"""

import os
import sys
import shutil
import argparse
from datetime import datetime

SOURCE_DIR = r"C:\Users\hp\Desktop\car_test\ablation_results\compression_augmented"
DEST_DIR = r"C:\Users\hp\Desktop\car_test\processed_data"

FILE_PAIRS = [
    ("best_traditional_model.pkl", "scaler.pkl"),
    ("cnn_model.keras", "mel_stats.pkl"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = parser.parse_args()

    all_files = [f for pair in FILE_PAIRS for f in pair]

    print(f"Source (new, compression-augmented): {SOURCE_DIR}")
    print(f"Destination (REAL PRODUCTION):        {DEST_DIR}")
    print()
    for f in all_files:
        src = os.path.join(SOURCE_DIR, f)
        if not os.path.exists(src):
            print(f"!! Missing source file: {src} -- aborting, nothing was copied.")
            sys.exit(1)
    print("All source files present:", ", ".join(all_files))

    if not args.yes:
        answer = input("\nThis will overwrite live production model files (after backing up the "
                        "current ones). Type 'deploy' to continue: ")
        if answer.strip() != "deploy":
            print("Aborted -- nothing was changed.")
            sys.exit(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = os.path.join(DEST_DIR, f"_backup_before_compression_fix_{timestamp}")
    os.makedirs(backup_dir, exist_ok=True)

    print(f"\nBacking up current files to: {backup_dir}")
    for f in all_files:
        dest_path = os.path.join(DEST_DIR, f)
        if os.path.exists(dest_path):
            shutil.copy2(dest_path, os.path.join(backup_dir, f))
            print(f"  backed up: {f}")
        else:
            print(f"  (no existing {f} to back up)")

    print("\nCopying new compression-augmented files into production...")
    for f in all_files:
        shutil.copy2(os.path.join(SOURCE_DIR, f), os.path.join(DEST_DIR, f))
        print(f"  deployed: {f}")

    print(f"\nDone. Old files backed up at: {backup_dir}")
    print("Restart the garageai-audio-analysis microservice for it to pick up the new files "
          "(model_loader.py loads everything once at startup).")
    print("\nRecommended: sanity-check with a real recording through the actual app before "
          "considering this fully verified in production.")


if __name__ == "__main__":
    main()
