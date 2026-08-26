# -*- coding: utf-8 -*-
r"""
===================================================================
Data Leakage Check - do any vehicles appear in more than one file,
and if so, did their files end up split across train/val/test?
===================================================================
Reads directly from dataset_documentation.csv (produced by
preprocessing.py) instead of re-reading the Excel files and re-globbing
audio folders itself -- one less place where folder paths can drift
out of sync with preprocessing.py.

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python check_data_leakage.py
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from preprocessing import build_vehicle_groups

BASE_DIR = r"C:\Users\hp\Desktop\car_test"
DATA_DIR = os.path.join(BASE_DIR, "processed_data")


def main():
    print("=" * 70)
    print("Checking for potential data leakage (same vehicle in multiple files)")
    print("=" * 70)

    doc_path = os.path.join(DATA_DIR, "dataset_documentation.csv")
    if not os.path.exists(doc_path):
        print(f"\n!! Could not find {doc_path}. Run preprocessing.py first.")
        return

    meta = pd.read_csv(doc_path)
    required = {"class", "split_assigned", "matched_path"}
    if not required.issubset(meta.columns):
        print(f"\n!! dataset_documentation.csv is missing expected columns: {required - set(meta.columns)}")
        return

    for col in ["brand", "model", "year"]:
        if col not in meta.columns:
            print(f"\n!! Column '{col}' not found in your Excel metadata - cannot fully check vehicle grouping.")
            meta[col] = "unknown"

    # Reuses preprocessing.py's ACTUAL grouping logic (the one that really determined
    # the train/val/test split) instead of a separately hand-rolled string-concat here.
    # The two used to disagree on how to treat placeholder values ("Unknown", "N/A",
    # empty, ...): preprocessing.py's build_vehicle_groups() gives each such row its
    # own singleton group (so two different vehicles that both just happen to have
    # brand="Unknown" are never treated as the same vehicle), but this script's old
    # inline logic merged them by the literal string "unknown" -- which manufactured
    # false-positive "leakage" for placeholder-metadata rows that preprocessing.py's
    # real split never actually grouped together. Importing the same function makes
    # the two impossible to drift apart again, the same principle audio_common.py
    # already applies to feature extraction across this project's scripts.
    meta["vehicle_id"] = build_vehicle_groups(meta)

    print(f"\nTotal matched files analyzed: {len(meta)}")
    print(f"Total unique vehicles (class+brand+model+year combinations): {meta['vehicle_id'].nunique()}")

    vehicle_counts = meta.groupby("vehicle_id").size().sort_values(ascending=False)
    repeated_vehicles = vehicle_counts[vehicle_counts > 1]
    print(f"Vehicles that appear in MORE THAN ONE file: {len(repeated_vehicles)}")

    if len(repeated_vehicles) == 0:
        print("\n>>> GOOD NEWS: every vehicle appears in exactly one file. No leakage risk from repeated vehicles.")
        return

    print("\nChecking whether any repeated vehicle's files landed in more than one split (train/val/test)...")
    leak_report = []
    n_leaking = 0
    for vid in repeated_vehicles.index:
        rows = meta[meta["vehicle_id"] == vid]
        splits_involved = sorted(rows["split_assigned"].unique())
        if len(splits_involved) > 1:
            n_leaking += 1
            leak_report.append({
                "vehicle_id": vid,
                "num_files": len(rows),
                "splits_involved": ", ".join(splits_involved),
                "file_paths": ", ".join(rows["matched_path"].tolist()),
            })

    print(f"\n>>> Vehicles whose files are SPLIT ACROSS multiple partitions (real leakage): {n_leaking}")

    if n_leaking > 0:
        leak_df = pd.DataFrame(leak_report)
        leak_df.to_csv(os.path.join(DATA_DIR, "data_leakage_report.csv"), index=False, encoding="utf-8-sig")
        print(f"    -> details saved to processed_data/data_leakage_report.csv")
        print("\n    Example cases:")
        print(leak_df.head(10).to_string(index=False))
    else:
        print("    None of the repeated vehicles got split across partitions - your current train/val/test split is SAFE.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
