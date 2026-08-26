# -*- coding: utf-8 -*-
r"""
Consolidates every 5-seed x augmentation variance study run for the
graduation-project comprehensive benchmark (2026-08) into ONE clean
summary table: model x condition x mean/std accuracy/macro-F1 across
5 independent seeds, all on the FULL dataset (~890 files / ~626 train).

Reads:
  processed_data/traditional_ml_augmentation_variance.csv
  processed_data/cnn_augmentation_variance.csv
  processed_data/augmentation_variance_study_full_data.csv   (yamnet, panns, ast, clap, passt, beats)
  processed_data/efficientat_augmentation_study.csv

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python summarize_full_benchmark.py
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")


def load_all():
    rows = []

    tml = pd.read_csv(os.path.join(DATA_DIR, "traditional_ml_augmentation_variance.csv"))
    for cond, g in tml.groupby("condition"):
        rows.append(("Traditional ML (SVM/RF/XGBoost)", cond, g["test_accuracy"].values, g["test_macro_f1"].values))

    cnn = pd.read_csv(os.path.join(DATA_DIR, "cnn_augmentation_variance.csv"))
    for cond, g in cnn.groupby("condition"):
        rows.append(("CNN", cond, g["test_accuracy"].values, g["test_macro_f1"].values))

    frozen = pd.read_csv(os.path.join(DATA_DIR, "augmentation_variance_study_full_data.csv"))
    name_map = {"yamnet": "YAMNet", "panns": "PANNs", "ast": "AST", "clap": "CLAP",
                "passt": "PaSST", "beats": "BEATs"}
    for (model, augment), g in frozen.groupby(["model", "augment"]):
        cond = "with_augmentation" if augment else "no_augmentation"
        rows.append((name_map.get(model, model), cond, g["test_accuracy"].values, g["test_macro_f1"].values))

    eat = pd.read_csv(os.path.join(DATA_DIR, "efficientat_augmentation_study.csv"))
    for aug, g in eat.groupby("augment"):
        cond = "with_augmentation" if aug else "no_augmentation"
        rows.append(("EfficientAT (fine-tuned)", cond, g["test_accuracy"].values, g["test_macro_f1"].values))

    return rows


def main():
    rows = load_all()

    print("=" * 100)
    print("COMPREHENSIVE BENCHMARK -- all models x 5 seeds x augmentation on/off, FULL dataset")
    print("=" * 100)
    header = f"{'Model':38s} {'Condition':20s} {'n':>3s} {'mean_acc':>9s} {'std_acc':>8s} {'mean_f1':>9s} {'std_f1':>8s} {'min':>7s} {'max':>7s}"
    print(header)
    print("-" * len(header))

    summary_rows = []
    for model, cond, accs, f1s in sorted(rows, key=lambda r: (r[0], r[1])):
        mean_a, std_a = np.mean(accs), np.std(accs)
        mean_f, std_f = np.mean(f1s), np.std(f1s)
        print(f"{model:38s} {cond:20s} {len(accs):3d} {mean_a:9.4f} {std_a:8.4f} {mean_f:9.4f} {std_f:8.4f} "
              f"{np.min(accs):7.4f} {np.max(accs):7.4f}")
        summary_rows.append({"model": model, "condition": cond, "n_seeds": len(accs),
                              "mean_accuracy": mean_a, "std_accuracy": std_a,
                              "mean_macro_f1": mean_f, "std_macro_f1": std_f,
                              "min_accuracy": np.min(accs), "max_accuracy": np.max(accs)})

    out_df = pd.DataFrame(summary_rows)
    out_path = os.path.join(DATA_DIR, "full_benchmark_summary.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\n" + "=" * 100)
    print("AUGMENTATION VERDICT PER MODEL (mean accuracy gap = with_aug - no_aug; "
          "'real' if |gap| > pooled std of the two conditions)")
    print("=" * 100)
    by_model = {}
    for r in summary_rows:
        by_model.setdefault(r["model"], {})[r["condition"]] = r
    for model, conds in by_model.items():
        if "no_augmentation" not in conds or "with_augmentation" not in conds:
            continue
        a, b = conds["no_augmentation"], conds["with_augmentation"]
        gap = b["mean_accuracy"] - a["mean_accuracy"]
        pooled_std = np.sqrt((a["std_accuracy"] ** 2 + b["std_accuracy"] ** 2) / 2)
        verdict = "NOISE (not a real effect)" if pooled_std == 0 or abs(gap) < 1.5 * pooled_std else \
                  ("AUGMENTATION HELPS" if gap > 0 else "AUGMENTATION HURTS")
        print(f"{model:38s} gap={gap:+.4f}  pooled_std={pooled_std:.4f}  -> {verdict}")


if __name__ == "__main__":
    main()
