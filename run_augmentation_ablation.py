# -*- coding: utf-8 -*-
r"""
===================================================================
Augmentation ablation study - does data augmentation actually help?
===================================================================
Runs the FULL pipeline TWICE on the real dataset:
    - "no_augmentation"   : N_AUGMENTATIONS_PER_TRAIN_FILE = 0
    - "with_augmentation" : N_AUGMENTATIONS_PER_TRAIN_FILE = 2 (preprocessing.py's default)

using the EXACT SAME train/val/test split both times -- the group-aware,
class-balanced split (preprocessing.py's greedy_balanced_group_split) only
depends on the matched file set and RANDOM_STATE, never on augmentation, so
this is a fair, apples-to-apples comparison: same val/test files judged
both times, only the TRAIN data differs (624 original clips vs 1872
clips including augmented copies, for example).

Both the traditional ML models (SVM/RF/XGBoost) and the CNN are trained
under each condition, and final TEST metrics (overall macro-F1 + PER-CLASS
recall, since "sway" is the class augmentation is most hoped to help) are
compared side by side at the end.

Each condition writes to its own isolated folder under ablation_results/,
so this NEVER touches your real processed_data/ or any already-trained
models sitting there.

WHY SUBPROCESSES: each stage (preprocessing.py / train_traditional_ml.py /
train_cnn.py) runs as a completely separate OS process (not just a fresh
Python import in one long-lived process). TensorFlow and large numpy
arrays don't reliably release memory back to the OS between runs within
the same process, so a single long-lived driver process running 2
conditions x 3 stages back-to-back on the real dataset can accumulate
enough memory to get killed with no Python traceback at all (this is
exactly what happened the first time this script ran as an in-process
driver). A fresh OS process per stage guarantees a clean memory slate --
this is also exactly how you'd run these scripts by hand one at a time.

HOW TO RUN:
    python run_augmentation_ablation.py

By default this reads the real belt/brake/sway audio from
REAL_DATA_BASE_DIR below (edit it if your data lives somewhere else) --
it never writes anything back there, only reads the raw audio + Excel
metadata. Each stage's full console output is also saved to a .log file
next to its report, in case anything needs debugging.
"""

import os
import sys
import json
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable

REAL_DATA_BASE_DIR = r"C:\Users\hp\Desktop\car_test"  # where belt/brake/sway + xlsx actually live (read-only)
ABLATION_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "ablation_results")

CONDITIONS = {
    "no_augmentation": 0,
    "with_augmentation": 2,
}

STAGES = ["preprocessing.py", "train_traditional_ml.py", "train_cnn.py"]


def run_stage(script_name, out_dir, n_augment):
    env = os.environ.copy()
    env["CAR_TEST_RAW_DATA_DIR"] = REAL_DATA_BASE_DIR
    env["CAR_TEST_OUTPUT_DIR"] = out_dir
    env["PREPROCESSING_N_AUGMENTATIONS"] = str(n_augment)
    # unbuffered + force utf-8 stdout so the log file gets everything, in order
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    log_path = os.path.join(out_dir, script_name.replace(".py", ".log"))
    print(f"      running {script_name} (log: {log_path}) ...")
    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(
            [PYTHON_EXE, os.path.join(SCRIPT_DIR, script_name)],
            cwd=SCRIPT_DIR, env=env,
            stdout=log_file, stderr=subprocess.STDOUT,
            text=True,
        )
    if result.returncode != 0:
        print(f"      !! {script_name} FAILED (exit code {result.returncode}) -- see {log_path}")
        return False
    print(f"      -> {script_name} OK")
    return True


def run_condition(label, n_augment):
    out_dir = os.path.join(ABLATION_OUTPUT_DIR, label)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "#" * 70)
    print(f"CONDITION: {label} (N_AUGMENTATIONS_PER_TRAIN_FILE={n_augment})")
    print("#" * 70)

    for stage in STAGES:
        ok = run_stage(stage, out_dir, n_augment)
        if not ok:
            return False
    return True


def parse_test_block(report_text):
    marker = "FINAL TEST results"
    idx = report_text.find(marker)
    return report_text[idx:].strip() if idx != -1 else report_text.strip()


def main():
    if not os.path.isdir(REAL_DATA_BASE_DIR):
        print(f"!! REAL_DATA_BASE_DIR not found: {REAL_DATA_BASE_DIR}\n"
              f"   Edit REAL_DATA_BASE_DIR at the top of this script to point at your belt/brake/sway data.")
        sys.exit(1)

    summary = {}
    for label, n_aug in CONDITIONS.items():
        ok = run_condition(label, n_aug)
        out_dir = os.path.join(ABLATION_OUTPUT_DIR, label)

        cond_summary = {}
        trad_report_path = os.path.join(out_dir, "traditional_ml_report.txt")
        if os.path.exists(trad_report_path):
            with open(trad_report_path, encoding="utf-8") as f:
                cond_summary["traditional_ml"] = parse_test_block(f.read())

        cnn_report_path = os.path.join(out_dir, "cnn_report.txt")
        if os.path.exists(cnn_report_path):
            with open(cnn_report_path, encoding="utf-8") as f:
                cond_summary["cnn"] = f.read().strip()

        cond_summary["stages_completed_ok"] = ok
        summary[label] = cond_summary

    with open(os.path.join(ABLATION_OUTPUT_DIR, "ablation_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n\n" + "=" * 70)
    print("AUGMENTATION ABLATION -- SIDE BY SIDE COMPARISON")
    print("=" * 70)
    for model_key, title in [("traditional_ml", "TRADITIONAL ML (best of SVM/RF/XGBoost)"), ("cnn", "CNN")]:
        print(f"\n########## {title} ##########")
        for label in CONDITIONS:
            text = summary[label].get(model_key, "(no report found -- check the .log files)")
            print(f"\n--- {label} ---")
            print(text)

    print("\n" + "=" * 70)
    print(f"Full reports + per-stage logs saved under: {ABLATION_OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
