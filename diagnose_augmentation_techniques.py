# -*- coding: utf-8 -*-
"""
Fast, controlled comparison of augmentation TECHNIQUE choice (not just noise level) --
skips log-mel/CNN entirely (FAST_MODE_SKIP_MEL) and uses both CPU cores, since only
the MFCC/HPSS feature vector + traditional ML training is needed to see the effect.
Each config is still a completely fresh preprocessing.py run against the real dataset
(so still subject to un-seeded augmentation randomness -- a single run each, not
repeated trials -- but fast enough that the earlier "was this signal or noise" question
can at least be narrowed down cheaply before committing to a slower full-pipeline rerun).

Configs tested:
  no_augmentation       : N_AUGMENTATIONS=0 (baseline, reference)
  no_noise              : time_shift + volume only, no noise at all
  light_noise           : time_shift + volume + noise(0.001) -- current default
"""

import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXE = sys.executable
REAL_DATA_BASE_DIR = r"C:\Users\hp\Desktop\car_test"
OUT_ROOT = os.path.join(SCRIPT_DIR, "aug_technique_diagnostic")

CONFIGS = {
    "no_augmentation": {"N_AUGMENT": "0", "TECHNIQUES": "time_shift,volume,noise"},
    "no_noise": {"N_AUGMENT": "2", "TECHNIQUES": "time_shift,volume"},
    "light_noise": {"N_AUGMENT": "2", "TECHNIQUES": "time_shift,volume,noise"},
}


def run_stage(script_name, out_dir, extra_env):
    env = os.environ.copy()
    env["CAR_TEST_RAW_DATA_DIR"] = REAL_DATA_BASE_DIR
    env["CAR_TEST_OUTPUT_DIR"] = out_dir
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra_env)

    log_path = os.path.join(out_dir, script_name.replace(".py", ".log"))
    print(f"      running {script_name} (log: {log_path}) ...")
    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(
            [PYTHON_EXE, os.path.join(SCRIPT_DIR, script_name)],
            cwd=SCRIPT_DIR, env=env,
            stdout=log_file, stderr=subprocess.STDOUT, text=True,
        )
    ok = result.returncode == 0
    print(f"      -> {script_name} {'OK' if ok else 'FAILED, see log'}")
    return ok


def main():
    results = {}
    for label, cfg in CONFIGS.items():
        out_dir = os.path.join(OUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)
        print("\n" + "#" * 70)
        print(f"CONFIG: {label}  (N_AUGMENT={cfg['N_AUGMENT']}, techniques={cfg['TECHNIQUES']})")
        print("#" * 70)

        extra_env = {
            "PREPROCESSING_N_AUGMENTATIONS": cfg["N_AUGMENT"],
            "PREPROCESSING_AUG_TECHNIQUES": cfg["TECHNIQUES"],
            "PREPROCESSING_FAST_MODE_SKIP_MEL": "1",
            "PREPROCESSING_N_WORKERS": "2",
        }
        ok = run_stage("preprocessing.py", out_dir, extra_env)
        if ok:
            ok = run_stage("train_traditional_ml.py", out_dir, extra_env)

        report_path = os.path.join(out_dir, "traditional_ml_report.txt")
        if ok and os.path.exists(report_path):
            with open(report_path, encoding="utf-8") as f:
                text = f.read()
            idx = text.find("FINAL TEST results")
            results[label] = text[idx:idx + 500] if idx != -1 else text[:500]
        else:
            results[label] = "(FAILED -- see log)"

    print("\n\n" + "=" * 70)
    print("AUGMENTATION TECHNIQUE DIAGNOSTIC -- SUMMARY")
    print("=" * 70)
    for label in CONFIGS:
        print(f"\n--- {label} ---")
        print(results[label])


if __name__ == "__main__":
    main()
