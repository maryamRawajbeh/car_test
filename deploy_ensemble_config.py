# -*- coding: utf-8 -*-
r"""
Deploy the validated 5-model-always-on ensemble weights to the live
processed_data/ensemble_config.pkl.

Rationale (full story: SESSION_HANDOFF_20260827.md Section 3, and
nested_cv_ensemble_5model_alwayson.py which reproduces every number):

  * The deployed 3-model fixed weights (0.30 Traditional / 0.60 CNN /
    0.10 YAMNet) score 89.47% acc / 89.33% macro-F1 on the untouched
    test set; out-of-fold macro-F1 mean=0.8596, std=0.1054.
  * A nested 5-fold CV weight search over ONLY the five models that are
    already computed on every prediction today (Traditional ML, CNN,
    YAMNet, PANNs/CNN14, EfficientAT fine-tuned) gives out-of-fold
    macro-F1 mean=0.9066, std=0.0784 -- better mean AND lower variance.
  * The element-wise mean of the 5 folds' winning weight vectors is
    {Traditional 0.14, CNN 0.18, YAMNet 0.00, PANNs 0.04, EfficientAT
    0.64}. Evaluated EXACTLY ONCE on the untouched test set: 90.23% acc
    / 90.24% macro-F1 -- a real +0.76pt improvement at zero extra cost
    (all five models already run on every request; only fusion weights
    change; no code change; no RAM/latency change).
  * YAMNet is dropped from the config's `names` list because its weight
    is exactly 0.00 -- same convention used previously when PANNs sat at
    0.00 in the old config.

This script backs up the current config with a timestamp before writing.
"""
import datetime
import pickle
import shutil
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "processed_data" / "ensemble_config.pkl"

NEW_CONFIG = {
    "mode": "weights",
    "names": ["Traditional ML", "CNN", "PANNs (CNN14)", "EfficientAT (fine-tuned)"],
    "weights": (0.14, 0.18, 0.04, 0.64),
    "meta_model": None,
}


def main():
    assert abs(sum(NEW_CONFIG["weights"]) - 1.0) < 1e-9, "weights must sum to 1.0"
    assert len(NEW_CONFIG["names"]) == len(NEW_CONFIG["weights"])

    old = pickle.loads(CONFIG_PATH.read_bytes())
    print("current config:", old)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CONFIG_PATH.with_name(f"ensemble_config_BACKUP_{stamp}.pkl")
    shutil.copy(CONFIG_PATH, backup)
    print("backed up ->", backup.name)

    with open(CONFIG_PATH, "wb") as f:
        pickle.dump(NEW_CONFIG, f)

    written = pickle.loads(CONFIG_PATH.read_bytes())
    print("new config:    ", written)
    assert written == NEW_CONFIG
    print("OK - deployed. Restart garageai-audio-analysis to pick this up.")


if __name__ == "__main__":
    main()
