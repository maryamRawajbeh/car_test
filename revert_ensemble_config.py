# -*- coding: utf-8 -*-
r"""
Revert processed_data/ensemble_config.pkl to the original 3-model config
  {Traditional ML 0.30, CNN 0.60, YAMNet 0.10}

Why (see mic_pipeline_test/test_full_ensemble_compression_4model.py and its
CSV): the 2026-08-30 nested-CV retune to {Traditional 0.14, CNN 0.18,
PANNs 0.04, EfficientAT 0.64} raised CLEAN test accuracy 89.47% -> 90.23%
(+0.76pt), but end-to-end testing through the real WebM/Opus production
pipeline on all 133 test recordings showed it REGRESSES badly under
compression:

  condition   NEW (4-model)   OLD (3-model)
  clean          90.2%           89.5%
  control        76.7%           85.7%    (plain WAV round-trip, no lossy codec)
  webm 16k       75.9%           83.5%
  webm 32k       77.4%           82.7%
  webm 64k       75.2%           85.7%
  Brake@webm32   45.5%           72.7%

The dominant model in the new config (EfficientAT, weight 0.64) was never
compression-augmented, so it is confidently wrong on compressed audio and
drags the whole fusion with it. The live app feeds exactly this compressed
WebM/Opus audio, so the original {0.30/0.60/0.10} config -- which holds
84.0% under the same compression -- is the correct one to keep deployed.
"""
import datetime
import pickle
import shutil
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "processed_data" / "ensemble_config.pkl"

ORIGINAL_CONFIG = {
    "mode": "weights",
    "names": ["Traditional ML", "CNN", "Transfer Learning (YAMNet)"],
    "weights": (0.30, 0.60, 0.10),
    "meta_model": None,
}


def main():
    current = pickle.loads(CONFIG_PATH.read_bytes())
    print("current config:", current)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = CONFIG_PATH.with_name(f"ensemble_config_BEFORE_REVERT_{stamp}.pkl")
    shutil.copy(CONFIG_PATH, backup)
    print("backed up current ->", backup.name)

    with open(CONFIG_PATH, "wb") as f:
        pickle.dump(ORIGINAL_CONFIG, f)

    written = pickle.loads(CONFIG_PATH.read_bytes())
    print("reverted config:  ", written)
    assert written == ORIGINAL_CONFIG
    print("OK - reverted to the compression-robust 3-model config.")


if __name__ == "__main__":
    main()
