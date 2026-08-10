# -*- coding: utf-8 -*-
"""One-off re-run of just the 'with_augmentation' condition (preprocessing +
train_traditional_ml + train_cnn) after lowering aug_add_noise's noise_level,
to verify the fix actually recovers the accuracy regression. Reuses
run_augmentation_ablation.py's run_condition() so it's byte-identical to how
the full ablation study invokes each stage (isolated subprocesses, same env
var wiring)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_augmentation_ablation as ra

ok = ra.run_condition("with_augmentation", 2)
print("with_augmentation condition finished OK:", ok)
