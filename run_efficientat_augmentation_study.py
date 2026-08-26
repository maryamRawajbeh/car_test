# -*- coding: utf-8 -*-
r"""
===================================================================
Augmentation x seed variance study - EfficientAT (real fine-tuning)
===================================================================
Same motivation as run_augmentation_variance_study.py (AST/CLAP/PaSST/
BEATs), applied to the one model in this project that actually
fine-tunes (train_efficientat_finetune.py) instead of extracting
frozen embeddings: does the project's waveform augmentation help a
model that's really being trained end-to-end, and how much does
plain seed variance move the result?

WHY THIS SCRIPT IS SEPARATE FROM run_augmentation_variance_study.py:
  That script's models are frozen -- augmentation just expands a
  STATIC set of (embedding, label) pairs once per config, computed
  before any classifier training happens. EfficientAT's fine-tuning
  is a real training LOOP (backprop every epoch): augmentation here
  means generating a NEW random augmented waveform on every access to
  a training clip (every epoch sees a differently-augmented version),
  which is the standard "online augmentation" pattern for training
  loops -- not directly comparable in mechanism to the static-copies
  approach, even though the underlying technique (time shift/volume/
  noise from audio_common.make_augmented_version) is identical.

  Unlike the frozen-embedding study, train is NOT subsampled here --
  EfficientAT's backbone is cheap enough on CPU to fine-tune on the
  full 626-file train split every run (the original single run
  confirmed this takes tens of minutes, not hours), so there's no
  speed reason to shrink it, and shrinking it would also change what
  question is being answered.

RESULTS: appended to processed_data/efficientat_augmentation_study.csv
(separate file/schema from the frozen-embedding study's CSV -- this
one has no "classifier head" to report, just epochs/early-stopping
info). Does NOT touch best_efficientat_ft_model.pt / efficientat_ft_
report.txt (the main comparison-table artifact already validated via
predict.py) -- separate study, separate output.

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python run_efficientat_augmentation_study.py [--seeds 1,2,3,4,5]
"""

import os
import sys
import copy
import csv
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from audio_common import load_clean_audio as _shared_load_clean_audio, make_augmented_version

BASE_DIR = r"C:\Users\hp\Desktop\car_test"
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
EFFICIENTAT_VENDOR_DIR = os.path.join(BASE_DIR, "efficientat_vendor")
sys.path.insert(0, EFFICIENTAT_VENDOR_DIR)

EFFICIENTAT_SR = 32000
TARGET_DURATION = 5.0
PRETRAINED_NAME = "mn10_as"
N_UNFROZEN_BLOCKS = 3
BATCH_SIZE = 16
N_EPOCHS = 30
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 6
RESULTS_CSV = os.path.join(DATA_DIR, "efficientat_augmentation_study.csv")


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_clean_audio_32k(path):
    y, _ = _shared_load_clean_audio(
        path, target_sr=EFFICIENTAT_SR, target_duration=TARGET_DURATION, fallback_to_untrimmed=True,
    )
    return y.astype(np.float32)


class ClipDataset(torch.utils.data.Dataset):
    """Same as train_efficientat_finetune.py's ClipDataset, plus an `augment` flag:
    when True, applies audio_common.make_augmented_version to the waveform on EVERY
    access (so each epoch sees a fresh random augmentation -- online augmentation,
    not a static pre-expanded set). Only meaningful for the train split; val/test
    datasets are always constructed with augment=False."""

    def __init__(self, paths, labels, augment=False):
        self.paths = list(paths)
        self.labels = list(labels)
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        y = load_clean_audio_32k(self.paths[idx])
        if self.augment:
            y = make_augmented_version(y)
        return torch.from_numpy(y), int(self.labels[idx])


def build_model_and_mel():
    from models.mn.model import get_model as get_mn
    from models.preprocess import AugmentMelSTFT
    from helpers.utils import NAME_TO_WIDTH

    mel = AugmentMelSTFT(n_mels=128, sr=EFFICIENTAT_SR)
    model = get_mn(
        num_classes=3,
        pretrained_name=PRETRAINED_NAME,
        width_mult=NAME_TO_WIDTH(PRETRAINED_NAME),
        head_type="mlp",
    )
    return model, mel


def apply_freezing(model, n_unfrozen_blocks):
    for p in model.parameters():
        p.requires_grad = False
    for layer in model.features[-n_unfrozen_blocks:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def run_epoch(model, mel, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    mel.train(is_train)

    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for waveforms, labels in loader:
            spec = mel(waveforms).unsqueeze(1)
            logits, _ = model(spec)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            all_logits.append(logits.detach())
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    y_pred = all_logits.argmax(dim=1).numpy()
    y_true = all_labels.numpy()
    return {
        "loss": total_loss / len(y_true),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": (y_pred == y_true).mean(),
    }


def run_one_config(seed, augment, train_df, val_df, test_df, label_encoder):
    set_seed(seed)
    y_train = label_encoder.transform(train_df["class"])
    y_val = label_encoder.transform(val_df["class"])
    y_test = label_encoder.transform(test_df["class"])

    train_loader = torch.utils.data.DataLoader(
        ClipDataset(train_df["matched_path"], y_train, augment=augment), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        ClipDataset(val_df["matched_path"], y_val, augment=False), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        ClipDataset(test_df["matched_path"], y_test, augment=False), batch_size=BATCH_SIZE, shuffle=False)

    model, mel = build_model_and_mel()
    apply_freezing(model, N_UNFROZEN_BLOCKS)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)

    best_val_f1, best_state, epochs_without_improvement, epochs_trained = -1, None, 0, 0
    for epoch in range(1, N_EPOCHS + 1):
        run_epoch(model, mel, train_loader, criterion, optimizer)
        val_res = run_epoch(model, mel, val_loader, criterion, optimizer=None)
        epochs_trained = epoch
        if val_res["f1_macro"] > best_val_f1:
            best_val_f1 = val_res["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    test_res = run_epoch(model, mel, test_loader, criterion, optimizer=None)
    return {
        "epochs_trained": epochs_trained,
        "best_val_macro_f1": best_val_f1,
        "test_accuracy": test_res["accuracy"],
        "test_macro_f1": test_res["f1_macro"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="1,2,3,4,5")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    import pickle
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)

    doc_df = pd.read_csv(os.path.join(DATA_DIR, "dataset_documentation.csv"))
    corrupted_path = os.path.join(DATA_DIR, "corrupted_files.csv")
    if os.path.exists(corrupted_path):
        corrupted_df = pd.read_csv(corrupted_path)
        if "matched_path" in corrupted_df.columns:
            doc_df = doc_df[~doc_df["matched_path"].isin(corrupted_df["matched_path"])].reset_index(drop=True)

    train_df = doc_df[doc_df["split_assigned"] == "train"]
    val_df = doc_df[doc_df["split_assigned"] == "val"]
    test_df = doc_df[doc_df["split_assigned"] == "test"]
    print(f"train={len(train_df)} (FULL, not subsampled), val={len(val_df)}, test={len(test_df)}")

    is_new_file = not os.path.exists(RESULTS_CSV)
    csv_file = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if is_new_file:
        writer.writerow(["seed", "augment", "epochs_trained", "best_val_macro_f1", "test_accuracy", "test_macro_f1"])

    for seed in seeds:
        for augment in (False, True):
            print(f"\n--- seed={seed} augment={augment} ---")
            res = run_one_config(seed, augment, train_df, val_df, test_df, label_encoder)
            print(f"   -> epochs={res['epochs_trained']} best_val_f1={res['best_val_macro_f1']:.4f} "
                  f"test_accuracy={res['test_accuracy']:.4f} test_macro_f1={res['test_macro_f1']:.4f}")
            writer.writerow([seed, augment, res["epochs_trained"], f"{res['best_val_macro_f1']:.4f}",
                              f"{res['test_accuracy']:.4f}", f"{res['test_macro_f1']:.4f}"])
            csv_file.flush()

    csv_file.close()
    print(f"\nDONE. Results appended to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
