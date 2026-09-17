# -*- coding: utf-8 -*-
r"""
===================================================================
EfficientAT (MobileNetV3, AudioSet-pretrained) - REAL fine-tuning
===================================================================
Every other transfer-learning script in this project (train_panns.py,
train_ast.py, train_clap.py, train_passt.py, train_beats.py) freezes
the pretrained backbone and only trains a small classifier head on
top of its embeddings. All of them lose to the traditional ML model
(86.47% test accuracy) for the same reason: their backbones are
pretrained on AudioSet (general/YouTube sounds), and a frozen backbone
can never close that domain gap, no matter how big or modern it is.

THIS script is different on purpose: it actually FINE-TUNES a
pretrained backbone (unfreezes the last few blocks + trains a new
classification head, real backprop through the backbone) instead of
just extracting frozen embeddings. This is the one experiment in the
project with a real mechanism to close the domain gap, not just
another data point confirming it exists. The backbone chosen
(EfficientAT's `mn10_as`, a MobileNetV3 pretrained on AudioSet) is
deliberately lightweight -- unlike the transformer comparison points
(AST/CLAP/PaSST/BEATs), it's cheap enough on CPU to actually backprop
through repeatedly over multiple epochs in reasonable time.

WHY THIS SCRIPT VENDORS CODE:
  EfficientAT (github.com/fschmid56/EfficientAT) has no pip package.
  This project vendors the handful of source files actually needed
  for loading + fine-tuning the `mn10_as` checkpoint, under
  `efficientat_vendor/` (MIT-licensed, headers preserved):
    - models/mn/model.py, block_types.py, attention_pooling.py, utils.py
    - models/preprocess.py (the AugmentMelSTFT mel-spectrogram frontend
      -- also provides SpecAugment-style freqm/timem masking automatically
      when in .train() mode, same idea already used by this project's own
      CNN via audio_common/train_cnn.py)
    - helpers/utils.py is TRIMMED to just NAME_TO_WIDTH (see that file's
      own header comment for why: the original also loads AudioSet's
      527-class label CSV as an unrelated import-time side effect)
  The pretrained checkpoint itself (mn10_as, a few MB) downloads
  automatically via torch.hub on first run, from EfficientAT's GitHub
  Releases (not vendored, not committed -- see model_dir="resources"
  inside models/mn/model.py).

WHAT THIS SCRIPT DOES:
  1) Reads dataset_documentation.csv -- SAME leakage-safe split as
     everywhere else. Unlike the frozen-embedding scripts, this one
     uses the FULL train split (no subsampling) -- fine-tuning
     benefits from more data, and MobileNetV3 is cheap enough on CPU
     that the full split is still fast (unlike the transformers).
  2) Loads audio at 32kHz (mn10_as's pretraining rate), fixed to 5
     seconds, via the shared audio_common.load_clean_audio.
  3) Loads mn10_as with a FRESH 3-class head (the pretrained 527-class
     AudioSet head is dropped automatically -- see get_model()'s own
     class-count-mismatch handling in models/mn/model.py).
  4) Freezes all of `features` except its last few blocks, keeps the
     new classifier head fully trainable. Fine-tunes with AdamW + a
     small learning rate, class-weighted cross-entropy, for up to
     N_EPOCHS with early stopping on VALIDATION macro-F1 (mirrors this
     project's own CNN training philosophy in train_cnn.py).
  5) Reports final metrics on the TEST set (only evaluated once, at
     the very end, with the best-val-macro-F1 checkpoint).

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python train_efficientat_finetune.py
"""

import os
import sys
import copy
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

from audio_common import load_clean_audio as _shared_load_clean_audio

BASE_DIR = os.environ.get("CAR_TEST_RAW_DATA_DIR", r"C:\Users\hp\Desktop\car_test")
DATA_DIR = os.environ.get("CAR_TEST_OUTPUT_DIR", os.path.join(BASE_DIR, "processed_data"))
EFFICIENTAT_VENDOR_DIR = os.path.join(BASE_DIR, "efficientat_vendor")
sys.path.insert(0, EFFICIENTAT_VENDOR_DIR)  # so `models.mn.model` / `helpers.utils` resolve

EFFICIENTAT_SR = 32000     # mn10_as's pretraining rate (same as PANNs/PaSST)
TARGET_DURATION = 5.0      # keep consistent with the rest of the project
PRETRAINED_NAME = "mn10_as"
N_UNFROZEN_BLOCKS = 3      # how many of the LAST blocks in `features` get fine-tuned
BATCH_SIZE = 16
N_EPOCHS = 30
LEARNING_RATE = 1e-4
EARLY_STOP_PATIENCE = 6    # epochs without val macro-F1 improvement before stopping
SEED = 42


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_clean_audio_32k(path):
    """Shared with preprocessing.py/predict.py via audio_common (same trim/
    normalize/pad + empty-clip fallback logic), just at mn10_as's 32kHz rate."""
    y, _ = _shared_load_clean_audio(
        path, target_sr=EFFICIENTAT_SR, target_duration=TARGET_DURATION, fallback_to_untrimmed=True,
    )
    return y.astype(np.float32)


class ClipDataset(torch.utils.data.Dataset):
    """Loads raw waveforms on-the-fly (cheap: just audio_common.load_clean_audio) --
    mel-spectrogram conversion (with SpecAugment when training) happens later, in
    the training loop, via the shared AugmentMelSTFT module (so .train()/.eval()
    on that ONE module controls augmentation for the whole epoch, matching how
    EfficientAT's own reference scripts use it)."""

    def __init__(self, paths, labels):
        self.paths = list(paths)
        self.labels = list(labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        y = load_clean_audio_32k(self.paths[idx])
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
    """Freezes everything, then unfreezes the LAST n_unfrozen_blocks layers of
    `features` (a flat nn.Sequential: first conv, N inverted-residual blocks, last
    conv) plus the entire classifier head (already fresh/random-init for 3 classes
    -- see get_model()'s class-count-mismatch handling)."""
    for p in model.parameters():
        p.requires_grad = False
    for layer in model.features[-n_unfrozen_blocks:]:
        for p in layer.parameters():
            p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"   Fine-tuning {trainable:,} / {total:,} parameters "
          f"({trainable/total*100:.1f}%) -- last {n_unfrozen_blocks} feature blocks + classifier head")


def run_epoch(model, mel, loader, criterion, optimizer=None):
    """optimizer given -> training epoch (mel/model in .train() mode, SpecAugment
    active, backprop happens). optimizer=None -> eval epoch (.eval() mode, no
    augmentation, no gradient updates)."""
    is_train = optimizer is not None
    model.train(is_train)
    mel.train(is_train)

    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.set_grad_enabled(is_train):
        for waveforms, labels in loader:
            spec = mel(waveforms).unsqueeze(1)  # (batch, n_mels, time) -> (batch, 1, n_mels, time)
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
    probs = torch.softmax(all_logits, dim=1).numpy()
    return {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "probs": probs,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    set_seed(SEED)
    print("=" * 70)
    print("EfficientAT (mn10_as) - real fine-tuning (not frozen embeddings)")
    print("=" * 70)

    doc_path = os.path.join(DATA_DIR, "dataset_documentation.csv")
    if not os.path.exists(doc_path):
        print(f"\n!! {doc_path} not found. Run preprocessing.py first.")
        return
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)

    doc_df = pd.read_csv(doc_path)
    corrupted_path = os.path.join(DATA_DIR, "corrupted_files.csv")
    if os.path.exists(corrupted_path):
        corrupted_df = pd.read_csv(corrupted_path)
        if "matched_path" in corrupted_df.columns:
            doc_df = doc_df[~doc_df["matched_path"].isin(corrupted_df["matched_path"])].reset_index(drop=True)

    train_df = doc_df[doc_df["split_assigned"] == "train"]
    val_df = doc_df[doc_df["split_assigned"] == "val"]
    test_df = doc_df[doc_df["split_assigned"] == "test"]
    print(f"Using the SAME leakage-safe split as the rest of the project: "
          f"train={len(train_df)} (FULL, not subsampled -- see module docstring), "
          f"val={len(val_df)}, test={len(test_df)}")

    y_train = label_encoder.transform(train_df["class"])
    y_val = label_encoder.transform(val_df["class"])
    y_test = label_encoder.transform(test_df["class"])

    train_loader = torch.utils.data.DataLoader(
        ClipDataset(train_df["matched_path"], y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        ClipDataset(val_df["matched_path"], y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        ClipDataset(test_df["matched_path"], y_test), batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n[1] Loading {PRETRAINED_NAME} (downloads checkpoint on first run, a few MB)...")
    model, mel = build_model_and_mel()
    apply_freezing(model, N_UNFROZEN_BLOCKS)

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)

    print(f"\n[2] Fine-tuning for up to {N_EPOCHS} epochs "
          f"(early stopping on val macro-F1, patience={EARLY_STOP_PATIENCE})...\n")
    best_val_f1 = -1
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, N_EPOCHS + 1):
        train_res = run_epoch(model, mel, train_loader, criterion, optimizer)
        val_res = run_epoch(model, mel, val_loader, criterion, optimizer=None)
        print(f"   Epoch {epoch:2d}/{N_EPOCHS}: train_loss={train_res['loss']:.4f} "
              f"train_acc={train_res['accuracy']:.4f} | "
              f"val_loss={val_res['loss']:.4f} val_acc={val_res['accuracy']:.4f} "
              f"val_macroF1={val_res['f1_macro']:.4f}")

        if val_res["f1_macro"] > best_val_f1:
            best_val_f1 = val_res["f1_macro"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"   (early stopping -- no val macro-F1 improvement in {EARLY_STOP_PATIENCE} epochs)")
                break

    print(f"\n[3] Restoring best checkpoint (val macro-F1={best_val_f1:.4f}) and evaluating on TEST...")
    model.load_state_dict(best_state)
    test_res = run_epoch(model, mel, test_loader, criterion, optimizer=None)
    val_res_final = run_epoch(model, mel, val_loader, criterion, optimizer=None)  # for saved val_probs.npy

    report = classification_report(test_res["y_true"], test_res["y_pred"], target_names=class_names, zero_division=0)
    precision_macro = precision_score(test_res["y_true"], test_res["y_pred"], average="macro", zero_division=0)
    recall_macro = recall_score(test_res["y_true"], test_res["y_pred"], average="macro", zero_division=0)

    print(f"   Test accuracy: {test_res['accuracy']:.4f}")
    print(f"   Test macro-F1: {test_res['f1_macro']:.4f}")
    print(f"   Test macro-precision: {precision_macro:.4f}")
    print(f"   Test macro-recall: {recall_macro:.4f}")
    print("\n" + report)

    with open(os.path.join(DATA_DIR, "efficientat_ft_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"REAL FINE-TUNING (not frozen embeddings) -- {PRETRAINED_NAME}, last "
                f"{N_UNFROZEN_BLOCKS} feature blocks + classifier head unfrozen, "
                f"{N_EPOCHS} max epochs, early stopping on val macro-F1 (patience={EARLY_STOP_PATIENCE}).\n")
        f.write(f"Best validation macro-F1 during training: {best_val_f1:.4f}\n\n")
        f.write("=== FINAL TEST results ===\n")
        f.write(f"accuracy={test_res['accuracy']:.4f}, macro-F1={test_res['f1_macro']:.4f}, "
                f"macro-precision={precision_macro:.4f}, macro-recall={recall_macro:.4f}\n\n")
        f.write(report)
        f.write("\n\nThis is the ONE experiment in the project that actually fine-tunes a "
                 "pretrained backbone (real backprop) instead of just extracting frozen "
                 "embeddings -- see README.md for how this compares against traditional ML "
                 "and against the frozen-embedding transfer-learning models.")

    cm = confusion_matrix(test_res["y_true"], test_res["y_pred"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax, cmap="Oranges", values_format="d")
    ax.set_title(f"Confusion Matrix (Test) - {PRETRAINED_NAME} fine-tuned")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "efficientat_ft_confusion_matrix.png"), dpi=150)
    plt.close()

    torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_efficientat_ft_model.pt"))
    np.save(os.path.join(DATA_DIR, "efficientat_ft_val_probs.npy"), val_res_final["probs"])
    np.save(os.path.join(DATA_DIR, "efficientat_ft_test_probs.npy"), test_res["probs"])

    print("\n" + "=" * 70)
    print("DONE. Saved:")
    print("   - best_efficientat_ft_model.pt")
    print("   - efficientat_ft_report.txt")
    print("   - efficientat_ft_confusion_matrix.png")
    print("   - efficientat_ft_val_probs.npy / efficientat_ft_test_probs.npy (for evaluate_ensemble.py)")
    print("=" * 70)
    print("\nUnlike every other transfer-learning script here, this ONE actually fine-tunes -- "
          "compare against traditional_ml_report.txt (86.47%) to see whether real fine-tuning "
          "(vs. frozen embeddings) closes the domain gap.")


if __name__ == "__main__":
    main()
