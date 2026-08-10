# -*- coding: utf-8 -*-
r"""
===================================================================
Deep Learning training - Belt / Brake / Sway (CNN on Log-Mel Spectrograms)
Matches proposal section 5, "Model 2: Deep Learning"
===================================================================
Trains a small Convolutional Neural Network on the log-mel spectrogram
images produced by preprocessing.py.

HOW TO RUN:
    pip install tensorflow
    cd C:\Users\hp\Desktop\car_test
    python train_cnn.py

NOTE: tensorflow is a large package (~500MB) and the first import can
take a minute. This is normal.
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras import callbacks

from cnn_model import build_cnn

BASE_DIR = os.environ.get("CAR_TEST_RAW_DATA_DIR", r"C:\Users\hp\Desktop\car_test")
DATA_DIR = os.environ.get("CAR_TEST_OUTPUT_DIR", os.path.join(BASE_DIR, "processed_data"))

EPOCHS = 40
BATCH_SIZE = 16
RANDOM_STATE = 42

# Adds a small BiGRU after the conv blocks instead of pooling straight to a single
# frequency-and-time-agnostic vector (GlobalAveragePooling2D). "Sway" is consistently
# the hardest class across every experiment in this project (see README) because its
# clicks are intermittent rather than a continuous tone like belt/brake squeal -- a
# model that can see HOW the sound evolves over the 5s clip, not just an averaged
# summary, should have an easier time telling "several sharp clicks" apart from
# "steady noise floor". Set to False to get the original architecture back.
USE_TEMPORAL_HEAD = True

# Focal loss (Lin et al., 2017) reweights each sample by how confidently WRONG the
# model currently is on it, on top of (not instead of) the class_weight="balanced"
# rebalancing already used below -- class_weight only corrects for class FREQUENCY,
# it doesn't make training focus extra hard on individual examples the model keeps
# getting wrong. Set to False to fall back to plain sparse_categorical_crossentropy.
USE_FOCAL_LOSS = True
FOCAL_GAMMA = 2.0

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train_mel.npy"))
    X_val = np.load(os.path.join(DATA_DIR, "X_val_mel.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test_mel.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder


def main():
    print("=" * 70)
    print("Training CNN on Log-Mel Spectrograms")
    print("=" * 70)

    X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = load_data()
    class_names = list(label_encoder.classes_)
    n_classes = len(class_names)

    print(f"Classes: {class_names}")
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

    model = build_cnn(input_shape=X_train.shape[1:], n_classes=n_classes,
                       use_temporal_head=USE_TEMPORAL_HEAD, use_focal_loss=USE_FOCAL_LOSS,
                       focal_gamma=FOCAL_GAMMA)
    model.summary()

    # class weights: the "sway" class is consistently the hardest (lowest recall) across
    # experiments, so we make misclassifying an under-represented class cost more during
    # training -- same idea as class_weight="balanced" already used for SVM/Random Forest.
    class_weight_values = compute_class_weight(
        class_weight="balanced", classes=np.arange(n_classes), y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weight_values)}
    print(f"\nClass weights (balanced): "
          f"{dict(zip(class_names, [round(w, 3) for w in class_weight_values]))}")

    early_stop = callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6
    )

    print("\n[1] Training the CNN...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight_dict,
        verbose=2,
    )

    # ---------- training curves ----------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train loss")
    axes[0].plot(history.history["val_loss"], label="val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"], label="train acc")
    axes[1].plot(history.history["val_accuracy"], label="val acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "cnn_training_curves.png"), dpi=150)
    plt.close()
    print("   saved cnn_training_curves.png")

    # ---------- final evaluation on TEST set ----------
    print("\n[2] Evaluating on the TEST set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

    print(f"   Test accuracy: {acc:.4f}")
    print(f"   Test macro-F1: {f1_macro:.4f}")
    print(f"   Test macro-precision: {precision_macro:.4f}")
    print(f"   Test macro-recall: {recall_macro:.4f}")
    print("\n" + report)

    with open(os.path.join(DATA_DIR, "cnn_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test accuracy: {acc:.4f}\n")
        f.write(f"Test macro-F1: {f1_macro:.4f}\n")
        f.write(f"Test macro-precision: {precision_macro:.4f}\n")
        f.write(f"Test macro-recall: {recall_macro:.4f}\n\n")
        f.write(report)

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names).plot(ax=ax, cmap="Greens", values_format="d")
    ax.set_title("Confusion Matrix (Test) - CNN")
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, "cnn_confusion_matrix.png"), dpi=150)
    plt.close()

    # save model + test-set probabilities (reused by evaluate_ensemble.py)
    model.save(os.path.join(DATA_DIR, "cnn_model.keras"))
    np.save(os.path.join(DATA_DIR, "cnn_test_probs.npy"), y_pred_probs)

    # also save validation-set probabilities -- needed to tune the ensemble fusion weight
    # on data the CNN wasn't evaluated on for its own reported metrics
    val_probs = model.predict(X_val, verbose=0)
    np.save(os.path.join(DATA_DIR, "cnn_val_probs.npy"), val_probs)

    print("\n" + "=" * 70)
    print("DONE. Saved:")
    print("   - cnn_model.keras (trained CNN, ready to use)")
    print("   - cnn_report.txt")
    print("   - cnn_confusion_matrix.png")
    print("   - cnn_training_curves.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
