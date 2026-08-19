# -*- coding: utf-8 -*-
r"""
===================================================================
Augmentation x seed variance study - AST / CLAP / PaSST / BEATs
===================================================================
The single-run numbers already in ast_report.txt / clap_report.txt /
passt_report.txt / beats_report.txt are each ONE run on ONE randomly
sampled train subset (seed=42), no waveform augmentation. This script
answers two follow-up questions properly instead of trusting one
lucky/unlucky draw:

  1) How much does test accuracy/macro-F1 swing just from WHICH ~200
     train files got sampled into the subset (5 different seeds)?
     (README.md section 6 already flags this as a real risk on a
     133-file test set -- this makes it concrete per model.)
  2) Does the project's own waveform augmentation (time shift / volume
     change / light noise -- see audio_common.make_augmented_version,
     the SAME technique train_transfer_learning.py/YAMNet already
     uses) actually help these frozen-embedding classifier heads, or
     hurt them? (train_panns.py's docstring CLAIMS "same lesson
     learned from YAMNet: augmenting hurts" -- but YAMNet's own
     current code actually DOES use augmentation (N=3 copies/file),
     so that claim was never actually verified for THESE models. This
     is the actual verification.)

EFFICIENCY NOTE: val/test are NEVER subsampled or augmented anywhere
in this project, so their embeddings are IDENTICAL across every
(seed, augment) combination for a given model. This script extracts
val/test embeddings ONCE per model (loading the heavy pretrained
model ONCE too), then only re-extracts TRAIN embeddings for each of
the 10 (5 seeds x 2 augment states) configurations -- instead of
naively re-running each full script 10x per model (40 full runs).

Results are appended incrementally to
processed_data/augmentation_variance_study.csv (one row per config,
flushed immediately) so a crash partway through doesn't lose prior
results -- this script is expected to take a long time.

Does NOT touch best_ast_model.pkl / ast_report.txt / ast_val_probs.npy
etc. (the main comparison-table artifacts already validated end-to-end
via predict.py) -- this is a separate study, separate output file.

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python run_augmentation_variance_study.py [--models ast,clap] [--seeds 1,2,3,4,5]
"""

import os
import sys
import csv
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from audio_common import make_augmented_version

BASE_DIR = r"C:\Users\hp\Desktop\car_test"
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
RESULTS_CSV = os.path.join(DATA_DIR, "augmentation_variance_study.csv")

TRAIN_SUBSET_PER_CLASS = 67  # same as the main comparison-table scripts (~200 total)
N_AUGMENTATIONS_PER_TRAIN_FILE = 3  # same convention as train_transfer_learning.py (YAMNet)
BATCH_SIZE = 8

CLASSIFIER_GRID = {
    "Logistic Regression (C=1)": lambda: LogisticRegression(C=1, max_iter=2000, class_weight="balanced"),
    "Logistic Regression (C=10)": lambda: LogisticRegression(C=10, max_iter=2000, class_weight="balanced"),
    "SVM (linear, C=1)": lambda: SVC(kernel="linear", C=1, probability=True, class_weight="balanced"),
    "SVM (rbf, C=10)": lambda: SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced"),
}


def batched_embed(waveforms, embed_batch_fn, desc):
    """waveforms: list of np.float32 arrays (already loaded+augmented). embed_batch_fn:
    (list_of_waveforms) -> np.array of shape (batch, dim). Generic batching wrapper
    reused for all 4 models -- only embed_batch_fn differs per model."""
    from tqdm import tqdm
    out = []
    for i in tqdm(range(0, len(waveforms), BATCH_SIZE), desc=desc):
        batch = waveforms[i:i + BATCH_SIZE]
        out.append(embed_batch_fn(batch))
    return np.concatenate(out, axis=0)


def build_model_context(model_key):
    """Loads the pretrained model ONCE and returns (loader_fn_for_audio, embed_batch_fn).
    Reuses the already-written, already-tested loader from each train_<model>.py script
    (importing a script only runs its module-level code, not main(), which is guarded)."""
    if model_key == "ast":
        from train_ast import load_ast_model, load_clean_audio_16k
        feature_extractor, model = load_ast_model()

        def embed_batch(batch):
            with torch.no_grad():
                inputs = feature_extractor(batch, sampling_rate=16000, return_tensors="pt")
                return model(**inputs).last_hidden_state.mean(dim=1).numpy()
        return load_clean_audio_16k, embed_batch

    elif model_key == "clap":
        from train_clap import load_clap_model, load_clean_audio_48k
        processor, model = load_clap_model()

        def embed_batch(batch):
            with torch.no_grad():
                inputs = processor(audio=batch, sampling_rate=48000, return_tensors="pt")
                return model.get_audio_features(**inputs).pooler_output.numpy()
        return load_clean_audio_48k, embed_batch

    elif model_key == "passt":
        from train_passt import load_passt_model, load_clean_audio_32k
        from hear21passt.base import get_scene_embeddings
        model = load_passt_model()

        def embed_batch(batch):
            with torch.no_grad():
                audio = torch.tensor(np.stack(batch))
                return get_scene_embeddings(audio, model).numpy()
        return load_clean_audio_32k, embed_batch

    elif model_key == "beats":
        from train_beats import load_beats_model, load_clean_audio_16k
        model = load_beats_model()

        def embed_batch(batch):
            with torch.no_grad():
                audio = torch.tensor(np.stack(batch))
                features, _ = model.extract_features(audio, padding_mask=None)
                return features.mean(dim=1).numpy()
        return load_clean_audio_16k, embed_batch

    else:
        raise ValueError(f"Unknown model_key: {model_key}")


def evaluate_config(load_audio_fn, embed_batch_fn, train_paths, train_labels,
                     X_val, y_val, X_test, y_test, class_names, augment, desc_prefix):
    """Extracts TRAIN embeddings for this (seed, augment) config, trains the classifier
    grid, picks the best on the FIXED val embeddings, evaluates on the FIXED test
    embeddings. Returns a dict of results."""
    waveforms, labels = [], []
    for path, label in zip(train_paths, train_labels):
        y = load_audio_fn(path)
        waveforms.append(y)
        labels.append(label)
        if augment:
            for _ in range(N_AUGMENTATIONS_PER_TRAIN_FILE):
                waveforms.append(make_augmented_version(y))
                labels.append(label)

    X_train = batched_embed(waveforms, embed_batch_fn, f"{desc_prefix} train embeddings (n={len(waveforms)})")
    y_train = np.array(labels)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    best_name, best_f1, best_model = None, -1, None
    for name, builder in CLASSIFIER_GRID.items():
        clf = builder()
        clf.fit(X_train_s, y_train)
        val_pred = clf.predict(X_val_s)
        f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_name, best_f1, best_model = name, f1, clf

    test_pred = best_model.predict(X_test_s)
    return {
        "n_train_used": len(waveforms),
        "best_classifier": best_name,
        "val_macro_f1": best_f1,
        "test_accuracy": accuracy_score(y_test, test_pred),
        "test_macro_f1": f1_score(y_test, test_pred, average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="ast,clap,passt,beats",
                         help="Comma-separated subset of: ast,clap,passt,beats")
    parser.add_argument("--seeds", default="1,2,3,4,5", help="Comma-separated seeds")
    args = parser.parse_args()
    model_keys = args.models.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    import pickle
    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    class_names = list(label_encoder.classes_)

    doc_df = pd.read_csv(os.path.join(DATA_DIR, "dataset_documentation.csv"))
    corrupted_path = os.path.join(DATA_DIR, "corrupted_files.csv")
    if os.path.exists(corrupted_path):
        corrupted_df = pd.read_csv(corrupted_path)
        if "matched_path" in corrupted_df.columns:
            doc_df = doc_df[~doc_df["matched_path"].isin(corrupted_df["matched_path"])].reset_index(drop=True)

    train_df_full = doc_df[doc_df["split_assigned"] == "train"]
    val_df = doc_df[doc_df["split_assigned"] == "val"]
    test_df = doc_df[doc_df["split_assigned"] == "test"]
    y_val = label_encoder.transform(val_df["class"])
    y_test = label_encoder.transform(test_df["class"])

    is_new_file = not os.path.exists(RESULTS_CSV)
    csv_file = open(RESULTS_CSV, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if is_new_file:
        writer.writerow(["model", "seed", "augment", "n_train_used", "best_classifier",
                          "val_macro_f1", "test_accuracy", "test_macro_f1"])

    for model_key in model_keys:
        print("\n" + "=" * 70)
        print(f"MODEL: {model_key}")
        print("=" * 70)
        load_audio_fn, embed_batch_fn = build_model_context(model_key)

        print(f"[{model_key}] Extracting FIXED val/test embeddings (never subsampled/augmented, "
              f"computed once)...")
        X_val = batched_embed([load_audio_fn(p) for p in val_df["matched_path"]],
                               embed_batch_fn, f"{model_key} val embeddings")
        X_test = batched_embed([load_audio_fn(p) for p in test_df["matched_path"]],
                                embed_batch_fn, f"{model_key} test embeddings")

        for seed in seeds:
            train_df = pd.concat([
                g.sample(n=min(len(g), TRAIN_SUBSET_PER_CLASS), random_state=seed)
                for _, g in train_df_full.groupby("class")
            ])
            train_labels = label_encoder.transform(train_df["class"])

            for augment in (False, True):
                desc = f"{model_key} seed={seed} augment={augment}"
                print(f"\n--- {desc} ---")
                res = evaluate_config(
                    load_audio_fn, embed_batch_fn, list(train_df["matched_path"]), train_labels,
                    X_val, y_val, X_test, y_test, class_names, augment, model_key,
                )
                print(f"   -> test_accuracy={res['test_accuracy']:.4f}  test_macro_f1={res['test_macro_f1']:.4f}  "
                      f"(best head: {res['best_classifier']}, n_train_used={res['n_train_used']})")
                writer.writerow([model_key, seed, augment, res["n_train_used"], res["best_classifier"],
                                  f"{res['val_macro_f1']:.4f}", f"{res['test_accuracy']:.4f}",
                                  f"{res['test_macro_f1']:.4f}"])
                csv_file.flush()

    csv_file.close()
    print("\n" + "=" * 70)
    print(f"DONE. Results appended to {RESULTS_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
