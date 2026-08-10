# -*- coding: utf-8 -*-
r"""
===================================================================
Data visualization - waveforms, spectrograms, and dataset statistics
Matches proposal Week 1 deliverable: "Visualize waveforms and spectrograms."
===================================================================
Picks sample recordings per class (belt / brake / sway) straight from
dataset_documentation.csv (produced by preprocessing.py) and produces a full
set of exploratory-data-analysis figures:

    Per-file plots
        - raw waveform + log-mel spectrogram (a few examples per class)

    Dataset-level plots
        - class distribution (bar chart)
        - train / validation / test split distribution (if a split column exists)
        - recording duration histogram (+ per-class boxplot)
        - average log-mel spectrogram per class (instead of a single file)
        - average MFCC heatmap per class
        - PCA projection of audio features, colored by class
        - t-SNE projection of audio features, colored by class (if sklearn available)
        - feature correlation heatmap
        - spectral centroid boxplot per class
        - RMS energy boxplot per class
        - zero crossing rate boxplot per class
        - spectral bandwidth boxplot per class (bonus)

This is a documentation/reporting tool only -- it does not feed into training.
Run it AFTER preprocessing.py (it needs dataset_documentation.csv).

HOW TO RUN:
    cd C:\Users\hp\Desktop\car_test
    python visualize_data.py
"""

import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

BASE_DIR = r"C:\Users\hp\Desktop\car_test"
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
OUT_DIR = os.path.join(DATA_DIR, "visualizations")

TARGET_SR = 22050
N_MELS = 128
N_MFCC = 13
HOP_LENGTH = 512
FIXED_DURATION_SEC = 5.0          # used only to align spectrograms/MFCCs for averaging
SAMPLES_PER_CLASS = 3             # example waveform/spectrogram plots per class
FEATURE_SAMPLES_PER_CLASS = 40    # how many files per class to use for stats/PCA/t-SNE
MAX_FILES_FOR_DURATION = 500      # cap for the duration histogram (speed)
RANDOM_STATE = 42

sns.set_theme(style="whitegrid")
CLASS_PALETTE = "Set2"


# ---------------------------------------------------------------------------
# Audio loading helpers
# ---------------------------------------------------------------------------

def load_raw_for_plot(path, sr=TARGET_SR, top_db=25):
    """Same cleaning as preprocessing.py (mono, resample, trim, normalize) but
    WITHOUT forcing a fixed 5-second length, so the plotted waveform shows the
    recording's natural shape."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    return y, sr


def load_fixed_for_stats(path, sr=TARGET_SR, top_db=25, duration=FIXED_DURATION_SEC):
    """Load + trim + normalize, then pad/truncate to a fixed duration so that
    spectrograms/MFCCs from different files can be averaged or stacked into a
    feature matrix of consistent shape."""
    y, sr = librosa.load(path, sr=sr, mono=True)
    y, _ = librosa.effects.trim(y, top_db=top_db)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    target_len = int(duration * sr)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y, sr


def get_duration_safe(path):
    try:
        return librosa.get_duration(path=path)
    except Exception:
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            return len(y) / sr
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Per-file plots (waveform + spectrogram)
# ---------------------------------------------------------------------------

def plot_one_file(path, class_name, out_path):
    y, sr = load_raw_for_plot(path)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6))

    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title(f"Waveform - {class_name} - {os.path.basename(path)}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")

    img = librosa.display.specshow(log_mel, sr=sr, hop_length=HOP_LENGTH,
                                    x_axis="time", y_axis="mel", ax=axes[1], cmap="magma")
    axes[1].set_title(f"Log-Mel Spectrogram - {class_name}")
    fig.colorbar(img, ax=axes[1], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 1) Class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(doc_df, out_path):
    counts = doc_df["class"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, palette=CLASS_PALETTE)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, str(v), ha="center", fontweight="bold")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Recordings")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 2) Train / Validation / Test distribution
# ---------------------------------------------------------------------------

def find_split_column(doc_df):
    """Looks for a common split-column name; returns None if not present."""
    candidates = ["split_assigned", "split", "set", "subset", "dataset_split", "train_test_split"]
    for c in candidates:
        if c in doc_df.columns:
            return c
    return None


def plot_split_distribution(doc_df, out_path):
    split_col = find_split_column(doc_df)
    if split_col is None:
        print("   !! No train/val/test split column found in dataset_documentation.csv - skipping this plot.")
        return

    cross = pd.crosstab(doc_df["class"], doc_df[split_col])
    cross.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="viridis")
    plt.title("Train / Validation / Test Distribution per Class")
    plt.xlabel("Class")
    plt.ylabel("Number of Recordings")
    plt.legend(title=split_col)
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 3) Recording duration histogram (+ per-class boxplot)
# ---------------------------------------------------------------------------

def compute_durations(doc_df, max_files=MAX_FILES_FOR_DURATION, rng=None):
    rng = rng or random.Random(RANDOM_STATE)
    rows = doc_df.to_dict("records")
    if len(rows) > max_files:
        rows = rng.sample(rows, max_files)

    records = []
    for row in rows:
        path = row["matched_path"]
        if not os.path.exists(path):
            continue
        dur = get_duration_safe(path)
        if dur is not None:
            records.append({"class": row["class"], "duration": dur})
    return pd.DataFrame(records)


def plot_duration_histogram(dur_df, out_path):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=dur_df, x="duration", hue="class", multiple="stack",
                 palette=CLASS_PALETTE, bins=30)
    plt.title("Recording Duration Distribution")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_duration_boxplot(dur_df, out_path):
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=dur_df, x="class", y="duration", palette=CLASS_PALETTE)
    plt.title("Recording Duration per Class")
    plt.xlabel("Class")
    plt.ylabel("Duration (seconds)")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 4) Average spectrogram per class (instead of just the first file)
# ---------------------------------------------------------------------------

def plot_average_spectrogram_per_class(doc_df, out_path, samples_per_class=FEATURE_SAMPLES_PER_CLASS, rng=None):
    rng = rng or random.Random(RANDOM_STATE)
    classes = sorted(doc_df["class"].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        cls_paths = list(doc_df[doc_df["class"] == cls]["matched_path"])
        rng.shuffle(cls_paths)
        cls_paths = [p for p in cls_paths if os.path.exists(p)][:samples_per_class]

        mel_stack = []
        sr_used = TARGET_SR
        for path in cls_paths:
            y, sr = load_fixed_for_stats(path)
            sr_used = sr
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
            log_mel = librosa.power_to_db(mel, ref=np.max)
            mel_stack.append(log_mel)

        if not mel_stack:
            ax.set_title(f"{cls.upper()} (no files found)")
            continue

        avg_mel = np.mean(np.stack(mel_stack), axis=0)
        img = librosa.display.specshow(avg_mel, sr=sr_used, hop_length=HOP_LENGTH,
                                        x_axis="time", y_axis="mel", ax=ax, cmap="magma")
        ax.set_title(f"{cls.upper()} - Average of {len(mel_stack)} files")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 5) Average MFCC heatmap per class
# ---------------------------------------------------------------------------

def plot_average_mfcc_per_class(doc_df, out_path, samples_per_class=FEATURE_SAMPLES_PER_CLASS, rng=None):
    rng = rng or random.Random(RANDOM_STATE)
    classes = sorted(doc_df["class"].unique())
    fig, axes = plt.subplots(1, len(classes), figsize=(5 * len(classes), 4))
    if len(classes) == 1:
        axes = [axes]

    for ax, cls in zip(axes, classes):
        cls_paths = list(doc_df[doc_df["class"] == cls]["matched_path"])
        rng.shuffle(cls_paths)
        cls_paths = [p for p in cls_paths if os.path.exists(p)][:samples_per_class]

        mfcc_stack = []
        sr_used = TARGET_SR
        for path in cls_paths:
            y, sr = load_fixed_for_stats(path)
            sr_used = sr
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
            mfcc_stack.append(mfcc)

        if not mfcc_stack:
            ax.set_title(f"{cls.upper()} (no files found)")
            continue

        avg_mfcc = np.mean(np.stack(mfcc_stack), axis=0)
        img = librosa.display.specshow(avg_mfcc, sr=sr_used, hop_length=HOP_LENGTH,
                                        x_axis="time", ax=ax, cmap="coolwarm")
        ax.set_ylabel("MFCC coefficient")
        ax.set_title(f"{cls.upper()} - Average MFCC ({len(mfcc_stack)} files)")
        fig.colorbar(img, ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Feature extraction shared by boxplots / correlation / PCA / t-SNE
# ---------------------------------------------------------------------------

def extract_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)
    zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=HOP_LENGTH)

    feats = {}
    for i in range(N_MFCC):
        feats[f"mfcc_{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc_{i+1}_std"] = float(np.std(mfcc[i]))
    feats["spectral_centroid_mean"] = float(np.mean(centroid))
    feats["spectral_bandwidth_mean"] = float(np.mean(bandwidth))
    feats["rms_mean"] = float(np.mean(rms))
    feats["zcr_mean"] = float(np.mean(zcr))
    return feats


def build_feature_dataframe(doc_df, samples_per_class=FEATURE_SAMPLES_PER_CLASS, rng=None):
    rng = rng or random.Random(RANDOM_STATE)
    rows = []

    for cls in sorted(doc_df["class"].unique()):
        cls_paths = list(doc_df[doc_df["class"] == cls]["matched_path"])
        rng.shuffle(cls_paths)
        cls_paths = [p for p in cls_paths if os.path.exists(p)][:samples_per_class]

        for path in cls_paths:
            try:
                y, sr = load_raw_for_plot(path)
                if len(y) == 0:
                    continue
                feats = extract_features(y, sr)
                feats["class"] = cls
                feats["file"] = os.path.basename(path)
                rows.append(feats)
            except Exception as e:
                print(f"   !! skipping {path} (feature extraction failed: {e})")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6) Boxplots: spectral centroid / RMS / ZCR / bandwidth per class
# ---------------------------------------------------------------------------

def plot_feature_boxplot(features_df, column, title, ylabel, out_path):
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=features_df, x="class", y=column, palette=CLASS_PALETTE)
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 7) Feature correlation heatmap
# ---------------------------------------------------------------------------

def plot_feature_correlation_heatmap(features_df, out_path):
    numeric_cols = [c for c in features_df.columns if c not in ("class", "file")]
    corr = features_df[numeric_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True,
                cbar_kws={"shrink": 0.7})
    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 8) PCA projection
# ---------------------------------------------------------------------------

def plot_pca(features_df, out_path):
    if not SKLEARN_AVAILABLE:
        print("   !! scikit-learn not installed - skipping PCA plot.")
        return

    numeric_cols = [c for c in features_df.columns if c not in ("class", "file")]
    X = features_df[numeric_cols].values
    y = features_df["class"].values

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette=CLASS_PALETTE, s=60)
    var1, var2 = pca.explained_variance_ratio_[:2] * 100
    plt.title("PCA Projection of Audio Features")
    plt.xlabel(f"PC1 ({var1:.1f}% variance)")
    plt.ylabel(f"PC2 ({var2:.1f}% variance)")
    plt.legend(title="Class")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# 9) t-SNE projection
# ---------------------------------------------------------------------------

def plot_tsne(features_df, out_path):
    if not TSNE_AVAILABLE:
        print("   !! scikit-learn (TSNE) not installed - skipping t-SNE plot.")
        return

    numeric_cols = [c for c in features_df.columns if c not in ("class", "file")]
    X = features_df[numeric_cols].values
    y = features_df["class"].values

    n_samples = X.shape[0]
    if n_samples < 5:
        print("   !! Not enough samples for t-SNE - skipping.")
        return

    X_scaled = StandardScaler().fit_transform(X)
    perplexity = min(30, max(5, n_samples // 4))
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=perplexity, init="pca")
    X_tsne = tsne.fit_transform(X_scaled)

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=CLASS_PALETTE, s=60)
    plt.title("t-SNE Projection of Audio Features")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Class")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Visualizing waveforms, spectrograms, and dataset statistics")
    print("=" * 70)

    doc_path = os.path.join(DATA_DIR, "dataset_documentation.csv")
    if not os.path.exists(doc_path):
        print(f"\n!! {doc_path} not found. Run preprocessing.py first.")
        return

    doc_df = pd.read_csv(doc_path)
    if "matched_path" not in doc_df.columns or "class" not in doc_df.columns:
        print("\n!! dataset_documentation.csv is missing required columns (matched_path, class).")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(RANDOM_STATE)

    # ---- Per-file waveform + spectrogram examples ----
    for cls in sorted(doc_df["class"].unique()):
        cls_df = doc_df[doc_df["class"] == cls]
        paths = list(cls_df["matched_path"])
        rng.shuffle(paths)
        chosen = paths[:SAMPLES_PER_CLASS]

        print(f"\n[{cls}] plotting {len(chosen)} example(s)...")
        for i, path in enumerate(chosen, 1):
            if not os.path.exists(path):
                print(f"   skipping missing file: {path}")
                continue
            out_path = os.path.join(OUT_DIR, f"{cls}_example_{i}.png")
            plot_one_file(path, cls, out_path)
            print(f"   saved {out_path}")

    # ---- 1) Class distribution ----
    print("\n[dataset] plotting class distribution...")
    out_path = os.path.join(OUT_DIR, "class_distribution.png")
    plot_class_distribution(doc_df, out_path)
    print(f"   saved {out_path}")

    # ---- 2) Train / val / test distribution ----
    print("\n[dataset] plotting train/val/test split distribution...")
    out_path = os.path.join(OUT_DIR, "split_distribution.png")
    plot_split_distribution(doc_df, out_path)

    # ---- 3) Duration histogram + boxplot ----
    print("\n[dataset] computing durations (this reads audio headers/files)...")
    dur_df = compute_durations(doc_df, rng=random.Random(RANDOM_STATE))
    if not dur_df.empty:
        out_path = os.path.join(OUT_DIR, "duration_histogram.png")
        plot_duration_histogram(dur_df, out_path)
        print(f"   saved {out_path}")

        out_path = os.path.join(OUT_DIR, "duration_boxplot_per_class.png")
        plot_duration_boxplot(dur_df, out_path)
        print(f"   saved {out_path}")
    else:
        print("   !! Could not compute any durations - skipping duration plots.")

    # ---- 4) Average spectrogram per class ----
    print("\n[dataset] plotting average spectrogram per class...")
    out_path = os.path.join(OUT_DIR, "class_overview_spectrograms.png")
    plot_average_spectrogram_per_class(doc_df, out_path, rng=random.Random(RANDOM_STATE))
    print(f"   saved {out_path}")

    # ---- 5) Average MFCC heatmap per class ----
    print("\n[dataset] plotting average MFCC heatmap per class...")
    out_path = os.path.join(OUT_DIR, "class_overview_mfcc.png")
    plot_average_mfcc_per_class(doc_df, out_path, rng=random.Random(RANDOM_STATE))
    print(f"   saved {out_path}")

    # ---- Build shared feature table for boxplots / correlation / PCA / t-SNE ----
    print(f"\n[dataset] extracting audio features (~{FEATURE_SAMPLES_PER_CLASS} files/class) for "
          f"boxplots, correlation, PCA and t-SNE...")
    features_df = build_feature_dataframe(doc_df, rng=random.Random(RANDOM_STATE))

    if features_df.empty:
        print("   !! No features could be extracted - skipping feature-based plots.")
    else:
        features_csv = os.path.join(OUT_DIR, "extracted_features.csv")
        features_df.to_csv(features_csv, index=False)
        print(f"   saved feature table -> {features_csv}")

        # ---- 6) Spectral centroid / RMS / ZCR / bandwidth boxplots ----
        print("\n[dataset] plotting spectral centroid boxplot...")
        out_path = os.path.join(OUT_DIR, "spectral_centroid_boxplot.png")
        plot_feature_boxplot(features_df, "spectral_centroid_mean",
                              "Spectral Centroid per Class", "Spectral Centroid (Hz)", out_path)
        print(f"   saved {out_path}")

        print("\n[dataset] plotting RMS energy boxplot...")
        out_path = os.path.join(OUT_DIR, "rms_energy_boxplot.png")
        plot_feature_boxplot(features_df, "rms_mean",
                              "RMS Energy per Class", "RMS Energy", out_path)
        print(f"   saved {out_path}")

        print("\n[dataset] plotting zero crossing rate boxplot...")
        out_path = os.path.join(OUT_DIR, "zcr_boxplot.png")
        plot_feature_boxplot(features_df, "zcr_mean",
                              "Zero Crossing Rate per Class", "Zero Crossing Rate", out_path)
        print(f"   saved {out_path}")

        print("\n[dataset] plotting spectral bandwidth boxplot (bonus)...")
        out_path = os.path.join(OUT_DIR, "spectral_bandwidth_boxplot.png")
        plot_feature_boxplot(features_df, "spectral_bandwidth_mean",
                              "Spectral Bandwidth per Class", "Spectral Bandwidth (Hz)", out_path)
        print(f"   saved {out_path}")

        # ---- 7) Feature correlation heatmap ----
        print("\n[dataset] plotting feature correlation heatmap...")
        out_path = os.path.join(OUT_DIR, "feature_correlation_heatmap.png")
        plot_feature_correlation_heatmap(features_df, out_path)
        print(f"   saved {out_path}")

        # ---- 8) PCA ----
        print("\n[dataset] plotting PCA projection...")
        out_path = os.path.join(OUT_DIR, "pca_projection.png")
        plot_pca(features_df, out_path)
        print(f"   saved {out_path}")

        # ---- 9) t-SNE ----
        print("\n[dataset] plotting t-SNE projection...")
        out_path = os.path.join(OUT_DIR, "tsne_projection.png")
        plot_tsne(features_df, out_path)
        print(f"   saved {out_path}")

    print("\n" + "=" * 70)
    print(f"DONE. All figures saved in: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
