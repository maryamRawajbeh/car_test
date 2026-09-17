# -*- coding: utf-8 -*-
"""v16 presentation assets:
 1. Per-class log-mel panels for THREE user-supplied clips (C:\\Users\\hp\\Downloads\\pro\\
    Belt.wav / brake.wav / Sway.wav): the 5 s waveform, the exact 128x216 log-mel
    image (audio_common.extract_log_mel, ref=np.max) and the deployed XGBoost's
    prediction + per-class confidence for that same clip.
 2. A combined 3-panel figure for one slide + one PNG per clip.
 3. The feature-extraction pipeline diagram.
"""
import os
import pickle
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display

from audio_common import (load_clean_audio, compute_shared_spectra,
                          extract_log_mel, extract_mfcc_vector)

warnings.filterwarnings("ignore")
DATA = "processed_data"
OUT = "processed_data/report_charts/v16"
os.makedirs(OUT, exist_ok=True)

BG = "#22394B"
FG = "#FFFFFF"
ACC = "#FF3F4A"
TEAL = "#16A89F"

config = pickle.load(open(os.path.join(DATA, "config.pkl"), "rb"))
le = pickle.load(open(os.path.join(DATA, "label_encoder.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(DATA, "scaler.pkl"), "rb"))
xgb = pickle.load(open(os.path.join(DATA, "best_traditional_model.pkl"), "rb"))["model"]
classes = list(le.classes_)                      # ['belt','brake','sway']
SR = config["target_sr"]
DUR = config["target_duration"]

CLIPS = [
    ("belt",  r"C:\Users\hp\Downloads\pro\Belt.wav"),
    ("brake", r"C:\Users\hp\Downloads\pro\brake.wav"),
    ("sway",  r"C:\Users\hp\Downloads\pro\Sway.wav"),
]
TITLES = {"belt": "BELT", "brake": "BRAKE", "sway": "SWAY"}
SUBT = {"belt": "drive-belt squeal",
        "brake": "brake noise (screech / grind)",
        "sway": "suspension / sway-bar clunk"}


def analyse(path):
    y, sr = load_clean_audio(path, SR, DUR, fallback_to_untrimmed=True)
    shared = compute_shared_spectra(y, sr, n_fft=2048, hop_length=512, n_mels=128)
    logmel = extract_log_mel(y, sr, shared=shared)                 # (128, T) dB
    feat = extract_mfcc_vector(y, sr, config["n_mfcc"]).reshape(1, -1)
    prob = xgb.predict_proba(scaler.transform(feat))[0]            # [belt,brake,sway]
    return y, sr, logmel, prob


def style_ax(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#9DB2C0")
    ax.tick_params(colors="#C9D6DF", labelsize=8)


def result_box(ax, prob):
    """Small 3-bar confidence readout drawn inside the log-mel axes."""
    pred = classes[int(prob.argmax())]
    txt = "model -> %s   (%s)" % (
        pred.upper(),
        "  ".join("%s %.0f%%" % (c, 100 * p) for c, p in zip(classes, prob)))
    ax.text(0.5, -0.42, txt, transform=ax.transAxes, ha="center", va="top",
            color=FG, fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35",
                      fc="#0E1B26",
                      ec=(TEAL if pred == cname_current[0] else ACC), lw=1.4))


cache = {}
for cname, path in CLIPS:
    cache[cname] = analyse(path)
    _, _, _, pr = cache[cname]
    print("%-6s %-12s -> model: %-5s  belt %.2f  brake %.2f  sway %.2f"
          % (cname, os.path.basename(path), classes[int(pr.argmax())], *pr))

# ---------- 1 + 2 : combined 3-column figure ----------
fig, axes = plt.subplots(2, 3, figsize=(13.8, 5.6), facecolor=BG,
                         gridspec_kw={"height_ratios": [1, 1.7],
                                      "hspace": 0.5, "wspace": 0.34})
for col, (cname, path) in enumerate(CLIPS):
    cname_current = (cname,)
    y, sr, logmel, prob = cache[cname]

    axw = axes[0, col]
    t = np.linspace(0, len(y) / sr, len(y))
    axw.plot(t, y, color=ACC, lw=0.5)
    axw.set_xlim(0, DUR)
    axw.set_ylim(-1.05, 1.05)
    axw.set_title(TITLES[cname] + "\n" + SUBT[cname], color=FG, fontsize=10,
                  fontweight="bold", pad=6)
    style_ax(axw)
    if col == 0:
        axw.set_ylabel("amplitude", color="#C9D6DF", fontsize=8)

    axm = axes[1, col]
    im = librosa.display.specshow(logmel, sr=sr, hop_length=512, x_axis="time",
                                  y_axis="mel", ax=axm, cmap="magma")
    axm.set_title("128 x 216 log-mel  (CNN input)", color="#C9D6DF", fontsize=8.5, pad=4)
    style_ax(axm)
    axm.set_xlabel("time (s)", color="#C9D6DF", fontsize=8)
    axm.set_ylabel("mel frequency (Hz)" if col == 0 else "", color="#C9D6DF", fontsize=8)
    axm.text(0.98, 0.06, os.path.basename(path), transform=axm.transAxes,
             ha="right", va="bottom", color=FG, fontsize=7,
             bbox=dict(boxstyle="round,pad=0.3", fc="#0E1B26", ec="none", alpha=0.75))

cbar = fig.colorbar(im, ax=axes[1, :].tolist(), fraction=0.02, pad=0.01)
cbar.ax.tick_params(colors="#C9D6DF", labelsize=7)
cbar.set_label("dB (ref = max)", color="#C9D6DF", fontsize=7)
fig.savefig(os.path.join(OUT, "logmel_three_classes.png"), dpi=200,
            facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("saved logmel_three_classes.png")

# ---------- one PNG per clip ----------
for cname, path in CLIPS:
    cname_current = (cname,)
    y, sr, logmel, prob = cache[cname]
    f2, (a1, a2) = plt.subplots(2, 1, figsize=(5.0, 4.6), facecolor=BG,
                                gridspec_kw={"height_ratios": [1, 1.8], "hspace": 0.75})
    t = np.linspace(0, len(y) / sr, len(y))
    a1.plot(t, y, color=ACC, lw=0.5)
    a1.set_xlim(0, DUR)
    a1.set_ylim(-1.05, 1.05)
    a1.set_title(TITLES[cname] + " - " + SUBT[cname], color=FG, fontsize=10, fontweight="bold")
    style_ax(a1)
    im = librosa.display.specshow(logmel, sr=sr, hop_length=512, x_axis="time",
                                  y_axis="mel", ax=a2, cmap="magma")
    style_ax(a2)
    a2.set_xlabel("time (s)", color="#C9D6DF", fontsize=8)
    a2.set_ylabel("mel frequency (Hz)", color="#C9D6DF", fontsize=8)
    a2.text(0.98, 0.06, os.path.basename(path), transform=a2.transAxes,
            ha="right", va="bottom", color=FG, fontsize=7,
            bbox=dict(boxstyle="round,pad=0.3", fc="#0E1B26", ec="none", alpha=0.75))
    f2.savefig(os.path.join(OUT, "logmel_%s.png" % cname), dpi=200,
               facecolor=BG, bbox_inches="tight")
    plt.close(f2)
print("saved per-clip panels")

# ---------- 3 : feature-extraction pipeline diagram ----------
fig, ax = plt.subplots(figsize=(12.4, 3.0), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 100)
ax.set_ylim(0, 30)
ax.margins(0)
ax.axis("off")


def box(x, y, w, h, text, fc="#2E4A5F", ec="#9DB2C0", tc=FG, fs=9.5):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.5, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
            fontsize=fs, zorder=3)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=2.2), zorder=1)


MID = 15
box(0.5, MID - 5, 16, 10, "5 s clip\nmono - 22 050 Hz\ntrim - peak-norm", fc="#1F3140", fs=9)
box(21, MID - 5, 15.5, 10, "STFT\nn_fft 2048\nhop 512", fs=9)
box(41.5, MID - 5, 16, 10, "Mel filterbank\n128 bands\n-> power", fs=9)
arrow(16.8, MID, 20.7, MID)
arrow(36.8, MID, 41.2, MID)

box(63.5, 14.5, 36, 14.5,
    "MFCC path   ->   269-feature vector\n"
    "log-power -> DCT -> 40 MFCC ;  + 40 delta  + 40 delta-delta\n"
    "    each summarised by mean & std over 5 s ........  240\n"
    "spectral contrast, 7 bands x (mean, std) ..........   14\n"
    "ZCR, centroid, bandwidth, rolloff, RMS (mean,std) ..  10\n"
    "HPSS: harmonic & percussive energy (mean,std) + ratio   5\n"
    "->  SVM - RF - XGBoost",
    fc="#10212B", ec=TEAL, fs=7.6)
box(64, 1.5, 35.5, 11,
    "Log-mel path   ->   128 x 216 image\n"
    "log-power, ref = max  (dB), then z-scored\nwith the train-set mean / std\n"
    "kept as the whole time-frequency picture\n->  CNN",
    fc="#10212B", ec=ACC, fs=8.2)
arrow(57.8, MID, 63.7, 22)
arrow(57.8, MID, 63.7, 7.5)
fig.savefig(os.path.join(OUT, "feature_pipeline.png"), dpi=200, facecolor=BG,
            bbox_inches="tight", pad_inches=0.05)
plt.close(fig)
print("saved feature_pipeline.png")
print("\nALL DONE ->", OUT)
