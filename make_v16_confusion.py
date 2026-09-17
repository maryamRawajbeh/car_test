# -*- coding: utf-8 -*-
"""Deck-styled confusion matrix for the deployed XGBoost model (fresh test
extraction -> 93.23%). Matches the v15 slide-14 figure numbers exactly."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#22394B"; FG = "#FFFFFF"
cm = np.array([[43, 0, 2], [0, 44, 0], [7, 0, 37]])
labels = ["Belt", "Brake", "Sway"]
row_pct = cm / cm.sum(1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(5.6, 4.4), facecolor=BG)
ax.set_facecolor(BG)
im = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i,j]}\n({row_pct[i,j]:.0f}%)", ha="center", va="center",
                fontsize=13, fontweight="bold",
                color="white" if row_pct[i, j] > 45 else "#0E1B26")
ax.set_xticks(range(3)); ax.set_yticks(range(3))
ax.set_xticklabels(labels, color=FG, fontsize=11)
ax.set_yticklabels(labels, color=FG, fontsize=11)
ax.set_xlabel("Predicted", color=FG, fontsize=11)
ax.set_ylabel("Actual", color=FG, fontsize=11)
ax.set_title("XGBoost (deployed) — 93.23% test accuracy / macro-F1\n133 held-out recordings   ·   cell = count (row %)",
             color=FG, fontsize=10.5, pad=12)
for s in ax.spines.values():
    s.set_color("#9DB2C0")
ax.tick_params(colors="#C9D6DF")
cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
cbar.ax.tick_params(colors="#C9D6DF", labelsize=8)
fig.tight_layout()
fig.savefig("processed_data/report_charts/v16/xgboost_confusion.png", dpi=200, facecolor=BG)
print("saved processed_data/report_charts/v16/xgboost_confusion.png")
