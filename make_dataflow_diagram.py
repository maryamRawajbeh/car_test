# -*- coding: utf-8 -*-
r"""Data-flow diagram: the audio path from capture to result (documentation ch.3).

Clean 'snake' layout -- row 1 left-to-right, a straight vertical drop into row 2,
row 2 right-to-left -- so no arrow ever crosses a box.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

NAVY = "#14304a"; TEAL = "#17a398"; EDGE = "#4a72b0"; ARROW = "#555"
fig, ax = plt.subplots(figsize=(12, 6.4), dpi=200)
ax.set_xlim(0, 24); ax.set_ylim(0, 12); ax.axis("off")

BW, BH = 3.7, 2.0


def box(cx, cy, text, w=BW, h=BH, fc=TEAL, tc="white", fs=8.6):
    ax.add_patch(mp.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                   boxstyle="round,pad=0.04", linewidth=1.3,
                                   edgecolor=EDGE, facecolor=fc))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc)


def store(cx, cy, text, w=4.6, h=1.5):
    ax.add_patch(mp.Rectangle((cx - w / 2, cy - h / 2), w, h, linewidth=1.3,
                              edgecolor=EDGE, facecolor="#eef3fb"))
    ax.plot([cx - w / 2, cx + w / 2], [cy + h / 2, cy + h / 2], color=EDGE, lw=1.3)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.4, color=NAVY)


def arrow(x0, y0, x1, y1, label=None, dx=0.0, dy=0.5, va="bottom"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=1.6))
    if label:
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, ha="center", va=va,
                fontsize=7.8, style="italic", color="#333")


TOP, BOT = 9.0, 3.4
xs = [2.6, 7.3, 12.0, 16.7, 21.0]           # row-1 columns
# ---- row 1 : left -> right ----
box(xs[0], TOP, "User\n(browser)", fc="white", tc=NAVY)
box(xs[1], TOP, "1  Capture audio\n(record / upload)")
box(xs[2], TOP, "2  Convert to WAV\n(ffmpeg)")
box(xs[3], TOP, "3  Extract features\n(MFCC / log-mel /\nembeddings)", fs=8.0)
box(xs[4], TOP, "4  “Other” gate\n(reject non-fault\naudio)", fs=8.2)
for i, lbl in enumerate(["audio", "", "WAV", "features"]):
    arrow(xs[i] + BW / 2, TOP, xs[i + 1] - BW / 2, TOP, lbl)

# ---- vertical drop 4 -> 5 ----
box(xs[4], BOT, "5  Run models\n(Traditional ML, CNN,\nYAMNet, PANNs, EfficientAT)", w=4.6, fs=7.6)
arrow(xs[4], TOP - BH / 2, xs[4], BOT + BH / 2 + 0.25, "not “other”", dx=1.5, dy=0.0)

# ---- row 2 : right -> left ----
box(xs[2], BOT, "6  Weighted fusion\n(0.30 / 0.60 / 0.10)", w=4.2, fs=8.4)
box(xs[0] + 0.4, BOT, "7  Return result\n(class + confidence,\nper-model breakdown)", w=4.4, fs=7.8)
arrow(xs[4] - 2.3, BOT, xs[2] + 2.1, BOT, "probabilities")
arrow(xs[2] - 2.1, BOT, xs[0] + 0.4 + 2.2, BOT, "fused class")

# ---- data stores ----
store(xs[4], 0.95, "D2  Trained model artifacts")
arrow(xs[4], 0.95 + 0.75, xs[4], BOT - BH / 2, "load", dx=1.1, dy=0.0)
store(xs[0] + 0.4, 0.95, "D1  Prediction history (SQLite)")
arrow(xs[0] + 0.4, BOT - BH / 2, xs[0] + 0.4, 0.95 + 0.75, "save / list", dx=1.4, dy=0.0)

ax.text(12, 11.3, "GarageAI — Data Flow: audio path from capture to result",
        ha="center", fontsize=13, fontweight="bold", color=NAVY)
fig.savefig("processed_data/report_charts/data_flow_diagram.png", bbox_inches="tight")
print("ok")
