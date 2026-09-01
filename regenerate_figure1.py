# -*- coding: utf-8 -*-
r"""Figure 3.2 -- UML use-case diagram, redrawn to cover all fourteen use cases
(the eleven original flows plus Ask GarageAI, Check Severity and Manage Account).

  processed_data/report_charts/use_case_diagram.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

matplotlib.rcParams["font.family"] = "DejaVu Sans"
OUT = r"processed_data/report_charts"
os.makedirs(OUT, exist_ok=True)

NAVY = "#16324a"; EDGE = "#3d63a0"; OVAL = "#eef3fb"; GREY = "#666"

GROUPS = [
    ("Diagnosis", [
        "Capture Audio Input", "Record Audio via Microphone", "Upload Audio File",
        "Classify Vehicle Sound", "View Classification Result", "View Prediction History",
        "Ask GarageAI (Chat)", "Check Severity",
    ]),
    ("Account", ["Sign Up", "Log In", "Reset Password", "Manage Account"]),
    ("Settings", ["Change Display Theme"]),
    ("Info", ["View About Page"]),
]

fig, ax = plt.subplots(figsize=(11.5, 9.2), dpi=200)
ax.set_xlim(0, 20); ax.set_ylim(0, 18); ax.axis("off")
ax.set_title("Use Case Diagram - GarageAI", fontsize=17, fontweight="bold",
             color=NAVY, pad=12)

# ---- actor (stick figure) ----
ax_x, ax_y = 1.7, 9.0
ax.add_patch(mp.Circle((ax_x, ax_y + 1.7), 0.42, fill=False, lw=2, ec=NAVY))
ax.plot([ax_x, ax_x], [ax_y + 1.28, ax_y - 0.2], lw=2, color=NAVY)
ax.plot([ax_x - 0.9, ax_x + 0.9], [ax_y + 0.8, ax_y + 0.8], lw=2, color=NAVY)
ax.plot([ax_x, ax_x - 0.8], [ax_y - 0.2, ax_y - 1.3], lw=2, color=NAVY)
ax.plot([ax_x, ax_x + 0.8], [ax_y - 0.2, ax_y - 1.3], lw=2, color=NAVY)
ax.text(ax_x, ax_y - 2.0, "Driver / User", ha="center", va="center",
        fontsize=12, fontweight="bold", color=NAVY)

# ---- system boundary ----
bx0, bx1, by0, by1 = 4.6, 19.4, 0.7, 16.2
ax.add_patch(mp.FancyBboxPatch((bx0, by0), bx1 - bx0, by1 - by0,
             boxstyle="round,pad=0.02", fill=False, lw=1.8, ec=NAVY))
ax.text((bx0 + bx1) / 2, by1 - 0.5, "GarageAI System", ha="center", va="center",
        fontsize=13, fontweight="bold", color=NAVY, style="italic")
ax.plot([bx0, bx1], [by1 - 1.0, by1 - 1.0], lw=1.0, color=NAVY)

# ---- ovals in two columns ----
col_x = [8.6, 15.4]
ow, oh = 6.4, 0.95
left = GROUPS[0][1]
right = sum([g[1] for g in GROUPS[1:]], [])
group_of = {}
for gname, ucs in GROUPS:
    for u in ucs:
        group_of[u] = gname


def place(items, cx, y_top):
    ys = []
    y = y_top
    last_group = None
    for u in items:
        g = group_of[u]
        if g != last_group:
            ax.text(cx - ow / 2 - 0.1, y + 0.72, g, ha="left", va="center",
                    fontsize=10.5, fontweight="bold", color=GREY)
            last_group = g
            y -= 0.55
        ax.add_patch(mp.FancyBboxPatch((cx - ow / 2, y - oh / 2), ow, oh,
                     boxstyle="round,pad=0.02,rounding_size=0.48", lw=1.4,
                     ec=EDGE, fc=OVAL))
        ax.text(cx, y, u, ha="center", va="center", fontsize=10, color=NAVY)
        ys.append((cx - ow / 2, y))
        y -= 1.5
    return ys


anchors = []
anchors += place(left, col_x[0], 14.2)
anchors += place(right, col_x[1], 14.2)

# ---- association lines from the actor ----
for xL, yL in anchors:
    ax.plot([ax_x + 0.95, xL], [ax_y, yL], lw=0.9, color="#8a97a4", zorder=0)

fig.savefig(f"{OUT}/use_case_diagram.png", bbox_inches="tight")
plt.close(fig)
print("use_case_diagram.png written")
