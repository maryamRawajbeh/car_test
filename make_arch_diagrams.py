# -*- coding: utf-8 -*-
r"""Figure 3.1 (three-tier architecture) and Figure 3.5 (component diagram),
drawn large and legible.

  processed_data/report_charts/architecture_3tier.png   -- 3 tiers left to right
  processed_data/report_charts/component_diagram.png     -- components stacked
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

matplotlib.rcParams["font.family"] = "DejaVu Sans"
OUT = r"processed_data/report_charts"
os.makedirs(OUT, exist_ok=True)

NAVY = "#16324a"; TEAL = "#0f8b82"; SLATE = "#2c5578"; GREY = "#555"


# ======================================================= Figure 3.1
def architecture():
    fig, ax = plt.subplots(figsize=(11.5, 5.0), dpi=200)
    ax.set_xlim(0, 24); ax.set_ylim(0, 11); ax.axis("off")
    ax.set_title("Three-Tier GarageAI System Architecture",
                 fontsize=19, fontweight="bold", color=NAVY, pad=14)

    W, H, Y = 6.6, 3.2, 3.0
    tiers = [
        (0.6, SLATE, "React Frontend", "garageai-frontend", "Vite / React / TS  ·  5173"),
        (8.7, TEAL, "Node.js Gateway", "garageai-backend", "Express / Swagger  ·  5000"),
        (16.8, NAVY, "Python Inference", "garageai-audio-analysis", "FastAPI / Swagger  ·  8001"),
    ]
    for x, c, t1, t2, sub in tiers:
        ax.add_patch(mp.FancyBboxPatch((x, Y), W, H, boxstyle="round,pad=0.05",
                                       linewidth=0, facecolor=c))
        ax.text(x + W / 2, Y + H * 0.68, t1, ha="center", va="center",
                color="white", fontsize=14.5, fontweight="bold")
        ax.text(x + W / 2, Y + H * 0.44, f"({t2})", ha="center", va="center",
                color="white", fontsize=10)
        ax.text(x + W / 2, Y + H * 0.19, sub, ha="center", va="center",
                color="white", fontsize=10)

    for x0 in (7.2, 15.3):
        ax.annotate("", xy=(x0 + 1.5, Y + H / 2), xytext=(x0, Y + H / 2),
                    arrowprops=dict(arrowstyle="<->", color=GREY, lw=2.0, mutation_scale=22))
        ax.text(x0 + 0.75, Y + H + 0.55, "audio file /\nJSON result",
                ha="center", va="bottom", fontsize=10, color="#333")

    ax.annotate("", xy=(0.6, Y + H / 2), xytext=(0.6, Y + H + 1.6),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8, mutation_scale=18))
    ax.text(0.6, Y + H + 1.9, "Microphone recording / file upload", ha="left",
            va="bottom", fontsize=10, style="italic", color="#333")

    ax.text(19.9, Y + H + 1.5,
            "Always-on: Traditional ML, CNN, YAMNet, PANNs,\n"
            "EfficientAT, “other” gate, Ensemble\n"
            "Opt-in: AST, CLAP, PaSST, BEATs",
            ha="center", va="bottom", fontsize=9.5, style="italic", color="#333")

    fig.savefig(f"{OUT}/architecture_3tier.png", bbox_inches="tight")
    plt.close(fig)


# ======================================================= Figure 3.5
def component():
    fig, ax = plt.subplots(figsize=(9.5, 11.5), dpi=200)
    ax.set_xlim(0, 17); ax.set_ylim(0, 25); ax.axis("off")
    ax.set_title("Component Diagram - GarageAI", fontsize=18, fontweight="bold",
                 color=NAVY, pad=12)

    def comp(x, y, w, h, title, body):
        ax.add_patch(mp.Rectangle((x, y), w, h, linewidth=1.7, edgecolor=NAVY,
                                  facecolor="#f3f7fc"))
        for ty in (y + h - 1.0, y + h - 2.0):
            ax.add_patch(mp.Rectangle((x - 0.5, ty), 1.0, 0.6, linewidth=1.5,
                                      edgecolor=NAVY, facecolor="white"))
        ax.text(x + w / 2, y + h - 0.9, title, ha="center", va="center",
                fontsize=13, fontweight="bold", color=NAVY)
        ax.text(x + w / 2, y + h * 0.36, body, ha="center", va="center",
                fontsize=10, color="#333")

    CW = 13.0
    X = 1.6
    comp(X, 19.4, CW, 3.6, "Frontend  (React + Vite)",
         "Home, History, Settings, About,\nLogin / Sign-Up  ·  port 5173")
    comp(X, 13.2, CW, 3.6, "Gateway Backend  (Node.js + Express)",
         "Auth + JWT sessions, rate limiting,\nSQLite persistence  ·  port 5000")
    comp(X, 6.6, CW, 4.2, "Inference Service  (Python + FastAPI)",
         "Traditional ML, CNN, YAMNet, PANNs, EfficientAT,\n"
         "“other” gate, Ensemble  ·  AST/CLAP/PaSST/BEATs (opt-in)\nport 8001")

    def iface(y, label):
        x = X + CW / 2
        ax.annotate("", xy=(x, y - 0.6), xytext=(x, y + 0.6),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.9, mutation_scale=18))
        ax.plot(x, y + 0.08, "o", ms=12, mfc="white", mec=GREY, mew=1.7)
        ax.text(x + 0.7, y, label, ha="left", va="center", fontsize=10, color="#333")

    iface(18.2, "IClassificationAPI  (REST / JSON)")
    iface(12.0, "IInferenceAPI  (REST / multipart)")

    # Trained model artifacts, directly below the inference component
    comp(X + 2.4, 1.4, 8.2, 3.2, "Trained Model Artifacts",
         "car_test/processed_data  (.pkl / .keras / configs)")
    ax.annotate("", xy=(X + CW / 2, 4.6), xytext=(X + CW / 2, 6.6),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8, ls="--",
                                mutation_scale=18))
    ax.text(X + CW / 2 + 0.4, 5.6, "«uses»", ha="left", va="center", fontsize=9.5,
            style="italic", color="#333")

    fig.savefig(f"{OUT}/component_diagram.png", bbox_inches="tight")
    plt.close(fig)


architecture()
component()
print("done ->", OUT)
