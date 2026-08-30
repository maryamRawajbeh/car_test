# -*- coding: utf-8 -*-
"""Regenerates Figure 3.4 (sequence diagram, fixes '264 features' -> '269') and
Figure 3.5 (component diagram, fixes the stale 'XGBoost | YAMNet | PANNs' model
list inside the Inference Service box), matching each original's layout/notation
as closely as practical, then swaps the bytes into the docx in place."""
import zipfile
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DOCX = r"C:\Users\hp\Downloads\GarageAI_Documentation.docx"

# ============================== Figure 3.4 (sequence) ==============================
MEDIA_34 = "word/media/64d0c1fcab949e7644ea8a4206be1b00b822651e.png"
OUT_34 = "fig34_regenerated.png"

lifelines = ["User", ":HomePage (React\nFrontend)", ":APIGateway (Node.js /\nExpress)",
             ":InferenceService\n(FastAPI)", ":EnsembleModel (ML\nPipeline)"]
xs = [0.7, 2.7, 4.9, 7.1, 9.3]

fig, ax = plt.subplots(figsize=(11, 8), dpi=200)
ax.set_xlim(0, 10.4)
ax.set_ylim(0, 12)
ax.axis("off")

for x, label in zip(xs, lifelines):
    ax.add_patch(mpatches.FancyBboxPatch((x - 0.85, 11.1), 1.7, 0.75, boxstyle="round,pad=0.03",
                                          linewidth=1, edgecolor="#4a72b0", facecolor="#dbe7f7"))
    ax.text(x, 11.475, label, ha="center", va="center", fontsize=8.5)
    ax.plot([x, x], [0.3, 11.1], color="#888", lw=1, ls=(0, (4, 3)))

def arrow(x0, x1, y, text, dashed=False, self_call=False):
    style = dict(arrowstyle="-|>", color="black", lw=1.3,
                 linestyle="dashed" if dashed else "solid")
    if self_call:
        ax.annotate("", xy=(x0 + 0.55, y - 0.35), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2,
                                     connectionstyle="arc3,rad=-1.3"))
        ax.text(x0 + 0.65, y - 0.15, text, fontsize=8, va="center")
    else:
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=style)
        ax.text((x0 + x1) / 2, y + 0.18, text, ha="center", fontsize=8)

y = 10.5
arrow(xs[0], xs[1], y, "recordAudio() / uploadFile()"); y -= 0.85
arrow(xs[1], xs[1], y, "captureAudioBlob()", self_call=True); y -= 0.85
arrow(xs[1], xs[2], y, "POST /api/classify (audio file)"); y -= 0.85
arrow(xs[2], xs[3], y, "forward request (multipart/form-data)"); y -= 0.85
arrow(xs[3], xs[3], y, "convertToWav() [ffmpeg]", self_call=True); y -= 0.85
arrow(xs[3], xs[3], y, "extractFeatures() [269 features]", self_call=True); y -= 0.85
arrow(xs[3], xs[4], y, "predict(features)"); y -= 0.85
arrow(xs[4], xs[3], y, "predictedClass, confidence, modelBreakdown", dashed=True); y -= 0.85
arrow(xs[3], xs[2], y, "JSON response", dashed=True); y -= 0.85
arrow(xs[2], xs[1], y, "relay JSON response", dashed=True); y -= 0.85
arrow(xs[1], xs[0], y, "display result (class + confidence)", dashed=True); y -= 0.6

for x in xs:
    ax.annotate("", xy=(x, y - 0.15), xytext=(x, y + 0.15),
                arrowprops=dict(arrowstyle="-|>", color="#888", lw=1))

plt.tight_layout()
plt.savefig(OUT_34, dpi=200, facecolor="white")
plt.close()
print(f"Regenerated: {OUT_34}")

# ============================== Figure 3.5 (component) ==============================
MEDIA_35 = "word/media/c17b66ae6716047c7616a1272019d6dcaa465654.png"
OUT_35 = "fig35_regenerated.png"

fig, ax = plt.subplots(figsize=(15, 4.2), dpi=200)
ax.set_xlim(0, 15.6)
ax.set_ylim(0, 4.2)
ax.axis("off")


def component_box(x, y, w, h, title, sub=""):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                          linewidth=1.3, edgecolor="black", facecolor="white"))
    for dy in (0.72, 0.55):
        ax.add_patch(mpatches.Rectangle((x - 0.06, y + h - dy), 0.14, 0.09,
                                         linewidth=1, edgecolor="black", facecolor="white"))
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="top", fontsize=10.5, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + 0.28, sub, ha="center", va="center", fontsize=7.5)


component_box(0.2, 1.5, 2.6, 1.6, "Frontend (React +\nVite, TS, Tailwind)",
              "Home, History, Settings,\nAbout, Login/Sign-Up")
component_box(5.9, 1.5, 3.0, 1.6, "Gateway Backend\n(Node.js + Express)")
component_box(11.2, 1.5, 3.9, 1.6,
              "Inference Service\n(Python + FastAPI)",
              "Traditional ML (SVM) | CNN | YAMNet | PANNs |\n"
              "EfficientAT | AST/CLAP/PaSST/BEATs (opt-in) |\n"
              "\"other\"-gate + Ensemble Combiner")

ax.annotate("", xy=(5.9, 2.3), xytext=(2.8, 2.3),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, linestyle="dashed"))
ax.add_patch(mpatches.Circle((4.35, 2.3), 0.08, facecolor="white", edgecolor="black", lw=1.3))
ax.text(4.35, 2.75, "IClassificationAPI (REST / JSON)", ha="center", fontsize=8)

ax.annotate("", xy=(11.2, 2.3), xytext=(8.9, 2.3),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, linestyle="dashed"))
ax.add_patch(mpatches.Circle((10.05, 2.3), 0.08, facecolor="white", edgecolor="black", lw=1.3))
ax.text(10.05, 2.75, "IInferenceAPI (REST / multipart)", ha="center", fontsize=8)

component_box(12.0, -1.0, 2.4, 1.1, "Trained Model\nArtifacts", "(processed_data/*)")
ax.annotate("", xy=(13.2, 0.1), xytext=(13.2, 1.5),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, linestyle="dashed"))
ax.text(13.55, 0.8, "uses", fontsize=8)

ax.set_ylim(-1.3, 4.2)
plt.tight_layout()
plt.savefig(OUT_35, dpi=200, facecolor="white")
plt.close()
print(f"Regenerated: {OUT_35}")

# ============================== swap both into the docx ==============================
tmp_out = DOCX + ".tmp_new"
with zipfile.ZipFile(DOCX, "r") as zin, zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == MEDIA_34:
            with open(OUT_34, "rb") as f:
                data = f.read()
            print("Replaced", item.filename)
        elif item.filename == MEDIA_35:
            with open(OUT_35, "rb") as f:
                data = f.read()
            print("Replaced", item.filename)
        zout.writestr(item, data)
shutil.move(tmp_out, DOCX)
print("Saved.")
