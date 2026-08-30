# -*- coding: utf-8 -*-
"""Regenerates Figure 1 (three-tier architecture diagram) with the current,
accurate model list, matching the original's visual style, then overwrites the
embedded image bytes directly inside the .docx zip (docx stores images as plain
files in word/media/, referenced by relationship id -- overwriting the file in
place keeps the existing embedding/position/caption untouched)."""
import zipfile
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DOCX = r"C:\Users\hp\Downloads\GarageAI_Research_Paper.docx"
MEDIA_PATH = "word/media/91ac28b81c4fd5b1163f63f84ea69f6d4543ff54.png"
OUT_PNG = "figure1_regenerated.png"

fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.5)
ax.axis("off")
ax.set_title("Three-Tier GarageAI System Architecture", fontsize=20, fontweight="bold",
              color="#1a3a5c", pad=20)

boxes = [
    (0.3, 2.0, 3.0, 1.6, "#2c5578", "React Frontend\n(garageai-frontend)",
     "Vite + React + TS\nport 5173"),
    (4.0, 2.0, 3.0, 1.6, "#1a9e94", "Node.js Gateway\n(garageai-backend)",
     "Express + Swagger\nport 5000"),
    (7.7, 2.0, 3.0, 1.6, "#16324a", "Python Inference\n(garageai-audio-analysis)",
     "FastAPI + Swagger\nport 8001"),
]
for x, y, w, h, color, title, sub in boxes:
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                          linewidth=0, facecolor=color))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            color="white", fontsize=13, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
            color="white", fontsize=10.5)

ax.annotate("", xy=(4.0, 2.9), xytext=(3.3, 2.9),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.8))
ax.annotate("", xy=(7.7, 2.9), xytext=(7.0, 2.9),
            arrowprops=dict(arrowstyle="<->", color="#555", lw=1.8))

ax.text(1.8, 4.2, "Mic recording /\nFile upload", ha="center", fontsize=10.5,
        style="italic", color="#333")
ax.text(3.65, 3.55, "audio file /\nJSON result", ha="center", fontsize=10, color="#333")
ax.text(7.35, 3.55, "audio file /\nJSON result", ha="center", fontsize=10, color="#333")
ax.text(9.2, 4.5,
        "Always-on: Traditional ML, CNN,\nYAMNet, PANNs, EfficientAT,\n"
        "\"other\"-class gate, Ensemble\nOpt-in: AST, CLAP, PaSST, BEATs",
        ha="center", fontsize=9.5, style="italic", color="#333")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=200, facecolor="white")
plt.close()
print(f"Regenerated: {OUT_PNG}")

shutil.copy(DOCX, DOCX + ".tmp_before_fig1")
with zipfile.ZipFile(DOCX, "r") as zin:
    names = zin.namelist()
    assert MEDIA_PATH in names, f"{MEDIA_PATH} not found in {DOCX}"

# Rewrite the zip, replacing only the one media file's bytes.
tmp_out = DOCX + ".tmp_new"
with zipfile.ZipFile(DOCX, "r") as zin, zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == MEDIA_PATH:
            with open(OUT_PNG, "rb") as f:
                data = f.read()
            print(f"Replaced {item.filename}: {len(zin.read(item.filename))} -> {len(data)} bytes")
        zout.writestr(item, data)

shutil.move(tmp_out, DOCX)
print("Saved.")
