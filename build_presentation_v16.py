# -*- coding: utf-8 -*-
"""Build GarageAI_Presentation_v16.pptx from v11_final.pptx (the editable
ancestor of the v15 PDF -- same Slidesgo template, near-identical content).

Applies the requested v16 edit set:
  1  Preprocessing slide expanded, step by step.
  2  Data-leakage content removed entirely.
  3  Feature-extraction slide rewritten: what MFCC and log-mel actually are +
     the exact extraction pipeline (feature_pipeline.png).
  4  New slide: log-mel spectrogram of one representative clip per class
     (logmel_three_classes.png) -- the actual CNN input.
  5  Deck reordered:  Feature Extraction -> Log-mel examples -> Augmentation
     -> Seed & Stability -> Models -> Results (chart) -> Results (table).
  6  The whole "audio fingerprint / compression" topic removed
     (dedicated slide deleted; robustness framing scrubbed from ensemble,
     conclusion, future-work).
  7  New slide: full held-out-test metrics table (Accuracy, Macro P/R/F1,
     Macro-AUC, per-class F1) for all 10 configs + the ensemble.
  8  Ensemble evaluated in full (its own row in the table + head-to-head on
     the "Why XGBoost replaced the ensemble" slide).
  9  Freesound slide: why it was tried + how each model moved when the data
     was added to training.
"""
import copy
import os

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

SRC = r"C:\Users\hp\Downloads\GarageAI_Presentation_v11_final.pptx"
OUT = r"C:\Users\hp\Desktop\car_test\GarageAI_Presentation_v16.pptx"
FIG = r"C:\Users\hp\Desktop\car_test\processed_data\report_charts\v16"

WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0xD9, 0xE2, 0xE8)
RED = RGBColor(0xFF, 0x3F, 0x4A)
TEAL = RGBColor(0x16, 0xA8, 0x9F)
DARK = RGBColor(0x1B, 0x2E, 0x3C)
ROWA = RGBColor(0x24, 0x3B, 0x4B)
ROWB = RGBColor(0x1C, 0x2E, 0x3B)
HEADFILL = RGBColor(0x12, 0x22, 0x2E)

prs = Presentation(SRC)
S = prs.slides


# ----------------------------------------------------------------------------- helpers
def shp(slide, idx):
    return list(slide.shapes)[idx]


def _apply(run, ref):
    run.font.name = ref.font.name
    if ref.font.size:
        run.font.size = ref.font.size
    run.font.bold = ref.font.bold
    try:
        if ref.font.color and ref.font.color.type is not None:
            run.font.color.rgb = ref.font.color.rgb
    except Exception:
        pass


def set_text(shape, text, size=None, bold=None, color=None):
    """Replace a text-frame with a single paragraph, keeping p0/r0 formatting."""
    tf = shape.text_frame
    p0 = tf.paragraphs[0]
    ref = p0.runs[0] if p0.runs else None
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    for r in list(p0.runs)[1:]:
        r._r.getparent().remove(r._r)
    if not p0.runs:
        p0.add_run()
    r = p0.runs[0]
    r.text = text
    if ref is not None:
        _apply(r, ref)
    if size is not None:
        r.font.size = Pt(size)
    if bold is not None:
        r.font.bold = bold
    if color is not None:
        r.font.color.rgb = color


def set_lines(shape, lines, size=None, space=None):
    """Replace a text-frame with several paragraphs, reusing p0 formatting."""
    tf = shape.text_frame
    p0 = tf.paragraphs[0]
    ref = p0.runs[0] if p0.runs else None
    align = p0.alignment
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    for r in list(p0.runs)[1:]:
        r._r.getparent().remove(r._r)
    for i, line in enumerate(lines):
        p = p0 if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.level = 0
        # kill any inherited list indent / bullet so multi-line blocks stay flush
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", "0")
        pPr.set("indent", "0")
        from pptx.oxml.ns import qn as _qn
        for _tag in ("a:buChar", "a:buAutoNum"):
            for _e in pPr.findall(_qn(_tag)):
                pPr.remove(_e)
        if pPr.find(_qn("a:buNone")) is None:
            pPr.append(pPr.makeelement(_qn("a:buNone"), {}))
        if space is not None:
            p.space_after = Pt(space)
        if not p.runs:
            p.add_run()
        r = p.runs[0]
        r.text = line
        if ref is not None:
            _apply(r, ref)
        if size is not None:
            r.font.size = Pt(size)


def two_run(shape, bold_part, rest):
    """label:  bold_part (bold) + rest (normal) in one paragraph -- matches the
    template's numbered-row style."""
    tf = shape.text_frame
    p0 = tf.paragraphs[0]
    refs = list(p0.runs)
    rb = refs[0] if refs else None
    rn = refs[1] if len(refs) > 1 else refs[0] if refs else None
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    for r in list(p0.runs)[2:]:
        r._r.getparent().remove(r._r)
    while len(p0.runs) < 2:
        p0.add_run()
    p0.runs[0].text = bold_part
    p0.runs[1].text = rest
    if rb is not None:
        _apply(p0.runs[0], rb)
    if rn is not None:
        _apply(p0.runs[1], rn)


def clear_body(slide, keep_names=("Title", "Title 1")):
    """Remove every shape whose name is not in keep_names (keeps the title)."""
    for s in list(slide.shapes):
        if s.name not in keep_names:
            s._element.getparent().remove(s._element)


def add_text(slide, l, t, w, h, text, size=12, bold=False, color=WHITE,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    lines = text if isinstance(text, (list, tuple)) else [text]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = "Aptos"
    return tb


# =========================================================================== #
# SLIDE 5  (idx 4) -- Preprocessing, step by step  (leakage removed)          #
# =========================================================================== #
s5 = S[4]
set_text(shp(s5, 0), "Preprocessing — Clip to Model Input")
# three column headers + bodies (shapes 4,5,6 = headers ; 1,2,3 = bodies)
set_text(shp(s5, 4), "1 · Clean the signal")
set_lines(shp(s5, 1), [
    "Decode to mono and resample every file to 22 050 Hz.",
    "Trim leading / trailing silence (anything > 25 dB below the peak).",
    "Peak-normalise the waveform to [-1, 1] so loud and quiet recordings are comparable.",
], size=11.5)
set_text(shp(s5, 5), "2 · Fix the length to 5 s")
set_lines(shp(s5, 2), [
    "Longer clips are cut to the first 5 s; shorter clips are zero-padded at the end.",
    "A fault sound is periodic, so 5 s still captures several repetitions — and every model then gets the same input size.",
], size=11.5)
set_text(shp(s5, 6), "3 · Guard & scale")
set_lines(shp(s5, 3), [
    "Files with < 0.3 s of real audio after trimming, or that fail to decode, are dropped (corrupted_files.csv).",
    "269 features → StandardScaler; the log-mel image → z-scored — both fit on the train split only.",
], size=11.5)


# =========================================================================== #
# SLIDE 6  (idx 5) -- Feature Extraction: what MFCC & log-mel are             #
# =========================================================================== #
s6 = S[5]
set_text(shp(s6, 0), "Feature Extraction — MFCC & Log-Mel")
clear_body(s6, keep_names=(shp(s6, 0).name,))
add_text(s6, 0.55, 1.15, 8.9, 0.7, [
    "Both paths start from one shared spectrogram: STFT (n_fft 2048, hop 512) → 128-band mel "
    "filterbank (a perceptual pitch scale) → power. Then they split.",
], size=10, color=MUTED)
s6.shapes.add_picture(os.path.join(FIG, "feature_pipeline.png"),
                      Inches(0.55), Inches(1.95), width=Inches(8.9))
add_text(s6, 0.55, 4.35, 4.4, 1.2, [
    "MFCC — a compact timbre descriptor",
    "Log the mel energies, then a DCT: the spectral shape becomes ~40 coefficients "
    "(low ones = coarse shape). Δ / ΔΔ track how they change. 40 + 40 + 40 curves, "
    "each reduced to mean & std, + contrast / centroid / rolloff / RMS / ZCR / HPSS = 269.",
], size=8.6, color=MUTED)
add_text(s6, 5.15, 4.35, 4.4, 1.2, [
    "Log-mel — the picture, kept whole",
    "The same mel energies on a dB scale (ref = max), z-scored with the train mean/std. "
    "No summary — the full 128 × 216 frequency × time image goes straight to the CNN.",
], size=8.6, color=MUTED)


# =========================================================================== #
# SLIDE 22 (idx 21) -> repurpose as LOG-MEL EXAMPLES                          #
# =========================================================================== #
s_lm = S[21]
clear_body(s_lm, keep_names=(shp(s_lm, 0).name,))
set_text(shp(s_lm, 0), "Log-Mel Spectrogram — One per Class")
s_lm.shapes.add_picture(os.path.join(FIG, "logmel_three_classes.png"),
                        Inches(0.55), Inches(1.2), width=Inches(8.9))
add_text(s_lm, 0.55, 5.12, 9.0, 0.4,
         "One example clip per class, run through the same preprocessing: the 5 s waveform "
         "and the exact 128 × 216 log-mel image the CNN receives.",
         size=9, color=MUTED)


# =========================================================================== #
# SLIDE 11 (idx 10) -- audio fingerprint  ->  DELETE later                    #
# =========================================================================== #

# =========================================================================== #
# SLIDE 12 (idx 11) -- Why XGBoost Replaced the Ensemble                      #
# =========================================================================== #
s12 = S[11]
set_text(shp(s12, 6), "XGBoost Replaced the Ensemble")               # TextBox 7 title
set_text(shp(s12, 2), "93.23%")                                       # big stat 1
set_text(shp(s12, 3), "deployed XGBoost — accuracy & macro-F1")
set_text(shp(s12, 0), "89.47%")                                       # big stat 2
set_text(shp(s12, 1), "old 3-model ensemble — now a comparison view")
set_text(shp(s12, 4), "+3.8 pts")                                     # big stat 3
set_text(shp(s12, 5), "best single model vs the fused ensemble")
set_lines(shp(s12, 7), [
    "Full-data 5-seed study: XGBoost on the 269 features is the single best model "
    "(92.6% mean); no frozen or fine-tuned backbone beats it.",
    "The old ensemble fused Traditional ML 0.30 / CNN 0.60 / YAMNet 0.10 and reached "
    "89.47% (macro-F1 0.893) — below XGBoost on every metric.",
    "XGBoost is also deterministic (no seed lottery), needs no probability-calibration "
    "step, and has no fragile checkpoint dependency.",
    "The ensemble stays in the code only as an individual-model comparison shown to the user.",
], size=12.5, space=10)


# =========================================================================== #
# SLIDE 13 (idx 12) -> repurpose as FULL METRICS TABLE                        #
# =========================================================================== #
s_tbl = S[12]
clear_body(s_tbl, keep_names=(shp(s_tbl, 0).name,))
set_text(shp(s_tbl, 0), "Full Metrics — Held-Out Test Set")

headers = ["Configuration", "Acc", "Macro\nP", "Macro\nR", "Macro\nF1",
           "Macro\nAUC", "Belt\nF1", "Brake\nF1", "Sway\nF1"]
data = [
    ("XGBoost  (Traditional ML, deployed)", 93.23, 93.62, 93.22, 93.23, 96.50, 90.53, 100.00, 89.16),
    ("Ensemble  (SVM+CNN+YAMNet) *",        89.47, 90.22, 89.41, 89.33, 95.13, 89.80, 93.18, 85.00),
    ("EfficientAT  (fine-tuned)",           88.72, 89.86, 88.65, 88.76, 98.06, 88.89, 91.36, 86.05),
    ("AST",                                 87.97, 88.07, 87.90, 87.83, 95.08, 92.63, 88.10, 82.76),
    ("CLAP",                                84.96, 85.97, 84.97, 85.13, 96.14, 83.52, 88.89, 82.98),
    ("PaSST",                               84.21, 84.72, 84.18, 84.23, 94.26, 85.11, 85.37, 82.22),
    ("PANNs / CNN14",                       83.46, 83.67, 83.38, 83.27, 93.67, 87.50, 80.49, 81.82),
    ("CNN  (from scratch)",                 83.46, 83.76, 83.35, 83.12, 93.02, 90.72, 80.00, 78.65),
    ("BEATs",                               81.95, 81.85, 81.87, 81.77, 93.68, 90.32, 78.57, 76.40),
    ("YAMNet",                              78.20, 78.02, 78.10, 77.86, 89.63, 85.42, 72.29, 75.86),
]
nrows, ncols = len(data) + 1, len(headers)
gt = s_tbl.shapes.add_table(nrows, ncols, Inches(0.42), Inches(1.18),
                            Inches(9.16), Inches(3.5))
tbl = gt.table
tbl.columns[0].width = Inches(3.06)
for c in range(1, ncols):
    tbl.columns[c].width = Inches((9.16 - 3.06) / (ncols - 1))
for r in range(nrows):
    tbl.rows[r].height = Inches(0.3)

def _cell(cell, text, size, bold, color, fill, align=PP_ALIGN.CENTER):
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.margin_left = cell.margin_right = Inches(0.04)
    cell.margin_top = cell.margin_bottom = Inches(0.02)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    tf.paragraphs[0].text = ""
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
        r.font.name = "Aptos"

for c, h in enumerate(headers):
    _cell(tbl.cell(0, c), h, 8.5, True, WHITE, HEADFILL,
          PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
for ri, row in enumerate(data, start=1):
    fill = ROWA if ri % 2 else ROWB
    deployed = ri == 1
    _cell(tbl.cell(ri, 0), row[0], 8, deployed, RED if deployed else WHITE, fill, PP_ALIGN.LEFT)
    for ci in range(1, ncols):
        val = row[ci]
        txt = f"{val:.2f}" if ci != 1 else f"{val:.2f}"
        _cell(tbl.cell(ri, ci), txt, 8, deployed, WHITE, fill)

add_text(s_tbl, 0.62, 4.82, 8.8, 0.7, [
    "Single held-out run (133 recordings), each model's original configuration. Deployed XGBoost = features re-extracted from audio.",
    "CLAP / PaSST / BEATs reach 90.2 / 90.2 / 86.5 in the 5-seed chart (previous slide) once retrained on the full 626-set.   "
    "* ensemble superseded 2026-09-08, shown for comparison.",
], size=8, color=MUTED)


# =========================================================================== #
# SLIDE 14 (idx 13) -- External Data (Freesound): why + per-model move        #
# =========================================================================== #
s_fs = S[13]
clear_body(s_fs, keep_names=(shp(s_fs, 0).name,))
set_text(shp(s_fs, 0), "External Data (Freesound) — Rejected")
add_text(s_fs, 0.5, 1.08, 9.0, 0.8, [
    "Why try it:  only 893 in-house recordings. We curated 520 general + 183 brake-specific "
    "license-filtered Freesound clips (135 brake after dropping non-commercial licences), "
    "agreement-scored them, and added them to TRAINING ONLY - the test split was untouched.",
], size=10, color=MUTED)

fs_headers = ["Model", "Clean", "+ Freesound in training", "delta"]
fs_rows = [
    ("Traditional ML (XGBoost)", "93.23%", "93.23% best-filtered  /  91.0% unfiltered merge", "0  /  -2.3"),
    ("CNN (from scratch)", "83.5%", "hurt in all 6 configs tried; best result about 81.2%", "-2.3"),
    ("PANNs / CNN14", "81.95%", "84.96%, heavily agreement-filtered subset only", "+3.0"),
    ("YAMNet", "78.2%", "small, inconsistent movement (133-clip test set)", "about 0"),
]
gt = s_fs.shapes.add_table(len(fs_rows) + 1, 4, Inches(0.5), Inches(1.95),
                           Inches(9.0), Inches(1.9))
t = gt.table
t.columns[0].width = Inches(2.4)
t.columns[1].width = Inches(0.9)
t.columns[2].width = Inches(4.55)
t.columns[3].width = Inches(1.15)
for c, h in enumerate(fs_headers):
    _cell(t.cell(0, c), h, 9, True, WHITE, HEADFILL,
          PP_ALIGN.LEFT if c in (0, 2) else PP_ALIGN.CENTER)
for ri, row in enumerate(fs_rows, start=1):
    fill = ROWA if ri % 2 else ROWB
    for ci, v in enumerate(row):
        al = PP_ALIGN.LEFT if ci in (0, 2) else PP_ALIGN.CENTER
        _cell(t.cell(ri, ci), v, 8.5, ci == 3, WHITE, fill, al)

add_text(s_fs, 0.5, 3.95, 9.0, 1.5, [
    "Only PANNs gained, and only from the most heavily filtered subset. The deployed model also "
    "scored 0 / 194 on Freesound “brake” clips (mean p(brake) ~ 0.007) despite 100% recall "
    "on its own brake test set - a domain / recording mismatch, never confirmed by listening.",
    "Why rejected:  no consistent benefit across models  +  clear evidence of domain mismatch  +  "
    "crowd-sourced labels we could not verify. Excluded from the final training pipeline.",
], size=9.5, color=MUTED)


# =========================================================================== #
# SLIDE 15 (idx 14) -- Per-Class Performance (deployed XGBoost)               #
# =========================================================================== #
s_pc = S[14]
set_text(shp(s_pc, 0), "Per-Class Performance — XGBoost")
set_text(shp(s_pc, 1), "Belt")
set_text(shp(s_pc, 2), "Precision 86.00%  ·  Recall 95.56%  ·  F1 90.53%")
set_text(shp(s_pc, 3), "Brake")
set_text(shp(s_pc, 4), "Precision 100%  ·  Recall 100%  ·  F1 100%")
set_text(shp(s_pc, 5), "Sway  (hardest)")
set_text(shp(s_pc, 6), "Precision 94.87%  ·  Recall 84.09%  ·  F1 89.16%")
# swap the confusion picture for the XGBoost one (fixed height so it never
# runs into the note text below it)
old_pic = shp(s_pc, 7)
old_pic._element.getparent().remove(old_pic._element)
s_pc.shapes.add_picture(os.path.join(FIG, "xgboost_confusion.png"),
                        Inches(4.95), Inches(1.28), height=Inches(2.95))
set_lines(shp(s_pc, 7), [
    "Main confusion: 7 of 44 Sway clips read as Belt; 2 Belt read as Sway.",
    "5-fold CV macro-F1 93.44% vs test 93.23% — no train–test gap.",
], size=11)


# =========================================================================== #
# SLIDE 17 (idx 16) -> Ask GarageAI (was System & Deployment)                 #
# =========================================================================== #
s_ai = S[16]
clear_body(s_ai, keep_names=(shp(s_ai, 0).name,))
set_text(shp(s_ai, 0), "Ask GarageAI — RAG Diagnosis")


def col(l, head, body):
    add_text(s_ai, l, 1.3, 2.85, 0.35, head, size=13, bold=True, color=TEAL)
    add_text(s_ai, l, 1.75, 2.85, 2.0, body, size=10.5, color=WHITE)

col(0.6, "Knowledge base",
    "2 curated Arabic glossaries — faults + severity (235 lines).")
col(3.62, "Retrieval",
    "multilingual-e5-small embeddings, cosine similarity, top-k = 8; the LLM then re-reads the hits.")
col(6.64, "Generation",
    "Gemini gemini-3.5-flash-lite, local Arabic dialect. Advisory only — never invents a diagnosis.")

add_text(s_ai, 0.6, 3.35, 8.8, 1.9, [
    "•  <<NO_MATCH>> gate: if nothing fits, it asks a follow-up instead of forcing an answer.",
    "•  Skips text search when an audio result already exists;  50–75% confidence → confirm, >75% → direct advice.",
    "•  <<SEVERITY>> tag stripped before the reply;  honest failure messages for no-match / quota / service error.",
    "•  Tested across the stack:  67 pytest  +  62 Jest  +  11 Playwright.",
], size=10.5, color=MUTED)


# =========================================================================== #
# SLIDE 19 (idx 18) -- Conclusion  (compression line removed)                 #
# =========================================================================== #
s_cc = S[18]
set_text(shp(s_cc, 2), "92.6%")
set_text(shp(s_cc, 3), "XGBoost — mean across 5 random seeds, full data")
set_text(shp(s_cc, 0), "93.23%")
set_text(shp(s_cc, 1), "deployed XGBoost — held-out test set")
set_text(shp(s_cc, 4), "10")
set_text(shp(s_cc, 5), "configurations compared under one protocol")
set_lines(shp(s_cc, 7), [
    "Answer to the research question: a carefully engineered, task-specific pipeline "
    "beat every large pre-trained backbone on this problem.",
    "Frozen general-purpose embeddings under-perform here; only genuine fine-tuning or a "
    "strong transformer gets close, and none passes.",
    "Honest negatives reported on purpose: waveform augmentation, SMOTE, mic-response sim, "
    "external Freesound data, and ensemble stacking.",
    "The 3-model ensemble (89.47%) is retained only as a comparison view.",
], size=14, space=10)


# =========================================================================== #
# SLIDE 20 (idx 19) -- Future Work  (fingerprint bullet removed)              #
# =========================================================================== #
s_fw = S[19]
two_run(shp(s_fw, 2), "More Sway data   ",
        "Genuinely new, correctly-labelled Sway recordings — the clearest lever for the main confusion.")
two_run(shp(s_fw, 4), "Significance testing   ",
        "Formal tests on the model-to-model accuracy gaps, which seed variation alone can rival.")
two_run(shp(s_fw, 6), "Backbones   ",
        "Fine-tune AST / EfficientAT further and re-compare against the deployed XGBoost.")
two_run(shp(s_fw, 8), "Cost benchmark   ",
        "Formal latency / RAM / model-size profiling across all 10 configs.")
two_run(shp(s_fw, 10), "Fix overfitting   ",
        "Add regularisation for the SVM / YAMNet train–test gap.")
two_run(shp(s_fw, 12), "On-device use   ",
        "Explore mobile / embedded deployment for fully offline operation.")


# =========================================================================== #
# SLIDE 18 (idx 17) -- Engineering Challenges: keep only #2 and #3            #
# =========================================================================== #
s_ec = S[17]
set_text(shp(s_ec, 2), "Seed discipline")
set_lines(shp(s_ec, 3), [
    "Never trust a single-run CNN accuracy delta - seed variance alone can move it "
    "by ~8 points (30 in one unlucky init). Always >= 5 seeds before a difference counts.",
], size=11)
set_text(shp(s_ec, 5), "Staleness trap")
set_lines(shp(s_ec, 6), [
    "A saved report (or a cached feature matrix) can describe an older model than the one "
    "deployed - always re-verify the loaded model against fresh inputs. It bit us again here: "
    "the old X_test cache scored the new XGBoost at 44%; fresh extraction gives 93.23%.",
], size=11)
set_text(shp(s_ec, 1), "01")
set_text(shp(s_ec, 4), "02")
for _ in range(6):                      # remove challenge #3 (03) and #4 (04) shape groups
    shp(s_ec, 7)._element.getparent().remove(shp(s_ec, 7)._element)
for k in (1, 2, 3, 4, 5, 6):           # nudge the two remaining items toward centre
    sh = shp(s_ec, k)
    sh.top = sh.top + Inches(1.1)


# =========================================================================== #
# SLIDE 3 (idx 2) -- overview: soften 'weighted fusion' wording               #
# =========================================================================== #
set_lines(shp(S[2], 20), ["The \u201cother\u201d gate runs first and can short-circuit; XGBoost makes the call."], size=10)
set_lines(shp(S[2], 19), ["Decision & Gate"], size=13.5)
set_lines(shp(S[2], 4), ["893 clips · 70 / 15 / 15 train / val / test."], size=10)

# ---- scrub remaining leakage / compression-robustness phrasing -------------
set_lines(shp(S[6], 8), ["Grid-searched blend; 5 seeds, full 626-recording set."], size=11.5)   # Ten Models: Weighted Ensemble
set_lines(shp(S[9], 20), ["Device-filter sim only moved a synthetic axis \u2014 no real-world gain."], size=10)  # Augmentation: Mic-Response Sim


# =========================================================================== #
# SLIDE 4 (idx 3) -- Dataset: add the mechanic / Reddit source split          #
# =========================================================================== #
s_ds = S[3]
# shrink the right-hand prose and drop it to the bottom
p9 = shp(s_ds, 9)
p9.top = Inches(4.68)
p9.height = Inches(0.7)
set_lines(p9, [
    "Real vehicle audio, not synthetic. Every clip's label was reviewed by a mechanic "
    "listening to it; no independent physical inspection.",
], size=9)
add_text(s_ds, 5.11, 2.62, 4.15, 0.32, "Where the audio came from",
         size=12, bold=True, color=RED)
_src_h = ["Class", "Mechanics", "Reddit", "Total"]
_src = [("Belt", "35", "267", "302"),
        ("Brake", "25", "269", "294"),
        ("Sway", "21", "276", "297")]
gt = s_ds.shapes.add_table(4, 4, Inches(5.11), Inches(3.0), Inches(4.15), Inches(1.4))
_t = gt.table
_t.columns[0].width = Inches(1.3)
_t.columns[1].width = Inches(1.15)
_t.columns[2].width = Inches(1.0)
_t.columns[3].width = Inches(0.7)
for c, h in enumerate(_src_h):
    _cell(_t.cell(0, c), h, 9, True, WHITE, HEADFILL, PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER)
for ri, row in enumerate(_src, start=1):
    fill = ROWA if ri % 2 else ROWB
    for ci, v in enumerate(row):
        _cell(_t.cell(ri, ci), v, 9, ci == 3, WHITE, fill,
              PP_ALIGN.LEFT if ci == 0 else PP_ALIGN.CENTER)


# =========================================================================== #
# SLIDE 8 chart (idx 7) -- relabel the two stale categories                   #
# =========================================================================== #
for sh in S[7].shapes:
    if sh.has_chart:
        cs = sh.chart._chartSpace
        for v in cs.iter():
            if v.tag.endswith("}v") and v.text:
                if v.text.strip() == "XGBoost \u2014 best single":
                    v.text = "XGBoost \u2014 deployed"
                elif v.text.strip() == "Ensemble \u2014 deployed":
                    v.text = "Ensemble (superseded)"


# =========================================================================== #
# SLIDE 21 (idx 20) -- Mechanics table: fill the real names (8 rows)          #
# =========================================================================== #
MECH = [
    ("\u0639\u0631\u0648\u0629 \u0645\u0635\u0627\u0631\u0648\u0629", "0595181050"),
    ("\u0645\u062c\u0627\u0647\u062f \u0635\u0627\u0644\u062d", "0597409364"),
    ("\u064a\u0648\u0633\u0641 \u0633\u0631\u064a\u062c\u064a", "0597058433"),
    ("\u0623\u062d\u0645\u062f \u0633\u0641\u0627\u0631\u064a\u0646\u064a", "0528215461"),
    ("\u0639\u0645\u0627\u062f \u064a\u062f\u0643", "0597018473"),
    ("\u0631\u0636\u0648\u0627\u0646 \u0634\u0646\u0627\u0631\u0629", "0595166555"),
    ("\u0646\u0647\u0627\u062f \u062d\u0645\u062f\u0627\u0646", "0598609818"),
    ("\u0631\u062c\u0627\u0626\u064a \u0627\u0644\u0642\u064a\u0633\u064a", "0598311113"),
]
for sh in S[20].shapes:
    if sh.has_table:
        mt = sh.table
        while len(mt.rows) < len(MECH) + 1:                 # grow by cloning last <a:tr>
            last = mt._tbl.tr_lst[-1]
            mt._tbl.append(copy.deepcopy(last))
        sh.top = Inches(1.5)
        sh.height = Inches(3.7)
        for r in mt.rows:
            r.height = Inches(0.4)
        for i, (nm, ph) in enumerate(MECH, start=1):
            for ci, val in ((0, nm), (1, ph)):
                cell = mt.cell(i, ci)
                para = cell.text_frame.paragraphs[0]
                ref = para.runs[0] if para.runs else None
                for extra in list(para.runs)[1:]:
                    extra._r.getparent().remove(extra._r)
                if not para.runs:
                    para.add_run()
                para.runs[0].text = val
                if ref is not None and ref is not para.runs[0]:
                    pass


# =========================================================================== #
#  DELETE slide 11 (audio fingerprint)  +  REORDER                            #
# =========================================================================== #
sldIdLst = S._sldIdLst
ids = list(sldIdLst)                       # 23 elements, index-aligned to S
sldIdLst.remove(ids[10])                   # drop the fingerprint slide
remaining = [e for i, e in enumerate(ids) if i != 10]   # 22 elements, new order 0..21

# new-position -> old-remaining-index
# order: ... Features, Log-mel examples, Augmentation, Models, Results chart,
#        Results table, Seed & Stability, Per-Class, Why-XGBoost, Freesound, ...
perm = [0, 1, 2, 3, 4, 5, 20, 9, 6, 7, 11, 8, 13, 10, 12, 14, 15, 16, 17, 18, 19, 21]
assert sorted(perm) == list(range(22))
for e in list(sldIdLst):
    sldIdLst.remove(e)
for pos in perm:
    sldIdLst.append(remaining[pos])

prs.save(OUT)
print("saved", OUT)
print("slides:", len(Presentation(OUT).slides))
