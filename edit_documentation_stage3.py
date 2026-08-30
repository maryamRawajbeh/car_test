# -*- coding: utf-8 -*-
"""Stage 3: structural additions to Appendix A -- same content already added to
the standalone research paper (Section 6.3 rewrite, CLAP/PaSST full-data
finding, confusion matrix + discussion, other-class gate paragraph, future-work
fixes), located by text-match instead of fixed index."""
import copy
import docx

SRC = r"C:\Users\hp\Downloads\GarageAI_Documentation.docx"


def set_text(paragraph, new_text):
    if not paragraph.runs:
        paragraph.add_run(new_text)
        return
    paragraph.runs[0].text = new_text
    for r in paragraph.runs[1:]:
        r.text = ""


def find_one(paragraphs, startswith):
    hits = [p for p in paragraphs if p.text.startswith(startswith)]
    assert len(hits) == 1, f"expected 1 match for {startswith!r}, got {len(hits)}"
    return hits[0]


def insert_paragraph_after(anchor, text):
    new_p = anchor.insert_paragraph_before("")
    anchor._p.addnext(new_p._p)
    new_p.add_run(text)
    return new_p


def main():
    d = docx.Document(SRC)
    P = d.paragraphs

    # --- 6.3 heading ---
    h = find_one(P, "6.3 Augmentation and Seed-Stability Study")
    set_text(h, "6.3 Augmentation and Seed-Stability Study (Comprehensive, All Nine Models)")

    # --- 6.3 intro paragraph ---
    p1 = find_one(P, "Beyond the qualitative analysis above, a dedicated stability study")
    set_text(p1, "The stability study summarized in the previous version of this "
             "section (four transfer-learning backbones, a 201-recording training "
             "subset for three of them) was extended on 2026-08-27 into a "
             "comprehensive follow-up covering all nine individual models -- "
             "Traditional ML, CNN, YAMNet, PANNs/CNN14, AST, CLAP, PaSST, BEATs, and "
             "EfficientAT -- each retrained from five different random seeds, both "
             "with and without waveform augmentation, for a total of 90 independent "
             "training runs. Every model was trained on the FULL 626-recording "
             "training set in this follow-up (not a 201-recording subset), removing "
             "the training-size confound flagged as a limitation of the original "
             "study. For the frozen-embedding models, the seed only varies which "
             "augmented copies are generated (the underlying training files no "
             "longer change with the seed, since none are subsampled); for "
             "Traditional ML, the seed varies both the augmentation draw and the "
             "model's own internal randomness (bootstrap sampling, cross-validation "
             "fold assignment); for CNN and EfficientAT, the seed varies weight "
             "initialization, batch order, and augmentation. The validation and "
             "test partitions were held fixed throughout, as before.")

    # --- 6.3 results paragraph ---
    p2 = find_one(P, "The results were decisive on both questions")
    set_text(p2, "The results sharpen rather than overturn the original conclusion. "
             "Across all 90 runs, no model showed a statistically real benefit from "
             "waveform augmentation: three models -- Traditional ML (-7.1 accuracy "
             "points, mean 92.6% to 85.6%), AST (-1.6 points, 88.0% to 86.3%), and "
             "PaSST (-0.8 points, 90.2% to 89.5%) -- showed a real, consistent "
             "degradation (the gap exceeds 1.5x the pooled standard deviation of "
             "the two conditions); the remaining six models (CNN, YAMNet, "
             "PANNs/CNN14, CLAP, BEATs, EfficientAT) showed no statistically "
             "distinguishable effect either way. This is a stronger and more "
             "general result than the original four-model study reported: it now "
             "covers the Traditional ML model directly (previously only mentioned "
             "qualitatively in Section 3.3.1) and confirms none of the newer "
             "backbones benefit either. The seed-sensitivity finding was also more "
             "dramatic than the original seven-point estimate: CLAP's test "
             "accuracy under augmentation ranged from 84.2% to 92.5% across five "
             "seeds (an 8.3-point spread), and the CNN -- without augmentation "
             "specifically -- produced one catastrophic outlier run at 56.4% "
             "accuracy against a typical 78-86% for the other four seeds, a "
             "30-point swing traced to unlucky weight initialization rather than "
             "any change in data or method (Table 8). Augmentation measurably "
             "stabilized the CNN even though it did not reliably improve its mean "
             "accuracy: run-to-run standard deviation fell from 11.0 points "
             "without augmentation to 4.4 points with it.")

    # --- 6.3 table-summary paragraph ---
    p3 = find_one(P, "Table 7 summarizes the mean test accuracy across the five seeds")
    set_text(p3, "Table 7 summarizes the mean test accuracy across the five seeds "
             "for each of the nine models, with and without augmentation. Table 8 "
             "reports every individual run underlying this summary. A separate "
             "engineering finding from this follow-up, unrelated to augmentation "
             "itself: the classifier-selection code for six of the nine models "
             "(all except CNN and EfficientAT, which do not use SVC) was found to "
             "hang for over 17 CPU-hours on one configuration, traced to "
             "scikit-learn's SVC(probability=True) triggering an internal 5-fold "
             "Platt-scaling calibration that was never actually needed for model "
             "selection (only .predict() is called during ranking). Fixed by "
             "deferring probability=True to a refit of the winning model only, "
             "protecting any future retrain from the same failure mode.")

    # --- Table 7 caption ---
    for p in P:
        if p.text.startswith("Table 7. Mean test accuracy across five random seeds"):
            set_text(p, "Table 7. Mean test accuracy (+/- standard deviation) across "
                     "five random seeds, with and without waveform augmentation, for "
                     "all nine individual models. All nine now trained on the full "
                     "626-recording training set (2026-08-27 follow-up).")
            break

    # --- CLAP/PaSST full-data finding, inserted after 7.1's first paragraph ---
    anchor = find_one(P, "Among the frozen, general-purpose backbones (YAMNet, PANNs/CNN14, CLAP, PaSST, BEATs), none exceeded")
    insert_paragraph_after(anchor,
        "This training-set-size explanation is directly testable, and the "
        "2026-08-27 full-data follow-up (Section 6.3) confirms it: when CLAP and "
        "PaSST are retrained on the same full 626-recording set used by AST and "
        "EfficientAT, rather than the 201-recording subset behind their originally "
        "reported single-run numbers in Table 4, both also exceed the traditional "
        "ML baseline (90.2% each, versus 86.5% for SVM), and BEATs matches it "
        "(86.5%) instead of trailing it (81.95% on the subset). Only YAMNet and "
        "PANNs/CNN14 -- both already trained on the full set in their original "
        "single run -- remain below the baseline regardless. This means "
        "training-set size, not architecture alone, explains a meaningful share "
        "of why CLAP, PaSST, and BEATs originally appeared to underperform in "
        "Table 4: four of the seven transfer-learning backbones tested (AST, "
        "CLAP, PaSST, and arguably BEATs) match or exceed the traditional "
        "baseline once given the same amount of training data, not just the two "
        "(AST, EfficientAT) Table 4 reports on its own. Table 4's numbers are "
        "left unchanged here since they document the models' originally "
        "reported, single-run configuration; Table 7 (Section 6.3) is the "
        "fuller, apples-to-apples comparison.")

    # --- Confusion-matrix-related text (Results section) ---
    p_res = find_one(P, "The per-class breakdown (Table 6) shows Sway achieving")
    set_text(p_res, "The per-class breakdown (Table 6) shows Sway achieving the "
             "lowest recall (0.75) and Belt showing markedly lower precision (0.81) "
             "than recall (0.96). The confusion matrix (Table 5) explains why: "
             "Belt's low precision is driven mainly by Sway recordings being "
             "misclassified as Belt (7 of 44), more than by Brake (3 of 44) -- the "
             "opposite pairing from the earlier XGBoost-era matrix, where Belt was "
             "the cleanest class and confusion concentrated entirely between Brake "
             "and Sway. The current SVM model still confuses Brake and Sway with "
             "each other (4 and 2 recordings respectively), but a comparable "
             "number of Sway recordings are now lost to Belt instead, consistent "
             "with the moderate overlap already visible in the PCA projection "
             "(Figure 2b).")

    p_perclass = find_one(P, "Belt shows high recall (0.96) but comparatively lower precision")
    set_text(p_perclass, "Belt shows high recall (0.96) but comparatively lower "
             "precision (0.81), indicating that recordings from other classes -- "
             "Sway especially (Table 5) -- are sometimes misclassified as Belt; "
             "Sway shows the lowest recall (0.75) despite high precision (0.92), "
             "indicating a meaningful share of true Sway recordings are being "
             "missed, split between Belt and Brake; Brake is the most evenly "
             "balanced class (0.89/0.89/0.89). This differs from the earlier "
             "XGBoost-era matrix (Table 5, Section 6 footnote), where confusion "
             "was concentrated almost entirely between Brake and Sway and Belt "
             "was nearly error-free -- the current SVM model's decision boundary "
             "trades some of that Belt precision for its other gains.")

    # Insert the real confusion matrix table right after p_res, cloning the
    # per-class breakdown table's formatting (find it by locating the table
    # immediately following p_perclass's own table -- simplest robust way:
    # clone the FIRST 4x4 table found in the document, matching the research
    # paper's approach).
    template = None
    for t in d.tables:
        if len(t.rows) == 4 and len(t.columns) == 4 and t.rows[0].cells[0].text == "Class":
            template = t._tbl
            break
    assert template is not None, "could not locate the 4x4 per-class table to clone"
    new_tbl = copy.deepcopy(template)
    cm_table = docx.table.Table(new_tbl, d)
    header = ["Predicted -> / Actual v", "Belt", "Brake", "Sway"]
    for c, text in enumerate(header):
        cm_table.cell(0, c).text = text
    class_names = ["Belt", "Brake", "Sway"]
    cm_per_100 = [[96, 2, 2], [7, 89, 5], [16, 9, 75]]
    for r, cls in enumerate(class_names):
        cm_table.cell(r + 1, 0).text = cls
        for c in range(3):
            cm_table.cell(r + 1, c + 1).text = str(cm_per_100[r][c])
    p_res._p.addnext(new_tbl)
    caption_p = p_res.insert_paragraph_before("")
    new_tbl.addnext(caption_p._p)
    set_text(caption_p, "Table 5. Confusion matrix (per 100 samples) for the "
             "deployed SVM model, computed 2026-08-27 on freshly re-extracted "
             "test-set features.")
    if caption_p.runs:
        caption_p.runs[0].italic = True

    # --- Other-class gate paragraph, inserted after the inference-backend paragraph ---
    anchor2 = find_one(P, "Implemented with FastAPI (Swagger at /docs), loading the actual trained model artifacts")
    insert_paragraph_after(anchor2,
        "A separate, genuinely supervised 4th class -- \"other\" -- is checked "
        "before any of the belt/brake/sway models run: a RandomForest classifier "
        "trained on synthetic non-car signals (noise, tones, chirps, chords, "
        "engine hum, impulse trains, road noise, formant-approximated speech) "
        "plus real non-car audio (Windows system sounds), reaching 95.0% 4-class "
        "test accuracy and 100% held-out generalization to entirely unseen "
        "non-car sound types. Deployment uses a probability threshold (P(other) "
        ">= 0.46) rather than plain argmax, tuned against both the model's own "
        "held-out \"other\" samples and a separate, stricter generalization probe "
        "set. The gate is wired into predict.py, this backend's ensemble.py "
        "(where it short-circuits before any heavier model runs, saving "
        "compute), and the frontend, which renders a distinct \"this doesn't "
        "sound like a car fault\" state instead of forcing a belt/brake/sway "
        "guess.")

    # --- Future work: EfficientAT deployment status ---
    fw = find_one(P, "Evaluate deploying AST and/or EfficientAT")
    set_text(fw, "EfficientAT is now always-on in the live service and AST, CLAP, "
             "PaSST, and BEATs are reachable as an opt-in comparison model "
             "(Section 5.1); the remaining open question is whether any of them "
             "should move from opt-in to always-on, or be folded into the fused "
             "ensemble weights (Section 3.4.4), given the RAM/latency cost "
             "observed for PANNs and EfficientAT and the accuracy gains "
             "identified in the full-data follow-up (Section 6.3, Table 7).")

    d.save(SRC)
    print("Stage 3 done.")


if __name__ == "__main__":
    main()
