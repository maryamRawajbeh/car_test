# -*- coding: utf-8 -*-
"""Applies the full set of corrections/updates to GarageAI_Research_Paper.docx,
identified by cross-referencing the paper against the actual codebase, the
2026-08-27 comprehensive 90-run benchmark, and the fresh overfitting check.
Edits a COPY (reads from Downloads, writes back to Downloads) -- a backup was
already made separately before this script runs.
"""
import copy
import docx

SRC = r"C:\Users\hp\Downloads\GarageAI_Research_Paper.docx"


def set_text(paragraph, new_text):
    """Replace a paragraph's visible text while preserving the first run's
    formatting (bold/italic/font) -- safest way to edit body text/captions
    without losing style, since these paragraphs are single-run."""
    if not paragraph.runs:
        paragraph.add_run(new_text)
        return
    paragraph.runs[0].text = new_text
    for r in paragraph.runs[1:]:
        r.text = ""


def set_table_cell(table, row, col, text):
    table.cell(row, col).text = text


def resize_table_rows(table, n_rows_needed):
    """Add or remove data rows (below the header) so the table has exactly
    n_rows_needed data rows, by cloning the last row's XML for new rows."""
    current_data_rows = len(table.rows) - 1
    if current_data_rows < n_rows_needed:
        template_row = table.rows[-1]._tr
        for _ in range(n_rows_needed - current_data_rows):
            new_row = copy.deepcopy(template_row)
            template_row.addnext(new_row)
    elif current_data_rows > n_rows_needed:
        for row in table.rows[n_rows_needed + 1:][::-1]:
            row._tr.getparent().remove(row._tr)


def main():
    d = docx.Document(SRC)
    P = d.paragraphs

    # ------------------------------------------------------------------
    # 1) GPU claim -- factually wrong (confirmed CPU-only, no GPU at all,
    #    torch.cuda.is_available()==False, tf GPU devices==[]); also
    #    contradicted the paper's OWN later sentence (5.1) about the
    #    extended study being "CPU-only".
    # ------------------------------------------------------------------
    assert "GPU acceleration used for CNN training" in P[103].text
    set_text(P[103], "CPU only for every training run and inference call in this "
             "project -- no GPU was available on the development machine at any "
             "point (confirmed directly: torch.cuda.is_available() == False, "
             "tf.config.list_physical_devices('GPU') == []).")

    # ------------------------------------------------------------------
    # 2) Confusion matrix -- was "pending re-verification"; now computed
    #    directly from the deployed SVM model on freshly re-extracted test
    #    features (never touches the cached, potentially stale arrays).
    # ------------------------------------------------------------------
    assert "pending re-verification" in P[114].text
    set_text(P[114], "The per-class breakdown (Table 6) shows Sway achieving the "
             "lowest recall (0.75) and Belt showing markedly lower precision (0.81) "
             "than recall (0.96). The confusion matrix (Table 5) explains why: Belt's "
             "low precision is driven mainly by Sway recordings being misclassified as "
             "Belt (7 of 44), more than by Brake (3 of 44) -- the opposite pairing from "
             "the earlier XGBoost-era matrix, where Belt was the cleanest class and "
             "confusion concentrated entirely between Brake and Sway. The current SVM "
             "model still confuses Brake and Sway with each other (4 and 2 recordings "
             "respectively), but a comparable number of Sway recordings are now lost to "
             "Belt instead, consistent with the moderate overlap already visible in the "
             "PCA projection (Figure 2b).")

    assert "will be examined further once an updated confusion matrix" in P[117].text
    set_text(P[117], "Belt shows high recall (0.96) but comparatively lower precision "
             "(0.81), indicating that recordings from other classes -- Sway "
             "especially (Table 5) -- are sometimes misclassified as Belt; Sway shows "
             "the lowest recall (0.75) despite high precision (0.92), indicating a "
             "meaningful share of true Sway recordings are being missed, split between "
             "Belt and Brake; Brake is the most evenly balanced class (0.89/0.89/0.89). "
             "This differs from the earlier XGBoost-era matrix (Table 5, Section 6 "
             "footnote), where confusion was concentrated almost entirely between Brake "
             "and Sway and Belt was nearly error-free -- the current SVM model's "
             "decision boundary trades some of that Belt precision for its other gains.")

    # Insert the real confusion matrix as a new table right after P114's paragraph.
    class_names = ["Belt", "Brake", "Sway"]
    cm_per_100 = [[96, 2, 2], [7, 89, 5], [16, 9, 75]]
    template = d.tables[4]._tbl  # clone the existing 4x4 per-class table's formatting
    new_tbl = copy.deepcopy(template)
    cm_table = docx.table.Table(new_tbl, d)
    resize_table_rows(cm_table, 4)  # header + 3 class rows, 4 cols
    header = ["Predicted -> / Actual v", "Belt", "Brake", "Sway"]
    for c, text in enumerate(header):
        set_table_cell(cm_table, 0, c, text)
    for r, cls in enumerate(class_names):
        set_table_cell(cm_table, r + 1, 0, cls)
        for c in range(3):
            set_table_cell(cm_table, r + 1, c + 1, str(cm_per_100[r][c]))
    P[114]._p.addnext(new_tbl)
    caption_p = P[114].insert_paragraph_before("")
    # move caption to just after the table instead (addnext again)
    new_tbl.addnext(caption_p._p)
    set_text(caption_p, "Table 5. Confusion matrix (per 100 samples) for the deployed "
             "SVM model, computed 2026-08-27 on freshly re-extracted test-set features.")
    if caption_p.runs:
        caption_p.runs[0].italic = True

    # ------------------------------------------------------------------
    # 3) Section 6.3 -- superseded by the comprehensive follow-up study
    #    (all nine models, full 626-recording training set, 90 runs total).
    # ------------------------------------------------------------------
    assert P[129].text.startswith("6.3 Augmentation")
    set_text(P[129], "6.3 Augmentation and Seed-Stability Study (Comprehensive, "
             "All Nine Models)")

    assert P[130].text.startswith("Beyond the qualitative analysis")
    set_text(P[130], "The stability study summarized in the previous version of this "
             "section (four transfer-learning backbones, a 201-recording training "
             "subset for three of them) was extended on 2026-08-27 into a comprehensive "
             "follow-up covering all nine individual models -- Traditional ML, CNN, "
             "YAMNet, PANNs/CNN14, AST, CLAP, PaSST, BEATs, and EfficientAT -- each "
             "retrained from five different random seeds, both with and without "
             "waveform augmentation, for a total of 90 independent training runs. "
             "Every model was trained on the FULL 626-recording training set in this "
             "follow-up (not a 201-recording subset), removing the training-size "
             "confound flagged as a limitation of the original study. For the "
             "frozen-embedding models, the seed only varies which augmented copies "
             "are generated (the underlying training files no longer change with the "
             "seed, since none are subsampled); for Traditional ML, the seed varies "
             "both the augmentation draw and the model's own internal randomness "
             "(bootstrap sampling, cross-validation fold assignment); for CNN and "
             "EfficientAT, the seed varies weight initialization, batch order, and "
             "augmentation. The validation and test partitions were held fixed "
             "throughout, as before.")

    p132_new = ("The results sharpen rather than overturn the original conclusion. "
                "Across all 90 runs, no model showed a statistically real benefit from "
                "waveform augmentation: three models -- Traditional ML (-7.1 accuracy "
                "points, mean 92.6% to 85.6%), AST (-1.6 points, 88.0% to 86.3%), and "
                "PaSST (-0.8 points, 90.2% to 89.5%) -- showed a real, consistent "
                "degradation (the gap exceeds 1.5x the pooled standard deviation of the "
                "two conditions); the remaining six models (CNN, YAMNet, PANNs/CNN14, "
                "CLAP, BEATs, EfficientAT) showed no statistically distinguishable "
                "effect either way. This is a stronger and more general result than the "
                "original four-model study reported: it now covers the Traditional ML "
                "model directly (previously only mentioned qualitatively in Section "
                "3.3.1) and confirms none of the newer backbones benefit either. The "
                "seed-sensitivity finding was also more dramatic than the original "
                "seven-point estimate: CLAP's test accuracy under augmentation ranged "
                "from 84.2% to 92.5% across five seeds (an 8.3-point spread), and the "
                "CNN -- without augmentation specifically -- produced one catastrophic "
                "outlier run at 56.4% accuracy against a typical 78-86% for the other "
                "four seeds, a 30-point swing traced to unlucky weight initialization "
                "rather than any change in data or method (Table 8). Augmentation "
                "measurably stabilized the CNN even though it did not reliably improve "
                "its mean accuracy: run-to-run standard deviation fell from 11.0 points "
                "without augmentation to 4.4 points with it.")
    assert P[131].text.startswith("The results were decisive")
    set_text(P[131], p132_new)

    p133_extra = (" A separate engineering finding from this follow-up, unrelated to "
                   "augmentation itself: the classifier-selection code for six of the "
                   "nine models (all except CNN and EfficientAT, which do not use SVC) "
                   "was found to hang for over 17 CPU-hours on one configuration, "
                   "traced to scikit-learn's SVC(probability=True) triggering an "
                   "internal 5-fold Platt-scaling calibration that was never actually "
                   "needed for model selection (only .predict() is called during "
                   "ranking). Fixed by deferring probability=True to a refit of the "
                   "winning model only, protecting any future retrain from the same "
                   "failure mode.")
    assert P[132].text.startswith("Table 7 summarizes")
    set_text(P[132], "Table 7 summarizes the mean test accuracy across the five seeds "
              "for each of the nine models, with and without augmentation. Table 8 "
              "reports every individual run underlying this summary." + p133_extra)

    assert P[133].text.startswith("Table 7. Mean test accuracy")
    set_text(P[133], "Table 7. Mean test accuracy (+/- standard deviation) across "
             "five random seeds, with and without waveform augmentation, for all "
             "nine individual models. All nine now trained on the full 626-recording "
             "training set (2026-08-27 follow-up).")

    # locate Table 8 caption robustly by scanning nearby paragraphs
    for idx in range(132, 142):
        if P[idx].text.startswith("Table 8. Full results"):
            set_text(P[idx], "Table 8. Full results of the 90-run comprehensive "
                     "stability study: nine models, five random seeds each, with and "
                     "without waveform augmentation. Every model trained on the full "
                     "626-recording training set; validation and test partitions held "
                     "fixed throughout.")
            break

    d.save(SRC)
    print("Saved stage 1 (text edits + confusion matrix table).")


if __name__ == "__main__":
    main()
