# -*- coding: utf-8 -*-
"""Stage 3: fix remaining stale "40-run/four models" references, and add the
single most important new finding from the full-data follow-up -- that CLAP and
PaSST (and nearly BEATs) also exceed the traditional ML baseline once trained on
the same full 626-recording set as AST/EfficientAT, not just a 201-recording
subset. This directly extends Section 7.1's own causal argument with real data."""
import docx

SRC = r"C:\Users\hp\Downloads\GarageAI_Research_Paper.docx"


def set_text(paragraph, new_text):
    if not paragraph.runs:
        paragraph.add_run(new_text)
        return
    paragraph.runs[0].text = new_text
    for r in paragraph.runs[1:]:
        r.text = ""


def insert_paragraph_after(anchor, text, italic=False):
    new_p = anchor.insert_paragraph_before("")
    anchor._p.addnext(new_p._p)
    new_p.add_run(text)
    if italic and new_p.runs:
        new_p.runs[0].italic = True
    return new_p


def main():
    d = docx.Document(SRC)
    P = d.paragraphs

    # --- Contributions bullet (Section 1.5) ---
    assert P[37].text.startswith("A 40-run stability study")
    set_text(P[37], "A 90-run stability study across all nine individual models, "
             "five random seeds, and with/without waveform augmentation -- extended "
             "2026-08-27 from an initial 40-run/four-model version -- quantifying "
             "how sensitive small-test-set accuracy estimates are to training-sample "
             "variation, confirming that waveform augmentation did not benefit any "
             "of the nine models tested (three showed a statistically real "
             "degradation), and resolving the original study's training-set-size "
             "confound by retraining every model on the full 626-recording set.")

    # --- New finding: CLAP/PaSST/BEATs on full data also close the gap ---
    assert P[139].text.startswith("Among the frozen, general-purpose backbones")
    insert_paragraph_after(P[139],
        "This training-set-size explanation is directly testable, and the "
        "2026-08-27 full-data follow-up (Section 6.3) confirms it: when CLAP and "
        "PaSST are retrained on the same full 626-recording set used by AST and "
        "EfficientAT, rather than the 201-recording subset behind their originally "
        "reported single-run numbers in Table 4, both also exceed the traditional "
        "ML baseline (90.2% each, versus 86.5% for SVM), and BEATs matches it "
        "(86.5%) instead of trailing it (81.95% on the subset). Only YAMNet and "
        "PANNs/CNN14 -- both already trained on the full set in their original "
        "single run -- remain below the baseline regardless. This means training-set "
        "size, not architecture alone, explains a meaningful share of why CLAP, "
        "PaSST, and BEATs originally appeared to underperform in Table 4: four of "
        "the seven transfer-learning backbones tested (AST, CLAP, PaSST, and "
        "arguably BEATs) match or exceed the traditional baseline once given the "
        "same amount of training data, not just the two (AST, EfficientAT) Table 4 "
        "reports on its own. Table 4's numbers are left unchanged here since they "
        "document the models' originally reported, single-run configuration; Table "
        "7 (Section 6.3) is the fuller, apples-to-apples comparison.")

    # --- 7.3 Statistical Significance: update the "seven points" headline number ---
    assert P[144].text.startswith("The single-run results in Table 4")
    old = P[144].text
    new = old.replace(
        "for four comparable transfer-learning backbones, changing only the "
        "training seed shifted test accuracy by up to approximately seven "
        "points, with a typical standard deviation of 1-3% across five seeds.",
        "across all nine models, changing only the training seed shifted test "
        "accuracy by up to 8.3 points among the frozen-embedding models (CLAP, "
        "under augmentation) and, in one pathological case, by 30 points for the "
        "CNN without augmentation (one unlucky seed scored 56.4% against a typical "
        "78-86%) -- with a typical standard deviation of 0-3% across five seeds for "
        "most configurations, but as high as 11% for the CNN specifically."
    )
    if new == old:
        # exact phrase not found verbatim (formatting differences) -- fall back
        # to a direct rewrite of the sentence carrying the old "seven points" claim
        import re
        new = re.sub(
            r"for four comparable transfer-learning backbones.*?across five seeds\.",
            "across all nine models, changing only the training seed shifted test "
            "accuracy by up to 8.3 points among the frozen-embedding models (CLAP, "
            "under augmentation) and, in one pathological case, by 30 points for the "
            "CNN without augmentation (one unlucky seed scored 56.4% against a "
            "typical 78-86%) -- with a typical standard deviation of 0-3% across "
            "five seeds for most configurations, but as high as 11% for the CNN "
            "specifically.",
            old, count=1, flags=re.DOTALL)
    assert new != old, "P144 seven-points sentence not found/replaced -- check text manually"
    set_text(P[144], new)

    # --- Conclusion ---
    assert P[171].text.startswith("This research evaluated")
    old = P[171].text
    marker = "A dedicated 40-run stability study further showed that waveform augmentation did not benefit any of four additional transfer-learning backbones tested, and that seed-to-seed variation alone can shift test accuracy by several percentage points on this study's fixed 133-recording test set — a methodological caveat that should temper strong claims about small differences between models."
    replacement = ("A follow-up 90-run stability study across all nine individual "
        "models further showed that waveform augmentation did not statistically "
        "benefit any of them (three -- Traditional ML, AST, and PaSST -- were "
        "measurably hurt by it), that CLAP and PaSST also exceed the traditional "
        "baseline once trained on the same full recording set as AST and "
        "EfficientAT, and that seed-to-seed variation alone can shift test accuracy "
        "by up to 8 points for frozen embeddings and 30 points in one CNN outlier "
        "run -- a methodological caveat that should temper strong claims about "
        "small differences between models. A subsequent overfitting check also "
        "found a large gap between training and held-out accuracy for the deployed "
        "SVM (99.7% vs 86.5%) and YAMNet (100.0% vs 78.2%) models, not previously "
        "measured in this study, suggesting real-world accuracy on more varied data "
        "than the current 133-recording test set could be lower than reported for "
        "those two models specifically.")
    if marker in old:
        new = old.replace(marker, replacement)
    else:
        new = old + " " + replacement
    set_text(P[171], new)

    d.save(SRC)
    print("Saved stage 3.")


if __name__ == "__main__":
    main()
