# -*- coding: utf-8 -*-
"""Stage 2: replace the old 4-model/40-run study tables with the comprehensive
9-model/90-run results, and update the now-resolved Limitations/Future-Work items."""
import copy
import docx
import pandas as pd
import numpy as np

SRC = r"C:\Users\hp\Downloads\GarageAI_Research_Paper.docx"


def set_text(paragraph, new_text):
    if not paragraph.runs:
        paragraph.add_run(new_text)
        return
    paragraph.runs[0].text = new_text
    for r in paragraph.runs[1:]:
        r.text = ""


def resize_table_rows(table, n_data_rows_needed):
    current_data_rows = len(table.rows) - 1
    if current_data_rows < n_data_rows_needed:
        template_row = table.rows[-1]._tr
        for _ in range(n_data_rows_needed - current_data_rows):
            new_row = copy.deepcopy(template_row)
            template_row.addnext(new_row)
            template_row = new_row
    elif current_data_rows > n_data_rows_needed:
        for row in table.rows[n_data_rows_needed + 1:][::-1]:
            row._tr.getparent().remove(row._tr)


def build_summary_rows():
    name_map = {"yamnet": "YAMNet", "panns": "PANNs/CNN14", "ast": "AST",
                "clap": "CLAP", "passt": "PaSST", "beats": "BEATs"}
    frozen = pd.read_csv("processed_data/augmentation_variance_study_full_data.csv")
    tml = pd.read_csv("processed_data/traditional_ml_augmentation_variance.csv")
    cnn = pd.read_csv("processed_data/cnn_augmentation_variance.csv")
    eat = pd.read_csv("processed_data/efficientat_augmentation_study.csv")

    stats = {}
    for (model, aug), g in frozen.groupby(["model", "augment"]):
        stats.setdefault(name_map[model], {})[bool(aug)] = g["test_accuracy"].values * 100
    for cond, g in tml.groupby("condition"):
        stats.setdefault("Traditional ML (SVM/XGBoost)", {})[cond == "with_augmentation"] = \
            g["test_accuracy"].values * 100
    for cond, g in cnn.groupby("condition"):
        stats.setdefault("CNN", {})[cond == "with_augmentation"] = g["test_accuracy"].values * 100
    for aug, g in eat.groupby("augment"):
        stats.setdefault("EfficientAT (fine-tuned)", {})[bool(aug)] = g["test_accuracy"].values * 100

    order = ["YAMNet", "PANNs/CNN14", "CNN", "BEATs", "PaSST", "CLAP",
             "Traditional ML (SVM/XGBoost)", "AST", "EfficientAT (fine-tuned)"]
    rows = []
    for name in order:
        no_aug = stats[name][False]
        with_aug = stats[name][True]
        m0, s0 = no_aug.mean(), no_aug.std()
        m1, s1 = with_aug.mean(), with_aug.std()
        gap = m1 - m0
        pooled = np.sqrt((s0 ** 2 + s1 ** 2) / 2)
        if pooled > 0 and abs(gap) > 1.5 * pooled:
            verdict = "Hurts" if gap < 0 else "Helps"
        else:
            verdict = "No effect (noise)"
        rows.append([name, f"{m0:.1f}% (+/-{s0:.1f})", f"{m1:.1f}% (+/-{s1:.1f})",
                     f"{gap:+.1f} pts -- {verdict}"])
    return rows


def build_raw_rows():
    name_map = {"yamnet": "YAMNet", "panns": "PANNs/CNN14", "ast": "AST",
                "clap": "CLAP", "passt": "PaSST", "beats": "BEATs"}
    frozen = pd.read_csv("processed_data/augmentation_variance_study_full_data.csv")
    tml = pd.read_csv("processed_data/traditional_ml_augmentation_variance.csv")
    cnn = pd.read_csv("processed_data/cnn_augmentation_variance.csv")
    eat = pd.read_csv("processed_data/efficientat_augmentation_study.csv")

    order = ["YAMNet", "PANNs/CNN14", "CNN", "BEATs", "PaSST", "CLAP",
              "Traditional ML (SVM/XGBoost)", "AST", "EfficientAT (fine-tuned)"]
    by_model = {name: [] for name in order}

    for _, r in frozen.iterrows():
        name = name_map[r["model"]]
        n = 2504 if r["augment"] else 626
        by_model[name].append([name, str(int(r["seed"])), "Yes" if r["augment"] else "No",
                                str(n), f"{r['test_accuracy']*100:.2f}%", f"{r['test_macro_f1']:.4f}"])
    for _, r in tml.iterrows():
        aug = r["condition"] == "with_augmentation"
        n = 1872 if aug else 624
        by_model["Traditional ML (SVM/XGBoost)"].append([
            "Traditional ML (SVM/XGBoost)", str(int(r["seed"])), "Yes" if aug else "No",
            str(n), f"{r['test_accuracy']*100:.2f}%", f"{r['test_macro_f1']:.4f}"])
    for _, r in cnn.iterrows():
        aug = r["condition"] == "with_augmentation"
        n = 1872 if aug else 624
        by_model["CNN"].append(["CNN", str(int(r["seed"])), "Yes" if aug else "No",
                                 str(n), f"{r['test_accuracy']*100:.2f}%", f"{r['test_macro_f1']:.4f}"])
    for _, r in eat.iterrows():
        by_model["EfficientAT (fine-tuned)"].append([
            "EfficientAT (fine-tuned)", str(int(r["seed"])), "Yes" if r["augment"] else "No",
            "626 (online aug.)" if r["augment"] else "626",
            f"{r['test_accuracy']*100:.2f}%", f"{r['test_macro_f1']:.4f}"])

    rows = []
    for name in order:
        rows.extend(sorted(by_model[name], key=lambda x: (int(x[1]), x[2])))
    return rows


def main():
    d = docx.Document(SRC)
    P = d.paragraphs
    T = d.tables

    # --- Table 6 (old 4-model summary) -> new 9-model summary ---
    summary_rows = build_summary_rows()
    t6 = T[6]
    resize_table_rows(t6, len(summary_rows))
    header = ["Model", "Mean Accuracy (no augmentation)", "Mean Accuracy (with augmentation)", "Effect"]
    for c, text in enumerate(header):
        t6.cell(0, c).text = text
    for r, row in enumerate(summary_rows):
        for c, val in enumerate(row):
            t6.cell(r + 1, c).text = val
    print(f"Table 6 (summary) resized to {len(summary_rows)} data rows.")

    # --- Table 7 (old 40-row raw) -> new 90-row raw ---
    raw_rows = build_raw_rows()
    t7 = T[7]
    resize_table_rows(t7, len(raw_rows))
    header = ["Model", "Seed", "Augmentation", "Train Size", "Test Accuracy", "Test Macro-F1"]
    for c, text in enumerate(header):
        t7.cell(0, c).text = text
    for r, row in enumerate(raw_rows):
        for c, val in enumerate(row):
            t7.cell(r + 1, c).text = val
    print(f"Table 7 (raw) resized to {len(raw_rows)} data rows.")

    # --- Limitations: resolved items ---
    assert P[150].text.startswith("Two important comparisons remain incomplete")
    set_text(P[150], "This was resolved in a 2026-08-27 follow-up: a comprehensive "
             "90-run stability study now covers all nine individual models (not just "
             "four), each trained on the full 626-recording training set (not a "
             "201-recording subset for three of them), removing both gaps at once "
             "(Section 6.3, Tables 7-8). A new gap surfaced by the same follow-up "
             "check is more concerning: the currently deployed SVM and YAMNet models "
             "show a large train-vs-test accuracy gap when evaluated on their own "
             "literal training recordings (SVM: 99.7% train vs 86.5% test, a 13-point "
             "gap; YAMNet: 100.0% train vs 78.2% test, a 22-point gap; CNN's gap is "
             "smaller, about 1 point on this metric) -- both models have enough "
             "capacity to nearly memorize their ~626 training examples, which was not "
             "previously measured or discussed in this paper. The reported test-set "
             "numbers remain honest (the test set was never used for fitting), but "
             "this gap suggests real-world accuracy on data more varied than the "
             "current 133-recording test set could plausibly be lower than reported, "
             "particularly for YAMNet.")

    assert P[153].text.startswith("The full confusion matrix")
    set_text(P[153], "The full confusion matrix for the current best model (SVM) is "
             "now included (Table 5, Section 6), computed on freshly re-extracted "
             "test-set features rather than any cached array, to rule out a "
             "known class of staleness bug in this project's own history.")

    # --- Future work: mark EfficientAT augmentation study as done, add new items ---
    assert P[164].text.startswith("Run the same five-seed")
    set_text(P[164], "Investigate the large train-vs-test accuracy gap identified for "
             "the deployed SVM and YAMNet models (Section 8) -- likely candidates are "
             "stronger regularization (lower SVM C, L2 penalty on the YAMNet "
             "classifier head) or more training data; determine whether it changes "
             "real-world accuracy on recordings outside the current 133-file test set. "
             "[The five-seed x augmentation stability study for EfficientAT, "
             "previously listed here, was completed 2026-08-27 -- see Section 6.3, "
             "Tables 7-8; it showed no statistically significant effect either way, "
             "unlike the six-model average.]")

    d.save(SRC)
    print("Saved stage 2.")


if __name__ == "__main__":
    main()
