# -*- coding: utf-8 -*-
"""Fixes GarageAI_Documentation.docx. Appendix A duplicates the research paper's
OLD (pre-2026-08-27-fix) text almost verbatim, so most fixes here are the exact
same substitutions already validated on GarageAI_Research_Paper.docx, just
located by text-match instead of fixed paragraph index (this doc's numbering
differs). Chapters 1-9 (the summary) have their own independently-worded
versions of the same stale facts and are fixed separately in stage 2."""
import docx

SRC = r"C:\Users\hp\Downloads\GarageAI_Documentation.docx"


def set_text(paragraph, new_text):
    if not paragraph.runs:
        paragraph.add_run(new_text)
        return
    paragraph.runs[0].text = new_text
    for r in paragraph.runs[1:]:
        r.text = ""


def find_para(paragraphs, startswith, occurrence=0):
    hits = [p for p in paragraphs if p.text.startswith(startswith)]
    if not hits:
        raise ValueError(f"No paragraph starts with: {startswith!r}")
    return hits[occurrence]


def find_all(paragraphs, startswith):
    return [p for p in paragraphs if p.text.startswith(startswith)]


def main():
    d = docx.Document(SRC)
    P = d.paragraphs

    fixes = 0

    # === Appendix A (duplicates old research-paper text) ===

    # GPU claim
    for p in find_all(P, "GPU acceleration used for CNN training"):
        set_text(p, "CPU only for every training run and inference call in this "
                 "project -- no GPU was available on the development machine at any "
                 "point (confirmed directly: torch.cuda.is_available() == False, "
                 "tf.config.list_physical_devices('GPU') == []).")
        fixes += 1

    # Split ratio
    for p in find_all(P, "The dataset consists of approximately 900 short vehicle"):
        old = p.runs[0].text
        new = old.replace(
            "allocated whole vehicle groups to train/validation/test partitions (80/20 train/test) while keeping class distribution balanced.",
            "allocated whole vehicle groups to train/validation/test partitions (70%/15%/15%) while keeping class distribution balanced -- yielding 626 training, 134 validation, and 133 test recordings."
        )
        assert new != old
        set_text(p, new)
        fixes += 1

    # Abstract (identical text to the paper's original abstract)
    for p in find_all(P, "Against initial expectations, two of the seven transfer-learning backbones outperformed"):
        set_text(p,
            'Against initial expectations, the traditional ML baseline (SVM, 86.47% accuracy, 86.30% macro-F1) was matched or exceeded by several transfer-learning backbones once training-set size was controlled for: in the original single-run comparison, only the Audio Spectrogram Transformer, AST (87.97% accuracy, 87.83% macro-F1), and EfficientAT -- the only model given genuine fine-tuning rather than a frozen embedding (88.72% accuracy, 88.76% macro-F1) -- exceeded it, while YAMNet, PANNs/CNN14, CLAP, PaSST, and BEATs scored at or below it. A 2026-08-27 follow-up retrained all nine models five times each, with and without waveform augmentation, on the full 626-recording training set (90 runs total, up from an initial 40-run/four-model pilot), and found that CLAP and PaSST also exceed the traditional baseline once given the same amount of training data as AST and EfficientAT (90.2% each), and BEATs matches it (86.5%) -- narrowing the domain-transfer gap to specifically YAMNet and PANNs/CNN14 among the seven backbones tested. This follow-up also found that none of the nine models benefited from waveform augmentation (three -- Traditional ML, AST, and PaSST -- were measurably hurt by it), that seed-to-seed variation alone can shift test accuracy by up to 8.3 points for a frozen-embedding model and by 30 points in one CNN outlier run, and that the currently deployed SVM and YAMNet models show a large gap between training and held-out accuracy (99.7% vs 86.5%, and 100.0% vs 78.2%, respectively) not previously measured in this study. The weighted ensemble of the traditional ML, CNN, and YAMNet models -- the only ensemble evaluated in production, not yet incorporating AST, CLAP, PaSST, or EfficientAT -- still achieved the best overall result at 89.47% accuracy (89.33% macro-F1). The central finding of this work is therefore more nuanced than a simple traditional-versus-deep-learning contrast: frozen, general-purpose audio embeddings pretrained at limited scale on this task can underperform a carefully engineered traditional model, but a sufficiently powerful transformer, genuine fine-tuning, or simply enough task-specific training data can close -- and in several cases exceed -- that gap. The trained pipeline was further deployed as a working three-tier system consisting of a Python inference backend, a Node.js gateway, and a React-based web frontend, demonstrating the practical feasibility of the proposed approach for real-world use.')
        fixes += 1

    # Contributions bullet (40-run)
    for p in find_all(P, "A dedicated 40-run stability study"):
        if "backend inference service" not in p.text:  # avoid accidental cross-match
            set_text(p, "A 90-run stability study across all nine individual models, "
                     "five random seeds, and with/without waveform augmentation -- "
                     "extended 2026-08-27 from an initial 40-run/four-model version -- "
                     "quantifying how sensitive small-test-set accuracy estimates are "
                     "to training-sample variation, confirming that waveform "
                     "augmentation did not benefit any of the nine models tested "
                     "(three showed a statistically real degradation), and resolving "
                     "the original study's training-set-size confound by retraining "
                     "every model on the full 626-recording set.")
            fixes += 1

    # 3.3.1 augmentation paragraph: pointer to "40-run stability study, Section 6.3" -- leave as
    # historical narrative (matches what was done in the standalone paper), no change needed.

    # 5.1 Hardware -- "only PANNs loaded"
    for p in find_all(P, "Sufficient RAM to hold the full engineered feature matrix"):
        old = p.runs[0].text
        new = old.replace(
            "PANNs was the only additional backbone actually loaded into the live "
            "inference service and profiled there, with RAM usage peaking at 233MB "
            "during model loading before settling at approximately 527MB. AST, CLAP, "
            "PaSST, BEATs, and EfficientAT were evaluated offline and are reported but "
            "not yet loaded into the live service.",
            "PANNs and EfficientAT were the first two additional backbones loaded "
            "into the live inference service (RAM peaking at 233MB during PANNs "
            "loading, settling at approximately 527MB); both now run on every "
            "prediction by default, alongside the original Traditional ML, CNN, and "
            "YAMNet models. As of 2026-08-27, AST, CLAP, PaSST, and BEATs are also "
            "reachable in the live service, but as an opt-in \"compare with an extra "
            "model\" selector (one loaded per request, not run by default) rather than "
            "always-on. None of these five are yet included in the production "
            "ensemble's fused weights (Section 3.4.4), which remain Traditional "
            "ML/CNN/YAMNet only."
        )
        if new != old:
            set_text(p, new)
            fixes += 1

    d.save(SRC)
    print(f"Stage 1 done. {fixes} paragraph(s) fixed.")


if __name__ == "__main__":
    main()
