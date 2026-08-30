# -*- coding: utf-8 -*-
"""Stage 4: reconcile the paper against the ACTUAL current state of all four
project folders (car_test, garageai-audio-analysis, garageai-backend,
garageai-frontend) checked directly on 2026-08-27 -- not just car_test's own
results. Two real gaps found: (1) the paper says only PANNs is loaded into the
live service and AST/CLAP/PaSST/BEATs/EfficientAT are "not yet" deployed --
false, all six are now reachable (EfficientAT + PANNs always-on via .env,
AST/CLAP/PaSST/BEATs as an opt-in extra_model selector wired into the FastAPI
service, the Node.js gateway, and the React frontend's dropdown). (2) A whole
deployed feature -- a real supervised 4th "other" class gate, spanning all
three tiers -- is never mentioned anywhere in the paper."""
import docx

SRC = r"C:\Users\hp\Downloads\GarageAI_Research_Paper.docx"


def set_text(paragraph, new_text):
    if not paragraph.runs:
        paragraph.add_run(new_text)
        return
    paragraph.runs[0].text = new_text
    for r in paragraph.runs[1:]:
        r.text = ""


def insert_paragraph_after(anchor, text):
    new_p = anchor.insert_paragraph_before("")
    anchor._p.addnext(new_p._p)
    new_p.add_run(text)
    return new_p


def main():
    d = docx.Document(SRC)
    P = d.paragraphs

    # --- 4.1: add the "other" class gate, a real deployed feature across all
    #     three tiers that this paper never otherwise mentions ---
    assert P[90].text.startswith("Implemented with FastAPI")
    insert_paragraph_after(P[90],
        "A separate, genuinely supervised 4th class -- \"other\" -- is checked "
        "before any of the belt/brake/sway models run: a RandomForest classifier "
        "trained on synthetic non-car signals (noise, tones, chirps, chords, "
        "engine hum, impulse trains, road noise, formant-approximated speech) plus "
        "real non-car audio (Windows system sounds), reaching 95.0% 4-class test "
        "accuracy and 100% held-out generalization to entirely unseen non-car sound "
        "types. Deployment uses a probability threshold (P(other) >= 0.46) rather "
        "than plain argmax -- tuned against both the model's own held-out \"other\" "
        "samples and a separate, stricter generalization probe set, after an "
        "initial threshold choice checked against only one of the two populations "
        "caused a real regression that a live spot-check caught before shipping. "
        "The gate is wired into predict.py, this backend's ensemble.py (where it "
        "short-circuits before any heavier model runs, saving compute), and the "
        "frontend, which renders a distinct \"this doesn't sound like a car fault\" "
        "state instead of forcing a belt/brake/sway guess. An earlier unsupervised "
        "approach (IsolationForest on MFCC/embedding features) was tried first and "
        "abandoned after it failed to generalize.")

    # --- 5.1: fix "only PANNs is loaded" -- now six models reachable ---
    assert "PANNs was the only additional backbone actually loaded" in P[104].text
    old = P[104].text
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
        "always-on -- the RAM/latency cost of running all four transformers on "
        "every request was judged not worth it given their offline results in "
        "Table 4 do not clearly exceed the always-on models. None of these five "
        "are yet included in the production ensemble's fused weights (Section "
        "3.4.4), which remain Traditional ML/CNN/YAMNet only."
    )
    assert new != old
    set_text(P[104], new)

    # --- Future work: "evaluate deploying AST/EfficientAT" is now half-done ---
    assert P[166].text.startswith("Evaluate deploying AST")
    set_text(P[166], "EfficientAT is now always-on in the live service and AST, "
             "CLAP, PaSST, and BEATs are reachable as an opt-in comparison model "
             "(Section 5.1); the remaining open question is whether any of them "
             "should move from opt-in to always-on, or be folded into the fused "
             "ensemble weights (Section 3.4.4), given the RAM/latency cost "
             "observed for PANNs and EfficientAT and the accuracy gains identified "
             "in the full-data follow-up (Section 6.3, Table 7).")

    d.save(SRC)
    print("Saved stage 4.")


if __name__ == "__main__":
    main()
