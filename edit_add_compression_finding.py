# -*- coding: utf-8 -*-
"""Adds the compression-robustness / domain-shift investigation (a major, real
finding from this project's history that was never documented in either the
paper or the documentation) as a proper new subsection 3.3.2, plus mentions in
the Abstract, Contributions, and Conclusion. Numbers verified directly from
mic_pipeline_test/compare_old_vs_fix_results.csv and
full_ensemble_compression_results.csv (not from memory)."""
import copy
import docx
import docx.table

SECTION_HEADING = "3.3.2 Compression-Robustness Investigation and Fix"

DIAGNOSTIC_PARA = (
    "A gap was observed between this project's own held-out test accuracy "
    "(89.5% clean, traditional ML) and the system's perceived accuracy in "
    "informal real-world use through the actual web application. To "
    "investigate, a dedicated diagnostic took real, held-out test recordings "
    "(never seen during training, labels certain) and ran them through the "
    "exact same production audio pipeline the live app uses -- encoding to "
    "WebM/Opus via the identical ffmpeg binary and flags the browser-to-backend "
    "path relies on, at bitrates bracketing typical browser MediaRecorder "
    "output (16/32/64 kbps), then decoding back exactly as the backend does "
    "before feature extraction. The result was a large, previously invisible "
    "accuracy collapse on the same 133 test recordings: overall accuracy fell "
    "from 89.5% clean to 62.4-65.4% once compressed, and the Brake class "
    "specifically was devastated -- from 97.7% clean accuracy to just 18.2% "
    "at 16 and 32 kbps, barely above chance. Since the true labels were never "
    "in doubt and the model itself was unchanged, this pattern indicated the "
    "model had partly learned to rely on artifacts of the original recording "
    "and encoding chain of its training data (much of it collected from "
    "public online sources, not one consistent recording setup) rather than "
    "the acoustic signature of the fault itself -- effectively a spurious "
    "audio fingerprint of how a clip was originally recorded and re-encoded, "
    "rather than the belt/brake/sway sound proper."
)

FIX_PARA = (
    "The fix round-trips each TRAINING recording (never validation or test) "
    "through the identical real WebM/Opus encode-decode pipeline at multiple "
    "bitrates, adding these compressed variants as extra training examples "
    "alongside the original. This is deliberately narrower and more "
    "evidence-driven than the generic waveform augmentation in Section 3.3.1 "
    "(already shown not to help): rather than adding arbitrary perturbations, "
    "it specifically forces the model to keep relying on features that "
    "survive the real compression pipeline, directly countering the "
    "mechanism diagnosed above. Retraining the Traditional ML and CNN models "
    "on this compression-augmented data and re-testing on the identical 133 "
    "recordings -- compressed through the identical bytes for both the old "
    "and new model, eliminating any confound -- confirmed the fix (Table 1a)."
)

TABLE_CAPTION = (
    "Table 1a. Test accuracy before and after compression-augmented "
    "retraining, on the identical 133 recordings compressed through the "
    "identical real WebM/Opus pipeline for both models."
)

TABLE_ROWS = [
    ["Condition", "Original model", "Compression-augmented model", "Delta"],
    ["Clean (no compression)", "89.5%", "86.5%", "-3.0 pts"],
    ["WebM/Opus 16 kbps", "63.9%", "85.0%", "+21.1 pts"],
    ["WebM/Opus 32 kbps", "62.4%", "88.0%", "+25.6 pts"],
    ["WebM/Opus 64 kbps", "65.4%", "85.0%", "+19.5 pts"],
    ["Brake class @ 32 kbps specifically", "18.2%", "88.6%", "+70.4 pts"],
]

ENSEMBLE_PARA = (
    "The full production ensemble (Traditional ML + CNN + YAMNet, weighted "
    "0.30/0.60/0.10) was independently re-tested end-to-end on the same real "
    "compression pipeline and confirmed robust: 86.7% clean accuracy actually "
    "rose to 93.3% under WebM compression at all three bitrates tested, the "
    "CNN's dominant ensemble weight acting as a safety net on the cases where "
    "the traditional model alone still flips. This compression-augmented "
    "configuration is the one currently deployed in production."
)

REJECTED_PARA = (
    "A further robustness step -- additionally simulating microphone/device "
    "frequency-response differences on top of compression augmentation -- "
    "was tested and rejected: it traded 3.0-4.5 points of the confirmed "
    "real-pipeline compression gain for a smaller gain on a synthetic "
    "device-filter axis not representative of real recording hardware, the "
    "same pattern by which generic waveform augmentation was already "
    "rejected in Section 3.3.1."
)

ABSTRACT_ADDITION = (
    " A separate investigation diagnosed and fixed a real-world robustness "
    "gap: the originally deployed model's accuracy on the project's own "
    "production audio pipeline (browser microphone recording compressed via "
    "WebM/Opus) collapsed under compression -- most severely for Brake "
    "(97.7% clean to 18.2% at 32 kbps) -- traced to reliance on recording-chain "
    "artifacts rather than the acoustic fault signature itself. A targeted, "
    "pipeline-matched compression augmentation recovered this gap almost "
    "entirely (88.0% vs. 62.4% at 32 kbps, Brake specifically to 88.6%) at a "
    "small cost to clean-audio accuracy, and this compression-augmented "
    "configuration is what is currently deployed in production."
)

CONTRIBUTIONS_BULLET = (
    "A diagnosed and fixed real-world domain-shift failure mode: the "
    "originally deployed model's accuracy collapsed under the WebM/Opus "
    "compression the live application's own audio pipeline actually applies "
    "(most severely for Brake, 97.7% to 18.2% at 32 kbps), traced to reliance "
    "on recording-chain artifacts rather than the acoustic fault signature; a "
    "targeted, pipeline-matched compression augmentation recovered the gap "
    "(88.0% vs. 62.4% at 32 kbps) and is the configuration now in production."
)

CONCLUSION_ADDITION = (
    " A separate investigation also diagnosed and fixed a real-world "
    "robustness gap invisible to the standard clean-audio test-set evaluation: "
    "accuracy collapsed under the actual WebM/Opus compression the live "
    "application's audio pipeline applies (Brake fell to 18.2% at 32 kbps), "
    "traced to the model relying on recording-chain artifacts rather than the "
    "acoustic fault signature -- a targeted, pipeline-matched compression "
    "augmentation recovered this almost entirely (88.0% vs. 62.4% at 32 kbps) "
    "and is the configuration currently deployed in production."
)


def set_text(p, text):
    if not p.runs:
        p.add_run(text)
        return
    p.runs[0].text = text
    for r in p.runs[1:]:
        r.text = ""


def clone_pPr(target_p, source_p):
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}pPr"
    old = target_p._p.find(ns)
    if old is not None:
        target_p._p.remove(old)
    src = source_p._p.find(ns)
    if src is not None:
        target_p._p.insert(0, copy.deepcopy(src))


def insert_after(anchor, text, style_source=None):
    new_p = anchor.insert_paragraph_before("")
    anchor._p.addnext(new_p._p)
    new_p.add_run(text)
    if style_source is not None:
        clone_pPr(new_p, style_source)
        new_p.style = style_source.style
    return new_p


def insert_table_after(anchor_p, doc, rows, template_table):
    new_tbl_elm = copy.deepcopy(template_table._tbl)
    tbl = docx.table.Table(new_tbl_elm, doc)
    # resize to len(rows) rows (including header)
    current = len(tbl.rows)
    if current < len(rows):
        template_row = tbl.rows[-1]._tr
        for _ in range(len(rows) - current):
            new_row = copy.deepcopy(template_row)
            template_row.addnext(new_row)
            template_row = new_row
    elif current > len(rows):
        for row in tbl.rows[len(rows):][::-1]:
            row._tr.getparent().remove(row._tr)
    ncols = len(tbl.columns)
    for r, row_vals in enumerate(rows):
        for c in range(ncols):
            tbl.cell(r, c).text = row_vals[c] if c < len(row_vals) else ""
    anchor_p._p.addnext(new_tbl_elm)
    return tbl


def add_section(doc):
    P = doc.paragraphs
    heading_src = next(p for p in P if p.text.startswith("3.3.1 Data Augmentation"))
    body_src = next(p for p in P if p.text.startswith("Augmented copies were generated"))
    # find a 4-column table to use as formatting template (the confusion-matrix one)
    template_table = None
    for t in doc.tables:
        if len(t.columns) == 4 and len(t.rows) >= 2:
            template_table = t
            break
    assert template_table is not None

    anchor = body_src
    anchor = insert_after(anchor, SECTION_HEADING, style_source=heading_src)
    anchor = insert_after(anchor, DIAGNOSTIC_PARA, style_source=body_src)
    anchor = insert_after(anchor, FIX_PARA, style_source=body_src)
    tbl = insert_table_after(anchor, doc, TABLE_ROWS, template_table)
    caption_anchor = anchor.insert_paragraph_before("")
    tbl._tbl.addnext(caption_anchor._p)
    set_text(caption_anchor, TABLE_CAPTION)
    if caption_anchor.runs:
        caption_anchor.runs[0].italic = True
    anchor = caption_anchor
    anchor = insert_after(anchor, ENSEMBLE_PARA, style_source=body_src)
    anchor = insert_after(anchor, REJECTED_PARA, style_source=body_src)

    # Abstract addition
    for p in P:
        if p.text.startswith("Against initial expectations, the traditional ML baseline"):
            if p.runs:
                p.runs[-1].text += ABSTRACT_ADDITION
            break

    # Contributions bullet
    for p in P:
        if p.text.startswith("A 90-run stability study across all nine individual models"):
            insert_after(p, CONTRIBUTIONS_BULLET, style_source=p)
            break

    # Conclusion addition
    for p in P:
        if p.text.startswith("This research evaluated an audio-based automotive fault classification system across ten configurations"):
            if p.runs:
                p.runs[-1].text += CONCLUSION_ADDITION
            break


if __name__ == "__main__":
    d = docx.Document(r"C:\Users\hp\Downloads\GarageAI_Research_Paper.docx")
    add_section(d)
    d.save(r"C:\Users\hp\Downloads\GarageAI_Research_Paper.docx")
    print("Research paper: compression-robustness section added.")
