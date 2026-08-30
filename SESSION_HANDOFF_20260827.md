# GarageAI — Session Handoff (2026-08-27/28)

Read this file fully before doing anything. It captures everything decided/built/found
in this session so a fresh conversation can continue without re-deriving it. This
supersedes the older `SESSION_HANDOFF.md` in scope (that file is still historically
accurate for what it covers, but this session did substantially more).

## 0. The four project folders (unchanged from before)

| Path | What it is |
|---|---|
| `C:\Users\hp\Desktop\car_test` | Model training/research code, raw audio, `processed_data/` (the deployed model artifacts). |
| `C:\Users\hp\Desktop\garageai-audio-analysis` | FastAPI inference microservice (port 8001). Loads models from `car_test\processed_data` via `MODELS_DIR` in `.env`. |
| `C:\Users\hp\Desktop\garageai-backend` | Node/Express gateway (port 5000). |
| `C:\Users\hp\Desktop\garageai-frontend` | React/Vite frontend (port 5173). |

Two deliverable documents live in `C:\Users\hp\Downloads\`:
- `GarageAI_Research_Paper.docx` — the academic paper.
- `GarageAI_Documentation.docx` — full engineering documentation (Chapters 1-9 summary + Appendix A = the paper reproduced in full + Appendix B = UML diagrams).
Both have `.BACKUP_BEFORE_EDIT.docx` copies from before this session's edits, in the
same folder, in case anything needs reverting.

## 1. What this session did, in order

1. **Ran a comprehensive 90-run augmentation/seed-stability study** across all nine
   individual models (Traditional ML, CNN, YAMNet, PANNs, AST, CLAP, PaSST, BEATs,
   EfficientAT), 5 seeds each, with/without waveform augmentation, all on the FULL
   626-recording training set (previously only 4 models had this study, and only on a
   201-file subset for 3 of them). New scripts: `run_traditional_ml_augmentation_variance.py`,
   `summarize_full_benchmark.py`. Extended `run_augmentation_variance_study.py` (added
   yamnet/panns model keys + `--full-data` flag) and `run_cnn_augmentation_variance.py`
   (added `--conditions` arg). Results: `processed_data/full_benchmark_summary.csv` and
   the per-model CSVs alongside it.
   - **Verdict: no model benefited from augmentation.** Traditional ML, AST, and PaSST
     showed a real, statistically significant regression; the other six were within noise.
   - **Found and fixed a real bug** while running this: `SVC(probability=True)` in the
     classifier-selection code of 7 files (`run_augmentation_variance_study.py` +
     `train_ast/beats/clap/panns/passt/transfer_learning/other_class_detector.py`)
     triggers scikit-learn's internal 5-fold Platt-scaling calibration even when only
     `.predict()` is used — a known pathological slowdown, confirmed to hang for **17+
     CPU-hours** on one config before being caught and killed. Fixed everywhere by
     deferring `probability=True` to a refit of the winning model only. **This fix is
     committed** (see Section 4).
2. **Ran a direct overfitting check** (`check_overfitting.py`) on the actually-deployed
   models, using freshly re-extracted features (not cached arrays, to avoid a known
   staleness-trap class of bug). Found a real, previously-unmeasured gap: SVM 99.7% train
   vs 86.5% test/val (13-16pt gap), YAMNet 100.0% train vs 78.2%/75.4% (22-25pt gap). CNN's
   gap is much smaller (~1-10pt). This is now documented as a Limitation + Future Work item
   in both `.docx` files, but **no mitigation (e.g. stronger regularization) has been applied
   yet** — still open.
3. **Verified the confusion matrix** for the deployed SVM model directly
   (`compute_confusion_matrix.py`), computed on freshly re-extracted test features. Matches
   the already-published per-class precision/recall/F1 numbers exactly.
4. **Wired PaSST and BEATs into the live inference service** as opt-in "compare with an
   extra model" options (AST and CLAP were already there from an earlier session). Changes
   span all three tiers:
   - `garageai-audio-analysis`: new `app/services/optional_models.py` entries, `beats_vendor/`
     and `efficientat_vendor/` folders, `ensemble.py`/`predict.py`/`config.py`/`model_loader.py`
     updates (see Section 4 for exact commit/uncommitted status).
   - `garageai-frontend`: `ExtraModel` type extended to `'ast'|'clap'|'passt'|'beats'` in
     `api.ts`, dropdown in `Home.tsx`.
   - **Also found, while reviewing the diff, substantial unrelated reliability engineering
     already present but never committed**: a `PREDICT_MAX_CONCURRENCY` semaphore, a
     temp-file-cleanup race-condition fix (Windows PermissionError on timeout), an ffmpeg
     subprocess timeout, and `vite.config.ts`'s `strictPort: true`. All verified via the
     full test suite (73/73 backend tests pass) and a clean frontend production build.
   - **Also found the whole "other"-class gate feature** (4th class: RandomForest, 95%
     4-class test accuracy, `P(other)>=0.46` threshold, wired into `predict.py`/`ensemble.py`/
     the frontend) had been built in an earlier session but was **never documented anywhere**.
5. **Rewrote large parts of both `.docx` deliverables** to fix real factual errors and add
   genuinely major, previously-undocumented findings from the project's history. Full list
   in Section 2 below.
6. **Reviewed and tested (but did not commit)** all pending changes in
   `garageai-audio-analysis` and `garageai-frontend` — see Section 4, this is the single
   most important pending action.
7. **Investigated whether the production ensemble should include
   AST/CLAP/PaSST/BEATs/EfficientAT** (the user explicitly asked for this). Found a real,
   validated improvement achievable **at zero extra cost** — see Section 3, this is the
   second most important pending action (the change is validated but NOT YET WRITTEN to
   the live `ensemble_config.pkl` — a permission classifier blocked the write, see Section 3).

## 2. Full list of `.docx` fixes/additions this session (both files, unless noted)

**Factual corrections** (both files had these wrong, inherited from an earlier version):
- Train/val/test split: was claimed "80/20 train/test", actually 70%/15%/15%
  (626/134/133 recordings) — `preprocessing.py`'s real split ratio.
- Hardware: was claimed "GPU acceleration used for CNN training" — this machine is
  **CPU-only**, confirmed directly (`torch.cuda.is_available()==False`).
- The live service's deployed model roster (Section 5.1/similar) was stale — updated to
  reflect PANNs+EfficientAT always-on, AST/CLAP/PaSST/BEATs opt-in (as of this session).
- Software/libraries table was missing `transformers`, `hear21passt`, `torch`,
  `better-sqlite3`, `jsonwebtoken`, `nodemailer` — added with verified exact names from
  `requirements.txt`/`package.json`.
- Confusion matrix was listed as "pending re-verification" in ~4 different places across
  both documents — now includes the real, freshly-computed one (a proper Table + discussion).
- `FR-6` and the Python-service description (documentation.docx only) still enumerated
  only the original 4 models — updated.
- One embedded PNG diagram (the 3-tier architecture figure, `Figure 1`/`Figure 3.1`, same
  image reused in both files) had a stale "Loads trained models: ...PANNs, Ensemble" caption
  baked into the image — regenerated with matplotlib and swapped in place (same technique
  used for two more stale diagrams in documentation.docx's Appendix B: `Figure 3.4` said
  "264 features" instead of 269, `Figure 3.5`'s component diagram listed a stale model list).

**Major new content — three previously-undocumented findings from the project's real
history, each added as a full subsection with real numbers (not summaries)**:
1. **Compression-robustness / "audio fingerprint" investigation** (paper Section 3.3.2,
   documentation Section 4.5 + Appendix A copy). The deployed model's accuracy collapsed
   under the real production audio pipeline (browser → WebM/Opus compression → ffmpeg
   decode) — Brake fell from 97.7% clean to 18.2% at 32kbps — traced to the model relying
   on recording-chain artifacts rather than the acoustic signature. Fixed via
   `compression_augment.py` (round-trip TRAIN files through the real pipeline). Verified
   numbers (recomputed fresh from `mic_pipeline_test/compare_old_vs_fix_results.csv` and
   `full_ensemble_compression_results.csv`, not from memory): clean 89.5%→86.5%,
   webm32 62.4%→88.0%, Brake@webm32 18.2%→88.6%. Full ensemble: 86.7% clean → 93.3% under
   compression. **This compression-augmented config is what's actually deployed today.**
2. **Ensemble stacking-vs-weighted-average fix** (paper Section 3.4.5, documentation
   Section 4.6). A real user-reported bug ("Brake" at 55% confidence with wildly disagreeing
   models) traced to (a) a stale-artifact reproducibility trap and (b) the ensemble's own
   stacking-vs-weights selection being noise-driven on a 134-sample validation set (single
   split: stacking "won" by 0.67 points; nested 5-fold CV: stacking won only 3/5 folds, not
   a real advantage). Fixed by switching to weighted averaging (0.30 Traditional/0.60 CNN/
   0.10 YAMNet — **this is the currently-deployed config, as of before this session's new
   finding in Section 3 below**).
3. **Freesound external-data investigation, rejected** (paper Section 3.3.3, documentation
   Section 4.7). Tested merging 520+183 license-filtered Freesound recordings across 4
   models; only PANNs benefited (81.95%→84.96%, heavily-filtered subset only); found the
   deployed model scores 0/194 on Freesound Brake clips despite 100% on its own test set.
   Excluded from training as a result — reported as a rigorous negative result, not a gap.

Both files' **Abstract, Contributions list (research paper), Conclusion, and
Limitations/Future Work sections** were updated to reference all of the above. All heading
numbering was checked end-to-end after each insertion (no gaps/duplicates). All new tables
use the project's real, verified numbers (never fabricated).

**Known still-open gaps in the `.docx` files** (told to the user already, not fixed):
- 3 screenshot placeholders in documentation.docx Chapter 7 (User Guide) are still just
  text placeholders ("🖼 Insert Screenshot: ..."), not real images. Could be filled by
  actually running the app and capturing screenshots — not done yet.
- `GarageAI_Research_Paper_Final.pdf` in Downloads is stale (pre-dates all of this
  session's edits) — needs re-exporting from the now-current `.docx` via Word before
  submission, if a PDF is what gets submitted.
- The two new ensemble-related findings from Section 3 below (if deployed) are NOT YET
  reflected in either `.docx`'s ensemble-weight numbers (0.30/0.60/0.10, 89.47%) — those
  numbers appear in many places (Abstract, 3.4.4, Table 4, Discussion, Conclusion, and the
  documentation's mirrors of all of these) and would need a careful, thorough find-and-update
  pass if the new config gets deployed. **Not done — flagged as the most likely next
  documentation task.**

## 3. Ensemble expansion investigation — validated, NOT YET deployed

The user explicitly asked to investigate folding AST/CLAP/PaSST/BEATs/EfficientAT into the
production ensemble's fused weights (not just showing them standalone). Full methodology:

1. `evaluate_ensemble.py` already auto-detects all 9 models' saved val/test probability
   files (`processed_data/*_val_probs.npy` / `*_test_probs.npy` all already exist) and has
   a code comment warning not to naively re-run it now that all 9 are available (overwrites
   the live `ensemble_config.pkl` with an untested result). Ran it via a **safe copy**
   (`evaluate_ensemble_9model_analysis.py`, writes to `*_9model_ANALYSIS.*` files, never
   touches the real config) to see what a naive single-split search finds.
2. **Naive result**: weights concentrate entirely on CLAP=0.60 + EfficientAT=0.40 (everything
   else zero), validation macro-F1 looks amazing (0.9402) — but on the untouched TEST set
   this scores only **88.72%, actually WORSE than the currently-deployed 89.47%**. Classic
   overfitting-to-a-134-sample-validation-set signature, exactly the failure mode already
   documented from the 2026-08-11 ensemble fix (Section 2, item 2 above) — expanding to 9
   models makes this worse (more free weight parameters, same tiny validation set).
3. Built `nested_cv_ensemble_9model.py`: proper nested 5-fold CV (search weights on 4 folds,
   score on the held-out 5th, repeat) comparing (a) deployed fixed weights, (b) re-searching
   among just the 4 already-deployed models, (c) searching among all 9. Result:
   - Deployed fixed: mean=0.8596, std=0.1054
   - 4-model re-search: mean=0.8448, std=0.1087 (confirms re-searching the SAME 4 models
     doesn't help — matches the 2026-08-11 finding)
   - 9-model search: mean=0.9030, std=0.0713 (better mean AND more stable — a REAL signal,
     properly out-of-fold validated, unlike the naive single-split result)
4. Took the **average of the 5 folds' own winning weight vectors** (a form of regularization,
   avoiding the naive search's winner-take-all extremity) as the final candidate, evaluated
   it ONCE on the untouched test set (disciplined — no further test-set iteration after this):
   `{Traditional ML: 0.08, CNN: 0.06, YAMNet: 0.02, CLAP: 0.54, BEATs: 0.04, EfficientAT: 0.26}`
   → **91.0% test accuracy / 91.0% macro-F1** — a real +1.5pt improvement over deployed.
5. **The catch, raised directly by the user**: CLAP (weight 0.54, the dominant model in this
   config) is currently opt-in-only in the live service, specifically because loading it costs
   ~20-30s extra latency per request (or, if kept resident instead, real RAM on an already
   RAM-constrained 6GB dev machine — free RAM was measured as low as 62MB with just AST
   resident). Deploying this config would require breaking the "opt-in for RAM/latency
   reasons" architecture decision for CLAP specifically.
6. **Resolved this cleanly**: re-ran the same nested-CV analysis restricted to ONLY the
   models that are ALREADY always-loaded (Traditional ML, CNN, YAMNet, PANNs, EfficientAT —
   no CLAP, no architecture change of any kind needed). Result:
   - 5-model-always-on search: mean=0.9066, std=0.0784 (as good as or better than the
     9-model version!)
   - Averaged fold weights: `{Traditional ML: 0.14, CNN: 0.18, YAMNet: 0.0, PANNs: 0.04,
     EfficientAT: 0.64}`
   - **Test set result: 90.23% accuracy / 90.24% macro-F1** — a real +0.76pt improvement
     over the deployed 89.47%, at **zero additional cost** (every one of these 5 models is
     already computed on every single prediction today; only the fusion weights change).
7. **Recommendation, already communicated to the user and agreed**: deploy the free 5-model
   config `{Traditional ML: 0.14, CNN: 0.18, PANNs (CNN14): 0.04, EfficientAT (fine-tuned): 0.64}`
   (YAMNet dropped since its weight is exactly 0.0, matching how PANNs was previously excluded
   at 0.0 weight in the old config's `names` list). This is strictly better than the current
   deployed config with no downside — the CLAP-containing 91.0% config was explicitly
   rejected in favor of this free option given the latency/RAM tradeoff.

**BLOCKED: the actual write to `processed_data/ensemble_config.pkl` (and even just a backup
copy of the old one) was refused by Claude Code's auto-mode permission classifier** — writing
to this specific production model-config file via a Bash/`python -c` one-liner was flagged
and denied, twice, even for a pure backup copy with no destructive write. **This is the
single most important pending action for the next session.**

### Exact next steps to finish this (for the next session)

```python
# 1. BACK UP the current config first (do this as an explicit, visible step the user can
#    see and approve, not hidden inside a larger one-liner -- that may be what triggered
#    the classifier block).
import pickle, shutil, datetime
stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
shutil.copy('processed_data/ensemble_config.pkl', f'processed_data/ensemble_config_BACKUP_{stamp}.pkl')

# 2. Write the new, validated config.
new_cfg = {
    'mode': 'weights',
    'names': ['Traditional ML', 'CNN', 'PANNs (CNN14)', 'EfficientAT (fine-tuned)'],
    'weights': (0.14, 0.18, 0.04, 0.64),
    'meta_model': None,
}
with open('processed_data/ensemble_config.pkl', 'wb') as f:
    pickle.dump(new_cfg, f)
```

Then:
- **Restart** the `garageai-audio-analysis` service (it reads `ensemble_config.pkl` once at
  startup into `state`, not per-request -- a running service needs a restart to pick this up).
  `ensemble.py`'s `predict()` function already reads weights/names dynamically from
  `state["ensemble_names"]`/`state["ensemble_weights"]` -- **no code changes needed there**,
  this is purely a data-file change.
- Verify with a real end-to-end prediction (a known belt/brake/sway test file through
  `/predict`, confirm `final_prediction`/`final_confidence` look sane and
  `individual_models` still lists all 5 always-on models plus whatever `extra_model` was
  requested).
- Update `car_test/evaluate_ensemble.py`'s honest report file (`ensemble_report.txt`) to
  reflect the new config if you want `processed_data/` to stay internally consistent (low
  priority, cosmetic).
- **Update both `.docx` files**: search for `89.47`, `89.33`, `0.30`, `0.60`, `0.10` (as
  ensemble weights specifically, not other unrelated numbers) across every location listed
  in Section 2's "known still-open gaps" above, and replace with the new
  90.23%/90.24%/0.14/0.18/0.04/0.64 numbers, plus a short paragraph explaining the
  free-improvement rationale (reuse the explanation in Section 3 above, condensed). This
  touches many places (Abstract, 3.4.4/3.4.5, Table 4, Discussion 7.2, Conclusion, and
  documentation's Ch4.4/4.6 + its Appendix A mirror) — go through them methodically, the
  same way the other big documentation passes in this session were done (search first, list
  every hit, then fix each one, then re-read the whole document once at the end to check
  for consistency).

## 4. Git status — the single most important pending action overall

**Neither `garageai-audio-analysis` nor `garageai-frontend` has been committed.** Both were
fully reviewed (every diff read line-by-line) and tested in this session:
- `garageai-audio-analysis`: **73/73 pytest tests pass**, including new tests for the
  "other" gate, optional-model loading, and the concurrency/timeout logic.
- `garageai-frontend`: `npm run build` succeeds cleanly.

Modified/new files (as of this session, git status via PowerShell -- **note: this session's
Bash tool had a broken PATH for `git`/coreutils partway through; use PowerShell for git
commands if Bash's `git`/`rm`/`cat`/`tail`/`wc` report "command not found"**):

`garageai-audio-analysis`: `.env.example`, `app/core/config.py`, `app/routes/predict.py`,
`app/services/audio_converter.py`, `app/services/ensemble.py`, `app/services/model_loader.py`,
`requirements.txt`, `tests/test_ensemble.py`, `tests/test_predict.py` (all modified) +
`app/services/beats_vendor/`, `app/services/efficientat_vendor/`,
`app/services/optional_models.py`, `tests/test_model_loader.py` (all new/untracked). A stray
`backend_startup.log` was already cleaned up.

`garageai-frontend`: `src/app/context/AudioAnalysisContext.tsx`, `src/app/lib/api.ts`,
`src/app/pages/Chat.tsx`, `src/app/pages/History.tsx`, `src/app/pages/Home.tsx`,
`vite.config.ts` (all modified, no new files).

**Recommended commit split** (matching the pattern already used successfully for `car_test`
earlier in this session -- see that repo's `git log` for the style/message format to match):
1. One commit for the PaSST/BEATs wiring + `optional_models.py` (this session's own work).
2. One commit for the reliability engineering (concurrency cap, temp-file race fix, ffmpeg
   timeout, `strictPort`) -- pre-existing, uncommitted work from an earlier session, reviewed
   and verified working in this one.
3. One commit for `tests/test_model_loader.py` + the "other"-gate-related pieces of
   `model_loader.py`/`ensemble.py` if not already covered by commit 2 (check for overlap
   before splitting -- `model_loader.py`'s diff includes BOTH `_load_other_gate` and the
   `_load_panns_model`/`_load_efficientat_model` refactor in one diff; decide whether to
   split them or keep as one commit, whichever produces a cleaner history).
4. Frontend: likely one commit is fine (`AudioAnalysisContext.tsx`/`api.ts`/`Home.tsx` are
   the extra-model dropdown; `Chat.tsx`/`History.tsx`/`vite.config.ts` are unrelated
   pre-existing fixes -- could split into 2 commits if a cleaner history is wanted).

Ask the user before pushing to any remote, same as always.

## 5. car_test's own git status (already handled, for reference only)

`car_test` itself was already fully committed and pushed to `origin/main` earlier in this
session (8 commits: the 90-run benchmark, the "other"-class detector, a `check_data_leakage.py`
fix, `run_efficientat_augmentation_study.py`, and the large/regenerable-data untracking +
`.gitignore` update). Nothing pending there as of this writing, **except** whatever new
scripts this session added afterward that haven't been committed yet:
`check_overfitting.py`, `compute_confusion_matrix.py`, `edit_documentation_stage*.py`,
`edit_research_paper*.py`, `regenerate_figure1.py`, `regenerate_fig34_35.py`,
`evaluate_ensemble_9model_analysis.py`, `evaluate_ensemble_ANALYSIS_ONLY.py` (a leftover copy
attempt, safe to delete -- superseded by `evaluate_ensemble_9model_analysis.py`),
`nested_cv_ensemble_9model.py`, and `SESSION_HANDOFF_20260827.md` (this file) -- plus the
`evaluate_ensemble_report_9model_ANALYSIS.txt` / `ensemble_confusion_matrix_9model_ANALYSIS.png`
/ `ensemble_config_9model_ANALYSIS.pkl` analysis-output files these produced. Worth a look
before deciding what's worth committing vs. is just scratch work (the `edit_*.py` scripts
are genuinely useful as a record of exactly what was changed in the `.docx` files and why --
probably worth keeping and committing, similar to how the paper itself treats its own
methodology scripts as first-class project artifacts).

## 6. How to resume in a new conversation

Paste/reference this file's path (`C:\Users\hp\Desktop\car_test\SESSION_HANDOFF_20260827.md`).
Priority order:
1. Deploy the ensemble-weights fix (Section 3) -- back up, write, restart service, verify.
2. Commit `garageai-audio-analysis` and `garageai-frontend` (Section 4).
3. Update both `.docx` files with the new ensemble numbers (Section 3's last bullet).
4. Take real screenshots for documentation.docx's Chapter 7 placeholders, if wanted.
5. Re-export the research paper PDF from the updated `.docx`, if a PDF is what gets submitted.
