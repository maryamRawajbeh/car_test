# Car Fault Sound Classification — Session Handoff

Read this file fully before doing anything. It captures everything decided/built so work can continue without re-deriving it.

## 1. The real system (4 connected projects)

| Path | What it is |
|---|---|
| `C:\Users\hp\Desktop\car_test` | **Working/dev copy AND the real production model source** (see correction below) — all the improvements below were made HERE. |
| `C:\Users\hp\Downloads\car_test` | **STALE / NO LONGER EXISTS.** This file previously said the live microservice loads models from here — that was true at some earlier point but is **not current**. `Downloads\car_test` doesn't exist on this machine anymore (renamed/removed at some point outside this session's knowledge). A leftover `Downloads\car_test1\processed_data` exists but is an OLD artifact (84-dim scaler — predates even the 264/269-dim feature set) — **do not use it for anything.** |
| `C:\Users\hp\Desktop\car_test\processed_data` | **THE REAL DEPLOYED MODEL DIRECTORY** — confirmed directly from `garageai-audio-analysis\.env` (`MODELS_DIR=C:\Users\hp\Desktop\car_test\processed_data`) and `app\core\config.py`. This is what real users' predictions actually come from. It is a separate, independently-trained artifact set from `ablation_results\no_augmentation\` (different XGBoost hyperparams: deployed is `n_estimators=200, max_depth=6, lr=0.05`; the ablation reference used all session is `n_estimators=300, max_depth=3, lr=0.1`) — both are legitimate/valid, just don't confuse "the reference baseline used for today's A/B tests" with "what's actually live." |
| `C:\Users\hp\Desktop\garageai-audio-analysis` | FastAPI microservice (port 8001). Loads models from `car_test\processed_data` (see above — NOT Downloads). Has its own venv with its own `feature_extraction.py` that must stay byte-identical to `car_test`'s `audio_common.py`. |
| `C:\Users\hp\Desktop\garageai-backend` | Node/Express API (port 5000), proxies audio uploads to the FastAPI service (`PYTHON_SERVICE_URL` in `.env`). No changes needed for anything in this file. |
| `C:\Users\hp\Desktop\garageai-frontend` | React/Vite app (port 5173). Records mic audio via plain `new MediaRecorder(stream)` (no explicit mimeType/bitrate — browser default, `src/app/pages/Home.tsx`) → uploads → backend → FastAPI → `audio_converter.py`'s `convert_to_wav()` (ffmpeg, mono, resample to `config["target_sr"]`). No code changes needed for anything in this file — the domain-shift fix below is entirely model-side.

**Deployed ensemble weights (`car_test\processed_data\ensemble_config.pkl`): `{'names': ['Traditional ML', 'CNN', 'Transfer Learning (YAMNet)'], 'weights': (0.8, 0.2, 0.0)}`.** This matters a lot: the REAL final prediction returned to users is **0.8×Traditional-ML + 0.2×CNN only** — YAMNet is weighted zero and PANNs isn't even in the list (see `garageai-audio-analysis/app/services/ensemble.py`'s `predict()` — it filters ensemble weights to `state["ensemble_names"]`, so PANNs/YAMNet get computed and returned in `individual_models` for display but never influence `final_prediction`). Practical upshot: fixing Traditional ML + CNN (which this session did, see section 5) fixes 100% of what currently reaches real users. YAMNet/PANNs are not currently load-bearing.

**Original training data provenance (learned directly from the user, not previously known):** most of the belt/brake/sway audio this whole project trains on was itself sourced from **Reddit** (not clean mechanic-recorded audio). This matters for interpreting section 3/5's findings — a model trained on one internet platform's audio can end up sensitive to *that platform's* compression/encoding fingerprint rather than purely the acoustic event, which is consistent with (and possibly part of the explanation for) the brake-class fragility found in section 5.

## 2. What was built/fixed this session (all in `Desktop\car_test` unless noted)

- **`audio_common.py`**, **`cnn_model.py`**, **`preprocessing.py`** (parallelized, HPSS features, fixed `aug_time_shift`), **`train_traditional_ml.py`**, **`train_cnn.py`**, **`train_transfer_learning.py`** (YAMNet + `CalibratedClassifierCV`), **`evaluate_ensemble.py`**, **`predict.py`**, **`run_augmentation_ablation.py`**, **`diagnose_augmentation_techniques.py`** — see git history / file headers for details, unchanged from earlier in this session.
- **`sounds_curation\`** (select_candidates.py, review_app.py, score_candidates.py, select_brake_dataset_candidates.py, run_ab_test.py, run_ab_test_cnn.py, run_ab_test_panns.py, run_ab_test_yamnet.py, curation_common.py) — the full Freesound-data investigation. **Concluded, see section 3.** Tooling still exists and works but the review app was never actually used for manual review (see section 3 — the automated A/B tests answered the question before manual review was needed).
- **`compression_augment.py`** (new) — `compress_roundtrip(y, sr, bitrate_kbps)`: round-trips a waveform through WebM/Opus encode + decode using the **exact same ffmpeg binary and flags** as the real app's `audio_converter.py` (via `imageio-ffmpeg`, installed into `car_test\venv`). Has a 30s subprocess timeout (added after code review) so a hung ffmpeg can't stall a whole preprocessing run silently. This is the training-side fix for the domain-shift problem in section 5.
- **`preprocessing.py`** — extended with `COMPRESSION_AUGMENT_ENABLED` (env: `PREPROCESSING_COMPRESSION_AUGMENT=1`) and `COMPRESSION_BITRATES_KBPS` (env: `PREPROCESSING_COMPRESSION_BITRATES`, default `16,32,64`). When enabled, each TRAIN file (never val/test) gets one extra variant per configured bitrate, built by compression-round-tripping the clean deterministic crop. **Deliberately independent** of the existing `N_AUGMENTATIONS_PER_TRAIN_FILE` waveform augmentation (time_shift/volume/noise) — that one is generic and already proven to hurt (section 3); this one is narrow and evidence-driven (section 5). Run with `PREPROCESSING_N_AUGMENTATIONS=0` to test compression augmentation in isolation, which is what this session did.
- **`mic_pipeline_test\`** (new folder) — `test_mic_pipeline_domain_shift.py` (the original 15-file diagnostic) and `compare_old_vs_compression_fix.py` (the full 133-file old-vs-new verdict). See section 5.
- **`run_cnn_augmentation_variance.py`** — generalized to accept any two `ablation_results\<name>\` conditions via the `CONDITIONS` list (currently set to `["no_augmentation", "compression_augmented"]`). Retrains the CNN N times (`--repeats`, default 5) per condition with a different seed each time and reports mean/std, not a single run — built after discovering real run-to-run CNN training variance (one `no_augmentation` run landed at 57.14% accuracy vs. 82-86% for other seeds, same everything else).
- Installed into `car_test\venv` this session: `imageio-ffmpeg` (self-contained ffmpeg for compression_augment.py). Microsoft Visual C++ Redistributable x64 was also installed **system-wide** (fixed both `soxr` — librosa's resampler — and TensorFlow/PyTorch, which were all failing with DLL-not-found errors; root cause was the same missing redistributable for all three).

## 3. Freesound external-data investigation — CONCLUDED, do not re-litigate without new evidence

**Verdict: do not blanket-merge Freesound data into training.** Tested empirically across all 4 models (not assumed):

Sources tested: `C:\Users\hp\Desktop\sounds\` (520 files, belt/brake/sway, confirmed via `freesound_downloader (2).py` to be Freesound.org preview downloads) + `C:\Users\hp\Desktop\brake_dataset_freesound\accepted\` (183 files, brake only, license-filtered to 135 commercial-safe CC0/CC-BY — 48 were CC-BY-NC and excluded outright). Both pools scored against the deployed-style model for label agreement, then A/B tested by actually adding curated candidates to TRAIN and evaluating on the untouched original TEST split:

| Model | Baseline (clean) | Best arm found | Best result | Verdict |
|---|---|---|---|---|
| XGBoost (traditional ML) | 93.23% | both pools, model-agreement-filtered (237 rows) | 93.23% | No help, parity at best; unfiltered data measurably hurts (-2.26%) |
| CNN | 87.22% (single run — see variance caveat below) | any arm | 84.96% best | **Hurts in 6/6 arms tried**, no exceptions |
| **PANNs** | 81.95% | both pools, model-agreement-filtered (237 rows) | 84.96% | **The only model that benefited, consistently** |
| YAMNet | 81.95% | brake_dataset_freesound, unfiltered | 83.46% | Mixed/inconsistent, small effect (test set is only 133 files — a couple of flipped predictions moves the number ~1.5%) |

Also notable en route: the deployed-style model has **100% brake mismatch** (0/194) against `sounds\`'s Freesound brake clips (mean predicted probability for "brake" ≈ 0.007, i.e. never even close) despite **100% recall/precision on the real held-out test set**. Against the more rigorously pre-QA'd `brake_dataset_freesound` pool, it does slightly better (11/135 correct) but still mostly fails. This was the first clue (later confirmed decisively, see section 5) that the model may be relying on non-causal recording-condition artifacts for the brake class specifically, not purely the acoustic signature.

**Practical takeaway:** the `sounds_curation\` tooling (candidate window selection + model-agreement scoring + review app) is real, working infrastructure, but manual review was never needed — the automated A/B test gave a clean enough answer first. If ever revisited: only PANNs showed genuine benefit, and only from the heavily-filtered (237-row) combined-pool subset.

## 4. Real-world domain-shift discovery + fix — DONE, pending final verification (section 6) before deploy

**The actual root cause of "the app feels inaccurate in real use."** Not a data-quantity problem — a recording-condition sensitivity problem.

### Discovery
Built `mic_pipeline_test\test_mic_pipeline_domain_shift.py`: took known TEST-split files (labels certain, never trained on) from the REAL deployed model's own split, ran them through the **exact real production pipeline** (encode to WebM/Opus via the same bundled ffmpeg the app uses, at bitrates bracketing typical browser `MediaRecorder` output, then decode via the same `convert_to_wav()` ffmpeg command `audio_converter.py` uses), and compared predictions before/after.

Result on 15 files: clean 93.3% → even a **lossless** ffmpeg-resample-only control dropped to 80.0% → webm 32kbps dropped to **66.7%**, 6/15 predictions flipped. Per-class breakdown showed brake devastated specifically (100% clean → 20% at webm32kbps) while belt/sway held up much better — strong evidence the model had partly learned to rely on recording-chain artifacts for brake, not the acoustic squeal itself (consistent with section 3's Freesound brake-mismatch finding).

### Fix built and validated
`compression_augment.py` + `preprocessing.py`'s `COMPRESSION_AUGMENT_ENABLED` (see section 2): retrain using the SAME real training files, each with 3 extra TRAIN-only copies that have actually been round-tripped through the real webm/opus pipeline at 16/32/64kbps, forcing the model to only rely on features that survive compression. Output lives at `ablation_results\compression_augmented\` (626 orig train files → 2496 with the 3 extra variants each; val/test untouched, 134/133).

Trained fresh XGBoost + CNN on this, then ran `mic_pipeline_test\compare_old_vs_compression_fix.py` — **same 133 real test files, same exact compressed bytes, OLD deployed model vs NEW compression-augmented model, side by side:**

| Condition | OLD model acc | NEW model acc | Delta |
|---|---|---|---|
| clean | 89.5% | 86.5% | -3.0% (expected small trade-off) |
| webm 16kbps | 63.9% | 85.0% | **+21.1%** |
| webm 32kbps | 62.4% | 88.0% | **+25.6%** |
| webm 64kbps | 65.4% | 85.0% | **+19.5%** |
| **brake @ webm32 specifically** | **18.2%** | **88.6%** | **+70.4%** |

Verified clean (no confound): OLD/NEW test sets are byte-identical file lists (133/133, confirmed via `dataset_documentation.csv` comparison), zero train/test leakage in either model, both models scored on the literal same compressed audio bytes (not independently re-compressed per model), XGBoost is deterministic (fixed `random_state`) so this isn't a lucky single run for that model.

### What's NOT yet done (blocking deploy)
- **CNN variance re-verification is IN PROGRESS as of end of this session** (`run_cnn_augmentation_variance.py --repeats 5`, comparing `no_augmentation` vs `compression_augmented`, 10 CNN trainings total). Reason: this session separately discovered CNN training has real run-to-run variance (one seed produced 57.14% accuracy vs 76-86% for other seeds on identical data/settings) — the compression-augmented CNN's 83.46% test result so far is a SINGLE run and needs to be confirmed as a real, stable improvement (not a lucky seed) before trusting it for deployment. XGBoost doesn't need this (deterministic).
- **Not yet deployed.** Once the variance check confirms the CNN result is real:
  - Copy `ablation_results\compression_augmented\best_traditional_model.pkl` + `scaler.pkl` (as a pair) → `car_test\processed_data\`
  - Copy `ablation_results\compression_augmented\cnn_model.keras` + `mel_stats.pkl` (as a pair — mel normalization stats MUST match the exact model they were computed alongside) → `car_test\processed_data\`
  - `config.pkl` and `label_encoder.pkl` verified byte-identical between old and new (same target_sr/duration/n_mfcc/classes) — no need to replace, but harmless if done for consistency.
  - **Do NOT touch** `best_transfer_model.pkl` (YAMNet), `best_panns_model.pkl`, `ensemble_config.pkl` — untouched by this fix, and (see section 1) don't affect the real final prediction anyway since their ensemble weight is 0.0 / not listed.
  - **Back up the old files first** (straightforward copy, not yet done) before overwriting anything in `car_test\processed_data` — this is live production, treat it with the same care as any prod deploy.
- YAMNet/PANNs were NOT retrained with compression augmentation and remain fragile to it — not urgent (they don't affect the current real output), but would need the same fix if their ensemble weight is ever raised above 0.

## 5. Other pending items (lower priority)

- Once CNN variance check (section 4) completes and deploy happens: consider re-running `evaluate_ensemble.py` on the new models to see if the 0.8/0.2/0.0 weights are still optimal, or retrain YAMNet/PANNs with compression augmentation too if they're going to be given nonzero weight.
- `feature_version` key in `config.pkl`, to make a future silent feature-dim mismatch fail loudly — still not done, now doubly relevant given `car_test\processed_data` vs `ablation_results\*\` are already-diverged artifact sets.
- Rename `break_*` → `brake_*` in `C:\Users\hp\Desktop\sounds\` — cosmetic, still not done, doesn't block anything (superseded in priority by section 3's conclusion anyway).
- Unresolved from earlier: a paper/report PDF with reviewer feedback was referenced but never located/shared — still blocked on the user.

## 6. How to resume in a new conversation

Paste/reference this file's path (`C:\Users\hp\Desktop\car_test\SESSION_HANDOFF.md`). Most likely next steps, in order:
1. Check whether `run_cnn_augmentation_variance.py --repeats 5` (comparing `no_augmentation` vs `compression_augmented`) finished and whether the compression-augmented CNN's improvement held up across all 5 seeds.
2. If confirmed: back up `car_test\processed_data`'s current traditional-ML and CNN files, then deploy the compression-augmented versions (see exact file list in section 4).
3. After deploying, sanity-check the live microservice end-to-end once (real mic recording through the actual app, not just the offline test scripts).
