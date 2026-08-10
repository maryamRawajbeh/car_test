# -*- coding: utf-8 -*-
r"""
===================================================================
Step 1.5 of the sounds\ curation plan (SESSION_HANDOFF.md, section 4)
===================================================================
Runs the CURRENT BEST real-data traditional-ML model (XGBoost, 93.2%
acc, from ablation_results\no_augmentation\ -- see SESSION_HANDOFF.md
section 3 item 6) against every candidate clip produced by
select_candidates.py, and writes a confidence/agreement signal into
manifest.json so the review app can put the most SUSPICIOUS clips
first instead of making the user listen to all 1193 in file order.

This is a triage aid, not a substitute for listening:
  - model agrees + high confidence  -> probably fine, quick-listen/approve
  - model disagrees, or low confidence on EITHER label -> listen carefully,
    this is exactly the kind of mislabeled-by-search-query clip the
    review pass exists to catch
  - the model itself is only 93.2% accurate on real held-out data, so a
    "disagreement" is a hint, not a verdict -- it still needs a human ear

Adds two fields to every candidate dict in manifest.json:
  "model_predicted_label", "model_predicted_prob"  (argmax over classes)
  "model_expected_prob"                            (prob assigned to the
                                                      item's OWN declared label)

Run (after select_candidates.py has produced manifest.json + candidates\):
  cd C:\Users\hp\Desktop\car_test\sounds_curation
  python score_candidates.py [--workers N]
"""

import os
import sys
import json
import pickle
import tempfile
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAR_TEST_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, CAR_TEST_DIR)  # audio_common.py lives in car_test\, one level up from here

from audio_common import load_clean_audio, extract_mfcc_vector

CANDIDATES_DIR = os.path.join(BASE_DIR, "candidates")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")

MODEL_DIR = os.path.join(CAR_TEST_DIR, "ablation_results", "no_augmentation")

N_WORKERS = int(os.environ.get("SOUNDS_CURATION_N_WORKERS", max(1, (os.cpu_count() or 2) - 1)))

# module-level so each worker process loads it once (via _init_worker), not once per file
_model = None
_scaler = None
_label_encoder = None
_config = None


def _init_worker():
    global _model, _scaler, _label_encoder, _config
    with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
        _config = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "best_traditional_model.pkl"), "rb") as f:
        _model = pickle.load(f)["model"]
    with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
        _scaler = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        _label_encoder = pickle.load(f)


def score_one(args):
    item_id, expected_label, filename = args
    path = os.path.join(CANDIDATES_DIR, filename)
    try:
        # fallback_to_untrimmed=True: a couple of candidates are the flagged
        # "low_energy" ones -- if silence-trim strips the whole clip, still
        # score *something* rather than crashing this worker
        y, sr = load_clean_audio(path, target_sr=_config["target_sr"],
                                  target_duration=_config["target_duration"],
                                  fallback_to_untrimmed=True)
        feats = extract_mfcc_vector(y, sr, n_mfcc=_config["n_mfcc"]).reshape(1, -1)
        feats_scaled = _scaler.transform(feats)
        probs = _model.predict_proba(feats_scaled)[0]
        classes = list(_label_encoder.classes_)

        predicted_idx = int(np.argmax(probs))
        predicted_label = classes[predicted_idx]
        predicted_prob = float(probs[predicted_idx])
        expected_prob = float(probs[classes.index(expected_label)]) if expected_label in classes else None

        return item_id, filename, {
            "model_predicted_label": predicted_label,
            "model_predicted_prob": round(predicted_prob, 4),
            "model_expected_prob": round(expected_prob, 4) if expected_prob is not None else None,
        }, None
    except Exception as e:
        return item_id, filename, None, f"{filename}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="only score first N source items (debug)")
    args = parser.parse_args()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    items = manifest["items"]
    if args.limit:
        items = items[:args.limit]

    tasks = []
    for it in items:
        for cand in it["candidates"]:
            tasks.append((it["id"], it["label"], cand["filename"]))

    print(f"Scoring {len(tasks)} candidate clips from {len(items)} source files "
          f"with the model in {MODEL_DIR} using {args.workers} worker(s).")

    scores_by_item = {}  # item_id -> {filename: score_dict}
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker) as ex:
        futures = [ex.submit(score_one, t) for t in tasks]
        done = 0
        for fut in as_completed(futures):
            item_id, filename, score, err = fut.result()
            done += 1
            if err:
                errors.append(err)
            else:
                scores_by_item.setdefault(item_id, {})[filename] = score
            if done % 100 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}] scored")

    n_mismatch = 0
    for it in manifest["items"]:
        for cand in it["candidates"]:
            score = scores_by_item.get(it["id"], {}).get(cand["filename"])
            if score:
                cand.update(score)
                if score["model_predicted_label"] != it["label"]:
                    n_mismatch += 1

    fd, tmp_path = tempfile.mkstemp(dir=BASE_DIR, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp_path, MANIFEST_PATH)

    print(f"\nDone. manifest.json updated in place with model_predicted_label / "
          f"model_predicted_prob / model_expected_prob per candidate.")
    print(f"  candidates where the model's top guess != the item's declared label: {n_mismatch}/{len(tasks)}")
    if errors:
        print(f"\n{len(errors)} clip(s) failed to score:")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
