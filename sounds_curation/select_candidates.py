# -*- coding: utf-8 -*-
r"""
===================================================================
Step 1 of the sounds\ curation plan (SESSION_HANDOFF.md, section 4)
===================================================================
Scans C:\Users\hp\Desktop\sounds\*.wav (the 520-file Freesound-preview
candidate pool -- sounds2\ is intentionally ignored, it's a confirmed
redundant duplicate) and, for each file, proposes the best-looking
5-second window(s) using short-time energy -- NOT a substitute for
human judgment, just narrows down what to listen to.

The actual window-selection logic lives in curation_common.py, shared
with select_brake_dataset_candidates.py (the second Freesound pool,
brake_dataset_freesound\accepted) so both stay identical by construction.

Output (this script only READS from sounds\, never modifies it):
  sounds_curation/candidates/<label>_<idx>_c<rank>.wav  -- trimmed clips,
      mono, resampled to CANDIDATE_SR, for fast review playback
  sounds_curation/manifest.json -- one entry per source file, with up to
      TOP_K candidate windows each (rank 0 = best guess)

Labels are normalized break -> brake here (cosmetic rename from
SESSION_HANDOFF.md section 4 item 5) without touching the original
sounds\break_*.wav filenames.

Run:
  cd C:\Users\hp\Desktop\car_test\sounds_curation
  python select_candidates.py [--limit N] [--force] [--workers N]
"""

import os
import json
import glob
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from curation_common import extract_candidates_for_file, WINDOW_SEC, CANDIDATE_SR, TOP_K, LOW_ENERGY_DBFS

SOUNDS_DIR = r"C:\Users\hp\Desktop\sounds"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_DIR = os.path.join(OUT_DIR, "candidates")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")

LABEL_MAP = {"belt": "belt", "break": "brake", "sway": "sway"}

N_WORKERS = int(os.environ.get("SOUNDS_CURATION_N_WORKERS", max(1, (os.cpu_count() or 2) - 1)))


def list_source_files():
    files = sorted(glob.glob(os.path.join(SOUNDS_DIR, "*.wav")))
    items = []
    for path in files:
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        parts = name.rsplit("_", 1)
        if len(parts) != 2 or parts[0] not in LABEL_MAP:
            print(f"  [skip] unrecognized filename pattern: {base}")
            continue
        raw_label, idx = parts
        items.append({
            "source_path": path,
            "source_filename": base,
            "raw_label": raw_label,
            "label": LABEL_MAP[raw_label],
            "idx": idx,
        })
    return items


def process_one(item, force):
    path = item["source_path"]
    label = item["label"]
    idx = item["idx"]
    try:
        source_meta, candidates = extract_candidates_for_file(
            path, out_prefix=f"{label}_{idx}", candidates_dir=CANDIDATES_DIR,
            force=force, top_k=TOP_K)
        return {
            "id": f"{label}_{idx}",
            "label": label,
            "pool": "sounds",
            "source_filename": item["source_filename"],
            **source_meta,
            "candidates": candidates,
        }, None
    except Exception as e:
        return None, f"{item['source_filename']}: {e}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only process first N files (debug)")
    parser.add_argument("--force", action="store_true", help="regenerate candidate clips even if they already exist")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    os.makedirs(CANDIDATES_DIR, exist_ok=True)

    items = list_source_files()
    if args.limit:
        items = items[:args.limit]

    print(f"Found {len(items)} source files in {SOUNDS_DIR}. Using {args.workers} worker process(es).")

    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_one, item, args.force): item for item in items}
        done = 0
        for fut in as_completed(futures):
            entry, err = fut.result()
            done += 1
            if err:
                errors.append(err)
                print(f"  [{done}/{len(items)}] ERROR: {err.splitlines()[0]}")
            else:
                results.append(entry)
                if done % 25 == 0 or done == len(items):
                    print(f"  [{done}/{len(items)}] processed")

    results.sort(key=lambda r: (r["label"], r["source_filename"]))

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({"window_sec": WINDOW_SEC, "candidate_sr": CANDIDATE_SR, "items": results}, f, indent=2)

    n_low_energy = sum(1 for r in results if r["candidates"] and r["candidates"][0]["low_energy"])
    n_short = sum(1 for r in results if r["short_file"])
    print(f"\nDone. {len(results)} files -> manifest written to {MANIFEST_PATH}")
    print(f"  short files (<= {WINDOW_SEC}s, no cropping needed): {n_short}")
    print(f"  best-candidate flagged low_energy (< {LOW_ENERGY_DBFS} dBFS): {n_low_energy}")
    if errors:
        print(f"\n{len(errors)} file(s) failed to process:")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
