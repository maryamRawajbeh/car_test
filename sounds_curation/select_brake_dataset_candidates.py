# -*- coding: utf-8 -*-
r"""
===================================================================
Second Freesound pool for brake: C:\Users\hp\Desktop\brake_dataset_freesound
===================================================================
Separate from sounds\break_*.wav (confirmed zero content-identical overlap
by hash between the two pools, despite both being Freesound-sourced brake
audio -- different collection scripts/QA). This one already ships with
per-file metadata in dataset_labels.csv: an automated accept/reject QA
pass (confidence, frequency-band checks), car_type, fault_type, AND
crucially the Freesound LICENSE for each file.

License filter (the whole reason this is a separate script rather than
just pointing select_candidates.py at a second folder): of the 183 files
physically in accepted\, ~26% are Freesound CC-BY-NC (non-commercial),
which garageai (a commercial product) cannot use. Those are EXCLUDED here
before any window-selection/scoring work is spent on them -- no point
reviewing audio that can never ship regardless of how it sounds.
14 of the 183 accepted\ files aren't in the CSV directly (they're
byte-identical duplicates of a CSV-listed file under a different
generated filename); their license is resolved via content-hash match to
their CSV-listed twin.

Appends into the SAME sounds_curation/manifest.json used by the sounds\
pool (adds a "pool": "brake_dataset_freesound" field per item so the two
sources stay distinguishable) -- the existing review_app/score_candidates.py
work on it unchanged, since they only key off item["candidates"] / item["label"].

Run:
  cd C:\Users\hp\Desktop\car_test\sounds_curation
  python select_brake_dataset_candidates.py [--force] [--workers N]
"""

import os
import csv
import json
import glob
import hashlib
import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from curation_common import extract_candidates_for_file, WINDOW_SEC, CANDIDATE_SR, TOP_K, LOW_ENERGY_DBFS

BDF_DIR = r"C:\Users\hp\Desktop\brake_dataset_freesound"
ACCEPTED_DIR = os.path.join(BDF_DIR, "accepted")
CSV_PATH = os.path.join(BDF_DIR, "dataset_labels.csv")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_DIR = os.path.join(OUT_DIR, "candidates")
MANIFEST_PATH = os.path.join(OUT_DIR, "manifest.json")

POOL_NAME = "brake_dataset_freesound"

N_WORKERS = int(os.environ.get("SOUNDS_CURATION_N_WORKERS", max(1, (os.cpu_count() or 2) - 1)))


def sha1_of(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def is_commercial_safe(license_url):
    # all 5 licenses observed in dataset_labels.csv are CC0 or CC-BY[-NC], so
    # "-nc" is an unambiguous marker (doesn't appear anywhere else in the URLs)
    return "-nc" not in license_url.lower()


def load_csv_rows():
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        return {r["filename"]: r for r in csv.DictReader(f)}


def resolve_items():
    """Returns list of dicts: filename, license, car_type, fault_type, label_csv
    for every file in accepted\\ that resolves to a commercial-safe license."""
    rows_by_name = load_csv_rows()
    accepted_files = sorted(glob.glob(os.path.join(ACCEPTED_DIR, "*.wav")))

    # first pass: hash every accepted file so unmatched ones can be resolved
    # via a content-identical CSV-listed twin
    hash_to_names = {}
    for path in accepted_files:
        base = os.path.basename(path)
        h = sha1_of(path)
        hash_to_names.setdefault(h, []).append(base)

    resolved = []
    unresolved = []
    for path in accepted_files:
        base = os.path.basename(path)
        row = rows_by_name.get(base)
        if row is None:
            h = sha1_of(path)
            twin = next((n for n in hash_to_names[h] if n != base and n in rows_by_name), None)
            row = rows_by_name.get(twin) if twin else None
        if row is None:
            unresolved.append(base)
            continue
        resolved.append({
            "path": path,
            "filename": base,
            "license": row["license"],
            "car_type": row.get("car_type", ""),
            "fault_type": row.get("fault_type", ""),
            "label_csv": row.get("label", ""),
        })

    commercial_safe = [r for r in resolved if is_commercial_safe(r["license"])]
    excluded_nc = [r for r in resolved if not is_commercial_safe(r["license"])]

    print(f"accepted\\ files: {len(accepted_files)}")
    print(f"  resolved to a CSV license: {len(resolved)}  (unresolved/no license info: {len(unresolved)})")
    print(f"  commercial-safe (CC0 / CC-BY): {len(commercial_safe)}")
    print(f"  excluded as non-commercial (CC-BY-NC): {len(excluded_nc)}")
    if unresolved:
        print(f"  [warning] no license could be determined for: {unresolved} -- excluded to be safe")

    return commercial_safe


def process_one(item, force):
    stem = os.path.splitext(item["filename"])[0]
    try:
        source_meta, candidates = extract_candidates_for_file(
            item["path"], out_prefix=f"bdf_{stem}", candidates_dir=CANDIDATES_DIR,
            force=force, top_k=TOP_K)
        return {
            "id": f"bdf_{stem}",
            "label": "brake",
            "pool": POOL_NAME,
            "source_filename": item["filename"],
            "source_license": item["license"],
            "car_type": item["car_type"],
            "fault_type": item["fault_type"],
            **source_meta,
            "candidates": candidates,
        }, None
    except Exception as e:
        return None, f"{item['filename']}: {e}\n{traceback.format_exc()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    os.makedirs(CANDIDATES_DIR, exist_ok=True)

    items = resolve_items()
    print(f"\nExtracting candidates for {len(items)} commercial-safe files using {args.workers} worker(s)...")

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

    # merge into the EXISTING manifest.json (which already has sounds\ pool
    # scoring from score_candidates.py) -- never overwrite it wholesale
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    existing_ids = {it["id"] for it in manifest["items"]}
    new_count = 0
    for entry in results:
        if entry["id"] in existing_ids:
            # replace (e.g. re-run after --force) rather than duplicate
            manifest["items"] = [it for it in manifest["items"] if it["id"] != entry["id"]]
        else:
            new_count += 1
        manifest["items"].append(entry)

    manifest["items"].sort(key=lambda r: (r.get("pool", "sounds"), r["label"], r["source_filename"]))

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(results)} brake_dataset_freesound items merged into {MANIFEST_PATH} "
          f"({new_count} new, {len(results) - new_count} replaced).")
    if errors:
        print(f"\n{len(errors)} file(s) failed to process:")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
