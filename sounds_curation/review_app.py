# -*- coding: utf-8 -*-
r"""
===================================================================
Step 2 of the sounds\ curation plan (SESSION_HANDOFF.md, section 4)
===================================================================
Local review app for the candidate 5-second windows produced by
select_candidates.py. Run this, then open the printed URL in a
browser on THIS machine -- it serves local files directly, so it is
not (and should not be) published as a claude.ai artifact.

  cd C:\Users\hp\Desktop\car_test\sounds_curation
  python review_app.py
  -> open http://127.0.0.1:5050

For each of the 520 source files, listen to the proposed best 5s
window (energy-based rank 0), optionally cycle to rank 1/2 if the
top pick clipped a real event or grabbed something irrelevant, then
Approve / Reject / Skip. Decisions are written to
review_decisions.json immediately on every click (atomic replace),
so closing the browser / restarting the server never loses progress.

review_decisions.json format:
  { "<item_id>": {"decision": "approved"|"rejected"|"skip",
                   "candidate_filename": "...", "ts": "..."} , ... }
"""

import os
import json
import tempfile
from datetime import datetime, timezone

from flask import Flask, jsonify, request, send_from_directory, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES_DIR = os.path.join(BASE_DIR, "candidates")
MANIFEST_PATH = os.path.join(BASE_DIR, "manifest.json")
DECISIONS_PATH = os.path.join(BASE_DIR, "review_decisions.json")

app = Flask(__name__)


def load_manifest():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_decisions():
    if not os.path.exists(DECISIONS_PATH):
        return {}
    with open(DECISIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_decisions(decisions):
    # atomic write: never leave review_decisions.json half-written if the
    # process gets killed mid-save
    fd, tmp_path = tempfile.mkstemp(dir=BASE_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(decisions, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, DECISIONS_PATH)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


@app.route("/")
def index():
    if not os.path.exists(MANIFEST_PATH):
        return ("manifest.json not found -- run select_candidates.py first "
                f"(expected at {MANIFEST_PATH})"), 500
    manifest = load_manifest()
    return render_template("review.html", manifest_json=json.dumps(manifest))


@app.route("/audio/<path:filename>")
def audio(filename):
    return send_from_directory(CANDIDATES_DIR, filename)


@app.route("/api/decisions")
def api_decisions():
    return jsonify(load_decisions())


@app.route("/api/decision", methods=["POST"])
def api_decision():
    body = request.get_json(force=True)
    item_id = body.get("id")
    decision = body.get("decision")
    if not item_id or decision not in ("approved", "rejected", "skip"):
        return jsonify({"error": "need id and decision in (approved, rejected, skip)"}), 400

    decisions = load_decisions()
    decisions[item_id] = {
        "decision": decision,
        "candidate_filename": body.get("candidate_filename"),
        "note": body.get("note", ""),
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    save_decisions(decisions)
    return jsonify({"ok": True, "count_decided": len(decisions)})


if __name__ == "__main__":
    if not os.path.exists(MANIFEST_PATH):
        print(f"manifest.json not found at {MANIFEST_PATH} -- run select_candidates.py first.")
    print("Review app starting at http://127.0.0.1:5050")
    app.run(host="127.0.0.1", port=5050, debug=False)
