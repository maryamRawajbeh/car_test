# -*- coding: utf-8 -*-
r"""Activity + sequence diagrams for the eleven use cases (documentation 3.9),
drawn to match the real code in garageai-frontend / garageai-backend /
garageai-audio-analysis. 22 PNGs -> processed_data/report_charts/uml/.

Rendered with large fonts at a size close to the final display size, so the
text stays crisp and legible after the doc scales the image down.
"""
import os
import textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp

matplotlib.rcParams["font.family"] = "DejaVu Sans"

OUT = r"processed_data/report_charts/uml"
os.makedirs(OUT, exist_ok=True)

NAVY = "#14304a"; TEAL = "#0f8b82"; LILAC = "#dbe4f3"; EDGE = "#3d63a0"
GREY = "#444"; RED = "#a23636"; REDBG = "#f5e3e3"

# font sizes (large on purpose)
F_TITLE = 15
F_BOX = 12
F_DIA = 11
F_LBL = 11
F_ACTOR = 12
F_MSG = 10.5


def wrap(t, w):
    return "\n".join(textwrap.wrap(t, width=w)) or t


# ---------------------------------------------------------------- activity
def activity(name, items, title):
    cx, bx = 4.7, 12.0
    GAP = 0.55                      # vertical clearance between shapes

    # ---- pass 1 : shape + height per item ----
    plan = []
    for it in items:
        kind = it[0]
        if kind in ("start", "end"):
            label = "Start" if kind == "start" else (it[1] if len(it) > 1 else "End")
            plan.append((kind, label, 0.75))
        elif kind == "action":
            txt = wrap(it[1], 26)
            plan.append(("action", txt, 0.8 + 0.4 * txt.count("\n")))
        elif kind == "decision":
            q = wrap(it[1], 22)
            no_text = wrap(it[3], 20)
            plan.append(("decision", (q, it[2], no_text), 1.45 + 0.3 * q.count("\n")))

    total = sum(h for _, _, h in plan) + GAP * (len(plan) - 1)
    fig, ax = plt.subplots(figsize=(7.6, total * 0.5 + 0.9), dpi=200)
    ax.set_xlim(0, 16); ax.set_ylim(0, total + 1.1); ax.axis("off")
    ax.set_title(title, fontsize=F_TITLE, fontweight="bold", color=NAVY, pad=12)

    def edge(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.7,
                                    mutation_scale=18))

    y = total + 0.4
    prev_bottom = None
    for kind, payload, h in plan:
        cy = y - h / 2
        if prev_bottom is not None:
            edge(cx, prev_bottom, cx, cy + h / 2)

        if kind in ("start", "end"):
            w = 2.8
            ax.add_patch(mp.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.4", linewidth=1.6,
                         edgecolor=EDGE, facecolor=NAVY if kind == "start" else "#37485a"))
            ax.text(cx, cy, payload, ha="center", va="center", fontsize=F_BOX, color="white")

        elif kind == "action":
            w = 7.8
            ax.add_patch(mp.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                         boxstyle="round,pad=0.03", linewidth=1.5,
                         edgecolor=EDGE, facecolor=TEAL))
            ax.text(cx, cy, payload, ha="center", va="center", fontsize=F_BOX, color="white")

        else:                                        # decision
            q, no_label, no_text = payload
            w = 7.0
            ax.add_patch(mp.Polygon([(cx, cy + h / 2), (cx + w / 2, cy),
                                     (cx, cy - h / 2), (cx - w / 2, cy)],
                                    closed=True, linewidth=1.5,
                                    edgecolor=EDGE, facecolor=LILAC))
            ax.text(cx, cy, q, ha="center", va="center", fontsize=F_DIA, color=NAVY)
            ax.annotate("", xy=(bx - 2.1, cy), xytext=(cx + w / 2, cy),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.6,
                                        mutation_scale=18))
            ax.text(cx + w / 2 + 0.25, cy + 0.55, no_label, fontsize=F_LBL,
                    style="italic", color=RED, ha="left", va="bottom")
            bw, bh = 4.1, 1.05 + 0.34 * no_text.count("\n")
            ax.add_patch(mp.FancyBboxPatch((bx - bw / 2, cy - bh / 2), bw, bh,
                         boxstyle="round,pad=0.03", linewidth=1.4,
                         edgecolor=RED, facecolor=REDBG))
            ax.text(bx, cy, no_text, ha="center", va="center", fontsize=F_MSG, color=RED)
            ax.text(cx + 0.55, cy - h / 2 - GAP / 2, "yes", fontsize=F_LBL,
                    style="italic", color=GREY, ha="left", va="center")

        prev_bottom = cy - h / 2
        y -= h + GAP

    fig.savefig(f"{OUT}/{name}_activity.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- sequence
def sequence(name, actors, messages, title):
    na, nm = len(actors), len(messages)
    ns = sum(1 for s, dd, _, _ in messages if s == dd)
    xstep, ystep = 4.2, 1.15
    extra = 0.35 * ns
    fig, ax = plt.subplots(figsize=(max(7.4, 2.1 * na), 0.72 * nm + extra + 2.4), dpi=200)
    ax.set_xlim(-1.2, (na - 1) * xstep + 3.6)
    ax.set_ylim(0, nm * ystep + extra + 2.8)
    ax.axis("off")
    ax.set_title(title, fontsize=F_TITLE, fontweight="bold", color=NAVY, pad=14)

    xs = [i * xstep + 1.3 for i in range(na)]
    top = nm * ystep + extra + 2.0
    for x, a in zip(xs, actors):
        ax.add_patch(mp.FancyBboxPatch((x - 1.55, top - 0.5), 3.1, 1.0,
                     boxstyle="round,pad=0.03", linewidth=1.5,
                     edgecolor=EDGE, facecolor=LILAC))
        ax.text(x, top, a, ha="center", va="center", fontsize=F_ACTOR, color=NAVY)
        ax.plot([x, x], [0.4, top - 0.6], linestyle=(0, (4, 3)), color="#9aa7b4", lw=1.1)

    y = nm * ystep + extra + 0.8
    for src, dst, label, dashed in messages:
        x0, x1 = xs[src], xs[dst]
        ls = "--" if dashed else "-"
        if src == dst:
            side = -1 if src == na - 1 else 1
            ax.plot([x0, x0 + side * 1.7, x0 + side * 1.7, x0],
                    [y, y, y - 0.4, y - 0.4], color=GREY, lw=1.6, ls=ls)
            ax.annotate("", xy=(x0, y - 0.4), xytext=(x0 + side * 0.9, y - 0.4),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.6,
                                        mutation_scale=16))
            ax.text(x0 + side * 2.0, y - 0.2, wrap(label, 24), fontsize=F_MSG,
                    color=NAVY, ha="left" if side == 1 else "right", va="center")
            y -= ystep + 0.35
        else:
            ax.annotate("", xy=(x1, y), xytext=(x0, y),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.7, ls=ls,
                                        mutation_scale=18))
            ax.text((x0 + x1) / 2, y + 0.2, wrap(label, 26), fontsize=F_MSG,
                    color=NAVY, ha="center", va="bottom")
            y -= ystep

    fig.savefig(f"{OUT}/{name}_sequence.png", bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 3.9.1  DIAGNOSIS
# ============================================================================
activity("capture_audio", [
    ("start",),
    ("action", "Open the Home page"),
    ("decision", "Record or upload?", "either", "Both paths reach 'audio ready'"),
    ("action", "Provide audio: record a clip or choose a file"),
    ("action", "Audio held in the browser, ready to analyse"),
    ("end",),
], "Activity - Capture Audio Input")
sequence("capture_audio", ["User", ":Frontend"], [
    (0, 1, "open Home page", False),
    (0, 1, "choose Record or Upload", False),
    (1, 1, "hold selected audio blob", False),
], "Sequence - Capture Audio Input")

activity("record_audio", [
    ("start",),
    ("action", "Tap Record"),
    ("decision", "Microphone permission granted?", "denied", "Show 'mic unavailable' error"),
    ("action", "Start MediaRecorder and the live level meter"),
    ("action", "User taps Stop; build the WebM/Opus blob"),
    ("decision", "Peak level below the silence threshold (0.03)?", "yes",
     "Show 'very quiet' warning (clip still usable)"),
    ("action", "Audio ready to analyse"),
    ("end",),
], "Activity - Record Audio via Microphone")
sequence("record_audio", ["User", ":Frontend", "MediaRecorder\n(browser)"], [
    (0, 1, "tap Record", False),
    (1, 2, "getUserMedia(audio)", False),
    (2, 1, "permission result", True),
    (1, 2, "start(); read level (loop)", False),
    (0, 1, "tap Stop", False),
    (2, 1, "onstop -> WebM blob", True),
    (1, 1, "silence check (peak < 0.03)", False),
], "Sequence - Record Audio via Microphone")

activity("upload_audio", [
    ("start",),
    ("action", "Click Upload File"),
    ("action", "Operating-system file picker opens"),
    ("action", "User selects an audio file"),
    ("action", "Frontend stores the file (name shown, never used)"),
    ("action", "Audio ready to analyse"),
    ("end",),
], "Activity - Upload Audio File")
sequence("upload_audio", ["User", ":Frontend", "OS file\npicker"], [
    (0, 1, "click Upload File", False),
    (1, 2, "open picker", False),
    (0, 2, "select audio file", False),
    (2, 1, "File object", True),
    (1, 1, "selectFile() -> hold blob", False),
], "Sequence - Upload Audio File")

activity("classify_sound", [
    ("start",),
    ("action", "User clicks Analyze"),
    ("action", "POST audio to the gateway  /api/v1/audio/analyze"),
    ("action", "Gateway forwards to the inference service  POST /predict"),
    ("decision", "Valid audio type and size <= 20 MB?", "no", "Return 400 error"),
    ("action", "Convert to WAV (ffmpeg) and extract features"),
    ("decision", "'Other' gate: P(other) >= 0.46?", "yes",
     "Return class = 'other' (heavy models skipped)"),
    ("action", "Run Traditional ML + CNN + YAMNet"),
    ("action", "Weighted fusion  0.30 / 0.60 / 0.10"),
    ("action", "Return class + confidence + per-model votes"),
    ("action", "Gateway saves the result to history (if signed in)"),
    ("action", "Frontend renders the prediction"),
    ("end",),
], "Activity - Classify Vehicle Sound")
sequence("classify_sound", ["User", ":Frontend", ":Gateway", ":Inference", "SQLite"], [
    (0, 1, "click Analyze", False),
    (1, 2, "POST /audio/analyze", False),
    (2, 3, "POST /predict (internal key)", False),
    (3, 3, "validate; ffmpeg -> WAV; features", False),
    (3, 3, "'other' gate; models; fusion", False),
    (3, 2, "class, confidence, per-model", True),
    (2, 4, "INSERT analyses (if signed in)", False),
    (2, 1, "JSON result", True),
    (1, 0, "show prediction + breakdown", False),
], "Sequence - Classify Vehicle Sound")

activity("view_result", [
    ("start",),
    ("action", "Analysis result received by the Frontend"),
    ("decision", "Overall confidence >= 0.5?", "no", "Show 'No Clear Match' + closest guess"),
    ("decision", "Predicted class = 'other'?", "yes", "Show 'not a car-fault sound' state"),
    ("action", "Show class + colour-banded confidence badge"),
    ("action", "Show per-class confidence bars and each model's vote"),
    ("action", "Offer Assess Severity / Ask GarageAI about this"),
    ("end",),
], "Activity - View Classification Result")
sequence("view_result", ["User", ":Frontend"], [
    (1, 1, "receive result JSON", False),
    (1, 1, "confidence < 0.5 -> No Clear Match", False),
    (1, 1, "class == other -> not-a-fault state", False),
    (1, 0, "render class, badge, bars, votes", False),
    (0, 1, "(optional) Assess Severity / chat", False),
], "Sequence - View Classification Result")

activity("view_history", [
    ("start",),
    ("action", "Open the History page"),
    ("decision", "Signed in? (ProtectedRoute)", "no", "Redirect to Login"),
    ("action", "GET /api/v1/history  (Bearer token)"),
    ("action", "Gateway: SELECT analyses WHERE user_id, newest first"),
    ("action", "Render the list; optional client-side search filter"),
    ("action", "Open an entry, or delete it  (DELETE /history/{id})"),
    ("end",),
], "Activity - View Prediction History")
sequence("view_history", ["User", ":Frontend", ":Gateway", "SQLite"], [
    (0, 1, "open History", False),
    (1, 2, "GET /history (Bearer)", False),
    (2, 2, "requireAuth", False),
    (2, 3, "SELECT analyses WHERE user_id", False),
    (3, 2, "rows", True),
    (2, 1, "{ analyses: [...] }", True),
    (1, 0, "render history list", False),
    (0, 1, "delete an entry", False),
    (1, 2, "DELETE /history/{id}", False),
    (2, 3, "DELETE WHERE id AND user_id", False),
], "Sequence - View Prediction History")

# ============================================================================
# 3.9.2  ACCOUNT
# ============================================================================
activity("sign_up", [
    ("start",),
    ("action", "Fill name, e-mail, password"),
    ("action", "POST /api/v1/auth/signup"),
    ("decision", "Name + valid e-mail + password >= 6 chars?", "no", "Return 400"),
    ("decision", "E-mail already registered?", "yes", "Return 409"),
    ("action", "bcrypt-hash the password (cost 10)"),
    ("action", "INSERT user; create session (jti); sign JWT"),
    ("action", "Return { token, user }"),
    ("action", "Frontend stores the token and sets the current user"),
    ("end",),
], "Activity - Sign Up")
sequence("sign_up", ["User", ":Frontend", ":Gateway", "SQLite"], [
    (0, 1, "submit sign-up form", False),
    (1, 2, "POST /auth/signup", False),
    (2, 2, "validate input", False),
    (2, 3, "SELECT id WHERE email", False),
    (2, 2, "bcrypt.hash(pw, 10)", False),
    (2, 3, "INSERT users; INSERT sessions (jti)", False),
    (2, 2, "signToken({ sub, jti })", False),
    (2, 1, "{ token, user }", True),
    (1, 1, "store token; set user", False),
], "Sequence - Sign Up")

activity("log_in", [
    ("start",),
    ("action", "Enter e-mail + password"),
    ("action", "POST /api/v1/auth/login"),
    ("decision", "E-mail found?", "no", "Return 401"),
    ("decision", "bcrypt.compare matches the stored hash?", "no", "Return 401"),
    ("action", "Create session (jti); sign JWT"),
    ("action", "Return { token, user }"),
    ("action", "Frontend stores the token and sets the current user"),
    ("end",),
], "Activity - Log In")
sequence("log_in", ["User", ":Frontend", ":Gateway", "SQLite"], [
    (0, 1, "submit login form", False),
    (1, 2, "POST /auth/login", False),
    (2, 3, "SELECT * FROM users WHERE email", False),
    (3, 2, "user row", True),
    (2, 2, "bcrypt.compare(pw, hash)", False),
    (2, 3, "INSERT sessions (jti)", False),
    (2, 2, "signToken", False),
    (2, 1, "{ token, user }", True),
    (1, 1, "store token; set user", False),
], "Sequence - Log In")

activity("reset_password", [
    ("start",),
    ("action", "Enter e-mail on Forgot-Password"),
    ("action", "POST /auth/forgot-password  (always a generic reply)"),
    ("decision", "Account exists?", "no", "Stop: no e-mail, same generic reply"),
    ("action", "Store SHA-256 hash of a 32-byte token, 1-hour expiry"),
    ("action", "E-mail the reset link"),
    ("action", "User opens the link -> Reset-Password page"),
    ("action", "POST /auth/reset-password { token, newPassword }"),
    ("decision", "Hash matches, unused, not expired?", "no", "Return 400 (invalid / expired)"),
    ("action", "bcrypt-hash new password; UPDATE user + mark token used (one transaction)"),
    ("end",),
], "Activity - Reset Password")
sequence("reset_password", ["User", ":Frontend", ":Gateway", "SQLite", ":Mailer"], [
    (0, 1, "submit e-mail (Forgot Password)", False),
    (1, 2, "POST /auth/forgot-password", False),
    (2, 3, "SELECT id WHERE email", False),
    (2, 3, "INSERT password_resets (hash, +1h)", False),
    (2, 4, "send reset link", False),
    (2, 1, "generic 200", True),
    (0, 1, "open link; submit new password", False),
    (1, 2, "POST /auth/reset-password", False),
    (2, 3, "SELECT reset WHERE hash, unused, valid", False),
    (2, 3, "UPDATE users; UPDATE reset used=1", False),
    (2, 1, "200 success", True),
], "Sequence - Reset Password")

# ============================================================================
# 3.9.3  SETTINGS
# ============================================================================
activity("change_theme", [
    ("start",),
    ("action", "Open Settings, or the navigation-bar toggle"),
    ("action", "Pick Light / Dark / System"),
    ("action", "next-themes applies the theme class"),
    ("action", "Choice persisted to the browser's localStorage"),
    ("action", "UI re-renders in the new theme"),
    ("end",),
], "Activity - Change Display Theme")
sequence("change_theme", ["User", ":Frontend", "next-themes", "localStorage"], [
    (0, 1, "choose theme (Settings / nav)", False),
    (1, 2, "setTheme(light|dark|system)", False),
    (2, 3, "persist choice", False),
    (2, 1, "theme class updated", True),
    (1, 0, "UI re-rendered", False),
], "Sequence - Change Display Theme")

# ============================================================================
# 3.9.4  INFO
# ============================================================================
activity("view_about", [
    ("start",),
    ("action", "Click About in the navigation"),
    ("action", "Router loads the public /about route"),
    ("action", "Static project description + credits render"),
    ("end",),
], "Activity - View About Page")
sequence("view_about", ["User", ":Frontend", ":Router"], [
    (0, 1, "click About", False),
    (1, 2, "navigate /about", False),
    (2, 1, "About component (public route)", True),
    (1, 0, "render static content (no backend call)", False),
], "Sequence - View About Page")

# ============================================================================
# 3.9.1  DIAGNOSIS  (added: chat + severity)
# ============================================================================
activity("ask_garageai", [
    ("start",),
    ("action", "User sends a question (fresh, or opened from a result / checklist)"),
    ("action", "POST /diagnose/text with the full message history"),
    ("action", "Gateway forwards to the inference service /diagnose/text"),
    ("action", "Embed the question; rank the knowledge-base entries by similarity"),
    ("decision", "Any fault category above the similarity floor?", "no",
     "Return a no-diagnosis reply"),
    ("action", "LLM answers, grounded only in the retrieved text"),
    ("decision", "LLM marks it as not a supported fault?", "yes",
     "Return a no-diagnosis reply"),
    ("action", "Return the grounded answer (+ optional severity level)"),
    ("action", "Save / update the conversation (if signed in)"),
    ("end",),
], "Activity - Ask GarageAI (Chat)")
sequence("ask_garageai", ["User", ":Frontend", ":Gateway", ":Inference", "LLM", "SQLite"], [
    (0, 1, "send message", False),
    (1, 2, "POST /diagnose/text (history)", False),
    (2, 3, "POST /diagnose/text (internal key)", False),
    (3, 3, "embed query; rank glossary entries", False),
    (3, 4, "retrieved context + conversation", False),
    (4, 3, "grounded answer (+severity / no-match)", True),
    (3, 2, "{ answer, category, severity }", True),
    (2, 5, "upsert diagnose_conversations (if signed in)", False),
    (2, 1, "reply", True),
    (1, 0, "show answer", False),
], "Sequence - Ask GarageAI (Chat)")

activity("check_severity", [
    ("start",),
    ("action", "Open the severity checklist from a result"),
    ("action", "Answer the yes/no questions for that fault class"),
    ("action", "Score = weighted fraction of 'yes' answers (0-100)"),
    ("action", "Map to Low (<= 30) / Medium (<= 65) / High"),
    ("action", "Show the risk level, percentage and advice"),
    ("decision", "Take the result into Ask GarageAI?", "no", "Stay on the result view"),
    ("action", "Open Ask GarageAI seeded with the confirmed findings"),
    ("end",),
], "Activity - Check Severity")
sequence("check_severity", ["User", ":Frontend", ":Gateway"], [
    (0, 1, "open the checklist", False),
    (0, 1, "answer yes / no questions", False),
    (1, 1, "scoreSeverity(answers, class) -> percent, level", False),
    (1, 0, "show level + percentage + advice", False),
    (0, 1, "(optional) ask the assistant", False),
    (1, 2, "POST /diagnose/text (formResult)", False),
    (2, 1, "answer built on the checklist result", True),
], "Sequence - Check Severity")

# ============================================================================
# 3.9.2  ACCOUNT  (added: manage account)
# ============================================================================
activity("manage_account", [
    ("start",),
    ("action", "Open Settings (signed in)"),
    ("decision", "Delete the account?", "yes",
     "DELETE /users/me -> cascade-remove history + conversations"),
    ("action", "Edit name / e-mail, change password, or toggle notification prefs"),
    ("action", "PUT /users/me  /  /me/password  /  /me/settings"),
    ("decision", "Was it a password change?", "no", "Keep all sessions"),
    ("action", "Revoke the account's other active sessions"),
    ("action", "Show the saved values"),
    ("end",),
], "Activity - Manage Account")
sequence("manage_account", ["User", ":Frontend", ":Gateway", "SQLite"], [
    (0, 1, "edit profile / password / prefs", False),
    (1, 2, "PUT /users/me (or /me/password, /me/settings)", False),
    (2, 2, "requireAuth; validate", False),
    (2, 3, "UPDATE users ...", False),
    (2, 3, "revoke other sessions (password change only)", False),
    (2, 1, "saved values", True),
    (0, 1, "delete account", False),
    (1, 2, "DELETE /users/me", False),
    (2, 3, "DELETE users (cascade history + conversations)", False),
    (2, 1, "204 No Content", True),
], "Sequence - Manage Account")

print("done ->", OUT)
