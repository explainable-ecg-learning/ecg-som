"""Generate minimalist architecture and loss overview figures for docs/.

Run from repo root:
    python src/utils/make_figures.py
Outputs:
    docs/architecture.png
    docs/loss.png
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Minimalist light palette (publication style) ──────────────────────────────
BG          = "#ffffff"
TEXT_MAIN   = "#0f172a"   # slate-900
TEXT_SUB    = "#64748b"   # slate-500
BORDER_NEUT = "#cbd5e1"   # slate-300
FILL_NEUT   = "#f8fafc"   # slate-50
ARROW_COL   = "#94a3b8"   # slate-400

# Subtle semantic accents: (fill, border, text/accent)
ACC_INPUT = ("#f1f5f9", "#94a3b8", "#334155")   # neutral slate
ACC_MORPH = ("#eff6ff", "#60a5fa", "#1d4ed8")   # soft blue
ACC_SOM   = ("#fefce8", "#eab308", "#a16207")   # soft gold
ACC_DEC   = ("#fff7ed", "#fb923c", "#c2410c")   # soft orange
ACC_DEMO  = ("#f0fdfa", "#2dd4bf", "#0f766e")   # soft teal
ACC_CORR  = ("#fdf2f8", "#f472b6", "#be185d")   # soft rose


def card(ax, xy, w, h, title, subtitle=None, accent=ACC_INPUT, title_size=9.5, sub_size=7.6, radius=0.016):
    x, y = xy
    fill, border, _ = accent
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.3,
        edgecolor=border,
        facecolor=fill,
        zorder=3,
    )
    ax.add_patch(patch)
    dy = 0.015 if subtitle else 0.0
    ax.text(
        x, y + dy, title,
        ha="center", va="center",
        fontsize=title_size, fontweight="semibold", color=TEXT_MAIN,
        fontfamily="sans-serif", zorder=4,
    )
    if subtitle:
        ax.text(
            x, y - 0.020, subtitle,
            ha="center", va="center",
            fontsize=sub_size, color=TEXT_SUB,
            fontfamily="sans-serif", zorder=4,
        )


def arrow(ax, x0, y0, x1, y1, color=ARROW_COL, lw=1.4, rad=0.0, ls="-"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="-|>,head_width=0.28,head_length=0.42",
            mutation_scale=11,
            color=color,
            lw=lw,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=1,
            shrinkB=1,
        ),
        zorder=2,
    )


def elbow(ax, pts, color=ARROW_COL, lw=1.2, ls="-"):
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="round")
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    arrow(ax, x0, y0, x1, y1, color=color, lw=lw, ls=ls)


def pill(ax, xy, text, color=TEXT_SUB, bg="#ffffff", border="#e2e8f0", fontsize=7.0):
    x, y = xy
    ax.text(
        x, y, text,
        ha="center", va="center",
        fontsize=fontsize, color=color, fontfamily="sans-serif",
        bbox=dict(boxstyle="round,pad=0.32,rounding_size=0.35", facecolor=bg, edgecolor=border, lw=0.9),
        zorder=5,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Minimalist Architecture Overview
# ─────────────────────────────────────────────────────────────────────────────

def make_architecture():
    fig, ax = plt.subplots(figsize=(12.5, 6.6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title + concise subtitle
    ax.text(0.03, 0.965, "DPSOM-ECG Architecture",
            ha="left", va="top", fontsize=15, fontweight="bold",
            color=TEXT_MAIN, fontfamily="sans-serif")
    ax.text(0.03, 0.92, "Unsupervised morphology clustering on a toroidal SOM with supervised demographic branches",
            ha="left", va="top", fontsize=9.1, color=TEXT_SUB, fontfamily="sans-serif")

    # Subtle column positions
    X_DEC   = 0.13
    X_MORPH = 0.36
    X_AGE   = 0.61
    X_SEX   = 0.83

    Y_IN  = 0.80
    Y_ENC = 0.60
    Y_LAT = 0.435
    Y_OUT = 0.245
    Y_REC = 0.08

    BW, BH = 0.175, 0.088

    # ── Subtle background zones: Unsupervised vs Supervised ──────────────────
    unsup_zone = FancyBboxPatch(
        (0.025, 0.175), 0.445, 0.515,
        boxstyle="round,pad=0,rounding_size=0.020",
        facecolor="#f8fbff", edgecolor="#93c5fd", linewidth=1.1, linestyle="--", zorder=0,
    )
    ax.add_patch(unsup_zone)
    pill(ax, (0.145, 0.690), "UNSUPERVISED  (no disease labels)",
         color="#2563eb", bg="#eff6ff", border="#93c5fd", fontsize=7.0)

    sup_zone = FancyBboxPatch(
        (0.50, 0.022), 0.472, 0.668,
        boxstyle="round,pad=0,rounding_size=0.020",
        facecolor="#f5fefa", edgecolor="#5eead4", linewidth=1.1, linestyle="--", zorder=0,
    )
    ax.add_patch(sup_zone)
    pill(ax, (0.725, 0.690), "SUPERVISED  (age & sex labels)",
         color="#0d9488", bg="#f0fdfa", border="#5eead4", fontsize=7.0)

    # ── Row 1: Input & preprocessing ─────────────────────────────────────────
    card(ax, (0.14, Y_IN), 0.18, 0.080, "12-Lead ECG", "10 s @ 100 Hz", accent=ACC_INPUT)
    arrow(ax, 0.23, Y_IN, 0.29, Y_IN)
    card(ax, (0.40, Y_IN), 0.22, 0.080, "Beat Segm. & Norm", "R-peaks · 1.2 s windows", accent=ACC_INPUT)

    # Split from Beat Segm. & Norm → Normalized Beats (to encoders) & Amplitude Stats (pre-norm)
    arrow(ax, 0.51, Y_IN, 0.555, Y_IN)
    card(ax, (0.645, Y_IN), 0.18, 0.080, "Normalized Beats", "N × 12 × 120", accent=ACC_INPUT)

    # Pre-norm amplitude stats branch off Beat Segm. & Norm cleanly over the top
    elbow(ax, [(0.40, Y_IN + 0.040), (0.40, 0.872), (0.87, 0.872), (0.87, Y_IN + 0.040)],
          color=ARROW_COL, lw=1.2)
    card(ax, (0.87, Y_IN), 0.18, 0.080, "Amplitude Stats", "Pre-norm σ & max", accent=ACC_INPUT)

    # ── Row 2: Three independent encoders ────────────────────────────────────
    for xc in (X_MORPH, X_AGE, X_SEX):
        arrow(ax, 0.645, Y_IN - 0.040, xc, Y_ENC + BH / 2)

    card(ax, (X_MORPH, Y_ENC), BW, BH, "Morphology Encoder", "Lead-wise Conv + SE", accent=ACC_MORPH)
    card(ax, (X_AGE,   Y_ENC), BW, BH, "Age Encoder",        "Lead-wise Conv + SE", accent=ACC_DEMO)
    card(ax, (X_SEX,   Y_ENC), BW, BH, "Sex Encoder",        "Lead-wise Conv + SE", accent=ACC_DEMO)

    # ── Row 3: Latents ───────────────────────────────────────────────────────
    arrow(ax, X_MORPH, Y_ENC - BH / 2, X_MORPH, Y_LAT + BH / 2)
    arrow(ax, X_AGE,   Y_ENC - BH / 2, X_AGE,   Y_LAT + BH / 2)
    arrow(ax, X_SEX,   Y_ENC - BH / 2, X_SEX,   Y_LAT + BH / 2)

    card(ax, (X_MORPH, Y_LAT), BW, BH, "Morphology  z", "32-d VAE  (μ, σ)", accent=ACC_MORPH)
    card(ax, (X_AGE,   Y_LAT), BW, BH, "Age  z_age",    "8-d latent",       accent=ACC_DEMO)
    card(ax, (X_SEX,   Y_LAT), BW, BH, "Sex  z_sex",    "8-d latent",       accent=ACC_DEMO)

    # ── Row 4: Decoder, SOM, Age Head, Sex Head ──────────────────────────────
    # Decoder from unmodified z
    arrow(ax, X_MORPH - BW / 2, Y_LAT, X_DEC, Y_OUT + BH / 2, rad=0.22)
    card(ax, (X_DEC, Y_OUT), 0.175, BH, "ECG Decoder", "Conv + SE + FiLM(z)", accent=ACC_DEC)

    # Residualization → Toroidal SOM
    arrow(ax, X_MORPH, Y_LAT - BH / 2, X_MORPH, Y_OUT + BH / 2)
    pill(ax, (X_MORPH, (Y_LAT + Y_OUT) / 2), "− sup. age/sex probes", color="#0f766e", bg="#f0fdfa", border="#5eead4", fontsize=6.7)
    card(ax, (X_MORPH, Y_OUT), BW, BH, "Toroidal SOM", "8×8 grid  ·  q(z)", accent=ACC_SOM)

    # Age & Sex heads
    arrow(ax, X_AGE, Y_LAT - BH / 2, X_AGE, Y_OUT + BH / 2)
    card(ax, (X_AGE, Y_OUT), BW, BH, "Age Head", "88 bins (CE) + λ·corr", accent=ACC_DEMO)

    arrow(ax, X_SEX, Y_LAT - BH / 2, X_SEX, Y_OUT + BH / 2)
    card(ax, (X_SEX, Y_OUT), BW, BH, "Sex Head", "Binary logit (BCE)", accent=ACC_DEMO)

    # Age correction bridge (SOM → Age Head, stop-grad)
    arrow(ax, X_MORPH + BW / 2, Y_OUT, X_AGE - BW / 2, Y_OUT, color=ACC_CORR[1], lw=1.6)
    pill(ax, ((X_MORPH + X_AGE) / 2, Y_OUT + 0.044), "top-k q(μ.detach)", color=ACC_CORR[2], bg="#fdf2f8", border="#fbcfe8", fontsize=6.6)

    # Amplitude Stats rail → Age Head, Sex Head, and Record Attention Pooling
    X_RAIL = 0.948
    elbow(ax, [(0.948, Y_IN - 0.040), (X_RAIL, 0.335), (X_AGE + 0.055, 0.335), (X_AGE + 0.055, Y_OUT + BH / 2)],
          color=ARROW_COL, lw=1.0, ls="--")
    elbow(ax, [(X_RAIL, 0.355), (X_SEX + 0.055, 0.355), (X_SEX + 0.055, Y_OUT + BH / 2)],
          color=ARROW_COL, lw=1.0, ls="--")
    elbow(ax, [(X_RAIL, 0.335), (X_RAIL, Y_REC), (0.915, Y_REC)],
          color=ARROW_COL, lw=1.0, ls="--")

    # ── Row 5: Record-level pooling ──────────────────────────────────────────
    arrow(ax, X_AGE, Y_OUT - BH / 2, 0.65, Y_REC + 0.040, color=ACC_DEMO[1], lw=1.3)
    arrow(ax, X_SEX, Y_OUT - BH / 2, 0.79, Y_REC + 0.040, color=ACC_DEMO[1], lw=1.3)
    card(
        ax, (0.72, Y_REC), 0.39, 0.078,
        "Record-Level Attention Pooling",
        "Pools detached beat preds → Record Age & Sex  (via μ.detach & amp)",
        accent=ACC_CORR, title_size=9.1, sub_size=7.0,
    )

    fig.tight_layout(pad=0.6)
    out = os.path.join(OUT_DIR, "architecture.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Minimalist Loss Overview
# ─────────────────────────────────────────────────────────────────────────────

def make_loss():
    fig, ax = plt.subplots(figsize=(12.2, 4.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.03, 0.95, "DPSOM-ECG Training Objectives",
            ha="left", va="top", fontsize=15, fontweight="bold",
            color=TEXT_MAIN, fontfamily="sans-serif")
    ax.text(0.03, 0.895, "Phase 3 joint loss combining reconstruction, toroidal clustering, temporal consistency, and demographics",
            ha="left", va="top", fontsize=9.0, color=TEXT_SUB, fontfamily="sans-serif")

    columns = [
        ("Reconstruction", ACC_DEC, [
            ("θ · L_recon", "Masked Huber + outside L1"),
            ("θ · w_KL · L_KL", "VAE prior KL  (w_KL = 2.5)"),
        ]),
        ("SOM Clustering", ACC_SOM, [
            ("γ · L_commit", "KL(p ‖ q) commitment"),
            ("β · L_smooth", "Toroidal neighbourhood"),
        ]),
        ("Temporal", ACC_MORPH, [
            ("τ · L_temporal", "Ordered intra-record beat diff"),
        ]),
        ("Demographics", ACC_DEMO, [
            ("δ_age · L_age", "Beat age (88-bin CE)"),
            ("δ_sex · L_sex", "Beat sex (BCE)"),
        ]),
        ("Record Pooling", ACC_CORR, [
            ("η · L_record", "Attention-pooled age (L1)\n& sex (BCE)"),
        ]),
    ]

    n_cols = len(columns)
    margin = 0.03
    gap    = 0.016
    col_w  = (1.0 - 2 * margin - (n_cols - 1) * gap) / n_cols

    Y_HEAD = 0.70
    Y_TOP  = 0.535
    DY     = 0.160
    CARD_H = 0.120

    # Subtle supervision group bars above columns 0-2 (Unsupervised) and 3-4 (Supervised)
    unsup_x0 = margin
    unsup_x1 = margin + 3 * col_w + 2 * gap
    sup_x0   = margin + 3 * (col_w + gap)
    sup_x1   = 1.0 - margin

    ax.add_patch(FancyBboxPatch(
        (unsup_x0, 0.785), unsup_x1 - unsup_x0, 0.046,
        boxstyle="round,pad=0,rounding_size=0.012",
        facecolor="#eff6ff", edgecolor="#93c5fd", lw=1.0, zorder=1,
    ))
    ax.text((unsup_x0 + unsup_x1) / 2, 0.808, "UNSUPERVISED  (no disease labels)",
            ha="center", va="center", fontsize=7.6, fontweight="bold",
            color="#2563eb", fontfamily="sans-serif", zorder=2)

    ax.add_patch(FancyBboxPatch(
        (sup_x0, 0.785), sup_x1 - sup_x0, 0.046,
        boxstyle="round,pad=0,rounding_size=0.012",
        facecolor="#f0fdfa", edgecolor="#5eead4", lw=1.0, zorder=1,
    ))
    ax.text((sup_x0 + sup_x1) / 2, 0.808, "SUPERVISED  (age & sex labels)",
            ha="center", va="center", fontsize=7.6, fontweight="bold",
            color="#0d9488", fontfamily="sans-serif", zorder=2)

    for i, (heading, accent, items) in enumerate(columns):
        cx = margin + i * (col_w + gap) + col_w / 2
        fill, border, accent_text = accent

        # Minimal column header underline
        ax.text(cx, Y_HEAD + 0.018, heading, ha="center", va="bottom",
                fontsize=9.3, fontweight="bold", color=accent_text, fontfamily="sans-serif")
        ax.plot([cx - col_w * 0.44, cx + col_w * 0.44], [Y_HEAD, Y_HEAD],
                color=border, lw=2.0, solid_capstyle="round")

        for j, (term, desc) in enumerate(items):
            cy = Y_TOP - j * DY
            patch = FancyBboxPatch(
                (cx - col_w / 2, cy - CARD_H / 2), col_w, CARD_H,
                boxstyle="round,pad=0,rounding_size=0.015",
                linewidth=1.2, edgecolor=border, facecolor=fill, zorder=3,
            )
            ax.add_patch(patch)
            ax.text(cx, cy + 0.024, term, ha="center", va="center",
                    fontsize=8.8, fontweight="semibold", color=TEXT_MAIN,
                    fontfamily="sans-serif", zorder=4)
            ax.text(cx, cy - 0.022, desc, ha="center", va="center",
                    fontsize=7.1, color=TEXT_SUB,
                    fontfamily="sans-serif", zorder=4)

    # Clean summary equation bar
    eq_box = FancyBboxPatch(
        (margin, 0.115), 1.0 - 2 * margin, 0.115,
        boxstyle="round,pad=0,rounding_size=0.016",
        facecolor=FILL_NEUT, edgecolor=BORDER_NEUT, linewidth=1.1, zorder=2,
    )
    ax.add_patch(eq_box)

    eq = (
        r"$\mathcal{L}_{\mathrm{total}} \;=\; "
        r"\theta(\mathcal{L}_{\mathrm{recon}} + w_{\mathrm{KL}}\mathcal{L}_{\mathrm{KL}})"
        r" \;+\; \gamma\,\mathcal{L}_{\mathrm{commit}}"
        r" \;+\; \beta\,\mathcal{L}_{\mathrm{smooth}}"
        r" \;+\; \tau\,\mathcal{L}_{\mathrm{temporal}}"
        r" \;+\; \delta_{\mathrm{age}}\mathcal{L}_{\mathrm{age}}"
        r" \;+\; \delta_{\mathrm{sex}}\mathcal{L}_{\mathrm{sex}}"
        r" \;+\; \eta\,\mathcal{L}_{\mathrm{record}}$"
    )
    ax.text(0.5, 0.172, eq, ha="center", va="center", fontsize=10.4, color=TEXT_MAIN, zorder=4)

    ax.text(
        0.5, 0.055,
        "α = 10.0 is the Student-t degrees-of-freedom parameter in q(z), not a loss weight.",
        ha="center", va="center", fontsize=7.8, color=TEXT_SUB,
        fontfamily="sans-serif", style="italic",
    )

    fig.tight_layout(pad=0.6)
    out = os.path.join(OUT_DIR, "loss.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_architecture()
    make_loss()
