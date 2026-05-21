"""Generate architecture and loss schematic figures for the README.

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
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_IN    = "#89b4fa"   # input / data – blue
C_ENC   = "#a6e3a1"   # encoder – green
C_LATENT= "#cba6f7"   # latent / VAE – purple
C_SOM   = "#f9e2af"   # SOM – yellow
C_DEC   = "#fab387"   # decoder – peach
C_PRED  = "#89dceb"   # prediction heads – sky
C_CORR  = "#f38ba8"   # age-correction – red/pink
C_LOSS  = "#313244"   # loss boxes background (dark)
BG      = "#1e1e2e"   # figure background
FG      = "#cdd6f4"   # foreground / text
ARROW   = "#9399b2"

FONT = dict(family="monospace")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def box(ax, xy, w, h, label, color, fontsize=8.5, sublabel=None, radius=0.015):
    x, y = xy
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.2,
        edgecolor=FG,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(patch)
    dy = 0.012 if sublabel else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, color="black", fontweight="bold",
            fontfamily="monospace", zorder=4)
    if sublabel:
        ax.text(x, y - 0.022, sublabel, ha="center", va="center",
                fontsize=7, color="#313244", fontfamily="monospace", zorder=4)


def arrow(ax, x0, y0, x1, y1, label="", color=ARROW):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle="->,head_width=0.015,head_length=0.012",
            color=color, lw=1.3,
        ),
        zorder=2,
    )
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        ax.text(mx + 0.008, my, label, fontsize=6.5, color=FG,
                fontfamily="monospace", zorder=5)


def arrow_h(ax, x0, x1, y, label="", color=ARROW):
    arrow(ax, x0, y, x1, y, label=label, color=color)


def arrow_v(ax, x, y0, y1, label="", color=ARROW):
    arrow(ax, x, y0, x, y1, label=label, color=color)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – Architecture
# ─────────────────────────────────────────────────────────────────────────────

def make_architecture():
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.5, 0.97, "DPSOM-ECG Architecture",
             ha="center", va="top", fontsize=13, fontweight="bold",
             color=FG, fontfamily="monospace")

    # ── row y positions ──────────────────────────────────────────────────────
    Y_DATA  = 0.82
    Y_ENC   = 0.62
    Y_LAT   = 0.42
    Y_BOT   = 0.22

    BW, BH  = 0.13, 0.09   # default box width / height
    SM      = 0.10          # small box width

    # ── ECG Record ───────────────────────────────────────────────────────────
    box(ax, (0.12, Y_DATA), BW, BH, "ECG Record", C_IN,
        sublabel="12 leads × T")

    arrow_h(ax, 0.19, 0.29, Y_DATA)
    box(ax, (0.35, Y_DATA), BW + 0.02, BH, "Beat Segmentation", C_IN,
        sublabel="R-peak detection")

    arrow_h(ax, 0.43, 0.52, Y_DATA)
    box(ax, (0.57, Y_DATA), BW, BH, "Beat Repr.", C_IN,
        sublabel="[B × N × C × T]")

    # amplitude context (AMP) side branch
    arrow_v(ax, 0.57, Y_DATA - 0.045, Y_ENC + 0.045, color="#a6adc8")
    box(ax, (0.57, Y_ENC), BW + 0.04, BH, "LeadWise Conv\nEncoder", C_ENC,
        sublabel="k=7, ch 32→64→512")

    # AMP context line from encoder outward
    # draw AMP context box to the right of encoder
    ax.annotate("", xy=(0.72, Y_ENC), xytext=(0.64, Y_ENC),
                arrowprops=dict(arrowstyle="->,head_width=0.015,head_length=0.012",
                                color=ARROW, lw=1.3), zorder=2)
    box(ax, (0.79, Y_ENC), SM + 0.04, BH, "AMP Context", C_IN,
        sublabel="ampl.+phase (2C)")

    # three latent branches  ─────────────────────────────────────────────────
    # z_morph (left), z_age (centre), z_sex (right)
    X_MORPH = 0.35
    X_AGE   = 0.57
    X_SEX   = 0.79

    # arrows from encoder down to latents
    for xd in [X_MORPH, X_AGE, X_SEX]:
        ax.annotate("", xy=(xd, Y_LAT + 0.045), xytext=(0.57, Y_ENC - 0.045),
                    arrowprops=dict(arrowstyle="->,head_width=0.015,head_length=0.012",
                                    color=ARROW, lw=1.1, connectionstyle="arc3,rad=0.0"),
                    zorder=2)

    box(ax, (X_MORPH, Y_LAT), BW + 0.02, BH, "z  (morphology)", C_LATENT,
        sublabel="μ,σ  dim=32  VAE")
    box(ax, (X_AGE,   Y_LAT), BW, BH, "z_age", C_LATENT,
        sublabel="dim=8 (25%)")
    box(ax, (X_SEX,   Y_LAT), BW, BH, "z_sex", C_LATENT,
        sublabel="dim=8 (25%)")

    # ── SOM (below z_morph) ──────────────────────────────────────────────────
    arrow_v(ax, X_MORPH, Y_LAT - 0.045, Y_BOT + 0.045)
    box(ax, (X_MORPH, Y_BOT), BW + 0.04, BH, "Toroidal SOM", C_SOM,
        sublabel="8×8 grid  q(z)")

    # ── Decoder (left of SOM) ────────────────────────────────────────────────
    ax.annotate("", xy=(0.12, Y_BOT + 0.0), xytext=(X_MORPH - 0.08, Y_BOT),
                arrowprops=dict(arrowstyle="->,head_width=0.015,head_length=0.012",
                                color=ARROW, lw=1.1), zorder=2)
    box(ax, (0.12, Y_BOT), BW, BH, "Decoder", C_DEC,
        sublabel="ConvTranspose")

    # ── Age prediction head ──────────────────────────────────────────────────
    # age-correction feeds into age head
    arrow_v(ax, X_AGE, Y_LAT - 0.045, Y_BOT + 0.045)
    box(ax, (X_AGE, Y_BOT), BW + 0.02, BH, "Age Head", C_PRED,
        sublabel="CE  88 bins")

    # age correction side-feed from SOM
    ax.annotate("", xy=(X_AGE - 0.075, Y_BOT), xytext=(X_MORPH + 0.075, Y_BOT),
                arrowprops=dict(arrowstyle="->,head_width=0.013,head_length=0.010",
                                color=C_CORR, lw=1.2,
                                connectionstyle="arc3,rad=-0.3"),
                zorder=2)
    ax.text((X_MORPH + X_AGE) / 2, Y_BOT + 0.07,
            "top-k SOM\n embeddings", fontsize=6.5, ha="center",
            color=C_CORR, fontfamily="monospace", zorder=5)

    # ── Sex prediction head ──────────────────────────────────────────────────
    arrow_v(ax, X_SEX, Y_LAT - 0.045, Y_BOT + 0.045)
    box(ax, (X_SEX, Y_BOT), BW, BH, "Sex Head", C_PRED,
        sublabel="BCE  binary")

    # ── AMP context → heads ──────────────────────────────────────────────────
    for xh in [X_AGE, X_SEX]:
        ax.annotate("", xy=(xh + BW / 2, Y_BOT + 0.01),
                    xytext=(0.79 - (SM + 0.04) / 2, Y_ENC - 0.045),
                    arrowprops=dict(arrowstyle="->,head_width=0.012,head_length=0.010",
                                    color="#a6adc8", lw=0.9,
                                    connectionstyle="arc3,rad=0.15"),
                    zorder=2)

    # ── Record attention pooling label ────────────────────────────────────────
    ax.text(0.5, 0.07,
            "Record-level attention pooling  (η)  aggregates beat predictions → record-level age / sex",
            ha="center", va="center", fontsize=7.5, color=FG,
            fontfamily="monospace", style="italic")

    # ── legend ────────────────────────────────────────────────────────────────
    legend_items = [
        (C_IN,     "Input / data"),
        (C_ENC,    "Encoder"),
        (C_LATENT, "Latent space (VAE)"),
        (C_SOM,    "SOM"),
        (C_DEC,    "Decoder"),
        (C_PRED,   "Prediction head"),
    ]
    for i, (col, lbl) in enumerate(legend_items):
        px = 0.01 + i * 0.165
        rect = FancyBboxPatch((px, 0.01), 0.015, 0.025,
                              boxstyle="round,pad=0,rounding_size=0.004",
                              facecolor=col, edgecolor=FG, linewidth=0.8, zorder=3)
        ax.add_patch(rect)
        ax.text(px + 0.02, 0.022, lbl, fontsize=6.5, va="center",
                color=FG, fontfamily="monospace")

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "architecture.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – Loss breakdown
# ─────────────────────────────────────────────────────────────────────────────

def make_loss():
    # Loss terms with (symbol, weight, description, colour, group)
    terms = [
        # group A – reconstruction
        ("L_recon",   "θ",      "Huber reconstruction",         C_DEC,    "Reconstruction"),
        ("L_KL",      "θ · p",  "KL(q(z) ‖ N(0,I))",           C_LATENT, "Reconstruction"),
        # group B – SOM
        ("L_commit",  "γ",      "KL(p ‖ q)  SOM commitment",   C_SOM,    "SOM"),
        ("L_smooth",  "β",      "neighbourhood smoothing",      C_SOM,    "SOM"),
        # group C – temporal
        ("L_temporal","τ",      "consecutive-beat smoothness",  C_ENC,    "Temporal"),
        # group D – demographics (beat-level)
        ("L_age_beat","δ_age",  "beat-level age  (CE, 88 bins)",C_PRED,   "Demographics"),
        ("L_sex_beat","δ_sex",  "beat-level sex  (BCE)",        C_PRED,   "Demographics"),
        # group E – record-level
        ("L_age_rec", "η·δ_age·0.4", "record-level age  (L1, attn pool)", C_CORR, "Record-level"),
        ("L_sex_rec", "η·δ_sex",     "record-level sex  (BCE, attn pool)",C_CORR, "Record-level"),
    ]

    groups = ["Reconstruction", "SOM", "Temporal", "Demographics", "Record-level"]
    group_colors = {
        "Reconstruction": "#313244",
        "SOM":            "#2a2a3e",
        "Temporal":       "#2e2e40",
        "Demographics":   "#292940",
        "Record-level":   "#2c2c3e",
    }

    fig, ax = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.text(0.5, 0.97, "DPSOM-ECG  ·  Total Loss Breakdown",
             ha="center", va="top", fontsize=13, fontweight="bold",
             color=FG, fontfamily="monospace")

    # ── group headers ─────────────────────────────────────────────────────────
    COL_W  = 0.18
    COL_PAD = 0.01
    N_COLS  = len(groups)
    total_w = N_COLS * COL_W + (N_COLS - 1) * COL_PAD
    x_start = (1 - total_w) / 2

    group_xs = {}
    for i, g in enumerate(groups):
        gx = x_start + i * (COL_W + COL_PAD) + COL_W / 2
        group_xs[g] = gx

    HEADER_Y = 0.84
    TERM_Y_START = 0.70
    TERM_DY      = 0.135
    BOX_W  = COL_W - 0.01
    BOX_H  = 0.08

    # draw group headers
    for g in groups:
        gx = group_xs[g]
        hpatch = FancyBboxPatch(
            (gx - BOX_W / 2, HEADER_Y - 0.035), BOX_W, 0.065,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=group_colors[g], edgecolor=FG, linewidth=1.0, zorder=3,
        )
        ax.add_patch(hpatch)
        ax.text(gx, HEADER_Y, g, ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=FG,
                fontfamily="monospace", zorder=4)

    # draw term boxes per group
    group_term_count = {g: 0 for g in groups}
    for (sym, weight, desc, color, group) in terms:
        gx  = group_xs[group]
        idx = group_term_count[group]
        ty  = TERM_Y_START - idx * TERM_DY

        # box
        tpatch = FancyBboxPatch(
            (gx - BOX_W / 2, ty - BOX_H / 2), BOX_W, BOX_H,
            boxstyle="round,pad=0,rounding_size=0.010",
            facecolor=color, edgecolor=FG, linewidth=1.0, zorder=3,
        )
        ax.add_patch(tpatch)
        # symbol + weight
        ax.text(gx, ty + 0.014,
                f"{weight} · {sym}", ha="center", va="center",
                fontsize=8.5, fontweight="bold", color="black",
                fontfamily="monospace", zorder=4)
        # description
        ax.text(gx, ty - 0.016, desc, ha="center", va="center",
                fontsize=6.8, color="#313244",
                fontfamily="monospace", zorder=4)

        # connector line from header to first term
        if idx == 0:
            ax.plot([gx, gx], [HEADER_Y - 0.035, ty + BOX_H / 2],
                    color=FG, lw=0.8, zorder=2, alpha=0.4)
        else:
            prev_ty = TERM_Y_START - (idx - 1) * TERM_DY
            ax.plot([gx, gx], [prev_ty - BOX_H / 2, ty + BOX_H / 2],
                    color=FG, lw=0.8, zorder=2, alpha=0.4)

        group_term_count[group] += 1

    # ── total loss equation ────────────────────────────────────────────────────
    eq = (
        r"$\mathcal{L}_\mathrm{total}$"
        "  =  "
        r"$\theta\,\mathcal{L}_\mathrm{recon}$"
        "  +  "
        r"$\gamma\,\mathcal{L}_\mathrm{commit}$"
        "  +  "
        r"$\beta\,\mathcal{L}_\mathrm{smooth}$"
        "  +  "
        r"$\tau\,\mathcal{L}_\mathrm{temporal}$"
        "  +  "
        r"$\delta_\mathrm{age}\,\mathcal{L}_\mathrm{age}$"
        "  +  "
        r"$\delta_\mathrm{sex}\,\mathcal{L}_\mathrm{sex}$"
        "  +  "
        r"$\eta\,\mathcal{L}_\mathrm{record}$"
    )
    ax.text(0.5, 0.085, eq, ha="center", va="center",
            fontsize=9.5, color=FG, zorder=5)

    # bounding box for equation
    eq_box = FancyBboxPatch(
        (0.03, 0.055), 0.94, 0.06,
        boxstyle="round,pad=0,rounding_size=0.012",
        facecolor="#1a1a2e", edgecolor=FG, linewidth=0.9, zorder=2,
    )
    ax.add_patch(eq_box)

    fig.tight_layout()
    out = os.path.join(OUT_DIR, "loss.png")
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    make_architecture()
    make_loss()
