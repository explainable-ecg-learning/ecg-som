"""Generate publication figures for datasets, ablations, SOM topology, latent space, and ECG reconstructions.

Figures produced (all saved at 180 dpi):
  1. docs/datasets_overview.png       -- Cohort demographics, beat alignment, and class distributions.
  2. ablation/head_ablations.png      -- 3-seed residualization & head-pruning comparison + legacy sweeps.
  3. docs/som_overview.png            -- 2D toroidal U-matrix, class purity, age, occupancy & 0-shot maps.
  4. docs/som_prototypes.png          -- Decoded 2D SOM prototype beat morphologies (Lead II + V2).
  5. ablation/cross_dataset_topology.png -- Transfer retention, TE/QE/active health, +UDA deltas & scale.
  6. docs/latent_disentanglement.png  -- Shared t-SNE mu_res, z_age PCA, branch leakage & age/sex calibration.
  7. docs/ecg_reconstructions.png     -- 6-lead beat reconstructions & SOM prototypes across all 3 cohorts.

Usage:
    PYTHONPATH=. python -u src/utils/make_paper_figures.py
    PYTHONPATH=. python -u src/utils/make_paper_figures.py --only transfer_diag latent recon
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from src.config import DPSOM_Config
from src.data.dataset import ECG_Dataset
from src.data.generator import ECG_DataGenerator
from src.training.trainer import _get_device, load_dpsom_checkpoint
from src.utils.metrics import compute_som_clustering_metrics


COHORT_COLORS = {
    "ptbxl":   "#2563eb",
    "georgia": "#0d9488",
    "chapman": "#d97706",
}


def _collect_cohort_stats(records: list) -> dict:
    """Extract per-record demographics, beat counts, labels, and Lead-II beat waveforms."""
    ages, sexes, labels, beat_counts = [], [], [], []
    lead2_beats = []
    for r in records:
        if getattr(r, "beat_representations", None) is None or len(r.beat_representations) == 0:
            continue
        age = float(getattr(r, "age", np.nan))
        sex = int(getattr(r, "sex", -1))
        if np.isnan(age) or age > 100 or sex not in (0, 1):
            continue
        beats = np.asarray(r.beat_representations, dtype=np.float32)  # [N, T, C]
        ages.append(age)
        sexes.append(sex)
        labels.append(str(r.label))
        beat_counts.append(len(beats))
        lead2_beats.append(beats[:, :, 1])
    return {
        "ages":        np.asarray(ages, dtype=np.float32),
        "sexes":       np.asarray(sexes, dtype=np.int64),
        "labels":      labels,
        "beat_counts": np.asarray(beat_counts, dtype=np.int64),
        "lead2":       np.concatenate(lead2_beats, axis=0) if lead2_beats else np.zeros((0, 120), dtype=np.float32),
    }


def make_datasets_overview(out_png: str = "docs/datasets_overview.png") -> None:
    """Plot cohort demographics, beat morphology alignment, and diagnostic distributions."""
    cfg = DPSOM_Config()
    print("[fig] Loading PTB-XL, Georgia, and Chapman-Shaoxing for datasets_overview...")
    ptbxl_ds = ECG_Dataset.load("data/processed/ptbxl_100_T-12ms.pkl")
    if os.path.exists(cfg.error_labels_path):
        ptbxl_ds.apply_error_labels(
            cfg.error_labels_path,
            max_error_severity=cfg.max_error_severity,
            exclude_error_flags=cfg.exclude_error_flags,
            verbose=False,
        )
    ptb_recs = (
        ptbxl_ds.get_data("train")
        + ptbxl_ds.get_data("val")
        + ptbxl_ds.get_data("test")
    )
    geo_ds = ECG_Dataset.load("data/processed/georgia_100.pkl")
    chp_ds = ECG_Dataset.load("data/processed/chapman_shaoxing_100.pkl")

    cohorts = [
        ("ptbxl",   "PTB-XL (curated)",        COHORT_COLORS["ptbxl"],   _collect_cohort_stats(ptb_recs)),
        ("georgia", "Georgia (0-shot)",         COHORT_COLORS["georgia"], _collect_cohort_stats(geo_ds.get_data("test"))),
        ("chapman", "Chapman-Shaoxing (0-shot)",COHORT_COLORS["chapman"], _collect_cohort_stats(chp_ds.get_data("test"))),
    ]

    fig = plt.figure(figsize=(16.8, 9.6), dpi=180)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, hspace=0.40, wspace=0.30)

    # (a) Age distributions
    ax_age = fig.add_subplot(gs[0, 0])
    bins = np.arange(0, 100, 5)
    for _key, title, col, st in cohorts:
        med = float(np.median(st["ages"]))
        n_rec = len(st["ages"])
        ax_age.hist(
            st["ages"], bins=bins, density=True, histtype="stepfilled",
            alpha=0.22, color=col, edgecolor=col, linewidth=1.4,
            label=f"{title} (n={n_rec:,}, med={med:.0f}y)",
        )
        ax_age.axvline(med, color=col, ls="--", lw=1.3, alpha=0.9)
    ax_age.set_title("(a) Record Age Distribution", fontsize=11, fontweight="bold")
    ax_age.set_xlabel("Age (years)", fontsize=9.5)
    ax_age.set_ylabel("Density", fontsize=9.5)
    ax_age.set_xlim(0, 95)
    ax_age.grid(True, axis="y", alpha=0.25, ls=":")
    ax_age.spines[["top", "right"]].set_visible(False)
    ax_age.legend(fontsize=7.8, loc="upper left", frameon=True)

    # (b) Sex split & beat segmentation yield
    ax_sex = fig.add_subplot(gs[0, 1])
    x = np.arange(len(cohorts))
    w = 0.34
    male_pcts = [100.0 * float((st["sexes"] == 0).mean()) for _, _, _, st in cohorts]
    fem_pcts  = [100.0 * float((st["sexes"] == 1).mean()) for _, _, _, st in cohorts]
    bm = ax_sex.bar(x - w / 2, male_pcts, w, color="#3b82f6", alpha=0.88, label="Male (%)")
    bf = ax_sex.bar(x + w / 2, fem_pcts,  w, color="#ec4899", alpha=0.82, label="Female (%)")
    for idx, (_key, _title, _col, st) in enumerate(cohorts):
        n_rec = len(st["ages"])
        n_b = int(st["beat_counts"].sum())
        mu_b = float(st["beat_counts"].mean())
        sd_b = float(st["beat_counts"].std(ddof=1))
        ax_sex.text(bm[idx].get_x() + w / 2, male_pcts[idx] + 1.5, f"{male_pcts[idx]:.1f}%", ha="center", va="bottom", fontsize=7.8)
        ax_sex.text(bf[idx].get_x() + w / 2, fem_pcts[idx] + 1.5,  f"{fem_pcts[idx]:.1f}%",  ha="center", va="bottom", fontsize=7.8)
        top_y = max(male_pcts[idx], fem_pcts[idx]) + 9.0
        ax_sex.text(
            idx, top_y, f"{n_b:,} beats\n({mu_b:.1f}±{sd_b:.1f}/rec)",
            ha="center", va="bottom", fontsize=7.5, color="#334155",
            bbox=dict(boxstyle="round,pad=0.22", fc="#f8fafc", ec="#cbd5e1", lw=0.7),
        )
    ax_sex.set_title("(b) Sex Ratio & Segmented Beats", fontsize=11, fontweight="bold")
    ax_sex.set_xticks(x)
    ax_sex.set_xticklabels(["PTB-XL", "Georgia", "Chapman-\nShaoxing"], fontsize=9.0)
    ax_sex.set_ylabel("Share of records (%)", fontsize=9.5)
    ax_sex.set_ylim(0, 86)
    ax_sex.grid(True, axis="y", alpha=0.25, ls=":")
    ax_sex.spines[["top", "right"]].set_visible(False)
    ax_sex.legend(fontsize=8.2, loc="upper right", frameon=True)

    # (c) Lead-II normalized beat alignment (1.2 s @ 100 Hz)
    ax_beat = fig.add_subplot(gs[0, 2])
    rng = np.random.default_rng(42)
    t_s = np.linspace(-0.6, 0.6, 120)
    for _key, title, col, st in cohorts:
        arr = st["lead2"]
        if len(arr) > 3000:
            arr = arr[rng.choice(len(arr), size=3000, replace=False)]
        med = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        ax_beat.fill_between(t_s, q25, q75, color=col, alpha=0.14)
        ax_beat.plot(t_s, med, color=col, lw=1.9, label=title)
    ax_beat.axvline(0.0, color="#0f172a", ls=":", lw=1.1, alpha=0.75, label="R-peak (t = 0 s)")
    ax_beat.set_title("(c) Lead-II Beat Morphology (Median + IQR)", fontsize=11, fontweight="bold")
    ax_beat.set_xlabel("Time relative to R-peak (s)", fontsize=9.5)
    ax_beat.set_ylabel("Normalized amplitude", fontsize=9.5)
    ax_beat.set_xlim(-0.6, 0.6)
    ax_beat.grid(True, alpha=0.25, ls=":")
    ax_beat.spines[["top", "right"]].set_visible(False)
    ax_beat.legend(fontsize=7.8, loc="upper right", frameon=True)

    # Bottom row (d, e, f): Diagnostic class bars for each dataset
    bar_specs = [
        (gs[1, 0], "(d) PTB-XL Subclasses (16 classes)", cohorts[0][3]["labels"], COHORT_COLORS["ptbxl"], 16),
        (gs[1, 1], "(e) Georgia Classes (Top 12 of 17)", cohorts[1][3]["labels"], COHORT_COLORS["georgia"], 12),
        (gs[1, 2], "(f) Chapman-Shaoxing Classes (9)",   cohorts[2][3]["labels"], COHORT_COLORS["chapman"], 12),
    ]
    for slot, title, lbls, col, max_bars in bar_specs:
        ax = fig.add_subplot(slot)
        counts = Counter(lbls).most_common(max_bars)
        names = [c[0] for c in counts][::-1]
        vals  = [c[1] for c in counts][::-1]
        total = max(1, len(lbls))
        y = np.arange(len(names))
        bars = ax.barh(y, vals, color=col, alpha=0.86, height=0.72)
        vmax = max(vals) if vals else 1
        for bar, v in zip(bars, vals):
            ax.text(
                v + vmax * 0.015, bar.get_y() + bar.get_height() / 2,
                f"{v:,} ({100.0 * v / total:.1f}%)",
                va="center", ha="left", fontsize=7.4, color="#1e293b",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.0)
        ax.set_xlim(0, vmax * 1.28)
        ax.set_xlabel("Records", fontsize=9.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.25, ls=":")
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "ECG Benchmark Cohorts Overview: Curated PTB-XL vs. Zero-Shot Georgia & Chapman-Shaoxing",
        fontsize=13.5, fontweight="bold", y=0.985,
    )
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved datasets overview -> {out_png}")


def make_head_ablations(out_png: str = "ablation/head_ablations.png") -> None:
    """Plot 3-seed residualization/pruning results and legacy head-ablation sweeps."""
    paper = json.loads(Path("ablation/paper_results_all.json").read_text())
    summary = {s["model"]: s for s in paper["summary"]}

    with open("ablation/ablation_results_all.csv", newline="") as f:
        legacy_rows = {r["variant"]: r for r in csv.DictReader(f)}

    fig = plt.figure(figsize=(16.6, 10.2), dpi=180)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.26)

    variants = [
        ("dpsom_pruned_8x8_no_res", "No Res.\n(probe=0)"),
        ("dpsom_pruned_8x8",        "Pruned Default\n(1-shot res.)"),
        ("dpsom_legacy_full_8x8",   "Legacy Full\n(refit+η+λ+RR)"),
    ]
    x = np.arange(len(variants))

    # (a) 3-seed clustering across residualization & auxiliary head pruning
    ax_a = fig.add_subplot(gs[0, 0])
    cohorts = [
        ("",                     "PTB-XL (in-dist)",         COHORT_COLORS["ptbxl"]),
        ("ext_",                 "Georgia (0-shot)",         COHORT_COLORS["georgia"]),
        ("ext_chapman_shaoxing_","Chapman-Shaoxing (0-shot)",COHORT_COLORS["chapman"]),
    ]
    wd = 0.25
    offsets = [-wd, 0.0, wd]
    max_y = 0.0
    for off, (pfx, lbl, col) in zip(offsets, cohorts):
        means = [summary[k][f"{pfx}Rec_AMI_mean"] for k, _ in variants]
        stds  = [summary[k][f"{pfx}Rec_AMI_std"]  for k, _ in variants]
        max_y = max(max_y, max(m + s for m, s in zip(means, stds)))
        bars = ax_a.bar(x + off, means, wd, yerr=stds, capsize=3, color=col, alpha=0.88, label=lbl)
        for bar, m, s in zip(bars, means, stds):
            ax_a.text(bar.get_x() + bar.get_width() / 2, m + s + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=7.2)
    ax_a.set_title("(a) Residualization & Auxiliary Head Pruning (8×8 SOM, 3 Seeds)", fontsize=11, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([lbl for _, lbl in variants], fontsize=8.8)
    ax_a.set_ylabel("Record-level AMI ↑", fontsize=9.5)
    ax_a.set_ylim(0, max_y + 0.055)
    ax_a.grid(True, axis="y", alpha=0.25, ls=":")
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.legend(fontsize=8.0, loc="upper left", frameon=True)

    # (b) Linear demographic leakage in morphology latent (mu -> mu_res)
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.28)
    ax_r2  = fig.add_subplot(gs_b[0, 0])
    ax_auc = fig.add_subplot(gs_b[0, 1])
    wb = 0.34
    for ax_sub, metric_stem, title, yl, chance_line in [
        (ax_r2,  "age_r2",  "(b1) Linear Age R² ↓\n(mu → mu_res)",   (0.0, 0.50), None),
        (ax_auc, "sex_auc", "(b2) Linear Sex AUROC ↓\n(mu → mu_res)",(0.45, 0.80), 0.50),
    ]:
        raw_m = [summary[k][f"{metric_stem}_raw_mean"] for k, _ in variants]
        raw_s = [summary[k][f"{metric_stem}_raw_std"]  for k, _ in variants]
        res_m = [summary[k][f"{metric_stem}_res_mean"] for k, _ in variants]
        res_s = [summary[k][f"{metric_stem}_res_std"]  for k, _ in variants]
        b_raw = ax_sub.bar(x - wb / 2, raw_m, wb, yerr=raw_s, capsize=3, color="#93c5fd", edgecolor="#2563eb", lw=1.0, label="Raw mu")
        b_res = ax_sub.bar(x + wb / 2, res_m, wb, yerr=res_s, capsize=3, color="#1e40af", label="Residualized mu_res")
        for bar, m, s in zip(b_raw, raw_m, raw_s):
            ax_sub.text(bar.get_x() + wb / 2 - 0.02, m + s + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=6.4)
        for bar, m, s in zip(b_res, res_m, res_s):
            ax_sub.text(bar.get_x() + wb / 2 + 0.02, m + s + 0.008, f"{m:.3f}", ha="center", va="bottom", fontsize=6.4, fontweight="bold")
        if chance_line is not None:
            ax_sub.axhline(chance_line, color="#0f172a", ls=":", lw=1.2, alpha=0.8, label="Random (0.50)")
        ax_sub.set_title(title, fontsize=10.5, fontweight="bold")
        ax_sub.set_xticks(x)
        ax_sub.set_xticklabels(["No Res.", "Pruned\nDefault", "Legacy\nFull"], fontsize=8.3)
        ax_sub.set_ylim(*yl)
        ax_sub.grid(True, axis="y", alpha=0.25, ls=":")
        ax_sub.spines[["top", "right"]].set_visible(False)
        ax_sub.legend(fontsize=7.3, loc="upper right", frameon=True)

    # (c) Legacy demographic supervision head ablations (delta_age, delta_sex, z_age_dim_factor)
    demo_specs = [
        ("baseline",         "baseline (δ=1, f=0.25)"),
        ("no_disentangle",   "no_disentangle (δ=0,0)"),
        ("no_age",           "no_age (δ_age=0)"),
        ("no_sex",           "no_sex (δ_sex=0)"),
        ("delta_age_low",    "δ_age = 0.1"),
        ("delta_age_high",   "δ_age = 5.0"),
        ("delta_sex_low",    "δ_sex = 0.1"),
        ("delta_sex_high",   "δ_sex = 5.0"),
        ("small_age_latent", "z_age frac = 0.125"),
        ("large_age_latent", "z_age frac = 0.50"),
    ][::-1]
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.14)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[0, 1])

    y_c = np.arange(len(demo_specs))
    mae_vals = [float(legacy_rows[k]["MAE_Age"]) for k, _ in demo_specs]
    auc_vals = [float(legacy_rows[k]["AUC_Sex"]) for k, _ in demo_specs]
    base_mae = float(legacy_rows["baseline"]["MAE_Age"])
    base_auc = float(legacy_rows["baseline"]["AUC_Sex"])

    c1_cols = ["#1e40af" if k == "baseline" else ("#dc2626" if v > base_mae + 2.0 else "#60a5fa") for (k, _), v in zip(demo_specs, mae_vals)]
    c2_cols = ["#1e40af" if k == "baseline" else ("#dc2626" if v < base_auc - 0.05 else "#34d399") for (k, _), v in zip(demo_specs, auc_vals)]

    bars_c1 = ax_c1.barh(y_c, mae_vals, color=c1_cols, height=0.68, alpha=0.9)
    ax_c1.axvline(base_mae, color="#0f172a", ls="--", lw=1.1, alpha=0.75)
    for bar, v in zip(bars_c1, mae_vals):
        ax_c1.text(v + 0.35, bar.get_y() + bar.get_height() / 2, f"{v:.2f}", va="center", ha="left", fontsize=7.4)
    ax_c1.set_yticks(y_c)
    ax_c1.set_yticklabels([lbl for _, lbl in demo_specs], fontsize=8.3)
    ax_c1.set_xlim(0, 23.8)
    ax_c1.set_xlabel("Age MAE (yr) ↓", fontsize=9.0)
    ax_c1.set_title("(c1) Age Head (Legacy 1-seed)", fontsize=10.2, fontweight="bold")
    ax_c1.grid(True, axis="x", alpha=0.25, ls=":")
    ax_c1.spines[["top", "right"]].set_visible(False)

    bars_c2 = ax_c2.barh(y_c, auc_vals, color=c2_cols, height=0.68, alpha=0.9)
    ax_c2.axvline(base_auc, color="#0f172a", ls="--", lw=1.1, alpha=0.75)
    ax_c2.axvline(0.5, color="#64748b", ls=":", lw=1.0, alpha=0.7)
    for bar, v in zip(bars_c2, auc_vals):
        ax_c2.text(v + 0.014, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", ha="left", fontsize=7.4)
    ax_c2.set_yticks(y_c)
    ax_c2.set_yticklabels([])
    ax_c2.set_xlim(0.30, 1.04)
    ax_c2.set_xlabel("Sex AUROC ↑", fontsize=9.0)
    ax_c2.set_title("(c2) Sex Head (Legacy 1-seed)", fontsize=10.2, fontweight="bold")
    ax_c2.grid(True, axis="x", alpha=0.25, ls=":")
    ax_c2.spines[["top", "right"]].set_visible(False)

    # (d) Legacy auxiliary heads (age_corr + record attention eta) as deltas vs baseline
    aux_specs = [
        ("no_age_corr",    "no_age_corr (λ=0)"),
        ("less_topk",      "age_corr top-k = 2"),
        ("more_topk",      "age_corr top-k = 8"),
        ("fast_ramp",      "age_corr ramp = 3 ep"),
        ("slow_ramp",      "age_corr ramp = 20 ep"),
        ("no_record_attn", "no_record_attn (η=0)"),
        ("eta_low",        "record attn η = 0.1"),
        ("eta_high",       "record attn η = 5.0"),
    ][::-1]
    gs_d = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.14)
    ax_d1 = fig.add_subplot(gs_d[0, 0])
    ax_d2 = fig.add_subplot(gs_d[0, 1])

    y_d = np.arange(len(aux_specs))
    base_nmi = float(legacy_rows["baseline"]["NMI"])
    dnmi = [float(legacy_rows[k]["NMI"]) - base_nmi for k, _ in aux_specs]
    dmae = [float(legacy_rows[k]["MAE_Age"]) - base_mae for k, _ in aux_specs]

    bars_d1 = ax_d1.barh(y_d, dnmi, color=["#16a34a" if v >= 0 else "#ef4444" for v in dnmi], height=0.64, alpha=0.86)
    ax_d1.axvline(0.0, color="#0f172a", lw=1.0)
    for bar, v in zip(bars_d1, dnmi):
        pad = 0.0006 if v >= 0 else -0.0006
        ax_d1.text(v + pad, bar.get_y() + bar.get_height() / 2, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.4)
    ax_d1.set_yticks(y_d)
    ax_d1.set_yticklabels([lbl for _, lbl in aux_specs], fontsize=8.3)
    ax_d1.set_xlim(-0.017, 0.013)
    ax_d1.set_xlabel("Δ Beat NMI vs baseline (↑)", fontsize=9.0)
    ax_d1.set_title("(d1) Aux Heads: Clustering Δ", fontsize=10.2, fontweight="bold")
    ax_d1.grid(True, axis="x", alpha=0.25, ls=":")
    ax_d1.spines[["top", "right"]].set_visible(False)

    bars_d2 = ax_d2.barh(y_d, dmae, color=["#16a34a" if v <= 0 else "#ef4444" for v in dmae], height=0.64, alpha=0.86)
    ax_d2.axvline(0.0, color="#0f172a", lw=1.0)
    for bar, v in zip(bars_d2, dmae):
        pad = 0.012 if v >= 0 else -0.012
        ax_d2.text(v + pad, bar.get_y() + bar.get_height() / 2, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.4)
    ax_d2.set_yticks(y_d)
    ax_d2.set_yticklabels([])
    ax_d2.set_xlim(-0.38, 0.08)
    ax_d2.set_xlabel("Δ Age MAE vs baseline (↓ better)", fontsize=9.0)
    ax_d2.set_title("(d2) Aux Heads: Age MAE Δ", fontsize=10.2, fontweight="bold")
    ax_d2.grid(True, axis="x", alpha=0.25, ls=":")
    ax_d2.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Head & Residualization Ablations: Cross-Covariance Disentanglement, Demographic Heads & Pruned Modules",
        fontsize=13.5, fontweight="bold", y=0.985,
    )
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved head ablations -> {out_png}")


def _encode_split_to_som(model, records: list, device: torch.device, batch_beats: int = 1024) -> dict:
    """Encode all beats of `records` deterministically onto SOM nodes without dropping any beats."""
    x_list, y_lbl, y_age, y_sex = [], [], [], []
    for r in records:
        if getattr(r, "beat_representations", None) is None or len(r.beat_representations) == 0:
            continue
        beats = np.asarray(r.beat_representations, dtype=np.float32)  # [N, T, C]
        x_list.append(beats.transpose(0, 2, 1))
        n = len(beats)
        y_lbl.extend([str(r.label)] * n)
        y_age.extend([float(r.age)] * n)
        y_sex.extend([int(r.sex)] * n)

    x_arr = np.concatenate(x_list, axis=0)
    mu_chunks, k_chunks = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_arr), batch_beats):
            xb = torch.from_numpy(x_arr[start:start + batch_beats]).to(device)
            mu, _, _, _ = model._encode(xb)
            k = model.k(mu)
            mu_chunks.append(mu.float().cpu().numpy())
            k_chunks.append(k.cpu().numpy())
    mu_np = np.concatenate(mu_chunks, axis=0)
    k_np  = np.concatenate(k_chunks, axis=0).astype(np.int64)
    with torch.inference_mode():
        mu_res_np = model.residualize_latent(torch.from_numpy(mu_np).to(device)).float().cpu().numpy()
    return {
        "mu":     mu_np,
        "mu_res": mu_res_np,
        "k":      k_np,
        "labels": np.asarray(y_lbl),
        "ages":   np.asarray(y_age, dtype=np.float32),
        "sexes":  np.asarray(y_sex, dtype=np.int64),
    }


def _grid_class_stats(k_arr: np.ndarray, labels: np.ndarray, ages: np.ndarray, H: int, W: int) -> tuple:
    """Compute per-cell counts, majority class, purity, and mean age for an H×W SOM."""
    K = H * W
    counts = np.zeros((H, W), dtype=np.int64)
    purity = np.full((H, W), np.nan, dtype=np.float32)
    mean_age = np.full((H, W), np.nan, dtype=np.float32)
    maj_label = np.full((H, W), "", dtype=object)

    for cell in range(K):
        mask = k_arr == cell
        r, c = divmod(cell, W)
        cnt = int(mask.sum())
        counts[r, c] = cnt
        if cnt > 0:
            uniq, cnts = np.unique(labels[mask], return_counts=True)
            j = int(np.argmax(cnts))
            maj_label[r, c] = str(uniq[j])
            purity[r, c] = float(cnts[j] / cnt)
            mean_age[r, c] = float(ages[mask].mean())
    return counts, maj_label, purity, mean_age


def _toroidal_umatrix(E: np.ndarray) -> np.ndarray:
    """Compute mean toroidal 4-neighbor Euclidean distance in latent space for codebook [H, W, D]."""
    H, W, _ = E.shape
    umat = np.zeros((H, W), dtype=np.float32)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nb = np.roll(np.roll(E, shift=dr, axis=0), shift=dc, axis=1)
        umat += np.linalg.norm(E - nb, axis=-1)
    return umat / 4.0


def _draw_som_heatmap(
    ax,
    mat: np.ndarray,
    cmap_name: str,
    title: str,
    cbar_label: str,
    text_grid: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    light_text_thresh: float = 0.58,
    fontsize: float = 6.7,
) -> None:
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color="#f1f5f9")
    lo = float(np.nanmin(mat)) if vmin is None else vmin
    hi = float(np.nanmax(mat)) if vmax is None else vmax
    im = ax.imshow(mat, cmap=cmap, vmin=lo, vmax=hi, aspect="equal")
    H, W = mat.shape
    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(labelsize=7.5)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)

    if text_grid is not None:
        span = hi - lo if hi > lo else 1.0
        for r in range(H):
            for c in range(W):
                val = mat[r, c]
                txt = str(text_grid[r, c])
                if not txt:
                    continue
                if np.isnan(val):
                    tcol = "#64748b"
                else:
                    norm = (float(val) - lo) / span
                    if cmap_name == "coolwarm":
                        tcol = "#0f172a"
                    else:
                        tcol = "white" if norm >= light_text_thresh else "#0f172a"
                ax.text(c, r, txt, ha="center", va="center", fontsize=fontsize, color=tcol, fontweight="medium")

    cbar = plt.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label(cbar_label, fontsize=8.0)


def make_som_visualizations(
    ckpt_path: str = "models/gpu_fast_seed42/gpu_fast_seed42_best.ckpt",
    out_overview: str = "docs/som_overview.png",
    out_prototypes: str = "docs/som_prototypes.png",
    make_overview: bool = True,
    make_prototypes: bool = True,
) -> None:
    """Render 2D toroidal SOM topology/occupancy/age/transfer overview and prototype wave grid."""
    device = _get_device()
    model, meta = load_dpsom_checkpoint(ckpt_path, device)
    H, W = tuple(int(v) for v in meta["som_dim"])
    D = int(meta["latent_dim"])
    C = int(meta["input_channels"])
    T = int(meta["input_length"])

    print(f"[fig] Encoding PTB-XL test, Georgia, and Chapman-Shaoxing onto {H}×{W} SOM ({os.path.basename(ckpt_path)})...")
    ptb_ds = ECG_Dataset.load("data/processed/ptbxl_100_T-12ms.pkl")
    geo_ds = ECG_Dataset.load("data/processed/georgia_100.pkl")
    chp_ds = ECG_Dataset.load("data/processed/chapman_shaoxing_100.pkl")

    enc_ptb = _encode_split_to_som(model, ptb_ds.get_data("test"), device)
    enc_geo = _encode_split_to_som(model, geo_ds.get_data("test"), device)
    enc_chp = _encode_split_to_som(model, chp_ds.get_data("test"), device)

    E = model._embeddings.detach().float().cpu().numpy().reshape(H, W, D)
    umat = _toroidal_umatrix(E)

    som_m = compute_som_clustering_metrics(
        enc_ptb["mu_res"], E.reshape(H * W, D), som_shape=(H, W), labels=enc_ptb["labels"], knn_k=5
    )
    qe = som_m["topology"]["quantization_error"]
    te = som_m["topology"]["topographic_error"]
    act = som_m["utilization"]["active_units"]
    occ_ent = som_m["utilization"]["occupancy_entropy"]

    cnt_ptb, maj_ptb, pur_ptb, age_ptb = _grid_class_stats(enc_ptb["k"], enc_ptb["labels"], enc_ptb["ages"], H, W)
    cnt_geo, maj_geo, pur_geo, _       = _grid_class_stats(enc_geo["k"], enc_geo["labels"], enc_geo["ages"], H, W)
    cnt_chp, maj_chp, pur_chp, _       = _grid_class_stats(enc_chp["k"], enc_chp["labels"], enc_chp["ages"], H, W)

    if make_overview:
        fig, axes = plt.subplots(2, 3, figsize=(17.4, 10.6), dpi=180)
        fig.patch.set_facecolor("white")

        txt_u = np.array([[f"{umat[r, c]:.2f}" for c in range(W)] for r in range(H)], dtype=object)
        _draw_som_heatmap(
            axes[0, 0], umat, "viridis",
            "(a) Toroidal U-Matrix (Mean 4-Nbr Latent Dist.)",
            "Euclidean distance", text_grid=txt_u, light_text_thresh=0.62, fontsize=6.4,
        )

        txt_ptb = np.array([
            [f"{maj_ptb[r, c][:5]}\n{cnt_ptb[r, c]}" if cnt_ptb[r, c] > 0 else "" for c in range(W)]
            for r in range(H)
        ], dtype=object)
        _draw_som_heatmap(
            axes[0, 1], pur_ptb, "YlGnBu",
            f"(b) PTB-XL Test Majority Class & Purity ({len(enc_ptb['k']):,} beats)",
            "Cluster purity", text_grid=txt_ptb, vmin=0.0, vmax=1.0, light_text_thresh=0.55, fontsize=6.0,
        )

        med_age = float(np.median(enc_ptb["ages"]))
        txt_age = np.array([
            [f"{age_ptb[r, c]:.0f}" if cnt_ptb[r, c] > 0 else "" for c in range(W)]
            for r in range(H)
        ], dtype=object)
        _draw_som_heatmap(
            axes[0, 2], age_ptb, "coolwarm",
            f"(c) Mean Patient Age per Node (PTB-XL med={med_age:.0f}y)",
            "Mean age (yr)", text_grid=txt_age, vmin=35.0, vmax=80.0, fontsize=6.6,
        )

        log_cnt = np.log10(1.0 + cnt_ptb.astype(np.float32))
        txt_cnt = np.array([
            [f"{cnt_ptb[r, c]}" if cnt_ptb[r, c] > 0 else "0" for c in range(W)]
            for r in range(H)
        ], dtype=object)
        _draw_som_heatmap(
            axes[1, 0], log_cnt, "Blues",
            f"(d) PTB-XL Beat Occupancy ({act}/{H*W} active, H_occ={occ_ent:.3f})",
            "log10(1 + beats)", text_grid=txt_cnt, vmin=0.0, light_text_thresh=0.58, fontsize=6.8,
        )

        txt_geo = np.array([
            [f"{maj_geo[r, c][:5]}\n{cnt_geo[r, c]}" if cnt_geo[r, c] > 0 else "" for c in range(W)]
            for r in range(H)
        ], dtype=object)
        _draw_som_heatmap(
            axes[1, 1], pur_geo, "PuBuGn",
            f"(e) Zero-Shot Georgia Map ({int((cnt_geo > 0).sum())}/{H*W} nodes hit)",
            "Cluster purity", text_grid=txt_geo, vmin=0.0, vmax=1.0, light_text_thresh=0.55, fontsize=6.0,
        )

        txt_chp = np.array([
            [f"{maj_chp[r, c][:5]}\n{cnt_chp[r, c]}" if cnt_chp[r, c] > 0 else "" for c in range(W)]
            for r in range(H)
        ], dtype=object)
        _draw_som_heatmap(
            axes[1, 2], pur_chp, "OrRd",
            f"(f) Zero-Shot Chapman-Shaoxing Map ({int((cnt_chp > 0).sum())}/{H*W} nodes hit)",
            "Cluster purity", text_grid=txt_chp, vmin=0.0, vmax=1.0, light_text_thresh=0.55, fontsize=6.0,
        )

        fig.suptitle(
            f"Toroidal {H}×{W} SOM Map Analysis ({os.path.basename(ckpt_path)} — QE={qe:.2f}, TE={te:.3f}, Active={act}/{H*W})",
            fontsize=13.0, fontweight="bold", y=0.99,
        )
        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
        os.makedirs(os.path.dirname(out_overview) or ".", exist_ok=True)
        fig.savefig(out_overview, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] Saved SOM overview -> {out_overview}")

    if make_prototypes:
        with torch.inference_mode():
            E_flat = torch.from_numpy(E.reshape(H * W, D)).to(device)
            decoded = model.decoder(E_flat).view(H, W, C, T).cpu().numpy()

        crop = 10
        x_plot = decoded[:, :, :, crop:T - crop]  # [H, W, C, 100] -> 1.0 s centered on R-peak
        lead2 = x_plot[:, :, 1, :]
        leadv2 = x_plot[:, :, 7, :]
        combined_waves = np.concatenate([lead2, leadv2], axis=-1)
        ylo = float(np.min(combined_waves)) - 0.15
        yhi = float(np.max(combined_waves)) + 0.15
        t_axis = np.linspace(-0.5, 0.5, x_plot.shape[-1])

        top_classes = [c for c, _ in Counter(enc_ptb["labels"]).most_common(7)]
        palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0d9488", "#ca8a04"]
        cls_color = {c: palette[i] for i, c in enumerate(top_classes)}
        bg_tint = {
            "#2563eb": "#eff6ff", "#dc2626": "#fef2f2", "#16a34a": "#f0fdf4",
            "#9333ea": "#faf5ff", "#ea580c": "#fff7ed", "#0d9488": "#f0fdfa", "#ca8a04": "#fefce8",
        }

        fig, axes = plt.subplots(H, W, figsize=(15.4, 14.4), dpi=180, sharex=True, sharey=True)
        fig.patch.set_facecolor("white")
        for r in range(H):
            for c in range(W):
                ax = axes[r, c]
                tot = int(cnt_ptb[r, c])
                mcls = str(maj_ptb[r, c]) if tot > 0 else ""
                col = cls_color.get(mcls, "#475569")
                ax.set_facecolor(bg_tint.get(col, "#f8fafc") if tot > 0 else "#f1f5f9")
                ax.axvline(0.0, color="#94a3b8", ls=":", lw=0.7, zorder=1)
                ax.axhline(0.0, color="#cbd5e1", lw=0.6, zorder=1)
                ax.plot(t_axis, leadv2[r, c], color="#64748b", lw=0.95, ls="--", alpha=0.75, zorder=2)
                ax.plot(t_axis, lead2[r, c],  color=col,       lw=1.45, zorder=3)
                ax.set_xlim(t_axis[0], t_axis[-1])
                ax.set_ylim(ylo, yhi)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#cbd5e1")
                    sp.set_linewidth(0.8)
                ttl = f"({r},{c}) {mcls} ({tot})" if tot > 0 else f"({r},{c}) empty"
                ax.set_title(ttl, fontsize=7.6, fontweight="bold" if tot > 0 else "normal", color="#0f172a" if tot > 0 else "#64748b", pad=3)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        handles = [
            Line2D([0], [0], color="#0f172a", lw=1.6, label="Lead II prototype"),
            Line2D([0], [0], color="#64748b", lw=1.1, ls="--", label="Lead V2 prototype"),
        ] + [Patch(facecolor=cls_color[c], label=c) for c in top_classes] + [
            Patch(facecolor="#475569", label="Other / mixed"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.958), ncol=5, fontsize=8.5, frameon=True)
        fig.suptitle(
            f"Decoded {H}×{W} Toroidal SOM Beat Prototypes (Colored by PTB-XL Test Majority Subclass)",
            fontsize=13.5, fontweight="bold", y=0.985,
        )
        fig.subplots_adjust(left=0.02, right=0.985, bottom=0.02, top=0.905, wspace=0.10, hspace=0.28)
        os.makedirs(os.path.dirname(out_prototypes) or ".", exist_ok=True)
        fig.savefig(out_prototypes, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"[fig] Saved SOM prototypes -> {out_prototypes}")


def make_cross_dataset_diagnostics(
    json_path: str = "ablation/cross_dataset_results.json",
    out_png: str = "ablation/cross_dataset_topology.png",
) -> None:
    """Plot transfer retention %, SOM topology health, +UDA deltas, and source scale comparison."""
    from matplotlib.patches import Rectangle

    d = json.loads(Path(json_path).read_text())
    smap = {(r["source"], r["target"], r["regime"]): r for r in d["summary"]}
    in_src = {"ptbxl": "ptbxl_full", "georgia": "georgia", "chapman": "chapman"}

    fig = plt.figure(figsize=(17.0, 10.4), dpi=180)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.28)

    # (a) Cross-Dataset Transfer Retention (% of In-Domain Target)
    ax_a = fig.add_subplot(gs[0, 0])
    pairs_a = [
        ("ptbxl_full", "georgia",  "PTB\n→Geo"),
        ("ptbxl_full", "chapman",  "PTB\n→Chp"),
        ("georgia",    "ptbxl",    "Geo\n→PTB"),
        ("chapman",    "ptbxl",    "Chp\n→PTB"),
        ("georgia",    "chapman",  "Geo\n→Chp"),
        ("chapman",    "georgia",  "Chp\n→Geo"),
    ]
    xa = np.arange(len(pairs_a))
    wa = 0.25
    ret_ami = [100.0 * smap[(s, t, "zero_shot")]["Rec_AMI_mean"] / smap[(in_src[t], t, "zero_shot")]["Rec_AMI_mean"] for s, t, _ in pairs_a]
    ret_age = [100.0 * smap[(in_src[t], t, "zero_shot")]["MAE_Age_mean"] / smap[(s, t, "zero_shot")]["MAE_Age_mean"] for s, t, _ in pairs_a]
    ret_sex = [100.0 * smap[(s, t, "zero_shot")]["AUC_Sex_mean"] / smap[(in_src[t], t, "zero_shot")]["AUC_Sex_mean"] for s, t, _ in pairs_a]

    b1 = ax_a.bar(xa - wa, ret_ami, wa, color="#2563eb", alpha=0.88, label="Clustering (Rec AMI %)")
    b2 = ax_a.bar(xa,      ret_age, wa, color="#0d9488", alpha=0.88, label="Age Pred (MAE_in / MAE_tr %)")
    b3 = ax_a.bar(xa + wa, ret_sex, wa, color="#d97706", alpha=0.88, label="Sex Class (AUC %)")
    ax_a.axhline(100.0, color="#0f172a", ls="--", lw=1.1, alpha=0.8, label="In-domain parity (100%)")
    for bars, vals in ((b1, ret_ami), (b2, ret_age), (b3, ret_sex)):
        for bar, v in zip(bars, vals):
            ax_a.text(bar.get_x() + wa / 2, v + 1.4, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5)
    ax_a.set_title("(a) Zero-Shot Retention Relative to In-Domain Target (%)", fontsize=11.0, fontweight="bold")
    ax_a.set_xticks(xa)
    ax_a.set_xticklabels([lbl for _, _, lbl in pairs_a], fontsize=8.8)
    ax_a.set_ylabel("% of Target In-Domain Score ↑", fontsize=9.5)
    ax_a.set_ylim(0, 128)
    ax_a.grid(True, axis="y", alpha=0.25, ls=":")
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.legend(fontsize=7.6, loc="upper right", ncol=2, frameon=True)

    # (b) Zero-shot SOM Topology & Codebook Health Heatmap (TE, Active/64, QE)
    ax_b = fig.add_subplot(gs[0, 1])
    src_keys = ["ptbxl_full", "ptbxl_2k", "georgia", "chapman"]
    src_lbls = ["PTB-XL Full", "PTB-XL 2k", "Georgia", "Chapman"]
    tgt_keys = ["ptbxl", "georgia", "chapman"]
    tgt_lbls = ["PTB-XL (test)", "Georgia (test)", "Chapman (test)"]

    te_mat = np.zeros((len(src_keys), len(tgt_keys)), dtype=np.float32)
    txt_b = np.full((len(src_keys), len(tgt_keys)), "", dtype=object)
    for i, sk in enumerate(src_keys):
        src_b = "ptbxl" if sk.startswith("ptbxl") else sk
        for j, tk in enumerate(tgt_keys):
            r = smap[(sk, tk, "zero_shot")]
            te_mat[i, j] = r["TE_mean"]
            txt_b[i, j] = f"TE {r['TE_mean']:.3f}\n{r['Active_Nodes_mean']:.0f}/64 · QE {r['QE_mean']:.1f}"

    im = ax_b.imshow(te_mat, cmap="YlOrRd", vmin=0.12, vmax=0.22, aspect="auto")
    ax_b.set_xticks(np.arange(len(tgt_keys)))
    ax_b.set_xticklabels(tgt_lbls, fontsize=8.8)
    ax_b.set_yticks(np.arange(len(src_keys)))
    ax_b.set_yticklabels(src_lbls, fontsize=8.8)
    ax_b.set_title("(b) SOM Topology Health: Topographic Error (TE ↓), Active & QE", fontsize=11.0, fontweight="bold")
    for i, sk in enumerate(src_keys):
        src_b = "ptbxl" if sk.startswith("ptbxl") else sk
        for j, tk in enumerate(tgt_keys):
            norm = (float(te_mat[i, j]) - 0.12) / 0.10
            tcol = "white" if norm >= 0.62 else "#0f172a"
            in_dom = src_b == tk
            ax_b.text(j, i, txt_b[i, j], ha="center", va="center", fontsize=7.6,
                      fontweight="bold" if in_dom else "normal", color=tcol)
            if in_dom:
                ax_b.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec="#0f172a", lw=2.2))
    cbar = plt.colorbar(im, ax=ax_b, fraction=0.044, pad=0.03)
    cbar.set_label("Topographic Error ↓", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.8)

    # (c) Unlabeled SOM Adaptation (+UDA - Zero-Shot) across all 8 off-diagonal pairs
    uda_pairs = [
        ("ptbxl_full", "georgia",  "PTB Full → Geo"),
        ("ptbxl_full", "chapman",  "PTB Full → Chp"),
        ("ptbxl_2k",   "georgia",  "PTB 2k → Geo"),
        ("ptbxl_2k",   "chapman",  "PTB 2k → Chp"),
        ("georgia",    "ptbxl",    "Geo → PTB"),
        ("georgia",    "chapman",  "Geo → Chp"),
        ("chapman",    "ptbxl",    "Chp → PTB"),
        ("chapman",    "georgia",  "Chp → Geo"),
    ][::-1]
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.14)
    ax_c1 = fig.add_subplot(gs_c[0, 0])
    ax_c2 = fig.add_subplot(gs_c[0, 1])

    yc = np.arange(len(uda_pairs))
    dnmi = [smap[(s, t, "uda")]["NMI_mean"] - smap[(s, t, "zero_shot")]["NMI_mean"] for s, t, _ in uda_pairs]
    dte  = [smap[(s, t, "uda")]["TE_mean"]  - smap[(s, t, "zero_shot")]["TE_mean"]  for s, t, _ in uda_pairs]

    b_c1 = ax_c1.barh(yc, dnmi, color=["#16a34a" if v >= 0 else "#ef4444" for v in dnmi], height=0.64, alpha=0.88)
    ax_c1.axvline(0.0, color="#0f172a", lw=1.0)
    for bar, v in zip(b_c1, dnmi):
        ax_c1.text(v + 0.0005, bar.get_y() + bar.get_height() / 2, f"{v:+.4f}", va="center", ha="left", fontsize=7.2)
    ax_c1.set_yticks(yc)
    ax_c1.set_yticklabels([lbl for _, _, lbl in uda_pairs], fontsize=8.2)
    ax_c1.set_xlim(-0.002, max(dnmi) + 0.0065)
    ax_c1.set_xlabel("Δ Beat NMI (↑ better)", fontsize=9.0)
    ax_c1.set_title("(c1) +UDA Clustering Gain", fontsize=10.5, fontweight="bold")
    ax_c1.grid(True, axis="x", alpha=0.25, ls=":")
    ax_c1.spines[["top", "right"]].set_visible(False)

    b_c2 = ax_c2.barh(yc, dte, color=["#16a34a" if v <= 0 else "#ef4444" for v in dte], height=0.64, alpha=0.88)
    ax_c2.axvline(0.0, color="#0f172a", lw=1.0)
    for bar, v in zip(b_c2, dte):
        ax_c2.text(v - 0.0008, bar.get_y() + bar.get_height() / 2, f"{v:+.4f}", va="center", ha="right", fontsize=7.2)
    ax_c2.set_yticks(yc)
    ax_c2.set_yticklabels([])
    ax_c2.set_xlim(min(dte) - 0.012, 0.004)
    ax_c2.set_xlabel("Δ Topographic Error (↓ better)", fontsize=9.0)
    ax_c2.set_title("(c2) +UDA Topology Gain", fontsize=10.5, fontweight="bold")
    ax_c2.grid(True, axis="x", alpha=0.25, ls=":")
    ax_c2.spines[["top", "right"]].set_visible(False)

    # (d) Source Scale vs Domain Shift across Targets
    ax_d = fig.add_subplot(gs[1, 1])
    xt = np.arange(len(tgt_keys))
    wd = 0.19
    src_styles = [
        ("ptbxl_full", "PTB-XL Full (13.5k)", "#1d4ed8"),
        ("ptbxl_2k",   "PTB-XL 2k (1.8k)",    "#93c5fd"),
        ("georgia",    "Georgia (~1.9k)",     "#0d9488"),
        ("chapman",    "Chapman (~1.9k)",     "#d97706"),
    ]
    offs = np.linspace(-1.5, 1.5, len(src_styles)) * wd
    ymax_d = 0.0
    for off, (sk, slbl, col) in zip(offs, src_styles):
        ms = [smap[(sk, tk, "zero_shot")]["Rec_AMI_mean"] for tk in tgt_keys]
        ss = [smap[(sk, tk, "zero_shot")]["Rec_AMI_std"]  for tk in tgt_keys]
        ymax_d = max(ymax_d, max(m + s for m, s in zip(ms, ss)))
        bars = ax_d.bar(xt + off, ms, wd, yerr=ss, capsize=2.5, color=col, edgecolor="#1e293b" if sk == "ptbxl_2k" else "none", lw=0.6, alpha=0.9, label=slbl)
        for bar, m, s in zip(bars, ms, ss):
            ax_d.text(bar.get_x() + wd / 2, m + s + 0.004, f"{m:.3f}", ha="center", va="bottom", fontsize=6.8)
    ax_d.set_title("(d) Source Scale vs. Domain Shift (Zero-Shot Record AMI ↑)", fontsize=11.0, fontweight="bold")
    ax_d.set_xticks(xt)
    ax_d.set_xticklabels(["PTB-XL Target", "Georgia Target", "Chapman Target"], fontsize=9.2)
    ax_d.set_ylabel("Record-level AMI ↑", fontsize=9.5)
    ax_d.set_ylim(0, max(0.25, ymax_d + 0.055))
    ax_d.grid(True, axis="y", alpha=0.25, ls=":")
    ax_d.spines[["top", "right"]].set_visible(False)
    ax_d.legend(fontsize=7.8, loc="upper left", ncol=2, frameon=True)

    fig.suptitle(
        "Cross-Dataset Transfer Diagnostics: Retention, SOM Topology, Unlabeled Adaptation (+UDA) & Source Scale",
        fontsize=13.2, fontweight="bold", y=0.985,
    )
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved cross-dataset diagnostics -> {out_png}")


def _extract_cohort_latents_and_preds(model, ds: ECG_Dataset, use_rr: bool, device: torch.device) -> dict:
    """Collect beat latents (mu, mu_res, z_age, z_sex) and record-level predictions on `test`."""
    gen = ECG_DataGenerator(ds, batch_size=64, use_rr_features=use_rr, preload_to_gpu=False)
    rec_gen = gen.get_record_batch_generator(mode="test", max_beats=None, shuffle=False)
    nb = gen.get_num_batches("test")
    class_names = list(ds.class_names) if getattr(ds, "class_names", None) else []

    mu_l, mures_l, zage_l, zsex_l = [], [], [], []
    bage_l, bsex_l, blbl_l, brec_l = [], [], [], []
    rage_t, rage_p, rsex_t, rsex_p = [], [], [], []
    rec_offset = 0

    model.eval()
    with torch.inference_mode():
        for _ in range(nb):
            beats, _, beat_meta, beat_valid, _, age_t, sex_t, labels_t = next(rec_gen)
            beats = beats.to(device)
            beat_meta = beat_meta.to(device)
            beat_valid = beat_valid.to(device)

            B, N, C, T = beats.shape
            _, zage, zsex, mu, _, _ = model.forward_batch(beats, beat_valid)
            age_pr, sex_lr, _, _ = model.predict_dual_age(
                beats, beat_valid, beat_meta, precomputed=(zage, zsex, mu)
            )
            has_b = (beat_valid.sum(dim=1) > 0.5).cpu().numpy()
            rage_t.append(age_t.cpu().numpy()[has_b])
            rage_p.append(age_pr.float().cpu().numpy()[has_b])
            rsex_t.append(sex_t.cpu().numpy()[has_b])
            rsex_p.append(torch.sigmoid(sex_lr.float()).cpu().numpy()[has_b])

            valid = (beat_valid.reshape(B * N) > 0.5).cpu().numpy()
            mu_f = mu.reshape(B * N, -1).float()[valid]
            mu_res_f = model.residualize_latent(mu_f).cpu().numpy()
            mu_l.append(mu_f.cpu().numpy())
            mures_l.append(mu_res_f)
            zage_l.append(zage.reshape(B * N, -1).float().cpu().numpy()[valid])
            zsex_l.append(zsex.reshape(B * N, -1).float().cpu().numpy()[valid])

            age_exp = age_t.unsqueeze(1).expand(B, N).reshape(B * N).cpu().numpy()[valid]
            sex_exp = sex_t.unsqueeze(1).expand(B, N).reshape(B * N).cpu().numpy()[valid]
            lbl_exp = labels_t.unsqueeze(1).expand(B, N).reshape(B * N).cpu().numpy()[valid]
            rec_exp = (torch.arange(rec_offset, rec_offset + B).unsqueeze(1).expand(B, N).reshape(B * N)).numpy()[valid]
            rec_offset += B
            bage_l.append(age_exp)
            bsex_l.append(sex_exp)
            brec_l.append(rec_exp)
            blbl_l.extend([class_names[int(li)] if 0 <= int(li) < len(class_names) else str(li) for li in lbl_exp])

    return {
        "mu_raw":       np.concatenate(mu_l, axis=0),
        "mu_res":       np.concatenate(mures_l, axis=0),
        "z_age":        np.concatenate(zage_l, axis=0),
        "z_sex":        np.concatenate(zsex_l, axis=0),
        "beat_age":     np.concatenate(bage_l, axis=0),
        "beat_sex":     np.concatenate(bsex_l, axis=0),
        "beat_rec_id":  np.concatenate(brec_l, axis=0),
        "beat_lbl":     np.asarray(blbl_l),
        "rec_age_true": np.concatenate(rage_t, axis=0),
        "rec_age_pred": np.concatenate(rage_p, axis=0),
        "rec_sex_true": np.concatenate(rsex_t, axis=0),
        "rec_sex_prob": np.concatenate(rsex_p, axis=0),
    }


def make_latent_disentanglement(
    ckpt_path: str = "models/gpu_fast_seed42/gpu_fast_seed42_best.ckpt",
    out_png: str = "docs/latent_disentanglement.png",
) -> None:
    """Plot shared t-SNE alignment, z_age PCA, branch leakage profile, and age/sex calibration."""
    device = _get_device()
    model, meta = load_dpsom_checkpoint(ckpt_path, device)
    use_rr = int(meta.get("amp_context_dim", 26)) == 26

    print("[fig] Extracting latent branches & demographic predictions across all 3 cohorts...")
    ptb_ds = ECG_Dataset.load("data/processed/ptbxl_100_T-12ms.pkl")
    geo_ds = ECG_Dataset.load("data/processed/georgia_100.pkl")
    chp_ds = ECG_Dataset.load("data/processed/chapman_shaoxing_100.pkl")

    ext_ptb = _extract_cohort_latents_and_preds(model, ptb_ds, use_rr, device)
    ext_geo = _extract_cohort_latents_and_preds(model, geo_ds, use_rr, device)
    ext_chp = _extract_cohort_latents_and_preds(model, chp_ds, use_rr, device)

    rng = np.random.default_rng(42)
    n_per = 800
    idx_p = rng.choice(len(ext_ptb["mu_res"]), size=min(n_per, len(ext_ptb["mu_res"])), replace=False)
    idx_g = rng.choice(len(ext_geo["mu_res"]), size=min(n_per, len(ext_geo["mu_res"])), replace=False)
    idx_c = rng.choice(len(ext_chp["mu_res"]), size=min(n_per, len(ext_chp["mu_res"])), replace=False)

    comb_mu = np.concatenate([
        ext_ptb["mu_res"][idx_p], ext_geo["mu_res"][idx_g], ext_chp["mu_res"][idx_c]
    ], axis=0)
    coords = TSNE(n_components=2, perplexity=35, random_state=42, init="pca", learning_rate="auto").fit_transform(comb_mu)
    n1, n2 = len(idx_p), len(idx_p) + len(idx_g)
    xy_p, xy_g, xy_c = coords[:n1], coords[n1:n2], coords[n2:]

    fig = plt.figure(figsize=(17.2, 10.2), dpi=180)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 3, hspace=0.38, wspace=0.28)

    # (a) Shared t-SNE colored by Cohort
    ax_a = fig.add_subplot(gs[0, 0])
    for xy, lbl, col in [
        (xy_p, "PTB-XL (in-dist)",        COHORT_COLORS["ptbxl"]),
        (xy_g, "Georgia (0-shot)",         COHORT_COLORS["georgia"]),
        (xy_c, "Chapman-Shaoxing (0-shot)",COHORT_COLORS["chapman"]),
    ]:
        ax_a.scatter(xy[:, 0], xy[:, 1], s=10, alpha=0.45, color=col, label=lbl, edgecolors="none")
    ax_a.set_title("(a) Shared Morphology Space mu_res (t-SNE)", fontsize=10.8, fontweight="bold")
    ax_a.set_xlabel("t-SNE 1", fontsize=9.0)
    ax_a.set_ylabel("t-SNE 2", fontsize=9.0)
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.legend(fontsize=7.8, loc="upper right", markerscale=1.6, frameon=True)

    # (b) PTB-XL t-SNE colored by Subclass
    ax_b = fig.add_subplot(gs[0, 1])
    lbl_p = ext_ptb["beat_lbl"][idx_p]
    top_cls = [c for c, _ in Counter(lbl_p).most_common(6)]
    pal = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0d9488"]
    other_mask = ~np.isin(lbl_p, top_cls)
    ax_b.scatter(xy_p[other_mask, 0], xy_p[other_mask, 1], s=9, alpha=0.28, color="#94a3b8", label="Other", edgecolors="none")
    for c_name, col in zip(top_cls, pal):
        m = lbl_p == c_name
        ax_b.scatter(xy_p[m, 0], xy_p[m, 1], s=12, alpha=0.65, color=col, label=c_name, edgecolors="none")
    ax_b.set_title("(b) PTB-XL Morphology Space mu_res by Subclass", fontsize=10.8, fontweight="bold")
    ax_b.set_xlabel("t-SNE 1", fontsize=9.0)
    ax_b.set_ylabel("t-SNE 2", fontsize=9.0)
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.legend(fontsize=7.5, loc="upper right", ncol=2, markerscale=1.5, frameon=True)

    # (c) z_age PCA colored by True Age
    ax_c = fig.add_subplot(gs[0, 2])
    idx_pca = rng.choice(len(ext_ptb["z_age"]), size=min(4000, len(ext_ptb["z_age"])), replace=False)
    pca_age = PCA(n_components=2, random_state=42)
    zage_2d = pca_age.fit_transform(ext_ptb["z_age"][idx_pca])
    sc = ax_c.scatter(
        zage_2d[:, 0], zage_2d[:, 1], c=ext_ptb["beat_age"][idx_pca],
        cmap="viridis", vmin=20, vmax=88, s=8, alpha=0.55, edgecolors="none",
    )
    cbar = plt.colorbar(sc, ax=ax_c, fraction=0.045, pad=0.03)
    cbar.set_label("True age (yr)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.8)
    vr = pca_age.explained_variance_ratio_
    ax_c.set_title("(c) Dedicated Age Branch z_age (PCA, Colored by Age)", fontsize=10.8, fontweight="bold")
    ax_c.set_xlabel(f"PC1 ({100 * vr[0]:.1f}% var)", fontsize=9.0)
    ax_c.set_ylabel(f"PC2 ({100 * vr[1]:.1f}% var)", fontsize=9.0)
    ax_c.spines[["top", "right"]].set_visible(False)

    # (d) Branch-to-Factor Disentanglement Profile (Record-grouped linear probes)
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ax_d = fig.add_subplot(gs[1, 0])
    branches = [("mu_raw", "Raw\nmu"), ("mu_res", "Residualized\nmu_res"), ("z_age", "Age branch\nz_age"), ("z_sex", "Sex branch\nz_sex")]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
    tr_idx, te_idx = next(gss.split(ext_ptb["mu_raw"], ext_ptb["beat_age"], ext_ptb["beat_rec_id"]))
    y_age_tr, y_age_te = ext_ptb["beat_age"][tr_idx], ext_ptb["beat_age"][te_idx]
    y_sex_tr, y_sex_te = ext_ptb["beat_sex"][tr_idx], ext_ptb["beat_sex"][te_idx]

    r2_age, som_sex = [], []
    for bk, _ in branches:
        Xtr, Xte = ext_ptb[bk][tr_idx], ext_ptb[bk][te_idx]
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(Xtr, y_age_tr)
        r2_age.append(max(0.0, float(metrics.r2_score(y_age_te, ridge.predict(Xte)))))
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(Xtr, y_sex_tr)
        auc_v = float(metrics.roc_auc_score(y_sex_te, clf.predict_proba(Xte)[:, 1]))
        som_sex.append(max(0.0, 2.0 * auc_v - 1.0))  # Gini / Somer's D in [0, 1]

    xd = np.arange(len(branches))
    wb = 0.34
    ba = ax_d.bar(xd - wb / 2, r2_age,  wb, color="#2563eb", alpha=0.88, label="Age predictivity (Ridge R²)")
    bs = ax_d.bar(xd + wb / 2, som_sex, wb, color="#ec4899", alpha=0.84, label="Sex predictivity (2·AUROC − 1)")
    for bar, v in zip(ba, r2_age):
        ax_d.text(bar.get_x() + wb / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.4)
    for bar, v in zip(bs, som_sex):
        ax_d.text(bar.get_x() + wb / 2, v + 0.015, f"{v:.3f}", ha="center", va="bottom", fontsize=7.4)
    ax_d.set_title("(d) Linear Demographic Predictivity by Branch", fontsize=10.5, fontweight="bold")
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([lbl for _, lbl in branches], fontsize=8.8)
    ax_d.set_ylabel("Predictivity score [0..1]", fontsize=9.5)
    ax_d.set_ylim(0, max(r2_age + som_sex) + 0.16)
    ax_d.grid(True, axis="y", alpha=0.25, ls=":")
    ax_d.spines[["top", "right"]].set_visible(False)
    ax_d.legend(fontsize=7.6, loc="upper left", frameon=True)

    # (e) Record-Level Age Calibration (Predicted vs True)
    ax_e = fig.add_subplot(gs[1, 1])
    ax_e.scatter(ext_ptb["rec_age_true"], ext_ptb["rec_age_pred"], s=7, alpha=0.14, color="#93c5fd", edgecolors="none", label="_nolegend_")
    edges = np.arange(20, 95, 10)
    for ext_c, title_c, col_c in [
        (ext_ptb, "PTB-XL",   COHORT_COLORS["ptbxl"]),
        (ext_geo, "Georgia",  COHORT_COLORS["georgia"]),
        (ext_chp, "Chapman",  COHORT_COLORS["chapman"]),
    ]:
        yt, yp = ext_c["rec_age_true"], ext_c["rec_age_pred"]
        mae_v = float(metrics.mean_absolute_error(yt, yp))
        r_v = float(np.corrcoef(yt, yp)[0, 1])
        bx, by, bs_err = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (yt >= lo) & (yt < hi)
            if int(m.sum()) >= 5:
                bx.append(float(yt[m].mean()))
                by.append(float(yp[m].mean()))
                bs_err.append(float(yp[m].std(ddof=1) / np.sqrt(m.sum())))
        ax_e.errorbar(bx, by, yerr=bs_err, marker="o", ms=5, lw=2.0, capsize=3, color=col_c,
                      label=f"{title_c} (MAE={mae_v:.2f}y, r={r_v:.2f})")
    ax_e.plot([15, 90], [15, 90], "k--", lw=1.1, alpha=0.75, label="Ideal y = x")
    ax_e.set_title("(e) Record-Level Age Calibration (Decade Mean ± SEM)", fontsize=10.8, fontweight="bold")
    ax_e.set_xlabel("True Age (years)", fontsize=9.2)
    ax_e.set_ylabel("Predicted Age (years)", fontsize=9.2)
    ax_e.set_xlim(15, 92)
    ax_e.set_ylim(15, 92)
    ax_e.grid(True, alpha=0.25, ls=":")
    ax_e.spines[["top", "right"]].set_visible(False)
    ax_e.legend(fontsize=7.6, loc="upper left", frameon=True)

    # (f) Record-Level Sex ROC Curves
    ax_f = fig.add_subplot(gs[1, 2])
    for ext_c, title_c, col_c in [
        (ext_ptb, "PTB-XL (in-dist)", COHORT_COLORS["ptbxl"]),
        (ext_geo, "Georgia (0-shot)", COHORT_COLORS["georgia"]),
        (ext_chp, "Chapman (0-shot)", COHORT_COLORS["chapman"]),
    ]:
        fpr, tpr, _ = metrics.roc_curve(ext_c["rec_sex_true"], ext_c["rec_sex_prob"])
        auc_v = float(metrics.roc_auc_score(ext_c["rec_sex_true"], ext_c["rec_sex_prob"]))
        ax_f.plot(fpr, tpr, lw=2.0, color=col_c, label=f"{title_c} (AUC = {auc_v:.3f})")
    ax_f.plot([0, 1], [0, 1], "k:", lw=1.1, alpha=0.7, label="Random (AUC = 0.500)")
    ax_f.set_title("(f) Record-Level Sex ROC (In-Dist & 0-Shot)", fontsize=10.8, fontweight="bold")
    ax_f.set_xlabel("False Positive Rate", fontsize=9.2)
    ax_f.set_ylabel("True Positive Rate", fontsize=9.2)
    ax_f.set_xlim(-0.02, 1.02)
    ax_f.set_ylim(-0.02, 1.02)
    ax_f.grid(True, alpha=0.25, ls=":")
    ax_f.spines[["top", "right"]].set_visible(False)
    ax_f.legend(fontsize=8.0, loc="lower right", frameon=True)

    fig.suptitle(
        f"Latent Disentanglement, Cross-Cohort Alignment & Demographic Calibration ({os.path.basename(ckpt_path)})",
        fontsize=13.2, fontweight="bold", y=0.985,
    )
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved latent disentanglement -> {out_png}")


def _pick_representative_record(records: list, prefer_labels: tuple[str, ...]):
    """Select the first valid test record matching `prefer_labels` (with >= 5 beats)."""
    valid_recs = [
        r for r in records
        if getattr(r, "beat_representations", None) is not None and len(r.beat_representations) >= 5
        and getattr(r, "age", 999) <= 100
    ]
    for lbl in prefer_labels:
        for r in valid_recs:
            if str(r.label) == lbl:
                return r
    return valid_recs[0]


def make_ecg_reconstructions(
    ckpt_path: str = "models/gpu_fast_seed42/gpu_fast_seed42_best.ckpt",
    out_png: str = "docs/ecg_reconstructions.png",
) -> None:
    """Plot 6-lead original beat vs. DPSOM reconstruction vs. assigned SOM prototype across 3 cohorts."""
    device = _get_device()
    model, meta = load_dpsom_checkpoint(ckpt_path, device)
    use_rr = int(meta.get("amp_context_dim", 26)) == 26
    H, W = tuple(int(v) for v in meta["som_dim"])

    ptb_ds = ECG_Dataset.load("data/processed/ptbxl_100_T-12ms.pkl")
    geo_ds = ECG_Dataset.load("data/processed/georgia_100.pkl")
    chp_ds = ECG_Dataset.load("data/processed/chapman_shaoxing_100.pkl")

    picks = [
        ("PTB-XL (in-dist)",         ptb_ds, _pick_representative_record(ptb_ds.get_data("test"), ("CLBBB", "IMI", "LVH"))),
        ("Georgia (0-shot)",         geo_ds, _pick_representative_record(geo_ds.get_data("test"), ("RBBB", "AFIB", "TInv"))),
        ("Chapman-Shaoxing (0-shot)",chp_ds, _pick_representative_record(chp_ds.get_data("test"), ("AFL", "SB", "STach"))),
    ]

    lead_indices = [0, 1, 3, 6, 7, 10]
    lead_labels = ["I", "II", "aVR", "V1", "V2", "V5"]
    crop = 10

    fig = plt.figure(figsize=(17.0, 9.4), dpi=180)
    fig.patch.set_facecolor("white")
    outer = gridspec.GridSpec(3, 1, hspace=0.46)

    for row_idx, (cohort_title, ds_ref, rec) in enumerate(picks):
        sub_ds = ECG_Dataset(fs=ds_ref.fs)
        sub_ds.records = [rec]
        sub_ds.class_names = list(ds_ref.class_names)
        sub_ds.label_encoder = ds_ref.label_encoder
        gen = ECG_DataGenerator(sub_ds, batch_size=1, use_rr_features=use_rr, preload_to_gpu=False)
        batch = next(gen.get_record_batch_generator(mode="test", max_beats=None, shuffle=False))
        beats, _, beat_meta, beat_valid, _, age_t, sex_t, _ = batch
        beats = beats.to(device)
        beat_meta = beat_meta.to(device)
        beat_valid = beat_valid.to(device)

        with torch.inference_mode():
            age_pr, sex_lr, _, _ = model.predict_dual_age(beats, beat_valid, beat_meta)
            n_b = int(beat_valid[0].sum().item())
            xb = beats[0, :n_b]  # [n_b, C, T]
            mu_b, _, _, _ = model._encode(xb)
            mu_res_b = model.residualize_latent(mu_b.float())
            med_idx = int(torch.argmin(torch.norm(mu_res_b - mu_res_b.mean(dim=0, keepdim=True), dim=-1)).item())

            _, C_in, T_in = xb.shape
            k_sel = int(model.k(mu_b[med_idx:med_idx + 1]).item())
            sr, sc = divmod(k_sel, W)
            proto_emb = model._embeddings[sr, sc].unsqueeze(0)
            recon = model.decoder(mu_b[med_idx:med_idx + 1]).view(C_in, T_in).float().cpu().numpy()
            proto_rec = model.decoder(proto_emb).view(C_in, T_in).float().cpu().numpy()

        orig = xb[med_idx].float().cpu().numpy()
        rmse = float(np.sqrt(np.mean((orig - recon) ** 2)))

        age_p = float(age_pr[0].item())
        sex_p_fem = float(torch.sigmoid(sex_lr[0]).item())
        sex_true_str = "F" if int(sex_t[0].item()) == 1 else "M"
        sex_pred_str = f"{'F' if sex_p_fem >= 0.5 else 'M'} (p={sex_p_fem:.2f})"

        t_axis = np.linspace(-0.5, 0.5, orig.shape[1] - 2 * crop)
        orig_c  = orig[:, crop:-crop]
        recon_c = recon[:, crop:-crop]
        proto_c = proto_rec[:, crop:-crop]
        y_all = np.concatenate([orig_c[lead_indices], recon_c[lead_indices], proto_c[lead_indices]], axis=-1)
        ylo, yhi = float(np.min(y_all)) - 0.15, float(np.max(y_all)) + 0.15

        # Row title + 6 lead panels
        ax_hdr = fig.add_subplot(outer[row_idx])
        ax_hdr.axis("off")
        ax_hdr.set_title(
            f"{cohort_title}  |  Dx: {rec.label}  |  Age: {rec.age:.0f}y → pred {age_p:.1f}y  |  "
            f"Sex: {sex_true_str} → {sex_pred_str}  |  SOM Node: ({sr},{sc})  |  12-lead RMSE = {rmse:.3f}",
            fontsize=10.6, fontweight="bold", pad=16, loc="left", color="#0f172a",
        )
        inner = gridspec.GridSpecFromSubplotSpec(1, len(lead_indices), subplot_spec=outer[row_idx], wspace=0.14)
        for li_pos, (li, lname) in enumerate(zip(lead_indices, lead_labels)):
            ax = fig.add_subplot(inner[0, li_pos])
            ax.axvline(0.0, color="#cbd5e1", ls=":", lw=0.8)
            ax.axhline(0.0, color="#e2e8f0", lw=0.7)
            l1, = ax.plot(t_axis, orig_c[li],  color="#2563eb", lw=1.45, label="Original beat (100 Hz)")
            l2, = ax.plot(t_axis, recon_c[li], color="#ea580c", lw=1.30, ls="--", label="DPSOM reconstruction dec(mu)")
            l3, = ax.plot(t_axis, proto_c[li], color="#475569", lw=1.10, ls=":", label="Assigned SOM prototype dec(e_k)")
            ax.set_title(f"Lead {lname}", fontsize=9.0, fontweight="bold", pad=3)
            ax.set_xlim(t_axis[0], t_axis[-1])
            ax.set_ylim(ylo, yhi)
            ax.grid(True, alpha=0.20, ls=":")
            ax.spines[["top", "right"]].set_visible(False)
            if li_pos > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("Norm. amp.", fontsize=8.5)
            ax.tick_params(labelsize=7.5)
            if row_idx == 2:
                ax.set_xlabel("Time (s)", fontsize=8.2)
            if row_idx == 0 and li_pos == 0:
                legend_handles = [l1, l2, l3]

    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, fontsize=9.2, frameon=True)
    fig.suptitle(
        f"Representative Multi-Cohort 6-Lead Beat Reconstructions & SOM Node Prototypes ({os.path.basename(ckpt_path)})",
        fontsize=13.2, fontweight="bold", y=0.985,
    )
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved ECG reconstructions -> {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="Generate dataset, ablation, SOM, latent, and reconstruction figures")
    p.add_argument(
        "--only", nargs="+",
        choices=["datasets", "heads", "som", "prototypes", "transfer_diag", "latent", "recon"],
        default=["datasets", "heads", "som", "prototypes", "transfer_diag", "latent", "recon"],
    )
    p.add_argument("--ckpt", type=str, default="models/gpu_fast_seed42/gpu_fast_seed42_best.ckpt")
    args = p.parse_args()

    targets = set(args.only)
    if "datasets" in targets:
        make_datasets_overview()
    if "heads" in targets:
        make_head_ablations()
    if "som" in targets or "prototypes" in targets:
        make_som_visualizations(
            ckpt_path=args.ckpt,
            make_overview=("som" in targets),
            make_prototypes=("prototypes" in targets),
        )
    if "transfer_diag" in targets:
        make_cross_dataset_diagnostics()
    if "latent" in targets:
        make_latent_disentanglement(ckpt_path=args.ckpt)
    if "recon" in targets:
        make_ecg_reconstructions(ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
