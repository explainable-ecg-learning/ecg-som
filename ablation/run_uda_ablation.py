"""Stronger Unsupervised Domain Adaptation (UDA) Ablation for DPSOM-ECG.

Benchmarks 7 adaptation regimes across all 8 off-diagonal transfers (PTB-XL, Georgia, Chapman):
  1. zero_shot         : Frozen source model evaluated directly on target test split
  2. uda_vanilla       : Baseline UDA (_adapt_unsupervised: beat-level DEC + SOM + ELBO)
  3. uda_bw_ot         : Closed-form Bures-Wasserstein Optimal Transport (0.1s, 0 backprop, 0 labels)
  4. uda_sf_strong     : Source-Free Hybrid Beat+Record Damped-DEC + SWA (0 source data, 0 labels)
  5. uda_ot_strong     : BW-OT Demo Alignment + Hybrid Beat+Record Damped-DEC + SWA (0 labels)
  6. uda_replay_strong : BW-OT Demo Init + Joint Source-Replay Hybrid Damped-DEC + SWA (0 labels)
  7. fewshot_100_head  : BW-OT + Head-Only 100-Shot Demographic Calibration (CNNs & SOM frozen)

Usage:
    # Parallel 2-GPU run:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u ablation/run_uda_ablation.py \
        --sources ptbxl_full georgia --seeds 42 123 456 --epochs 30 --output ablation/uda_gpu0.json
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u ablation/run_uda_ablation.py \
        --sources ptbxl_2k chapman --seeds 42 123 456 --epochs 30 --output ablation/uda_gpu1.json

    # Merge & plot:
    .venv/bin/python -u ablation/run_uda_ablation.py \
        --merge ablation/uda_gpu0.json ablation/uda_gpu1.json \
        --output ablation/uda_ablation_results.json \
        --plot ablation/uda_ablation.png
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from src.config import DPSOM_Config
from src.data.dataset import ECG_Dataset
from src.training.uda import (
    adapt_fewshot_demo_heads,
    adapt_uda_bw_ot,
    adapt_uda_ot_strong,
    adapt_uda_replay_strong,
    adapt_uda_sf_strong,
    extract_domain_moments,
)
from run_cross_dataset_ablation import (
    SOURCES,
    TARGETS,
    _adapt_unsupervised,
    _compute_chance,
    _eval_on_target,
    _source_base,
    _subsample_ptbxl,
    _train_source_model,
    _write_outputs,
)


REGIMES = (
    "zero_shot",
    "uda_vanilla",
    "uda_bw_ot",
    "uda_sf_strong",
    "uda_ot_strong",
    "uda_replay_strong",
    "fewshot_100_head",
)


def plot_uda_ablation(json_path: str, out_png: str = "ablation/uda_ablation.png") -> None:
    """Create 3-panel strong-UDA diagnostic comparison figure."""
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    d = json.loads(Path(json_path).read_text())
    smap = {(r["source"], r["target"], r["regime"]): r for r in d["summary"]}

    pairs = [
        ("ptbxl_full", "georgia",  "PTB-XL Full → Georgia"),
        ("ptbxl_full", "chapman",  "PTB-XL Full → Chapman"),
        ("ptbxl_2k",   "georgia",  "PTB-XL 2k → Georgia"),
        ("ptbxl_2k",   "chapman",  "PTB-XL 2k → Chapman"),
        ("georgia",    "ptbxl",    "Georgia → PTB-XL"),
        ("chapman",    "ptbxl",    "Chapman → PTB-XL"),
        ("georgia",    "chapman",  "Georgia → Chapman"),
        ("chapman",    "georgia",  "Chapman → Georgia"),
    ]
    strong_pairs = [p for p in pairs if p[0] == "ptbxl_full"]

    comp_regimes = [
        "uda_vanilla",
        "uda_bw_ot",
        "uda_sf_strong",
        "uda_ot_strong",
        "uda_replay_strong",
    ]
    reg_labels = {
        "zero_shot":         "Zero-Shot",
        "uda_vanilla":       "Vanilla\nUDA",
        "uda_bw_ot":         "BW-OT\n(0-ep)",
        "uda_sf_strong":     "SF Hybrid\nDEC+SWA",
        "uda_ot_strong":     "OT+Hybrid\nDEC+SWA",
        "uda_replay_strong": "Replay+\nDEC+SWA",
        "fewshot_100_head":  "100-Shot\nHead Cal.",
    }

    fig = plt.figure(figsize=(21.5, 6.8), dpi=180)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.08, 1.34, 1.04], wspace=0.52)

    # Panel (a): Mean Δ Rec AMI vs Zero-Shot
    ax_a = fig.add_subplot(gs[0, 0])
    all_deltas = [
        float(np.mean([smap[(s, t, reg)]["Rec_AMI_mean"] - smap[(s, t, "zero_shot")]["Rec_AMI_mean"] for s, t, _ in pairs]))
        for reg in comp_regimes
    ]
    strong_deltas = [
        float(np.mean([smap[(s, t, reg)]["Rec_AMI_mean"] - smap[(s, t, "zero_shot")]["Rec_AMI_mean"] for s, t, _ in strong_pairs]))
        for reg in comp_regimes
    ]
    x = np.arange(len(comp_regimes)) * 1.15
    w = 0.42
    b1 = ax_a.bar(x - w / 2, all_deltas, w, color="#2563eb", alpha=0.90, label="All 8 Transfer Pairs")
    b2 = ax_a.bar(x + w / 2, strong_deltas, w, color="#f97316", alpha=0.90, label="Strong Source (PTB-XL Full → *)")
    ax_a.axhline(0.0, color="#0f172a", lw=1.0, ls="--", alpha=0.75)

    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            va = "bottom" if val >= 0 else "top"
            off = 0.00035 if val >= 0 else -0.00045
            ax_a.text(
                bar.get_x() + bar.get_width() / 2,
                val + off,
                f"{val:+.3f}",
                ha="center",
                va=va,
                fontsize=7.2,
            )

    ymin = min(-0.005, min(all_deltas + strong_deltas) - 0.0025)
    ymax = max(0.010, max(all_deltas + strong_deltas) + 0.0045)
    ax_a.set_ylim(ymin, ymax)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([reg_labels[r] for r in comp_regimes], fontsize=7.8)
    ax_a.set_ylabel("Mean Δ Rec AMI vs. Zero-Shot ↑", fontsize=9.2)
    ax_a.set_title("(a) Mean Rec AMI Gain Over Zero-Shot", fontsize=10.5, fontweight="bold", pad=8)
    ax_a.grid(True, axis="y", alpha=0.25, ls=":")
    ax_a.spines[["top", "right"]].set_visible(False)
    ax_a.legend(fontsize=7.8, loc="upper left", frameon=True)

    # Panel (b): Heatmap of Δ Rec AMI across 8 pairs x 5 zero-label UDA regimes
    ax_b = fig.add_subplot(gs[0, 1])
    uda_cols = ["uda_vanilla", "uda_bw_ot", "uda_sf_strong", "uda_ot_strong", "uda_replay_strong"]
    mat = np.zeros((len(pairs), len(uda_cols)), dtype=np.float32)
    te_mat = np.zeros((len(pairs), len(uda_cols)), dtype=np.float32)
    for i, (s, t, _) in enumerate(pairs):
        zs_ami = smap[(s, t, "zero_shot")]["Rec_AMI_mean"]
        zs_te = smap[(s, t, "zero_shot")]["TE_mean"]
        for j, reg in enumerate(uda_cols):
            mat[i, j] = smap[(s, t, reg)]["Rec_AMI_mean"] - zs_ami
            te_mat[i, j] = smap[(s, t, reg)]["TE_mean"] - zs_te

    lim = max(0.015, float(np.max(np.abs(mat))))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    im = ax_b.imshow(mat, cmap="RdBu", norm=norm, aspect="auto")
    ax_b.set_xticks(np.arange(len(uda_cols)))
    ax_b.set_xticklabels([reg_labels[r] for r in uda_cols], fontsize=7.6)
    ax_b.set_yticks(np.arange(len(pairs)))
    ax_b.set_yticklabels([lbl for _, _, lbl in pairs], fontsize=8.0)
    ax_b.set_title("(b) Per-Pair Δ Rec AMI (top) & Δ TE (bottom)", fontsize=10.5, fontweight="bold", pad=8)

    for i in range(len(pairs)):
        best_j = int(np.argmax(mat[i]))
        for j in range(len(uda_cols)):
            val = float(mat[i, j])
            tval = float(te_mat[i, j])
            tcol = "white" if abs(val) > 0.58 * lim else "#0f172a"
            ax_b.text(
                j,
                i,
                f"{val:+.3f}\n(TE {tval:+.3f})",
                ha="center",
                va="center",
                fontsize=6.8,
                fontweight="bold" if j == best_j else "normal",
                color=tcol,
            )
    cbar = plt.colorbar(im, ax=ax_b, fraction=0.044, pad=0.03)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label("Δ Rec AMI", fontsize=8.2)

    # Panel (c): Demographic adaptation comparison (Age MAE ↓ & Sex AUROC ↑)
    ax_c = fig.add_subplot(gs[0, 2])
    demo_regs = ["zero_shot", "uda_vanilla", "uda_bw_ot", "uda_ot_strong", "fewshot_100_head"]
    age_means = [float(np.mean([smap[(s, t, reg)]["MAE_Age_mean"] for s, t, _ in pairs])) for reg in demo_regs]
    sex_means = [float(np.mean([smap[(s, t, reg)]["AUC_Sex_mean"] for s, t, _ in pairs])) for reg in demo_regs]

    xc = np.arange(len(demo_regs))
    wc = 0.36
    ax_c2 = ax_c.twinx()

    bars_age = ax_c.bar(xc - wc / 2, age_means, wc, color="#dc2626", alpha=0.84, label="Age MAE (yr) ↓")
    bars_sex = ax_c2.bar(xc + wc / 2, sex_means, wc, color="#0d9488", alpha=0.88, label="Sex AUROC ↑")

    ax_c.set_ylim(0.0, max(age_means) * 1.34)
    ax_c2.set_ylim(0.65, min(1.0, max(sex_means) + 0.075))

    for bar, v in zip(bars_age, age_means):
        ax_c.text(bar.get_x() + bar.get_width() / 2, v + 0.25, f"{v:.2f}", ha="center", va="bottom",
                  fontsize=7.3, color="#991b1b", fontweight="bold")
    for bar, v in zip(bars_sex, sex_means):
        ax_c2.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom",
                   fontsize=7.3, color="#115e59", fontweight="bold")

    ax_c.set_xticks(xc)
    ax_c.set_xticklabels([reg_labels[r] for r in demo_regs], fontsize=7.7)
    ax_c.set_ylabel("Age MAE (yr) ↓", fontsize=9.2, color="#991b1b")
    ax_c2.set_ylabel("Sex AUROC ↑", fontsize=9.2, color="#115e59")
    ax_c.set_title("(c) Demographic Shift: BW-OT vs. 100-Shot", fontsize=10.5, fontweight="bold", pad=8)
    ax_c.grid(True, axis="y", alpha=0.22, ls=":")
    ax_c.spines["top"].set_visible(False)
    ax_c2.spines["top"].set_visible(False)

    h1, l1 = ax_c.get_legend_handles_labels()
    h2, l2 = ax_c2.get_legend_handles_labels()
    ax_c.legend(h1 + h2, l1 + l2, fontsize=7.6, loc="upper center", ncol=2, frameon=True)

    fig.suptitle(
        "BW-OT + Hybrid Damped-DEC UDA for DPSOM-ECG Across PTB-XL, Georgia & Chapman (8 Pairs × 3 Seeds)",
        fontsize=12.8,
        fontweight="bold",
        y=1.01,
    )
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[uda-abl] Saved figure -> {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stronger UDA ablation for DPSOM-ECG")
    p.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--output", type=str, default="ablation/uda_ablation_results.json")
    p.add_argument("--merge", nargs="+", default=None)
    p.add_argument("--plot", type=str, default=None)
    args = p.parse_args()

    cfg = DPSOM_Config()
    print("[uda-abl] Loading benchmark datasets (PTB-XL, Georgia transfer, Chapman transfer)...")
    ptb_full = ECG_Dataset.load("data/processed/ptbxl_100_T-12ms.pkl")
    if os.path.exists(cfg.error_labels_path):
        ptb_full.apply_error_labels(
            cfg.error_labels_path,
            max_error_severity=cfg.max_error_severity,
            exclude_error_flags=cfg.exclude_error_flags,
        )
    ptb_2k = _subsample_ptbxl(ptb_full, n_train=1800, n_val=225, seed=42)
    geo_ds = ECG_Dataset.load("data/processed/georgia_transfer_100.pkl")
    chp_ds = ECG_Dataset.load("data/processed/chapman_transfer_100.pkl")

    source_ds = {
        "ptbxl_full": ptb_full,
        "ptbxl_2k":   ptb_2k,
        "georgia":    geo_ds,
        "chapman":    chp_ds,
    }
    target_ds = {
        "ptbxl":   ptb_full,
        "georgia": geo_ds,
        "chapman": chp_ds,
    }
    chance = _compute_chance(target_ds)

    if args.merge:
        merged: list[dict] = []
        seen: set[tuple[str, str, str, int]] = set()
        for path in args.merge:
            d = json.loads(Path(path).read_text())
            for r in d["runs"]:
                key = (r["source"], r["target"], r["regime"], int(r["seed"]))
                if key not in seen:
                    seen.add(key)
                    merged.append(r)
        s_ord = {k: i for i, k in enumerate(SOURCES)}
        t_ord = {k: i for i, k in enumerate(TARGETS)}
        r_ord = {k: i for i, k in enumerate(REGIMES)}
        merged.sort(key=lambda r: (
            s_ord.get(r["source"], 99),
            t_ord.get(r["target"], 99),
            r_ord.get(r["regime"], 99),
            int(r["seed"]),
        ))
        _write_outputs(merged, args.output, chance)
        print(f"[uda-abl] Merged {len(merged)} rows -> {args.output}")
        if args.plot:
            plot_uda_ablation(args.output, args.plot)
        return

    rows: list[dict] = []
    for src in args.sources:
        for seed in args.seeds:
            t0 = time.time()
            print(f"\n{'=' * 66}\n  SOURCE: {src}  |  SEED: {seed}\n{'=' * 66}")
            sds = source_ds[src]
            src_model = _train_source_model(src, sds, seed, args.epochs)
            src_moments = extract_domain_moments(src_model, sds, use_rr_features=False)

            for tgt in TARGETS:
                if _source_base(src) == tgt:
                    continue
                tds = target_ds[tgt]

                builders = [
                    ("zero_shot",         lambda: src_model),
                    ("uda_vanilla",       lambda: _adapt_unsupervised(src_model, tds, batch_size=64, adapt_epochs=6, lr=1e-4, use_amp=cfg.use_amp)),
                    ("uda_bw_ot",         lambda: adapt_uda_bw_ot(src_model, tds, src_moments, s_morph=0.35, s_age=0.25, s_sex=0.75, use_rr_features=False)),
                    ("uda_sf_strong",     lambda: adapt_uda_sf_strong(src_model, tds, use_amp=cfg.use_amp, use_rr_features=False)),
                    ("uda_ot_strong",     lambda: adapt_uda_ot_strong(src_model, tds, src_moments, use_amp=cfg.use_amp, use_rr_features=False)),
                    ("uda_replay_strong", lambda: adapt_uda_replay_strong(src_model, sds, tds, src_moments, use_amp=cfg.use_amp, use_rr_features=False)),
                    ("fewshot_100_head",  lambda: adapt_fewshot_demo_heads(src_model, tds, src_moments=src_moments, n_shots=100, seed=seed, use_amp=cfg.use_amp, use_rr_features=False)),
                ]

                for reg, build_fn in builders:
                    model_v = build_fn()
                    res = _eval_on_target(model_v, tds, use_rr=False, batch_size=64)
                    rows.append({"source": src, "target": tgt, "regime": reg, "seed": seed, **res})
                    print(
                        f"  [{reg:17s}] {src:10s}->{tgt:7s}: RecAMI={res['Rec_AMI']:.4f}  "
                        f"NMI={res['NMI']:.4f}  MAE={res['MAE_Age']:.2f}  AUC={res['AUC_Sex']:.4f}  "
                        f"TE={res['TE']:.3f}  Act={res['Active_Nodes']}/64"
                    )

            print(f"[uda-abl] Completed {src} s={seed} in {time.time() - t0:.1f}s")
            _write_outputs(rows, args.output, chance)

    if args.plot:
        plot_uda_ablation(args.output, args.plot)


if __name__ == "__main__":
    main()
