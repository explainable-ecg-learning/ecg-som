"""DPSOM-ECG parameter ablation & sensitivity analysis runner (pruned defaults).

Evaluates core DPSOM loss ablations and 1-D parameter sensitivity sweeps across:
  - SOM grid size       : 4x4, 6x6, 8x8*, 12x12, 16x16
  - Student-t df (alpha): 1.0, 5.0, 10.0*, 25.0, 50.0
  - SOM smoothing (beta): 0.0, 0.1, 0.5*, 2.0, 5.0
  - SOM commit (gamma)  : 0.0, 1.0, 5.0, 10.0*, 25.0, 50.0
  - Temporal (tau)      : 0.0, 0.4, 1.6*, 4.0, 8.0
  - ELBO/recon (theta)  : 0.0, 0.02, 0.1*, 0.5, 1.0
  - Latent dim          : 16, 32*, 64
  - Residualization     : num_epochs_probe = 0 vs 3*

Reports PTB-XL test (NMI, AMI, Purity, MAE_Age, AUC_Sex), toroidal SOM topology
(QE, TE, Active_Frac, Occ_Entropy), and zero-shot Georgia 12-Lead ECG transfer.

Usage:
    # Run in parallel across 2 GPUs:
    CUDA_VISIBLE_DEVICES=0 python ablation/run_dpsom_sensitivity.py --shard 0/2 --output ablation/sens_gpu0.json
    CUDA_VISIBLE_DEVICES=1 python ablation/run_dpsom_sensitivity.py --shard 1/2 --output ablation/sens_gpu1.json

    # Merge shards & plot sensitivity curves:
    python ablation/run_dpsom_sensitivity.py --merge ablation/sens_gpu0.json ablation/sens_gpu1.json \
        --output ablation/dpsom_sensitivity_results.json --plot ablation/dpsom_sensitivity.png
"""
from __future__ import annotations

import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import csv
import json
import shutil
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.config import DPSOM_Config
from src.data.dataset import ECG_Dataset
from src.data.generator import ECG_DataGenerator
from src.model.dpsom import DPSOM_ECG
from src.training.trainer import (
    _amp_ctx,
    _get_device,
    _load_checkpoint_state,
    _set_global_determinism,
    evaluate_model,
    train_model,
)
from src.utils.metrics import compute_som_clustering_metrics
from run_paper_experiments import _eval_external_dpsom


# ---------------------------------------------------------------------------
# Variant registry (all relative to pruned default DPSOM_Config)
# ---------------------------------------------------------------------------

VARIANTS: OrderedDict[str, tuple[str, str, dict]] = OrderedDict([
    # (sweep_group, display_value, config_overrides)
    ("default",          ("default", "8x8 / default", {})),

    # --- Core loss & residualization ablations ---
    ("gamma_0",          ("gamma",   "0",    {"gamma": 0.0})),
    ("beta_0",           ("beta",    "0",    {"beta": 0.0})),
    ("tau_0",            ("tau",     "0",    {"tau": 0.0})),
    ("theta_0",          ("theta",   "0",    {"theta": 0.0})),
    ("probe_0",          ("probe",   "0",    {"num_epochs_probe": 0})),

    # --- SOM grid size ---
    ("som_4x4",          ("som_dim", "4x4",   {"som_dim": (4, 4)})),
    ("som_6x6",          ("som_dim", "6x6",   {"som_dim": (6, 6)})),
    ("som_12x12",        ("som_dim", "12x12", {"som_dim": (12, 12)})),
    ("som_16x16",        ("som_dim", "16x16", {"som_dim": (16, 16)})),

    # --- Student-t degrees of freedom (alpha) ---
    ("alpha_1",          ("alpha",   "1",    {"alpha": 1.0})),
    ("alpha_5",          ("alpha",   "5",    {"alpha": 5.0})),
    ("alpha_25",         ("alpha",   "25",   {"alpha": 25.0})),
    ("alpha_50",         ("alpha",   "50",   {"alpha": 50.0})),

    # --- SOM neighbourhood smoothing (beta) ---
    ("beta_0.1",         ("beta",    "0.1",  {"beta": 0.1})),
    ("beta_2.0",         ("beta",    "2.0",  {"beta": 2.0})),
    ("beta_5.0",         ("beta",    "5.0",  {"beta": 5.0})),

    # --- SOM commitment KL(p || q) weight (gamma) ---
    ("gamma_1",          ("gamma",   "1",    {"gamma": 1.0})),
    ("gamma_5",          ("gamma",   "5",    {"gamma": 5.0})),
    ("gamma_25",         ("gamma",   "25",   {"gamma": 25.0})),
    ("gamma_50",         ("gamma",   "50",   {"gamma": 50.0})),

    # --- Temporal beat-sequence smoothness (tau) ---
    ("tau_0.4",          ("tau",     "0.4",  {"tau": 0.4})),
    ("tau_4.0",          ("tau",     "4.0",  {"tau": 4.0})),
    ("tau_8.0",          ("tau",     "8.0",  {"tau": 8.0})),

    # --- ELBO / reconstruction weight (theta) ---
    ("theta_0.02",       ("theta",   "0.02", {"theta": 0.02})),
    ("theta_0.5",        ("theta",   "0.5",  {"theta": 0.5})),
    ("theta_1.0",        ("theta",   "1.0",  {"theta": 1.0})),

    # --- Latent dimensionality ---
    ("latent_16",        ("latent",  "16",   {"latent_dim": 16})),
    ("latent_64",        ("latent",  "64",   {"latent_dim": 64})),
])


@torch.no_grad()
def _eval_som_topology(model: DPSOM_ECG, gen: ECG_DataGenerator, cfg: DPSOM_Config, device: torch.device) -> dict:
    model.eval()
    z_list, lbl_list = [], []
    test_records = gen.ecg_dataset.get_data("test")
    nb = int(np.ceil(len(test_records) / gen.batch_size))
    test_gen = gen.get_record_batch_generator(mode="test", shuffle=False)
    for _ in range(nb):
        beats, _, _, beat_valid, _, _, _, rec_labels = next(test_gen)
        beats = beats.to(device, non_blocking=True)
        beat_valid = beat_valid.to(device, non_blocking=True)
        B, N, _, _ = beats.shape
        _, _, _, mu, _, _ = model.forward_batch(beats, beat_valid)
        mu_res = model.residualize_latent(mu.reshape(B * N, -1))
        vf = (beat_valid.reshape(B * N) > 0.5).cpu().numpy()
        if not vf.any():
            continue
        z_list.append(mu_res.float().cpu().numpy()[vf])
        lbl_list.append(rec_labels.unsqueeze(1).expand(B, N).reshape(B * N).cpu().numpy()[vf])

    z_all = np.concatenate(z_list, axis=0)
    lbl_all = np.concatenate(lbl_list, axis=0)
    emb = getattr(model, "embeddings", model._embeddings)
    codebook = emb.detach().float().cpu().numpy().reshape(-1, z_all.shape[1])
    som_m = compute_som_clustering_metrics(z_all, codebook, cfg.som_dim, labels=lbl_all, knn_k=10)
    util = som_m["utilization"]
    topo = som_m["topology"]
    return {
        "Active_Nodes": int(util["active_units"]),
        "Total_Nodes": int(util["total_units"]),
        "Active_Frac": float(util["active_units"] / max(1, util["total_units"])),
        "Occ_Entropy": float(util["occupancy_entropy"]),
        "QE": float(topo["quantization_error"]),
        "TE": float(topo["topographic_error"]),
        "Nbr_Consistency": float(topo["neighborhood_consistency"]),
    }


def run_variant(
    name: str,
    group: str,
    val_label: str,
    overrides: dict,
    ds: ECG_Dataset,
    ext_ds: ECG_Dataset | None,
    seed: int,
    epochs: int,
    device: torch.device,
) -> dict:
    t0 = time.time()
    cfg = DPSOM_Config()
    cfg.random_seed = seed
    cfg.num_epochs = epochs
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.ex_name = f"sens_{name}_s{seed}"
    cfg.logdir = f"logs/{cfg.ex_name}"
    cfg.modelpath = f"models/{cfg.ex_name}/{cfg.ex_name}.ckpt"

    _set_global_determinism(seed, fast_gpu=cfg.fast_gpu)
    gen = ECG_DataGenerator(
        ds, batch_size=cfg.batch_size, use_rr_features=cfg.use_rr_features, preload_to_gpu=cfg.preload_to_gpu
    )
    sample = gen.get_flat_beats_tensor("train")
    input_channels, input_length = sample.shape[1], sample.shape[2]
    amp_dim = 2 * input_channels + (2 if cfg.use_rr_features else 0)

    model = DPSOM_ECG(
        latent_dim=cfg.latent_dim,
        som_dim=cfg.som_dim,
        input_length=input_length,
        input_channels=input_channels,
        alpha=cfg.alpha,
        beta=cfg.beta,
        theta=cfg.theta,
        gamma=cfg.gamma,
        tau=cfg.tau,
        eta=cfg.eta,
        delta_age=cfg.delta_age,
        delta_sex=cfg.delta_sex,
        dropout=cfg.dropout,
        prior_var=cfg.prior_var,
        prior=cfg.prior,
        encoder_base_channels_1=cfg.encoder_base_channels_1,
        encoder_base_channels_2=cfg.encoder_base_channels_2,
        encoder_kernel_size=cfg.encoder_kernel_size,
        encoder_fc_hidden_dim=cfg.encoder_fc_hidden_dim,
        z_age_dim_factor=cfg.z_age_dim_factor,
        z_sex_dim_factor=cfg.z_sex_dim_factor,
        age_corr_topk=cfg.age_corr_topk,
        age_corr_lambda_max=cfg.age_corr_lambda_max,
        age_corr_ramp_epochs=cfg.age_corr_ramp_epochs,
        som_init_std=cfg.som_init_std,
        amp_context_dim=amp_dim,
    )

    best_ckpt = train_model(
        model, gen,
        num_epochs=cfg.num_epochs,
        batch_size=cfg.batch_size,
        ex_name=cfg.ex_name,
        model_path=cfg.modelpath,
        learning_rate=cfg.learning_rate,
        epochs_pretrain=cfg.num_epochs_pretrain,
        som_dim=cfg.som_dim,
        learning_rate_pretrain=cfg.learning_rate_pretrain,
        decay_factor=cfg.decay_factor,
        decay_steps=cfg.decay_steps,
        num_used_beats=cfg.num_beats,
        weight_decay=cfg.weight_decay,
        gradient_clip_norm=cfg.gradient_clip_norm,
        num_epochs_probe=cfg.num_epochs_probe,
        lr_probe=cfg.lr_probe,
        lr_meta_factor=cfg.lr_meta_factor,
        probe_refit_interval=cfg.probe_refit_interval,
        probe_refit_epochs=cfg.probe_refit_epochs,
        use_amp=cfg.use_amp,
        fast_gpu=cfg.fast_gpu,
    )
    if cfg.use_best_checkpoint and best_ckpt and os.path.exists(best_ckpt):
        state, _ = _load_checkpoint_state(best_ckpt, device)
        model.load_state_dict(state, strict=True)

    eval_res = evaluate_model(model, gen, compute_disentanglement=False)
    topo_res = _eval_som_topology(model, gen, cfg, device)
    row = {
        "variant": name,
        "group": group,
        "value": val_label,
        "seed": seed,
        "NMI": float(eval_res["NMI"]),
        "AMI": float(eval_res["AMI"]),
        "Purity": float(eval_res["Purity"]),
        "MAE_Age": float(eval_res["MAE_Age"]),
        "AUC_Sex": float(eval_res["AUC_Sex"]),
        **topo_res,
        "elapsed_s": round(time.time() - t0, 1),
    }
    if ext_ds is not None:
        row.update(_eval_external_dpsom(model, ext_ds, cfg.use_rr_features, cfg.batch_size))

    # Clean up temporary checkpoint & log dir immediately
    shutil.rmtree(os.path.dirname(cfg.modelpath), ignore_errors=True)
    shutil.rmtree(cfg.logdir, ignore_errors=True)
    return row


def _save_outputs(rows: list[dict], out_json: str) -> None:
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"runs": rows}, f, indent=2)
    out_csv = out_json[:-5] + ".csv" if out_json.endswith(".json") else out_json + ".csv"
    if rows:
        keys = list(rows[0].keys())
        for r in rows[1:]:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def _load_paper_baselines(paper_json: str = "ablation/paper_results_all.json") -> dict[str, dict]:
    if not os.path.exists(paper_json):
        return {}
    d = json.loads(Path(paper_json).read_text())
    return {s["model"]: s for s in d.get("summary", [])}


def plot_sensitivity(
    rows: list[dict],
    out_png: str,
    paper_json: str = "ablation/paper_results_all.json",
) -> None:
    import matplotlib.pyplot as plt

    by_name = {r["variant"]: r for r in rows}
    default = by_name["default"]
    baselines = _load_paper_baselines(paper_json)
    vae = baselines.get("vae_kmeans")
    resnet = baselines.get("resnet1d_supervised")
    best_16 = baselines.get("dpsom_pruned_16x16") or baselines.get("dpsom_16x16")

    panels = [
        ("SOM Grid Size (H×W)", [
            ("4×4", by_name.get("som_4x4")),
            ("6×6", by_name.get("som_6x6")),
            ("8×8*", default),
            ("12×12", by_name.get("som_12x12")),
            ("16×16", by_name.get("som_16x16")),
        ]),
        ("Student-t Degrees of Freedom (α)", [
            ("1", by_name.get("alpha_1")),
            ("5", by_name.get("alpha_5")),
            ("10*", default),
            ("25", by_name.get("alpha_25")),
            ("50", by_name.get("alpha_50")),
        ]),
        ("SOM Smoothing Weight (β)", [
            ("0", by_name.get("beta_0")),
            ("0.1", by_name.get("beta_0.1")),
            ("0.5*", default),
            ("2.0", by_name.get("beta_2.0")),
            ("5.0", by_name.get("beta_5.0")),
        ]),
        ("SOM Commitment Weight (γ)", [
            ("0", by_name.get("gamma_0")),
            ("1", by_name.get("gamma_1")),
            ("5", by_name.get("gamma_5")),
            ("10*", default),
            ("25", by_name.get("gamma_25")),
            ("50", by_name.get("gamma_50")),
        ]),
        ("Temporal Smoothness Weight (τ)", [
            ("0", by_name.get("tau_0")),
            ("0.4", by_name.get("tau_0.4")),
            ("1.6*", default),
            ("4.0", by_name.get("tau_4.0")),
            ("8.0", by_name.get("tau_8.0")),
        ]),
        ("ELBO / Recon Weight (θ)", [
            ("0", by_name.get("theta_0")),
            ("0.02", by_name.get("theta_0.02")),
            ("0.1*", default),
            ("0.5", by_name.get("theta_0.5")),
            ("1.0", by_name.get("theta_1.0")),
        ]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.8), dpi=180)
    fig.patch.set_facecolor("white")

    legend_handles = []
    legend_labels = []

    for idx_panel, (ax, (title, items)) in enumerate(zip(axes.flat, panels)):
        valid_items = [(lbl, r) for lbl, r in items if r is not None]
        xs = np.arange(len(valid_items))
        labels = [lbl for lbl, _ in valid_items]
        nmi = [r["NMI"] for _, r in valid_items]
        ext_nmi = [r.get("ext_NMI", np.nan) for _, r in valid_items]
        mae = [r["MAE_Age"] for _, r in valid_items]

        l1 = ax.plot(xs, nmi, marker="o", lw=2.1, color="#1d4ed8", label="DPSOM (1-seed) PTB-XL Beat NMI ↑", zorder=4)
        l2 = ax.plot(xs, ext_nmi, marker="s", lw=2.1, ls="--", color="#0d9488", label="DPSOM (1-seed) Georgia Beat NMI ↑", zorder=4)

        ref_lines = []
        if vae:
            r1 = ax.axhline(vae["NMI_mean"], color="#64748b", ls=":", lw=1.3, alpha=0.9,
                            label=f"VAE+KM (k=64) PTB-XL NMI ({vae['NMI_mean']:.3f})")
            r2 = ax.axhline(vae["ext_NMI_mean"], color="#0f766e", ls=":", lw=1.3, alpha=0.65,
                            label=f"VAE+KM (k=64) Georgia NMI ({vae['ext_NMI_mean']:.3f})")
            ref_lines.extend([r1, r2])
        if resnet:
            r3 = ax.axhline(resnet["NMI_mean"], color="#dc2626", ls="-.", lw=1.3, alpha=0.8,
                            label=f"1D-ResNet (16-cls) PTB-XL NMI ({resnet['NMI_mean']:.3f})")
            r4 = ax.axhline(resnet["ext_NMI_mean"], color="#be185d", ls=":", lw=1.4, alpha=0.8,
                            label=f"1D-ResNet (16-cls) Georgia NMI ({resnet['ext_NMI_mean']:.3f})")
            ref_lines.extend([r3, r4])
        if best_16 and idx_panel > 0:
            r5 = ax.axhline(best_16["NMI_mean"], color="#1e3a8a", ls=(0, (5, 2)), lw=1.1, alpha=0.55,
                            label=f"DPSOM 16×16 PTB-XL Beat NMI ({best_16['NMI_mean']:.3f})")
            r6 = ax.axhline(best_16["ext_NMI_mean"], color="#047857", ls=(0, (5, 2)), lw=1.1, alpha=0.55,
                            label=f"DPSOM 16×16 Georgia Beat NMI ({best_16['ext_NMI_mean']:.3f})")
            ref_lines.extend([r5, r6])

        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Beat NMI ↑ (Note: k-dependent)", fontsize=8.5, color="#1e293b")
        ax.set_ylim(0.12, 0.44)
        ax.grid(True, axis="y", alpha=0.22, ls=":")
        ax.spines["top"].set_visible(False)

        for idx, lbl in enumerate(labels):
            if lbl.endswith("*"):
                ax.axvline(idx, color="#94a3b8", ls=":", lw=1.2, zorder=0)

        ax2 = ax.twinx()
        l3 = ax2.plot(xs, mae, marker="^", lw=1.5, ls="-.", color="#d97706", alpha=0.8, label="DPSOM (1-seed) PTB-XL Age MAE ↓", zorder=3)
        ax2.set_ylabel("Age MAE (yr) ↓", fontsize=8.5, color="#b45309")
        ax2.tick_params(axis="y", labelcolor="#b45309", labelsize=8.5)
        ax2.spines["top"].set_visible(False)

        if idx_panel == 1:
            all_h = l1 + l2 + l3 + ref_lines
            legend_handles = all_h
            legend_labels = [h.get_label() for h in all_h]

    fig.suptitle(
        "DPSOM-ECG Parameter Sensitivity (Single-Seed, 24 ep) vs. Fair 3-Seed References (* = 8×8 Default)",
        fontsize=12.5, fontweight="bold", y=0.99,
    )
    if legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc="upper center", bbox_to_anchor=(0.5, 0.955),
            ncol=3, fontsize=8.0, frameon=True, facecolor="#f8fafc", edgecolor="#cbd5e1",
        )
    plt.tight_layout(rect=[0, 0, 1, 0.86])
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[sens] Saved sensitivity figure -> {out_png}")


def plot_baseline_comparison(
    out_png: str,
    paper_json: str = "ablation/paper_results_all.json",
) -> None:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    if not os.path.exists(paper_json):
        return
    d = json.loads(Path(paper_json).read_text())
    baselines = {s["model"]: s for s in d.get("summary", [])}
    chance = d.get("chance", {})
    vae = baselines.get("vae_kmeans")
    resnet = baselines.get("resnet1d_supervised")
    dp4 = baselines.get("dpsom_pruned_4x4")
    dp8 = baselines.get("dpsom_pruned_8x8")
    dp16 = baselines.get("dpsom_pruned_16x16")
    if not (vae and resnet and dp4 and dp8 and dp16):
        return

    has_chapman = "ext_chapman_shaoxing_Rec_AMI_mean" in dp8
    ks = (16, 64, 256)
    xk = np.arange(len(ks))
    wk = 0.25
    dpsom_by_k = {16: dp4, 64: dp8, 256: dp16}

    if has_chapman:
        fig = plt.figure(figsize=(16.0, 8.2), dpi=180)
        fig.patch.set_facecolor("white")
        gs = gridspec.GridSpec(2, 6, hspace=0.40, wspace=0.55)
        clust_specs = [
            (fig.add_subplot(gs[0, 0:2]), "", "PTB-XL Record AMI ↑ (Matched k)", "ptbxl_test"),
            (fig.add_subplot(gs[0, 2:4]), "ext_", "Georgia 0-Shot Record AMI ↑ (Matched k)", "georgia_test"),
            (fig.add_subplot(gs[0, 4:6]), "ext_chapman_shaoxing_", "Chapman-Shaoxing 0-Shot Record AMI ↑ (Matched k)", "chapman_shaoxing_test"),
        ]
        ax_age = fig.add_subplot(gs[1, 0:3])
        ax_sex = fig.add_subplot(gs[1, 3:6])
        demo_cohorts = [
            ("", "#2563eb", "PTB-XL (in-dist)"),
            ("ext_", "#0d9488", "Georgia (0-shot)"),
            ("ext_chapman_shaoxing_", "#d97706", "Chapman-Shaoxing (0-shot)"),
        ]
        wd = 0.26
    else:
        fig, axes = plt.subplots(1, 4, figsize=(16.8, 4.6), dpi=180)
        fig.patch.set_facecolor("white")
        clust_specs = [
            (axes[0], "", "PTB-XL Record AMI ↑ (Matched k)", "ptbxl_test"),
            (axes[1], "ext_", "Georgia 0-Shot Record AMI ↑ (Matched k)", "georgia_test"),
        ]
        ax_age, ax_sex = axes[2], axes[3]
        demo_cohorts = [
            ("", "#2563eb", "PTB-XL (in-dist)"),
            ("ext_", "#0d9488", "Georgia (0-shot)"),
        ]
        wd = 0.36

    # Top / left panels: Matched-k Record-Level AMI (k = 16, 64, 256)
    for ax, prefix, title, ch_key in clust_specs:
        vae_m = [vae[f"{prefix}Rec_AMI_k{k}_mean"] for k in ks]
        vae_s = [vae[f"{prefix}Rec_AMI_k{k}_std"] for k in ks]
        dp_m = [dpsom_by_k[k][f"{prefix}Rec_AMI_mean"] for k in ks]
        dp_s = [dpsom_by_k[k][f"{prefix}Rec_AMI_std"] for k in ks]
        rn_m = [resnet[f"{prefix}Rec_AMI_k{k}_mean"] for k in ks]
        rn_s = [resnet[f"{prefix}Rec_AMI_k{k}_std"] for k in ks]

        b_vae = ax.bar(xk - wk, vae_m, wk, yerr=vae_s, capsize=3, color="#64748b", alpha=0.9, label="VAE + K-Means (Unsup)")
        b_dp = ax.bar(xk, dp_m, wk, yerr=dp_s, capsize=3, color="#2563eb", alpha=0.9, label="DPSOM-ECG Pruned (Unsup)")
        b_rn = ax.bar(xk + wk, rn_m, wk, yerr=rn_s, capsize=3, color="#dc2626", alpha=0.85, label="1D-ResNet h+KM (Sup)")

        ch_vals = [chance.get(ch_key, {}).get(f"k{k}", {}).get("random_record", {}).get("AMI", 0.0) for k in ks]
        ax.axhline(float(np.mean(ch_vals)), color="#0f172a", ls=":", lw=1.2, alpha=0.7, label="Random-record chance (~0.000)")

        ymax = max([v + s for v, s in zip(vae_m + dp_m + rn_m, vae_s + dp_s + rn_s)] + [0.22])
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xticks(xk)
        ax.set_xticklabels([f"k={k}" for k in ks], fontsize=9)
        ax.set_ylim(0, max(0.37 if not prefix else 0.30, ymax + (0.11 if not prefix else 0.08)))
        ax.grid(True, axis="y", alpha=0.25, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
        for bars, vals, errs in ((b_vae, vae_m, vae_s), (b_dp, dp_m, dp_s), (b_rn, rn_m, rn_s)):
            for bar, v, s in zip(bars, vals, errs):
                ax.text(bar.get_x() + bar.get_width() / 2, v + s + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=6.7)
        if not prefix:
            ax.legend(fontsize=7.3, loc="upper right", frameon=True)

    # Demographic Prediction panels (Age MAE ↓ & Sex AUROC ↑)
    demo_specs = [
        ("VAE\n(mu only)", vae, "MAE_Age", "AUC_Sex"),
        ("VAE\n(mu+amp)", vae, "MAE_Age_amp", "AUC_Sex_amp"),
        ("DPSOM\n8×8", dp8, "MAE_Age", "AUC_Sex"),
        ("DPSOM\n16×16", dp16, "MAE_Age", "AUC_Sex"),
        ("1D-ResNet\n(Sup)", resnet, "MAE_Age", "AUC_Sex"),
    ]
    xd = np.arange(len(demo_specs))
    xlabels = [lbl for lbl, _, _, _ in demo_specs]
    offsets = np.linspace(-(len(demo_cohorts) - 1) / 2, (len(demo_cohorts) - 1) / 2, len(demo_cohorts)) * wd

    # Age MAE ↓
    max_age = 0.0
    for off, (pfx, color, lbl) in zip(offsets, demo_cohorts):
        m = [row[f"{pfx}{age_k}_mean"] for _, row, age_k, _ in demo_specs]
        s = [row[f"{pfx}{age_k}_std"] for _, row, age_k, _ in demo_specs]
        max_age = max(max_age, max(v + err for v, err in zip(m, s)))
        bars = ax_age.bar(xd + off, m, wd, yerr=s, capsize=2.5, color=color, alpha=0.88, label=lbl)
        for bar, v, err in zip(bars, m, s):
            ax_age.text(bar.get_x() + bar.get_width() / 2, v + err + 0.25, f"{v:.1f}", ha="center", va="bottom", fontsize=6.6)
    ax_age.set_title("Age Prediction MAE (yr) ↓", fontsize=10.5, fontweight="bold")
    ax_age.set_xticks(xd)
    ax_age.set_xticklabels(xlabels, fontsize=8.0)
    ax_age.set_ylim(0, max(15.0, max_age + 2.4))
    ax_age.grid(True, axis="y", alpha=0.25, ls=":")
    ax_age.spines[["top", "right"]].set_visible(False)
    ax_age.legend(fontsize=7.5, loc="upper right", frameon=True)

    # Sex AUROC ↑
    for off, (pfx, color, _lbl) in zip(offsets, demo_cohorts):
        m = [row[f"{pfx}{sex_k}_mean"] for _, row, _, sex_k in demo_specs]
        s = [row[f"{pfx}{sex_k}_std"] for _, row, _, sex_k in demo_specs]
        bars = ax_sex.bar(xd + off, m, wd, yerr=s, capsize=2.5, color=color, alpha=0.88)
        for bar, v, err in zip(bars, m, s):
            ax_sex.text(bar.get_x() + bar.get_width() / 2, v + err + 0.010, f"{v:.2f}", ha="center", va="bottom", fontsize=6.6)
    ax_sex.set_title("Sex Classification AUROC ↑", fontsize=10.5, fontweight="bold")
    ax_sex.set_xticks(xd)
    ax_sex.set_xticklabels(xlabels, fontsize=8.0)
    ax_sex.set_ylim(0.5, 1.02)
    ax_sex.grid(True, axis="y", alpha=0.25, ls=":")
    ax_sex.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Fair Matched-k & Demographic Benchmark (3-Seed Mean ± Std, max_beats=None, Val-Selected)",
        fontsize=12, fontweight="bold", y=0.98 if has_chapman else 1.02,
    )
    if not has_chapman:
        plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[sens] Saved baseline comparison figure -> {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="DPSOM parameter ablation & sensitivity analysis")
    p.add_argument("--variants", nargs="+", default=None, help="Specific variant names to run")
    p.add_argument("--shard", type=str, default=None, help="Shard index/total, e.g. 0/2 or 1/2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--ext-data", type=str, default="data/processed/georgia_100.pkl")
    p.add_argument("--output", type=str, default="ablation/dpsom_sensitivity_results.json")
    p.add_argument("--merge", nargs="+", default=None, help="Merge multiple JSON shard files")
    p.add_argument("--plot", type=str, default=None, help="Output PNG path for sensitivity plot")
    p.add_argument("--plot-baselines", type=str, default=None, help="Output PNG path for baseline comparison bar chart")
    args = p.parse_args()

    if args.merge:
        merged: list[dict] = []
        seen: set[str] = set()
        for path in args.merge:
            d = json.loads(Path(path).read_text())
            for r in d["runs"]:
                if r["variant"] not in seen:
                    seen.add(r["variant"])
                    merged.append(r)
        order = list(VARIANTS.keys())
        merged.sort(key=lambda r: order.index(r["variant"]) if r["variant"] in order else 999)
        _save_outputs(merged, args.output)
        print(f"[sens] Merged {len(merged)} runs -> {args.output}")
        if args.plot:
            plot_sensitivity(merged, args.plot)
        if args.plot_baselines:
            plot_baseline_comparison(args.plot_baselines)
        return

    names = args.variants if args.variants else list(VARIANTS.keys())
    if args.shard:
        idx_s, tot_s = args.shard.split("/")
        idx, tot = int(idx_s), int(tot_s)
        names = [n for i, n in enumerate(names) if i % tot == idx]

    cfg_base = DPSOM_Config()
    print(f"[sens] Loading PTB-XL once... ({len(names)} variants scheduled)")
    ds = ECG_Dataset.load("data/processed/ptbxl_100_T-12ms.pkl")
    if os.path.exists(cfg_base.error_labels_path):
        ds.apply_error_labels(
            cfg_base.error_labels_path,
            max_error_severity=cfg_base.max_error_severity,
            exclude_error_flags=cfg_base.exclude_error_flags,
        )

    ext_ds = None
    if args.ext_data and os.path.exists(args.ext_data):
        ext_ds = ECG_Dataset.load(args.ext_data)
        print(f"[sens] Loaded external dataset: {args.ext_data} ({len(ext_ds.records)} records)")

    device = _get_device()
    rows: list[dict] = []
    for i, name in enumerate(names, 1):
        group, val_label, overrides = VARIANTS[name]
        print(f"\n[{i}/{len(names)}] Running {name} ({group}={val_label}) ...")
        row = run_variant(name, group, val_label, overrides, ds, ext_ds, args.seed, args.epochs, device)
        rows.append(row)
        print(
            f"  -> NMI={row['NMI']:.4f}  Pur={row['Purity']:.4f}  MAE={row['MAE_Age']:.2f}  AUC={row['AUC_Sex']:.4f}  "
            f"Act={row['Active_Nodes']}/{row['Total_Nodes']}  QE={row['QE']:.3f}  TE={row['TE']:.3f}  |  "
            f"extNMI={row.get('ext_NMI', float('nan')):.4f}  extMAE={row.get('ext_MAE_Age', float('nan')):.2f}  ({row['elapsed_s']}s)"
        )
        _save_outputs(rows, args.output)

    if args.plot:
        plot_sensitivity(rows, args.plot)


if __name__ == "__main__":
    main()
