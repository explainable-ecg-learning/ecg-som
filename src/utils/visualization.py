"""Visualization utilities for DPSOM-ECG (SOM heatmaps + signal reconstruction figures)."""

from __future__ import annotations

import math
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.model.dpsom import DPSOM_ECG

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# ---------------------------------------------------------------------------
# Signal layout helper
# ---------------------------------------------------------------------------

def _resolve_signal_layout(num_samples: int, sampling_rate: float, num_plots: int):
    duration = num_samples / sampling_rate
    is_short = duration < 2.5
    if is_short:
        cols = 4 if num_plots >= 4 else num_plots
        rows = math.ceil(num_plots / cols)
        fw, fh = 4 * cols, 3 * rows
    else:
        cols, rows = 1, num_plots
        fw, fh = max(12, duration * 2), 2.0 * rows
    return duration, is_short, rows, cols, fw, fh


# ---------------------------------------------------------------------------
# Signal reconstruction figure
# ---------------------------------------------------------------------------

def draw_signal_reconstruction_figure(original, recon, sampling_rate, subtitle, leads=None):
    """Return a Matplotlib figure overlaying original vs reconstructed signal.

    Args:
        original: [T, C] array.
        recon:    [T, C] array.
    """
    if leads is None:
        leads = list(range(original.shape[1]))

    num_samples = original.shape[0]
    num_plots   = len(leads)
    duration, is_short, rows, cols, fw, fh = _resolve_signal_layout(num_samples, sampling_rate, num_plots)
    t = np.arange(num_samples) / sampling_rate

    if is_short:
        fig, axes = plt.subplots(rows, cols, figsize=(fw, fh), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(rows, cols, figsize=(fw, fh), sharex=True)
    axes_flat = axes.flatten() if num_plots > 1 else [axes]

    for i, ax in enumerate(axes_flat):
        if i >= num_plots:
            ax.axis('off')
            continue
        li = leads[i]
        ax.plot(t, original[:, li], linewidth=1.2, color='#1f77b4', label='Original')
        ax.plot(t, recon[:, li],    linewidth=1.2, color='#ff7f0e', alpha=0.9, label='Reconstruction')
        ax.set_title(LEAD_NAMES[li], fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6)
        if is_short:
            if i % cols == 0:
                ax.set_ylabel('mV', fontsize=9)
            if i >= (rows - 1) * cols:
                ax.set_xlabel('Time (s)', fontsize=9)
        else:
            ax.set_ylabel('mV', fontsize=9)
            if i == num_plots - 1:
                ax.set_xlabel('Time (s)', fontsize=10)
        if i == 0:
            ax.legend(loc='upper right', fontsize=8, frameon=True)

    fig.suptitle(f"{subtitle} ({duration:.2f}s)", fontsize=14, fontweight='bold',
                 y=0.98 if is_short else 1.005)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SOM location figure
# ---------------------------------------------------------------------------

def draw_som_location_figure(som_dim, rc, subtitle="SOM location"):
    """Return a figure marking cell `rc` in the SOM grid.

    Args:
        som_dim: (H, W).
        rc: (row, col) of the cell to mark.
    """
    H, W = som_dim
    r, c = rc

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.zeros((H, W), dtype=np.float32), cmap='Greys', vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=1.0)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.scatter([c], [r], s=250, marker='x', color='red', linewidths=3)
    ax.set_xticks(np.arange(W))
    ax.set_yticks(np.arange(H))
    ax.set_title(subtitle, fontsize=12, fontweight='bold')
    return fig


# ---------------------------------------------------------------------------
# SOM visualization logging
# ---------------------------------------------------------------------------

def log_som_visualizations(
    dataset,
    generator,
    ex_name: str,
    ckpt_path: str,
    device,
    single_label_only: bool = False,
    crop: int = 10,
):
    """Load checkpoint, run test-set inference, save SOM heatmaps and centroid figures."""
    print("\nVisualizations...")

    X_test, X_mask, y_test, _, _, _, _, _ = generator.get_all_beats_representation(split="test")

    if single_label_only:
        test_records = generator.ecg_dataset.get_data("test")
        keep = []
        for rec in test_records:
            ok = len(getattr(rec, "extra_labels", [])) == 0
            keep.extend([ok] * len(rec.beat_representations))
        keep = np.asarray(keep, dtype=bool)
        n = min(keep.shape[0], X_test.shape[0], np.asarray(y_test).shape[0])
        X_test = X_test[:n][keep[:n]]
        y_test = np.asarray(y_test)[:n][keep[:n]]
        if X_test.shape[0] == 0:
            print("No single-label test beats available. Skipping visualization.")
            return

    class_names = getattr(dataset, "class_names", ['CD', 'HYP', 'MI', 'NORM', 'STTC'])
    num_classes = len(class_names)

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Skipping.")
        return

    meta  = checkpoint.get("meta", {}) if isinstance(checkpoint, dict) else {}
    state = checkpoint
    if isinstance(state, dict):
        for k in ["state_dict", "model", "net", "weights"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    def _strip(sd, prefix):
        return {k[len(prefix):]: v for k, v in sd.items()} if any(k.startswith(prefix) for k in sd) else sd

    state = _strip(_strip(state, "module."), "model.")

    # Remap legacy enc_* / dec_* layer names to current convention
    _LEGACY = {
        "enc_conv1": "conv1", "enc_bn1": "bn1", "enc_se1": "se1",
        "enc_conv2": "conv2", "enc_bn2": "bn2", "enc_se2": "se2",
        "dec_conv1": "conv1", "dec_bn1": "bn1", "dec_se1": "se1",
        "dec_film1": "film1", "dec_conv2": "conv2", "dec_bn2": "bn2",
        "dec_se2": "se2", "dec_film2": "film2", "dec_conv_out": "conv_out",
    }
    state = {
        ".".join(_LEGACY.get(p, p) for p in k.split(".")): v
        for k, v in state.items()
    }

    input_length   = int(meta.get("input_length",   X_test.shape[1]))
    input_channels = int(meta.get("input_channels", X_test.shape[2]))
    som_dim        = tuple(meta.get("som_dim", ()))
    latent_dim     = int(meta.get("latent_dim", 0))

    if len(som_dim) != 2 or latent_dim <= 0:
        embed_key = next(
            (k for k, v in state.items() if isinstance(v, torch.Tensor) and v.ndim == 3 and "embeddings" in k),
            None,
        )
        if embed_key is None:
            print("Error: Cannot infer SOM dimensions from checkpoint. Skipping.")
            return
        H_size, W_size, latent_dim = state[embed_key].shape
        som_dim = (int(H_size), int(W_size))

    H_size, W_size = som_dim

    model = DPSOM_ECG(
        latent_dim=latent_dim, som_dim=som_dim,
        input_length=input_length, input_channels=input_channels,
        encoder_base_channels_1=int(meta.get("encoder_base_channels_1", 32)),
        encoder_base_channels_2=int(meta.get("encoder_base_channels_2", 64)),
        encoder_kernel_size=int(meta.get("encoder_kernel_size", 7)),
        encoder_fc_hidden_dim=int(meta.get("encoder_fc_hidden_dim", 512)),
        z_age_dim_factor=float(meta.get("z_age_dim_factor", 0.25)),
        z_sex_dim_factor=float(meta.get("z_sex_dim_factor", 0.25)),
    ).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    x_t = torch.from_numpy(X_test.transpose(0, 2, 1)).contiguous().float().to(device)

    with torch.no_grad():
        enc_out = model._encode(x_t)
        mu      = enc_out[0]
        k_all   = model.k(mu).cpu().numpy()
        E_flat  = model._embeddings.view(-1, latent_dim)
        logits  = model.decoder(E_flat)
        x_hat_all = logits.view(H_size, W_size, input_channels, input_length).cpu().numpy()

    k_flat  = k_all.flatten()
    labels  = y_test.flatten()
    K       = H_size * W_size
    counts  = np.zeros((K, num_classes), dtype=np.int64)
    valid   = labels < num_classes
    np.add.at(counts, (k_flat[valid], labels[valid].astype(np.int64)), 1)
    totals  = counts.sum(axis=1)

    majority = np.full(K, np.nan, dtype=np.float32)
    nonempty = totals > 0
    majority[nonempty] = counts[nonempty].argmax(axis=1).astype(np.float32)
    clust_matr = majority.reshape(H_size, W_size)

    annot_labels = np.full((H_size, W_size), "", dtype=object)
    for i in range(H_size):
        for j in range(W_size):
            val = clust_matr[i, j]
            if np.isnan(val):
                continue
            cell   = i * W_size + j
            total  = int(totals[cell])
            if total == 0:
                continue
            mi     = int(val)
            correct = int(counts[cell, mi])
            short  = str(class_names[mi])[:3] if 0 <= mi < num_classes else "NA"
            annot_labels[i, j] = f"{short}\n{correct}/{total}"

    log_dir = f"logs/{ex_name}/test"
    os.makedirs(log_dir, exist_ok=True)

    # Crop centroids
    crop = max(0, min(int(crop), max(0, (input_length - 1) // 2)))
    x_hat_plot = x_hat_all[:, :, :, crop:input_length - crop] if crop > 0 else x_hat_all

    # Per-class heatmaps
    for c in range(num_classes):
        cname  = class_names[c] if c < len(class_names) else f"Class {c}"
        cc     = counts[:, c].reshape(H_size, W_size)
        annot  = cc.astype(str)
        annot[cc == 0] = ""
        fig, _ = plt.subplots(figsize=(10, 8))
        sns.heatmap(cc, cmap="YlGnBu", annot=annot, fmt="")
        plt.title(f"Heatmap: {cname}")
        safe = cname.replace("/", "_").replace(" ", "_")
        fig.savefig(os.path.join(log_dir, f"heatmap_{safe}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Majority class heatmap
    fig_hm = plt.figure(figsize=(20, 15))
    sns.heatmap(clust_matr, cmap="YlGnBu", annot=annot_labels, fmt="")
    plt.title("Majority Class per SOM Cell")
    fig_hm.savefig(os.path.join(log_dir, "heatmap_majority.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_hm)

    # Lead-0 centroid grid
    ymin = float(np.min(x_hat_plot[:, :, 0, :]))
    ymax = float(np.max(x_hat_plot[:, :, 0, :]))
    plen = x_hat_plot.shape[-1]

    fig_c, axes = plt.subplots(H_size, W_size, figsize=(W_size * 1.5, H_size * 1.5), sharex=True, sharey=True)
    if H_size == 1 and W_size == 1:
        axes = np.array([[axes]])
    elif H_size == 1 or W_size == 1:
        axes = axes.reshape(H_size, W_size)
    fig_c.subplots_adjust(left=0.02, right=0.995, bottom=0.02, top=0.995, wspace=0.05, hspace=0.05)

    for i in range(H_size):
        for j in range(W_size):
            ax = axes[i, j]
            ax.plot(x_hat_plot[i, j, 0, :], linewidth=0.8)
            ax.set_xlim(0, plen - 1)
            ax.set_ylim(ymin, ymax)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0: ax.set_title(str(j), fontsize=10, pad=6)
            if j == 0: ax.set_ylabel(str(i), rotation=0, fontsize=10, labelpad=10, va="center")

    fig_c.supxlabel("Time", y=0.01)
    fig_c.savefig(os.path.join(log_dir, "som_centroids_lead0.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_c)

    # All-leads prototype grid
    n_leads  = input_channels
    cell_h   = max(1.8, n_leads * 0.28)
    fig_p, axes_p = plt.subplots(H_size, W_size, figsize=(W_size * 2.0, H_size * cell_h), sharex=True)
    if H_size == 1 and W_size == 1:
        axes_p = np.array([[axes_p]])
    elif H_size == 1 or W_size == 1:
        axes_p = axes_p.reshape(H_size, W_size)
    fig_p.subplots_adjust(left=0.06, right=0.995, bottom=0.02, top=0.98, wspace=0.08, hspace=0.12)

    g_amp   = float(np.percentile(np.abs(x_hat_plot), 95))
    g_amp   = g_amp if g_amp > 1e-6 else 1.0
    step    = g_amp * 2.2

    for i in range(H_size):
        for j in range(W_size):
            ax = axes_p[i, j]
            for li in range(n_leads - 1, -1, -1):
                ax.plot(x_hat_plot[i, j, li, :] + (n_leads - 1 - li) * step, linewidth=0.6, color="#2166ac")
            ax.set_yticks([(n_leads - 1 - k) * step for k in range(n_leads)])
            if j == 0:
                ax.set_yticklabels([LEAD_NAMES[k] if k < len(LEAD_NAMES) else str(k) for k in range(n_leads)], fontsize=5)
            else:
                ax.set_yticklabels([])
            ax.set_xticks([])
            cell = i * W_size + j
            tot  = int(totals[cell])
            if tot > 0:
                mi  = int(counts[cell].argmax())
                lbl = class_names[mi][:4] if mi < len(class_names) else ""
                ax.set_title(f"{i},{j} {lbl} ({tot})", fontsize=5, pad=2)
            else:
                ax.set_title(f"{i},{j}", fontsize=5, pad=2)

    fig_p.suptitle("SOM Cell Prototypes — all leads", fontsize=9, y=0.995)
    fig_p.savefig(os.path.join(log_dir, "som_prototypes_all_leads.png"), dpi=150, bbox_inches="tight")
    plt.close(fig_p)
    print(f"Visualizations saved to {log_dir}/")
