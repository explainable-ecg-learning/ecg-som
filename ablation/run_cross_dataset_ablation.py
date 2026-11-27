"""Cross-Dataset Transfer Ablation for DPSOM-ECG (PTB-XL <-> Georgia <-> Chapman-Shaoxing).

Evaluates bidirectional domain transfer when training pruned 8x8 DPSOM-ECG on each dataset
and testing on all other datasets under two regimes:
  1. zero_shot : frozen source checkpoint evaluated directly on target test split (max_beats=None)
  2. uda       : 8 epochs of unlabeled target SOM fine-tuning (demographic heads frozen,
                 delta_age = delta_sex = 0, zero target labels or demographic supervision used)

Also includes a size-matched PTB-XL control (`ptbxl_2k`, train=1800, val=225) matching the
~1.8k training size of Georgia/Chapman transfer cohorts to separate domain shift from sample size.

Usage:
    # Parallel 2-GPU run:
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u ablation/run_cross_dataset_ablation.py \
        --sources ptbxl_full georgia --seeds 42 123 456 --epochs 30 --output ablation/xds_gpu0.json
    CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u ablation/run_cross_dataset_ablation.py \
        --sources ptbxl_2k chapman --seeds 42 123 456 --epochs 30 --output ablation/xds_gpu1.json

    # Merge & plot:
    .venv/bin/python -u ablation/run_cross_dataset_ablation.py \
        --merge ablation/xds_gpu0.json ablation/xds_gpu1.json \
        --output ablation/cross_dataset_results.json \
        --plot ablation/cross_dataset_transfer.png
"""
from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import argparse
import copy
import csv
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import DPSOM_Config
from src.data.dataset import ECG_Dataset
from src.data.generator import ECG_DataGenerator
from src.model.dpsom import DPSOM_ECG
from src.training.trainer import (
    _compute_target_distribution,
    _get_device,
    _load_checkpoint_state,
    _set_global_determinism,
    evaluate_model,
    train_model,
)
from src.utils.metrics import cluster_purity, compute_som_clustering_metrics


SOURCES = ("ptbxl_full", "ptbxl_2k", "georgia", "chapman")
TARGETS = ("ptbxl", "georgia", "chapman")


def _source_base(src: str) -> str:
    return "ptbxl" if src.startswith("ptbxl") else src


def _subsample_ptbxl(ptb_full: ECG_Dataset, n_train: int = 1800, n_val: int = 225, seed: int = 42) -> ECG_Dataset:
    """Create a size-matched stratified subset of PTB-XL train/val while keeping the full test split."""
    sub = ECG_Dataset(fs=ptb_full.fs)
    sub.class_names = list(ptb_full.class_names)
    sub.label_encoder = ptb_full.label_encoder

    tr_recs = ptb_full.get_data("train")
    va_recs = ptb_full.get_data("val")
    te_recs = ptb_full.get_data("test")

    if len(tr_recs) > n_train:
        y_tr = np.array([r.label for r in tr_recs])
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_train, random_state=seed)
        idx_tr, _ = next(sss.split(np.zeros(len(tr_recs)), y_tr))
        tr_sub = [tr_recs[int(i)] for i in idx_tr]
    else:
        tr_sub = list(tr_recs)

    if len(va_recs) > n_val:
        y_va = np.array([r.label for r in va_recs])
        sss = StratifiedShuffleSplit(n_splits=1, train_size=n_val, random_state=seed)
        idx_va, _ = next(sss.split(np.zeros(len(va_recs)), y_va))
        va_sub = [va_recs[int(i)] for i in idx_va]
    else:
        va_sub = list(va_recs)

    sub.records = tr_sub + va_sub + list(te_recs)
    return sub


def _eval_on_target(model: DPSOM_ECG, tgt_ds: ECG_Dataset, use_rr: bool, batch_size: int = 64) -> dict:
    """Evaluate a trained/adapted DPSOM on a target cohort test split (all beats per record)."""
    gen = ECG_DataGenerator(tgt_ds, batch_size=batch_size, use_rr_features=use_rr, preload_to_gpu=True)
    res = evaluate_model(
        model, gen, compute_disentanglement=False, compute_linear_probes=False, split="test", max_beats=None
    )

    device = next(model.parameters()).device
    rec_gen = gen.get_record_batch_generator(mode="test", max_beats=None, shuffle=False)
    nb = gen.get_num_batches("test")
    z_res_all, lbl_all = [], []
    model.eval()
    with torch.inference_mode():
        for _ in range(nb):
            beats, _, _, beat_valid, _, _, _, labels_t = next(rec_gen)
            B, N, C, T = beats.shape
            valid = beat_valid.reshape(B * N) > 0.5
            mu, _, _, _ = model._encode(beats.reshape(B * N, C, T)[valid])
            z_res = model.residualize_latent(mu.float()).cpu().numpy()
            z_res_all.append(z_res)
            lbl_all.append(labels_t.unsqueeze(1).expand(B, N).reshape(B * N)[valid].cpu().numpy())

    z_res_np = np.concatenate(z_res_all, axis=0)
    lbl_np = np.concatenate(lbl_all, axis=0).astype(str)
    H, W = tuple(int(v) for v in model.som_dim)
    codebook = model._embeddings.detach().float().cpu().numpy().reshape(H * W, -1)
    som_m = compute_som_clustering_metrics(z_res_np, codebook, som_shape=(H, W), labels=lbl_np, knn_k=5)

    return {
        "NMI":         float(res["NMI"]),
        "AMI":         float(res["AMI"]),
        "Purity":      float(res["Purity"]),
        "Rec_NMI":     float(res["Rec_NMI"]),
        "Rec_AMI":     float(res["Rec_AMI"]),
        "Rec_Purity":  float(res["Rec_Purity"]),
        "MAE_Age":     float(res["MAE_Age"]),
        "AUC_Sex":     float(res["AUC_Sex"]),
        "Active_Nodes": int(som_m["utilization"]["active_units"]),
        "Occ_Entropy":  float(som_m["utilization"]["occupancy_entropy"]),
        "QE":           float(som_m["topology"]["quantization_error"]),
        "TE":           float(som_m["topology"]["topographic_error"]),
    }


def _adapt_unsupervised(
    src_model: DPSOM_ECG,
    tgt_ds: ECG_Dataset,
    batch_size: int = 64,
    adapt_epochs: int = 6,
    lr: float = 1e-4,
    use_amp: bool = True,
) -> DPSOM_ECG:
    """Unsupervised target SOM/morphology adaptation with demographic heads frozen and zero labels."""
    device = next(src_model.parameters()).device
    adapted = copy.deepcopy(src_model).to(device)
    adapted.delta_age = 0.0
    adapted.delta_sex = 0.0

    trainable = []
    for name, param in adapted.named_parameters():
        if any(k in name for k in ("age", "sex")):
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable.append(param)

    tgt_gen = ECG_DataGenerator(tgt_ds, batch_size=batch_size, use_rr_features=False, preload_to_gpu=True)
    num_batches = tgt_gen.get_num_batches("train")
    num_used_beats = 4
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

    amp_enabled = bool(use_amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16

    for _ep in range(adapt_epochs):
        p_target = _compute_target_distribution(adapted, tgt_gen, "train", device, use_amp=use_amp)
        train_gen = tgt_gen.get_record_batch_generator(mode="train", max_beats=num_used_beats, shuffle=True)
        adapted.train()
        for _ in range(num_batches):
            batch = next(train_gen)
            beats, beat_mask, beat_meta, beat_valid, global_beat_idx = batch[:5]
            B, N = beat_valid.shape
            valid_mask = global_beat_idx >= 0
            p_batch = torch.zeros(B, N, p_target.shape[1], device=device, dtype=p_target.dtype)
            p_batch[valid_mask] = p_target[global_beat_idx[valid_mask].long()]

            dummy_age = torch.zeros(B, device=device, dtype=torch.float32)
            dummy_sex = torch.zeros(B, device=device, dtype=torch.long)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                losses = adapted.loss_batch(
                    beats, beat_mask, beat_meta, beat_valid, dummy_age, dummy_sex, p_batch
                )
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
        adapted.inc_epoch()

    return adapted


def _train_source_model(
    src_name: str,
    src_ds: ECG_Dataset,
    seed: int,
    epochs: int,
) -> DPSOM_ECG:
    """Train a pruned default 8x8 DPSOM-ECG model on `src_name` and return its val-best checkpoint."""
    cfg = DPSOM_Config()
    cfg.random_seed = seed
    cfg.num_epochs = epochs
    cfg.som_dim = (8, 8)
    cfg.eta = 0.0
    cfg.age_corr_lambda_max = 0.0
    cfg.probe_refit_interval = 0
    cfg.use_rr_features = False
    cfg.num_epochs_probe = 3
    cfg.batch_size = 300 if src_name == "ptbxl_full" else 64
    cfg.ex_name = f"xds_{src_name}_s{seed}"
    cfg.logdir = f"logs/{cfg.ex_name}"
    cfg.modelpath = f"models/{cfg.ex_name}/{cfg.ex_name}.ckpt"
    _set_global_determinism(cfg.random_seed, fast_gpu=cfg.fast_gpu)

    gen = ECG_DataGenerator(
        src_ds, batch_size=cfg.batch_size, use_rr_features=False, preload_to_gpu=cfg.preload_to_gpu
    )
    sample = gen.get_flat_beats_tensor("train")
    input_channels, input_length = sample.shape[1], sample.shape[2]
    amp_dim = 2 * input_channels

    model = DPSOM_ECG(
        latent_dim=cfg.latent_dim, som_dim=cfg.som_dim,
        input_length=input_length, input_channels=input_channels,
        alpha=cfg.alpha, beta=cfg.beta, theta=cfg.theta, gamma=cfg.gamma,
        tau=cfg.tau, eta=cfg.eta, delta_age=cfg.delta_age, delta_sex=cfg.delta_sex,
        dropout=cfg.dropout, prior_var=cfg.prior_var, prior=cfg.prior,
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
    if best_ckpt and os.path.exists(best_ckpt):
        state, _ = _load_checkpoint_state(best_ckpt, _get_device())
        model.load_state_dict(state, strict=True)

    shutil.rmtree(os.path.dirname(cfg.modelpath), ignore_errors=True)
    shutil.rmtree(cfg.logdir, ignore_errors=True)
    return model


def _compute_chance(datasets: dict[str, ECG_Dataset], k: int = 64, n_draws: int = 10) -> dict:
    rng = np.random.default_rng(0)
    chance: dict = {}
    for name in TARGETS:
        ds = datasets[name]
        recs = ds.get_data("test")
        enc = ds.label_encoder
        y_rec, y_beat, rids = [], [], []
        for idx, r in enumerate(recs):
            if r.beat_representations is None or len(r.beat_representations) == 0:
                continue
            lbl = int(enc.transform([r.label])[0])
            nb = len(r.beat_representations)
            y_rec.append(lbl)
            y_beat.extend([lbl] * nb)
            rids.extend([idx] * nb)
        y_rec_a = np.asarray(y_rec, dtype=np.int64)
        y_beat_a = np.asarray(y_beat, dtype=np.int64)
        rid_a = np.asarray(rids, dtype=np.int64)
        rr_ami, rr_nmi, rb_nmi = [], [], []
        for _ in range(n_draws):
            cr = rng.integers(0, k, size=len(y_rec_a))
            rr_ami.append(metrics.adjusted_mutual_info_score(y_rec_a, cr, average_method="geometric"))
            rr_nmi.append(metrics.normalized_mutual_info_score(y_rec_a, cr, average_method="geometric"))
            rc = rng.integers(0, k, size=len(recs))
            rb_nmi.append(metrics.normalized_mutual_info_score(y_beat_a, rc[rid_a], average_method="geometric"))
        chance[name] = {
            "n_records": int(len(y_rec_a)),
            "n_beats": int(len(y_beat_a)),
            "n_classes": int(len(np.unique(y_rec_a))),
            "random_record_AMI": float(np.mean(rr_ami)),
            "random_record_NMI": float(np.mean(rr_nmi)),
            "rec_hash_beat_NMI": float(np.mean(rb_nmi)),
        }
    return chance


def summarize(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["source"], r["target"], r["regime"])].append(r)

    skip = {"source", "target", "regime", "seed"}
    out: list[dict] = []
    for (src, tgt, reg), grp in by_key.items():
        s: dict = {"source": src, "target": tgt, "regime": reg, "n_seeds": len(grp)}
        metric_keys = [k for k in grp[0] if k not in skip and isinstance(grp[0][k], (int, float))]
        for k in metric_keys:
            vals = [float(r[k]) for r in grp if np.isfinite(float(r[k]))]
            if vals:
                s[f"{k}_mean"] = float(np.mean(vals))
                s[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out.append(s)
    return out


def _write_outputs(rows: list[dict], out_json: str, chance: dict) -> None:
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    payload = {"runs": rows, "summary": summarize(rows), "chance": chance}
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_json[:-5] + ".csv" if out_json.endswith(".json") else out_json + ".csv"
    if rows:
        keys = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


def plot_cross_dataset_transfer(json_path: str, out_png: str = "ablation/cross_dataset_transfer.png") -> None:
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    d = json.loads(Path(json_path).read_text())
    smap = {(r["source"], r["target"], r["regime"]): r for r in d["summary"]}

    src_keys = ["ptbxl_full", "ptbxl_2k", "georgia", "chapman"]
    src_labels = ["PTB-XL Full\n(n=13.5k)", "PTB-XL 2k\n(n=1.8k)", "Georgia\n(n~1.9k)", "Chapman\n(n~1.9k)"]
    tgt_keys = ["ptbxl", "georgia", "chapman"]
    tgt_labels = ["PTB-XL\n(test)", "Georgia\n(test)", "Chapman\n(test)"]

    fig = plt.figure(figsize=(17.2, 10.2), dpi=180)
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(2, 6, hspace=0.42, wspace=0.72)

    def _draw_matrix(ax, metric: str, title: str, cmap_name: str, fmt: str, cbar_lbl: str,
                     vmin: float | None = None, vmax: float | None = None, reverse_txt: bool = False,
                     show_yticks: bool = True):
        mat = np.zeros((len(src_keys), len(tgt_keys)), dtype=np.float32)
        std = np.zeros((len(src_keys), len(tgt_keys)), dtype=np.float32)
        for i, sk in enumerate(src_keys):
            for j, tk in enumerate(tgt_keys):
                row = smap.get((sk, tk, "zero_shot"))
                if row:
                    mat[i, j] = row[f"{metric}_mean"]
                    std[i, j] = row[f"{metric}_std"]

        lo = float(np.min(mat)) if vmin is None else vmin
        hi = float(np.max(mat)) if vmax is None else vmax
        im = ax.imshow(mat, cmap=cmap_name, vmin=lo, vmax=hi, aspect="auto")
        ax.set_xticks(np.arange(len(tgt_keys)))
        ax.set_xticklabels(tgt_labels, fontsize=8.8)
        ax.set_yticks(np.arange(len(src_keys)))
        ax.set_yticklabels(src_labels if show_yticks else [], fontsize=8.5)
        ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)

        span = hi - lo if hi > lo else 1.0
        for i, sk in enumerate(src_keys):
            for j, tk in enumerate(tgt_keys):
                norm = (float(mat[i, j]) - lo) / span
                tcol = ("#0f172a" if norm >= 0.52 else "white") if reverse_txt else ("white" if norm >= 0.55 else "#0f172a")
                in_dom = _source_base(sk) == tk
                lbl_str = f"{format(mat[i, j], fmt)}\n±{format(std[i, j], fmt)}"
                ax.text(j, i, lbl_str, ha="center", va="center", fontsize=8.0,
                        fontweight="bold" if in_dom else "normal", color=tcol)
                if in_dom:
                    ax.add_patch(Rectangle((j - 0.48, i - 0.48), 0.96, 0.96, fill=False, ec="#0f172a", lw=2.2))

        cbar = plt.colorbar(im, ax=ax, fraction=0.048, pad=0.03)
        cbar.ax.tick_params(labelsize=7.8)
        cbar.set_label(cbar_lbl, fontsize=8.2)

    _draw_matrix(fig.add_subplot(gs[0, 0:2]), "Rec_AMI", "(a) Transfer Matrix: Record AMI ↑ (Zero-Shot)",
                 "YlGnBu", ".3f", "Rec AMI ↑", vmin=0.05, show_yticks=True)
    _draw_matrix(fig.add_subplot(gs[0, 2:4]), "MAE_Age", "(b) Transfer Matrix: Age MAE (yr) ↓",
                 "YlOrRd", ".1f", "Age MAE (yr) ↓", vmin=7.5, show_yticks=False)
    _draw_matrix(fig.add_subplot(gs[0, 4:6]), "AUC_Sex", "(c) Transfer Matrix: Sex AUROC ↑",
                 "PuBuGn", ".3f", "Sex AUROC ↑", vmin=0.70, vmax=0.92, show_yticks=False)

    # Panel (d): Zero-shot vs +UDA vs In-Domain Target->Target across 6 cross pairs
    ax_d = fig.add_subplot(gs[1, :])
    pairs = [
        ("ptbxl_full", "georgia",  "PTB-XL\n→ Georgia"),
        ("ptbxl_full", "chapman",  "PTB-XL\n→ Chapman"),
        ("georgia",    "ptbxl",    "Georgia\n→ PTB-XL"),
        ("chapman",    "ptbxl",    "Chapman\n→ PTB-XL"),
        ("georgia",    "chapman",  "Georgia\n→ Chapman"),
        ("chapman",    "georgia",  "Chapman\n→ Georgia"),
    ]
    xp = np.arange(len(pairs))
    wb = 0.25
    zs_m = [smap[(s, t, "zero_shot")]["Rec_AMI_mean"] for s, t, _ in pairs]
    zs_s = [smap[(s, t, "zero_shot")]["Rec_AMI_std"]  for s, t, _ in pairs]
    ud_m = [smap[(s, t, "uda")]["Rec_AMI_mean"]       for s, t, _ in pairs]
    ud_s = [smap[(s, t, "uda")]["Rec_AMI_std"]        for s, t, _ in pairs]
    in_src = {"ptbxl": "ptbxl_full", "georgia": "georgia", "chapman": "chapman"}
    id_m = [smap[(in_src[t], t, "zero_shot")]["Rec_AMI_mean"] for _s, t, _ in pairs]
    id_s = [smap[(in_src[t], t, "zero_shot")]["Rec_AMI_std"]  for _s, t, _ in pairs]

    b1 = ax_d.bar(xp - wb, zs_m, wb, yerr=zs_s, capsize=3, color="#64748b", alpha=0.90, label="Zero-Shot Frozen Source")
    b2 = ax_d.bar(xp,      ud_m, wb, yerr=ud_s, capsize=3, color="#2563eb", alpha=0.92, label="+Unlabeled SOM Adapt (6 ep, 0 labels)")
    b3 = ax_d.bar(xp + wb, id_m, wb, yerr=id_s, capsize=3, color="#0d9488", alpha=0.88, label="Target In-Domain Baseline (Target → Target)")

    ymax = max(v + err for v, err in zip(zs_m + ud_m + id_m, zs_s + ud_s + id_s))
    for bars, vals, errs in ((b1, zs_m, zs_s), (b2, ud_m, ud_s), (b3, id_m, id_s)):
        for bar, v, e in zip(bars, vals, errs):
            ax_d.text(bar.get_x() + bar.get_width() / 2, v + e + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7.6)

    ax_d.set_title("(d) Zero-Shot Transfer vs. Unlabeled SOM Adaptation (+UDA) vs. In-Domain Target (Record AMI ↑)",
                   fontsize=11.0, fontweight="bold")
    ax_d.set_xticks(xp)
    ax_d.set_xticklabels([lbl for _, _, lbl in pairs], fontsize=9.2)
    ax_d.set_ylabel("Record-level AMI ↑", fontsize=9.5)
    ax_d.set_ylim(0, max(0.30, ymax + 0.055))
    ax_d.grid(True, axis="y", alpha=0.25, ls=":")
    ax_d.spines[["top", "right"]].set_visible(False)
    ax_d.legend(fontsize=8.6, loc="upper left", ncol=3, frameon=True)

    fig.suptitle(
        "Cross-Dataset Transfer Ablation: 8×8 Pruned DPSOM-ECG Across PTB-XL, Georgia & Chapman-Shaoxing (3 Seeds)",
        fontsize=13.2, fontweight="bold", y=0.985,
    )
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[xds] Saved cross-dataset figure -> {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description="Cross-dataset transfer ablation for DPSOM-ECG")
    p.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--adapt-epochs", type=int, default=6)
    p.add_argument("--adapt-lr", type=float, default=1e-4)
    p.add_argument("--output", type=str, default="ablation/cross_dataset_results.json")
    p.add_argument("--merge", nargs="+", default=None)
    p.add_argument("--plot", type=str, default=None)
    args = p.parse_args()

    cfg = DPSOM_Config()
    print("[xds] Loading benchmark datasets (PTB-XL, Georgia transfer, Chapman transfer)...")
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
        merged.sort(key=lambda r: (s_ord.get(r["source"], 99), r["regime"], t_ord.get(r["target"], 99), int(r["seed"])))
        _write_outputs(merged, args.output, chance)
        print(f"[xds] Merged {len(merged)} rows -> {args.output}")
        if args.plot:
            plot_cross_dataset_transfer(args.output, args.plot)
        return

    rows: list[dict] = []
    for src in args.sources:
        for seed in args.seeds:
            t0 = time.time()
            print(f"\n{'=' * 66}\n  SOURCE: {src}  |  SEED: {seed}\n{'=' * 66}")
            model = _train_source_model(src, source_ds[src], seed, args.epochs)

            for tgt in TARGETS:
                res = _eval_on_target(model, target_ds[tgt], use_rr=False, batch_size=64)
                row = {"source": src, "target": tgt, "regime": "zero_shot", "seed": seed, **res}
                rows.append(row)
                print(
                    f"  [zero_shot] {src} -> {tgt}: RecAMI={res['Rec_AMI']:.4f}  BeatNMI={res['NMI']:.4f}  "
                    f"MAE={res['MAE_Age']:.2f}  AUC={res['AUC_Sex']:.4f}  Act={res['Active_Nodes']}/64"
                )
                if _source_base(src) != tgt:
                    adapted = _adapt_unsupervised(
                        model, target_ds[tgt], batch_size=64, adapt_epochs=args.adapt_epochs, lr=args.adapt_lr,
                        use_amp=cfg.use_amp,
                    )
                    res_uda = _eval_on_target(adapted, target_ds[tgt], use_rr=False, batch_size=64)
                    rows.append({"source": src, "target": tgt, "regime": "uda", "seed": seed, **res_uda})
                    print(
                        f"  [+UDA {args.adapt_epochs}ep] {src} -> {tgt}: RecAMI={res_uda['Rec_AMI']:.4f}  "
                        f"BeatNMI={res_uda['NMI']:.4f}  Act={res_uda['Active_Nodes']}/64"
                    )
            print(f"[xds] Completed {src} s={seed} in {time.time() - t0:.1f}s")
            _write_outputs(rows, args.output, chance)

    if args.plot:
        plot_cross_dataset_transfer(args.output, args.plot)


if __name__ == "__main__":
    main()
