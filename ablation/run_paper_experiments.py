"""Multi-seed fair benchmark runner: Matched-k Baselines, Residualization & External Transfer.

All methods are evaluated on the exact same beat set (``max_beats=None`` — all beats per
record), reporting both record-level and beat-level NMI / AMI / Purity at matched cluster
counts ``k in {16, 64, 256}``, empirical chance baselines, deterministic raw-mu vs
residualized-mu linear demographic leakage across all seeds, and zero-shot Georgia transfer.

Experiment configs are pinned explicitly (independent of future config.py edits).

Usage:
    # Parallel 2-GPU run:
    CUDA_VISIBLE_DEVICES=0 python ablation/run_paper_experiments.py \
        --models vae_kmeans resnet1d_supervised dpsom_pruned_4x4 dpsom_pruned_8x8 \
        --seeds 42 123 456 --epochs 30 --output ablation/paper_v2_gpu0.json
    CUDA_VISIBLE_DEVICES=1 python ablation/run_paper_experiments.py \
        --models dpsom_pruned_16x16 dpsom_pruned_8x8_no_res dpsom_legacy_full_8x8 \
        --seeds 42 123 456 --epochs 30 --output ablation/paper_v2_gpu1.json

    # Merge shards & compute chance baselines:
    python ablation/run_paper_experiments.py --merge ablation/paper_v2_gpu0.json ablation/paper_v2_gpu1.json \
        --output ablation/paper_results_all.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from sklearn import metrics

from src.config import DPSOM_Config
from src.data.dataset import ECG_Dataset
from src.data.generator import ECG_DataGenerator
from src.model.dpsom import DPSOM_ECG
from src.training.baselines import _cluster_metrics_for_k, train_eval_resnet1d, train_eval_vae_kmeans
from src.training.trainer import (
    _get_device,
    _load_checkpoint_state,
    _set_global_determinism,
    evaluate_model,
    train_model,
)
from src.utils.metrics import cluster_purity

KS: tuple[int, ...] = (16, 64, 256)

_PRUNED = {
    "eta": 0.0,
    "age_corr_lambda_max": 0.0,
    "probe_refit_interval": 0,
    "use_rr_features": False,
    "num_epochs_probe": 3,
}
_LEGACY_FULL = {
    "eta": 1.0,
    "age_corr_lambda_max": 0.30,
    "probe_refit_interval": 6,
    "use_rr_features": True,
    "num_epochs_probe": 3,
}

# Pinned self-contained experiment registry
EXPERIMENTS: dict[str, dict] = {
    "vae_kmeans":              {"kind": "vae_kmeans", "overrides": {"use_rr_features": False}},
    "resnet1d_supervised":     {"kind": "resnet1d",   "overrides": {"use_rr_features": False}},
    "dpsom_pruned_4x4":        {"kind": "dpsom",      "overrides": {**_PRUNED, "som_dim": (4, 4)}},
    "dpsom_pruned_8x8":        {"kind": "dpsom",      "overrides": {**_PRUNED, "som_dim": (8, 8)}},
    "dpsom_pruned_16x16":      {"kind": "dpsom",      "overrides": {**_PRUNED, "som_dim": (16, 16)}},
    "dpsom_pruned_8x8_no_res": {"kind": "dpsom",      "overrides": {**_PRUNED, "som_dim": (8, 8), "num_epochs_probe": 0}},
    "dpsom_legacy_full_8x8":   {"kind": "dpsom",      "overrides": {**_LEGACY_FULL, "som_dim": (8, 8)}},
}


def _ext_dataset_key(path: str) -> str:
    stem = Path(path).stem
    return stem[:-4] if stem.endswith("_100") else stem


def _eval_external_dpsom(
    model: DPSOM_ECG, ext_ds: ECG_Dataset, use_rr: bool, batch_size: int, prefix: str = "ext_"
) -> dict:
    gen = ECG_DataGenerator(ext_ds, batch_size=batch_size, use_rr_features=use_rr, preload_to_gpu=True)
    res = evaluate_model(
        model, gen, compute_disentanglement=False, compute_linear_probes=False, split="test", max_beats=None
    )
    out = {f"{prefix}{k}": v for k, v in res.items()}
    K = int(model.som_dim[0]) * int(model.som_dim[1])
    for m in ("NMI", "AMI", "Purity", "Rec_NMI", "Rec_AMI", "Rec_Purity"):
        if m in res:
            out[f"{prefix}{m}_k{K}"] = res[m]
    return out


def _eval_external_vae(
    model,
    kmeans_b: dict,
    kmeans_r: dict,
    probes: dict,
    ext_ds: ECG_Dataset,
    batch_size: int,
    use_rr: bool = False,
    prefix: str = "ext_",
) -> dict:
    device = next(model.parameters()).device
    gen = ECG_DataGenerator(ext_ds, batch_size=batch_size, use_rr_features=use_rr, preload_to_gpu=True)
    rec_gen = gen.get_record_batch_generator(mode="test", max_beats=None, shuffle=False)
    nb = int(np.ceil(len(ext_ds.get_data("test")) / batch_size))
    mu_b, lbl_b, mu_r, amp_r, lbl_r, age_r, sex_r = [], [], [], [], [], [], []
    model.eval()
    with torch.inference_mode():
        for _ in range(nb):
            beats, _, beat_meta, beat_valid, _, age_t, sex_t, labels_t = next(rec_gen)
            B, N, C, T = beats.shape
            valid = beat_valid.reshape(B * N) > 0.5
            mu_v, _ = model.encode(beats.reshape(B * N, C, T)[valid])
            mu_v = mu_v.float()
            mu_b.append(mu_v.cpu().numpy())
            lbl_b.append(labels_t.unsqueeze(1).expand(B, N).reshape(B * N)[valid].cpu().numpy())

            mu_full = torch.zeros(B * N, model.latent_dim, device=device)
            mu_full[valid] = mu_v
            denom = beat_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            keep_rec = (beat_valid.sum(dim=1) > 0.5).cpu().numpy()
            mu_mean = (mu_full.view(B, N, -1) * beat_valid.unsqueeze(-1)).sum(dim=1) / denom
            amp_mean = (beat_meta.float() * beat_valid.unsqueeze(-1)).sum(dim=1) / denom
            mu_r.append(mu_mean.cpu().numpy()[keep_rec])
            amp_r.append(amp_mean.cpu().numpy()[keep_rec])
            lbl_r.append(labels_t.cpu().numpy()[keep_rec])
            age_r.append(age_t.cpu().numpy()[keep_rec])
            sex_r.append(sex_t.cpu().numpy()[keep_rec])

    mu_b, lbl_b = np.concatenate(mu_b), np.concatenate(lbl_b)
    mu_r, amp_r, lbl_r = np.concatenate(mu_r), np.concatenate(amp_r), np.concatenate(lbl_r)
    age_r, sex_r = np.concatenate(age_r), np.concatenate(sex_r)

    out: dict = {}
    for k, kmb in kmeans_b.items():
        kmr = kmeans_r[k]
        out.update(_cluster_metrics_for_k(lbl_b, kmb.predict(mu_b), lbl_r, kmr.predict(mu_r), k, prefix=prefix))
    k_def = 64 if 64 in kmeans_b else next(iter(kmeans_b))
    for m in ("NMI", "AMI", "Purity", "Rec_NMI", "Rec_AMI", "Rec_Purity"):
        out[f"{prefix}{m}"] = out[f"{prefix}{m}_k{k_def}"]

    te_full = np.concatenate([mu_r, amp_r], axis=1)
    out.update({
        f"{prefix}MAE_Age":     float(metrics.mean_absolute_error(age_r, np.clip(probes["age_probe"].predict(mu_r), 2.0, 89.0))),
        f"{prefix}AUC_Sex":     float(metrics.roc_auc_score(sex_r, probes["sex_probe"].predict_proba(mu_r)[:, 1])),
        f"{prefix}MAE_Age_amp": float(metrics.mean_absolute_error(age_r, np.clip(probes["age_probe_amp"].predict(te_full), 2.0, 89.0))),
        f"{prefix}AUC_Sex_amp": float(metrics.roc_auc_score(sex_r, probes["sex_probe_amp"].predict_proba(te_full)[:, 1])),
    })
    return out


def _eval_external_resnet(
    model,
    kmeans_b: dict,
    kmeans_r: dict,
    ext_ds: ECG_Dataset,
    use_rr: bool,
    batch_size: int,
    prefix: str = "ext_",
) -> dict:
    device = next(model.parameters()).device
    gen = ECG_DataGenerator(ext_ds, batch_size=batch_size, use_rr_features=use_rr, preload_to_gpu=True)
    rec_gen = gen.get_record_batch_generator(mode="test", max_beats=None, shuffle=False)
    nb = int(np.ceil(len(ext_ds.get_data("test")) / batch_size))
    h_b, lbl_b, pred_cls_b = [], [], []
    h_r, lbl_r, pred_cls_r = [], [], []
    age_t_all, age_p_all, sex_t_all, sex_p_all = [], [], [], []
    model.eval()
    with torch.inference_mode():
        for _ in range(nb):
            beats, _, beat_meta, beat_valid, _, age_t, sex_t, labels_t = next(rec_gen)
            B, N, C, T = beats.shape
            valid = beat_valid.reshape(B * N) > 0.5
            cls_l, age_l, sex_l, h = model(beats.reshape(B * N, C, T)[valid], beat_meta.reshape(B * N, -1)[valid])
            h = h.float()
            h_b.append(h.cpu().numpy())
            pred_cls_b.append(cls_l.argmax(dim=-1).cpu().numpy())
            lbl_b.append(labels_t.unsqueeze(1).expand(B, N).reshape(B * N)[valid].cpu().numpy())

            h_full = torch.zeros(B * N, h.shape[1], device=device)
            cls_full = torch.zeros(B * N, cls_l.shape[1], device=device)
            age_full = torch.zeros(B * N, device=device)
            sex_full = torch.zeros(B * N, device=device)
            h_full[valid] = h
            cls_full[valid] = torch.softmax(cls_l.float(), dim=-1)
            age_full[valid] = (torch.softmax(age_l.float(), dim=-1) * model.age_bin_values).sum(dim=-1)
            sex_full[valid] = sex_l.float()

            denom = beat_valid.sum(dim=1).clamp_min(1.0)
            keep_rec = (beat_valid.sum(dim=1) > 0.5).cpu().numpy()
            h_mean = (h_full.view(B, N, -1) * beat_valid.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
            cls_mean = (cls_full.view(B, N, -1) * beat_valid.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
            h_r.append(h_mean.cpu().numpy()[keep_rec])
            pred_cls_r.append(cls_mean.argmax(dim=-1).cpu().numpy()[keep_rec])
            lbl_r.append(labels_t.cpu().numpy()[keep_rec])
            age_p_all.append(((age_full.view(B, N) * beat_valid).sum(dim=1) / denom).cpu().numpy()[keep_rec])
            sex_p_all.append(torch.sigmoid((sex_full.view(B, N) * beat_valid).sum(dim=1) / denom).cpu().numpy()[keep_rec])
            age_t_all.append(age_t.cpu().numpy()[keep_rec])
            sex_t_all.append(sex_t.cpu().numpy()[keep_rec])

    h_b, lbl_b, pred_cls_b = np.concatenate(h_b), np.concatenate(lbl_b), np.concatenate(pred_cls_b)
    h_r, lbl_r, pred_cls_r = np.concatenate(h_r), np.concatenate(lbl_r), np.concatenate(pred_cls_r)

    out: dict = {}
    for k, kmb in kmeans_b.items():
        kmr = kmeans_r[k]
        out.update(_cluster_metrics_for_k(lbl_b, kmb.predict(h_b), lbl_r, kmr.predict(h_r), k, prefix=prefix))
    out.update({
        f"{prefix}NMI":        float(metrics.normalized_mutual_info_score(lbl_b, pred_cls_b, average_method="geometric")),
        f"{prefix}AMI":        float(metrics.adjusted_mutual_info_score(lbl_b, pred_cls_b, average_method="geometric")),
        f"{prefix}Purity":     float(cluster_purity(lbl_b, pred_cls_b)),
        f"{prefix}Rec_NMI":    float(metrics.normalized_mutual_info_score(lbl_r, pred_cls_r, average_method="geometric")),
        f"{prefix}Rec_AMI":    float(metrics.adjusted_mutual_info_score(lbl_r, pred_cls_r, average_method="geometric")),
        f"{prefix}Rec_Purity": float(cluster_purity(lbl_r, pred_cls_r)),
        f"{prefix}MAE_Age":    float(metrics.mean_absolute_error(np.concatenate(age_t_all), np.concatenate(age_p_all))),
        f"{prefix}AUC_Sex":    float(metrics.roc_auc_score(np.concatenate(sex_t_all), np.concatenate(sex_p_all))),
    })
    return out


def _run_dpsom_once(
    name: str,
    overrides: dict,
    seed: int,
    epochs: int,
    ds: ECG_Dataset,
    ext_datasets: list[tuple[str, str, ECG_Dataset]] | None,
    compute_dis: bool = False,
) -> dict:
    t0 = time.time()
    cfg = DPSOM_Config()
    cfg.random_seed = seed
    cfg.num_epochs = epochs
    for k, v in overrides.items():
        setattr(cfg, k, v)
    cfg.ex_name = f"paper_{name}_s{seed}"
    cfg.logdir = f"logs/{cfg.ex_name}"
    cfg.modelpath = f"models/{cfg.ex_name}/{cfg.ex_name}.ckpt"
    _set_global_determinism(cfg.random_seed, fast_gpu=cfg.fast_gpu)

    gen = ECG_DataGenerator(
        ds, batch_size=cfg.batch_size, use_rr_features=cfg.use_rr_features, preload_to_gpu=cfg.preload_to_gpu
    )
    sample = gen.get_flat_beats_tensor("train")
    input_channels, input_length = sample.shape[1], sample.shape[2]
    amp_dim = 2 * input_channels + (2 if cfg.use_rr_features else 0)

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

    res = evaluate_model(
        model, gen, compute_disentanglement=compute_dis, compute_linear_probes=True, split="test", max_beats=None
    )
    val_res = evaluate_model(
        model, gen, compute_disentanglement=False, compute_linear_probes=False, split="val", max_beats=None
    )
    K = int(cfg.som_dim[0]) * int(cfg.som_dim[1])
    row = {
        "NMI": res["NMI"],
        "AMI": res["AMI"],
        "Purity": res["Purity"],
        "Rec_NMI": res["Rec_NMI"],
        "Rec_AMI": res["Rec_AMI"],
        "Rec_Purity": res["Rec_Purity"],
        f"NMI_k{K}": res["NMI"],
        f"AMI_k{K}": res["AMI"],
        f"Purity_k{K}": res["Purity"],
        f"Rec_NMI_k{K}": res["Rec_NMI"],
        f"Rec_AMI_k{K}": res["Rec_AMI"],
        f"Rec_Purity_k{K}": res["Rec_Purity"],
        "MAE_Age": res["MAE_Age"],
        "AUC_Sex": res["AUC_Sex"],
        "val_NMI": val_res["NMI"],
        "val_AMI": val_res["AMI"],
        "val_Purity": val_res["Purity"],
        "val_Rec_NMI": val_res["Rec_NMI"],
        "val_Rec_AMI": val_res["Rec_AMI"],
        "val_Rec_Purity": val_res["Rec_Purity"],
        "val_MAE_Age": val_res["MAE_Age"],
        "val_AUC_Sex": val_res["AUC_Sex"],
        "elapsed_s": round(time.time() - t0, 1),
    }
    if "Linear_Probes" in res:
        row.update(res["Linear_Probes"])
    if compute_dis:
        row["SAP_age_raw"] = res["Disentanglement"]["z_main"]["SAP"]["SAP_age"]
        row["SAP_sex_raw"] = res["Disentanglement"]["z_main"]["SAP"]["SAP_sex"]
        row["SAP_age_res"] = res["Disentanglement_Residualized"]["z_main"]["SAP"]["SAP_age"]
        row["SAP_sex_res"] = res["Disentanglement_Residualized"]["z_main"]["SAP"]["SAP_sex"]
        row["DCI_info_raw"] = res["Disentanglement"]["z_main"]["DCI"]["DCI_informativeness"]
        row["DCI_info_res"] = res["Disentanglement_Residualized"]["z_main"]["DCI"]["DCI_informativeness"]

    if ext_datasets:
        for _key, prefix, ext_ds in ext_datasets:
            row.update(_eval_external_dpsom(model, ext_ds, cfg.use_rr_features, cfg.batch_size, prefix=prefix))

    shutil.rmtree(os.path.dirname(cfg.modelpath), ignore_errors=True)
    shutil.rmtree(cfg.logdir, ignore_errors=True)
    return row


def compute_chance_baselines(
    ds: ECG_Dataset,
    ext_datasets: list[tuple[str, str, ECG_Dataset]] | None,
    ks: tuple[int, ...] = KS,
    n_draws: int = 5,
) -> dict:
    """Empirical random-beat & random-record chance baselines using exact evaluation beats (max_beats=None)."""
    rng = np.random.default_rng(0)
    chance: dict = {}
    targets = [("ptbxl_test", ds, "test")]
    if ext_datasets:
        for key, _prefix, ext_ds in ext_datasets:
            targets.append((f"{key}_test", ext_ds, "test"))

    for name, dset, split in targets:
        recs = dset.get_data(split)
        enc = dset.label_encoder
        y_rec, y_beat, rec_ids = [], [], []
        for idx, r in enumerate(recs):
            if r.beat_representations is None or len(r.beat_representations) == 0:
                continue
            lbl = int(enc.transform([r.label])[0])
            nb = len(r.beat_representations)
            y_rec.append(lbl)
            y_beat.extend([lbl] * nb)
            rec_ids.extend([idx] * nb)
        y_rec_arr = np.asarray(y_rec, dtype=np.int64)
        y_beat_arr = np.asarray(y_beat, dtype=np.int64)
        rid_arr = np.asarray(rec_ids, dtype=np.int64)
        n_recs = len(recs)
        maj_rec = float(np.bincount(y_rec_arr).max() / len(y_rec_arr))
        maj_beat = float(np.bincount(y_beat_arr).max() / len(y_beat_arr))

        entry: dict = {
            "n_records": int(len(y_rec_arr)),
            "n_beats": int(len(y_beat_arr)),
            "n_classes": int(len(np.unique(y_rec_arr))),
            "majority_purity_rec": maj_rec,
            "majority_purity_beat": maj_beat,
        }
        for k in ks:
            rand_beat_nmi, rand_beat_ami, rand_beat_pur = [], [], []
            rec_hash_beat_nmi, rec_hash_beat_ami, rec_hash_beat_pur = [], [], []
            rand_rec_nmi, rand_rec_ami, rand_rec_pur = [], [], []
            for _ in range(n_draws):
                cb = rng.integers(0, k, size=len(y_beat_arr))
                rand_beat_nmi.append(metrics.normalized_mutual_info_score(y_beat_arr, cb, average_method="geometric"))
                rand_beat_ami.append(metrics.adjusted_mutual_info_score(y_beat_arr, cb, average_method="geometric"))
                rand_beat_pur.append(cluster_purity(y_beat_arr, cb))

                rc_all = rng.integers(0, k, size=n_recs)
                cb_rec = rc_all[rid_arr]
                rec_hash_beat_nmi.append(metrics.normalized_mutual_info_score(y_beat_arr, cb_rec, average_method="geometric"))
                rec_hash_beat_ami.append(metrics.adjusted_mutual_info_score(y_beat_arr, cb_rec, average_method="geometric"))
                rec_hash_beat_pur.append(cluster_purity(y_beat_arr, cb_rec))

                cr = rng.integers(0, k, size=len(y_rec_arr))
                rand_rec_nmi.append(metrics.normalized_mutual_info_score(y_rec_arr, cr, average_method="geometric"))
                rand_rec_ami.append(metrics.adjusted_mutual_info_score(y_rec_arr, cr, average_method="geometric"))
                rand_rec_pur.append(cluster_purity(y_rec_arr, cr))

            entry[f"k{k}"] = {
                "random_beat":     {"NMI": float(np.mean(rand_beat_nmi)), "AMI": float(np.mean(rand_beat_ami)), "Purity": float(np.mean(rand_beat_pur))},
                "random_rec_hash": {"NMI": float(np.mean(rec_hash_beat_nmi)), "AMI": float(np.mean(rec_hash_beat_ami)), "Purity": float(np.mean(rec_hash_beat_pur))},
                "random_record":   {"NMI": float(np.mean(rand_rec_nmi)), "AMI": float(np.mean(rand_rec_ami)), "Purity": float(np.mean(rand_rec_pur))},
            }
        chance[name] = entry
    return chance


def summarize(rows: list[dict]) -> list[dict]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    skip = {"model", "seed", "elapsed_s"}
    summary_rows = []
    for model_name, mrows in by_model.items():
        s: dict = {"model": model_name, "n_seeds": len(mrows)}
        all_keys: list[str] = []
        for r in mrows:
            for k in r:
                if k not in skip and k not in all_keys and isinstance(r[k], (int, float)):
                    all_keys.append(k)
        for k in all_keys:
            vals = [float(r[k]) for r in mrows if k in r and r[k] is not None and np.isfinite(float(r[k]))]
            if vals:
                s[f"{k}_mean"] = float(np.mean(vals))
                s[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(s)
    return summary_rows


def _write_outputs(all_rows: list[dict], out_json: str, chance: dict | None = None) -> None:
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    payload: dict = {"runs": all_rows, "summary": summarize(all_rows)}
    if chance:
        payload["chance"] = chance
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_json[:-5] + ".csv" if out_json.endswith(".json") else out_json + ".csv"
    keys: list[str] = []
    for r in all_rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Fair matched-k multi-seed benchmark runner")
    p.add_argument("--models", nargs="+", default=list(EXPERIMENTS.keys()))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--ext-data", nargs="*", default=["data/processed/georgia_100.pkl"])
    p.add_argument("--output", type=str, default="ablation/paper_results_all.json")
    p.add_argument("--compute-disentanglement", action="store_true")
    p.add_argument("--merge", nargs="+", default=None, help="Merge shard JSON files in canonical model order")
    args = p.parse_args()

    cfg_base = DPSOM_Config()
    print("[paper] Loading PTB-XL dataset once...")
    ds = ECG_Dataset.load("./data/processed/ptbxl_100_T-12ms.pkl")
    if os.path.exists(cfg_base.error_labels_path):
        ds.apply_error_labels(
            cfg_base.error_labels_path,
            max_error_severity=cfg_base.max_error_severity,
            exclude_error_flags=cfg_base.exclude_error_flags,
        )

    ext_datasets: list[tuple[str, str, ECG_Dataset]] = []
    for idx_ext, ext_path in enumerate(args.ext_data or []):
        if ext_path and os.path.exists(ext_path):
            key = _ext_dataset_key(ext_path)
            prefix = "ext_" if idx_ext == 0 else f"ext_{key}_"
            eds = ECG_Dataset.load(ext_path)
            ext_datasets.append((key, prefix, eds))
            print(f"[paper] Loaded external dataset '{key}' ({prefix}): {ext_path} ({len(eds.get_data('test'))} records)")

    chance = compute_chance_baselines(ds, ext_datasets)

    if args.merge:
        order = list(EXPERIMENTS.keys())
        merged_rows: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for path in args.merge:
            d = json.loads(Path(path).read_text())
            for r in d["runs"]:
                key = (r["model"], int(r["seed"]))
                if key not in seen:
                    seen.add(key)
                    merged_rows.append(r)
        merged_rows.sort(key=lambda r: (order.index(r["model"]) if r["model"] in order else 999, int(r["seed"])))
        _write_outputs(merged_rows, args.output, chance=chance)
        print(f"[paper] Merged {len(merged_rows)} runs + chance baselines -> {args.output}")
        return

    all_rows: list[dict] = []
    for model_name in args.models:
        if model_name not in EXPERIMENTS:
            print(f"[paper] Unknown model '{model_name}', skipping.")
            continue
        spec = EXPERIMENTS[model_name]
        for idx, seed in enumerate(args.seeds):
            print(f"\n{'=' * 66}\n  MODEL: {model_name}  |  SEED: {seed}\n{'=' * 66}")
            cfg = DPSOM_Config()
            cfg.random_seed = seed
            for k, v in spec.get("overrides", {}).items():
                setattr(cfg, k, v)

            if spec["kind"] == "vae_kmeans":
                res, vae, km_b, km_r, probes = train_eval_vae_kmeans(
                    cfg, ks=KS, num_epochs=args.epochs + 10, ds=ds
                )
                for _key, prefix, ext_ds in ext_datasets:
                    res.update(_eval_external_vae(
                        vae, km_b, km_r, probes, ext_ds, cfg.batch_size, use_rr=cfg.use_rr_features, prefix=prefix
                    ))
            elif spec["kind"] == "resnet1d":
                res, rnet, km_b, km_r = train_eval_resnet1d(
                    cfg, ks=KS, num_epochs=args.epochs + 10, ds=ds
                )
                for _key, prefix, ext_ds in ext_datasets:
                    res.update(_eval_external_resnet(
                        rnet, km_b, km_r, ext_ds, cfg.use_rr_features, cfg.batch_size, prefix=prefix
                    ))
            else:
                res = _run_dpsom_once(
                    model_name, spec["overrides"], seed, args.epochs, ds, ext_datasets,
                    compute_dis=(args.compute_disentanglement and idx == 0),
                )

            row = {"model": model_name, "seed": seed, **res}
            all_rows.append(row)
            r2_str = (
                f"  ageR2={row['age_r2_raw']:.3f}->{row['age_r2_res']:.3f}"
                if "age_r2_raw" in row else ""
            )
            print(
                f"[result] {model_name} (s={seed}): RecAMI={row['Rec_AMI']:.4f}  RecNMI={row['Rec_NMI']:.4f}  "
                f"BeatNMI={row['NMI']:.4f}  MAE={row['MAE_Age']:.2f}  AUC={row['AUC_Sex']:.4f}{r2_str}"
                + (f"  | ext RecAMI={row['ext_Rec_AMI']:.4f}  ext MAE={row['ext_MAE_Age']:.2f}"
                   if "ext_MAE_Age" in row else "")
            )
            _write_outputs(all_rows, args.output, chance=chance)

    print(f"\n[paper] Saved {len(all_rows)} runs -> {args.output}")


if __name__ == "__main__":
    main()
