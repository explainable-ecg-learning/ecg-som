"""Training and evaluation routines for VAE+KMeans and Supervised 1D-ResNet baselines."""
from __future__ import annotations

import copy
import os
import time

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DPSOM_Config
from src.data.dataset import ECG_Dataset
from src.data.generator import ECG_DataGenerator
from src.model.baselines import ECG_VAE, ResNet1D_ECG
from src.training.trainer import _amp_ctx, _get_device, _set_global_determinism
from src.utils.metrics import cluster_purity


def _load_curated_dataset(cfg: DPSOM_Config) -> ECG_Dataset:
    ds = ECG_Dataset.load("./data/processed/ptbxl_100_T-12ms.pkl")
    err_path = getattr(cfg, "error_labels_path", None)
    max_sev = getattr(cfg, "max_error_severity", None)
    exc_flg = getattr(cfg, "exclude_error_flags", None)
    if err_path and os.path.exists(err_path) and (max_sev is not None or exc_flg):
        ds.apply_error_labels(err_path, max_error_severity=max_sev, exclude_error_flags=exc_flg, verbose=False)
    return ds


def _cluster_metrics_for_k(
    labels_b: np.ndarray,
    preds_b: np.ndarray,
    labels_r: np.ndarray,
    preds_r: np.ndarray,
    k: int,
    prefix: str = "",
) -> dict:
    return {
        f"{prefix}NMI_k{k}":        float(metrics.normalized_mutual_info_score(labels_b, preds_b, average_method="geometric")),
        f"{prefix}AMI_k{k}":        float(metrics.adjusted_mutual_info_score(labels_b, preds_b, average_method="geometric")),
        f"{prefix}Purity_k{k}":     float(cluster_purity(labels_b, preds_b)),
        f"{prefix}Rec_NMI_k{k}":    float(metrics.normalized_mutual_info_score(labels_r, preds_r, average_method="geometric")),
        f"{prefix}Rec_AMI_k{k}":    float(metrics.adjusted_mutual_info_score(labels_r, preds_r, average_method="geometric")),
        f"{prefix}Rec_Purity_k{k}": float(cluster_purity(labels_r, preds_r)),
    }


def train_eval_vae_kmeans(
    cfg: DPSOM_Config,
    ks: tuple[int, ...] = (16, 64, 256),
    num_epochs: int = 40,
    ds: ECG_Dataset | None = None,
) -> tuple[dict, ECG_VAE, dict[int, MiniBatchKMeans], dict[int, MiniBatchKMeans], dict]:
    """Train ECG_VAE with val-ELBO selection, fit matched-k K-Means & linear probes on train, eval on val/test."""
    t0 = time.time()
    _set_global_determinism(cfg.random_seed, fast_gpu=cfg.fast_gpu)
    device = _get_device()
    ds = ds if ds is not None else _load_curated_dataset(cfg)
    gen = ECG_DataGenerator(
        ds, batch_size=cfg.batch_size, use_rr_features=cfg.use_rr_features, preload_to_gpu=cfg.preload_to_gpu
    )

    sample = gen.get_flat_beats_tensor("train")
    input_channels, input_length = sample.shape[1], sample.shape[2]

    model = ECG_VAE(
        latent_dim=cfg.latent_dim,
        input_length=input_length,
        input_channels=input_channels,
        dropout=cfg.dropout,
        prior=cfg.prior,
        encoder_base_channels_1=cfg.encoder_base_channels_1,
        encoder_base_channels_2=cfg.encoder_base_channels_2,
        encoder_kernel_size=cfg.encoder_kernel_size,
        encoder_fc_hidden_dim=cfg.encoder_fc_hidden_dim,
    ).to(device)

    use_fused = cfg.fast_gpu and device.type == "cuda"
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
                     **({"fused": True} if use_fused else {}))
    train_gen = gen.get_record_batch_generator(mode="train", max_beats=cfg.num_beats, shuffle=True)
    num_batches = gen.get_num_batches("train")
    val_gen = gen.get_record_batch_generator(mode="val", max_beats=cfg.num_beats, shuffle=False)
    num_val_batches = gen.get_num_batches("val")

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for _ in range(num_epochs):
        model.train()
        for _ in range(num_batches):
            beats, beat_mask, _, beat_valid = next(train_gen)[:4]
            opt.zero_grad(set_to_none=True)
            with _amp_ctx(device, cfg.use_amp):
                loss, _, _ = model.loss_batch(beats, beat_mask, beat_valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradient_clip_norm)
            opt.step()

        model.eval()
        val_sum = 0.0
        with torch.inference_mode(), _amp_ctx(device, cfg.use_amp):
            for _ in range(num_val_batches):
                vb = next(val_gen)
                vl, _, _ = model.loss_batch(vb[0], vb[1], vb[3])
                val_sum += float(vl.item())
        val_avg = val_sum / max(1, num_val_batches)
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    model.eval()

    def _extract_mu(split: str):
        rec_gen = gen.get_record_batch_generator(mode=split, max_beats=None, shuffle=False)
        n_rec = len(ds.get_data(split))
        nb = int(np.ceil(n_rec / gen.batch_size))
        mu_beats, lbl_beats = [], []
        mu_rec, amp_rec, lbl_rec, age_rec, sex_rec = [], [], [], [], []
        with torch.inference_mode():
            for _ in range(nb):
                beats, _, beat_meta, beat_valid, _, age_t, sex_t, labels_t = next(rec_gen)
                B, N, C, T = beats.shape
                valid = beat_valid.reshape(B * N) > 0.5
                mu_v, _ = model.encode(beats.reshape(B * N, C, T)[valid])
                mu_v = mu_v.float()
                mu_beats.append(mu_v.cpu().numpy())
                lbl_beats.append(labels_t.unsqueeze(1).expand(B, N).reshape(B * N)[valid].cpu().numpy())

                mu_full = torch.zeros(B * N, cfg.latent_dim, device=device)
                mu_full[valid] = mu_v
                denom = beat_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
                mu_mean = (mu_full.view(B, N, cfg.latent_dim) * beat_valid.unsqueeze(-1)).sum(dim=1) / denom
                amp_mean = (beat_meta.float() * beat_valid.unsqueeze(-1)).sum(dim=1) / denom
                keep_rec = (beat_valid.sum(dim=1) > 0.5).cpu().numpy()
                mu_rec.append(mu_mean.cpu().numpy()[keep_rec])
                amp_rec.append(amp_mean.cpu().numpy()[keep_rec])
                lbl_rec.append(labels_t.cpu().numpy()[keep_rec])
                age_rec.append(age_t.cpu().numpy()[keep_rec])
                sex_rec.append(sex_t.cpu().numpy()[keep_rec])
        return (
            np.concatenate(mu_beats),
            np.concatenate(lbl_beats),
            np.concatenate(mu_rec),
            np.concatenate(amp_rec),
            np.concatenate(lbl_rec),
            np.concatenate(age_rec),
            np.concatenate(sex_rec),
        )

    mu_tr_b, _, mu_tr_r, amp_tr_r, _, age_tr_r, sex_tr_r = _extract_mu("train")
    mu_va_b, lbl_va_b, mu_va_r, amp_va_r, lbl_va_r, age_va_r, sex_va_r = _extract_mu("val")
    mu_te_b, lbl_te_b, mu_te_r, amp_te_r, lbl_te_r, age_te_r, sex_te_r = _extract_mu("test")

    kmeans_b: dict[int, MiniBatchKMeans] = {}
    kmeans_r: dict[int, MiniBatchKMeans] = {}
    res: dict = {}

    for k in ks:
        kmb = MiniBatchKMeans(n_clusters=k, random_state=cfg.random_seed, batch_size=2048, n_init="auto").fit(mu_tr_b)
        kmr = MiniBatchKMeans(n_clusters=k, random_state=cfg.random_seed, batch_size=2048, n_init="auto").fit(mu_tr_r)
        kmeans_b[k] = kmb
        kmeans_r[k] = kmr
        res.update(_cluster_metrics_for_k(lbl_va_b, kmb.predict(mu_va_b), lbl_va_r, kmr.predict(mu_va_r), k, prefix="val_"))
        res.update(_cluster_metrics_for_k(lbl_te_b, kmb.predict(mu_te_b), lbl_te_r, kmr.predict(mu_te_r), k, prefix=""))

    # Default k=64 aliases for backward compatibility
    k_def = 64 if 64 in ks else ks[0]
    for m in ("NMI", "AMI", "Purity", "Rec_NMI", "Rec_AMI", "Rec_Purity"):
        res[m] = res[f"{m}_k{k_def}"]
        res[f"val_{m}"] = res[f"val_{m}_k{k_def}"]

    # Probes on mu only, and probes on [mu, amp] (same input signals as DPSOM's age/sex heads)
    age_probe = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(mu_tr_r, age_tr_r)
    sex_probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(mu_tr_r, sex_tr_r)

    tr_full = np.concatenate([mu_tr_r, amp_tr_r], axis=1)
    va_full = np.concatenate([mu_va_r, amp_va_r], axis=1)
    te_full = np.concatenate([mu_te_r, amp_te_r], axis=1)
    age_probe_amp = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(tr_full, age_tr_r)
    sex_probe_amp = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr_full, sex_tr_r)

    res.update({
        "MAE_Age":         float(metrics.mean_absolute_error(age_te_r, np.clip(age_probe.predict(mu_te_r), 2.0, 89.0))),
        "AUC_Sex":         float(metrics.roc_auc_score(sex_te_r, sex_probe.predict_proba(mu_te_r)[:, 1])),
        "MAE_Age_amp":     float(metrics.mean_absolute_error(age_te_r, np.clip(age_probe_amp.predict(te_full), 2.0, 89.0))),
        "AUC_Sex_amp":     float(metrics.roc_auc_score(sex_te_r, sex_probe_amp.predict_proba(te_full)[:, 1])),
        "val_MAE_Age":     float(metrics.mean_absolute_error(age_va_r, np.clip(age_probe.predict(mu_va_r), 2.0, 89.0))),
        "val_AUC_Sex":     float(metrics.roc_auc_score(sex_va_r, sex_probe.predict_proba(mu_va_r)[:, 1])),
        "val_MAE_Age_amp": float(metrics.mean_absolute_error(age_va_r, np.clip(age_probe_amp.predict(va_full), 2.0, 89.0))),
        "val_AUC_Sex_amp": float(metrics.roc_auc_score(sex_va_r, sex_probe_amp.predict_proba(va_full)[:, 1])),
        "elapsed_s":       round(time.time() - t0, 1),
    })
    probes = {
        "age_probe": age_probe, "sex_probe": sex_probe,
        "age_probe_amp": age_probe_amp, "sex_probe_amp": sex_probe_amp,
    }
    return res, model, kmeans_b, kmeans_r, probes


def train_eval_resnet1d(
    cfg: DPSOM_Config,
    ks: tuple[int, ...] = (16, 64, 256),
    num_epochs: int = 40,
    ds: ECG_Dataset | None = None,
) -> tuple[dict, ResNet1D_ECG, dict[int, MiniBatchKMeans], dict[int, MiniBatchKMeans]]:
    """Train supervised 1D-ResNet with val selection; evaluate cls head and matched-k penultimate h clustering."""
    t0 = time.time()
    _set_global_determinism(cfg.random_seed, fast_gpu=cfg.fast_gpu)
    device = _get_device()
    ds = ds if ds is not None else _load_curated_dataset(cfg)
    gen = ECG_DataGenerator(
        ds, batch_size=cfg.batch_size, use_rr_features=cfg.use_rr_features, preload_to_gpu=cfg.preload_to_gpu
    )

    sample = gen.get_flat_beats_tensor("train")
    input_channels = sample.shape[1]
    amp_dim = 2 * input_channels + (2 if cfg.use_rr_features else 0)
    num_classes = len(ds.class_names)

    model = ResNet1D_ECG(
        input_channels=input_channels,
        num_classes=num_classes,
        amp_context_dim=amp_dim,
        dropout=cfg.dropout,
    ).to(device)

    use_fused = cfg.fast_gpu and device.type == "cuda"
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4,
                     **({"fused": True} if use_fused else {}))
    train_gen = gen.get_record_batch_generator(mode="train", max_beats=cfg.num_beats, shuffle=True)
    num_batches = gen.get_num_batches("train")
    val_gen = gen.get_record_batch_generator(mode="val", max_beats=cfg.num_beats, shuffle=False)
    num_val_batches = gen.get_num_batches("val")

    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    def _batch_loss(batch):
        beats, _, beat_meta, beat_valid, _, age_t, sex_t, labels_t = batch
        B, N, C, T = beats.shape
        valid = beat_valid.reshape(B * N) > 0.5
        x = beats.reshape(B * N, C, T)[valid]
        meta = beat_meta.reshape(B * N, -1)[valid]
        y_cls = labels_t.unsqueeze(1).expand(B, N).reshape(B * N)[valid]
        y_age = age_t.unsqueeze(1).expand(B, N).reshape(B * N)[valid]
        y_sex = sex_t.unsqueeze(1).expand(B, N).reshape(B * N)[valid].float()
        age_bins = ((y_age - 2.0) / 87.0 * 87.0).clamp(0, 87).long()
        cls_logits, age_logits, sex_logits, _ = model(x, meta)
        return (
            F.cross_entropy(cls_logits, y_cls)
            + F.cross_entropy(age_logits, age_bins)
            + F.binary_cross_entropy_with_logits(sex_logits, y_sex)
        )

    for _ in range(num_epochs):
        model.train()
        for _ in range(num_batches):
            opt.zero_grad(set_to_none=True)
            with _amp_ctx(device, cfg.use_amp):
                loss = _batch_loss(next(train_gen))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradient_clip_norm)
            opt.step()

        model.eval()
        val_sum = 0.0
        with torch.inference_mode(), _amp_ctx(device, cfg.use_amp):
            for _ in range(num_val_batches):
                val_sum += float(_batch_loss(next(val_gen)).item())
        val_avg = val_sum / max(1, num_val_batches)
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state, strict=True)
    model.eval()

    def _extract_split(split: str):
        rec_gen = gen.get_record_batch_generator(mode=split, max_beats=None, shuffle=False)
        n_rec = len(ds.get_data(split))
        nb = int(np.ceil(n_rec / gen.batch_size))
        h_beats, lbl_beats, pred_cls_b = [], [], []
        h_rec, pred_cls_r, lbl_rec = [], [], []
        age_true_r, age_pred_r, sex_true_r, sex_prob_r = [], [], [], []
        with torch.inference_mode():
            for _ in range(nb):
                beats, _, beat_meta, beat_valid, _, age_t, sex_t, labels_t = next(rec_gen)
                B, N, C, T = beats.shape
                valid = beat_valid.reshape(B * N) > 0.5
                x = beats.reshape(B * N, C, T)[valid]
                meta = beat_meta.reshape(B * N, -1)[valid]
                cls_logits, age_logits, sex_logits, h = model(x, meta)
                h = h.float()
                h_beats.append(h.cpu().numpy())
                pred_cls_b.append(cls_logits.argmax(dim=-1).cpu().numpy())
                lbl_beats.append(labels_t.unsqueeze(1).expand(B, N).reshape(B * N)[valid].cpu().numpy())

                h_full = torch.zeros(B * N, h.shape[1], device=device)
                cls_full = torch.zeros(B * N, cls_logits.shape[1], device=device)
                age_full = torch.zeros(B * N, device=device)
                sex_full = torch.zeros(B * N, device=device)
                h_full[valid] = h
                cls_full[valid] = torch.softmax(cls_logits.float(), dim=-1)
                age_probs = torch.softmax(age_logits.float(), dim=-1)
                age_full[valid] = (age_probs * model.age_bin_values).sum(dim=-1)
                sex_full[valid] = sex_logits.float()

                denom = beat_valid.sum(dim=1).clamp_min(1.0)
                keep_rec = (beat_valid.sum(dim=1) > 0.5).cpu().numpy()
                h_mean = (h_full.view(B, N, -1) * beat_valid.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
                cls_mean = (cls_full.view(B, N, -1) * beat_valid.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)
                age_r = (age_full.view(B, N) * beat_valid).sum(dim=1) / denom
                sex_r = (sex_full.view(B, N) * beat_valid).sum(dim=1) / denom

                h_rec.append(h_mean.cpu().numpy()[keep_rec])
                pred_cls_r.append(cls_mean.argmax(dim=-1).cpu().numpy()[keep_rec])
                lbl_rec.append(labels_t.cpu().numpy()[keep_rec])
                age_true_r.append(age_t.cpu().numpy()[keep_rec])
                age_pred_r.append(age_r.cpu().numpy()[keep_rec])
                sex_true_r.append(sex_t.cpu().numpy()[keep_rec])
                sex_prob_r.append(torch.sigmoid(sex_r).cpu().numpy()[keep_rec])
        return {
            "h_b": np.concatenate(h_beats),
            "lbl_b": np.concatenate(lbl_beats),
            "pred_cls_b": np.concatenate(pred_cls_b),
            "h_r": np.concatenate(h_rec),
            "lbl_r": np.concatenate(lbl_rec),
            "pred_cls_r": np.concatenate(pred_cls_r),
            "age_t": np.concatenate(age_true_r),
            "age_p": np.concatenate(age_pred_r),
            "sex_t": np.concatenate(sex_true_r),
            "sex_p": np.concatenate(sex_prob_r),
        }

    tr = _extract_split("train")
    va = _extract_split("val")
    te = _extract_split("test")

    kmeans_b: dict[int, MiniBatchKMeans] = {}
    kmeans_r: dict[int, MiniBatchKMeans] = {}
    res: dict = {}

    for k in ks:
        kmb = MiniBatchKMeans(n_clusters=k, random_state=cfg.random_seed, batch_size=2048, n_init="auto").fit(tr["h_b"])
        kmr = MiniBatchKMeans(n_clusters=k, random_state=cfg.random_seed, batch_size=2048, n_init="auto").fit(tr["h_r"])
        kmeans_b[k] = kmb
        kmeans_r[k] = kmr
        res.update(_cluster_metrics_for_k(va["lbl_b"], kmb.predict(va["h_b"]), va["lbl_r"], kmr.predict(va["h_r"]), k, prefix="val_"))
        res.update(_cluster_metrics_for_k(te["lbl_b"], kmb.predict(te["h_b"]), te["lbl_r"], kmr.predict(te["h_r"]), k, prefix=""))

    # Supervised head metrics (16 classes)
    res.update({
        "NMI":          float(metrics.normalized_mutual_info_score(te["lbl_b"], te["pred_cls_b"], average_method="geometric")),
        "AMI":          float(metrics.adjusted_mutual_info_score(te["lbl_b"], te["pred_cls_b"], average_method="geometric")),
        "Purity":       float(cluster_purity(te["lbl_b"], te["pred_cls_b"])),
        "Accuracy":     float(metrics.accuracy_score(te["lbl_b"], te["pred_cls_b"])),
        "Rec_NMI":      float(metrics.normalized_mutual_info_score(te["lbl_r"], te["pred_cls_r"], average_method="geometric")),
        "Rec_AMI":      float(metrics.adjusted_mutual_info_score(te["lbl_r"], te["pred_cls_r"], average_method="geometric")),
        "Rec_Purity":   float(cluster_purity(te["lbl_r"], te["pred_cls_r"])),
        "Rec_Accuracy": float(metrics.accuracy_score(te["lbl_r"], te["pred_cls_r"])),
        "MAE_Age":      float(metrics.mean_absolute_error(te["age_t"], te["age_p"])),
        "AUC_Sex":      float(metrics.roc_auc_score(te["sex_t"], te["sex_p"])),
        "val_NMI":      float(metrics.normalized_mutual_info_score(va["lbl_b"], va["pred_cls_b"], average_method="geometric")),
        "val_AMI":      float(metrics.adjusted_mutual_info_score(va["lbl_b"], va["pred_cls_b"], average_method="geometric")),
        "val_Purity":   float(cluster_purity(va["lbl_b"], va["pred_cls_b"])),
        "val_Rec_NMI":  float(metrics.normalized_mutual_info_score(va["lbl_r"], va["pred_cls_r"], average_method="geometric")),
        "val_Rec_AMI":  float(metrics.adjusted_mutual_info_score(va["lbl_r"], va["pred_cls_r"], average_method="geometric")),
        "val_MAE_Age":  float(metrics.mean_absolute_error(va["age_t"], va["age_p"])),
        "val_AUC_Sex":  float(metrics.roc_auc_score(va["sex_t"], va["sex_p"])),
        "elapsed_s":    round(time.time() - t0, 1),
    })
    return res, model, kmeans_b, kmeans_r
