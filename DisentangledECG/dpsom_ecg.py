from tqdm import tqdm
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import os
import time
import random
import numpy.random as nprand
from typing import Tuple, List

# Must be set before CUDA context initialization for deterministic cuBLAS kernels.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# Disable torch.compile graph lowering for strict run-to-run reproducibility.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.optim as optim
import torch.nn as nn

from data_generator import ECG_DataGenerator
from decay_scheduler import ExponentialDecayScheduler
from dpsom_config import DPSOM_Config
from dpsom_ecg_model import DPSOM_ECG
from utils import cluster_purity, compute_disentanglement_metrics
from ECG_Dataset import ECG_Dataset
from visual_utils import log_som_visualizations

config = DPSOM_Config()


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    nprand.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False

    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True, warn_only=False)


def _np_to_torch(x_np, device):
    x = torch.from_numpy(x_np).float()
    if x.ndim == 3:
        x = x.permute(0, 2, 1).contiguous()  # [B, T, C] → [B, C, T]
    return x.to(device)


def _build_checkpoint(model):
    return {
        "model": model.state_dict(),
        "meta": {
            "latent_dim": int(model.latent_dim),
            "som_dim": tuple(int(v) for v in model.som_dim),
            "input_length": int(model.input_length),
            "input_channels": int(model.input_channels),
        },
    }


def _load_checkpoint_state(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt["model"], ckpt.get("meta", {})


def _run_pretraining_phase(
    model: nn.Module,
    generator: ECG_DataGenerator,
    optimizer: optim.Optimizer,
    step: int,
    device: torch.device,
    epochs: int,
    batch_size: int,
    som_dim: Tuple[int, int],
    learning_rate_pretrain: float,
    pbar: tqdm,
    num_batches: int,
    num_used_beats: int = 4
) -> int:
    train_gen = generator.get_record_batch_generator(mode="train", max_beats=num_used_beats, shuffle=True)
    val_gen = generator.get_record_batch_generator(mode="val", max_beats=num_used_beats, shuffle=False)

    print("\n[Phase 1] Autoencoder Pretraining...\n")

    model.train()
    dummy_som_targets = np.zeros((batch_size, som_dim[0] * som_dim[1]), dtype=np.float32)

    for epoch in range(epochs):
        for _ in range(num_batches):
            batch = next(train_gen)
            beats, beat_mask, beat_meta, beat_valid, _, record_age, record_sex = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

            B, N = beat_valid.shape
            model.set_p(torch.from_numpy(dummy_som_targets[:B * N]).to(device))

            for g in optimizer.param_groups:
                g["lr"] = float(learning_rate_pretrain)

            optimizer.zero_grad()
            loss_rec, rc, kl, pred_loss, loss_age, loss_sex = model.loss_reconstruction_batch(
                beats, beat_mask, beat_meta, beat_valid, record_age, record_sex
            )
            (loss_rec + pred_loss).backward()
            optimizer.step()

            if step % 100 == 0:
                val_batch = next(val_gen)
                v_beats, v_mask, v_meta, v_valid, _, v_age, v_sex = val_batch[0], val_batch[1], val_batch[2], val_batch[3], val_batch[4], val_batch[5], val_batch[6]

                with torch.no_grad():
                    elbo_v, rc_v, kl_v, pred_v, age_v, sex_v = model.loss_reconstruction_batch(
                        v_beats, v_mask, v_meta, v_valid, v_age, v_sex
                    )

                pbar.set_postfix(epoch=epoch, train_loss=loss_rec.item(), val_loss=elbo_v.item())

            step += 1
            pbar.update(1)

        val_metrics = _compute_val_cluster_age_metrics(model, generator, device, mode="val", num_used_beats=num_used_beats)

    return step


def _evaluate_latent_kmeans_after_pretraining(
    model: nn.Module,
    generator: ECG_DataGenerator,
    device: torch.device,
    mode: str = "val",
    num_used_beats: int = 4,
    random_state: int = 2025,
    max_samples: int = 50000,
) -> dict:
    was_training = model.training
    model.eval()

    latents_all, labels_all = [], []
    val_gen = generator.get_record_batch_generator(mode=mode, max_beats=num_used_beats, shuffle=False)
    num_records = len(generator.ecg_dataset.get_data(mode))
    num_batches = int(np.ceil(num_records / generator.batch_size))

    with torch.inference_mode():
        for _ in range(num_batches):
            batch = next(val_gen)
            beats, beat_valid, record_labels = batch[0], batch[3], batch[7]

            B, N = beat_valid.shape
            _, _, _, mu, _, _ = model.forward_batch(beats.to(device), beat_valid.to(device))

            mu_flat = mu.reshape(B * N, -1).cpu().numpy()
            valid_mask = (beat_valid.reshape(B * N) > 0.5).cpu().numpy()
            latents_all.append(mu_flat[valid_mask])
            labels_all.append(record_labels.unsqueeze(1).expand(B, N).reshape(B * N).cpu().numpy()[valid_mask])

    if was_training:
        model.train()

    z = np.concatenate(latents_all, axis=0)
    y = np.concatenate(labels_all, axis=0)

    if z.shape[0] > max_samples:
        idx = np.random.default_rng(random_state).choice(z.shape[0], size=max_samples, replace=False)
        z, y = z[idx], y[idx]

    n_clusters = int(np.unique(y).size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    pred = kmeans.fit_predict(z)

    nmi = metrics.normalized_mutual_info_score(y, pred, average_method="geometric")
    ami = metrics.adjusted_mutual_info_score(y, pred, average_method="geometric")
    purity = cluster_purity(y, pred)
    sil_sample_size = min(10000, z.shape[0])
    silhouette = metrics.silhouette_score(z, pred, sample_size=sil_sample_size, random_state=random_state)
    calinski_harabasz = metrics.calinski_harabasz_score(z, pred)
    davies_bouldin = metrics.davies_bouldin_score(z, pred)

    results = {
        "samples": int(z.shape[0]), "clusters": int(n_clusters),
        "NMI": float(nmi), "AMI": float(ami), "Purity": float(purity),
        "Silhouette": float(silhouette), "CalinskiHarabasz": float(calinski_harabasz),
        "DaviesBouldin": float(davies_bouldin), "Inertia": float(kmeans.inertia_),
    }
    print(
        f"\n[Phase 1] K-Means latent clustering (mode={mode}, samples={results['samples']}, k={results['clusters']}):\n"
        f"  NMI={nmi:.4f}, AMI={ami:.4f}, Purity={purity:.4f}, "
        f"Silhouette={silhouette:.4f}, CH={calinski_harabasz:.2f}, "
        f"DB={davies_bouldin:.4f}, Inertia={kmeans.inertia_:.2f}"
    )
    return results


def _run_age_probe_fit_phase(
    model: nn.Module,
    generator: ECG_DataGenerator,
    step: int,
    device: torch.device,
    epochs: int,
    lr: float,
    pbar: tqdm,
    num_batches: int,
    num_used_beats: int = 4,
) -> int:
    train_gen = generator.get_record_batch_generator(mode="train", max_beats=num_used_beats, shuffle=True)
    val_gen = generator.get_record_batch_generator(mode="val", max_beats=num_used_beats, shuffle=False)

    print("\n[Phase 1.5] Fitting Age Probe for SOM Residualization...\n")

    was_training = model.training
    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad = False
    for p in model.age_probe.parameters():
        p.requires_grad = True

    model.eval()
    model.age_probe.train()
    probe_optimizer = optim.Adam(model.age_probe.parameters(), lr=lr)

    for epoch in range(epochs):
        for _ in range(num_batches):
            batch = next(train_gen)
            beats, beat_valid, record_age = batch[0], batch[3], batch[5]

            B, N = beat_valid.shape
            valid_mask = beat_valid > 0.5
            beats_valid = beats[valid_mask]
            with torch.no_grad():
                mu_valid, _, _, _ = model._encode(beats_valid)

            age_targets = record_age.unsqueeze(1).expand(B, N)[valid_mask].float()
            probe_loss = torch.nn.functional.mse_loss(model.age_probe(mu_valid).squeeze(-1), age_targets)

            probe_optimizer.zero_grad()
            probe_loss.backward()
            probe_optimizer.step()

            if step % 100 == 0:
                with torch.no_grad():
                    val_batch = next(val_gen)
                    v_beats, v_valid, v_age = val_batch[0], val_batch[3], val_batch[5]
                    Bv, Nv = v_valid.shape
                    v_mask = v_valid > 0.5
                    v_mu, _, _, _ = model._encode(v_beats[v_mask])
                    probe_val_loss = torch.nn.functional.mse_loss(
                        model.age_probe(v_mu).squeeze(-1),
                        v_age.unsqueeze(1).expand(Bv, Nv)[v_mask].float()
                    )
                pbar.set_postfix(epoch=epoch, probe_train=probe_loss.item(), probe_val=probe_val_loss.item())

            step += 1
            pbar.update(1)

    model.freeze_age_probe()

    for name, p in model.named_parameters():
        p.requires_grad = False if name.startswith("age_probe.") else original_requires_grad[name]

    if was_training:
        model.train()
    else:
        model.eval()

    return step


def _run_sex_probe_fit_phase(
    model: nn.Module,
    generator: ECG_DataGenerator,
    step: int,
    device: torch.device,
    epochs: int,
    lr: float,
    pbar: tqdm,
    num_batches: int,
    num_used_beats: int = 4,
) -> int:
    train_gen = generator.get_record_batch_generator(mode="train", max_beats=num_used_beats, shuffle=True)
    val_gen = generator.get_record_batch_generator(mode="val", max_beats=num_used_beats, shuffle=False)

    print("\n[Phase 1.6] Fitting Sex Probe for SOM Residualization...\n")

    was_training = model.training
    original_requires_grad = {name: p.requires_grad for name, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad = False
    for p in model.sex_probe.parameters():
        p.requires_grad = True

    model.eval()
    model.sex_probe.train()
    probe_optimizer = optim.Adam(model.sex_probe.parameters(), lr=lr)

    for epoch in range(epochs):
        for _ in range(num_batches):
            batch = next(train_gen)
            beats, beat_valid, record_sex = batch[0], batch[3], batch[6]

            B, N = beat_valid.shape
            valid_mask = beat_valid > 0.5
            beats_valid = beats[valid_mask]
            with torch.no_grad():
                mu_valid, _, _, _ = model._encode(beats_valid)

            sex_targets = record_sex.unsqueeze(1).expand(B, N)[valid_mask].float()
            probe_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model.sex_probe(mu_valid).squeeze(-1), sex_targets
            )

            probe_optimizer.zero_grad()
            probe_loss.backward()
            probe_optimizer.step()

            if step % 100 == 0:
                with torch.no_grad():
                    val_batch = next(val_gen)
                    v_beats, v_valid, v_sex = val_batch[0], val_batch[3], val_batch[6]
                    Bv, Nv = v_valid.shape
                    v_mask = v_valid > 0.5
                    v_mu, _, _, _ = model._encode(v_beats[v_mask])
                    probe_val_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        model.sex_probe(v_mu).squeeze(-1),
                        v_sex.unsqueeze(1).expand(Bv, Nv)[v_mask].float()
                    )
                pbar.set_postfix(epoch=epoch, sex_probe_train=probe_loss.item(), sex_probe_val=probe_val_loss.item())

            step += 1
            pbar.update(1)

    model.freeze_sex_probe()

    for name, p in model.named_parameters():
        p.requires_grad = False if name.startswith("sex_probe.") else original_requires_grad[name]

    if was_training:
        model.train()
    else:
        model.eval()

    return step


def _run_som_initialization_phase(
    model: nn.Module,
    generator: ECG_DataGenerator,
    optimizer: optim.Optimizer,
    step: int,
    device: torch.device,
    som_dim: Tuple[int, int],
    batch_size: int,
    num_batches: int,
    pbar: tqdm,
    num_used_beats: int = 4
) -> int:
    train_gen = generator.get_record_batch_generator(mode="train", max_beats=num_used_beats, shuffle=True)
    val_gen = generator.get_record_batch_generator(mode="val", max_beats=num_used_beats, shuffle=False)

    print("\n[Phase 2] SOM Initialization...\n")

    dummy_som_targets = np.zeros((batch_size, som_dim[0] * som_dim[1]), dtype=np.float32)

    for lr_stage in [0.9, 0.3, 0.1, 0.01]:
        for epoch in range(2):
            for _ in range(num_batches):
                batch = next(train_gen)
                beats, beat_mask, beat_meta, beat_valid, _, record_age, record_sex = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5], batch[6]

                B, N = beat_valid.shape
                model.set_p(torch.from_numpy(dummy_som_targets[:B * N]).to(device))

                for g in optimizer.param_groups:
                    g["lr"] = float(lr_stage)

                optimizer.zero_grad()
                loss_init, loss_som_s, loss_commit_s = model.loss_a_batch(beats, beat_valid)
                loss_init.backward()
                optimizer.step()

                if step % 100 == 0:
                    val_batch = next(val_gen)
                    v_beats, v_mask, v_meta, v_valid, _, v_age, v_sex = val_batch[0], val_batch[1], val_batch[2], val_batch[3], val_batch[4], val_batch[5], val_batch[6]

                    with torch.no_grad():
                        val_loss_init, val_loss_som_s, val_loss_commit_s = model.loss_a_batch(v_beats, v_valid)
                        elbo_v, rc_v, kl_v, _, _, _ = model.loss_reconstruction_batch(v_beats, v_mask, v_meta, v_valid, v_age, v_sex)
                        elbo_t, rc_t, kl_t, _, _, _ = model.loss_reconstruction_batch(beats, beat_mask, beat_meta, beat_valid, record_age, record_sex)

                    pbar.set_postfix(epoch=epoch, init_loss=loss_init.item(), val_init_loss=val_loss_init.item())

                step += 1
                pbar.update(1)

            val_metrics = _compute_val_cluster_age_metrics(model, generator, device, mode="val", num_used_beats=num_used_beats)

    return step


def _compute_target_distribution(
    model: nn.Module,
    generator: ECG_DataGenerator,
    mode: str,
    device: torch.device,
    chunk_size: int = 5000
) -> np.ndarray:
    data, _, _, _, _, _, _, _ = generator.get_data(split=mode)
    model.eval()

    with torch.inference_mode():
        q_list = []
        for start in range(0, len(data), chunk_size):
            x_chunk = _np_to_torch(data[start:start + chunk_size], device)
            q_list.append(model.q_p(x_chunk).cpu().numpy())

    q = np.concatenate(q_list, axis=0)
    return model.target_distribution(q)


def _run_main_training_phase(
    model: nn.Module,
    generator: ECG_DataGenerator,
    optimizer: optim.Optimizer,
    scheduler: ExponentialDecayScheduler,
    step: int,
    device: torch.device,
    num_epochs: int,
    model_path: str,
    num_batches: int,
    pbar: tqdm,
    num_used_beats: int = 4
) -> int:
    train_gen = generator.get_record_batch_generator(mode="train", max_beats=num_used_beats, shuffle=True)
    val_gen = generator.get_record_batch_generator(mode="val", max_beats=num_used_beats, shuffle=False)

    print("\n[Phase 3] Joint Training (VAE + SOM)...\n")

    for epoch in range(num_epochs):
        p_train_all_t = torch.from_numpy(_compute_target_distribution(model, generator, "train", device)).to(device)
        p_val_all_t   = torch.from_numpy(_compute_target_distribution(model, generator, "val",   device)).to(device)

        model.train()
        for i in range(num_batches):
            batch = next(train_gen)
            beats, beat_mask, beat_meta, beat_valid, global_beat_idx, record_age, record_sex, rr = batch[:8]

            B, N = beat_valid.shape
            # Map global beat indices to pre-computed target distributions P
            valid_mask = global_beat_idx >= 0
            p_batch = torch.zeros(B, N, p_train_all_t.shape[1], device=device, dtype=p_train_all_t.dtype)
            p_batch[valid_mask] = p_train_all_t[global_beat_idx[valid_mask].long()]

            current_lr = scheduler.get_lr(step)
            optimizer.param_groups[0]["lr"] = current_lr
            optimizer.param_groups[1]["lr"] = current_lr / 5

            optimizer.zero_grad()
            losses = model.loss_batch(beats, beat_mask, beat_meta, beat_valid, record_age, record_sex, p_batch)
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            if step % 100 == 0:
                with torch.inference_mode():
                    z_train, _, _, _, _, _ = model.forward_batch(beats, beat_valid)
                    z_train_flat = z_train.reshape(B * N, -1)
                    v_idx_train = beat_valid.reshape(-1) > 0.5
                    _, t_som_s, t_commit_s = model.loss_a(z_train_flat[v_idx_train])

                    v_batch = next(val_gen)
                    vb, vm, vmt, vv, vi, va, vs, vrr = v_batch[:8]

                    Bv, Nv = vv.shape
                    valid_mask_v = vi > 0
                    p_batch_v = torch.zeros(Bv, Nv, p_val_all_t.shape[1], device=device, dtype=p_val_all_t.dtype)
                    p_batch_v[valid_mask_v] = p_val_all_t[vi[valid_mask_v].long()]

                    v_losses = model.loss_batch(vb, vm, vmt, vv, va, vs, p_batch_v)

                    z_val, _, _, _, _, _ = model.forward_batch(vb, vv)
                    z_val_flat = z_val.reshape(Bv * Nv, -1)
                    v_idx_val = vv.reshape(-1) > 0.5
                    _, v_som_s, v_commit_s = model.loss_a(z_val_flat[v_idx_val])

                pbar.set_postfix(
                    epoch=epoch, train_loss=losses[0].item(), test_loss=v_losses[0].item(),
                    ssom=v_losses[3].item(), cah=v_losses[2].item(), vae=v_losses[1].item(),
                    record_age=v_losses[11].item(), record_sex=v_losses[12].item(), refresh=False
                )

            step += 1
            pbar.update(1)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(_build_checkpoint(model), model_path)
        model.inc_epoch()

        val_metrics = _compute_val_cluster_age_metrics(model, generator, device, mode="val", num_used_beats=num_used_beats)

    return step


def _compute_val_cluster_age_metrics(
    model: nn.Module,
    generator: ECG_DataGenerator,
    device: torch.device,
    mode: str = "val",
    num_used_beats: int = 4,
) -> dict:
    was_training = model.training
    model.eval()

    labels_all, k_pred_all = [], []
    age_true_all, age_pred_all = [], []
    sex_true_all, sex_pred_all = [], []

    val_gen = generator.get_record_batch_generator(mode=mode, max_beats=num_used_beats, shuffle=False)
    num_records = len(generator.ecg_dataset.get_data(mode))
    num_batches = int(np.ceil(num_records / generator.batch_size))

    with torch.inference_mode():
        for _ in range(num_batches):
            batch = next(val_gen)
            beats, beat_mask, beat_meta, beat_valid, _, record_age, record_sex, record_labels = batch

            beats        = beats.to(device)
            beat_valid   = beat_valid.to(device)
            beat_meta    = beat_meta.to(device)
            record_age   = record_age.to(device)
            record_sex   = record_sex.to(device)
            record_labels = record_labels.to(device)

            B, N = beat_valid.shape

            _, _, _, mu, _, _ = model.forward_batch(beats, beat_valid)
            mu_flat = mu.reshape(B * N, -1)
            valid_mask = beat_valid.reshape(B * N) > 0.5

            k_pred_all.extend(model.k(mu_flat).cpu().numpy()[valid_mask.cpu().numpy()].tolist())
            labels_all.extend(
                record_labels.unsqueeze(1).expand(B, N).reshape(B * N).cpu().numpy()[valid_mask.cpu().numpy()].tolist()
            )

            age_record, sex_logits_record, _, _ = model.predict_dual_age(beats, beat_valid, beat_meta)
            age_true_all.extend(record_age.cpu().numpy().tolist())
            age_pred_all.extend(age_record.cpu().numpy().tolist())
            sex_true_all.extend(record_sex.cpu().numpy().tolist())
            sex_pred_all.extend(torch.sigmoid(sex_logits_record).reshape(-1).cpu().numpy().tolist())

    if was_training:
        model.train()

    labels_all   = np.array(labels_all)
    k_pred_all   = np.array(k_pred_all)
    age_true_all = np.array(age_true_all)
    age_pred_all = np.array(age_pred_all)
    sex_true_all = np.array(sex_true_all)
    sex_pred_all = np.array(sex_pred_all)

    nmi    = metrics.normalized_mutual_info_score(labels_all, k_pred_all, average_method="geometric")
    ami    = metrics.adjusted_mutual_info_score(labels_all, k_pred_all, average_method="geometric")
    purity = cluster_purity(labels_all, k_pred_all)
    mae_age = metrics.mean_absolute_error(age_true_all, age_pred_all)
    try:
        auc_sex = metrics.roc_auc_score(sex_true_all, sex_pred_all)
    except ValueError:
        auc_sex = 0.0

    return {
        "NMI": float(nmi), "AMI": float(ami), "Purity": float(purity),
        "MAE_Age": float(mae_age), "AUC_Sex": float(auc_sex),
    }


def train_model(
    model: nn.Module,
    generator: ECG_DataGenerator,
    num_epochs: int = config.num_epochs,
    batch_size: int = config.batch_size,
    ex_name: str = config.ex_name,
    model_path: str = config.modelpath,
    learning_rate: float = config.learning_rate,
    epochs_pretrain: int = config.num_epochs_pretrain,
    som_dim: Tuple[int, int] = config.som_dim,
    learning_rate_pretrain: float = config.learning_rate_pretrain,
    decay_factor: float = config.decay_factor,
    decay_steps: int = config.decay_steps,
    num_used_beats: int = config.num_beats
):
    device = _get_device()
    model = model.to(device)
    num_batches = generator.get_num_batches("train")

    age_probe_fit_epochs = 3
    age_probe_fit_lr = 1e-3
    sex_probe_fit_epochs = 3
    sex_probe_fit_lr = 1e-3

    optimizer_vae = optim.Adam(model.parameters(), lr=learning_rate_pretrain)
    optimizer_som = optim.Adam(model.parameters(), lr=0.9)

    meta_keywords = ['age', 'sex']
    meta_params = [p for name, p in model.named_parameters() if any(k in name for k in meta_keywords)]
    base_params  = [p for name, p in model.named_parameters() if not any(k in name for k in meta_keywords)]
    optimizer_joint = optim.Adam([
        {'params': base_params,  'lr': learning_rate},
        {'params': meta_params,  'lr': learning_rate / 5},
    ])

    scheduler = ExponentialDecayScheduler(
        initial_lr=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_factor,
        staircase=True
    )

    print(f"\n[Training Session] Experiment: {ex_name}")
    print(f"[Training Session] Config: Latent={model.latent_dim}, SOM={som_dim[0]}x{som_dim[1]}")

    # 8 = 4 lr stages × 2 epochs from SOM init
    total_epochs = num_epochs + epochs_pretrain + age_probe_fit_epochs + sex_probe_fit_epochs + 8
    pbar = tqdm(total=total_epochs * num_batches, desc="Total Training")
    step = 0

    step = _run_pretraining_phase(
        model, generator, optimizer_vae, step, device,
        epochs_pretrain, batch_size, som_dim, learning_rate_pretrain, pbar, num_batches, num_used_beats
    )

    _evaluate_latent_kmeans_after_pretraining(
        model=model, generator=generator, device=device,
        mode="val", num_used_beats=num_used_beats, random_state=config.random_seed,
    )

    step = _run_age_probe_fit_phase(
        model, generator, step, device,
        age_probe_fit_epochs, age_probe_fit_lr, pbar, num_batches, num_used_beats
    )

    step = _run_sex_probe_fit_phase(
        model, generator, step, device,
        sex_probe_fit_epochs, sex_probe_fit_lr, pbar, num_batches, num_used_beats
    )

    step = _run_som_initialization_phase(
        model, generator, optimizer_som, step,
        device, som_dim, batch_size, num_batches, pbar, num_used_beats
    )

    _run_main_training_phase(
        model, generator, optimizer_joint, scheduler, step,
        device, num_epochs, model_path, num_batches, pbar, num_used_beats
    )

    pbar.close()


def evaluate_model(model, generator, record_filter: list = None):
    """
    Evaluate on test set.
    record_filter: optional boolean list [num_test_records], True = include.
                   None means evaluate on all records.
    """
    device = _get_device()
    model = model.to(device)
    model.eval()

    labels_test_all, k_pred_all = [], []
    age_true_all, age_pred_record_all = [], []
    sex_true_all, sex_pred_all = [], []
    z_all, z_age_all, z_sex_all = [], [], []
    age_per_beat_all, sex_per_beat_all, record_id_per_beat_all = [], [], []

    print("\n\nEvaluation...\n")

    test_records = generator.ecg_dataset.get_data("test")
    num_batches = int(np.ceil(len(test_records) / generator.batch_size))
    test_gen = generator.get_record_batch_generator(mode="test", shuffle=False)

    with torch.no_grad():
        for batch_idx in range(num_batches):
            beats, beat_mask, beat_meta, beat_valid, global_beat_idx, age_target, sex_target, record_labels = next(test_gen)

            start_r = batch_idx * generator.batch_size
            end_r   = min(start_r + generator.batch_size, len(test_records))
            batch_record_ids = np.arange(start_r, end_r, dtype=np.int64)

            if record_filter is not None:
                keep = torch.tensor(record_filter[start_r:end_r], dtype=torch.bool)
                if not keep.any():
                    continue
                beats            = beats[keep]
                beat_mask        = beat_mask[keep]
                beat_meta        = beat_meta[keep]
                beat_valid       = beat_valid[keep]
                age_target       = age_target[keep]
                sex_target       = sex_target[keep]
                record_labels    = record_labels[keep]
                batch_record_ids = batch_record_ids[keep.numpy()]

            beats        = beats.to(device)
            beat_valid   = beat_valid.to(device)
            beat_meta    = beat_meta.to(device)
            age_target   = age_target.to(device)
            sex_target   = sex_target.to(device)
            record_labels = record_labels.to(device)

            B, N, C, T = beats.shape

            age_record, sex_logits_record, attn_age, attn_sex = model.predict_dual_age(beats, beat_valid, beat_meta)
            age_true_all.extend(age_target.cpu().numpy().tolist())
            age_pred_record_all.extend(age_record.cpu().numpy().tolist())
            sex_pred_all.extend(torch.sigmoid(sex_logits_record).cpu().numpy().reshape(-1).tolist())
            sex_true_all.extend(sex_target.cpu().numpy().tolist())

            z, z_age, z_sex, mu, logvar, _ = model.forward_batch(beats, beat_valid)
            mu_flat = mu.reshape(B * N, -1)
            beat_valid_flat = beat_valid.reshape(B * N).cpu().numpy()
            valid_mask = beat_valid_flat > 0.5

            k_pred_all.extend(model.k(mu_flat).cpu().numpy()[valid_mask].tolist())
            labels_all_exp = record_labels.unsqueeze(1).expand(B, N).reshape(B * N).cpu().numpy()
            labels_test_all.extend(labels_all_exp[valid_mask].tolist())

            z_flat    = z.reshape(B * N, -1).cpu().numpy()
            z_age_flat = z_age.reshape(B * N, -1).cpu().numpy()
            z_sex_flat = z_sex.reshape(B * N, -1).cpu().numpy()
            z_all.append(z_flat[valid_mask])
            z_age_all.append(z_age_flat[valid_mask])
            z_sex_all.append(z_sex_flat[valid_mask])

            age_expanded = age_target.unsqueeze(1).expand(B, N).reshape(B * N).cpu().numpy()
            sex_expanded = sex_target.unsqueeze(1).expand(B, N).reshape(B * N).cpu().numpy()
            age_per_beat_all.append(age_expanded[valid_mask])
            sex_per_beat_all.append(sex_expanded[valid_mask])
            record_id_per_beat_all.append(np.repeat(batch_record_ids, N)[valid_mask])

    labels_test_all      = np.array(labels_test_all)
    k_pred_all           = np.array(k_pred_all)
    age_true_all         = np.array(age_true_all)
    age_pred_record_all  = np.array(age_pred_record_all)
    sex_true_all         = np.array(sex_true_all)
    sex_pred_all         = np.array(sex_pred_all)
    z_all                = np.concatenate(z_all, axis=0)
    z_age_all            = np.concatenate(z_age_all, axis=0)
    z_sex_all            = np.concatenate(z_sex_all, axis=0)
    age_per_beat         = np.concatenate(age_per_beat_all, axis=0)
    sex_per_beat         = np.concatenate(sex_per_beat_all, axis=0)
    record_id_per_beat   = np.concatenate(record_id_per_beat_all, axis=0)

    test_nmi   = metrics.normalized_mutual_info_score(labels_test_all, k_pred_all, average_method="geometric")
    test_ami   = metrics.adjusted_mutual_info_score(labels_test_all, k_pred_all, average_method="geometric")
    test_purity = cluster_purity(labels_test_all, k_pred_all)
    mae_record = metrics.mean_absolute_error(age_true_all, age_pred_record_all)
    try:
        auc_sex = metrics.roc_auc_score(sex_true_all, sex_pred_all)
    except ValueError:
        auc_sex = 0.0

    print("\n\nComputing Disentanglement Metrics (MIG, SAP, DCI)...\n")
    disentanglement_results = compute_disentanglement_metrics(
        z_all, z_age_all, z_sex_all, age_per_beat, sex_per_beat, groups=record_id_per_beat
    )

    if disentanglement_results:
        print("\nDisentanglement Interpretation:")
        for rep_key in ("z_main", "z_age", "z_sex"):
            rep_block = disentanglement_results.get(rep_key, {})
            interp = rep_block.get("interpretation", {})
            if not interp:
                continue
            print(f"\n{rep_key} interpretation:")
            for key in ("MIG", "SAP", "DCI", "Informativeness"):
                if key in interp:
                    print(f"  {key}: {interp[key]}")

    return {
        "NMI": float(test_nmi),
        "AMI": float(test_ami),
        "Purity": float(test_purity),
        "MAE_Age": float(mae_record),
        "AUC_Sex": float(auc_sex),
        "Disentanglement": disentanglement_results,
    }


def main(config=config, model_path=None):
    fs = 100

    _set_global_determinism(config.random_seed)

    start = time.time()
    os.makedirs("./models", exist_ok=True)

    ecg_dataset_path = f"./data/ptbxl_{fs}_T-12ms.pkl"
    if config.use_data_cache:
        print(f"Loading ECG_Dataset from {ecg_dataset_path} ...")
        ds: ECG_Dataset = ECG_Dataset.load(ecg_dataset_path)
    else:
        ds = ECG_Dataset(fs)
        ds.import_ptbxl(base_path="/nfs/data8/schlegel/git/ecg-cbm/data/ptb-xl")
        ds.save(ecg_dataset_path)

    data_generator = ECG_DataGenerator(ds)
    sample_rec = data_generator.get_all_beats_representation(split="train")
    input_length   = sample_rec[0].shape[1]  # T
    input_channels = sample_rec[0].shape[2]  # C = 12

    if model_path is not None:
        device = _get_device()
        state_dict, meta = _load_checkpoint_state(model_path, device)
        latent_dim     = int(meta.get("latent_dim", config.latent_dim))
        _sd = meta.get("som_dim", config.som_dim)
        som_dim: Tuple[int, int] = (int(_sd[0]), int(_sd[1]))
        input_length   = int(meta.get("input_length", input_length))
        input_channels = int(meta.get("input_channels", input_channels))
        model = DPSOM_ECG(
            latent_dim=latent_dim, som_dim=som_dim,
            input_length=input_length, input_channels=input_channels,
            alpha=config.alpha, beta=config.beta, theta=config.theta, gamma=config.gamma,
            tau=config.tau, eta=config.eta, delta_age=config.delta_age, delta_sex=config.delta_sex,
            dropout=config.dropout, prior_var=config.prior_var, prior=config.prior,
        )
        model.load_state_dict(state_dict, strict=True)
        eval_ckpt_path = model_path
    else:
        model = DPSOM_ECG(
            latent_dim=config.latent_dim, som_dim=config.som_dim,
            input_length=input_length, input_channels=input_channels,
            alpha=config.alpha, beta=config.beta, theta=config.theta, gamma=config.gamma,
            tau=config.tau, eta=config.eta, delta_age=config.delta_age, delta_sex=config.delta_sex,
            dropout=config.dropout, prior_var=config.prior_var, prior=config.prior,
        )
        train_model(model, data_generator)
        eval_ckpt_path = config.modelpath

    results = evaluate_model(model, data_generator)
    print(f"NMI: {results['NMI']:.4f}, AMI: {results['AMI']:.4f}, PUR: {results['Purity']:.4f}, "
          f"MAE Age: {results['MAE_Age']:.4f}, AUC Sex: {results['AUC_Sex']:.4f}.")

    if "Disentanglement" in results:
        dis = results["Disentanglement"]
        print("\n=== Disentanglement Metrics ===")

        print("\nMain representation z (should NOT encode age/sex):")
        print(f"  MIG_age: {dis['z_main']['MIG']['age']['mig_norm']:.4f}, MIG_sex: {dis['z_main']['MIG']['sex']['mig_norm']:.4f}")
        print(f"  SAP_age: {dis['z_main']['SAP']['SAP_age']:.4f}, SAP_sex: {dis['z_main']['SAP']['SAP_sex']:.4f}")
        print(f"  DCI: disentanglement={dis['z_main']['DCI']['DCI_disentanglement']:.4f}, completeness={dis['z_main']['DCI']['DCI_completeness']:.4f}, informativeness={dis['z_main']['DCI']['DCI_informativeness']:.4f}")

        print("\nz_age representation (should encode age, NOT sex):")
        print(f"  MIG_age: {dis['z_age']['MIG']['age']['mig_norm']:.4f}, MIG_sex: {dis['z_age']['MIG']['sex']['mig_norm']:.4f}")
        print(f"  SAP_age: {dis['z_age']['SAP']['SAP_age']:.4f}, SAP_sex: {dis['z_age']['SAP']['SAP_sex']:.4f}")
        print(f"  DCI: disentanglement={dis['z_age']['DCI']['DCI_disentanglement']:.4f}, completeness={dis['z_age']['DCI']['DCI_completeness']:.4f}, informativeness={dis['z_age']['DCI']['DCI_informativeness']:.4f}")


        print("\nz_sex representation (should encode sex, NOT age):")
        print(f"  MIG_age: {dis['z_sex']['MIG']['age']['mig_norm']:.4f}, MIG_sex: {dis['z_sex']['MIG']['sex']['mig_norm']:.4f}")
        print(f"  SAP_age: {dis['z_sex']['SAP']['SAP_age']:.4f}, SAP_sex: {dis['z_sex']['SAP']['SAP_sex']:.4f}")
        print(f"  DCI: disentanglement={dis['z_sex']['DCI']['DCI_disentanglement']:.4f}, completeness={dis['z_sex']['DCI']['DCI_completeness']:.4f}, informativeness={dis['z_sex']['DCI']['DCI_informativeness']:.4f}")


    # Evaluate on single-label records only
    single_label_filter = [len(r.extra_labels) == 0 for r in ds.get_data("test")]
    results_r = evaluate_model(model, data_generator, record_filter=single_label_filter)
    print(f"NMI_r: {results_r['NMI']:.4f}, AMI_r: {results_r['AMI']:.4f}, PUR_r: {results_r['Purity']:.4f}, "
          f"MAE Age_r: {results_r['MAE_Age']:.4f}, AUC Sex_r: {results_r['AUC_Sex']:.4f}.")

    log_som_visualizations(ds, data_generator, config.ex_name, eval_ckpt_path, _get_device(), single_label_only=True)

    elapsed_time_fl = time.time() - start
    print(f"Elapsed Time: {elapsed_time_fl}")


    return results


if __name__ == "__main__":
    main()
