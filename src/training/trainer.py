"""Training and evaluation routines for DPSOM-ECG."""
from tqdm import tqdm
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import os
import time
import random
import numpy.random as nprand
from typing import Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import torch
import torch.optim as optim
import torch.nn as nn

from src.config import DPSOM_Config
from src.scheduler import ExponentialDecayScheduler
from src.model.dpsom import DPSOM_ECG
from src.data.dataset import ECG_Dataset
from src.data.generator import ECG_DataGenerator
from src.utils.metrics import cluster_purity, compute_disentanglement_metrics
from src.utils.visualization import log_som_visualizations

config = DPSOM_Config()


# ---------------------------------------------------------------------------
# Device / determinism helpers
# ---------------------------------------------------------------------------

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_global_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    nprand.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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
        x = x.permute(0, 2, 1).contiguous()
    return x.to(device)


def _build_checkpoint(model):
    return {
        "model": model.state_dict(),
        "meta": {
            "latent_dim":               int(model.latent_dim),
            "som_dim":                  tuple(int(v) for v in model.som_dim),
            "input_length":             int(model.input_length),
            "input_channels":           int(model.input_channels),
            "encoder_base_channels_1":  int(getattr(model, "_encoder_base_channels_1", 32)),
            "encoder_base_channels_2":  int(getattr(model, "_encoder_base_channels_2", 64)),
            "encoder_kernel_size":      int(getattr(model, "_encoder_kernel_size", 7)),
            "encoder_fc_hidden_dim":    int(getattr(model, "_encoder_fc_hidden_dim", 512)),
            "z_age_dim_factor":         float(getattr(model, "_z_age_dim_factor", 0.25)),
            "z_sex_dim_factor":         float(getattr(model, "_z_sex_dim_factor", 0.25)),
        },
    }


_LEGACY_KEY_MAP = [
    # encoder / encoder_age / encoder_sex
    ("enc_conv1",  "conv1"),
    ("enc_bn1",    "bn1"),
    ("enc_se1",    "se1"),
    ("enc_conv2",  "conv2"),
    ("enc_bn2",    "bn2"),
    ("enc_se2",    "se2"),
    # decoder
    ("dec_conv1",  "conv1"),
    ("dec_bn1",    "bn1"),
    ("dec_se1",    "se1"),
    ("dec_film1",  "film1"),
    ("dec_conv2",  "conv2"),
    ("dec_bn2",    "bn2"),
    ("dec_se2",    "se2"),
    ("dec_film2",  "film2"),
    ("dec_conv_out", "conv_out"),
]

def _remap_legacy_keys(state_dict: dict) -> dict:
    """Rename old ``enc_*`` / ``dec_*`` layer names to the current convention."""
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in _LEGACY_KEY_MAP:
            # Replace only the last segment of each dotted name to avoid
            # false positives (e.g. "encoder" containing "enc").
            parts = new_k.split(".")
            parts = [new if p == old else p for p in parts]
            new_k = ".".join(parts)
        new_sd[new_k] = v
    return new_sd


def _load_checkpoint_state(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = _remap_legacy_keys(state)
    meta  = ckpt.get("meta", {}) if isinstance(ckpt, dict) else {}
    return state, meta


# ---------------------------------------------------------------------------
# Training phases
# ---------------------------------------------------------------------------

def _run_pretraining_phase(
    model, generator, optimizer, step, device,
    epochs, batch_size, som_dim, lr_pretrain, pbar, num_batches, num_used_beats=4,
) -> int:
    train_gen = generator.get_record_batch_generator(mode="train", max_beats=num_used_beats, shuffle=True)
    val_gen   = generator.get_record_batch_generator(mode="val",   max_beats=num_used_beats, shuffle=False)
    print("\n[Phase 1] Autoencoder Pretraining...\n")
    model.train()
    dummy_p = np.zeros((batch_size, som_dim[0] * som_dim[1]), dtype=np.float32)

    for epoch in range(epochs):
        for _ in range(num_batches):
            batch = next(train_gen)
            beats, beat_mask, beat_meta, beat_valid, _, record_age, record_sex = batch[:7]
            B, N = beat_valid.shape
            model.set_p(torch.from_numpy(dummy_p[:B * N]).to(device))
            for g in optimizer.param_groups:
                g["lr"] = float(lr_pretrain)
            optimizer.zero_grad()
            loss_rec, rc, kl, pred_loss, _, _ = model.loss_reconstruction_batch(
                beats, beat_mask, beat_meta, beat_valid, record_age, record_sex)
            (loss_rec + pred_loss).backward()
            optimizer.step()

            if step % 100 == 0:
                vb = next(val_gen)
                with torch.no_grad():
                    elbo_v, *_ = model.loss_reconstruction_batch(vb[0], vb[1], vb[2], vb[3], vb[5], vb[6])
                pbar.set_postfix(epoch=epoch, train=loss_rec.item(), val=elbo_v.item())

            step += 1
            pbar.update(1)

        _compute_val_cluster_age_metrics(model, generator, device, mode="val", num_used_beats=num_used_beats)

    return step


def _run_probe_fit_phase(model, generator, step, device, epochs, lr, pbar, num_batches,
                         num_used_beats, target, label):
    train_gen = generator.get_record_batch_generator(mode="train", max_beats=num_used_beats, shuffle=True)
    val_gen   = generator.get_record_batch_generator(mode="val",   max_beats=num_used_beats, shuffle=False)
    print(f"\n[Phase 1.{5 if target == 'age' else 6}] Fitting {target.capitalize()} Probe...\n")

    was_training = model.training
    saved_grad = {n: p.requires_grad for n, p in model.named_parameters()}
    for p in model.parameters():
        p.requires_grad = False

    probe = model.age_probe if target == "age" else model.sex_probe
    for p in probe.parameters():
        p.requires_grad = True

    model.eval()
    probe.train()
    opt = optim.Adam(probe.parameters(), lr=lr)

    for epoch in range(epochs):
        for _ in range(num_batches):
            batch = next(train_gen)
            beats, beat_valid = batch[0], batch[3]
            demo = batch[5] if target == "age" else batch[6]
            B, N = beat_valid.shape
            valid_mask = beat_valid > 0.5
            with torch.no_grad():
                mu_v, _, _, _ = model._encode(beats[valid_mask])
            tgt = demo.unsqueeze(1).expand(B, N)[valid_mask].float()
            if target == "age":
                loss = nn.functional.mse_loss(probe(mu_v).squeeze(-1), tgt)
            else:
                loss = nn.functional.binary_cross_entropy_with_logits(probe(mu_v).squeeze(-1), tgt)
            opt.zero_grad(); loss.backward(); opt.step()

            if step % 100 == 0:
                pbar.set_postfix(epoch=epoch, **{f"{target}_probe_train": loss.item()})
            step += 1
            pbar.update(1)

    if target == "age":
        model.freeze_age_probe()
    else:
        model.freeze_sex_probe()

    for name, p in model.named_parameters():
        p.requires_grad = False if name.startswith(f"{target}_probe.") else saved_grad[name]
    if was_training:
        model.train()
    return step


def _run_som_initialization_phase(
    model, generator, optimizer, step, device, som_dim,
    batch_size, num_batches, pbar, num_used_beats=4,
) -> int:
    train_gen = generator.get_record_batch_generator(mode="train", max_beats=num_used_beats, shuffle=True)
    val_gen   = generator.get_record_batch_generator(mode="val",   max_beats=num_used_beats, shuffle=False)
    print("\n[Phase 2] SOM Initialization...\n")
    dummy_p = np.zeros((batch_size, som_dim[0] * som_dim[1]), dtype=np.float32)

    for lr_stage in [0.9, 0.3, 0.1, 0.01]:
        for epoch in range(2):
            for _ in range(num_batches):
                batch = next(train_gen)
                beats, beat_mask, beat_meta, beat_valid = batch[0], batch[1], batch[2], batch[3]
                record_age, record_sex = batch[5], batch[6]
                B, N = beat_valid.shape
                model.set_p(torch.from_numpy(dummy_p[:B * N]).to(device))
                for g in optimizer.param_groups:
                    g["lr"] = float(lr_stage)
                optimizer.zero_grad()
                loss_init, _, _ = model.loss_a_batch(beats, beat_valid)
                loss_init.backward()
                optimizer.step()

                if step % 100 == 0:
                    vb = next(val_gen)
                    with torch.no_grad():
                        val_loss, _, _ = model.loss_a_batch(vb[0], vb[3])
                    pbar.set_postfix(epoch=epoch, init=loss_init.item(), val_init=val_loss.item())

                step += 1
                pbar.update(1)

            _compute_val_cluster_age_metrics(model, generator, device, mode="val", num_used_beats=num_used_beats)

    return step


def _compute_target_distribution(model, generator, mode, device, chunk_size=5000):
    data, _, _, _, _, _, _, _ = generator.get_data(split=mode)
    model.eval()
    q_list = []
    with torch.inference_mode():
        for start in range(0, len(data), chunk_size):
            x = _np_to_torch(data[start:start + chunk_size], device)
            q_list.append(model.q_p(x).cpu().numpy())
    return model.target_distribution(np.concatenate(q_list, axis=0))


def _run_main_training_phase(
    model, generator, optimizer, scheduler, step,
    device, num_epochs, model_path, num_batches, pbar, num_used_beats=4,
    gradient_clip_norm: float = 5.0,
    lr_meta_factor: float = 5.0,
) -> int:
    train_gen = generator.get_record_batch_generator(mode="train", max_beats=num_used_beats, shuffle=True)
    val_gen   = generator.get_record_batch_generator(mode="val",   max_beats=num_used_beats, shuffle=False)
    print("\n[Phase 3] Joint Training (VAE + SOM)...\n")

    for epoch in range(num_epochs):
        p_train = torch.from_numpy(_compute_target_distribution(model, generator, "train", device)).to(device)
        p_val   = torch.from_numpy(_compute_target_distribution(model, generator, "val",   device)).to(device)

        model.train()
        for _ in range(num_batches):
            batch = next(train_gen)
            beats, beat_mask, beat_meta, beat_valid, global_beat_idx, record_age, record_sex = batch[:7]
            B, N = beat_valid.shape

            valid_mask = global_beat_idx >= 0
            p_batch = torch.zeros(B, N, p_train.shape[1], device=device, dtype=p_train.dtype)
            p_batch[valid_mask] = p_train[global_beat_idx[valid_mask].long()]

            current_lr = scheduler.get_lr(step)
            optimizer.param_groups[0]["lr"] = current_lr
            optimizer.param_groups[1]["lr"] = current_lr / lr_meta_factor

            optimizer.zero_grad()
            losses = model.loss_batch(beats, beat_mask, beat_meta, beat_valid, record_age, record_sex, p_batch)
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            optimizer.step()

            if step % 100 == 0:
                vb = next(val_gen)
                Bv, Nv = vb[3].shape
                vm = vb[4] >= 0
                p_batch_v = torch.zeros(Bv, Nv, p_val.shape[1], device=device, dtype=p_val.dtype)
                p_batch_v[vm] = p_val[vb[4][vm].long()]
                with torch.inference_mode():
                    v_losses = model.loss_batch(vb[0], vb[1], vb[2], vb[3], vb[5], vb[6], p_batch_v)
                pbar.set_postfix(
                    epoch=epoch, train=losses[0].item(), val=v_losses[0].item(),
                    vae=v_losses[1].item(), commit=v_losses[2].item(), som=v_losses[3].item(),
                )

            step += 1
            pbar.update(1)

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(_build_checkpoint(model), model_path)
        model.inc_epoch()
        _compute_val_cluster_age_metrics(model, generator, device, mode="val", num_used_beats=num_used_beats)

    return step


# ---------------------------------------------------------------------------
# Validation metrics helper
# ---------------------------------------------------------------------------

def _compute_val_cluster_age_metrics(model, generator, device, mode="val", num_used_beats=4):
    was_training = model.training
    model.eval()

    labels_all, k_pred_all = [], []
    age_true_all, age_pred_all = [], []
    sex_true_all, sex_pred_all = [], []

    val_gen   = generator.get_record_batch_generator(mode=mode, max_beats=num_used_beats, shuffle=False)
    num_records = len(generator.ecg_dataset.get_data(mode))
    num_batches = int(np.ceil(num_records / generator.batch_size))

    with torch.inference_mode():
        for _ in range(num_batches):
            batch = next(val_gen)
            beats, beat_mask, beat_meta, beat_valid, _, record_age, record_sex, record_labels = batch

            B, N = beat_valid.shape
            _, _, _, mu, _, _ = model.forward_batch(beats, beat_valid)
            mu_flat    = mu.reshape(B * N, -1)
            valid_mask = beat_valid.reshape(B * N) > 0.5

            k_pred_all.extend(model.k(mu_flat).cpu().numpy()[valid_mask.cpu().numpy()].tolist())
            labels_all.extend(
                record_labels.unsqueeze(1).expand(B, N).reshape(B * N)
                .cpu().numpy()[valid_mask.cpu().numpy()].tolist()
            )

            age_r, sex_logits_r, _, _ = model.predict_dual_age(beats, beat_valid, beat_meta)
            age_true_all.extend(record_age.cpu().numpy().tolist())
            age_pred_all.extend(age_r.cpu().numpy().tolist())
            sex_true_all.extend(record_sex.cpu().numpy().tolist())
            sex_pred_all.extend(torch.sigmoid(sex_logits_r).reshape(-1).cpu().numpy().tolist())

    if was_training:
        model.train()

    nmi    = metrics.normalized_mutual_info_score(labels_all, k_pred_all, average_method="geometric")
    ami    = metrics.adjusted_mutual_info_score(labels_all, k_pred_all, average_method="geometric")
    purity = cluster_purity(np.array(labels_all), np.array(k_pred_all))
    mae    = metrics.mean_absolute_error(age_true_all, age_pred_all)
    try:
        auc = metrics.roc_auc_score(sex_true_all, sex_pred_all)
    except ValueError:
        auc = 0.0

    return {"NMI": float(nmi), "AMI": float(ami), "Purity": float(purity),
            "MAE_Age": float(mae), "AUC_Sex": float(auc)}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    num_used_beats: int = config.num_beats,
    weight_decay: float = config.weight_decay,
    gradient_clip_norm: float = config.gradient_clip_norm,
    num_epochs_probe: int = config.num_epochs_probe,
    lr_probe: float = config.lr_probe,
    lr_meta_factor: float = config.lr_meta_factor,
):
    device = _get_device()
    model  = model.to(device)
    num_batches = generator.get_num_batches("train")

    optimizer_vae   = optim.Adam(model.parameters(), lr=learning_rate_pretrain,
                                 weight_decay=weight_decay)
    optimizer_som   = optim.Adam(model.parameters(), lr=0.9)

    meta_kw   = ["age", "sex"]
    meta_p    = [p for n, p in model.named_parameters() if any(k in n for k in meta_kw)]
    base_p    = [p for n, p in model.named_parameters() if not any(k in n for k in meta_kw)]
    optimizer_joint = optim.Adam([
        {"params": base_p, "lr": learning_rate},
        {"params": meta_p, "lr": learning_rate / lr_meta_factor},
    ], weight_decay=weight_decay)

    scheduler = ExponentialDecayScheduler(learning_rate, decay_steps, decay_factor)

    print(f"\n[Training] {ex_name}  latent={model.latent_dim}  SOM={som_dim[0]}x{som_dim[1]}")

    extra_epochs = num_epochs_probe * 2 + 8
    total_epochs = num_epochs + epochs_pretrain + extra_epochs
    pbar = tqdm(total=total_epochs * num_batches, desc="Training")
    step = 0

    step = _run_pretraining_phase(
        model, generator, optimizer_vae, step, device,
        epochs_pretrain, batch_size, som_dim, learning_rate_pretrain, pbar, num_batches, num_used_beats,
    )

    step = _run_probe_fit_phase(
        model, generator, step, device, num_epochs_probe, lr_probe, pbar, num_batches, num_used_beats,
        target="age", label="age",
    )

    step = _run_probe_fit_phase(
        model, generator, step, device, num_epochs_probe, lr_probe, pbar, num_batches, num_used_beats,
        target="sex", label="sex",
    )

    step = _run_som_initialization_phase(
        model, generator, optimizer_som, step,
        device, som_dim, batch_size, num_batches, pbar, num_used_beats,
    )

    _run_main_training_phase(
        model, generator, optimizer_joint, scheduler, step,
        device, num_epochs, model_path, num_batches, pbar, num_used_beats,
        gradient_clip_norm=gradient_clip_norm,
        lr_meta_factor=lr_meta_factor,
    )

    pbar.close()


def evaluate_model(model, generator, record_filter=None):
    """Evaluate on the test set and return clustering + demographic metrics."""
    device = _get_device()
    model  = model.to(device)
    model.eval()

    labels_all, k_pred_all = [], []
    age_true_all, age_pred_all = [], []
    sex_true_all, sex_pred_all = [], []
    z_all, z_age_all, z_sex_all = [], [], []
    age_pb, sex_pb, rec_id_pb = [], [], []

    test_records = generator.ecg_dataset.get_data("test")
    nb           = int(np.ceil(len(test_records) / generator.batch_size))
    test_gen     = generator.get_record_batch_generator(mode="test", shuffle=False)

    with torch.no_grad():
        for idx in range(nb):
            batch = next(test_gen)
            beats, beat_mask, beat_meta, beat_valid, _, age_t, sex_t, rec_labels = batch

            s = idx * generator.batch_size
            e = min(s + generator.batch_size, len(test_records))
            rid = np.arange(s, e, dtype=np.int64)

            if record_filter is not None:
                keep = torch.tensor(record_filter[s:e], dtype=torch.bool)
                if not keep.any():
                    continue
                beats, beat_mask, beat_meta, beat_valid = beats[keep], beat_mask[keep], beat_meta[keep], beat_valid[keep]
                age_t, sex_t, rec_labels = age_t[keep], sex_t[keep], rec_labels[keep]
                rid = rid[keep.numpy()]

            B, N, C, T = beats.shape
            age_r, sex_logits_r, _, _ = model.predict_dual_age(beats, beat_valid, beat_meta)
            age_true_all.extend(age_t.cpu().numpy().tolist())
            age_pred_all.extend(age_r.cpu().numpy().tolist())
            sex_true_all.extend(sex_t.cpu().numpy().tolist())
            sex_pred_all.extend(torch.sigmoid(sex_logits_r).cpu().numpy().reshape(-1).tolist())

            z, z_age, z_sex, mu, _, _ = model.forward_batch(beats, beat_valid)
            mu_flat      = mu.reshape(B * N, -1)
            valid_mask   = beat_valid.reshape(B * N).cpu().numpy() > 0.5

            k_pred_all.extend(model.k(mu_flat).cpu().numpy()[valid_mask].tolist())
            labels_all.extend(
                rec_labels.unsqueeze(1).expand(B, N).reshape(B * N).cpu().numpy()[valid_mask].tolist()
            )

            z_all.append(z.reshape(B * N, -1).cpu().numpy()[valid_mask])
            z_age_all.append(z_age.reshape(B * N, -1).cpu().numpy()[valid_mask])
            z_sex_all.append(z_sex.reshape(B * N, -1).cpu().numpy()[valid_mask])
            age_pb.append(age_t.unsqueeze(1).expand(B, N).reshape(B * N).cpu().numpy()[valid_mask])
            sex_pb.append(sex_t.unsqueeze(1).expand(B, N).reshape(B * N).cpu().numpy()[valid_mask])
            rec_id_pb.append(np.repeat(rid, N)[valid_mask])

    labels_all  = np.array(labels_all)
    k_pred_all  = np.array(k_pred_all)
    z_all       = np.concatenate(z_all)
    z_age_all   = np.concatenate(z_age_all)
    z_sex_all   = np.concatenate(z_sex_all)
    age_pb      = np.concatenate(age_pb)
    sex_pb      = np.concatenate(sex_pb)
    rec_id_pb   = np.concatenate(rec_id_pb)

    nmi    = metrics.normalized_mutual_info_score(labels_all, k_pred_all, average_method="geometric")
    ami    = metrics.adjusted_mutual_info_score(labels_all, k_pred_all, average_method="geometric")
    purity = cluster_purity(labels_all, k_pred_all)
    mae    = metrics.mean_absolute_error(age_true_all, age_pred_all)
    try:
        auc = metrics.roc_auc_score(sex_true_all, sex_pred_all)
    except ValueError:
        auc = 0.0

    dis = compute_disentanglement_metrics(z_all, z_age_all, z_sex_all, age_pb, sex_pb, groups=rec_id_pb)

    return {
        "NMI": float(nmi), "AMI": float(ami), "Purity": float(purity),
        "MAE_Age": float(mae), "AUC_Sex": float(auc),
        "Disentanglement": dis,
    }


def main(cfg=None, model_path=None):
    if cfg is None:
        cfg = config
    fs = 100
    _set_global_determinism(cfg.random_seed)
    start = time.time()
    os.makedirs("./models", exist_ok=True)

    cache_path = f"./data/ptbxl_{fs}_T-12ms.pkl"
    if cfg.use_data_cache:
        ds = ECG_Dataset.load(cache_path)
    else:
        ds = ECG_Dataset(fs)
        ds.import_ptbxl(base_path="/nfs/data8/schlegel/git/ecg-cbm/data/ptb-xl")
        ds.save(cache_path)

    gen = ECG_DataGenerator(ds)
    sample = gen.get_all_beats_representation(split="train")
    input_length, input_channels = sample[0].shape[1], sample[0].shape[2]

    if model_path is not None:
        device = _get_device()
        state, meta = _load_checkpoint_state(model_path, device)
        latent_dim     = int(meta.get("latent_dim", cfg.latent_dim))
        sd             = meta.get("som_dim", cfg.som_dim)
        som_dim        = (int(sd[0]), int(sd[1]))
        input_length   = int(meta.get("input_length",   input_length))
        input_channels = int(meta.get("input_channels", input_channels))
        model = DPSOM_ECG(
            latent_dim=latent_dim, som_dim=som_dim,
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
        )
        model.load_state_dict(state, strict=True)
        eval_ckpt = model_path
    else:
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
        )
        train_model(
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
        )
        eval_ckpt = cfg.modelpath

    results = evaluate_model(model, gen)
    print(f"NMI={results['NMI']:.4f}  AMI={results['AMI']:.4f}  Purity={results['Purity']:.4f}  "
          f"MAE_Age={results['MAE_Age']:.4f}  AUC_Sex={results['AUC_Sex']:.4f}")

    single_label_filter = [len(r.extra_labels) == 0 for r in ds.get_data("test")]
    results_r = evaluate_model(model, gen, record_filter=single_label_filter)
    print(f"(single-label) NMI={results_r['NMI']:.4f}  Purity={results_r['Purity']:.4f}  "
          f"MAE_Age={results_r['MAE_Age']:.4f}  AUC_Sex={results_r['AUC_Sex']:.4f}")

    log_som_visualizations(ds, gen, cfg.ex_name, eval_ckpt, _get_device(), single_label_only=True)

    print(f"Elapsed: {time.time() - start:.1f}s")
    return results


if __name__ == "__main__":
    main()
