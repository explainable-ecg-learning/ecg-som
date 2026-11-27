"""Strong Unsupervised Domain Adaptation (UDA) and demographic calibration for DPSOM-ECG.

Combines closed-form Bures-Wasserstein optimal transport (BW-OT) alignment for zero-backprop
domain shift correction with Hybrid Beat+Record Damped-DEC self-training, Cosine LR, and
Stochastic Weight Averaging (SWA) across cohorts.
"""
from __future__ import annotations

import copy
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from src.data.dataset import ECG_Dataset
from src.data.generator import ECG_DataGenerator
from src.model.dpsom import DPSOM_ECG


DEMO_HEAD_ADAPT_PREFIXES: Tuple[str, ...] = (
    "amp_proj_age.",
    "amp_proj_sex.",
    "age_head.",
    "sex_head.",
)


def _amp_settings(device: torch.device, use_amp: bool) -> Tuple[bool, torch.dtype]:
    enabled = bool(use_amp and device.type == "cuda")
    dtype = torch.bfloat16 if (enabled and torch.cuda.is_bf16_supported()) else torch.float16
    return enabled, dtype


def _is_morph_or_som_param(name: str) -> bool:
    return not any(k in name for k in ("age", "sex"))


def _snapshot_params(model: DPSOM_ECG, prefixes: Sequence[str]) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().float().clone()
        for name, param in model.named_parameters()
        if any(name.startswith(p) for p in prefixes)
    }


def _moments(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute FP32 mean [D] and unbiased covariance [D, D] of feature matrix [N, D]."""
    x32 = x.float()
    mean = x32.mean(dim=0)
    xc = x32 - mean
    denom = float(max(x32.shape[0] - 1, 1))
    cov = (xc.T @ xc) / denom
    return mean.detach(), cov.detach()


def extract_domain_moments(
    model: DPSOM_ECG,
    ds: ECG_Dataset,
    batch_size: int = 128,
    use_rr_features: bool = False,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Compute train-split FP32 mean and covariance for mu, z_age, z_sex, and beat_meta."""
    device = next(model.parameters()).device
    gen = ECG_DataGenerator(
        ds,
        batch_size=batch_size,
        device=device,
        pin_memory=False,
        use_rr_features=use_rr_features,
        preload_to_gpu=(device.type == "cuda"),
    )
    rec_gen = gen.get_record_batch_generator(mode="train", max_beats=4, shuffle=False)
    num_batches = gen.get_num_batches("train")

    mu_list, age_list, sex_list, meta_list = [], [], [], []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(num_batches):
                beats, _, beat_meta, beat_valid = next(rec_gen)[:4]
                B, N, C, T = beats.shape
                valid = beat_valid.reshape(B * N) > 0.5
                if not bool(valid.any()):
                    continue
                beats_v = beats.reshape(B * N, C, T)[valid]
                meta_v = beat_meta.reshape(B * N, -1)[valid].float()

                mu, _, z_age, z_sex = model._encode(beats_v)
                mu_list.append(mu.float())
                age_list.append(z_age.float())
                sex_list.append(z_sex.float())
                meta_list.append(meta_v)
    finally:
        model.train(was_training)

    return {
        "mu": _moments(torch.cat(mu_list, dim=0)),
        "z_age": _moments(torch.cat(age_list, dim=0)),
        "z_sex": _moments(torch.cat(sex_list, dim=0)),
        "meta": _moments(torch.cat(meta_list, dim=0)),
    }


extract_source_moments = extract_domain_moments


def bures_wasserstein_ot_map(
    src_mean: torch.Tensor,
    src_cov: torch.Tensor,
    tgt_mean: torch.Tensor,
    tgt_cov: torch.Tensor,
    reg: float = 1e-2,
    shrink: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Closed-form FP64 Bures-Wasserstein Gaussian Monge optimal transport map (A, b)."""
    device = src_cov.device
    dim = int(src_cov.shape[0])
    eye = torch.eye(dim, device=device, dtype=torch.float64)

    cs = 0.5 * (src_cov.double() + src_cov.double().T) + reg * eye
    ct = 0.5 * (tgt_cov.double() + tgt_cov.double().T) + reg * eye

    evals_t, evecs_t = torch.linalg.eigh(ct)
    evals_t = evals_t.clamp_min(1e-6)
    ct_half = (evecs_t * evals_t.sqrt()) @ evecs_t.T
    ct_invhalf = (evecs_t * evals_t.rsqrt()) @ evecs_t.T

    mid = ct_half @ cs @ ct_half
    evals_m, evecs_m = torch.linalg.eigh(0.5 * (mid + mid.T))
    evals_m = evals_m.clamp_min(1e-6)
    mid_half = (evecs_m * evals_m.sqrt()) @ evecs_m.T

    a_full = ct_invhalf @ mid_half @ ct_invhalf
    a_s = (1.0 - float(shrink)) * eye + float(shrink) * a_full
    ms = src_mean.double()
    mt = tgt_mean.double()
    b_s = (1.0 - float(shrink)) * mt + float(shrink) * ms - a_s @ mt
    return a_s.float(), b_s.float()


def _fold_linear_post(layer: torch.nn.Linear, mat: torch.Tensor, vec: torch.Tensor) -> None:
    """Fold y_new = A (W x + bias) + b into linear layer weights in-place."""
    with torch.no_grad():
        a = mat.to(device=layer.weight.device, dtype=layer.weight.dtype)
        b = vec.to(device=layer.weight.device, dtype=layer.weight.dtype)
        layer.weight.copy_(a @ layer.weight)
        if layer.bias is not None:
            layer.bias.copy_(a @ layer.bias + b)


def _fold_linear_pre(layer: torch.nn.Linear, mat: torch.Tensor, vec: torch.Tensor) -> None:
    """Fold y_new = W (A x + b) + bias into linear layer weights in-place."""
    with torch.no_grad():
        a = mat.to(device=layer.weight.device, dtype=layer.weight.dtype)
        b = vec.to(device=layer.weight.device, dtype=layer.weight.dtype)
        if layer.bias is not None:
            layer.bias.copy_(layer.bias + layer.weight @ b)
        layer.weight.copy_(layer.weight @ a)


def adapt_uda_bw_ot(
    src_model: DPSOM_ECG,
    tgt_ds: ECG_Dataset,
    src_moments: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    s_morph: float = 0.35,
    s_age: float = 0.25,
    s_sex: float = 0.75,
    reg: float = 1e-2,
    use_rr_features: bool = False,
) -> DPSOM_ECG:
    """Zero-backprop closed-form Bures-Wasserstein OT folded into linear projection layers."""
    device = next(src_model.parameters()).device
    adapted = copy.deepcopy(src_model).to(device)
    tgt_moments = extract_domain_moments(adapted, tgt_ds, use_rr_features=use_rr_features)

    for layer, key, shrink in (
        (adapted.enc_mu, "mu", s_morph),
        (adapted.enc_age, "z_age", s_age),
        (adapted.enc_sex, "z_sex", s_sex),
    ):
        if shrink > 0.0:
            a, b = bures_wasserstein_ot_map(
                *src_moments[key], *tgt_moments[key], reg=reg, shrink=shrink
            )
            _fold_linear_post(layer, a, b)

    if s_age > 0.0:
        a_m, b_m = bures_wasserstein_ot_map(
            *src_moments["meta"], *tgt_moments["meta"], reg=reg, shrink=s_age
        )
        _fold_linear_pre(adapted.amp_proj_age[0], a_m, b_m)

    if s_sex > 0.0:
        a_m, b_m = bures_wasserstein_ot_map(
            *src_moments["meta"], *tgt_moments["meta"], reg=reg, shrink=s_sex
        )
        _fold_linear_pre(adapted.amp_proj_sex[0], a_m, b_m)

    return adapted


def compute_hybrid_target_distribution(
    model: DPSOM_ECG,
    gen: ECG_DataGenerator,
    split: str = "train",
    alpha_rec: float = 0.35,
    rho: float = 0.75,
    device: torch.device | None = None,
    use_amp: bool = True,
) -> torch.Tensor:
    """Compute Hybrid Beat+Record Damped-DEC target distribution table ``p_table`` [N_beats, K]."""
    if device is None:
        device = next(model.parameters()).device

    num_batches = gen.get_num_batches(split)
    total_beats = int(gen.get_flat_beats_tensor(split).shape[0])
    num_nodes = int(model.som_dim[0] * model.som_dim[1])
    if total_beats == 0 or num_batches == 0:
        return torch.zeros((total_beats, num_nodes), device=device, dtype=torch.float32)

    amp_enabled, amp_dtype = _amp_settings(device, use_amp)
    rec_gen = gen.get_record_batch_generator(mode=split, max_beats=None, shuffle=False)
    q_table = torch.zeros((total_beats, num_nodes), device=device, dtype=torch.float32)

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(num_batches):
                beats, _, _, beat_valid, global_beat_idx = next(rec_gen)[:5]
                B, N, C, T = beats.shape
                valid_flat = beat_valid.reshape(B * N) > 0.5
                q_bn = torch.zeros((B * N, num_nodes), device=device, dtype=torch.float32)
                if bool(valid_flat.any()):
                    beats_v = beats.reshape(B * N, C, T)[valid_flat]
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                        mu_v = model._encode(beats_v)[0]
                        q_v = model.q(mu_v.float())
                    q_bn[valid_flat] = q_v.float()
                q_bn = q_bn.view(B, N, num_nodes)
                w = (beat_valid > 0.5).float().unsqueeze(-1)
                q_rec = (q_bn * w).sum(dim=1, keepdim=True) / w.sum(dim=1, keepdim=True).clamp_min(1.0)
                q_hyb = (1.0 - float(alpha_rec)) * q_bn + float(alpha_rec) * q_rec
                valid_gidx = global_beat_idx >= 0
                if bool(valid_gidx.any()):
                    q_table[global_beat_idx[valid_gidx].long()] = q_hyb[valid_gidx]
    finally:
        model.train(was_training)

    f = q_table.sum(dim=0, keepdim=True).clamp_min(1e-8)
    p_table = (q_table ** 2) / (f ** float(rho))
    p_table = p_table / p_table.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return p_table


def l2_sp_loss(model: DPSOM_ECG, ref_params: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Layer-normalized L2-SP weight anchor toward initial parameters."""
    device = next(model.parameters()).device
    total = torch.zeros((), device=device, dtype=torch.float32)
    for name, param in model.named_parameters():
        ref = ref_params.get(name)
        if ref is not None:
            total = total + (param.float() - ref.to(device=device, dtype=torch.float32)).pow(2).mean()
    return total


def _gather_p_batch(
    p_table: torch.Tensor,
    global_beat_idx: torch.Tensor,
    beat_valid: torch.Tensor,
) -> torch.Tensor:
    B, N = beat_valid.shape
    p_batch = torch.zeros((B, N, p_table.shape[1]), device=p_table.device, dtype=p_table.dtype)
    valid_gidx = global_beat_idx >= 0
    if bool(valid_gidx.any()):
        p_batch[valid_gidx] = p_table[global_beat_idx[valid_gidx].long()]
    return p_batch


def _unsupervised_step_terms(
    adapted: DPSOM_ECG,
    beats: torch.Tensor,
    beat_mask: torch.Tensor,
    beat_valid: torch.Tensor,
    p_batch: torch.Tensor,
) -> torch.Tensor:
    """Compute unsupervised ELBO + DEC commit + SOM + temporal smoothness loss."""
    B, N, C, T = beats.shape
    BN = B * N
    z, _, _, mu, logvar, logits = adapted.forward_batch(beats, beat_valid)
    valid = beat_valid.reshape(BN) > 0.5

    mu_v = mu.reshape(BN, -1)[valid]
    z_v = z.reshape(BN, -1)[valid]
    loss_elbo, _, _ = adapted.loss_reconstruction(
        logits[valid],
        beats.reshape(BN, C, T)[valid],
        beat_mask.reshape(BN, T)[valid],
        mu_v,
        logvar.reshape(BN, -1)[valid],
    )
    adapted.set_p(p_batch.reshape(BN, -1)[valid])
    loss_commit = adapted.gamma * adapted.loss_commit(z_v)
    loss_som = adapted.beta * adapted.loss_som(z_v)
    loss_temp = adapted.loss_temporal_smoothness(z, beat_valid)
    return loss_elbo + loss_commit + loss_som + loss_temp


def _accumulate_swa(
    model: DPSOM_ECG,
    swa_state: Dict[str, torch.Tensor] | None,
    swa_count: int,
) -> Tuple[Dict[str, torch.Tensor], int]:
    swa_count += 1
    if swa_state is None:
        return {k: v.detach().clone() for k, v in model.state_dict().items()}, swa_count
    for k, v in model.state_dict().items():
        if v.is_floating_point():
            swa_state[k].mul_((swa_count - 1) / swa_count).add_(v.detach(), alpha=1.0 / swa_count)
    return swa_state, swa_count


def _freeze_to_morph_and_som(model: DPSOM_ECG) -> list[torch.nn.Parameter]:
    trainable: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if _is_morph_or_som_param(name):
            param.requires_grad = True
            trainable.append(param)
        else:
            param.requires_grad = False
    return trainable


def adapt_uda_sf_strong(
    src_model: DPSOM_ECG,
    tgt_ds: ECG_Dataset,
    batch_size: int = 64,
    adapt_epochs: int = 8,
    lr: float = 1e-4,
    alpha_rec: float = 0.35,
    rho: float = 0.75,
    tau_mult: float = 1.5,
    use_swa: bool = True,
    use_amp: bool = True,
    use_rr_features: bool = False,
) -> DPSOM_ECG:
    """Source-free UDA: Hybrid Beat+Record Damped-DEC + Cosine LR + SWA."""
    device = next(src_model.parameters()).device
    adapted = copy.deepcopy(src_model).to(device)
    orig_tau = float(adapted.tau)
    adapted.delta_age = 0.0
    adapted.delta_sex = 0.0
    adapted.tau = orig_tau * float(tau_mult)

    trainable = _freeze_to_morph_and_som(adapted)
    tgt_gen = ECG_DataGenerator(
        tgt_ds, batch_size=batch_size, device=device, pin_memory=False,
        use_rr_features=use_rr_features, preload_to_gpu=(device.type == "cuda"),
    )
    num_batches = tgt_gen.get_num_batches("train")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, adapt_epochs), eta_min=0.1 * lr
    )
    amp_enabled, amp_dtype = _amp_settings(device, use_amp)

    swa_state: Dict[str, torch.Tensor] | None = None
    swa_count = 0
    for ep in range(adapt_epochs):
        p_table = compute_hybrid_target_distribution(
            adapted, tgt_gen, split="train", alpha_rec=alpha_rec, rho=rho,
            device=device, use_amp=use_amp,
        )
        train_gen = tgt_gen.get_record_batch_generator(mode="train", max_beats=4, shuffle=True)
        adapted.train()
        for _ in range(num_batches):
            beats, beat_mask, _, beat_valid, global_beat_idx = next(train_gen)[:5]
            if not bool((beat_valid > 0.5).any()):
                continue
            p_batch = _gather_p_batch(p_table, global_beat_idx, beat_valid)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                loss = _unsupervised_step_terms(adapted, beats, beat_mask, beat_valid, p_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
        scheduler.step()
        adapted.inc_epoch()
        if use_swa and ep >= adapt_epochs // 2:
            swa_state, swa_count = _accumulate_swa(adapted, swa_state, swa_count)

    if use_swa and swa_state is not None:
        adapted.load_state_dict(swa_state, strict=True)
    adapted.tau = orig_tau
    return adapted


def adapt_uda_ot_strong(
    src_model: DPSOM_ECG,
    tgt_ds: ECG_Dataset,
    src_moments: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    s_age: float = 0.25,
    s_sex: float = 0.75,
    batch_size: int = 64,
    adapt_epochs: int = 8,
    lr: float = 1e-4,
    alpha_rec: float = 0.35,
    rho: float = 0.75,
    tau_mult: float = 1.5,
    use_amp: bool = True,
    use_rr_features: bool = False,
) -> DPSOM_ECG:
    """BW-OT demographic alignment followed by Hybrid Damped-DEC + SWA on morphology/SOM."""
    m_ot = adapt_uda_bw_ot(
        src_model,
        tgt_ds,
        src_moments,
        s_morph=0.0,
        s_age=s_age,
        s_sex=s_sex,
        use_rr_features=use_rr_features,
    )
    return adapt_uda_sf_strong(
        m_ot,
        tgt_ds,
        batch_size=batch_size,
        adapt_epochs=adapt_epochs,
        lr=lr,
        alpha_rec=alpha_rec,
        rho=rho,
        tau_mult=tau_mult,
        use_swa=True,
        use_amp=use_amp,
        use_rr_features=use_rr_features,
    )


def adapt_uda_replay_strong(
    src_model: DPSOM_ECG,
    src_ds: ECG_Dataset,
    tgt_ds: ECG_Dataset,
    src_moments: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    s_age: float = 0.25,
    s_sex: float = 0.75,
    batch_size: int = 64,
    adapt_epochs: int = 8,
    lr: float = 1e-4,
    alpha_rec: float = 0.35,
    rho: float = 0.75,
    tau_mult: float = 1.5,
    w_src: float = 0.35,
    max_steps_per_epoch: int = 80,
    use_amp: bool = True,
    use_rr_features: bool = False,
) -> DPSOM_ECG:
    """BW-OT demographic init + Joint Source-Replay Hybrid Damped-DEC + SWA (0 target labels)."""
    adapted = adapt_uda_bw_ot(
        src_model,
        tgt_ds,
        src_moments,
        s_morph=0.0,
        s_age=s_age,
        s_sex=s_sex,
        use_rr_features=use_rr_features,
    )
    device = next(adapted.parameters()).device
    orig_tau = float(adapted.tau)
    adapted.delta_age = 0.0
    adapted.delta_sex = 0.0
    adapted.tau = orig_tau * float(tau_mult)

    trainable = _freeze_to_morph_and_som(adapted)
    preload = device.type == "cuda"
    tgt_gen = ECG_DataGenerator(
        tgt_ds, batch_size=batch_size, device=device, pin_memory=False,
        use_rr_features=use_rr_features, preload_to_gpu=preload,
    )
    src_gen = ECG_DataGenerator(
        src_ds, batch_size=batch_size, device=device, pin_memory=False,
        use_rr_features=use_rr_features, preload_to_gpu=preload,
    )
    num_steps = min(int(tgt_gen.get_num_batches("train")), int(max_steps_per_epoch))
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, adapt_epochs), eta_min=0.1 * lr
    )
    amp_enabled, amp_dtype = _amp_settings(device, use_amp)

    src_rec_gen = src_gen.get_record_batch_generator(mode="train", max_beats=4, shuffle=True)
    swa_state: Dict[str, torch.Tensor] | None = None
    swa_count = 0

    for ep in range(adapt_epochs):
        p_table_tgt = compute_hybrid_target_distribution(
            adapted, tgt_gen, split="train", alpha_rec=alpha_rec, rho=rho,
            device=device, use_amp=use_amp,
        )
        p_table_src = compute_hybrid_target_distribution(
            adapted, src_gen, split="train", alpha_rec=alpha_rec, rho=rho,
            device=device, use_amp=use_amp,
        )
        tgt_rec_gen = tgt_gen.get_record_batch_generator(mode="train", max_beats=4, shuffle=True)
        adapted.train()
        for _ in range(num_steps):
            beats, beat_mask, _, beat_valid, global_beat_idx = next(tgt_rec_gen)[:5]
            if not bool((beat_valid > 0.5).any()):
                continue
            p_batch_tgt = _gather_p_batch(p_table_tgt, global_beat_idx, beat_valid)

            try:
                src_batch = next(src_rec_gen)
            except StopIteration:
                src_rec_gen = src_gen.get_record_batch_generator(mode="train", max_beats=4, shuffle=True)
                src_batch = next(src_rec_gen)
            s_beats, s_mask, _, s_valid, s_gidx = src_batch[:5]
            if not bool((s_valid > 0.5).any()):
                continue
            p_batch_src = _gather_p_batch(p_table_src, s_gidx, s_valid)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                l_tgt = _unsupervised_step_terms(adapted, beats, beat_mask, beat_valid, p_batch_tgt)
                l_src = _unsupervised_step_terms(adapted, s_beats, s_mask, s_valid, p_batch_src)
                total = l_tgt + float(w_src) * l_src
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
        scheduler.step()
        adapted.inc_epoch()
        if ep >= adapt_epochs // 2:
            swa_state, swa_count = _accumulate_swa(adapted, swa_state, swa_count)

    if swa_state is not None:
        adapted.load_state_dict(swa_state, strict=True)
    adapted.tau = orig_tau
    return adapted


def _build_fewshot_dataset(tgt_ds: ECG_Dataset, n_shots: int, seed: int) -> ECG_Dataset:
    tr_recs = tgt_ds.get_data("train")
    k = min(n_shots, len(tr_recs))
    if len(tr_recs) > k:
        y_tr = np.array([r.label for r in tr_recs])
        try:
            sss = StratifiedShuffleSplit(n_splits=1, train_size=k, random_state=seed)
            idx, _ = next(sss.split(np.zeros(len(tr_recs)), y_tr))
        except ValueError:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(tr_recs), size=k, replace=False)
        shot_recs = [tr_recs[int(i)] for i in idx]
    else:
        shot_recs = list(tr_recs)

    shot_ds = ECG_Dataset(fs=tgt_ds.fs)
    shot_ds.class_names = list(tgt_ds.class_names)
    shot_ds.label_encoder = tgt_ds.label_encoder
    shot_ds.records = shot_recs
    return shot_ds


def adapt_fewshot_demo_heads(
    src_model: DPSOM_ECG,
    tgt_ds: ECG_Dataset,
    src_moments: Dict[str, Tuple[torch.Tensor, torch.Tensor]] | None = None,
    n_shots: int = 100,
    seed: int = 42,
    batch_size: int = 32,
    adapt_epochs: int = 10,
    lr: float = 5e-4,
    lambda_sp: float = 0.5,
    use_amp: bool = True,
    use_rr_features: bool = False,
) -> DPSOM_ECG:
    """BW-OT init + head-only 100-shot demographic calibration keeping CNNs and SOM frozen."""
    if src_moments is not None:
        adapted = adapt_uda_bw_ot(
            src_model,
            tgt_ds,
            src_moments,
            s_morph=0.0,
            s_age=0.25,
            s_sex=0.75,
            use_rr_features=use_rr_features,
        )
    else:
        device = next(src_model.parameters()).device
        adapted = copy.deepcopy(src_model).to(device)

    device = next(adapted.parameters()).device
    adapted.delta_age = float(max(adapted.delta_age, 1.0))
    adapted.delta_sex = float(max(adapted.delta_sex, 1.0))

    trainable: list[torch.nn.Parameter] = []
    for name, param in adapted.named_parameters():
        if name.startswith(DEMO_HEAD_ADAPT_PREFIXES):
            param.requires_grad = True
            trainable.append(param)
        else:
            param.requires_grad = False

    ref_heads = _snapshot_params(adapted, DEMO_HEAD_ADAPT_PREFIXES)
    shot_ds = _build_fewshot_dataset(tgt_ds, n_shots=n_shots, seed=seed)

    shot_gen = ECG_DataGenerator(
        shot_ds,
        batch_size=min(batch_size, max(1, len(shot_ds.records))),
        device=device,
        pin_memory=False,
        use_rr_features=use_rr_features,
        preload_to_gpu=(device.type == "cuda"),
    )
    num_batches = shot_gen.get_num_batches("train")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, adapt_epochs), eta_min=0.1 * lr
    )
    amp_enabled, amp_dtype = _amp_settings(device, use_amp)

    for _ in range(adapt_epochs):
        train_gen = shot_gen.get_record_batch_generator(mode="train", max_beats=4, shuffle=True)
        adapted.train()
        for _ in range(num_batches):
            beats, _, beat_meta, beat_valid, _, record_age, record_sex, _ = next(train_gen)
            B, N, _, _ = beats.shape
            BN = B * N
            valid = beat_valid.reshape(BN) > 0.5
            if not bool(valid.any()):
                continue

            age_t = record_age.unsqueeze(1).expand(B, N).reshape(BN)[valid]
            sex_t = record_sex.unsqueeze(1).expand(B, N).reshape(BN)[valid]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                _, z_age, z_sex, mu, _, _ = adapted.forward_batch(beats, beat_valid)
                pred_loss, _, _ = adapted.meta_loss(
                    z_age.reshape(BN, -1)[valid],
                    z_sex.reshape(BN, -1)[valid],
                    beat_meta.reshape(BN, -1)[valid],
                    age_t,
                    sex_t,
                    z_morph=mu.reshape(BN, -1)[valid].detach(),
                )
                total = pred_loss + float(lambda_sp) * l2_sp_loss(adapted, ref_heads)
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
        scheduler.step()

    return adapted
