import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from src.model.codec import LeadWiseConvEncoder, LeadWiseConvDecoder


class DPSOM_ECG(nn.Module):
    """
    Disentangled Probabilistic SOM for ECG beat sequences (toroidal grid).

    Morphology encoder (z) drives SOM assignments via soft t-distribution assignments q(z).
    Age/sex encoders (z_age, z_sex) predict demographic labels from separate feature streams.
    SOM residualization projects out the age/sex directions from the morphology latent space.
    Disease-conditioned age correction uses TopK SOM node embeddings gated by detached q(z)
    so the correction does not push gradients into the morphology encoder or SOM.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        som_dim: Tuple[int, int] = (8, 8),
        input_length: int = 120,
        input_channels: int = 12,
        alpha: float = 10.0,
        beta: float = 20.0,
        gamma: float = 20.0,
        theta: float = 1.0,
        tau: float = 1.0,
        eta: float = 1.0,
        delta_age: float = 1.0,
        delta_sex: float = 1.0,
        dropout: float = 0.2,
        prior_var: float = 1.0,
        prior: float = 0.5,
        # Encoder architecture
        encoder_base_channels_1: int = 32,
        encoder_base_channels_2: int = 64,
        encoder_kernel_size: int = 7,
        encoder_fc_hidden_dim: int = 512,
        # Demographic latent space fractions
        z_age_dim_factor: float = 0.25,
        z_sex_dim_factor: float = 0.25,
        # Age-correction module
        age_corr_topk: int = 4,
        age_corr_lambda_max: float = 0.30,
        age_corr_ramp_epochs: int = 10,
        # SOM initialisation
        som_init_std: float = 0.05,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.som_dim = som_dim
        self.input_length = input_length
        self.input_channels = input_channels

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.tau = tau
        self.eta = eta
        self.delta_age = delta_age
        self.delta_sex = delta_sex
        self.dropout = dropout
        self.prior_var = prior_var
        self.prior = prior

        self.z_shape_dim = latent_dim
        self.z_age_dim = max(1, int(latent_dim * z_age_dim_factor))
        self.z_sex_dim = max(1, int(latent_dim * z_sex_dim_factor))

        self.min_age = 2
        self.max_age = 89
        self.num_age_bins = 88

        self.amp_context_dim = 2 * self.input_channels
        self.amp_proj_dim = 12
        self.amp_proj_age = nn.Sequential(
            nn.Linear(self.amp_context_dim, self.amp_proj_dim),
            nn.LayerNorm(self.amp_proj_dim),
            nn.LeakyReLU(0.2)
        )
        self.amp_proj_sex = nn.Sequential(
            nn.Linear(self.amp_context_dim, self.amp_proj_dim),
            nn.LayerNorm(self.amp_proj_dim),
            nn.LeakyReLU(0.2)
        )
        self.amp_attn_proj = nn.Sequential(
            nn.Linear(self.amp_context_dim, self.amp_proj_dim),
            nn.LayerNorm(self.amp_proj_dim),
            nn.Tanh()
        )

        h_som, w_som = self.som_dim
        self._embeddings = nn.Parameter(torch.empty(h_som, w_som, self.latent_dim))
        nn.init.trunc_normal_(self._embeddings, mean=0.0, std=som_init_std,
                               a=-2 * som_init_std, b=2 * som_init_std)

        # Use same-padding (kernel_size // 2) so out_length is independent of kernel_size
        _enc_kwargs = dict(
            input_channels=self.input_channels,
            input_length=self.input_length,
            base_channels_1=encoder_base_channels_1,
            base_channels_2=encoder_base_channels_2,
            kernel_size=encoder_kernel_size,
            padding=encoder_kernel_size // 2,
        )
        # Store architecture params for checkpoint metadata
        self._encoder_base_channels_1 = encoder_base_channels_1
        self._encoder_base_channels_2 = encoder_base_channels_2
        self._encoder_kernel_size = encoder_kernel_size
        self._encoder_fc_hidden_dim = encoder_fc_hidden_dim
        self._z_age_dim_factor = z_age_dim_factor
        self._z_sex_dim_factor = z_sex_dim_factor
        self.encoder     = LeadWiseConvEncoder(**_enc_kwargs)
        self.encoder_age = LeadWiseConvEncoder(**_enc_kwargs)
        self.encoder_sex = LeadWiseConvEncoder(**_enc_kwargs)

        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.enc_fc_hidden = nn.Linear(int(self.encoder.feature_dim), encoder_fc_hidden_dim)
        self.enc_fc_hidden_age = nn.Linear(int(self.encoder_age.feature_dim), encoder_fc_hidden_dim)
        self.enc_fc_hidden_sex = nn.Linear(int(self.encoder_sex.feature_dim), encoder_fc_hidden_dim)

        self.enc_mu = nn.Linear(encoder_fc_hidden_dim, self.z_shape_dim)
        self.enc_logvar = nn.Linear(encoder_fc_hidden_dim, self.z_shape_dim)
        self.enc_age = nn.Linear(encoder_fc_hidden_dim, self.z_age_dim)
        self.enc_sex = nn.Linear(encoder_fc_hidden_dim, self.z_sex_dim)

        self.decoder = LeadWiseConvDecoder(
            z_dim=self.z_shape_dim,
            input_channels=self.input_channels,
            input_length=self.input_length,
            enc_out_channels=int(self.encoder.out_channels),
            conv_out_len=int(self.encoder.out_length),
            dropout=self.dropout,
        )

        self.age_head = nn.Sequential(
            nn.Linear(self.z_age_dim + self.amp_proj_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, self.num_age_bins),
        )
        self.sex_head = nn.Sequential(
            nn.Linear(self.z_sex_dim + self.amp_proj_dim, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1),
        )

        self.rec_attn_age = nn.Sequential(
            nn.Linear(self.latent_dim + self.amp_proj_dim, self.latent_dim // 2),
            nn.Tanh(),
            nn.Linear(self.latent_dim // 2, 1),
        )
        self.rec_attn_sex = nn.Sequential(
            nn.Linear(self.latent_dim + self.amp_proj_dim, self.latent_dim // 2),
            nn.Tanh(),
            nn.Linear(self.latent_dim // 2, 1),
        )

        self.num_som_nodes = h_som * w_som
        self.age_corr_topk = age_corr_topk
        self.age_corr_lambda_max = age_corr_lambda_max
        self.age_corr_ramp_epochs = age_corr_ramp_epochs

        self.age_node_emb_dim = 24
        self.age_node_emb = nn.Embedding(self.num_som_nodes, self.age_node_emb_dim)

        corr_in = self.amp_proj_dim + self.age_node_emb_dim
        self.age_corr_mlp = nn.Sequential(
            nn.Linear(corr_in, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, self.num_age_bins),
        )
        self.age_corr_amp_shortcut = nn.Linear(self.amp_proj_dim, self.num_age_bins, bias=False)

        self.age_probe = nn.Linear(self.latent_dim, 1, bias=True)
        self.sex_probe = nn.Linear(self.latent_dim, 1, bias=True)

        self.register_buffer("_prior_scale", torch.ones(self.z_shape_dim) * float(self.prior_var))
        self.register_buffer("age_bin_values", torch.linspace(self.min_age, self.max_age, steps=self.num_age_bins))

        self._train_epoch = 0
        self._p_tensor = None
        self.register_buffer("_probe_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_age_dir", torch.zeros(self.latent_dim))
        self.register_buffer("_sex_probe_ready", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("_sex_dir", torch.zeros(self.latent_dim))

    def get_epoch(self):
        return self._train_epoch

    def inc_epoch(self):
        self._train_epoch += 1

    def _encode(self, x: torch.Tensor):
        feat_m = self.encoder(x)
        hm = F.leaky_relu(self.enc_fc_hidden(feat_m.flatten(1)), 0.2)
        hm = self.dropout_layer(hm)
        mu = self.enc_mu(hm)
        logvar = self.enc_logvar(hm).clamp(-10.0, 10.0)

        feat_a = self.encoder_age(x)
        ha = F.leaky_relu(self.enc_fc_hidden_age(feat_a.flatten(1)), 0.2)
        ha = self.dropout_layer(ha)
        z_age = self.enc_age(ha)

        feat_s = self.encoder_sex(x)
        hs = F.leaky_relu(self.enc_fc_hidden_sex(feat_s.flatten(1)), 0.2)
        hs = self.dropout_layer(hs)
        z_sex = self.enc_sex(hs)

        return mu, logvar, z_age, z_sex

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def set_age_direction_from_probe(self):
        w = self.age_probe.weight.detach().view(-1)
        self._age_dir.copy_((w / w.norm(p=2)).to(device=self._age_dir.device, dtype=self._age_dir.dtype))
        self._probe_ready.fill_(True)

    def residualize_latent(self, z: torch.Tensor) -> torch.Tensor:
        v_age = self._age_dir.to(device=z.device, dtype=z.dtype).detach()
        z = z - (z * v_age).sum(dim=1, keepdim=True) * v_age * self._probe_ready.float()

        v_sex = self._sex_dir.to(device=z.device, dtype=z.dtype).detach()
        z = z - (z * v_sex).sum(dim=1, keepdim=True) * v_sex * self._sex_probe_ready.float()

        return z

    def freeze_age_probe(self):
        for p in self.age_probe.parameters():
            p.requires_grad = False
        self.age_probe.eval()
        self.set_age_direction_from_probe()

    def set_sex_direction_from_probe(self):
        w = self.sex_probe.weight.detach().view(-1)
        self._sex_dir.copy_((w / w.norm(p=2)).to(device=self._sex_dir.device, dtype=self._sex_dir.dtype))
        self._sex_probe_ready.fill_(True)

    def freeze_sex_probe(self):
        for p in self.sex_probe.parameters():
            p.requires_grad = False
        self.sex_probe.eval()
        self.set_sex_direction_from_probe()

    @torch.compile()
    def forward_batch(self, beats: torch.Tensor, beat_valid: torch.Tensor):
        B, N, C, T = beats.shape
        BN = B * N

        valid_mask = beat_valid.reshape(BN) > 0.5
        valid_idx = valid_mask.nonzero(as_tuple=False).squeeze(1)
        beats_valid = beats.reshape(BN, C, T)[valid_mask]

        mu_v, logvar_v, zage_v, zsex_v = self._encode(beats_valid)
        z_v = self._reparameterize(mu_v, logvar_v)
        logits_v = self.decoder(z_v)

        z      = torch.zeros(BN, self.latent_dim, device=beats.device, dtype=z_v.dtype)
        mu     = torch.zeros(BN, self.latent_dim, device=beats.device, dtype=mu_v.dtype)
        logvar = torch.zeros(BN, self.latent_dim, device=beats.device, dtype=logvar_v.dtype)
        zage   = torch.zeros(BN, self.z_age_dim,  device=beats.device, dtype=zage_v.dtype)
        zsex   = torch.zeros(BN, self.z_sex_dim,  device=beats.device, dtype=zsex_v.dtype)
        logits = torch.zeros(BN, C * T,            device=beats.device, dtype=logits_v.dtype)

        z      = z.index_copy(0, valid_idx, z_v)
        mu     = mu.index_copy(0, valid_idx, mu_v)
        logvar = logvar.index_copy(0, valid_idx, logvar_v)
        zage   = zage.index_copy(0, valid_idx, zage_v)
        zsex   = zsex.index_copy(0, valid_idx, zsex_v)
        logits = logits.index_copy(0, valid_idx, logits_v)

        return (
            z.view(B, N, self.latent_dim),
            zage.view(B, N, self.z_age_dim),
            zsex.view(B, N, self.z_sex_dim),
            mu.view(B, N, self.latent_dim),
            logvar.view(B, N, self.latent_dim),
            logits,
        )

    def _age_corr_lambda(self) -> float:
        t = min(1.0, float(self._train_epoch) / float(self.age_corr_ramp_epochs))
        return float(self.age_corr_lambda_max) * t

    def age_logits_with_disease_correction(
        self,
        z_age: torch.Tensor,
        amp_context: torch.Tensor,
        z_morph: torch.Tensor,
    ) -> torch.Tensor:
        amp_proj = self.amp_proj_age(amp_context)
        age_logits_base = self.age_head(torch.cat([z_age, amp_proj], dim=1))

        q = self.q(z_morph.detach())
        q_top, idx_top = torch.topk(q, k=self.age_corr_topk, dim=1)
        q_top = q_top / q_top.sum(dim=1, keepdim=True).clamp_min(1e-8)

        node_e = self.age_node_emb(idx_top)
        amp_e = amp_proj.unsqueeze(1).expand(-1, self.age_corr_topk, -1)
        corr_in = torch.cat([amp_e, node_e], dim=-1)
        node_corr = self.age_corr_mlp(
            corr_in.reshape(-1, corr_in.shape[-1])
        ).view(amp_context.shape[0], self.age_corr_topk, self.num_age_bins)

        corr_logits = (q_top.unsqueeze(-1) * node_corr).sum(dim=1)
        corr_logits = corr_logits + self.age_corr_amp_shortcut(amp_proj)

        return age_logits_base + self._age_corr_lambda() * corr_logits

    def _kl_divergence_diag(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        prior_var = self._prior_scale.to(mu.device)
        var_q = torch.exp(logvar)
        kl = 0.5 * (
            (var_q / prior_var).sum(dim=1)
            + ((mu ** 2) / prior_var).sum(dim=1)
            - self.latent_dim
            + prior_var.log().sum() - logvar.sum(dim=1)
        )
        return kl.mean()

    def loss_reconstruction_ze(self, logits, x, mask, mu, logvar):
        B, C, T = x.shape
        pred = logits.view(B, C, T)
        target = x.to(pred.dtype)

        ramp = 0.2
        t = torch.arange(T, device=pred.device, dtype=pred.dtype)
        left = int(round(ramp * (T - 1)))
        right = T - 1 - left
        w = torch.ones(T, device=pred.device, dtype=pred.dtype)
        edge_weight = 0.1
        tl = t[:left + 1] / float(left)
        w[:left + 1] = edge_weight + (1.0 - edge_weight) * 0.5 * (1.0 - torch.cos(torch.pi * tl))
        tr = (t[right:] - float(right)) / float((T - 1) - right)
        w[right:] = 1.0 - (1.0 - edge_weight) * 0.5 * (1.0 - torch.cos(torch.pi * tr))

        m = (mask.to(pred.dtype) * w).unsqueeze(1).expand(-1, C, -1)
        denom = m.sum(dim=(1, 2)).clamp_min(1e-8)
        rec_huber = (
            F.huber_loss(pred, target, reduction="none", delta=0.5) * m
        ).sum(dim=(1, 2)) / denom * float(C * T)
        rec_huber = rec_huber.mean()

        outside = (mask <= 0).to(pred.dtype).unsqueeze(1).expand(-1, C, -1)
        rec_l1_outside = (
            pred.abs() * outside
        ).sum(dim=(1, 2)) / outside.sum(dim=(1, 2)).clamp_min(1e-8)
        rec_l1_outside = rec_l1_outside.mean()

        rec = rec_huber + rec_l1_outside
        kl = self._kl_divergence_diag(mu, logvar)
        return rec + self.prior * kl, rec, self.prior * kl

    def loss_reconstruction(self, logits, x, mask, mu, logvar):
        l, rl, kl = self.loss_reconstruction_ze(logits, x, mask, mu, logvar)
        return self.theta * l, self.theta * rl, self.theta * kl

    def loss_temporal_smoothness(self, z: torch.Tensor, beat_valid: torch.Tensor) -> torch.Tensor:
        valid_pairs = beat_valid[:, :-1] * beat_valid[:, 1:]
        z_diff = z[:, 1:, :] - z[:, :-1, :]
        loss_z = (z_diff.norm(dim=-1) ** 2 * valid_pairs).sum() / valid_pairs.sum().clamp_min(1.0)
        return self.tau * loss_z

    @torch.compile
    def z_dist_flat(self, z: torch.Tensor) -> torch.Tensor:
        H, W = self.som_dim
        z_use = self.residualize_latent(z)
        diff = z_use[:, None, None, :] - self._embeddings[None, :, :, :]
        return (diff * diff).sum(dim=-1).view(z.shape[0], H * W)

    def k(self, z: torch.Tensor) -> torch.Tensor:
        return torch.argmin(self.z_dist_flat(z), dim=-1)

    def z_q(self, z: torch.Tensor) -> torch.Tensor:
        H, W = self.som_dim
        k = self.k(z)
        return self._embeddings[k // W, k % W, :]

    def som_neighbors(self, k: torch.Tensor):
        H, W = self.som_dim
        k1, k2 = k // W, k % W
        k_up    = torch.remainder(k1 + 1, H) * W + k2
        k_down  = torch.remainder(k1 - 1, H) * W + k2
        k_right = k1 * W + torch.remainder(k2 + 1, W)
        k_left  = k1 * W + torch.remainder(k2 - 1, W)
        return k_up, k_down, k_right, k_left

    def z_q_neighbors(self, z: torch.Tensor) -> torch.Tensor:
        H, W = self.som_dim
        k = self.k(z)
        k_up, k_down, k_right, k_left = self.som_neighbors(k)
        return torch.stack([
            self._embeddings[k // W,       k % W,       :],
            self._embeddings[k_up // W,    k_up % W,    :],
            self._embeddings[k_down // W,  k_down % W,  :],
            self._embeddings[k_right // W, k_right % W, :],
            self._embeddings[k_left // W,  k_left % W,  :],
        ], dim=1)

    def q(self, z: torch.Tensor) -> torch.Tensor:
        dist_flat = self.z_dist_flat(z)
        eps = torch.finfo(dist_flat.dtype).eps
        # Clamp alpha away from zero to avoid NaN when alpha=0 (no_som_smooth variant)
        alpha = max(float(self.alpha), 1e-4)
        q = eps + 1.0 / torch.pow(1.0 + dist_flat / alpha, (alpha + 1.0) / 2.0)
        return q / q.sum(dim=1, keepdim=True)

    def q_p(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar, _, _ = self._encode(x)
        return self.q(self._reparameterize(mu, logvar))

    @property
    def p(self):
        return self._p_tensor

    def set_p(self, p_tensor: torch.Tensor):
        self._p_tensor = p_tensor

    def loss_commit(self, z: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        q = self.q(z)
        p = self.p.to(q.dtype).to(q.device)
        return (p.clamp_min(eps) * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(dim=1).mean()

    @torch.no_grad()
    def target_distribution(self, q_np):
        p = q_np ** 2 / q_np.sum(axis=0, keepdims=False)
        return p / p.sum(axis=1, keepdims=True)

    def loss_som(self, z: torch.Tensor) -> torch.Tensor:
        H, W = self.som_dim
        k = torch.arange(H * W, device=self._embeddings.device, dtype=torch.long)
        k_up, k_down, k_right, k_left = self.som_neighbors(k)

        z_ng = z.detach()
        q_t = self.q(z_ng).transpose(0, 1)
        q_neighbours = torch.stack([
            q_t[k_up], q_t[k_down], q_t[k_right], q_t[k_left]
        ], dim=2).transpose(0, 1)

        eps = torch.finfo(q_neighbours.dtype).eps
        new_q = self.q(z)
        return -torch.mean((torch.log(q_neighbours + eps).sum(dim=-1) * new_q).sum(dim=-1))

    def loss_commit_s(self, z_ng: torch.Tensor) -> torch.Tensor:
        return torch.mean((z_ng - self.z_q(z_ng)) ** 2)

    def loss_som_s(self, z_ng: torch.Tensor) -> torch.Tensor:
        return torch.mean((z_ng.unsqueeze(1) - self.z_q_neighbors(z_ng)) ** 2)

    def loss_a(self, z: torch.Tensor):
        z_ng = z.detach()
        a = self.loss_som_s(z_ng)
        b = self.loss_commit_s(z_ng)
        return a + b, a, b

    def meta_loss(self, z_age, z_sex, amp_context, age_target, sex_target, z_morph):
        age_logits = self.age_logits_with_disease_correction(z_age, amp_context, z_morph)
        t = ((age_target - self.min_age) / (self.max_age - self.min_age + 1e-8) * (self.num_age_bins - 1)).clamp(0, self.num_age_bins - 1).long()
        l_age = F.cross_entropy(age_logits, t)

        amp_proj_s = self.amp_proj_sex(amp_context)
        l_sex = F.binary_cross_entropy_with_logits(
            self.sex_head(torch.cat([z_sex, amp_proj_s], dim=1)).reshape(-1),
            sex_target.float().reshape(-1)
        )

        da = self.delta_age * l_age
        ds = self.delta_sex * l_sex
        return da + ds, da, ds

    def loss(self, logits, x, mask, z, mu, logvar, z_age, z_sex, amp_context, age_target, sex_target):
        a, rc, kl = self.loss_reconstruction(logits, x, mask, mu, logvar)
        supervised_loss, da, ds = self.meta_loss(z_age, z_sex, amp_context, age_target, sex_target, z_morph=mu)
        b = self.gamma * self.loss_commit(z)
        c = self.beta * self.loss_som(z)
        return a + b + c + supervised_loss, a, b, c, rc, kl, supervised_loss, da, ds

    @torch.compile()
    def loss_reconstruction_batch(self, beats, beat_mask, beat_meta, beat_valid, record_age, record_sex):
        B, N, C, T = beats.shape
        BN = B * N

        z, z_age, z_sex, mu, logvar, logits = self.forward_batch(beats, beat_valid)

        num_valid = (beat_valid.reshape(BN) > 0.5).float().sum().clamp_min(1.0)
        loss_rec, rc, kl = self.loss_reconstruction(
            logits,
            beats.reshape(BN, C, T),
            beat_mask.reshape(BN, T),
            mu.reshape(BN, -1),
            logvar.reshape(BN, -1),
        )
        scale = BN / num_valid
        loss_rec, rc, kl = loss_rec * scale, rc * scale, kl * scale

        valid = beat_valid.reshape(BN) > 0.5
        pred_loss, loss_age, loss_sex = self.meta_loss(
            z_age.reshape(BN, -1)[valid],
            z_sex.reshape(BN, -1)[valid],
            beat_meta.reshape(BN, -1)[valid],
            record_age.unsqueeze(1).expand(B, N).reshape(BN)[valid],
            record_sex.unsqueeze(1).expand(B, N).reshape(BN)[valid],
            z_morph=mu.reshape(BN, -1)[valid],
        )
        return loss_rec, rc, kl, pred_loss, loss_age, loss_sex

    def loss_a_batch(self, beats: torch.Tensor, beat_valid: torch.Tensor):
        B, N, C, T = beats.shape
        z, _, _, _, _, _ = self.forward_batch(beats, beat_valid)

        valid = beat_valid.reshape(B * N) > 0.5
        z_ng = z.reshape(B * N, -1)[valid].detach()
        loss_som_s = self.loss_som_s(z_ng)
        loss_commit_s = self.loss_commit_s(z_ng)
        return loss_som_s + loss_commit_s, loss_som_s, loss_commit_s

    def loss_batch(self, beats, beat_mask, beat_meta, beat_valid, record_age, record_sex, p_batch):
        B, N, C, T = beats.shape
        BN = B * N

        z, z_age, z_sex, mu, logvar, logits = self.forward_batch(beats, beat_valid)

        loss_temporal = self.loss_temporal_smoothness(z, beat_valid)

        beats_flat      = beats.reshape(BN, C, T)
        beat_mask_flat  = beat_mask.reshape(BN, T)
        beat_meta_flat  = beat_meta.reshape(BN, -1)
        mu_flat         = mu.reshape(BN, -1)
        logvar_flat     = logvar.reshape(BN, -1)
        z_flat          = z.reshape(BN, -1)
        zage_flat       = z_age.reshape(BN, -1)
        zsex_flat       = z_sex.reshape(BN, -1)
        p_batch_flat    = p_batch.reshape(BN, -1)
        beat_valid_flat = beat_valid.reshape(BN)

        valid = beat_valid_flat > 0.5
        self.set_p(p_batch_flat[valid])

        age_target = record_age.unsqueeze(1).expand(B, N).reshape(BN)
        sex_target = record_sex.unsqueeze(1).expand(B, N).reshape(BN)

        loss, loss_elbo, loss_commit, loss_som, rc, kl, sl, da, ds = self.loss(
            logits[valid], beats_flat[valid], beat_mask_flat[valid],
            z_flat[valid], mu_flat[valid], logvar_flat[valid],
            zage_flat[valid], zsex_flat[valid], beat_meta_flat[valid],
            age_target[valid], sex_target[valid],
        )

        age_logits_all = self.age_logits_with_disease_correction(
            z_age=zage_flat, amp_context=beat_meta_flat, z_morph=mu_flat
        )
        age_probs_beat = torch.softmax(age_logits_all.view(B, N, self.num_age_bins), dim=-1)
        age_pred_beat = (age_probs_beat * self.age_bin_values).sum(dim=-1)

        amp_proj_sex_flat = self.amp_proj_sex(beat_meta_flat)
        sex_logits_beat = self.sex_head(
            torch.cat([zsex_flat, amp_proj_sex_flat], dim=-1)
        ).view(B, N)

        attn_age = self.record_attention_age(mu.detach(), beat_valid, beat_meta)
        attn_sex = self.record_attention_sex(mu.detach(), beat_valid, beat_meta)

        age_pred_record = (age_pred_beat.detach() * attn_age).sum(dim=1)
        sex_logits_record = (sex_logits_beat.detach() * attn_sex).sum(dim=1)

        record_age_loss = 0.4 * self.delta_age * F.l1_loss(age_pred_record, record_age.float())
        record_sex_loss = self.delta_sex * F.binary_cross_entropy_with_logits(sex_logits_record, record_sex.float())
        record_loss = self.eta * (record_age_loss + record_sex_loss)

        total_loss = loss + loss_temporal + record_loss

        return (total_loss, loss_elbo, loss_commit, loss_som,
                rc, kl, sl, da, ds, loss_temporal, record_loss, record_age_loss, record_sex_loss)

    def predict_dual_age(self, beats, beat_valid, amp_context):
        B, N, C, T = beats.shape
        z, zage, zsex, mu, _, _ = self.forward_batch(beats, beat_valid)

        amp_flat  = amp_context.reshape(B * N, -1)
        mu_flat   = mu.reshape(B * N, -1)
        zage_flat = zage.reshape(B * N, -1)

        age_logits_all = self.age_logits_with_disease_correction(
            z_age=zage_flat, amp_context=amp_flat, z_morph=mu_flat
        )
        age_probs_beat = torch.softmax(age_logits_all.view(B, N, self.num_age_bins), dim=-1)
        age_pred_beat = (age_probs_beat * self.age_bin_values).sum(dim=-1)

        amp_proj_sex = self.amp_proj_sex(amp_flat).view(B, N, -1)
        sex_logits_beat = self.sex_head(
            torch.cat([zsex, amp_proj_sex], dim=-1)
        ).squeeze(-1)

        attn_age = self.record_attention_age(mu, beat_valid, amp_context)
        attn_sex = self.record_attention_sex(mu, beat_valid, amp_context)

        age_pred_record = (age_pred_beat * attn_age).sum(dim=1)
        sex_logits_record = (sex_logits_beat * attn_sex).sum(dim=1)

        return age_pred_record, sex_logits_record, attn_age, attn_sex

    def record_attention_age(self, beat_emb, beat_valid, amp_context):
        B, N, _ = beat_emb.shape
        amp_attn = self.amp_attn_proj(amp_context.reshape(B * N, -1)).view(B, N, -1)
        logits = self.rec_attn_age(torch.cat([beat_emb, amp_attn], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(beat_valid <= 0.5, -1e9)
        # Guard against all-masked records producing NaN in softmax
        all_masked = (beat_valid <= 0.5).all(dim=1, keepdim=True)
        logits = logits.masked_fill(all_masked, 0.0)
        return torch.softmax(logits, dim=1)

    def record_attention_sex(self, beat_emb, beat_valid, amp_context):
        B, N, _ = beat_emb.shape
        amp_attn = self.amp_attn_proj(amp_context.reshape(B * N, -1)).view(B, N, -1)
        logits = self.rec_attn_sex(torch.cat([beat_emb, amp_attn], dim=-1)).squeeze(-1)
        logits = logits.masked_fill(beat_valid <= 0.5, -1e9)
        # Guard against all-masked records producing NaN in softmax
        all_masked = (beat_valid <= 0.5).all(dim=1, keepdim=True)
        logits = logits.masked_fill(all_masked, 0.0)
        return torch.softmax(logits, dim=1)
