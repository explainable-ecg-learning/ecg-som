"""Baseline models for benchmarking against DPSOM-ECG on PTB-XL.

1. ECG_VAE:
   Same LeadWiseConvEncoder + LeadWiseConvDecoder as DPSOM-ECG (unsupervised),
   paired with K-Means clustering on latent means (mu) and linear demographic probes.
2. ResNet1D_ECG:
   Standard supervised 1D-ResNet multi-task upper bound predicting diagnostic
   subclass (16 classes), age (88 bins), and sex (binary) from 12-lead beats.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.codec import LeadWiseConvDecoder, LeadWiseConvEncoder


# ---------------------------------------------------------------------------
# Baseline 1: Unsupervised VAE (+ K-Means at evaluation)
# ---------------------------------------------------------------------------

class ECG_VAE(nn.Module):
    """Beat-level VAE sharing DPSOM-ECG's lead-wise conv encoder/decoder."""

    def __init__(
        self,
        latent_dim: int = 32,
        input_length: int = 120,
        input_channels: int = 12,
        dropout: float = 0.2,
        prior: float = 2.5,
        encoder_base_channels_1: int = 32,
        encoder_base_channels_2: int = 64,
        encoder_kernel_size: int = 7,
        encoder_fc_hidden_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_length = input_length
        self.input_channels = input_channels
        self.prior = prior

        self.encoder = LeadWiseConvEncoder(
            input_channels=input_channels,
            input_length=input_length,
            base_channels_1=encoder_base_channels_1,
            base_channels_2=encoder_base_channels_2,
            kernel_size=encoder_kernel_size,
            padding=encoder_kernel_size // 2,
        )
        self.fc_hidden = nn.Linear(int(self.encoder.feature_dim), encoder_fc_hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.fc_mu = nn.Linear(encoder_fc_hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(encoder_fc_hidden_dim, latent_dim)

        self.decoder = LeadWiseConvDecoder(
            z_dim=latent_dim,
            input_channels=input_channels,
            input_length=input_length,
            enc_out_channels=int(self.encoder.out_channels),
            conv_out_len=int(self.encoder.out_length),
            dropout=dropout,
            base_channels_1=encoder_base_channels_1,
            kernel_size=encoder_kernel_size,
            padding=encoder_kernel_size // 2,
        )

    def encode(self, x: torch.Tensor):
        h = F.leaky_relu(self.fc_hidden(self.encoder(x).flatten(1)), 0.2)
        h = self.dropout(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10.0, 10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def loss_batch(self, beats: torch.Tensor, beat_mask: torch.Tensor, beat_valid: torch.Tensor):
        B, N, C, T = beats.shape
        BN = B * N
        valid = beat_valid.reshape(BN) > 0.5
        x = beats.reshape(BN, C, T)[valid]
        m = beat_mask.reshape(BN, T)[valid]

        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred = self.decoder(z).view(-1, C, T)

        m_exp = m.unsqueeze(1).expand(-1, C, -1)
        denom = m_exp.sum(dim=(1, 2)).clamp_min(1e-8)
        rec_huber = (F.huber_loss(pred, x, reduction="none", delta=0.5) * m_exp).sum(dim=(1, 2)) / denom * float(C * T)
        outside = (m <= 0).to(pred.dtype).unsqueeze(1).expand(-1, C, -1)
        rec_out = (pred.abs() * outside).sum(dim=(1, 2)) / outside.sum(dim=(1, 2)).clamp_min(1e-8)
        rec = (rec_huber + rec_out).mean()

        kl = 0.5 * (torch.exp(logvar).sum(dim=1) + (mu ** 2).sum(dim=1) - self.latent_dim - logvar.sum(dim=1)).mean()
        return rec + self.prior * kl, rec, kl


# ---------------------------------------------------------------------------
# Baseline 2: Supervised 1D-ResNet (upper bound)
# ---------------------------------------------------------------------------

class _ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(p=dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.shortcut = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(self.drop(h)))
        return F.relu(h + self.shortcut(x), inplace=True)


class ResNet1D_ECG(nn.Module):
    """Multi-task supervised 1D-ResNet predicting disease subclass, age, and sex."""

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 16,
        amp_context_dim: int = 26,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.min_age = 2
        self.max_age = 89
        self.num_age_bins = 88

        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _ResBlock1D(64, 64, stride=1, dropout=dropout)
        self.layer2 = _ResBlock1D(64, 128, stride=2, dropout=dropout)
        self.layer3 = _ResBlock1D(128, 256, stride=2, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.amp_proj = nn.Sequential(
            nn.Linear(amp_context_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(inplace=True),
        )
        feat_dim = 256 + 16
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.age_head = nn.Sequential(nn.Linear(feat_dim, 64), nn.ReLU(inplace=True), nn.Linear(64, self.num_age_bins))
        self.sex_head = nn.Sequential(nn.Linear(feat_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 1))
        self.register_buffer("age_bin_values", torch.linspace(self.min_age, self.max_age, steps=self.num_age_bins))

    def forward(self, x: torch.Tensor, amp_context: torch.Tensor):
        h = self.pool(self.layer3(self.layer2(self.layer1(self.stem(x))))).squeeze(-1)
        f = torch.cat([h, self.amp_proj(amp_context)], dim=1)
        return self.cls_head(f), self.age_head(f), self.sex_head(f).squeeze(-1), h
