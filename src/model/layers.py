"""Reusable building blocks shared across encoder/decoder architectures."""
import torch
from torch import nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for 1-D signals."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        reduced = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Produces (gamma, beta) from latent z and applies:
        h = x * (1 + gamma) + beta
    Initialized near identity (zero weight / bias).
    """

    def __init__(self, latent_dim: int, channels: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 2 * channels)
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T], z: [B, latent_dim]
        params = self.fc(z).unsqueeze(2)          # [B, 2*C, 1]
        gamma, beta = params.chunk(2, dim=1)      # [B, C, 1]
        return x * (1.0 + gamma) + beta
