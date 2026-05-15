import torch
from torch import nn


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Produces gamma and beta from latent z and applies:
    x * (1 + gamma) + beta
    """
    def __init__(self, latent_dim: int, channels: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 2 * channels)
        # initialize near identity
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T], z: [B, latent_dim]
        params = self.fc(z).unsqueeze(2)  # [B, 2*C, 1]
        gamma, beta = params.chunk(2, dim=1)  # [B, C, 1]
        return x * (1.0 + gamma) + beta
