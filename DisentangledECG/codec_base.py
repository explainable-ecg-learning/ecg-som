# Pluggable encoder/decoder base class
import torch
from torch import nn


class ECGEncoderBase(nn.Module):
    """
    Encoder interface expected by DPSOM_ECG.

    forward(x) MUST return a feature map: [B, out_channels, out_length]

    Implementations must set:
      - out_channels: int
      - out_length: int
      - feature_dim: int (out_channels * out_length)
    """
    out_channels: int
    out_length: int
    feature_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class ECGDecoderBase(nn.Module):
    """
    Minimal interface expected by DPSOM_ECG.
    Implementations must implement forward(z) -> recon_flat [B, C*T]
    where C=input_channels and T=input_length from the parent model.
    """
    def forward(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError
