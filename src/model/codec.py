"""
ECG encoder / decoder base classes and lead-wise convolutional implementations.
"""
import torch
from torch import nn
import torch.nn.functional as F

from src.model.layers import FiLMLayer, SEBlock


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class ECGEncoderBase(nn.Module):
    """
    Interface expected by DPSOM_ECG.

    ``forward(x)`` must return a feature map ``[B, out_channels, out_length]``.
    Implementations must set:
        out_channels : int
        out_length   : int
        feature_dim  : int  (= out_channels * out_length)
    """
    out_channels: int
    out_length: int
    feature_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class ECGDecoderBase(nn.Module):
    """
    Interface expected by DPSOM_ECG.
    ``forward(z)`` must return ``[B, C*T]`` (flattened reconstruction).
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Lead-wise convolutional encoder
# ---------------------------------------------------------------------------

class LeadWiseConvEncoder(ECGEncoderBase):
    """
    Two-stage lead-wise (depthwise) convolutional encoder.
    Returns feature map ``[B, out_channels, out_length]``.
    """

    def __init__(
        self,
        input_channels: int,
        input_length: int,
        base_channels_1: int = 32,
        base_channels_2: int = 64,
        kernel_size: int = 7,
        stride: int = 2,
        padding: int = 3,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_length = input_length

        def _conv_len(length):
            return ((length + 2 * padding - kernel_size) // stride) + 1

        L = _conv_len(_conv_len(input_length))
        self.out_length = L

        fpl1 = max(base_channels_1 // input_channels, 1)
        out1 = fpl1 * input_channels
        fpl2 = max(base_channels_2 // input_channels, 1)
        out2 = fpl2 * input_channels

        self.out_channels = out2
        self.feature_dim = out2 * L

        self.conv1 = nn.Conv1d(input_channels, out1, kernel_size, stride, padding, groups=input_channels)
        self.bn1   = nn.GroupNorm(input_channels, out1)
        self.se1   = SEBlock(out1)

        self.conv2 = nn.Conv1d(out1, out2, kernel_size, stride, padding, groups=input_channels)
        self.bn2   = nn.GroupNorm(input_channels, out2)
        self.se2   = SEBlock(out2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.se1(self.bn1(self.conv1(x))), 0.2)
        h = F.leaky_relu(self.se2(self.bn2(self.conv2(h))), 0.2)
        return h  # [B, out_channels, out_length]


# ---------------------------------------------------------------------------
# Lead-wise convolutional decoder
# ---------------------------------------------------------------------------

class LeadWiseConvDecoder(ECGDecoderBase):
    """
    Decoder: latent z → waveform ``[B, C*T]`` via lead-wise conv + FiLM.
    """

    def __init__(
        self,
        z_dim: int,
        input_channels: int,
        input_length: int,
        enc_out_channels: int,
        conv_out_len: int,
        dropout: float = 0.2,
        base_channels_1: int = 32,
        base_channels_3: int = 16,
        kernel_size: int = 7,
        padding: int = 3,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.input_channels = input_channels
        self.input_length = input_length
        self.enc_out_channels = enc_out_channels
        self.conv_out_len = conv_out_len

        self.dropout = nn.Dropout(p=dropout)
        self.dec_fc  = nn.Linear(z_dim, enc_out_channels * conv_out_len)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        fpl1 = max(base_channels_1 // input_channels, 1)
        out1 = fpl1 * input_channels
        fpl3 = max(base_channels_3 // input_channels, 1)
        out3 = fpl3 * input_channels

        self.conv1  = nn.Conv1d(enc_out_channels, out1, kernel_size, 1, padding, groups=input_channels)
        self.bn1    = nn.GroupNorm(input_channels, out1)
        self.se1    = SEBlock(out1)
        self.film1  = FiLMLayer(z_dim, out1)

        self.conv2  = nn.Conv1d(out1, out3, kernel_size, 1, padding, groups=input_channels)
        self.bn2    = nn.GroupNorm(input_channels, out3)
        self.se2    = SEBlock(out3)
        self.film2  = FiLMLayer(z_dim, out3)

        self.conv_out = nn.Conv1d(out3, input_channels, kernel_size, 1, padding, groups=input_channels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = F.leaky_relu(self.dropout(self.dec_fc(z)), 0.2)
        h = h.view(B, self.enc_out_channels, self.conv_out_len)

        h = self.upsample(h)
        h = F.leaky_relu(self.film1(self.se1(self.bn1(self.conv1(h))), z), 0.2)

        h = self.upsample(h)
        h = F.leaky_relu(self.film2(self.se2(self.bn2(self.conv2(h))), z), 0.2)

        return self.conv_out(h).reshape(B, -1)  # [B, C*T]
