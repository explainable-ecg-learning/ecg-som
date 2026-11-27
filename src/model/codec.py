"""
ECG encoder / decoder base classes and lead-wise / multi-scale residual implementations.
"""
import math
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


# ---------------------------------------------------------------------------
# Multi-scale cross-lead residual encoder & smooth decoder (DPSOM-ECG v2)
# ---------------------------------------------------------------------------

def _gn_groups(channels: int, target_groups: int = 8) -> int:
    return math.gcd(channels, target_groups) or 1


class _ResBlock1D(nn.Module):
    """Pre-act / post-SE residual 1D convolution block with optional downsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_1: int = 5,
        stride: int = 1,
        dilation_1: int = 1,
        kernel_2: int = 3,
        dilation_2: int = 1,
        se_reduction: int = 8,
    ):
        super().__init__()
        pad1 = ((kernel_1 - 1) * dilation_1) // 2
        pad2 = ((kernel_2 - 1) * dilation_2) // 2
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_1, stride=stride, padding=pad1, dilation=dilation_1,
        )
        self.norm1 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_2, stride=1, padding=pad2, dilation=dilation_2,
        )
        self.norm2 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.se = SEBlock(out_channels, reduction=se_reduction)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(_gn_groups(out_channels), out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.se(self.norm2(self.conv2(h)))
        return F.silu(h + self.skip(x))


class MultiScaleResConvEncoder(ECGEncoderBase):
    """
    Multi-scale lead-wise stem + cross-lead residual dilated backbone +
    optional dual adaptive (avg + max) temporal segment pooling.
    """

    def __init__(
        self,
        input_channels: int,
        input_length: int,
        base_channels: int = 48,
        stage_channels: int = 64,
        pool_bins: int | None = None,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.input_length = input_length
        self.pool_bins = pool_bins

        # Parallel lead-wise filters: narrow (QRS ~50 ms) + wide (P/ST/T ~150 ms)
        stem_short_ch = input_channels * 2
        stem_long_ch = input_channels * 2
        self.stem_short = nn.Conv1d(
            input_channels, stem_short_ch, kernel_size=5, padding=2, groups=input_channels,
        )
        self.stem_long = nn.Conv1d(
            input_channels, stem_long_ch, kernel_size=15, padding=7, groups=input_channels,
        )
        self.stem_mix = nn.Sequential(
            nn.Conv1d(stem_short_ch + stem_long_ch, base_channels, kernel_size=1),
            nn.GroupNorm(_gn_groups(base_channels), base_channels),
            nn.SiLU(),
        )

        self.stage1 = _ResBlock1D(base_channels, base_channels, kernel_1=7, stride=2, dilation_1=1, kernel_2=5, dilation_2=2)
        self.stage2 = _ResBlock1D(base_channels, stage_channels, kernel_1=5, stride=2, dilation_1=1, kernel_2=5, dilation_2=2)
        self.stage3 = _ResBlock1D(stage_channels, stage_channels, kernel_1=5, stride=1, dilation_1=4, kernel_2=3, dilation_2=1)

        s1_len = ((input_length + 2 * 3 - 7) // 2) + 1
        s2_len = ((s1_len + 2 * 2 - 5) // 2) + 1
        if pool_bins is not None:
            self.avg_pool = nn.AdaptiveAvgPool1d(pool_bins)
            self.max_pool = nn.AdaptiveMaxPool1d(pool_bins)
            self.out_length = pool_bins * 2
        else:
            self.avg_pool = None
            self.max_pool = None
            self.out_length = s2_len

        self.out_channels = stage_channels
        self.feature_dim = self.out_channels * self.out_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([self.stem_short(x), self.stem_long(x)], dim=1)
        h = self.stem_mix(h)
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        if self.pool_bins is None or self.avg_pool is None or self.max_pool is None:
            return h  # [B, stage_channels, T//4]
        return torch.cat([self.avg_pool(h), self.max_pool(h)], dim=-1)


class _ResDecoderStage(nn.Module):
    """Upsampled residual conv + SE + FiLM conditioning stage for smooth ECG reconstruction."""

    def __init__(self, in_channels: int, out_channels: int, z_dim: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(_gn_groups(out_channels), out_channels)
        self.se = SEBlock(out_channels, reduction=8)
        self.film = FiLMLayer(z_dim, out_channels)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.film(self.se(self.norm2(self.conv2(h))), z)
        return F.silu(h + self.skip(x))


class ResidualECGDecoder(ECGDecoderBase):
    """
    Smooth cross-lead residual FiLM decoder using linear interpolation upsampling
    to eliminate nearest-neighbor staircase artifacts on reconstructed ECG beats.
    """

    def __init__(
        self,
        z_dim: int,
        input_channels: int,
        input_length: int,
        conv_out_len: int | None = None,
        dropout: float = 0.2,
        base_channels: int = 64,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.input_channels = input_channels
        self.input_length = input_length
        self.conv_out_len = int(conv_out_len) if conv_out_len is not None else max(1, input_length // 4)
        self.base_channels = base_channels

        self.dropout = nn.Dropout(p=dropout)
        self.dec_fc = nn.Linear(z_dim, base_channels * self.conv_out_len)

        mid_channels = max(32, base_channels // 2)
        self.stage1 = _ResDecoderStage(base_channels, base_channels, z_dim, kernel_size=7)
        self.stage2 = _ResDecoderStage(base_channels, mid_channels, z_dim, kernel_size=7)
        self.refine = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size=5, padding=2),
            nn.GroupNorm(_gn_groups(mid_channels), mid_channels),
            nn.SiLU(),
            nn.Conv1d(mid_channels, input_channels, kernel_size=5, padding=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        h = F.silu(self.dropout(self.dec_fc(z))).view(B, self.base_channels, self.conv_out_len)

        h = F.interpolate(h, scale_factor=2, mode="linear", align_corners=False)
        h = self.stage1(h, z)

        h = F.interpolate(h, scale_factor=2, mode="linear", align_corners=False)
        h = self.stage2(h, z)

        out = self.refine(h)
        if out.shape[-1] != self.input_length:
            out = F.interpolate(out, size=self.input_length, mode="linear", align_corners=False)
        return out.reshape(B, -1)  # [B, C*T]
