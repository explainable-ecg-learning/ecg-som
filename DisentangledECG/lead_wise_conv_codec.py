# lead_wise_conv_codec.py
import torch
from torch import nn
import torch.nn.functional as F

from codec_base import ECGEncoderBase, ECGDecoderBase
from film_layer import FiLMLayer
from se_block import SEBlock


class LeadWiseConvEncoder(ECGEncoderBase):
    """
    Lead-wise convolutional encoder.
    Returns feature map [B, C_enc, L_enc].
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

        def calc_conv_output_length(length, kernel_size=kernel_size, stride=stride, padding=padding):
            return ((length + 2 * padding - kernel_size) // stride) + 1

        L = calc_conv_output_length(input_length)
        L = calc_conv_output_length(L)
        self.out_length = L

        filters_per_lead_1 = base_channels_1 // input_channels
        if filters_per_lead_1 < 1:
            filters_per_lead_1 = 1
        out_channels_1 = filters_per_lead_1 * input_channels

        filters_per_lead_2 = base_channels_2 // input_channels
        if filters_per_lead_2 < 1:
            filters_per_lead_2 = 1
        out_channels_2 = filters_per_lead_2 * input_channels

        self.out_channels = out_channels_2
        self.feature_dim = self.out_channels * self.out_length

        self.enc_conv1 = nn.Conv1d(
            input_channels,
            out_channels_1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=input_channels,
        )
        self.enc_bn1 = nn.GroupNorm(input_channels, out_channels_1)
        self.enc_se1 = SEBlock(out_channels_1)

        self.enc_conv2 = nn.Conv1d(
            out_channels_1,
            out_channels_2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=input_channels,
        )
        self.enc_bn2 = nn.GroupNorm(input_channels, out_channels_2)
        self.enc_se2 = SEBlock(out_channels_2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc_conv1(x)
        h = self.enc_bn1(h)
        h = self.enc_se1(h)
        h = F.leaky_relu(h, 0.2)

        h = self.enc_conv2(h)
        h = self.enc_bn2(h)
        h = self.enc_se2(h)
        h = F.leaky_relu(h, 0.2)

        return h  # [B, out_channels, out_length]


class LeadWiseConvDecoder(ECGDecoderBase):
    """
    Decoder that takes a latent vector z_in (any meaning: morphology z or z_age)
    and reconstructs a waveform [B, C*T] using lead-wise conv + FiLM(z_in).
    """
    def __init__(
        self,
        z_dim: int,                 # latent dim used for FiLM conditioning
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

        self.dropout_layer = nn.Dropout(p=dropout)

        # latent -> feature map
        self.dec_fc = nn.Linear(z_dim, enc_out_channels * conv_out_len)
        self.dec_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        filters_per_lead_1 = base_channels_1 // input_channels
        if filters_per_lead_1 < 1:
            filters_per_lead_1 = 1
        out_channels_1 = filters_per_lead_1 * input_channels

        self.dec_conv1 = nn.Conv1d(
            enc_out_channels,
            out_channels_1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=input_channels,
        )
        self.dec_bn1 = nn.GroupNorm(input_channels, out_channels_1)
        self.dec_se1 = SEBlock(out_channels_1)
        self.dec_film1 = FiLMLayer(z_dim, out_channels_1)

        filters_per_lead_3 = base_channels_3 // input_channels
        if filters_per_lead_3 < 1:
            filters_per_lead_3 = 1
        out_channels_3 = filters_per_lead_3 * input_channels

        self.dec_conv2 = nn.Conv1d(
            out_channels_1,
            out_channels_3,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=input_channels,
        )
        self.dec_bn2 = nn.GroupNorm(input_channels, out_channels_3)
        self.dec_se2 = SEBlock(out_channels_3)
        self.dec_film2 = FiLMLayer(z_dim, out_channels_3)

        self.dec_conv_out = nn.Conv1d(
            out_channels_3,
            input_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=input_channels,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z_in: [B, z_dim]
        returns: [B, C*T]
        """
        B = z.size(0)

        h = self.dec_fc(z)
        h = F.leaky_relu(h, 0.2)
        h = self.dropout_layer(h)
        h = h.view(B, self.enc_out_channels, self.conv_out_len)

        h = self.dec_upsample(h)
        h = self.dec_conv1(h)
        h = self.dec_bn1(h)
        h = self.dec_se1(h)
        h = self.dec_film1(h, z)
        h = F.leaky_relu(h, 0.2)

        h = self.dec_upsample(h)
        h = self.dec_conv2(h)
        h = self.dec_bn2(h)
        h = self.dec_se2(h)
        h = self.dec_film2(h, z)
        h = F.leaky_relu(h, 0.2)

        h = self.dec_conv_out(h)
        return h.reshape(B, -1)
