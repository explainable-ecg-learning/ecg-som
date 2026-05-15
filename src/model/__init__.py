"""
src.model — Neural network components for DPSOM-ECG.

Exports
-------
SEBlock             Squeeze-and-Excitation block (1-D).
FiLMLayer           Feature-wise Linear Modulation layer.
ECGEncoderBase      Abstract base for ECG encoders.
ECGDecoderBase      Abstract base for ECG decoders.
LeadWiseConvEncoder Lead-wise depthwise convolutional encoder.
LeadWiseConvDecoder Lead-wise depthwise convolutional decoder.
DPSOM_ECG           Full disentangled probabilistic SOM model.
"""
from src.model.layers import SEBlock, FiLMLayer
from src.model.codec import ECGEncoderBase, ECGDecoderBase, LeadWiseConvEncoder, LeadWiseConvDecoder
from src.model.dpsom import DPSOM_ECG

__all__ = [
    "SEBlock",
    "FiLMLayer",
    "ECGEncoderBase",
    "ECGDecoderBase",
    "LeadWiseConvEncoder",
    "LeadWiseConvDecoder",
    "DPSOM_ECG",
]
