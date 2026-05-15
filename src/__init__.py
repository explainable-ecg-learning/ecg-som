"""
src — Refactored DisentangledECG package.

Subpackages
-----------
model       Core neural network (DPSOM_ECG, encoder/decoder, FiLM, SE).
data        Data loading and preprocessing (ECG_Record, ECG_Dataset, DataGenerator).
training    Training and evaluation loop (Trainer).
utils       Metrics and visualization helpers.
"""
from src.config import DPSOM_Config
from src.scheduler import ExponentialDecayScheduler

__all__ = ["DPSOM_Config", "ExponentialDecayScheduler"]
