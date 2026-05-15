"""
src.data — ECG data loading and preprocessing.

Exports
-------
ECG_Record      Single ECG record with peak detection and beat segmentation.
ECG_Dataset     Collection of ECG records with I/O and PTB-XL import.
ECG_DataGenerator   Batch generator for model training.
calc_sqi_metrics    Signal quality index computation.
"""
from src.data.signal import calc_sqi_metrics
from src.data.record import ECG_Record
from src.data.dataset import ECG_Dataset
from src.data.generator import ECG_DataGenerator

__all__ = [
    "calc_sqi_metrics",
    "ECG_Record",
    "ECG_Dataset",
    "ECG_DataGenerator",
]
