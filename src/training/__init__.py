"""
src.training — Training and evaluation routines.

Exports
-------
train_model       Run the full multi-phase training pipeline.
evaluate_model    Evaluate a trained model on the test set.
main              Entry-point: load data, train, evaluate, visualize.
"""
from src.training.trainer import train_model, evaluate_model, main

__all__ = ["train_model", "evaluate_model", "main"]
