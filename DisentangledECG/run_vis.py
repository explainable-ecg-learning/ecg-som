"""
Standalone visualization script for a saved DPSOM-ECG checkpoint.

Usage
-----
    python run_vis.py                                       # latest checkpoint in models/
    python run_vis.py --ckpt models/<exp>/<exp>.ckpt        # explicit checkpoint
    python run_vis.py --ckpt <path> --data data/ptbxl_100_T-12ms.pkl
    python run_vis.py --all-labels                          # include multi-label records
"""

import argparse
import glob
import os

os.environ.setdefault("MPLBACKEND", "Agg")

from dpsom_ecg import _get_device
from ECG_Dataset import ECG_Dataset
from data_generator import ECG_DataGenerator
from visual_utils import log_som_visualizations


def _latest_checkpoint() -> str:
    """Return the most recently modified .ckpt file under models/."""
    ckpts = glob.glob("models/**/*.ckpt", recursive=True)
    if not ckpts:
        raise FileNotFoundError("No checkpoints found under models/")
    return max(ckpts, key=os.path.getmtime)


def _ex_name_from_ckpt(ckpt_path: str) -> str:
    """Derive experiment name from the checkpoint path stem."""
    return os.path.splitext(os.path.basename(ckpt_path))[0]


def main():
    parser = argparse.ArgumentParser(description="Run SOM visualizations for a saved checkpoint")
    parser.add_argument("--ckpt", default=None,
                        help="Path to .ckpt file (default: latest under models/)")
    parser.add_argument("--data", default="./data/ptbxl_100_T-12ms.pkl",
                        help="Path to cached ECG_Dataset pickle")
    parser.add_argument("--all-labels", action="store_true",
                        help="Include multi-label records (default: single-label only)")
    args = parser.parse_args()

    ckpt = args.ckpt or _latest_checkpoint()
    ex_name = _ex_name_from_ckpt(ckpt)
    single_label_only = not args.all_labels

    print(f"Checkpoint : {ckpt}")
    print(f"Experiment : {ex_name}")
    print(f"Dataset    : {args.data}")
    print(f"Single-label only: {single_label_only}")

    device = _get_device()
    ds = ECG_Dataset.load(args.data)
    gen = ECG_DataGenerator(ds)
    log_som_visualizations(ds, gen, ex_name, ckpt, device, single_label_only=single_label_only)
    print("Done.")


if __name__ == "__main__":
    main()
