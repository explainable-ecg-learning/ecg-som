"""Export compact per-split .npz files for fast ECG-SOM Explorer loading.

Replaces unpickling the 3.6 GB ptbxl_100_T-12ms.pkl with lightweight float16
per-split archives (~60 MB for test, ~50 MB for val, ~400 MB for train).

Usage:
    python explorer/export_compact_data.py [--input data/ptbxl_100_T-12ms.pkl] [--out-dir data]
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def export_splits(pkl_path: str, out_dir: str) -> None:
    t0 = time.time()
    print(f"[export] Loading {pkl_path} ...")
    with open(pkl_path, "rb") as f:
        dataset = pickle.load(f)
    print(f"[export] Loaded pickle in {time.time() - t0:.1f}s")

    class_names = list(getattr(dataset, "class_names", ["CD", "HYP", "MI", "NORM", "STTC"]))
    label_to_idx = {str(name): i for i, name in enumerate(class_names)}
    os.makedirs(out_dir, exist_ok=True)

    for split in ("test", "val", "train"):
        ts = time.time()
        records = dataset.get_data(split)
        beats_list, labels_list, ages_list, sexes_list = [], [], [], []
        rec_ids, beat_local = [], []

        for rec in records:
            if rec.beat_representations is None:
                continue
            nb = len(rec.beat_representations)
            beats_list.append(rec.beat_representations.astype(np.float16))
            lbl = label_to_idx.get(str(rec.label), -1)
            labels_list.extend([lbl] * nb)
            ages_list.extend([float(rec.age)] * nb)
            sexes_list.extend([int(rec.sex)] * nb)
            rec_ids.extend([str(rec.id)] * nb)
            beat_local.extend(range(nb))

        beats_arr = np.concatenate(beats_list, axis=0)  # [N, T, C] float16
        out_path = os.path.join(out_dir, f"explorer_{split}.npz")
        np.savez(
            out_path,
            beats=beats_arr,
            labels=np.asarray(labels_list, dtype=np.int16),
            ages=np.asarray(ages_list, dtype=np.float32),
            sexes=np.asarray(sexes_list, dtype=np.int8),
            rec_ids=np.asarray(rec_ids),
            beat_local=np.asarray(beat_local, dtype=np.int16),
            class_names=np.asarray(class_names),
        )
        mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"[export] {split:<5}: {len(beats_arr):>6} beats -> {out_path} ({mb:.1f} MB, {time.time() - ts:.1f}s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export compact per-split .npz files for ECG-SOM Explorer")
    p.add_argument("--input", default=os.path.join(_REPO_ROOT, "data", "ptbxl_100_T-12ms.pkl"))
    p.add_argument("--out-dir", default=os.path.join(_REPO_ROOT, "data"))
    args = p.parse_args()
    export_splits(args.input, args.out_dir)
