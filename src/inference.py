"""Converter and inference pipeline for predicting on new 12-lead ECG data.

Supports raw NumPy arrays, `.npy` / `.npz`, `.csv`, and WFDB (`.hea` / `.dat`) records
at any sampling rate (automatically resampled to 100 Hz).

CLI examples
------------
Predict on a single ECG file (WFDB, .npy, .npz, or .csv):
    python -m src.inference --ckpt models/.../model_best.ckpt --input path/to/ecg.npy --fs 500
    python -m src.inference --ckpt models/.../model_best.ckpt --input path/to/00001_hr

Convert a directory of ECG files into an Explorer-compatible ECG_Dataset pickle:
    python -m src.inference --convert-dir path/to/ecgs/ --output data/processed/custom_ecgs.pkl --fs 500
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import Counter
from math import gcd
from typing import Optional, Sequence

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch
from scipy.signal import resample_poly

from src.data.dataset import ECG_Dataset
from src.data.generator import ECG_DataGenerator
from src.data.record import ECG_Record
from src.model.dpsom import DPSOM_ECG
from src.training.trainer import load_dpsom_checkpoint

TARGET_FS = 100
TARGET_BEAT_LEN = 120
STANDARD_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


# ---------------------------------------------------------------------------
# Signal loading & conversion
# ---------------------------------------------------------------------------

def load_raw_ecg(path: str, fs: Optional[float] = None) -> tuple[np.ndarray, float]:
    """Load a 12-lead ECG from WFDB, .npy, .npz, or .csv -> (signal [T, 12], fs)."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".hea", ".dat") or (ext == "" and os.path.exists(path + ".hea")):
        import wfdb
        stem = os.path.splitext(path)[0] if ext in (".hea", ".dat") else path
        sig, meta = wfdb.rdsamp(stem)
        detected_fs = float(meta.get("fs", fs or TARGET_FS))
        return np.asarray(sig, dtype=np.float32), float(fs or detected_fs)

    if ext == ".npy":
        arr = np.load(path)
        return np.asarray(arr, dtype=np.float32), float(fs or TARGET_FS)

    if ext == ".npz":
        z = np.load(path)
        key = "signal" if "signal" in z else list(z.keys())[0]
        arr = z[key]
        detected_fs = float(z["fs"]) if "fs" in z else float(fs or TARGET_FS)
        return np.asarray(arr, dtype=np.float32), float(fs or detected_fs)

    if ext == ".csv":
        import pandas as pd
        df = pd.read_csv(path)
        # If first row was actually numeric (no header), reload with header=None
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 12:
            df = pd.read_csv(path, header=None)
            num_df = df.select_dtypes(include=[np.number])
        return num_df.to_numpy(dtype=np.float32), float(fs or TARGET_FS)

    raise ValueError(f"Unsupported ECG file format: '{path}' (supported: WFDB .hea/.dat, .npy, .npz, .csv)")


def normalize_signal_layout(
    signal: np.ndarray,
    fs: float = TARGET_FS,
    lead_names: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Convert raw ECG array to float32 [T, 12] at 100 Hz in standard lead order."""
    sig = np.asarray(signal, dtype=np.float32)
    if sig.ndim == 3 and sig.shape[0] == 1:
        sig = sig[0]
    if sig.ndim != 2:
        raise ValueError(f"Expected 2D ECG array ([T, 12] or [12, T]), got shape {sig.shape}")

    if sig.shape[1] != 12 and sig.shape[0] == 12:
        sig = sig.T
    if sig.shape[1] != 12:
        raise ValueError(f"Expected 12 ECG leads, got shape {sig.shape}")

    sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)

    if lead_names is not None:
        upper = [str(n).strip().upper() for n in lead_names]
        if set(STANDARD_LEADS).issubset(set(upper)):
            idx = [upper.index(ld) for ld in STANDARD_LEADS]
            sig = sig[:, idx]

    fs_int = int(round(float(fs)))
    if fs_int > 0 and fs_int != TARGET_FS:
        g = gcd(TARGET_FS, fs_int)
        sig = resample_poly(sig, up=TARGET_FS // g, down=fs_int // g, axis=0).astype(np.float32)

    return sig


def convert_signal_to_record(
    signal: np.ndarray,
    fs: float = TARGET_FS,
    record_id: str | int = "new_ecg",
    lead_names: Optional[Sequence[str]] = None,
    age: float = 0.0,
    sex: int = 0,
    label: str = "UNKNOWN",
    fold: int = 10,
) -> ECG_Record:
    """Convert a raw 12-lead ECG array into a segmented, normalized ECG_Record."""
    sig_100 = normalize_signal_layout(signal, fs=fs, lead_names=lead_names)
    rec = ECG_Record(
        id=record_id,
        signal=sig_100,
        age=age,
        sex=sex,
        label=label,
        fold=fold,
        fs=TARGET_FS,
    )
    rec.detect_peaks(0)
    if not rec.segment_beats(target_length=TARGET_BEAT_LEN):
        rec.is_valid = False
        rec.reason = "Segmentation failed"
        raise ValueError(
            f"Beat segmentation failed for '{record_id}' (detected {len(rec.r_peaks)} R-peaks; need >= 3)."
        )
    rec.normalize()
    return rec


def convert_directory_to_dataset(
    input_dir: str,
    output_pkl: str,
    fs: Optional[float] = None,
    label: str = "UNKNOWN",
) -> ECG_Dataset:
    """Convert all WFDB / .npy / .npz / .csv files in a directory into an Explorer-compatible ECG_Dataset."""
    patterns = ["*.hea", "*.npy", "*.npz", "*.csv"]
    files: list[str] = []
    for pat in patterns:
        files.extend(sorted(glob.glob(os.path.join(input_dir, "**", pat), recursive=True)))

    if not files:
        raise FileNotFoundError(f"No supported ECG files found in {input_dir}")

    ds = ECG_Dataset(fs=TARGET_FS)
    failed = 0
    for fpath in files:
        rec_id = os.path.splitext(os.path.basename(fpath))[0]
        try:
            sig, det_fs = load_raw_ecg(fpath, fs=fs)
            rec = convert_signal_to_record(sig, fs=det_fs, record_id=rec_id, label=label, fold=10)
            ds.add_record(rec)
        except Exception as exc:
            failed += 1
            print(f"[convert] Skipping {fpath}: {exc}")

    if not ds.records:
        raise RuntimeError("No valid ECG records could be converted.")

    ds.label_encoder.fit([r.label for r in ds.records])
    ds.class_names = list(ds.label_encoder.classes_)
    ds.save(output_pkl)
    print(f"[convert] Saved {len(ds.records)} records ({failed} skipped) -> {output_pkl}")
    return ds


# ---------------------------------------------------------------------------
# Model loading & prediction
# ---------------------------------------------------------------------------

def find_best_checkpoint(
    models_dir: str = "models",
    ablation_csv: str = os.path.join("ablation", "ablation_results_all.csv"),
) -> str:
    """Return the highest-NMI checkpoint from models/ (using ablation_results_all.csv).

    Falls back to the newest ``*_best.ckpt`` or newest ``*.ckpt`` if no ablation
    metrics match.
    """
    ckpts = sorted(glob.glob(os.path.join(models_dir, "**", "*.ckpt"), recursive=True))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under '{models_dir}'.")

    scores: dict[str, tuple[float, float]] = {}
    if os.path.exists(ablation_csv):
        with open(ablation_csv, newline="") as fh:
            for row in csv.DictReader(fh):
                var = (row.get("variant") or "").strip()
                try:
                    nmi = float(row["NMI"])
                    ami = float(row.get("AMI") or 0.0)
                except (KeyError, TypeError, ValueError):
                    continue
                scores[var] = (nmi, ami)

    best_ckpt: Optional[str] = None
    best_key: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    for p in ckpts:
        dirname = os.path.basename(os.path.dirname(p))
        if dirname.startswith("abl_"):
            m = re.search(r"_\d+_\d+-\d+_", dirname[4:])
            var = dirname[4:4 + m.start()] if m else None
            if var and var in scores:
                nmi, ami = scores[var]
                key = (nmi, ami, os.path.getmtime(p))
                if key > best_key:
                    best_key = key
                    best_ckpt = p

    if best_ckpt is not None:
        return best_ckpt

    best_named = [p for p in ckpts if p.endswith("_best.ckpt")]
    pool = best_named or ckpts
    return max(pool, key=os.path.getmtime)


def load_model(
    ckpt_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> tuple[DPSOM_ECG, dict]:
    """Load a trained DPSOM_ECG checkpoint (defaults to the best model when ckpt_path is None)."""
    ckpt_path = ckpt_path or find_best_checkpoint()
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_dpsom_checkpoint(ckpt_path, dev)


@torch.no_grad()
def predict_record(model: DPSOM_ECG, record: ECG_Record, crop: int = 10) -> dict:
    """Run age, sex, and toroidal SOM prediction on a preprocessed ECG_Record."""
    device = next(model.parameters()).device
    beats_np = record.beat_representations.astype(np.float32)  # [N, T, C]
    N, T, C = beats_np.shape

    beats = torch.from_numpy(beats_np.transpose(0, 2, 1)).unsqueeze(0).to(device)  # [1, N, C, T]
    beat_valid = torch.ones((1, N), dtype=torch.float32, device=device)

    meta_parts = [
        torch.from_numpy(record.beat_sigma.astype(np.float32)),
        torch.from_numpy(record.beat_amax.astype(np.float32)),
    ]
    if model.amp_context_dim > 2 * C:
        meta_parts.append(ECG_DataGenerator._compute_rr_features(record))
    amp_context = torch.cat(meta_parts, dim=1).unsqueeze(0).to(device)  # [1, N, D]

    _, z_age, z_sex, mu, _, _ = model.forward_batch(beats, beat_valid)
    age_r, sex_logits_r, attn_age, attn_sex = model.predict_dual_age(
        beats, beat_valid, amp_context, precomputed=(z_age, z_sex, mu)
    )

    k_beats = model.k(mu.reshape(N, -1)).cpu().numpy().tolist()
    _, W = model.som_dim
    cells = [{"row": int(k // W), "col": int(k % W)} for k in k_beats]

    dom_k = Counter(k_beats).most_common(1)[0][0]
    dom_cell = {"row": int(dom_k // W), "col": int(dom_k % W)}

    sex_prob_f = float(torch.sigmoid(sex_logits_r[0]).item())
    attn_a = attn_age[0].cpu().numpy().tolist()
    attn_s = attn_sex[0].cpu().numpy().tolist()

    c = max(0, min(int(crop), (T - 1) // 2))
    beats_crop = beats_np[:, c:T - c, :] if c > 0 else beats_np

    return {
        "record_id": str(record.id),
        "num_beats": int(N),
        "predicted_age": round(float(age_r[0].item()), 2),
        "predicted_sex": "Female" if sex_prob_f >= 0.5 else "Male",
        "predicted_sex_prob_female": round(sex_prob_f, 4),
        "dominant_cell": dom_cell,
        "beat_cells": [
            {
                "beat_idx": i,
                "row": cells[i]["row"],
                "col": cells[i]["col"],
                "attn_age": round(float(attn_a[i]), 4),
                "attn_sex": round(float(attn_s[i]), 4),
                "leads": [beats_crop[i, :, li].tolist() for li in range(C)],
            }
            for i in range(N)
        ],
    }


def predict_signal(
    model: DPSOM_ECG,
    signal: np.ndarray,
    fs: float = TARGET_FS,
    record_id: str | int = "new_ecg",
    lead_names: Optional[Sequence[str]] = None,
) -> dict:
    """Convert a raw 12-lead ECG array and predict age, sex, and SOM placement."""
    rec = convert_signal_to_record(signal, fs=fs, record_id=record_id, lead_names=lead_names)
    return predict_record(model, rec)


def predict_file(
    ckpt_path: Optional[str],
    input_path: str,
    fs: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load checkpoint (defaults to best model) + ECG file and return predictions."""
    model, meta = load_model(ckpt_path, device=device)
    sig, det_fs = load_raw_ecg(input_path, fs=fs)
    rec_id = os.path.splitext(os.path.basename(input_path))[0]
    res = predict_signal(model, sig, fs=det_fs, record_id=rec_id)
    res["ckpt_path"] = meta.get("ckpt_path", ckpt_path)
    return res


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Convert new 12-lead ECGs and predict age, sex & SOM placement")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (.ckpt); defaults to best model")
    p.add_argument("--input", type=str, help="Single ECG file (WFDB .hea/.dat, .npy, .npz, or .csv)")
    p.add_argument("--fs", type=float, default=None, help="Sampling rate in Hz (auto for WFDB, default 100)")
    p.add_argument("--convert-dir", type=str, help="Convert all ECGs in a directory into an ECG_Dataset pickle")
    p.add_argument("--output", type=str, default="data/processed/custom_ecgs.pkl", help="Output .pkl path for --convert-dir")
    args = p.parse_args()

    if args.convert_dir:
        convert_directory_to_dataset(args.convert_dir, args.output, fs=args.fs)
        return

    if not args.input:
        p.error("Provide --input for prediction (or --convert-dir for batch conversion).")

    res = predict_file(args.ckpt, args.input, fs=args.fs)
    summary = {k: v for k, v in res.items() if k != "beat_cells"}
    summary["beat_cells"] = [
        {k: v for k, v in b.items() if k != "leads"} for b in res["beat_cells"]
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
