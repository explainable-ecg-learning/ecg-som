"""ECG-SOM Explorer — FastAPI backend.

Launch (from repo root):
    python explorer/server.py [--port 8050] [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import pickle
import sys
import time
import traceback

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.data.dataset import ECG_Dataset
from src.data.generator import ECG_DataGenerator
from src.inference import (
    find_best_checkpoint as _find_best_checkpoint,
    load_model as _load_dpsom_model,
    load_raw_ecg,
    predict_signal,
)
from src.model.dpsom import DPSOM_ECG
from src.training.trainer import _get_device, evaluate_model, load_dpsom_checkpoint
from src.training.uda import (
    adapt_uda_bw_ot,
    adapt_uda_ot_strong,
    adapt_uda_sf_strong,
    extract_domain_moments,
)
from src.utils.metrics import compute_som_clustering_metrics
from explorer.metadata import (
    FIGURE_MAP as _FIGURE_MAP,
    REGIME_META as _REGIME_META,
    build_overview_payload,
    build_uda_results_payload,
    compute_uda_overall_means as _compute_uda_overall_means,
    safe_float as _safe_float,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS              = 100.0
MAX_SAMPLES     = 20
CLASS_NAMES_DEF = ["CD", "HYP", "MI", "NORM", "STTC"]
_STATIC_DIR     = os.path.join(os.path.dirname(__file__), "static")
_CACHE_DIR      = os.path.join(os.path.dirname(__file__), "cache")
_DATA_DIR       = os.path.realpath(os.path.join(_REPO_ROOT, "data"))
_MODELS_DIR     = os.path.realpath(os.path.join(_REPO_ROOT, "models"))
_ABLATION_DIR   = os.path.realpath(os.path.join(_REPO_ROOT, "ablation"))
_DOCS_DIR       = os.path.realpath(os.path.join(_REPO_ROOT, "docs"))
_STATE: dict    = {}
_SPLIT_CACHE: dict[tuple[str, str], dict] = {}
_DATASET_CACHE: dict[str, object] = {}
_MOMENTS_CACHE: dict[tuple[str, float, str], dict] = {}
_UDA_STATE: dict = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="ECG-SOM Explorer")
app.add_middleware(GZipMiddleware, minimum_size=1000)
_cors_env = os.environ.get("EXPLORER_CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_ckpts() -> list[str]:
    return sorted(glob.glob(os.path.join(_MODELS_DIR, "**", "*.ckpt"), recursive=True))


def _discover_datasets() -> tuple[list[dict], str]:
    """List selectable dataset files strictly inside the data/ subdirectory."""
    options: list[dict] = []
    compact_splits = ("explorer_test.npz", "explorer_val.npz", "explorer_train.npz")
    compact_candidates = [
        ("explorer/explorer_test.npz", os.path.join(_DATA_DIR, "explorer", "explorer_test.npz")),
        ("explorer_test.npz", os.path.join(_DATA_DIR, "explorer_test.npz")),
    ]
    for rel_val, abs_path in compact_candidates:
        if os.path.isfile(abs_path):
            options.append({
                "label": "PTB-XL (compact .npz — fast)",
                "value": rel_val,
            })
            break

    scan_dirs = [
        ("processed", os.path.join(_DATA_DIR, "processed")),
        ("explorer", os.path.join(_DATA_DIR, "explorer")),
        ("", _DATA_DIR),
    ]
    seen: set[str] = set()
    for prefix, dpath in scan_dirs:
        if not os.path.isdir(dpath):
            continue
        for fname in sorted(os.listdir(dpath)):
            if not fname.endswith((".npz", ".pkl")):
                continue
            if fname in compact_splits:
                continue
            fpath = os.path.join(dpath, fname)
            if not os.path.isfile(fpath):
                continue
            rel = f"{prefix}/{fname}" if prefix else fname
            if rel in seen:
                continue
            seen.add(rel)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            size_str = f"{size_mb / 1024:.1f} GB" if size_mb >= 1024 else f"{size_mb:.0f} MB"
            options.append({
                "label": f"{rel}  ({size_str})",
                "value": rel,
            })

    default_val = options[0]["value"] if options else ""
    return options, default_val


def _discover_pkl_datasets() -> list[dict]:
    """List processed .pkl cohorts inside data/ suitable for live UDA."""
    out: list[dict] = []
    proc_dir = os.path.join(_DATA_DIR, "processed")
    if not os.path.isdir(proc_dir):
        return out
    for fname in sorted(os.listdir(proc_dir)):
        if not fname.endswith(".pkl"):
            continue
        fpath = os.path.join(proc_dir, fname)
        if not os.path.isfile(fpath):
            continue
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        size_str = f"{size_mb / 1024:.1f} GB" if size_mb >= 1024 else f"{size_mb:.0f} MB"
        rel = f"processed/{fname}"
        out.append({"label": f"{rel} ({size_str})", "value": rel})
    return out


def _resolve_data_path(data_name: str) -> str:
    """Resolve a dataset filename strictly inside data/ (i.e. /app/data in Docker)."""
    cleaned = (data_name or "").strip()
    if not cleaned:
        raise HTTPException(400, "Dataset name is empty.")
    if os.path.isabs(cleaned):
        real = os.path.realpath(cleaned)
    else:
        rel = cleaned[5:] if cleaned.startswith("data/") else cleaned
        real = os.path.realpath(os.path.join(_DATA_DIR, rel))
    if real != _DATA_DIR and not real.startswith(_DATA_DIR + os.sep):
        raise HTTPException(400, "Dataset path must be inside the data/ directory.")
    if not os.path.isfile(real):
        raise HTTPException(404, f"Dataset file not found in data/: {os.path.basename(real)}")
    return real


def _resolve_ckpt_path(ckpt_path: str) -> str:
    """Resolve a checkpoint filename strictly inside models/."""
    cleaned = (ckpt_path or "").strip()
    if not cleaned:
        raise HTTPException(400, "Checkpoint path is empty.")
    if os.path.isabs(cleaned):
        real = os.path.realpath(cleaned)
    else:
        rel = cleaned[7:] if cleaned.startswith("models/") else cleaned
        real = os.path.realpath(os.path.join(_MODELS_DIR, rel))
    if real != _MODELS_DIR and not real.startswith(_MODELS_DIR + os.sep):
        raise HTTPException(400, "Checkpoint path must be inside the models/ directory.")
    if not os.path.isfile(real):
        raise HTTPException(404, f"Checkpoint file not found in models/: {os.path.basename(real)}")
    return real


def _load_ablation_index() -> dict[str, dict]:
    """Return {variant_name: {NMI, AMI, Purity, MAE_Age, AUC_Sex}} from ablation_results_all.csv."""
    csv_path = os.path.join(_ABLATION_DIR, "ablation_results_all.csv")
    if not os.path.exists(csv_path):
        return {}
    try:
        import csv
        index: dict[str, dict] = {}
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                variant = row.get("variant", "").strip()
                if not variant:
                    continue
                index[variant] = {
                    "NMI":     _safe_float(row.get("NMI")),
                    "AMI":     _safe_float(row.get("AMI")),
                    "Purity":  _safe_float(row.get("Purity")),
                    "MAE_Age": _safe_float(row.get("MAE_Age")),
                    "AUC_Sex": _safe_float(row.get("AUC_Sex")),
                }
        return index
    except Exception:
        return {}


def _variant_from_path(ckpt_path: str) -> str | None:
    import re
    dirname = os.path.basename(os.path.dirname(ckpt_path))
    if not dirname.startswith("abl_"):
        return None
    body = dirname[4:]
    m = re.search(r"_\d+_\d+-\d+_", body)
    return body[: m.start()] if m else None


_CACHE_SCHEMA_VERSION = "v4"  # v4: include ckpt mtime + canonical strict loader


def _cache_path(ckpt_path: str, data_path: str, split: str) -> str:
    mtime = os.path.getmtime(ckpt_path) if os.path.exists(ckpt_path) else 0.0
    key = hashlib.sha256(
        f"{_CACHE_SCHEMA_VERSION}|{ckpt_path}|{mtime:.3f}|{data_path}|{split}".encode()
    ).hexdigest()[:16]
    cache_dir = _CACHE_DIR
    try:
        os.makedirs(cache_dir, exist_ok=True)
        probe = os.path.join(cache_dir, ".write_test")
        with open(probe, "wb"):
            pass
        os.unlink(probe)
    except OSError:
        cache_dir = "/tmp/ecg-som-explorer-cache"
        os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{key}.pkl")


def _load_cache(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except OSError:
        return None


def _save_cache(path: str, payload: dict) -> None:
    try:
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError as exc:
        print(f"[explorer] Warning: could not write cache {path}: {exc}")


def _resolve_compact_npz(data_path: str, split: str) -> list[str] | None:
    """Return matching compact explorer_<split>.npz file(s) inside data/ when selected."""
    fname = os.path.basename(data_path)
    base_dir = os.path.dirname(data_path)
    if fname.startswith("explorer_") and fname.endswith(".npz"):
        if split == "all":
            paths = [os.path.join(base_dir, f"explorer_{s}.npz") for s in ("train", "val", "test")]
            return paths if all(os.path.exists(p) for p in paths) else None
        cand = os.path.join(base_dir, f"explorer_{split}.npz")
        return [cand] if os.path.exists(cand) else [data_path]
    if data_path.endswith(".npz") and os.path.exists(data_path):
        return [data_path]
    return None


def _load_pkl_dataset(data_path: str) -> ECG_Dataset:
    abs_pkl = os.path.abspath(data_path)
    dataset = _DATASET_CACHE.get(abs_pkl)
    if dataset is None:
        with open(abs_pkl, "rb") as f:
            dataset = pickle.load(f)
        _DATASET_CACHE[abs_pkl] = dataset
    return dataset  # type: ignore[return-value]


def _dataset_class_names(dataset: object) -> list[str]:
    names = [str(c) for c in (getattr(dataset, "class_names", None) or [])]
    if not names and hasattr(dataset, "label_encoder") and hasattr(dataset.label_encoder, "classes_"):
        names = [str(c) for c in dataset.label_encoder.classes_]
    return names if names else list(CLASS_NAMES_DEF)


def _records_to_split_arrays(records: list, class_names: list[str], data_label: str, split: str) -> dict:
    label_to_idx = {str(name): i for i, name in enumerate(class_names)}
    beats_list, labels_list, ages_list, sexes_list = [], [], [], []
    rec_ids, beat_local = [], []

    for rec in records:
        if rec.beat_representations is None:
            continue
        for i in range(len(rec.beat_representations)):
            beats_list.append(rec.beat_representations[i])
            labels_list.append(label_to_idx.get(str(rec.label), -1))
            ages_list.append(rec.age)
            sexes_list.append(rec.sex)
            rec_ids.append(str(rec.id))
            beat_local.append(i)

    if not beats_list:
        raise HTTPException(
            400,
            f"Split '{split}' has no records in {os.path.basename(data_label)} (try 'test' or 'all').",
        )

    return {
        "beats":       np.stack(beats_list).astype(np.float16),
        "labels":      np.asarray(labels_list, dtype=np.int64),
        "ages":        np.asarray(ages_list, dtype=np.float32),
        "sexes":       np.asarray(sexes_list, dtype=np.int64),
        "rec_ids":     np.asarray(rec_ids),
        "beat_local":  np.asarray(beat_local, dtype=np.int64),
        "class_names": class_names,
    }


def _load_split_arrays(data_path: str, split: str) -> dict:
    """Load beat arrays for a split from compact .npz (preferred) or .pkl, cached in RAM."""
    cache_key = (os.path.abspath(data_path), split)
    if cache_key in _SPLIT_CACHE:
        return _SPLIT_CACHE[cache_key]

    npz_paths = _resolve_compact_npz(data_path, split)
    if npz_paths:
        parts = [np.load(p, allow_pickle=False) for p in npz_paths]
        payload = {
            "beats":       np.concatenate([z["beats"] for z in parts], axis=0) if len(parts) > 1 else parts[0]["beats"],
            "labels":      np.concatenate([z["labels"] for z in parts], axis=0).astype(np.int64),
            "ages":        np.concatenate([z["ages"] for z in parts], axis=0).astype(np.float32),
            "sexes":       np.concatenate([z["sexes"] for z in parts], axis=0).astype(np.int64),
            "rec_ids":     np.concatenate([z["rec_ids"] for z in parts], axis=0),
            "beat_local":  np.concatenate([z["beat_local"] for z in parts], axis=0).astype(np.int64),
            "class_names": [str(c) for c in parts[0]["class_names"].tolist()],
        }
        _SPLIT_CACHE[cache_key] = payload
        return payload

    dataset = _load_pkl_dataset(data_path)
    class_names = _dataset_class_names(dataset)
    records = dataset.get_data(split)
    payload = _records_to_split_arrays(records, class_names, data_path, split)
    _SPLIT_CACHE[cache_key] = payload
    return payload


def _compute_cell_detail(row: int, col: int) -> dict:
    """Lazily compute percentile bands and sample beats for a single clicked SOM cell."""
    H, W = _STATE["som_dim"]
    flat = row * W + col
    beat_indices = _STATE["cell_beats"][flat]
    n_total = len(beat_indices)
    crop = _STATE["crop"]
    input_length = _STATE["input_length"]
    input_channels = _STATE["input_channels"]
    proto = _STATE["prototype_signals"][row, col]
    proto_leads = [proto[:, li].tolist() for li in range(input_channels)]

    split_data  = _load_split_arrays(_STATE["data_path"], _STATE["split"])
    beats_np    = split_data["beats"]
    labels_enc  = split_data["labels"]
    ages_np     = split_data["ages"]
    sexes_np    = split_data["sexes"]
    rec_ids     = split_data["rec_ids"]
    beat_local  = split_data["beat_local"]
    class_names = split_data["class_names"]

    percentiles = None
    if n_total > 0:
        cb = beats_np[beat_indices, crop:input_length - crop, :].astype(np.float32)
        pcts = np.percentile(cb, [5, 25, 50, 75, 95], axis=0)
        percentiles = {
            "p5":  [pcts[0, :, li].tolist() for li in range(input_channels)],
            "p25": [pcts[1, :, li].tolist() for li in range(input_channels)],
            "p50": [pcts[2, :, li].tolist() for li in range(input_channels)],
            "p75": [pcts[3, :, li].tolist() for li in range(input_channels)],
            "p95": [pcts[4, :, li].tolist() for li in range(input_channels)],
        }

    samples = []
    for gi in beat_indices[:MAX_SAMPLES]:
        beat = beats_np[gi, crop:input_length - crop, :].astype(np.float32)
        lbl_enc = int(labels_enc[gi])
        samples.append({
            "leads":     [beat[:, li].tolist() for li in range(input_channels)],
            "record_id": str(rec_ids[gi]),
            "beat_idx":  int(beat_local[gi]),
            "label":     class_names[lbl_enc] if 0 <= lbl_enc < len(class_names) else "?",
            "age":       float(ages_np[gi]),
            "sex":       "F" if int(sexes_np[gi]) == 1 else "M",
        })

    return {
        "row":         row,
        "col":         col,
        "n_total":     n_total,
        "n_shown":     len(samples),
        "proto_leads": proto_leads,
        "percentiles": percentiles,
        "samples":     samples,
        "t_axis":      _STATE["t_axis"],
    }


def _run_inference_with_model(
    model: DPSOM_ECG,
    meta: dict,
    ckpt_path: str,
    data_path: str,
    split: str,
    *,
    dataset_override: ECG_Dataset | None = None,
    info_prefix: str | None = None,
) -> dict:
    """Compute SOM assignments, prototypes, and state from an already-loaded model."""
    device = torch.device("cpu")
    model = copy.deepcopy(model).to(device)
    model.eval()

    if dataset_override is not None:
        class_names = _dataset_class_names(dataset_override)
        split_data = _records_to_split_arrays(
            dataset_override.get_data(split), class_names, data_path, split
        )
        _SPLIT_CACHE[(os.path.abspath(data_path), split)] = split_data
    else:
        split_data = _load_split_arrays(data_path, split)

    beats_np    = split_data["beats"]
    labels_enc  = split_data["labels"]
    class_names = split_data["class_names"]

    latent_dim     = int(meta["latent_dim"])
    som_dim        = tuple(int(v) for v in meta["som_dim"])
    input_length   = int(meta["input_length"])
    input_channels = int(meta["input_channels"])
    H, W = som_dim

    k_list = []
    with torch.inference_mode():
        for s in range(0, len(beats_np), 512):
            xb = torch.from_numpy(beats_np[s:s + 512].transpose(0, 2, 1).astype(np.float32)).to(device)
            mu, _, _, _ = model._encode(xb)
            k_list.append(model.k(mu).cpu().numpy())
    k_all = np.concatenate(k_list)

    CROP = 10
    with torch.inference_mode():
        E_flat = model._embeddings.view(-1, latent_dim)
        logits = model.decoder(E_flat)
        proto  = logits.view(H, W, input_channels, input_length).cpu().numpy()
    crop = max(0, min(CROP, (input_length - 1) // 2))
    proto_cropped = proto[:, :, :, crop:input_length - crop] if crop > 0 else proto
    prototype_signals = proto_cropped.transpose(0, 1, 3, 2)
    t_axis = (np.arange(proto_cropped.shape[-1]) / FS).tolist()

    K           = H * W
    num_classes = len(class_names)
    cell_beats: dict[int, list[int]] = {k: [] for k in range(K)}
    counts = np.zeros((K, num_classes), dtype=np.int64)
    for i, ki in enumerate(k_all):
        cell_beats[int(ki)].append(i)
        lbl = int(labels_enc[i])
        if 0 <= lbl < num_classes:
            counts[int(ki), lbl] += 1

    totals        = counts.sum(axis=1)
    cell_majority = np.full(K, -1, dtype=np.int64)
    nonempty      = totals > 0
    if bool(nonempty.any()) and num_classes > 0:
        cell_majority[nonempty] = counts[nonempty].argmax(axis=1)
    cell_predicted = cell_majority

    cells = []
    for r in range(H):
        for c in range(W):
            flat = r * W + c
            maj  = int(cell_majority[flat])
            pred = int(cell_predicted[flat])
            proto_rc = prototype_signals[r, c]
            cells.append({
                "row":             r,
                "col":             c,
                "count":           int(totals[flat]),
                "majority":        maj,
                "majority_label":  class_names[maj] if 0 <= maj < num_classes else "",
                "predicted":       pred,
                "predicted_label": class_names[pred] if 0 <= pred < num_classes else "",
                "class_counts":    counts[flat].tolist(),
                "proto_leads":     [proto_rc[:, li].tolist() for li in range(input_channels)],
            })

    cell_majority_map = {(c["row"], c["col"]): c["majority_label"] for c in cells}
    _STATE.clear()
    _STATE.update({
        "som_dim":           som_dim,
        "cell_beats":        cell_beats,
        "prototype_signals": prototype_signals,
        "t_axis":            t_axis,
        "crop":              crop,
        "input_length":      input_length,
        "input_channels":    input_channels,
        "data_path":         data_path,
        "split":             split,
        "ckpt_path":         ckpt_path,
        "model":             model,
        "embeddings":        model._embeddings.detach().float().cpu().numpy(),
        "cell_majority":     cell_majority_map,
        "cell_detail_cache": {},
    })

    head = info_prefix or f"Loaded: {os.path.basename(ckpt_path)}"
    return {
        "som_dim":     list(som_dim),
        "class_names": class_names,
        "cells":       cells,
        "info": (
            f"{head}  |  Split: {split}  |  Beats: {len(beats_np)}  |  SOM: {H}×{W}"
        ),
        "cache_hit": False,
    }


def _run_inference(ckpt_path: str, data_path: str, split: str) -> dict:
    model, meta = load_dpsom_checkpoint(ckpt_path, torch.device("cpu"))
    return _run_inference_with_model(model, meta, ckpt_path, data_path, split)


def _ensure_uda_splits(ds: ECG_Dataset) -> ECG_Dataset:
    """Return `ds` with non-empty train and test splits (deterministic 80/20 fallback if needed)."""
    if ds.get_data("train") and ds.get_data("test"):
        return ds
    out = ECG_Dataset(fs=ds.fs)
    out.class_names = list(ds.class_names)
    out.label_encoder = ds.label_encoder
    split_recs = list(ds.records)
    n_test = max(1, len(split_recs) // 5)
    for i, rec in enumerate(split_recs):
        rc = copy.copy(rec)
        rc.fold = 10 if i >= len(split_recs) - n_test else 1
        out.add_record(rc)
    return out


def _evaluate_model_for_explorer(model: DPSOM_ECG, ds: ECG_Dataset, use_rr: bool) -> dict:
    """Evaluate clustering, demographic, topology, and cell grid summary on target test split."""
    device = _get_device()
    model_eval = model.to(device)
    model_eval.eval()
    gen = ECG_DataGenerator(
        ds, batch_size=64, device=device, pin_memory=False,
        use_rr_features=use_rr, preload_to_gpu=(device.type == "cuda"),
    )
    res = evaluate_model(
        model_eval, gen, compute_disentanglement=False, compute_linear_probes=False,
        split="test", max_beats=None,
    )

    rec_gen = gen.get_record_batch_generator(mode="test", max_beats=None, shuffle=False)
    nb = gen.get_num_batches("test")
    z_res_all, lbl_all, k_all, y_idx_all = [], [], [], []
    with torch.inference_mode():
        for _ in range(nb):
            beats, _, _, beat_valid, _, _, _, labels_t = next(rec_gen)
            B, N, C, T = beats.shape
            valid = beat_valid.reshape(B * N) > 0.5
            if not bool(valid.any()):
                continue
            mu, _, _, _ = model_eval._encode(beats.reshape(B * N, C, T)[valid])
            z_res = model_eval.residualize_latent(mu.float()).cpu().numpy()
            k_ids = model_eval.k(mu).cpu().numpy()
            y_ids = labels_t.unsqueeze(1).expand(B, N).reshape(B * N)[valid].cpu().numpy()
            z_res_all.append(z_res)
            k_all.append(k_ids)
            y_idx_all.append(y_ids)
            lbl_all.append(y_ids)

    z_res_np = np.concatenate(z_res_all, axis=0)
    k_np = np.concatenate(k_all, axis=0).astype(np.int64)
    y_idx_np = np.concatenate(y_idx_all, axis=0).astype(np.int64)
    lbl_np = np.concatenate(lbl_all, axis=0).astype(str)

    H, W = tuple(int(v) for v in model_eval.som_dim)
    K = H * W
    codebook = model_eval._embeddings.detach().float().cpu().numpy().reshape(K, -1)
    som_m = compute_som_clustering_metrics(z_res_np, codebook, som_shape=(H, W), labels=lbl_np, knn_k=5)

    class_names = _dataset_class_names(ds)
    num_classes = len(class_names)
    counts = np.zeros((K, num_classes), dtype=np.int64)
    for ki, yi in zip(k_np, y_idx_np):
        if 0 <= ki < K and 0 <= yi < num_classes:
            counts[ki, yi] += 1

    cells = []
    for r in range(H):
        for c in range(W):
            flat = r * W + c
            cnt = int(counts[flat].sum())
            maj = int(counts[flat].argmax()) if cnt > 0 else -1
            cells.append({
                "row":            r,
                "col":            c,
                "count":          cnt,
                "majority":       maj,
                "majority_label": class_names[maj] if 0 <= maj < num_classes else "",
                "class_counts":   counts[flat].tolist(),
            })

    return {
        "metrics": {
            "Rec_AMI":      float(res["Rec_AMI"]),
            "NMI":          float(res["NMI"]),
            "Rec_Purity":   float(res["Rec_Purity"]),
            "MAE_Age":      float(res["MAE_Age"]),
            "AUC_Sex":      float(res["AUC_Sex"]),
            "Active_Nodes": int(som_m["utilization"]["active_units"]),
            "Occ_Entropy":  float(som_m["utilization"]["occupancy_entropy"]),
            "QE":           float(som_m["topology"]["quantization_error"]),
            "TE":           float(som_m["topology"]["topographic_error"]),
        },
        "som_dim":     [H, W],
        "class_names": class_names,
        "cells":       cells,
    }


def _compute_cell_demographics(
    bidx: list[int] | np.ndarray,
    labels_enc: np.ndarray,
    ages_np: np.ndarray,
    sexes_np: np.ndarray,
    class_names: list[str],
) -> dict:
    """Compute class purity, top classes, age histogram, and sex breakdown for one SOM cell."""
    num_classes = len(class_names)
    bidx_arr = np.asarray(bidx, dtype=np.int64)
    counts = np.zeros(num_classes, dtype=np.int64)
    if len(bidx_arr) > 0 and num_classes > 0:
        lbls = labels_enc[bidx_arr]
        valid_lbls = lbls[(lbls >= 0) & (lbls < num_classes)]
        if len(valid_lbls) > 0:
            counts = np.bincount(valid_lbls, minlength=num_classes)

    total_lbl = int(counts.sum())
    maj_idx = int(counts.argmax()) if total_lbl > 0 else -1
    purity = float(counts.max() / total_lbl) if total_lbl > 0 else 0.0
    top_classes = [
        {
            "label": class_names[ci],
            "count": int(counts[ci]),
            "pct":   round(100.0 * float(counts[ci]) / float(total_lbl), 1),
        }
        for ci in np.argsort(-counts)
        if counts[ci] > 0
    ][:3]

    age_bin_edges = [0, 30, 40, 50, 60, 70, 80, 120]
    age_bin_labels = ["<30", "30s", "40s", "50s", "60s", "70s", "80+"]
    if len(bidx_arr) > 0:
        cell_ages = ages_np[bidx_arr].astype(np.float64)
        valid_ages = cell_ages[np.isfinite(cell_ages) & (cell_ages > 0)]
        cell_sexes = sexes_np[bidx_arr]
        valid_sexes = cell_sexes[(cell_sexes == 0) | (cell_sexes == 1)]
    else:
        valid_ages = np.array([], dtype=np.float64)
        valid_sexes = np.array([], dtype=np.int64)

    age_bins = (
        np.histogram(valid_ages, bins=age_bin_edges)[0].tolist()
        if len(valid_ages) > 0
        else [0] * len(age_bin_labels)
    )
    return {
        "counts":         counts,
        "majority_label": class_names[maj_idx] if 0 <= maj_idx < num_classes else "",
        "purity":         purity,
        "top_classes":    top_classes,
        "age_mean":       float(np.mean(valid_ages)) if len(valid_ages) > 0 else None,
        "age_std":        float(np.std(valid_ages)) if len(valid_ages) > 0 else None,
        "age_median":     float(np.median(valid_ages)) if len(valid_ages) > 0 else None,
        "age_bins":       age_bins,
        "age_bin_labels": age_bin_labels,
        "female_pct":     float(100.0 * np.mean(valid_sexes == 1)) if len(valid_sexes) > 0 else None,
    }


# ---------------------------------------------------------------------------
# Routes — SOM Explorer (Existing + Enhanced)
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(
        os.path.join(_STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/_vercel/insights/script.js")
def vercel_analytics_noop():
    """Serve harmless local stub so local/Docker Explorer avoids a 404; Vercel edge replaces this in prod."""
    return Response(
        content="/* Vercel Web Analytics disabled on local FastAPI */",
        media_type="application/javascript",
    )


@app.get("/api/models")
def get_models():
    ckpts     = _discover_ckpts()
    abl_index = _load_ablation_index()
    datasets, default_data = _discover_datasets()

    entries = []
    for p in ckpts:
        rel     = os.path.relpath(p, _REPO_ROOT)
        variant = _variant_from_path(p)
        metrics = abl_index.get(variant) if variant else None

        if variant and metrics:
            nmi = metrics["NMI"]
            ami = metrics["AMI"]
            nmi_str = f"{nmi:.3f}" if nmi is not None else "?"
            ami_str = f"{ami:.3f}" if ami is not None else "?"
            label = f"[abl] {variant}  (NMI={nmi_str}  AMI={ami_str})"
        elif variant:
            label = f"[abl] {variant}"
        else:
            label = rel

        entry = {"label": label, "value": p, "rel_path": rel}
        if variant:
            entry["variant"] = variant
        if metrics:
            entry["metrics"] = metrics
        entries.append(entry)

    def _sort_key(e: dict):
        m = e.get("metrics") or {}
        nmi = m.get("NMI")
        ami = m.get("AMI")
        if nmi is None:
            return (1, 0.0, 0.0)
        return (0, -(nmi), -(ami or 0.0))

    entries.sort(key=_sort_key)

    return {
        "models":       entries,
        "datasets":     datasets,
        "default_data": default_data,
    }


class LoadRequest(BaseModel):
    ckpt:      str
    data_path: str
    split:     str = "test"


@app.post("/api/load")
def load_model(req: LoadRequest, force_recompute: bool = False):
    try:
        resolved_ckpt = _resolve_ckpt_path(req.ckpt)
        resolved_data = _resolve_data_path(req.data_path)
        cache_file = _cache_path(resolved_ckpt, resolved_data, req.split)
        if not force_recompute:
            cached = _load_cache(cache_file)
            if cached is not None:
                cell_majority_map = {
                    (c["row"], c["col"]): c["majority_label"]
                    for c in cached["_result"]["cells"]
                }
                _STATE.clear()
                _STATE.update({
                    "som_dim":           cached["_som_dim"],
                    "cell_beats":        cached["_cell_beats"],
                    "prototype_signals": cached["_prototype_signals"],
                    "t_axis":            cached["_t_axis"],
                    "crop":              cached["_crop"],
                    "input_length":      cached["_input_length"],
                    "input_channels":    cached["_input_channels"],
                    "data_path":         resolved_data,
                    "split":             req.split,
                    "ckpt_path":         resolved_ckpt,
                    "cell_majority":     cell_majority_map,
                    "cell_detail_cache": {},
                })
                result = cached["_result"].copy()
                result["cache_hit"] = True
                result["info"] = result["info"] + "  |  (from cache)"
                return result
        result = _run_inference(resolved_ckpt, resolved_data, req.split)
        _save_cache(cache_file, {
            "_som_dim":           _STATE["som_dim"],
            "_cell_beats":        _STATE["cell_beats"],
            "_prototype_signals": _STATE["prototype_signals"],
            "_t_axis":            _STATE["t_axis"],
            "_crop":              _STATE["crop"],
            "_input_length":      _STATE["input_length"],
            "_input_channels":    _STATE["input_channels"],
            "_result":            result,
        })
        return result
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/cell/{row}/{col}")
def get_cell(row: int, col: int):
    if not _STATE:
        raise HTTPException(400, "No model loaded — POST /api/load first.")
    H, W = _STATE["som_dim"]
    if not (0 <= row < H and 0 <= col < W):
        raise HTTPException(404, "Cell out of range.")
    cache = _STATE.setdefault("cell_detail_cache", {})
    key = (row, col)
    if key not in cache:
        cache[key] = _compute_cell_detail(row, col)
    return cache[key]


def _get_loaded_model() -> DPSOM_ECG:
    ckpt_path = _STATE.get("ckpt_path")
    if not ckpt_path:
        ckpt_path = _find_best_checkpoint(
            models_dir=_MODELS_DIR,
            ablation_csv=os.path.join(_ABLATION_DIR, "ablation_results_all.csv"),
        )
        _STATE["ckpt_path"] = ckpt_path
    model = _STATE.get("model")
    if model is None:
        model, _ = _load_dpsom_model(ckpt_path, device=torch.device("cpu"))
        _STATE["model"] = model
    return model


def _enrich_prediction(pred: dict) -> dict:
    maj_map = _STATE.get("cell_majority", {})
    dom = pred["dominant_cell"]
    pred["dominant_label"] = maj_map.get((dom["row"], dom["col"]), "")
    for bc in pred["beat_cells"]:
        bc["cell_label"] = maj_map.get((bc["row"], bc["col"]), "")
    return pred


class PredictPathRequest(BaseModel):
    path: str
    fs:   float | None = None


@app.post("/api/predict_path")
def predict_from_path(req: PredictPathRequest):
    """Predict age, sex & SOM placement for an ECG file inside data/."""
    try:
        model = _get_loaded_model()
        cleaned = (req.path or "").strip()
        if not cleaned:
            raise HTTPException(400, "File path is empty.")
        rel = cleaned[5:] if cleaned.startswith("data/") else cleaned
        real = os.path.realpath(cleaned if os.path.isabs(cleaned) else os.path.join(_DATA_DIR, rel))
        if real != _DATA_DIR and not real.startswith(_DATA_DIR + os.sep):
            raise HTTPException(400, "ECG file path must be inside the data/ directory.")
        sig, det_fs = load_raw_ecg(real, fs=req.fs)
        rec_id = os.path.splitext(os.path.basename(real))[0]
        pred = predict_signal(model, sig, fs=det_fs, record_id=rec_id)
        return _enrich_prediction(pred)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


class PredictUploadRequest(BaseModel):
    filename:    str
    content_b64: str
    fs:          float = 100.0


@app.post("/api/predict_upload")
def predict_from_upload(req: PredictUploadRequest):
    """Predict age, sex & SOM placement for an uploaded ECG file (.npy, .npz, or .csv)."""
    import base64
    import tempfile
    try:
        model = _get_loaded_model()
        suffix = os.path.splitext(req.filename or "")[1].lower() or ".npy"
        raw = base64.b64decode(req.content_b64)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            sig, det_fs = load_raw_ecg(tmp_path, fs=req.fs)
        finally:
            os.unlink(tmp_path)
        rec_id = os.path.splitext(req.filename or "uploaded_ecg")[0]
        pred = predict_signal(model, sig, fs=det_fs, record_id=rec_id)
        return _enrich_prediction(pred)
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


class CompareCellsRequest(BaseModel):
    cells: list[list[int]]


def _get_som_embeddings() -> np.ndarray:
    """Return cached [H, W, D] SOM embeddings, loading active model lazily on cache hits."""
    emb = _STATE.get("embeddings")
    if emb is not None:
        return emb
    model = _get_loaded_model()
    emb = model._embeddings.detach().float().cpu().numpy()
    _STATE["embeddings"] = emb
    return emb


def _toroidal_grid_dist(r1: int, c1: int, r2: int, c2: int, H: int, W: int) -> tuple[int, float]:
    dr = min(abs(r1 - r2), H - abs(r1 - r2))
    dc = min(abs(c1 - c2), W - abs(c1 - c2))
    return int(dr + dc), float(np.hypot(dr, dc))


def _js_divergence(counts_p: np.ndarray, counts_q: np.ndarray) -> float:
    """Base-2 Jensen-Shannon divergence in [0, 1] between two non-negative count vectors."""
    sp = float(counts_p.sum())
    sq = float(counts_q.sum())
    if sp <= 0.0 or sq <= 0.0:
        return 0.0
    p = counts_p.astype(np.float64) / sp
    q = counts_q.astype(np.float64) / sq
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return float(max(0.0, min(1.0, 0.5 * (_kl(p, m) + _kl(q, m)))))


@app.post("/api/cells/compare")
def compare_cells(req: CompareCellsRequest):
    """Return full prototypes, percentile bands, demographics, and pairwise distances for 1–6 SOM cells."""
    if not _STATE:
        raise HTTPException(400, "No model loaded — load a model in SOM Explorer first.")
    if not req.cells or len(req.cells) > 6:
        raise HTTPException(400, "Select between 1 and 6 SOM cells to compare.")

    H, W = _STATE["som_dim"]
    coords: list[tuple[int, int]] = []
    seen_coords: set[tuple[int, int]] = set()
    for pair in req.cells:
        if len(pair) != 2:
            raise HTTPException(400, f"Invalid cell coordinate: {pair}")
        r, c = int(pair[0]), int(pair[1])
        if not (0 <= r < H and 0 <= c < W):
            raise HTTPException(404, f"Cell ({r}, {c}) out of range for {H}×{W} grid.")
        if (r, c) not in seen_coords:
            seen_coords.add((r, c))
            coords.append((r, c))

    split_data = _load_split_arrays(_STATE["data_path"], _STATE["split"])
    labels_enc = split_data["labels"]
    ages_np = split_data["ages"]
    sexes_np = split_data["sexes"]
    class_names = split_data["class_names"]
    num_classes = len(class_names)

    cache = _STATE.setdefault("cell_detail_cache", {})
    cell_payloads = []
    counts_vecs = []
    for r, c in coords:
        key = (r, c)
        if key not in cache:
            cache[key] = _compute_cell_detail(r, c)
        detail = cache[key]

        flat = r * W + c
        stats = _compute_cell_demographics(
            _STATE["cell_beats"][flat], labels_enc, ages_np, sexes_np, class_names,
        )
        counts_vecs.append(stats["counts"])
        cell_payloads.append({
            "row":            r,
            "col":            c,
            "count":          int(detail["n_total"]),
            "majority_label": stats["majority_label"],
            "purity":         stats["purity"],
            "class_counts":   stats["counts"].tolist(),
            "top_classes":    stats["top_classes"],
            "age_mean":       stats["age_mean"],
            "age_std":        stats["age_std"],
            "age_median":     stats["age_median"],
            "age_bins":       stats["age_bins"],
            "age_bin_labels": stats["age_bin_labels"],
            "female_pct":     stats["female_pct"],
            "proto_leads":    detail["proto_leads"],
            "percentiles":    detail["percentiles"],
        })

    emb = _get_som_embeddings()
    protos = _STATE["prototype_signals"]
    pairwise = []
    for i, (r1, c1) in enumerate(coords):
        row_list = []
        e1 = emb[r1, c1]
        p1 = protos[r1, c1]
        n1 = float(np.linalg.norm(e1))
        for j, (r2, c2) in enumerate(coords):
            e2 = emb[r2, c2]
            p2 = protos[r2, c2]
            n2 = float(np.linalg.norm(e2))
            tl1, tl2 = _toroidal_grid_dist(r1, c1, r2, c2, H, W)
            row_list.append({
                "toroidal_l1":   tl1,
                "toroidal_l2":   tl2,
                "latent_l2":     float(np.linalg.norm(e1 - e2)),
                "latent_cos":    float(np.dot(e1, e2) / max(n1 * n2, 1e-8)),
                "waveform_rmse": float(np.sqrt(np.mean((p1 - p2) ** 2))),
                "js_divergence": _js_divergence(counts_vecs[i], counts_vecs[j]),
            })
        pairwise.append(row_list)

    return {
        "som_dim":     [H, W],
        "class_names": class_names,
        "t_axis":      _STATE["t_axis"],
        "cells":       cell_payloads,
        "pairwise":    pairwise,
    }


# ---------------------------------------------------------------------------
# Routes — Figures, System Overview & UDA Explorer
# ---------------------------------------------------------------------------

@app.get("/api/figures/{fig_id}")
def get_figure(fig_id: str):
    spec = _FIGURE_MAP.get(fig_id)
    if spec is None:
        raise HTTPException(404, f"Unknown figure ID: {fig_id}")
    rel_path = spec[0]
    root_base = _DOCS_DIR if rel_path.startswith("docs" + os.sep) else _ABLATION_DIR
    abs_path = os.path.realpath(os.path.join(_REPO_ROOT, rel_path))
    if abs_path != root_base and not abs_path.startswith(root_base + os.sep):
        raise HTTPException(400, "Invalid figure path.")
    if not os.path.isfile(abs_path):
        raise HTTPException(404, f"Figure file not found on disk: {rel_path}")
    return FileResponse(abs_path, media_type="image/png")


@app.get("/api/overview")
def get_overview():
    """Return structured DPSOM-ECG architecture, cohort specs, benchmarks, UDA summary, and figures."""
    return build_overview_payload(_REPO_ROOT, figure_url_template="/api/figures/{fig_id}")


@app.get("/api/uda/results")
def get_uda_results():
    """Return 3-seed UDA benchmark results, cross-dataset matrices, and selectable .pkl cohorts."""
    return build_uda_results_payload(_REPO_ROOT, pkl_datasets=_discover_pkl_datasets())


class UdaRunRequest(BaseModel):
    ckpt:         str
    tgt_data:     str
    src_data:     str | None = None
    method:       str = "uda_bw_ot"
    adapt_epochs: int = 4
    s_morph:      float = 0.35
    s_age:        float = 0.25
    s_sex:        float = 0.75
    alpha_rec:    float = 0.35
    rho:          float = 0.75


@app.post("/api/uda/run")
def run_live_uda(req: UdaRunRequest):
    """Run live zero-label UDA on a selected checkpoint and target .pkl dataset."""
    valid_methods = {"uda_bw_ot", "uda_sf_strong", "uda_ot_strong", "uda_vanilla"}
    if req.method not in valid_methods:
        raise HTTPException(400, f"Unsupported UDA method '{req.method}'. Choose one of {sorted(valid_methods)}.")

    try:
        t0 = time.time()
        ckpt_real = _resolve_ckpt_path(req.ckpt)
        tgt_real = _resolve_data_path(req.tgt_data)
        tgt_ds = _ensure_uda_splits(_load_pkl_dataset(tgt_real))

        src_real = None
        src_ds = None
        needs_src = req.method in ("uda_bw_ot", "uda_ot_strong")
        if needs_src:
            if not req.src_data:
                raise HTTPException(400, f"Method '{req.method}' requires src_data for source-domain moments.")
            src_real = _resolve_data_path(req.src_data)
            src_ds = _ensure_uda_splits(_load_pkl_dataset(src_real))

        device = _get_device()
        src_model, meta = load_dpsom_checkpoint(ckpt_real, device)
        use_rr = (int(meta.get("amp_context_dim", 2 * int(meta["input_channels"]))) == 2 * int(meta["input_channels"]) + 2)

        before = _evaluate_model_for_explorer(src_model, tgt_ds, use_rr=use_rr)

        src_moments = None
        if needs_src and src_ds is not None and src_real is not None:
            mtime = os.path.getmtime(ckpt_real)
            mkey = (ckpt_real, round(mtime, 3), src_real)
            src_moments = _MOMENTS_CACHE.get(mkey)
            if src_moments is None:
                src_moments = extract_domain_moments(src_model, src_ds, batch_size=128, use_rr_features=use_rr)
                _MOMENTS_CACHE[mkey] = src_moments

        epochs = max(1, min(int(req.adapt_epochs), 15))
        use_amp = (device.type == "cuda")

        if req.method == "uda_bw_ot":
            adapted = adapt_uda_bw_ot(
                src_model, tgt_ds, src_moments,  # type: ignore[arg-type]
                s_morph=float(req.s_morph), s_age=float(req.s_age), s_sex=float(req.s_sex),
                use_rr_features=use_rr,
            )
        elif req.method == "uda_sf_strong":
            adapted = adapt_uda_sf_strong(
                src_model, tgt_ds, batch_size=64, adapt_epochs=epochs,
                alpha_rec=float(req.alpha_rec), rho=float(req.rho),
                use_amp=use_amp, use_rr_features=use_rr,
            )
        elif req.method == "uda_ot_strong":
            adapted = adapt_uda_ot_strong(
                src_model, tgt_ds, src_moments,  # type: ignore[arg-type]
                s_age=float(req.s_age), s_sex=float(req.s_sex),
                batch_size=64, adapt_epochs=epochs,
                alpha_rec=float(req.alpha_rec), rho=float(req.rho),
                use_amp=use_amp, use_rr_features=use_rr,
            )
        else:  # uda_vanilla
            adapted = adapt_uda_sf_strong(
                src_model, tgt_ds, batch_size=64, adapt_epochs=epochs,
                lr=1e-4, alpha_rec=0.0, rho=1.0, tau_mult=1.0,
                use_swa=False, use_amp=use_amp, use_rr_features=use_rr,
            )

        after = _evaluate_model_for_explorer(adapted, tgt_ds, use_rr=use_rr)
        adapted_cpu = adapted.cpu()
        deltas = {
            k: round(float(after["metrics"][k]) - float(before["metrics"][k]), 6)
            for k in before["metrics"]
        }

        _UDA_STATE.clear()
        _UDA_STATE.update({
            "model":       adapted_cpu,
            "meta":        meta,
            "ckpt_path":   ckpt_real,
            "tgt_data":    tgt_real,
            "tgt_ds":      tgt_ds,
            "method":      req.method,
            "som_dim":     after["som_dim"],
            "class_names": after["class_names"],
        })

        return {
            "before":      before,
            "after":       after,
            "deltas":      deltas,
            "som_dim":     after["som_dim"],
            "class_names": after["class_names"],
            "elapsed_s":   round(time.time() - t0, 2),
            "method":      req.method,
            "use_rr":      use_rr,
            "can_apply":   True,
        }
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/uda/apply")
def apply_live_uda():
    """Install the most recently adapted UDA model into SOM Explorer state."""
    if not _UDA_STATE or "model" not in _UDA_STATE:
        raise HTTPException(400, "No live UDA run in memory — POST /api/uda/run first.")
    try:
        method_lbl = _REGIME_META.get(_UDA_STATE["method"], {}).get("label", _UDA_STATE["method"])
        ckpt_base = os.path.basename(_UDA_STATE["ckpt_path"])
        return _run_inference_with_model(
            _UDA_STATE["model"],
            _UDA_STATE["meta"],
            _UDA_STATE["ckpt_path"],
            _UDA_STATE["tgt_data"],
            "test",
            dataset_override=_UDA_STATE.get("tgt_ds"),
            info_prefix=f"Adapted [{method_lbl}]: {ckpt_base}",
        )
    except HTTPException:
        raise
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="ECG-SOM Explorer (FastAPI + D3)")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
