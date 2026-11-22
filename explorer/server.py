"""ECG-SOM Explorer — FastAPI backend.

Launch (from repo root):
    python explorer/server.py [--port 8050] [--host 0.0.0.0]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import pickle
import sys
import traceback

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.inference import (
    find_best_checkpoint as _find_best_checkpoint,
    load_model as _load_dpsom_model,
    load_raw_ecg,
    predict_signal,
)
from src.model.dpsom import DPSOM_ECG
from src.training.trainer import _load_checkpoint_state

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS              = 100.0
MAX_SAMPLES     = 20
CLASS_NAMES_DEF = ["CD", "HYP", "MI", "NORM", "STTC"]
_STATIC_DIR     = os.path.join(os.path.dirname(__file__), "static")
_CACHE_DIR      = os.path.join(os.path.dirname(__file__), "cache")
_DATA_DIR       = os.path.realpath(os.path.join(_REPO_ROOT, "data"))
_STATE: dict    = {}
_SPLIT_CACHE: dict[tuple[str, str], dict] = {}
_DATASET_CACHE: dict[str, object] = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="ECG-SOM Explorer")
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_ckpts() -> list[str]:
    return sorted(glob.glob(os.path.join(_REPO_ROOT, "models", "**", "*.ckpt"), recursive=True))


def _discover_datasets() -> tuple[list[dict], str]:
    """List selectable dataset files strictly inside the data/ subdirectory."""
    options: list[dict] = []
    compact_splits = ("explorer_test.npz", "explorer_val.npz", "explorer_train.npz")
    has_compact = os.path.exists(os.path.join(_DATA_DIR, "explorer_test.npz"))

    if has_compact:
        options.append({
            "label": "PTB-XL (compact .npz — fast)",
            "value": "explorer_test.npz",
        })

    if os.path.isdir(_DATA_DIR):
        for fname in sorted(os.listdir(_DATA_DIR)):
            if not fname.endswith((".npz", ".pkl")):
                continue
            if fname in compact_splits:
                continue
            fpath = os.path.join(_DATA_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            size_str = f"{size_mb / 1024:.1f} GB" if size_mb >= 1024 else f"{size_mb:.0f} MB"
            options.append({
                "label": f"{fname}  ({size_str})",
                "value": fname,
            })

    default_val = options[0]["value"] if options else ""
    return options, default_val


def _resolve_data_path(data_name: str) -> str:
    """Resolve a dataset filename strictly inside data/ (i.e. /app/data in Docker)."""
    cleaned = (data_name or "").strip()
    if not cleaned:
        raise HTTPException(400, "Dataset name is empty.")
    # Allow passing either 'foo.npz', 'data/foo.npz', or '/app/data/foo.npz' — strip to relative under data/
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


def _load_ablation_index() -> dict[str, dict]:
    """Return {variant_name: {NMI, AMI, Purity, MAE_Age, AUC_Sex}} from ablation_results_all.csv."""
    csv_path = os.path.join(_REPO_ROOT, "ablation", "ablation_results_all.csv")
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


def _safe_float(v: object) -> float | None:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _variant_from_path(ckpt_path: str) -> str | None:
    import re
    dirname = os.path.basename(os.path.dirname(ckpt_path))
    if not dirname.startswith("abl_"):
        return None
    body = dirname[4:]
    m = re.search(r"_\d+_\d+-\d+_", body)
    return body[: m.start()] if m else None


_CACHE_SCHEMA_VERSION = "v3"  # v3: lazy per-cell percentiles + compact float16 split loading


def _cache_path(ckpt_path: str, data_path: str, split: str) -> str:
    key = hashlib.sha256(
        f"{_CACHE_SCHEMA_VERSION}|{ckpt_path}|{data_path}|{split}".encode()
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
    # Only auto-switch across explorer_{split}.npz when the user selected the compact PTB-XL option
    if fname.startswith("explorer_") and fname.endswith(".npz"):
        if split == "all":
            paths = [os.path.join(_DATA_DIR, f"explorer_{s}.npz") for s in ("train", "val", "test")]
            return paths if all(os.path.exists(p) for p in paths) else None
        cand = os.path.join(_DATA_DIR, f"explorer_{split}.npz")
        return [cand] if os.path.exists(cand) else [data_path]
    if data_path.endswith(".npz") and os.path.exists(data_path):
        return [data_path]
    return None


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

    # Fallback: load .pkl (cached in memory across splits/checkpoints)
    abs_pkl = os.path.abspath(data_path)
    dataset = _DATASET_CACHE.get(abs_pkl)
    if dataset is None:
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
        _DATASET_CACHE[abs_pkl] = dataset

    class_names  = list(getattr(dataset, "class_names", CLASS_NAMES_DEF))
    label_to_idx = {str(name): i for i, name in enumerate(class_names)}
    records      = dataset.get_data(split)

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

    payload = {
        "beats":       np.stack(beats_list).astype(np.float16),
        "labels":      np.asarray(labels_list, dtype=np.int64),
        "ages":        np.asarray(ages_list, dtype=np.float32),
        "sexes":       np.asarray(sexes_list, dtype=np.int64),
        "rec_ids":     np.asarray(rec_ids),
        "beat_local":  np.asarray(beat_local, dtype=np.int64),
        "class_names": class_names,
    }
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


def _run_inference(ckpt_path: str, data_path: str, split: str) -> dict:
    device = torch.device("cpu")
    split_data  = _load_split_arrays(data_path, split)
    beats_np    = split_data["beats"]
    labels_enc  = split_data["labels"]
    class_names = split_data["class_names"]

    state_dict, meta = _load_checkpoint_state(ckpt_path, device)
    latent_dim     = int(meta.get("latent_dim", 32))
    som_dim        = tuple(int(v) for v in meta.get("som_dim", (8, 8)))
    input_length   = int(meta.get("input_length", 120))
    input_channels = int(meta.get("input_channels", 12))
    H, W = som_dim

    model = DPSOM_ECG(
        latent_dim=latent_dim, som_dim=som_dim,
        input_length=input_length, input_channels=input_channels,
        encoder_base_channels_1=int(meta.get("encoder_base_channels_1", 32)),
        encoder_base_channels_2=int(meta.get("encoder_base_channels_2", 64)),
        encoder_kernel_size=int(meta.get("encoder_kernel_size", 7)),
        encoder_fc_hidden_dim=int(meta.get("encoder_fc_hidden_dim", 512)),
        z_age_dim_factor=float(meta.get("z_age_dim_factor", 0.25)),
        z_sex_dim_factor=float(meta.get("z_sex_dim_factor", 0.25)),
        amp_context_dim=meta.get("amp_context_dim", None),
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

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
        "cell_majority":     cell_majority_map,
        "cell_detail_cache": {},
    })

    return {
        "som_dim":     list(som_dim),
        "class_names": class_names,
        "cells":       cells,
        "info": (
            f"Loaded: {os.path.basename(ckpt_path)}  |  "
            f"Split: {split}  |  Beats: {len(beats_np)}  |  SOM: {H}×{W}"
        ),
        "cache_hit": False,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


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
        resolved_data = _resolve_data_path(req.data_path)
        cache_file = _cache_path(req.ckpt, resolved_data, req.split)
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
                    "ckpt_path":         req.ckpt,
                    "cell_majority":     cell_majority_map,
                    "cell_detail_cache": {},
                })
                result = cached["_result"].copy()
                result["cache_hit"] = True
                result["info"] = result["info"] + "  |  (from cache)"
                return result
        result = _run_inference(req.ckpt, resolved_data, req.split)
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
            models_dir=os.path.join(_REPO_ROOT, "models"),
            ablation_csv=os.path.join(_REPO_ROOT, "ablation", "ablation_results_all.csv"),
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
