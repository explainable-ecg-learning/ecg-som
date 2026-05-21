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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
_STATE: dict    = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="ECG-SOM Explorer")
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_ckpts() -> list[str]:
    return sorted(glob.glob(os.path.join(_REPO_ROOT, "models", "**", "*.ckpt"), recursive=True))


def _load_ablation_index() -> dict[str, dict]:
    """Return a mapping {variant_name: {NMI, AMI, Purity, MAE_Age, AUC_Sex}}
    from ablation_results_all.csv, or an empty dict if the file is absent."""
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
    """Extract the ablation variant name from a checkpoint path.
    Directory names follow the pattern: abl_{variant}_{latent}_{som}_{date}_{hash}
    e.g. abl_no_som_commit_32_8-8_2026-05-05_19f21
    We strip 'abl_' and then drop the trailing numeric/date/hash suffix.
    """
    import re
    dirname = os.path.basename(os.path.dirname(ckpt_path))
    # Must start with 'abl_'
    if not dirname.startswith("abl_"):
        return None
    # Strip 'abl_' prefix, then remove trailing _<latent_dim>_<som>_<date>_<hash>
    body = dirname[4:]  # e.g. 'no_som_commit_32_8-8_2026-05-05_19f21'
    # The suffix starts at the first _<digits>_ block (latent dim)
    m = re.search(r"_\d+_\d+-\d+_", body)
    if m:
        return body[: m.start()]
    return None


def _cache_path(ckpt_path: str, data_path: str, split: str) -> str:
    key = hashlib.sha256(f"{ckpt_path}|{data_path}|{split}".encode()).hexdigest()[:16]
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, f"{key}.pkl")


def _load_cache(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_cache(path: str, payload: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _run_inference(ckpt_path: str, data_path: str, split: str) -> dict:
    device = torch.device("cpu")

    with open(data_path, "rb") as f:
        dataset = pickle.load(f)

    state_dict, meta = _load_checkpoint_state(ckpt_path, device)
    latent_dim     = int(meta.get("latent_dim", 32))
    som_dim        = tuple(int(v) for v in meta.get("som_dim", (8, 8)))
    input_length   = int(meta.get("input_length", 120))
    input_channels = int(meta.get("input_channels", 12))
    H, W = som_dim

    model = DPSOM_ECG(
        latent_dim=latent_dim, som_dim=som_dim,
        input_length=input_length, input_channels=input_channels,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    class_names = getattr(dataset, "class_names", CLASS_NAMES_DEF)
    label_enc   = dataset.label_encoder
    records     = dataset.get_data(split)

    beats_list, labels_list, ages_list, sexes_list = [], [], [], []
    rec_ids, beat_local = [], []

    for rec in records:
        for i in range(len(rec.beat_representations)):
            beats_list.append(rec.beat_representations[i])  # [T, C]
            labels_list.append(rec.label)
            ages_list.append(rec.age)
            sexes_list.append(rec.sex)
            rec_ids.append(rec.id)
            beat_local.append(i)

    beats_np   = np.stack(beats_list).astype(np.float32)   # [N, T, C]
    labels_enc = label_enc.transform(labels_list)           # [N]

    # Encode batched
    x_t    = torch.from_numpy(beats_np.transpose(0, 2, 1))  # [N, C, T]
    k_list = []
    with torch.no_grad():
        for s in range(0, len(x_t), 512):
            xb = x_t[s:s + 512].to(device)
            mu, _, _, _ = model._encode(xb)
            k_list.append(model.k(mu).cpu().numpy())
    k_all = np.concatenate(k_list)  # [N]

    # Decode SOM prototypes
    CROP = 10  # remove boundary artifacts from decoder output, matching visualization.py
    with torch.no_grad():
        E_flat = model._embeddings.view(-1, latent_dim)
        logits = model.decoder(E_flat)
        proto  = logits.view(H, W, input_channels, input_length).cpu().numpy()
    # proto shape: [H, W, C, T] — crop T edges, then transpose to [H, W, T_cropped, C]
    crop = max(0, min(CROP, (input_length - 1) // 2))
    proto_cropped = proto[:, :, :, crop:input_length - crop] if crop > 0 else proto
    prototype_signals = proto_cropped.transpose(0, 1, 3, 2)  # [H, W, T_cropped, C]
    t_axis = (np.arange(proto_cropped.shape[-1]) / FS).tolist()

    # Build per-cell index
    K           = H * W
    num_classes = len(class_names)
    cell_beats: dict[int, list[int]] = {k: [] for k in range(K)}
    counts = np.zeros((K, num_classes), dtype=np.int64)
    for i, ki in enumerate(k_all):
        cell_beats[int(ki)].append(i)
        lbl = int(labels_enc[i])
        if 0 <= lbl < num_classes:
            counts[int(ki), lbl] += 1

    totals       = counts.sum(axis=1)
    cell_majority = np.full(K, -1, dtype=np.int64)
    nonempty     = totals > 0
    cell_majority[nonempty] = counts[nonempty].argmax(axis=1)

    ages_np  = np.array(ages_list, dtype=np.float32)
    sexes_np = np.array(sexes_list, dtype=np.int64)
    # t_axis already defined above (cropped length)

    # Pre-compute per-cell data (percentiles + samples)
    cell_data: dict[tuple[int, int], dict] = {}
    cells = []
    for r in range(H):
        for c in range(W):
            flat        = r * W + c
            maj         = int(cell_majority[flat])
            beat_indices = cell_beats[flat]
            n_total     = len(beat_indices)
            proto       = prototype_signals[r, c]           # [T, C]
            proto_leads = [proto[:, li].tolist() for li in range(input_channels)]

            percentiles = None
            if n_total > 0:
                cb = beats_np[beat_indices]                 # [N, T, C]
                cb = cb[:, crop:input_length - crop, :]     # crop edges
                pcts = np.percentile(cb, [5, 25, 50, 75, 95], axis=0)  # [5, T_crop, C]
                percentiles = {
                    "p5":  [pcts[0, :, li].tolist() for li in range(input_channels)],
                    "p25": [pcts[1, :, li].tolist() for li in range(input_channels)],
                    "p50": [pcts[2, :, li].tolist() for li in range(input_channels)],
                    "p75": [pcts[3, :, li].tolist() for li in range(input_channels)],
                    "p95": [pcts[4, :, li].tolist() for li in range(input_channels)],
                }

            samples = []
            for gi in beat_indices[:MAX_SAMPLES]:
                beat    = beats_np[gi, crop:input_length - crop, :]  # [T_crop, C]
                lbl_enc = int(labels_enc[gi])
                samples.append({
                    "leads":     [beat[:, li].tolist() for li in range(input_channels)],
                    "record_id": str(rec_ids[gi]),
                    "beat_idx":  int(beat_local[gi]),
                    "label":     class_names[lbl_enc] if 0 <= lbl_enc < len(class_names) else "?",
                    "age":       float(ages_np[gi]),
                    "sex":       "F" if int(sexes_np[gi]) == 1 else "M",
                })

            cell_data[(r, c)] = {
                "row":         r,
                "col":         c,
                "n_total":     n_total,
                "n_shown":     len(samples),
                "proto_leads": proto_leads,
                "percentiles": percentiles,
                "samples":     samples,
                "t_axis":      t_axis,
            }

            cells.append({
                "row":           r,
                "col":           c,
                "count":         int(totals[flat]),
                "majority":      maj,
                "majority_label": class_names[maj] if 0 <= maj < num_classes else "",
                "class_counts":  counts[flat].tolist(),
                "proto_leads":   proto_leads,
            })

    _STATE.clear()
    _STATE.update({
        "som_dim":    som_dim,
        "cell_data":  cell_data,
    })

    result = {
        "som_dim":     list(som_dim),
        "class_names": class_names,
        "cells":       cells,
        "info": (
            f"Loaded: {os.path.basename(ckpt_path)}  |  "
            f"Split: {split}  |  Beats: {len(beats_np)}  |  SOM: {H}×{W}"
        ),
        "cache_hit": False,
    }
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(os.path.join(_STATIC_DIR, "index.html"))


@app.get("/api/models")
def get_models():
    ckpts        = _discover_ckpts()
    abl_index    = _load_ablation_index()
    default_data = os.path.join(_REPO_ROOT, "data", "ptbxl_100_T-12ms.pkl")

    entries = []
    for p in ckpts:
        rel      = os.path.relpath(p, _REPO_ROOT)
        variant  = _variant_from_path(p)
        metrics  = abl_index.get(variant) if variant else None

        # Build a human-readable label
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
        # Entries with metrics sort first (descending NMI, then AMI); rest go last
        if nmi is None:
            return (1, 0.0, 0.0)
        return (0, -(nmi), -(ami or 0.0))

    entries.sort(key=_sort_key)

    return {
        "models":       entries,
        "default_data": default_data if os.path.exists(default_data) else "",
    }


class LoadRequest(BaseModel):
    ckpt:      str
    data_path: str
    split:     str = "test"


@app.post("/api/load")
def load_model(req: LoadRequest, force_recompute: bool = False):
    try:
        cache_file = _cache_path(req.ckpt, req.data_path, req.split)
        if not force_recompute:
            cached = _load_cache(cache_file)
            if cached is not None:
                _STATE.clear()
                _STATE.update({
                    "som_dim":   cached["_som_dim"],
                    "cell_data": cached["_cell_data"],
                })
                result = cached["_result"].copy()
                result["cache_hit"] = True
                result["info"] = result["info"] + "  |  (from cache)"
                return result
        result = _run_inference(req.ckpt, req.data_path, req.split)
        _save_cache(cache_file, {
            "_som_dim":   _STATE["som_dim"],
            "_cell_data": _STATE["cell_data"],
            "_result":    result,
        })
        return result
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/cell/{row}/{col}")
def get_cell(row: int, col: int):
    if not _STATE:
        raise HTTPException(400, "No model loaded — POST /api/load first.")
    H, W = _STATE["som_dim"]
    if not (0 <= row < H and 0 <= col < W):
        raise HTTPException(404, "Cell out of range.")
    data = _STATE["cell_data"].get((row, col))
    if data is None:
        raise HTTPException(404, "Cell data not found.")
    return data


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
