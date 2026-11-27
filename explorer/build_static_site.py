"""Static site builder for deploying ECG-SOM Explorer to Vercel or any static CDN.

Generates a self-contained static bundle inside ``explorer/dist/``:
  - Pure Python stdlib when ``--lightweight`` is passed or when ``torch`` /
    checkpoints / datasets are absent (allowing zero-config GitHub -> Vercel builds).
  - Full interactive SOM + Cell Compare + UDA Playground export when local
    checkpoints and datasets are present.

Usage:
    python explorer/build_static_site.py [--out-dir explorer/dist] [--lightweight] [--if-missing]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from explorer.metadata import (
        FIGURE_MAP,
        REGIME_META,
        build_overview_payload,
        build_uda_results_payload,
    )
except ImportError:
    from metadata import (
        FIGURE_MAP,
        REGIME_META,
        build_overview_payload,
        build_uda_results_payload,
    )


def _sanitize_json_value(obj: object, ndigits: int = 3) -> object:
    """Recursively round floats and replace non-finite floats with None."""
    if isinstance(obj, float):
        return round(obj, ndigits) if math.isfinite(obj) else None
    if isinstance(obj, list):
        return [_sanitize_json_value(v, ndigits) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_json_value(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_value(v, ndigits) for k, v in obj.items()}
    return obj


def _write_json(path: str, payload: object, *, ndigits: int | None = 3) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cleaned = _sanitize_json_value(payload, ndigits=ndigits) if ndigits is not None else payload
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cleaned, fh, allow_nan=False, separators=(",", ":"))


def _default_bundle_specs(repo_root: str) -> list[dict]:
    """Return candidate SOM checkpoint/dataset pairs to precompute when available on disk."""
    ptbxl_compact = os.path.join(repo_root, "data", "explorer", "explorer_test.npz")
    ptbxl_pkl = os.path.join(repo_root, "data", "processed", "ptbxl_100_T-12ms.pkl")
    if os.path.isfile(ptbxl_compact):
        ptbxl_abs, ptbxl_val, ptbxl_lbl = ptbxl_compact, "explorer/explorer_test.npz", "PTB-XL (compact .npz — fast)"
    else:
        ptbxl_abs, ptbxl_val, ptbxl_lbl = ptbxl_pkl, "processed/ptbxl_100_T-12ms.pkl", "PTB-XL test (processed)"

    ckpt_best = "models/gpu_fast_seed42/gpu_fast_seed42_best.ckpt"
    ckpt_large = "models/abl_large_som_32_16-16_2026-05-04_dc308/abl_large_som_32_16-16_2026-05-04_dc308.ckpt"

    return [
        {
            "id":         "gpu_fast_seed42_best_ptbxl",
            "ckpt_abs":   os.path.join(repo_root, ckpt_best),
            "ckpt_val":   ckpt_best,
            "ckpt_label": ckpt_best,
            "data_abs":   ptbxl_abs,
            "data_val":   ptbxl_val,
            "data_label": ptbxl_lbl,
            "split":      "test",
        },
        {
            "id":         "gpu_fast_seed42_best_georgia",
            "ckpt_abs":   os.path.join(repo_root, ckpt_best),
            "ckpt_val":   ckpt_best,
            "ckpt_label": ckpt_best,
            "data_abs":   os.path.join(repo_root, "data", "processed", "georgia_transfer_100.pkl"),
            "data_val":   "processed/georgia_transfer_100.pkl",
            "data_label": "processed/georgia_transfer_100.pkl",
            "split":      "test",
        },
        {
            "id":         "gpu_fast_seed42_best_chapman",
            "ckpt_abs":   os.path.join(repo_root, ckpt_best),
            "ckpt_val":   ckpt_best,
            "ckpt_label": ckpt_best,
            "data_abs":   os.path.join(repo_root, "data", "processed", "chapman_transfer_100.pkl"),
            "data_val":   "processed/chapman_transfer_100.pkl",
            "data_label": "processed/chapman_transfer_100.pkl",
            "split":      "test",
        },
        {
            "id":          "abl_large_som_ptbxl",
            "ckpt_abs":    os.path.join(repo_root, ckpt_large),
            "ckpt_val":    ckpt_large,
            "ckpt_label":  "[abl] large_som_32_16-16 (16×16)",
            "data_abs":    ptbxl_abs,
            "data_val":    ptbxl_val,
            "data_label":  ptbxl_lbl,
            "split":       "test",
            "max_samples": 2,
        },
    ]


def _copy_static_shell(repo_root: str, out_dir: str) -> None:
    """Copy HTML/CSS/JS assets and mark the bundle as running in static mode."""
    src_static = os.path.join(repo_root, "explorer", "static")
    dst_static = os.path.join(out_dir, "static")
    if os.path.isdir(dst_static):
        shutil.rmtree(dst_static)
    shutil.copytree(src_static, dst_static)

    shutil.copy2(os.path.join(src_static, "index.html"), os.path.join(out_dir, "index.html"))
    with open(os.path.join(dst_static, "site-config.js"), "w", encoding="utf-8") as fh:
        fh.write("window.EXPLORER_STATIC_MODE = true;\nwindow.EXPLORER_API_BASE = '';\n")


def _export_figures_and_overview(repo_root: str, out_dir: str) -> None:
    fig_dir = os.path.join(out_dir, "static-data", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for fig_id, (rel_path, _, _) in FIGURE_MAP.items():
        src = os.path.join(repo_root, rel_path)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(fig_dir, f"{fig_id}.png"))

    overview = build_overview_payload(
        repo_root,
        figure_url_template="/static-data/figures/{fig_id}.png",
    )
    _write_json(os.path.join(out_dir, "static-data", "overview.json"), overview, ndigits=5)


def _export_single_som_bundle(
    srv,
    out_dir: str,
    bundle_id: str,
    inf_result: dict,
    *,
    max_samples: int = 4,
) -> None:
    """Write ``static-data/som/<bundle_id>.json`` and lazy cell files from ``srv._STATE``."""
    som_dim = tuple(int(v) for v in srv._STATE["som_dim"])
    H, W = som_dim
    split_data = srv._load_split_arrays(srv._STATE["data_path"], srv._STATE["split"])
    labels_enc = split_data["labels"]
    ages_np = split_data["ages"]
    sexes_np = split_data["sexes"]
    class_names = split_data["class_names"]

    cell_stats = []
    cells_dir = os.path.join(out_dir, "static-data", "cells", bundle_id)
    os.makedirs(cells_dir, exist_ok=True)

    prev_max = srv.MAX_SAMPLES
    effective_samples = min(int(max_samples), 2) if H * W > 100 else int(max_samples)
    srv.MAX_SAMPLES = max(1, effective_samples)
    try:
        for r in range(H):
            for c in range(W):
                flat = r * W + c
                bidx = srv._STATE["cell_beats"][flat]
                stats = srv._compute_cell_demographics(bidx, labels_enc, ages_np, sexes_np, class_names)
                cell_stats.append({
                    "purity":         stats["purity"],
                    "top_classes":    stats["top_classes"],
                    "age_mean":       stats["age_mean"],
                    "age_std":        stats["age_std"],
                    "age_median":     stats["age_median"],
                    "age_bins":       stats["age_bins"],
                    "age_bin_labels": stats["age_bin_labels"],
                    "female_pct":     stats["female_pct"],
                })
                detail = srv._compute_cell_detail(r, c)
                _write_json(os.path.join(cells_dir, f"{r}_{c}.json"), detail, ndigits=3)
    finally:
        srv.MAX_SAMPLES = prev_max

    som_payload = {
        **inf_result,
        "t_axis":     srv._STATE["t_axis"],
        "embeddings": srv._STATE["embeddings"].tolist(),
        "cell_stats": cell_stats,
    }
    _write_json(os.path.join(out_dir, "static-data", "som", f"{bundle_id}.json"), som_payload, ndigits=3)


def _export_uda_demos(
    srv,
    out_dir: str,
    repo_root: str,
    *,
    max_samples: int,
) -> tuple[list[dict], list[dict]]:
    """Export precomputed UDA Playground demos for PTB-XL -> Georgia/Chapman."""
    import torch
    from src.training.trainer import load_dpsom_checkpoint
    from src.training.uda import (
        adapt_uda_bw_ot,
        adapt_uda_ot_strong,
        adapt_uda_sf_strong,
        extract_domain_moments,
    )

    ckpt_val = "models/gpu_fast_seed42/gpu_fast_seed42_best.ckpt"
    ckpt_abs = os.path.join(repo_root, ckpt_val)
    src_val = "processed/ptbxl_100_T-12ms.pkl"
    src_abs = os.path.join(repo_root, "data", src_val)
    targets = [
        ("georgia", "processed/georgia_transfer_100.pkl", os.path.join(repo_root, "data", "processed", "georgia_transfer_100.pkl")),
        ("chapman", "processed/chapman_transfer_100.pkl", os.path.join(repo_root, "data", "processed", "chapman_transfer_100.pkl")),
    ]

    if not (os.path.isfile(ckpt_abs) and os.path.isfile(src_abs)):
        return [], []
    available_targets = [(t_name, t_val, t_abs) for t_name, t_val, t_abs in targets if os.path.isfile(t_abs)]
    if not available_targets:
        return [], []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_model, meta = load_dpsom_checkpoint(ckpt_abs, device)
    use_rr = (int(meta.get("amp_context_dim", 2 * int(meta["input_channels"]))) == 2 * int(meta["input_channels"]) + 2)
    src_ds = srv._ensure_uda_splits(srv._load_pkl_dataset(src_abs))
    src_moments = extract_domain_moments(src_model, src_ds, batch_size=128, use_rr_features=use_rr)

    demos_index: list[dict] = []
    pkl_datasets: list[dict] = [{"label": f"{src_val} (PTB-XL source)", "value": src_val}]

    for t_name, t_val, t_abs in available_targets:
        pkl_datasets.append({"label": f"{t_val} ({t_name.title()} target)", "value": t_val})
        tgt_ds = srv._ensure_uda_splits(srv._load_pkl_dataset(t_abs))
        before = srv._evaluate_model_for_explorer(src_model, tgt_ds, use_rr=use_rr)

        for method in ("uda_bw_ot", "uda_sf_strong", "uda_ot_strong"):
            if method == "uda_bw_ot":
                adapted = adapt_uda_bw_ot(src_model, tgt_ds, src_moments, s_morph=0.35, s_age=0.25, s_sex=0.75, use_rr_features=use_rr)
                elapsed = 0.12
            elif method == "uda_sf_strong":
                adapted = adapt_uda_sf_strong(
                    src_model, tgt_ds, batch_size=64, adapt_epochs=4,
                    alpha_rec=0.35, rho=0.75, use_amp=(device.type == "cuda"), use_rr_features=use_rr,
                )
                elapsed = 4.5
            else:
                adapted = adapt_uda_ot_strong(
                    src_model, tgt_ds, src_moments, s_age=0.25, s_sex=0.75,
                    batch_size=64, adapt_epochs=4, alpha_rec=0.35, rho=0.75,
                    use_amp=(device.type == "cuda"), use_rr_features=use_rr,
                )
                elapsed = 4.6

            after = srv._evaluate_model_for_explorer(adapted, tgt_ds, use_rr=use_rr)
            deltas = {
                k: round(float(after["metrics"][k]) - float(before["metrics"][k]), 6)
                for k in before["metrics"]
            }

            demo_id = f"uda_{t_name}_{method}"
            bundle_id = demo_id
            method_lbl = REGIME_META.get(method, {}).get("label", method)
            inf_res = srv._run_inference_with_model(
                adapted.cpu(), meta, ckpt_abs, t_abs, "test",
                dataset_override=tgt_ds,
                info_prefix=f"Adapted [{method_lbl}]: {os.path.basename(ckpt_abs)}",
            )
            _export_single_som_bundle(srv, out_dir, bundle_id, inf_res, max_samples=max_samples)

            demo_payload = {
                "before":            before,
                "after":             after,
                "deltas":            deltas,
                "som_dim":           after["som_dim"],
                "class_names":       after["class_names"],
                "elapsed_s":         elapsed,
                "method":            method,
                "use_rr":            use_rr,
                "can_apply":         True,
                "precomputed":       True,
                "adapted_bundle_id": bundle_id,
            }
            demo_url = f"/static-data/uda_demos/{demo_id}.json"
            _write_json(os.path.join(out_dir, "static-data", "uda_demos", f"{demo_id}.json"), demo_payload, ndigits=4)
            demos_index.append({
                "id":                demo_id,
                "ckpt":              ckpt_val,
                "tgt_data":          t_val,
                "src_data":          src_val if method != "uda_sf_strong" else None,
                "method":            method,
                "url":               demo_url,
                "adapted_bundle_id": bundle_id,
            })

    return pkl_datasets, demos_index


def build_static_site(
    out_dir: str,
    *,
    repo_root: str = _REPO_ROOT,
    lightweight: bool = False,
    max_samples: int = 4,
    skip_uda_demos: bool = False,
    custom_bundles: list[dict] | None = None,
) -> dict:
    """Build static Vercel/CDN directory under ``out_dir`` and return summary info."""
    os.makedirs(out_dir, exist_ok=True)
    _copy_static_shell(repo_root, out_dir)
    _export_figures_and_overview(repo_root, out_dir)

    models_list: list[dict] = []
    datasets_list: list[dict] = []
    seen_models: set[str] = set()
    seen_datasets: set[str] = set()
    bundles_index: dict[str, str] = {}
    bundle_meta: dict[str, dict] = {}
    pkl_datasets: list[dict] = []
    static_demos: list[dict] = []

    if not lightweight:
        candidate_specs = custom_bundles if custom_bundles is not None else _default_bundle_specs(repo_root)
        valid_specs = [
            s for s in candidate_specs
            if os.path.isfile(s["ckpt_abs"]) and os.path.isfile(s["data_abs"])
        ]
        if valid_specs or (custom_bundles is None and not skip_uda_demos):
            try:
                import torch  # noqa: F401
                import explorer.server as srv
            except ImportError as exc:
                print(f"[build_static_site] Heavy ML deps unavailable ({exc}); falling back to lightweight static build.")
                valid_specs = []
                srv = None
        else:
            srv = None

        if srv is not None:
            for spec in valid_specs:
                b_id = spec["id"]
                print(f"[build_static_site] Exporting SOM bundle '{b_id}'...")
                inf_res = srv._run_inference(spec["ckpt_abs"], spec["data_abs"], spec.get("split", "test"))
                _export_single_som_bundle(
                    srv,
                    out_dir,
                    b_id,
                    inf_res,
                    max_samples=int(spec.get("max_samples", max_samples)),
                )
                ckpt_val = spec["ckpt_val"]
                data_val = spec["data_val"]
                split_val = spec.get("split", "test")
                if ckpt_val not in seen_models:
                    seen_models.add(ckpt_val)
                    models_list.append({
                        "label":    spec.get("ckpt_label", ckpt_val),
                        "value":    ckpt_val,
                        "rel_path": ckpt_val,
                    })
                if data_val not in seen_datasets:
                    seen_datasets.add(data_val)
                    datasets_list.append({
                        "label": spec.get("data_label", data_val),
                        "value": data_val,
                    })
                bundles_index[f"{ckpt_val}|{data_val}|{split_val}"] = b_id
                bundle_meta[b_id] = {
                    "ckpt":      ckpt_val,
                    "data_path": data_val,
                    "split":     split_val,
                    "url":       f"/static-data/som/{b_id}.json",
                }

            if custom_bundles is None and not skip_uda_demos:
                print("[build_static_site] Exporting precomputed UDA Playground demos...")
                pkl_datasets, static_demos = _export_uda_demos(srv, out_dir, repo_root, max_samples=max_samples)

    uda_payload = build_uda_results_payload(
        repo_root,
        pkl_datasets=pkl_datasets,
        static_demos=static_demos,
        static_mode=True,
    )
    _write_json(os.path.join(out_dir, "static-data", "uda_results.json"), uda_payload, ndigits=5)

    models_payload = {
        "static_mode":  True,
        "models":       models_list,
        "datasets":     datasets_list,
        "default_data": datasets_list[0]["value"] if datasets_list else "",
        "bundles":      bundles_index,
        "bundle_meta":  bundle_meta,
    }
    _write_json(os.path.join(out_dir, "static-data", "models.json"), models_payload, ndigits=None)

    return {
        "out_dir":       out_dir,
        "n_bundles":     len(bundle_meta),
        "n_uda_demos":   len(static_demos),
        "lightweight":   lightweight or len(bundle_meta) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static ECG-SOM Explorer site for Vercel/CDN hosting.")
    parser.add_argument("--out-dir", default=os.path.join(_SCRIPT_DIR, "dist"))
    parser.add_argument("--lightweight", action="store_true", help="Skip model/dataset inference; export Overview, UDA benchmarks, and figures only.")
    parser.add_argument("--if-missing", action="store_true", help="Exit 0 without overwriting if static-data/models.json already exists.")
    parser.add_argument("--max-samples", type=int, default=4, help="Max sample beats saved per cell JSON (keeps static bundle compact).")
    parser.add_argument("--skip-uda-demos", action="store_true", help="Skip exporting precomputed UDA Playground adapted maps.")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    sentinel = os.path.join(out_dir, "static-data", "models.json")
    if args.if_missing and os.path.isfile(sentinel):
        _copy_static_shell(_REPO_ROOT, out_dir)
        print(f"[build_static_site] Found existing snapshots at {sentinel}; refreshed HTML/CSS/JS shell (--if-missing).")
        return

    summary = build_static_site(
        out_dir,
        repo_root=_REPO_ROOT,
        lightweight=args.lightweight,
        max_samples=args.max_samples,
        skip_uda_demos=args.skip_uda_demos,
    )
    print(f"[build_static_site] Done: {summary}")


if __name__ == "__main__":
    main()
