"""Export a self-contained static Vercel bundle under ``explorer/vercel/``.

Follows the zero-cold-start architecture of ``ecg-heartbeat-cbm``: precomputes
SOM inference grids, per-cell demographic stats & waveforms, UDA Playground
demos, benchmarks, and figures into ``explorer/vercel/data/`` alongside a standalone
``vercel.json``, ``README.md``, ``index.html``, and ``static/`` bundle so Vercel
deploys directly from Root Directory ``explorer/vercel`` with zero build step.

Usage:
    python explorer/build_vercel.py [--out-dir explorer/vercel] [--lightweight]
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

VERCEL_DIR = os.path.join(_SCRIPT_DIR, "vercel")

_VERCEL_CONFIG = {
    "$schema": "https://openapi.vercel.sh/vercel.json",
    "cleanUrls": True,
    "headers": [
        {
            "source": "/static/(.*)",
            "headers": [
                {
                    "key": "Cache-Control",
                    "value": "public, max-age=3600, stale-while-revalidate=86400",
                }
            ],
        },
        {
            "source": "/data/(.*)",
            "headers": [
                {
                    "key": "Cache-Control",
                    "value": "public, max-age=86400, stale-while-revalidate=604800",
                }
            ],
        },
    ],
}

_VERCEL_README = """# ECG-SOM Explorer — Vercel Static Deployment

Zero-cold-start, static CDN build of the interactive **ECG-SOM Explorer**
supporting **SOM Explorer**, **Cell Compare**, **UDA Explorer** (168-run
benchmark + precomputed UDA Playground demos), and **System Overview** across
**PTB-XL**, **Georgia (CinC 2020)**, and **Chapman-Shaoxing**.

## Why Static Export?

PyTorch CPU wheels exceed Vercel's `250 MB` serverless bundle limit and add
slow cold starts. Instead, [`explorer/build_vercel.py`](../build_vercel.py)
exports precomputed SOM grids, per-cell embeddings & demographic histograms
(for client-side Cell Compare), lazy per-cell waveforms/percentiles, and
precomputed UDA adapted maps directly into `./data/`.

You can also connect a live FastAPI GPU/CPU backend at any time by appending
`?api=https://your-fastapi-host` to the deployed Vercel URL.

## Deploy to Vercel

### Option 1 — Vercel Dashboard / GitHub Auto-Deploy (Recommended)
1. Push this repository to GitHub.
2. Import the repository in the Vercel Dashboard.
3. Set **Root Directory** to `explorer/vercel`, leave **Framework Preset** as `Other`,
   leave **Build Command** empty, and click **Deploy**.
4. Open the project's **Analytics** tab and click **Enable** to activate Vercel
   Web Analytics (`/_vercel/insights/script.js` is already wired in `index.html`).

### Option 2 — Vercel CLI
```bash
cd explorer/vercel
npx vercel --prod
```

## Regenerating the Static Bundle

Whenever checkpoints, showcase splits, or benchmark results change, regenerate
`explorer/vercel/`:

```bash
python explorer/build_vercel.py
```

And preview locally before deploying:

```bash
python3 -m http.server 8041 --directory explorer/vercel
```
"""


def _sanitize_json_value(obj: object, ndigits: int = 3) -> object:
    """Recursively round floats and replace non-finite floats with None."""
    if isinstance(obj, float):
        return round(obj, ndigits) if math.isfinite(obj) else None
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_value(v, ndigits) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_value(v, ndigits) for k, v in obj.items()}
    return obj


def _write_json(
    path: str,
    payload: object,
    *,
    ndigits: int | None = 3,
    pretty: bool = False,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cleaned = _sanitize_json_value(payload, ndigits=ndigits) if ndigits is not None else payload
    with open(path, "w", encoding="utf-8") as fh:
        if pretty:
            json.dump(cleaned, fh, allow_nan=False, indent=2)
            fh.write("\n")
        else:
            json.dump(cleaned, fh, allow_nan=False, separators=(",", ":"))


def _default_bundle_specs(repo_root: str) -> list[dict]:
    """Return candidate SOM checkpoint/dataset pairs to precompute when available on disk."""
    ptbxl_compact = os.path.join(repo_root, "data", "explorer", "explorer_test.npz")
    ptbxl_pkl = os.path.join(repo_root, "data", "processed", "ptbxl_100_T-12ms.pkl")
    if os.path.isfile(ptbxl_compact):
        ptbxl_abs, ptbxl_val, ptbxl_lbl = ptbxl_compact, "explorer/explorer_test.npz", "PTB-XL (compact .npz — fast)"
    else:
        ptbxl_abs, ptbxl_val, ptbxl_lbl = ptbxl_pkl, "processed/ptbxl_100_T-12ms.pkl", "PTB-XL test (processed)"

    ckpt_best = "models/dpsom_pruned_8x8/dpsom_pruned_8x8_best.ckpt"
    ckpt_large = "models/dpsom_v2_16x16/dpsom_v2_16x16_best.ckpt"

    return [
        {
            "id":         "dpsom_pruned_8x8_best_ptbxl",
            "ckpt_abs":   os.path.join(repo_root, ckpt_best),
            "ckpt_val":   ckpt_best,
            "ckpt_label": "dpsom_pruned_8x8_best (8×8 · NMI=0.1801)",
            "data_abs":   ptbxl_abs,
            "data_val":   ptbxl_val,
            "data_label": ptbxl_lbl,
            "split":      "test",
        },
        {
            "id":         "dpsom_pruned_8x8_best_georgia",
            "ckpt_abs":   os.path.join(repo_root, ckpt_best),
            "ckpt_val":   ckpt_best,
            "ckpt_label": "dpsom_pruned_8x8_best (8×8 · NMI=0.1801)",
            "data_abs":   os.path.join(repo_root, "data", "processed", "georgia_transfer_100.pkl"),
            "data_val":   "processed/georgia_transfer_100.pkl",
            "data_label": "processed/georgia_transfer_100.pkl",
            "split":      "test",
        },
        {
            "id":         "dpsom_pruned_8x8_best_chapman",
            "ckpt_abs":   os.path.join(repo_root, ckpt_best),
            "ckpt_val":   ckpt_best,
            "ckpt_label": "dpsom_pruned_8x8_best (8×8 · NMI=0.1801)",
            "data_abs":   os.path.join(repo_root, "data", "processed", "chapman_transfer_100.pkl"),
            "data_val":   "processed/chapman_transfer_100.pkl",
            "data_label": "processed/chapman_transfer_100.pkl",
            "split":      "test",
        },
        {
            "id":          "dpsom_v2_16x16_best_ptbxl",
            "ckpt_abs":    os.path.join(repo_root, ckpt_large),
            "ckpt_val":    ckpt_large,
            "ckpt_label":  "dpsom_v2_16x16_best (16×16 v2 · RMSE=0.389 · Sex=0.895)",
            "data_abs":    ptbxl_abs,
            "data_val":    ptbxl_val,
            "data_label":  ptbxl_lbl,
            "split":       "test",
            "max_samples": 2,
        },
    ]


def _copy_static_shell(repo_root: str, out_dir: str) -> None:
    """Copy HTML/CSS/JS assets into ``out_dir`` with relative paths + static config."""
    src_static = os.path.join(repo_root, "explorer", "static")
    dst_static = os.path.join(out_dir, "static")
    if os.path.isdir(dst_static):
        shutil.rmtree(dst_static)
    shutil.copytree(src_static, dst_static)

    with open(os.path.join(src_static, "index.html"), encoding="utf-8") as fh:
        html = fh.read()

    preload_tags = (
        '  <link rel="preload" href="./data/models.json" as="fetch" crossorigin="anonymous">\n'
        '  <link rel="preload" href="./data/overview.json" as="fetch" crossorigin="anonymous">\n'
    )
    html = html.replace('  <link rel="stylesheet" href="/static/style.css?v=3">', preload_tags + '  <link rel="stylesheet" href="./static/style.css?v=3">')
    html = html.replace('"/static/', '"./static/')

    for target_html in (os.path.join(out_dir, "index.html"), os.path.join(dst_static, "index.html")):
        with open(target_html, "w", encoding="utf-8") as fh:
            fh.write(html)

    with open(os.path.join(dst_static, "site-config.js"), "w", encoding="utf-8") as fh:
        fh.write("window.EXPLORER_STATIC_MODE = true;\nwindow.EXPLORER_API_BASE = '';\n")

    _write_json(os.path.join(out_dir, "vercel.json"), _VERCEL_CONFIG, ndigits=None, pretty=True)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(_VERCEL_README)


def _export_figures_and_overview(repo_root: str, out_dir: str) -> None:
    fig_dir = os.path.join(out_dir, "data", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    for fig_id, (rel_path, _, _) in FIGURE_MAP.items():
        src = os.path.join(repo_root, rel_path)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(fig_dir, f"{fig_id}.png"))

    overview = build_overview_payload(
        repo_root,
        figure_url_template="./data/figures/{fig_id}.png",
    )
    _write_json(os.path.join(out_dir, "data", "overview.json"), overview, ndigits=5)


def _export_single_som_bundle(
    srv,
    out_dir: str,
    bundle_id: str,
    inf_result: dict,
    *,
    max_samples: int = 4,
) -> None:
    """Write ``data/som/<bundle_id>.json`` and lazy cell files from ``srv._STATE``."""
    som_dim = tuple(int(v) for v in srv._STATE["som_dim"])
    H, W = som_dim
    split_data = srv._load_split_arrays(srv._STATE["data_path"], srv._STATE["split"])
    labels_enc = split_data["labels"]
    ages_np = split_data["ages"]
    sexes_np = split_data["sexes"]
    class_names = split_data["class_names"]

    cell_stats = []
    cells_dir = os.path.join(out_dir, "data", "cells", bundle_id)
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
    _write_json(os.path.join(out_dir, "data", "som", f"{bundle_id}.json"), som_payload, ndigits=3)


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

    ckpt_val = "models/dpsom_pruned_8x8/dpsom_pruned_8x8_best.ckpt"
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
            demo_url = f"./data/uda_demos/{demo_id}.json"
            _write_json(os.path.join(out_dir, "data", "uda_demos", f"{demo_id}.json"), demo_payload, ndigits=4)
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


def build_vercel_bundle(
    out_dir: str = VERCEL_DIR,
    *,
    repo_root: str = _REPO_ROOT,
    lightweight: bool = False,
    max_samples: int = 4,
    skip_uda_demos: bool = False,
    custom_bundles: list[dict] | None = None,
) -> dict:
    """Build self-contained Vercel deployment directory under ``out_dir``."""
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
                print(f"[build_vercel] Heavy ML deps unavailable ({exc}); falling back to lightweight static build.")
                valid_specs = []
                srv = None
        else:
            srv = None

        if srv is not None:
            for spec in valid_specs:
                b_id = spec["id"]
                print(f"[build_vercel] Exporting SOM bundle '{b_id}'...")
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
                    "url":       f"./data/som/{b_id}.json",
                }

            if custom_bundles is None and not skip_uda_demos:
                print("[build_vercel] Exporting precomputed UDA Playground demos...")
                pkl_datasets, static_demos = _export_uda_demos(srv, out_dir, repo_root, max_samples=max_samples)

    uda_payload = build_uda_results_payload(
        repo_root,
        pkl_datasets=pkl_datasets,
        static_demos=static_demos,
        static_mode=True,
    )
    _write_json(os.path.join(out_dir, "data", "uda_results.json"), uda_payload, ndigits=5)

    models_payload = {
        "static_mode":  True,
        "models":       models_list,
        "datasets":     datasets_list,
        "default_data": datasets_list[0]["value"] if datasets_list else "",
        "bundles":      bundles_index,
        "bundle_meta":  bundle_meta,
    }
    _write_json(os.path.join(out_dir, "data", "models.json"), models_payload, ndigits=None)

    return {
        "out_dir":       out_dir,
        "n_bundles":     len(bundle_meta),
        "n_uda_demos":   len(static_demos),
        "lightweight":   lightweight or len(bundle_meta) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static ECG-SOM Explorer bundle under explorer/vercel/.")
    parser.add_argument("--out-dir", default=VERCEL_DIR)
    parser.add_argument("--lightweight", action="store_true", help="Skip model/dataset inference; export Overview, UDA benchmarks, and figures only.")
    parser.add_argument("--max-samples", type=int, default=4, help="Max sample beats saved per cell JSON (keeps static bundle compact).")
    parser.add_argument("--skip-uda-demos", action="store_true", help="Skip exporting precomputed UDA Playground adapted maps.")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    summary = build_vercel_bundle(
        out_dir,
        repo_root=_REPO_ROOT,
        lightweight=args.lightweight,
        max_samples=args.max_samples,
        skip_uda_demos=args.skip_uda_demos,
    )
    print(f"[build_vercel] Done: {summary}")


if __name__ == "__main__":
    main()
