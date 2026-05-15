"""
Ablation study for DPSOM-ECG.

Each ablation variant is defined as an override dict applied to DPSOM_Config.
The main() function from dpsom_ecg.py is reused so every variant goes through
the same train / eval / visualisation pipeline.

Variants
--------
baseline          : full model as configured in dpsom_config.py
no_disentangle    : remove demographic disentanglement (delta_age=0, delta_sex=0)
no_temporal       : remove temporal smoothness loss      (tau=0)
no_record_attn    : remove record-level attention        (eta=0)
no_som_commit     : remove SOM commitment loss           (beta=0)
no_som_smooth     : remove SOM neighbourhood loss        (alpha=0)
no_kl             : remove KL term                       (gamma=0)
small_latent      : latent_dim=16  (half of baseline)
large_latent      : latent_dim=64  (double of baseline)
small_som         : som_dim=(4,4)
large_som         : som_dim=(16,16)

Loss weight sweeps  (low = 0.1× baseline, high = 5–10× baseline)
-----------------------------------------------------------------
theta_low/high     : reconstruction weight      (baseline 0.1)
alpha_low/high     : SOM neighbourhood smooth   (baseline 10.0)
beta_low/high      : SOM commitment             (baseline 0.5)
gamma_low/high     : KL divergence              (baseline 10.0)
tau_low/high       : temporal smoothness        (baseline 1.6)
eta_low/high       : record-level attention     (baseline 1.0)
delta_age_low/high : age disentanglement        (baseline 1.0)
delta_sex_low/high : sex disentanglement        (baseline 1.0)

Usage
-----
    cd DisentangledECG
    source .venv/bin/activate
    python3 ablation.py                       # all variants
    python3 ablation.py --variants baseline no_temporal no_disentangle
    python3 ablation.py --list                # print available variants
"""

import argparse
import copy
import csv
import json
import os
import sys
from datetime import date

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from dpsom_config import DPSOM_Config
from dpsom_ecg import main as _train_eval_main


# ---------------------------------------------------------------------------
# Variant registry
# Each entry: (display_name, {config_attr: value, ...})
# ---------------------------------------------------------------------------
VARIANTS: dict[str, dict] = {
    "baseline": {},

    # --- Disentanglement components ---
    "no_disentangle": {"delta_age": 0.0, "delta_sex": 0.0},
    "no_age":         {"delta_age": 0.0},
    "no_sex":         {"delta_sex": 0.0},

    # --- Loss term ablations ---
    "no_temporal":    {"tau":   0.0},
    "no_record_attn": {"eta":   0.0},
    "no_som_commit":  {"beta":  0.0},
    "no_som_smooth":  {"alpha": 0.0},
    "no_kl":          {"gamma": 0.0},

    # --- Architecture ablations ---
    "small_latent":   {"latent_dim": 16},
    "large_latent":   {"latent_dim": 64},
    "small_som":      {"som_dim": (4, 4)},
    "large_som":      {"som_dim": (16, 16)},

    # --- Loss weight sweeps ---
    # theta: reconstruction weight (baseline 0.1)
    "theta_low":          {"theta": 0.01},
    "theta_high":         {"theta": 1.0},

    # alpha: SOM neighbourhood smoothness (baseline 10.0)
    "alpha_low":          {"alpha": 1.0},
    "alpha_high":         {"alpha": 50.0},

    # beta: SOM commitment (baseline 0.5)
    "beta_low":           {"beta": 0.05},
    "beta_high":          {"beta": 5.0},

    # gamma: KL divergence (baseline 10.0)
    "gamma_low":          {"gamma": 1.0},
    "gamma_high":         {"gamma": 50.0},

    # tau: temporal smoothness (baseline 1.6)
    "tau_low":            {"tau": 0.2},
    "tau_high":           {"tau": 8.0},

    # eta: record-level attention (baseline 1.0)
    "eta_low":            {"eta": 0.1},
    "eta_high":           {"eta": 5.0},

    # delta_age: age disentanglement (baseline 1.0)
    "delta_age_low":      {"delta_age": 0.1},
    "delta_age_high":     {"delta_age": 5.0},

    # delta_sex: sex disentanglement (baseline 1.0)
    "delta_sex_low":      {"delta_sex": 0.1},
    "delta_sex_high":     {"delta_sex": 5.0},
}


def _make_config(variant_name: str, overrides: dict) -> DPSOM_Config:
    cfg = DPSOM_Config()
    cfg.use_data_cache = True   # reuse the cached dataset for all variants
    for k, v in overrides.items():
        setattr(cfg, k, v)
    # Rebuild derived fields that depend on changed attributes
    import uuid
    cfg.ex_name = (
        f"abl_{variant_name}_{cfg.latent_dim}"
        f"_{cfg.som_dim[0]}-{cfg.som_dim[1]}"
        f"_{str(date.today())}_{uuid.uuid4().hex[:5]}"
    )
    cfg.logdir    = f"logs/{cfg.ex_name}"
    cfg.modelpath = f"models/{cfg.ex_name}/{cfg.ex_name}.ckpt"
    return cfg


def _flatten_results(variant: str, results: dict) -> dict:
    """Flatten nested disentanglement sub-dict into a single-level row."""
    row = {"variant": variant}
    for k, v in results.items():
        if k == "Disentanglement":
            if isinstance(v, dict):
                for rep_key, rep_block in v.items():
                    interp = rep_block.get("interpretation", {}) if isinstance(rep_block, dict) else {}
                    for metric, val in interp.items():
                        row[f"{rep_key}/{metric}"] = val
        else:
            row[k] = v
    return row


def run_ablation(variants_to_run: list[str], results_path: str = "ablation_results.csv") -> None:
    all_rows = []

    for variant_name in variants_to_run:
        if variant_name not in VARIANTS:
            print(f"[ablation] Unknown variant '{variant_name}', skipping.", file=sys.stderr)
            continue

        overrides = VARIANTS[variant_name]
        print(f"\n{'='*70}")
        print(f"  ABLATION VARIANT: {variant_name}")
        print(f"  Overrides: {overrides if overrides else '(none – baseline)'}")
        print(f"{'='*70}\n")

        cfg = _make_config(variant_name, overrides)

        try:
            results = _train_eval_main(config=cfg)
        except Exception as exc:
            print(f"[ablation] Variant '{variant_name}' FAILED: {exc}", file=sys.stderr)
            import traceback; traceback.print_exc()
            results = {"error": str(exc)}

        row = _flatten_results(variant_name, results or {})
        all_rows.append(row)

        # Save after every variant so partial results survive a crash
        _write_csv(all_rows, results_path)
        print(f"\n[ablation] Results so far saved to {results_path}")

    # Also write a pretty JSON summary
    json_path = results_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"[ablation] JSON summary written to {json_path}")


def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})


def main() -> None:
    parser = argparse.ArgumentParser(description="DPSOM-ECG ablation study")
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS.keys()),
        help="Which variants to run (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available variants and exit",
    )
    parser.add_argument(
        "--output", default="ablation_results.csv",
        help="CSV path for results (default: ablation_results.csv)",
    )
    args = parser.parse_args()

    if args.list:
        print("Available ablation variants:")
        for name, overrides in VARIANTS.items():
            desc = "baseline (no changes)" if not overrides else str(overrides)
            print(f"  {name:<20} {desc}")
        return

    run_ablation(args.variants, results_path=args.output)


if __name__ == "__main__":
    main()
