"""
Ablation study for DPSOM-ECG (src/ version).

Each ablation variant is defined as an override dict applied to DPSOM_Config.
The main() function from src.training.trainer is reused so every variant goes
through the same train / eval / visualisation pipeline.

Variants
--------
baseline           : full model as configured in src/config.py

Loss term ablations
-------------------
no_disentangle     : delta_age=0, delta_sex=0
no_age             : delta_age=0
no_sex             : delta_sex=0
no_temporal        : tau=0
no_record_attn     : eta=0
no_som_commit      : beta=0
no_som_smooth      : alpha=0
no_kl              : gamma=0

Architecture ablations
----------------------
small_latent       : latent_dim=16
large_latent       : latent_dim=64
small_som          : som_dim=(4,4)
large_som          : som_dim=(16,16)
narrow_encoder     : encoder_base_channels_1=16, encoder_base_channels_2=32
wide_encoder       : encoder_base_channels_1=64, encoder_base_channels_2=128
small_fc           : encoder_fc_hidden_dim=256
large_fc           : encoder_fc_hidden_dim=1024
small_kernel       : encoder_kernel_size=3
large_kernel       : encoder_kernel_size=11
small_age_latent   : z_age_dim_factor=0.125
large_age_latent   : z_age_dim_factor=0.5

Age-correction ablations
------------------------
no_age_corr        : age_corr_lambda_max=0.0
more_topk          : age_corr_topk=8
less_topk          : age_corr_topk=2
fast_ramp          : age_corr_ramp_epochs=3
slow_ramp          : age_corr_ramp_epochs=20

Loss weight sweeps  (low = 0.1×, high = 5–10× baseline)
---------------------------------------------------------
theta_low/high     : reconstruction weight      (baseline 0.1)
alpha_low/high     : SOM neighbourhood smooth   (baseline 10.0)
beta_low/high      : SOM commitment             (baseline 0.5)
gamma_low/high     : KL divergence              (baseline 10.0)
tau_low/high       : temporal smoothness        (baseline 1.6)
eta_low/high       : record-level attention     (baseline 1.0)
delta_age_low/high : age disentanglement        (baseline 1.0)
delta_sex_low/high : sex disentanglement        (baseline 1.0)

Optimiser sweeps
----------------
lr_low/high        : learning_rate
wd_low/high        : weight_decay
dropout_low/high   : dropout

Usage
-----
    cd /nfs/data8/schlegel/git/ecg-som
    source DisentangledECG/.venv/bin/activate
    python ablation.py                          # all variants
    python ablation.py --variants baseline no_temporal no_disentangle
    python ablation.py --list                   # print available variants
"""

import argparse
import csv
import json
import os
import sys
import uuid
from datetime import date

# Resolve script location so imports and default paths work from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

from src.config import DPSOM_Config
from src.training.trainer import main as _train_eval_main


# ---------------------------------------------------------------------------
# Variant registry
# Each entry: variant_name -> {config_attr: override_value, ...}
# ---------------------------------------------------------------------------
VARIANTS: dict[str, dict] = {
    "baseline": {},

    # --- Loss term ablations ---
    "no_disentangle":   {"delta_age": 0.0, "delta_sex": 0.0},
    "no_age":           {"delta_age": 0.0},
    "no_sex":           {"delta_sex": 0.0},
    "no_temporal":      {"tau":   0.0},
    "no_record_attn":   {"eta":   0.0},
    "no_som_commit":    {"beta":  0.0},
    "no_som_smooth":    {"alpha": 0.0},
    "no_kl":            {"gamma": 0.0},

    # --- Architecture: latent / SOM size ---
    "small_latent":     {"latent_dim": 16},
    "large_latent":     {"latent_dim": 64},
    "small_som":        {"som_dim": (4, 4)},
    "large_som":        {"som_dim": (16, 16)},

    # --- Architecture: encoder width ---
    "narrow_encoder":   {"encoder_base_channels_1": 16, "encoder_base_channels_2": 32},
    "wide_encoder":     {"encoder_base_channels_1": 64, "encoder_base_channels_2": 128},

    # --- Architecture: FC hidden dim ---
    "small_fc":         {"encoder_fc_hidden_dim": 256},
    "large_fc":         {"encoder_fc_hidden_dim": 1024},

    # --- Architecture: conv kernel ---
    "small_kernel":     {"encoder_kernel_size": 3},
    "large_kernel":     {"encoder_kernel_size": 11},

    # --- Demographic latent fraction ---
    "small_age_latent": {"z_age_dim_factor": 0.125},
    "large_age_latent": {"z_age_dim_factor": 0.5},

    # --- Age-correction module ---
    "no_age_corr":      {"age_corr_lambda_max": 0.0},
    "more_topk":        {"age_corr_topk": 8},
    "less_topk":        {"age_corr_topk": 2},
    "fast_ramp":        {"age_corr_ramp_epochs": 3},
    "slow_ramp":        {"age_corr_ramp_epochs": 20},

    # --- Loss weight sweeps ---
    "theta_low":        {"theta": 0.01},
    "theta_high":       {"theta": 1.0},

    "alpha_low":        {"alpha": 1.0},
    "alpha_high":       {"alpha": 50.0},

    "beta_low":         {"beta": 0.05},
    "beta_high":        {"beta": 5.0},

    "gamma_low":        {"gamma": 1.0},
    "gamma_high":       {"gamma": 50.0},

    "tau_low":          {"tau": 0.2},
    "tau_high":         {"tau": 8.0},

    "eta_low":          {"eta": 0.1},
    "eta_high":         {"eta": 5.0},

    "delta_age_low":    {"delta_age": 0.1},
    "delta_age_high":   {"delta_age": 5.0},

    "delta_sex_low":    {"delta_sex": 0.1},
    "delta_sex_high":   {"delta_sex": 5.0},

    # --- Optimiser sweeps ---
    "lr_low":           {"learning_rate": 1e-4, "learning_rate_pretrain": 1e-4},
    "lr_high":          {"learning_rate": 3e-3, "learning_rate_pretrain": 3e-3},
    "wd_low":           {"weight_decay": 1e-5},
    "wd_high":          {"weight_decay": 1e-3},
    "dropout_low":      {"dropout": 0.0},
    "dropout_high":     {"dropout": 0.5},
}


def _make_config(variant_name: str, overrides: dict) -> DPSOM_Config:
    cfg = DPSOM_Config()
    cfg.use_data_cache = True
    for k, v in overrides.items():
        setattr(cfg, k, v)
    # Rebuild derived name / path fields after any attribute changes
    cfg.ex_name = (
        f"abl_{variant_name}_{cfg.latent_dim}"
        f"_{cfg.som_dim[0]}-{cfg.som_dim[1]}"
        f"_{str(date.today())}_{uuid.uuid4().hex[:5]}"
    )
    cfg.logdir    = f"logs/{cfg.ex_name}"
    cfg.modelpath = f"models/{cfg.ex_name}/{cfg.ex_name}.ckpt"
    return cfg


def _flatten_results(variant: str, overrides: dict, results: dict) -> dict:
    """Flatten nested disentanglement sub-dict into a single-level row."""
    row: dict = {"variant": variant}
    # Record which keys were overridden
    row["overrides"] = json.dumps(overrides, default=str)
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


def run_ablation(variants_to_run: list[str], results_path: str = "ablation_results.csv") -> None:
    all_rows: list[dict] = []

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
            results = _train_eval_main(cfg=cfg)
        except Exception as exc:
            print(f"[ablation] Variant '{variant_name}' FAILED: {exc}", file=sys.stderr)
            import traceback; traceback.print_exc()
            results = {"error": str(exc)}
        finally:
            # Reset torch.compile cache so compiled graphs from this variant
            # do not interfere with the next variant (different architecture).
            try:
                import torch
                torch._dynamo.reset()
            except Exception:
                pass

        row = _flatten_results(variant_name, overrides, results or {})
        all_rows.append(row)

        _write_csv(all_rows, results_path)
        print(f"\n[ablation] Results so far saved to {results_path}")

    json_path = results_path.replace(".csv", ".json")
    with open(json_path, "w") as f:
        json.dump(all_rows, f, indent=2, default=str)
    print(f"[ablation] JSON summary written to {json_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DPSOM-ECG ablation study (src/ stack)")
    parser.add_argument(
        "--variants", nargs="+", default=list(VARIANTS.keys()),
        help="Variants to run (default: all). Pass --list to see options.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available variants and exit.",
    )
    parser.add_argument(
        "--output", default=os.path.join(_HERE, "ablation_results.csv"),
        help="Output CSV path (default: ablation/ablation_results.csv).",
    )
    args = parser.parse_args()

    if args.list:
        print("Available ablation variants:")
        for name, overrides in VARIANTS.items():
            desc = "baseline (no changes)" if not overrides else str(overrides)
            print(f"  {name:<22} {desc}")
        return

    run_ablation(args.variants, results_path=args.output)


if __name__ == "__main__":
    main()
