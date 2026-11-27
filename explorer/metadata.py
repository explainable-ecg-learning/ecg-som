"""Pure-stdlib metadata, architecture specs, and benchmark helpers for ECG-SOM Explorer.

Shared between ``explorer/server.py`` (FastAPI backend) and
``explorer/build_static_site.py`` (Vercel static bundle exporter) so the
lightweight GitHub -> Vercel build path has zero heavy runtime dependencies.
"""
from __future__ import annotations

import json
import math
import os

FIGURE_MAP: dict[str, tuple[str, str, str]] = {
    "architecture": (
        os.path.join("docs", "architecture.png"),
        "DPSOM-ECG Architecture",
        "Three disentangled Conv1D streams (Morphology VAE + Age/Sex encoders), Gram-Schmidt residualization, toroidal SOM, and FiLM decoder.",
    ),
    "loss": (
        os.path.join("docs", "loss.png"),
        "3-Phase Training Loss Landscape",
        "Autoencoder pretraining, closed-form demographic linear probes, and joint ELBO + toroidal SOM + DEC optimization.",
    ),
    "datasets_overview": (
        os.path.join("docs", "datasets_overview.png"),
        "Three-Cohort Morphology & Demographic Overview",
        "Class distributions, demographic histograms, and standardized beat overlays across PTB-XL, Georgia (CinC 2020), and Chapman-Shaoxing.",
    ),
    "som_overview": (
        os.path.join("docs", "som_overview.png"),
        "Toroidal 8×8 SOM Topology & Occupancy",
        "Node occupancy entropy, dominant-class regions, and U-matrix boundary distances across the toroidal lattice.",
    ),
    "som_prototypes": (
        os.path.join("docs", "som_prototypes.png"),
        "Decoded 8×8 SOM Cell Prototypes",
        "Decoder waveform reconstructions across all 64 SOM prototypes showing smooth neighborhood morphological transitions.",
    ),
    "latent_disentanglement": (
        os.path.join("docs", "latent_disentanglement.png"),
        "Latent Disentanglement & Domain Overlap",
        "Cross-cohort latent projection showing shared mu_res manifold geometry alongside separated age/sex demographic subspaces.",
    ),
    "ecg_reconstructions": (
        os.path.join("docs", "ecg_reconstructions.png"),
        "12-Lead Beat Reconstructions",
        "Original vs. LeadWiseConvDecoder waveform reconstructions across representative clinical categories.",
    ),
    "head_ablations": (
        os.path.join("ablation", "head_ablations.png"),
        "Auxiliary Head & Residualization Ablations",
        "Impact of record attention, age-correction MLP, periodic probe refit, RR context, and Gram-Schmidt residualization across 3 seeds.",
    ),
    "baseline_comparison": (
        os.path.join("ablation", "baseline_comparison.png"),
        "Three-Cohort Matched-k Baseline Comparison",
        "PTB-XL, Georgia, and Chapman-Shaoxing clustering & demographic benchmarks comparing VAE+k-means, 1D-ResNet, and DPSOM-ECG variants.",
    ),
    "cross_dataset_transfer": (
        os.path.join("ablation", "cross_dataset_transfer.png"),
        "Cross-Dataset Transfer Matrix",
        "Zero-shot bidirectional transfer matrices and unlabeled SOM adaptation (+UDA) across PTB-XL, Georgia, and Chapman-Shaoxing.",
    ),
    "cross_dataset_topology": (
        os.path.join("ablation", "cross_dataset_topology.png"),
        "Cross-Dataset Topology & Codebook Health",
        "Active SOM nodes, quantization error (QE), topographic error (TE), and beat NMI across zero-shot and UDA transfers.",
    ),
    "uda_ablation": (
        os.path.join("ablation", "uda_ablation.png"),
        "Stronger UDA Ablation (BW-OT + Hybrid Damped-DEC + SWA)",
        "Comparison of Vanilla UDA, closed-form BW-OT, Source-Free/OT Hybrid Damped-DEC+SWA, Replay, and 100-shot head calibration (8 pairs × 3 seeds).",
    ),
}

REGIME_META: dict[str, dict] = {
    "zero_shot":         {"label": "Zero-Shot",           "target_labels": 0,   "desc": "Frozen source checkpoint evaluated directly on target test split."},
    "uda_vanilla":       {"label": "Vanilla UDA",         "target_labels": 0,   "desc": "6-epoch beat-level DEC + SOM + ELBO fine-tuning with demographic heads frozen."},
    "uda_bw_ot":         {"label": "BW-OT (0.1 s)",       "target_labels": 0,   "desc": "Closed-form FP64 Bures-Wasserstein optimal transport folded into linear projections (0 backprop)."},
    "uda_sf_strong":     {"label": "SF Hybrid DEC+SWA",   "target_labels": 0,   "desc": "Source-free Hybrid Beat+Record Damped-DEC + Cosine LR + Stochastic Weight Averaging."},
    "uda_ot_strong":     {"label": "OT + Hybrid DEC+SWA", "target_labels": 0,   "desc": "BW-OT demographic alignment followed by Hybrid Damped-DEC + SWA on morphology/SOM."},
    "uda_replay_strong": {"label": "Replay + DEC+SWA",    "target_labels": 0,   "desc": "BW-OT demographic alignment + joint source-replay Hybrid Damped-DEC + SWA."},
    "fewshot_100_head":  {"label": "100-Shot Head Cal.",  "target_labels": 100, "desc": "BW-OT initialization + L2-SP head-only calibration on 100 stratified target records."},
}

ARCHITECTURE_SPECS: list[dict] = [
    {
        "id":      "input",
        "title":   "12-Lead Beat Extraction & Normalization",
        "badge":   "Preprocessing",
        "file":    "src/data/preprocessing.py · src/data/cinc.py",
        "dims":    "[B, N, 12, 120] @ 100 Hz (1.2 s R-peak centered) + beat_meta [24]",
        "summary": "Resamples 12-lead ECGs to 100 Hz, detects R-peaks with Pan-Tompkins + NeuroKit2 cross-validation, crops 1.2 s R-peak windows with cosine-tapered masks, normalizes morphology per beat, and extracts 24 lead-wise scale features (12 IQR σ + 12 max |x|).",
        "formula": "x_norm = (x - median) / sigma,  beat_meta = [log(sigma), amax] in R^24",
        "finding": "Standardizing morphology scales enables strong zero-shot cross-continental transfer across PTB-XL, Georgia, and Chapman.",
    },
    {
        "id":      "encoders",
        "title":   "Three Disentangled Lead-Wise Conv1D Encoders",
        "badge":   "Representation",
        "file":    "src/model/codec.py · src/model/dpsom.py",
        "dims":    "mu, logvar in R^32  |  z_age in R^8  |  z_sex in R^8",
        "summary": "Grouped depthwise Conv1d blocks (32 -> 64 channels, kernel=7, GroupNorm, Squeeze-and-Excitation) run independently for each of the 12 leads across three decoupled streams: morphology VAE (z), age stream (z_age), and sex stream (z_sex).",
        "formula": "z ~ N(mu(x), diag(sigma^2)),  pred_age(z_age, meta),  pred_sex(z_sex, meta)",
        "finding": "Separating demographic streams prevents age/sex gradients from polluting unsupervised beat clustering on the SOM.",
    },
    {
        "id":      "residualization",
        "title":   "Closed-Form Probe Residualization",
        "badge":   "Disentanglement",
        "file":    "src/model/dpsom.py (residualize_latent) · src/training/trainer.py",
        "dims":    "2D subspace projection in R^32 (Gram-Schmidt u_age _|_ u_sex)",
        "summary": "Fits closed-form ridge age regression and L2 logistic sex classification probes directly on frozen Phase-1 latent codes, then projects out the orthonormalized age and sex subspaces before SOM distance computation.",
        "formula": "z_res = z - (z . u_age) u_age - (z . u_sex_|_) u_sex_|_",
        "finding": "Drops demographic leakage to R^2 -> 0.00 / AUROC -> 0.50 in z_res while preserving clinical Rec AMI (matched-k).",
    },
    {
        "id":      "som",
        "title":   "Toroidal 8×8 Probabilistic SOM + FiLM Decoder",
        "badge":   "Clustering & Prototypes",
        "file":    "src/model/dpsom.py · src/model/layers.py",
        "dims":    "Codebook E in R^(8x8x32)  |  Wrap-around Von-Neumann neighborhood",
        "summary": "Maps residualized latents z_res to a 64-unit toroidal SOM using Student-t soft assignments q_ik, periodic wrap-around neighbor coupling (beta), DEC cluster sharpening KL(p || q) (gamma), and temporal smoothness along consecutive beats (tau). FiLM-conditioned LeadWiseConvDecoder decodes every cell embedding to a 12-lead ECG prototype.",
        "formula": "q_ik propto (1 + ||z_res - e_k||^2 / alpha)^(-(alpha+1)/2),  L = theta*ELBO + gamma*KL(p||q) + beta*L_som + tau*L_temp",
        "finding": "Pruned 8×8 SOM achieves 100% interpretable prototypes and outperforms 4×4 and 16×16 grids on matched-k record AMI.",
    },
    {
        "id":      "training",
        "title":   "3-Phase Training & Pruned Architecture",
        "badge":   "Optimization",
        "file":    "src/training/trainer.py · src/config.py",
        "dims":    "10 pretrain ep -> Closed-form probes -> 24 joint ep (bf16 + GPU preload)",
        "summary": "Phase 1 trains morphology VAE + demographic predictors; Phase 1.5/1.6 fits and freezes age/sex residualization probes; Phase 2 initializes SOM prototypes via k-means/grad steps; Phase 3 jointly refines VAE, SOM, and heads with val-NMI checkpoint selection.",
        "formula": "Pruned defaults: eta=0, age_corr_lambda_max=0, use_rr=False, probe_refit=0",
        "finding": "3-seed ablation proved pruned defaults match full auxiliary complexity with cleaner cross-cohort robustness (0.1914 Georgia / 0.1989 Chapman Rec AMI).",
    },
    {
        "id":      "uda",
        "title":   "Stronger UDA (BW-OT + Hybrid Damped-DEC + SWA)",
        "badge":   "Domain Adaptation",
        "file":    "src/training/uda.py · ablation/run_uda_ablation.py",
        "dims":    "0-ep analytical BW-OT (~0.1s) + 8-ep Hybrid DEC (alpha_rec=0.35, rho=0.75) + SWA",
        "summary": "Solves demographic shift analytically via closed-form FP64 Bures-Wasserstein Gaussian optimal transport folded into linear weights (enc_sex, enc_age, amp_proj_*), paired with Hybrid Beat+Record Damped-DEC + Cosine LR + Stochastic Weight Averaging on morphology/SOM.",
        "formula": "A = C_t^(-1/2) (C_t^(1/2) C_s C_t^(1/2))^(1/2) C_t^(-1/2),  p_hyb = q_hyb^2 / f_k^rho",
        "finding": "BW-OT raises Sex AUROC on 8/8 transfer pairs (0.7988 -> 0.8074, 0 labels); Hybrid DEC+SWA lifts PTB-XL Full -> Georgia Rec AMI (0.1186 -> 0.1282) & cuts TE (0.165 -> 0.152).",
    },
]

DATASET_SPECS: list[dict] = [
    {
        "id":       "ptbxl",
        "name":     "PTB-XL",
        "region":   "Germany / Europe (21.8k total)",
        "records":  "13,504 train · 1,714 val · 2,156 test (curated 17,374)",
        "classes":  "16 clinical diagnostic subclasses (NORM, IMI, AMI, STTC, LAFB/LPFB, LVH, CLBBB, CRBBB, ...)",
        "age_sex":  "Age 59.3±16.8 yr · 48.3% Female",
        "notes":    "Filtered via ecg-data-repo ptbxl_error_labels.csv (severity <= 2, excluding zero-diagnostic records).",
    },
    {
        "id":       "georgia",
        "name":     "Georgia (CinC 2020)",
        "region":   "Southeastern USA (Emory / Grady)",
        "records":  "1,930 train · 241 val · 241 test (stratified transfer) · 2,412 full",
        "classes":  "17 SNOMED-CT clinical diagnoses (NSR, AFIB, IAVB, RBBB, SB, STach, TInv, ...)",
        "age_sex":  "Age 60.2±15.3 yr · 48.1% Female",
        "notes":    "High multi-ethnic structural diversity; strong bidirectional zero-shot symmetry with PTB-XL.",
    },
    {
        "id":       "chapman",
        "name":     "Chapman-Shaoxing",
        "region":   "China (PhysioNet ecg-arrhythmia 1.0.0)",
        "records":  "1,864 train · 233 val · 233 test (stratified transfer) · 2,330 full",
        "classes":  "8 SNOMED-CT rhythm/conduction classes (82.1% SB/NSR/AFL/STach)",
        "age_sex":  "Age 58.1±19.2 yr · 42.8% Female",
        "notes":    "Rhythm-skewed class mix tests UDA robustness against strong class-imbalance shift.",
    },
]


def safe_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def read_json_if_exists(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def compute_uda_overall_means(summary: list[dict]) -> list[dict]:
    """Compute overall 8-pair mean metrics across all UDA regimes."""
    by_reg: dict[str, list[dict]] = {}
    for row in summary:
        by_reg.setdefault(row["regime"], []).append(row)

    metric_keys = [
        "Rec_AMI", "NMI", "Rec_Purity", "MAE_Age", "AUC_Sex",
        "TE", "QE", "Active_Nodes", "Occ_Entropy",
    ]
    ordered = [r for r in REGIME_META if r in by_reg] + [r for r in sorted(by_reg) if r not in REGIME_META]
    zs_rows = by_reg.get("zero_shot", [])
    zs_ami = sum(float(r["Rec_AMI_mean"]) for r in zs_rows) / len(zs_rows) if zs_rows else 0.0

    out = []
    for reg in ordered:
        grp = by_reg[reg]
        strong_grp = [r for r in grp if r["source"] == "ptbxl_full"]
        item: dict = {
            "regime":        reg,
            "label":         REGIME_META.get(reg, {}).get("label", reg),
            "target_labels": REGIME_META.get(reg, {}).get("target_labels", 0),
            "desc":          REGIME_META.get(reg, {}).get("desc", ""),
            "n_pairs":       len(grp),
        }
        for mk in metric_keys:
            vals = [float(r[f"{mk}_mean"]) for r in grp if f"{mk}_mean" in r]
            item[mk] = (sum(vals) / len(vals)) if vals else None
        if item["Rec_AMI"] is not None:
            item["delta_Rec_AMI"] = float(item["Rec_AMI"] - zs_ami)
        if strong_grp:
            item["strong_Rec_AMI"] = sum(float(r["Rec_AMI_mean"]) for r in strong_grp) / len(strong_grp)
        out.append(item)
    return out


def build_overview_payload(
    repo_root: str,
    *,
    figure_url_template: str = "/api/figures/{fig_id}",
) -> dict:
    """Build structured System Overview payload matching ``GET /api/overview``."""
    ablation_dir = os.path.join(repo_root, "ablation")
    paper = read_json_if_exists(os.path.join(ablation_dir, "paper_results_all.json")) or {}
    uda_res = read_json_if_exists(os.path.join(ablation_dir, "uda_ablation_results.json")) or {}

    benchmarks = []
    for row in paper.get("summary", []):
        benchmarks.append({
            "model":             row.get("model"),
            "n_seeds":           row.get("n_seeds", 3),
            "ptbxl_Rec_AMI":     safe_float(row.get("Rec_AMI_k64_mean", row.get("Rec_AMI_mean"))),
            "ptbxl_Rec_AMI_std": safe_float(row.get("Rec_AMI_k64_std", row.get("Rec_AMI_std"))),
            "ptbxl_MAE_Age":     safe_float(row.get("MAE_Age_mean")),
            "ptbxl_AUC_Sex":     safe_float(row.get("AUC_Sex_mean")),
            "georgia_Rec_AMI":   safe_float(row.get("ext_Rec_AMI_k64_mean", row.get("ext_Rec_AMI_mean"))),
            "georgia_MAE_Age":   safe_float(row.get("ext_MAE_Age_mean")),
            "georgia_AUC_Sex":   safe_float(row.get("ext_AUC_Sex_mean")),
            "chapman_Rec_AMI":   safe_float(row.get("ext_chapman_shaoxing_Rec_AMI_k64_mean", row.get("ext_chapman_shaoxing_Rec_AMI_mean"))),
            "chapman_MAE_Age":   safe_float(row.get("ext_chapman_shaoxing_MAE_Age_mean")),
            "chapman_AUC_Sex":   safe_float(row.get("ext_chapman_shaoxing_AUC_Sex_mean")),
        })

    uda_highlights = compute_uda_overall_means(uda_res.get("summary", [])) if uda_res.get("summary") else []

    figures = []
    for fig_id, (rel_path, title, caption) in FIGURE_MAP.items():
        if os.path.isfile(os.path.join(repo_root, rel_path)):
            figures.append({
                "id":      fig_id,
                "url":     figure_url_template.format(fig_id=fig_id),
                "title":   title,
                "caption": caption,
            })

    return {
        "architecture":   ARCHITECTURE_SPECS,
        "datasets":       DATASET_SPECS,
        "benchmarks":     benchmarks,
        "uda_highlights": uda_highlights,
        "figures":        figures,
    }


def build_uda_results_payload(
    repo_root: str,
    *,
    pkl_datasets: list[dict] | None = None,
    static_demos: list[dict] | None = None,
    static_mode: bool = False,
) -> dict:
    """Build structured UDA benchmark payload matching ``GET /api/uda/results``."""
    ablation_dir = os.path.join(repo_root, "ablation")
    uda_res = read_json_if_exists(os.path.join(ablation_dir, "uda_ablation_results.json")) or {
        "runs": [], "summary": [], "chance": {},
    }
    xds_res = read_json_if_exists(os.path.join(ablation_dir, "cross_dataset_results.json")) or {
        "summary": [], "chance": {},
    }
    payload: dict = {
        "uda_ablation": {
            "runs":          uda_res.get("runs", []),
            "summary":       uda_res.get("summary", []),
            "chance":        uda_res.get("chance", {}),
            "regime_meta":   REGIME_META,
            "overall_means": compute_uda_overall_means(uda_res.get("summary", [])),
        },
        "cross_dataset": {
            "summary": xds_res.get("summary", []),
            "chance":  xds_res.get("chance", {}),
        },
        "pkl_datasets": pkl_datasets or [],
    }
    if static_mode:
        payload["static_mode"] = True
        payload["static_demos"] = static_demos or []
    return payload
