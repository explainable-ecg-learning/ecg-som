"""
Disentanglement metrics (MIG, SAP, DCI) with reproducibility.
Normalized SAP scores (comparable between regression and classification)

Implements:
- Purity (for clustering labels).
- MIG: normalized MI top-1 vs top-2 gap per factor (discrete MI via binning).
- SAP: gap of prediction *error* between best and 2nd-best single-dim predictors.
  * For regression: uses normalized MSE (MSE / variance) in [0, inf)
  * For classification: uses error rate in [0, 1]
- DCI: disentanglement/completeness from importance matrix + informativeness on test set.
"""

from __future__ import annotations

import numpy as np
from sklearn import metrics
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from scipy.stats import entropy as scipy_entropy

from sklearn.metrics import mutual_info_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


# Utilities

def _check_2d(z: np.ndarray, name: str = "z") -> np.ndarray:
    z = np.asarray(z)
    if z.ndim != 2:
        raise ValueError(f"{name} must be 2D array (n_samples, latent_dim). Got shape {z.shape}.")
    if z.shape[0] < 2 or z.shape[1] < 1:
        raise ValueError(f"{name} must have at least 2 samples and 1 dim. Got shape {z.shape}.")
    return z


def _check_1d(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1D array (n_samples,). Got shape {x.shape}.")
    if x.shape[0] < 2:
        raise ValueError(f"{name} must have at least 2 samples. Got shape {x.shape}.")
    return x


def compute_per_lead_pearson(x: np.ndarray, x_hat: np.ndarray) -> Dict[str, Any]:
    """
    Compute reconstruction Pearson correlation per lead.

    Supported shapes:
      - Single beat: (C, T)
      - Batch: (B, C, T)

    For constant signals (std == 0), correlation is returned as NaN for that lead/beat.
    """
    x = np.asarray(x, dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)

    if x.shape != x_hat.shape:
        raise ValueError(f"x and x_hat must have the same shape. Got {x.shape} vs {x_hat.shape}.")
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected shape (C, T) or (B, C, T). Got {x.shape}.")

    if x.ndim == 2:
        x = x[None, ...]
        x_hat = x_hat[None, ...]

    beat_count, lead_count, _ = x.shape
    corr_bc = np.full((beat_count, lead_count), np.nan, dtype=np.float64)

    for b in range(beat_count):
        for c in range(lead_count):
            lead_x = x[b, c, :]
            lead_x_hat = x_hat[b, c, :]
            if np.std(lead_x) == 0.0 or np.std(lead_x_hat) == 0.0:
                continue
            corr_bc[b, c] = np.corrcoef(lead_x, lead_x_hat)[0, 1]

    mean_per_lead = np.full(lead_count, np.nan, dtype=np.float64)
    median_per_lead = np.full(lead_count, np.nan, dtype=np.float64)
    for c in range(lead_count):
        valid = corr_bc[:, c][np.isfinite(corr_bc[:, c])]
        if valid.size == 0:
            continue
        mean_per_lead[c] = float(np.mean(valid))
        median_per_lead[c] = float(np.median(valid))

    valid_lead_means = mean_per_lead[np.isfinite(mean_per_lead)]
    mean_over_leads = float(np.mean(valid_lead_means)) if valid_lead_means.size else float("nan")
    median_over_leads = float(np.median(valid_lead_means)) if valid_lead_means.size else float("nan")

    return {
        "per_lead": mean_per_lead.astype(float).tolist(),
        "aggregates": {
            "mean_over_leads": mean_over_leads,
            "median_over_leads": median_over_leads,
            "mean_per_lead": mean_per_lead.astype(float).tolist(),
            "median_per_lead": median_per_lead.astype(float).tolist(),
        },
        "beat_count": int(beat_count),
        "sample_count": int(beat_count),
        "per_beat_per_lead": corr_bc.astype(float).tolist(),
    }


def som_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): som_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [som_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return som_json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return None if not np.isfinite(v) else v
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _som_pairwise_dist2(z: np.ndarray, e: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    e = np.asarray(e, dtype=np.float32)
    z_norm = np.sum(z * z, axis=1, keepdims=True)
    e_norm = np.sum(e * e, axis=1, keepdims=True).T
    dist2 = z_norm + e_norm - 2.0 * (z @ e.T)
    return np.maximum(dist2, 0.0)


def _som_gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).flatten()
    if x.size == 0:
        return float("nan")
    total = float(x.sum())
    if total <= 0.0:
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(i * x_sorted) / (n * total)) - (n + 1.0) / n)


def _som_neighbor_consistency(z: np.ndarray, bmu: np.ndarray, H: int, W: int, k: int = 10) -> float:
    n = int(z.shape[0])
    if n <= 1:
        return float("nan")
    k = max(1, min(int(k), n - 1))
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(z)
        _, idx = nn.kneighbors(z, return_distance=True)
        neigh = idx[:, 1:]
    except Exception:
        return float("nan")

    bmu_row = bmu // W
    bmu_col = bmu % W
    neigh_bmu = bmu[neigh]
    dr = np.abs((neigh_bmu // W) - bmu_row[:, None])
    dc = np.abs((neigh_bmu % W) - bmu_col[:, None])
    same_or_adj = (dr + dc) <= 1
    return float(np.mean(same_or_adj))


def compute_som_clustering_metrics(
    z: np.ndarray,
    codebook: np.ndarray,
    som_shape: Tuple[int, int],
    labels: Optional[np.ndarray] = None,
    knn_k: int = 10,
) -> Dict[str, Any]:
    z = _check_2d(np.asarray(z), name="z")
    codebook = _check_2d(np.asarray(codebook), name="codebook")
    H, W = int(som_shape[0]), int(som_shape[1])
    total_units = int(H * W)
    if codebook.shape[0] != total_units:
        raise ValueError(f"Codebook first dimension must equal H*W ({total_units}), got {codebook.shape[0]}.")
    if z.shape[1] != codebook.shape[1]:
        raise ValueError(
            f"Latent dim mismatch between z ({z.shape[1]}) and codebook ({codebook.shape[1]})."
        )

    dist2 = _som_pairwise_dist2(z, codebook)
    top2_unsorted = np.argpartition(dist2, kth=1, axis=1)[:, :2]
    ridx = np.arange(z.shape[0])[:, None]
    top2_dist = dist2[ridx, top2_unsorted]
    order = np.argsort(top2_dist, axis=1)
    top2 = top2_unsorted[ridx, order]
    bmu = top2[:, 0].astype(np.int64)
    second_bmu = top2[:, 1].astype(np.int64)

    q_err = float(np.mean(np.sqrt(dist2[np.arange(z.shape[0]), bmu])))
    bmu_r, bmu_c = bmu // W, bmu % W
    b2_r, b2_c = second_bmu // W, second_bmu % W
    adjacent = (np.abs(bmu_r - b2_r) + np.abs(bmu_c - b2_c)) == 1
    topographic_error = float(np.mean(~adjacent))
    neighborhood_consistency = _som_neighbor_consistency(z, bmu, H, W, k=knn_k)

    occupancy = np.bincount(bmu, minlength=total_units).astype(np.int64)
    active_units = int(np.sum(occupancy > 0))
    dead_units = int(total_units - active_units)
    dead_ratio = float(dead_units / total_units) if total_units > 0 else float("nan")

    p = occupancy.astype(np.float64)
    p_sum = float(p.sum())
    if p_sum > 0.0:
        p /= p_sum
        p_nz = p[p > 0.0]
        occ_entropy = float(-np.sum(p_nz * np.log(p_nz)))
        if total_units > 1:
            occ_entropy /= float(np.log(total_units))
    else:
        occ_entropy = float("nan")

    label_metrics = None
    unit_dominant = [None] * total_units
    unit_purity = [None] * total_units
    unit_label_rows = []
    neighbor_edge_rows = []
    if labels is not None:
        labels = np.asarray(labels)
        if labels.shape[0] == bmu.shape[0]:
            labels = labels.astype(str)
            weighted_purity_num = 0.0
            weighted_purity_den = 0.0

            for u in range(total_units):
                cnt = int(occupancy[u])
                if cnt > 0:
                    unit_labels = labels[bmu == u]
                    uniq, cnts = np.unique(unit_labels, return_counts=True)
                    j = int(np.argmax(cnts))
                    dom = str(uniq[j])
                    pur = float(cnts[j] / cnt)
                    unit_dominant[u] = dom
                    unit_purity[u] = pur
                    weighted_purity_num += pur * cnt
                    weighted_purity_den += cnt
                unit_label_rows.append({
                    "unit_id": int(u),
                    "x": int(u // W),
                    "y": int(u % W),
                    "dominant_label": unit_dominant[u],
                    "unit_purity": unit_purity[u],
                    "count": int(cnt),
                })

            global_purity_by_units = (
                float(weighted_purity_num / weighted_purity_den) if weighted_purity_den > 0 else float("nan")
            )

            agreement = []
            transition = {}
            for x in range(H):
                for y in range(W):
                    u = x * W + y
                    if occupancy[u] <= 0 or unit_dominant[u] is None:
                        continue
                    for nx, ny in ((x + 1, y), (x, y + 1)):
                        if nx >= H or ny >= W:
                            continue
                        v = nx * W + ny
                        if occupancy[v] <= 0 or unit_dominant[v] is None:
                            continue
                        la = unit_dominant[u]
                        lb = unit_dominant[v]
                        same = int(la == lb)
                        agreement.append(same)
                        key = f"{la}|{lb}"
                        transition[key] = int(transition.get(key, 0) + 1)
                        neighbor_edge_rows.append({
                            "unit_id_a": int(u),
                            "unit_id_b": int(v),
                            "label_a": la,
                            "label_b": lb,
                            "same_label": int(same),
                        })

            label_metrics = {
                "per_unit_dominant_label": unit_dominant,
                "per_unit_purity": unit_purity,
                "global_purity_by_units": global_purity_by_units,
                "neighbor_label_agreement": float(np.mean(agreement)) if agreement else float("nan"),
                "neighbor_transition_matrix": transition,
            }

    return {
        "topology": {
            "quantization_error": q_err,
            "topographic_error": topographic_error,
            "neighborhood_consistency": neighborhood_consistency,
        },
        "utilization": {
            "total_units": int(total_units),
            "active_units": int(active_units),
            "dead_units": int(dead_units),
            "dead_ratio": dead_ratio,
            "occupancy_counts": occupancy.astype(int).tolist(),
            "occupancy_entropy": occ_entropy,
            "occupancy_gini": _som_gini(occupancy.astype(np.float64)),
            "occupancy_p50": float(np.percentile(occupancy, 50)),
            "occupancy_p90": float(np.percentile(occupancy, 90)),
            "occupancy_max": int(np.max(occupancy)) if occupancy.size else 0,
        },
        "labels": label_metrics,
        "unit_label_rows": unit_label_rows,
        "neighbor_edge_rows": neighbor_edge_rows,
    }


def _encode_as_int_labels(x: np.ndarray) -> np.ndarray:
    """Encode arbitrary labels into contiguous int labels 0..K-1."""
    x = np.asarray(x)
    _, inv = np.unique(x, return_inverse=True)
    return inv.astype(np.int32, copy=False)


def _digitize_uniform(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Uniform-width binning into <= n_bins bins. Returns int labels."""
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Input contains NaN/inf; clean data before discretization.")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    xmin, xmax = float(x.min()), float(x.max())
    if xmin == xmax:
        return np.zeros_like(x, dtype=np.int32)
    edges = np.linspace(xmin, xmax, n_bins + 1)[1:-1]  # internal edges, length n_bins-1
    return np.digitize(x, edges, right=False).astype(np.int32, copy=False)


def _digitize_quantile(x: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Quantile binning into <= n_bins bins. Handles constant features and duplicated quantiles.
    Returns int labels in [0, num_edges].
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Input contains NaN/inf; clean data before discretization.")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    if float(x.min()) == float(x.max()):
        return np.zeros_like(x, dtype=np.int32)

    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]  # internal quantiles
    edges = np.quantile(x, qs, method="linear")
    edges = np.unique(edges)  # drop duplicates (common with discrete/peaky data)
    if edges.size == 0:
        return np.zeros_like(x, dtype=np.int32)
    return np.digitize(x, edges, right=False).astype(np.int32, copy=False)


# Purity
def cluster_purity(labels_true, labels_pred):
    """
    Calculate clustering purity
    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        purity, in [0,1]
    """
    labels_true = labels_true.astype(np.int64)
    assert labels_pred.size == labels_true.size
    D = max(labels_pred.max(), labels_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(labels_pred.size):
        w[labels_pred[i], labels_true[i]] += 1
    label_mapping = w.argmax(axis=1)
    y_pred_voted = labels_pred.copy()
    for i in range(labels_pred.size):
        y_pred_voted[i] = label_mapping[labels_pred[i]]
    return metrics.accuracy_score(y_pred_voted, labels_true)

# MIG

def _quantile_discretize(z, n_bins=10):
    """
    Quantile discretization per latent dimension.
    Returns integer bins [0..n_bins-1]
    """
    z = np.asarray(z)
    z_disc = np.zeros_like(z, dtype=np.int32)

    for j in range(z.shape[1]):
        edges = np.quantile(z[:, j], np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)

        if len(edges) <= 2:
            continue

        z_disc[:, j] = np.digitize(z[:, j], edges[1:-1])

    return z_disc


def _mig_for_factor(z_disc, factor_labels):
    """
    Computes normalized MIG for one factor
    """
    n_latents = z_disc.shape[1]
    mi = np.zeros(n_latents)

    for j in range(n_latents):
        mi[j] = mutual_info_score(z_disc[:, j], factor_labels)

    mi_sorted = np.sort(mi)[::-1]

    if mi_sorted[0] < 1e-12:
        return 0.0, mi_sorted

    gap = mi_sorted[0] - mi_sorted[1]
    mig_norm = gap / (mi_sorted[0] + 1e-12)
    return float(mig_norm), mi_sorted


def compute_mig(z_main, z_age, z_sex, age, sex, n_bins_z=10):
    """
    Normalized MIG (post-β-TCVAE style) for mixed latent groups.

    Parameters
    -
    z_main : [N, Dm]
    z_age  : [N, Da]
    z_sex  : [N, Ds]
    age    : integer labels (2..89)
    sex    : 0/1 labels
    """

    results = {}

    age_labels = age.astype(np.int32)
    age_labels -= age_labels.min()
    sex_labels = sex.astype(np.int32)
    # concatenate latents
    z_age_all = np.concatenate([z_main, z_age], axis=1)
    z_sex_all = np.concatenate([z_main, z_sex], axis=1)
    # discretize latents (quantile bins)
    z_age_disc = _quantile_discretize(z_age_all, n_bins_z)
    z_sex_disc = _quantile_discretize(z_sex_all, n_bins_z)
    # AGE MIG
    mig_age, mi_age_sorted = _mig_for_factor(z_age_disc, age_labels)
    # SEX MIG
    mig_sex, mi_sex_sorted = _mig_for_factor(z_sex_disc, sex_labels)
    # leakage check (main should NOT encode factors)
    z_main_disc = _quantile_discretize(z_main, n_bins_z)
    mig_main_age, _ = _mig_for_factor(z_main_disc, age_labels)
    mig_main_sex, _ = _mig_for_factor(z_main_disc, sex_labels)
    # results
    results["age"] = {
        "mig_norm": mig_age,
        "top1_mi": float(mi_age_sorted[0]),
        "top2_mi": float(mi_age_sorted[1]),
    }

    results["sex"] = {
        "mig_norm": mig_sex,
        "top1_mi": float(mi_sex_sorted[0]),
        "top2_mi": float(mi_sex_sorted[1]),
    }

    results["main_leakage"] = {
        "age_mig": mig_main_age,
        "sex_mig": mig_main_sex,
    }

    return results


# SAP

@dataclass(frozen=True)
class SAPSettings:
    test_size: float = 0.33
    max_train_samples: int = 15000
    random_state: int = 0
    # For discrete factor prediction:
    svc_c: float = 0.01
    svc_max_iter: int = 5000


def compute_sap(
    z: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    settings: SAPSettings = SAPSettings(),
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Separated Attribute Predictability (SAP) with train/test split.

    Compute per-dimension *predictive score* on the test set,
    then SAP_factor = best_score - second_best_score (larger is better).

    Scores:
      - age (continuous): R^2 of LinearRegression on single dimension
      - sex (discrete): accuracy of LinearSVC on single dimension

    Notes:
      - R^2 can be negative (worse than predicting mean).

    Args:
        z: Latent codes, shape (n_samples, latent_dim)
        age: Age values, shape (n_samples,)
        sex: Sex labels, shape (n_samples,)
        settings: SAPSettings configuration
        groups: Optional group IDs for group-aware split (e.g., record IDs)

    Returns:
        Dictionary with SAP scores and per-dimension scores (+ diagnostics)
    """
    z = _check_2d(z, "z")
    age = _check_1d(age, "age").astype(np.float64, copy=False)
    sex = _check_1d(sex, "sex")
    if z.shape[0] != age.size or z.shape[0] != sex.size:
        raise ValueError("z, age, sex must have matching n_samples.")
    if groups is not None:
        groups = _check_1d(groups, "groups")
        if groups.size != z.shape[0]:
            raise ValueError("groups must have the same n_samples as z.")

    # Subsample for speed/reproducibility
    n = z.shape[0]
    rng = np.random.default_rng(settings.random_state)
    if n > settings.max_train_samples:
        idx = rng.choice(n, size=settings.max_train_samples, replace=False)
        z = z[idx]
        age = age[idx]
        sex = sex[idx]
        if groups is not None:
            groups = groups[idx]

    sex_disc = _encode_as_int_labels(sex.astype(np.int64, copy=False))
    # stratify on sex so the split is stable even if imbalanced
    if groups is not None and np.unique(groups).size > 1:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=settings.test_size,
            random_state=settings.random_state,
        )
        train_idx, test_idx = next(splitter.split(z, sex_disc, groups=groups))
        X_train, X_test = z[train_idx], z[test_idx]
        age_train, age_test = age[train_idx], age[test_idx]
        sex_train, sex_test = sex_disc[train_idx], sex_disc[test_idx]
    else:
        X_train, X_test, age_train, age_test, sex_train, sex_test = train_test_split(
            z, age, sex_disc,
            test_size=settings.test_size,
            random_state=settings.random_state,
            shuffle=True,
            stratify=sex_disc if np.unique(sex_disc).size > 1 else None,
        )

    def _scores_for_factor(
        y_train: np.ndarray,
        y_test: np.ndarray,
        kind: str,
    ) -> np.ndarray:
        scores = np.zeros(X_train.shape[1], dtype=np.float64)
        for j in range(X_train.shape[1]):
            xtr = X_train[:, j].reshape(-1, 1)
            xte = X_test[:, j].reshape(-1, 1)

            if kind == "regression":
                model = make_pipeline(StandardScaler(), LinearRegression())
                model.fit(xtr, y_train)
                # sklearn's score() for LinearRegression is R^2
                scores[j] = float(model.score(xte, y_test))
            elif kind == "classification":
                model = make_pipeline(
                    StandardScaler(),
                    LinearSVC(C=settings.svc_c, max_iter=settings.svc_max_iter, random_state=settings.random_state),
                )
                # If train split has 1 class, classifier is undefined -> worst informative score.
                if np.unique(y_train).size < 2:
                    scores[j] = 0.0
                    continue
                model.fit(xtr, y_train)
                # score() is accuracy for classifiers
                scores[j] = float(model.score(xte, y_test))
            else:
                raise ValueError("kind must be 'regression' or 'classification'.")
        return scores

    age_scores = _scores_for_factor(age_train, age_test, "regression")      # R^2 per dim
    sex_scores = _scores_for_factor(sex_train, sex_test, "classification")  # accuracy per dim

    def _sap_gap(scores: np.ndarray, factor_name: str) -> Tuple[float, Dict]:
        if scores.size < 2:
            return 0.0, {}
        sorted_indices = np.argsort(scores)[::-1]  # descending; best is largest score
        sorted_scores = scores[sorted_indices]
        gap = float(sorted_scores[0] - sorted_scores[1])

        diagnostics = {
            f"best_dims_{factor_name}": sorted_indices[:5].tolist(),
            f"best_scores_{factor_name}": sorted_scores[:5].tolist(),
        }
        return gap, diagnostics

    sap_age, diag_age = _sap_gap(age_scores, "age")
    sap_sex, diag_sex = _sap_gap(sex_scores, "sex")

    return {
        "SAP_age": sap_age,
        "SAP_sex": sap_sex,
        "age_score_per_dim": age_scores.tolist(),
        "sex_score_per_dim": sex_scores.tolist(),
        **diag_age,
        **diag_sex,
        "settings": {
            "test_size": settings.test_size,
            "max_train_samples": settings.max_train_samples,
            "random_state": settings.random_state,
            "svc_c": settings.svc_c,
        },
    }


# DCI

@dataclass(frozen=True)
class DCISettings:
    test_size: float = 0.33
    max_samples: int = 15000
    random_state: int = 0
    model: str = "gbt"  # 'gbt' or 'rf' (rf not implemented here, keep API room)
    clip_regression_r2_to_unit: bool = True  # map R2 to [0,1] for averaging


def compute_dci(
    z: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    settings: DCISettings = DCISettings(),
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Disentanglement, Completeness, and Informativeness (DCI) with train/test split.

    DCI builds an importance matrix from feature importances and computes:
    - Disentanglement: Each code should be important for only one factor (high = good)
    - Completeness: Each factor should be captured by only one code (high = good)
    - Informativeness: Overall prediction performance on test set (high = good)

    - age is treated as continuous -> GradientBoostingRegressor
    - sex is treated as discrete -> GradientBoostingClassifier

    Args:
        z: Latent codes, shape (n_samples, latent_dim)
        age: Age values, shape (n_samples,)
        sex: Sex labels, shape (n_samples,)
        settings: DCISettings configuration
        groups: Optional group IDs for group-aware split (e.g., record IDs)

    Returns:
        Dictionary with DCI metrics, importance matrices, and diagnostics
    """
    z = _check_2d(z, "z")
    age = _check_1d(age, "age").astype(np.float64, copy=False)
    sex = _check_1d(sex, "sex")
    if z.shape[0] != age.size or z.shape[0] != sex.size:
        raise ValueError("z, age, sex must have matching n_samples.")
    if groups is not None:
        groups = _check_1d(groups, "groups")
        if groups.size != z.shape[0]:
            raise ValueError("groups must have the same n_samples as z.")

    # Subsample
    n = z.shape[0]
    rng = np.random.default_rng(settings.random_state)
    if n > settings.max_samples:
        idx = rng.choice(n, size=settings.max_samples, replace=False)
        z = z[idx]
        age = age[idx]
        sex = sex[idx]
        if groups is not None:
            groups = groups[idx]

    sex_disc = _encode_as_int_labels(sex.astype(np.int64, copy=False))

    if groups is not None and np.unique(groups).size > 1:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=settings.test_size,
            random_state=settings.random_state,
        )
        train_idx, test_idx = next(splitter.split(z, sex_disc, groups=groups))
        X_train, X_test = z[train_idx], z[test_idx]
        age_train, age_test = age[train_idx], age[test_idx]
        sex_train, sex_test = sex_disc[train_idx], sex_disc[test_idx]
    else:
        X_train, X_test, age_train, age_test, sex_train, sex_test = train_test_split(
            z, age, sex_disc,
            test_size=settings.test_size,
            random_state=settings.random_state,
            shuffle=True,
            stratify=sex_disc if np.unique(sex_disc).size > 1 else None,
        )

    # Standardize once for the tree models (not strictly necessary but harmless)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Fit models
    if settings.model != "gbt":
        raise ValueError("Only model='gbt' is implemented in this ready-to-go file.")

    age_model = GradientBoostingRegressor(random_state=settings.random_state)
    sex_model = GradientBoostingClassifier(random_state=settings.random_state)

    age_model.fit(X_train_s, age_train)
    sex_model.fit(X_train_s, sex_train)

    # Importances (non-negative for GBT)
    # Shape: [n_factors=2, latent_dim]
    # Row 0 = age importances, Row 1 = sex importances
    R_raw = np.vstack([
        np.abs(age_model.feature_importances_),
        np.abs(sex_model.feature_importances_),
    ]).astype(np.float64)

    # Test-set informativeness
    age_r2 = float(age_model.score(X_test_s, age_test))  # R^2
    sex_acc = float(sex_model.score(X_test_s, sex_test))  # accuracy

    def _r2_to_unit(r2: float) -> float:
        if not settings.clip_regression_r2_to_unit:
            return r2
        # Clip to [0,1] to make averaging with accuracy meaningful and bounded.
        return float(np.clip(r2, 0.0, 1.0))

    informativeness_avg = float((_r2_to_unit(age_r2) + sex_acc) / 2.0)

    # DCI aggregation
    n_factors, latent_dim = R_raw.shape

    def _col_entropy_score(col: np.ndarray) -> float:
        """Compute normalized entropy for a column (code) of importance matrix.
        Lower entropy = more concentrated = better disentanglement."""
        s = float(col.sum())
        if s <= 1e-12:
            return 0.0
        p = col / s
        # Add epsilon BEFORE computing entropy to avoid log(0)
        h = float(scipy_entropy(p + 1e-12))
        if n_factors <= 1:
            return 1.0  # Perfect concentration when only 1 factor
        return float(1.0 - h / np.log(n_factors))

    def _row_entropy_score(row: np.ndarray) -> float:
        """Compute normalized entropy for a row (factor) of importance matrix.
        Lower entropy = more concentrated = better completeness."""
        s = float(row.sum())
        if s <= 1e-12:
            return 0.0
        p = row / s
        # Add epsilon BEFORE computing entropy to avoid log(0)
        h = float(scipy_entropy(p + 1e-12))
        if latent_dim <= 1:
            return 1.0  # Perfect concentration when only 1 code
        return float(1.0 - h / np.log(latent_dim))

    # Disentanglement: weighted by relative importance of each code (column)
    col_sums = R_raw.sum(axis=0)
    total_sum = float(col_sums.sum()) + 1e-12
    rho = col_sums / total_sum  # relative importance per code
    disentanglement_scores = np.array([_col_entropy_score(R_raw[:, j]) for j in range(latent_dim)], dtype=np.float64)
    dci_disentanglement = float(np.sum(rho * disentanglement_scores))

    # Completeness: mean over factors
    completeness_scores = np.array([_row_entropy_score(R_raw[i, :]) for i in range(n_factors)], dtype=np.float64)
    dci_completeness = float(np.mean(completeness_scores))

    # Row-normalized importance matrix (useful for inspection)
    R_norm = R_raw / (R_raw.sum(axis=1, keepdims=True) + 1e-12)

    # Top dimensions per factor
    age_top_dims = np.argsort(R_raw[0, :])[::-1][:5].tolist()
    sex_top_dims = np.argsort(R_raw[1, :])[::-1][:5].tolist()
    age_top_importance = R_raw[0, age_top_dims].tolist()
    sex_top_importance = R_raw[1, sex_top_dims].tolist()

    return {
        "DCI_disentanglement": dci_disentanglement,
        "DCI_completeness": dci_completeness,
        "DCI_informativeness": informativeness_avg,
        "informativeness": {
            "age_r2_test": age_r2,
            "sex_acc_test": sex_acc,
            "avg": informativeness_avg,
        },
        "importance_matrix_raw": R_raw.tolist(),
        "importance_matrix_row_normalized": R_norm.tolist(),
        "disentanglement_per_code": disentanglement_scores.tolist(),
        "completeness_per_factor": completeness_scores.tolist(),
        "top_dims_age": age_top_dims,
        "top_importance_age": age_top_importance,
        "top_dims_sex": sex_top_dims,
        "top_importance_sex": sex_top_importance,
        "settings": {
            "test_size": settings.test_size,
            "max_samples": settings.max_samples,
            "random_state": settings.random_state,
            "model": settings.model,
            "clip_regression_r2_to_unit": settings.clip_regression_r2_to_unit,
        },
    }


# All metrics wrapper
def compute_disentanglement_metrics(
    z: np.ndarray,
    z_age: np.ndarray,
    z_sex: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    groups: Optional[np.ndarray] = None,
    mig_bins_z: int = 10,
    sap_settings: SAPSettings = SAPSettings(),
    dci_settings: DCISettings = DCISettings(),
) -> Dict[str, Any]:
    """
    Compute normalized MIG / SAP / DCI for z, z_age, z_sex.

    MIG:
        - Uses quantile discretization for latents
        - Uses true discrete labels for age (2..89) and sex (0/1)
        - Normalization: (I1 - I2) / I1  -> range [0,1]

    Interpretation intent:
        z_main -> low for age/sex
        z_age  -> high for age, low for sex
        z_sex  -> high for sex, low for age
    """

    results: Dict[str, Any] = {}

    # MIG per representation
    def _compute_mig_for_rep(rep: np.ndarray):
        """
        We pass the representation as both main and target block
        so MIG is computed over this representation only.
        """
        dummy_empty = np.zeros((rep.shape[0], 0))
        mig = compute_mig(
            z_main=rep,
            z_age=dummy_empty,
            z_sex=dummy_empty,
            age=age,
            sex=sex,
            n_bins_z=mig_bins_z,
        )
        return mig

    # Packing helper
    def _pack(rep: np.ndarray, rep_name: str) -> Dict[str, Any]:

        mig = _compute_mig_for_rep(rep)

        sap = compute_sap(rep, age, sex, settings=sap_settings, groups=groups)
        dci = compute_dci(rep, age, sex, settings=dci_settings, groups=groups)

        return {
            "MIG": mig,
            "SAP": sap,
            "DCI": dci,
        }
    # Compute per representation
    results["z_main"] = _pack(z, "z_main")
    results["z_age"] = _pack(z_age, "z_age")
    results["z_sex"] = _pack(z_sex, "z_sex")

    return results
