"""
Disentanglement metrics (MIG, SAP, DCI) with reproducibility.

Implements:
- cluster_purity: label-matching purity for clustering evaluation.
- compute_mig: normalized Mutual Information Gap per factor.
- compute_sap: Separated Attribute Predictability.
- compute_dci: Disentanglement / Completeness / Informativeness.
- compute_disentanglement_metrics: wrapper combining MIG, SAP, DCI.
- compute_per_lead_pearson: per-lead reconstruction correlation.
- compute_som_clustering_metrics: topology / utilization / label metrics.
- som_json_safe: JSON-serializable conversion of SOM metric dicts.
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


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Clustering purity
# ---------------------------------------------------------------------------

def cluster_purity(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute clustering purity in [0, 1]."""
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


# ---------------------------------------------------------------------------
# MIG
# ---------------------------------------------------------------------------

def _quantile_discretize(z: np.ndarray, n_bins: int = 10) -> np.ndarray:
    z = np.asarray(z)
    z_disc = np.zeros_like(z, dtype=np.int32)
    for j in range(z.shape[1]):
        edges = np.quantile(z[:, j], np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) <= 2:
            continue
        z_disc[:, j] = np.digitize(z[:, j], edges[1:-1])
    return z_disc


def _mig_for_factor(z_disc: np.ndarray, factor_labels: np.ndarray):
    n_latents = z_disc.shape[1]
    mi = np.array([mutual_info_score(z_disc[:, j], factor_labels) for j in range(n_latents)])
    mi_sorted = np.sort(mi)[::-1]
    if mi_sorted[0] < 1e-12:
        return 0.0, mi_sorted
    mig_norm = (mi_sorted[0] - mi_sorted[1]) / (mi_sorted[0] + 1e-12)
    return float(mig_norm), mi_sorted


def compute_mig(
    z_main: np.ndarray,
    z_age: np.ndarray,
    z_sex: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    n_bins_z: int = 10,
) -> Dict[str, Any]:
    """Normalized MIG for mixed latent groups."""
    age_labels = age.astype(np.int32) - age.astype(np.int32).min()
    sex_labels = sex.astype(np.int32)

    z_age_all = np.concatenate([z_main, z_age], axis=1) if z_age.shape[1] > 0 else z_main
    z_sex_all = np.concatenate([z_main, z_sex], axis=1) if z_sex.shape[1] > 0 else z_main

    z_age_disc  = _quantile_discretize(z_age_all, n_bins_z)
    z_sex_disc  = _quantile_discretize(z_sex_all, n_bins_z)
    z_main_disc = _quantile_discretize(z_main, n_bins_z)

    mig_age, mi_age_sorted   = _mig_for_factor(z_age_disc, age_labels)
    mig_sex, mi_sex_sorted   = _mig_for_factor(z_sex_disc, sex_labels)
    mig_main_age, _          = _mig_for_factor(z_main_disc, age_labels)
    mig_main_sex, _          = _mig_for_factor(z_main_disc, sex_labels)

    return {
        "age":  {"mig_norm": mig_age,  "top1_mi": float(mi_age_sorted[0]),  "top2_mi": float(mi_age_sorted[1])},
        "sex":  {"mig_norm": mig_sex,  "top1_mi": float(mi_sex_sorted[0]),  "top2_mi": float(mi_sex_sorted[1])},
        "main_leakage": {"age_mig": mig_main_age, "sex_mig": mig_main_sex},
    }


# ---------------------------------------------------------------------------
# SAP
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SAPSettings:
    test_size: float = 0.33
    max_train_samples: int = 15000
    random_state: int = 0
    svc_c: float = 0.01
    svc_max_iter: int = 5000


def _encode_as_int_labels(x: np.ndarray) -> np.ndarray:
    _, inv = np.unique(np.asarray(x), return_inverse=True)
    return inv.astype(np.int32, copy=False)


def compute_sap(
    z: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    settings: SAPSettings = SAPSettings(),
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Separated Attribute Predictability (SAP) with train/test split."""
    z   = _check_2d(z, "z")
    age = _check_1d(age, "age").astype(np.float64, copy=False)
    sex = _check_1d(sex, "sex")

    n = z.shape[0]
    rng = np.random.default_rng(settings.random_state)
    if n > settings.max_train_samples:
        idx = rng.choice(n, size=settings.max_train_samples, replace=False)
        z, age, sex = z[idx], age[idx], sex[idx]
        if groups is not None:
            groups = groups[idx]

    sex_disc = _encode_as_int_labels(sex.astype(np.int64, copy=False))

    if groups is not None and np.unique(groups).size > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=settings.test_size, random_state=settings.random_state)
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

    def _scores(y_tr, y_te, kind):
        scores = np.zeros(X_train.shape[1], dtype=np.float64)
        for j in range(X_train.shape[1]):
            xtr, xte = X_train[:, j:j+1], X_test[:, j:j+1]
            if kind == "regression":
                m = make_pipeline(StandardScaler(), LinearRegression())
                m.fit(xtr, y_tr)
                scores[j] = float(m.score(xte, y_te))
            else:
                if np.unique(y_tr).size < 2:
                    scores[j] = 0.0
                    continue
                m = make_pipeline(StandardScaler(), LinearSVC(C=settings.svc_c, max_iter=settings.svc_max_iter, random_state=settings.random_state))
                m.fit(xtr, y_tr)
                scores[j] = float(m.score(xte, y_te))
        return scores

    age_scores = _scores(age_train, age_test, "regression")
    sex_scores = _scores(sex_train, sex_test, "classification")

    def _gap(scores, name):
        s = np.sort(scores)[::-1]
        idx = np.argsort(scores)[::-1]
        return float(s[0] - s[1]) if scores.size >= 2 else 0.0, idx[:5].tolist(), s[:5].tolist()

    sap_age, top_age_dims, top_age_sc = _gap(age_scores, "age")
    sap_sex, top_sex_dims, top_sex_sc = _gap(sex_scores, "sex")

    return {
        "SAP_age": sap_age, "SAP_sex": sap_sex,
        "age_score_per_dim": age_scores.tolist(), "sex_score_per_dim": sex_scores.tolist(),
        "best_dims_age": top_age_dims, "best_scores_age": top_age_sc,
        "best_dims_sex": top_sex_dims, "best_scores_sex": top_sex_sc,
        "settings": {"test_size": settings.test_size, "max_train_samples": settings.max_train_samples,
                     "random_state": settings.random_state, "svc_c": settings.svc_c},
    }


# ---------------------------------------------------------------------------
# DCI
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DCISettings:
    test_size: float = 0.33
    max_samples: int = 15000
    random_state: int = 0
    model: str = "gbt"
    clip_regression_r2_to_unit: bool = True


def compute_dci(
    z: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
    settings: DCISettings = DCISettings(),
    groups: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """DCI: Disentanglement, Completeness, Informativeness."""
    z   = _check_2d(z, "z")
    age = _check_1d(age, "age").astype(np.float64, copy=False)
    sex = _check_1d(sex, "sex")

    n = z.shape[0]
    rng = np.random.default_rng(settings.random_state)
    if n > settings.max_samples:
        idx = rng.choice(n, size=settings.max_samples, replace=False)
        z, age, sex = z[idx], age[idx], sex[idx]
        if groups is not None:
            groups = groups[idx]

    sex_disc = _encode_as_int_labels(sex.astype(np.int64, copy=False))

    if groups is not None and np.unique(groups).size > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=settings.test_size, random_state=settings.random_state)
        train_idx, test_idx = next(splitter.split(z, sex_disc, groups=groups))
        X_train, X_test = z[train_idx], z[test_idx]
        age_train, age_test = age[train_idx], age[test_idx]
        sex_train, sex_test = sex_disc[train_idx], sex_disc[test_idx]
    else:
        X_train, X_test, age_train, age_test, sex_train, sex_test = train_test_split(
            z, age, sex_disc,
            test_size=settings.test_size, random_state=settings.random_state,
            shuffle=True, stratify=sex_disc if np.unique(sex_disc).size > 1 else None,
        )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    age_model = GradientBoostingRegressor(random_state=settings.random_state)
    sex_model = GradientBoostingClassifier(random_state=settings.random_state)
    age_model.fit(X_train_s, age_train)
    sex_model.fit(X_train_s, sex_train)

    R_raw = np.vstack([
        np.abs(age_model.feature_importances_),
        np.abs(sex_model.feature_importances_),
    ]).astype(np.float64)

    age_r2  = float(age_model.score(X_test_s, age_test))
    sex_acc = float(sex_model.score(X_test_s, sex_test))

    def _r2_unit(r2): return float(np.clip(r2, 0.0, 1.0)) if settings.clip_regression_r2_to_unit else r2
    info_avg = float((_r2_unit(age_r2) + sex_acc) / 2.0)

    n_factors, latent_dim = R_raw.shape

    def _col_score(col):
        s = float(col.sum())
        if s <= 1e-12: return 0.0
        h = float(scipy_entropy(col / s + 1e-12))
        return 1.0 if n_factors <= 1 else float(1.0 - h / np.log(n_factors))

    def _row_score(row):
        s = float(row.sum())
        if s <= 1e-12: return 0.0
        h = float(scipy_entropy(row / s + 1e-12))
        return 1.0 if latent_dim <= 1 else float(1.0 - h / np.log(latent_dim))

    col_sums  = R_raw.sum(axis=0)
    rho       = col_sums / (float(col_sums.sum()) + 1e-12)
    dis_scores = np.array([_col_score(R_raw[:, j]) for j in range(latent_dim)])
    com_scores = np.array([_row_score(R_raw[i, :]) for i in range(n_factors)])
    R_norm = R_raw / (R_raw.sum(axis=1, keepdims=True) + 1e-12)

    age_top = np.argsort(R_raw[0])[::-1][:5].tolist()
    sex_top = np.argsort(R_raw[1])[::-1][:5].tolist()

    return {
        "DCI_disentanglement": float(np.sum(rho * dis_scores)),
        "DCI_completeness":    float(np.mean(com_scores)),
        "DCI_informativeness": info_avg,
        "informativeness":     {"age_r2_test": age_r2, "sex_acc_test": sex_acc, "avg": info_avg},
        "importance_matrix_raw":            R_raw.tolist(),
        "importance_matrix_row_normalized": R_norm.tolist(),
        "disentanglement_per_code":         dis_scores.tolist(),
        "completeness_per_factor":          com_scores.tolist(),
        "top_dims_age": age_top, "top_importance_age": R_raw[0, age_top].tolist(),
        "top_dims_sex": sex_top, "top_importance_sex": R_raw[1, sex_top].tolist(),
        "settings": {"test_size": settings.test_size, "max_samples": settings.max_samples,
                     "random_state": settings.random_state, "model": settings.model},
    }


# ---------------------------------------------------------------------------
# Combined wrapper
# ---------------------------------------------------------------------------

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
    """Compute MIG / SAP / DCI for z, z_age, z_sex."""
    def _pack(rep):
        dummy = np.zeros((rep.shape[0], 0))
        return {
            "MIG": compute_mig(z_main=rep, z_age=dummy, z_sex=dummy, age=age, sex=sex, n_bins_z=mig_bins_z),
            "SAP": compute_sap(rep, age, sex, settings=sap_settings, groups=groups),
            "DCI": compute_dci(rep, age, sex, settings=dci_settings, groups=groups),
        }
    return {"z_main": _pack(z), "z_age": _pack(z_age), "z_sex": _pack(z_sex)}


# ---------------------------------------------------------------------------
# Per-lead Pearson
# ---------------------------------------------------------------------------

def compute_per_lead_pearson(x: np.ndarray, x_hat: np.ndarray) -> Dict[str, Any]:
    """Compute reconstruction Pearson correlation per ECG lead."""
    x     = np.asarray(x,     dtype=np.float64)
    x_hat = np.asarray(x_hat, dtype=np.float64)
    if x.shape != x_hat.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {x_hat.shape}.")
    if x.ndim == 2:
        x, x_hat = x[None], x_hat[None]

    B, C, _ = x.shape
    corr_bc = np.full((B, C), np.nan)
    for b in range(B):
        for c in range(C):
            if np.std(x[b, c]) > 0 and np.std(x_hat[b, c]) > 0:
                corr_bc[b, c] = np.corrcoef(x[b, c], x_hat[b, c])[0, 1]

    mean_per_lead   = np.array([np.mean(corr_bc[:, c][np.isfinite(corr_bc[:, c])]) if np.any(np.isfinite(corr_bc[:, c])) else np.nan for c in range(C)])
    median_per_lead = np.array([np.median(corr_bc[:, c][np.isfinite(corr_bc[:, c])]) if np.any(np.isfinite(corr_bc[:, c])) else np.nan for c in range(C)])
    valid = mean_per_lead[np.isfinite(mean_per_lead)]
    return {
        "per_lead": mean_per_lead.tolist(),
        "aggregates": {
            "mean_over_leads": float(np.mean(valid)) if valid.size else float("nan"),
            "median_over_leads": float(np.median(valid)) if valid.size else float("nan"),
            "mean_per_lead": mean_per_lead.tolist(),
            "median_per_lead": median_per_lead.tolist(),
        },
        "beat_count": B,
        "per_beat_per_lead": corr_bc.tolist(),
    }


# ---------------------------------------------------------------------------
# SOM metrics
# ---------------------------------------------------------------------------

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
    z_n = np.sum(z * z, axis=1, keepdims=True)
    e_n = np.sum(e * e, axis=1, keepdims=True).T
    return np.maximum(z_n + e_n - 2.0 * (z @ e.T), 0.0)


def _som_gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).flatten()
    if x.size == 0: return float("nan")
    total = float(x.sum())
    if total <= 0.0: return 0.0
    xs = np.sort(x)
    n = xs.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(i * xs) / (n * total)) - (n + 1.0) / n)


def _som_neighbor_consistency(z: np.ndarray, bmu: np.ndarray, H: int, W: int, k: int = 10) -> float:
    n = z.shape[0]
    if n <= 1: return float("nan")
    k = max(1, min(k, n - 1))
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(z)
        _, idx = nn.kneighbors(z)
        neigh = idx[:, 1:]
    except Exception:
        return float("nan")
    bmu_row, bmu_col = bmu // W, bmu % W
    nb_bmu = bmu[neigh]
    dr = np.abs((nb_bmu // W) - bmu_row[:, None])
    dc = np.abs((nb_bmu % W) - bmu_col[:, None])
    return float(np.mean((dr + dc) <= 1))


def compute_som_clustering_metrics(
    z: np.ndarray,
    codebook: np.ndarray,
    som_shape: Tuple[int, int],
    labels: Optional[np.ndarray] = None,
    knn_k: int = 10,
) -> Dict[str, Any]:
    z        = _check_2d(np.asarray(z),        name="z")
    codebook = _check_2d(np.asarray(codebook), name="codebook")
    H, W = int(som_shape[0]), int(som_shape[1])
    total_units = H * W
    if codebook.shape[0] != total_units:
        raise ValueError(f"Codebook first dim must equal H*W={total_units}, got {codebook.shape[0]}.")
    if z.shape[1] != codebook.shape[1]:
        raise ValueError(f"Latent dim mismatch: z={z.shape[1]}, codebook={codebook.shape[1]}.")

    dist2 = _som_pairwise_dist2(z, codebook)
    top2u = np.argpartition(dist2, kth=1, axis=1)[:, :2]
    ridx  = np.arange(z.shape[0])[:, None]
    order = np.argsort(dist2[ridx, top2u], axis=1)
    top2  = top2u[ridx, order]
    bmu   = top2[:, 0].astype(np.int64)
    second_bmu = top2[:, 1].astype(np.int64)

    q_err = float(np.mean(np.sqrt(dist2[np.arange(z.shape[0]), bmu])))
    bmu_r, bmu_c = bmu // W, bmu % W
    b2_r,  b2_c  = second_bmu // W, second_bmu % W
    topo_err = float(np.mean((np.abs(bmu_r - b2_r) + np.abs(bmu_c - b2_c)) != 1))
    nc = _som_neighbor_consistency(z, bmu, H, W, k=knn_k)

    occupancy = np.bincount(bmu, minlength=total_units).astype(np.int64)
    active    = int(np.sum(occupancy > 0))
    dead      = total_units - active
    p = occupancy.astype(np.float64)
    p_s = float(p.sum())
    if p_s > 0.0:
        p /= p_s
        p_nz = p[p > 0.0]
        occ_ent = float(-np.sum(p_nz * np.log(p_nz)))
        if total_units > 1:
            occ_ent /= float(np.log(total_units))
    else:
        occ_ent = float("nan")

    label_metrics = None
    unit_dominant = [None] * total_units
    unit_purity   = [None] * total_units
    unit_label_rows, neighbor_edge_rows = [], []

    if labels is not None:
        labels = np.asarray(labels).astype(str)
        if labels.shape[0] == bmu.shape[0]:
            wp_num, wp_den = 0.0, 0.0
            for u in range(total_units):
                cnt = int(occupancy[u])
                if cnt > 0:
                    ul = labels[bmu == u]
                    uniq, cnts = np.unique(ul, return_counts=True)
                    j = int(np.argmax(cnts))
                    unit_dominant[u] = str(uniq[j])
                    unit_purity[u]   = float(cnts[j] / cnt)
                    wp_num += unit_purity[u] * cnt
                    wp_den += cnt
                unit_label_rows.append({
                    "unit_id": u, "x": u // W, "y": u % W,
                    "dominant_label": unit_dominant[u], "unit_purity": unit_purity[u],
                    "count": int(occupancy[u]),
                })

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
                        la, lb = unit_dominant[u], unit_dominant[v]
                        same = int(la == lb)
                        agreement.append(same)
                        transition[f"{la}|{lb}"] = int(transition.get(f"{la}|{lb}", 0) + 1)
                        neighbor_edge_rows.append({"unit_id_a": u, "unit_id_b": v, "label_a": la, "label_b": lb, "same_label": same})

            label_metrics = {
                "per_unit_dominant_label": unit_dominant,
                "per_unit_purity":         unit_purity,
                "global_purity_by_units":  float(wp_num / wp_den) if wp_den > 0 else float("nan"),
                "neighbor_label_agreement": float(np.mean(agreement)) if agreement else float("nan"),
                "neighbor_transition_matrix": transition,
            }

    return {
        "topology":  {"quantization_error": q_err, "topographic_error": topo_err, "neighborhood_consistency": nc},
        "utilization": {
            "total_units": total_units, "active_units": active, "dead_units": dead,
            "dead_ratio": float(dead / total_units) if total_units > 0 else float("nan"),
            "occupancy_counts": occupancy.tolist(), "occupancy_entropy": occ_ent,
            "occupancy_gini": _som_gini(occupancy.astype(np.float64)),
            "occupancy_p50": float(np.percentile(occupancy, 50)),
            "occupancy_p90": float(np.percentile(occupancy, 90)),
            "occupancy_max": int(np.max(occupancy)) if occupancy.size else 0,
        },
        "labels": label_metrics,
        "unit_label_rows": unit_label_rows,
        "neighbor_edge_rows": neighbor_edge_rows,
    }
