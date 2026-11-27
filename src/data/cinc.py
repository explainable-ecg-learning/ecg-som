"""CinC 12-Lead ECG dataset importers & downloaders (compatible with ecg-data-repo).

Supports PhysioNet CinC 2020 / arrhythmia 12-lead ECG datasets (Georgia 12-Lead ECG,
Chapman-Shaoxing) whose WFDB `.hea` headers contain `#Age:`, `#Sex:`, and `#Dx:`
SNOMED-CT codes, plus `ecg-data-repo` per-record `error_labels.csv` quality filtering.
Converts records to 100 Hz, 1.2 s masked beats stored as an `ECG_Dataset` pickle
for zero-shot external evaluation and Explorer visualization.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import wfdb

from src.data.dataset import ECG_Dataset
from src.inference import convert_signal_to_record

GEORGIA_BASE_URL = "https://physionet.org/files/challenge-2020/1.0.2/training/georgia"
GEORGIA_ERROR_LABELS_URL = (
    "https://raw.githubusercontent.com/explainable-ecg-learning/"
    "ecg-data-repo/main/georgia_ecg/quality/error_labels.csv"
)
CHAPMAN_BASE_URL = "https://physionet.org/files/ecg-arrhythmia/1.0.0"
CHAPMAN_ERROR_LABELS_URL = (
    "https://raw.githubusercontent.com/explainable-ecg-learning/"
    "ecg-data-repo/main/chapman_shaoxing/quality/error_labels.csv"
)

# Common CinC 2020 SNOMED-CT codes -> concise clinical labels
SNOMED_MAP = {
    "426783006": "NSR",      # Normal sinus rhythm
    "164889003": "AFIB",     # Atrial fibrillation
    "164890007": "AFL",      # Atrial flutter
    "426177001": "SB",       # Sinus bradycardia
    "427084000": "STach",    # Sinus tachycardia
    "427393009": "SA",       # Sinus arrhythmia
    "164909002": "LBBB",     # Left bundle branch block
    "59118001":  "RBBB",     # Right bundle branch block
    "713427006": "CRBBB",    # Complete RBBB
    "713426002": "IRBBB",    # Incomplete RBBB
    "270492004": "IAVB",     # 1st-degree AV block
    "164934002": "TInv",     # T-wave inversion
    "429622005": "STDep",    # ST depression
    "164931005": "STElev",   # ST elevation
    "164873001": "LVH",      # Left ventricular hypertrophy
    "39732003":  "LAD",      # Left axis deviation
    "47665007":  "RAD",      # Right axis deviation
    "284470004": "PAC",      # Premature atrial contraction
    "427172004": "PVC",      # Premature ventricular contraction
    "164865005": "MI",       # Myocardial infarction
    "164861001": "MI",       # Myocardial ischemia
}


def load_snomed_csv(csv_path: str | Path) -> dict[str, str]:
    """Load SNOMED-CT code -> acronym mapping from a ConditionNames_SNOMED-CT CSV."""
    path = Path(csv_path)
    if not path.is_file():
        return {}
    mapping: dict[str, str] = {}
    with open(path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return {}
        fields = [c.strip() for c in reader.fieldnames]
        code_col = next((c for c in fields if "snomed" in c.lower()), fields[0])
        name_col = next(
            (c for c in fields if any(k in c.lower() for k in ("acronym", "abbrev", "name"))),
            fields[min(1, len(fields) - 1)],
        )
        for row in reader:
            cleaned = {k.strip(): (v.strip() if v is not None else "") for k, v in row.items() if k is not None}
            code = cleaned.get(code_col, "")
            name = cleaned.get(name_col, "")
            if code and name and code not in mapping:
                mapping[code] = name
    return mapping


def _load_error_severity_map(csv_path: str | Path) -> dict[str, int]:
    """Load record_id -> integer severity tier (0..3) from an ecg-data-repo error_labels CSV."""
    sev_map: dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        for row in csv.DictReader(f):
            rec_id = (row.get("record_id") or "").strip()
            tier = (row.get("severity_tier") or "").strip()
            if not rec_id or not tier:
                continue
            try:
                sev_map[rec_id] = int(tier[0])
            except ValueError:
                continue
    return sev_map


def _expand_records(root: Path, records_file: Path, rel_prefix: str = "") -> list[str]:
    """Recursively expand CinC RECORDS files when entries point to subdirectories."""
    out: list[str] = []
    with open(records_file, "r", encoding="utf-8", errors="ignore") as f:
        entries = [ln.strip() for ln in f if ln.strip()]
    for entry in entries:
        clean = entry.strip("/")
        rel = f"{rel_prefix}/{clean}" if rel_prefix else clean
        nested = root / rel / "RECORDS"
        if entry.endswith("/") or nested.is_file():
            if nested.is_file():
                out.extend(_expand_records(root, nested, rel_prefix=rel))
        else:
            out.append(rel)
    return out


def parse_cinc_header(hea_path: str) -> tuple[list[str], float | None, int | None]:
    """Parse #Dx, #Age, and #Sex from a PhysioNet CinC .hea header."""
    dx_codes: list[str] = []
    age: float | None = None
    sex: int | None = None
    with open(hea_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("#"):
                continue
            body = line.lstrip("#").strip()
            low = body.lower()
            if low.startswith("age:"):
                val = body.split(":", 1)[1].strip()
                try:
                    age = float(val)
                except ValueError:
                    age = None
            elif low.startswith("sex:"):
                val = body.split(":", 1)[1].strip().lower()
                if val.startswith("m"):
                    sex = 0
                elif val.startswith("f"):
                    sex = 1
            elif low.startswith("dx:"):
                raw = body.split(":", 1)[1].strip()
                dx_codes = [c.strip() for c in re.split(r"[,\s]+", raw) if c.strip()]
    return dx_codes, age, sex


def download_georgia_records(
    dest_dir: str,
    max_records: int = 500,
    workers: int = 8,
    max_error_severity: int | None = 2,
) -> str:
    """Download an evenly stratified quality-filtered subset of Georgia 12-Lead ECG records."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    err_csv = dest / "error_labels.csv"
    if not err_csv.exists() or err_csv.stat().st_size == 0:
        try:
            urllib.request.urlretrieve(GEORGIA_ERROR_LABELS_URL, err_csv)
        except Exception as exc:
            print(f"[georgia] Warning: could not download error_labels.csv ({exc}); falling back to g1/RECORDS")

    selected: list[str] = []
    if err_csv.is_file() and err_csv.stat().st_size > 0:
        candidates: list[str] = []
        with open(err_csv, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            for row in csv.DictReader(f):
                rec_path = (row.get("record_path") or "").strip()
                tier = (row.get("severity_tier") or "").strip()
                if not rec_path or not tier:
                    continue
                if (
                    (row.get("err_dx_empty") or "0").strip() != "0"
                    or (row.get("err_age_missing") or "0").strip() != "0"
                    or (row.get("err_sex_missing_or_unknown") or "0").strip() != "0"
                ):
                    continue
                try:
                    sev = int(tier[0])
                except ValueError:
                    continue
                if max_error_severity is not None and sev > max_error_severity:
                    continue
                candidates.append(rec_path)
        if candidates:
            n = min(max_records, len(candidates))
            idx = np.linspace(0, len(candidates) - 1, n, dtype=int)
            selected = [candidates[i] for i in idx]

    if not selected:
        with urllib.request.urlopen(f"{GEORGIA_BASE_URL}/g1/RECORDS", timeout=30) as resp:
            all_recs = [ln.strip() for ln in resp.read().decode().splitlines() if ln.strip()]
        selected = [f"g1/{r}" for r in all_recs[:max_records]]

    def _fetch_one(rel: str) -> str | None:
        for ext in ("hea", "mat"):
            out = dest / f"{rel}.{ext}"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists() or out.stat().st_size == 0:
                url = f"{GEORGIA_BASE_URL}/{rel}.{ext}"
                try:
                    urllib.request.urlretrieve(url, out)
                except Exception:
                    if out.exists():
                        out.unlink()
                    return None
        return rel

    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = [r for r in ex.map(_fetch_one, selected) if r is not None]

    with open(dest / "RECORDS", "w") as f:
        f.write("\n".join(ok) + "\n")

    print(f"[georgia] Downloaded {len(ok)}/{len(selected)} records to {dest}")
    return str(dest)


def download_chapman_records(
    dest_dir: str,
    max_records: int = 500,
    workers: int = 8,
    max_error_severity: int | None = 2,
) -> str:
    """Download an evenly stratified subset of Chapman-Shaoxing records from PhysioNet."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    cond_csv = dest / "ConditionNames_SNOMED-CT.csv"
    if not cond_csv.exists() or cond_csv.stat().st_size == 0:
        try:
            urllib.request.urlretrieve(f"{CHAPMAN_BASE_URL}/ConditionNames_SNOMED-CT.csv", cond_csv)
        except Exception as exc:
            print(f"[chapman_shaoxing] Warning: could not download ConditionNames_SNOMED-CT.csv ({exc})")

    err_csv = dest / "error_labels.csv"
    if not err_csv.exists() or err_csv.stat().st_size == 0:
        try:
            urllib.request.urlretrieve(CHAPMAN_ERROR_LABELS_URL, err_csv)
        except Exception as exc:
            print(f"[chapman_shaoxing] Warning: could not download error_labels.csv ({exc}); falling back to RECORDS")

    selected: list[str] = []
    if err_csv.is_file() and err_csv.stat().st_size > 0:
        candidates: list[str] = []
        with open(err_csv, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
            for row in csv.DictReader(f):
                rec_path = (row.get("record_path") or "").strip()
                tier = (row.get("severity_tier") or "").strip()
                if not rec_path or not tier:
                    continue
                if (
                    (row.get("err_dx_empty") or "0").strip() != "0"
                    or (row.get("err_age_missing") or "0").strip() != "0"
                    or (row.get("err_sex_missing_or_unknown") or "0").strip() != "0"
                ):
                    continue
                try:
                    sev = int(tier[0])
                except ValueError:
                    continue
                if max_error_severity is not None and sev > max_error_severity:
                    continue
                candidates.append(rec_path)
        if candidates:
            n = min(max_records, len(candidates))
            idx = np.linspace(0, len(candidates) - 1, n, dtype=int)
            selected = [candidates[i] for i in idx]

    if not selected:
        with urllib.request.urlopen(f"{CHAPMAN_BASE_URL}/RECORDS", timeout=30) as resp:
            subdirs = [ln.strip().strip("/") for ln in resp.read().decode().splitlines() if ln.strip()]
        n_dirs = min(len(subdirs), max(1, (max_records + 9) // 10))
        chosen_dirs = [subdirs[i] for i in np.linspace(0, len(subdirs) - 1, n_dirs, dtype=int)]
        per_dir = max(1, (max_records + n_dirs - 1) // n_dirs)
        for sub in chosen_dirs:
            if len(selected) >= max_records:
                break
            try:
                with urllib.request.urlopen(f"{CHAPMAN_BASE_URL}/{sub}/RECORDS", timeout=30) as resp:
                    recs = [ln.strip() for ln in resp.read().decode().splitlines() if ln.strip()]
                for r in recs[:per_dir]:
                    selected.append(f"{sub}/{r}")
                    if len(selected) >= max_records:
                        break
            except Exception:
                continue

    def _fetch_one(rel: str) -> str | None:
        for ext in ("hea", "mat"):
            out = dest / f"{rel}.{ext}"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists() or out.stat().st_size == 0:
                url = f"{CHAPMAN_BASE_URL}/{rel}.{ext}"
                try:
                    urllib.request.urlretrieve(url, out)
                except Exception:
                    if out.exists():
                        out.unlink()
                    return None
        return rel

    with ThreadPoolExecutor(max_workers=workers) as ex:
        ok = [r for r in ex.map(_fetch_one, selected) if r is not None]

    with open(dest / "RECORDS", "w") as f:
        f.write("\n".join(ok) + "\n")

    print(f"[chapman_shaoxing] Downloaded {len(ok)}/{len(selected)} records to {dest}")
    return str(dest)


def import_cinc_dataset(
    base_path: str,
    output_pkl: str | None = None,
    max_records: int | None = None,
    min_class_count: int = 5,
    error_labels_path: str | None = None,
    max_error_severity: int | None = None,
    stratified_folds: bool = False,
    fold_seed: int = 42,
) -> ECG_Dataset:
    """Import a CinC-style 12-lead ECG directory (Georgia, Chapman-Shaoxing) into an ECG_Dataset."""
    root = Path(base_path)
    records_file = root / "RECORDS"
    rel_paths = _expand_records(root, records_file)
    if max_records is not None:
        rel_paths = rel_paths[:max_records]

    csv_map = load_snomed_csv(root / "ConditionNames_SNOMED-CT.csv")
    snomed_lookup = {**csv_map, **SNOMED_MAP}

    sev_map: dict[str, int] = {}
    if max_error_severity is not None:
        err_csv = Path(error_labels_path) if error_labels_path else (root / "error_labels.csv")
        if err_csv.is_file():
            sev_map = _load_error_severity_map(err_csv)

    ds = ECG_Dataset(fs=100)
    skipped = 0
    for rel in rel_paths:
        rec_id = os.path.basename(rel)
        if sev_map and rec_id in sev_map and sev_map[rec_id] > max_error_severity:
            skipped += 1
            continue
        stem = str(root / rel)
        hea = stem + ".hea"
        if not os.path.exists(hea):
            skipped += 1
            continue
        dx_codes, age, sex = parse_cinc_header(hea)
        if age is None or sex is None or np.isnan(age) or age < 2 or age > 100:
            skipped += 1
            continue

        mapped = [snomed_lookup[c] for c in dx_codes if c in snomed_lookup]
        primary = mapped[0] if mapped else (dx_codes[0] if dx_codes else "OTHER")
        extra = mapped[1:] if len(mapped) > 1 else []

        try:
            sig, meta = wfdb.rdsamp(stem)
            fs = float(meta.get("fs", 500.0))
            lead_names = meta.get("sig_name")
            rec = convert_signal_to_record(
                sig,
                fs=fs,
                record_id=rec_id,
                lead_names=lead_names,
                age=float(age),
                sex=int(sex),
                label=primary,
                fold=10,  # external test split unless stratified_folds=True
            )
            rec.extra_labels = extra
            ds.add_record(rec)
        except Exception:
            skipped += 1

    if not ds.records:
        raise RuntimeError(f"No valid records imported from {base_path}")

    # Prune very rare labels for clean clustering metrics
    labels_all = np.array([r.label for r in ds.records])
    uniq, cnts = np.unique(labels_all, return_counts=True)
    keep = set(uniq[cnts >= min_class_count])
    if keep:
        ds.records = [r for r in ds.records if r.label in keep]

    if stratified_folds:
        from collections import defaultdict
        from sklearn.model_selection import StratifiedKFold

        y_str = np.array([r.label for r in ds.records])
        uniq_l, cnts_l = np.unique(y_str, return_counts=True)
        n_splits = 10
        idx_ge10 = [i for i, lbl in enumerate(y_str) if cnts_l[np.searchsorted(uniq_l, lbl)] >= n_splits]
        idx_lt10 = [i for i in range(len(ds.records)) if i not in set(idx_ge10)]

        if len(idx_ge10) >= n_splits:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=fold_seed)
            for fold_idx, (_, test_pos) in enumerate(skf.split(np.zeros(len(idx_ge10)), y_str[idx_ge10]), start=1):
                for pos in test_pos:
                    ds.records[idx_ge10[pos]].fold = fold_idx

        by_lbl_small: dict[str, list[int]] = defaultdict(list)
        for i in idx_lt10:
            by_lbl_small[y_str[i]].append(i)
        rng = np.random.default_rng(fold_seed)
        for _, idxs in by_lbl_small.items():
            rng.shuffle(idxs)
            for j, rec_idx in enumerate(idxs):
                ds.records[rec_idx].fold = (j % n_splits) + 1

    ds.label_encoder.fit([r.label for r in ds.records])
    ds.class_names = list(ds.label_encoder.classes_)
    split_msg = ""
    if stratified_folds:
        n_tr = sum(1 for r in ds.records if r.fold <= 8)
        n_va = sum(1 for r in ds.records if r.fold == 9)
        n_te = sum(1 for r in ds.records if r.fold == 10)
        split_msg = f" [splits train={n_tr}, val={n_va}, test={n_te}]"
    print(
        f"[cinc] Imported {len(ds.records)} valid records ({skipped} skipped), "
        f"{len(ds.class_names)} classes{split_msg}: {ds.class_names}"
    )

    if output_pkl:
        ds.save(output_pkl)
        print(f"[cinc] Saved dataset -> {output_pkl}")
    return ds


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Download & import CinC 12-lead ECG datasets (ecg-data-repo compatible)")
    p.add_argument("--dataset", choices=["georgia", "chapman_shaoxing"], default="georgia")
    p.add_argument("--download-dir", default=None)
    p.add_argument("--max-records", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--min-class-count", type=int, default=5)
    p.add_argument("--max-error-severity", type=int, default=2, help="Max severity tier (0..3); -1 disables filtering")
    p.add_argument("--stratified-folds", action="store_true", help="Assign stratified 10-fold splits (folds 1-8 train, 9 val, 10 test)")
    p.add_argument("--fold-seed", type=int, default=42)
    p.add_argument("--output", default=None)
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    dl_dir = args.download_dir or f"data/raw/{args.dataset}"
    out_pkl = args.output or f"data/processed/{args.dataset}_100.pkl"
    max_sev = None if args.max_error_severity < 0 else args.max_error_severity

    if not args.skip_download:
        if args.dataset == "georgia":
            download_georgia_records(
                dl_dir, max_records=args.max_records, workers=args.workers, max_error_severity=max_sev
            )
        elif args.dataset == "chapman_shaoxing":
            download_chapman_records(
                dl_dir, max_records=args.max_records, workers=args.workers, max_error_severity=max_sev
            )

    import_cinc_dataset(
        dl_dir,
        output_pkl=out_pkl,
        max_records=args.max_records,
        min_class_count=args.min_class_count,
        max_error_severity=max_sev,
        stratified_folds=args.stratified_folds,
        fold_seed=args.fold_seed,
    )
