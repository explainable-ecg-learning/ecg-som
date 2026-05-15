import os
import ast
import pickle
import numpy as np
import pandas as pd
import wfdb
from sklearn.preprocessing import LabelEncoder

from src.data.record import ECG_Record


class ECG_Dataset:
    """
    Collection of ECG_Records with file I/O and PTB-XL import support.
    """

    def __init__(self, fs=100):
        self.records = []
        self.label_encoder = LabelEncoder()
        self.class_names = []
        self.fs = fs

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

    def __getitem__(self, idx):
        return self.records[idx]

    def add_record(self, record):
        self.records.append(record)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    def get_record(self, idx):
        return self.records[idx]

    def get_record_by_id(self, idx):
        for r in self.records:
            if r.id == idx:
                return r
        return None

    def get_data(self, split="all"):
        """
        Return records for a split.  Folds 1-8 = train, 9 = val, 10 = test.
        """
        if not self.records:
            raise ValueError("Dataset is empty.")

        if split == "train":
            filtered = [r for r in self.records if r.fold <= 8]
        elif split == "val":
            filtered = [r for r in self.records if r.fold == 9]
        elif split == "test":
            filtered = [r for r in self.records if r.fold == 10]
        else:
            filtered = self.records

        if split != "test":
            filtered = [r for r in filtered if r.is_valid]
        else:
            filtered = [r for r in filtered if r.reason != "Age > 100"]

        if not filtered:
            raise ValueError(f"No records found for split '{split}'")
        return filtered

    def import_record(self, row, from_path, is_valid=True, reason=None):
        sig, _ = wfdb.rdsamp(from_path)
        if np.isnan(sig).any():
            sig = np.nan_to_num(sig)

        extra = row.get("extra_labels", [])
        if isinstance(extra, float) and np.isnan(extra):
            extra = []

        record = ECG_Record(
            id=row["ecg_id"],
            signal=sig,
            age=row["age"],
            sex=row["sex"],
            label=row["diagnostic_class"],
            extra_labels=extra,
            fold=row["strat_fold"],
            fs=self.fs,
            is_valid=is_valid,
            reason=reason,
        )

        if record.age > 100:
            record.is_valid = False
            record.reason = "Age > 100"

        record.detect_peaks(0)
        if record.segment_beats(target_length=int(1.2 * self.fs)):
            record.normalize()
        else:
            record.is_valid = False
            record.reason = "Segmentation failed"

        return record

    def import_ptbxl(self, base_path, verbose=True):
        """Import and preprocess the PTB-XL dataset from ``base_path``."""
        print(f"Starting PTB-XL Import from {base_path}...")

        df = pd.read_csv(os.path.join(base_path, "ptbxl_database.csv"))
        df_scps = pd.read_csv(os.path.join(base_path, "scp_statements.csv"), index_col=0)
        df_scps = df_scps[df_scps["diagnostic"] == 1]

        df = df.dropna(subset=["age", "sex"])
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
        df[["diagnostic_class", "extra_labels"]] = df["scp_codes"].apply(
            lambda x: pd.Series(self._pick_subclass(x, df_scps))
        )

        filename_col = "filename_lr" if self.fs == 100 else "filename_hr"
        noise_cols = ["static_noise", "burst_noise", "electrodes_problems"]
        existing_cols = [c for c in noise_cols if c in df.columns]
        count = 0

        for _, row in df.iterrows():
            f_path = os.path.join(base_path, row[filename_col])
            is_valid, reason = True, None
            if existing_cols:
                noise_info = row[existing_cols]
                if noise_info.notnull().any():
                    is_valid = False
                    reason = "; ".join(
                        f"{c}: {row[c]}" for c in existing_cols if pd.notnull(row[c])
                    )
            record = self.import_record(row, from_path=f_path, is_valid=is_valid, reason=reason)
            if record is not None:
                self.records.append(record)
                count += 1
            if verbose and count % 1000 == 0:
                print(f"Processed {count} records...")

        # Prune rare classes (< 0.29 % of total)
        total = len(self.records)
        if total == 0:
            return
        labels_all = np.array([r.label for r in self.records])
        unique, counts = np.unique(labels_all, return_counts=True)
        keep = set(unique[counts / total >= 0.0029])
        removed = set(unique) - keep
        if removed:
            self.records = [r for r in self.records if r.label in keep]

        all_labels = [r.label for r in self.records]
        self.label_encoder.fit(all_labels)
        self.class_names = list(self.label_encoder.classes_)
        print(f"Import complete. Valid records: {len(self.records)}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_subclass(self, subclass):
        if subclass in {"ISCA", "ISCI", "ISC_"}:
            return "ISC"
        if subclass == "LMI":
            return "IMI"
        return subclass

    def _pick_subclass(self, scp_dict, scp_df):
        candidates = []
        total_weight = 0.0
        for code, weight in scp_dict.items():
            if code not in scp_df.index:
                continue
            sub = self._normalize_subclass(scp_df.loc[code, "diagnostic_subclass"])
            candidates.append((sub, weight))
            total_weight += float(weight)

        if not candidates:
            return "NORM", []

        best_by_sub = {}
        for sub, w in candidates:
            best_by_sub[sub] = max(best_by_sub.get(sub, w), w)

        best = max(best_by_sub.items(), key=lambda x: (x[1], x[0]))[0]
        extra = sorted(s for s in best_by_sub if s != best) if total_weight > 100.0 else []
        return best, extra
