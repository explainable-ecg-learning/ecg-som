import numpy as np
import neurokit2 as nk

from src.data.signal import calc_sqi_metrics


class ECG_Record:
    """
    Single 12-lead ECG record with metadata, R-peak detection, beat
    segmentation, and per-lead normalization.
    """

    def __init__(self, id, signal, age, sex, label=None, extra_labels=None,
                 fold=None, fs=100, is_valid=True, reason=None):
        self.id = id
        self.signal = signal.astype(np.float32)  # [T, C]
        self.beats = None               # [B, T, C]
        self.beat_representations = None  # [B, T, C]
        self.beat_masks = None          # [B, T]
        self.beat_sigma = None          # [B, C]
        self.beat_amax = None           # [B, C]

        self.age = float(age)
        self.sex = sex                  # 0 = Male, 1 = Female
        self.label = label
        self.extra_labels = extra_labels or []
        self.fold = fold
        self.fs = fs

        self.r_peaks = []
        self.r_peak_candidates = []

        self.is_valid = is_valid
        self.reason = reason

    def is_acceptable(self, min_kurtosis=5.0, min_psqi=0.5):
        """Return True if average lead quality meets thresholds."""
        m = calc_sqi_metrics(self.signal, self.fs)
        return (np.mean(m["kSQI"]) >= min_kurtosis) and (np.mean(m["pSQI"]) >= min_psqi)

    def detect_peaks(self, lead_idx=0):
        """
        Detect R-peaks in lead ``lead_idx``, then search all other leads for
        additional peak candidates not already close to primary peaks.
        """
        candidate_list = []

        try:
            cleaned = nk.ecg_clean(self.signal[:, lead_idx], sampling_rate=self.fs)
            _, info = nk.ecg_peaks(cleaned, sampling_rate=self.fs)
            peaks = sorted([int(p) for p in info["ECG_R_Peaks"] if not np.isnan(p)])
        except Exception:
            peaks = []

        self.r_peaks = np.array(peaks)
        if len(peaks) <= 2:
            return False

        threshold = self.fs / 5
        for lead in range(self.signal.shape[1]):
            if lead == lead_idx:
                continue
            try:
                cleaned = nk.ecg_clean(self.signal[:, lead], sampling_rate=self.fs)
                _, info = nk.ecg_peaks(cleaned, sampling_rate=self.fs)
                other_peaks = [int(p) for p in info["ECG_R_Peaks"] if not np.isnan(p)]
                for cp in other_peaks:
                    if not any(abs(cp - rp) < threshold for rp in self.r_peaks):
                        candidate_list.append((cp, lead, abs(self.signal[cp, lead])))
            except Exception:
                continue

        # Filter: keep candidates seen in >= 6 leads within threshold
        candidate_list.sort(key=lambda x: x[0])
        filtered = []
        i = 0
        while i < len(candidate_list):
            group = [candidate_list[i]]
            j = i + 1
            while j < len(candidate_list) and abs(candidate_list[j][0] - candidate_list[i][0]) < threshold:
                group.append(candidate_list[j])
                j += 1
            if len({g[1] for g in group}) >= 6:
                filtered.append(max(group, key=lambda x: x[2])[0])
            i = j

        self.r_peak_candidates = np.array(filtered)

        # Accept filtered candidates that are >= 40% amplitude of both neighbors
        cand = np.unique(np.array(filtered, dtype=int)) if filtered else np.array([], dtype=int)
        if cand.size > 0 and self.r_peaks.size >= 3:
            base = np.sort(self.r_peaks.astype(int))
            accepted = []
            for p in np.sort(cand):
                a = float(self.signal[p, lead_idx])
                if a < -0.1:
                    continue
                pos = int(np.searchsorted(base, p))
                if pos >= len(base):
                    continue
                a_prev = float(abs(self.signal[int(base[pos - 1]), lead_idx]))
                a_next = float(abs(self.signal[int(base[pos]), lead_idx]))
                if abs(a) >= 0.4 * a_prev and abs(a) >= 0.4 * a_next:
                    accepted.append(p)
            if accepted:
                self.r_peaks = np.unique(np.concatenate([base, np.array(accepted, dtype=int)]))
                self.r_peaks.sort()

        # Remove "weak" peaks (< 25% of both neighbors and < 0.2 mV)
        peaks_arr = self.r_peaks
        amps = np.abs(self.signal[peaks_arr, lead_idx]).astype(np.float32)
        keep_mask = np.ones(len(peaks_arr), dtype=bool)
        for i in range(1, len(peaks_arr) - 1):
            a, a_prev, a_next = float(amps[i]), float(amps[i - 1]), float(amps[i + 1])
            if (a < 0.25 * a_prev) and (a_prev > 0.1) and (a < 0.2) and (a_next > 0.1) and (a < 0.25 * a_next):
                keep_mask[i] = False
        self.r_peaks = peaks_arr[keep_mask]

        return True

    def segment_beats(self, target_length=120):
        """Extract beats centered on R-peaks with fixed length ``target_length``."""
        if len(self.r_peaks) <= 2:
            return False

        peaks = self.r_peaks.tolist()
        segments, masks = [], []
        half = target_length // 2
        sig_len = self.signal.shape[0]
        num_peaks = len(peaks)

        for i in range(1, num_peaks - 1):
            p = peaks[i]
            ws, we = p - half, p + (target_length - half)

            if ws >= 0 and we <= sig_len:
                seg = self.signal[ws:we].copy()
            else:
                pad_l = max(0, -ws)
                pad_r = max(0, we - sig_len)
                seg = np.pad(
                    self.signal[max(0, ws):min(sig_len, we)],
                    ((pad_l, pad_r), (0, 0)),
                    mode="constant", constant_values=0,
                )

            nat_start = (p + peaks[i - 1]) // 2
            nat_end   = sig_len if i == num_peaks - 1 else (p + peaks[i + 1]) // 2

            rel_s = nat_start - ws
            rel_e = nat_end - ws

            if rel_s > 0:
                seg[:max(0, int(rel_s)), :] = 0
            if rel_e < target_length:
                seg[min(target_length, int(rel_e)):, :] = 0

            m = np.zeros(target_length, dtype=np.float32)
            m[int(np.clip(rel_s, 0, target_length)):int(np.clip(rel_e, 0, target_length))] = 1.0

            segments.append(seg)
            masks.append(m)

        if segments:
            self.beats = np.stack(segments, axis=0)
            self.beat_masks = np.stack(masks, axis=0)
            return True
        return False

    def normalize(self):
        """Per-record, per-lead normalization using the beat mask."""
        beats = self.beats.astype(np.float32)   # [B, T, C]
        masks = self.beat_masks.astype(np.float32)  # [B, T]
        m = masks[:, :, None]  # [B, T, 1]

        denom = np.maximum(np.sum(m, axis=1, keepdims=True), 1.0)
        mean = np.sum(beats * m, axis=1, keepdims=True) / denom
        var  = np.sum(((beats - mean) ** 2) * m, axis=1, keepdims=True) / denom
        std  = np.sqrt(np.maximum(var, 1e-6))

        self.beat_sigma = std[:, 0, :]                           # [B, C]
        self.beat_amax  = np.max(np.abs(beats * m), axis=1)     # [B, C]
        self.beat_representations = ((beats - mean) / std) * m  # [B, T, C]
        return self

    def get_sex(self):
        return "M" if self.sex == 0 else "F"

    def __repr__(self):
        return f"ECG_Record(id={self.id}, age={self.age}, sex={self.get_sex()}, label={self.label})"
