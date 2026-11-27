import numpy as np
import torch


class ECG_DataGenerator:
    """
    Wraps ECG_Dataset and provides batched generators for model training.

    Pre-converts record data to torch tensors with optional pinned memory
    for efficient GPU transfer.
    """

    def __init__(self, ecg_dataset, batch_size=300, device=None, pin_memory=True,
                 use_rr_features=False, preload_to_gpu=True):
        self.ecg_dataset = ecg_dataset
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pin_memory = pin_memory and (self.device.type == "cuda")
        self.use_rr_features = use_rr_features
        self.preload_to_gpu = preload_to_gpu and (self.device.type == "cuda")
        self._record_cache = {}
        self._gpu_split_cache = {}

    @staticmethod
    def _compute_rr_features(record) -> torch.Tensor:
        """Compute per-beat (RR_prev, RR_next) in seconds, clipped to [0.2, 2.5] s."""
        nb = len(record.beat_representations)
        rr = np.full((nb, 2), 0.8, dtype=np.float32)  # fallback ~75 bpm
        peaks = getattr(record, "r_peaks", None)
        fs = float(getattr(record, "fs", 100.0))
        if peaks is not None and len(peaks) >= nb + 2 and fs > 0:
            p = np.asarray(peaks, dtype=np.float32)
            rr_prev = (p[1:nb + 1] - p[0:nb]) / fs
            rr_next = (p[2:nb + 2] - p[1:nb + 1]) / fs
            rr[:, 0] = np.clip(rr_prev, 0.2, 2.5)
            rr[:, 1] = np.clip(rr_next, 0.2, 2.5)
        return torch.from_numpy(rr)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preload_records_to_torch(self, split="train"):
        if split in self._record_cache:
            return self._record_cache[split]

        torch_records = []
        for record in self.ecg_dataset.get_data(split):
            beats = torch.from_numpy(record.beat_representations).float()
            if beats.ndim == 3:
                beats = beats.permute(0, 2, 1).contiguous()  # [B, C, T]

            masks   = torch.from_numpy(record.beat_masks).float()
            sigmas  = torch.from_numpy(record.beat_sigma).float()
            amaxes  = torch.from_numpy(record.beat_amax).float()
            rr      = self._compute_rr_features(record) if self.use_rr_features else None

            if self.pin_memory:
                beats  = beats.pin_memory()
                masks  = masks.pin_memory()
                sigmas = sigmas.pin_memory()
                amaxes = amaxes.pin_memory()
                if rr is not None:
                    rr = rr.pin_memory()

            torch_records.append({
                "beats":     beats,
                "masks":     masks,
                "sigmas":    sigmas,
                "amaxes":    amaxes,
                "rr":        rr,
                "age":       record.age,
                "sex":       record.sex,
                "label":     self.ecg_dataset.label_encoder.transform([record.label])[0],
                "num_beats": len(record.beat_representations),
            })

        self._record_cache[split] = torch_records
        return torch_records

    def _np_to_torch(self, x_np):
        x = torch.from_numpy(x_np).float()
        if x.ndim == 3:
            x = x.permute(0, 2, 1).contiguous()
        return x.to(self.device)

    def _meta_to_torch(self, sigmas_np, amaxes_np):
        return torch.cat([
            torch.from_numpy(sigmas_np).float(),
            torch.from_numpy(amaxes_np).float(),
        ], dim=1).to(self.device)

    def build_global_beat_index(self, split="train"):
        records = self.ecg_dataset.get_data(split)
        rec_to_global, g = [], 0
        for r in records:
            nb = len(r.beat_representations)
            rec_to_global.append(np.arange(g, g + nb, dtype=np.int64))
            g += nb
        return records, rec_to_global, g

    def _preload_split_tensors(self, split="train"):
        """Pack an entire split into contiguous flat tensors (resident on GPU when enabled).

        Eliminates per-batch Python loops and CPU->GPU copies during training/eval.
        """
        if split in self._gpu_split_cache:
            return self._gpu_split_cache[split]

        records = self.ecg_dataset.get_data(split)
        target_dev = self.device if self.preload_to_gpu else torch.device("cpu")

        beats_all, masks_all, meta_all = [], [], []
        num_beats = np.empty(len(records), dtype=np.int64)
        ages_np   = np.empty(len(records), dtype=np.float32)
        sex_np    = np.empty(len(records), dtype=np.int64)
        labels_np = self.ecg_dataset.label_encoder.transform([r.label for r in records]).astype(np.int64)

        for i, r in enumerate(records):
            nb = len(r.beat_representations)
            num_beats[i] = nb
            ages_np[i]   = float(r.age)
            sex_np[i]    = int(r.sex)

            b = torch.from_numpy(r.beat_representations).float()
            if b.ndim == 3:
                b = b.permute(0, 2, 1).contiguous()  # [nb, C, T]
            m = torch.from_numpy(r.beat_masks).float()
            parts = [
                torch.from_numpy(r.beat_sigma).float(),
                torch.from_numpy(r.beat_amax).float(),
            ]
            if self.use_rr_features:
                parts.append(self._compute_rr_features(r))
            beats_all.append(b)
            masks_all.append(m)
            meta_all.append(torch.cat(parts, dim=1))

        max_nb = int(num_beats.max())
        rec_gidx = np.full((len(records), max_nb), -1, dtype=np.int64)
        offset = 0
        for i, nb in enumerate(num_beats):
            rec_gidx[i, :nb] = np.arange(offset, offset + nb, dtype=np.int64)
            offset += int(nb)

        pack = {
            "all_beats": torch.cat(beats_all, dim=0).to(target_dev, non_blocking=True),
            "all_masks": torch.cat(masks_all, dim=0).to(target_dev, non_blocking=True),
            "all_meta":  torch.cat(meta_all,  dim=0).to(target_dev, non_blocking=True),
            "rec_gidx":  torch.from_numpy(rec_gidx).to(target_dev, non_blocking=True),
            "num_beats": torch.from_numpy(num_beats).to(target_dev, non_blocking=True),
            "ages":      torch.from_numpy(ages_np).to(target_dev, non_blocking=True),
            "sex":       torch.from_numpy(sex_np).to(target_dev, non_blocking=True),
            "labels":    torch.from_numpy(labels_np).to(target_dev, non_blocking=True),
            "max_nb":    max_nb,
            "num_records": len(records),
        }
        self._gpu_split_cache[split] = pack
        return pack

    def get_flat_beats_tensor(self, split="train") -> torch.Tensor:
        """Return contiguous [total_beats, C, T] tensor (on GPU when preloaded)."""
        return self._preload_split_tensors(split)["all_beats"]

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------

    def get_record_batch_generator(self, mode="train", max_beats=8, shuffle=True):
        """
        Yield batches of records using vectorized tensor gathering.

        Each batch is a tuple::

            (beats, beat_mask, beat_meta, beat_valid,
             global_beat_idx, record_age, record_sex, labels)
        """
        pack = self._preload_split_tensors(split=mode)
        all_beats = pack["all_beats"]
        all_masks = pack["all_masks"]
        all_meta  = pack["all_meta"]
        rec_gidx  = pack["rec_gidx"]
        num_beats = pack["num_beats"]
        ages_all  = pack["ages"]
        sex_all   = pack["sex"]
        labels_all = pack["labels"]
        num_records = pack["num_records"]
        max_nb      = pack["max_nb"]

        N = max_nb if max_beats is None else int(max_beats)
        dev = all_beats.device
        pos_range = torch.arange(N, device=dev)

        while True:
            if mode == "train" and shuffle:
                perm = torch.randperm(num_records, device=dev)
            else:
                perm = torch.arange(num_records, device=dev)

            for s in range(0, num_records, self.batch_size):
                ridx = perm[s:s + self.batch_size]
                B = ridx.numel()
                nb = num_beats[ridx]  # [B]
                gidx_row = rec_gidx[ridx]  # [B, max_nb]

                if mode == "train" and max_nb > N:
                    # Vectorized random without-replacement selection (sorted chronologically)
                    scores = torch.rand(B, max_nb, device=dev)
                    invalid_pos = torch.arange(max_nb, device=dev).unsqueeze(0) >= nb.unsqueeze(1)
                    scores = scores.masked_fill(invalid_pos, 2.0)
                    sel_pos = scores.topk(N, dim=1, largest=False).indices
                    sel_pos, _ = torch.sort(sel_pos, dim=1)
                    gidx_t = gidx_row.gather(1, sel_pos)
                else:
                    if max_nb >= N:
                        gidx_t = gidx_row[:, :N]
                    else:
                        pad = torch.full((B, N - max_nb), -1, dtype=torch.long, device=dev)
                        gidx_t = torch.cat([gidx_row, pad], dim=1)

                valid_bool = pos_range.unsqueeze(0) < nb.unsqueeze(1)  # [B, N]
                gidx_t = gidx_t.masked_fill(~valid_bool, -1)
                safe_idx = gidx_t.clamp_min(0)

                valid_t = valid_bool.float()
                beats_t = all_beats[safe_idx] * valid_t.unsqueeze(-1).unsqueeze(-1)
                mask_t  = all_masks[safe_idx] * valid_t.unsqueeze(-1)
                meta_t  = all_meta[safe_idx]  * valid_t.unsqueeze(-1)
                age_t   = ages_all[ridx]
                sex_t   = sex_all[ridx]
                labels_t = labels_all[ridx]

                if dev != self.device:
                    beats_t  = beats_t.to(self.device, non_blocking=self.pin_memory)
                    mask_t   = mask_t.to(self.device, non_blocking=self.pin_memory)
                    meta_t   = meta_t.to(self.device, non_blocking=self.pin_memory)
                    valid_t  = valid_t.to(self.device, non_blocking=self.pin_memory)
                    gidx_t   = gidx_t.to(self.device, non_blocking=self.pin_memory)
                    age_t    = age_t.to(self.device, non_blocking=self.pin_memory)
                    sex_t    = sex_t.to(self.device, non_blocking=self.pin_memory)
                    labels_t = labels_t.to(self.device, non_blocking=self.pin_memory)

                yield beats_t, mask_t, meta_t, valid_t, gidx_t, age_t, sex_t, labels_t

    def get_all_beats_representation(self, split="all"):
        """
        Flatten all beats from all records into per-beat arrays.

        Returns:
            X          [N, T, C]
            X_mask     [N, T]
            y          [N]  (encoded labels)
            ages       [N]
            sexes      [N]
            sigmas     [N, C]
            amaxes     [N, C]
            rhythmic   [N]
        """
        X_list, mask_list, y_list = [], [], []
        ages_list, sexes_list, sig_list, amax_list, rhy_list = [], [], [], [], []

        for record in self.ecg_dataset.get_data(split):
            nb = len(record.beat_representations)
            rh = float(int(getattr(record, "rhythmic", 0)))
            for i in range(nb):
                X_list.append(record.beat_representations[i])
                mask_list.append(record.beat_masks[i])
                y_list.append(record.label)
                ages_list.append(record.age)
                sexes_list.append(record.sex)
                sig_list.append(record.beat_sigma[i])
                amax_list.append(record.beat_amax[i])
                rhy_list.append(rh)

        y_enc = self.ecg_dataset.label_encoder.transform(y_list)
        return (
            np.stack(X_list),
            np.stack(mask_list),
            y_enc,
            np.array(ages_list,  dtype=np.float32),
            np.array(sexes_list, dtype=np.int64),
            np.array(sig_list,   dtype=np.float32),
            np.array(amax_list,  dtype=np.float32),
            np.array(rhy_list,   dtype=np.float32),
        )

    def get_data(self, split="train"):
        return self.get_all_beats_representation(split=split)

    def get_num_batches(self, split="train"):
        data = self.ecg_dataset.get_data(split)
        return int(np.ceil(len(data) / self.batch_size))

    def clear_cache(self, split=None):
        if split is None:
            self._record_cache.clear()
            self._gpu_split_cache.clear()
        else:
            self._record_cache.pop(split, None)
            self._gpu_split_cache.pop(split, None)
