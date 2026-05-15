import numpy as np
import torch


class ECG_DataGenerator:
    """
    Wraps ECG_Dataset and provides batched generators for model training.

    Pre-converts record data to torch tensors with optional pinned memory
    for efficient GPU transfer.
    """

    def __init__(self, ecg_dataset, batch_size=300, device=None, pin_memory=True):
        self.ecg_dataset = ecg_dataset
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pin_memory = pin_memory and (self.device.type == "cuda")
        self._record_cache = {}

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

            if self.pin_memory:
                beats  = beats.pin_memory()
                masks  = masks.pin_memory()
                sigmas = sigmas.pin_memory()
                amaxes = amaxes.pin_memory()

            torch_records.append({
                "beats":     beats,
                "masks":     masks,
                "sigmas":    sigmas,
                "amaxes":    amaxes,
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

    # ------------------------------------------------------------------
    # Generators
    # ------------------------------------------------------------------

    def get_record_batch_generator(self, mode="train", max_beats=8, shuffle=True):
        """
        Yield batches of records.

        Each batch is a tuple::

            (beats, beat_mask, beat_meta, beat_valid,
             global_beat_idx, record_age, record_sex, labels)

        Shapes:
            beats          [B, N, C, T]
            beat_mask      [B, N, T]
            beat_meta      [B, N, 2*C]
            beat_valid     [B, N]
            global_beat_idx [B, N]  (-1 = padding)
            record_age     [B]
            record_sex     [B]
            labels         [B]
        """
        torch_records = self._preload_records_to_torch(split=mode)
        _, rec_to_global, _ = self.build_global_beat_index(split=mode)

        if max_beats is None:
            max_beats = max(r["num_beats"] for r in torch_records)

        r0 = torch_records[0]
        C, T = r0["beats"].shape[1], r0["beats"].shape[2]
        rec_indices = np.arange(len(torch_records))

        while True:
            if mode == "train" and shuffle:
                np.random.shuffle(rec_indices)

            for s in range(0, len(rec_indices), self.batch_size):
                batch_ridx = rec_indices[s:s + self.batch_size]
                B, N = len(batch_ridx), max_beats

                beats_list, mask_list, sig_list, amax_list = [], [], [], []
                valid_np = np.zeros((B, N), np.float32)
                gidx_np  = -np.ones((B, N), np.int64)
                ages_np  = np.zeros(B, np.float32)
                sex_np   = np.zeros(B, np.int64)
                labels_np = np.zeros(B, np.int64)

                for b, ridx in enumerate(batch_ridx):
                    r  = torch_records[ridx]
                    nb = r["num_beats"]
                    take = min(nb, N)

                    ages_np[b]   = float(r["age"])
                    sex_np[b]    = int(r["sex"])
                    labels_np[b] = int(r["label"])

                    sel = (
                        np.random.choice(nb, take, replace=False)
                        if mode == "train" and nb > take
                        else np.arange(take)
                    )

                    b_sel = r["beats"][sel]
                    m_sel = r["masks"][sel]
                    s_sel = r["sigmas"][sel]
                    a_sel = r["amaxes"][sel]

                    if take < N:
                        def _pad(t, shape):
                            p = torch.zeros(shape, dtype=t.dtype)
                            return torch.cat([t, p.pin_memory() if self.pin_memory else p], dim=0)

                        b_sel = _pad(b_sel, (N - take, C, T))
                        m_sel = _pad(m_sel, (N - take, T))
                        s_sel = _pad(s_sel, (N - take, C))
                        a_sel = _pad(a_sel, (N - take, C))

                    beats_list.append(b_sel)
                    mask_list.append(m_sel)
                    sig_list.append(s_sel)
                    amax_list.append(a_sel)

                    valid_np[b, :take] = 1.0
                    gidx_np[b, :take]  = rec_to_global[ridx][sel]

                nk = self.pin_memory
                beats_t = torch.stack(beats_list).to(self.device, non_blocking=nk)
                mask_t  = torch.stack(mask_list).to(self.device, non_blocking=nk)
                meta_t  = torch.cat(
                    [torch.stack(sig_list), torch.stack(amax_list)], dim=2
                ).to(self.device, non_blocking=nk)

                valid_t  = torch.from_numpy(valid_np).float().to(self.device)
                gidx_t   = torch.from_numpy(gidx_np).long().to(self.device)
                age_t    = torch.from_numpy(ages_np).float().to(self.device)
                sex_t    = torch.from_numpy(sex_np).long().to(self.device)
                labels_t = torch.from_numpy(labels_np).long().to(self.device)

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
        total = sum(len(r.beat_representations) for r in data)
        return int(np.ceil(total / self.batch_size))

    def clear_cache(self, split=None):
        if split is None:
            self._record_cache.clear()
        elif split in self._record_cache:
            del self._record_cache[split]
