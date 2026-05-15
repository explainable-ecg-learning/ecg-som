import numpy as np
import torch


class ECG_DataGenerator:
    """
    Data generator that encapsulates ECG_Dataset and provides both
    batch generation and direct data access methods.
    Uses num_beats from each record.
    Converts data to torch tensors with pinned memory only once
    for get_record_batch_generator.

    Returns torch tensors directly.
    """

    def __init__(self, ecg_dataset, batch_size=300, device=None, pin_memory=True):
        self.ecg_dataset = ecg_dataset
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pin_memory = pin_memory and (self.device.type == 'cuda')

        # Cache for pre-converted torch tensors (record-level)
        self._record_cache = {}

    def _preload_records_to_torch(self, split="train"):
        """
        Pre-convert all record data to torch tensors with pinned memory once.
        Stores beat representations, masks, and meta features as torch tensors.

        Also stores a per-beat rhythmic flag (0/1) replicated from the record,
        so it can be used to mask losses (e.g. time_smoothness_loss).
        """
        if split in self._record_cache:
            return self._record_cache[split]

        records = self.ecg_dataset.get_data(split)

        # Pre-convert each record's beat data to torch tensors
        torch_records = []
        for record in records:
            # Convert beat representations [num_beats, T, C] -> [num_beats, C, T]
            beats_np = record.beat_representations
            beats_torch = torch.from_numpy(beats_np).float()
            if beats_torch.ndim == 3:
                beats_torch = beats_torch.permute(0, 2, 1).contiguous()  # [num_beats, C, T]

            # Convert masks [num_beats, T]
            masks_torch = torch.from_numpy(record.beat_masks).float()

            # Convert meta features [num_beats, C]
            sigmas_torch = torch.from_numpy(record.beat_sigma).float()
            amaxes_torch = torch.from_numpy(record.beat_amax).float()

            # Per-beat rhythmic flag (replicated from record)
            num_beats = len(record.beat_representations)

            # Pin memory if using CUDA
            if self.pin_memory:
                beats_torch = beats_torch.pin_memory()
                masks_torch = masks_torch.pin_memory()
                sigmas_torch = sigmas_torch.pin_memory()
                amaxes_torch = amaxes_torch.pin_memory()


            torch_records.append({
                'beats': beats_torch,
                'masks': masks_torch,
                'sigmas': sigmas_torch,
                'amaxes': amaxes_torch,
                'age': record.age,
                'sex': record.sex,
                'label': self.ecg_dataset.label_encoder.transform([record.label])[0],
                'num_beats': num_beats
            })

        self._record_cache[split] = torch_records
        return torch_records

    def _np_to_torch(self, x_np):
        """Convert ECG data [B, T, C] to torch tensor [B, C, T]"""
        x = torch.from_numpy(x_np).float()
        if x.ndim == 3:
            x = x.permute(0, 2, 1).contiguous()  # [B, C, T]
        return x.to(self.device)

    def _meta_to_torch(self, sigmas_np, amaxes_np):
        """
        Convert meta-features (sigma, amax) to a single tensor.
        Shapes:
          sigmas: [B, C]
          amaxes: [B, C]
        Returns: [B, 2*C] float32
        """
        sigmas = torch.from_numpy(sigmas_np).float()  # [B,C]
        amaxes = torch.from_numpy(amaxes_np).float()  # [B,C]
        meta = torch.cat([sigmas, amaxes], dim=1)  # [B, 2*C]
        return meta.to(self.device)

    def build_global_beat_index(self, split="train"):
        """
        Builds a deterministic mapping:
          record_idx -> np.array(global beat indices)
        where 'global beat index' matches the flattening order of
        get_all_beats_representation(split).
        """
        records = self.ecg_dataset.get_data(split)
        rec_to_global = []
        g = 0
        for r in records:
            nb = len(r.beat_representations)
            rec_to_global.append(np.arange(g, g + nb, dtype=np.int64))
            g += nb
        return records, rec_to_global, g  # total beats in split

    def get_record_batch_generator(self, mode="train", max_beats=8, shuffle=True):
        """
        OPTIMIZED: Pre-converts all record data to torch tensors once.

        Yields:
          beats: [B, N, C, T] torch.float
          beat_time_mask: [B, N, T] torch.float
          beat_meta: [B, N, 2C] torch.float
          beat_valid: [B, N] torch.float (1 real, 0 pad)
          global_beat_idx: [B, N] torch.long (-1 pad)
          record_age: [B] torch.float
          record_sex: [B] torch.long
          rr: [B, N, 3] torch.float
          labels: [B] torch.long
          beat_rhythmic: [B, N] torch.float (0/1, padded with 0)
        """
        # Pre-convert all records to torch tensors once
        torch_records = self._preload_records_to_torch(split=mode)
        _, rec_to_global, _ = self.build_global_beat_index(split=mode)

        rec_indices = np.arange(len(torch_records))

        # None case - use maximum beats across all records
        if max_beats is None:
            max_beats = max(r['num_beats'] for r in torch_records)

        # Get dimensions from first record
        r0 = torch_records[0]
        C, T = r0['beats'].shape[1], r0['beats'].shape[2]  # [num_beats, C, T]

        while True:
            if mode == "train" and shuffle:
                np.random.shuffle(rec_indices)

            for s in range(0, len(rec_indices), self.batch_size):
                batch_ridx = rec_indices[s:s + self.batch_size]
                B = len(batch_ridx)
                N = max_beats

                # Pre-allocate tensors on CPU
                beats_list = []
                mask_list = []
                sig_list = []
                amax_list = []
                rr_list = []
                rh_list = []

                valid_np = np.zeros((B, N), np.float32)
                gidx_np = -np.ones((B, N), np.int64)
                ages_np = np.zeros((B,), np.float32)
                sex_np = np.zeros((B,), np.int64)
                labels_np = np.zeros((B,), np.int64)

                for b, ridx in enumerate(batch_ridx):
                    r = torch_records[ridx]
                    ages_np[b] = float(r['age'])
                    sex_np[b] = int(r['sex'])
                    labels_np[b] = int(r['label'])
                    nb = r['num_beats']
                    take = min(nb, N)

                    # Select which beats to use
                    if mode == "train" and nb > take:
                        sel = np.random.choice(nb, take, replace=False)
                    else:
                        sel = np.arange(take)

                    # Index pre-converted torch tensors (much faster!)
                    beats_sel = r['beats'][sel]      # [take, C, T]
                    masks_sel = r['masks'][sel]      # [take, T]
                    sig_sel = r['sigmas'][sel]       # [take, C]
                    amax_sel = r['amaxes'][sel]      # [take, C]

                    # Pad to N beats if necessary
                    if take < N:
                        pad_beats = torch.zeros((N - take, C, T), dtype=beats_sel.dtype)
                        pad_masks = torch.zeros((N - take, T), dtype=masks_sel.dtype)
                        pad_sig = torch.zeros((N - take, C), dtype=sig_sel.dtype)
                        pad_amax = torch.zeros((N - take, C), dtype=amax_sel.dtype)

                        if self.pin_memory:
                            pad_beats = pad_beats.pin_memory()
                            pad_masks = pad_masks.pin_memory()
                            pad_sig = pad_sig.pin_memory()
                            pad_amax = pad_amax.pin_memory()

                        beats_sel = torch.cat([beats_sel, pad_beats], dim=0)
                        masks_sel = torch.cat([masks_sel, pad_masks], dim=0)
                        sig_sel = torch.cat([sig_sel, pad_sig], dim=0)
                        amax_sel = torch.cat([amax_sel, pad_amax], dim=0)

                    beats_list.append(beats_sel)
                    mask_list.append(masks_sel)
                    sig_list.append(sig_sel)
                    amax_list.append(amax_sel)

                    valid_np[b, :take] = 1.0
                    gsel = rec_to_global[ridx][sel]
                    gidx_np[b, :take] = gsel

                # Stack all batches
                beats_stacked = torch.stack(beats_list, dim=0)  # [B, N, C, T]
                masks_stacked = torch.stack(mask_list, dim=0)   # [B, N, T]
                sig_stacked = torch.stack(sig_list, dim=0)      # [B, N, C]
                amax_stacked = torch.stack(amax_list, dim=0)    # [B, N, C]

                # Combine meta features
                meta_stacked = torch.cat([sig_stacked, amax_stacked], dim=2)  # [B, N, 2*C]

                # Transfer to device (non-blocking with pinned memory)
                beats_t = beats_stacked.to(self.device, non_blocking=self.pin_memory)
                mask_t = masks_stacked.to(self.device, non_blocking=self.pin_memory)
                meta_t = meta_stacked.to(self.device, non_blocking=self.pin_memory)

                # Convert remaining numpy arrays
                valid_t = torch.from_numpy(valid_np).float().to(self.device)
                gidx_t = torch.from_numpy(gidx_np).long().to(self.device)
                age_t = torch.from_numpy(ages_np).float().to(self.device)
                sex_t = torch.from_numpy(sex_np).long().to(self.device)
                labels_t = torch.from_numpy(labels_np).long().to(self.device)

                yield beats_t, mask_t, meta_t, valid_t, gidx_t, age_t, sex_t, labels_t

    def get_batch_generator(self, mode="train"):
        """Generate batches using instance batch_size. Returns torch tensors."""
        X, X_mask, y, ages, sexes, sigmas, amaxes, rhythmic = self.get_all_beats_representation(split=mode)

        num_samples = len(X)

        while True:
            indices = np.arange(num_samples)
            if mode == "train":
                np.random.shuffle(indices)

            # Calculate batches including partial last batch
            num_batches = int(np.ceil(num_samples / self.batch_size))

            for i in range(num_batches):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_samples)
                batch_indices = indices[start_idx:end_idx]

                # Convert to torch tensors
                x_batch = self._np_to_torch(X[batch_indices])
                mask_batch = torch.from_numpy(X_mask[batch_indices]).float().to(self.device)
                meta_batch = self._meta_to_torch(sigmas[batch_indices], amaxes[batch_indices])
                age_target = torch.from_numpy(ages[batch_indices]).to(self.device)
                sex_target = torch.from_numpy(sexes[batch_indices]).to(self.device)
                labels_batch = y[batch_indices]
                rhythmic_batch = torch.from_numpy(rhythmic[batch_indices]).float().to(self.device)

                yield (
                    x_batch,
                    mask_batch,
                    labels_batch,
                    age_target,
                    sex_target,
                    meta_batch,
                    rhythmic_batch,
                    batch_indices
                )

    def get_all_beats_representation(self, split='all'):
        """
        Extract all beats from all records, flattening them into individual samples.
        Each beat becomes a separate sample with replicated metadata.

        Returns:
          X: [N, T, C]
          X_mask: [N, T]
          y: [N]
          ages: [N]
          sexes: [N]
          sigmas: [N, C]
          amaxes: [N, C]
          rhythmic: [N] float32 0/1
        """
        filtered = self.ecg_dataset.get_data(split)

        X_list = []
        X_mask_list = []
        y_list = []
        ages_list = []
        sexes_list = []
        sigmas_list = []
        amaxes_list = []
        rhythmic_list = []

        for record in filtered:
            num_beats = len(record.beat_representations)
            rh_val = float(int(getattr(record, "rhythmic", 0)))
            for beat_idx in range(num_beats):
                X_list.append(record.beat_representations[beat_idx])
                X_mask_list.append(record.beat_masks[beat_idx])
                y_list.append(record.label)
                ages_list.append(record.age)
                sexes_list.append(record.sex)
                sigmas_list.append(record.beat_sigma[beat_idx])
                amaxes_list.append(record.beat_amax[beat_idx])
                rhythmic_list.append(rh_val)

        X = np.stack(X_list)
        X_mask = np.stack(X_mask_list)
        y = self.ecg_dataset.label_encoder.transform(y_list)
        ages = np.array(ages_list, dtype=np.float32)
        sexes = np.array(sexes_list, dtype=np.int64)
        sigmas = np.array(sigmas_list, dtype=np.float32)
        amaxes = np.array(amaxes_list, dtype=np.float32)
        rhythmic = np.array(rhythmic_list, dtype=np.float32)

        return X, X_mask, y, ages, sexes, sigmas, amaxes, rhythmic

    def get_data(self, split="train"):
        """Get full dataset for a split without batching."""
        return self.get_all_beats_representation(split=split)

    def get_num_batches(self, split="train"):
        """Get number of batches (including partial last batch)."""
        data = self.get_all_beats_representation(split=split)
        return int(np.ceil(len(data[0]) / self.batch_size))

    def clear_cache(self, split=None):
        """Clear cached torch tensors to free memory."""
        if split is None:
            self._record_cache.clear()
        elif split in self._record_cache:
            del self._record_cache[split]
