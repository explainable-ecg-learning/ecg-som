"""Unit and integration smoke tests for DPSOM-ECG."""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch

from src.model.dpsom import DPSOM_ECG
from src.training.trainer import _build_checkpoint, _load_checkpoint_state
from src.utils.metrics import compute_som_clustering_metrics
from src.data.dataset import ECG_Dataset
from src.data.record import ECG_Record
from src.data.generator import ECG_DataGenerator
from ablation.ablation import _flatten_results


def _make_dummy_batch(B=4, N=5, C=12, T=120, amp_dim=26, som_nodes=16):
    beats = torch.randn(B, N, C, T)
    beat_mask = torch.ones(B, N, T)
    beat_meta = torch.randn(B, N, amp_dim)
    beat_valid = torch.ones(B, N)
    beat_valid[:, -1] = 0.0  # include padded slot
    record_age = torch.randint(20, 80, (B,)).float()
    record_sex = torch.randint(0, 2, (B,))
    p_batch = torch.softmax(torch.randn(B, N, som_nodes), dim=-1)
    return beats, beat_mask, beat_meta, beat_valid, record_age, record_sex, p_batch


def test_dpsom_forward_backward_with_rr_and_custom_arch():
    model = DPSOM_ECG(
        latent_dim=16,
        som_dim=(4, 4),
        input_length=120,
        input_channels=12,
        encoder_base_channels_1=16,
        encoder_base_channels_2=32,
        encoder_kernel_size=5,
        encoder_fc_hidden_dim=128,
        z_age_dim_factor=0.5,
        amp_context_dim=26,
    )
    beats, beat_mask, beat_meta, beat_valid, age, sex, p_batch = _make_dummy_batch(amp_dim=26, som_nodes=16)

    # Phase 1 loss
    loss_rec, _, _, pred_loss, _, _ = model.loss_reconstruction_batch(
        beats, beat_mask, beat_meta, beat_valid, age, sex
    )
    (loss_rec + pred_loss).backward()
    model.zero_grad()

    # Phase 2 SOM init loss
    loss_init, _, _ = model.loss_a_batch(beats, beat_valid)
    loss_init.backward()
    model.zero_grad()

    # Phase 3 joint loss
    losses = model.loss_batch(beats, beat_mask, beat_meta, beat_valid, age, sex, p_batch)
    losses[0].backward()
    assert torch.isfinite(losses[0])

    # Record prediction
    age_r, sex_r, attn_a, attn_s = model.predict_dual_age(beats, beat_valid, beat_meta)
    assert age_r.shape == (4,)
    assert sex_r.shape == (4,)
    assert torch.allclose(attn_a.sum(dim=1), torch.ones(4), atol=1e-5)
    assert torch.allclose(attn_s.sum(dim=1), torch.ones(4), atol=1e-5)


def test_probe_residualization_orthogonality():
    model = DPSOM_ECG(latent_dim=16, som_dim=(4, 4), input_length=120, input_channels=12)
    with torch.no_grad():
        model.age_probe.weight.copy_(torch.eye(1, 16))
        model.sex_probe.weight.copy_(torch.roll(torch.eye(1, 16), shifts=1, dims=1))
    model.freeze_age_probe()
    model.freeze_sex_probe()

    z = torch.randn(32, 16)
    z_res = model.residualize_latent(z)
    assert torch.allclose((z_res * model._age_dir).sum(dim=1), torch.zeros(32), atol=1e-5)
    assert torch.allclose((z_res * model._sex_dir).sum(dim=1), torch.zeros(32), atol=1e-5)


def test_checkpoint_roundtrip_preserves_epoch_and_arch():
    model = DPSOM_ECG(
        latent_dim=16,
        som_dim=(4, 4),
        input_length=120,
        input_channels=12,
        encoder_base_channels_1=16,
        encoder_base_channels_2=32,
        encoder_kernel_size=3,
        encoder_fc_hidden_dim=64,
        z_age_dim_factor=0.5,
        amp_context_dim=26,
    )
    for _ in range(10):
        model.inc_epoch()
    assert model.get_epoch() == 10
    assert abs(model._age_corr_lambda() - 0.30) < 1e-6

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "test.ckpt")
        torch.save(_build_checkpoint(model), ckpt_path)
        state, meta = _load_checkpoint_state(ckpt_path, torch.device("cpu"))

    reloaded = DPSOM_ECG(
        latent_dim=meta["latent_dim"],
        som_dim=tuple(meta["som_dim"]),
        input_length=meta["input_length"],
        input_channels=meta["input_channels"],
        encoder_base_channels_1=meta["encoder_base_channels_1"],
        encoder_base_channels_2=meta["encoder_base_channels_2"],
        encoder_kernel_size=meta["encoder_kernel_size"],
        encoder_fc_hidden_dim=meta["encoder_fc_hidden_dim"],
        z_age_dim_factor=meta["z_age_dim_factor"],
        z_sex_dim_factor=meta["z_sex_dim_factor"],
        amp_context_dim=meta["amp_context_dim"],
    )
    reloaded.load_state_dict(state, strict=True)
    assert reloaded.get_epoch() == 10
    assert abs(reloaded._age_corr_lambda() - 0.30) < 1e-6


def test_toroidal_som_topology_metrics_wrap_around():
    # 4x4 grid where unit 0 (0,0) and unit 3 (0,3) are toroidal neighbors
    codebook = np.zeros((16, 2), dtype=np.float32)
    for i in range(16):
        codebook[i] = [100.0 + i, 100.0 + i]
    codebook[0] = [0.0, 0.0]
    codebook[3] = [1.0, 0.0]  # toroidal neighbor of (0,0) via wrap-around
    z = np.array([[0.2, 0.0], [0.8, 0.0]], dtype=np.float32)
    m = compute_som_clustering_metrics(z, codebook, som_shape=(4, 4), knn_k=1)
    assert m["topology"]["topographic_error"] == 0.0
    assert m["topology"]["neighborhood_consistency"] == 1.0


def test_rr_features_and_record_batch_count():
    ds = ECG_Dataset(fs=100)
    for idx in range(5):
        rec = ECG_Record(id=idx, signal=np.zeros((1000, 12), dtype=np.float32),
                         age=50, sex=0, label="NORM", fold=1, fs=100)
        rec.r_peaks = np.array([100, 180, 270, 350, 440, 520])
        rec.beat_representations = np.zeros((4, 120, 12), dtype=np.float32)
        rec.beat_masks = np.ones((4, 120), dtype=np.float32)
        rec.beat_sigma = np.ones((4, 12), dtype=np.float32)
        rec.beat_amax = np.ones((4, 12), dtype=np.float32)
        ds.add_record(rec)
    ds.label_encoder.fit(["NORM"])

    gen = ECG_DataGenerator(ds, batch_size=2, device=torch.device("cpu"), pin_memory=False, use_rr_features=True)
    assert gen.get_num_batches("train") == 3  # ceil(5 records / 2)
    batch = next(gen.get_record_batch_generator(mode="train", max_beats=4, shuffle=False))
    beat_meta = batch[2]
    assert beat_meta.shape == (2, 4, 26)
    # First beat corresponds to r_peaks[1]=180: rr_prev=0.8s, rr_next=0.9s
    assert np.allclose(beat_meta[0, 0, -2:].numpy(), [0.8, 0.9], atol=1e-5)


def test_ablation_flattener_extracts_disentanglement():
    dummy = {
        "NMI": 0.21,
        "Disentanglement": {
            "z_main": {
                "MIG": {"age": {"mig_norm": 0.1}, "sex": {"mig_norm": 0.2}},
                "SAP": {"SAP_age": 0.3, "SAP_sex": 0.4},
                "DCI": {"DCI_disentanglement": 0.5, "DCI_completeness": 0.6, "DCI_informativeness": 0.7},
            }
        },
    }
    row = _flatten_results("v", {"latent_dim": 32}, dummy)
    assert row["z_main/MIG_age"] == 0.1
    assert row["z_main/SAP_sex"] == 0.4
    assert row["z_main/DCI_info"] == 0.7


def test_inference_converter_and_prediction():
    import neurokit2 as nk
    from src.inference import convert_signal_to_record, load_raw_ecg, predict_signal

    # Synthesize a realistic 10 s 12-lead ECG at 500 Hz in [12, T] layout
    lead0 = nk.ecg_simulate(duration=10, sampling_rate=500, heart_rate=72, random_state=42)
    sig_500 = np.stack([lead0 * (0.8 + 0.05 * i) for i in range(12)], axis=0).astype(np.float32)  # [12, 5000]

    with tempfile.TemporaryDirectory() as tmp:
        npy_path = os.path.join(tmp, "sample.npy")
        np.save(npy_path, sig_500)
        loaded, fs = load_raw_ecg(npy_path, fs=500)
        assert loaded.shape == (12, 5000) and fs == 500.0

    rec = convert_signal_to_record(loaded, fs=500, record_id="synth_500hz")
    assert rec.fs == 100
    assert rec.beat_representations.shape[1:] == (120, 12)
    assert len(rec.beat_representations) >= 5

    model = DPSOM_ECG(latent_dim=16, som_dim=(4, 4), input_length=120, input_channels=12, amp_context_dim=26)
    model.eval()
    pred = predict_signal(model, sig_500, fs=500, record_id="synth_500hz")
    assert 2.0 <= pred["predicted_age"] <= 89.0
    assert pred["predicted_sex"] in ("Male", "Female")
    assert 0.0 <= pred["predicted_sex_prob_female"] <= 1.0
    assert pred["num_beats"] == len(pred["beat_cells"])
    assert 0 <= pred["dominant_cell"]["row"] < 4 and 0 <= pred["dominant_cell"]["col"] < 4
