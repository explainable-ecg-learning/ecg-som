"""Unit and integration smoke tests for DPSOM-ECG."""
from __future__ import annotations

import os
import tempfile

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")

import numpy as np
import torch

from src.model.dpsom import DPSOM_ECG
from src.training.trainer import (
    _build_checkpoint,
    _load_checkpoint_state,
    _run_probe_fit_phase,
    load_dpsom_checkpoint,
)
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


def test_pruned_defaults_shortcircuit_and_mean_pool():
    model = DPSOM_ECG(
        latent_dim=16, som_dim=(4, 4), input_length=120, input_channels=12,
        eta=0.0, age_corr_lambda_max=0.0, amp_context_dim=24,
    )
    for _ in range(10):
        model.inc_epoch()
    assert model._age_corr_lambda() == 0.0

    # Corrupt corr weights massively; lambda=0 must completely ignore them
    with torch.no_grad():
        for p in model.age_corr_mlp.parameters():
            p.fill_(1e4)
        model.age_corr_amp_shortcut.weight.fill_(1e4)

    beats, beat_mask, beat_meta, beat_valid, age, sex, p_batch = _make_dummy_batch(amp_dim=24, som_nodes=16)
    beat_valid[:, 3:] = 0.0  # first 3 beats valid per record
    age_r, _, attn_a, attn_s = model.predict_dual_age(beats, beat_valid, beat_meta)
    assert torch.isfinite(age_r).all()
    expected_w = torch.tensor([1 / 3, 1 / 3, 1 / 3, 0.0, 0.0]).unsqueeze(0).expand(4, 5)
    assert torch.allclose(attn_a, expected_w, atol=1e-6)
    assert torch.allclose(attn_s, expected_w, atol=1e-6)

    losses = model.loss_batch(beats, beat_mask, beat_meta, beat_valid, age, sex, p_batch)
    assert losses[10].item() == 0.0  # record_loss == 0


def test_load_dpsom_checkpoint_roundtrip_and_strict():
    model = DPSOM_ECG(
        latent_dim=16, som_dim=(4, 4), input_length=120, input_channels=12,
        eta=0.0, age_corr_lambda_max=0.0, age_corr_topk=2, age_corr_ramp_epochs=5,
        amp_context_dim=24,
    )
    for _ in range(10):
        model.inc_epoch()

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = os.path.join(tmp, "pruned.ckpt")
        torch.save(_build_checkpoint(model), ckpt_path)
        reloaded, meta = load_dpsom_checkpoint(ckpt_path, torch.device("cpu"))

    assert reloaded.eta == 0.0
    assert reloaded.age_corr_lambda_max == 0.0
    assert reloaded._age_corr_lambda() == 0.0
    assert reloaded.amp_context_dim == 24
    assert meta["age_corr_lambda_max"] == 0.0
    assert meta["eta"] == 0.0


def test_non_orthogonal_probe_residualization():
    model = DPSOM_ECG(latent_dim=16, som_dim=(4, 4), input_length=120, input_channels=12)
    v1 = torch.zeros(16); v1[0] = 1.0
    v2 = torch.zeros(16); v2[0] = 1.0; v2[1] = 1.0  # cos(v1, v2) ≈ 0.707
    model.set_age_direction(v1, 0.0)
    model.set_sex_direction(v2, 0.0)

    z = torch.randn(64, 16)
    z_res = model.residualize_latent(z)
    assert torch.allclose((z_res * (v1 / v1.norm())).sum(dim=1), torch.zeros(64), atol=1e-5)
    assert torch.allclose((z_res * (v2 / v2.norm())).sum(dim=1), torch.zeros(64), atol=1e-5)


def test_closed_form_probe_removes_linear_age_and_zero_epochs_skips():
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    ds = ECG_Dataset(fs=100)
    for idx in range(20):
        age = 20.0 + 3.0 * idx
        sex = idx % 2
        sig = np.zeros((1000, 12), dtype=np.float32)
        beats = np.zeros((3, 120, 12), dtype=np.float32)
        # Embed an age- and sex-correlated DC pattern across beats
        beats[:, 20:40, 0] = (age - 50.0) / 10.0 + rng.normal(0, 0.05, size=(3, 20))
        beats[:, 40:60, 1] = (sex - 0.5) * 2.0 + rng.normal(0, 0.05, size=(3, 20))
        rec = ECG_Record(id=idx, signal=sig, age=age, sex=sex, label="NORM", fold=1, fs=100)
        rec.beat_representations = beats
        rec.beat_masks = np.ones((3, 120), dtype=np.float32)
        rec.beat_sigma = np.ones((3, 12), dtype=np.float32)
        rec.beat_amax = np.ones((3, 12), dtype=np.float32)
        ds.add_record(rec)
    ds.label_encoder.fit(["NORM"])

    gen = ECG_DataGenerator(ds, batch_size=20, device=torch.device("cpu"), pin_memory=False, use_rr_features=False)
    model_no = DPSOM_ECG(latent_dim=16, som_dim=(4, 4), input_length=120, input_channels=12, amp_context_dim=24)
    _run_probe_fit_phase(model_no, gen, 0, torch.device("cpu"), epochs=0, lr=1e-3, pbar=None,
                         num_batches=1, num_used_beats=3, target="age", label="age")
    assert not bool(model_no._probe_ready.item())
    z = torch.randn(8, 16)
    assert torch.allclose(model_no.residualize_latent(z), z)

    model_fit = DPSOM_ECG(latent_dim=16, som_dim=(4, 4), input_length=120, input_channels=12, amp_context_dim=24)
    # Quickly imprint linear age signal into model_fit.enc_mu (20 tiny steps)
    opt_tmp = torch.optim.Adam(model_fit.parameters(), lr=5e-3)
    batch0 = next(gen.get_record_batch_generator(mode="train", max_beats=3, shuffle=False))
    b0, v0, a0 = batch0[0], batch0[3], batch0[5]
    a_tgt = a0.unsqueeze(1).expand_as(v0)[v0 > 0.5]
    for _ in range(25):
        opt_tmp.zero_grad()
        mu0, _, _, _ = model_fit._encode(b0[v0 > 0.5])
        loss0 = torch.nn.functional.mse_loss(mu0[:, 0] * 15.0 + 50.0, a_tgt)
        loss0.backward()
        opt_tmp.step()

    _run_probe_fit_phase(model_fit, gen, 0, torch.device("cpu"), epochs=3, lr=1e-3, pbar=None,
                         num_batches=1, num_used_beats=3, target="age", label="age")
    _run_probe_fit_phase(model_fit, gen, 0, torch.device("cpu"), epochs=3, lr=1e-3, pbar=None,
                         num_batches=1, num_used_beats=3, target="sex", label="sex")
    assert bool(model_fit._probe_ready.item()) and bool(model_fit._sex_probe_ready.item())

    model_fit.eval()
    with torch.no_grad():
        mu_v, _, _, _ = model_fit._encode(b0[v0 > 0.5])
        mu_res = model_fit.residualize_latent(mu_v)
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    y_b = a_tgt.numpy()
    r2_raw = r2_score(y_b, Ridge(alpha=1e-2).fit(mu_v.numpy(), y_b).predict(mu_v.numpy()))
    r2_res = r2_score(y_b, Ridge(alpha=1e-2).fit(mu_res.numpy(), y_b).predict(mu_res.numpy()))
    assert r2_raw > 0.8 and r2_res < 0.02, f"Expected age R2 drop, got raw={r2_raw:.3f}, res={r2_res:.3f}"


def test_cinc_chapman_nested_records_snomed_and_error_filtering():
    import neurokit2 as nk
    import src.data.cinc as cinc_mod

    lead0 = nk.ecg_simulate(duration=10, sampling_rate=500, heart_rate=72, random_state=42)
    sig_500 = np.stack([lead0 * (0.8 + 0.05 * i) for i in range(12)], axis=1).astype(np.float32)  # [5000, 12]
    orig_rdsamp = cinc_mod.wfdb.rdsamp

    try:
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "WFDBRecords", "01", "010")
            os.makedirs(sub, exist_ok=True)
            with open(os.path.join(tmp, "RECORDS"), "w") as f:
                f.write("WFDBRecords/01/010/\n")
            with open(os.path.join(sub, "RECORDS"), "w") as f:
                f.write("JS00001\nJS00002\n")

            for rid in ("JS00001", "JS00002"):
                with open(os.path.join(sub, f"{rid}.hea"), "w") as f:
                    f.write(f"{rid} 12 500 5000\n#Age: 65\n#Sex: Female\n#Dx: 426761007\n")

            with open(os.path.join(tmp, "ConditionNames_SNOMED-CT.csv"), "w", encoding="utf-8-sig") as f:
                f.write("Acronym Name,Full Name,Snomed_CT\nSVT,Supraventricular Tachycardia,426761007\n")

            with open(os.path.join(tmp, "error_labels.csv"), "w") as f:
                f.write("record_id,record_path,severity_tier,err_dx_empty\n")
                f.write("JS00001,WFDBRecords/01/010/JS00001,0_clean,0\n")
                f.write("JS00002,WFDBRecords/01/010/JS00002,3_severe_error,0\n")

            expanded = cinc_mod._expand_records(cinc_mod.Path(tmp), cinc_mod.Path(tmp) / "RECORDS")
            assert expanded == ["WFDBRecords/01/010/JS00001", "WFDBRecords/01/010/JS00002"]
            assert cinc_mod.load_snomed_csv(os.path.join(tmp, "ConditionNames_SNOMED-CT.csv")) == {"426761007": "SVT"}

            cinc_mod.wfdb.rdsamp = lambda _stem: (sig_500, {"fs": 500.0, "sig_name": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]})
            ds = cinc_mod.import_cinc_dataset(tmp, min_class_count=1, max_error_severity=2)
            assert len(ds.records) == 1
            rec = ds.records[0]
            assert rec.id == "JS00001"
            assert rec.label == "SVT"
            assert rec.age == 65.0 and rec.sex == 1 and rec.fold == 10
    finally:
        cinc_mod.wfdb.rdsamp = orig_rdsamp


def _make_synth_ecg_dataset(n_records: int = 16, seed: int = 0) -> ECG_Dataset:
    rng = np.random.default_rng(seed)
    ds = ECG_Dataset(fs=100)
    labels = ["NORM", "AFIB"]
    for idx in range(n_records):
        lbl = labels[idx % 2]
        n_beats = 4
        beats = rng.normal(0, 0.2, size=(n_beats, 120, 12)).astype(np.float32)
        beats[:, 30:40, :] += (1.0 if lbl == "NORM" else -1.0)
        rec = ECG_Record(
            id=f"r{idx}",
            signal=np.zeros((1000, 12), dtype=np.float32),
            age=float(40 + (idx % 4) * 10),
            sex=idx % 2,
            label=lbl,
            fold=1,
            fs=100,
        )
        rec.beat_representations = beats
        rec.beat_masks = np.ones((n_beats, 120), dtype=np.float32)
        rec.beat_sigma = np.ones((n_beats, 12), dtype=np.float32)
        rec.beat_amax = np.ones((n_beats, 12), dtype=np.float32)
        ds.add_record(rec)
    ds.label_encoder.fit(labels)
    return ds


def test_uda_adaptation_methods():
    from src.training.uda import (
        adapt_fewshot_demo_heads,
        adapt_uda_bw_ot,
        adapt_uda_ot_strong,
        adapt_uda_replay_strong,
        adapt_uda_sf_strong,
        bures_wasserstein_ot_map,
        compute_hybrid_target_distribution,
        extract_domain_moments,
    )

    torch.manual_seed(0)
    src_ds = _make_synth_ecg_dataset(n_records=12, seed=1)
    tgt_ds = _make_synth_ecg_dataset(n_records=12, seed=2)
    src_model = DPSOM_ECG(
        latent_dim=16, som_dim=(4, 4), input_length=120, input_channels=12,
        encoder_base_channels_1=16, encoder_base_channels_2=32, encoder_fc_hidden_dim=64,
        eta=0.0, age_corr_lambda_max=0.0, amp_context_dim=24,
    )
    src_snap = {k: v.detach().clone() for k, v in src_model.state_dict().items()}

    src_moments = extract_domain_moments(src_model, src_ds, batch_size=6)
    tgt_moments = extract_domain_moments(src_model, tgt_ds, batch_size=6)
    assert set(src_moments.keys()) == {"mu", "z_age", "z_sex", "meta"}
    for m, c in src_moments.values():
        assert torch.isfinite(m).all() and torch.isfinite(c).all()

    a_ot, b_ot = bures_wasserstein_ot_map(*src_moments["mu"], *tgt_moments["mu"], reg=1e-2, shrink=0.5)
    assert a_ot.shape == (16, 16) and b_ot.shape == (16,)
    assert torch.isfinite(a_ot).all() and torch.isfinite(b_ot).all()

    tgt_gen = ECG_DataGenerator(
        tgt_ds, batch_size=6, device=torch.device("cpu"), pin_memory=False, use_rr_features=False
    )
    p_table = compute_hybrid_target_distribution(
        src_model, tgt_gen, split="train", device=torch.device("cpu"), use_amp=False
    )
    assert p_table.shape == (48, 16)
    assert torch.allclose(p_table.sum(dim=1), torch.ones(48), atol=1e-4)

    adapted_variants = [
        adapt_uda_bw_ot(src_model, tgt_ds, src_moments, s_morph=0.35, s_age=0.25, s_sex=0.75),
        adapt_uda_sf_strong(src_model, tgt_ds, batch_size=6, adapt_epochs=2, use_amp=False),
        adapt_uda_ot_strong(src_model, tgt_ds, src_moments, batch_size=6, adapt_epochs=2, use_amp=False),
        adapt_uda_replay_strong(src_model, src_ds, tgt_ds, src_moments, batch_size=6, adapt_epochs=2, use_amp=False),
        adapt_fewshot_demo_heads(src_model, tgt_ds, src_moments=src_moments, n_shots=8, seed=0, batch_size=4, adapt_epochs=2, use_amp=False),
    ]

    for k, v in src_model.state_dict().items():
        assert torch.equal(v, src_snap[k])

    batch = next(tgt_gen.get_record_batch_generator("train", max_beats=4, shuffle=False))
    beats, _, beat_meta, beat_valid = batch[:4]
    valid = beat_valid.reshape(-1) > 0.5
    b_flat = beats.reshape(-1, 12, 120)[valid]
    for adapted in adapted_variants:
        for p in adapted.parameters():
            assert torch.isfinite(p).all()
        adapted.eval()
        with torch.no_grad():
            mu = adapted._encode(b_flat)[0]
            k_ids = adapted.k(mu)
            age_r, sex_r, _, _ = adapted.predict_dual_age(beats, beat_valid, beat_meta)
        assert torch.isfinite(mu).all() and torch.isfinite(age_r).all() and torch.isfinite(sex_r).all()
        assert int(k_ids.unique().numel()) >= 2


def test_explorer_uda_and_overview_endpoints():
    import pickle
    from fastapi import HTTPException
    import explorer.server as srv

    # 1. Overview & UDA results & whitelisted figures endpoints
    ov = srv.get_overview()
    assert {"architecture", "datasets", "benchmarks", "uda_highlights", "figures"}.issubset(ov.keys())
    assert len(ov["architecture"]) >= 6 and len(ov["datasets"]) == 3

    uda_res = srv.get_uda_results()
    assert {"uda_ablation", "cross_dataset", "pkl_datasets"}.issubset(uda_res.keys())
    assert len(uda_res["uda_ablation"]["summary"]) > 0

    fig_resp = srv.get_figure("uda_ablation")
    assert os.path.isfile(fig_resp.path)

    # 2. Path traversal checks
    for bad_fn, arg in (
        (srv._resolve_ckpt_path, "../README.md"),
        (srv._resolve_data_path, "../README.md"),
    ):
        raised = False
        try:
            bad_fn(arg)
        except HTTPException as exc:
            raised = (exc.status_code == 400)
        assert raised, f"Expected HTTP 400 for traversal input {arg}"

    # 3. End-to-end live BW-OT run + apply into SOM Explorer state
    orig_models_dir = srv._MODELS_DIR
    orig_data_dir = srv._DATA_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            models_tmp = os.path.join(tmp, "models")
            data_tmp = os.path.join(tmp, "data")
            os.makedirs(models_tmp, exist_ok=True)
            os.makedirs(data_tmp, exist_ok=True)
            srv._MODELS_DIR = os.path.realpath(models_tmp)
            srv._DATA_DIR = os.path.realpath(data_tmp)

            m = DPSOM_ECG(
                latent_dim=16, som_dim=(4, 4), input_length=120, input_channels=12,
                encoder_base_channels_1=16, encoder_base_channels_2=32, encoder_fc_hidden_dim=64,
                eta=0.0, age_corr_lambda_max=0.0, amp_context_dim=24,
            )
            ckpt_p = os.path.join(models_tmp, "tiny.ckpt")
            torch.save(_build_checkpoint(m), ckpt_p)

            def _write_split_pkl(fname: str, seed: int):
                ds = _make_synth_ecg_dataset(n_records=12, seed=seed)
                for idx, r in enumerate(ds.records):
                    r.fold = 10 if idx >= 8 else 1
                p = os.path.join(data_tmp, fname)
                with open(p, "wb") as fh:
                    pickle.dump(ds, fh)
                return p

            _write_split_pkl("src.pkl", seed=11)
            _write_split_pkl("tgt.pkl", seed=22)

            req = srv.UdaRunRequest(
                ckpt="models/tiny.ckpt",
                src_data="src.pkl",
                tgt_data="tgt.pkl",
                method="uda_bw_ot",
                s_morph=0.35,
                s_age=0.25,
                s_sex=0.75,
            )
            res = srv.run_live_uda(req)
            assert "error" not in res, f"Unexpected UDA error: {res}"
            assert res["som_dim"] == [4, 4]
            assert res["can_apply"] is True
            assert len(res["before"]["cells"]) == 16 and len(res["after"]["cells"]) == 16
            for k, v in res["after"]["metrics"].items():
                assert np.isfinite(float(v)), f"Non-finite metric {k}: {v}"

            applied = srv.apply_live_uda()
            assert "error" not in applied, f"Unexpected apply error: {applied}"
            assert applied["som_dim"] == [4, 4]
            assert len(applied["cells"]) == 16
            assert "Adapted" in applied["info"]
            assert srv._STATE["som_dim"] == (4, 4)
    finally:
        srv._MODELS_DIR = orig_models_dir
        srv._DATA_DIR = orig_data_dir
        srv._UDA_STATE.clear()
        srv._MOMENTS_CACHE.clear()
        srv._STATE.clear()
        srv._SPLIT_CACHE.clear()
        srv._DATASET_CACHE.clear()


def test_explorer_compare_cells_endpoint():
    import pickle
    from fastapi import HTTPException
    import explorer.server as srv

    srv._STATE.clear()
    try:
        raised = False
        try:
            srv.compare_cells(srv.CompareCellsRequest(cells=[[0, 0]]))
        except HTTPException as exc:
            raised = (exc.status_code == 400)
        assert raised, "Expected HTTP 400 when compare_cells called without loaded model"

        orig_data_dir = srv._DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as tmp:
                srv._DATA_DIR = os.path.realpath(tmp)
                ds = _make_synth_ecg_dataset(n_records=12, seed=9)
                for idx, r in enumerate(ds.records):
                    r.fold = 10 if idx >= 6 else 1
                pkl_path = os.path.join(tmp, "cmp_test.pkl")
                with open(pkl_path, "wb") as fh:
                    pickle.dump(ds, fh)

                m = DPSOM_ECG(
                    latent_dim=16, som_dim=(4, 4), input_length=120, input_channels=12,
                    encoder_base_channels_1=16, encoder_base_channels_2=32, encoder_fc_hidden_dim=64,
                    eta=0.0, age_corr_lambda_max=0.0, amp_context_dim=24,
                )
                meta = {
                    "latent_dim": 16, "som_dim": (4, 4), "input_length": 120, "input_channels": 12,
                }
                srv._run_inference_with_model(
                    m, meta, "models/dummy.ckpt", pkl_path, "test", dataset_override=ds
                )

                # Test wrap-around toroidal neighbors (0,0) and (0,3) + (2,2), with duplicate deduplication
                res = srv.compare_cells(srv.CompareCellsRequest(cells=[[0, 0], [0, 3], [2, 2], [0, 0]]))
                assert res["som_dim"] == [4, 4]
                assert len(res["cells"]) == 3
                pw = res["pairwise"]
                assert len(pw) == 3 and len(pw[0]) == 3
                assert pw[0][1]["toroidal_l1"] == 1
                assert pw[0][2]["toroidal_l1"] == 4
                for i in range(3):
                    for j in range(3):
                        for k in ("latent_l2", "latent_cos", "waveform_rmse", "js_divergence"):
                            assert np.isfinite(float(pw[i][j][k])), f"Non-finite {k}"
                for c in res["cells"]:
                    assert len(c["proto_leads"]) == 12
                    assert 0.0 <= c["purity"] <= 1.0

                out_of_range = False
                try:
                    srv.compare_cells(srv.CompareCellsRequest(cells=[[9, 0]]))
                except HTTPException as exc:
                    out_of_range = (exc.status_code == 404)
                assert out_of_range, "Expected HTTP 404 for out-of-range coordinate"
        finally:
            srv._DATA_DIR = orig_data_dir
    finally:
        srv._STATE.clear()
        srv._SPLIT_CACHE.clear()
        srv._DATASET_CACHE.clear()


def test_static_site_builder_for_vercel():
    import json
    import pickle
    from explorer.build_vercel import build_vercel_bundle

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    with tempfile.TemporaryDirectory() as tmp:
        # 1. Lightweight static build (zero ML inference, pure metadata + figures)
        lite_dir = os.path.join(tmp, "lite_vercel")
        res_lite = build_vercel_bundle(lite_dir, repo_root=repo_root, lightweight=True)
        assert res_lite["lightweight"] is True
        idx_html_path = os.path.join(lite_dir, "index.html")
        assert os.path.isfile(idx_html_path)
        with open(idx_html_path, encoding="utf-8") as fh:
            idx_html = fh.read()
        assert "/_vercel/insights/script.js" in idx_html and "window.va" in idx_html
        assert "./static/style.css" in idx_html and "./data/models.json" in idx_html
        with open(os.path.join(lite_dir, "vercel.json"), encoding="utf-8") as fh:
            vcfg = json.load(fh)
        assert vcfg.get("cleanUrls") is True
        assert "buildCommand" not in vcfg
        assert any(h.get("source") == "/data/(.*)" for h in vcfg.get("headers", []))
        assert os.path.isfile(os.path.join(lite_dir, "README.md"))
        with open(os.path.join(lite_dir, "static", "site-config.js"), encoding="utf-8") as fh:
            cfg_js = fh.read()
        assert "EXPLORER_STATIC_MODE = true" in cfg_js

        with open(os.path.join(lite_dir, "data", "models.json"), encoding="utf-8") as fh:
            m_lite = json.load(fh)
        assert m_lite["static_mode"] is True and m_lite["models"] == []

        with open(os.path.join(lite_dir, "data", "overview.json"), encoding="utf-8") as fh:
            ov = json.load(fh)
        assert len(ov["figures"]) >= 1
        assert ov["figures"][0]["url"].startswith("./data/figures/")

        # 2. Full bundle build on synthetic checkpoint + dataset
        full_dir = os.path.join(tmp, "full_vercel")
        ckpt_p = os.path.join(tmp, "tiny.ckpt")
        ds_p = os.path.join(tmp, "synth.pkl")
        m = DPSOM_ECG(
            latent_dim=16, som_dim=(4, 4), input_length=120, input_channels=12,
            encoder_base_channels_1=16, encoder_base_channels_2=32, encoder_fc_hidden_dim=64,
            eta=0.0, age_corr_lambda_max=0.0, amp_context_dim=24,
        )
        torch.save(_build_checkpoint(m), ckpt_p)
        ds = _make_synth_ecg_dataset(n_records=10, seed=5)
        for idx, r in enumerate(ds.records):
            r.fold = 10 if idx >= 6 else 1
        with open(ds_p, "wb") as fh:
            pickle.dump(ds, fh)

        res_full = build_vercel_bundle(
            full_dir,
            repo_root=repo_root,
            lightweight=False,
            max_samples=2,
            skip_uda_demos=True,
            custom_bundles=[{
                "id":         "tiny_synth",
                "ckpt_abs":   ckpt_p,
                "ckpt_val":   "models/tiny.ckpt",
                "ckpt_label": "Tiny Synthetic",
                "data_abs":   ds_p,
                "data_val":   "synth.pkl",
                "data_label": "Synthetic ECG",
                "split":      "test",
            }],
        )
        assert res_full["n_bundles"] == 1
        with open(os.path.join(full_dir, "data", "models.json"), encoding="utf-8") as fh:
            m_full = json.load(fh)
        assert m_full["bundles"]["models/tiny.ckpt|synth.pkl|test"] == "tiny_synth"
        with open(os.path.join(full_dir, "data", "som", "tiny_synth.json"), encoding="utf-8") as fh:
            som = json.load(fh)
        assert som["som_dim"] == [4, 4]
        assert len(som["embeddings"]) == 4 and len(som["cell_stats"]) == 16
        with open(os.path.join(full_dir, "data", "cells", "tiny_synth", "0_0.json"), encoding="utf-8") as fh:
            c00 = json.load(fh)
        assert {"row", "col", "n_total", "proto_leads", "percentiles", "samples", "t_axis"}.issubset(c00.keys())


if __name__ == "__main__":
    import sys, traceback
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception:
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
