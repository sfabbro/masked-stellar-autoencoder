"""Synthetic contracts only: no catalogue inputs, pretrained models or checkpoints."""

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_expanded_checkpoint_evaluation as evaluate

from masked_stellar_autoencoder.training.five_wave_heads import build_head


def valid_producer():
    cfg = dict(
        producer_source_sha="a" * 40,
        large_parent_sha256="b" * 64,
        original_prepared_sha256="c" * 64,
        library_revision="d" * 40,
    )
    arm = dict(producer_run_id="source-head")
    state = dict(
        source_sha=cfg["producer_source_sha"],
        parent_sha256=cfg["large_parent_sha256"],
        input_sha256=cfg["original_prepared_sha256"],
        arm=dict(run_id="source-head"),
        completed_presentations=150_000_000,
        completed_updates=4578,
    )
    manifest = dict(
        status="complete",
        source_sha=cfg["producer_source_sha"],
        completed_presentations=450_000_000,
        completed_updates=13734,
        checkpoint_sha256="e" * 64,
        selected_checkpoint=dict(
            sha256="f" * 64,
            presentations=150_000_000,
            updates=4578,
            all_saved_state_values_reload_equal=True,
        ),
        selected_model_counters=dict(presentations=150_000_000, updates=4578),
        full_state_reload_equal=True,
        policy=dict(library_revision=cfg["library_revision"]),
    )
    return manifest, state, cfg, arm


def test_selected_checkpoint_not_last_training_state():
    manifest, state, cfg, arm = valid_producer()
    evaluate.validate_producer(manifest, state, "e" * 64, cfg, arm)
    for field, bad in [
        ("status", "running"),
        ("completed_presentations", 150_000_000),
        ("full_state_reload_equal", False),
        ("checkpoint_sha256", "f" * 64),
    ]:
        changed = copy.deepcopy(manifest)
        changed[field] = bad
        with pytest.raises(ValueError):
            evaluate.validate_producer(changed, state, "e" * 64, cfg, arm)
    changed = copy.deepcopy(state)
    changed["completed_presentations"] = 450_000_000
    with pytest.raises(ValueError, match="Selected"):
        evaluate.validate_producer(manifest, changed, "e" * 64, cfg, arm)


def test_exact_id_matching_and_no_repeat_baseline():
    ids = np.array([7, 3, 11], dtype="i8")
    old = np.array([3, 7], dtype="i8")
    index = evaluate.matched_indices(ids, old)
    np.testing.assert_array_equal(index, [1, 0, -1])
    extra = np.zeros((3, 6), bool)
    extra[0, 4] = True
    np.testing.assert_array_equal(
        evaluate.new_inference_rows(index, extra, "quantile3"), [False, False, True]
    )
    np.testing.assert_array_equal(
        evaluate.new_inference_rows(index, extra, "gaussian"), [True, False, True]
    )
    with pytest.raises(ValueError, match="duplicate"):
        evaluate.matched_indices(ids, np.array([3, 3]))


def test_old_point_and_existing_scores_reused_new_labels_scored():
    old = dict(
        source_id=np.array([10]),
        point_native=np.arange(6)[None].astype(float),
        quantiles_native=np.ones((1, 3, 6)),
        intrinsic_pit=np.full((1, 6), 0.3),
        intrinsic_energy_full6_fit_standardized=np.array([np.nan]),
    )
    computed = dict(
        point_native=np.full((2, 6), 20.0),
        quantiles_native=np.full((2, 3, 6), 20.0),
        intrinsic_pit=np.full((2, 6), 0.8),
        intrinsic_energy_full6_fit_standardized=np.array([5.0, 6.0]),
    )
    mask = np.ones((2, 6), bool)
    previous = np.zeros_like(mask)
    previous[0, :3] = True
    r = evaluate.combine_cached(old, np.array([0, -1]), computed, mask, previous)
    np.testing.assert_array_equal(r["point_native"][0], old["point_native"][0])
    np.testing.assert_array_equal(r["intrinsic_pit"][0, :3], [0.3] * 3)
    np.testing.assert_array_equal(r["intrinsic_pit"][0, 3:], [0.8] * 3)
    assert r["intrinsic_energy_full6_fit_standardized"][0] == 5


def test_hdf5_unsorted_row_reads_and_array_store(tmp_path):
    path = tmp_path / "prepared.h5"
    with h5py.File(path, "w") as h:
        h["input"] = np.arange(12).reshape(4, 3)
        h["label_source"] = np.array([b"a", b"b", b"c", b"d"])
    source = evaluate.Prepared(path)
    np.testing.assert_array_equal(
        source.take("input", [3, 1, 3]), [[9, 10, 11], [3, 4, 5], [9, 10, 11]]
    )
    assert source.take("label_source", [2, 0]).tolist() == ["c", "a"]
    source.close()
    store = evaluate.ArrayStore(tmp_path / "arrays", 4)
    store.put(np.array([0, 1]), {"point": np.ones((2, 6))})
    store.put(np.array([2, 3]), {"point": np.full((2, 6), 2.0)})
    result = store.finish()
    assert isinstance(result["point"], np.memmap)
    np.testing.assert_array_equal(result["point"][:, 0], [1, 1, 2, 2])


def test_invalid_batch_rejected():
    part = dict(
        point_native=np.ones((2, 6)),
        quantiles_native=np.ones((2, 3, 6)),
        intrinsic_pit=np.full((2, 6), 0.5),
    )
    evaluate.validate_batch(part, np.ones((2, 6), bool))
    part["intrinsic_pit"][0, 0] = 1.1
    with pytest.raises(FloatingPointError, match="CDF"):
        evaluate.validate_batch(part, np.ones((2, 6), bool))


def test_cpu_guard_before_catalogue_or_checkpoint_access(tmp_path, monkeypatch):
    monkeypatch.setenv("MSA_EXPERIMENT_CPU_ONLY", "1")
    with pytest.raises(RuntimeError, match="CANFAR"):
        evaluate.run(
            SimpleNamespace(
                config=tmp_path / "missing", run_id="none", output=tmp_path / "out"
            )
        )
    assert not (tmp_path / "out").exists()


def test_synthetic_gaussian_scoring_and_external_age():
    torch.manual_seed(3)
    encoder = torch.nn.Linear(3, 2).double()
    model = SimpleNamespace(encoder=encoder)
    head = build_head(dict(kind="gaussian"), 2).double().eval()
    values = np.array(
        [[5000.0, 4.0, -0.5, 0.1, 4.0, 1.0], [5500.0, 3.0, -1.0, 0.2, 8.0, 2.0]]
    )
    center = np.array([5000.0, 3.5, -0.7, 0.15, 1.6, 0.2])
    scale = np.array([1000.0, 1.0, 0.5, 0.2, 0.5, 0.5])
    errors = np.zeros_like(values)
    known = np.zeros_like(values, dtype=bool)
    mask = np.ones_like(known)
    saved = []
    part, draws, contract = evaluate.score_new(
        model,
        head,
        np.zeros((2, 3)),
        values,
        errors,
        known,
        mask,
        center,
        scale,
        "cpu",
        7,
        capture=lambda v: saved.append(set(v)),
        external_age=np.array([5.0, 9.0]),
    )
    evaluate.validate_batch(part, mask)
    np.testing.assert_allclose(
        part["intrinsic_pit"], part["observed_pit"], atol=1e-12, rtol=0
    )
    assert saved and "intrinsic_log_prob_native" in saved[0]
    assert draws["intrinsic"][0].shape == (512, 2, 6)
    assert contract["mask"].shape == (2, 6)
    assert np.isfinite(part["external_age_intrinsic_crps_gyr"]).all()


def test_source_masks_exclude_cross_target_exposure_and_keep_unknown_photometry_diagnostic(
    tmp_path,
):
    path = tmp_path / "prepared.h5"
    with h5py.File(path, "w") as h:
        h["observed"] = np.ones((3, 6), bool)
        h["quality"] = np.ones((3, 6), bool)
        h["primary_score_eligible"] = np.ones((3, 6), bool)
        h["diagnostic_eligible"] = np.ones((3, 6), bool)
        h["strict_unexposed_star"] = [False, True, True]
        h["ancestor_feh_supervised_fit_exposed"] = [True, False, False]
        h["original_test_member"] = [True, False, False]
        h["xp_status"] = np.array(
            [b"complete_xp", b"photometry_only_publication_unknown", b"official_no_xp"]
        )
    data = evaluate.Prepared(path)
    mask, diagnostic, raw, uncertain = evaluate.source_reference_masks(
        data, np.arange(3)
    )
    assert not mask[0].any() and not diagnostic[0].any()
    assert not mask[1].any() and diagnostic[1].all() and uncertain[1]
    assert mask[2].all()
    assert raw[1] == "photometry_only_publication_unknown"
    assert evaluate.report_xp_status(raw).tolist() == [
        "complete_xp",
        "unknown",
        "official_no_xp",
    ]
    data.close()
    with h5py.File(path, "r+") as h:
        h["strict_unexposed_star"][0] = True
    data = evaluate.Prepared(path)
    with pytest.raises(ValueError, match="ancestry"):
        evaluate.source_reference_masks(data, np.arange(3))
    data.close()


def test_synthetic_external_age_only_scoring_has_no_invented_primary_label():
    torch.manual_seed(4)
    model = SimpleNamespace(encoder=torch.nn.Linear(2, 2).double())
    head = build_head(dict(kind="gaussian"), 2).double().eval()
    values = np.full((2, 6), np.nan)
    center = np.zeros(6)
    scale = np.ones(6)
    mask = np.zeros((2, 6), bool)
    part, _, _ = evaluate.score_new(
        model,
        head,
        np.zeros((2, 2)),
        values,
        values.copy(),
        mask,
        mask,
        center,
        scale,
        "cpu",
        14,
        external_age=np.array([5.0, np.nan]),
    )
    evaluate.validate_batch(part, mask)
    assert np.isnan(part["intrinsic_pit"]).all()
    assert np.isfinite(part["external_age_intrinsic_pit"][0])
    assert np.isnan(part["external_age_intrinsic_pit"][1])
    assert np.isfinite(part["external_age_intrinsic_crps_gyr"][0])


def test_external_age_report_accepts_readonly_memmap_without_mutating_it(tmp_path):
    import json

    path = tmp_path / "external-mask.npy"
    np.save(path, np.array([True, True, True]))
    original_bytes = path.read_bytes()
    readonly = np.load(path, mmap_mode="r", allow_pickle=False)
    point = np.ones((3, 6))
    point[:, 4] = [5.1, 5.8, 7.0]
    predictions = dict(
        external_age_observed=readonly,
        external_age_value=np.array([5.0, 6.0, np.nan]),
        point_native=point,
    )
    evaluate.external_report(predictions, tmp_path / "report")
    summary = json.loads((tmp_path / "report/summary.json").read_text())
    assert summary["count"] == 2 and summary["pooled_with_teacher"] is False
    assert path.read_bytes() == original_bytes
    np.testing.assert_array_equal(readonly, [True, True, True])
