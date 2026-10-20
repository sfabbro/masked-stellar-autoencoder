"""Actual tiny HDF5 staging, cache reuse, provenance and fail-closed policy."""

import h5py
import numpy as np
import pytest

from masked_stellar_autoencoder.training.large_reconstruction import (
    prepare_original_reconstruction,
)


def fixture(tmp_path):
    source = tmp_path / "original.h5"
    dtype = [("source_id", "i8"), ("G", "f4"), ("bp_0", "f4")]
    with h5py.File(source, "w") as handle:
        for name, ids in (
            ("train0", range(1, 11)),
            ("train1", range(11, 21)),
            ("valid", range(21, 24)),
        ):
            array = np.zeros(len(ids), dtype=dtype)
            array["source_id"] = list(ids)
            array["G"] = array["source_id"] + 10
            array["bp_0"] = array["source_id"] / 10
            handle.create_dataset(name, data=array)
    signature = dict(
        train_keys=["train0", "train1"],
        train_rows_by_key={"train0": 10, "train1": 10},
        feature_cols=["G", "bp_0"],
        scaler_center=[10.0, 0.0],
        scaler_scale=[2.0, 1.0],
    )
    suite = dict(
        reconstruction_source="original_sample",
        reconstruction_training_rows=12,
        reconstruction_presentations=24,
        reconstruction_original=str(source),
        reconstruction_cache=str(tmp_path / "cache"),
        reconstruction_validation_keys=["valid"],
        reconstruction_chunk_rows=4,
        checkpoint_sha256="a" * 64,
    )
    return (
        suite,
        {"run_signature": signature},
        signature["feature_cols"],
        [10.0, 0.0],
        [2.0, 1.0],
    )


def test_actual_staging_identity_counts_normalization_and_reuse(tmp_path):
    args = fixture(tmp_path)
    cache, counts, manifest = prepare_original_reconstruction(*args)
    assert cache.shape == (12, 2) and not cache.flags.writeable
    assert counts == {"train0": 6, "train1": 6}
    assert manifest["sample_source_ids_unique"]
    assert manifest["original_validation_ids_excluded"]
    ids = np.load(tmp_path / "cache/source_id.npy")
    assert len(np.unique(ids)) == 12 and not np.isin(ids, [21, 22, 23]).any()
    np.testing.assert_allclose(cache[:, 0], ids / 2)
    np.testing.assert_allclose(cache[:, 1], ids / 10)
    again, _, same = prepare_original_reconstruction(*args)
    np.testing.assert_array_equal(again, cache)
    assert manifest["request_sha256"] == same["request_sha256"]
    args[0]["reconstruction_sample_seed"] = 43
    with pytest.raises(ValueError, match="expected identity"):
        prepare_original_reconstruction(*args)


def test_source_count_change_and_partial_epoch_refused(tmp_path):
    args = fixture(tmp_path)
    args[0]["reconstruction_presentations"] = 13
    with pytest.raises(ValueError, match="complete declared"):
        prepare_original_reconstruction(*args)
    args[0]["reconstruction_presentations"] = 24
    args[1]["run_signature"]["train_rows_by_key"]["train0"] = 9
    with pytest.raises(ValueError, match="row count differs"):
        prepare_original_reconstruction(*args)


def test_validation_id_overlap_refused_before_cache_publication(tmp_path):
    args = fixture(tmp_path)
    with h5py.File(args[0]["reconstruction_original"], "r+") as handle:
        values = handle["valid"][:]
        values["source_id"] = [1, 2, 3]
        handle["valid"][:] = values
    args[0]["reconstruction_training_rows"] = 20
    args[0]["reconstruction_presentations"] = 40
    with pytest.raises(ValueError, match="overlaps original validation"):
        prepare_original_reconstruction(*args)
    assert not (tmp_path / "cache").exists()
    assert (tmp_path / "cache.building").exists()
