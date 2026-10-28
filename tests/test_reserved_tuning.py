"""Role selection never sees labels; parent inputs/scalers stay fixed."""

import json

import h5py
import numpy as np
import pytest

from masked_stellar_autoencoder.training.reserved_tuning import (
    evaluate_reserved_predictions,
    load_reserved_tuning,
    reserved_native_rows,
)


def split(ids):
    return np.asarray(ids) % 3


def test_selection_is_row_order_independent_and_excludes_other_roles_partition():
    ids = np.arange(100, 160, dtype="i8")
    rows = np.arange(len(ids))
    available = np.ones(len(ids), bool)
    available[0] = False
    partition = np.zeros(len(ids), int)
    partition[3] = 1
    a = reserved_native_rows(
        ids, rows, available, partition, identity_split=split, size=8
    )
    reverse = rows[::-1]
    b = reserved_native_rows(
        ids, reverse, available, partition, identity_split=split, size=8
    )
    np.testing.assert_array_equal(ids[rows[a]], ids[reverse[b]])
    assert (split(ids[a]) == 1).all()
    assert available[a].all() and not partition[a].any()
    with pytest.raises(ValueError, match="require 4096"):
        reserved_native_rows(ids, rows, available, partition, identity_split=split)


def test_loader_is_label_blind_preserves_scaler_policy_and_source_pairing(tmp_path):
    path = tmp_path / "cohort.h5"
    ids = np.arange(100, 160, dtype="i8")
    raw = np.arange(len(ids) * 3, dtype="f4").reshape(-1, 3)
    raw[0, 0] = np.nan
    records = np.array(
        list(zip(ids, ids / 100)), dtype=[("source_id", "i8"), ("value", "f8")]
    )
    with h5py.File(path, "w") as f:
        for name, value in {
            "features/source_id": ids,
            "features/values": raw,
            "features/feature_available": np.ones(len(ids), bool),
            "features/benchmark_partition": np.zeros(len(ids), "u1"),
            "features/label_split": split(ids),
            "layer_index/native/feature_row": np.arange(len(ids))[::-1],
            "raw_layers/native/measurements": records[::-1],
            "reference_scaler/center": np.array([10, 20, 30]),
            "reference_scaler/scale": np.array([2, 3, 4]),
        }.items():
            f.create_dataset(name, data=value)
        f.attrs["feature_columns_json"] = json.dumps(["G", "RA", "EBV"])
    a = load_reserved_tuning(path, identity_split=split, size=8)
    with h5py.File(path, "r+") as f:
        changed = f["raw_layers/native/measurements"][:]
        changed["value"] = -100
        f["raw_layers/native/measurements"][:] = changed
    b = load_reserved_tuning(path, identity_split=split, size=8)
    np.testing.assert_array_equal(a["source_id"], b["source_id"])
    np.testing.assert_array_equal(a["safe"], b["safe"])
    expected = (a["raw_values"] - a["center"]) / a["scale"]
    np.testing.assert_array_equal(a["safe"][:, 1], -9999)
    np.testing.assert_allclose(a["safe"][:, [0, 2]], expected[:, [0, 2]])
    metrics = evaluate_reserved_predictions(
        a, a["measurements"]["value"] + 0.2, a["source_id"]
    )
    assert metrics["mae_dex"] == pytest.approx(0.2)
    assert not metrics["label_scaler_refitted"]
    with pytest.raises(AssertionError):
        evaluate_reserved_predictions(
            a, a["measurements"]["value"], a["source_id"][::-1]
        )
