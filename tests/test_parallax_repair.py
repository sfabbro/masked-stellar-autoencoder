import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "repair",
    Path(__file__).resolve().parents[1] / "scripts/repair_expanded_parallax.py",
)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def test_exact_large_integer_ids_and_duplicate_rejection():
    ids = np.array([6000000000000000001, 6000000000000000003], dtype="i8")
    block = np.zeros(
        3, dtype=[("source_id", "i8"), ("PARALLAX", "f8"), ("e_parallax", "f8")]
    )
    block["source_id"] = [ids[1], ids[0] + 1, ids[0]]
    block["PARALLAX"] = [-0.3, 9.0, 0.0]
    pos, records = repair.match_block(ids, block)
    np.testing.assert_array_equal(pos, [1, 0])
    np.testing.assert_array_equal(records["PARALLAX"], [-0.3, 0.0])
    block["source_id"][1] = ids[0]
    with pytest.raises(ValueError, match="Repeated"):
        repair.match_block(ids, block)


def test_unchanged_audit_catches_label_mutation(tmp_path):
    paths = [tmp_path / "parent.h5", tmp_path / "child.h5"]
    for path in paths:
        with h5py.File(path, "w") as f:
            f["role"] = np.array([b"fit", b"devtest"])
            f["labels/value"] = np.array([np.nan, 3.0])
            f["measured_parallax/input_value"] = [0.0, 1.0]
    with h5py.File(paths[0]) as a, h5py.File(paths[1], "r+") as b:
        b["measured_parallax/input_value"][0] = -9999
        assert repair.verify_unchanged(a, b) == ["labels/value", "role"]
        b["labels/value"][1] = 4
        with pytest.raises(AssertionError):
            repair.verify_unchanged(a, b)
