import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "extension",
    Path(__file__).resolve().parents[1] / "scripts/extend_full_raw_source.py",
)
extension = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extension)


def test_existing_roles_and_values_survive_sorted_union_and_new_references_are_external(
    tmp_path,
):
    old_ids = np.array([6000000000000000001, 6000000000000000005], dtype="i8")
    new_ids = np.array([6000000000000000003, 6000000000000000007], dtype="i8")
    old_roles = np.array([b"cal", b"devtest"], dtype="S8")
    ids, op, newpos, roles = extension.union_layout(
        old_ids, new_ids, old_roles, new_ids[:1]
    )
    np.testing.assert_array_equal(ids[op], old_ids)
    np.testing.assert_array_equal(roles[op], old_roles)
    assert roles[newpos[0]] == b"external"
    with h5py.File(tmp_path / "merge.h5", "w") as f:
        original = f.create_dataset(
            "original",
            data=np.array([[np.nan, -9999.0], [-0.3, 2.0]]),
            fillvalue=np.nan,
        )
        destination = f.create_dataset(
            "merged", shape=(4, 2), dtype="f8", fillvalue=np.nan
        )
        extension.aligned_copy(
            destination, original, op, newpos, np.array([[3.0, 4.0], [5.0, 6.0]])
        )
        np.testing.assert_array_equal(destination[op], original[:])
        np.testing.assert_array_equal(destination[newpos], [[3.0, 4.0], [5.0, 6.0]])
        labels = f.create_dataset("labels", shape=(4, 2), dtype="f8", fillvalue=np.nan)
        extension.aligned_copy(labels, original, op, newpos)
        assert np.isnan(labels[newpos]).all()
    with pytest.raises(ValueError, match="disjoint"):
        extension.union_layout(old_ids, old_ids, old_roles, new_ids)


def test_mixed_signed_unsigned_ids_cannot_silently_round_to_float():
    old = np.array([6000000000000000001], dtype="i8")
    new = np.array([6000000000000000003], dtype="u8")
    ids, op, np_, roles = extension.union_layout(
        old, new, np.array([b"devtest"]), np.array([], dtype="i8")
    )
    assert ids.dtype == np.dtype("i8") and len(np.unique(ids)) == 2
    assert int(ids[np_[0]]) == 6000000000000000003
    with pytest.raises(ValueError, match="integer"):
        extension.exact_ids(new.astype("f8"))
