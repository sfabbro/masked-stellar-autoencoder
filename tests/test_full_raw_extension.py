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


def test_new_config_source_records_join_by_exact_id_and_keep_three_columns():
    ids = np.array([11, 22, 33, 44], dtype="i8")
    sid = np.array([44, 11, 99, 22], dtype="i8")
    value = np.array(
        [[5000, 2, -2.5], [6000, 4, -0.2], [5500, 3, -1], [4800, 1.5, np.nan]]
    )
    error = np.array([[50, 0.1, 0.1], [np.nan] * 3, [60, 0.2, 0.2], [70, 0.2, np.nan]])
    observed = np.isfinite(value)
    observed[3] = [False, False, False]  # star 22 observes nothing -> dropped
    known = np.isfinite(error) & observed
    rows, arrays, outside = extension.native_source_records(
        ids, sid, value, error, observed, known
    )
    np.testing.assert_array_equal(rows, [0, 3])  # stars 11 and 44, sorted
    assert outside == 1  # star 99 is not in the source
    assert arrays["value"].shape == (2, 6)
    np.testing.assert_array_equal(arrays["value"][1, :3], [5000, 2, -2.5])
    assert not arrays["observed"][:, 3:].any() and not arrays["error_known"][0].any()


def test_new_config_source_records_reject_duplicate_stars():
    ids = np.array([11, 22], dtype="i8")
    value = np.array([[5000, 2, -1.0], [5100, 2, -1.1]])
    with pytest.raises(ValueError, match="two records"):
        extension.native_source_records(
            ids,
            np.array([11, 11]),
            value,
            np.full((2, 3), np.nan),
            np.ones((2, 3), bool),
            np.zeros((2, 3), bool),
        )


def test_xp_presence_is_read_from_input_sentinels(tmp_path):
    names = [f"{arm}_{j}" for arm in ("bp", "rp") for j in range(1, 56)] + ["G", "BP"]
    cols = extension.xp_columns(names)
    assert len(cols) == 110 and cols[-1] == 109
    data = np.full((3, len(names)), -9999.0, dtype="f4")
    data[:, 110] = 15.0  # G always present
    data[0, 5] = 0.3  # one XP coefficient present
    data[2, :110] = np.nan
    with h5py.File(tmp_path / "x.h5", "w") as f:
        f["input"] = data
        has = extension.input_has_any(f["input"], cols, chunk=2)
    np.testing.assert_array_equal(has, [True, False, False])


def test_native_records_accept_four_columns_for_declared_chemistry():
    ids = np.array([11, 22], dtype="i8")
    value = np.array([[np.nan, np.nan, -1.2, 0.3], [np.nan, np.nan, np.nan, 0.1]])
    error = np.array([[np.nan, np.nan, 0.1, 0.05], [np.nan, np.nan, np.nan, 0.05]])
    observed = np.isfinite(value)
    rows, arrays, outside = extension.native_source_records(
        ids, np.array([22, 11]), value, error, observed, observed
    )
    np.testing.assert_array_equal(rows, [0, 1])
    # rows are sorted by star: row 0 is star 11 (second input), row 1 is star 22
    np.testing.assert_array_equal(arrays["value"][1, 2:4], [-1.2, 0.3])
    assert arrays["observed"][0].tolist() == [False, False, False, True, False, False]
