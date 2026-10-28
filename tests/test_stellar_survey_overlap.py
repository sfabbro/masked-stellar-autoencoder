import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from stellar_survey_overlap import DTYPE, paired, statistics


def test_pairing_uses_exact_ids_and_best_error_without_reordering_values():
    left = np.array(
        [(9, -2.1, 0.2, 10), (2, -1.0, 0.1, 11), (9, -2.2, 0.1, 12)], dtype=DTYPE
    )
    right = np.array(
        [(9, -2.3, 0.1, 30), (7, -3.0, 0.1, 31), (2, -1.1, 0.2, 32)], dtype=DTYPE
    )
    matched_left, r, split = paired(left, right)
    assert matched_left["source_id"].tolist() == [2, 9]
    assert matched_left["source_row"].tolist() == [11, 12]
    assert r["source_row"].tolist() == [32, 30]
    np.testing.assert_allclose(matched_left["value"] - r["value"], [0.1, 0.1])
    assert len(split) == 2
    a, b, s = paired(left, np.empty(0, dtype=DTYPE))
    assert len(a) == len(b) == len(s) == 0


def test_summary_known_offset_and_empty_sample():
    d = statistics(np.array([0.1, 0.1, 0.1]), np.array([0.2, 0.2, 0.2]))
    assert d["n"] == 3 and d["median_delta_dex"] == 0.1
    assert d["robust_scatter_dex"] == 0
    assert d["fraction_abs_delta_gt_3_reported_sigma"] == 0
    assert statistics(np.array([]), np.array([])) == {"n": 0}


def test_read_only_audit_preserves_pairs_and_input_provenance(tmp_path):
    import json

    import h5py
    from stellar_survey_overlap import build_audit

    root = tmp_path / "inputs"
    paths = [
        root / "label_registry_v2/labels.h5",
        root / "pigs_labels_v1/pigs-labels.h5",
        root / "desi_labels_v1/desi-labels.h5",
    ]
    for path in paths:
        path.parent.mkdir(parents=True)
    native = np.array(
        [
            (9, -2.0, 0.1, b"galah", True, b"FeH"),
            (9, -2.2, 0.2, b"apogee", True, b"FeH"),
        ],
        dtype=[
            ("source_id", "i8"),
            ("value", "f8"),
            ("uncertainty", "f8"),
            ("survey", "S32"),
            ("quality_pass", "?"),
            ("quantity", "S8"),
        ],
    )
    pigs = np.array(
        [(9, -2.1, 0.13, 17)],
        dtype=[
            ("source_id", "i8"),
            ("value", "f8"),
            ("uncertainty", "f8"),
            ("source_row", "i8"),
        ],
    )
    desi = np.array(
        [(9, -2.3, 0.15, 21)],
        dtype=[
            ("source_id", "i8"),
            ("feh_corrected", "f8"),
            ("feh_err_native", "f8"),
            ("source_row", "i8"),
        ],
    )
    for path, key, values in zip(
        paths,
        ["measurements", "selected/labels", "selected/measurements"],
        [native, pigs, desi],
    ):
        with h5py.File(path, "w") as f:
            f.create_dataset(key, data=values)
    out = tmp_path / "out"
    out.mkdir()
    result = build_audit(out, tmp_path, root)
    assert len(result["comparisons"]) == 6
    assert all(v["pairs"] == 1 for v in result["comparisons"].values())
    assert set(json.loads((out / "input-provenance.json").read_text())) == {
        "native",
        "PIGS",
        "DESI",
    }
    with np.load(out / "overlap-pairs.npz") as a:
        assert a["galah_minus_DESI__right_source_row"].tolist() == [21]
        np.testing.assert_allclose(a["galah_minus_DESI__right_feh"], [-2.3])
    with h5py.File(paths[0]) as f:
        np.testing.assert_array_equal(f["measurements"][:], native)
