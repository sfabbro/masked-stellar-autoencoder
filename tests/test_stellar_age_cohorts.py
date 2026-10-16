import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from stellar_age_cohorts import cohort_masks, feature_index, quality_masks
from stellar_catalog_jobs import identity_split


def test_warnings_regime_and_asymmetric_uncertainty():
    age = np.tile([1, 2.5, 3, 3.5, 5.0], (6, 1))
    age[3, 1] = 0.5
    age[4, 4] = 2
    flag = np.array([b"SGB"] * 5 + [b"RGB"])
    empty = np.array([b""] * 6)
    warning = np.array([b"", b"WARN_diff_inout", b"", b"", b"", b""])
    inputs = np.array([b"PARALLAX", b"", b"WARN_input", b"", b"", b""])
    m = quality_masks(age, age / 2, flag, inputs, empty, warning)
    assert m["age_candidate"].tolist() == [True, False, False, False, False, False]
    assert m["inputs_include_parallax"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_all_reference_and_validation_ids_excluded_from_fit_and_tune():
    ids = np.r_[0, np.arange(2**60, 2**60 + 200, dtype="i8")]
    splits = identity_split(ids)
    train = ids[(splits == 0) & (ids > 0)][:5]
    validation = ids[(splits == 0) & (ids > 0)][5:8]
    reference = np.r_[train[:2], ids[(splits == 1) & (ids > 0)][:3]]
    m = cohort_masks(ids, np.ones(len(ids), dtype=bool), train, validation, reference)
    for key in ["strict_fit", "broader_fit", "tune"]:
        assert not m[key][np.isin(ids, reference)].any()
        assert not m[key][np.isin(ids, validation)].any()
        assert not m[key][0]
    assert m["strict_fit"].sum() == 3
    assert m["benchmark_partition"][0] == 254
    assert m["label_split"][0] == 255


def test_feature_rows_keep_original_first_and_reject_missing_xp(tmp_path):
    path = tmp_path / "features.fits"
    ids = np.array([2**60 + 3, 2**60 + 1, 2**60 + 3, 0, 2**60 + 7], dtype="i8")
    fits.BinTableHDU.from_columns(
        [
            fits.Column(name="source_id", format="K", array=ids),
            fits.Column(name="bp_1", format="E", array=[np.nan, 1, 2, 4, 3]),
            fits.Column(name="rp_1", format="E", array=[1, 2, 3, 4, -9999]),
        ]
    ).writeto(path)
    got, rows, available, dropped = feature_index(path, ["bp_1", "rp_1"])
    np.testing.assert_array_equal(got, ids[[0, 1, 4]])
    np.testing.assert_array_equal(rows, [0, 1, 4])
    assert available.tolist() == [False, True, False]
    assert dropped == 2
