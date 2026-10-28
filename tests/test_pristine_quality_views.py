import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from pristine_quality_views import exact_rows, quality_masks


def fixture():
    names = [
        "Flag",
        "CaHKmag0",
        "e_CaHKmag0",
        "Pvar",
        "RUWE",
        "Cstar",
        "s_Cstar",
        "E(B-V)",
        "[Fe/H]phPgm",
        "b_[Fe/H]phP",
        "[Fe/H]phP",
        "B_[Fe/H]phP",
        "MCfCaHKP",
    ]
    d = np.zeros(3, dtype=[(k, "f8") for k in names])
    for k, v in {
        "Flag": -1,
        "CaHKmag0": 17.0,
        "e_CaHKmag0": 0.03,
        "Pvar": 0.01,
        "RUWE": 1.0,
        "s_Cstar": 0.05,
        "E(B-V)": 0.1,
        "[Fe/H]phPgm": -2.5,
        "b_[Fe/H]phP": -2.7,
        "[Fe/H]phP": -2.5,
        "B_[Fe/H]phP": -2.3,
        "MCfCaHKP": 1.0,
    }.items():
        d[k] = v
    return d


def test_signed_flags_and_within_grid_fraction_not_inverted():
    d = fixture()
    d["Flag"][1] = 1
    d["MCfCaHKP"][2] = 0.2
    phot, weak, bits, names = quality_masks(d)
    assert phot.tolist() == [True, False, True]
    assert weak.tolist() == [True, False, False]
    assert bits[1] & (1 << names.index("point_source"))
    d["[Fe/H]phPgm"][0] = -4.2
    assert quality_masks(d)[0][0] and not quality_masks(d)[1][0]


def test_undefined_quality_and_bad_percentiles_reject_only_applicable_mask():
    d = fixture()
    d["Pvar"][0] = np.nan
    d["e_CaHKmag0"][1] = 0
    d["b_[Fe/H]phP"][2] = -2.1
    phot, weak, _, _ = quality_masks(d)
    assert phot.tolist() == [False, False, True]
    assert not weak.any()


def test_exact_id_mapping_keeps_original_rows_and_missing():
    np.testing.assert_array_equal(
        exact_rows([90, 3, 17], [17, 5, 90, 3, 0]), [2, -1, 0, 1, -1]
    )
    np.testing.assert_array_equal(exact_rows([], [1]), [-1])
    with pytest.raises(ValueError, match="unique"):
        exact_rows([3, 3], [3])


def test_gaia_ids_above_float_precision_remain_exact_with_mixed_integer_types():
    high = 6000000000000000001
    reference = np.array([high, high + 3], dtype="i8")
    query = np.array([high + 1, high + 3], dtype="u8")
    np.testing.assert_array_equal(exact_rows(reference, query), [-1, 1])
    with pytest.raises(ValueError, match="integer"):
        exact_rows(reference, np.array([float(high)]))
