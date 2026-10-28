"""Synthetic provenance/ID contracts only; no native models or datasets."""

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_five_job_source import (
    ROW_KEYS,
    ancestry_proof,
    fixed_corners,
    join_catalogue,
    publication_flag,
    unchanged_training,
    vac_projection,
    xp_evidence,
)

NAMES = ["G", "BP", "RP", "W1", "W2"] + [
    f"{arm}_{i}" for arm in ("bp", "rp") for i in range(1, 56)
]


def test_missing_is_not_genuine_no_xp_without_official_false():
    raw = np.full((5, len(NAMES)), np.nan)
    raw[3, 5] = 0.3
    raw[4, 5:] = 0.2
    complete, no_xp, status = xp_evidence(
        raw,
        NAMES,
        np.array([False, False, True, False, False]),
        np.array([True, False, True, True, True]),
    )
    assert no_xp.tolist() == [True, False, False, False, False]
    assert complete.tolist() == [False, False, False, False, True]
    assert status.tolist() == [
        "official_no_xp",
        "unknown",
        "xp_published_not_staged",
        "partial_xp",
        "complete_xp",
    ]


def test_unknown_publication_values_and_masks_are_not_false_evidence():
    flag, known = publication_flag(
        np.ma.array([0, 1, -1, 0], mask=[False, False, False, True])
    )
    assert flag.tolist() == [False, True, False, False]
    assert known.tolist() == [True, True, False, False]
    flag, known = publication_flag(np.array(["false", "TRUE", "missing"]))
    assert known.tolist() == [True, True, False]
    assert flag.tolist() == [False, True, False]


def test_Andrae_join_keeps_MH_proxy_and_known_training_membership():
    source = dict(
        source_id=np.array([30, 10]),
        other=np.array([[5300, 3.1, -0.8], [4500, 2.0, -1.4]]),
        heldout=np.array([False, True]),
        heldout_available=np.array(True),
    )
    joined = join_catalogue(np.array([10, 20, 30]), source)
    np.testing.assert_allclose(
        joined["andrae_prediction_native"][[0, 2]], source["other"][[1, 0]]
    )
    assert np.isnan(joined["andrae_prediction_native"][1]).all()
    assert joined["andrae_matched"].tolist() == [True, False, True]
    assert joined["andrae_in_training"].tolist() == [False, False, True]
    assert joined["andrae_training_membership_known"].tolist() == [True, False, True]


def minimal_rows():
    data = {key: np.zeros((4, 1)) for key in ROW_KEYS}
    data["role"] = np.array(["fit", "tune", "cal", "test"])
    return data


def test_original_training_rows_byte_identical_including_NaNs():
    base = minimal_rows()
    base["value"][1, 0] = np.nan
    expanded = {key: np.concatenate([value, value[-1:]]) for key, value in base.items()}
    assert len(unchanged_training(base, expanded)) == len(ROW_KEYS)
    expanded["value"][0, 0] = 1
    with pytest.raises(ValueError, match="changed"):
        unchanged_training(base, expanded)


def test_staged_Gaia_VAC_conflicting_duplicate_flags_are_unknown(tmp_path):
    columns = [
        fits.Column(name="source_id", format="K", array=np.array([11, 12, 12])),
        fits.Column(
            name="has_xp_continuous", format="L", array=np.array([False, False, True])
        ),
    ]
    for name in [
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "W1mag",
        "W2mag",
    ]:
        columns.append(
            fits.Column(name=name, format="D", array=np.array([12.0, 999.0, 999.0]))
        )
    for name in ["e_W1mag", "e_W2mag"]:
        columns.append(fits.Column(name=name, format="D", array=np.full(3, 0.03)))
    for prefix in ("phot_g", "phot_bp", "phot_rp"):
        columns.extend(
            [
                fits.Column(
                    name=prefix + "_mean_flux", format="D", array=np.full(3, 100.0)
                ),
                fits.Column(
                    name=prefix + "_mean_flux_error", format="D", array=np.full(3, 10.0)
                ),
            ]
        )
    path = tmp_path / "staged-synthetic-vac.fits"
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU.from_columns(columns)]).writeto(
        path
    )
    raw = np.full((2, len(NAMES)), np.nan)
    error = raw.copy()
    published, known, rows, _ = vac_projection(
        dict(path=path, bytes=path.stat().st_size, rows=3),
        np.array([11, 12]),
        raw,
        error,
        NAMES,
    )
    assert known.tolist() == [True, False]
    assert published.tolist() == [False, True]
    assert rows.tolist() == [0, 1]
    np.testing.assert_allclose(raw[0, :5], 12)
    assert np.isnan(raw[1]).all()
    assert error[0, 0] == pytest.approx(2.5 / np.log(10) * 0.1)


def test_corner_rows_and_axes_model_blind_and_empty_groups_explicit():
    data = dict(
        source_id=np.array([1, 2, 3, 4]),
        role=np.array(["fit", "fit", "test", "test"]),
        quality=np.ones((4, 6), bool),
        observed=np.ones((4, 6), bool),
        value=np.array(
            [
                [5000, 2, -0.5, 0.1, 5, 1],
                [6000, 3, 0, 0.2, 9, 2],
                [5100, 2.1, -2.4, 0.1, 6, 1],
                [5200, 2.2, -2.5, 0.1, 7, 1.2],
            ]
        ),
        real_xp_present=np.array([True, True, True, False]),
        real_no_xp=np.array([False, False, False, True]),
    )
    rows, limits, cells = fixed_corners(data)
    assert set(rows) == {2, 3}
    assert limits.shape == (2, 6, 2)
    assert np.isfinite(limits).all() and (limits[:, :, 1] > limits[:, :, 0]).all()
    assert cells[0]["source_id"] == [3]
    assert cells[1]["source_id"] == [4]
    assert cells[2]["empty"]


def test_direct_Feh_holdout_overlap_rejected_but_cross_target_flagged():
    data = dict(
        source_id=np.array([1, 2, 3]),
        role=np.array(["fit", "cal", "test"]),
        observed=np.ones((3, 6), bool),
        quality=np.ones((3, 6), bool),
    )
    data["quality"][1, 2] = False
    exposed, counts = ancestry_proof(data, [1, 2])
    assert exposed.tolist() == [True, True, False]
    assert counts["cal"][0] == 1
    assert counts["cal"][2] == 0
    with pytest.raises(ValueError, match="Direct FeH"):
        ancestry_proof(data, [3])
