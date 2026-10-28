"""Gaia GSP-Spec partial-label and quality policy checks on synthetic data."""

import hashlib

import h5py
import numpy as np
import pytest
from astropy.io import fits

from masked_stellar_autoencoder.training.multisurvey_gaia_gspspec import (
    build_gaia_sidecar,
    quality_masks,
    summarize_gaia_roles,
)


def flag(*changes):
    s = list("0" * 41)
    for pos, value in changes:
        s[pos] = value
    return "".join(s)


def test_partial_labels_and_quality_flags_are_per_target():
    t = np.array([5000, np.nan, 5000, 909, 5000, 5000, 5000, 5000], dtype=float)
    g = np.array([4, 3, np.nan, 5.71, 4, 4, 4, 4], dtype=float)
    f = np.array(
        [
            flag(),
            flag(),
            flag(),
            flag(),
            flag((1, "2")),
            flag((6, "4")),
            flag((8, "1")),
            flag((10, "9")),
        ]
    )
    result = quality_masks(t, g, f)
    assert result["selected"].tolist() == [
        [True, True, False],
        [False, True, False],
        [True, False, False],
        [False, False, False],
        [True, False, False],
        [False, False, False],
        [True, True, False],
        [False, False, False],
    ]
    assert result["raw"][3].tolist() == [True, True, False]
    assert result["flags_ascii"].shape == (8, 41)


def test_missing_or_malformed_flags_do_not_become_good_quality():
    r = quality_masks([5000, 5000], [4, 4], np.array(["000", "00000000000X"]))
    assert not r["selected"].any()
    with pytest.raises(ValueError, match="aligned"):
        quality_masks([5000], [4, 4], np.array(["0" * 41]))
    with pytest.raises(ValueError, match="exceed"):
        quality_masks([5000], [4], np.array(["0" * 42]))


def test_sidecar_keeps_raw_quantiles_and_never_renames_mh_to_feh(tmp_path):
    path = tmp_path / "gaia.fits"
    cols = [
        fits.Column(name="source_id", format="K", array=np.array([10, 20], dtype="i8"))
    ]
    values = {
        "teff_gspspec": [5000, 909],
        "teff_gspspec_lower": [4900, 800],
        "teff_gspspec_upper": [5100, 1000],
        "logg_gspspec": [4, 5.71],
        "logg_gspspec_lower": [3.9, 5.6],
        "logg_gspspec_upper": [4.1, 5.8],
        "mh_gspspec": [-0.5, 0.1],
        "mh_gspspec_lower": [-0.6, 0],
        "mh_gspspec_upper": [-0.4, 0.2],
        "alphafe_gspspec": [0.2, 0.1],
        "alphafe_gspspec_lower": [0.1, 0],
        "alphafe_gspspec_upper": [0.3, 0.2],
    }
    cols.extend(
        fits.Column(name=k, format="E", array=np.array(v, dtype="f4"))
        for k, v in values.items()
    )
    cols.append(
        fits.Column(
            name="flags_gspspec",
            format="41A",
            array=np.array([flag(), flag()], dtype="S41"),
        )
    )
    fits.BinTableHDU.from_columns(cols).writeto(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = build_gaia_sidecar(path, sha, tmp_path / "sidecar.h5", expected_rows=2)
    assert (
        receipt["counts"]["selected_teff"] == 1
        and receipt["counts"]["selected_logg"] == 1
    )
    assert (
        receipt["counts"]["raw_teff"] == 2 and receipt["counts"]["selected_fe_h"] == 0
    )
    with h5py.File(tmp_path / "sidecar.h5") as h:
        g = h["native_observations/gaia_dr3_gspspec"]
        assert g["observed"][:].tolist() == [[True, True, False], [False, False, False]]
        assert np.isnan(g["value"][:, 2]).all() and np.isnan(g["error"][:]).all()
        assert np.isclose(g["mh_gspspec"][0], -0.5)
        assert g["percentile16"][0, 0] == 4900 and g["percentile84"][0, 0] == 5100
        assert g["value"][1, 0] == 909 and g["raw_observed"][1, 0]
    base = tmp_path / "base.h5"
    with h5py.File(base, "w") as h:
        h["source_id"] = np.array([10, 30], dtype="i8")
        h["role"] = np.array([b"fit", b"tune"])
        h["primary_input_eligible"] = np.array([True, True])
    roles = summarize_gaia_roles(base, tmp_path / "sidecar.h5")
    assert roles["matched_base_ids"] == 1 and roles["outside_base_ids"] == 1
    assert roles["roles"]["fit"]["teff"] == 1
    assert roles["roles"]["fit"]["logg"] == 1
    assert roles["roles"]["fit"]["mh_source_specific"] == 1
    assert roles["roles"]["fit"]["alpha_source_specific"] == 1
    assert roles["roles"]["fit"]["fe_h_direct"] == 0
    with pytest.raises(ValueError, match="content"):
        build_gaia_sidecar(path, "0" * 64, tmp_path / "bad.h5")


def test_abundance_masks_keep_mh_and_alpha_distinct_from_feh():
    flags = np.array([flag(), flag((12, "1")), flag((2, "2"))])
    masks = quality_masks(
        [5000, 5000, 5000],
        [4, 4, 4],
        flags,
        mh=[-0.4, -0.5, -0.6],
        alpha=[0.2, 0.3, 0.4],
    )
    assert masks["selected_mh"].tolist() == [True, True, False]
    assert masks["selected_alpha"].tolist() == [True, False, False]
    assert masks["raw_alpha"].tolist() == [True, True, True]
    assert not masks["selected"][:, 2].any()


def test_missing_temperature_sentinel_keeps_other_partial_labels():
    masks = quality_masks(
        [-9999, np.nan, 909, np.inf],
        [3, 3, 3, 3],
        np.array([flag(), flag(), flag(), flag()]),
        mh=[-0.5, -0.5, -0.5, -0.5],
        alpha=[0.2, 0.2, 0.2, 0.2],
    )
    assert masks["selected"][:, 1].tolist() == [True, True, False, False]
    assert masks["selected_mh"].tolist() == [True, True, False, False]
    assert masks["selected_alpha"].tolist() == [True, True, False, False]
