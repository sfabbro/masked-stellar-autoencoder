import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_alpha_source_evidence import alpha_eligibility


def test_signed_alpha_and_error_quality_policy():
    mask = alpha_eligibility(
        np.array([2**60 + 1, 2, 3, 4, 5, 6]),
        np.array([-0.4, 0.1, 0.2, np.nan, 0.2, 0.3]),
        np.array([0.01, 0, 0.02, 0.03, 0.02, 0.02]),
        np.zeros(6),
        np.array([0, 0, 1, 0, 0, 0]),
        np.ones(6, bool),
        np.ones(6, bool),
        np.array([10, 20, 20, 20, 9, 10]),
        np.array([1, 1, 1, 1, 1, 0], bool),
    )
    np.testing.assert_array_equal(mask, [True, False, False, False, False, False])


def test_actual_fits_projection_and_raw_sha_gate(tmp_path):
    import hashlib
    import json

    import h5py
    import pytest
    from astropy.io import fits
    from prepare_alpha_source_evidence import prepare

    source_id = 2**60 + 1
    values = {
        "TARGETID": np.array([21], dtype="i8"),
        "ALPHAFE": np.array([-0.3]),
        "ALPHAFE_ERR": np.array([0.03]),
        "FEH": np.array([-2.2]),
        "FEH_ERR": np.array([0.1]),
        "TEFF": np.array([5000.0]),
        "LOGG": np.array([2.0]),
        "RVS_WARN": np.array([0], dtype="i8"),
        "SUCCESS": np.array([True]),
        "PRIMARY": np.array([True]),
        "SN_R": np.array([20.0]),
        "RR_SPECTYPE": np.array(["STAR"]),
        "TARGET_RA": np.array([10.0]),
        "TARGET_DEC": np.array([20.0]),
    }
    columns = [
        fits.Column(
            name=k,
            array=v,
            format="K"
            if v.dtype.kind in "iu"
            else "L"
            if v.dtype.kind == "b"
            else "4A"
            if v.dtype.kind == "U"
            else "D",
            unit="dex" if k.startswith("ALPHAFE") else None,
        )
        for k, v in values.items()
    ]
    gaia = [
        fits.Column(
            name="SOURCE_ID", array=np.array([source_id], dtype="i8"), format="K"
        ),
        fits.Column(name="RA", array=np.array([10.0]), format="D"),
        fits.Column(name="DEC", array=np.array([20.0]), format="D"),
    ]
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(columns, name="RVTAB"),
        fits.BinTableHDU.from_columns(gaia, name="GAIA"),
    ]
    hdus += [
        fits.BinTableHDU.from_columns(
            [fits.Column(name="TARGETID", array=values["TARGETID"], format="K")], name=n
        )
        for n in ("FIBERMAP", "SPTAB")
    ]
    raw = tmp_path / "raw.fits"
    fits.HDUList(hdus).writeto(raw)
    with pytest.raises(ValueError, match="SHA"):
        prepare(raw, tmp_path / "bad", expected_sha="0" * 64)
    assert not (tmp_path / "bad").exists()
    report = prepare(
        raw,
        tmp_path / "good",
        expected_sha=hashlib.sha256(raw.read_bytes()).hexdigest(),
    )
    assert report["counts"]["eligible_rows"] == 1
    with h5py.File(tmp_path / "good/desi-alpha-native.h5") as h5:
        assert int(h5["measurements"][0]["source_id"]) == source_id
        assert (
            json.loads(h5.attrs["original_rvtab_columns_json"])["ALPHAFE"]["unit"]
            == "dex"
        )
