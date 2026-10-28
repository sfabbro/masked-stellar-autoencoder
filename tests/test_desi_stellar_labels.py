import hashlib
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "desi_labels", Path(__file__).parents[1] / "scripts/desi_stellar_labels.py"
)
desi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(desi)


def test_published_correction_fixed_points_bounds_and_derivative():
    value, valid, _ = desi.corrected_feh(
        [-2] * 5, [5000, 5000, 6600, 5000, 0], [2.5, 4.3, 2.5, 1.79, 2.5]
    )
    np.testing.assert_allclose(value[:2], [-2.197, -2.039])
    assert valid.tolist() == [True, True, False, False, False]
    assert np.isnan(value[2:]).all()
    _, _, derivative = desi.corrected_feh(-2, 5300, 2.5)
    plus = desi.corrected_feh(-2, 5300.01, 2.5)[0]
    minus = desi.corrected_feh(-2, 5299.99, 2.5)[0]
    np.testing.assert_allclose(derivative, (plus - minus) / 0.02, rtol=1e-8)
    np.testing.assert_allclose(
        desi.separation_arcsec(359.999, 0, 0.001, 0), 7.2, rtol=1e-8
    )


def test_desi_ingestion_validates_alignment_and_preserves_rejected_nan_rows(tmp_path):
    n = 5
    rv = np.zeros(
        n,
        dtype=[
            ("TARGETID", "i8"),
            ("TARGET_RA", "f8"),
            ("TARGET_DEC", "f8"),
            ("FEH", "f8"),
            ("TEFF", "f8"),
            ("LOGG", "f8"),
            ("FEH_ERR", "f8"),
            ("TEFF_ERR", "f8"),
            ("LOGG_ERR", "f8"),
            ("SUCCESS", "?"),
            ("PRIMARY", "?"),
            ("RVS_WARN", "i8"),
            ("RR_SPECTYPE", "S6"),
            ("SN_R", "f8"),
        ],
    )
    rv["TARGETID"] = np.arange(n) + 10
    rv["TARGET_RA"] = 30
    rv["TARGET_DEC"] = 5
    rv["FEH"] = -2.5
    rv["TEFF"] = 5000
    rv["LOGG"] = [2.5, 2.5, 0.5, 4.5, 3]
    rv["FEH_ERR"] = [0.1, 0.2, 0.2, 0.1, 0.3]
    rv["TEFF_ERR"] = 50
    rv["LOGG_ERR"] = 0.1
    rv["SUCCESS"] = rv["PRIMARY"] = True
    rv["RVS_WARN"][3] = 1
    rv["RR_SPECTYPE"] = b"STAR"
    rv["SN_R"] = 20
    gaia = np.zeros(n, dtype=[("SOURCE_ID", "i8"), ("RA", "f8"), ("DEC", "f8")])
    gaia["SOURCE_ID"] = [11, 11, 22, 33, 44]
    gaia["RA"], gaia["DEC"] = 30, 5
    gaia["RA"][-1] += 0.01
    other = np.zeros(n, dtype=[("TARGETID", "i8")])
    other["TARGETID"] = rv["TARGETID"]
    bad = other.copy()
    bad["TARGETID"] = bad["TARGETID"][::-1]
    with pytest.raises(ValueError, match="alignment"):
        desi.check_target_alignment(rv, bad)
    root = tmp_path / "raw"
    root.mkdir()
    source = root / "mwsall-pix-iron.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU(rv, name="RVTAB"),
            fits.BinTableHDU(other, name="SPTAB"),
            fits.BinTableHDU(other, name="FIBERMAP"),
            fits.BinTableHDU(gaia, name="GAIA"),
        ]
    ).writeto(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    (root / "desi-publisher.sha256sum").write_text(digest + "  mwsall-pix-iron.fits\n")
    output, scratch = tmp_path / "out", tmp_path / "scratch"
    output.mkdir()
    scratch.mkdir()
    result = desi.build_labels(output, scratch, root, digest)
    assert result["counts"]["eligible"] == 2
    assert result["counts"]["selected_unique_stars"] == 1
    with h5py.File(output / "desi-labels.h5") as f:
        assert len(f["measurements"]) == 5
        assert np.isnan(f["measurements"][2]["feh_corrected"])
        assert f["selected/measurements"]["source_id"].tolist() == [11]
        np.testing.assert_allclose(
            f["selected/measurements"]["feh_corrected"], [-2.697]
        )
        np.testing.assert_allclose(f["selected/measurements"]["feh_err_native"], [0.1])
        assert "NOT included" in f.attrs["uncertainty_definition"]
