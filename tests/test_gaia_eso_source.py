import importlib.util
import sys

import h5py
import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location(
    "gaia_eso_source", "scripts/gaia_eso_source.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_preserves_values_and_no_invented_identity_or_epoch(tmp_path):
    source = tmp_path / "s.fits"
    out = tmp_path / "s.h5"
    cols = []
    for name in m.REQUIRED:
        if name == "OBJECT":
            cols.append(
                fits.Column(name=name, format="16A", array=["star_a", "star_b"])
            )
        else:
            cols.append(
                fits.Column(
                    name=name,
                    format="D",
                    unit="deg" if name in ("RA", "DECLINATION") else None,
                    array=[1.0, 2.0],
                )
            )
    fits.BinTableHDU.from_columns(cols).writeto(source)
    result = m.produce(source, out, m.sha256_file(source), 2)
    assert result["counts"]["position_epoch"] == "unknown_per_row"
    with h5py.File(out) as f:
        assert f["raw/OBJECT"][:].tolist() == [b"star_a", b"star_b"]
        assert f["raw/RA"].attrs["fits_unit"] == "deg"
        assert "gaia_source_id" not in f["raw"]
        assert np.array_equal(f["raw/FEH"][:], [1, 2])
    with pytest.raises(ValueError):
        m.produce(source, tmp_path / "bad.h5", "0" * 64, 2)
