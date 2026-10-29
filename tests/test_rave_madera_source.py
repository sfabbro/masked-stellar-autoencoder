import importlib.util
import sys

import h5py
import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location(
    "rave_madera_source", "scripts/rave_madera_source.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_independent_rows_and_ambiguous_keys(tmp_path):
    src = tmp_path / "m.fits"
    cross = tmp_path / "x.fits"
    out = tmp_path / "out.h5"
    cols = [
        fits.Column(name="rave_obs_id", format="32A", array=np.array(["a", "b", "c"]))
    ]
    for name in m.COLUMNS:
        cols.append(
            fits.Column(
                name=name,
                format="J" if name == "algo_conv_madera" else "E",
                array=np.array([1, 2, 3]),
            )
        )
    fits.BinTableHDU.from_columns(cols).writeto(src)
    fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="rave_obs_id", format="32A", array=np.array(["a", "b", "b"])
            ),
            fits.Column(
                name="source_id",
                format="K",
                null=np.iinfo(np.int64).max,
                array=np.array([101, 102, 103], dtype=np.int64),
            ),
        ]
    ).writeto(cross)
    result = m.produce(src, cross, out, m.sha256_file(src), m.sha256_file(cross), 3, 3)
    assert result["counts"]["identity_candidate_stars"] == 1
    assert result["counts"]["ambiguous_observation_key_rows"] == 1
    assert result["counts"]["missing_observation_key_rows"] == 1
    with h5py.File(out) as f:
        assert f["published_gaia_edr3_crossmatch/gaia_edr3_dr3_source_id"][
            :
        ].tolist() == [101, -1, -1]
        assert f["madera_raw/teff_cal_madera"][:].tolist() == [1, 2, 3]
        assert "MADERA" in f.attrs["scope"]
    with pytest.raises(FileExistsError):
        m.produce(src, cross, out, m.sha256_file(src), m.sha256_file(cross), 3, 3)
    with pytest.raises(ValueError):
        m.produce(src, cross, tmp_path / "bad.h5", "0" * 64, m.sha256_file(cross), 3, 3)
