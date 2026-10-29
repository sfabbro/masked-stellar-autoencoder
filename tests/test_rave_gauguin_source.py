import gzip
import importlib.util
import sys

import h5py
import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location(
    "rave_gauguin_source", "scripts/rave_gauguin_source.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def make_line(obs, fe, alpha):
    b = bytearray(b" " * 225)
    b[0 : len(obs)] = obs.encode()
    b[22:37] = f"{fe:15.8e}".encode()
    b[176:191] = f"{alpha:15.8e}".encode()
    return bytes(b) + b"\n"


def test_native_fields_missing_and_join(tmp_path):
    source = tmp_path / "g.gz"
    x = tmp_path / "x.fits"
    out = tmp_path / "out.h5"
    with gzip.open(source, "wb") as f:
        f.write(make_line("a", -2.5, 0.4) + make_line("b", -0.2, 0.1))
    fits.BinTableHDU.from_columns(
        [
            fits.Column(name="rave_obs_id", format="32A", array=["a", "b", "b"]),
            fits.Column(
                name="source_id",
                format="K",
                null=np.iinfo(np.int64).max,
                array=np.array([101, 102, 103], dtype=np.int64),
            ),
        ]
    ).writeto(x)
    r = m.produce(source, x, out, m.sha256_file(source), m.sha256_file(x), 2, 3)
    assert r["counts"]["identity_candidate_stars"] == 1
    assert r["counts"]["ambiguous_observation_key_rows"] == 1
    with h5py.File(out) as f:
        assert f["gauguin_raw/fe_h"][:].tolist() == pytest.approx([-2.5, -0.2])
        assert f["gauguin_raw/alpha_fe"][:].tolist() == pytest.approx([0.4, 0.1])
        assert np.isnan(f["gauguin_raw/fe_h_error"][:]).all()
        assert f["published_gaia_edr3_crossmatch/gaia_edr3_dr3_source_id"][
            :
        ].tolist() == [101, -1]
    with pytest.raises(ValueError):
        m.produce(source, x, tmp_path / "bad.h5", "0" * 64, m.sha256_file(x), 2, 3)


def test_truncated_input_rejected(tmp_path):
    source = tmp_path / "bad.gz"
    with gzip.open(source, "wb") as f:
        f.write(b"too short\n")
    with pytest.raises(ValueError):
        m.read_gauguin(source, 1)


def test_empty_input_handles_row_counts(tmp_path):
    source = tmp_path / "empty.gz"
    with gzip.open(source, "wb"):
        pass
    obs, values = m.read_gauguin(source, 0)
    assert obs.size == 0
    assert values.shape == (0, len(m.FIELDS))
    with pytest.raises(ValueError):
        m.read_gauguin(source, 1)
