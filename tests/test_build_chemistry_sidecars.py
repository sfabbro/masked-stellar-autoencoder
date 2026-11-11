import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

from masked_stellar_autoencoder.training.multisurvey_overlay import (
    write_observation_sidecar,
)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "chem", SCRIPTS / "build_chemistry_sidecars.py"
)
chem = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chem)


def test_gspspec_declares_mh_and_alpha_and_keeps_low_extrapolation(tmp_path):
    flags = [
        "0" * 9 + "0" + "0" * 31,
        "0" * 9 + "2" + "0" * 31,
        "0" * 9 + "1" + "0" * 31,
    ]
    cols = [
        fits.Column("source_id", "K", array=[1, 2, 3]),
        fits.Column("mh_gspspec", "D", array=[-1.0, -2.0, -0.5]),
        fits.Column("mh_gspspec_lower", "D", array=[-1.1, -2.2, -0.6]),
        fits.Column("mh_gspspec_upper", "D", array=[-0.9, -1.8, -0.4]),
        fits.Column("alphafe_gspspec", "D", array=[0.3, 0.4, np.nan]),
        fits.Column("alphafe_gspspec_lower", "D", array=[0.25, 0.3, np.nan]),
        fits.Column("alphafe_gspspec_upper", "D", array=[0.35, 0.5, np.nan]),
        fits.Column("flags_gspspec", "41A", array=flags),
    ]
    path = tmp_path / "gsp.fits"
    fits.BinTableHDU.from_columns(cols).writeto(path)
    block, n = chem.gspspec(str(path))
    assert n == 3 and block["source_id"].tolist() == [1, 3]  # star 2 extrapolated
    np.testing.assert_allclose(block["chem_error"][0], [0.1, 0.05])
    assert block["chem_observed"][1].tolist() == [True, False]  # star 3 has no alpha
    k = len(block["source_id"])
    block.update(
        value=np.full((k, 3), np.nan),
        error=np.full((k, 3), np.nan),
        observed=np.zeros((k, 3), bool),
    )
    registry = {
        "gsp": dict(
            release="r",
            pipeline="p",
            dependency_id="d",
            targets=dict(teff=None, logg=None, fe_h=None),
            chem_quantities={"fe_h": "m_h", "alpha": "alpha_fe"},
        )
    }
    receipt = write_observation_sidecar(
        tmp_path / "s.h5", {"gsp": block}, registry, provenance=dict(test=True)
    )
    with h5py.File(receipt["path"]) as f:
        g = f["native_observations/gsp"]
        assert g["chem_value"].shape == (2, 2) and not g["observed"][:].any()
