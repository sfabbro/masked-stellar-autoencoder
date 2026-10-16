import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from pristine_async_fetch import verify_catalogue


def test_full_pristine_requires_real_schema_exact_rows_unique_ids(tmp_path):
    cols = [fits.Column(name="GaiaDR3", format="K", array=[11, 12])]
    for name in [
        "RAICRS",
        "DEICRS",
        "CaHKmag0",
        "e_CaHKmag0",
        "[Fe/H]phP",
        "b_[Fe/H]phP",
        "B_[Fe/H]phP",
        "Flag",
    ]:
        cols.append(fits.Column(name=name, format="D", array=np.ones(2)))
    path = tmp_path / "real.fits"
    table = fits.BinTableHDU.from_columns(cols)
    table.writeto(path)
    assert verify_catalogue(path, 2)["unique_positive_DR3_IDs"] == 2
    with pytest.raises(ValueError, match="row count"):
        verify_catalogue(path, 3)
    table.data["GaiaDR3"][1] = 11
    table.writeto(path, overwrite=True)
    with pytest.raises(ValueError, match="duplicated"):
        verify_catalogue(path, 2)
