import importlib.util
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "loose", SCRIPTS / "build_loose_survey_sidecars.py"
)
loose = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loose)


def _lrs(path):
    cols = [
        fits.Column(
            "gaia_source_id", "20A", array=np.array(["11", "22", "22", "33", "44", ""])
        ),
        fits.Column("snrg", "D", array=[50.0, 12.0, 15.0, 8.0, 40.0, 60.0]),
        fits.Column("teff", "D", array=[5000.0, 5100, 5150, 5200, 5300, 5400]),
        fits.Column("logg", "D", array=[4.0, 2.5, 2.6, 3.0, 4.5, 4.4]),
        fits.Column("feh", "D", array=[-0.5, -2.4, -2.3, -1.0, 0.1, 0.0]),
        fits.Column("teff_err", "D", array=[50.0] * 6),
        fits.Column("logg_err", "D", array=[0.1] * 6),
        fits.Column("feh_err", "D", array=[0.1] * 6),
    ]
    fits.BinTableHDU.from_columns(cols).writeto(path)


def _desi(path):
    n = 4
    rv = fits.BinTableHDU.from_columns(
        [
            fits.Column("TARGETID", "K", array=np.arange(n)),
            fits.Column("TEFF", "D", array=[5000.0, 3800.0, 5200.0, 5500.0]),
            fits.Column("LOGG", "D", array=[4.5, 4.8, 2.0, 4.0]),
            fits.Column("FEH", "D", array=[-1.0, -0.2, -2.8, -0.5]),
            fits.Column("TEFF_ERR", "D", array=[50.0] * n),
            fits.Column("LOGG_ERR", "D", array=[0.2] * n),
            fits.Column("FEH_ERR", "D", array=[0.9, 0.1, 0.2, 0.1]),
            fits.Column("SUCCESS", "L", array=[True, True, True, False]),
            fits.Column("PRIMARY", "L", array=[True] * n),
            fits.Column("RVS_WARN", "K", array=[0, 0, 0, 0]),
            fits.Column("RR_SPECTYPE", "6A", array=["STAR"] * n),
            fits.Column("SN_R", "D", array=[3.0, 30.0, 20.0, 50.0]),
            fits.Column("TARGET_RA", "D", array=[10.0] * n),
            fits.Column("TARGET_DEC", "D", array=[0.0] * n),
        ],
        name="RVTAB",
    )
    gaia = fits.BinTableHDU.from_columns(
        [
            fits.Column("SOURCE_ID", "K", array=[101, 102, 103, 104]),
            fits.Column("RA", "D", array=[10.0] * n),
            fits.Column("DEC", "D", array=[0.0] * n),
        ],
        name="GAIA",
    )
    other = lambda name: fits.BinTableHDU.from_columns(  # noqa: E731
        [fits.Column("TARGETID", "K", array=np.arange(n))], name=name
    )
    fits.HDUList(
        [fits.PrimaryHDU(), rv, gaia, other("FIBERMAP"), other("SPTAB")]
    ).writeto(path)


def test_table_block_keeps_low_snr_converged_fits_one_per_star(tmp_path):
    path = tmp_path / "lrs.fits"
    _lrs(path)
    spec = dict(loose.TABLES[0], path=str(path))
    block, rows = loose.table_block(spec, snr_min=10.0)
    block = loose.one_per_star(block)
    assert rows == 6
    assert block["source_id"].tolist() == [
        11,
        22,
        44,
    ]  # 33 below S/N 10; blank id dropped
    assert block["value"][1, 2] == -2.3  # star 22: higher-S/N repeat kept
    kept, overlap = loose.drop_existing(block, np.array([11]))
    assert kept["source_id"].tolist() == [22, 44] and overlap == 1


def test_desi_split_by_correction_box_with_failed_fit_cuts_only(tmp_path):
    path = tmp_path / "desi.fits"
    _desi(path)
    inside, outside, rows = loose.desi_blocks(str(path))
    assert rows == 4
    # 101: S/N 3 and FEH_ERR 0.9 (dropped before) now kept, inside box -> corrected
    assert inside["source_id"].tolist() == [101, 103]
    assert inside["value"][0, 2] != -1.0  # correction applied
    # 102: Teff 3800 outside box -> raw [Fe/H]; 104: SUCCESS false -> dropped
    assert outside["source_id"].tolist() == [102]
    assert outside["value"][0, 2] == -0.2
