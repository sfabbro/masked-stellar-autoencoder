import numpy as np
import pandas as pd

from masked_stellar_autoencoder.training.stopgap_labels import (
    choose_stars,
    select_labels,
)


def rec(**kw):
    base = dict(
        survey="lamost_dr12_lrs",
        tier="survey",
        source_id=1,
        fit_ok=True,
        teff=5000.0,
        e_teff=100.0,
        logg=4.0,
        e_logg=0.2,
        fe_h=0.0,
        e_fe_h=0.1,
        fe_h_kind="fe_h",
        fe_h_corrected=np.nan,
        alpha=np.nan,
        e_alpha=np.nan,
        alpha_kind="",
    )
    base.update(kw)
    return base


def test_priority_error_and_kinds():
    records = pd.DataFrame(
        [
            rec(source_id=1, teff=5100.0, fe_h=-0.5),
            rec(source_id=1, survey="apogee_dr17", teff=5050.0, fe_h=np.nan,
                alpha=0.1, e_alpha=0.02, alpha_kind="alpha_m"),
            rec(source_id=1, survey="hr", tier="highres_reference", teff=np.nan,
                fe_h=-0.7, e_fe_h=0.05),
            rec(source_id=2, survey="gaia_dr3_gspspec", fe_h=-1.0, fe_h_kind="m_h"),
            rec(source_id=2, teff=6000.0, fe_h=-0.2, fit_ok=False),
            rec(source_id=3, survey="desi_dr1", fe_h=-1.2, fe_h_corrected=-1.1),
            rec(source_id=4, survey="desi_dr1_sp", fe_h=-3.49),
            rec(source_id=0, survey="x", fe_h=-3.0),
        ]
    )  # fmt: skip
    lab = select_labels(records)
    assert list(lab.index) == [1, 2, 3]
    assert lab.loc[1, "teff"] == 5050.0 and lab.loc[1, "src_teff"] == "apogee_dr17"
    assert lab.loc[1, "fe_h"] == -0.7 and lab.loc[1, "src_fe_h"] == "hr"
    assert lab.loc[1, "alpha"] == 0.1
    assert np.isnan(lab.loc[2, "fe_h"])  # [M/H] is not [Fe/H]; failed fit dropped
    assert lab.loc[2, "teff"] == 5000.0  # GSP-Spec teff kept
    assert lab.loc[3, "fe_h"] == -1.1  # published DESI correction


def test_choose_stars_keeps_priority_and_metal_poor():
    rng = np.random.default_rng(0)
    n = 100
    lab = pd.DataFrame(
        {
            "fe_h": rng.normal(-0.2, 0.3, n),
            **{f"prio_{p}": np.full(n, 3) for p in ("teff", "logg", "fe_h", "alpha")},
        },
        index=pd.Index(np.arange(n) + 10, name="source_id"),
    )
    lab.loc[10, "prio_teff"] = 1
    lab.loc[11, "fe_h"] = -2.5
    chosen = choose_stars(lab, 20)
    assert len(chosen) == 20 and {10, 11} <= set(chosen)
    assert list(chosen) == sorted(chosen)
