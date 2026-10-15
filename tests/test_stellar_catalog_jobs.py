import importlib.util
from pathlib import Path

import numpy as np
from astropy.io import fits

spec = importlib.util.spec_from_file_location(
    "catalog_jobs", Path(__file__).parents[1] / "scripts/stellar_catalog_jobs.py"
)
jobs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jobs)


def test_native_apogee_never_uses_m_h_as_iron_and_requires_flags():
    table = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="GAIAEDR3_SOURCE_ID", format="K", array=[1, 2]),
            fits.Column(name="M_H", format="D", array=[-3.0, -2.0]),
            fits.Column(name="FE_H", format="D", array=[-2.5, -1.8]),
            fits.Column(name="FE_H_ERR", format="D", array=[0.1, 0.1]),
            fits.Column(name="ASPCAPFLAG", format="K", array=[0, 1]),
            fits.Column(name="STARFLAG", format="K", array=[0, 0]),
        ]
    ).data
    _, value, _, eligible = jobs.selected_native(table, "apogee")
    np.testing.assert_allclose(value, [-2.5, -1.8])
    assert eligible.tolist() == [True, False]
    _, value, _, eligible = jobs.selected_native(
        table[["M_H"]]
        if False
        else fits.BinTableHDU.from_columns(
            [fits.Column(name="source_id", format="K", array=[1])]
        ).data,
        "galah",
    )
    assert not eligible.any()


def test_duplicate_source_ids_have_identical_split():
    ids = np.array([123, 456, 123, 789], dtype="i8")
    assert jobs.identity_split(ids)[0] == jobs.identity_split(ids)[2]


def test_registry_publication_preserves_mh_and_excludes_unflagged_galah(tmp_path):
    import argparse

    ids = [100, 200, 300]
    columns = [
        fits.Column(name="source_id", format="K", array=ids),
        fits.Column(name="fe_h", format="D", array=[-1.0, -2.0, -3.0]),
        fits.Column(name="e_fe_h", format="D", array=[0.1, 0.1, 0.1]),
        fits.Column(
            name="spec_source",
            format="20A",
            array=["galah", "apogee-latentage", "li_et_al_vmps"],
        ),
    ]
    legacy = tmp_path / "legacy.fits"
    fits.BinTableHDU.from_columns(columns).writeto(legacy)
    output = tmp_path / "out"
    output.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    result = jobs.build_registry(
        argparse.Namespace(legacy=legacy, output=output), [], scratch
    )
    import h5py

    with h5py.File(output / "labels.h5") as f:
        assert f["measurements"]["quantity"].tolist() == [b"FeH", b"MH", b"FeH"]
        assert f["selected/labels"]["source_id"].tolist() == [300]
    assert result["selected_FeH_stars"] == 1
