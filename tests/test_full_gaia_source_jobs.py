import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "correction_sources",
    Path(__file__).parents[1] / "scripts/run_full_gaia_correction_sources.py",
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_join_keeps_exact_gaia_ids_and_no_xp_predicate():
    ids = np.array([100000000000000001, 100000000000000002], dtype="i8")
    rows, pos = m.match_positions(ids, np.array([ids[1], ids[0], -1, 20], dtype="i8"))
    np.testing.assert_array_equal(rows, [0, 1])
    np.testing.assert_array_equal(pos, [1, 0])
    import pytest

    with pytest.raises(ValueError):
        m.match_positions(ids, ids.astype(float))


def test_stage_preserves_native_age_and_alpha_definitions_and_integer_ids():
    a = np.array(
        [(100000000000000001, 3.5, 0.2)],
        dtype=[("source_id", ">i8"), ("Agergb", ">f4"), ("alpha_m", ">f4")],
    )
    frame = m.table_frame(a, list(a.dtype.names), 0, 1)
    assert frame.source_id.iloc[0] == 100000000000000001
    assert frame.Agergb.iloc[0] == 3.5
    assert m.native_fields(["Agergb", "e_Agerc", "alpha_m", "mg_fe", "source_id"]) == [
        "Agergb",
        "e_Agerc",
        "alpha_m",
        "mg_fe",
    ]


def test_age_stage_reads_ascii_fits_tables(tmp_path, monkeypatch):
    from astropy.io import fits
    import pyarrow.parquet as pq

    # Native AMBRE and LAMOST-Kepler releases use ASCII TABLE extensions.
    source = fits.TableHDU.from_columns([
        fits.Column(name="GaiaDR2", format="I20", array=[100000000000000001]),
        fits.Column(name="Age", format="F8.3", array=[3.5]),
        fits.Column(name="e_Age", format="F8.3", array=[0.2]),
    ])
    source.writeto(tmp_path / "ambre-ages.fits")
    monkeypatch.setattr(m, "CAT", tmp_path)
    output = tmp_path / "staged"
    output.mkdir()
    manifest = {"sources": []}
    m.stage_labels("ages", output, manifest, lambda: None)

    table = manifest["sources"][0]["tables"][0]
    assert table["selected_fields"] == ["Age", "e_Age"]
    assert table["training_admitted"] is False
    frame = pq.read_table(output / "000-01/part-0000000000.parquet").to_pandas()
    assert frame.GaiaDR2.iloc[0] == 100000000000000001
    assert frame.Age.iloc[0] == 3.5
    assert frame.source_row.tolist() == [0]
    assert table["hdu_type"] == "TableHDU"
    assert table["hdu_index"] == 1


def test_granulation_timescales_are_not_stellar_age_candidates():
    assert m.native_fields(["TAU_GRAN_SSI", "TAU_GRAN_SSI_PERR", "AGE_DW"]) == [
        "AGE_DW"
    ]


def test_ascii_missing_marker_preserves_nulls_and_exact_integer_ids(tmp_path, monkeypatch):
    from astropy.io import fits
    import pyarrow.parquet as pq

    path = tmp_path / "lamost-kepler-ages.fits"
    fits.TableHDU.from_columns([
        fits.Column(name="gaia_source_id", format="I20", array=[100000000000000001, 100000000000000002]),
        fits.Column(name="Agergb", format="F8.3", array=[3.5, 0.0]),
    ]).writeto(path)
    with fits.open(path) as hd:
        h = hd[1]
        position = h.fileinfo()["datLoc"] + h.header["NAXIS1"] + h.columns["Agergb"].start - 1
    with path.open("r+b") as stream:
        stream.seek(position)
        stream.write(b"     ---")  # Native LAMOST uses this without a TNULL card.
    monkeypatch.setattr(m, "CAT", tmp_path)
    output = tmp_path / "staged"
    output.mkdir()
    manifest = {"sources": []}
    m.stage_labels("ages", output, manifest, lambda: None)
    table = manifest["sources"][1]["tables"][0]
    assert table["finite_not_quality_selected"]["Agergb"] == 1
    assert table["ascii_null_policy"]["additional_numeric_null_tokens"] == ["---"]
    frame = pq.read_table(output / "001-01/part-0000000000.parquet").to_pandas()
    assert frame.gaia_source_id.tolist() == [100000000000000001, 100000000000000002]
    assert frame.Agergb.iloc[0] == 3.5
    assert np.isnan(frame.Agergb.iloc[1])
    assert frame.source_row.tolist() == [0, 1]


def test_raw_native_parent_retains_stars_outside_quality_and_feature_sidecars(tmp_path):
    import h5py
    from astropy.io import fits

    ids = np.arange(100000000000000001, 100000000000000005, dtype="i8")
    for product, raw in [("lrs", ids[[0, 1, 1]]), ("mrs", ids[[2]])]:
        directory = tmp_path / f"multisurvey_lamost_dr12_{product}_source_direct_v2"
        directory.mkdir()
        with h5py.File(directory / "source.h5", "w") as data:
            data["observations/source_id"] = raw
            # Selected features/labels deliberately omit native identities.
            data["selected/source_id"] = np.array([], dtype="i8")
    directory = tmp_path / "desi_collection_v1"
    directory.mkdir()
    fits.BinTableHDU.from_columns([
        fits.Column(name="SOURCE_ID", format="K", array=ids[[1, 3]]),
    ], name="GAIA").writeto(directory / "mwsall-pix-iron.fits")
    arrays, provenance = m.raw_native_candidate_ids(tmp_path)
    np.testing.assert_array_equal(np.unique(np.concatenate(arrays)), ids)
    assert sum(source["rows"] for source in provenance) == 6


def test_explicit_audited_parent_requires_exact_unique_positive_integer_ids(tmp_path):
    import pytest

    path = tmp_path / "parent.npy"
    ids = np.array([100000000000000001, 100000000000000002], dtype="i8")
    np.save(path, ids)
    actual, provenance = m.candidates(path)
    np.testing.assert_array_equal(actual, ids)
    assert len(provenance[0]["sha256"]) == 64
    for invalid in [ids.astype(float), ids[::-1], np.r_[ids, ids], np.array([0, 1])]:
        np.save(path, invalid)
        with pytest.raises(ValueError):
            m.candidates(path)
