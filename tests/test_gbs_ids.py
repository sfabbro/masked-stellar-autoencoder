"""GBS v3 Gaia DR3 ids: refusing float-rounded ids and recovering exact ones."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from masked_stellar_autoencoder.training.anchor_labels import (
    exact_gaia_ids,
    gaia_benchmarks,
)

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location(
        "fix_gbs_dr3_ids", ROOT / "scripts" / "fix_gbs_dr3_ids.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


F = _script()
RNG = np.random.default_rng(3)
# Gaia-like ids: HEALPix level-12 index in the high bits, small source number below
IDS = (RNG.integers(0, 12 * 4**12, 200) << 35) + RNG.integers(1, 2**20, 200)


def test_float_columns_are_refused():
    with pytest.raises(ValueError, match="float"):
        exact_gaia_ids(IDS.astype(np.float64), "ids")


def test_float_rounded_integer_ids_are_refused():
    rounded = IDS.astype(np.float64).astype(np.int64)
    assert not np.array_equal(rounded, IDS)
    with pytest.raises(ValueError, match="rounded"):
        exact_gaia_ids(pd.array(rounded, dtype="Int64"), "ids")


def test_exact_ids_pass_with_missing_values():
    ids = pd.array(list(IDS[:60]) + [None], dtype="Int64")
    out = exact_gaia_ids(ids, "ids")
    assert out.dtype == np.int64 and out[-1] == 0
    np.testing.assert_array_equal(out[:60], IDS[:60])


def _gbs_row(sid, ra=10.0, dec=-20.0, plx=50.0):
    return dict(
        Name="star", HD="HD1", GaiaDR3=sid, RA_ICRS=ra, DE_ICRS=dec, Plx=plx,
        Teff=5500, logg=4.4, **{"[Fe/H]": 0.0},
    )  # fmt: skip


def test_loader_reads_corrected_table_and_refuses_original(tmp_path):
    true = IDS[:60]
    rows = [_gbs_row(int(s)) for s in true.astype(np.float64).astype(np.int64)]
    original = pd.DataFrame(rows).astype({"GaiaDR3": "Int64"})
    original.to_parquet(tmp_path / "orig.parquet")
    with pytest.raises(ValueError, match="rounded"):
        gaia_benchmarks(tmp_path, path=tmp_path / "orig.parquet")
    fixed = original.drop(columns=["GaiaDR3"])
    fixed.insert(0, "source_id", pd.array(true, dtype="Int64"))
    fixed.to_parquet(tmp_path / "gaia_fgk_benchmarks_v3_dr3ids.parquet")
    out = gaia_benchmarks(tmp_path)  # finds the corrected table in root
    np.testing.assert_array_equal(np.sort(out.source_id), np.sort(true))
    assert set(out.role) == {"validation"}


def test_rounding_interval_is_exact():
    for sid in IDS[:20]:
        lo, hi = F.rounding_interval(int(sid))
        stored = int(np.float64(sid))
        assert lo <= sid <= hi
        # interior ids round to the stored float; the endpoints are ties
        # (round half to even), outside ids never do
        assert F.same_float([lo + 1, hi - 1, sid], stored).all()
        assert not F.same_float([lo - 1, hi + 1], stored).any()


def _fake_query(table):
    def query(adql):
        if "BETWEEN" in adql:
            lo, hi = (int(x) for x in adql.split("BETWEEN")[1].split("AND"))
            return table[(table.source_id >= lo) & (table.source_id <= hi)]
        return table  # cone: everything nearby

    return query


def _gaia(sid, ra, dec, plx, g, pmra=0.0, pmdec=0.0):
    return dict(
        source_id=sid, ra=ra, dec=dec, pmra=pmra, pmdec=pmdec, parallax=plx,
        phot_g_mean_mag=g,
    )  # fmt: skip


def test_superset_candidate_with_position_and_parallax_is_recovered():
    true = int(IDS[0])
    stored = int(np.float64(true))
    # the star moved 1"/yr in RA: its 2016 position is 16" from the J2000 one
    pm = 1000.0
    ra16 = 10.0 + pm * 16 / 3.6e6 / np.cos(np.radians(-20.0))
    gaia = pd.DataFrame([_gaia(true, ra16, -20.0, 50.2, 5.1, pmra=pm)])
    out = F.resolve_row(pd.Series(_gbs_row(stored)), query=_fake_query(gaia))
    assert out["id_method"] == "superset" and out["source_id"] == true
    assert out["sep_arcsec"] < 0.1


def test_wrong_star_in_superset_falls_back_to_cone():
    true, wrong = int(IDS[1]), int(IDS[2])
    stored = int(np.float64(wrong))
    lo, hi = F.rounding_interval(stored)
    gaia = pd.DataFrame(
        [
            _gaia(wrong, 10.0, -20.0005, 1.0, 11.2),  # faint neighbour, 1.8" away
            _gaia(true, 10.0, -20.0, 49.8, 4.9),  # the benchmark, id outside range
        ]
    )
    assert not lo <= true <= hi
    out = F.resolve_row(pd.Series(_gbs_row(stored)), query=_fake_query(gaia))
    assert out["id_method"] == "cone" and out["source_id"] == true


def test_row_without_parallax_or_match_stays_unmatched():
    gaia = pd.DataFrame([_gaia(int(IDS[3]), 50.0, 50.0, 10.0, 12.0)])
    row = _gbs_row(None, plx=np.nan)
    out = F.resolve_row(pd.Series(row), query=_fake_query(gaia))
    assert out["id_method"] == "unmatched" and out["source_id"] is None


def test_single_two_parameter_candidate_is_accepted_within_cone():
    true = int(IDS[4])
    stored = int(np.float64(true))
    gaia = pd.DataFrame([_gaia(true, 10.0, -20.0016, np.nan, 2.5, np.nan, np.nan)])
    out = F.resolve_row(pd.Series(_gbs_row(stored)), query=_fake_query(gaia))
    assert out["id_method"] == "superset_2p" and out["source_id"] == true


def test_fix_table_keeps_exact_ids_next_to_unmatched_rows():
    true = [int(IDS[5]), int(IDS[6])]
    gaia = pd.DataFrame(
        [_gaia(true[0], 10.0, -20.0, 50.0, 5.0), _gaia(true[1], 30.0, 5.0, 20.0, 6.0)]
    )
    t = pd.DataFrame(
        [
            _gbs_row(int(np.float64(true[0]))),
            _gbs_row(int(np.float64(true[1])), ra=30.0, dec=5.0, plx=20.0),
            _gbs_row(None, ra=80.0, dec=80.0, plx=np.nan),  # unmatched: a null id
        ]
    ).astype({"GaiaDR3": "Int64"})
    fixed = F.fix_table(t, query=_fake_query(gaia))
    assert str(fixed["source_id"].dtype) == "Int64"
    assert fixed["source_id"].tolist()[:2] == true  # not rounded through float64
    assert pd.isna(fixed["source_id"].iloc[2])
    s = F.summary(fixed)
    assert s["recovered"] == 2 and s["unmatched_names"] == ["star"]
