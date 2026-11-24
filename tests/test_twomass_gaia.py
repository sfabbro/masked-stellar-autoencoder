"""Gaia-keyed 2MASS replacement (data/twomass_gaia.py, build_twomass_sidecar)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from masked_stellar_autoencoder.data.twomass_gaia import (
    apply_sidecar,
    ks_g_check,
    lookup,
    replace_columns,
)
from masked_stellar_autoencoder.training.finetune_data import ExtraColumns

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(7)


def _stars(n=400):
    sid = np.sort(RNG.choice(10**15, n, replace=False)) + 10**17
    g = RNG.uniform(8, 17, n)
    ks = g - RNG.uniform(0.5, 3.5, n)  # G - Ks colour of FGK/M stars
    return sid, g, ks


def _twomass_fits(path, sid, ks, *, qual=None, extra_rows=50, duplicate=None):
    """Synthetic twomass.fits: tmass_oid order, plus unrelated rows."""
    n = len(sid)
    other = np.arange(extra_rows) + 10**18
    ids = np.concatenate([sid, other])
    ksall = np.concatenate([ks, RNG.uniform(8, 15, extra_rows)])
    q = np.array(["AAA"] * (n + extra_rows))
    if qual is not None:
        q[: len(qual)] = qual
    t = Table()
    order = RNG.permutation(len(ids))  # not sorted by source_id
    t["source_id"] = ids[order]
    t["tmass_oid"] = np.arange(len(ids))
    t["ph_qual"] = q[order]
    t["j_m"] = (ksall + 0.8)[order].astype("f4")
    t["j_msigcom"] = np.full(len(ids), 0.02, "f4")
    t["h_m"] = (ksall + 0.1)[order].astype("f4")
    t["h_msigcom"] = np.full(len(ids), 0.03, "f4")
    t["ks_m"] = ksall[order].astype("f4")
    t["ks_msigcom"] = np.full(len(ids), 0.025, "f4")
    if duplicate is not None:
        dup = t[t["source_id"] == duplicate]
        dup["ks_m"] = 99.0
        t = Table(np.concatenate([t.as_array(), dup.as_array()]))
    t.write(path, overwrite=True)
    return path


def test_lookup_returns_values_aligned_with_requested_ids(tmp_path):
    sid, g, ks = _stars()
    path = _twomass_fits(tmp_path / "tm.fits", sid[:300], ks[:300], duplicate=sid[0])
    want = np.concatenate([sid[::-1], [0]])  # reversed, a star without 2MASS, id 0
    cols, info = lookup(want, path, chunk_rows=97)
    ks_out = cols["KS"]
    np.testing.assert_allclose(ks_out[:400][::-1][:300], ks[:300], rtol=1e-6)
    assert np.isnan(ks_out[:100]).all() and np.isnan(ks_out[-1])
    assert info["matched_ids"] == 300 and info["duplicate_rows"] == 1
    np.testing.assert_allclose(cols["E_KS"][100:400], 0.025)


def test_upper_limits_and_bad_quality_become_missing(tmp_path):
    sid, g, ks = _stars(4)
    path = _twomass_fits(
        tmp_path / "tm.fits", sid, ks, qual=["AAA", "AAU", "UAA", "XAE"]
    )
    cols, _ = lookup(sid, path)
    # the synthetic file is shuffled: find each star's row through its id
    t = Table.read(path)
    q = {int(s): str(x) for s, x in zip(t["source_id"], t["ph_qual"], strict=True)}
    for i, s in enumerate(sid):
        jq, _, kq = q[int(s)][0], q[int(s)][1], q[int(s)][2]
        assert np.isfinite(cols["J"][i]) == (jq in "ABCD")
        assert np.isfinite(cols["KS"][i]) == (kq in "ABCD")
        assert np.isfinite(cols["E_KS"][i]) == (kq in "ABCD")


def test_check_separates_mismatched_from_matched_ks():
    sid, g, ks = _stars(2000)
    shuffled = RNG.permutation(ks)  # Ks of other stars, as in the Andrae file
    check = ks_g_check(g, shuffled, ks)
    assert abs(check["old"]["corr"]) < 0.1
    assert check["new"]["corr"] > 0.9 and check["passed"]
    assert not ks_g_check(g, None, shuffled)["passed"]


def test_replace_columns_and_sidecar_join(tmp_path):
    sid, g, ks = _stars(50)
    fields = ["G", "J", "H", "KS", "E_J", "E_H", "E_KS", "EBV"]
    matrix = RNG.normal(size=(50, len(fields))).astype("f4")
    cols = {
        c: np.full(50, i, "f4")
        for i, c in enumerate(["J", "H", "KS", "E_J", "E_H", "E_KS"])
    }
    assert replace_columns(fields, matrix, cols) == [
        "J",
        "H",
        "KS",
        "E_J",
        "E_H",
        "E_KS",
    ]
    np.testing.assert_array_equal(matrix[:, 3], 2.0)
    side = tmp_path / "side.npz"
    np.savez(side, source_id=sid[::-1], **{c: v[::-1] + 10 for c, v in cols.items()})
    sub = RNG.permutation(50)[:20]
    m2 = np.zeros((20, len(fields)), "f4")
    apply_sidecar(sid[sub], fields, m2, side)
    np.testing.assert_array_equal(m2[:, 3], 12.0)
    with pytest.raises(ValueError, match="missing"):
        apply_sidecar(np.array([1, 2]), fields, m2[:2], side)
    extra = ExtraColumns([side])
    np.testing.assert_array_equal(extra.get("KS", sid[sub]), 12.0)
    assert extra.get("G", sid[sub]) is None


def test_sidecar_script_end_to_end(tmp_path):
    sid, g, ks = _stars(300)
    tm = _twomass_fits(tmp_path / "tm.fits", sid, ks)
    table = Table()
    table["source_id"] = sid
    table["G"] = g.astype("f4")
    table["KS"] = RNG.permutation(ks).astype("f4")  # mismatched, as in the raw file
    table.write(tmp_path / "finetune.fits")
    fields = np.array(["G", "KS"], dtype="U40")
    feats = np.column_stack([g[:100], RNG.permutation(ks[:100])]).astype("f4")
    np.savez(
        tmp_path / "unlabelled.npz", source_id=sid[:100], fields=fields, features=feats
    )
    out = tmp_path / "side"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_twomass_sidecar.py"),
         "--table", str(tmp_path / "finetune.fits"), "--table",
         str(tmp_path / "unlabelled.npz"), "--output-dir", str(out), "--twomass", str(tm)],
        check=True, capture_output=True, text=True,
    )  # fmt: skip
    report = json.loads((out / "twomass_check.json").read_text())
    for rec in report["tables"].values():
        assert rec["check"]["passed"] and rec["check"]["new"]["corr"] > 0.9
    with np.load(out / "finetune_twomass_gaia.npz") as z:
        np.testing.assert_array_equal(z["source_id"], sid)
        np.testing.assert_allclose(z["KS"], ks, rtol=1e-6)


def test_miskeyed_adql_files_are_refused(tmp_path):
    from masked_stellar_autoencoder.data.twomass_gaia import refuse_miskeyed

    for bad in (
        "/arc/projects/k-pop/catalogues/adql_matches/twomass.fits",
        "/arc/projects/k-pop/catalogues/adql_matches/tmass_adql_match_files/tmass_0.fits",
    ):
        with pytest.raises(ValueError, match="mis-keyed"):
            refuse_miskeyed(bad)
        with pytest.raises(ValueError, match="mis-keyed"):
            lookup([1, 2], bad)
    refuse_miskeyed(tmp_path / "twomass_dr3_designation.parquet")  # fine


def test_archive_lookup_aligns_job_results():
    import pandas as pd

    from masked_stellar_autoencoder.data.twomass_gaia import archive_lookup

    sid, g, ks = _stars(30)
    calls = []

    def job(ids):
        calls.append(len(ids))
        keep = ids[::2]  # half the stars have a 2MASS best neighbour
        k = ks[np.searchsorted(sid, keep)]
        return pd.DataFrame(dict(
            source_id=keep, designation="x", ph_qual="AAA", j_m=k + 0.8,
            j_msigcom=0.02, h_m=k + 0.1, h_msigcom=0.03, ks_m=k, ks_msigcom=0.02,
        ))  # fmt: skip

    cols, info = archive_lookup(sid[::-1], chunk=8, job=job)
    assert calls == [8, 8, 8, 6] and info["matched_ids"] == 15
    got = cols["KS"][::-1]
    np.testing.assert_allclose(got[::2], ks[::2], rtol=1e-6)
    assert np.isnan(got[1::2]).all()


def test_builder_masks_twomass_by_default():
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_stopgap_finetune_table import TWOMASS_DEFAULT, replace_twomass

    sid, g, ks = _stars(20)
    fields = ["G", "J", "KS", "E_KS"]
    matrix = np.column_stack([g, ks + 1, RNG.permutation(ks), np.full(20, 0.02)])
    rec = replace_twomass(
        TWOMASS_DEFAULT, fields, {"table": (sid, matrix)}, log=lambda m: None
    )
    assert rec["replaced"] and np.isnan(matrix[:, 1:]).all()
    assert np.isfinite(matrix[:, 0]).all()
    assert replace_twomass("none", fields, {"table": (sid, matrix)}) == dict(
        replaced=False
    )
