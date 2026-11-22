"""Synthetic-catalogue checks for scripts/age_label_audit.py (no downloads)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _audit():
    path = Path(__file__).parents[1] / "scripts" / "age_label_audit.py"
    spec = importlib.util.spec_from_file_location("age_label_audit_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through it
    spec.loader.exec_module(module)
    return module


def _catalogue(audit, source, sid, lage, sig, *, alt=None, feh=None, stage="giant"):
    n = len(lage)
    age = 10.0 ** np.asarray(lage)
    frame = audit._frame(
        source,
        source_id=np.asarray(sid, dtype="i8"),
        alt_kind="kic" if alt is not None else "",
        alt_id=np.asarray(alt, dtype=str) if alt is not None else "",
        age=age,
        age16=age / 10.0**sig,
        age84=age * 10.0**sig,
        feh=np.zeros(n) if feh is None else feh,
        stage=stage,
    )
    frame["lage"], frame["sig"] = audit.log_age_columns(
        frame.age, frame.age16, frame.age84
    )
    return frame


def test_log_age_columns_recovers_symmetric_dex_width():
    audit = _audit()
    lage, sig = audit.log_age_columns(
        [2.0, 5.0, -1.0], [1.0, 2.5, 1.0], [4.0, 10.0, 2.0]
    )
    np.testing.assert_allclose(lage[:2], np.log10([2.0, 5.0]))
    np.testing.assert_allclose(sig[:2], np.log10(2.0))
    assert np.isnan(lage[2])


def test_crossmatch_uses_gaia_then_alt_ids_and_known_offset():
    audit = _audit()
    rng = np.random.default_rng(1)
    truth = rng.uniform(-0.3, 1.1, 400)
    sid = np.arange(1, 401)
    a = _catalogue(audit, "a", sid, truth + rng.normal(0, 0.05, 400), 0.05)
    b_lage = truth - 0.1 + rng.normal(0, 0.05, 400)
    b_sid = sid.copy()
    b_sid[:50] = 0  # unresolved Gaia IDs: match through the alternative identifier
    b = _catalogue(audit, "b", b_sid, b_lage, 0.05, alt=sid)
    a.loc[:49, "alt_kind"], a.loc[:49, "alt_id"] = "kic", sid[:50].astype(str)
    a.loc[:49, "source_id"] = 0
    m = audit.crossmatch(a, b)
    assert len(m) == 400
    assert (m.match == "gaia_dr3").sum() == 350 and (m.match == "alt:kic").sum() == 50
    s = audit.summarize(m.delta, m.sig_pair)
    assert abs(s["median"] - 0.1) < 0.02
    assert abs(s["nmad"] - np.hypot(0.05, 0.05)) < 0.015
    assert 0.8 < s["inflation"] < 1.2


def test_binned_offsets_follow_an_age_dependent_bias():
    audit = _audit()
    rng = np.random.default_rng(2)
    lage = rng.uniform(np.log10(0.5), np.log10(13.0), 20000)
    age = 10**lage
    delta = 0.02 * (age - 5.0) + rng.normal(0, 0.03, len(age))
    t = audit.binned(delta, age, audit.AGE_EDGES_GYR, min_n=50)
    centre = 0.5 * (t.lo + np.minimum(t.hi, 13.0))
    expected = 0.02 * (centre - 5.0)
    assert np.all(np.abs(t["median"] - expected) < 0.03)
    assert np.all(np.diff(t["median"]) > 0)
    assert (t.n >= 50).all()
    # an empty or sparse bin is dropped, not reported with n < min_n
    sparse = audit.binned(delta[:5], age[:5], audit.AGE_EDGES_GYR, min_n=10)
    assert sparse.empty


def test_pairwise_and_dedupe_on_three_synthetic_catalogues():
    audit = _audit()
    rng = np.random.default_rng(3)
    n = 600
    truth = rng.uniform(-0.2, 1.1, n)
    feh = rng.uniform(-1.5, 0.4, n)
    sid = np.arange(10, 10 + n)
    # A: reference-like; B: offset growing toward low [Fe/H]; C: noisy, partial overlap
    a = _catalogue(audit, "a", sid, truth + rng.normal(0, 0.03, n), 0.03, feh=feh)
    b = _catalogue(audit, "b", sid, truth - 0.1 * (feh < -0.5) + rng.normal(0, 0.03, n),
                   0.03, feh=feh)  # fmt: skip
    c = _catalogue(
        audit,
        "c",
        sid[:200],
        truth[:200] + rng.normal(0, 0.15, 200),
        0.05,
        feh=feh[:200],
    )
    dup = a.iloc[:5].copy()
    dup["sig"] = 0.5  # worse duplicate: dedupe keeps the better one
    rec = pd.concat([a, dup, b, c], ignore_index=True)
    rec["good"] = True
    rec["stage_class"] = "giant"
    assert len(audit.dedupe(rec)) == n + n + 200
    summary, bins, matched = audit.pairwise(rec, min_pairs=30, min_bin=20)
    pairs = {(r.a, r.b): r for r in summary.itertuples()}
    assert set(pairs) == {("a", "b"), ("a", "c"), ("b", "c")}
    assert pairs[("a", "b")].n == n and pairs[("a", "c")].n == 200
    assert pairs[("a", "c")].inflation > 2.0  # C's quoted errors are too small
    fe = bins[(bins.a == "a") & (bins.b == "b") & (bins.dim == "feh")].set_index("lo")
    assert abs(fe.loc[-1.0, "median"] - 0.1) < 0.02
    assert abs(fe.loc[-0.2, "median"]) < 0.02
    assert "stage:giant" in set(bins.dim)


def test_id_map_drops_ambiguous_keys_and_normalises_2mass():
    audit = _audit()
    ids = audit.IdMap()
    ids.add("kic", ["1", "2", "2", "3"], [11, 22, 23, 33], "x")
    ids.add("kic", ["3", "4"], [33, 44], "y")
    np.testing.assert_array_equal(
        ids.resolve("kic", ["1", "2", "3", "4", "5"]), [11, 0, 33, 44, 0]
    )
    # a filtered pandas input keeps gaps in its index; the map must stay positional
    frame = pd.DataFrame({"k": ["7", "8", "9"], "g": [70, 80, 90]}).iloc[[0, 2]]
    ids.add("kic", frame.k, frame.g, "filtered")
    np.testing.assert_array_equal(ids.resolve("kic", ["7", "9"]), [70, 90])
    tm = audit.normalise_2mass(
        ["2M00000002+7417074", "J19250937+3644599", "VESTA", " 00000211+6327470"]
    )
    assert list(tm) == ["00000002+7417074", "19250937+3644599", "", "00000211+6327470"]


def test_pair_stage_prefers_higher_ranked_catalogue():
    audit = _audit()
    out = audit.pair_stage(["MSTO", "unknown", "SGB"], ["RC", "RGB", "unknown"], 2, 5)
    assert list(out) == ["giant", "giant", "subgiant"]


def test_sky_fraction_bounds():
    audit = _audit()
    rng = np.random.default_rng(4)
    l_deg = rng.uniform(0, 360, 200000)
    b_deg = np.degrees(np.arcsin(rng.uniform(-1, 1, 200000)))
    assert audit.sky_fraction(l_deg, b_deg) == 1.0
    assert audit.sky_fraction(np.full(100, 76.0), np.full(100, 13.0)) < 0.01
