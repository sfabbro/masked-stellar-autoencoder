"""The single evaluator (scripts/evaluate_model.py) and the qa package, on synthetic data."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from masked_stellar_autoencoder.data.msa_split import hash_partition
from masked_stellar_autoencoder.qa import competitors as C
from masked_stellar_autoencoder.qa import conatal as K
from masked_stellar_autoencoder.qa import core as Q
from masked_stellar_autoencoder.qa.predictions import (
    Predictions,
    load_predictions,
    write_predictions,
)
from masked_stellar_autoencoder.qa.stats import (
    assign,
    pit_from_quantiles,
    wilson_by_stratum,
)

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(11)


def _script():
    spec = importlib.util.spec_from_file_location(
        "evaluate_model", ROOT / "scripts" / "evaluate_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ids(n):
    return np.unique(
        (RNG.integers(0, 12 * 4**12, 3 * n) << 35) + RNG.integers(1, 2**20, 3 * n)
    )[:n]


# --------------------------------------------------------------------- stats


def test_pit_from_gaussian_quantiles_matches_cdf():
    levels = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
    mu, sd = RNG.normal(size=500), RNG.uniform(0.5, 2, 500)
    q = mu[:, None] + sd[:, None] * norm.ppf(levels)[None]
    y = mu + sd * RNG.uniform(-1.5, 1.5, 500)
    pit = pit_from_quantiles(levels, q, y)
    np.testing.assert_allclose(pit, norm.cdf((y - mu) / sd), atol=0.03)
    far = pit_from_quantiles(levels, q, mu + 10 * sd)
    assert np.all(far == 1.0)


def test_wilson_strata_flag_under_coverage():
    lab = np.array(["a"] * 2000 + ["b"] * 2000 + ["c"] * 5)
    cov = np.r_[RNG.random(2000) < 0.68, RNG.random(2000) < 0.45, np.ones(5, bool)]
    rows = {r["stratum"]: r for r in wilson_by_stratum(lab, cov, 0.68)}
    assert rows["a"]["within_003"] and not rows["a"]["under"]
    assert rows["b"]["under"] and not rows["b"]["within_003"]
    assert rows["c"]["descriptive"]
    assert assign("G", [12.0, 15.5, np.nan, 20.0]).tolist() == [
        "<=13.5",
        "15-16",
        "",
        ">19",
    ]


# ---------------------------------------------------------------- co-natal


def _pred_table(n=4000, labelled_frac=0.2):
    sid = _ids(n)
    teff = RNG.uniform(3500, 6500, n)
    logg = np.where(
        RNG.random(n) < 0.2, RNG.uniform(1.5, 3.2, n), RNG.uniform(4.0, 4.7, n)
    )
    feh = RNG.normal(-0.2, 0.3, n)
    G = RNG.uniform(10, 18, n)
    lab = sid[RNG.random(n) < labelled_frac]
    return sid, teff, logg, feh, G, lab


def test_wide_binaries_cuts_and_bias():
    sid, teff, logg, feh, G, lab = _pred_table()
    a, b = sid[:1000], sid[1000:2000]
    feh = feh.copy()
    feh[1000:2000] = (
        feh[:1000] - 0.2 * (G[1000:2000] > G[:1000]) + 0.2 * (G[1000:2000] <= G[:1000])
    )
    P = K.PredTable.from_arrays(sid, teff, logg, feh, np.full(len(sid), 0.1), G, lab)
    r = np.where(np.arange(1000) < 900, 0.01, 0.5)
    out = K.wide_binaries(P, a, b, r)
    assert out["pairs_scored"] == 900
    assert out["all"]["median"] == pytest.approx(0.2, abs=1e-9)
    assert 0 < out["pairs_neither_labelled"] < 900
    keep = np.arange(1000) < 300
    assert K.wide_binaries(P, a, b, r, keep=keep)["pairs_scored"] == 300


def test_binary_pool_and_clusters():
    sid, teff, logg, feh, G, lab = _pred_table()
    feh = feh.copy()
    P0 = K.PredTable.from_arrays(sid, teff, logg, feh, np.full(len(sid), 0.1), G, lab)
    comp, prim = sid[:500], sid[500:1000]
    pool = K.binary_pool_heldout(P0, comp, prim, teff[:500], G[:500])
    assert pool["scored"] == 500 and np.isfinite(pool["all"]["median"])
    cid = np.repeat(np.arange(40), 50)
    member = sid[:2000]
    feh[:2000] = np.repeat(RNG.normal(0, 0.2, 40), 50) + 0.3 * (
        logg[:2000] < 3.5
    )  # giants +0.3
    P = K.PredTable.from_arrays(sid, teff, logg, feh, np.full(len(sid), 0.05), G, lab)
    table = pd.DataFrame(
        dict(
            ID=np.arange(40),
            Type=["o"] * 38 + ["g", "g"],
            Name=[f"C{i}" for i in range(40)],
        )
    )
    r = K.clusters(P, member, cid, np.full(2000, 0.9), table, {"C39": -1.0})
    oc = K.open_cluster_summary(r)
    assert oc["clusters"] == 38 and oc["giant_minus_dwarf"] == pytest.approx(
        0.3, abs=1e-6
    )
    gc = K.globular_cluster_summary(r)
    assert gc["clusters"] == 2 and gc["with_harris"] == 1
    low = K.clusters(
        P, member, cid, np.full(2000, 0.5), table, keep_cluster=lambda c: c < 38
    )
    assert low.empty  # members below Prob 0.7 are dropped


# --------------------------------------------------------------------- core


def _records(n_rows=300):
    names = np.array(["bensby2014", "saga_mw", "apogee_dr17"])
    rows = np.r_[np.arange(n_rows), np.arange(0, n_rows, 2), np.arange(n_rows)]
    survey = np.r_[np.zeros(n_rows, int), np.ones(n_rows // 2, int), np.full(n_rows, 2)]
    val = np.zeros((len(rows), 3))
    val[:, 2] = np.r_[
        np.linspace(-4, 0.4, n_rows), np.full(n_rows // 2, 9.0), np.full(n_rows, 7.0)
    ]
    return dict(survey_names=names, parameters=np.array(["teff", "logg", "fe_h"]), value=val,
                error=np.full_like(val, 0.1), row=rows, survey=survey)  # fmt: skip


def test_core_references_and_tables():
    rec = _records()
    ref, eref, src = Q.core_references(rec, ("bensby2014", "saga_mw"))
    np.testing.assert_allclose(ref, np.linspace(-4, 0.4, 300))  # bensby wins over saga
    G = RNG.uniform(10, 17.5, 300)
    pred = ref + np.where(ref < -2.5, 0.3, 0.0)
    t = Q.core_table(pred, ref, G)
    assert t["[-9.0,-3.0)"]["G<=16"]["bias"] == pytest.approx(0.3)
    assert Q.worst_bias(t)["bias"] == pytest.approx(0.3)
    s = Q.selection(ref, G, pred < -2.5)
    assert s["all"]["purity"] == 1.0
    tp = Q.tail_probability(ref, (ref < -2.5).astype(float))
    assert tp["brier"] == 0.0 and tp["auc"] == 1.0


# -------------------------------------------------------------- adapters


def test_adapters_posterior_s2_s3_and_canonical(tmp_path):
    n = 50
    sid = _ids(n)
    quant = np.sort(RNG.normal(size=(n, 5, 3)), axis=2)
    np.savez(tmp_path / "post.npz", source_id=sid, truth=RNG.normal(size=(n, 4)),
             truth_names=np.array(["teff", "logg", "fe_h", "ebv_star"]), median=quant[..., 1],
             quantiles=quant, G=RNG.uniform(10, 17, n), EBV=RNG.uniform(0, 1, n))  # fmt: skip
    p = load_predictions(tmp_path / "post.npz")
    assert (
        set(p.q) == {"teff", "logg", "fe_h", "ebv", "parallax"} and "ebv_ref" in p.extra
    )
    np.savez(tmp_path / "s2.npz", source_id=sid, table_row=np.arange(n), split=np.full(n, 2),
             median=quant[:, :3, 1], q16=quant[:, :3, 0], q84=quant[:, :3, 2],
             conformal_group=np.zeros(n), p_mp=np.zeros(n))  # fmt: skip
    assert set(load_predictions(tmp_path / "s2.npz").q) == {"teff", "logg", "fe_h"}
    np.savez(tmp_path / "s3.npz", source_id=sid, table_row=np.arange(n), split=np.full(n, 2),
             G=np.full(n, 15.0), s_q=np.tile([0.0, 0.1, 0.2], (n, 1)), e_q=quant[:, 0],
             pit_plx=RNG.random(n), pit_e=RNG.random(n))  # fmt: skip
    s3 = load_predictions(tmp_path / "s3.npz")
    assert s3.q["dist_mod"][0, 0] == pytest.approx(10.0)  # 1 kpc
    mle = RNG.normal(size=(n, 3))
    np.savez(tmp_path / "s1.npz", source_id=sid, table_row=np.arange(n), split=np.full(n, 2),
             mle=mle, smax=np.zeros(n), post_median=mle, neyman68_lo=mle - 1, neyman68_hi=mle + 1,
             neyman95_lo=mle - 2, neyman95_hi=mle + 2,
             in_neyman68_param=np.ones((n, 3), bool))  # fmt: skip
    s1 = load_predictions(tmp_path / "s1.npz")
    assert s1.kind == "A" and s1.levels.tolist() == [0.025, 0.16, 0.5, 0.84, 0.975]
    np.testing.assert_allclose(s1.q["logg"][:, 1], mle[:, 1] - 1)
    assert s1.extra["fe_h_cov68"].all()
    write_predictions(tmp_path / "c.npz", p)
    c = load_predictions(tmp_path / "c.npz")
    np.testing.assert_array_equal(c.source_id, sid)
    np.testing.assert_allclose(c.q["fe_h"], p.q["fe_h"])


# ------------------------------------------------------------ competitors


def test_competitor_readers_and_compare(tmp_path):
    from astropy.table import Table

    sid = _ids(200)
    t = Table(dict(source_id=sid, in_training_sample=np.arange(200) < 50,
                   mh_xgboost=np.zeros(200), teff_xgboost=np.full(200, 5000.0),
                   logg_xgboost=np.full(200, 4.0)))  # fmt: skip
    t.write(tmp_path / "andrae.fits")
    a = C.read_andrae(tmp_path / "andrae.fits", sid[::2])
    assert len(a) == 100 and a.in_training.sum() == 25
    pd.DataFrame(dict(source_id=sid, xgb_logteff=np.log10(np.full(200, 5500.0)), xgb_logg=4.0,
                      xgb_met=-0.5, xgb_av=0.31, dist=1.0)).to_parquet(tmp_path / "sh.parquet")  # fmt: skip
    sh = C.read_shboost(tmp_path / "sh.parquet", sid[:10])
    assert (
        len(sh) == 10
        and sh.teff.iloc[0] == pytest.approx(5500)
        and sh.dist_mod.iloc[0] == pytest.approx(10)
    )
    f = lambda v: np.full(200, v)  # noqa: E731
    Table(dict(GaiaEDR3=sid, Teff50=f(5000.0), logg50=f(4.0), Met50=f(0.0), Dist50=f(0.1),
               AV50=f(0.31))).write(tmp_path / "gesdr5.fits")  # fmt: skip
    st = C.read_starhorse(tmp_path / "gesdr5.fits", sid[:5])
    assert len(st) == 5 and st.dist_mod.iloc[0] == pytest.approx(5.0)
    staged = tmp_path / "andrae2023.parquet"
    a.to_parquet(staged)
    assert len(C.load_competitor("andrae2023", staged, sid[:20])) == 10
    ref = dict(fe_h=np.zeros(200))
    model = dict(fe_h=RNG.normal(0, 0.05, 200), dist_mod=np.full(200, 10.0))
    out = C.compare(sid, model, ref, a)
    assert out["overlap"] == 100 and out["overlap_in_competitor_training"] == 25
    assert out["fe_h"]["competitor"]["nmad"] == 0.0 and out["fe_h"]["model"]["n"] == 75


# ------------------------------------------------------------ end to end


def _vizier(path, cols, rows):
    lines = [
        "#",
        "\t".join(cols),
        "\t".join(" " for _ in cols),
        "\t".join("-" * 5 for _ in cols),
    ]
    lines += ["\t".join(str(v) for v in r) for r in rows]
    Path(path).write_text("\n".join(lines) + "\n")


def test_evaluator_end_to_end(tmp_path):
    E = _script()
    n = 6000
    sid = _ids(n)
    part = hash_partition(sid)
    freeze = tmp_path / "freeze"
    freeze.mkdir()
    res = sid[:30]  # reserved benchmarks
    pd.DataFrame(dict(source_id=res, partition=np.full(30, 4, np.uint8), group_id=np.full(30, -1, np.int32),
                      input_class=np.zeros(30, np.uint8), reserved_reason=["gbs_v3"] * 30)
                 ).to_parquet(freeze / "overrides.parquet")  # fmt: skip
    levels = np.array([0.025, 0.16, 0.5, 0.84, 0.975])
    truth = dict(
        teff=RNG.uniform(4000, 6500, n),
        logg=RNG.uniform(1.5, 4.7, n),
        fe_h=RNG.normal(-0.3, 0.5, n),
    )
    sd = dict(teff=60.0, logg=0.1, fe_h=0.08)
    q, extra = {}, {}
    for p, t in truth.items():
        mu = t + RNG.normal(0, sd[p], n)
        q[p] = mu[:, None] + sd[p] * norm.ppf(levels)[None]
        extra[f"{p}_ref"] = t
    G = RNG.uniform(10, 18, n)
    extra.update(G=G, table_row=np.arange(n), table_split=np.where(np.arange(n) < 600, 0, 2),
                 teff_irfm=truth["teff"] + RNG.normal(0, 50, n))  # fmt: skip
    write_predictions(
        tmp_path / "pred.npz", Predictions("toy", sid, levels, q, extra, "B")
    )
    cov = pd.DataFrame(dict(source_id=sid, G=G, bp_rp=RNG.uniform(0.3, 2.5, n), b=RNG.uniform(-90, 90, n),
                            ruwe=RNG.uniform(0.8, 2, n), has_xp=RNG.random(n) < 0.7))  # fmt: skip
    cov.to_parquet(tmp_path / "cov.parquet")
    rec = _records(n)
    np.savez(tmp_path / "records.npz", **rec)
    (tmp_path / "cfg.yaml").write_text(
        "multisurvey:\n  fe_h_core: [bensby2014, saga_mw]\n"
    )
    viz = tmp_path / "viz"
    viz.mkdir()
    pairs = [(sid[i], sid[i + 1], 1e-3, "MSMS") for i in range(1000, 3000, 2)]
    _vizier(viz / "elbadry.tsv", ["Source1", "Source2", "R", "BinType"], pairs)
    members = [(f"C{i // 40}", i // 40, sid[3000 + i], 0.9) for i in range(1600)]
    _vizier(viz / "hr23_members.tsv", ["Name", "ID", "GaiaDR3", "Prob"], members)
    _vizier(
        viz / "hr23_clusters.tsv",
        ["Name", "ID", "Type"],
        [(f"C{c}", c, "o") for c in range(40)],
    )
    _vizier(viz / "harris.tsv", ["ID", "[Fe/H]"], [("NGC 104", -0.76)])
    np.savez(tmp_path / "labelled.npz", source_id=sid[:100])
    np.savez(tmp_path / "pool.npz", source_id=sid[4600:4800], primary_id=sid[4800:5000],
             heldout=np.ones(200, bool), teff_companion=RNG.uniform(3000, 6000, 200),
             g_companion=RNG.uniform(14, 18, 200))  # fmt: skip
    gbs = pd.DataFrame(dict(source_id=pd.array(res, dtype="Int64"), Name="b", HD="h",
                            Teff=truth["teff"][:30], logg=truth["logg"][:30], **{"[Fe/H]": truth["fe_h"][:30]}))  # fmt: skip
    gbs.to_parquet(tmp_path / "gbs.parquet")
    comp = pd.DataFrame(
        dict(source_id=sid, teff=truth["teff"] + 100, fe_h=truth["fe_h"])
    )
    comp.to_parquet(tmp_path / "andrae2023.parquet")
    out = tmp_path / "qa"
    E.main([
        "--predictions", f"toy={tmp_path / 'pred.npz'}", "--core", "toy",
        "--core-records", str(tmp_path / "records.npz"), "--core-config", str(tmp_path / "cfg.yaml"),
        "--conatal", str(tmp_path / "pred.npz"), "--vizier-dir", str(viz),
        "--labelled", str(tmp_path / "labelled.npz"), "--binary-pool", str(tmp_path / "pool.npz"),
        "--covariates", str(tmp_path / "cov.parquet"), "--gbs", str(tmp_path / "gbs.parquet"),
        "--physical-catalogues", str(tmp_path / "none"),
        "--competitor", f"andrae2023={tmp_path / 'andrae2023.parquet'}",
        "--split-dir", str(freeze), "--exclude-table-splits", "0,1", "--out", str(out),
    ])  # fmt: skip
    m = json.loads((out / "metrics.json").read_text())
    s = m["sets"]["toy"]
    test = (part == 3) & ~np.isin(sid, res)
    assert s["rows"]["evaluated"] == int((test & (np.arange(n) >= 600)).sum())
    assert s["rows"]["excluded_seen_in_partition"] == int(
        (test & (np.arange(n) < 600)).sum()
    )
    h = s["heldout"]["fe_h"]
    assert (
        abs(h["overall"]["bias"]) < 0.02
        and abs(h["coverage"]["68"]["overall"] - 0.68) < 0.06
    )
    assert "input_class" in h["coverage"]["68"]["by_stratum"]
    assert s["physical"]["gbs_v3"]["predicted"] == 30  # reserved stars are report-only
    assert s["competitors"]["andrae2023"]["teff"]["competitor"][
        "bias"
    ] == pytest.approx(100)
    assert m["core"]["toy"]["with_reference"] > 0 and "table" in m["core"]["toy"]
    assert m["conatal"]["wide_binaries"]["pairs_scored"] > 0
    assert "competitor:survey_of_surveys_dr3" in m["missing"]
    report = (out / "report.md").read_text()
    assert "Frozen metal-poor core" in report and "Wide binaries" in report
    assert (out / "plots" / "qa_toy_by_G.png").exists()
    with pytest.raises(SystemExit, match="allow-dev"):
        E.main(["--predictions", f"toy={tmp_path / 'pred.npz'}", "--partition", "dev",
                "--split-dir", str(freeze), "--out", str(out)])  # fmt: skip


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
