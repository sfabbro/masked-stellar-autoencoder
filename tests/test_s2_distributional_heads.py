"""S2 distributional heads: synthetic bimodal recovery, conformal coverage, and
an end-to-end run of ``scripts/s2_distributional_heads.py`` on a tiny stand-in
for the S1 run directory (CPU)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

from masked_stellar_autoencoder.training import distributional_heads as dh

ROOT = Path(__file__).resolve().parents[1]


def _bimodal(n, rng, d=8):
    """Latents z ~ N(0, I); y1 is bimodal (+-1.5) where z0 > 0, unimodal else."""
    z = rng.normal(size=(n, d)).astype("f2")
    zf = z.astype("f8")
    y = np.zeros((n, 3))
    y[:, 0] = 0.5 * zf[:, 1] + rng.normal(0, 0.2, n)
    sign = np.where(rng.random(n) < 0.5, 1.0, -1.0)
    y[:, 1] = np.where(zf[:, 0] > 0, 1.5 * sign, 0.0) + rng.normal(0, 0.25, n)
    y[:, 2] = 0.3 * zf[:, 2] + rng.normal(0, 0.2, n)
    return z, y


def test_gather_rows_matches_fancy_indexing(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.normal(size=(1000, 5)).astype("f2")
    np.save(tmp_path / "a.npy", arr)
    mm = np.load(tmp_path / "a.npy", mmap_mode="r")
    rows = rng.choice(1000, 300, replace=False)
    assert np.array_equal(dh.gather_rows(mm, rows, chunk_rows=64), arr[rows])
    assert np.array_equal(dh.gather_rows(mm, rows[:3], chunk_rows=500), arr[rows[:3]])


def test_mondrian_groups_partition_with_calibration_rows():
    p = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2])
    s = np.array([0, 0, 1, 2, 0, 0, 0, 1, 0, 1])
    cal = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], bool)
    g = dh.mondrian_groups(p, s, cal, min_count=2)
    # dense cells keep their id; bin 0's two sparse cells pool (2 calibration
    # rows); bin 1's empty cell joins its bin's largest group; bin 2 (no
    # calibration row) joins the largest group overall.
    assert g.tolist() == [0, 0, 1000, 1000, 100, 100, 100, 100, 100, 100]
    assert set(g[~cal]) <= set(g[cal])
    g = dh.mondrian_groups(p, None, cal, min_count=4)
    assert g.tolist() == [1000] * 4 + [-1] * 6
    assert (
        dh.mondrian_groups(p, None, cal, min_count=3).tolist()
        == [1000] * 4 + [1001] * 4 + [1000] * 2
    )


def test_flow_recovers_bimodality_and_conformal_coverage_is_nominal():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    z, y = _bimodal(8000, rng)
    std = dh.Standardizer.fit(z.astype("f8"), y)
    zt = torch.as_tensor(z)
    yt = torch.as_tensor(std.y(y), dtype=torch.float32)
    s = torch.full((len(z), 3), 0.02)
    tr, st, cal, te = (
        slice(0, 3500),
        slice(3500, 4000),
        slice(4000, 6000),
        slice(6000, 8000),
    )
    member = dh.Member("flow", z.shape[1], hidden=(64, 64), context=16,
                       flow_hidden=(64, 64), bins=8, tail_bound=6.0)  # fmt: skip
    hist = dh.train_member(member, std, zt[tr], yt[tr], s[tr], zt[st], yt[st], s[st],
                           eiv=False, batch=256, max_epochs=40, patience=5, lr=3e-3,
                           log=lambda *_: None)  # fmt: skip
    assert np.isfinite(hist["best_valid"])
    ens = dh.EnsemblePredictor([member.requires_grad_(False)], std, eiv=False)
    with torch.no_grad():
        pred = ens.predict(zt[te], "cpu")
        draws = pred.sample(256)
        lp = pred.log_prob_many(draws)
        second, peak = dh.secondary_mode(draws, lp, pred.log_prob_many)
    z0 = z[te, 0].astype("f8")
    bimodal = (second >= 0.1).numpy()
    assert bimodal[z0 > 0.3].mean() > 0.8  # the +-1.5 modes are found
    assert bimodal[z0 < -0.3].mean() < 0.3  # unimodal stars are not flagged
    found = np.isfinite(peak.numpy()).all(1)
    assert (
        np.all(np.isfinite(peak.numpy()[bimodal]).all(1))
        and found.sum() == bimodal.sum()
    )
    # Split-conformal level sets of -log p on held-out synthetic data.
    with torch.no_grad():
        lp_cal = ens.predict(zt[cal], "cpu").log_prob(yt[cal]).numpy()
        lp_te = ens.predict(zt[te], "cpu").log_prob(yt[te]).numpy()
    groups = (z[:, 0] > 0).astype(int)
    for level in (0.68, 0.95):
        inside, thr = dh.conformal_level_sets(lp_cal, lp_te, level)
        assert abs(inside.mean() - level) < 0.05, (level, inside.mean())
        assert set(thr) == {None}
        inside, thr = dh.conformal_level_sets(
            lp_cal, lp_te, level, groups[cal], groups[te]
        )
        assert set(thr) == {0, 1}
        for g in (0, 1):
            assert abs(inside[groups[te] == g].mean() - level) < 0.06
    # Ensemble bookkeeping: quantiles, mode and stratified draws.
    with torch.no_grad():
        q, smp = pred.quantiles((0.16, 0.5, 0.84), n_samples=64)
    assert q.shape == (2000, 3, 3) and smp.shape == (2000, 64, 3)
    assert torch.all(q[..., 0] <= q[..., 2])
    mode, mlp = dh.find_mode(pred, smp, steps=5)
    assert mode.shape == (2000, 3) and torch.isfinite(mlp).all()


def test_mixture_eiv_label_density_is_wider_than_clean():
    torch.manual_seed(1)
    rng = np.random.default_rng(1)
    z, y = _bimodal(600, rng)
    std = dh.Standardizer.fit(z.astype("f8"), y)
    m1 = dh.Member("tmix", z.shape[1], hidden=(16,), components=2).requires_grad_(False)
    m2 = dh.Member("t", z.shape[1], hidden=(16,)).requires_grad_(False)
    ens = dh.EnsemblePredictor([m1, dh.Member("tmix", z.shape[1], hidden=(16,),
                                              components=2).requires_grad_(False)], std, True)  # fmt: skip
    with torch.no_grad():
        pred = ens.predict(z[:50], "cpu")
        assert pred.mixture.loc.shape == (50, 4, 3)  # 2 members x 2 components
        y0 = pred.mixture.loc[:, 0]
        clean = pred.log_prob(y0)
        noisy = pred.label_log_prob(y0, torch.full((50, 3), 2.0))
        assert torch.all(noisy < clean)  # broad label noise lowers the peak density
        many = pred.label_log_prob(pred.sample(7), torch.full((50, 3), 0.1))
        assert many.shape == (50, 7)
        single = dh.EnsemblePredictor([m2], std, True).predict(z[:5], "cpu")
        assert single.quantiles((0.5,))[0].shape == (5, 3, 1)
    post = dh.posterior_prediction(y[:5], y[:5] - 0.1, y[:5] + 0.1, 5.0, std, "cpu")
    q = (
        post.quantiles((0.16, 0.5, 0.84))[0].numpy() * std.y_std[:, None]
        + std.y_mean[:, None]
    )
    assert np.allclose(q[:, :, 1], y[:5], atol=1e-4)
    assert np.allclose(q[:, :, 2] - q[:, :, 0], 0.2, atol=1e-3)


def _write_stand_in(tmp, n=1500, d=12, seed=0):
    """Synthetic S1 run directory: latents, posterior predictions, labels,
    frozen-core rows, anchor records and an S1 baseline."""
    rng = np.random.default_rng(seed)
    zf = rng.normal(size=(n, d))
    teff = 5200 + 600 * np.tanh(zf[:, 0]) + rng.normal(0, 60, n)
    giant = zf[:, 1] > 0
    logg = np.where(giant & (rng.random(n) < 0.6), 2.5, 4.4) + rng.normal(0, 0.15, n)
    logg = np.where(giant, logg, 4.3 + rng.normal(0, 0.1, n))
    feh = np.clip(-0.6 + 0.7 * zf[:, 2], -4.0, 0.8) + rng.normal(0, 0.08, n)
    G = rng.uniform(11, 18, n)
    np.save(tmp / "latent.npy", zf.astype("f2"))
    split = rng.choice([0, 1, 2], n, p=[0.6, 0.2, 0.2]).astype("i1")
    split[:20] = -1
    theta = np.column_stack([teff, logg, feh])
    core = np.flatnonzero(rng.random(n) < 0.25)
    ref = np.full(n, np.nan)
    ref[core] = feh[core] + rng.normal(0, 0.05, len(core))
    np.save(tmp / "rows.npy", core)
    sid = np.arange(n, dtype="i8") + 10**15
    lab = dict(source_id=sid, table_row=np.arange(n), split=split,
               teff=teff + rng.normal(0, 50, n), e_teff=np.full(n, 50.0),
               logg=logg + rng.normal(0, 0.05, n), e_logg=np.full(n, 0.05),
               fe_h=feh + rng.normal(0, 0.05, n), e_fe_h=np.full(n, 0.05),
               n_teff=np.ones(n), n_logg=np.ones(n), n_feh=np.ones(n),
               has_core_feh=np.isfinite(ref), fe_h_ref=ref, fe_h_core=ref)  # fmt: skip
    lab["e_logg"][::50] = np.nan  # missing errors get the train median
    lab["fe_h"][::97] = np.nan  # incomplete labels: not labelled
    np.savez(tmp / "labels.npz", **lab)
    med = theta + rng.normal(0, 1, (n, 3)) * [80, 0.2, 0.1]
    half = np.array([120.0, 0.25, 0.15])
    np.savez(tmp / "predictions.npz", source_id=sid, G=G, post_median=med,
             post_q16=med - half, post_q84=med + half, post_nu=np.float64(5.0),
             post_balanced=True)  # fmt: skip
    anchor = rng.choice(n, 200, replace=False)
    val = np.full((400, 4), np.nan)
    val[:200, 0] = teff[anchor] + rng.normal(0, 40, 200)
    val[200:, 1] = logg[anchor] + rng.normal(0, 0.03, 200)
    np.savez(tmp / "records.npz", row=np.r_[anchor, anchor],
             survey=np.r_[np.zeros(200), np.ones(200)].astype("i4"), value=val,
             error=np.ones_like(val), survey_names=np.array(["irfm", "seismic"]),
             survey_is_anchor=np.array([True, True]),
             parameters=np.array(["teff", "logg", "fe_h", "alpha"]))  # fmt: skip
    base = tmp / "s1"
    base.mkdir()
    np.savez(base / "labels.npz", source_id=sid)
    held = np.flatnonzero(np.isin(split, (1, 2)))
    np.savez(base / "infer_all.npz", table_row=held,
             mle=theta[held] + rng.normal(0, 1, (len(held), 3)) * [100, 0.3, 0.15])  # fmt: skip
    return base


def test_s2_script_end_to_end(tmp_path, monkeypatch):
    base = _write_stand_in(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "s2_distributional_heads", ROOT / "scripts" / "s2_distributional_heads.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "s2", "--inputs", str(tmp_path), "--output", str(out),
        "--records", str(tmp_path / "records.npz"), "--core-rows", str(tmp_path / "rows.npy"),
        "--baseline", str(base), "--heads", "flow,tmix,gmix,t,flow_noeiv", "--ensemble", "2",
        "--device", "cpu", "--epochs", "3", "--batch", "128", "--hidden", "32,32",
        "--context", "8", "--flow-hidden", "32,32", "--flow-bins", "8",
        "--components", "2", "--eiv-samples", "2", "--stop-mc", "2", "--eval-mc", "4",
        "--hpd-mc", "2", "--summary-samples", "16", "--score-samples", "32",
        "--n-score", "150", "--mode-steps", "3", "--eval-chunk", "256",
        "--score-chunk", "64", "--mondrian-min", "15", "--speed-rows", "100",
        "--speed-rows-cpu", "100", "--example-samples", "50", "--max-stop", "100",
    ])  # fmt: skip
    mod.main()
    sel = json.loads((out / "selection.json").read_text())
    assert sel["counts"]["train"] > 500 and sel["counts"]["core_heldout"] > 50
    m = json.loads((out / "metrics.json").read_text())
    assert set(m["scores"]) == {"posterior", "flow", "tmix", "gmix", "t", "flow_noeiv"}
    for nm, s in m["scores"].items():
        assert np.isfinite(s["nll_label"]) and np.isfinite(s["energy"]), nm
    assert set(m["decision"]) == {"flow", "tmix", "gmix", "t", "flow_noeiv"}
    assert "s1_all" in m["frozen_core"] and "flow_mode" in m["frozen_core"]
    assert (
        "teff_anchor" in m["anchors"] and "flow_median" in m["anchors"]["teff_anchor"]
    )
    cov = m["coverage"]["flow"]
    for k in ("raw_0.68", "raw_clean_0.68", "marginal_0.95", "mondrian_gfeh_0.68"):
        assert k in cov, k
    # Conformal marginal coverage of the test labels is close to nominal.
    for nm in m["coverage"]:
        c = m["coverage"][nm]["marginal_0.95"]["all"]["coverage"]
        assert c > 0.85, (nm, c)
    assert "flow" in m["bimodality"] and "t" not in m["bimodality"]
    assert set(m["speed"]["flow"]) == {"cpu"}
    assert m["speed"]["tmix"]["cpu"]["summaries_per_s"] > 0
    report = (out / "report.md").read_text()
    for title in ("## Decision", "## Scores", "## Coverage", "## Frozen held-out core",
                  "## Anchors", "## Non-Gaussianity", "## Speed", "## QA plots"):  # fmt: skip
        assert title in report, title
    for f in ("qa_coverage.png", "qa_bias_nmad.png", "qa_pit.png", "qa_examples.png"):
        assert (out / f).stat().st_size > 5000, f
    pr = np.load(out / "s2_predictions_flow.npz")
    assert pr["median"].shape[1] == 3 and np.isfinite(pr["median"]).all()
    assert np.isfinite(pr["conformal68_nlp"]).all()
    ex = np.load(out / "examples.npz")
    assert ex["samples_flow"].shape == (4, 50, 3) and "samples_posterior" in ex.files
    thr = json.loads((out / "conformal_tmix.json").read_text())["thresholds"]
    assert set(thr) == set(mod.VARIANTS)


def test_gaussian_mixture_head_is_the_df_inf_mixture():
    """gmix: a full-covariance Gaussian mixture (df = inf, not learned)."""
    torch.manual_seed(2)
    rng = np.random.default_rng(2)
    z, y = _bimodal(200, rng)
    std = dh.Standardizer.fit(z.astype("f8"), y)
    m = dh.Member("gmix", z.shape[1], hidden=(16,), components=2).requires_grad_(False)
    assert not any(n.endswith("raw_df") for n, _ in m.named_parameters())
    with torch.no_grad():
        pred = dh.EnsemblePredictor([m], std, True).predict(z[:20], "cpu")
        assert torch.isinf(pred.mixture.df).all()
        assert torch.isfinite(pred.log_prob(pred.mixture.loc[:, 0])).all()
