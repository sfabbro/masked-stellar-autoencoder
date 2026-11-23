"""S3 joint distance-extinction heads: quadrature and PIT against brute force,
recovery of the distance relation from parallaxes alone (low-S/N and negative
parallaxes included) against a parallax cut and naive inversion, coverage of
held-out parallaxes, and an end-to-end run of ``scripts/s3_distance_extinction.py``
on a tiny synthetic stand-in (CPU)."""

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training import distance_extinction as de
from masked_stellar_autoencoder.training.distributional_heads import Standardizer

ROOT = Path(__file__).resolve().parents[1]
LN10 = math.log(10.0)


def _stars(n, rng, d_noise=3):
    """Synthetic stars with known s = ln(d/kpc) and E(B-V).

    The latent carries G (apparent), M (absolute magnitude, the "spectroscopic"
    information, noisy) and E (noisy colour excess) plus noise dimensions, so
    s = (G - M - 2.7 E - 10) ln10 / 5 is predictable to ~0.08. Parallax errors
    grow with G: most faint stars have plx/err < 1 and many negative parallaxes.
    """
    M = rng.uniform(-0.5, 7.5, n)
    s = rng.normal(0.4, 0.8, n)
    E = rng.gamma(2.0, 0.05, n) * np.exp(0.4 * s)
    G = M + 10 + 5 * s / LN10 + 2.7 * E
    z = np.column_stack([G + rng.normal(0, 0.01, n), M + rng.normal(0, 0.15, n),
                         E + rng.normal(0, 0.03, n), rng.normal(size=(n, d_noise))])  # fmt: skip
    sig = np.maximum(0.02 * 10 ** (0.2 * (G - 13)), 0.01)
    plx = np.exp(-s) + sig * rng.normal(size=n)
    e_err = np.full(n, 0.03)
    ebv = E + e_err * rng.normal(size=n)
    return dict(z=z, s=s, E=E, G=G, plx=plx, sig=sig, ebv=ebv, e_ebv=e_err)


def _obs(st, idx, with_e=True):
    o = dict(parallax_corrected=st["plx"], parallax_error=st["sig"],
             kappa_bin=de.kappa_bin(st["G"]),
             ebv=st["ebv"] if with_e else np.full(len(st["G"]), np.nan), e_ebv=st["e_ebv"])  # fmt: skip
    return de.observation_tensors(o, idx, "cpu")


def _std(z):
    return Standardizer(z.mean(0), z.std(0), np.array([0.0, 0.2]), np.array([1.0, 0.2]))


def _member(kind, d, **kw):
    base = dict(hidden=(64, 64), context=16, flow_hidden=(32, 32), bins=8, components=3,
                e_loc=0.2, e_scale=0.2)  # fmt: skip
    base.update(kw)
    return de.Member(kind, d, **base)


@pytest.mark.parametrize("kind", ["gmix", "flow"])
def test_marginal_likelihood_and_pit_match_brute_force(kind):
    torch.manual_seed(0)
    m = _member(kind, 6, hidden=(16,), context=8, flow_hidden=(16, 16))
    B = 6
    x = torch.randn(B, 6)
    ob = dict(plx=torch.tensor([1.5, 0.3, -0.2, 0.05, 2.0, 0.8]),
              sig=torch.tensor([0.01, 0.2, 0.3, 0.5, 0.05, 0.4]),
              gbin=torch.zeros(B, dtype=torch.long),
              e_obs=torch.tensor([0.1, 0.5, 0.2, 0.0, 0.3, 0.9]),
              e_sig=torch.tensor([0.03, 0.05, 0.02, 0.04, 0.03, 0.03]),
              has_plx=torch.ones(B, dtype=torch.bool),
              has_e=torch.tensor([True, True, False, True, False, True]))  # fmt: skip
    with torch.no_grad():
        p = m(x)
        ll = de.log_marginal_likelihood(m, p, ob, n_s=40, n_e=16)
        s = torch.linspace(-7, 7, 2801)
        E = torch.linspace(-3, 4, 1401)
        S, EE = s[:, None].expand(2801, B), E[:, None].expand(1401, B)
        lj = m.log_prob_joint(p, S, EE)
        lpl = de.parallax_log_likelihood(S, ob["plx"], ob["sig"])
        lle = -0.5 * ((ob["e_obs"] - EE) / ob["e_sig"]) ** 2 - torch.log(ob["e_sig"])
        lle = torch.where(
            ob["has_e"], lle - 0.5 * math.log(2 * math.pi), torch.zeros_like(lle)
        )
        dsde = math.log(14 / 2800) + math.log(7 / 1400)
        ref = torch.logsumexp(lj + lpl[:, None] + lle[None] + dsde, (0, 1))
        assert torch.allclose(ll, ref, atol=2e-3), (ll, ref)
        # The joint density is normalised; the s marginal is exact.
        assert torch.allclose(
            torch.logsumexp(lj + dsde, (0, 1)), torch.zeros(B), atol=2e-3
        )
        lps = m.log_prob_s(p, S)
        assert torch.allclose(
            torch.logsumexp(lj + math.log(7 / 1400), 1), lps, atol=2e-3
        )
        # PIT of the parallax under the convolved predictive vs Monte Carlo.
        pit = de.parallax_pit(m, p, ob)
        w = torch.softmax(lps, 0)
        gen = torch.Generator().manual_seed(1)
        idx = torch.multinomial(w.T, 100_000, replacement=True, generator=gen).T
        rep = torch.exp(-s[idx]) + ob["sig"] * torch.randn(idx.shape, generator=gen)
        mc = (rep <= ob["plx"]).float().mean(0)
        assert torch.allclose(pit, mc, atol=6e-3), (pit, mc)


def test_windows_resolve_precise_and_negative_parallaxes():
    lo_p, hi_p = torch.full((4,), -1.0), torch.full((4,), 1.0)
    plx = torch.tensor([2.0, -0.1, -0.4, 50.0])
    sig = torch.tensor([0.01, 0.3, 0.1, 0.1])
    lo_m, hi_m, inf = de.parallax_window(plx, sig, torch.ones(4, dtype=torch.bool))
    lo, hi, fb = de.combine_windows(lo_p, hi_p, lo_m, hi_m, inf, de.S_BOUNDS)
    assert bool(inf.all())
    # Precise parallax: the likelihood's narrow window (h = 2 +- 7 x 0.01 mas).
    assert hi[0] - lo[0] < 0.08 and lo[0] > -0.8 and hi[0] < -0.6 and not fb[0]
    # Negative parallax within errors: h < 2 mas allowed, the far side open
    # (box edge) and cut by the predictive.
    assert torch.isclose(lo[1], torch.tensor(-math.log(2.0))) and torch.isclose(
        hi[1], torch.tensor(1.0)
    )
    # -0.4 +- 0.1 mas: the likelihood grows towards h -> 0, beyond the
    # predictive: the gap plus the (narrower) predictive window.
    assert (
        bool(fb[2])
        and torch.isclose(lo[2], torch.tensor(-1.0))
        and torch.isclose(hi[2], torch.tensor(-math.log(0.3)))
    )
    # 50 mas, far in the predictive's tail: the (narrow) measurement window,
    # which holds the product's mass; flagged as tension.
    assert bool(fb[3]) and lo[3] < -3.92 and hi[3] < -3.89


def _fit(member, st, tr, st_rows, with_e=True, epochs=10, batch=512, lr=3e-3):
    z = torch.as_tensor(st["z"], dtype=torch.float32)
    std = _std(st["z"])
    hist = de.train_member(member, std, z[tr], _obs(st, tr, with_e), z[st_rows],
                           _obs(st, st_rows, with_e), n_s=32, n_e=12, batch=batch,
                           max_epochs=epochs, patience=3, lr=lr, log=lambda *_: None)  # fmt: skip
    return std, hist


def _regress_s(z, y, rows, epochs=60, seed=0):
    """Point regression z -> s (MSE), the parallax-cut / inversion baselines."""
    torch.manual_seed(seed)
    zt = torch.as_tensor(z, dtype=torch.float32)
    mu, sd = zt[rows].mean(0), zt[rows].std(0)
    net = torch.nn.Sequential(torch.nn.Linear(z.shape[1], 64), torch.nn.GELU(),
                              torch.nn.Linear(64, 64), torch.nn.GELU(), torch.nn.Linear(64, 1))  # fmt: skip
    opt = torch.optim.Adam(net.parameters(), 3e-3)
    yt = torch.as_tensor(y, dtype=torch.float32)
    rows = torch.as_tensor(rows)
    for _ in range(epochs):
        for i in range(0, len(rows), 512):
            r = rows[torch.randperm(len(rows))[:512]]
            loss = (net((zt[r] - mu) / sd)[:, 0] - yt[r]).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return net((zt - mu) / sd)[:, 0].numpy()


def test_parallax_likelihood_recovers_distances_without_cut_bias():
    """Training on every parallax (low S/N and negative included) recovers the
    distance relation at the faint end; a parallax/error cut with inverted
    parallaxes and naive inversion of the positive parallaxes are biased there.
    Held-out parallaxes are covered at the nominal rate (PIT)."""
    rng = np.random.default_rng(1)
    st = _stars(36_000, rng)
    n = len(st["G"])
    tr, sp, te = np.arange(0, 26_000), np.arange(26_000, 28_000), np.arange(28_000, n)
    snr = st["plx"] / st["sig"]
    faint = te[st["G"][te] > 18.5]
    assert np.mean(snr[faint] < 1) > 0.5 and np.mean(st["plx"][faint] < 0) > 0.2
    torch.manual_seed(1)
    m = _member("gmix", st["z"].shape[1])
    std, hist = _fit(m, st, tr, sp, with_e=False, epochs=12)
    ens = de.EnsembleDE([m.requires_grad_(False)], std)
    z = torch.as_tensor(st["z"][te], dtype=torch.float32)
    r = ens.predict(z, _obs(st, te, with_e=False), n_joint=32, n_e=32, n_s_lik=32)
    err_like = r["s_q"][:, 1] - st["s"][te]
    fz = st["G"][te] > 18.5
    bias_like = float(np.median(err_like[fz]))
    # Baselines: inverted parallaxes as labels, (a) plx/err > 5, (b) every plx > 0.
    with np.errstate(invalid="ignore", divide="ignore"):
        s_inv = -np.log(st["plx"])
    cut = tr[snr[tr] > 5]
    pos = tr[st["plx"][tr] > 0]
    bias_cut = float(
        np.median(_regress_s(st["z"], s_inv, cut)[te][fz] - st["s"][te][fz])
    )
    bias_naive = float(
        np.median(_regress_s(st["z"], s_inv, pos)[te][fz] - st["s"][te][fz])
    )
    print(
        f"faint bias: likelihood {bias_like:+.3f}, cut {bias_cut:+.3f}, naive {bias_naive:+.3f}"
    )
    assert abs(bias_like) < 0.04, bias_like
    assert abs(bias_naive) > 0.15, bias_naive
    assert abs(bias_cut) > 0.08 and abs(bias_cut) > 2.5 * abs(bias_like), bias_cut
    # Precision near the true scatter (~0.08 in s) and calibration of the
    # held-out parallaxes (every S/N, no selection on the parallax).
    assert np.median(0.5 * (r["s_q"][:, 2] - r["s_q"][:, 0])) < 0.15
    pit = r["pit_plx"]
    assert abs(np.mean(np.abs(pit - 0.5) <= 0.34) - 0.68) < 0.03
    assert abs(np.mean(np.abs(pit - 0.5) <= 0.475) - 0.95) < 0.02
    assert abs(np.mean(np.abs(pit[fz] - 0.5) <= 0.34) - 0.68) < 0.05
    # True errors are exact: the fitted inflation stays near 1 where constrained.
    kap = m.kappa().detach().numpy()
    assert np.all(np.abs(kap[2:] - 1.0) < 0.15), kap
    # The combined (spectrophotometric + parallax) distance is tighter (a
    # conflicting measurement may widen a mixture posterior, so on aggregate),
    # and much tighter for precise parallaxes.
    hc = 0.5 * (r["comb_s_q"][:, 2] - r["comb_s_q"][:, 0])
    hs = 0.5 * (r["s_q"][:, 2] - r["s_q"][:, 0])
    assert np.mean(hc <= hs + 1e-3) > 0.97 and np.median(hc) < np.median(hs)
    precise = snr[te] > 20
    assert np.median(hc[precise]) < 0.5 * np.median(hs[precise])


def test_flow_joint_head_learns_extinction_and_correlation():
    """The flow learns E from its labels (errors in variables) jointly with s."""
    rng = np.random.default_rng(2)
    st = _stars(12_000, rng)
    tr, sp, te = np.arange(0, 9000), np.arange(9000, 10_000), np.arange(10_000, 12_000)
    torch.manual_seed(2)
    m = _member("flow", st["z"].shape[1], e_loc=float(st["ebv"].mean()),
                e_scale=float(st["ebv"].std()))  # fmt: skip
    std, hist = _fit(m, st, tr, sp, with_e=True, epochs=8, batch=256)
    assert np.isfinite(hist["best_valid"])
    ens = de.EnsembleDE([m.requires_grad_(False)], std)
    z = torch.as_tensor(st["z"][te], dtype=torch.float32)
    r = ens.predict(z, _obs(st, te), n_joint=32, n_e=32, n_s_lik=32, n_e_lik=16)
    de_ = r["e_q"][:, 1] - st["E"][te]
    assert abs(np.median(de_)) < 0.02 and 1.4826 * np.median(np.abs(de_)) < 0.05
    ds = r["s_q"][:, 1] - st["s"][te]
    assert abs(np.median(ds)) < 0.05
    assert np.all(np.isfinite(r["lp_obs"])) and np.all(np.isfinite(r["pit_e"]))
    assert abs(np.mean(np.abs(r["pit_plx"] - 0.5) <= 0.34) - 0.68) < 0.06


# ── end-to-end ──────────────────────────────────────────────────────────────


def _write_stand_in(tmp, n=3000, seed=0):
    from astropy.io import fits

    rng = np.random.default_rng(seed)
    st = _stars(n, rng, d_noise=2)
    # Five clusters of 12 members each: shared s and E.
    cl_rows = rng.choice(n, 60, replace=False).reshape(5, 12)
    for c, rows in enumerate(cl_rows):
        st["s"][rows] = -0.5 + 0.2 * c
        st["E"][rows] = 0.05 + 0.03 * c
    M = st["z"][:, 1]
    st["G"] = M + 10 + 5 * st["s"] / LN10 + 2.7 * st["E"]
    st["z"][:, 0] = st["G"]
    st["z"][:, 2] = st["E"] + rng.normal(0, 0.03, n)
    st["plx"] = np.exp(-st["s"]) + st["sig"] * rng.normal(size=n)
    st["ebv"] = st["E"] + st["e_ebv"] * rng.normal(size=n)
    inp = tmp / "s1"
    inp.mkdir()
    np.save(inp / "latent.npy", st["z"].astype("f2"))
    (inp / "latent_meta.json").write_text(json.dumps(dict(hidden_inputs=["PARALLAX"])))
    sid = np.arange(n, dtype="i8") * 7 + 10**15
    split = rng.choice([0, 1, 2], n, p=[0.6, 0.2, 0.2]).astype("i1")
    split[cl_rows[:, :6].ravel()] = 2
    split[:40] = -1
    feh = rng.normal(-0.5, 0.6, n)
    ref = np.where(rng.random(n) < 0.3, feh + rng.normal(0, 0.05, n), np.nan)
    np.savez(inp / "predictions.npz", source_id=sid, G=st["G"])
    np.savez(inp / "labels.npz", source_id=sid, split=split, fe_h=feh, fe_h_ref=ref)
    np.save(tmp / "rows.npy", np.flatnonzero(np.isfinite(ref)))
    ra, dec = rng.uniform(0, 360, n), np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    cols = [fits.Column(name="source_id", format="K", array=sid),
            fits.Column(name="PARALLAX", format="D", array=st["plx"]),
            fits.Column(name="e_parallax", format="D", array=st["sig"]),
            fits.Column(name="G", format="D", array=st["G"]),
            fits.Column(name="RA", format="D", array=ra),
            fits.Column(name="DEC", format="D", array=dec),
            fits.Column(name="EBV", format="D", array=st["E"] + rng.normal(0, 0.01, n))]  # fmt: skip
    fits.BinTableHDU.from_columns(cols).writeto(tmp / "table.fits")
    (tmp / "config.yaml").write_text(json.dumps(dict(data=dict(
        feature_cols=["G", "PARALLAX", "bp_1", "rp_1"],
        error_cols=[None, "e_parallax", None, None], ft_datafile=str(tmp / "table.fits")))))  # fmt: skip
    ebv = st["ebv"].copy()
    ebv[rng.random(n) < 0.3] = np.nan
    np.savez(tmp / "reddening.npz", source_id=sid, ebv_star=ebv, e_ebv_star=st["e_ebv"])
    mem = [fits.Column(name="GaiaDR3", format="K", array=np.r_[sid[cl_rows.ravel()], 1, 2]),
           fits.Column(name="ID", format="K", array=np.r_[np.repeat(np.arange(5), 12), 0, 9]),
           fits.Column(name="Prob", format="D", array=np.r_[np.full(60, 0.9), 0.9, 0.9])]  # fmt: skip
    fits.BinTableHDU.from_columns(mem).writeto(tmp / "members.fits")
    clus = [fits.Column(name="ID", format="K", array=np.arange(10)),
            fits.Column(name="Name", format="10A", array=np.array([f"C{i}" for i in range(10)])),
            fits.Column(name="Type", format="1A", array=np.array(["o"] * 5 + ["g"] * 5))]  # fmt: skip
    fits.BinTableHDU.from_columns(clus).writeto(tmp / "clusters.fits")
    noxp = tmp / "noxp"
    noxp.mkdir()
    zx = st["z"].copy()
    zx[:, 1] += rng.normal(0, 0.4, n)  # less spectroscopic information
    np.save(noxp / "latent.npy", zx.astype("f2"))
    (noxp / "latent_meta.json").write_text(
        json.dumps(dict(hidden_inputs=["PARALLAX", "bp_1", "rp_1"]))
    )
    return inp


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "s3_distance_extinction", ROOT / "scripts" / "s3_distance_extinction.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(tmp, inp, out, *extra):
    return [
        "s3", "--inputs", str(inp), "--config", str(tmp / "config.yaml"), "--output", str(out),
        "--reddening", str(tmp / "reddening.npz"), "--core-rows", str(tmp / "rows.npy"),
        "--members", str(tmp / "members.fits"), "--clusters-table", str(tmp / "clusters.fits"),
        "--cluster-min-n", "3", "--noxp-inputs", str(tmp / "noxp"), "--device", "cpu",
        "--ensemble", "2", "--epochs", "3", "--batch", "256", "--hidden", "32,32",
        "--context", "8", "--flow-hidden", "16,16", "--flow-bins", "8", "--components", "2",
        "--train-ns", "32", "--train-ne", "16", "--pred-ns", "128", "--pred-nsj", "24",
        "--pred-ne", "24", "--quad-rows", "200", "--eval-chunk", "256", "--max-stop", "300",
        *extra,
    ]  # fmt: skip


def test_s3_script_end_to_end(tmp_path, monkeypatch):
    inp = _write_stand_in(tmp_path)
    mod = _load_script()
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", _args(
        tmp_path, inp, out, "--stages", "data,clusters,train,predict,noxp,report"))  # fmt: skip
    mod.main()
    sel = json.loads((out / "selection.json").read_text())
    c = sel["counts"]
    assert (
        c["train"] > 1000 and c["negative_parallax"] > 50 and c["hash_split_rows"] == 40
    )
    assert c["train_negative_parallax"] > 0 and c["train_with_ebv"] < c["train"]
    m = json.loads((out / "metrics.json").read_text())
    assert set(m["scores"]) == {"gmix", "flow", "gmix_noxp", "flow_noxp"}
    for nm, s in m["scores"].items():
        assert np.isfinite(s["lp_obs"]["nll"]) and np.isfinite(s["lp_plx"]["nll"]), nm
    for f in ("gmix", "flow"):
        d = m["distance"][f]
        assert d["n"] > 300 and d["n_precise"] > 20
        assert 0.4 < d["all"]["cov68"] < 0.9
        assert (
            len(m["kappa"][f]) == 2
            and len(m["kappa"][f][0]) == len(de.KAPPA_G_EDGES) + 1
        )
        assert m["clusters"][f]["n_clusters"] == 5
        assert m["quadrature"][f]["rows"] == 200 and m["quadrature"][f]["p99"] < 0.1
        assert set(m["noxp"][f]) == {"full", "noxp", "ebv_full", "ebv_noxp"}
        assert m["ebv"][f]["high_lat_vs_sfd"]["prediction"]["n"] > 10
    assert set(m["decision"]) == {"gmix", "flow"}
    report = (out / "report.md").read_text()
    for title in ("## Decision", "## Scores", "## Distances", "## Held-out frozen core",
                  "## E(B-V)", "## Open clusters", "## Beyond XP", "## Quadrature",
                  "## QA plots"):  # fmt: skip
        assert title in report, title
    for f in ("qa_distance_hexbin.png", "qa_distance_bias.png", "qa_parallax_calibration.png",
              "qa_ebv.png", "qa_examples.png", "qa_clusters.png", "qa_noxp.png"):  # fmt: skip
        assert (out / f).stat().st_size > 5000, f
    pr = np.load(out / "s3_predictions_flow.npz")
    assert pr["s_q"].shape[1] == 3 and np.isfinite(pr["s_q"]).all()
    assert np.isfinite(pr["e_q"]).all() and np.isfinite(pr["rho"]).all()
    has = np.isfinite(pr["pit_plx"])
    assert has.mean() > 0.95 and np.isfinite(pr["comb_s_q"][has]).all()
    assert (np.load(out / "s3_noxp_predictions_gmix.npz")["split"] == 2).all()
    ex = np.load(out / "examples_gmix.npz")
    assert ex["log_p"].shape[:2] == (96, 64)


def test_s3_refuses_latents_with_parallax_input(tmp_path, monkeypatch):
    inp = _write_stand_in(tmp_path)
    (inp / "latent_meta.json").write_text(json.dumps(dict(hidden_inputs=[])))
    mod = _load_script()
    monkeypatch.setattr(
        sys, "argv", _args(tmp_path, inp, tmp_path / "out", "--stages", "data")
    )
    with pytest.raises(SystemExit, match="PARALLAX hidden"):
        mod.main()


def test_micro_batch_accumulation_matches_the_full_batch_step():
    """Gradient accumulation over micro-batches is the same optimisation step."""
    rng = np.random.default_rng(5)
    st = _stars(600, rng)
    z = torch.as_tensor(st["z"], dtype=torch.float32)
    std = _std(st["z"])
    tr, sr = np.arange(500), np.arange(500, 600)
    params = []
    for micro in (None, 64):
        torch.manual_seed(0)
        member = _member("gmix", z.shape[1])
        de.train_member(member, std, z[tr], _obs(st, tr), z[sr], _obs(st, sr), n_s=16,
                        n_e=8, batch=256, max_epochs=1, patience=3, lr=3e-3,
                        micro_batch=micro, log=lambda *_: None)  # fmt: skip
        params.append(torch.cat([p.detach().reshape(-1) for p in member.parameters()]))
    torch.testing.assert_close(params[0], params[1], rtol=1e-4, atol=1e-5)


def test_non_finite_gradients_are_skipped_and_parameters_stay_finite():
    """A step with a non-finite gradient is skipped instead of poisoning the model."""
    rng = np.random.default_rng(6)
    st = _stars(400, rng)
    z = torch.as_tensor(st["z"], dtype=torch.float32)
    std = _std(st["z"])
    tr, sr = np.arange(300), np.arange(300, 400)
    torch.manual_seed(0)
    member = _member("gmix", z.shape[1])
    before = [q.detach().clone() for q in member.parameters()]
    first = next(q for q in member.parameters() if q.requires_grad)
    first.register_hook(lambda g: g * float("nan"))
    hist = de.train_member(member, std, z[tr], _obs(st, tr), z[sr], _obs(st, sr), n_s=8,
                           n_e=4, batch=100, max_epochs=1, patience=3, lr=3e-3,
                           log=lambda *_: None)  # fmt: skip
    assert hist["skipped"] == [3]
    assert all(torch.isfinite(q).all() for q in member.parameters())
    for q, q0 in zip(member.parameters(), before):
        torch.testing.assert_close(q.detach(), q0)
    # An epoch whose every step was skipped is unhealthy: rolled back (to the
    # initial state here) and never taken as the best state.
    assert hist["rollbacks"] == 1 and hist["best_valid"] == math.inf


# ── S3 v2: fitted label noise, zero-point floor, XP-hidden mixing ───────────


def test_fitted_noise_recovers_overstated_e_errors_and_a_parallax_floor():
    """No label is truth: quoted E errors twice the real ones (spread over a
    decade, as on real data: that spread identifies the scale) plus 5 %
    outliers are recovered by the fitted kappa_E and outlier fraction, and an
    unmodelled parallax noise floor by the fitted sig_zp; the held-out E labels
    and parallaxes are then calibrated."""
    rng = np.random.default_rng(5)
    n = 10_000
    st = _stars(n, rng)
    floor = 0.05
    st["e_ebv"] = np.exp(rng.uniform(np.log(0.02), np.log(0.2), n))  # quoted
    real = 0.5 * st["e_ebv"] * np.where(rng.random(n) < 0.05, de.E_OUTLIER_SCALE, 1.0)
    st["ebv"] = st["E"] + real * rng.normal(size=n)
    st["plx"] = np.exp(-st["s"]) + np.sqrt(st["sig"] ** 2 + floor**2) * rng.normal(
        size=n
    )
    tr, sp, te = np.arange(0, 7000), np.arange(7000, 8000), np.arange(8000, n)
    torch.manual_seed(5)
    m = _member("gmix", st["z"].shape[1], fit_e_noise=True, fit_zp=True,
                e_loc=float(st["ebv"].mean()), e_scale=float(st["ebv"].std()))  # fmt: skip
    std, hist = _fit(m, st, tr, sp, with_e=True, epochs=10, batch=256)
    ke = m.kappa_e().detach().numpy()
    occupied = np.bincount(de.kappa_bin(st["G"][tr]), minlength=len(ke)) > 500
    assert np.all(np.abs(ke[occupied] - 0.5) < 0.15), ke
    # The floor is weakly identified (bright parallaxes are far tighter than the
    # predictive): it must move from its 0.005 mas start towards the truth.
    zp = float(m.zp_sigma().detach())
    assert 0.01 < zp < floor + 0.02, zp
    assert 0.02 < float(m.e_outlier_frac().detach()) < 0.1
    # Outliers have 4 x the real error = 2 x the quoted one: lambda ~ 2.
    assert 1.4 < float(m.e_outlier_lambda().detach()) < 3.0
    assert hist["kappa_e_final"] == pytest.approx(ke.tolist())
    ens = de.EnsembleDE([m.requires_grad_(False)], std)
    z = torch.as_tensor(st["z"][te], dtype=torch.float32)
    r = ens.predict(z, _obs(st, te), n_joint=32, n_e=32, n_s_lik=32)
    assert abs(np.mean(np.abs(r["pit_e"] - 0.5) <= 0.34) - 0.68) < 0.05
    assert abs(np.mean(np.abs(r["pit_plx"] - 0.5) <= 0.34) - 0.68) < 0.04
    # Old heads (no fitted noise terms) still load and give unit scales.
    old = _member("gmix", st["z"].shape[1])
    assert torch.allclose(old.kappa_e(), torch.ones(len(de.KAPPA_G_EDGES) + 1))
    assert float(old.zp_sigma(0.01)) == pytest.approx(0.01)


def test_xp_hidden_mixing_gives_one_head_calibrated_with_and_without_xp():
    """Trained on a random mix of full and information-poor ("XP hidden")
    latents with a 0/1 flag, one head is sharp on full latents and wider but
    still calibrated on the XP-hidden ones."""
    rng = np.random.default_rng(6)
    n = 12_000
    st = _stars(n, rng, d_noise=2)
    zx = st["z"].copy()
    zx[:, 1] = zx[:, 1] + rng.normal(0, 1.0, n)  # absolute magnitude much noisier
    zx[:, 2] = zx[:, 2] + rng.normal(0, 0.15, n)
    tr, sp, te = np.arange(0, 9000), np.arange(9000, 10_000), np.arange(10_000, n)
    torch.manual_seed(6)
    d = st["z"].shape[1]
    m = _member("gmix", d + 1, xp_flag=True, e_loc=float(st["ebv"].mean()),
                e_scale=float(st["ebv"].std()))  # fmt: skip
    std = _std(st["z"][tr])
    f32 = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
    hist = de.train_member(m, std, f32(st["z"][tr]), _obs(st, tr), f32(st["z"][sp]),
                           _obs(st, sp), n_s=32, batch=256, max_epochs=10, patience=3,
                           z_alt_train=f32(zx[tr]), z_alt_stop=f32(zx[sp]), alt_prob=0.5,
                           log=lambda *_: None)  # fmt: skip
    assert hist["alt_prob"] == 0.5
    ens = de.EnsembleDE([m.requires_grad_(False)], std)
    out = {}
    for name, lat, flag in (("full", st["z"], 0.0), ("noxp", zx, 1.0)):
        ens.flag_value = flag
        r = ens.predict(f32(lat[te]), _obs(st, te), n_joint=32, n_e=32, n_s_lik=32)
        out[name] = r
        cov = np.mean(np.abs(r["pit_plx"] - 0.5) <= 0.34)
        assert abs(cov - 0.68) < 0.05, (name, cov)
    hw = {k: np.median(0.5 * (r["s_q"][:, 2] - r["s_q"][:, 0])) for k, r in out.items()}
    assert hw["noxp"] > 1.5 * hw["full"], hw
    with pytest.raises(ValueError, match="xp_flag"):
        de.train_member(_member("gmix", d), std, f32(st["z"][tr[:300]]), _obs(st, tr[:300]),
                        f32(st["z"][sp[:100]]), _obs(st, sp[:100]), max_epochs=1,
                        z_alt_train=f32(zx[tr[:300]]), alt_prob=0.5, log=lambda *_: None)  # fmt: skip


def test_s3_script_v2_options_end_to_end(tmp_path, monkeypatch):
    """--e-noise fit, --fit-zp-sigma, --xp-mix and --train-members (parallel sessions)."""
    inp = _write_stand_in(tmp_path)
    mod = _load_script()
    out = tmp_path / "out"
    v2 = ("--families", "gmix", "--e-noise", "fit", "--fit-zp-sigma", "--xp-mix", "0.5")
    for members in ("0", "1"):  # two "sessions", one member each
        monkeypatch.setattr(sys, "argv", _args(
            tmp_path, inp, out, "--stages", "data,clusters,train", "--train-members", members,
            *v2))  # fmt: skip
        mod.main()
    assert {p.name for p in (out / "heads").glob("gmix_*.pt")} == {
        "gmix_0.pt",
        "gmix_1.pt",
    }
    monkeypatch.setattr(sys, "argv", _args(
        tmp_path, inp, out, "--stages", "predict,noxp,report", "--wait-for", str(out / "selection.npz"), *v2))  # fmt: skip
    mod.main()
    m = json.loads((out / "metrics.json").read_text())
    assert len(m["kappa_e"]["gmix"]) == 2 and len(m["zp_sigma"]["gmix"]) == 2
    assert set(m["scores"]) == {"gmix", "gmix_noxp"}
    report = (out / "report.md").read_text()
    assert "## Fitted E-label error scale kappa_E" in report
    assert "## Fitted parallax zero-point uncertainty" in report
    assert "XP-hidden latents with probability 0.5" in report


def test_heads_load_with_and_without_the_fitted_outlier_width(tmp_path):
    """v2 heads (no outlier-width parameter) and v2.1 heads both load."""
    std = _std(np.random.default_rng(0).normal(size=(50, 4)))
    for fit_scale in (False, True):
        m = _member("gmix", 4, fit_e_noise=True, fit_e_out_scale=fit_scale)
        cfg = dict(m.config)
        cfg.pop("fit_e_out_scale")  # configs saved before the key existed
        path = tmp_path / f"m{int(fit_scale)}.pt"
        torch.save(
            dict(config=cfg, state_dict=m.state_dict(), standardizer=std.state()), path
        )
        loaded, _ = de.load_member(path)
        assert hasattr(loaded, "e_out_scale_raw") == fit_scale
        e = torch.full((3,), 0.05)
        g = torch.zeros(3, dtype=torch.long)
        want = m.e_outlier_sigma(e, g)
        torch.testing.assert_close(loaded.e_outlier_sigma(e, g), want)
