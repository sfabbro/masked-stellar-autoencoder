"""Tests for the data-driven XP forward model (synthetic emulator, known covariances)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training import xp_forward_model as xf

CPU = torch.device("cpu")
BOUNDS = np.array([[4500.0, 6500.0], [1.5, 4.5], [-2.5, 0.3]])
BANDS = ("G", "J", "KS")
SMOKE = Path(
    "/tmp/claude-0/-home-user/d60e9c7a-7b13-5bb2-ade1-fb8da7adbc26/scratchpad/v9_smoke/finetune.fits"
)


def _true_model(nc: int = 6, seed: int = 0) -> xf.XPForwardModel:
    """A fixed random emulator with known scatter (2 % fractional, 0.005 floor)."""
    torch.manual_seed(seed)
    model = xf.XPForwardModel(
        BANDS,
        theta_loc=[5500.0, 3.0, -1.0],
        theta_scale=[600.0, 1.0, 0.8],
        xp_scale=np.ones(2 * nc),
        mag_loc=np.array([4.0, 2.5, 2.0]),
        mag_scale=np.array([1.5, 1.2, 1.2]),
        n_coeff=nc,
        config=xf.ModelConfig(hidden=32, depth=2, rank=1),
    )
    with torch.no_grad():
        for loss in model.xp_losses:
            loss.log_fraction.fill_(math.log(0.02))
            loss.log_floor.fill_(math.log(0.005))
            loss.factor_weight.zero_()
        model.log_mag_scatter.fill_(math.log(0.01))
        model.intrinsic[-1].bias.add_(
            torch.cat([torch.full((2 * nc,), 1.5), torch.zeros(3)])
        )
        # A realistic extinction signal: dc/dE ~ 0.2 per coefficient, and nearly
        # spectrum-independent A_b / E = (2.7, 0.8, 0.35) in (G, J, Ks).
        last = model.extinction[-1]
        last.weight[: 2 * nc] *= 4.0
        last.bias[: 2 * nc] *= 4.0
        last.weight[2 * nc :] *= 0.05
        a_b = torch.tensor([2.7, 0.8, 0.35])
        last.bias[2 * nc :] = torch.log(torch.expm1(a_b))
    return model.double().eval()


def _simulate(model, n, seed=1, dense=True, noise=(0.5, 3.0), sfd_factor=1.25,
              label_sigma=0.03, unlabelled=0.25):  # fmt: skip
    """Stars drawn from ``model`` with per-star known covariances (S/N varies by 6x).

    ``EBV`` (the SFD stand-in) is ``sfd_factor`` times the true ``E``; reddening labels
    are ``E + N(0, label_sigma)`` with a fraction ``unlabelled`` missing.
    """
    rng = np.random.default_rng(seed)
    nc = model.n_coeff
    span = BOUNDS[:, 1] - BOUNDS[:, 0]
    theta = rng.uniform(
        BOUNDS[:, 0] + 0.1 * span, BOUNDS[:, 1] - 0.1 * span, size=(n, 3)
    )
    ebv = rng.uniform(0.0, 0.5, n)
    plx = rng.uniform(0.5, 5.0, n)
    plx_err = 0.08 * plx
    with torch.no_grad():
        xp, mag = model(torch.as_tensor(theta), torch.as_tensor(ebv))
    xp, mag = xp.numpy(), mag.numpy()
    level = rng.uniform(*noise, n)
    covs = np.zeros((n, 2, nc, nc))
    obs = np.zeros((n, 2 * nc))
    for i in range(n):
        for k in range(2):
            c = xp[i, k * nc : (k + 1) * nc]
            sd = level[i] * (0.01 + 0.01 * np.abs(c))
            r = np.eye(nc)
            if dense:
                a = rng.normal(size=(nc, nc))
                r = a @ a.T + nc * np.eye(nc)
                d = np.sqrt(np.diag(r))
                r = r / d[:, None] / d[None]
            covs[i, k] = sd[:, None] * sd[None] * r
            model_sd = np.sqrt((0.02 * c) ** 2 + 0.005**2)
            obs[i, k * nc : (k + 1) * nc] = (
                c
                + np.linalg.cholesky(covs[i, k]) @ rng.normal(size=nc)
                + model_sd * rng.normal(size=nc)
            )
    mag_err = rng.uniform(0.01, 0.05, (n, 3))
    m_obs = (
        mag
        + 5 * np.log10(100 / plx)[:, None]
        + np.sqrt(mag_err**2 + 0.01**2) * rng.normal(size=(n, 3))
    )
    diag = np.concatenate(
        [np.diagonal(covs[:, k], axis1=1, axis2=2) for k in range(2)], 1
    )
    missing = rng.random(n) < unlabelled
    data = xf.StarData(
        rows=np.arange(n),
        source_id=np.arange(n) + 1000,
        xp=obs.astype(np.float32),
        xp_mask=np.ones((n, 2 * nc), bool),
        xp_cov=xf.pack_covariance(covs).astype(np.float32) if dense else None,
        xp_var=None if dense else diag.astype(np.float32),
        mag=m_obs.astype(np.float32),
        mag_err=mag_err.astype(np.float32),
        mag_mask=np.ones((n, 3), bool),
        parallax=plx + plx_err * rng.normal(size=n),
        parallax_err=plx_err,
        parallax_zp=np.zeros(n),
        g_mag=m_obs[:, 0].astype(np.float64),
        ebv_sfd=sfd_factor * ebv,
        theta=theta,
        theta_err=np.tile([50.0, 0.1, 0.05], (n, 1)),
        band_names=BANDS,
        ebv_label=np.where(missing, np.nan, ebv + label_sigma * rng.normal(size=n)),
        ebv_label_err=np.where(missing, np.nan, label_sigma),
        split=np.zeros(n, np.int8),
    )
    return data, {"theta": theta, "ebv": ebv, "parallax": plx, "xp": xp, "mag": mag}


def _coverage(res, theta):
    in68 = (theta >= res["lo68"]) & (theta <= res["hi68"])
    in95 = (theta >= res["lo95"]) & (theta <= res["hi95"])
    return float(in68.mean()), float(in95.mean())


def _small_inference(n):
    return xf.InferenceConfig(
        chunk_rows=n,
        n_candidates=32,
        bank_theta=256,
        steps=30,
        profile_scan=4,
        profile_steps=10,
    )


def test_nuisance_box_and_unit_round_trip() -> None:
    sfd = np.array([0.0, 0.1, -0.2, np.nan])
    plx = np.array([2.0, 0.01, -0.5, np.nan])
    err = np.array([0.02, 0.5, 0.3, np.nan])
    lo, hi, e_max = xf.nuisance_box(sfd, plx, err)
    np.testing.assert_allclose(e_max, [0.05, 0.17, 0.05, 0.05])
    np.testing.assert_allclose(10 ** hi[0], 2.0 + 6 * 0.02)
    np.testing.assert_allclose(10 ** lo[0], 2.0 - 6 * 0.02)
    assert np.all(10**lo >= xf.PARALLAX_MIN_MAS - 1e-12) and np.all(hi > lo)
    np.testing.assert_allclose(10 ** hi[3], xf.PARALLAX_MAX_MAS)
    u = xf.nuisance_to_unit(
        np.array([0.02, 0.1, 0.0, 0.03]), np.array([2.0, 0.3, 0.1, 1.0]), lo, hi, e_max
    )
    batch = {
        k: torch.as_tensor(v)
        for k, v in (("e_max", e_max), ("log_plx_lo", lo), ("log_plx_hi", hi))
    }
    ebv, parallax = xf.unit_to_nuisance(torch.as_tensor(u), batch)
    np.testing.assert_allclose(ebv.numpy(), [0.02, 0.1, 0.0, 0.03], atol=1e-12)
    np.testing.assert_allclose(parallax.numpy()[[0, 1, 3]], [2.0, 0.3, 1.0], rtol=1e-10)
    # E may go below 0 down to E_MIN (zero-point uncertainty), u = 0 maps there.
    low = xf.unit_to_nuisance(torch.zeros(4, 2, dtype=torch.float64), batch)[0]
    np.testing.assert_allclose(low.numpy(), xf.E_MIN)
    assert (
        xf.nuisance_to_unit(
            np.array([-0.02]), np.array([1.0]), lo[:1], hi[:1], e_max[:1]
        )[0, 0]
        > 0
    )


def test_diagonal_fallback_equals_dense_diagonal_covariance() -> None:
    model = _true_model()
    dense, truth = _simulate(model, 20, dense=False)
    covs = np.zeros((20, 2, 6, 6))
    var = dense.xp_var.astype(np.float64)
    for k in range(2):
        covs[:, k] = np.einsum("ni,ij->nij", var[:, k * 6 : (k + 1) * 6], np.eye(6))
    dense_cov = dense.subset(np.arange(20))
    dense_cov.xp_cov, dense_cov.xp_var = (
        xf.pack_covariance(covs).astype(np.float32),
        None,
    )
    theta = torch.as_tensor(truth["theta"])
    ebv = torch.as_tensor(truth["ebv"])
    plx = torch.as_tensor(truth["parallax"])
    a = xf.star_nll(
        model, theta, ebv, plx, dense.batch(np.arange(20), CPU, torch.float64)
    )
    b = xf.star_nll(
        model, theta, ebv, plx, dense_cov.batch(np.arange(20), CPU, torch.float64)
    )
    torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)


def test_masked_bands_and_coefficients_are_marginalised() -> None:
    model = _true_model()
    data, truth = _simulate(model, 10)
    theta = torch.as_tensor(truth["theta"])
    ebv = torch.as_tensor(truth["ebv"])
    plx = torch.as_tensor(truth["parallax"])
    full = xf.star_nll(
        model,
        theta,
        ebv,
        plx,
        data.batch(np.arange(10), CPU, torch.float64),
        parts=True,
    )
    masked = data.subset(np.arange(10))
    masked.mag_mask = masked.mag_mask.copy()
    masked.mag_mask[:, 1] = False  # J missing everywhere
    masked.mag = masked.mag.copy()
    masked.mag[:, 1] = np.nan
    batch = masked.batch(np.arange(10), CPU, torch.float64)
    part = xf.star_nll(model, theta, ebv, plx, batch, parts=True)
    # Diagonal photometry: removing a band subtracts exactly its Gaussian term.
    with torch.no_grad():
        _, mag_hat = model(theta, ebv)
    resid = torch.as_tensor(data.mag[:, 1], dtype=torch.float64) - (
        mag_hat[:, 1] + 5 * torch.log10(100 / plx)
    )
    var = torch.as_tensor(data.mag_err[:, 1], dtype=torch.float64) ** 2 + 0.01**2
    term = 0.5 * (resid**2 / var + torch.log(var) + math.log(2 * math.pi))
    torch.testing.assert_close(
        part["nll_phot"], full["nll_phot"] - term, atol=1e-6, rtol=1e-6
    )
    assert torch.equal(part["dof_phot"], full["dof_phot"] - 1)
    # Missing XP coefficients with NaN values: finite NLL and gradients, dof reduced.
    masked.xp_mask = masked.xp_mask.copy()
    masked.xp = masked.xp.copy()
    masked.xp_mask[:3, 2] = False
    masked.xp[:3, 2] = np.nan
    masked.xp_mask[5, 6:] = False  # whole RP band missing
    masked.xp[5, 6:] = np.nan
    batch = masked.batch(np.arange(10), CPU, torch.float64)
    th = theta.clone().requires_grad_(True)
    out = xf.star_nll(model, th, ebv, plx, batch, parts=True)
    out["nll"].sum().backward()
    assert torch.isfinite(out["nll"]).all() and torch.isfinite(th.grad).all()
    assert out["dof_xp"][0] == 11 and out["dof_xp"][5] == 6 and out["dof_xp"][9] == 12


def test_label_noise_hook() -> None:
    model = _true_model().float()
    data, _ = _simulate(model.double(), 16)
    model = model.float()
    batch = data.batch(np.arange(16), CPU, torch.float32)
    ebv, plx = xf.unit_to_nuisance(torch.full((16, 2), 0.5), batch)
    gen = torch.Generator().manual_seed(0)
    marginal = xf.TrainConfig(label_noise="marginal", label_noise_samples=4)
    plug = xf.batch_loss(model, batch, ebv, plx, xf.TrainConfig(), gen)
    marg = xf.batch_loss(model, batch, ebv, plx, marginal, gen)
    assert plug.shape == marg.shape == (16,) and torch.isfinite(marg).all()
    batch["theta_err"] = torch.zeros_like(batch["theta_err"])
    exact = xf.batch_loss(model, batch, ebv, plx, marginal, gen)
    torch.testing.assert_close(exact, plug)
    with pytest.raises(ValueError, match="label_noise"):
        xf.batch_loss(model, batch, ebv, plx, xf.TrainConfig(label_noise="bogus"), gen)


def test_prepared_likelihood_matches_torchregress() -> None:
    """The fixed-factor fast path equals the torchregress training loss, masks included."""
    model = _true_model()
    data, truth = _simulate(model, 12)
    data.xp_mask = data.xp_mask.copy()
    data.xp_mask[:2, 3] = False
    data.xp_mask[4, 6:] = False
    data.mag_mask = data.mag_mask.copy()
    data.mag_mask[1, 2] = False
    data.parallax_err = data.parallax_err.copy()
    data.parallax_err[3] = np.nan  # no parallax
    batch = data.batch(np.arange(12), CPU, torch.float64)
    theta = torch.as_tensor(truth["theta"])
    ebv = torch.as_tensor(truth["ebv"]) + 0.03
    plx = torch.as_tensor(truth["parallax"]) * 1.05
    ref = xf.star_nll(model, theta, ebv, plx, batch, parts=True)
    reference = xf.model_reference(model, theta, ebv)
    prep = xf.prepare_stars(model, batch, reference=reference)
    fast = xf.star_nll_prepared(model, theta, ebv, plx, prep, parts=True)
    for key in ("nll", "chi2_xp", "dof_xp", "nll_phot", "nll_parallax"):
        torch.testing.assert_close(fast[key], ref[key], rtol=1e-9, atol=1e-8)
    prep32 = xf.prepare_stars(model, batch, reference=reference, dtype=torch.float32)
    m32 = copy.deepcopy(model).float()
    fast32 = xf.star_nll_prepared(m32, theta.float(), ebv.float(), plx.float(), prep32)
    torch.testing.assert_close(fast32.double(), ref["nll"], rtol=1e-4, atol=1e-2)


def test_fit_nuisances_recovers_extinction_and_distance() -> None:
    """Projected Gauss-Newton at the true theta: reaches the profile optimum of the
    data likelihood (as found by multistart Newton) and recovers E and the parallax."""
    model = _true_model()
    data, truth = _simulate(model, 200, seed=7)
    batch = data.batch(np.arange(200), CPU, torch.float64)
    theta = batch["theta"]
    ref_e = torch.as_tensor(truth["ebv"])
    prep = xf.prepare_stars(
        model, batch, reference=xf.model_reference(model, theta, ref_e)
    )
    ebv0 = prep.e_max.clone()  # deliberately poor start: the upper bound
    mu0 = xf.distance_modulus_start(model, theta, ebv0, prep)
    ebv, mu, f = xf.fit_nuisances(model, theta, prep, ebv0, mu0, steps=12, anchor=False)
    plx = 100.0 * 10.0 ** (-mu.numpy() / 5.0)
    err_e = ebv.numpy() - truth["ebv"]
    assert np.median(np.abs(err_e)) < 0.02, np.median(np.abs(err_e))
    assert abs(np.median(err_e)) < 0.01
    assert np.median(np.abs(plx / truth["parallax"] - 1)) < 0.05
    # Same optimum as torchregress' multistart projected Newton on the same score
    # (same fixed covariance), from the poor start.
    x = torch.arange(200, dtype=torch.float64).unsqueeze(1)
    res = xf.maximize_score(xf._score_fn(model, prep, fixed_theta=theta), x,
                            [[0.0, 1.0], [0.0, 1.0]], n_candidates=32, n_starts=3, steps=50)  # fmt: skip
    with torch.no_grad():
        nll_gn = xf.star_nll_prepared(
            model, theta, ebv, 100.0 * torch.pow(10.0, -mu / 5.0), prep
        )
    gap = nll_gn.numpy() - (-res.score.numpy())
    assert np.quantile(gap, 0.95) < 1e-3, np.sort(gap)[-5:]
    # The validation profile (three starts, covariance refreshed at its end point)
    # recovers E as well.
    prof = xf.profile_nuisance_at_labels(model, data, device=CPU, steps=12)
    assert np.median(np.abs(prof["ebv"] - truth["ebv"])) < 0.02
    # The anchor pulls E toward the labels, by their weight only.
    ea, _, _ = xf.fit_nuisances(model, theta, prep, ebv0, mu0, steps=12, anchor=True)
    lab = data.anchored()
    d_free = np.abs(ebv.numpy() - data.ebv_label)[lab]
    d_anch = np.abs(ea.numpy() - data.ebv_label)[lab]
    assert np.median(d_anch) <= np.median(d_free) + 1e-9
    np.testing.assert_allclose(ea.numpy()[~lab], ebv.numpy()[~lab], atol=1e-6)


def test_extinction_gauge_is_degenerate_without_labels() -> None:
    """The old parametrisation's degeneracy, exactly: E -> a E with the extinction
    network rescaled leaves every data likelihood unchanged; the anchor breaks it."""
    model = _true_model()
    data, truth = _simulate(model, 50, seed=3)
    a = 2.0
    gauge = copy.deepcopy(model)
    with torch.no_grad():  # h1(t, E) -> h1(t, E / a) / a, i.e. c(theta, a E) unchanged
        first = gauge.extinction[0]
        first.weight[:, 3] /= a
        last = gauge.extinction[-1]
        n = 2 * gauge.n_coeff
        last.weight[:n] /= a
        last.bias[:n] /= a
    theta = torch.as_tensor(truth["theta"])
    ebv = torch.as_tensor(truth["ebv"])
    plx = torch.as_tensor(truth["parallax"])
    batch = data.batch(np.arange(50), CPU, torch.float64)
    with torch.no_grad():
        xp1, _ = model(theta, ebv)
        xp2, _ = gauge(theta, a * ebv)
        nll1 = xf.star_nll(model, theta, ebv, plx, batch, parts=True)["nll_xp"]
        nll2 = xf.star_nll(gauge, theta, a * ebv, plx, batch, parts=True)["nll_xp"]
    torch.testing.assert_close(xp1, xp2)
    torch.testing.assert_close(nll1, nll2)  # the XP data cannot tell the two apart
    lab, err = batch["ebv_label"], batch["ebv_label_err"]
    assert (
        xf.anchor_nll(a * ebv, lab, err).sum() - xf.anchor_nll(ebv, lab, err).sum()
    ) > 100


@pytest.fixture(scope="module")
def trained() -> dict:
    """Emulators trained on the same synthetic stars, whose SFD stand-in is twice the
    true E: ``anchored`` (reddening labels, E fitted per star after one warm-up
    epoch) and ``sfd_gauge`` (no labels, E held at its SFD start: the first run's
    regime, where one sparse step per epoch left E near SFD)."""
    truth_model = _true_model()
    kw = dict(noise=(0.3, 1.5), sfd_factor=2.0)
    train, _ = _simulate(truth_model, 2500, seed=2, **kw)
    valid, _ = _simulate(truth_model, 100, seed=3, **kw)
    test, truth = _simulate(truth_model, 40, seed=4, **kw)
    out = {"truth_model": truth_model, "train": train, "test": test, "truth": truth}
    epochs = 24
    for name, anchor, warmup in (("anchored", True, 1), ("sfd_gauge", False, epochs)):
        torch.manual_seed(0)
        model = xf.build_model(train, xf.ModelConfig(hidden=64, depth=2, rank=1))
        cfg = xf.TrainConfig(epochs=epochs, batch_size=128, lr=3e-3, scatter_lr=1e-2, valid_rows=100,
                             log_every=0, anchor=anchor, nuisance_warmup_epochs=warmup)  # fmt: skip
        hist = xf.train_forward_model(
            model, train, valid, cfg, device=CPU, log=lambda s: None
        )
        out[name] = (model.double().eval(), hist)
    return out


def _extinction_scale(model, truth_model, theta, ebv) -> tuple[float, np.ndarray]:
    """Least-squares ratio of fitted to true dc/dE (table units), and the ratios of
    the colour-excess coefficients d(M_G - M_b)/dE (the grey part of A_b trades
    with the distance and is constrained only by the parallaxes)."""
    th, e = torch.as_tensor(theta), torch.as_tensor(ebv)
    dxp_f, dmag_f = model.extinction_coefficients(th, e)
    dxp_t, dmag_t = truth_model.extinction_coefficients(th, e)
    dxp_f = dxp_f * model.xp_scale.double()
    s_xp = float((dxp_f * dxp_t).sum() / dxp_t.square().sum())
    colour_f = (dmag_f[:, :1] - dmag_f[:, 1:]).mean(0)
    colour_t = (dmag_t[:, :1] - dmag_t[:, 1:]).mean(0)
    return s_xp, (colour_f / colour_t).numpy()


def test_training_with_labels_recovers_the_extinction_coefficients(trained) -> None:
    model, hist = trained["anchored"]
    truth, tm = trained["truth"], trained["truth_model"]
    epochs = hist["epochs"]
    assert epochs[0]["scatter_frozen"] and not epochs[-1]["scatter_frozen"]
    assert epochs[-1]["train_nll"] < epochs[0]["train_nll"] - 50
    assert 0.6 < epochs[-1]["valid_chi2_xp_per_dof"] < 1.3
    assert 0.7 < hist["anchored_fraction"] < 0.8
    frac = model.describe()["bp_fraction_median"]
    assert 0.012 < frac < 0.03, frac  # true fractional scatter 0.02
    s_xp, s_colour = _extinction_scale(model, tm, truth["theta"], np.full(40, 0.2))
    assert abs(s_xp - 1) < 0.1, s_xp
    assert np.all(np.abs(s_colour - 1) < 0.1), s_colour
    # Training nuisances: E of the training stars on the labels' scale.
    train = trained["train"]
    e_fit = hist["train_nuisance"]["ebv"]
    ok = train.anchored()
    assert abs(np.median(e_fit[ok] - train.ebv_label[ok])) < 0.01
    assert abs(np.polyfit(train.ebv_label[ok], e_fit[ok], 1)[0] - 1) < 0.1
    # Validation stars: E profiled from the data alone matches their labels.
    rec = epochs[-1]["valid_ebv_vs_label"]
    assert abs(rec["median_offset"]) < 0.02 and abs(rec["slope"] - 1) < 0.15, rec
    # Emulator spectra within the noise.
    with torch.no_grad():
        xp_hat, _ = model(
            torch.as_tensor(truth["theta"]), torch.as_tensor(truth["ebv"])
        )
    err = np.abs(xp_hat.numpy() * model.xp_scale.numpy() - truth["xp"])
    test = trained["test"]
    var = np.concatenate(
        [np.diagonal(xf.unpack_covariance(torch.as_tensor(test.xp_cov))[:, k].numpy(),
                     axis1=1, axis2=2) for k in range(2)], 1,
    )  # fmt: skip
    noise = np.sqrt(var + (0.02 * truth["xp"]) ** 2 + 0.005**2)
    assert np.median(err / noise) < 1.0, np.median(err / noise)


def test_training_without_labels_inherits_the_sfd_gauge(trained) -> None:
    """Without the anchor, the E scale is whatever fixes the gauge: here the SFD
    start (SFD = 2 E), so every extinction coefficient comes out at half its value
    (the data fit is as good: the gauge is invisible to the likelihood)."""
    model, hist = trained["sfd_gauge"]
    s_xp, s_colour = _extinction_scale(model, trained["truth_model"], trained["truth"]["theta"],
                                       np.full(40, 0.2))  # fmt: skip
    assert 0.35 < s_xp < 0.65, s_xp
    assert np.all((s_colour > 0.35) & (s_colour < 0.65)), s_colour
    assert hist["epochs"][-1]["valid_chi2_xp_per_dof"] < 1.3


def test_inference_recovers_extinction_and_covers(trained) -> None:
    model, _ = trained["anchored"]
    test, truth, train = trained["test"], trained["truth"], trained["train"]
    bounds = xf.theta_bounds_from_labels(train.theta)
    res = xf.infer_stars(model, test, bounds, train.theta, _small_inference(40), device=CPU,
                         log=lambda s: None)  # fmt: skip
    cov68, cov95 = _coverage(res, truth["theta"])
    assert 0.5 <= cov68 <= 0.88, cov68  # 120 intervals
    assert cov95 >= 0.88, cov95
    d = res["E_mle"] - truth["ebv"]
    assert abs(np.median(d)) < 0.02 and np.median(np.abs(d)) < 0.04, d
    assert (res["E_bound"] != 0).mean() < 0.2
    # The SFD-gauge emulator measures E in its own units: twice too large.
    res_g = xf.infer_stars(trained["sfd_gauge"][0], test, bounds, train.theta,
                           _small_inference(40), device=CPU, log=lambda s: None)  # fmt: skip
    ok = truth["ebv"] > 0.15
    ratio = np.median(res_g["E_mle"][ok] / truth["ebv"][ok])
    assert ratio > 1.6, ratio


def test_inference_with_the_true_emulator_covers_at_nominal_rate() -> None:
    model = _true_model()
    data, truth = _simulate(model, 48)
    pool = np.random.default_rng(5).uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(2000, 3))
    res = xf.infer_stars(
        model, data, BOUNDS, pool, _small_inference(48), device=CPU, log=lambda s: None
    )
    cov68, cov95 = _coverage(res, truth["theta"])
    assert 0.55 <= cov68 <= 0.82, cov68  # 144 intervals: binomial sd 0.04
    assert 0.90 <= cov95 <= 1.0, cov95
    assert np.all(res["lo68"] <= res["mle"] + 1e-9) and np.all(
        res["mle"] <= res["hi68"] + 1e-9
    )
    assert np.all(res["lo95"] <= res["lo68"] + 1e-9) and np.all(
        res["hi68"] <= res["hi95"] + 1e-9
    )
    assert np.median(np.abs(res["parallax_mle"] / truth["parallax"] - 1)) < 0.08
    assert np.all(res["E_mle"] <= 1.2 * data.ebv_sfd + 0.05 + 1e-9)
    assert np.all(res["E_mle"] >= xf.E_MIN - 1e-9)
    assert np.median(np.abs(res["E_mle"] - truth["ebv"])) < 0.05
    assert res["dof_xp"].min() == 12 and np.isfinite(res["chi2_xp"]).all()
    # float32 inference (factors from float64) agrees with float64.
    res32 = xf.infer_stars(
        model.float(), data, BOUNDS, pool, _small_inference(48), device=CPU,
        dtype=torch.float32, log=lambda s: None,
    )  # fmt: skip
    model.double()
    width = res["hi68"] - res["lo68"]
    assert np.median(np.abs(res32["mle"] - res["mle"]) / width) < 0.05
    assert np.median(np.abs((res32["hi68"] - res32["lo68"]) / width - 1)) < 0.05


def test_local_profile_intervals_match_the_global_box() -> None:
    """The per-star local box only moves the scan points: intervals equal those of
    the global box, also when a local edge is hit (0.5 sigma box: global fallback)."""
    model = _true_model()
    data, truth = _simulate(model, 24, seed=9)
    batch = data.batch(np.arange(24), CPU, torch.float64)
    prep = xf.prepare_stars(model, batch, reference=xf.observed_reference(model, batch))
    score = xf._score_fn(model, prep)
    x = torch.arange(24, dtype=torch.float64).unsqueeze(1)
    bounds = np.concatenate([BOUNDS, [[0.0, 1.0], [0.0, 1.0]]])
    pool = np.random.default_rng(5).uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(2000, 3))
    bank = torch.as_tensor(xf.candidate_bank(pool, 256, (0.0, 0.1, 0.25, 0.5), 0))
    init = xf.coarse_init(model, prep, bank)
    res = xf.maximize_score(
        score, x, bounds, n_candidates=8, n_starts=1, steps=40, init=init
    )
    thr = [xf.wilks_threshold(0.6827), xf.wilks_threshold(0.95)]
    cfg = xf.InferenceConfig(profile_tol=1e-6)
    ref = xf.profile_intervals(score, x, res.theta, res.score, bounds, thr,
                               xf.InferenceConfig(profile_local_sigma=0.0, profile_tol=1e-6))  # fmt: skip
    cov = xf.inverse_information(xf.observed_information(score, res.theta, x))
    for sigma in (8.0, 0.5):
        cfg.profile_local_sigma = sigma
        got = xf.profile_intervals(
            score, x, res.theta, res.score, bounds, thr, cfg, covariance=cov
        )
        for (lo_r, hi_r, b_r), (lo_g, hi_g, b_g) in zip(ref, got, strict=True):
            width = (hi_r - lo_r).clamp_min(1e-9)
            assert float(((lo_g - lo_r).abs() / width).max()) < 1e-3
            assert float(((hi_g - hi_r).abs() / width).max()) < 1e-3
            assert torch.equal(b_r, b_g)
    # The inverse observed information approximates the 68 % profile half-widths.
    half = 0.5 * (ref[0][1] - ref[0][0])
    sd = torch.diagonal(cov, dim1=1, dim2=2)[:, :3].sqrt()
    ratio = (sd / half).numpy()
    assert 0.8 < np.median(ratio) < 1.25, np.median(ratio, axis=0)


def test_select_rows_keeps_held_out_core_out() -> None:
    n = 400
    rng = np.random.default_rng(0)
    split = rng.integers(0, 3, n).astype(np.int8)
    split[:6] = [0, 0, 1, 2, 1, -1]
    labels = {"table_row": np.arange(n), "split": split,
              "teff": rng.uniform(4000, 6000, n), "logg": rng.uniform(1, 5, n),
              "fe_h": rng.uniform(-3, 0.3, n), "has_core_feh": np.zeros(n, bool)}  # fmt: skip
    labels["teff"][7] = np.nan  # unlabelled
    core = np.array([0, 1, 2, 3, 4, 5, 7])
    labels["has_core_feh"][[10, 11]] = True  # core by flag only
    sel = xf.select_rows(labels, core, max_train=50, max_valid=30, max_heldout=40)
    core_all = set(core) | {10, 11}
    held_core = {r for r in core_all if split[r] in (1, 2)}
    train_core = {r for r in core_all if split[r] == 0 and r != 7}
    assert train_core <= set(sel["train"])  # every train-split core star trains
    assert not held_core & set(sel["train"])
    assert not core_all & (set(sel["valid"]) | set(sel["heldout"]))
    assert (
        len(sel["train"]) == 50
        and len(sel["valid"]) == 30
        and len(sel["heldout"]) == 40
    )
    assert (split[sel["train"]] == 0).all() and (split[sel["valid"]] == 1).all()
    assert (split[sel["heldout"]] == 2).all()
    np.testing.assert_array_equal(sel["core"], np.unique(core))
    np.testing.assert_array_equal(sel["core_train"], sorted(train_core))


def test_reddening_labels_lookup(tmp_path) -> None:
    sid = np.array([30, 10, 20, 40], np.int64)
    np.savez(tmp_path / "r.npz", source_id=sid, ebv_star=np.array([0.3, 0.1, np.nan, 0.4], "f4"),
             e_ebv_star=np.array([0.03, 0.02, 0.02, -1.0], "f4"))  # fmt: skip
    red = xf.ReddeningLabels(tmp_path / "r.npz")
    ebv, err = red.lookup(np.array([10, 20, 30, 40, 50]))
    np.testing.assert_allclose(ebv[[0, 2]], [0.1, 0.3], rtol=1e-6)
    assert np.isnan(ebv[[1, 3, 4]]).all() and np.isnan(err[[1, 3, 4]]).all()
    assert red.summary()["labelled"] == 2


def test_checkpoint_round_trip(tmp_path) -> None:
    model = _true_model().float()
    xf.save_model(model, tmp_path / "m.pt", extra={"theta_bounds": BOUNDS})
    loaded, extra = xf.load_model(tmp_path / "m.pt")
    th = torch.as_tensor([[5000.0, 3.0, -1.0]])
    e = torch.as_tensor([0.2])
    for a, b in zip(model(th, e), loaded(th, e), strict=True):
        torch.testing.assert_close(a, b)
    np.testing.assert_array_equal(extra["theta_bounds"], BOUNDS)


# ─── table and sidecar I/O ──────────────────────────────────────────────────


def _write_table(path: Path, n: int = 150, seed: int = 0) -> dict:
    """A small finetune.fits look-alike with physically plausible XP and photometry."""
    from astropy.table import Table

    rng = np.random.default_rng(seed)
    model = _true_model(nc=xf.N_COEFF, seed=seed)
    data, truth = _simulate(model, n, seed=seed, dense=False, noise=(0.5, 2.0))
    g = rng.uniform(10.0, 17.0, n)
    f = 10 ** ((xf.G_REF - g) / 2.5)
    cols = {
        "source_id": np.arange(n, dtype=np.int64) + 10**15,
        "G": g.astype(np.float32),
    }
    for k, name in enumerate(xf.XP_COLUMNS):
        cols[name] = data.xp[:, k]
    raw_err = np.sqrt(data.xp_var.astype(np.float64)) * f[:, None]  # raw archive units
    for k, name in enumerate(xf.XP_ERROR_COLUMNS):
        cols[name] = raw_err[:, k].astype(np.float32)
    for b in xf.DEFAULT_BANDS:
        if b.name != "G":  # G is the XP normalisation magnitude written above
            cols[b.name] = (data.mag[:, 1] + rng.normal(0, 0.3, n)).astype(np.float32)
        if b.error:
            cols[b.error] = np.full(n, 0.03, np.float32)
    cols["J"][:5] = np.nan
    cols["PARALLAX"] = data.parallax.astype(np.float32)
    cols["e_parallax"] = data.parallax_err.astype(np.float32)
    cols["EBV"] = data.ebv_sfd.astype(np.float32)
    for k, p in enumerate(xf.PARAMS):
        values = truth["theta"][:, k].copy()
        values[-10:] = np.nan  # unlabelled rows
        cols[p] = values.astype(np.float32)
        cols[f"e_{p}"] = np.full(n, [50.0, 0.1, 0.05][k], np.float32)
    Table(cols).write(path, overwrite=True)
    return {"g": g, "raw_err": raw_err, "data": data}


def test_table_io_and_diagonal_raw_error_conversion(tmp_path) -> None:
    info = _write_table(tmp_path / "t.fits")
    rows = np.array([7, 3, 120, 3])
    out = xf.read_table_rows(tmp_path / "t.fits", rows, ["source_id", "G", "bp_1"])
    np.testing.assert_array_equal(out["source_id"], rows + 10**15)
    labels = xf.labels_from_table(tmp_path / "t.fits", core_rows=np.array([1, 2]))
    assert set(xf.LABEL_FIELDS) <= set(labels)
    assert set(np.unique(labels["split"])) <= {0, 1, 2}
    assert labels["has_core_feh"][[1, 2]].all() and labels["has_core_feh"].sum() == 2
    data, diag = xf.load_star_data(
        tmp_path / "t.fits",
        rows,
        labels=labels,
        zero_point="constant",
        log=lambda s: None,
    )
    expected = (
        info["raw_err"][rows] / 10 ** ((xf.G_REF - info["g"][rows]) / 2.5)[:, None]
    ) ** 2
    np.testing.assert_allclose(data.xp_var, expected, rtol=2e-6)
    np.testing.assert_allclose(data.parallax_zp, xf.L21_CONSTANT_ZP)
    expected_theta = [labels[p][7] for p in xf.PARAMS]
    np.testing.assert_allclose(data.theta[0], expected_theta, rtol=1e-6)
    ps1 = xf.read_table_rows(tmp_path / "t.fits", rows, ["G_PS1"])["G_PS1"]
    k = data.band_names.index("G_PS1")
    np.testing.assert_array_equal(data.mag_mask[:, k], ps1 >= 14.0)  # saturation limit
    j = data.band_names.index("J")
    assert not data.mag_mask[3, j]  # row 3 has NaN J
    assert diag["xp_covariance"].startswith("diagonal")
    with pytest.raises(ValueError, match="sidecar"):
        xf.load_star_data(
            tmp_path / "t.fits", rows, zero_point="sidecar", log=lambda s: None
        )


def _write_sidecar(path: Path, n: int, g: np.ndarray, seed: int = 0) -> dict:
    import h5py

    from masked_stellar_autoencoder.data import covariance_sidecar as cs

    rng = np.random.default_rng(seed)
    nc = cs.N_COEFF
    out = {}
    with h5py.File(path, "w") as f:
        f.attrs["format_version"] = cs.FORMAT_VERSION
        f["source_id"] = np.arange(n, dtype=np.int64) + 10**15
        f["row_index"] = np.arange(n, dtype=np.int64)
        f["found_xp"] = np.ones(n, bool)
        f["found_astrometry"] = np.ones(n, bool)
        for band in ("bp", "rp"):
            a = rng.normal(size=(n, nc, 3))
            corr = np.einsum("nik,njk->nij", a, a) + 3 * np.eye(nc)
            d = np.sqrt(np.einsum("nii->ni", corr))
            corr = corr / d[:, :, None] / d[:, None, :]
            f[f"{band}_coefficient_correlations"] = cs.pack_correlations(corr).astype(
                np.float32
            )
            err = rng.uniform(1.0, 5.0, (n, nc)).astype(np.float32)
            f[f"{band}_coefficient_errors"] = err
            f[f"{band}_standard_deviation"] = np.full(n, 1.2, np.float32)
            f[f"{band}_coefficients"] = rng.normal(size=(n, nc))
            out[band] = (err, corr)
        f["phot_g_mean_mag"] = g.astype(np.float32)
        f["phot_g_mean_flux_over_error"] = np.full(n, 1000.0, np.float32)
        f["parallax"] = np.linspace(0.5, 2.0, n)
        f["parallax_error"] = np.full(n, 0.05, np.float32)
        f["parallax_zpt_l21"] = np.full(n, -0.03, np.float32)
    return out


def test_sidecar_covariances_and_zero_point(tmp_path) -> None:
    info = _write_table(tmp_path / "t.fits", n=40)
    side = _write_sidecar(tmp_path / "s.h5", 40, info["g"])
    sidecar = xf.SidecarXPCovariance(tmp_path / "s.h5")
    rows = np.array([5, 1, 30])
    data, diag = xf.load_star_data(
        tmp_path / "t.fits", rows, sidecar=sidecar, log=lambda s: None
    )
    batch = data.batch(np.arange(3), CPU, torch.float64)
    f = 10 ** ((xf.G_REF - info["g"][rows].astype(np.float32).astype(np.float64)) / 2.5)
    err, corr = side["rp"]
    expected = (
        err[rows, :, None] * err[rows, None, :] * corr[rows] / (f**2)[:, None, None]
    )
    np.testing.assert_allclose(batch["xp_cov"][:, 1].numpy(), expected, rtol=1e-5)
    np.testing.assert_allclose(data.parallax, np.linspace(0.5, 2.0, 40)[rows])
    np.testing.assert_allclose(data.parallax_zp, -0.03, rtol=1e-6)
    assert diag["zero_point"] == "sidecar" and diag["sidecar_parallax_found"] == 3
    sidecar.close()


def test_cli_end_to_end(tmp_path) -> None:
    import importlib.util

    info = _write_table(tmp_path / "t.fits", n=150)
    from astropy.table import Table

    sid = np.asarray(Table.read(tmp_path / "t.fits")["source_id"])
    labels = xf.labels_from_table(tmp_path / "t.fits")
    labels["split"][:] = 0
    labels["split"][100:125] = 1
    labels["split"][125:] = 2
    core = np.array([0, 1, 2, 101, 126])  # train-split core 0-2, held-out core 101, 126
    labels["has_core_feh"][:] = False
    labels["has_core_feh"][core] = True
    labels["fe_h_ref"] = np.where(labels["has_core_feh"], labels["fe_h"], np.nan)
    np.savez(tmp_path / "labels.npz", **labels)
    np.save(tmp_path / "core.npy", core)
    ebv = info["data"].ebv_label
    np.savez(tmp_path / "red.npz", source_id=sid, ebv_star=ebv.astype("f4"),
             e_ebv_star=np.where(np.isfinite(ebv), 0.03, np.nan).astype("f4"))  # fmt: skip
    spec = importlib.util.spec_from_file_location(
        "run_xp_forward_model",
        Path(__file__).resolve().parents[1] / "scripts" / "run_xp_forward_model.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "out"
    manifest = mod.main([
        "--table", str(tmp_path / "t.fits"), "--labels", str(tmp_path / "labels.npz"),
        "--core-rows", str(tmp_path / "core.npy"), "--reddening-labels", str(tmp_path / "red.npz"),
        "--output", str(out), "--epochs", "2", "--batch-size", "32", "--hidden", "16", "--depth", "1",
        "--rank", "1", "--max-train-stars", "60", "--max-heldout-stars", "3", "--valid-rows", "8",
        "--n-candidates", "8", "--bank-theta", "16", "--infer-chunk-rows", "4",
        "--bands", "G,W1,G_PS1,R_PS1",
    ])  # fmt: skip
    pred = np.load(out / "xp_forward_predictions.npz")
    for key in ("source_id", "table_row", "mle", "lo68", "hi68", "lo95", "hi95", "E_mle",
                "E_bound", "parallax_mle", "chi2_xp", "dof_xp", "at_bound", "fe_h_ref",
                "split", "ebv_label"):  # fmt: skip
        assert key in pred.files, key
    n = len(pred["table_row"])
    assert pred["mle"].shape == (n, 3) and pred["at_bound"].shape == (n, 5)
    assert set(pred["subset"]) == {"heldout", "core"}
    is_core = pred["subset"] == "core"
    np.testing.assert_array_equal(np.sort(pred["table_row"][is_core]), core)
    np.testing.assert_array_equal(pred["split"][is_core], [0, 0, 0, 1, 2])
    assert np.isfinite(pred["mle"]).all() and (pred["dof_xp"] == 110).all()
    for name in (
        "model.pt",
        "history.json",
        "report.md",
        "report.json",
        "manifest.json",
        "train_nuisance.npz",
    ):
        assert (out / name).exists(), name
    assert manifest["rows"]["core"] == 5 and manifest["rows"]["core_train"] == 3
    assert manifest["rows"]["train"] == 60
    train = np.load(out / "train_nuisance.npz")
    assert np.isin([0, 1, 2], train["table_row"]).all()  # train-split core trains
    assert not np.isin([101, 126], train["table_row"]).any()  # held-out core never
    report = json.loads((out / "report.json").read_text())
    assert {"heldout", "core_heldout", "core_train"} <= set(report["recovery"])
    assert report["recovery"]["core_heldout"]["n_fe_h_ref"] == 2
    assert "ebv_at_lower_bound" in report["recovery"]["heldout"]
    assert "ebv_vs_label" in report["chi2_at_labels"]
    assert manifest["reddening_labels"]["labelled"] > 0


@pytest.mark.skipif(not SMOKE.exists(), reason="local smoke table not available")
def test_smoke_rows_cpu(tmp_path) -> None:
    labels = xf.labels_from_table(SMOKE)
    lab = np.isfinite(np.stack([labels[p] for p in xf.PARAMS], 1)).all(1)
    rows = np.flatnonzero(lab & (labels["split"] == 0))[:200]
    data, _ = xf.load_star_data(
        SMOKE, rows, labels=labels, zero_point="constant", log=lambda s: None
    )
    model = xf.build_model(data, xf.ModelConfig(hidden=16, depth=1, rank=1))
    hist = xf.train_forward_model(
        model,
        data,
        None,
        xf.TrainConfig(epochs=1, batch_size=64, log_every=0),
        device=CPU,
        log=lambda s: None,
    )
    assert math.isfinite(hist["epochs"][0]["train_nll"])
    test, _ = xf.load_star_data(SMOKE, np.flatnonzero(lab & (labels["split"] == 2))[:3], labels=labels,
                                zero_point="constant", log=lambda s: None)  # fmt: skip
    cfg = xf.InferenceConfig(
        chunk_rows=3,
        n_candidates=8,
        bank_theta=16,
        steps=10,
        profile_scan=2,
        profile_steps=5,
    )
    res = xf.infer_stars(
        model,
        test,
        xf.theta_bounds_from_labels(data.theta),
        data.theta,
        cfg,
        device=CPU,
        log=lambda s: None,
    )
    assert np.isfinite(res["mle"]).all() and np.isfinite(res["chi2_xp"]).all()


def test_prepared_covariance_does_not_depend_on_the_observation() -> None:
    """At a fixed reference the prepared XP covariance is the same whatever the
    observed coefficients are (a covariance that follows the observation can
    explain the residual away), and the training loss evaluates the learned terms
    at the prediction (its chi^2 equals the prepared one at that reference)."""
    model = _true_model()
    with torch.no_grad():
        for loss in model.xp_losses:
            for p in loss.parameters():
                p.add_(0.05 * torch.randn_like(p))  # non-trivial learned terms
    data, truth = _simulate(model, 16, seed=11)
    batch = data.batch(np.arange(16), CPU, torch.float64)
    theta = torch.as_tensor(truth["theta"])
    ebv = torch.as_tensor(truth["ebv"])
    reference = xf.model_reference(model, theta, ebv)
    a = xf.prepare_stars(model, batch, reference=reference)
    shifted = dict(batch, xp=batch["xp"] * 1.3 + 0.1 * batch["xp"].abs().mean())
    b = xf.prepare_stars(model, shifted, reference=reference)
    torch.testing.assert_close(a.xp_factor, b.xp_factor)
    torch.testing.assert_close(a.xp_q, b.xp_q)
    torch.testing.assert_close(a.xp_log_det, b.xp_log_det)
    plx = torch.as_tensor(truth["parallax"])
    ref = xf.star_nll(model, theta, ebv, plx, shifted, parts=True)
    fast = xf.star_nll_prepared(model, theta, ebv, plx, b, parts=True)
    torch.testing.assert_close(fast["chi2_xp"], ref["chi2_xp"], rtol=1e-9, atol=1e-8)
