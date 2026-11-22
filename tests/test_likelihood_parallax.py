"""Parallax nuisance and ensemble scores of the likelihood head (synthetic)."""

import math

import numpy as np
import torch
from torch import nn

from masked_stellar_autoencoder.training import likelihood_head as lh

U_BOUNDS = (-1.5, 2.5)


class ToyHead(nn.Module):
    """Analytic 'ratio head' over theta' = (theta1, theta2, u = log10 parallax).

    s = -0.5 ((z1 - theta1 - alpha 10**u) / sz)^2 - 0.5 ((z2 - theta2) / sz)^2
        + delta: z1 measures theta1 plus a parallax term, z2 measures theta2.
    """

    def __init__(self, alpha=1.0, sz=0.1, delta=0.0):
        super().__init__()
        self.alpha, self.sz, self.delta = alpha, sz, delta
        self.config = dict(embed_out=2, n_params=3)

    def embed(self, z):
        return z[:, :2]

    def score_embedded(self, theta, e):
        e = e.to(theta.dtype)
        r1 = e[:, 0] - theta[:, 0] - self.alpha * torch.pow(10.0, theta[:, 2])
        r2 = e[:, 1] - theta[:, 1]
        return -0.5 * (r1 / self.sz) ** 2 - 0.5 * (r2 / self.sz) ** 2 + self.delta


def _toy(n=64, seed=0):
    rng = np.random.default_rng(seed)
    plx = rng.uniform(1.0, 5.0, n)
    err = 0.05 * plx  # parallax / error = 20
    z = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n)])
    heads = [ToyHead(delta=0.3), ToyHead(delta=-0.3)]
    score = lh.EnsembleScore(heads, parallax=True)
    window = lh.parallax_window(plx, err, U_BOUNDS, k=8.0)
    emb = score.embed(torch.as_tensor(z, dtype=torch.float64))
    x = score.observation(emb, window)
    return score, x, z, plx, err


def test_support_in_four_dimensions_with_log_parallax():
    bounds = ((3000.0, 8000.0), (-0.5, 5.5), (-4.5, 1.0), U_BOUNDS)
    widths = (100.0, 0.1, 0.1, 0.1)
    # A 5^4-cell block of labels around (5000 K, 4.0, -0.5, log plx 0.5).
    g = np.stack(np.meshgrid(*[np.arange(5)] * 4, indexing="ij"), -1).reshape(-1, 4)
    labels = np.array([4800.0, 3.8, -0.7, 0.3]) + (g + 0.5) * np.array(widths)
    sup = lh.Support(labels, bounds, widths, dilate=1)
    assert sup.grid.shape == (50, 60, 55, 40) and sup.grid.sum() == 7**4
    pts = torch.tensor(
        [
            [5050.0, 4.05, -0.45, 0.55],  # block centre
            [5050.0, 4.05, -0.45, 0.25],  # one cell below in log plx: dilated ring
            [5050.0, 4.05, -0.45, 0.05],  # three cells below: outside
            [5050.0, 4.05, -0.45, 2.75],  # outside the box
        ]
    )
    assert sup.inside(pts).tolist() == [True, True, False, False]
    assert sup.on_edge(pts).tolist() == [False, True, False, False]
    # Support cells on the box boundary are edges (no wrap-around).
    corner = lh.Support(np.array([[3050.0, 4.0, 0.0, 2.45]]), bounds, widths)
    assert corner.on_edge(torch.tensor([[3050.0, 4.0, 0.0, 2.45]])).item()
    head = lh.RatioHead(4, n_params=4, embed_dims=(8,), embed_out=4, hidden=(8,))
    score = lh.EnsembleScore([head], support=sup)
    s = score.score_fn()(pts.float(), head.embed(torch.zeros(4, 4)))
    assert torch.isfinite(s[:2]).all() and torch.isnan(s[2:]).all()


def test_parallax_window_and_flat_term_without_measurement():
    plx = np.array([10.0, 0.1, np.nan, -2.0, 1.0])
    err = np.array([0.02, 0.1, 0.1, 0.1, np.nan])
    a, b, p0, w = lh.parallax_window(plx, err, U_BOUNDS, k=8.0)
    np.testing.assert_allclose(a[0], math.log10(10.0 - 0.16))
    np.testing.assert_allclose(a[0] + b[0], math.log10(10.0 + 0.16))
    np.testing.assert_allclose(w[:2], [50.0, 10.0])
    # Poor parallax: the window reaches the lower edge of the box.
    np.testing.assert_allclose([a[1], a[1] + b[1]], [U_BOUNDS[0], math.log10(0.9)])
    # No measurement: whole box, flat term (w = 0).
    assert w[2] == 0 and w[4] == 0 and p0[2] == 0
    np.testing.assert_allclose([a[2], b[2]], [U_BOUNDS[0], 4.0])
    # Measurement outside the box: whole box, term kept.
    np.testing.assert_allclose([a[3], b[3], w[3]], [U_BOUNDS[0], 4.0, 10.0])
    head = ToyHead()
    score = lh.EnsembleScore([head], parallax=True)
    x = score.observation(torch.zeros(5, 2, dtype=torch.float64), (a, b, p0, w))
    theta = torch.tensor([[0.1, 0.2, 0.5]] * 5, dtype=torch.float64)
    full = score.score_fn()(theta, x)
    ratio = score.score_fn(measurement=False)(theta, x)
    assert torch.equal(full[[2, 4]], ratio[[2, 4]])  # flat where unmeasured
    assert (full[[0, 1, 3]] < ratio[[0, 1, 3]]).all()


def test_ensemble_score_is_the_mean_log_ratio():
    torch.manual_seed(0)
    heads = [lh.RatioHead(6, 3, embed_dims=(8,), embed_out=4, hidden=(8,))
             for _ in range(3)]  # fmt: skip
    score = lh.EnsembleScore(heads)
    z, theta = torch.randn(7, 6), torch.randn(7, 3)
    x = lh.embed_rows(score, z, np.arange(7), device="cpu")
    assert x.shape == (7, 12)
    with torch.no_grad():
        mean = sum(h(theta, z) for h in heads) / 3
        torch.testing.assert_close(score.score_fn()(theta, x), mean)
        one = score.member(2)
        torch.testing.assert_close(
            one.score_fn()(theta, score.member_observation(x, 2)), heads[2](theta, z)
        )


def test_profiling_parallax_with_its_measurement_is_analytic():
    """Gaussian toy: the profile over the parallax adds alpha^2 sigma^2."""
    score, x, z, plx, err = _toy()
    sz, alpha = 0.1, 1.0
    box = [[-10.0, 5.0], [-5.0, 5.0], [0.0, 1.0]]
    # Ensemble = mean of the two heads: the same Gaussian, offset 0.
    th = torch.tensor([[0.3, -0.2, 0.0]] * len(x), dtype=torch.float64)
    with torch.no_grad():
        tp = score.physical(th, x)
        h0, h1 = score.heads
        mean = 0.5 * (
            h0.score_embedded(tp, x[:, :2]) + h1.score_embedded(tp, x[:, 2:4])
        )
        torch.testing.assert_close(score.log_ratio(tp, x), mean)
    t0 = lh.window_start(score, x, np.column_stack([z[:, 0] - plx, z[:, 1]]))
    init = np.column_stack([np.zeros((len(x), 2)), t0])
    mle, smax, atb = lh.maximize(score, x, box, init=init, n_candidates=64,
                                 log=lambda m: None)  # fmt: skip
    a, b = x[:, 4].numpy(), x[:, 5].numpy()
    varpi = 10 ** (a + b * mle[:, 2])
    np.testing.assert_allclose(varpi, plx, rtol=1e-5)
    np.testing.assert_allclose(mle[:, 0], z[:, 0] - alpha * plx, atol=1e-5)
    np.testing.assert_allclose(mle[:, 1], z[:, 1], atol=1e-5)
    np.testing.assert_allclose(smax, 0.0, atol=1e-8)
    assert not atb.any()
    s_tot = np.sqrt(sz**2 + alpha**2 * err**2)
    # Per-parameter profile statistic of theta1 (theta2 and parallax profiled).
    d = np.random.default_rng(1).normal(0, 1.5, len(x)) * s_tot
    lam1 = lh.profile_statistic(score, x, 0, mle[:, 0] + d, smax, mle, box,
                                n_candidates=16)  # fmt: skip
    np.testing.assert_allclose(lam1, 0.5 * (d / s_tot) ** 2, rtol=1e-4, atol=1e-8)
    # Joint statistic of (theta1, theta2), parallax profiled.
    d2 = 0.7 * sz
    lam = lh.profile_statistic(
        score, x, [0, 1], np.column_stack([mle[:, 0] + d, mle[:, 1] + d2]), smax,
        mle, box, n_candidates=16, n_starts=2,
    )  # fmt: skip
    np.testing.assert_allclose(lam, 0.5 * (d / s_tot) ** 2 + 0.5 * 0.7**2, rtol=1e-4)
    # Profile intervals at 0.5: +- the combined error (theta1), +- sz (theta2).
    lo, hi, lob, hib = lh.profiles(score, x, mle, smax, 0.5, box, params=(0, 1),
                                   log=lambda m: None)  # fmt: skip
    assert lo.shape == (len(x), 2) and not (lob.any() or hib.any())
    np.testing.assert_allclose(hi[:, 0] - mle[:, 0], s_tot, rtol=1e-4)
    np.testing.assert_allclose(mle[:, 0] - lo[:, 0], s_tot, rtol=1e-4)
    np.testing.assert_allclose(hi[:, 1] - lo[:, 1], 2 * sz, rtol=1e-4)
    # Without the measurement term theta1 and the parallax are degenerate:
    # the profile of theta1 is flat (lambda ~ 0) wherever 10**u can absorb it.
    flat = lh.EnsembleScore(score.heads, parallax=True)
    a2, b2, _, _ = lh.parallax_window(np.full(len(x), np.nan), err, U_BOUNDS)
    x2 = flat.observation(x[:, :4], (a2, b2, np.zeros(len(x)), np.zeros(len(x))))
    mle2 = mle.copy()
    mle2[:, 2] = (np.log10(plx) - a2) / b2
    lam2 = lh.profile_statistic(flat, x2, 0, mle[:, 0] + 0.5, np.zeros(len(x)), mle2,
                                box, n_candidates=64)  # fmt: skip
    np.testing.assert_allclose(lam2, 0.0, atol=1e-6)
