import numpy as np
import torch

from masked_stellar_autoencoder.training.conditional_survey_map import (
    ConditionalSurveyMaps,
    coverage_grid,
)


def make_maps(m=2, bins=4, **kw):
    support = np.ones((m, 4), bool)
    coverage = np.ones((m, 4, bins**3), "f4")
    return ConditionalSurveyMaps(support, coverage, bins=bins, **kw)


def test_starts_at_identity():
    maps = make_maps()
    q = torch.randn(5, 4, 3).sort(-1).values
    torch.testing.assert_close(maps(q, torch.tensor([0, 1, 0, 1, 1])), q)


def test_coverage_grid_fills_missing_coordinates_from_the_star():
    value = np.array([[0.1, 0.1, np.nan, 0.0], [0.1, 0.1, 0.1, np.nan]])
    cov = coverage_grid(value, np.array([0, 1]), np.array([0, 0]), 2, bins=2)
    # record 0 lacks [Fe/H] but takes the star's other record (0.1): cell (1,1,1)
    assert cov[0, 0, 7] > 0 and cov[0, 2, :].sum() == 0
    assert cov[1, 3].sum() == 0  # survey 1 has no alpha


def test_gauge_ignores_surveys_without_coverage():
    maps = make_maps(bins=2)
    maps.precision.fill_(1.0)
    with torch.no_grad():
        maps.bias[1, 2] = 0.5
    x = torch.zeros(3, 4)
    assert float(maps.gauge_penalty(x)) > 0
    maps.coverage[1] = 0.0  # survey 1 has no stars here: survey 0 is the scale
    assert float(maps.gauge_penalty(x)) == 0.0


def test_monotone_penalty_flags_decreasing_maps():
    maps = make_maps()
    x, s = torch.randn(8, 4), torch.zeros(8, dtype=torch.long)
    assert float(maps.monotone_penalty(x, s)) == 0.0
    with torch.no_grad():
        maps.linear[0, 2, 2] = -2.0
    assert float(maps.monotone_penalty(x, s)) > 0


def test_learns_cross_parameter_offset_with_trusted_reference():
    """Survey 1 reports [Fe/H] + 0.3 Teff; trusted survey 0 is unbiased."""
    torch.manual_seed(0)
    trust = np.ones((2, 4), "f4")
    trust[0] = 10.0
    maps = make_maps(trust=trust, basis=4, hidden=16)
    maps.precision.fill_(1.0)
    x = torch.randn(512, 4)
    y1 = x.clone()
    y1[:, 2] = y1[:, 2] + 0.3 * x[:, 0]
    s0 = torch.zeros(512, dtype=torch.long)
    opt = torch.optim.Adam(maps.parameters(), lr=0.02)
    for _ in range(400):
        opt.zero_grad()
        fit = (x + maps.delta(x, s0) - x).square().mean() + (
            x + maps.delta(x, s0 + 1) - y1
        ).square().mean()
        loss = fit + 0.1 * maps.gauge_penalty(x)
        loss.backward()
        opt.step()
    with torch.no_grad():
        d1 = maps.delta(x, s0 + 1)[:, 2]
        d0 = maps.delta(x, s0)
    assert float((d1 - 0.3 * x[:, 0]).abs().mean()) < 0.03
    assert float(d0.abs().mean()) < 0.03


def test_hard_targets_hold_the_weighted_mean_exactly():
    torch.manual_seed(1)
    prec = np.array([[1, 1, 4, 1], [1, 1, 1, 1], [1, 1, 9, 1]], "f4")
    maps = make_maps(m=3, hard_targets=[0, 2, 3], precision=prec)
    with torch.no_grad():
        for p in (maps.bias, maps.linear, maps.mix):
            p.uniform_(-0.5, 0.5)
    x = torch.randn(64, 4)
    d = maps.delta_all(x)
    w = maps._weights(x)
    mean = (w * d).sum(0) / w.sum(0)
    torch.testing.assert_close(mean[[0, 2, 3]], torch.zeros(3, 64), atol=1e-5, rtol=0)
    assert mean[1].abs().max() > 1e-3  # log g stays free (soft gauge only)
    s = torch.tensor([2] * 64)
    torch.testing.assert_close(maps.delta(x, s), d[2].T, atol=1e-5, rtol=1e-5)
    maps.update_precision(torch.ones(64, 4), s, torch.ones(64, 4, dtype=torch.bool))
    torch.testing.assert_close(maps.precision, torch.as_tensor(prec))  # fixed


def test_overlap_scatter_recovers_survey_noise():
    from masked_stellar_autoencoder.training.conditional_survey_map import (
        overlap_scatter,
    )

    rng = np.random.default_rng(0)
    n, sig = 4000, np.array([0.05, 0.1, 0.2])
    truth = rng.normal(size=n)
    rows, surveys, values = [], [], []
    for s, sg in enumerate(sig):
        rows.append(np.arange(n))
        surveys.append(np.full(n, s))
        v = np.full((n, 4), np.nan)
        v[:, 2] = truth + 0.3 * s + rng.normal(scale=sg, size=n)  # offsets drop out
        values.append(v)
    out = overlap_scatter(
        np.concatenate(values), np.concatenate(surveys), np.concatenate(rows), 3
    )
    np.testing.assert_allclose(out[:, 2], sig, rtol=0.15)
    assert np.isnan(out[:, 0]).all()


def test_overlap_precision_floors_and_fills():
    from masked_stellar_autoencoder.training.conditional_survey_map import (
        overlap_precision,
    )

    sigma = np.full((4, 4), np.nan)
    sigma[:, 2] = [0.0, 0.1, 0.2, np.nan]
    prec = overlap_precision(sigma, np.ones((4, 4), bool), floor_quantile=0.0)
    np.testing.assert_allclose(prec[:, 2], [100, 100, 25, 25], rtol=1e-5)
    assert (prec[:, 0] == 1).all()  # nothing measured: equal weights
