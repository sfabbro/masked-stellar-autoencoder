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
