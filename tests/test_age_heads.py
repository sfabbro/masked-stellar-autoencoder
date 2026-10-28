"""Age-head contract: proper quantiles, exact split normal, positive log-age.

The split-normal density and its quantile function are checked against their
own exact normalization and against the inverse-CDF identity (not against the
same code path), the pinball loss is checked to recover known Gaussian
quantiles on synthetic data, and every transform is checked to keep
``age = exp(tau)`` strictly positive without clipping.
"""

import math

import pytest
import torch

from masked_stellar_autoencoder.training import age_heads as ah

DOUBLE = torch.float64


def _split_normal(peak, sigma_lower, sigma_upper, grid):
    """Broadcast split-normal pieces over a value grid for one parameter set."""
    return ah.split_normal_log_prob(
        grid,
        torch.full_like(grid, peak),
        torch.full_like(grid, sigma_lower),
        torch.full_like(grid, sigma_upper),
    )


def test_log_age_roundtrip_and_positivity():
    ages = torch.tensor([1e-9, 0.01, 1.0, 13.8, 1e9], dtype=DOUBLE)
    tau = ah.log_age_gyr(ages)
    assert torch.allclose(ah.age_gyr_from_log_age(tau), ages, rtol=1e-14, atol=0)
    assert torch.equal(tau, ages.log())
    tiny = ah.age_gyr_from_log_age(torch.tensor([-20.0, 0.0, 25.0], dtype=DOUBLE))
    assert torch.all(tiny > 0) and torch.isfinite(tiny).all()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            ah.log_age_gyr(torch.tensor([bad], dtype=DOUBLE))
    with pytest.raises(ValueError):
        ah.age_gyr_from_log_age(torch.tensor([1000.0], dtype=DOUBLE))
    with pytest.raises(TypeError):
        ah.log_age_gyr(torch.tensor([1, 2]))


def test_split_normal_is_normalized_to_1e16():
    grid = torch.linspace(-12.0, 12.0, 400_001, dtype=DOUBLE)
    for peak, sigma_lower, sigma_upper in (
        (1.0, 0.2, 0.4),
        (0.0, 0.4, 0.2),
        (-1.5, 0.3, 0.3),
        (2.0, 0.05, 1.5),
    ):
        density = _split_normal(peak, sigma_lower, sigma_upper, grid).exp()
        total = torch.trapezoid(density, grid)
        assert total.item() == pytest.approx(1.0, abs=1e-10)
        # CDF endpoints and continuity at the mode.
        cdf = ah.split_normal_cdf(
            torch.tensor([peak], dtype=DOUBLE),
            torch.tensor([peak], dtype=DOUBLE),
            torch.tensor([sigma_lower], dtype=DOUBLE),
            torch.tensor([sigma_upper], dtype=DOUBLE),
        )
        assert cdf.item() == pytest.approx(
            sigma_lower / (sigma_lower + sigma_upper), abs=1e-14
        )
    with pytest.raises(ValueError):
        ah.split_normal_log_prob(
            torch.zeros(2, dtype=DOUBLE),
            torch.zeros(2, dtype=DOUBLE),
            torch.zeros(2, dtype=DOUBLE),
            torch.ones(2, dtype=DOUBLE),
        )


def test_split_normal_quantiles_are_the_inverse_cdf():
    peaks = torch.tensor([1.0, 0.0], dtype=DOUBLE)
    sigma_lower = torch.tensor([0.2, 0.4], dtype=DOUBLE)
    sigma_upper = torch.tensor([0.4, 0.2], dtype=DOUBLE)
    probabilities = (0.05, 0.16, 0.5, 0.84, 0.95)
    quantiles = ah.split_normal_quantiles(
        peaks, sigma_lower, sigma_upper, probabilities
    )
    assert quantiles.shape == (2, len(probabilities))
    for row in range(2):
        for column, probability in enumerate(probabilities):
            cdf = ah.split_normal_cdf(
                quantiles[row, column].reshape(1),
                peaks[row].reshape(1),
                sigma_lower[row].reshape(1),
                sigma_upper[row].reshape(1),
            )
            assert cdf.item() == pytest.approx(probability, abs=1e-12)
    # The 0.5 quantile is NOT the mode unless the widths are equal: this is the
    # documented approximation error of reading tau_16/tau_84 as tau_50 -/+ s.
    assert quantiles[0, 2].item() > peaks[0].item()
    assert quantiles[1, 2].item() < peaks[1].item()
    symmetric = ah.split_normal_quantiles(
        torch.tensor([0.0], dtype=DOUBLE),
        torch.tensor([0.3], dtype=DOUBLE),
        torch.tensor([0.3], dtype=DOUBLE),
        probabilities,
    )
    assert symmetric[0, 2].item() == pytest.approx(0.0, abs=1e-14)
    # For equal widths the density is a plain Gaussian, so the 16/84 quantiles
    # are sigma * Phi^{-1}(0.16/0.84) = -/+0.9945 sigma, not exactly -/+ sigma.
    assert symmetric[0, 1].item() == pytest.approx(-0.3, abs=2e-3)
    assert symmetric[0, 3].item() == pytest.approx(0.3, abs=2e-3)
    # ... and even then the 16th percentile is 0.1587, not exactly 0.16.
    equal_width_cdf = ah.split_normal_cdf(
        symmetric[0, 1].reshape(1),
        torch.tensor([0.0], dtype=DOUBLE),
        torch.tensor([0.3], dtype=DOUBLE),
        torch.tensor([0.3], dtype=DOUBLE),
    )
    assert equal_width_cdf.item() == pytest.approx(0.16, abs=2e-3)
    with pytest.raises(ValueError):
        ah.split_normal_quantiles(peaks, sigma_lower, sigma_lower, (0.16, 0.84, 0.5))


def test_split_normal_nll_is_minimized_at_the_observation():
    peaks = torch.linspace(-2.0, 3.0, 4001, dtype=DOUBLE)
    label = torch.tensor([0.7], dtype=DOUBLE)
    nll = -ah.split_normal_log_prob(
        label.expand_as(peaks),
        peaks,
        torch.full_like(peaks, 0.25),
        torch.full_like(peaks, 0.6),
    )
    best = peaks[torch.argmin(nll)].item()
    assert best == pytest.approx(0.7, abs=1e-3)
    losses = ah.split_normal_nll(
        peaks,
        torch.full_like(peaks, math.log(0.25)),
        torch.full_like(peaks, math.log(0.6)),
        label.expand_as(peaks),
    )
    assert losses.observation_count == peaks.numel()
    assert float(losses.numerator) == pytest.approx(float(nll.sum()))


def test_label_split_normal_widths_and_crossing_guard():
    sigma_lower, sigma_upper = ah.label_split_normal_widths(
        torch.tensor([0.2], dtype=DOUBLE),
        torch.tensor([1.0], dtype=DOUBLE),
        torch.tensor([1.5], dtype=DOUBLE),
    )
    assert sigma_lower.item() == pytest.approx(0.8)
    assert sigma_upper.item() == pytest.approx(0.5)
    with pytest.raises(ValueError):
        ah.label_split_normal_widths(
            torch.tensor([1.2], dtype=DOUBLE),
            torch.tensor([1.0], dtype=DOUBLE),
            torch.tensor([1.5], dtype=DOUBLE),
        )


def test_pinball_matches_the_manual_formula():
    predictions = torch.tensor([[0.0, 0.5, 1.0], [2.0, 2.0, 2.0]], dtype=DOUBLE)
    target = torch.tensor([0.3, 3.0], dtype=DOUBLE)
    quantiles = (0.16, 0.5, 0.84)
    result = ah.pinball_loss(predictions, target, quantiles)
    manual = 0.0
    for row, value in enumerate(target.tolist()):
        for column, q in enumerate(quantiles):
            error = value - predictions[row, column].item()
            manual += max(q * error, (q - 1.0) * error)
    assert float(result.numerator) == pytest.approx(manual, abs=1e-14)
    assert result.observation_count == 6
    assert result.row_count == 2
    assert result.crossing_fraction == 0.0
    assert result.mean.item() == pytest.approx(manual / 6.0, abs=1e-14)


def test_pinball_recovers_gaussian_quantiles():
    """Properness check: the pinball-optimal constants are the true quantiles."""
    generator = torch.Generator().manual_seed(7)
    samples = 1.5 + 0.4 * torch.randn(20_000, dtype=DOUBLE, generator=generator)
    quantiles = (0.16, 0.5, 0.84)
    fitted = torch.zeros(3, dtype=DOUBLE, requires_grad=True)
    optimizer = torch.optim.Adam([fitted], lr=0.05)
    for _ in range(1500):
        optimizer.zero_grad()
        predictions = fitted.expand(samples.numel(), 3)
        loss = ah.pinball_loss(predictions, samples, quantiles).mean
        loss.backward()
        optimizer.step()
    expected = torch.tensor(
        [
            1.5 + 0.4 * torch.special.ndtri(torch.tensor(q, dtype=DOUBLE)).item()
            for q in quantiles
        ],
        dtype=DOUBLE,
    )
    assert torch.allclose(fitted.detach(), expected, atol=0.02)
    empirical = torch.quantile(samples, torch.tensor(quantiles, dtype=DOUBLE))
    assert torch.allclose(fitted.detach(), empirical, atol=0.01)


def test_crossover_penalty_and_combined_loss():
    ordered = torch.tensor([[0.0, 1.0, 2.0]], dtype=DOUBLE)
    crossed = torch.tensor([[2.0, 0.5, 0.0]], dtype=DOUBLE)
    quantiles = (0.16, 0.5, 0.84)
    assert ah.crossover_penalty(ordered, quantiles).item() == 0.0
    assert ah.crossover_penalty(crossed, quantiles).item() == pytest.approx(2.0)
    target = torch.tensor([1.0], dtype=DOUBLE)
    plain = ah.age_quantile_loss(crossed, target, quantiles, crossover_weight=0.0)
    penalized = ah.age_quantile_loss(crossed, target, quantiles, crossover_weight=2.0)
    assert float(penalized.numerator) == pytest.approx(
        float(plain.numerator) + 2.0 * 2.0, abs=1e-12
    )
    assert penalized.crossing_fraction == 1.0
    with pytest.raises(ValueError):
        ah.age_quantile_loss(crossed, target, quantiles, crossover_weight=-1.0)
    with pytest.raises(ValueError):
        ah.pinball_loss(ordered, target, (0.16, 0.84, 0.5))


def test_gaussian_nll_is_the_closed_form():
    mean = torch.tensor([0.1, -0.4], dtype=DOUBLE)
    log_scale = torch.tensor([-1.0, -0.5], dtype=DOUBLE)
    label = torch.tensor([0.3, -0.9], dtype=DOUBLE)
    result = ah.gaussian_nll(mean, log_scale, label)
    scale = log_scale.exp()
    expected = (
        0.5 * ((label - mean) / scale).square()
        + scale.log()
        + 0.5 * math.log(2 * math.pi)
    ).sum()
    assert float(result.numerator) == pytest.approx(float(expected), abs=1e-14)
    assert result.observation_count == 2
    with pytest.raises(ValueError):
        ah.gaussian_nll(mean, torch.full_like(mean, float("inf")), label)


def test_empty_batches_give_differentiable_zero():
    predictions = torch.zeros(0, 3, dtype=DOUBLE, requires_grad=True)
    target = torch.zeros(0, dtype=DOUBLE)
    result = ah.pinball_loss(predictions, target, (0.16, 0.5, 0.84))
    assert result.observation_count == 0
    assert float(result.numerator.detach()) == 0.0
    (grad,) = torch.autograd.grad(result.mean, predictions)
    assert grad.shape == (0, 3)
    penalty = ah.crossover_penalty(predictions, (0.16, 0.5, 0.84))
    assert float(penalty.detach()) == 0.0
    mean = torch.zeros(0, dtype=DOUBLE, requires_grad=True)
    log_scale = torch.zeros(0, dtype=DOUBLE, requires_grad=True)
    empty = ah.gaussian_nll(mean, log_scale, target)
    assert empty.observation_count == 0
    (grad,) = torch.autograd.grad(empty.mean, mean)
    assert grad.shape == (0,)
    split = ah.split_normal_nll(mean, log_scale, log_scale, target)
    assert split.observation_count == 0
    # All-NaN labels are selected out, not imputed.
    valid = ah.pinball_loss(
        torch.zeros(4, 3, dtype=DOUBLE),
        torch.full((4,), float("nan"), dtype=DOUBLE),
        (0.16, 0.5, 0.84),
    )
    assert valid.observation_count == 0


def test_quantile_head_shapes_monotone_and_grid():
    head = ah.AgeQuantileHead(
        input_dim=12, width=16, quantiles=ah.WIDE_AGE_QUANTILES
    ).double()
    features = torch.randn(6, 12, dtype=DOUBLE)
    raw = head(features)
    assert raw.shape == (6, 5)
    assert head.quantile_grid.shape == (5,)
    assert tuple(head.quantiles) == ah.WIDE_AGE_QUANTILES
    monotone = ah.AgeQuantileHead(input_dim=12, width=16, monotone=True).double()
    ordered = monotone(features)
    assert (ordered[:, 1:] >= ordered[:, :-1]).all()
    label = torch.zeros(6, dtype=DOUBLE)
    result = head.loss(features, label)
    assert result.observation_count == 6 * 5
    assert torch.isfinite(result.numerator)
    with pytest.raises(ValueError):
        ah.AgeQuantileHead(quantiles=(0.5, 0.16, 0.84))
    with pytest.raises(ValueError):
        head(torch.zeros(6, 13, dtype=DOUBLE))


def test_split_normal_and_gaussian_heads_are_finite():
    features = torch.randn(4, 10, dtype=DOUBLE)
    split = ah.AgeSplitNormalHead(input_dim=10, width=16).double()
    outputs = split(features)
    assert outputs["sigma_lower"].shape == (4,)
    assert torch.all(outputs["sigma_lower"] > 0) and torch.all(
        outputs["sigma_upper"] > 0
    )
    assert torch.isfinite(split.loss(features, torch.zeros(4, dtype=DOUBLE)).numerator)
    gaussian = ah.AgeGaussianHead(input_dim=10, width=16).double()
    g = gaussian(features)
    assert g["scale"].shape == (4,)
    assert torch.isfinite(
        gaussian.loss(features, torch.zeros(4, dtype=DOUBLE)).numerator
    )
    with pytest.raises(ValueError):
        ah.AgeGaussianHead(initial_log_scale=float("nan"))


def test_interval_from_quantiles_reports_asymmetry():
    predictions = torch.tensor([[0.0, 0.5, 1.4]], dtype=DOUBLE)
    median, lower, upper, asymmetry = ah.interval_from_quantiles(
        predictions, (0.16, 0.5, 0.84)
    )
    assert median.item() == pytest.approx(0.5)
    assert lower.item() == pytest.approx(0.0)
    assert upper.item() == pytest.approx(1.4)
    assert asymmetry.item() == pytest.approx((1.4 - 0.5) - (0.5 - 0.0))
    with pytest.raises(ValueError):
        ah.interval_from_quantiles(predictions, (0.16, 0.5, 0.84), lower=0.05)
    with pytest.raises(ValueError):
        ah.interval_from_quantiles(
            predictions, (0.16, 0.5, 0.84), lower=0.84, upper=0.16
        )


def test_no_truth_claim_is_declared():
    assert "not physical truth" in ah.AGE_EVIDENCE_STATUS
    assert "observed_label" in ah.AGE_EVIDENCE_STATUS
    assert not any(name == "true" for name in dir(ah))
