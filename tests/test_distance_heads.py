"""Numerical contract of the convolved raw-parallax distance likelihood.

Every check here is one of the pre-launch gates of TRAINING_PLAN Sec. 5 and
``05-heads-losses-metrics.md`` Sec. 4: the narrow-error limit, GH convergence,
float64 gradients, negative parallaxes, empty batches, the unit identity and
batch invariance. Reference integrals come from `scipy.integrate.quad` and from
an independent dense grid, never from the same quadrature being tested.
"""

import math

import pytest
import torch
from scipy.integrate import quad

from masked_stellar_autoencoder.training import distance_heads as dh

DOUBLE = torch.float64


def _loss(mean, log_scale, plx, sigma, **kwargs):
    return dh.convolved_parallax_nll(mean, log_scale, plx, sigma, **kwargs)


def _reference_nll(mu, sigma_s, plx, sigma_eff, window=14.0):
    """Independent ``-log int p(s) N(plx|z+exp(-s), sigma) ds`` in float64."""

    def integrand(s):
        h = math.exp(-s)
        log_prior = -0.5 * ((s - mu) / sigma_s) ** 2 - math.log(sigma_s)
        log_like = -0.5 * ((plx - h) / sigma_eff) ** 2 - math.log(sigma_eff)
        return math.exp(log_prior + log_like - math.log(2.0 * math.pi))

    value, _ = quad(
        integrand,
        mu - window * sigma_s,
        mu + window * sigma_s,
        epsabs=1e-14,
        epsrel=1e-13,
        limit=400,
    )
    assert value > 0
    return -math.log(value)


def test_unit_identity_is_exact():
    """s = 0 <-> 1 mas <-> 1 kpc and s = log(10) <-> 0.1 mas: no 1000x factor."""
    assert dh.PARALLAX_MAS_AT_ONE_KPC == 1.0
    assert dh.LOG_DISTANCE_AT_ONE_KPC == 0.0
    assert (
        dh.parallax_mas_from_log_distance(torch.tensor(0.0, dtype=DOUBLE)).item() == 1.0
    )
    assert dh.parallax_mas_from_log_distance(
        torch.tensor(math.log(10.0), dtype=DOUBLE)
    ).item() == pytest.approx(0.1, rel=1e-15)
    assert (
        dh.log_distance_from_parallax_mas(torch.tensor(1.0, dtype=DOUBLE)).item() == 0.0
    )
    for log_distance in (-2.0, 0.0, 0.7, 3.0):
        value = torch.tensor([log_distance], dtype=DOUBLE)
        back = dh.log_distance_from_parallax_mas(
            dh.parallax_mas_from_log_distance(value)
        )
        assert back.item() == pytest.approx(log_distance, abs=1e-15)


def test_log_distance_inverse_rejects_non_positive_parallax():
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            dh.log_distance_from_parallax_mas(torch.tensor([bad], dtype=DOUBLE))


@pytest.mark.parametrize(
    "mu,sigma_s,plx",
    [
        (0.0, 0.30, 1.0),
        (0.5, 0.301, 1.0),
        (-0.4, 0.50, 2.0),
        (1.0, 0.20, 0.5),
        (0.0, 0.80, 4.0),
    ],
)
def test_narrow_error_limit_to_1e8(mu, sigma_s, plx):
    """sigma_eff -> 0 limit is -log p(s*) + log h* to 1e-8 at sigma_eff = 1e-6."""
    mean = torch.tensor([mu], dtype=DOUBLE)
    log_scale = torch.tensor([math.log(sigma_s)], dtype=DOUBLE)
    measured = torch.tensor([plx], dtype=DOUBLE)
    narrow = _loss(
        mean,
        log_scale,
        measured,
        torch.tensor([1e-6], dtype=DOUBLE),
        rule="gh_residual",
    )
    limit = dh.narrow_error_limit_nll(mean, log_scale, measured)
    assert narrow.observation_count == 1
    assert narrow.residual_rule_rows == 1
    assert float(narrow.numerator.item()) == pytest.approx(
        float(limit.item()), abs=1e-8
    )
    # The prior-space rule cannot resolve a 1e-6 mas likelihood: this is why
    # "auto" switches to the residual-space rule in that regime.
    prior = _loss(
        mean, log_scale, measured, torch.tensor([1e-6], dtype=DOUBLE), rule="gh_prior"
    )
    assert float(prior.numerator.item()) > 1.0


def test_residual_rule_matches_independent_reference():
    for sigma_eff in (0.005, 0.03, 0.09):
        got = _loss(
            torch.tensor([0.0], dtype=DOUBLE),
            torch.tensor([math.log(0.3)], dtype=DOUBLE),
            torch.tensor([1.0], dtype=DOUBLE),
            torch.tensor([sigma_eff], dtype=DOUBLE),
            rule="gh_residual",
        )
        assert float(got.numerator.item()) == pytest.approx(
            _reference_nll(0.0, 0.3, 1.0, sigma_eff), abs=1e-8
        )


def test_gh16_to_256_convergence_below_1e3():
    """Prior-space rule in its declared regime (sigma_eff = sigma_s h*)."""
    mean = torch.tensor([0.0], dtype=DOUBLE)
    log_scale = torch.tensor([math.log(0.3)], dtype=DOUBLE)
    plx = torch.tensor([1.0], dtype=DOUBLE)
    sigma = torch.tensor([0.3], dtype=DOUBLE)
    values = [
        float(
            _loss(
                mean, log_scale, plx, sigma, rule="gh_prior", order=order
            ).numerator.item()
        )
        for order in (16, 32, 64, 128, 256)
    ]
    reference = values[-1]
    for value in values[:-1]:
        assert abs(value - reference) / abs(reference) < 1e-3
    assert reference == pytest.approx(_reference_nll(0.0, 0.3, 1.0, 0.3), abs=1e-6)


def test_one_dimensional_gaussian_matches_scipy_to_1e10():
    mean = torch.tensor([0.2], dtype=DOUBLE)
    log_scale = torch.tensor([math.log(0.4)], dtype=DOUBLE)
    plx = torch.tensor([1.4], dtype=DOUBLE)
    sigma = torch.tensor([0.35], dtype=DOUBLE)
    got = float(
        _loss(mean, log_scale, plx, sigma, rule="gh_prior", order=256).numerator.item()
    )
    assert got == pytest.approx(_reference_nll(0.2, 0.4, 1.4, 0.35), abs=1e-10)


def test_float64_gradcheck_three_rows():
    plx = torch.tensor([1.0, 0.4, 2.3], dtype=DOUBLE)
    wide = torch.tensor([0.5, 0.8, 0.3], dtype=DOUBLE)
    narrow = torch.tensor([0.01, 0.02, 0.005], dtype=DOUBLE)
    mean = torch.tensor([0.1, -0.3, 0.7], dtype=DOUBLE, requires_grad=True)
    log_scale = torch.tensor([-1.1, -0.7, -1.4], dtype=DOUBLE, requires_grad=True)

    def prior(mean_, log_scale_):
        return _loss(mean_, log_scale_, plx, wide, rule="gh_prior", order=8).mean

    def residual(mean_, log_scale_):
        return _loss(mean_, log_scale_, plx, narrow, rule="auto", order=8).mean

    assert torch.autograd.gradcheck(prior, (mean, log_scale), eps=1e-6, atol=1e-8)
    assert torch.autograd.gradcheck(residual, (mean, log_scale), eps=1e-6, atol=1e-8)


def test_negative_parallax_finite_and_prefers_larger_distance():
    """A negative parallax is retained: finite, and the NLL falls as d grows."""
    means = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=DOUBLE)
    log_scale = torch.full_like(means, math.log(0.3))
    plx = torch.full_like(means, -5.0)
    sigma = torch.full_like(means, 0.1)
    losses = _loss(means, log_scale, plx, sigma, rule="auto")
    assert losses.observation_count == 6
    assert torch.isfinite(losses.numerator)
    per_row = [
        float(
            _loss(
                means[i : i + 1], log_scale[i : i + 1], plx[i : i + 1], sigma[i : i + 1]
            ).numerator
        )
        for i in range(len(means))
    ]
    assert per_row == sorted(per_row, reverse=True)
    assert dh.narrow_error_limit_nll(means, log_scale, plx).isinf().all()


def test_zero_parallax_is_finite_and_the_limit_is_undefined():
    mean = torch.tensor([0.0], dtype=DOUBLE)
    log_scale = torch.tensor([math.log(0.3)], dtype=DOUBLE)
    zero = torch.tensor([0.0], dtype=DOUBLE)
    got = _loss(mean, log_scale, zero, torch.tensor([0.1], dtype=DOUBLE), rule="auto")
    assert got.observation_count == 1
    assert torch.isfinite(got.numerator)
    assert dh.narrow_error_limit_nll(mean, log_scale, zero).isinf().all()


def test_empty_and_all_invalid_batches_are_differentiable_zero():
    cases = [
        (torch.zeros(0, dtype=DOUBLE), torch.zeros(0, dtype=DOUBLE)),
        (
            torch.full((4,), float("nan"), dtype=DOUBLE),
            torch.full((4,), 0.1, dtype=DOUBLE),
        ),
        (torch.full((4,), 1.0, dtype=DOUBLE), torch.zeros(4, dtype=DOUBLE)),
    ]
    for plx, sigma in cases:
        n = plx.shape[0]
        mean = torch.zeros(n, dtype=DOUBLE, requires_grad=True)
        log_scale = torch.zeros(n, dtype=DOUBLE, requires_grad=True)
        result = _loss(mean, log_scale, plx, sigma)
        assert result.observation_count == 0
        assert result.row_count == 0
        assert float(result.numerator.detach()) == 0.0
        (grad,) = torch.autograd.grad(result.mean, mean, allow_unused=True)
        assert grad is not None
        assert torch.isfinite(grad).all()
    # Eligibility can empty a non-empty batch as well.
    mean = torch.zeros(3, dtype=DOUBLE)
    log_scale = torch.zeros(3, dtype=DOUBLE)
    plx = torch.ones(3, dtype=DOUBLE)
    sigma = torch.full((3,), 0.1, dtype=DOUBLE)
    result = _loss(
        mean, log_scale, plx, sigma, eligible=torch.zeros(3, dtype=torch.bool)
    )
    assert result.observation_count == 0


def test_batch_one_equals_batch_128_in_float64():
    generator = torch.Generator().manual_seed(11)
    n = 128
    mean = torch.randn(n, dtype=DOUBLE, generator=generator)
    log_scale = -1.0 + 0.2 * torch.randn(n, dtype=DOUBLE, generator=generator)
    plx = 0.1 + 2.0 * torch.rand(n, dtype=DOUBLE, generator=generator)
    sigma = 0.01 + 0.05 * torch.rand(n, dtype=DOUBLE, generator=generator)
    full = _loss(mean, log_scale, plx, sigma, rule="auto")
    single = torch.stack(
        [
            _loss(
                mean[i : i + 1], log_scale[i : i + 1], plx[i : i + 1], sigma[i : i + 1]
            ).numerator
            for i in range(n)
        ]
    ).sum()
    assert full.observation_count == n
    assert float(full.numerator) == float(single)


def test_zero_point_scalar_and_tensor_agree():
    mean = torch.tensor([0.0, 0.4], dtype=DOUBLE)
    log_scale = torch.tensor([-1.0, -0.8], dtype=DOUBLE)
    plx = torch.tensor([1.0, 0.9], dtype=DOUBLE)
    sigma = torch.tensor([0.1, 0.2], dtype=DOUBLE)
    scalar = _loss(mean, log_scale, plx, sigma, zero_point_mas=0.03)
    tensor = _loss(
        mean, log_scale, plx, sigma, zero_point_mas=torch.full_like(plx, 0.03)
    )
    per_star = _loss(
        mean,
        log_scale,
        plx,
        sigma,
        zero_point_mas=torch.tensor([0.03, 0.03], dtype=DOUBLE),
    )
    assert (
        float(scalar.numerator) == float(tensor.numerator) == float(per_star.numerator)
    )
    with pytest.raises(ValueError):
        _loss(mean, log_scale, plx, sigma, zero_point_mas=torch.zeros(5, dtype=DOUBLE))
    with pytest.raises(ValueError):
        _loss(mean, log_scale, plx, sigma, zero_point_mas=float("nan"))


def test_effective_parallax_error_formula_and_failures():
    quoted = torch.tensor([0.1, 0.2], dtype=DOUBLE)
    sigma = dh.effective_parallax_error_mas(
        quoted, kappa_parallax=2.0, sigma_zp_mas=0.05
    )
    expected = torch.sqrt((2.0 * quoted).square() + 0.05**2)
    assert torch.allclose(sigma, expected, rtol=0, atol=1e-15)
    for bad in (0.0, -0.1, float("nan")):
        with pytest.raises(ValueError):
            dh.effective_parallax_error_mas(torch.tensor([bad], dtype=DOUBLE))
    with pytest.raises(ValueError):
        dh.effective_parallax_error_mas(quoted, kappa_parallax=0.0)
    with pytest.raises(ValueError):
        dh.effective_parallax_error_mas(quoted, sigma_zp_mas=-1.0)


def test_chi2_one_dof_quantile_matches_closed_form():
    assert dh.chi2_1dof_quantile(0.68) == pytest.approx(0.9889464814780231, rel=1e-12)
    assert dh.chi2_1dof_quantile(0.90) == pytest.approx(2.7055434540954155, rel=1e-12)
    assert dh.chi2_1dof_quantile(0.95) == pytest.approx(3.8414588206941223, rel=1e-12)
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            dh.chi2_1dof_quantile(bad)


def test_profile_confidence_set_exact_bounds():
    plx = torch.tensor([1.0], dtype=DOUBLE)
    sigma = torch.tensor([0.1], dtype=DOUBLE)
    ci = dh.parallax_profile_confidence_set(plx, sigma, level=0.90)
    half = 0.1 * math.sqrt(2.7055434540954155)
    assert ci.s_lower.item() == pytest.approx(-math.log(1.0 + half), abs=1e-14)
    assert ci.s_upper.item() == pytest.approx(-math.log(1.0 - half), abs=1e-14)
    assert ci.mode.item() == pytest.approx(0.0, abs=1e-15)
    assert not bool(ci.empty.item())
    assert not bool(ci.upper_unbounded.item())
    # 68% must be narrower than 90%.
    narrow = dh.parallax_profile_confidence_set(plx, sigma, level=0.68)
    assert narrow.s_lower > ci.s_lower and narrow.s_upper < ci.s_upper


def test_profile_confidence_set_empty_and_unbounded_are_never_capped():
    plx = torch.tensor([-5.0, 0.05, 1.0], dtype=DOUBLE)
    sigma = torch.tensor([0.1, 0.1, 0.1], dtype=DOUBLE)
    ci = dh.parallax_profile_confidence_set(plx, sigma, level=0.90)
    assert bool(ci.empty[0]) and not bool(ci.upper_unbounded[0])
    assert math.isnan(ci.s_lower[0].item())
    assert bool(ci.upper_unbounded[1]) and not bool(ci.empty[1])
    assert math.isinf(ci.s_upper[1].item())
    assert ci.mode[0].isinf()
    assert ci.mode[1].item() == pytest.approx(-math.log(0.05), abs=1e-14)
    assert ci.mode[2].item() == pytest.approx(0.0, abs=1e-15)
    fractions = ci.fractions()
    assert fractions["n"] == 3.0
    assert fractions["frac_ci_empty"] == pytest.approx(1.0 / 3.0)
    assert fractions["frac_ci_upper_unbounded"] == pytest.approx(1.0 / 3.0)
    assert fractions["n_ci_empty"] == 1.0
    # Exact boundary: h* + sigma sqrt(q) <= 0 is empty.
    boundary = dh.parallax_profile_confidence_set(
        torch.tensor([-0.1 * math.sqrt(2.7055434540954155)], dtype=DOUBLE),
        sigma[:1],
        level=0.90,
    )
    assert bool(boundary.empty[0])
    with pytest.raises(ValueError):
        dh.parallax_profile_confidence_set(plx, torch.zeros_like(sigma))


def test_fixed_qmc_nodes_are_saved_deterministic_and_antithetic():
    node, weight = dh.fixed_qmc_nodes(
        64, 2026, dtype=DOUBLE, device=torch.device("cpu")
    )
    other, other_weight = dh.fixed_qmc_nodes(
        64, 2026, dtype=DOUBLE, device=torch.device("cpu")
    )
    assert torch.equal(node, other) and torch.equal(weight, other_weight)
    assert node.shape == (64,) and weight.shape == (64,)
    assert node.min() < 0 < node.max()
    assert torch.allclose(weight, torch.full_like(weight, math.sqrt(2 * math.pi) / 64))
    # The node multiset is its own negation (antithetic pairs).
    assert torch.allclose(torch.sort(node).values, torch.sort(-node).values)
    assert len(torch.unique(node)) == 64
    for bad in (7, 0):
        with pytest.raises(ValueError):
            dh.fixed_qmc_nodes(bad, 2026, dtype=DOUBLE, device=torch.device("cpu"))
    with pytest.raises(ValueError):
        dh.fixed_qmc_nodes(8, -1, dtype=DOUBLE, device=torch.device("cpu"))


def test_qmc_fallback_is_exact_in_the_narrow_limit():
    mean = torch.tensor([0.0], dtype=DOUBLE)
    log_scale = torch.tensor([math.log(0.3)], dtype=DOUBLE)
    plx = torch.tensor([1.0], dtype=DOUBLE)
    sigma = torch.tensor([1e-6], dtype=DOUBLE)
    qmc = _loss(mean, log_scale, plx, sigma, rule="gh_residual", nodes="qmc")
    limit = dh.narrow_error_limit_nll(mean, log_scale, plx)
    assert float(qmc.numerator) == pytest.approx(float(limit), abs=1e-8)
    # ... and it stays a fallback, not a replacement, away from that limit.
    wide = torch.tensor([1.2], dtype=DOUBLE)
    gh = _loss(mean, log_scale, plx, wide, rule="gh_prior", order=256, nodes="gh")
    qmc_wide = _loss(mean, log_scale, plx, wide, rule="gh_prior", nodes="qmc")
    relative = abs(float(gh.numerator) - float(qmc_wide.numerator)) / abs(
        float(gh.numerator)
    )
    assert relative < 5e-3
    with pytest.raises(ValueError):
        _loss(mean, log_scale, plx, wide, nodes="not-a-node-set")


def test_no_reciprocal_of_the_measured_parallax():
    """Zero and negative measurements stay finite; only the model is inverted."""
    mean = torch.tensor([0.0], dtype=DOUBLE)
    log_scale = torch.tensor([math.log(0.3)], dtype=DOUBLE)
    for plx in (0.0, -0.5):
        result = _loss(
            mean,
            log_scale,
            torch.tensor([plx], dtype=DOUBLE),
            torch.tensor([0.2], dtype=DOUBLE),
        )
        assert torch.isfinite(result.numerator)
        assert result.observation_count == 1


def test_rule_diagnostics_expose_the_quadrature_regime():
    mean = torch.tensor([0.0], dtype=DOUBLE)
    log_scale = torch.tensor([math.log(0.3)], dtype=DOUBLE)
    plx = torch.tensor([1.0], dtype=DOUBLE)
    exact = _loss(mean, log_scale, plx, torch.tensor([0.05], dtype=DOUBLE))
    assert exact.residual_rule_rows == 1
    assert math.isinf(exact.min_prior_resolution)
    prior = _loss(mean, log_scale, plx, torch.tensor([0.2], dtype=DOUBLE))
    assert prior.residual_rule_rows == 0
    # sigma_eff / (sigma_s h*) = 0.2 / 0.3: below the ~0.4 "resolved" point, so
    # the diagnostic must say so instead of hiding it.
    assert prior.min_prior_resolution == pytest.approx(0.2 / 0.3, rel=1e-12)
    wide = _loss(mean, log_scale, plx, torch.tensor([0.6], dtype=DOUBLE))
    assert wide.min_prior_resolution == pytest.approx(2.0, rel=1e-12)
    with pytest.raises(ValueError):
        _loss(mean, log_scale, plx, torch.tensor([0.2], dtype=DOUBLE), order=1)


def test_rule_selection_and_strict_modes():
    mean = torch.tensor([0.0], dtype=DOUBLE)
    log_scale = torch.tensor([math.log(0.3)], dtype=DOUBLE)
    plx = torch.tensor([1.0], dtype=DOUBLE)
    wide = torch.tensor([0.5], dtype=DOUBLE)
    narrow = torch.tensor([0.01], dtype=DOUBLE)
    assert _loss(mean, log_scale, plx, narrow, rule="auto").residual_rule_rows == 1
    assert _loss(mean, log_scale, plx, wide, rule="auto").residual_rule_rows == 0
    with pytest.raises(ValueError):
        _loss(mean, log_scale, plx, wide, rule="gh_residual")
    with pytest.raises(ValueError):
        _loss(mean, log_scale, plx, wide, rule="not-a-rule")
    # A zero sigma_eff is an invalid measurement, so it is masked out (count 0
    # with a differentiable zero), never silently floored.
    invalid = _loss(mean, log_scale, plx, torch.zeros_like(plx), rule="gh_prior")
    assert invalid.observation_count == 0
    assert float(invalid.numerator) == 0.0


def test_mixture_k_one_equals_gaussian_and_is_normalized():
    generator = torch.Generator().manual_seed(3)
    n = 16
    mean = torch.randn(n, dtype=DOUBLE, generator=generator)
    log_scale = -1.0 + 0.1 * torch.randn(n, dtype=DOUBLE, generator=generator)
    plx = 0.1 + 2.0 * torch.rand(n, dtype=DOUBLE, generator=generator)
    sigma = 0.01 + 0.05 * torch.rand(n, dtype=DOUBLE, generator=generator)
    gaussian = _loss(mean, log_scale, plx, sigma, rule="gh_prior")
    mixture = dh.convolved_parallax_nll_mixture(
        torch.zeros(n, 1, dtype=DOUBLE),
        mean[:, None],
        log_scale[:, None],
        plx,
        sigma,
        rule="gh_prior",
    )
    assert float(mixture.numerator) == float(gaussian.numerator)

    head = dh.LogDistanceMixtureHead(input_dim=6, width=8, components=3).double()
    with torch.no_grad():
        head.trunk.output.bias[:3] = torch.tensor([0.4, -0.6, 0.1], dtype=DOUBLE)
        head.trunk.output.bias[3:6] = torch.tensor([-0.5, 0.2, 0.9], dtype=DOUBLE)
        head.trunk.output.bias[6:] = torch.log(
            torch.tensor([0.3, 0.15, 0.08], dtype=DOUBLE)
        )
    features = torch.zeros(1, 6, dtype=DOUBLE)
    grid = torch.linspace(-8.0, 8.0, 400_001, dtype=DOUBLE)
    log_prob = head.log_prob(features.expand(grid.numel(), 6), grid)
    total = torch.trapezoid(log_prob.exp(), grid)
    assert total.item() == pytest.approx(1.0, abs=1e-8)


def test_head_shapes_units_and_failures():
    head = dh.LogDistanceGaussianHead(input_dim=384, width=16).double()
    features = torch.zeros(5, 384, dtype=DOUBLE)
    outputs = head(features)
    assert outputs["mean"].shape == (5,)
    assert outputs["log_scale"].shape == (5,)
    assert torch.all(outputs["scale"] > 0)
    assert outputs["mean"].abs().max().item() == 0.0  # zero initialised
    assert outputs["scale"].max().item() == pytest.approx(math.exp(-1.0))
    plx = torch.full((5,), 0.5, dtype=DOUBLE)
    sigma = torch.full((5,), 0.1, dtype=DOUBLE)
    result = head.loss(features, plx, sigma, rule="gh_prior")
    assert result.observation_count == 5
    assert torch.isfinite(result.numerator)
    with pytest.raises(ValueError):
        head(torch.zeros(5, 383, dtype=DOUBLE))
    with pytest.raises(ValueError):
        head(torch.zeros(5, dtype=DOUBLE))
    mixture = dh.LogDistanceMixtureHead(input_dim=8, width=8, components=4)
    out = mixture(torch.zeros(2, 8))
    assert out["logits"].shape == (2, 4)
    assert out["mean"].shape == (2, 4)
    assert torch.allclose(torch.softmax(out["logits"], dim=1), torch.full((2, 4), 0.25))
    with pytest.raises(ValueError):
        dh.LogDistanceMixtureHead(input_dim=8, components=0)
    with pytest.raises(ValueError):
        dh.LogDistanceGaussianHead(input_dim=8, initial_log_distance=float("inf"))
