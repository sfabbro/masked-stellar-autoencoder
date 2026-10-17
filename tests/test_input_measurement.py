"""Known-slope checks for diagonal XP regression calibration and SIMEX."""

import numpy as np
import torch

from masked_stellar_autoencoder.training.input_measurement import (
    apply_per_star_rc,
    apply_shared_rc,
    fit_univariate_rc,
    perturb_xp,
    prepare_replicas,
    rational_extrapolate,
    shared_rms,
    simex_library_weights,
    xp_indices,
)


def _slope(x, y):
    x = np.ravel(x)
    y = np.ravel(y)
    x = x - x.mean()
    y = y - y.mean()
    return float((x @ y) / (x @ x))


def test_shared_rc_recovers_a_unit_noise_slope():
    torch.manual_seed(0)
    truth = torch.randn(8000, 1)
    observed = truth + torch.randn(8000, 1)
    target = 2 * truth
    calibrated = apply_shared_rc(
        fit_univariate_rc(observed.numpy(), np.array([1.0])), observed.numpy()
    )
    naive = _slope(observed.numpy(), target.numpy())
    corrected = _slope(calibrated, target.numpy())
    assert abs(naive - 1.0) < 0.05
    assert abs(corrected - 2.0) < 0.15


def test_per_star_rc_shrinks_the_noisier_half_more():
    torch.manual_seed(1)
    truth = torch.randn(4000, 1)
    sigma = torch.cat([torch.full((2000, 1), 0.2), torch.full((2000, 1), 2.0)])
    observed = truth + sigma * torch.randn(4000, 1)
    rms = shared_rms(np.repeat(sigma.numpy(), 110, axis=1))
    # One coefficient is enough for the shrinkage comparison; tile to the API.
    values = np.repeat(observed.numpy(), 110, axis=1)
    noise = np.repeat(sigma.numpy(), 110, axis=1)
    calibrators = fit_univariate_rc(values, rms)
    calibrated = apply_per_star_rc(calibrators, values, noise)
    center = float(calibrators[0].mu_w)
    precise = np.linalg.norm(calibrated[:2000, 0] - center) / np.linalg.norm(
        values[:2000, 0] - center
    )
    noisy = np.linalg.norm(calibrated[2000:, 0] - center) / np.linalg.norm(
        values[2000:, 0] - center
    )
    assert precise > noisy + 0.2


def test_simex_weights_match_the_default_quadratic_grid():
    lambdas, weights = simex_library_weights()
    assert lambdas == [0.0, 0.5, 1.0, 1.5, 2.0]
    np.testing.assert_allclose(weights, [3.0, -0.4, -1.8, -1.2, 1.4], atol=1e-6)
    assert abs(float(np.sum(weights)) - 1.0) < 1e-6


def test_library_quadratic_stops_halfway_and_the_rational_extrapolant_does_not():
    lambdas, weights = simex_library_weights()
    slope = 2.0 / (2.0 + np.asarray(lambdas))
    quadratic = float(weights @ slope)
    rational = float(rational_extrapolate(lambdas, slope[:, None])[0])
    assert abs(quadratic - 1.494285714285713) < 1e-6
    assert abs(rational - 2.0) < 0.02


def test_extrapolation_and_zero_lambda_noise_are_deterministic():
    lambdas, weights = simex_library_weights()
    rows = np.arange(5)
    base = np.zeros((5, 140), dtype="f4")
    columns = np.arange(5, 115)
    sigma = np.full(110, 0.3)
    untouched = perturb_xp(base, rows, columns, sigma, lambdas[0], seed=7)
    np.testing.assert_array_equal(untouched, base)
    noised = perturb_xp(base, rows, columns, sigma, lambdas[1], seed=7)
    again = perturb_xp(base, rows, columns, sigma, lambdas[1], seed=7)
    np.testing.assert_array_equal(noised, again)
    assert not np.array_equal(noised[:, columns], base[:, columns])
    stacked = np.stack([np.full(4, value) for value in lambdas])
    quadratic = stacked.T @ weights
    np.testing.assert_allclose(quadratic, np.full(4, -1.0), atol=1e-6)


def test_prepare_replicas_calibrates_xp_and_counts_simex_runs():
    names = (
        [f"pre_{i}" for i in range(5)]
        + [f"bp_{i}" for i in range(1, 56)]
        + [f"rp_{i}" for i in range(1, 56)]
        + ["EBV"]
    )
    generator = np.random.default_rng(0)
    matrix = generator.normal(size=(40, len(names))).astype("f4")
    raw_error = np.full(matrix.shape, np.nan)
    columns = xp_indices(names)
    raw_error[:, columns] = 0.2
    scale = np.ones(len(names))
    fit = np.arange(30)
    spec = {
        "features": "xp_coefficients",
        "noise": "fit_rms_standardized_diagonal",
    }
    replicas, _report = prepare_replicas(
        {**spec, "method": "rc_shared"}, matrix, raw_error, scale, names, fit
    )
    assert len(replicas) == 1
    np.testing.assert_allclose(replicas[0]["matrix"][:, :5], matrix[:, :5])
    assert not np.allclose(replicas[0]["matrix"][:, columns], matrix[:, columns])
    simex, report = prepare_replicas(
        {
            **spec,
            "method": "simex",
            "simex_n_simulations": 4,
            "simex_extrapolation_order": 2,
        },
        matrix,
        raw_error,
        scale,
        names,
        fit,
    )
    assert len(simex) == 17
    assert report["library_quadratic_extrapolant_rejected"] is True
    assert np.array_equal(simex[0]["matrix"], matrix)
