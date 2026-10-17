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
    standardized_xp_errors,
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
        {**spec, "method": "rc_shared"},
        matrix,
        raw_error,
        scale,
        names,
        fit,
        g_magnitude=np.full(len(matrix), 8.5),
        error_convention="raw_xp",
    )
    assert len(replicas) == 1
    np.testing.assert_allclose(replicas[0]["matrix"][:, :5], matrix[:, :5])
    assert not np.allclose(replicas[0]["matrix"][:, columns], matrix[:, columns])
    simex, report = prepare_replicas(
        {
            **spec,
            "method": "simex",
            "simex_extrapolant": "affine_rational",
            "simex_n_simulations": 4,
            "simex_extrapolation_order": 2,
        },
        matrix,
        raw_error,
        scale,
        names,
        fit,
        g_magnitude=np.full(len(matrix), 8.5),
        error_convention="raw_xp",
    )
    assert len(simex) == 17
    assert report["library_quadratic_extrapolant_rejected"] is True
    assert np.array_equal(simex[0]["matrix"], matrix)


def test_rational_simex_preserves_constants_and_label_origin():
    lambdas = [0.0, 0.5, 1.0, 1.5, 2.0]
    constants = np.array([-2.0, -0.2, 0.0, 0.3, 2.0])
    np.testing.assert_array_equal(
        rational_extrapolate(lambdas, np.tile(constants, (5, 1))), constants
    )
    curve = 2 / (2 + np.asarray(lambdas))
    base = rational_extrapolate(lambdas, curve[:, None])
    for shift in [-3.0, 0.7, 10.0]:
        np.testing.assert_allclose(
            rational_extrapolate(lambdas, (curve + shift)[:, None]),
            base + shift,
            atol=1e-10,
        )
    # Classical attenuation with an unknown nonzero response intercept.
    assert abs(float(rational_extrapolate(lambdas, (curve + 7)[:, None])[0]) - 9) < 0.02


def test_rational_simex_rejects_invalid_curves():
    import pytest

    for lambdas in [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [0.0, np.nan, 1.0],
        [1.0, 2.0, 3.0],
    ]:
        with pytest.raises(ValueError):
            rational_extrapolate(lambdas, np.ones((3, 1)))
    with pytest.raises(ValueError):
        rational_extrapolate([0.0, 1.0, 2.0], np.full((3, 1), np.nan))


def test_raw_xp_errors_need_G_flux_normalization_but_values_do_not():
    error = np.array([[2.0, 4.0, np.nan], [2.0, 4.0, np.nan]])
    scale = np.array([2.0, 4.0, 1.0])
    g = np.array([8.5, 13.5])
    sigma = standardized_xp_errors(
        error, scale, np.array([0, 1]), g_magnitude=g, error_convention="raw_xp"
    )
    np.testing.assert_allclose(sigma, [[1.0, 1.0], [100.0, 100.0]])
    # Same physical errors already expressed on coefficient G-normalised scale.
    normalized = error.copy()
    normalized[1, :2] *= 100
    same = standardized_xp_errors(
        normalized,
        scale,
        np.array([0, 1]),
        g_magnitude=g,
        error_convention="g_normalized_xp",
    )
    np.testing.assert_allclose(same, sigma)
    np.testing.assert_array_equal(error[:, :2], [[2.0, 4.0], [2.0, 4.0]])


def test_xp_errors_accept_explicit_mixed_conventions_and_reject_ambiguous_units():
    import pytest

    error = np.array([[2.0, 4.0], [200.0, 400.0]])
    kwargs = dict(g_magnitude=np.array([13.5, 13.5]))
    sigma = standardized_xp_errors(
        error,
        np.array([2.0, 4.0]),
        np.array([0, 1]),
        error_convention=np.array(["raw_xp", "g_normalized_xp"]),
        **kwargs,
    )
    np.testing.assert_allclose(sigma, np.full((2, 2), 100.0))
    for convention in ["unknown", "scaled", np.array(["raw_xp"])]:
        with pytest.raises(ValueError, match="convention"):
            standardized_xp_errors(
                error,
                np.ones(2),
                np.array([0, 1]),
                error_convention=convention,
                **kwargs,
            )
    for g in [np.array([np.nan, 13.5]), np.array([-9999.0, 13.5]), np.ones(3)]:
        with pytest.raises(ValueError, match="G magnitudes"):
            standardized_xp_errors(
                error,
                np.ones(2),
                np.array([0, 1]),
                g_magnitude=g,
                error_convention="raw_xp",
            )
    with pytest.raises(ValueError, match="scales"):
        standardized_xp_errors(
            error,
            np.array([0.0, 1.0]),
            np.array([0, 1]),
            error_convention="raw_xp",
            **kwargs,
        )


def test_actual_original_record_error_audit_fails_on_units_mismatch(tmp_path):
    import json

    import h5py
    import pytest
    from astropy.table import Table

    from masked_stellar_autoencoder.training.input_measurement import (
        audit_xp_error_sources,
    )

    names = [f"{b}_{i}" for b in ("bp", "rp") for i in range(1, 56)] + ["G"]
    errors = [f"{b}e_{i}" for b in ("bp", "rp") for i in range(1, 56)] + [None]
    dtype = [("source_id", "i8"), ("G", "f4")] + [
        (n, "f4") for n in names[:-1] + errors[:-1]
    ]
    original = np.zeros(3, dtype=dtype)
    original["source_id"] = [11, 22, 33]
    original["G"] = [10.0, 15.0, 13.0]
    for i, n in enumerate(names[:-1]):
        original[n] = np.arange(3) + i + 1
    for n in errors[:-1]:
        original[n] = [0.1, 0.2, 0.3]
    legacy = tmp_path / "legacy.fits"
    Table(original).write(legacy)
    benchmark = tmp_path / "benchmark.h5"
    with h5py.File(benchmark, "w") as f:
        f.create_dataset("train/records", data=original[:2])
        f.create_dataset("validation/records", data=original[:0])
    cohort = tmp_path / "cohort.h5"
    values = np.column_stack([original[n] for n in names])
    uncertainty = np.column_stack(
        [original[n] if n is not None else np.full(3, np.nan) for n in errors]
    )
    with h5py.File(cohort, "w") as f:
        f.attrs["error_columns_json"] = json.dumps(errors)
        for name, val in dict(
            source_id=original["source_id"],
            input_source=[0, 0, 2],
            input_original_row=[0, 1, 2],
            values=values,
            reported_feature_errors=uncertainty,
        ).items():
            f.create_dataset("features/" + name, data=val)
    result = audit_xp_error_sources(
        cohort, benchmark, legacy, np.arange(3), names, common_rows=2
    )
    assert result["status"] == "verified" and result["selected_rows"] == 3
    assert result["all110_errors_match_original_benchmark_and_legacy"]
    with h5py.File(cohort, "r+") as f:
        f["features/reported_feature_errors"][2, 0] *= 100
    with pytest.raises(ValueError, match="selected raw errors"):
        audit_xp_error_sources(
            cohort, benchmark, legacy, np.arange(3), names, common_rows=2
        )
    with h5py.File(cohort, "r+") as f:
        f["features/reported_feature_errors"][2, 0] = original[errors[0]][2]
    # Different legacy error convention would not be inferred from identical coefficients.
    original[errors[0]][0] *= 100
    Table(original).write(legacy, overwrite=True)
    with pytest.raises(ValueError, match="common raw bpe_1"):
        audit_xp_error_sources(
            cohort, benchmark, legacy, np.arange(3), names, common_rows=2
        )


def test_finite_large_parent_coefficients_are_not_arbitrarily_rejected():
    from masked_stellar_autoencoder.training.input_measurement import _as_float32

    expected = np.array([[146.5087, -200.0]], dtype="f4")
    np.testing.assert_array_equal(_as_float32(torch.tensor(expected), 1), expected)
