import numpy as np

from masked_stellar_autoencoder.training.export_mixture_diagnostics import (
    affine_survey_components,
)


def test_zero_sum_can_fit_biased_survey_average_exactly():
    physical = np.array([5000.0, 2.0, -0.5])
    common = physical.copy()
    common[1] += 0.2
    delta = np.zeros((1, 3, 3))
    delta[0, :, 1] = [-0.2, 0.4, -0.2]
    mu = common[None, None]
    lower = np.eye(3)[None, None]
    mapped, _ = affine_survey_components(
        mu, lower, common[None], delta, np.zeros((1, 3, 3, 3))
    )
    np.testing.assert_allclose(delta.mean(1), 0, atol=1e-15)
    np.testing.assert_allclose(mapped[0, :, 0, 1], [2.0, 2.6, 2.0])
    np.testing.assert_allclose(mapped.mean(1), mu)
    assert common[1] != physical[1]


def test_zero_sum_derivatives_do_not_ensure_invertible_survey_maps():
    mu = np.zeros((1, 1, 3))
    lower = np.eye(3)[None, None]
    derivative = np.zeros((1, 3, 3, 3))
    derivative[0, :, 1, 1] = [-1.0, 0.5, 0.5]
    _, factors = affine_survey_components(
        mu, lower, mu[:, 0], np.zeros((1, 3, 3)), derivative
    )
    np.testing.assert_allclose(derivative.mean(1), 0)
    assert np.linalg.det(factors[0, 0, 0]) == 0


def test_affine_gauge_change_preserves_all_survey_distribution_components():
    rng = np.random.default_rng(41)
    mu = rng.normal(size=(2, 4, 3))
    ref = rng.normal(size=(2, 3))
    lower = np.broadcast_to(np.eye(3), (2, 4, 3, 3)).copy()
    delta = rng.normal(size=(2, 3, 3)) * 0.1
    deriv = rng.normal(size=(2, 3, 3, 3)) * 0.1
    before_mu, before_lower = affine_survey_components(mu, lower, ref, delta, deriv)
    t = np.diag([1.0, 1.2, 1.0])
    shift = np.array([0.0, -0.3, 0.0])
    new_mu = mu @ t.T + shift
    new_ref = ref @ t.T + shift
    new_lower = t @ lower
    new_delta = ref[:, None] + delta - new_ref[:, None]
    new_deriv = (np.eye(3) + deriv) @ np.linalg.inv(t) - np.eye(3)
    after_mu, after_lower = affine_survey_components(
        new_mu, new_lower, new_ref, new_delta, new_deriv
    )
    np.testing.assert_allclose(after_mu, before_mu, atol=1e-14)
    np.testing.assert_allclose(after_lower, before_lower, atol=1e-14)
