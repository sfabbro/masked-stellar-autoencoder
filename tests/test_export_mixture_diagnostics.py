import numpy as np
import pytest
from scipy.special import ndtr

from masked_stellar_autoencoder.training.export_mixture_diagnostics import (
    marginal_scores,
    summarize_residuals,
)


def test_normal_closed_form_and_reference_noise():
    y = np.array([-1.0, 0.0, 2.0])
    w = np.ones((3, 1))
    m = np.zeros_like(w)
    s = marginal_scores(w, m, np.ones_like(w), y, np.ones(3))
    z = y / np.sqrt(2)
    np.testing.assert_allclose(s["pit"], ndtr(z))
    np.testing.assert_allclose(s["nll"], 0.5 * np.log(4 * np.pi) + y**2 / 4)
    assert s["crps"][1] == pytest.approx(np.sqrt(2) * (np.sqrt(2) - 1) / np.sqrt(np.pi))


def test_split_component_invariance_and_permutation():
    w = np.array([[0.2, 0.8], [0.3, 0.7]])
    m = np.array([[-2.0, 3.0], [1.0, 5.0]])
    v = np.array([[1.0, 2.0], [4.0, 3.0]])
    y = np.array([0.0, 2.0])
    a = marginal_scores(w, m, v, y)
    b = marginal_scores(
        np.repeat(w / 2, 2, axis=1)[:, ::-1],
        np.repeat(m, 2, axis=1)[:, ::-1],
        np.repeat(v, 2, axis=1)[:, ::-1],
        y,
    )
    for key in a:
        np.testing.assert_allclose(a[key], b[key])
    with pytest.raises(ValueError):
        marginal_scores(w, m, -v, y)
    assert summarize_residuals(np.ones(5), np.zeros(5))["bias"] == 1


def test_affine_mapping_preserves_component_covariance_and_reference_origin():
    from masked_stellar_autoencoder.training.export_mixture_diagnostics import (
        affine_survey_components,
    )

    mean = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    lower = np.broadcast_to(
        np.array([[2.0, 0, 0], [1.0, 3.0, 0], [0.5, 0.1, 1.0]]), (1, 2, 3, 3)
    ).copy()
    ref = np.array([[10.0, 20.0, 30.0]])
    offsets = np.array([[[0.1, 0.2, 0.3]]])
    jac = np.array([[2.0, 0.5, 0], [0, 1.0, 0], [0, 0, 3.0]])
    mu, factor = affine_survey_components(
        mean, lower, ref, offsets, (jac - np.eye(3))[None, None]
    )
    np.testing.assert_allclose(
        mu[0, 0, 0], ref[0] + offsets[0, 0] + jac @ (mean[0, 0] - ref[0])
    )
    cov = lower[0, 0] @ lower[0, 0].T
    np.testing.assert_allclose(factor[0, 0, 0] @ factor[0, 0, 0].T, jac @ cov @ jac.T)
    same, copied = affine_survey_components(
        mean, lower, ref, np.zeros((1, 1, 3)), np.zeros((1, 1, 3, 3))
    )
    np.testing.assert_allclose(same[:, 0], mean)
    np.testing.assert_allclose(copied[:, 0], lower)
