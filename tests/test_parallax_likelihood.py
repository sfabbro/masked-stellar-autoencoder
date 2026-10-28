"""Independent measurement-space and gradient checks for the proposed head."""

import math

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.parallax_likelihood import (
    distance_modulus_to_parallax_mas,
    signed_parallax_nll,
)


def test_physical_unit_conversion():
    mu = torch.tensor([0.0, 5.0, 10.0, 15.0, 20.0], dtype=torch.float64)
    # Independent parsec definition rather than copying the implementation.
    distance_pc = 10 ** ((mu.numpy() + 5) / 5)
    np.testing.assert_allclose(
        distance_modulus_to_parallax_mas(mu).numpy(), 1000 / distance_pc, rtol=1e-14
    )


def test_negative_zero_and_positive_measurements_and_analytic_gradient():
    mu = torch.tensor([10.0, 15.0, 20.0], dtype=torch.float64, requires_grad=True)
    measured = torch.tensor([-0.2, 0.0, 0.03], dtype=torch.float64)
    error = torch.tensor([0.1, 0.02, 0.03], dtype=torch.float64)
    zero_point = -0.017
    loss, count = signed_parallax_nll(mu, measured, error, zero_point_mas=zero_point)
    q = 1000 / (10 ** ((mu.detach().numpy() + 5) / 5))
    residual = measured.numpy() - zero_point - q
    reference = np.mean(
        residual**2 / (2 * error.numpy() ** 2)
        + np.log(error.numpy() * np.sqrt(2 * np.pi))
    )
    assert count == 3
    np.testing.assert_allclose(loss.detach().numpy(), reference, rtol=1e-13)
    loss.backward()
    derivative = residual * q * math.log(10) / (5 * error.numpy() ** 2 * count)
    np.testing.assert_allclose(mu.grad.numpy(), derivative, rtol=1e-13)
    assert mu.grad[0] < 0  # A negative observation pushes toward smaller q/larger d.


def test_missing_and_ineligible_labels_have_no_gradient_or_imputation():
    mu = torch.tensor([10.0, float("nan"), 12.0, 13.0, 14.0], requires_grad=True)
    measured = torch.tensor([1.0, float("nan"), 0.2, 0.3, 0.4])
    error = torch.tensor([0.1, 0.1, 0.0, float("inf"), 0.1])
    eligible = torch.tensor([True, True, True, True, False])
    loss, count = signed_parallax_nll(mu, measured, error, eligible=eligible)
    assert count == 1
    loss.backward()
    assert torch.isfinite(mu.grad).all()
    assert torch.equal(mu.grad[1:], torch.zeros(4))


def test_empty_valid_batch_has_differentiable_zero():
    mu = torch.tensor([float("nan"), 10.0], requires_grad=True)
    loss, count = signed_parallax_nll(
        mu, torch.tensor([float("nan"), 0.0]), torch.zeros(2)
    )
    assert count == 0 and loss.item() == 0
    loss.backward()
    assert torch.equal(mu.grad, torch.zeros(2))


@pytest.mark.parametrize("bad_mu", [float("nan"), float("inf"), -1e6, 1e6])
def test_invalid_model_predictions_are_rejected_instead_of_masked(bad_mu):
    with pytest.raises(ValueError, match="positive model parallaxes"):
        signed_parallax_nll(
            torch.tensor([bad_mu]), torch.tensor([0.0]), torch.tensor([0.1])
        )


def test_no_implicit_broadcast_or_nonboolean_eligibility():
    mu = torch.zeros(2)
    with pytest.raises(ValueError, match="shapes"):
        signed_parallax_nll(mu, torch.ones(2, 1), torch.ones(2))
    with pytest.raises(ValueError, match="boolean"):
        signed_parallax_nll(mu, torch.ones(2), torch.ones(2), eligible=torch.ones(2))


def test_torch_finite_difference_gradcheck():
    mu = torch.tensor([9.2, 12.7], dtype=torch.float64, requires_grad=True)
    observed = torch.tensor([-0.1, 0.3], dtype=torch.float64)
    errors = torch.tensor([0.2, 0.1], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda x: signed_parallax_nll(x, observed, errors)[0], (mu,)
    )
