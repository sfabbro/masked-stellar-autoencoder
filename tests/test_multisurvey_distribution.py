"""Independent observation-space and joint-sampling regression checks."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.multisurvey_distribution import (
    joint_gaussian_draws,
    observation_marginals,
)


def test_observation_variance_includes_projection_and_measurement_noise():
    mean = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    lower = torch.tensor([[[2.0, 0.0], [1.0, 3.0]]], dtype=torch.float64)
    jac = torch.tensor(
        [[[1.0, 2.0], [float("nan"), float("nan")]]], dtype=torch.float64
    )
    packed = SimpleNamespace(
        jacobian=jac,
        mask=torch.tensor([[True, False]]),
        measurement_covariance=torch.diag_embed(
            torch.tensor([[4.0, float("nan")]], dtype=torch.float64)
        ),
        mapped_mean=torch.tensor([[5.0, float("nan")]], dtype=torch.float64),
        values=torch.tensor([[5.0, float("nan")]], dtype=torch.float64),
    )
    mapped, variance, pit = observation_marginals(mean, lower, packed)
    # (1,2) L = (4,6), giving latent variance 52 plus noise variance 4.
    assert variance[0, 0].item() == pytest.approx(56.0)
    assert pit[0, 0].item() == 0.5
    assert torch.isnan(pit[0, 1])
    assert mapped[0, 0] == 5


def test_joint_draws_reproduce_covariance_not_independent_marginals():
    mean = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    lower = torch.tensor([[[2.0, 0.0], [1.0, 3.0]]], dtype=torch.float64)
    draws = joint_gaussian_draws(
        mean, lower, 100000, generator=torch.Generator().manual_seed(42)
    )
    np.testing.assert_allclose(draws[0].mean(0), mean[0], atol=0.025)
    np.testing.assert_allclose(
        np.cov(draws[0].numpy(), rowvar=False), [[4.0, 2.0], [2.0, 10.0]], atol=0.12
    )
    with pytest.raises(ValueError):
        joint_gaussian_draws(mean, lower, 0)
