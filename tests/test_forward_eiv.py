import math

import numpy as np
import torch
from torch import nn

from masked_stellar_autoencoder.training.forward_eiv import (
    IntegratedForward,
    conditional_log_prob,
    grid_predictions,
)


def test_integrated_forward_matches_analytic_gaussian_convolution():
    model = IntegratedForward(2).double()
    model.decoder = nn.Linear(1, 2, bias=False).double()
    with torch.no_grad():
        model.decoder.weight.copy_(torch.tensor([[1.0], [0.7]], dtype=torch.double))
        model.log_intrinsic_variance.fill_(math.log(0.4))
    y = torch.tensor([-0.6, 0.8], dtype=torch.double)
    sy = torch.tensor([0.12, 0.2], dtype=torch.double)
    x = torch.tensor([[-0.4, -0.5], [1.2, 0.4]], dtype=torch.double)
    sx = torch.full_like(x, 0.2)
    slope = model.decoder.weight.flatten()
    covariance = torch.diag_embed(sx.square() + 0.4) + sy[
        :, None, None
    ].square() * torch.outer(slope, slope)
    reference = (
        -torch.distributions.MultivariateNormal(
            y[:, None] * slope, covariance_matrix=covariance
        )
        .log_prob(x)
        .mean()
        / 2
    )
    torch.testing.assert_close(
        model.objective(x, sx, y, sy), reference, atol=1e-10, rtol=1e-10
    )


def test_grid_matches_independent_explicit_gaussian_and_uses_no_labels():
    decoder = nn.Linear(1, 2).double()
    with torch.no_grad():
        decoder.weight.copy_(torch.tensor([[1.0], [2.0]]))
        decoder.bias.zero_()
    grid = torch.linspace(-3, 3, 401, dtype=torch.double)
    x = torch.tensor([[0.2, 0.5], [-0.5, -0.9]], dtype=torch.double)
    sx = torch.full_like(x, 0.2)
    variance = torch.tensor([0.2, 0.3], dtype=torch.double)
    actual = grid_predictions(decoder, variance, x, sx, grid)
    explicit = conditional_log_prob(x, sx, decoder(grid[:, None])[None, ...], variance)
    weights = torch.ones_like(grid)
    weights[[0, -1]] = 0.5
    probability = torch.softmax(explicit + weights.log(), -1)
    np.testing.assert_allclose(actual[:, 0], grid[explicit.argmax(-1)].numpy())
    np.testing.assert_allclose(
        actual[:, 1], (probability @ grid).detach().numpy(), rtol=1e-10, atol=1e-10
    )


def test_flow_objective_has_finite_prior_decoder_and_mismatch_gradients():
    torch.manual_seed(42)
    model = IntegratedForward(3, flow=True)
    loss = model.objective(
        torch.randn(8, 3),
        torch.full((8, 3), 0.1),
        torch.randn(8),
        torch.full((8,), 0.2),
    )
    loss.backward()
    assert torch.isfinite(loss)
    for module in [model.decoder, model.prior]:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    assert torch.isfinite(model.log_intrinsic_variance.grad).all()


def test_flow_prior_integral_matches_joint_gaussian_high_snr():
    class GaussianPrior(nn.Module):
        def forward(self):
            return torch.distributions.Independent(
                torch.distributions.Normal(
                    torch.zeros(1, dtype=torch.double),
                    torch.ones(1, dtype=torch.double),
                ),
                1,
            )

    model = IntegratedForward(2).double()
    model.decoder = nn.Linear(1, 2, bias=False).double()
    model.prior = GaussianPrior()
    with torch.no_grad():
        model.decoder.weight.copy_(torch.tensor([[1.0], [0.7]], dtype=torch.double))
        model.log_intrinsic_variance.fill_(math.log(0.0004))
    x = torch.tensor([[0.2, 0.15], [-0.7, -0.5]], dtype=torch.double)
    sx = torch.full_like(x, 0.02)
    y = torch.tensor([0.1, -0.8], dtype=torch.double)
    sy = torch.full_like(y, 0.3)
    loading = torch.tensor([1.0, 0.7, 1.0], dtype=torch.double)
    covariance = torch.outer(loading, loading) + torch.diag(
        torch.tensor([0.0008, 0.0008, 0.09], dtype=torch.double)
    )
    joint = torch.cat([x, y[:, None]], -1)
    reference = (
        -torch.distributions.MultivariateNormal(
            torch.zeros(3, dtype=torch.double), covariance_matrix=covariance
        )
        .log_prob(joint)
        .mean()
        / 2
    )
    torch.testing.assert_close(
        model.objective(x, sx, y, sy), reference, atol=1e-9, rtol=1e-9
    )
