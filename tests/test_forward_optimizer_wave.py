import math

import torch
from torch import nn

from masked_stellar_autoencoder.training.forward_optimizer_wave import (
    measurement_nodes,
    observed_prior_log_prob,
    solve_one,
)


def test_independent_lbfgs_and_newton_recover_exact_affine_map():
    decoder = nn.Linear(3, 4, bias=False).double()
    with torch.no_grad():
        decoder.weight.copy_(
            torch.tensor([[1.0, 0.2, 0], [0, 2.0, 0.3], [0.1, 0, 1.0], [1.0, 1.0, 1.0]])
        )
    truth = torch.tensor([0.4, -0.7, 0.3], dtype=torch.double)
    x = decoder(truth).detach()
    variance = torch.tensor([0.1, 0.3, 0.2, 0.5], dtype=torch.double)
    prior = torch.distributions.MultivariateNormal(
        torch.zeros(3, dtype=torch.double), torch.eye(3, dtype=torch.double)
    )
    a = decoder.weight.detach()
    expected = torch.linalg.solve(
        a.T @ torch.diag(1 / variance) @ a + torch.eye(3), a.T @ (x / variance)
    )
    for method in ["lbfgs", "newton"]:
        result = solve_one(
            decoder,
            x,
            variance,
            torch.zeros((2, 3), dtype=torch.double),
            method=method,
            log_prior=prior.log_prob,
        )
        torch.testing.assert_close(result["prediction"], expected, atol=2e-7, rtol=2e-7)
        assert result["hessian_min_eigenvalue"] > 0


def test_prior_error_convolution_matches_analytic_normal():
    nodes = measurement_nodes(65536, dimensions=1)
    prior = torch.distributions.Independent(
        torch.distributions.Normal(
            torch.zeros(1, dtype=torch.double), torch.ones(1, dtype=torch.double)
        ),
        1,
    )
    y = torch.tensor([[0.2], [1.1]], dtype=torch.double)
    error = torch.full_like(y, 0.2)
    actual = observed_prior_log_prob(prior, y, error, nodes)
    expected = torch.distributions.Normal(0.0, math.sqrt(1.04)).log_prob(y[:, 0])
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(
        measurement_nodes(32), measurement_nodes(64)[:32], atol=0, rtol=0
    )


def test_laplace_importance_normalization_exact_affine_gaussian():
    from masked_stellar_autoencoder.training.forward_optimizer_wave import (
        laplace_importance_reference,
    )

    decoder = nn.Linear(1, 1, bias=False).double()
    with torch.no_grad():
        decoder.weight.fill_(1.5)
    prior = torch.distributions.Independent(
        torch.distributions.Normal(
            torch.zeros(1, dtype=torch.double), torch.ones(1, dtype=torch.double)
        ),
        1,
    )
    x = torch.tensor([0.7], dtype=torch.double)
    variance = torch.tensor([0.2], dtype=torch.double)
    result = solve_one(
        decoder,
        x,
        variance,
        torch.zeros((1, 1), dtype=torch.double),
        log_prior=prior.log_prob,
    )
    reference = laplace_importance_reference(
        decoder,
        x,
        variance,
        result["prediction"],
        result["hessian"],
        prior.log_prob,
        counts=(64, 128),
    )
    expected = torch.distributions.Normal(0.0, math.sqrt(2.45)).log_prob(x[0])
    for count, value in reference.items():
        assert abs(value["log_normalizer"] - expected) < 1e-7
        assert abs(value["effective_samples"] - int(count)) < 1e-7


def test_defensive_posterior_gaussian_reference():
    from masked_stellar_autoencoder.training.forward_optimizer_wave import (
        defensive_flow_posterior,
    )

    decoder = nn.Identity()
    prior = torch.distributions.MultivariateNormal(
        torch.zeros(1, dtype=torch.double), torch.ones((1, 1), dtype=torch.double)
    )
    result = defensive_flow_posterior(
        decoder,
        torch.tensor([0.7], dtype=torch.double),
        torch.tensor([0.25], dtype=torch.double),
        torch.tensor([0.56], dtype=torch.double),
        torch.tensor([[5.0]], dtype=torch.double),
        prior,
        count=32768,
    )
    torch.testing.assert_close(
        result["mean"], torch.tensor([0.56], dtype=torch.double), atol=0.01, rtol=0
    )
    torch.testing.assert_close(
        result["covariance"],
        torch.tensor([[0.2]], dtype=torch.double),
        atol=0.01,
        rtol=0,
    )
    assert result["effective_samples"] > 10000


def test_library_gauss_newton_matches_lbfgs_affine_likelihood():
    from torchregress.algorithms import ForwardInversionRegressor

    from masked_stellar_autoencoder.training.forward_optimizer_wave import (
        solve_library_gauss_newton,
    )

    decoder = nn.Linear(2, 3, bias=False).double()
    with torch.no_grad():
        decoder.weight.copy_(
            torch.tensor([[1.0, 0.3], [0.2, 2.0], [1.0, -1.0]], dtype=torch.double)
        )
    y = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-0.5, 0.5]], dtype=torch.double
    )
    x = decoder(y).detach()
    reg = ForwardInversionRegressor(
        decoder=decoder,
        epochs=1,
        lr=0.0,
        learn_noise=False,
        sigma_x=0.2,
        weight_decay=0.0,
    ).fit(x, y)
    reg.decoder_.requires_grad_(False)
    target = decoder(torch.tensor([0.7, -0.2], dtype=torch.double)).detach()
    starts = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double)
    gn = solve_library_gauss_newton(reg, target, torch.full_like(target, 0.2), starts)
    lbfgs = solve_one(reg.decoder_, target, torch.full_like(target, 0.04), starts)
    torch.testing.assert_close(
        gn["prediction"], lbfgs["prediction"], atol=1e-7, rtol=1e-7
    )
