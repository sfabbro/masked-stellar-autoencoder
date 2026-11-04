"""Independent likelihood, gradient and non-Gaussian-head admission checks."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.stats import multivariate_normal

from masked_stellar_autoencoder.training.multisurvey_flow import (
    CopulaResidualHead,
    GaussianResidualHead,
    MixtureResidualHead,
    NSFResidualHead,
    gaussian_observation_log_prob,
    importance_log_prob,
    normal_nodes,
    reference_posterior,
    whiten_observations,
)


def example():
    torch.manual_seed(7)
    lower = torch.tensor([[[2.0, 0.0], [0.5, 1.0]]], dtype=torch.float64).expand(
        3, -1, -1
    )
    design = torch.randn(3, 4, 2, dtype=torch.float64)
    noise_raw = torch.randn(3, 4, 4, dtype=torch.float64)
    noise = noise_raw @ noise_raw.transpose(-1, -2) + torch.eye(4)
    mask = torch.tensor([[1, 1, 1, 1], [1, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
    packed = SimpleNamespace(
        mask=mask,
        values=torch.randn(3, 4, dtype=torch.float64),
        mapped_mean=torch.randn(3, 4, dtype=torch.float64),
        jacobian=design,
        measurement_covariance=noise,
    )
    return lower, packed


def test_whitened_reference_matches_independent_scipy_with_missing_and_empty():
    lower, packed = example()
    observation = whiten_observations(lower, packed)
    mean, proposal_lower, log_prob = reference_posterior(observation)
    for b in range(3):
        mask = packed.mask[b].numpy()
        if not mask.any():
            assert log_prob[b] == 0
            torch.testing.assert_close(mean[b], torch.zeros(2, dtype=torch.float64))
            continue
        h = (packed.jacobian[b] @ lower[b]).numpy()[mask]
        r = packed.measurement_covariance[b].numpy()[np.ix_(mask, mask)]
        residual = (packed.values[b] - packed.mapped_mean[b]).numpy()[mask]
        expected = multivariate_normal.logpdf(residual, cov=h @ h.T + r)
        assert log_prob[b].item() == pytest.approx(expected, abs=1e-11)
        covariance = np.linalg.inv(np.eye(2) + h.T @ np.linalg.solve(r, h))
        np.testing.assert_allclose(
            (proposal_lower[b] @ proposal_lower[b].T).numpy(), covariance, atol=1e-12
        )
        np.testing.assert_allclose(
            mean[b].numpy(), covariance @ h.T @ np.linalg.solve(r, residual), atol=1e-12
        )


@pytest.mark.parametrize(
    "kind", [GaussianResidualHead, CopulaResidualHead, NSFResidualHead]
)
def test_flow_starts_at_reference_and_has_finite_gradients(kind):
    torch.manual_seed(20)
    head = kind(4, dims=3, width=8).double()
    context = torch.randn(3, 4, dtype=torch.float64)
    value = torch.randn(16, 3, 3, dtype=torch.float64)
    actual = head.log_prob(context, value)
    expected = -0.5 * (value.square().sum(-1) + 3 * math.log(2 * math.pi))
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    (-actual.mean()).backward()
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0
    assert head.sample(context, 32).shape == (32, 3, 3)


def test_importance_identity_and_gradients_agree_with_analytic_gaussian():
    lower, packed = example()
    obs = whiten_observations(lower, packed)
    proposal_mean, proposal_lower, lp0 = reference_posterior(obs)
    head = GaussianResidualHead(4, dims=2, width=8).double()
    context = torch.randn(3, 4, dtype=torch.float64)
    nodes = normal_nodes(32768, 3, 2, seed=44, like=context)
    estimate, ess = importance_log_prob(
        head, context, proposal_mean, proposal_lower, lp0, obs["count"], nodes
    )
    torch.testing.assert_close(estimate, lp0, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(ess, torch.ones_like(ess), atol=1e-11, rtol=1e-11)
    mean, chol = head(context)
    exact = gaussian_observation_log_prob(mean, chol, obs)
    out = head.gaussian.net[-1]
    exact_grad = torch.autograd.grad(exact.sum(), out.bias, retain_graph=True)[0]
    qmc_grad = torch.autograd.grad(estimate.sum(), out.bias)[0]
    torch.testing.assert_close(qmc_grad, exact_grad, atol=0.003, rtol=0.003)


def test_nonidentity_gaussian_integral_and_multimodal_mixture_are_normalized():
    head = MixtureResidualHead(2, dims=2, width=4, components=2).double()
    with torch.no_grad():
        # Separate modes at -2,+2 in coordinate one with narrow within-mode spread.
        raw = head.net[-1].bias.reshape(2, -1)
        raw[:, 1:3] = torch.tensor([[-2.0, 0.0], [2.0, 0.0]])
        raw[:, 3:] = 0
        raw[:, 3:][:, head.row == head.column] = math.log(math.expm1(0.3 - 1e-4))
    context = torch.zeros(1, 2, dtype=torch.float64)
    peaks = head.log_prob(
        context,
        torch.tensor([[[-2.0, 0.0]], [[0.0, 0.0]], [[2.0, 0.0]]], dtype=torch.float64),
    )
    assert peaks[0] > peaks[1] + 10 and peaks[2] > peaks[1] + 10
    assert head.sample(context, 64).shape == (64, 1, 2)


def test_non_gaussian_joint_noise_integral_matches_exact_mixture():
    observation = dict(
        design=torch.tensor([[[1.0], [2.0]], [[0.5], [1.0]]], dtype=torch.float64),
        residual=torch.tensor([[0.3, -0.2], [1.0, 0.2]], dtype=torch.float64),
        noise_logdet=torch.zeros(2, dtype=torch.float64),
        count=torch.full((2,), 2),
    )
    mean, lower, lp0 = reference_posterior(observation)
    head = MixtureResidualHead(4, dims=1, width=8, components=2).double()
    with torch.no_grad():
        raw = head.net[-1].bias.reshape(2, -1)
        raw[:, 0] = torch.tensor([-0.4, 0.4])
        raw[:, 1] = torch.tensor([-0.8, 0.8])
        raw[:, 2] = math.log(math.expm1(0.7 - 1e-4))
    context = torch.zeros(2, 4, dtype=torch.float64)
    nodes = normal_nodes(32768, 2, 1, seed=18, like=context)
    approximate, ess = importance_log_prob(
        head, context, mean, lower, lp0, observation["count"], nodes
    )
    exact = head.observation_log_prob(context, observation)
    torch.testing.assert_close(approximate, exact, atol=0.001, rtol=0.001)
    assert (ess > 0.1).all()


def test_measurement_unit_change_preserves_posterior_and_density_jacobian():
    lower, packed = example()
    reference = reference_posterior(whiten_observations(lower, packed))
    scale = torch.tensor([1000.0, 0.01, 10.0, 1.0], dtype=torch.float64)
    rescaled = SimpleNamespace(
        mask=packed.mask,
        values=packed.values * scale,
        mapped_mean=packed.mapped_mean * scale,
        jacobian=packed.jacobian * scale[None, :, None],
        measurement_covariance=packed.measurement_covariance
        * scale[None, :, None]
        * scale[None, None, :],
    )
    changed = reference_posterior(whiten_observations(lower, rescaled))
    torch.testing.assert_close(changed[0], reference[0], atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(changed[1], reference[1], atol=1e-11, rtol=1e-11)
    expected = reference[2] - (packed.mask * scale.log()).sum(-1)
    torch.testing.assert_close(changed[2], expected, atol=1e-11, rtol=1e-11)


def test_ensemble_crps_matches_independent_pairwise_definition():
    from masked_stellar_autoencoder.training.multisurvey_flow import ensemble_crps

    samples = torch.tensor(
        [[-2.0, 1.0], [-0.5, 2.0], [0.5, 4.0], [3.0, 5.0]], dtype=torch.float64
    )
    truth = torch.tensor([0.25, 3.0], dtype=torch.float64)
    pairs = (samples[:, None] - samples[None, :]).abs().sum((0, 1))
    expected = (samples - truth).abs().mean(0) - pairs / (2 * 4 * 3)
    torch.testing.assert_close(ensemble_crps(samples, truth), expected)
    torch.testing.assert_close(
        ensemble_crps(1000 * samples + 4, 1000 * truth + 4), 1000 * expected
    )


@pytest.mark.parametrize(
    "kind",
    [GaussianResidualHead, MixtureResidualHead, CopulaResidualHead, NSFResidualHead],
)
def test_fresh_instance_reload_preserves_joint_density(kind):
    torch.manual_seed(71)
    first = kind(4, width=8).double()
    x = torch.randn(3, 4, dtype=torch.float64)
    y = torch.randn(16, 3, 3, dtype=torch.float64)
    optimizer = torch.optim.Adam(first.parameters(), lr=0.001)
    optimizer.zero_grad()
    (-first.log_prob(x, y).mean()).backward()
    optimizer.step()
    expected = first.log_prob(x, y).detach()
    torch.manual_seed(999)
    second = kind(4, width=8).double()
    second.load_state_dict(first.state_dict(), strict=True)
    torch.testing.assert_close(second.log_prob(x, y), expected, atol=1e-12, rtol=1e-12)
