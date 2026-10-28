"""Synthetic probability/covariance checks; no native model or checkpoint runs."""

import math

import pytest
import torch

from masked_stellar_autoencoder.training.remediation_joint6 import (
    JointSixHead,
    importance_observed_log_prob,
    masked_gaussian_log_prob,
    normal_qmc_nodes,
)


def correlated_parameters(rows=1):
    lower = torch.eye(6, dtype=torch.float64)
    lower[1, 0], lower[2, 0], lower[4, 3], lower[5, 1] = 0.65, -0.3, 0.45, 0.2
    lower.diagonal().copy_(torch.tensor([1.1, 0.7, 0.8, 1.0, 0.9, 0.6]))
    return torch.linspace(-0.3, 0.4, 6).double().expand(rows, -1), lower.expand(
        rows, -1, -1
    )


def masks():
    return torch.tensor([[bool(k & (1 << j)) for j in range(6)] for k in range(64)])


def set_gaussian(head, mean, lower, component=0):
    with torch.no_grad():
        bias = head.gaussian.net[-1].bias.view(head.gaussian.components, 28)
        bias[component, 1:7] = mean
        packed = lower[head.gaussian.row, head.gaussian.column].clone()
        diagonal = head.gaussian.row == head.gaussian.column
        packed[diagonal] = packed[diagonal].log()
        bias[component, 7:] = packed


def independent_marginal(mean, lower, target, sigma, mask):
    out = []
    covariance = lower @ lower.transpose(-1, -2)
    for row in range(len(target)):
        observed = mask[row].nonzero().flatten()
        if not len(observed):
            out.append(target.new_zeros(()))
        else:
            cov = covariance[row][observed][:, observed] + torch.diag(
                sigma[row, observed] ** 2
            )
            dist = torch.distributions.MultivariateNormal(mean[row, observed], cov)
            out.append(dist.log_prob(target[row, observed]))
    return torch.stack(out)


def test_full_covariance_gaussian_exact_every_missingness_pattern_with_noise():
    mean, lower = correlated_parameters(64)
    torch.manual_seed(17)
    target = torch.randn(64, 6, dtype=torch.float64)
    sigma = torch.rand_like(target) * 0.3
    mask = masks()
    target[~mask] = float("nan")
    actual = masked_gaussian_log_prob(mean, lower, target, sigma, mask)
    expected = independent_marginal(mean, lower, target, sigma, mask)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert actual[0] == 0


def test_full_covariance_mixture_uses_joint_component_and_exact_subvector():
    head = JointSixHead("mdn", latent_dim=3, width=16, components=2).double()
    mean, lower = correlated_parameters()
    set_gaussian(head, mean[0], lower[0], 0)
    set_gaussian(head, -mean[0], lower[0] * 0.8, 1)
    context = torch.zeros(3, 3, dtype=torch.float64)
    target = torch.tensor([[0.4, -0.7, 0.2, 0.8, 0.1, -0.6]]).double().expand(3, -1)
    mask = torch.tensor([[1, 0, 1, 0, 0, 1], [1] * 6, [0] * 6], dtype=torch.bool)
    sigma = torch.full_like(target, 0.15)
    a = independent_marginal(
        mean.expand(3, -1), lower.expand(3, -1, -1), target, sigma, mask
    )
    b = independent_marginal(
        -mean.expand(3, -1), (lower * 0.8).expand(3, -1, -1), target, sigma, mask
    )
    expected = torch.logaddexp(a, b) - math.log(2)
    torch.testing.assert_close(head.log_prob(context, target, sigma, mask), expected)


def test_sparse_importance_matches_correlated_gaussian_all_patterns_exact_proposal():
    mean, lower = correlated_parameters(64)
    torch.manual_seed(3)
    target = torch.randn(64, 6, dtype=torch.float64) * 0.4
    mask = masks()
    sigma = torch.zeros_like(target)
    nodes = normal_qmc_nodes(16, seed=71, like=target)

    def density(rows, values):
        return torch.distributions.MultivariateNormal(
            mean[rows], scale_tril=lower[rows]
        ).log_prob(values)

    actual, diagnostic = importance_observed_log_prob(
        density, mean, lower, target, sigma, mask, nodes
    )
    expected = independent_marginal(mean, lower, target, sigma, mask)
    torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        diagnostic["relative_ess"], torch.ones(64).double(), rtol=2e-12, atol=2e-12
    )


def test_sparse_importance_convolves_noise_against_analytic_gaussian():
    mean, lower = correlated_parameters(4)
    target = torch.tensor([[0.3, 0.5, -0.6, 0.2, 0.1, -0.4]]).double().expand(4, -1)
    sigma = torch.full_like(target, 0.25)
    mask = torch.tensor(
        [[1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1], [1] * 6, [0, 0, 0, 1, 1, 0]],
        dtype=torch.bool,
    )
    nodes = normal_qmc_nodes(4096, seed=901, like=target)

    def density(rows, values):
        return torch.distributions.MultivariateNormal(
            mean[rows], scale_tril=lower[rows]
        ).log_prob(values)

    actual, _ = importance_observed_log_prob(
        density, mean, lower, target, sigma, mask, nodes
    )
    expected = independent_marginal(mean, lower, target, sigma, mask)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.002)


def test_omitting_importance_proposal_correction_is_a_failing_counterexample():
    target = torch.zeros(1, 6, dtype=torch.float64)
    mean, lower = torch.zeros_like(target), torch.eye(6).double()[None]
    mask = torch.tensor([[True, True, True, False, False, False]])
    nodes = normal_qmc_nodes(64, seed=9, like=target)
    distribution = torch.distributions.MultivariateNormal(mean, scale_tril=lower)
    actual, _ = importance_observed_log_prob(
        lambda rows, values: distribution.log_prob(values),
        mean,
        lower,
        target,
        target,
        mask,
        nodes,
    )
    expected = -1.5 * math.log(2 * math.pi)
    torch.testing.assert_close(actual, torch.tensor([expected]).double())
    naive_values = nodes.new_zeros(64, 1, 6)
    naive_values[:, :, 3:] = nodes[:, None, 9:]
    naive = torch.logsumexp(distribution.log_prob(naive_values), 0) - math.log(64)
    assert (naive - actual).abs().item() > 1.0


def test_whole_vector_gaussian_samples_recover_offdiagonal_covariance():
    head = JointSixHead("gaussian", latent_dim=3, width=16).double()
    mean, lower = correlated_parameters()
    set_gaussian(head, mean[0], lower[0])
    sample = head.sample(
        torch.zeros(1, 3).double(), 20000, torch.Generator().manual_seed(104)
    )[:, 0]
    empirical = torch.cov(sample.T)
    torch.testing.assert_close(empirical, lower[0] @ lower[0].T, rtol=0, atol=0.04)
    assert empirical[0, 1] > 0.5


def test_joint_nsf_is_six_dimensional_nonfactorized_and_has_finite_gradients():
    torch.manual_seed(123)
    head = JointSixHead("nsf", latent_dim=3, width=16, particles=16).double()
    assert head.flow.target_dim == 6
    context = torch.randn(4, 3, dtype=torch.float64)
    target = torch.randn(4, 6, dtype=torch.float64) * 0.5
    sigma = torch.full_like(target, 0.05)
    mask = torch.tensor(
        [[1] * 6, [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [1, 0, 1, 0, 1, 0]],
        dtype=torch.bool,
    )
    objective = head.criterion(context, target, sigma, mask)
    objective.backward()
    gradients = [p.grad for p in head.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(g.abs().sum().item() for g in gradients) > 0
    fixed = context[:1]
    point = torch.full((6,), 0.2, dtype=torch.float64, requires_grad=True)
    hessian = torch.autograd.functional.hessian(
        lambda y: head.flow.distribution(fixed).log_prob(y[None]).sum(), point
    )
    offdiagonal = hessian - torch.diag(torch.diag(hessian))
    assert offdiagonal.abs().max() > 1e-6
    generator = torch.Generator().manual_seed(913)
    expected_base = torch.randn((20, 1, 6), dtype=torch.float64, generator=generator)
    expected = head.flow.distribution(fixed).transform.inv(expected_base)
    actual = head.sample(fixed, 20, torch.Generator().manual_seed(913))
    torch.testing.assert_close(actual, expected)
    diagnostic = head.diagnostics(context, target, sigma, mask, particles=16)
    assert torch.isfinite(diagnostic["independent_replica_log_prob_gap"]).all()
    assert torch.isfinite(diagnostic["refinement_log_prob_gap"]).all()
    assert not diagnostic["exact"]


def test_proposal_is_detached_in_importance_branch_but_auxiliary_trains_it():
    head = JointSixHead("nsf", latent_dim=2, width=16, particles=16).double()
    x = torch.zeros(2, 2).double()
    target = torch.ones(2, 6).double() * 0.4
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]], dtype=torch.bool)
    sigma = torch.zeros_like(target)
    head.log_prob(x, target, sigma, mask).sum().backward()
    assert all(p.grad is None for p in head.gaussian.parameters())
    head.zero_grad()
    head.criterion(x, target, sigma, mask).backward()
    assert head.gaussian.net[-1].bias.grad.abs().sum() > 0


def test_native_jacobian_and_positive_log_coordinate_samples():
    head = JointSixHead("gaussian", latent_dim=2, width=16).double()
    x = torch.zeros(1, 2).double()
    native = torch.tensor([[1, 2, -0.5, 0.2, 3, 2]]).double()
    center, scale = torch.zeros(6).double(), torch.ones(6).double() * 2
    mask = torch.ones(1, 6, dtype=torch.bool)
    coordinate = native.clone()
    coordinate[:, 4:] = coordinate[:, 4:].log()
    expected = (
        torch.distributions.Normal(0.0, 1.0).log_prob(coordinate / scale).sum(-1)
        - scale.log().sum()
        - native[:, 4:].log().sum(-1)
    )
    actual = head.native_log_prob(
        x, native, torch.zeros_like(native), mask, center, scale
    )
    torch.testing.assert_close(actual, expected)
    samples = head.native_samples(x, 10, center, scale)
    assert torch.isfinite(samples).all() and (samples[..., 4:] > 0).all()
    with pytest.raises(ValueError, match="widths"):
        head.native_log_prob(x, native, torch.ones_like(native), mask, center, scale)


def test_invalid_masks_values_and_empty_rows():
    head = JointSixHead("gaussian", latent_dim=2, width=16).double()
    x, y = torch.zeros(3, 2).double(), torch.zeros(3, 6).double()
    mask = torch.zeros_like(y, dtype=torch.bool)
    torch.testing.assert_close(head.log_prob(x, y, y, mask), torch.zeros(3).double())
    assert head.criterion(x, y, y, mask) == 0
    with pytest.raises(ValueError, match="boolean"):
        head.log_prob(x, y, y, mask.double())
    with pytest.raises(ValueError, match="finite"):
        head.log_prob(x, y + float("nan"), y, ~mask)
    with pytest.raises(ValueError, match="particle"):
        JointSixHead("nsf", particles=3)
