import math

import pytest
import torch

from masked_stellar_autoencoder.training.conditional_flow_head import (
    ConditionalFlowHead,
)


@pytest.mark.parametrize("kind,bins", [("maf", 8), ("nsf", 8), ("nsf", 16)])
def test_identity_conditional_density_and_moments(kind, bins):
    torch.manual_seed(42)
    head = ConditionalFlowHead(kind=kind, bins=bins, context=5)
    latent = torch.randn(7, 5)
    y = torch.linspace(-2, 2, 7)
    output = head(latent)
    torch.testing.assert_close(output[:, :, 0], torch.zeros(7, 1), rtol=0, atol=2e-7)
    torch.testing.assert_close(output[:, :, 1], torch.zeros(7, 1), rtol=0, atol=2e-6)
    reference = 0.5 * y.square() + 0.5 * math.log(2 * math.pi)
    torch.testing.assert_close(
        head.distribution(latent).log_prob(y[:, None]), -reference, rtol=1e-5, atol=2e-6
    )
    torch.testing.assert_close(
        head.cdf(latent, y),
        torch.distributions.Normal(0, 1).cdf(y),
        rtol=1e-5,
        atol=2e-6,
    )


@pytest.mark.parametrize("kind", ["maf", "nsf"])
def test_nonidentity_change_of_variables_quantiles_and_reload(kind):
    torch.manual_seed(3)
    head = ConditionalFlowHead(kind=kind, context=4).double()
    with torch.no_grad():
        for transform in head.flow.transform.transforms:
            final = [
                m
                for m in transform.hyper.modules()
                if hasattr(m, "weight") and hasattr(m, "bias")
            ][-1]
            final.weight.normal_(std=0.02)
            final.bias.normal_(std=0.1)
    latent = torch.randn(5, 4, dtype=torch.double)
    base = torch.linspace(-1, 1, 5, dtype=torch.double)[:, None].requires_grad_(True)
    dist = head.distribution(latent)
    y = dist.transform.inv(base)
    derivative = torch.autograd.grad(y.sum(), base, retain_graph=True)[0]
    expected = (
        -0.5 * base.square() - 0.5 * math.log(2 * math.pi) - derivative.log()
    ).flatten()
    torch.testing.assert_close(dist.log_prob(y), expected, rtol=1e-7, atol=1e-7)
    q = head.quantiles(latent, [0.025, 0.5, 0.975])
    for i, p in enumerate([0.025, 0.5, 0.975]):
        torch.testing.assert_close(
            head.cdf(latent, q[:, i]),
            torch.full((5,), p, dtype=torch.double),
            rtol=1e-6,
            atol=1e-7,
        )
    other = ConditionalFlowHead(kind=kind, context=4).double()
    other.load_state_dict(head.state_dict(), strict=True)
    torch.testing.assert_close(other(latent), head(latent), rtol=0, atol=0)


def test_density_normalization_and_finite_conditional_training_gradient():
    torch.manual_seed(4)
    head = ConditionalFlowHead(context=3).double()
    latent = torch.randn(2, 3, dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.002)
    for _ in range(3):
        optimizer.zero_grad()
        head.criterion(latent, torch.tensor([-0.7, 0.8], dtype=torch.double)).backward(
            retain_graph=True
        )
        optimizer.step()
    loss = head.criterion(latent, torch.tensor([-0.7, 0.8], dtype=torch.double))
    gradient = torch.autograd.grad(loss, latent)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    grid = torch.linspace(-8, 8, 4097, dtype=torch.double)
    density = (
        head.distribution(latent.detach())
        .log_prob(grid[:, None, None].expand(-1, 2, 1))
        .exp()
    )
    torch.testing.assert_close(
        torch.trapezoid(density, grid, dim=0),
        torch.ones(2, dtype=torch.double),
        rtol=2e-5,
        atol=2e-5,
    )


def test_bound_includes_fit_tail_and_preserves_its_density_gradient():
    latent = torch.randn(2, 3)
    target = torch.tensor([7.0, 7.5])
    narrow = ConditionalFlowHead(context=3, bound=5)
    wide = ConditionalFlowHead(context=3, bound=10)
    narrow.criterion(latent, target).backward()
    wide.criterion(latent, target).backward()
    assert (
        sum(p.grad.abs().sum() for p in narrow.parameters() if p.grad is not None) == 0
    )
    assert sum(p.grad.abs().sum() for p in wide.parameters() if p.grad is not None) > 0


def test_composite_moments_match_independent_adaptive_quadrature():
    from scipy.integrate import quad

    torch.manual_seed(9)
    head = ConditionalFlowHead(context=3, bound=9).double()
    with torch.no_grad():
        for transform in head.flow.transform.transforms:
            final = [
                m
                for m in transform.hyper.modules()
                if hasattr(m, "weight") and hasattr(m, "bias")
            ][-1]
            final.bias.normal_(std=0.15)
    latent = torch.randn(1, 3, dtype=torch.double)
    with torch.no_grad():
        mean, var = head.moments(latent)
        distribution = head.distribution(latent)

        def integrand(z, power):
            y = float(
                distribution.transform.inv(
                    torch.tensor([[z]], dtype=torch.double)
                ).item()
            )
            return y**power * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)

        reference_mean = quad(integrand, -10, 10, args=(1,), epsabs=1e-9, limit=300)[0]
        reference_second = quad(integrand, -10, 10, args=(2,), epsabs=1e-9, limit=300)[
            0
        ]
    torch.testing.assert_close(
        mean, torch.tensor([reference_mean], dtype=torch.double), rtol=1e-5, atol=2e-6
    )
    torch.testing.assert_close(
        var,
        torch.tensor([reference_second - reference_mean**2], dtype=torch.double),
        rtol=1e-5,
        atol=2e-6,
    )
