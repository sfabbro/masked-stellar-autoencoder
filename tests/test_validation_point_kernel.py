import importlib.util
from pathlib import Path

import torch

s = importlib.util.spec_from_file_location(
    "validation_points", Path("experiments/xp/monitor/verify_validation_points.py")
)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def test_manual_mixture_matches_torch_distribution_and_convolution():
    torch.manual_seed(11)
    raw = torch.randn(7, 28, dtype=torch.float64, requires_grad=True)
    y = torch.randn(7, 3, dtype=torch.float64)
    sigma = torch.full_like(y, 0.2)
    mean, var, lp, obs = m.analytic(raw, y, sigma, "mdn", 4)
    w = raw[:, :4].softmax(-1)
    mu = raw[:, 4:16].reshape(7, 4, 3)
    scale = raw[:, 16:].reshape(7, 4, 3).exp()
    dist = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(probs=w),
        torch.distributions.Independent(torch.distributions.Normal(mu, scale), 1),
    )
    torch.testing.assert_close(lp, dist.log_prob(y))
    torch.testing.assert_close(mean, dist.mean)
    torch.testing.assert_close(var, dist.variance)
    noisy = torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(probs=w),
        torch.distributions.Independent(
            torch.distributions.Normal(
                mu, (scale.square() + sigma[:, None].square()).sqrt()
            ),
            1,
        ),
    )
    torch.testing.assert_close(obs, noisy.log_prob(y))
    obs.sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_manual_gaussian_density_and_noise():
    raw = torch.zeros(2, 6, dtype=torch.float64)
    y = torch.ones(2, 3, dtype=torch.float64)
    sigma = torch.ones_like(y)
    mean, var, lp, obs = m.analytic(raw, y, sigma, "gaussian")
    expected = torch.distributions.Independent(
        torch.distributions.Normal(mean, (var + 1).sqrt()), 1
    ).log_prob(y)
    torch.testing.assert_close(obs, expected)
    assert torch.all(var == 1)
