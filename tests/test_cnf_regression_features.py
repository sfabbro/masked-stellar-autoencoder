"""Independent identities for the CNF feature augmentation adaptation."""

import pytest
import torch
from torch import nn

from masked_stellar_autoencoder.training.cnf_regression_features import (
    CNFRegressionFeatures,
    density_shaping_loss,
)


class GaussianFlow(nn.Module):
    def __init__(self, mean):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor(mean, dtype=torch.float32))

    def forward(self):
        return torch.distributions.Independent(
            torch.distributions.Normal(self.mean, torch.ones_like(self.mean)), 1
        )


def test_density_shaping_matches_analytic_gaussian_and_nll_control():
    flow = GaussianFlow([0.0, 0.0])
    pos = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    neg = torch.tensor([[2.0, 1.0]])
    log_constant = torch.log(torch.tensor(2 * torch.pi))
    nll = log_constant + 0.25
    expected = (nll - 2 * (log_constant + 2.5)).clamp_min(-50)
    torch.testing.assert_close(
        density_shaping_loss(flow, pos, neg, contrastive_c=2), expected
    )
    torch.testing.assert_close(
        density_shaping_loss(flow, pos, None, contrastive_c=None), nll
    )
    density_shaping_loss(flow, pos, neg, contrastive_c=0.5).backward()
    assert torch.isfinite(flow.mean.grad).all()


def test_clamp_is_scalar_and_has_zero_gradient_below_guard():
    flow = GaussianFlow([0.0])
    loss = density_shaping_loss(
        flow, torch.zeros(2, 1), torch.full((2, 1), 100.0), contrastive_c=2
    )
    torch.testing.assert_close(loss, torch.tensor(-50.0))
    loss.backward()
    torch.testing.assert_close(flow.mean.grad, torch.zeros(1))


def test_features_preserve_latent_and_use_no_inference_labels_and_reload():
    first = CNFRegressionFeatures(GaussianFlow([0.0, 0.0]), GaussianFlow([1.0, 1.0]))
    first.freeze_flows()
    z = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    out = first(z)
    assert out.shape == (2, 4)
    torch.testing.assert_close(out[:, :2], z, rtol=0, atol=0)
    p = torch.exp(first.signal_flow().log_prob(z))
    torch.testing.assert_close(out[:, 2], p / (1 + p))
    other = CNFRegressionFeatures(GaussianFlow([3.0, 4.0]), GaussianFlow([5.0, 6.0]))
    other.load_state_dict(first.state_dict(), strict=True)
    torch.testing.assert_close(out, other(z), rtol=0, atol=0)
    assert not any(p.requires_grad for p in first.parameters())
    torch.testing.assert_close(
        first.partition(torch.tensor([-3.0, -2.0, 0.0])),
        torch.tensor([True, False, False]),
    )


def test_missing_regime_and_nonfinite_input_rejected():
    features = CNFRegressionFeatures(GaussianFlow([0.0]), GaussianFlow([1.0]))
    with pytest.raises(ValueError, match="Both"):
        features.partition(torch.tensor([0.0, 1.0]))
    with pytest.raises(ValueError, match="finite"):
        features(torch.tensor([[float("nan")]]))


def test_density_projection_keeps_full_regression_features():
    features = CNFRegressionFeatures(GaussianFlow([0.0]), GaussianFlow([1.0]))
    latent = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    density = torch.tensor([[0.0], [1.0]])
    out = features(latent, density)
    assert out.shape == (2, 5)
    torch.testing.assert_close(out[:, :3], latent, rtol=0, atol=0)
    torch.testing.assert_close(out[:, 3:], features(density)[:, 1:])
    with pytest.raises(ValueError, match="same star rows"):
        features(latent, density[:1])
