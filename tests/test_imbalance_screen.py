import torch
import torch.nn.functional as F

from masked_stellar_autoencoder.training.imbalance_screen import (
    imbalance_loss,
    imbalance_objective,
)


def test_bmc_matches_independent_batch_softmax_and_gradient():
    y = torch.tensor([-2.0, 0.1, 0.5, 1.0], dtype=torch.float64)
    mu = torch.tensor([-0.8, 0.0, 0.3, 0.7], dtype=torch.float64, requires_grad=True)
    objective = imbalance_objective("bmc", {"sigma_standardized": 0.5}).double()
    actual = imbalance_loss(objective, "bmc", mu[:, None, None], None, y)
    logits = -(mu[:, None] - y[None, :]).square() / 0.5
    expected = 0.5 * F.cross_entropy(logits, torch.arange(4))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, mu, retain_graph=True)[0],
        torch.autograd.grad(expected, mu)[0],
    )


def test_ranksim_reaches_encoder_and_is_added_once():
    torch.manual_seed(3)
    latent = torch.randn(256, 256, requires_grad=True)
    y = torch.linspace(-2, 1, 256)
    mu = latent[:, :1, None]
    objective = imbalance_objective("mse_ranksim", {"gamma": 0.1})
    loss = imbalance_loss(objective, "mse_ranksim", mu, latent, y)
    torch.testing.assert_close(loss, F.mse_loss(mu[:, 0, 0], y) + objective(latent, y))
    gradient = torch.autograd.grad(objective(latent, y), latent)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_huber_ranksim_uses_huber_and_representation_gradient():
    torch.manual_seed(3)
    latent = torch.randn(256, 256, requires_grad=True)
    y = torch.linspace(-2, 1, 256)
    output = latent[:, :1, None]
    objective = imbalance_objective("huber_ranksim", {"gamma": 0.1})
    actual = imbalance_loss(objective, "huber_ranksim", output, latent, y)
    expected = F.huber_loss(output[:, 0, 0], y, delta=1.0) + objective(latent, y)
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual, latent)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
