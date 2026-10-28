"""XP-only forward inversion and empirical-Bayes FeH integration.

The flow describes intrinsic FeH, not a 110-dimensional deconvolved XP law.
Reported diagonal XP/FeH errors are conditional inputs, not physical truth.
"""

import math

import numpy as np
import torch
from torch import nn


def decoder(width):
    return nn.Sequential(
        nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, width)
    )


def prior_flow():
    from zuko.flows import NSF

    return NSF(features=1, transforms=3, hidden_features=(32, 32), bins=8)


def hermite(order, reference):
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return (
        reference.new_tensor(nodes),
        reference.new_tensor(np.log(weights) - 0.5 * np.log(np.pi)),
    )


def conditional_log_prob(observed, sigma, mean, intrinsic_variance):
    """[row,node] log p(XPobs | FeHtrue), summing observable dimensions."""
    variance = sigma[:, None, :].square() + intrinsic_variance
    return -0.5 * (
        (observed[:, None, :] - mean).square() / variance
        + variance.log()
        + math.log(2 * math.pi)
    ).sum(-1)


class IntegratedForward(nn.Module):
    def __init__(self, width, *, flow=False):
        super().__init__()
        self.decoder = decoder(width)
        # Extra mismatch variance is learned, never added to quoted raw errors.
        self.log_intrinsic_variance = nn.Parameter(torch.zeros(width))
        self.prior = prior_flow() if flow else None

    @property
    def intrinsic_variance(self):
        return self.log_intrinsic_variance.exp().clamp_min(1e-8)

    def objective(self, observed, sigma, labels, label_sigma, *, order=64):
        if not torch.all(torch.isfinite(label_sigma) & (label_sigma > 0)):
            raise ValueError("Quoted label errors must be finite and positive")
        nodes, log_weights = hermite(order, labels)
        # A local linear likelihood gives an importance proposal much narrower
        # than the measurement kernel at high XP S/N. It is detached: gradients
        # differentiate the integrand, never the proposal construction.
        with torch.no_grad():
            local = self.decoder(labels[:, None])
            derivative = (
                self.decoder((labels + 0.001)[:, None])
                - self.decoder((labels - 0.001)[:, None])
            ) / 0.002
            variance = sigma.square() + self.intrinsic_variance
            precision = label_sigma.square().reciprocal() + (
                derivative.square() / variance
            ).sum(-1)
            proposal_variance = precision.reciprocal()
            proposal_mean = labels + proposal_variance * (
                derivative * (observed - local) / variance
            ).sum(-1)
            proposal_sigma = proposal_variance.sqrt()
        latent = proposal_mean[:, None] + math.sqrt(2) * proposal_sigma[:, None] * nodes
        mean = self.decoder(latent[..., None])
        logs = conditional_log_prob(observed, sigma, mean, self.intrinsic_variance)
        log_measurement = (
            -0.5 * ((latent - labels[:, None]) / label_sigma[:, None]).square()
            - label_sigma[:, None].log()
            - 0.5 * math.log(2 * math.pi)
        )
        log_proposal = (
            -0.5
            * ((latent - proposal_mean[:, None]) / proposal_sigma[:, None]).square()
            - proposal_sigma[:, None].log()
            - 0.5 * math.log(2 * math.pi)
        )
        logs = logs + log_measurement - log_proposal
        if self.prior is not None:
            logs = logs + self.prior().log_prob(latent[..., None])
        # Sum XP contributions before integration. Only then normalize by width.
        return -torch.logsumexp(logs + log_weights, dim=1).mean() / observed.shape[1]


@torch.no_grad()
def grid_predictions(
    decoder_model, intrinsic_variance, observed, sigma, grid, prior=None, chunk=256
):
    """Common finite-grid MLE and prior-weighted mean; no test label arguments."""
    if not torch.all(torch.diff(grid) > 0) or len(grid) < 3:
        raise ValueError("Prediction grid must be strictly increasing")
    spacing = torch.diff(grid)
    torch.testing.assert_close(
        spacing, spacing[0].expand_as(spacing), rtol=1e-4, atol=1e-6
    )
    decoder_model.eval()
    if prior is not None:
        prior.eval()
    mean = decoder_model(grid[:, None]).double()
    prior_log = (
        torch.zeros_like(grid) if prior is None else prior().log_prob(grid[:, None])
    )
    # Trapezoidal endpoint weights; common grid normalization is explicit.
    grid = grid.double()
    prior_log = prior_log.double()
    weights = torch.zeros_like(grid)
    weights[[0, -1]] = -math.log(2)
    outputs = []
    for x, sx in zip(observed.split(chunk), sigma.split(chunk), strict=True):
        x, sx = x.double(), sx.double()
        variance = sx.square() + intrinsic_variance.double()
        inverse = variance.reciprocal()
        # Equivalent to the full Gaussian, avoiding [row,grid,XP] allocation.
        logs = -0.5 * (
            (x.square() * inverse).sum(-1)[:, None]
            - 2 * (x * inverse) @ mean.T
            + inverse @ mean.square().T
            + (variance.log() + math.log(2 * math.pi)).sum(-1)[:, None]
        )
        posterior = torch.softmax(logs + prior_log + weights, dim=-1)
        mle = grid[logs.argmax(-1)]
        post_mean = posterior @ grid
        variance_y = (posterior * (grid[None, :] - post_mean[:, None]).square()).sum(-1)
        boundary = posterior[:, :4].sum(-1) + posterior[:, -4:].sum(-1)
        outputs.append(
            torch.stack((mle, post_mean, variance_y.sqrt(), boundary), dim=-1).cpu()
        )
    return torch.cat(outputs).numpy()
