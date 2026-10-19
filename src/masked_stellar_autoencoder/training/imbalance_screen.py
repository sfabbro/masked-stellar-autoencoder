"""Declared, point-only BMC and representation RankSim comparison objectives."""

import torch
from torchregress.losses import BalancedMSEBMC, RankSimLoss


def imbalance_objective(kind, policy):
    if kind == "bmc":
        return BalancedMSEBMC(sigma=policy["sigma_standardized"], prior="uniform")
    if kind == "mse_ranksim":
        return RankSimLoss(
            gamma=policy["gamma"],
            interpolation_strength=2.0,
            similarity="cosine",
            duplicate_policy="mean",
        )
    return None


def imbalance_loss(objective, kind, output, latent, target):
    prediction = output[:, 0, 0]
    if kind == "bmc":
        return 2 * objective.sigma.square() * objective(prediction, target)
    return torch.nn.functional.mse_loss(prediction, target) + objective(latent, target)
