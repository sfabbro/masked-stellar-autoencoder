"""Declared, point-only BMC and representation RankSim comparison objectives."""

import torch
from torchregress.losses import BalancedMSEBMC, RankSimLoss


def imbalance_objective(kind, policy):
    if kind == "bmc":
        return BalancedMSEBMC(sigma=policy["sigma_standardized"], prior="uniform")
    if kind in {"mse_ranksim", "huber_ranksim"}:
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
    task_loss = (
        torch.nn.functional.huber_loss(prediction, target, delta=1.0)
        if kind == "huber_ranksim"
        else torch.nn.functional.mse_loss(prediction, target)
    )
    return task_loss + objective(latent, target)
