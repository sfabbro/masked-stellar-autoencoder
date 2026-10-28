"""Bounded comparisons of dilution, ratio estimation and target-regime risk."""

import torch
from torch import nn

from .prediction_experiment_heads import ContrastiveFlowEstimator


class NeuralRatioEstimator(ContrastiveFlowEstimator):
    """NRE-B on all256 encoder channels; no learned density or 8D bottleneck."""

    def __init__(self):
        nn.Module.__init__(self)
        from torchregress.losses import NeuralRatioLoss, build_ratio_classifier

        self.summary = nn.Identity()
        self.flow_loss = NeuralRatioLoss(
            build_ratio_classifier(1, 256, hidden_dims=(128, 64)),
            criterion="multiclass",
        )
        self.shifts = (1, 2, 4, 8, 16, 32, 64, 128)

    def standardized_mle(self, latent, grid):
        if grid.ndim != 1 or len(grid) != 201 or not torch.isfinite(grid).all():
            raise ValueError("NRE requires the declared201-point finite grid")
        from torchregress.losses import neural_ratio_logits

        chosen = []
        with torch.no_grad():
            for block in latent.split(128):
                context = grid[None, :, None].expand(len(block), -1, 1).reshape(-1, 1)
                observable = (
                    block[:, None, :].expand(-1, len(grid), -1).reshape(-1, 256)
                )
                logits = neural_ratio_logits(
                    self.flow_loss.classifier, context, observable
                ).reshape(len(block), len(grid))
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite ratio grid scores")
                chosen.append(grid[logits.argmax(1)])
        return torch.cat(chosen) if chosen else grid.new_empty(0)


def ratio_grid(labels):
    return torch.linspace(
        float(labels.min()) - 0.5,
        float(labels.max()) + 0.5,
        201,
        device=labels.device,
        dtype=labels.dtype,
    )


def corrected_score_objective(model, head):
    from torchregress.losses import CorrectedScoreEIVLoss

    # Numeric eps=0: exact declared diagonal sigmas; no invented physical floor.
    return CorrectedScoreEIVLoss(
        lambda x: head(model.encoder(x))[:, 0, 0],
        sigma_x=0.0,
        eps=0.0,
        stabilize="none",
        jacobian_chunk_size=32,
    )


def group_dro_objective(labels, center, scale):
    from torchregress.algorithms import ContinuousGroupDRO

    cuts = labels.new_tensor(
        [(-2 - center) / scale, (-1 - center) / scale, (0 - center) / scale]
    )
    edges = torch.cat(
        [
            torch.minimum(labels.min(), cuts.min()).reshape(1) - 1,
            cuts,
            torch.maximum(labels.max(), cuts.max()).reshape(1) + 1,
        ]
    )
    if not torch.all(edges[1:] > edges[:-1]):
        raise ValueError("Physical target regimes must lie inside fit-only support")
    return ContinuousGroupDRO(
        edges,
        eta=0.01,
        base_loss=nn.HuberLoss(delta=1.0, reduction="none"),
        min_group_weight=0.01,
    ).to(labels.device)
