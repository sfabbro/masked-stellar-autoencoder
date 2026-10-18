"""Paper-style binary density shaping as features for a continuous FeH head.

The two densities separate a fit-only observed-label regime. Their outputs are
features, never a continuous FeH likelihood or posterior. Keep the MSA encoder
frozen while fitting these flows; freeze the flows before fitting the point head.
"""

import torch
from torch import Tensor, nn


def density_shaping_loss(
    flow: nn.Module,
    positive: Tensor,
    negative: Tensor | None,
    *,
    contrastive_c: float | None,
) -> Tensor:
    """Harness CNF objective, with its scalar batch-loss clamp at -50.

    ``contrastive_c=None`` is the matched ordinary-NLL flow control. The shaped
    objective is unbounded without the clamp and is not a normalized likelihood
    training criterion. The flow distribution itself remains normalized.
    """
    if positive.ndim != 2 or not len(positive) or not torch.isfinite(positive).all():
        raise ValueError("Positive flow batch must be a nonempty finite matrix")
    distribution = flow()
    log_positive = distribution.log_prob(positive)
    if (
        log_positive.shape != positive.shape[:1]
        or not torch.isfinite(log_positive).all()
    ):
        raise FloatingPointError("Flow must return one finite log density per star")
    loss = -log_positive.mean()
    if contrastive_c is not None:
        if contrastive_c <= 0 or not torch.isfinite(torch.tensor(contrastive_c)):
            raise ValueError("Density-shaping coefficient must be positive and finite")
        if (
            negative is None
            or negative.ndim != 2
            or not len(negative)
            or negative.shape[1] != positive.shape[1]
            or not torch.isfinite(negative).all()
        ):
            raise ValueError("Density shaping requires a finite negative batch")
        log_negative = distribution.log_prob(negative)
        if (
            log_negative.shape != negative.shape[:1]
            or not torch.isfinite(log_negative).all()
        ):
            raise FloatingPointError("Flow must return one finite negative log density")
        loss = (loss + contrastive_c * log_negative.mean()).clamp_min(-50.0)
    return loss


class CNFRegressionFeatures(nn.Module):
    """Append sigmoid(log-density) features from frozen signal/background flows.

    The signal definition is stored as a physical observed-FeH threshold. At
    inference no label enters the transform. Original latents are retained, so
    downstream regression can ignore the binary density features if unhelpful.
    ``density_features`` may be a fixed fit-only PCA projection; its transform
    and appended-Phi scaling must be preserved by the caller. Check clipping
    rates and reject constant Phi columns before fitting the point head.
    """

    def __init__(
        self, signal_flow: nn.Module, background_flow: nn.Module, *, cut_dex=-2.0
    ):
        super().__init__()
        if not torch.isfinite(torch.tensor(cut_dex)):
            raise ValueError("Regime threshold must be finite")
        self.signal_flow = signal_flow
        self.background_flow = background_flow
        self.register_buffer(
            "cut_dex", torch.tensor(float(cut_dex), dtype=torch.float64)
        )

    def partition(self, fit_labels_dex: Tensor) -> Tensor:
        """Use on strict-fit labels only; both regimes must have actual support."""
        if fit_labels_dex.ndim != 1 or not torch.isfinite(fit_labels_dex).all():
            raise ValueError("Fit labels must be a finite vector in dex")
        signal = fit_labels_dex < self.cut_dex
        if not signal.any() or signal.all():
            raise ValueError("Both fit-only density regimes require stars")
        return signal

    def freeze_flows(self):
        for flow in (self.signal_flow, self.background_flow):
            flow.eval()
            flow.requires_grad_(False)
        return self

    def forward(self, latent: Tensor, density_features: Tensor | None = None) -> Tensor:
        if latent.ndim != 2 or not torch.isfinite(latent).all():
            raise ValueError("Flow features require finite two-dimensional latents")
        if density_features is None:
            density_features = latent
        if (
            density_features.ndim != 2
            or len(density_features) != len(latent)
            or not torch.isfinite(density_features).all()
        ):
            raise ValueError("Density features must be finite with the same star rows")
        features = [latent]
        with torch.no_grad():
            for flow in (self.signal_flow, self.background_flow):
                log_density = flow().log_prob(density_features)
                if (
                    log_density.shape != latent.shape[:1]
                    or not torch.isfinite(log_density).all()
                ):
                    raise FloatingPointError(
                        "Flow returned nonfinite or malformed log density"
                    )
                # Equal to p/(1+p), evaluated stably with the harness's declared
                # [-30,30] log-density clipping. Not a class posterior.
                features.append(torch.sigmoid(log_density.clamp(-30, 30))[:, None])
        return torch.cat(features, dim=1)
