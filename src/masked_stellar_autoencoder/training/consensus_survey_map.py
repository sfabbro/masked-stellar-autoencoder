"""Symmetric coordinate convention for survey maps, with no fixed survey edge."""

import torch
from masked_stellar_autoencoder.training.six_survey_map import SixSurveyMap


class ConsensusSurveyMap(SixSurveyMap):
    """Overlap-weighted mean map equals identity at every spline segment.

    This fixes a coordinate gauge. It does not establish an absolute physical
    zero point, a source-noise model or identifiability outside the overlap
    domain. Weights must be supplied from an audited fixed comparison domain;
    raw differences of survey population means are not suitable weights.
    """

    def __init__(self, support, weights, *, kind="spline", bins=8):
        support = torch.as_tensor(support, dtype=torch.bool)
        w = torch.as_tensor(weights, dtype=torch.float32)
        if (
            w.shape != support.shape
            or not torch.isfinite(w).all()
            or (w < 0).any()
            or (w[~support] != 0).any()
            or (w.sum(0) <= 0).any()
        ):
            raise ValueError(
                "Positive total supported overlap weight for every target required"
            )
        super().__init__(support, [-1] * 6, kind=kind, bins=bins)
        self.register_buffer("consensus_weights", w / w.sum(0))

    def coefficients(self):
        b, a = super().coefficients()
        w = self.consensus_weights
        centered = b - (w * b).sum(0)[None]
        scaled = a / (w[:, :, None] * a).sum(0)[None]
        return torch.where(
            self.support, centered, torch.zeros_like(centered)
        ), torch.where(self.support[:, :, None], scaled, torch.ones_like(scaled))
