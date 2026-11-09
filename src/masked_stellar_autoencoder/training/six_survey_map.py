"""Identified own-target observation maps for sparse six-target sample scoring.

All maps act in declared normalized target coordinates. Age/distance use log
coordinates of the retained posterior teachers, not direct measurements.
"""

import torch
from torch import nn


class SixSurveyMap(nn.Module):
    """Positive affine or monotone linear-spline forward maps with explicit gauge.

    A per-target reference source fixes h_ref,j(theta)=theta_j. A value -1
    instead requires a separately observed physical anchor. A zero-sum target
    fixes the arithmetic mean map to the identity at every spline knot.
    Unsupported edges stay identity and must never receive a scored label.
    """

    def __init__(self, support, reference, *, zero_sum=(), kind="affine", bins=8):
        super().__init__()
        support = torch.as_tensor(support, dtype=torch.bool)
        reference = torch.as_tensor(reference, dtype=torch.long)
        if (
            support.ndim != 2
            or support.shape[1] != 6
            or reference.shape != (6,)
            or kind not in ["offset", "affine", "spline"]
            or bins < 2
            or bins % 2
        ):
            raise ValueError(
                "Six supported target edges, explicit references and even spline bins required"
            )
        for j, ref in enumerate(reference.tolist()):
            if ref < -1 or ref >= len(support) or (ref >= 0 and not support[ref, j]):
                raise ValueError(
                    "Reference must be a supported source-target edge or external anchor"
                )
        zero = torch.zeros(6, dtype=torch.bool)
        zero[list(zero_sum)] = True
        if (zero & (reference >= 0)).any():
            raise ValueError("Choose either a reference source or a zero-sum gauge")
        free = support.clone()
        for j, ref in enumerate(reference.tolist()):
            if ref >= 0:
                free[ref, j] = False
        self.register_buffer("support", support)
        self.register_buffer("free", free)
        self.register_buffer("zero_sum", zero)
        self.register_buffer("reference", reference)
        self.bias = nn.Parameter(torch.zeros(support.shape))
        self.raw_slope = nn.Parameter(
            torch.zeros((*support.shape, bins if kind == "spline" else 1))
        )
        self.kind = kind
        self.bins = bins if kind == "spline" else 1

    def coefficients(self):
        bias = 2 * torch.tanh(self.bias) * self.free
        slopes = torch.exp(0.5 * torch.tanh(self.raw_slope))
        slopes = torch.where(self.free[:, :, None], slopes, torch.ones_like(slopes))
        if self.kind == "offset":
            slopes = torch.ones_like(slopes)
        for j in torch.nonzero(self.zero_sum, as_tuple=False).flatten().tolist():
            active = self.support[:, j]
            bias = bias.clone()
            slopes = slopes.clone()
            bias[active, j] = bias[active, j] - bias[active, j].mean()
            slopes[active, j] = slopes[active, j] / slopes[active, j].mean(
                0, keepdim=True
            )
        return bias, slopes

    def forward(self, draws, source_index, observed=None):
        if (
            draws.ndim != 3
            or draws.shape[-1] != 6
            or source_index.shape != (draws.shape[1],)
            or source_index.dtype != torch.long
            or (source_index < 0).any()
            or (source_index >= len(self.support)).any()
        ):
            raise ValueError(
                "Require [draw,star,6] and aligned valid integer source indices"
            )
        if observed is not None and (
            observed.shape != draws.shape[1:]
            or (observed & ~self.support[source_index]).any()
        ):
            raise ValueError("Observed label on unsupported source-target edge")
        bias, slopes = self.coefficients()
        bias, slopes = bias[source_index], slopes[source_index]
        if self.kind != "spline":
            return draws * slopes[:, :, 0][None] + bias[None]
        # Exact continuous piecewise-linear map, including positive tangent tails.
        width = 8 / self.bins
        position = (draws + 4) / width
        index = position.floor().long().clamp(0, self.bins - 1)
        cumulative = torch.cat(
            [torch.zeros_like(slopes[:, :, :1]), slopes.cumsum(-1) * width], dim=-1
        )
        base = cumulative[:, :, self.bins // 2]
        expanded = slopes.unsqueeze(0).expand(draws.shape[0], -1, -1, -1)
        selected = expanded.gather(-1, index[:, :, :, None]).squeeze(-1)
        integrated = (
            cumulative.unsqueeze(0)
            .expand(draws.shape[0], -1, -1, -1)
            .gather(-1, index[:, :, :, None])
            .squeeze(-1)
        )
        return (
            integrated
            + selected * (draws - (-4 + index * width))
            - base[None]
            + bias[None]
        )

    def regularization(self):
        bias, slopes = self.coefficients()
        return (
            bias[self.support].square().mean()
            + torch.log(slopes[self.support]).square().mean()
        )
