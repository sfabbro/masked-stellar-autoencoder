"""XP order encoder, TabM ancillary trunk, quantile head, low-rank joint head."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torchregress.ensemble.layers import BatchEnsembleLinear
from torchregress.losses.nflows import HAS_ZUKO, NormalizingFlowLoss, create_flow_model

from masked_stellar_autoencoder.pipeline.registry import ColumnRegistry


class Forward(NamedTuple):
    latent: torch.Tensor
    recon_bp: torch.Tensor
    recon_rp: torch.Tensor
    quantiles: torch.Tensor
    joint_mean: torch.Tensor
    joint_factor: torch.Tensor
    joint_diag: torch.Tensor


class OrderEncoder(nn.Module):
    """Shallow 1D stack along coefficient order. Missing orders use a token."""

    def __init__(self, n_order: int, width: int) -> None:
        super().__init__()
        self.n_order = n_order
        self.mask_token = nn.Parameter(torch.zeros(width))
        self.in_proj = nn.Linear(1, width)
        self.order = nn.Embedding(n_order, width)
        self.conv = nn.Conv1d(width, width, kernel_size=3, padding=1)

    def forward(self, values: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.n_order:
            raise ValueError(f"expected {self.n_order} orders, got {values.shape[-1]}")
        hidden = self.in_proj(values.unsqueeze(-1)) + self.order.weight
        hidden = torch.where(missing.unsqueeze(-1), self.mask_token, hidden)
        hidden = torch.nn.functional.gelu(
            self.conv(hidden.transpose(1, 2)).transpose(1, 2)
        )
        weight = (~missing).to(hidden.dtype).unsqueeze(-1)
        return (hidden * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class AncillaryTabM(nn.Module):
    """TabM trunk. The missing bit is a feature, so an absent survey is not the median star."""

    def __init__(self, n_in: int, width: int, ensemble_size: int) -> None:
        super().__init__()
        if n_in < 1:
            raise ValueError("ancillary trunk needs at least one column")
        self.fc1 = BatchEnsembleLinear(n_in * 2, width, ensemble_size)
        self.fc2 = BatchEnsembleLinear(width, width, ensemble_size)

    def forward(self, values: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        filled = torch.where(missing, torch.zeros_like(values), values)
        bits = missing.to(values.dtype)
        hidden = torch.nn.functional.gelu(self.fc1(torch.cat([filled, bits], dim=-1)))
        hidden = torch.nn.functional.gelu(self.fc2(hidden))
        return hidden.mean(dim=1)


class StellarNet(nn.Module):
    def __init__(
        self,
        registry: ColumnRegistry,
        *,
        width: int = 32,
        ensemble_size: int = 4,
        rank: int = 1,
    ) -> None:
        super().__init__()
        n_bp = len(registry.names("xp_bp"))
        n_rp = len(registry.names("xp_rp"))
        n_anc = len(registry.names("photometry")) + len(registry.names("astrometry"))
        n_labels = len(registry.names("labels"))
        if min(n_bp, n_rp, n_labels) < 1:
            raise ValueError("registry needs xp_bp, xp_rp, and labels")
        self.n_labels = n_labels
        self.rank = rank
        self.bp = OrderEncoder(n_bp, width)
        self.rp = OrderEncoder(n_rp, width)
        self.ancillary = AncillaryTabM(n_anc, width, ensemble_size)
        self.fuse = nn.Linear(width * 3, width)
        self.recon_bp = nn.Linear(width, n_bp)
        self.recon_rp = nn.Linear(width, n_rp)
        self.quantile = BatchEnsembleLinear(width, 3 * n_labels, ensemble_size)
        self.joint_mean = nn.Linear(width, n_labels)
        self.joint_factor = nn.Linear(width, n_labels * rank)
        self.joint_log_diag = nn.Linear(width, n_labels)
        if HAS_ZUKO:
            flow = create_flow_model(
                n_features=n_labels,
                context_dim=width,
                n_transforms=2,
                hidden_features=[16, 16],
            )
            self.flow = NormalizingFlowLoss(flow=flow)
        else:
            self.flow = None

    def forward(
        self,
        xp_bp: torch.Tensor,
        miss_bp: torch.Tensor,
        xp_rp: torch.Tensor,
        miss_rp: torch.Tensor,
        ancillary: torch.Tensor,
        miss_anc: torch.Tensor,
    ) -> Forward:
        bp = self.bp(xp_bp, miss_bp)
        rp = self.rp(xp_rp, miss_rp)
        anc = self.ancillary(ancillary, miss_anc)
        latent = torch.nn.functional.gelu(self.fuse(torch.cat([bp, rp, anc], dim=-1)))
        quantiles = self.quantile(latent).mean(dim=1).view(-1, 3, self.n_labels)
        factor = self.joint_factor(latent).view(-1, self.n_labels, self.rank)
        diag = torch.nn.functional.softplus(self.joint_log_diag(latent)) + 1e-4
        return Forward(
            latent=latent,
            recon_bp=self.recon_bp(latent),
            recon_rp=self.recon_rp(latent),
            quantiles=quantiles,
            joint_mean=self.joint_mean(latent),
            joint_factor=factor,
            joint_diag=diag,
        )
