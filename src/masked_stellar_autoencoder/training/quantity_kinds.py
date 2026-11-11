"""Declared measurement quantities for the metallicity and alpha target slots.

The common stellar coordinates are [Fe/H] (slot 2) and [alpha/Fe] (slot 3).
Surveys measure different quantities in those slots: [M/H] instead of
[Fe/H], [alpha/M] or [Mg/Fe] instead of [alpha/Fe]. Each source declares its
quantity per slot, and the common draws are converted to that quantity in dex
before the source's own monotone survey map, using the Salaris et al. (1993)
relation shared by all sources:

    f([alpha/Fe]) = log10(0.638 * 10**[alpha/Fe] + 0.362)
    [M/H]       = [Fe/H] + f([alpha/Fe])
    [alpha/M]   = [alpha/Fe] - f([alpha/Fe])
    [Mg/Fe]     : no conversion (its own map relates it to [alpha/Fe])

Coordinates are the trainer's normalized target coordinates; ``center`` and
``scale`` convert to dex and back. Slot quantity codes:
metallicity 0 = fe_h, 1 = m_h; alpha 0 = alpha_fe, 1 = alpha_m, 2 = mg_fe.
"""

import torch
from torch import nn

METAL_KINDS = {"fe_h": 0, "m_h": 1}
ALPHA_KINDS = {"alpha_fe": 0, "alpha_m": 1, "mg_fe": 2}
METAL_SLOT, ALPHA_SLOT = 2, 3


def salaris(alpha_fe):
    return torch.log10(0.638 * torch.pow(10.0, alpha_fe) + 0.362)


class QuantityKinds(nn.Module):
    """Convert common draws to each source's declared metallicity/alpha quantity."""

    def __init__(self, metal_kinds, alpha_kinds, center, scale):
        super().__init__()
        if len(metal_kinds) != len(alpha_kinds):
            raise ValueError("One metallicity and one alpha kind per source")
        try:
            metal = [METAL_KINDS[k] for k in metal_kinds]
            alpha = [ALPHA_KINDS[k] for k in alpha_kinds]
        except KeyError as error:
            raise ValueError(f"Unknown quantity kind {error}") from None
        self.register_buffer(
            "metal", torch.tensor(metal, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "alpha", torch.tensor(alpha, dtype=torch.long), persistent=False
        )
        c = torch.as_tensor(center, dtype=torch.float32)
        s = torch.as_tensor(scale, dtype=torch.float32)
        if c.shape != (6,) or s.shape != (6,) or not (s > 0).all():
            raise ValueError("Six target centers and positive scales required")
        self.register_buffer("center", c, persistent=False)
        self.register_buffer("scale", s, persistent=False)
        self.active = bool((self.metal != 0).any() or (self.alpha == 1).any())

    def forward(self, draws, source_index):
        """draws [draw, star, 6] in normalized coordinates; per-star source index."""
        if not self.active:
            return draws
        metal = self.metal[source_index]
        alpha_kind = self.alpha[source_index]
        c, s = self.center, self.scale
        fe_h = draws[..., METAL_SLOT] * s[METAL_SLOT] + c[METAL_SLOT]
        alpha_fe = draws[..., ALPHA_SLOT] * s[ALPHA_SLOT] + c[ALPHA_SLOT]
        f = salaris(alpha_fe)
        m_h = (fe_h + f - c[METAL_SLOT]) / s[METAL_SLOT]
        alpha_m = (alpha_fe - f - c[ALPHA_SLOT]) / s[ALPHA_SLOT]
        out = draws.clone()
        out[..., METAL_SLOT] = torch.where(
            (metal == 1)[None], m_h, draws[..., METAL_SLOT]
        )
        out[..., ALPHA_SLOT] = torch.where(
            (alpha_kind == 1)[None], alpha_m, draws[..., ALPHA_SLOT]
        )
        return out
