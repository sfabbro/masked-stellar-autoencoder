"""Additive metadata conditioning of the encoder latent for prediction heads.

``z' = z + g([values * visible, visible])`` with the last layer of ``g``
zero-initialised. A warm-started head therefore sees exactly the same latent
at step 0 and its saved weights still load strictly; the head learns to use
EBV, astrometry and sky position as training proceeds. Missing metadata is
encoded by its mask, never as a known zero.
"""

import torch
from torch import nn


class ContextLatent(nn.Module):
    def __init__(self, latent_dim: int, context_dim: int, hidden: int = 64):
        super().__init__()
        self.context_dim = context_dim
        self.net = nn.Sequential(
            nn.Linear(2 * context_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, values, visible):
        if values.shape != visible.shape or values.shape[-1] != self.context_dim:
            raise ValueError("Context values/visible must be [row, context_dim]")
        visible = visible.to(z.dtype)
        values = torch.nan_to_num(values.to(z.dtype)) * visible
        return z + self.net(torch.cat([values, visible], dim=-1))
