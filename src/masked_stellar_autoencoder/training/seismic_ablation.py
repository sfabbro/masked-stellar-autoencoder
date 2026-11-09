"""Matched seismic observation-channel controls, with an explicit common gauge."""

import numpy as np
import torch
from torch import nn


class SeismicChannel(nn.Module):
    """Identity or constrained affine gravity calibration, optionally contextual.

    The offset is shared across stars; it is not an independent noise floor.
    The positive gravity derivative makes the scalar map invertible conditional
    on temperature and iron. Parameters are MAP estimates, not posterior draws
    of the common-mode systematic uncertainty.
    """

    def __init__(self, center, scale, kind="identity"):
        super().__init__()
        if kind not in ["identity", "affine", "context"]:
            raise ValueError("Undeclared seismic map")
        self.kind = kind
        self.register_buffer("center", torch.as_tensor(center, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.raw = nn.Parameter(torch.zeros(4), requires_grad=kind != "identity")

    def coefficients(self):
        if self.kind == "identity":
            return self.raw * 0
        values = torch.tanh(self.raw) * self.raw.new_tensor([0.15, 0.15, 0.10, 0.10])
        if self.kind == "affine":
            values = values * values.new_tensor([1, 1, 0, 0])
        return values

    def forward(self, samples):
        b, log_slope, t, f = self.coefficients().unbind()
        out = samples.clone()
        out[..., 1] = (
            torch.exp(log_slope) * samples[..., 1]
            + (b + t * samples[..., 0] + f * samples[..., 2]) / self.scale[1]
        )
        return out

    def inverse_gravity(self, seismic_g, standardized_teff, standardized_feh):
        b, log_slope, t, f = self.coefficients().unbind()
        return (
            seismic_g
            - self.center[1]
            - b
            - t * standardized_teff
            - f * standardized_feh
        ) / torch.exp(log_slope) + self.center[1]

    def regularization(self):
        values = self.coefficients()
        return ((values / values.new_tensor([0.03, 0.05, 0.03, 0.03])) ** 2).sum()


def matched_schedule(lengths, steps, seed, batch=256):
    """Uniform native-record sampling with replacement; independent of treatment.

    Fixed-size 32-seismic + 16/8/8 teacher auxiliary streams are appended by the
    caller using the same RNG and then verified by exact source/star hashes.
    No seismic loss changes sample counts or the native-loss denominator.
    """
    sizes = np.asarray(lengths, dtype="i8")
    if sizes.ndim != 1 or (sizes < 0).any() or sizes.sum() <= 0:
        raise ValueError("Valid nonempty native source lengths required")
    rng = np.random.default_rng(seed)
    bounds = sizes.cumsum()
    offsets = np.r_[0, bounds[:-1]]
    for _ in range(steps):
        index = rng.integers(int(bounds[-1]), size=batch)
        source = np.searchsorted(bounds, index, side="right")
        yield source, index - offsets[source], rng
