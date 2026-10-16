"""Proposed joint-head input policy; not wired into existing training jobs.

Allow named photometry, XP coefficients and optionally the existing EBV input.
All other columns, including any target/astrometry aliases, remain withheld.
Input values must already use the parent checkpoint scaler; genuine observed
masks come from the original values before standardization or sentinel filling.
This policy does not establish independence from the parent's training data or
from model-dependent photometric/reddening processing.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

_PHOTOMETRY = {
    "w1",
    "w2",
    "g",
    "bp",
    "rp",
    "j",
    "h",
    "ks",
    *[f"{band}_smss" for band in ("u", "v", "g", "r", "i", "z")],
    *[f"{band}_sdss" for band in ("u", "g", "r", "i", "z")],
    *[f"{band}_ps1" for band in ("g", "r", "i", "z", "y")],
}
_XP = {f"{arm}_{index}" for arm in ("bp", "rp") for index in range(1, 56)}


def spectrophotometric_inputs(
    scaled_values: Tensor,
    observed: Tensor,
    feature_names: Sequence[str],
    *,
    include_reddening: bool = True,
) -> tuple[Tensor, Tensor]:
    """Return sanitized parent-width values and the actual visible mask.

    No random masks or dropout are applied here. Missing/nonfinite/withheld
    entries use the parent's -9999 sentinel. Unknown fields are excluded by
    default so adding a new distance/label alias cannot silently expose it.
    The returned tensors do not modify the original reconstruction stream.
    """
    if scaled_values.ndim != 2 or len(feature_names) != scaled_values.shape[1]:
        raise ValueError("require a matrix and one feature name per column")
    if not scaled_values.is_floating_point():
        raise TypeError("scaled inputs must be floating point")
    if observed.shape != scaled_values.shape or observed.dtype != torch.bool:
        raise ValueError("observed must be a boolean mask with the input shape")
    if observed.device != scaled_values.device:
        raise ValueError("values and observed mask must use the same device")
    names = [str(name).casefold() for name in feature_names]
    if len(set(names)) != len(names):
        raise ValueError("ambiguous duplicate feature names")
    allowed_names = _PHOTOMETRY | _XP | ({"ebv"} if include_reddening else set())
    allowed = torch.tensor(
        [name in allowed_names for name in names],
        dtype=torch.bool,
        device=scaled_values.device,
    )
    visible = (
        observed
        & allowed.unsqueeze(0)
        & torch.isfinite(scaled_values)
        & (scaled_values != -9999)
    )
    return torch.where(visible, scaled_values, -9999.0), visible
