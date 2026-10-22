"""Fit-only context conditioner for the distance and age heads.

Inputs are the Tier-1 context vector of the plan: **14 continuous features +
8 mask/quality bits = 22**, mapped to 128 and concatenated *after* the 256-d
latent, so the parent86 checkpoint loads unmodified ("context injection point",
``07-inputs-architecture.md`` Sec. 3).

Two hard rules are enforced here:

1. **Fit-only normalization.** Medians and IQR scales are computed from rows
   explicitly marked as the fit role and from nothing else. Passing tune/cal/
   test rows into :meth:`ContextNormalizer.fit` is a caller error that cannot be
   detected from the numbers, so the role helper
   :func:`fit_mask_from_roles` exists and the docstrings say it out loud.
2. **No silent median fill.** A missing feature is normalized to ``0.0`` -- the
   fit median -- *only* together with its mask bit set to ``0``. The bit is
   mandatory and always concatenated, so the head can tell "at the median" from
   "not measured". Unobserved entries may be non-finite; observed entries must
   be finite or the transform raises.

The SFD ``E(B-V)`` slot is context only. It is a line-of-sight total, not
``A_V(d)``, and it is not admissible as a physical extinction until the
provenance gate passes (TRAINING_PLAN Sec. 3.1); ``flag_ebv_unphysical`` marks
the ``> 2`` rows observed in the audit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

#: Continuous context slots, in this frozen order.
CONTINUOUS_CONTEXT: tuple[str, ...] = (
    "parallax_mas",
    "e_parallax_mas",
    "zero_point_mas",
    "pmra_masyr",
    "pmdec_masyr",
    "e_pmra_masyr",
    "e_pmdec_masyr",
    "l_deg",
    "b_deg",
    "healpix_nside64",
    "ebv_sfd",
    "ruwe",
    "gaia_neighbor_count",
    "astrometric_params_solved",
)

#: Mask/quality bits, in this frozen order. Bits 0-4 are measurement-observed
#: flags, bits 5-7 are quality flags derived from the values.
MASK_BITS: tuple[str, ...] = (
    "mask_parallax",
    "mask_e_parallax",
    "mask_pm",
    "mask_ebv",
    "mask_ruwe",
    "flag_ebv_unphysical",
    "flag_parallax_negative",
    "flag_zero_point_applied",
)

#: Full context width (14 + 8).
CONTEXT_DIMENSIONS = len(CONTINUOUS_CONTEXT) + len(MASK_BITS)

#: E(B-V) above which the audited table contains non-physical values.
EBV_UNPHYSICAL_THRESHOLD = 2.0

#: Roles understood by :func:`fit_mask_from_roles`.
ROLES: tuple[str, ...] = ("fit", "tune", "calibration", "test")


def fit_mask_from_roles(roles, role: str = "fit") -> np.ndarray:
    """Return the boolean mask selecting exactly one role.

    ``roles`` is a sequence of role labels with no implicit default: an unknown
    label raises instead of being treated as "not fit", because a typo would
    otherwise silently move rows out of the fit set used for normalization.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    array = np.asarray(roles)
    if array.ndim != 1:
        raise ValueError("roles must be a 1-D sequence")
    unknown = set(np.unique(array).tolist()) - set(ROLES)
    if unknown:
        raise ValueError(f"unknown role labels: {sorted(unknown)}")
    mask = array == role
    if not mask.any():
        raise ValueError(f"no rows carry role {role!r}")
    return mask


def _as_float_array(values) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("context values must be a [row, feature] array")
    if array.shape[1] != len(CONTINUOUS_CONTEXT):
        raise ValueError(
            f"context values must have {len(CONTINUOUS_CONTEXT)} continuous columns"
        )
    if np.iscomplexobj(array):
        raise TypeError("complex context values are not supported")
    return array.astype(np.float64, copy=False)


@dataclass(frozen=True)
class ContextNormalizer:
    """Fit-only median/IQR standardizer for the 14 continuous context slots.

    ``transform`` maps the fit median to ``0`` and the fit interquartile range
    (``q75 - q25``) to a span of exactly ``1`` per feature. The quartiles
    themselves land wherever the sample's asymmetry puts them (``-0.5``/``+0.5``
    only for a symmetric sample; ``IQR = 1.349 sigma`` for a Gaussian, which is
    why a robust sigma is ``IQR/1.349`` and not ``IQR``). No clipping is applied
    and no variance floor is substituted (a degenerate fit IQR raises).
    """

    center: np.ndarray
    scale: np.ndarray
    n_fit: tuple[int, ...]

    @classmethod
    def fit(cls, values, fit_mask) -> ContextNormalizer:
        """Fit centers/scales on ``fit_mask`` rows only.

        Every feature needs at least two finite observed fit rows and a strictly
        positive IQR. Rows where ``fit_mask`` is False never contribute, even if
        they are finite.
        """
        array = _as_float_array(values)
        mask = np.asarray(fit_mask)
        if mask.dtype != np.bool_ or mask.shape != (array.shape[0],):
            raise ValueError("fit_mask must be a boolean [row] mask")
        selected = array[mask]
        if selected.shape[0] < 2:
            raise ValueError("at least two fit rows are required")
        finite = np.isfinite(selected)
        counts = finite.sum(0)
        if (counts < 2).any():
            bad = [CONTINUOUS_CONTEXT[i] for i in np.flatnonzero(counts < 2)]
            raise ValueError(f"fewer than two finite fit values for {bad}")
        center = np.zeros(len(CONTINUOUS_CONTEXT), dtype=np.float64)
        scale = np.zeros(len(CONTINUOUS_CONTEXT), dtype=np.float64)
        for index in range(len(CONTINUOUS_CONTEXT)):
            column = selected[finite[:, index], index]
            q25, median, q75 = np.percentile(column, [25.0, 50.0, 75.0])
            if not (
                math.isfinite(median) and math.isfinite(q25) and math.isfinite(q75)
            ):
                raise ValueError("non-finite fit percentiles")
            if not q75 - q25 > 0:
                raise ValueError(
                    f"degenerate fit IQR for {CONTINUOUS_CONTEXT[index]}; "
                    "no variance floor is substituted"
                )
            center[index] = median
            scale[index] = q75 - q25
        return cls(center=center, scale=scale, n_fit=tuple(int(c) for c in counts))

    def standardize(self, values, observed) -> np.ndarray:
        """Return the 14 standardized continuous columns (missing -> ``0.0``)."""
        array = _as_float_array(values)
        seen = np.asarray(observed)
        if seen.dtype != np.bool_ or seen.shape != array.shape:
            raise ValueError("observed must be a boolean mask of the values shape")
        if not np.isfinite(array[seen]).all():
            raise ValueError(
                "observed context values must be finite; mask them instead of "
                "relying on a silent fill"
            )
        standard = np.zeros(array.shape, dtype=np.float64)
        rows = np.arange(array.shape[0])
        for index in range(array.shape[1]):
            keep = seen[:, index]
            if keep.any():
                standard[rows[keep], index] = (
                    array[rows[keep], index] - self.center[index]
                ) / self.scale[index]
        return standard

    def bits(self, values, observed, *, zero_point_applied) -> np.ndarray:
        """Return the 8 mask/quality bits as ``float64`` ``0/1`` columns."""
        array = _as_float_array(values)
        seen = np.asarray(observed)
        if seen.dtype != np.bool_ or seen.shape != array.shape:
            raise ValueError("observed must be a boolean mask of the values shape")
        if not np.isfinite(array[seen]).all():
            raise ValueError("observed context values must be finite")
        n = array.shape[0]
        applied = np.asarray(zero_point_applied)
        if applied.ndim == 0:
            applied = np.full(n, bool(applied), dtype=bool)
        if applied.dtype != np.bool_ or applied.shape != (n,):
            raise ValueError(
                "zero_point_applied must be a boolean scalar or [row] mask"
            )
        plx_index = CONTINUOUS_CONTEXT.index("parallax_mas")
        ebv_index = CONTINUOUS_CONTEXT.index("ebv_sfd")
        ruwe_index = CONTINUOUS_CONTEXT.index("ruwe")
        pmra_index = CONTINUOUS_CONTEXT.index("pmra_masyr")
        pmdec_index = CONTINUOUS_CONTEXT.index("pmdec_masyr")
        bits = np.zeros((n, len(MASK_BITS)), dtype=np.float64)
        bits[:, 0] = seen[:, plx_index]
        bits[:, 1] = seen[:, CONTINUOUS_CONTEXT.index("e_parallax_mas")]
        bits[:, 2] = seen[:, pmra_index] & seen[:, pmdec_index]
        bits[:, 3] = seen[:, ebv_index]
        bits[:, 4] = seen[:, ruwe_index]
        bits[:, 5] = seen[:, ebv_index] & (
            array[:, ebv_index] > EBV_UNPHYSICAL_THRESHOLD
        )
        bits[:, 6] = seen[:, plx_index] & (array[:, plx_index] < 0)
        bits[:, 7] = applied
        return bits

    def transform(self, values, observed, *, zero_point_applied) -> np.ndarray:
        """Return the full ``[row, 22]`` context matrix (continuous + bits)."""
        continuous = self.standardize(values, observed)
        bits = self.bits(values, observed, zero_point_applied=zero_point_applied)
        return np.concatenate([continuous, bits], axis=1)


class ContextConditioner(nn.Module):
    """Map the 22-d context to 128 and fuse it with the 256-d latent.

    22 -> 64 -> 64 -> 128 with GELU and biases (13,952 parameters). Trained only
    through the heads; it has no loss of its own. The fused width is
    ``latent_dim + 128 = 384``, which is exactly the input width the distance
    and age heads expect.
    """

    def __init__(
        self,
        context_dim: int = CONTEXT_DIMENSIONS,
        width: int = 64,
        output_dim: int = 128,
        latent_dim: int = 256,
    ):
        super().__init__()
        for name, value in (
            ("context_dim", context_dim),
            ("width", width),
            ("output_dim", output_dim),
            ("latent_dim", latent_dim),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.context_dim = context_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(context_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, output_dim),
        )

    def forward(self, context: Tensor) -> Tensor:
        """Return the 128-d context summary for a ``[row, 22]`` tensor."""
        if context.ndim != 2 or context.shape[1] != self.context_dim:
            raise ValueError(f"context must be [row, {self.context_dim}]")
        if not context.is_floating_point():
            raise TypeError("context must be floating point")
        if not torch.isfinite(context).all():
            raise ValueError("context must be finite; mask missing values instead")
        summary = self.net(context)
        if not torch.isfinite(summary).all():
            raise ValueError("context conditioner produced nonfinite outputs")
        return summary

    def fuse(self, latent: Tensor, context: Tensor) -> Tensor:
        """Concatenate ``[latent (256) || context summary (128)]`` -> ``[row, 384]``."""
        if latent.ndim != 2 or latent.shape[1] != self.latent_dim:
            raise ValueError(f"latent must be [row, {self.latent_dim}]")
        if not latent.is_floating_point():
            raise TypeError("latent must be floating point")
        if not torch.isfinite(latent).all():
            raise ValueError("latent must be finite")
        if latent.device != context.device:
            raise ValueError("latent and context must share a device")
        summary = self(context)
        if summary.shape[0] != latent.shape[0]:
            raise ValueError("latent and context must share the row dimension")
        return torch.cat([latent, summary], dim=1)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
