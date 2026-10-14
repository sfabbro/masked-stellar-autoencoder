"""Linear XP photometry and training-calibrated, dimensionless flux objectives.

XP coefficients are G-normalised, whereas crossmatched magnitudes are apparent.
No standardisation offsets are applied: raw GaiaXPy/torchgaiaxp matrices are
linear, and survey differences are described by training-only bias and scatter.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class PhotometryProjection:
    matrix: np.ndarray  # [110, bands], BP then RP
    zero_points: np.ndarray
    columns: tuple[str, ...]
    bands: tuple[str, ...]

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            matrix=self.matrix,
            zero_points=self.zero_points,
            columns=np.asarray(self.columns),
            bands=np.asarray(self.bands),
        )

    @classmethod
    def load(cls, path: str | Path) -> PhotometryProjection:
        with np.load(path, allow_pickle=False) as arrays:
            obj = cls(
                arrays["matrix"],
                arrays["zero_points"],
                tuple(arrays["columns"].tolist()),
                tuple(arrays["bands"].tolist()),
            )
        if (
            obj.matrix.shape != (110, len(obj.columns))
            or not np.isfinite(obj.matrix).all()
        ):
            raise ValueError("invalid photometry projection matrix")
        if (
            obj.zero_points.shape != (len(obj.columns),)
            or not np.isfinite(obj.zero_points).all()
        ):
            raise ValueError("invalid photometry zero points")
        return obj


def build_projection() -> PhotometryProjection:
    """Build fixed raw g/r/i matrices; torchgaiaxp is needed only here."""
    from torchgaiaxp import _assets
    from torchgaiaxp.design import XPDesign

    selection = (
        ("SDSS", ("g", "r", "i"), ("G_SDSS", "R_SDSS", "I_SDSS")),
        ("PanSTARRS1", ("gp", "rp", "ip"), ("G_PS1", "R_PS1", "I_PS1")),
        ("Sky_Mapper", ("g", "r", "i"), ("G_SMSS", "R_SMSS", "I_SMSS")),
    )
    matrices, points, columns, bands = [], [], [], []
    for system, selected, catalog in selection:
        design = XPDesign.photometric(system, dtype=torch.float64)
        combined = (
            (design.design * design.merge[..., None]).permute(0, 2, 1).reshape(110, -1)
        )
        indices = [design.bands.index(band) for band in selected]
        matrices.append(combined[:, indices].numpy())
        points.extend(
            _assets.tensor("systems", f"{system}/zero_points")[indices].tolist()
        )
        columns.extend(catalog)
        bands.extend(f"{system}/{band}" for band in selected)
    return PhotometryProjection(
        np.concatenate(matrices, axis=1),
        np.asarray(points),
        tuple(columns),
        tuple(bands),
    )


def normalized_observed_flux(magnitude, g_magnitude, zero_points):
    """F_band / 10**((8.5-G)/2.5), computed as a single stable exponent."""
    return 10.0 ** (
        -0.4
        * (
            np.asarray(magnitude)
            - np.asarray(zero_points)
            + 8.5
            - np.asarray(g_magnitude)[:, None]
        )
    )


@dataclass(frozen=True)
class PhotometryCalibration:
    reference_flux: np.ndarray
    bias: np.ndarray  # median (synthetic-observed)/reference
    floor: np.ndarray  # robust residual scale, in reference units
    count: np.ndarray

    def as_dict(self):
        return {
            key: getattr(self, key).tolist()
            for key in ("reference_flux", "bias", "floor", "count")
        }


def fit_calibration(
    coefficients,
    magnitudes,
    g_magnitudes,
    projection,
    *,
    min_count=100,
    minimum_floor=0.02,
) -> PhotometryCalibration:
    """Fit only on designated training rows, never validation or predictions.

    1.4826*MAD describes irreducible cross-survey mismatch (including noise).
    The floor lower bound is relative to a per-band median flux, not an absolute
    physical-flux epsilon. Bands with insufficient coverage fail explicitly.
    """
    if minimum_floor <= 0 or min_count < 1:
        raise ValueError("positive calibration floor/count required")
    coefficients = np.asarray(coefficients)
    observed = normalized_observed_flux(
        magnitudes, g_magnitudes, projection.zero_points
    )
    synthetic = coefficients @ projection.matrix
    valid = (
        np.isfinite(coefficients).all(axis=1)[:, None]
        & np.isfinite(observed)
        & (observed > 0)
        & np.isfinite(synthetic)
        & (synthetic > 0)
    )
    reference, bias, floor, count = [], [], [], []
    for band in range(observed.shape[1]):
        rows = valid[:, band]
        n = int(rows.sum())
        if n < min_count:
            raise ValueError(
                f"{projection.columns[band]}: only {n} calibration matches"
            )
        ref = np.median(observed[rows, band])
        residual = (synthetic[rows, band] - observed[rows, band]) / ref
        median = np.median(residual)
        scatter = 1.4826 * np.median(np.abs(residual - median))
        reference.append(ref)
        bias.append(median)
        floor.append(max(float(scatter), minimum_floor))
        count.append(n)
    return PhotometryCalibration(
        np.asarray(reference), np.asarray(bias), np.asarray(floor), np.asarray(count)
    )


def photometry_consistency_sum(
    prediction,
    target,
    hidden,
    center,
    scale,
    matrix,
    observed_flux,
    magnitude_errors,
    calibration,
    *,
    band_mask=None,
    decoder_flux=None,
):
    """Return (weighted Huber numerator, valid band count).

    Arrays are XP-only [N,110]; observed_flux is already G-normalised [N,B].
    Visible XP comes from truth; gradients act only on hidden finite XP.
    Each star's band-mean loss is weighted by its hidden XP count. Divide this
    numerator by the same batch scored-entry count used for coefficient MAE.
    Optional decoder_flux substitutes a DETACHED photometry target, preventing
    the two predicted branches from jointly moving the target in this term.
    Nonfinite network outputs in scored XP raise rather than silently disappear.
    """
    dtype, device = prediction.dtype, prediction.device
    convert = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
    hidden = hidden.bool()
    scored = hidden & torch.isfinite(target)
    if not torch.isfinite(prediction[scored]).all():
        raise ValueError("nonfinite predicted hidden XP coefficient")
    matrix, center, scale = convert(matrix), convert(center), convert(scale)
    reference = convert(calibration.reference_flux)
    bias, floor = convert(calibration.bias), convert(calibration.floor)
    if not (
        torch.isfinite(reference).all()
        and (reference > 0).all()
        and torch.isfinite(bias).all()
        and torch.isfinite(floor).all()
        and (floor > 0).all()
    ):
        raise ValueError("invalid photometry calibration")
    complete = torch.isfinite(target).all(dim=1)
    combined = torch.where(
        scored, prediction, torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    )
    synthetic = ((combined * scale + center) @ matrix) / reference
    observed = convert(observed_flux) / reference
    valid = (
        complete[:, None]
        & torch.isfinite(observed)
        & (observed > 0)
        & (scored.sum(dim=1)[:, None] > 0)
    )
    if band_mask is not None:
        valid &= band_mask.bool()
    errors = convert(magnitude_errors)
    errors = torch.where(torch.isfinite(errors) & (errors >= 0), errors, 0.0)
    observed = torch.where(valid, observed, torch.zeros_like(observed))
    uncertainty = torch.sqrt(
        (observed * (np.log(10.0) / 2.5) * errors).square() + floor.square()
    )
    if decoder_flux is None:
        residual = (synthetic - observed - bias) / uncertainty
    else:
        decoder = convert(decoder_flux).detach() / reference
        valid &= torch.isfinite(decoder) & (decoder > 0)
        decoder = torch.where(valid, decoder, torch.zeros_like(decoder))
        residual = (synthetic - decoder - bias) / uncertainty
    residual = torch.where(valid, residual, torch.zeros_like(residual))
    absolute = residual.abs()
    huber = torch.where(absolute <= 1, 0.5 * residual.square(), absolute - 0.5)
    count = valid.sum(dim=1)
    star_mean = huber.sum(dim=1) / count.clamp_min(1)
    return (star_mean * scored.sum(dim=1)).sum(), count.sum()
