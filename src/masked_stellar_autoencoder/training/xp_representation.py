"""Experimental multiscale spectra and paired-view latent regularization.

These helpers do not change production pretraining. Spectral terms retain the
optimizer-batch coefficient denominator. VICReg statistics intentionally use
the fixed microbatch: their estimator is not invariant to microbatch size.
"""

import math

import torch


def continuum_operator(wavelength, width_nm=40.0):
    """Row-normalized local wavelength box average, including edge windows."""
    if not math.isfinite(width_nm) or width_nm <= 0:
        raise ValueError("Continuum width must be positive and finite")
    if wavelength.ndim != 1 or len(wavelength) < 2:
        raise ValueError("At least two wavelengths are required")
    if not torch.isfinite(wavelength).all() or not (wavelength.diff() > 0).all():
        raise ValueError("Wavelengths must be finite and strictly increasing")
    # Quadrature weights avoid overweighting densely sampled regions.
    delta = wavelength.diff()
    weights = torch.cat((delta[:1], delta[:-1] + delta[1:], delta[-1:])) / 2
    neighborhood = (wavelength[:, None] - wavelength[None, :]).abs() <= width_nm / 2
    operator = neighborhood.to(wavelength.dtype) * weights[None, :]
    return operator / operator.sum(1, keepdim=True)


def multiscale_spectral_relative_error(
    prediction,
    target,
    mask,
    xp,
    scale,
    center,
    basis,
    wavelength,
    *,
    smoothing=None,
    width_nm=40.0,
    shape_weight=1.0,
):
    """Continuum plus complementary local shape error from hidden XP only.

    Both terms divide by the *same full-spectrum* integrated absolute target
    flux. A flat target therefore has a defined shape loss, and rescaling flux
    units cannot change either term. Invalid target references are excluded;
    nonfinite predictions on eligible hidden coefficients raise immediately.
    """
    if not math.isfinite(shape_weight) or shape_weight < 0:
        raise ValueError("Shape weight must be finite and nonnegative")
    valid = torch.isfinite(target[:, xp]).all(1) & mask[:, xp].any(1)
    hidden = mask[:, xp] & valid[:, None]
    if not torch.isfinite(prediction[:, xp][hidden]).all():
        raise FloatingPointError("Nonfinite hidden XP prediction")
    safe_target = target[:, xp].masked_fill(~valid[:, None], 0)
    residual = (prediction[:, xp] - safe_target).masked_fill(~hidden, 0)
    flux = (safe_target * scale[xp] + center[xp]) @ basis[xp]
    residual_flux = (residual * scale[xp]) @ basis[xp]
    reference = torch.trapezoid(flux.abs(), wavelength, dim=1)
    valid = valid & torch.isfinite(reference) & (reference > 0)
    safe_reference = torch.where(valid, reference, torch.ones_like(reference))
    if smoothing is None:
        smoothing = continuum_operator(wavelength, width_nm)
    continuum = residual_flux @ smoothing.T
    shape = residual_flux - continuum
    continuum_error = (
        torch.trapezoid(continuum.abs(), wavelength, dim=1) / safe_reference
    )
    shape_error = torch.trapezoid(shape.abs(), wavelength, dim=1) / safe_reference
    combined = (continuum_error + shape_weight * shape_error).masked_fill(~valid, 0)
    counts = mask[:, xp].sum(1).masked_fill(~valid, 0)
    return combined, counts


def paired_latent_regularizer(
    first,
    second,
    reference_scale,
    *,
    invariance_weight=1.0,
    variance_weight=0.1,
    covariance_weight=0.01,
    std_floor=0.5,
):
    """VICReg-style terms in frozen parent training-standard-deviation units.

    Pair two masks for the SAME star, retaining its XP visibility group. The
    caller must use training-only parent scales frozen before optimization,
    and weight this microbatch mean by microbatch_rows / optimizer_batch_rows.
    No variance target is imposed on original physical latent units.
    """
    if first.ndim != 2 or first.shape != second.shape or len(first) < 2:
        raise ValueError("Paired latent statistics require matching batches of >=2")
    if reference_scale.shape != first.shape[1:]:
        raise ValueError("Reference scale must match latent dimensionality")
    if not torch.isfinite(reference_scale).all() or not (reference_scale > 0).all():
        raise ValueError("Frozen parent latent scales must be finite and positive")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise FloatingPointError("Nonfinite paired latent")
    for value in (invariance_weight, variance_weight, covariance_weight, std_floor):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Latent weights and floor must be finite and nonnegative")
    a, b = first / reference_scale, second / reference_scale
    invariance = (a - b).square().mean()
    variance = first.new_zeros(())
    covariance = first.new_zeros(())
    dimensions = first.shape[1]
    off_diagonal = ~torch.eye(dimensions, dtype=torch.bool, device=first.device)
    for values in (a, b):
        centered = values - values.mean(0)
        # Fixed epsilon is dimensionless after parent standardization.
        std = (centered.square().sum(0) / (len(values) - 1) + 1e-8).sqrt()
        variance = variance + torch.relu(std_floor - std).mean() / 2
        cov = centered.T @ centered / (len(values) - 1)
        # Average pairs rather than sum/D keeps the regularizer bounded as D grows.
        if dimensions > 1:
            covariance = covariance + cov[off_diagonal].square().mean() / 2
    return (
        invariance_weight * invariance
        + variance_weight * variance
        + covariance_weight * covariance
    )
