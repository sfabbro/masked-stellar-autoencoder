"""Objectives and spectral diagnostics for bounded XP reconstruction experiments."""

import math

import numpy as np
import torch


def noise_reliability(target, sigma):
    """Training-only signal-variance fraction; missing errors keep unit weight."""
    variance = np.nanvar(target, axis=0)
    noise = np.nanmedian(
        np.where(np.isfinite(sigma) & (sigma > 0), sigma**2, np.nan), axis=0
    )
    return np.where(
        np.isfinite(noise) & np.isfinite(variance) & (variance > 0),
        np.clip(1 - noise / np.maximum(variance, 1e-12), 0.05, 1),
        1,
    ).astype(np.float32)


def experiment_objective(
    prediction,
    target,
    mask,
    xp,
    experiment,
    global_count,
    *,
    sigma=None,
    reliability=None,
    correlations=None,
):
    """Global optimizer-batch denominator keeps microbatch gradients equivalent."""
    target = target.masked_fill(~mask, 0)
    safe_prediction = prediction.masked_fill(~mask, 0)
    errors = (safe_prediction - target).abs()
    objective = experiment.get("loss", "mae")
    if objective == "torchregress_covariance":
        from torchregress.losses.gaussian import MultivariateGaussianLoss

        if sigma is None or correlations is None or len(xp) != 110:
            raise ValueError(
                "Full XP covariance requires 110 errors and packed BP/RP correlations"
            )
        # The covariance of scored coefficients is a principal submatrix, not a
        # conditional covariance given visible coefficients. Identity padding
        # lets differently masked stars share a batched 55-dimensional solve.
        total = errors[:, [i for i in range(target.shape[1]) if i not in xp]].sum()
        triangle = torch.tril_indices(55, 55, -1, device=prediction.device)
        for band in range(2):
            columns = xp[band * 55 : (band + 1) * 55]
            scored = mask[:, columns]
            active = scored.any(1)
            if not active.any():
                continue
            scored = scored[active]
            uncertainty = sigma[active][:, columns].double()
            uncertainty = torch.where(
                torch.isfinite(uncertainty) & (uncertainty > 0), uncertainty, 0
            )
            corr = (
                torch.eye(55, device=prediction.device, dtype=torch.float64)
                .expand(int(active.sum()), -1, -1)
                .clone()
            )
            packed = correlations[active, band].double()
            corr[:, triangle[0], triangle[1]] = packed
            corr[:, triangle[1], triangle[0]] = packed
            covariance = corr * uncertainty[:, :, None] * uncertainty[:, None, :]
            covariance += torch.eye(
                55, device=prediction.device, dtype=torch.float64
            ) * experiment.get("variance_floor", 0.1)
            covariance *= scored[:, :, None] & scored[:, None, :]
            covariance += torch.diag_embed((~scored).double())
            # Reject indefinite matrices rather than silently repairing Gaia errors.
            if torch.linalg.cholesky_ex(covariance)[1].any():
                raise ValueError(
                    "XP covariance is not positive definite after the fixed variance floor"
                )
            losses = MultivariateGaussianLoss(reduction="none", jitter=0, eps=0)(
                safe_prediction[active][:, columns].double(),
                target[active][:, columns].double(),
                covariance_matrices=covariance,
            )
            total = (
                total
                + (
                    losses - (~scored).sum(1).double() * (0.5 * math.log(2 * math.pi))
                ).sum()
            )
        return total / global_count.clamp_min(1)
    if objective in {"torchregress_noise", "xp_mse"}:
        from torchregress.losses.uncertain_gt import NoisyTargetGaussianNLL

        variance = torch.zeros_like(target[:, xp])
        if objective == "torchregress_noise":
            if sigma is None:
                raise ValueError(
                    "The noisy-target objective requires coefficient errors"
                )
            uncertainty = sigma[:, xp]
            variance = torch.where(
                torch.isfinite(uncertainty) & (uncertainty > 0), uncertainty, 0
            ).square()
        log_variance = torch.full_like(
            variance, math.log(experiment.get("variance_floor", 0.1))
        )
        loss = NoisyTargetGaussianNLL(reduction="none")(
            (safe_prediction[:, xp], log_variance),
            target[:, xp],
            target_variance=variance,
            mask=mask[:, xp],
        )
        errors = errors.clone()
        errors[:, xp] = loss
    elif objective == "signal_weighted_mae":
        if reliability is None:
            raise ValueError("Reliability weights must be derived from training rows")
        errors = errors.clone()
        errors[:, xp] *= reliability[xp]
    elif objective != "mae":
        raise ValueError(f"Unknown experiment objective: {objective}")
    return errors.masked_fill(~mask, 0).sum() / global_count.clamp_min(1)


def project_spectral_residual(residual, scale, basis):
    """Residuals have no scaler-center term; basis is coefficients × wavelength."""
    return (np.asarray(residual) * np.asarray(scale)) @ np.asarray(basis)


def spectral_basis(names):
    """Load GaiaXPy's calibrated BP/RP sampling and merge matrices."""
    import gaiaxpy
    from gaiaxpy.core.config import load_xpmerge_from_xml, load_xpsampling_from_xml

    sampling = load_xpsampling_from_xml()
    merge = load_xpmerge_from_xml()
    # GaiaXPy returns (wavelength grid, band merge weights).
    wavelength, weights = merge
    basis = np.zeros((len(names), len(wavelength)), dtype=np.float64)
    for i, name in enumerate(names):
        if name.startswith(("bp_", "rp_")):
            band, order = name.split("_")
            basis[i] = sampling[band][int(order) - 1] * weights[band]
    return {
        "wavelength_nm": np.asarray(wavelength),
        "basis": basis,
        "gaiaxpy_version": gaiaxpy.__version__,
        "flux_normalization": "G=8.5 normalized coefficients; all 55 bases retained",
    }


def spectral_summary(target, prediction, mask, scale, center, basis):
    """Only hidden XP predictions replace observations in reconstructed spectra."""
    projection = basis["basis"]
    xp = np.any(projection != 0, axis=1)
    valid = np.isfinite(target[:, xp]).all(1) & mask[:, xp].any(1)
    if not valid.any():
        return {"rows": 0}
    target = target[valid][:, xp]
    prediction = prediction[valid][:, xp]
    mask = mask[valid][:, xp]
    raw_target = target * scale[xp] + center[xp]
    flux = raw_target @ projection[xp]
    residual = project_spectral_residual(
        np.where(mask, prediction - target, 0), scale[xp], projection[xp]
    )
    wavelength = basis["wavelength_nm"]
    absolute = np.trapezoid(abs(residual), wavelength, axis=1)
    reference = np.trapezoid(abs(flux), wavelength, axis=1)
    fraction = absolute[reference > 0] / reference[reference > 0]
    return {
        "rows": int(valid.sum()),
        "integrated_absolute_fraction_mean": float(fraction.mean())
        if len(fraction)
        else None,
        "integrated_absolute_fraction_median": float(np.median(fraction))
        if len(fraction)
        else None,
        "wavelength_mae": abs(residual).mean(axis=0).tolist(),
        "units": basis["flux_normalization"],
    }
