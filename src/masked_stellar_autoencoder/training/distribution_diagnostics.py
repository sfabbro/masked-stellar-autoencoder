"""Distribution diagnostics preserving the learned head and native supports."""

import math

import torch
from torchregress.metrics.distribution import (
    crps_from_samples,
    energy_score,
    pinball_loss,
    variogram_score,
)

LEVELS = (0.025, 0.05, 0.16, 0.5, 0.84, 0.95, 0.975)
GROUPS = {"native3": (0, 1, 2), "age_distance": (4, 5), "full6": tuple(range(6))}


def normal_cdf(value):
    return torch.special.ndtr(value)


def marginal_cdf(head, latent, target, sigma=None):
    """Exact intrinsic CDF; NSF observed convolution uses its fixed 32 nodes.

    Targets and quoted sigmas are in the checkpoint's standardized coordinates.
    Unknown sigmas must be represented separately and supplied as zero here.
    """
    error = torch.zeros_like(target) if sigma is None else sigma
    if target.shape != (len(latent), 6) or error.shape != target.shape:
        raise ValueError("Six-coordinate CDF tensor layout differs")
    if (
        not torch.isfinite(latent).all()
        or not torch.isfinite(target).all()
        or not torch.isfinite(error).all()
        or (error < 0).any()
    ):
        raise ValueError("Finite targets and nonnegative standard errors required")
    if head.kind == "gaussian":
        mean, log_variance = head.head(latent).chunk(2, -1)
        return normal_cdf(
            (target - mean) / (log_variance.exp() + error.square()).sqrt()
        )
    if head.kind == "mdn":
        log_weight, mean, log_std = head.head._mixture_parameters(head.head(latent))
        scale = ((2 * log_std).exp() + error[:, None].square()).sqrt()
        return (
            log_weight.exp()[:, :, None] * normal_cdf((target[:, None] - mean) / scale)
        ).sum(1)
    if head.kind != "factorized_nsf":
        raise ValueError("Unsupported distribution family")
    columns = []
    for c, scalar in enumerate(head.heads):
        distribution = scalar.distribution(latent)
        if sigma is None:
            columns.append(
                normal_cdf(distribution.transform(target[:, c : c + 1]))[:, 0]
            )
        else:
            nodes = scalar.measurement_nodes[:, 0].to(target)
            values = target[:, c][None] - nodes[:, None] * error[:, c][None]
            parts = [
                normal_cdf(distribution.transform(block[:, :, None]))[:, :, 0]
                for block in values.split(8)
            ]
            columns.append(torch.cat(parts).mean(0))
    result = torch.stack(columns, -1)
    if not torch.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise FloatingPointError("CDF outside [0,1]")
    return result


def predictive_draws(head, latent, count, generator):
    """Actual joint head samples, not a Normal fitted to its moments."""
    eps = torch.randn(
        (count, len(latent), 6),
        device=latent.device,
        dtype=latent.dtype,
        generator=generator,
    )
    if head.kind == "gaussian":
        mean, lv = head.head(latent).chunk(2, -1)
        return mean[None] + (lv * 0.5).exp()[None] * eps
    if head.kind == "mdn":
        log_weight, mean, log_std = head.head._mixture_parameters(head.head(latent))
        component = torch.multinomial(
            log_weight.exp(), count, replacement=True, generator=generator
        )
        index = torch.arange(len(latent), device=latent.device)[:, None]
        return (
            mean[index, component]
            + log_std[index, component].exp() * eps.permute(1, 0, 2)
        ).permute(1, 0, 2)
    if head.kind != "factorized_nsf":
        raise ValueError("Unsupported distribution family")
    values = []
    for c, scalar in enumerate(head.heads):
        distribution = scalar.distribution(latent)
        values.append(
            torch.cat(
                [
                    distribution.transform.inv(block[:, :, None])[:, :, 0]
                    for block in eps[:, :, c].split(8)
                ]
            )
        )
    return torch.stack(values, -1)


def native_coordinates(standardized, center, scale):
    coordinate = standardized * scale + center
    native = coordinate.clone()
    native[..., 4:] = native[..., 4:].exp()
    if not torch.isfinite(native).all() or (native[..., 4:] <= 0).any():
        raise FloatingPointError("Nonfinite predictive samples; no clipping allowed")
    return native


def sample_metrics(draws, native_target, center, scale, mask, *, energy_samples=128):
    """Proper empirical scores in native units, and FIT-scaled joint scores.

    Joint energy/variogram operate in standardized original coordinates: age and
    distance axes are natural logarithms. They do not mix K/dex/Gyr/kpc norms.
    """
    if draws.ndim != 3 or draws.shape[1:] != native_target.shape or draws.shape[0] < 2:
        raise ValueError("At least two samples [S,B,6] required")
    safe = torch.where(mask, native_target, torch.zeros_like(native_target))
    score = crps_from_samples(draws, safe, reduction="none")
    result = {
        "crps_native": score.masked_fill(~mask, float("nan")),
        "vario_rho1_native": (-score).masked_fill(~mask, float("nan")),
    }
    coordinate = draws.clone()
    coordinate[..., 4:] = coordinate[..., 4:].log()
    coordinate = (coordinate - center) / scale
    truth = safe.clone()
    truth[:, 4:] = torch.where(
        mask[:, 4:], safe[:, 4:], torch.ones_like(safe[:, 4:])
    ).log()
    truth = (truth - center) / scale
    for name, indices in GROUPS.items():
        valid = mask[:, indices].all(-1)
        es = torch.full(
            (len(truth),), float("nan"), device=draws.device, dtype=draws.dtype
        )
        vs = es.clone()
        if valid.any():
            forecast = coordinate[:, valid][:, :, indices]
            target = truth[valid][:, indices]
            es[valid] = energy_score(
                forecast[:energy_samples], target, reduction="none"
            )
            vs[valid] = variogram_score(forecast, target, p=0.5, reduction="none")
        result[f"energy_{name}_fit_standardized"] = es
        result[f"variogram_{name}_fit_standardized"] = vs
    return result


def predictive_quantile_metrics(draws, native_target, mask):
    """Native empirical quantiles and proper interval/pinball scores."""
    safe = torch.where(mask, native_target, torch.zeros_like(native_target))
    result = {}
    levels = draws.new_tensor(LEVELS)
    quantiles = torch.quantile(draws, levels, dim=0)
    for i, level in enumerate(LEVELS):
        result[f"quantile_{level:g}_native"] = quantiles[i]
        result[f"pinball_{level:g}_native"] = pinball_loss(
            level, quantiles[i], safe
        ).masked_fill(~mask, float("nan"))
    for level, lower, upper in [(90, 1, 5), (95, 0, 6)]:
        lo, hi = quantiles[lower], quantiles[upper]
        alpha = 1 - level / 100
        width = hi - lo
        interval_score = (
            width
            + 2 / alpha * (lo - safe).clamp_min(0)
            + 2 / alpha * (safe - hi).clamp_min(0)
        )
        result[f"predictive_{level}_covered"] = (
            ((safe >= lo) & (safe <= hi))
            .to(draws.dtype)
            .masked_fill(~mask, float("nan"))
        )
        result[f"predictive_{level}_width_native"] = width.masked_fill(
            ~mask, float("nan")
        )
        result[f"predictive_{level}_interval_score_native"] = (
            interval_score.masked_fill(~mask, float("nan"))
        )
    return result


def analytic_normal_crps(mean, variance, target):
    """Independent Normal reference for synthetic kernel checks."""
    std = variance.sqrt()
    z = (target - mean) / std
    return std * (
        z * (2 * normal_cdf(z) - 1)
        + 2 * torch.exp(-z.square() / 2) / math.sqrt(2 * math.pi)
        - 1 / math.sqrt(math.pi)
    )
