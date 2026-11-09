"""Explicit NSF score controls; historical energy jobs remain immutable."""

import numpy as np
import torch


def marginal_crps(samples, target, observed):
    """Unbiased sample CRPS, averaged over observed coordinates then stars."""
    if (
        samples.ndim != 3
        or samples.shape[0] < 2
        or target.shape != samples.shape[1:]
        or observed.shape != target.shape
        or observed.dtype != torch.bool
    ):
        raise ValueError("Require [draw,star,target] and aligned observed masks")
    if (
        not observed.any(1).all()
        or not torch.isfinite(samples).all()
        or not torch.isfinite(target[observed]).all()
    ):
        raise ValueError("Require finite draws and observed values per star")
    count = len(samples)
    safe = torch.where(observed, target, torch.zeros_like(target))
    first = (samples - safe[None]).abs().mean(0)
    ordered = samples.sort(dim=0).values
    ranks = torch.arange(1, count + 1, device=samples.device, dtype=samples.dtype)[
        :, None, None
    ]
    pair = ((2 * ranks - count - 1) * ordered).sum(0) / (count * (count - 1))
    score = torch.where(observed, first - pair, torch.zeros_like(first))
    return (score.sum(1) / observed.sum(1)).mean()


def confirmation_score(
    draws,
    target,
    mask,
    sigma,
    known,
    floor,
    generator,
    *,
    energy_weight,
    crps_weight,
    native_rows=None,
    correlation=0.0,
    native_student_t=False,
):
    from masked_stellar_autoencoder.training.multisurvey_masked_energy import (
        masked_energy_score,
    )

    if min(energy_weight, crps_weight) < 0 or energy_weight + crps_weight <= 0:
        raise ValueError("Positive explicit objective weights required")
    if (
        (known & ~mask).any()
        or not torch.isfinite(sigma[known]).all()
        or (sigma[known] <= 0).any()
    ):
        raise ValueError("Invalid known measurement errors")
    floor = torch.as_tensor(floor, device=draws.device, dtype=draws.dtype)
    if (
        floor.shape != (draws.shape[-1],)
        or not torch.isfinite(floor).all()
        or (floor <= 0).any()
    ):
        raise ValueError("Explicit positive unknown-error floors required")
    errors = torch.where(mask, torch.where(known, sigma, floor), 0.0)
    noise = torch.randn(
        draws.shape, device=draws.device, dtype=draws.dtype, generator=generator
    )
    if not -0.49 < correlation < 0.99:
        raise ValueError(
            "Correlation sensitivity must yield a positive-definite 3D block"
        )
    if native_rows is not None:
        native_rows = torch.as_tensor(
            native_rows, device=draws.device, dtype=torch.bool
        )
        if native_rows.shape != (draws.shape[1],) or draws.shape[-1] != 6:
            raise ValueError("Native atmospheric rows require six-target layout")
        if native_rows.any():
            block = noise[:, native_rows, :3]
            if correlation:
                matrix = torch.full(
                    (3, 3), correlation, device=draws.device, dtype=draws.dtype
                )
                matrix.fill_diagonal_(1.0)
                block = block @ torch.linalg.cholesky(matrix).T
            if native_student_t:
                # Student-t nu=4, scaled to unit covariance, shared within the atmospheric block.
                chi = (
                    torch.randn(
                        (4, draws.shape[0], int(native_rows.sum()), 1),
                        device=draws.device,
                        dtype=draws.dtype,
                        generator=generator,
                    )
                    .square()
                    .sum(0)
                )
                block = block * torch.sqrt(2.0 / chi)
            noise[:, native_rows, :3] = block
    noisy = draws + noise * errors[None]
    energy = masked_energy_score(
        noisy,
        target,
        mask,
        scale=torch.ones(draws.shape[-1], device=draws.device, dtype=draws.dtype),
    )
    crps = marginal_crps(noisy, target, mask)
    return energy_weight * energy + crps_weight * crps, dict(
        masked_energy_score=energy.detach(), marginal_crps=crps.detach()
    )


def make_head(latent_dim, cfg):
    from masked_stellar_autoencoder.training.remediation_energy6 import (
        EnergyJointSixHead,
    )

    class ConfirmationHead(EnergyJointSixHead):
        def criterion(
            self, context, target, sigma, mask, error_known, label_source=None
        ):
            if self.sample_generator is None or self.noise_generator is None:
                raise ValueError("Explicit generators required")
            samples = self.sample(
                context, self.energy_training_draws, self.sample_generator
            )
            native_rows = None
            if label_source is not None:
                source = np.asarray(label_source).astype("U")
                same = (source[:, :3] == source[:, :1]).all(1) & np.isin(
                    source[:, 0], ["apogee", "galah"]
                )
                native_rows = (
                    torch.as_tensor(same, device=target.device)
                    & mask[:, :3].all(1)
                    & error_known[:, :3].all(1)
                )
            value, components = confirmation_score(
                samples,
                target,
                mask,
                sigma,
                error_known,
                self.unknown_error_floor_normalized,
                self.noise_generator,
                energy_weight=cfg["confirmation_energy_weight"],
                crps_weight=cfg["confirmation_crps_weight"],
                native_rows=native_rows,
                correlation=cfg.get("confirmation_noise_correlation", 0.0),
                native_student_t=cfg.get("confirmation_native_student_t", False),
            )
            components["known_same_source_native_fraction"] = (
                target.new_tensor(0.0)
                if native_rows is None
                else native_rows.float().mean().detach()
            )
            self.last_loss_components = components
            return value

    return ConfirmationHead(
        latent_dim=latent_dim,
        width=cfg["head_width"],
        particles=cfg["particles"],
        seed=cfg["seed"],
        components=cfg["mixture_components"],
        energy_training_draws=cfg["energy_training_draws"],
        unknown_error_floor_normalized=cfg["unknown_error_floor_normalized"],
    )


def backward_energy_chunks(
    head, latent, target, sigma, mask, known, rows, *, chunk_rows, tensor, label_sources
):
    rows = np.asarray(rows)
    if rows.ndim != 1 or not len(rows) or np.any(np.diff(rows) <= 0) or chunk_rows < 1:
        raise ValueError("Sorted unique rows and positive chunk required")
    if np.asarray(label_sources).shape != np.asarray(target).shape:
        raise ValueError("Source provenance must align with every target")
    total = len(rows)
    value = None
    components = {}
    for start in range(0, total, chunk_rows):
        rr = rows[start : start + chunk_rows]
        loss = head.criterion(
            latent(rr),
            tensor(target[rr]),
            tensor(sigma[rr]),
            tensor(mask[rr], torch.bool),
            tensor(known[rr], torch.bool),
            label_source=label_sources[rr],
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite confirmation objective")
        fraction = len(rr) / total
        contribution = loss * fraction
        contribution.backward()
        value = (
            contribution.detach() if value is None else value + contribution.detach()
        )
        for key, v in head.last_loss_components.items():
            components[key] = components.get(key, 0.0) + float(v) * fraction
    return value, components
