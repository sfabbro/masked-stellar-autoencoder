"""Controlled native Teff/logg/FeH heads, calibration and label-free refinement.

All joint-density methods use the same complete-case native rows. Quoted errors
are diagonal survey measurement errors, not a physical-truth certificate.
"""

from __future__ import annotations

import hashlib
import math
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

TARGET_NAMES = ("teff", "logg", "fe_h")


def complete_roles(observed, quality, roles):
    observed, quality = np.asarray(observed), np.asarray(quality)
    if observed.shape != quality.shape or observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError("Three aligned native targets required")
    complete = (observed & quality).all(1)
    result = {
        role: np.flatnonzero(np.asarray(roles[role], dtype=bool) & complete)
        for role in ("fit", "tune", "calibration", "test")
    }
    for i, left in enumerate(result):
        for right in list(result)[i + 1 :]:
            if np.intersect1d(result[left], result[right]).size:
                raise ValueError("Native source roles overlap")
    return result


def label_scaling(value, fit):
    values = np.asarray(value)[fit]
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite complete fit labels required")
    center, scale = values.mean(0), values.std(0)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Degenerate fit-only target scale")
    return center, scale


def qmc_normal_nodes(count=256, dimensions=3, seed=2026):
    if count < 8 or count % 2:
        raise ValueError("Even antithetic QMC count of at least eight required")
    uniform = (
        torch.quasirandom.SobolEngine(dimensions, scramble=True, seed=seed)
        .draw(count // 2)
        .double()
    )
    if not ((uniform > 0) & (uniform < 1)).all():
        raise FloatingPointError("Sobol endpoint cannot be mapped to finite normal")
    normal = math.sqrt(2.0) * torch.erfinv(2.0 * uniform - 1.0)
    return torch.cat([normal, -normal]).float()


def supervised_fit_rows(rng, fit, batch_size=256):
    """Draw a bounded FIT minibatch without repeated source positions."""
    if batch_size < 1 or len(fit) < 2:
        raise ValueError("At least two FIT objects and a positive batch are required")
    return rng.choice(fit, size=min(batch_size, len(fit)), replace=False)


class MultiLabelHead(nn.Module):
    """Native three-output Huber, diagonal Gaussian or conditional joint flow."""

    def __init__(
        self,
        kind="gaussian",
        *,
        latent_dim=256,
        width=128,
        flow_type="nsf",
        noise_aware=False,
        mc_samples=32,
        flow_bound=8.0,
        flow_objective="nll",
        contrastive_weight=0.1,
    ):
        super().__init__()
        if kind not in {"huber", "gaussian", "flow"}:
            raise ValueError("Unsupported multilabel head")
        if noise_aware and kind == "huber":
            raise ValueError("Huber is not a density measurement likelihood")
        if flow_objective not in {"nll", "contrastive", "nll_contrastive"}:
            raise ValueError("Unsupported flow objective")
        if flow_objective != "nll" and (kind != "flow" or noise_aware):
            raise ValueError("Contrastive contexts require a clean joint flow")
        if contrastive_weight < 0:
            raise ValueError("Contrastive weight must be nonnegative")
        self.flow_objective = flow_objective
        self.contrastive_weight = float(contrastive_weight)
        self.kind, self.noise_aware, self.flow_type = kind, bool(noise_aware), flow_type
        self.input = nn.Linear(latent_dim, width)
        self.block = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width // 4),
            nn.GELU(),
            nn.Linear(width // 4, width),
        )
        self.output = nn.Linear(
            width, 16 if kind == "flow" else 6 if kind == "gaussian" else 3
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if kind == "gaussian":
            from torchregress.losses.gaussian import GaussianNLLLoss

            self.loss_fn = GaussianNLLLoss(
                log_variance=False,
                min_variance=torch.finfo(torch.float32).tiny,
                eps=0.0,
            )
        elif kind == "flow":
            from torchregress.losses.nflows import (
                NormalizingFlowLoss,
                create_flow_model,
            )

            kwargs = dict(
                n_features=3,
                context_dim=16,
                flow_type=flow_type,
                n_transforms=3,
                hidden_features=[64, 64],
            )
            if flow_type == "nsf":
                kwargs.update(bound=flow_bound, bins=16)
            if flow_type == "bpf":
                kwargs.update(degree=8)
            flow = create_flow_model(**kwargs)
            if flow_objective == "nll":
                base = NormalizingFlowLoss(flow)
            else:
                from torchregress.losses.nflows import ContrastiveFlowLoss

                base = ContrastiveFlowLoss(flow, temperature=1.0, margin=0.0)

            if noise_aware:
                from torchregress.losses.noise_aware_density import (
                    NoiseAwareDensityLoss,
                )

                self.loss_fn = NoiseAwareDensityLoss(
                    base, mode="mc", n_samples=mc_samples, reduction="none"
                )
            else:
                self.loss_fn = base
        self.register_buffer("point_nodes", qmc_normal_nodes())
        self.register_buffer("measurement_nodes", qmc_normal_nodes(mc_samples))

    def forward(self, latent):
        hidden = torch.nn.functional.gelu(self.input(latent))
        return self.output(hidden + self.block(hidden))

    def distribution(self, latent):
        if self.kind != "flow":
            raise ValueError("Joint flow required")
        loss = self.loss_fn.base_loss if self.noise_aware else self.loss_fn
        return loss.flow(self(latent))

    def criterion(self, latent, target, sigma, *, source_ids=None):
        if target.shape != (len(latent), 3) or sigma.shape != target.shape:
            raise ValueError("Complete standardized three-target rows required")
        if (
            not torch.isfinite(target).all()
            or not torch.isfinite(sigma).all()
            or not (sigma > 0).all()
        ):
            raise ValueError("Invalid native target/error")
        raw = self(latent)
        if self.kind == "huber":
            return torch.nn.functional.huber_loss(raw, target)
        if self.kind == "gaussian":
            mean, logvar = raw.chunk(2, -1)
            variance = logvar.exp()
            if self.noise_aware:
                variance = variance + sigma.square()
            if not torch.isfinite(variance).all() or not (variance > 0).all():
                raise FloatingPointError("Nonfinite Gaussian variance")
            return self.loss_fn((mean, variance), target)
        if self.noise_aware:
            return self.loss_fn(raw, target, sigma=sigma).mean() / 3
        if self.flow_objective != "nll":
            if source_ids is None or len(raw) < 2:
                raise ValueError("Contrastive batches require unique source IDs")
            if len(torch.unique(source_ids)) != len(raw):
                raise ValueError("Contrastive negatives cannot contain duplicate stars")
            # One fixed cyclic derangement; both context branches retain gradients.
            negative = torch.roll(raw, shifts=1, dims=0).unsqueeze(1)
            contrastive = self.loss_fn(raw, target, negative_context=negative)
            if self.flow_objective == "contrastive":
                return contrastive / 3
            nll = -self.loss_fn.log_prob(raw, target).mean()
            return (nll + self.contrastive_weight * contrastive) / 3
        return self.loss_fn(raw, target) / 3

    def moments(self, latent):
        if self.kind == "flow":
            dist = self.distribution(latent)
            # Preserve all fixed nodes and their reduction order while bounding
            # the spline inverse's temporary sample-by-context allocation.
            samples = [
                dist.transform.inv(nodes[:, None, :].expand(-1, len(latent), -1))
                for nodes in self.point_nodes.split(16)
            ]
            sample = torch.cat(samples, dim=0)
            mean = sample.double().mean(0).to(latent.dtype)
            covariance = (sample - mean).square().double().mean(0).to(latent.dtype)
            return mean, covariance
        raw = self(latent)
        if self.kind == "huber":
            return raw, torch.full_like(raw, float("nan"))
        mean, logvar = raw.chunk(2, -1)
        return mean, logvar.exp()

    def log_prob(self, latent, target, sigma=None):
        if self.kind == "flow":
            distribution = self.distribution(latent)
            if sigma is None:
                return distribution.log_prob(target)
            perturbed = (
                target.unsqueeze(0)
                + sigma.unsqueeze(0) * self.measurement_nodes[:, None, :]
            )
            return torch.logsumexp(distribution.log_prob(perturbed), 0) - math.log(
                len(self.measurement_nodes)
            )
        if self.kind == "huber":
            return torch.full((len(latent),), float("nan"), device=latent.device)
        mean, variance = self.moments(latent)
        if sigma is not None:
            variance = variance + sigma.square()
        return (
            -0.5
            * (
                (target - mean).square() / variance
                + variance.log()
                + math.log(2 * math.pi)
            )
        ).sum(-1)


def conformal_intervals(
    calibration_prediction, calibration_target, test_prediction, alpha=0.1
):
    """Marginal library CP and conservative simultaneous Bonferroni boxes."""
    from torchregress.losses.conformal import MultiTargetConformal

    marginal = MultiTargetConformal(alpha=alpha)
    marginal.calibrate(calibration_prediction, calibration_target)
    simultaneous = MultiTargetConformal(alpha=alpha / 3)
    simultaneous.calibrate(calibration_prediction, calibration_target)
    lower, upper = marginal.predict_interval(test_prediction)
    joint_lower, joint_upper = simultaneous.predict_interval(test_prediction)
    return dict(
        marginal_lower=lower,
        marginal_upper=upper,
        simultaneous_lower=joint_lower,
        simultaneous_upper=joint_upper,
        marginal_threshold=marginal.q_hat,
        simultaneous_threshold=simultaneous.q_hat,
    )


def hash_selected(ids, count=512, seed=1042):
    keys = np.asarray(
        [
            int.from_bytes(
                hashlib.sha256(f"{int(s)}:{seed}".encode()).digest()[:8], "little"
            )
            for s in ids
        ],
        dtype="u8",
    )
    return np.argsort(keys, kind="stable")[: min(len(ids), count)]


@contextmanager
def refinement_scope(model, head):
    """Freeze predictor/decoder; restore complete per-object LN parameter state."""
    parameters = list(model.parameters()) + list(head.parameters())
    flags = [p.requires_grad for p in parameters]
    modes = [
        (module, module.training) for root in (model, head) for module in root.modules()
    ]
    selected = [
        p
        for module in model.encoder.modules()
        if isinstance(module, nn.LayerNorm)
        for p in module.parameters(recurse=False)
    ]
    if not selected:
        raise ValueError("No encoder LayerNorm affine parameters")
    saved = [p.detach().clone() for p in selected]
    gradients = [
        None if p.grad is None else p.grad.detach().clone() for p in parameters
    ]
    try:
        for p in parameters:
            p.requires_grad_(False)
        for p in selected:
            p.requires_grad_(True)
        model.eval()
        head.eval()
        yield selected, saved
    finally:
        with torch.no_grad():
            for p, v in zip(selected, saved, strict=True):
                p.copy_(v)
        for p, flag, grad in zip(parameters, flags, gradients, strict=True):
            p.requires_grad_(flag)
            p.grad = grad
        for module, mode in modes:
            module.training = mode


def refine_one(
    model, head, x, visible, xp_columns, source_id, *, steps=3, lr=1e-4, seed=1042
):
    """MSA reconstruction TTT; labels/errors never enter this local objective."""
    if x.shape[0] != 1 or visible.shape != x.shape:
        raise ValueError("One object with its observed input mask required")
    with refinement_scope(model, head) as (parameters, original):
        valid = [int(c) for c in xp_columns if bool(visible[0, c])]
        if len(valid) < 2:
            raise ValueError("At least two observed XP coefficients required")
        key = hashlib.sha256(f"{int(source_id)}:{seed}:holdout".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
        chosen = [c for c in valid if rng.random() < 0.2]
        if not chosen:
            chosen = [valid[int(rng.integers(len(valid)))]]
        if len(chosen) == len(valid):
            chosen = chosen[:-1]
        hidden = x.clone()
        hidden[:, chosen] = -9999.0
        with torch.no_grad():
            baseline = head.moments(model.encoder(x))[0]
        optimizer = torch.optim.Adam(parameters, lr=lr)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            latent = model.encoder(hidden)
            reconstruction = model.reconstruction_layer(model.decoder(latent))
            if getattr(model, "heteroscedastic", False):
                reconstruction = reconstruction.chunk(2, -1)[0]
            loss = (reconstruction[:, chosen] - x[:, chosen]).abs().mean()
            drift = sum(
                (p - v).square().mean()
                for p, v in zip(parameters, original, strict=True)
            ) / len(parameters)
            mean = head.moments(model.encoder(x))[0]
            loss = loss + 0.1 * drift + 0.1 * (mean - baseline).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite local refinement")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            prediction = head.moments(model.encoder(x))[0].detach()
        return prediction, dict(
            steps=steps, heldout_count=len(chosen), final_objective=float(loss.detach())
        )


def native_xp_error_subset(errors, scale, xp, g_magnitude, used_rows):
    """Normalize only the model's actual rows, never unrelated missing-XP stars."""
    from masked_stellar_autoencoder.training.input_measurement import (
        standardized_xp_errors,
    )

    used = np.unique(np.asarray(used_rows, dtype="i8"))
    if not len(used) or used[0] < 0 or used[-1] >= len(errors):
        raise ValueError("Actual XP source rows required")
    sigma = standardized_xp_errors(
        np.asarray(errors)[used],
        scale,
        xp,
        g_magnitude=np.asarray(g_magnitude)[used],
        error_convention="raw_xp",
    )
    return used, sigma
