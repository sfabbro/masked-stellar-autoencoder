"""Decoder-frozen, label-free TTT with held-out XP and clean-state fallback.

The held-out reconstruction score selects an object-specific step, including
zero. It is not a guarantee of improved label prediction or calibrated density.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn

from .mae_test_time import object_adaptation_scope


def split_columns(visible, xp_columns, source_id, step, *, seed=2047):
    """Persistent validation columns and disjoint stateless update columns."""
    valid = np.asarray([int(c) for c in xp_columns if bool(visible[0, c])])
    if len(valid) < 4:
        raise ValueError("At least four observed XP coefficients required")
    fixed_key = hashlib.sha256(
        f"{int(source_id)}:{seed}:guarded-ttt-validation".encode()
    ).digest()
    fixed_rng = np.random.default_rng(int.from_bytes(fixed_key[:8], "little"))
    validation = np.sort(
        fixed_rng.choice(valid, max(1, len(valid) // 4), replace=False)
    )
    remaining = np.setdiff1d(valid, validation)
    step_key = hashlib.sha256(
        f"{int(source_id)}:{seed}:{step}:guarded-ttt-update".encode()
    ).digest()
    rng = np.random.default_rng(int.from_bytes(step_key[:8], "little"))
    update = np.sort(rng.choice(remaining, max(1, len(remaining) // 5), replace=False))
    return validation, update


def _reconstruct(model, x):
    z = model.encoder(x)
    output = model.reconstruction_layer(model.decoder(z))
    return z, output.chunk(2, -1)[0] if getattr(
        model, "heteroscedastic", False
    ) else output


def adapt_guarded(
    model,
    head,
    x,
    visible,
    xp_columns,
    source_id,
    *,
    baseline_mean,
    baseline_variance,
    scope="encoder_only",
    optimizer_name="sgd",
    steps=20,
    lr=1e-5,
    latent_anchor=0.1,
    maximum_relative_latent_drift=0.05,
    seed=2047,
):
    """Frozen decoder/head; SGD/Adam updates and held-out step selection.

    The same persistent XP validation columns are hidden during every gradient
    update and never contribute to the optimized loss. Reconstruction-based
    selection includes the exact clean step, whose saved predictions are reused.
    No target labels/errors are accepted. State, gradients, modes and RNG reset
    even on failure. Uses a fresh optimizer for each object.
    """
    if x.shape != visible.shape or len(x) != 1 or steps < 1:
        raise ValueError("One object and positive step count required")
    if scope not in {"encoder_only", "layernorm"} or optimizer_name not in {
        "sgd",
        "adam",
    }:
        raise ValueError("Unknown guarded TTT scope or optimizer")
    if lr <= 0 or latent_anchor < 0 or maximum_relative_latent_drift <= 0:
        raise ValueError("Invalid guarded TTT hyperparameters")
    if baseline_mean.shape != baseline_variance.shape or len(baseline_mean) != 1:
        raise ValueError("Saved clean mean and variance must have shape [1,D]")
    validation, _ = split_columns(visible, xp_columns, source_id, 0, seed=seed)
    context_scope = "encoder_decoder" if scope == "encoder_only" else "layernorm"
    with object_adaptation_scope(model, head, context_scope):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        selected = (
            list(model.encoder.parameters())
            if scope == "encoder_only"
            else [
                p
                for m in model.encoder.modules()
                if isinstance(m, nn.LayerNorm)
                for p in m.parameters(recurse=False)
            ]
        )
        if not selected:
            raise ValueError("No selected encoder parameters")
        for parameter in selected:
            parameter.requires_grad_(True)
        optimizer = (
            torch.optim.SGD(selected, lr=lr, momentum=0.9)
            if optimizer_name == "sgd"
            else torch.optim.Adam(selected, lr=lr)
        )
        monitor_input = x.detach().clone()
        monitor_input[:, validation] = -9999.0
        with torch.no_grad():
            clean_latent, initial_reconstruction = _reconstruct(model, monitor_input)
            initial_score = (
                (initial_reconstruction[:, validation] - x[:, validation])
                .square()
                .mean()
            )
        if not torch.isfinite(initial_score):
            raise FloatingPointError("Nonfinite clean held-out objective")
        best_score = float(initial_score)
        selected_step = 0
        best_state = None
        trace = []
        for step in range(steps):
            _, update = split_columns(visible, xp_columns, source_id, step, seed=seed)
            hidden = monitor_input.clone()
            hidden[:, update] = -9999.0
            optimizer.zero_grad(set_to_none=True)
            latent, reconstruction = _reconstruct(model, hidden)
            reconstruction_loss = (
                (reconstruction[:, update] - x[:, update]).square().mean()
            )
            anchor = (latent - clean_latent).square().mean()
            loss = reconstruction_loss + latent_anchor * anchor
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite guarded adaptation objective")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in selected
            ):
                raise FloatingPointError("Nonfinite guarded adaptation gradient")
            optimizer.step()
            with torch.no_grad():
                monitored_latent, reconstruction = _reconstruct(model, monitor_input)
                score = (
                    (reconstruction[:, validation] - x[:, validation]).square().mean()
                )
                drift = (
                    (monitored_latent - clean_latent).square().mean().sqrt()
                    / clean_latent.square().mean().sqrt().clamp_min(1e-6)
                )
            finite = bool(torch.isfinite(score) and torch.isfinite(drift))
            accepted = finite and float(drift) <= maximum_relative_latent_drift
            if accepted and float(score) < best_score:
                best_score, selected_step = float(score), step + 1
                best_state = [p.detach().clone() for p in selected]
            trace.append(
                dict(
                    step=step + 1,
                    update_columns=update.tolist(),
                    validation_columns=validation.tolist(),
                    training_objective=float(loss.detach()),
                    heldout_mse=float(score) if finite else None,
                    relative_latent_drift=float(drift) if finite else None,
                    admissible=accepted,
                )
            )
        if best_state is None:
            mean, variance = (
                baseline_mean.detach().clone(),
                baseline_variance.detach().clone(),
            )
        else:
            with torch.no_grad():
                for parameter, value in zip(selected, best_state, strict=True):
                    parameter.copy_(value)
                mean, variance = head.moments(model.encoder(x))
                mean, variance = mean.detach().clone(), variance.detach().clone()
        if (
            not torch.isfinite(mean).all()
            or not torch.isfinite(variance).all()
            or not (variance > 0).all()
        ):
            raise FloatingPointError("Invalid guarded predictive moments")
        return (
            mean,
            variance,
            dict(
                selected_step=selected_step,
                initial_heldout_mse=float(initial_score),
                selected_heldout_mse=best_score,
                trace=trace,
                clean_fallback=selected_step == 0,
            ),
        )


def gaussian_log_prob_from_moments(value, mean, variance):
    """Float64 diagonal Gaussian score from one coherent saved moment pair."""
    value, mean, variance = [np.asarray(v, dtype="f8") for v in (value, mean, variance)]
    if value.shape != mean.shape or value.shape != variance.shape or value.ndim != 2:
        raise ValueError("Gaussian values/moments must have equal [N,D] shapes")
    if (
        not all(np.isfinite(v).all() for v in (value, mean, variance))
        or not (variance > 0).all()
    ):
        raise ValueError("Finite positive Gaussian moments required")
    return (
        -0.5 * ((value - mean) ** 2 / variance + np.log(variance) + np.log(2 * np.pi))
    ).sum(-1)
