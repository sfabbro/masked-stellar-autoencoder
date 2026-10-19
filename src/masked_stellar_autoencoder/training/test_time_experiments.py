"""Object-local, label-free test-time probes for a frozen MSA predictor.

These operators never accept labels. LayerNorm adaptation restores its complete
parameter scope per object; latent refinement freezes the entire model/head.
Masks use the trained missing-value sentinel and score held-out observed XP only.
"""

import hashlib
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn


def _rng(source_id, seed, view, domain):
    key = f"{int(source_id)}:{int(seed)}:{int(view)}:{domain}".encode()
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(key).digest()[:8], "little")
    )


def augmented_view(
    values,
    visible,
    sigma_xp,
    source_ids,
    xp_columns,
    *,
    seed=1042,
    view=0,
    noise_scale=0.0,
    mask_probability=0.0,
):
    """Independent stateless per-source XP noise/missingness; caller untouched."""
    if not 0 <= mask_probability < 1 or noise_scale < 0:
        raise ValueError("invalid noise/masking policy")
    if values.ndim != 2 or visible.shape != values.shape or visible.dtype != torch.bool:
        raise ValueError("values/visible must be paired matrices")
    columns = list(map(int, xp_columns))
    if (
        len(set(columns)) != len(columns)
        or not columns
        or min(columns) < 0
        or max(columns) >= values.shape[1]
    ):
        raise ValueError("distinct XP columns required")
    if sigma_xp.shape != (len(values), len(columns)):
        raise ValueError("XP sigma has wrong shape")
    if (
        len(source_ids) != len(values)
        or not torch.isfinite(sigma_xp).all()
        or (sigma_xp < 0).any()
    ):
        raise ValueError("finite nonnegative XP sigma and source IDs required")
    out = values.clone()
    hidden = torch.zeros_like(visible)
    valid = (
        visible[:, columns]
        & torch.isfinite(values[:, columns])
        & (values[:, columns] != -9999)
    )
    noise, masks = [], []
    for source in source_ids:
        noise.append(_rng(source, seed, view, "noise").standard_normal(len(columns)))
        masks.append(
            _rng(source, seed, view, "mask").random(len(columns)) < mask_probability
        )
    perturbation = torch.as_tensor(
        np.asarray(noise), dtype=values.dtype, device=values.device
    )
    local_mask = (
        torch.as_tensor(np.asarray(masks), dtype=torch.bool, device=values.device)
        & valid
    )
    out[:, columns] = torch.where(
        valid, out[:, columns] + noise_scale * sigma_xp * perturbation, out[:, columns]
    )
    hidden[:, columns] = local_mask
    out[hidden] = -9999
    return out, hidden


@contextmanager
def frozen_predictor(model, head, *, layernorm=False):
    """Freeze everything except encoder LayerNorm affine terms; restore on exit."""
    parameters = list(model.parameters()) + list(head.parameters())
    flags = [p.requires_grad for p in parameters]
    gradients = [
        None if p.grad is None else p.grad.detach().clone() for p in parameters
    ]
    modes = [
        (module, module.training) for root in (model, head) for module in root.modules()
    ]
    selected = []
    if layernorm:
        for module in model.encoder.modules():
            if isinstance(module, nn.LayerNorm):
                selected.extend(module.parameters(recurse=False))
        if not selected:
            raise ValueError("encoder has no LayerNorm affine parameters")
    for p in parameters:
        p.requires_grad_(False)
    if layernorm:
        for p in selected:
            p.requires_grad_(True)
    saved = [p.detach().clone() for p in selected]
    model.eval()
    head.eval()
    try:
        yield selected
    finally:
        with torch.no_grad():
            for p, value in zip(selected, saved, strict=True):
                p.copy_(value)
        for p, flag, gradient in zip(parameters, flags, gradients, strict=True):
            p.requires_grad_(flag)
            p.grad = gradient
        for module, mode in modes:
            module.training = mode


def _mean(head, latent):
    return head(latent)[:, :, 0].mean(1)


def masked_object(values, visible, xp_columns, source_id, *, seed, probability):
    hidden, scored = augmented_view(
        values,
        visible,
        torch.zeros((1, len(xp_columns)), device=values.device),
        [source_id],
        xp_columns,
        seed=seed,
        mask_probability=probability,
    )
    valid = [int(c) for c in xp_columns if bool(visible[0, c])]
    if len(valid) < 2:
        raise ValueError("adaptation requires at least two visible XP coefficients")
    # A deterministic nonempty proper holdout, including rare RNG edge cases.
    if not scored.any():
        selected = valid[int(_rng(source_id, seed, 0, "fallback").integers(len(valid)))]
        scored[0, selected] = True
        hidden[0, selected] = -9999
    if all(bool(scored[0, c]) for c in valid):
        scored[0, valid[-1]] = False
        hidden[0, valid[-1]] = values[0, valid[-1]]
    return hidden, scored


def adapt_one(
    model,
    head,
    values,
    visible,
    xp_columns,
    source_id,
    *,
    mode,
    steps=3,
    lr=1e-4,
    mask_probability=0.2,
    anchor=0.1,
    consistency=0.0,
    seed=1042,
):
    """One-object inference. No label, label error, batch statistics or test stopping."""
    if values.shape[0] != 1 or steps < 0 or lr <= 0 or anchor < 0 or consistency < 0:
        raise ValueError("one object and nonnegative fixed adaptation budget required")
    if mode not in {"masked_view", "latent_recon", "layernorm_recon"}:
        raise ValueError("unknown adaptation mode")
    with frozen_predictor(model, head, layernorm=mode == "layernorm_recon") as selected:
        with torch.no_grad():
            baseline = _mean(head, model.encoder(values))
        if steps == 0:
            return baseline.detach(), {"steps": 0, "scored_features": 0}
        hidden, scored = masked_object(
            values,
            visible,
            xp_columns,
            source_id,
            seed=seed,
            probability=mask_probability,
        )
        with torch.no_grad():
            z0 = model.encoder(hidden)
        if mode == "masked_view":
            return _mean(head, z0).detach(), {
                "steps": 0,
                "scored_features": int(scored.sum()),
            }
        latent = z0.detach().clone().requires_grad_(mode == "latent_recon")
        initial = [p.detach().clone() for p in selected]
        optimizer = torch.optim.Adam(
            [latent] if mode == "latent_recon" else selected, lr=lr
        )
        last_loss = None
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            z = latent if mode == "latent_recon" else model.encoder(hidden)
            reconstruction = model.reconstruction_layer(model.decoder(z))
            if getattr(model, "heteroscedastic", False):
                reconstruction = reconstruction.chunk(2, dim=-1)[0]
            score = scored[:, : reconstruction.shape[1]]
            if not score.any():
                raise ValueError("no observed held-out reconstruction targets")
            loss = (
                (reconstruction - values[:, : reconstruction.shape[1]])
                .abs()[score]
                .mean()
            )
            if mode == "latent_recon":
                loss = loss + anchor * (latent - z0).square().mean()
                if consistency:
                    loss = (
                        loss
                        + consistency * (_mean(head, latent) - baseline).square().mean()
                    )
            else:
                drift = sum(
                    (p - v).square().mean()
                    for p, v in zip(selected, initial, strict=True)
                ) / len(selected)
                loss = loss + anchor * drift
                if consistency:
                    loss = (
                        loss
                        + consistency
                        * (_mean(head, model.encoder(values)) - baseline)
                        .square()
                        .mean()
                    )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite test-time objective")
            loss.backward()
            updates = [latent] if mode == "latent_recon" else selected
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in updates
            ):
                raise FloatingPointError("nonfinite test-time gradient")
            optimizer.step()
            last_loss = float(loss.detach())
        with torch.no_grad():
            point = _mean(
                head, latent if mode == "latent_recon" else model.encoder(values)
            )
        return point.detach(), dict(
            steps=steps, scored_features=int(scored.sum()), final_objective=last_loss
        )


@torch.no_grad()
def average_predictions(
    model,
    head,
    values,
    visible,
    sigma_xp,
    source_ids,
    xp_columns,
    *,
    views=8,
    noise_scale=0.0,
    mask_probability=0.0,
    seed=1042,
):
    if views < 1:
        raise ValueError("at least one view required")
    with frozen_predictor(model, head):
        baseline = _mean(head, model.encoder(values))
        if noise_scale == 0 and mask_probability == 0:
            return baseline.detach()
        points = [baseline]
        for view in range(views - 1):
            corrupted, _ = augmented_view(
                values,
                visible,
                sigma_xp,
                source_ids,
                xp_columns,
                seed=seed,
                view=view,
                noise_scale=noise_scale,
                mask_probability=mask_probability,
            )
            points.append(_mean(head, model.encoder(corrupted)))
        return torch.stack(points).double().mean(0).to(baseline.dtype)
