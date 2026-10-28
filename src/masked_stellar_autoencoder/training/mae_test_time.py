"""Label-free masked-reconstruction test-time training with exact object reset."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn


def stateless_columns(visible, xp_columns, source_id, step, fraction, seed=1042):
    """Observed XP only; no target value/error enters the mask or objective."""
    valid = np.array([int(c) for c in xp_columns if bool(visible[0, c])])
    if len(valid) < 2 or not 0 < fraction < 1:
        raise ValueError("At least two observed XP and fraction in (0,1) required")
    key = hashlib.sha256(f"{int(source_id)}:{seed}:{step}:mae-ttt".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(key[:8], "little"))
    count = min(len(valid) - 1, max(1, int(round(len(valid) * fraction))))
    return np.sort(rng.choice(valid, count, replace=False))


@contextmanager
def object_adaptation_scope(model, head, scope):
    """Restore parameters, buffers, gradients, flags, modes and RNG on any exit."""
    params = list(model.parameters()) + list(head.parameters())
    flags = [p.requires_grad for p in params]
    grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
    modes = [(m, m.training) for root in (model, head) for m in root.modules()]
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    if scope == "encoder_decoder":
        selected = (
            list(model.encoder.parameters())
            + list(model.decoder.parameters())
            + list(model.reconstruction_layer.parameters())
        )
    elif scope == "layernorm":
        selected = [
            p
            for m in model.encoder.modules()
            if isinstance(m, nn.LayerNorm)
            for p in m.parameters(recurse=False)
        ]
    else:
        raise ValueError("Unknown test-time parameter scope")
    selected = list({id(p): p for p in selected}.values())
    if not selected:
        raise ValueError("No trainable test-time parameters selected")
    try:
        for p in params:
            p.requires_grad_(False)
        for p in selected:
            p.requires_grad_(True)
        model.eval()
        head.eval()
        yield selected
    finally:
        model.load_state_dict(state, strict=True)
        for p, flag, grad in zip(params, flags, grads, strict=True):
            p.requires_grad_(flag)
            p.grad = grad
        for m, mode in modes:
            m.training = mode
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def masked_mse(reconstruction, x, columns):
    return (reconstruction[:, columns] - x[:, columns]).square().mean()


def adapt_masked_mse(
    model,
    head,
    x,
    visible,
    xp_columns,
    source_id,
    *,
    scope="encoder_decoder",
    steps=20,
    lr=1e-4,
    momentum=0.9,
    mask_fraction=0.2,
    seed=1042,
):
    """SGD momentum TTT; fixed prediction head and no pseudo-target/label loss.

    This is a tabular analogue of MAE-based TTT, not the paper's image MAE or
    frozen linear-probe protocol. Each object starts from the same checkpoint,
    optimizer momentum zero, and fresh deterministic masks per update. A
    3-step run is an exact prefix of its matched 20-step mask sequence.
    """
    if x.shape[0] != 1 or x.shape != visible.shape or steps < 1:
        raise ValueError("One object/visible mask and positive steps required")
    with object_adaptation_scope(model, head, scope) as params:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum)
        trace = []
        for step in range(steps):
            chosen = stateless_columns(
                visible, xp_columns, source_id, step, mask_fraction, seed
            )
            hidden = x.detach().clone()
            hidden[:, chosen] = -9999.0
            optimizer.zero_grad(set_to_none=True)
            latent = model.encoder(hidden)
            reconstruction = model.reconstruction_layer(model.decoder(latent))
            if getattr(model, "heteroscedastic", False):
                reconstruction = reconstruction.chunk(2, -1)[0]
            loss = masked_mse(reconstruction, x, chosen)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite test-time masked MSE")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all() for p in params
            ):
                raise FloatingPointError("Nonfinite test-time gradient")
            optimizer.step()
            trace.append(
                {
                    "step": step + 1,
                    "heldout_columns": chosen.tolist(),
                    "masked_mse": float(loss.detach()),
                }
            )
        with torch.no_grad():
            mean, variance = head.moments(model.encoder(x))
            if (
                not torch.isfinite(mean).all()
                or not torch.isfinite(variance).all()
                or not (variance > 0).all()
            ):
                raise FloatingPointError("Invalid adapted predictive moments")
        return mean.detach().clone(), variance.detach().clone(), trace
