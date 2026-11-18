"""Inference for anchored (Student-t head) checkpoints.

One path for training-time checks, evaluation and prediction: inputs are the
clean scaled features (NaN = missing), with the checkpoint's dropped input
features hidden from the encoder, optional head context, and optional
conditioning of the joint Student-t on a measured label (Gaia parallax).
"""

from __future__ import annotations

import numpy as np
import torch

from .anchored_survey_map import condition_on_observation


def encoder_input(xb: torch.Tensor, drop_idx=()) -> torch.Tensor:
    """Encoder tensor: dropped features hidden, NaN as the -9999 token."""
    if len(drop_idx):
        xb = xb.clone()
        xb[:, list(drop_idx)] = float("nan")
    return torch.nan_to_num(xb, nan=-9999.0)


@torch.no_grad()
def anchored_predict(
    model, head, X, *, device, E=None, context=None, drop_idx=(), condition=None,
    Y=None, eY=None, batch: int = 4096,
):  # fmt: skip
    """Locations and marginal scales (N, P) in scaled label units.

    ``E``: scaled feature errors (needed by a context that uses errors);
    ``Y``/``eY``: scaled labels and errors (a context with label inputs).
    ``condition``: (label index j, z (N,), var (N,)) in scaled label units;
    rows with a finite z are conditioned on it (``condition_on_observation``).
    """
    model.eval()
    head.eval()
    if context is not None:
        context.eval()
    X = np.asarray(X, dtype=np.float32)
    mus, sds = [], []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i : i + batch], device=device)
        z = model.encoder(encoder_input(xb, drop_idx))
        ctx = None
        if context is not None:

            def t(a):
                return (
                    None
                    if a is None
                    else torch.as_tensor(
                        np.asarray(a[i : i + batch], dtype=np.float32), device=device
                    )
                )

            ctx, _ = context(xb, t(E), t(Y), t(eY))
        mu, L = head.params(z, ctx)
        if condition is not None:
            j, zc, vc = condition
            sl = slice(i, i + batch)
            zt = torch.as_tensor(zc[sl], device=device)
            vt = torch.as_tensor(vc[sl], device=device)
            mu, sd = condition_on_observation(mu, L, j, zt, vt, head.nu)
        else:
            sd = L.square().sum(-1).sqrt()
        mus.append(mu.cpu())
        sds.append(sd.cpu())
    return torch.cat(mus).numpy(), torch.cat(sds).numpy()
