"""Functional 1/2/3-D NSF kernel from head state tensors, without flow-library calls.

Fixed contract: 3 transforms, 16 bins, bound 8, slope 1e-3, ReLU masked
MLPs with sixteen context coordinates. Low-dimensional copy: the old 3-D
verifier/kernel bytes and all their completed caches remain unchanged. Forward means target -> base normal. Supports clean
NormalizingFlowLoss and ContrastiveFlowLoss state layout loss_fn.flow.*.
No encoder, moment quadrature, model reconstruction, or checkpoint loading.
"""

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F

BINS = 16
BOUND = 8.0
SLOPE = 1e-3
PREFIX = "loss_fn.flow."


def parameters(x, context, state: Mapping, layer: int):
    """Masked autoregressive MLP; coordinate-major [16,16,15] outputs."""
    if x.shape[-1] not in (1, 2, 3) or context.shape[-1] != 16:
        raise ValueError("Expected one to three targets and 16 context coordinates")
    context = context.expand(*x.shape[:-1], 16)
    first = state[f"{PREFIX}transform.transforms.{layer}.hyper.0.weight"]
    h = context if first.shape[-1] == 16 else torch.cat((x, context), dim=-1)
    prefix = f"{PREFIX}transform.transforms.{layer}.hyper."
    for index in (0, 2, 4):
        key = prefix + str(index)
        h = F.linear(
            h,
            state[key + ".weight"]
            * state.get(key + ".mask", torch.ones_like(state[key + ".weight"])),
            state[key + ".bias"],
        )
        if index != 4:
            h = F.relu(h)
    h = h.reshape(*h.shape[:-1], x.shape[-1], 3 * BINS - 1)
    return h[..., :BINS], h[..., BINS : 2 * BINS], h[..., 2 * BINS :]


def knots(raw_width, raw_height, raw_derivative):
    """Exact local Zuko conventions: bounded logits, softmax, exp derivatives.

    There is no additive minimum-bin-width clamp in this implementation.
    Slope bounds raw logits before softmax; endpoint log derivatives are zero.
    """
    log_floor = math.log(SLOPE)
    rw = raw_width / (1 + torch.abs(2 * raw_width / log_floor))
    rh = raw_height / (1 + torch.abs(2 * raw_height / log_floor))
    rd = raw_derivative / (1 + torch.abs(raw_derivative / log_floor))
    horizontal = BOUND * (2 * F.pad(rw.softmax(-1), (1, 0)).cumsum(-1) - 1)
    vertical = BOUND * (2 * F.pad(rh.softmax(-1), (1, 0)).cumsum(-1) - 1)
    derivative = F.pad(rd, (1, 1)).exp()
    return horizontal, vertical, derivative


def spline(value, raw_width, raw_height, raw_derivative, *, inverse=False):
    """Elementwise rational-quadratic map and directional log derivative."""
    horizontal, vertical, derivative = knots(raw_width, raw_height, raw_derivative)
    edges = vertical if inverse else horizontal
    # Strict '<' chooses the bin to the left of an exact internal knot.
    index = (edges < value.unsqueeze(-1)).sum(-1) - 1
    inside = (index >= 0) & (index < BINS)
    index = index.remainder(BINS)

    def gather(values, offset=0):
        return values.gather(-1, (index + offset).unsqueeze(-1)).squeeze(-1)

    x0, x1 = gather(horizontal), gather(horizontal, 1)
    y0, y1 = gather(vertical), gather(vertical, 1)
    d0, d1 = gather(derivative), gather(derivative, 1)
    width, height = x1 - x0, y1 - y0
    secant = height / width
    bend = d0 + d1 - 2 * secant
    if inverse:
        delta = inside * (value - y0)
        a = height * (secant - d0) + delta * bend
        b = height * d0 - delta * bend
        c = -secant * delta
        theta = 2 * c / (-b - torch.sqrt(b.square() - 4 * a * c))
        result = x0 + theta * width
    else:
        theta = inside * (value - x0) / width
        numerator = secant * theta.square() + d0 * theta * (1 - theta)
        result = y0 + height * numerator / (secant + bend * theta * (1 - theta))
    denominator = secant + bend * theta * (1 - theta)
    jacobian = (
        secant.square()
        * (
            2 * secant * theta * (1 - theta)
            + d0 * (1 - theta).square()
            + d1 * theta.square()
        )
        / denominator.square()
    )
    logdet = inside * jacobian.log()
    if inverse:
        logdet = -logdet
    return torch.where(inside, result, value), logdet


def transform(x, context, state):
    """Target -> base, composing layers 0,1,2 with exact triangular logdet."""
    logdet = x.new_zeros(x.shape[:-1])
    for layer in (0, 1, 2):
        x, ld = spline(x, *parameters(x, context, state, layer))
        logdet = logdet + ld.sum(-1)
    return x, logdet


def inverse_with_logdet(z, context, state):
    """Base -> target, reverse layers then solve stored coordinate ordering."""
    logdet = z.new_zeros(z.shape[:-1])
    for layer in (2, 1, 0):
        x = torch.zeros_like(z)
        order = state.get(
            f"{PREFIX}transform.transforms.{layer}.order",
            torch.arange(z.shape[-1], device=z.device),
        )
        if sorted(order.tolist()) != list(range(z.shape[-1])):
            raise ValueError("Kernel requires fully autoregressive coordinate order")
        for coordinate in order.argsort().tolist():
            candidate, ld = spline(
                z, *parameters(x, context, state, layer), inverse=True
            )
            x = x.clone()
            x[..., coordinate] = candidate[..., coordinate]
            logdet = logdet + ld[..., coordinate]
        z = x
    return z, logdet


def density(target, context, state):
    """Conditional normal-base log density with forward change of variables."""
    z, logdet = transform(target, context, state)
    loc, scale = state[PREFIX + "base.loc"], state[PREFIX + "base.scale"]
    normal = (
        -0.5 * ((z - loc) / scale).square() - scale.log() - 0.5 * math.log(2 * math.pi)
    )
    return normal.sum(-1) + logdet


def inverse(z, context, state):
    """Base -> target API matching the independent MAF verifier."""
    return inverse_with_logdet(z, context, state)[0]
