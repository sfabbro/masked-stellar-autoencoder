"""Sparse six-target normalized heads and explicitly adapted regularizers.

No absent label is imputed into an objective. The NSF baseline is a product of
six conditional scalar densities, not an empirically identified six-D copula.
Age/distance use log coordinates; native density includes their Jacobian.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from masked_stellar_autoencoder.training.multilabel_experiments import (
    MultiLabelHead,
    qmc_normal_nodes,
)

NAMES = ("teff", "logg", "fe_h", "alpha", "age", "distance")


def model_coordinates(value, observed):
    value, observed = np.asarray(value), np.asarray(observed, dtype=bool)
    if value.shape != observed.shape or value.shape[-1] != 6:
        raise ValueError("Six aligned targets/masks required")
    mask = observed & np.isfinite(value)
    result = np.where(mask, value, np.nan).astype("f8")
    for column in (4, 5):
        if np.any(mask[:, column] & (value[:, column] <= 0)):
            raise ValueError("Observed age/distance must be positive")
        result[:, column] = np.where(
            mask[:, column],
            np.log(np.where(mask[:, column], value[:, column], 1)),
            np.nan,
        )
    return result, mask


def normalization(value, observed, fit):
    coordinate, mask = model_coordinates(value, observed)
    center, scale = [], []
    for column in range(6):
        values = coordinate[np.asarray(fit) & mask[:, column], column]
        if len(values) < 16 or not np.isfinite(values).all() or values.std() <= 0:
            raise ValueError(
                f"Insufficient nondegenerate FIT labels for {NAMES[column]}"
            )
        center.append(values.mean())
        scale.append(values.std())
    return np.array(center), np.array(scale)


def standardized_arrays(value, observed, error, error_observed, center, scale):
    coordinate, mask = model_coordinates(value, observed)
    standardized = np.where(mask, (coordinate - center) / scale, 0)
    known = mask & np.asarray(error_observed, dtype=bool)
    errors = np.asarray(error, dtype="f8")
    if np.any(known & (~np.isfinite(errors) | (errors <= 0))):
        raise ValueError("Known measurement errors must be finite and positive")
    # Posterior ages/distances are not independent measurement-error likelihoods.
    if known[:, 4:].any():
        raise ValueError("Age/distance posterior widths are not measurement errors")
    sigma = np.where(known, np.where(np.isfinite(errors), errors, 0) / scale, 0)
    jacobian = np.broadcast_to(np.log(scale), value.shape).copy()
    for column in (4, 5):
        jacobian[:, column] += np.log(np.where(mask[:, column], value[:, column], 1))
    return standardized, sigma, mask, jacobian


class SparseSixHead(nn.Module):
    """Exact sparse diagonal Gaussian/MDN marginals or factorized scalar NSFs."""

    def __init__(
        self, kind="gaussian", latent_dim=256, *, noise_aware=True, mixed_weight=0.0
    ):
        super().__init__()
        if kind not in {"gaussian", "mdn", "factorized_nsf"}:
            raise ValueError("Unsupported six-target distribution")
        if mixed_weight and kind != "factorized_nsf":
            raise ValueError("Contrastive objective requires scalar NSF components")
        self.kind, self.noise_aware, self.mixed_weight = kind, noise_aware, mixed_weight
        if kind == "factorized_nsf":
            self.heads = nn.ModuleList(
                [
                    MultiLabelHead(
                        "flow", target_dim=1, latent_dim=latent_dim, flow_type="nsf"
                    )
                    for _ in range(6)
                ]
            )
            for head in self.heads:
                head.point_nodes = qmc_normal_nodes(256, dimensions=1)
        else:
            self.head = MultiLabelHead(kind, target_dim=6, latent_dim=latent_dim)

    def row_log_prob(self, latent, target, sigma, mask, *, observed=True):
        if (
            target.shape != sigma.shape
            or target.shape != mask.shape
            or target.shape != (len(latent), 6)
        ):
            raise ValueError("Six-target tensor layout mismatch")
        if (
            not torch.isfinite(target).all()
            or not torch.isfinite(sigma).all()
            or (sigma < 0).any()
        ):
            raise ValueError(
                "Safe standardized targets and nonnegative errors required"
            )
        error = sigma if self.noise_aware and observed else torch.zeros_like(sigma)
        if self.kind == "gaussian":
            mean, lv = self.head(latent).chunk(2, -1)
            variance = lv.exp() + error.square()
            per_label = -0.5 * (
                (target - mean).square() / variance
                + variance.log()
                + math.log(2 * math.pi)
            )
            return per_label.masked_fill(~mask, 0).sum(-1)
        if self.kind == "mdn":
            weight, means, log_stds = self.head._mixture_parameters(self.head(latent))
            variance = (2 * log_stds).exp() + error[:, None].square()
            component = (
                (
                    -0.5
                    * (
                        (target[:, None] - means).square() / variance
                        + variance.log()
                        + math.log(2 * math.pi)
                    )
                )
                .masked_fill(~mask[:, None], 0)
                .sum(-1)
            )
            return torch.logsumexp(weight + component, -1)
        result = target.new_zeros(len(target))
        for column, head in enumerate(self.heads):
            keep = mask[:, column]
            if keep.any():
                density = head.log_prob(
                    latent[keep],
                    target[keep, column : column + 1],
                    error[keep, column : column + 1]
                    if self.noise_aware and observed
                    else None,
                )
                result = result.index_add(0, keep.nonzero().squeeze(1), density)
        return result

    def criterion(self, latent, target, sigma, mask, source_id):
        count = mask.sum()
        if not count:
            return latent.sum() * 0
        total = -self.row_log_prob(latent, target, sigma, mask).sum()
        if self.mixed_weight:
            for column, head in enumerate(self.heads):
                index = mask[:, column].nonzero().squeeze(1)
                if len(index) < 2:
                    continue
                if len(torch.unique(source_id[index])) != len(index):
                    raise ValueError("Source-distinct contrastive contexts required")
                context = latent[index]
                y, error = (
                    target[index, column : column + 1],
                    sigma[index, column : column + 1],
                )
                plus = head.log_prob(context, y, error if self.noise_aware else None)
                minus = head.log_prob(
                    context.roll(1, 0), y, error if self.noise_aware else None
                )
                # SAME error convolution for positive and negative contexts.
                total = (
                    total
                    + self.mixed_weight * (torch.logaddexp(plus, minus) - plus).sum()
                )
        return total / count

    def moments(self, latent):
        if self.kind != "factorized_nsf":
            return self.head.moments(latent)
        values = [head.moments(latent) for head in self.heads]
        return torch.cat([v[0] for v in values], -1), torch.cat(
            [v[1] for v in values], -1
        )

    def native_moments(self, latent, center, scale):
        if self.kind == "factorized_nsf":
            samples = []
            for head in self.heads:
                distribution = head.distribution(latent)
                blocks = [
                    distribution.transform.inv(
                        node[:, None, :].expand(-1, len(latent), -1)
                    )
                    for node in head.point_nodes.to(latent).split(16)
                ]
                samples.append(torch.cat(blocks, 0))
            draws = torch.cat(samples, -1)
            draws = center + scale * draws
            draws[..., 4:] = draws[..., 4:].exp()
            mean = draws.mean(0)
            return mean, (draws - mean).square().mean(0)
        if self.kind == "gaussian":
            mean, lv = self.head(latent).chunk(2, -1)
            variance = lv.exp() * scale.square()
            mean = center + scale * mean
            native_mean, native_variance = mean.clone(), variance.clone()
            native_mean[:, 4:] = (mean[:, 4:] + variance[:, 4:] / 2).exp()
            native_variance[:, 4:] = (
                variance[:, 4:].expm1() * (2 * mean[:, 4:] + variance[:, 4:]).exp()
            )
            return native_mean, native_variance
        weights, means, log_std = self.head._mixture_parameters(self.head(latent))
        means = center + scale * means
        variance = (2 * log_std).exp() * scale.square()
        first, component_variance = means.clone(), variance.clone()
        first[..., 4:] = (means[..., 4:] + variance[..., 4:] / 2).exp()
        component_variance[..., 4:] = (
            variance[..., 4:].expm1() * (2 * means[..., 4:] + variance[..., 4:]).exp()
        )
        mean = (weights.exp()[..., None] * first).sum(1)
        return mean, (
            weights.exp()[..., None]
            * (component_variance + (first - mean[:, None]).square())
        ).sum(1)


def prediction_order_loss(mean, target, mask, sigma, label_source):
    """Per-label scalar ordering, within source scale only, uncertain ties removed.

    One cyclic source-distinct pair per row. Missing/error-ambiguous pairs carry
    no constraint. This is a prediction-ranking regularizer, not RankSim.
    """
    losses = []
    for column in range(6):
        for group in torch.unique(label_source[:, column]):
            rows = (
                (mask[:, column] & (label_source[:, column] == group))
                .nonzero()
                .squeeze(1)
            )
            if len(rows) < 2:
                continue
            other = rows.roll(1)
            delta = target[rows, column] - target[other, column]
            threshold = torch.maximum(
                delta.new_full(delta.shape, 0.2),
                2
                * torch.sqrt(
                    sigma[rows, column].square() + sigma[other, column].square()
                ),
            )
            keep = delta.abs() > threshold
            if keep.any():
                margin = delta.sign() * (mean[rows, column] - mean[other, column])
                losses.append(F.softplus(-margin[keep] / 0.2).mean())
    return torch.stack(losses).mean() if losses else mean.sum() * 0


class AdaptedTriLoss(nn.Module):
    """FIT-standardized multi-label adaptation of Chang's binned tri-loss.

    Width-weighted midpoint CDF expression is the paper surrogate, NOT exact
    continuous CRPS. Separate label projections avoid demanding identical six
    feature orderings. Two independent masked encoder views are mandatory.
    """

    def __init__(self, latent_dim, edges):
        super().__init__()
        from torchregress.losses import RankNContrastLoss

        self.rnc = RankNContrastLoss(temperature=2.0)
        self.classifiers = nn.ModuleList()
        self.projections = nn.ModuleList()
        for j, edge in enumerate(edges):
            edge = torch.as_tensor(edge, dtype=torch.float32)
            if len(edge) < 3 or not (edge.diff() > 0).all():
                raise ValueError("At least two positive FIT-only bins required")
            self.register_buffer(f"edges_{j}", edge)
            self.classifiers.append(nn.Linear(latent_dim, len(edge) - 1))
            self.projections.append(nn.Linear(latent_dim, 32))

    @staticmethod
    def binned_terms(logits, value, edges):
        label = torch.bucketize(value.contiguous(), edges[1:-1]).clamp(
            0, len(edges) - 2
        )
        ce = F.cross_entropy(logits, label)
        cdf = logits.softmax(-1).cumsum(-1)
        indicator = ((edges[:-1] + edges[1:]) / 2 >= value[:, None]).to(logits)
        surrogate = ((cdf - indicator).square() * edges.diff()).sum(-1).mean()
        return ce, surrogate

    def forward(self, view1, view2, target, mask, label_source, include_rnc=True):
        losses = []
        for column in range(6):
            for group in torch.unique(label_source[:, column]):
                rows = (
                    (mask[:, column] & (label_source[:, column] == group))
                    .nonzero()
                    .squeeze(1)
                )
                if len(rows) < 2:
                    continue
                views = torch.cat([view1[rows], view2[rows]], 0)
                labels = target[rows, column].repeat(2)
                ce, cdf = self.binned_terms(
                    self.classifiers[column](views),
                    labels,
                    getattr(self, f"edges_{column}"),
                )
                term = 0.5 * ce + 0.5 * cdf
                if include_rnc:
                    term = term + 0.1 * self.rnc(
                        self.projections[column](views), labels
                    )
                losses.append(term)
        return torch.stack(losses).mean() if losses else view1.sum() * 0


def independent_masked_views(x, visible, xp, generator, fraction=0.15):
    """Two masked views with an augmentation RNG separate from reconstruction."""
    if x.shape != visible.shape or not 0 < fraction < 1:
        raise ValueError("Aligned actual input/visible arrays required")
    allowed = torch.zeros(x.shape[-1], device=x.device, dtype=torch.bool)
    allowed[xp] = True
    result = []
    for _ in range(2):
        hide = (
            (torch.rand(x.shape, device=x.device, generator=generator) < fraction)
            & visible
            & allowed
        )
        result.append(x.masked_fill(hide, -9999))
    return result
