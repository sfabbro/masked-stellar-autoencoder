"""Full-pool non-Gaussian consensus with a retained masked reconstruction path."""

from __future__ import annotations

import numpy as np
import torch

from .multisurvey_data import SparseSurveyDataset
from .multisurvey_flow import whiten_observations
from .multisurvey_model import pack_multisurvey_observations


class IndexedRows:
    """Read-only global-row lookup; rejects access outside its declared role."""

    def __init__(self, rows, values):
        self.rows = np.asarray(rows, dtype="i8").copy()
        self.values = np.asarray(values).copy()
        if len(self.rows) != len(self.values) or (np.diff(self.rows) <= 0).any():
            raise ValueError("Cached rows must be unique, sorted and aligned")
        self.rows.flags.writeable = self.values.flags.writeable = False

    def __getitem__(self, rows):
        rows = np.asarray(rows)
        pos = np.searchsorted(self.rows, rows)
        if (pos >= len(self.rows)).any() or not np.array_equal(self.rows[pos], rows):
            raise ValueError("Requested row is outside cached role")
        return self.values[pos].copy()


def read_rows(source, rows, chunk=65536):
    """Bounded contiguous HDF5 reads; retain only explicitly requested rows."""
    rows = np.asarray(rows, dtype="i8")
    result = np.empty((len(rows), *source.shape[1:]), dtype=source.dtype)
    if len(rows):
        for start in range(int(rows[0]) // chunk * chunk, int(rows[-1]) + 1, chunk):
            lo, hi = np.searchsorted(rows, [start, start + chunk])
            if hi > lo:
                result[lo:hi] = source[start : min(start + chunk, len(source))][
                    rows[lo:hi] - start
                ]
    return result


class RoleMemoryDataset(SparseSurveyDataset):
    """Same source reader/records, backed by role-restricted immutable RAM arrays.

    The owner checks original attachment bindings before/after construction and
    every optimizer batch. Native provenance and repeat resolution are unchanged.
    """

    def __init__(self, source):
        source._check_source_binding()
        self.__dict__.update(source.__dict__)
        self._file = None
        h5 = source._handle()
        self._indices = {}
        for name, (stars, native) in source._indices.items():
            keep = np.isin(stars, self.star_rows)
            self._indices[name] = (stars[keep], native[keep])
        self._memory = {
            name: IndexedRows(self.star_rows, read_rows(h5[name], self.star_rows))
            for name in ("input", "role")
        }
        self._memory["native_observations"] = {}
        for name in self.names:
            native = np.unique(self._indices[name][1])
            group = h5["native_observations"][name]
            self._memory["native_observations"][name] = {
                key: IndexedRows(native, read_rows(group[key], native))
                for key in ("value", "error", "observed", "source_row")
            }
        source._check_source_binding()

    def _handle(self):
        return self._memory


def mixture_terms(
    model,
    mixture,
    relative,
    batch,
    floors,
    center,
    scale,
    *,
    mask=None,
    reconstruction_width=None,
):
    """Analytic mixture likelihood under the shared affine survey observation model.

    One intrinsic 3D mixture per star; all available survey measurements share
    that same latent. Encoder gradients flow through reference and mixture paths.
    """
    device = next(model.parameters()).device
    inputs = batch["input"].to(device)
    latent = model.mae.encoder(inputs)
    mean, lower = model.head(latent.double())
    context = (latent.double() - center) / scale
    packed = pack_multisurvey_observations(
        mean,
        batch,
        relative,
        floors,
        expected_registry_sha256=batch["registry_sha256"],
        expected_source_registry=batch["source_registry"],
        assume_diagonal_catalog_covariance=True,
    )
    observation = whiten_observations(lower, packed)
    lp = mixture.observation_log_prob(context, observation)
    active = observation["count"] > 0
    result = dict(
        nll_sum=-lp[active].sum(),
        active=int(active.sum()),
        context=context,
        reference_mean=mean,
        reference_lower=lower,
        packed=packed,
        reconstruction_sum=lp.new_zeros(()),
        masked_count=0,
    )
    if mask is not None:
        mask = mask.to(device)
        available = batch["input_available"].to(device)
        if (
            mask.dtype != torch.bool
            or mask.shape != inputs.shape
            or (mask & ~available).any()
        ):
            raise ValueError("Reconstruction mask must select available input features")
        width = reconstruction_width
        if (
            not isinstance(width, int)
            or not 0 < width <= inputs.shape[1]
            or mask[:, width:].any()
        ):
            raise ValueError("Reconstruction mask exceeds the declared target prefix")
        prediction = model.reconstruct(inputs.masked_fill(mask, -9999))
        if isinstance(prediction, tuple):
            prediction = prediction[0]
        target = inputs[:, :width]
        hidden = mask[:, :width]
        if prediction.shape != target.shape:
            raise ValueError("Reconstruction dimensions differ")
        result["reconstruction_sum"] = (prediction[hidden] - target[hidden]).abs().sum()
        result["masked_count"] = int(hidden.sum())
    return result


def physical_components(mixture, context, mean, lower):
    """Complete common-reference mixture and its exact mean/covariance."""
    log_weights, mu, chol = mixture(context)
    means = mean[:, None] + torch.einsum("bij,bcj->bci", lower, mu)
    lowers = lower[:, None] @ chol
    weights = log_weights.exp()
    average = (weights[..., None] * means).sum(1)
    delta = means - average[:, None]
    covariance = (
        weights[..., None, None]
        * (lowers @ lowers.transpose(-1, -2) + delta[..., None] * delta[..., None, :])
    ).sum(1)
    return weights, means, lowers, average, covariance
