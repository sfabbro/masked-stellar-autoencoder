"""Flux scale, per-order asinh, and the diagonal Jacobian on BP and RP covariances.

Order of transforms, per coefficient j of one star:

    c1 = c / F,  F = 10**(-0.4*(G - G_ref))
    u  = (c1 - m_j) / s_j
    z  = asinh(u)

F is one scalar per star. m_j and s_j are the train-reservoir median and IQR of
that order. The Jacobian is diagonal:

    dz/dc = (1/F) * (1/s_j) / sqrt(1 + u^2)

and Sigma_z = J Sigma J^T, separately for BP and for RP. A file that only has
coefficient errors uses a diagonal Sigma.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from masked_stellar_autoencoder.pipeline.batch import StellarBatch
from masked_stellar_autoencoder.pipeline.registry import ColumnRegistry

FLUX_TAG = "pogson_div_v1"
# ponytail: prefix of the training iterator, cap ``max_rows`` (~1e6). Shuffle
# partitions before iterating. A second full pass replaces this if quantiles move.


def pogson_factor(g_mag: np.ndarray, g_ref: float) -> np.ndarray:
    """10**(-0.4*(G-G_ref)). A missing G stays NaN; it is not the reference flux."""
    g = np.asarray(g_mag, dtype=np.float64)
    return np.power(10.0, -0.4 * (g - g_ref))


def reservoir_matrix(
    reader: Iterable[StellarBatch], group: str, max_rows: int
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    n = 0
    for batch in reader:
        need = max_rows - n
        if need <= 0:
            break
        rows = np.asarray(batch.groups[group].values[:need], dtype=np.float64)
        chunks.append(rows)
        n += len(rows)
    if not chunks:
        raise ValueError(f"no rows in group {group}")
    return np.concatenate(chunks, axis=0)


@dataclass(frozen=True)
class OrderScaler:
    median: np.ndarray
    iqr: np.ndarray
    g_ref: float
    flux_tag: str = FLUX_TAG

    def __post_init__(self) -> None:
        if self.median.shape != self.iqr.shape:
            raise ValueError("median and iqr must share a shape")
        if np.any(self.iqr <= 0):
            raise ValueError("iqr must be positive")


def fit_order_scaler(flux_scaled: np.ndarray, g_ref: float) -> OrderScaler:
    data = np.asarray(flux_scaled, dtype=np.float64)
    median = np.nanmedian(data, axis=0)
    q1 = np.nanpercentile(data, 25, axis=0)
    q3 = np.nanpercentile(data, 75, axis=0)
    iqr = q3 - q1
    iqr = np.where(~np.isfinite(iqr) | (iqr < 1e-8), 1.0, iqr)
    median = np.where(np.isfinite(median), median, 0.0)
    return OrderScaler(median=median, iqr=iqr, g_ref=g_ref)


def diagonal_covariance(errors: np.ndarray) -> np.ndarray:
    """Sigma = diag(sigma^2). A missing error stays NaN, not a zero variance."""
    sigma = np.asarray(errors, dtype=np.float64)
    return np.eye(sigma.shape[1]) * np.square(sigma)[:, :, None]


def transform_coefficients(
    values: np.ndarray,
    cov: np.ndarray | None,
    g_mag: np.ndarray,
    scaler: OrderScaler,
    *,
    flux_state: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return z, Sigma_z, and the diagonal Jacobian dz/dc.

    ``flux_state='already_scaled'`` skips the Pogson factor so a file that was
    divided at build time is not divided again. The checkpoint stores the tag.
    """
    if flux_state == "raw":
        factor = pogson_factor(g_mag, scaler.g_ref)
    elif flux_state == "already_scaled":
        factor = np.ones(len(np.asarray(g_mag)), dtype=np.float64)
    else:
        raise ValueError(f"flux_state must be raw or already_scaled, got {flux_state}")
    scaled = np.asarray(values, dtype=np.float64) / factor[:, None]
    u = (scaled - scaler.median) / scaler.iqr
    z = np.arcsinh(u)
    jacobian = (1.0 / factor[:, None]) * (1.0 / scaler.iqr) / np.sqrt(1.0 + u * u)
    if cov is None:
        return z, None, jacobian
    sigma = np.asarray(cov, dtype=np.float64)
    cov_z = jacobian[:, :, None] * sigma * jacobian[:, None, :]
    return z, cov_z, jacobian


@dataclass(frozen=True)
class VectorScaler:
    median: np.ndarray
    iqr: np.ndarray
    kind: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.median.shape != self.iqr.shape
            or len(self.kind) != self.median.shape[0]
        ):
            raise ValueError("median, iqr, and kind must share a length")


def _nonlinear(values: np.ndarray, kind: tuple[str, ...]) -> np.ndarray:
    out = np.array(values, dtype=np.float64, copy=True)
    for j, name in enumerate(kind):
        col = out[:, j]
        if name == "log10":
            # A non-positive temperature is missing, not log10(1e-3).
            logged = np.full(col.shape, np.nan, dtype=np.float64)
            positive = col > 0
            logged[positive] = np.log10(col[positive])
            col = logged
        elif name == "asinh":
            col = np.arcsinh(col)
        elif name != "robust":
            raise ValueError(f"unknown label transform {name}")
        out[:, j] = col
    return out


def fit_vector_scaler(values: np.ndarray, kind: tuple[str, ...]) -> VectorScaler:
    data = _nonlinear(values, kind)
    median = np.nanmedian(data, axis=0)
    q1 = np.nanpercentile(data, 25, axis=0)
    q3 = np.nanpercentile(data, 75, axis=0)
    iqr = np.where((q3 - q1) < 1e-8, 1.0, q3 - q1)
    median = np.where(np.isfinite(median), median, 0.0)
    iqr = np.where(np.isfinite(iqr), iqr, 1.0)
    return VectorScaler(median=median, iqr=iqr, kind=kind)


def transform_vector(
    values: np.ndarray, missing: np.ndarray, scaler: VectorScaler
) -> np.ndarray:
    z = (_nonlinear(values, scaler.kind) - scaler.median) / scaler.iqr
    return np.where(missing, np.nan, z)


def inverse_vector(scaled: np.ndarray, scaler: VectorScaler) -> np.ndarray:
    x = np.asarray(scaled, dtype=np.float64) * scaler.iqr + scaler.median
    out = np.empty_like(x)
    for j, name in enumerate(scaler.kind):
        col = x[:, j]
        if name == "log10":
            col = np.power(10.0, col)
        elif name == "asinh":
            col = np.sinh(col)
        out[:, j] = col
    return out


def clipped_snr(
    values: np.ndarray, errors: np.ndarray, *, cap: float = 10.0
) -> np.ndarray:
    """Parallax and proper motion as clipped signal-to-noise. A non-positive error is missing."""
    sigma = np.asarray(errors, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(sigma) & (sigma > 0) & np.isfinite(values)
    snr = np.full(values.shape, np.nan, dtype=np.float64)
    snr[ok] = np.clip(values[ok] / sigma[ok], -cap, cap)
    return snr


def fit_batch_scalers(
    batch: StellarBatch, registry: ColumnRegistry
) -> tuple[OrderScaler, OrderScaler, VectorScaler, VectorScaler]:
    """Fit on one training batch. Callers pass the train reservoir, not the full sky."""
    g_ref = registry.g_ref
    bp = _flux_for_fit(batch, "xp_bp", g_ref)
    rp = _flux_for_fit(batch, "xp_rp", g_ref)
    photo = batch.groups["photometry"]
    labels = batch.groups["labels"]
    photo_kind = tuple(col.transform for col in registry.group("photometry"))
    label_kind = tuple(col.transform for col in registry.group("labels"))
    return (
        fit_order_scaler(bp, g_ref),
        fit_order_scaler(rp, g_ref),
        fit_vector_scaler(photo.values, photo_kind),
        fit_vector_scaler(labels.values, label_kind),
    )


def _flux_for_fit(batch: StellarBatch, group: str, g_ref: float) -> np.ndarray:
    values = batch.groups[group].values
    if batch.flux_state == "already_scaled":
        return values
    factor = pogson_factor(batch.g_mag, g_ref)
    return values / factor[:, None]
