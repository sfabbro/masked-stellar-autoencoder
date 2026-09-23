"""StellarBatch: the only object the training loop sees."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import h5py
import numpy as np

from masked_stellar_autoencoder.pipeline.registry import ColumnRegistry


@dataclass
class FeatureGroup:
    values: np.ndarray
    missing: np.ndarray
    errors: np.ndarray | None = None
    covariance: np.ndarray | None = None


@dataclass
class StellarBatch:
    source_id: np.ndarray
    healpix: np.ndarray
    survey: np.ndarray
    g_mag: np.ndarray
    groups: dict[str, FeatureGroup]
    n_relevant_bp: np.ndarray | None = None
    n_relevant_rp: np.ndarray | None = None
    flux_state: str = "raw"

    def __post_init__(self) -> None:
        n = int(self.source_id.shape[0])
        if (
            self.healpix.shape[0] != n
            or self.g_mag.shape[0] != n
            or self.survey.shape[0] != n
        ):
            raise ValueError(
                "source_id, healpix, survey, and g_mag must share the row count"
            )
        if self.flux_state not in ("raw", "already_scaled"):
            raise ValueError(
                f"flux_state must be raw or already_scaled, got {self.flux_state}"
            )
        for name, group in self.groups.items():
            if group.values.shape[0] != n:
                raise ValueError(
                    f"{name} has {group.values.shape[0]} rows, batch has {n}"
                )
            if group.missing.shape != group.values.shape:
                raise ValueError(
                    f"{name} missing mask shape {group.missing.shape} != values"
                )
            if group.covariance is not None:
                d = group.values.shape[1]
                expected = (n, d, d)
                if group.covariance.shape != expected:
                    raise ValueError(
                        f"{name} covariance shape {group.covariance.shape} != {expected}"
                    )
            if group.errors is not None and group.errors.shape != group.values.shape:
                raise ValueError(f"{name} errors shape {group.errors.shape} != values")

    def __len__(self) -> int:
        return int(self.source_id.shape[0])


@dataclass
class MemoryReader:
    """Partition iterator. An HDF shard or a HATS partition is one yield."""

    batches: list[StellarBatch] = field(default_factory=list)

    def __iter__(self) -> Iterator[StellarBatch]:
        yield from self.batches


def _column(columns: dict[str, np.ndarray], name: str, n: int) -> np.ndarray | None:
    if name not in columns:
        return None
    array = np.asarray(columns[name], dtype=np.float64)
    if array.ndim != 1 or len(array) != n:
        raise ValueError(f"{name} has shape {array.shape}, expected ({n},)")
    return array


def assemble(
    columns: dict[str, np.ndarray],
    registry: ColumnRegistry,
    *,
    source_id: np.ndarray,
    healpix: np.ndarray,
    survey: np.ndarray,
    g_mag: np.ndarray | None = None,
    cov_bp: np.ndarray | None = None,
    cov_rp: np.ndarray | None = None,
    n_relevant_bp: np.ndarray | None = None,
    n_relevant_rp: np.ndarray | None = None,
    flux_state: str = "raw",
) -> StellarBatch:
    """Build a batch by column name. A missing survey stays missing; width is unchanged."""
    source_id = np.asarray(source_id)
    n = int(source_id.shape[0])
    if g_mag is None:
        raw_g = _column(columns, registry.g_column, n)
        g_mag = np.full(n, np.nan) if raw_g is None else raw_g
    groups: dict[str, FeatureGroup] = {}
    cov = {"xp_bp": cov_bp, "xp_rp": cov_rp}
    for group_name in ("xp_bp", "xp_rp", "photometry", "astrometry", "labels"):
        specs = registry.group(group_name)
        if not specs:
            continue
        width = len(specs)
        values = np.full((n, width), np.nan, dtype=np.float64)
        errors = np.full((n, width), np.nan, dtype=np.float64)
        any_error = False
        for j, spec in enumerate(specs):
            col = _column(columns, spec.name, n)
            if col is None:
                continue
            values[:, j] = col
            if spec.error is not None:
                err = _column(columns, spec.error, n)
                if err is not None:
                    any_error = True
                    errors[:, j] = err
        missing = ~np.isfinite(values)
        group_cov = cov.get(group_name)
        if group_cov is not None:
            group_cov = np.asarray(group_cov, dtype=np.float64)
            if group_cov.shape[-1] != width:
                group_cov = _pad_cov(group_cov, width)
        groups[group_name] = FeatureGroup(
            values=values,
            missing=missing,
            errors=errors if any_error else None,
            covariance=group_cov,
        )
    return StellarBatch(
        source_id=np.asarray(source_id),
        healpix=np.asarray(healpix),
        survey=np.asarray(survey).astype(str),
        g_mag=np.asarray(g_mag, dtype=np.float64),
        groups=groups,
        n_relevant_bp=None if n_relevant_bp is None else np.asarray(n_relevant_bp),
        n_relevant_rp=None if n_relevant_rp is None else np.asarray(n_relevant_rp),
        flux_state=flux_state,
    )


def _pad_cov(cov: np.ndarray, width: int) -> np.ndarray:
    n, d, _ = cov.shape
    if d > width:
        raise ValueError(f"covariance order {d} exceeds registry width {width}")
    if d == width:
        return cov
    out = np.zeros((n, width, width), dtype=np.float64)
    out[:, :d, :d] = cov
    return out


def _columns_from_node(node: h5py.Group | h5py.Dataset) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    if isinstance(node, h5py.Dataset) and node.dtype.names:
        for name in node.dtype.names:
            columns[name] = node[name][:]
        return columns
    if isinstance(node, h5py.Group):
        for name, child in node.items():
            if isinstance(child, h5py.Dataset):
                columns[name] = child[:]
        return columns
    raise ValueError("node is not a named table")


def _batch_from_columns(
    columns: dict[str, np.ndarray],
    registry: ColumnRegistry,
    *,
    survey: str,
    flux_state: str,
    where: str,
) -> StellarBatch:
    if "source_id" not in columns:
        raise ValueError(f"{where} has no source_id column")
    source_id = columns["source_id"]
    n = len(source_id)
    return assemble(
        columns,
        registry,
        source_id=source_id,
        healpix=np.zeros(n, dtype=np.int64),
        survey=np.full(n, survey),
        flux_state=flux_state,
    )


def read_named_table(
    path: str,
    registry: ColumnRegistry,
    key: str,
    *,
    survey: str = "gaia",
    flux_state: str = "raw",
) -> StellarBatch:
    """Read one named partition. The key is required so the first group is not special."""
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"{path} has no partition {key}")
        return _batch_from_columns(
            _columns_from_node(handle[key]),
            registry,
            survey=survey,
            flux_state=flux_state,
            where=f"{path}:{key}",
        )


def iter_partitions(
    path: str,
    registry: ColumnRegistry,
    *,
    survey: str = "gaia",
    flux_state: str = "raw",
) -> Iterator[StellarBatch]:
    """One StellarBatch per HDF key. The caller does not concatenate them."""
    with h5py.File(path, "r") as handle:
        for key in handle.keys():
            yield _batch_from_columns(
                _columns_from_node(handle[key]),
                registry,
                survey=survey,
                flux_state=flux_state,
                where=f"{path}:{key}",
            )
