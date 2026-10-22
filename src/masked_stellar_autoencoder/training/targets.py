"""Frozen target registry: name, unit, transform, positivity and status.

One source of truth for the native target vector. Every target that may appear in a
config, a checkpoint signature or a predictions file is declared here; nothing may
carry a target outside this registry. Coordinates are looked up **by name**, never by
position, so reordering or adding a target cannot silently mis-invert a prediction
(the legacy `i == 5` / `i == 0` aliasing).

Model coordinate transform ``T`` (all targets use the same fit-only standardization
``u = (T(y) - c) / w`` afterwards):

===========  ==================================  ===============  ==================
name         native unit                          ``T(y)``         ``log|T'(y)|``
===========  ==================================  ===============  ==================
``teff``     K (kelvin)                           ``y``            ``0``
``logg``     dex (cgs)                            ``y``            ``0``
``fe_h``     dex                                  ``y``            ``0``
``distance`` kpc                                  ``log(y)``       ``-log y``
``age``      Gyr                                  ``log(y)``       ``-log y``
``alpha``    dex                                  ``y``            ``0``
===========  ==================================  ===============  ==================

The native-unit density therefore needs, per row and per **observed** coordinate a
subtraction of the offset ``log w_j - log|T'_j(y_j)|``::

    log p_Y(y|x) = log p_U(u|x) - sum_obs (log w_j - log|T'_j(y_j)|)
                 = log p_U(u|x) - sum_obs log w_j + sum_obs log|T'_j(y_j)|

(change of variables: ``p_Y = p_U * |T'| / w`` because ``u = (T(y) - c) / w``, checked
against an independent log-normal density in ``tests/test_targets.py``). For a log
transform ``log|T'(y)| = -log y``, so the offset is ``log w_j + log y_j`` and the
corrected native density is ``log p_U(u) - sum_obs log w_j - sum_obs log y_j``. A
single scalar Jacobian (the legacy ``sum(log(scale))``) is only valid for identity
transforms; it is too high by ``+log d`` / ``+log age`` on every row once
distance/age enter the vector.

Positivity is a declared property of the target (distance, age, Teff) and is
enforced exactly where the transform needs a positive argument, i.e. for the log
transforms. No ``asinh``/log-of-possibly-negative shortcut is admitted: negative or
zero native values for a log-transformed target raise instead of silently producing
``-inf``. ``positive`` on an identity target stays a declared property (it bounds the
support) and is not enforced on arbitrary synthetic scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

IDENTITY = "identity"
LOG_KPC = "log_kpc"
LOG_GYR = "log_gyr"
TRANSFORMS = (IDENTITY, LOG_KPC, LOG_GYR)

SYMMETRIC = "symmetric"
ASYMMETRIC_QUANTILES = "asymmetric_quantiles"
RAW_SIGNED_PARALLAX = "raw_signed_parallax"
ERROR_KINDS = (SYMMETRIC, ASYMMETRIC_QUANTILES, RAW_SIGNED_PARALLAX)

PHYSICAL = "physical"
PROXY = "proxy"
OBSERVABLE_KINDS = (PHYSICAL, PROXY)

TRAINABLE = "trainable"
INTERFACE_ONLY = "interface_only"
BLOCKED = "blocked"
STATUSES = (TRAINABLE, INTERFACE_ONLY, BLOCKED)


@dataclass(frozen=True)
class TargetSpec:
    """One frozen declaration of a native target."""

    name: str
    unit: str
    transform: str = IDENTITY
    positive: bool = False
    error_kind: str = SYMMETRIC
    source_tag: str | None = None
    observable_kind: str = PHYSICAL
    status: str = TRAINABLE
    description: str = ""

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("target name must be a non-empty string")
        if not isinstance(self.unit, str) or not self.unit:
            raise ValueError(f"{self.name}: unit must be a non-empty string")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"{self.name}: unsupported transform {self.transform!r}")
        if self.error_kind not in ERROR_KINDS:
            raise ValueError(f"{self.name}: unsupported error kind {self.error_kind!r}")
        if self.observable_kind not in OBSERVABLE_KINDS:
            raise ValueError(
                f"{self.name}: invalid observable kind {self.observable_kind!r}"
            )
        if self.status not in STATUSES:
            raise ValueError(f"{self.name}: invalid status {self.status!r}")
        if self.transform in (LOG_KPC, LOG_GYR) and not self.positive:
            raise ValueError(f"{self.name}: a log transform requires positivity")
        if not isinstance(self.positive, bool):
            raise ValueError(f"{self.name}: positivity must be a boolean")

    @property
    def log_transformed(self) -> bool:
        return self.transform in (LOG_KPC, LOG_GYR)

    def _native(self, native):
        value = np.asarray(native, dtype="f8")
        if not np.isfinite(value).all():
            raise ValueError(f"{self.name}: nonfinite native value")
        if self.log_transformed and not (value > 0).all():
            raise ValueError(
                f"{self.name} is log-transformed and positive; got a non-positive "
                "native value"
            )
        return value

    def to_coordinate(self, native):
        """Native value -> model coordinate ``T(y)`` (float64, no standardization)."""
        value = self._native(native)
        return np.log(value) if self.log_transformed else value

    def to_native(self, coordinate):
        """Model coordinate ``T(y)`` -> native value (inverse transform)."""
        value = np.asarray(coordinate, dtype="f8")
        if not np.isfinite(value).all():
            raise ValueError(f"{self.name}: nonfinite coordinate")
        return np.exp(value) if self.log_transformed else value

    def log_abs_det_jacobian(self, native):
        """``log|T'(y)|`` at the native value (0, ``-log d`` or ``-log age``)."""
        if not self.log_transformed:
            return np.zeros_like(np.asarray(native, dtype="f8"))
        value = self._native(native)
        return -np.log(value)


TARGET_SPECS = (
    TargetSpec(
        "teff",
        "K",
        IDENTITY,
        positive=True,
        error_kind=SYMMETRIC,
        description="effective temperature; identity in K, degenerate label term is K",
    ),
    TargetSpec(
        "logg",
        "dex",
        IDENTITY,
        error_kind=SYMMETRIC,
        description="surface gravity, dex cgs",
    ),
    TargetSpec(
        "fe_h",
        "dex",
        IDENTITY,
        error_kind=SYMMETRIC,
        description="metallicity [Fe/H]",
    ),
    TargetSpec(
        "distance",
        "kpc",
        LOG_KPC,
        positive=True,
        error_kind=RAW_SIGNED_PARALLAX,
        description=(
            "latent s=log(d/kpc); scored through raw signed parallax, never 1/varpi"
        ),
    ),
    TargetSpec(
        "age",
        "Gyr",
        LOG_GYR,
        positive=True,
        error_kind=ASYMMETRIC_QUANTILES,
        description="tau=log(age/Gyr); asymmetric 5/16/50/84/95 percentiles",
    ),
    TargetSpec(
        "alpha",
        "dex",
        IDENTITY,
        error_kind=SYMMETRIC,
        source_tag="per-survey",
        observable_kind=PROXY,
        status=INTERFACE_ONLY,
        description="alpha/M != alpha/Fe != alphafe; interface only, gated",
    ),
)

TARGET_NAMES = tuple(spec.name for spec in TARGET_SPECS)
TARGET_UNITS = tuple(spec.unit for spec in TARGET_SPECS)
TARGET_TRANSFORMS = tuple(spec.transform for spec in TARGET_SPECS)
REGISTRY = MappingProxyType({spec.name: spec for spec in TARGET_SPECS})


def target_spec(name: str) -> TargetSpec:
    """Registry lookup by name; unknown names never fall back to a position."""
    try:
        return REGISTRY[name]
    except (KeyError, TypeError):
        raise ValueError(
            f"target {name!r} is not in the frozen registry {TARGET_NAMES}"
        ) from None


def target_index(name: str) -> int:
    return TARGET_NAMES.index(target_spec(name).name)


def validate_targets(names) -> tuple[str, ...]:
    """Validate a declared target sequence: known, unique, registry-ordered.

    Registry order is required so a silently reordered vector is rejected rather
    than re-associated with the wrong native column.
    """
    declared = tuple(names)
    if not declared:
        raise ValueError("at least one target is required")
    specd = tuple(target_spec(name) for name in declared)
    resolved = tuple(spec.name for spec in specd)
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"duplicate targets in {resolved}")
    order = [TARGET_NAMES.index(name) for name in resolved]
    if order != sorted(order):
        raise ValueError(
            f"target order {resolved} differs from the frozen registry order {TARGET_NAMES}"
        )
    return resolved


def _as_matrix(names, values):
    names = validate_targets(names)
    array = np.asarray(values, dtype="f8")
    if array.ndim == 1 and len(names) == 1:
        array = array[:, None]
    if array.ndim != 2 or array.shape[1] != len(names):
        raise ValueError(f"expected [row, {len(names)}] values, got {array.shape}")
    return names, array


def model_coordinates(names, native):
    """Native matrix -> model-coordinate matrix (per declared transform)."""
    names, native = _as_matrix(names, native)
    return np.column_stack(
        [REGISTRY[name].to_coordinate(native[:, j]) for j, name in enumerate(names)]
    )


def native_from_coordinates(names, coordinates):
    """Model-coordinate matrix -> native matrix (registry-driven inverse)."""
    names, coordinates = _as_matrix(names, coordinates)
    return np.column_stack(
        [REGISTRY[name].to_native(coordinates[:, j]) for j, name in enumerate(names)]
    )


def native_from_standardized(names, standardized, center, scale):
    """Invert fit-only standardization and the transform, by name not position."""
    names, standardized = _as_matrix(names, standardized)
    center = np.asarray(center, dtype="f8")
    scale = np.asarray(scale, dtype="f8")
    if center.shape != (len(names),) or scale.shape != (len(names),):
        raise ValueError("center/scale must match the declared target count")
    if not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("fit-only target scale must be finite and positive")
    return native_from_coordinates(names, standardized * scale + center)


def log_abs_det_jacobian(names, native, *, observed=None):
    """Per-row, per-coordinate ``log|T'_j(y_j)|`` (0 for unobserved coordinates)."""
    names, native = _as_matrix(names, native)
    terms = np.column_stack(
        [
            REGISTRY[name].log_abs_det_jacobian(native[:, j])
            for j, name in enumerate(names)
        ]
    )
    if observed is None:
        return terms
    mask = np.asarray(observed, dtype=bool)
    if mask.shape != native.shape:
        raise ValueError("observed mask must match the native value matrix")
    return np.where(mask, terms, 0.0)


def density_offsets(names, native, scale, *, observed=None):
    """Per-row, per-observed-coordinate ``log w_j - log|T'_j(y_j)|``.

    Subtracting the row sum of this matrix from a standardized log density yields
    the native-unit log density (``p_Y = p_U |T'| / w``). For identity transforms
    the offset is ``log w``; for ``log_kpc``/``log_gyr`` it is ``log w + log y``.
    Unobserved coordinates contribute exactly zero.
    """
    names, native = _as_matrix(names, native)
    scale = np.asarray(scale, dtype="f8")
    if scale.shape != (len(names),):
        raise ValueError("fit-only scale must match the declared target count")
    if not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("fit-only target scale must be finite and positive")
    offsets = np.log(scale) - log_abs_det_jacobian(names, native, observed=observed)
    if observed is None:
        return offsets
    return np.where(np.asarray(observed, dtype=bool), offsets, 0.0)


def native_log_prob(names, log_prob_standardized, native, scale, *, observed=None):
    """Correct a standardized log density to native units, per row."""
    names = validate_targets(names)
    log_prob = np.asarray(log_prob_standardized, dtype="f8")
    if log_prob.shape != (len(native),):
        raise ValueError("standardized log density must be one value per row")
    return log_prob - density_offsets(names, native, scale, observed=observed).sum(1)
