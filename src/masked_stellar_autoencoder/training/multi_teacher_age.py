"""Multi-teacher age correction on a **named** age scale.

Implements the machinery specified in ``09-age-data-and-systematics.md``
Sec. 2.1-2.3 and ``08-losses-age-distance-alpha.md`` Sec. 2.5: per-teacher
bias/scale links, a shared latent age, cluster coevality, a predict-time
monotone meta-correction and the systematic-variance decomposition. Nothing
here calls a catalogue "truth" and nothing averages teacher values.

Coordinate
----------
``tau = ln(age / Gyr)``, exactly as :mod:`...training.age_heads`. Positivity of
the age is automatic; a Gaussian is never placed on Gyr.

Label model (per star ``i``, teacher ``s``)
------------------------------------------
::

    t_i                      ~ p_theta(tau | x_i)          # latent, here Gaussian
    y_{i,s} | t_i, x_i       ~ N( a_s(x_i) + b_s(x_i) t_i,
                                  sigma_s(x_i)^2 + kappa_s^2 sigma_quoted,is^2 )

    L_age = sum_s ( w_s / sum_i q_{i,s} ) sum_i q_{i,s}
            [ -log int p_theta(tau | x_i) N(y_{i,s} | a_s + b_s tau, v_{i,s}) dtau ]

``a_s(x), b_s(x), sigma_s(x)`` are the teacher's own offset, scale and intrinsic
scatter (``TeacherCalibration``: constant affine, small MLP, or stage-binned
affine over ``Teff/logg/FeH/alpha/stage``); ``kappa_s`` inflates the quoted
label error and ``q_{i,s}`` is the declared label quality weight. **Each teacher
keeps its own denominator**, so a teacher with a million labels has the same
weight in the objective as one with ten, and a teacher's term never depends on
another teacher's rows. Several teachers of one star enter as a *product* of
likelihoods (a sum of log-densities) given the shared latent ``t_i``; the
teacher values are never averaged into a pseudo-label. ``gaussian_convolution_nll``
is the exact closed form of that integral for a Gaussian latent and exists as
the numerical certificate of the quadrature (``teacher_marginal_nll`` with
``rule="auto"``), not as a replacement for a non-Gaussian latent.

Identifiability / the named scale
---------------------------------
Only *relative* teacher offsets and scales are identifiable, so the latent
coordinate must be defined by a **reference teacher**: its link is frozen at
``a = 0, b = 1`` (``TeacherCalibration.freeze_identity``) and every other
teacher is expressed relative to it. That teacher's ``age_scale`` string is the
scale the latent and every prediction are on - "age on a named scale", never
truth. Without a reference (``multi_teacher_age_loss`` used directly with all
links free) the likelihood is invariant under ``t -> lambda t``; that freedom is
the caller's declared gauge, not something this module hides.

Cluster coevality
-----------------
``cluster_coeval_loss`` ties members of one cluster to a shared age: supplied
(per-cluster tensor, possibly an ``nn.Parameter``, see ``LearnableClusterAges``)
or profiled out (closed-form weighted mean of the members). It is the bridge
used where teacher overlap is absent (MSTO/SGB StarHorse vs giant seismology),
and it is a *regularizer with cluster-level denominators*, not truth.

Predict-time meta-correction
----------------------------
``MetaCorrection`` implements ``tau_corr = tau_base + g_phi(tau_base, x,
target_scale)`` with a monotone-by-construction family (see the class
docstring). ``fit`` only ever sees the rows it is given, and ``anchor_mask``
restricts it to anchor rows: a few-shot re-fit for a new survey touches only
``g_phi`` and no test row.

Uncertainty
-----------
``systematic_variance_decomposition`` returns
``Var_model + Var_within_teacher + Var_between_teacher + Var_scale`` with the
plan's reporting labels, exact addition and no clipping. The between-teacher
term is a *systematic envelope*, not coverage of truth.

Runner hook (A1/A2)
-------------------
::

    head = MultiTeacherAgeHead(input_dim=256, teacher_names=("starhorse", "apokasc3", "willett"),
                               reference_teacher="apokasc3", calibration_mode="mlp")
    labels = TeacherLabelBatch.from_rows(rows, teacher_names=head.teacher_names)
    out = head.loss(latent_features, physical_features, labels)     # A1
    total = out.mean + cluster_weight * cluster_coeval_loss(head_tau, cluster_ids,
                                                            cluster_age=ages).mean
    correction = MetaCorrection(feature_dim=physical_features.shape[1],
                                target_scales=("cluster",)).double()
    fit = correction.fit(base_tau, physical_features, anchor_tau,
                         target_scale="cluster", anchor_mask=anchor_rows)   # A2
    corrected = correction.apply(base_tau, physical_features, "cluster")
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist

import torch
from torch import Tensor, nn

from .age_heads import AgeGaussianHead
from .distance_heads import fixed_qmc_nodes, gauss_hermite

__all__ = [
    "AGE_COORDINATE",
    "DEFAULT_QMC_COUNT",
    "DEFAULT_QMC_SEED",
    "DEFAULT_QUADRATURE_ORDER",
    "DEFAULT_TARGET_SCALE",
    "DEFAULT_TEACHER_FEATURE_NAMES",
    "MULTI_TEACHER_EVIDENCE_STATUS",
    "NODE_SETS",
    "QUADRATURE_RULES",
    "STAGE_FEATURE_NAME",
    "SYSTEMATIC_VARIANCE_LABELS",
    "TEACHER_CALIBRATION_MODES",
    "ClusterCoevalLoss",
    "LearnableClusterAges",
    "MetaCorrection",
    "MetaCorrectionFit",
    "MetaCorrectionOutput",
    "MultiTeacherAgeHead",
    "MultiTeacherAgeLoss",
    "QuadratureDiagnostics",
    "SystematicVariance",
    "TeacherAgeLabel",
    "TeacherCalibration",
    "TeacherLabelBatch",
    "TeacherMapping",
    "TeacherNLL",
    "cluster_coeval_loss",
    "gaussian_convolution_nll",
    "multi_teacher_age_loss",
    "systematic_variance_decomposition",
    "teacher_marginal_nll",
]

_LOG_2PI = math.log(2.0 * math.pi)
#: Gaussian-equivalent one-sigma half-width of the 5/95 interval
#: (``Phi^{-1}(0.95) = 1.6449``); used *only* when a label supplies 5/95
#: percentiles and no 16/84 pair and no quoted error.
_Z_95 = NormalDist().inv_cdf(0.95)

#: The age coordinate of this module (natural log, age in Gyr).
AGE_COORDINATE = "tau = ln(age / Gyr)"

#: Declared evidence status of every age product of this module.
MULTI_TEACHER_EVIDENCE_STATUS = (
    "observed_label (multi-teacher, named age scale); not physical truth"
)

#: Physical features the teacher links are allowed to depend on.
STAGE_FEATURE_NAME = "stage"
DEFAULT_TEACHER_FEATURE_NAMES: tuple[str, ...] = (
    "teff",
    "logg",
    "feh",
    "alpha",
    STAGE_FEATURE_NAME,
)

#: Teacher link families.
TEACHER_CALIBRATION_MODES: tuple[str, ...] = ("affine", "mlp", "stage_affine")
#: Latent-integration rules. Both are exact identities for any latent scale;
#: they differ in which limit they converge fastest (see ``teacher_marginal_nll``).
QUADRATURE_RULES: tuple[str, ...] = ("auto", "gh_latent", "gh_residual")
#: Fixed node sets: Gauss-Hermite, or the saved antithetic Sobol set.
NODE_SETS: tuple[str, ...] = ("gh", "qmc")
DEFAULT_QUADRATURE_ORDER = 32
DEFAULT_QMC_COUNT = 64
DEFAULT_QMC_SEED = 2026

#: Reporting labels of ``09`` Sec. 2.3, in the order they are summed.
SYSTEMATIC_VARIANCE_LABELS: tuple[str, ...] = (
    "Var_model",
    "Var_within_teacher",
    "Var_between_teacher",
    "Var_scale",
)

#: Default name of the correction target scale (a string the caller owns).
DEFAULT_TARGET_SCALE = "reference"


# --------------------------------------------------------------------------- #
# Small validators
# --------------------------------------------------------------------------- #
def _require_name(what: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string")
    return value


def _require_finite(what: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{what} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what} must be finite")
    return number


def _require_positive(what: str, value: float) -> float:
    number = _require_finite(what, value)
    if number <= 0.0:
        raise ValueError(f"{what} must be strictly positive; no clipping")
    return number


def _check_names(names: Sequence[str], what: str) -> tuple[str, ...]:
    checked = tuple(_require_name(what, name) for name in names)
    if not checked:
        raise ValueError(f"{what} must contain at least one name")
    if len(set(checked)) != len(checked):
        raise ValueError(f"{what} must be unique")
    return checked


def _check_1d(what: str, value: Tensor, rows: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{what} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{what} must be floating point")
    if value.ndim != 1:
        raise ValueError(f"{what} must be one-dimensional [row]")
    if value.shape[0] != rows:
        raise ValueError(f"{what} must have exactly {rows} rows")
    if not torch.isfinite(value).all():
        raise ValueError(f"{what} must be finite; no clipping or imputation")
    return value


def _check_features(
    features: Tensor,
    feature_names: Sequence[str],
    *,
    what: str = "teacher features",
) -> Tensor:
    if not isinstance(features, Tensor):
        raise TypeError(f"{what} must be a torch.Tensor")
    if not features.is_floating_point():
        raise TypeError(f"{what} must be floating point")
    if features.ndim != 2:
        raise ValueError(f"{what} must be [row, feature]")
    if features.shape[1] != len(feature_names):
        raise ValueError(
            f"{what} must have {len(feature_names)} columns {tuple(feature_names)}"
        )
    if not torch.isfinite(features).all():
        raise ValueError(f"{what} must be finite")
    return features


def _check_rule(rule: str, nodes: str) -> None:
    if rule not in QUADRATURE_RULES:
        raise ValueError(f"rule must be one of {QUADRATURE_RULES}")
    if nodes not in NODE_SETS:
        raise ValueError(f"nodes must be one of {NODE_SETS}")


def _zero_like(*tensors: Tensor) -> Tensor:
    """Differentiable zero that carries the graph of every argument."""
    total = tensors[0].reshape(-1)[:0].sum()
    for tensor in tensors:
        total = total + tensor.reshape(-1)[:0].sum()
    return total


# --------------------------------------------------------------------------- #
# Teacher labels
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TeacherAgeLabel:
    """One teacher's age label for one star, in ``tau = ln(age / Gyr)``.

    ``tau50`` is the central value (median/mode as the teacher reports it).
    Optional ``tau16``/``tau84`` (preferred) or ``tau5``/``tau95`` carry the
    reported asymmetry. ``sigma_quoted`` is the teacher's quoted one-sigma
    error in ``tau``; when it is omitted it is derived from the 16/84 half-width
    (or, failing that, from 5/95 divided by the Gaussian-equivalent 1.645) and
    the derivation is recorded in ``sigma_quoted_source``. A label with neither a
    quoted error nor a percentile pair is rejected: an error is never invented.

    ``age_scale`` names the scale the value is on (``"apokasc3"``,
    ``"willett_age_Dnu"``, ``"cluster"``, ...); it is carried into the batch and
    reported, and mixing scales inside one teacher is rejected there.
    ``flags``/``quality`` carry the source quality metadata; ``quality`` in
    ``[0, 1]`` is a declared reliability *weight*, never a variance.
    """

    teacher: str
    age_scale: str
    tau50: float
    sigma_quoted: float | None = None
    tau16: float | None = None
    tau84: float | None = None
    tau5: float | None = None
    tau95: float | None = None
    flags: tuple[str, ...] = ()
    quality: float = 1.0
    #: How ``sigma_quoted`` was obtained (``quoted``, ``derived_16_84``,
    #: ``derived_5_95_gaussian``); set in ``__post_init__``, never passed in.
    sigma_quoted_source: str = field(init=False, default="")

    def __post_init__(self) -> None:
        object.__setattr__(self, "teacher", _require_name("teacher", self.teacher))
        object.__setattr__(
            self, "age_scale", _require_name("age_scale", self.age_scale)
        )
        object.__setattr__(self, "tau50", _require_finite("tau50", self.tau50))
        percentiles: dict[str, float | None] = {}
        for name in ("tau16", "tau84", "tau5", "tau95"):
            value = getattr(self, name)
            if value is None:
                percentiles[name] = None
                continue
            checked = _require_finite(name, value)
            percentiles[name] = checked
            object.__setattr__(self, name, checked)
        sequence = [
            percentiles["tau5"],
            percentiles["tau16"],
            self.tau50,
            percentiles["tau84"],
            percentiles["tau95"],
        ]
        defined = [value for value in sequence if value is not None]
        if any(later < earlier for earlier, later in zip(defined, defined[1:])):
            raise ValueError(
                "teacher percentiles are crossed (5 <= 16 <= 50 <= 84 <= 95); "
                "refusing to reorder them"
            )
        source = "quoted"
        if self.sigma_quoted is None:
            if percentiles["tau16"] is not None and percentiles["tau84"] is not None:
                source = "derived_16_84"
                sigma = 0.5 * (percentiles["tau84"] - percentiles["tau16"])
            elif percentiles["tau5"] is not None and percentiles["tau95"] is not None:
                source = "derived_5_95_gaussian"
                sigma = 0.5 * (percentiles["tau95"] - percentiles["tau5"]) / _Z_95
            else:
                raise ValueError(
                    "a teacher label needs sigma_quoted or a 16/84 (or 5/95) "
                    "percentile pair; no uncertainty is invented"
                )
            if not sigma > 0.0:
                raise ValueError(
                    "the derived quoted uncertainty is not positive; "
                    "the percentile pair is degenerate"
                )
        else:
            sigma = _require_positive("sigma_quoted", self.sigma_quoted)
        object.__setattr__(self, "sigma_quoted", sigma)
        object.__setattr__(self, "sigma_quoted_source", source)
        flags = tuple(self.flags)
        if any(not isinstance(flag, str) or not flag for flag in flags):
            raise ValueError("flags must be non-empty strings")
        object.__setattr__(self, "flags", flags)
        quality = _require_finite("quality", self.quality)
        if not 0.0 <= quality <= 1.0:
            raise ValueError("quality must lie in [0, 1]")
        object.__setattr__(self, "quality", quality)

    # -- derived quantities ------------------------------------------------- #
    @property
    def age_gyr(self) -> float:
        """``exp(tau50)`` in Gyr; raises rather than clipping an overflow."""
        age = math.exp(self.tau50)
        if not math.isfinite(age) or age <= 0.0:
            raise ValueError("age overflowed; no silent clipping of log age")
        return age

    @property
    def sigma_quoted_from_percentiles(self) -> float | None:
        """Half-width of the reported 16/84 pair, when both are present."""
        if self.tau16 is None or self.tau84 is None:
            return None
        return 0.5 * (self.tau84 - self.tau16)

    @property
    def tau_percentiles(self) -> dict[str, float]:
        """Defined percentiles keyed by their probability string."""
        out = {}
        for probability, name in (
            ("0.05", "tau5"),
            ("0.16", "tau16"),
            ("0.50", "tau50"),
            ("0.84", "tau84"),
            ("0.95", "tau95"),
        ):
            value = getattr(self, name)
            if value is not None:
                out[probability] = value
        return out

    @classmethod
    def from_gyr(
        cls,
        teacher: str,
        age_scale: str,
        age_gyr: float,
        sigma_quoted: float | None = None,
        *,
        sigma_quoted_gyr: float | None = None,
        flags: tuple[str, ...] = (),
        quality: float = 1.0,
    ) -> TeacherAgeLabel:
        """Build from a central age in Gyr (and optionally a linear error).

        ``sigma_quoted`` is in ``tau``; ``sigma_quoted_gyr`` (mutually exclusive)
        is a linear age error converted as ``sigma_tau = sigma_gyr / age``, the
        first-order propagation stated in ``08`` Sec. 2.4. No error is invented
        when neither is given and no percentile pair exists.
        """
        age = _require_positive("age_gyr", age_gyr)
        if sigma_quoted is not None and sigma_quoted_gyr is not None:
            raise ValueError("give sigma_quoted (tau) or sigma_quoted_gyr, not both")
        if sigma_quoted_gyr is not None:
            sigma = _require_positive("sigma_quoted_gyr", sigma_quoted_gyr) / age
        else:
            sigma = sigma_quoted
        return cls(
            teacher=teacher,
            age_scale=age_scale,
            tau50=math.log(age),
            sigma_quoted=sigma,
            flags=flags,
            quality=quality,
        )

    @classmethod
    def from_percentiles_gyr(
        cls,
        teacher: str,
        age_scale: str,
        *,
        age_50: float,
        age_16: float | None = None,
        age_84: float | None = None,
        age_5: float | None = None,
        age_95: float | None = None,
        sigma_quoted: float | None = None,
        flags: tuple[str, ...] = (),
        quality: float = 1.0,
    ) -> TeacherAgeLabel:
        """Build from StarHorse-style posterior percentiles in Gyr."""

        def tau(value: float | None, name: str) -> float | None:
            if value is None:
                return None
            return math.log(_require_positive(name, value))

        return cls(
            teacher=teacher,
            age_scale=age_scale,
            tau50=tau(age_50, "age_50"),
            sigma_quoted=sigma_quoted,
            tau16=tau(age_16, "age_16"),
            tau84=tau(age_84, "age_84"),
            tau5=tau(age_5, "age_5"),
            tau95=tau(age_95, "age_95"),
            flags=flags,
            quality=quality,
        )


@dataclass(frozen=True, eq=False)
class TeacherLabelBatch:
    """Rectangular ``[teacher, row]`` view of sparse teacher labels.

    ``observed[i, r]`` is exactly ``isfinite(tau[i, r])``: an absent label is a
    NaN with ``observed=False``, never a zero and never an imputed value.
    ``quality`` is 0 wherever the label is absent. Sparse rows (a star with one
    of three teachers) are the normal case and are handled per teacher, not by
    dropping the star.
    """

    teacher_names: tuple[str, ...]
    age_scales: tuple[str, ...]
    tau: Tensor
    sigma_quoted: Tensor
    observed: Tensor
    quality: Tensor

    def __post_init__(self) -> None:
        names = _check_names(self.teacher_names, "teacher_names")
        if len(self.age_scales) != len(names):
            raise ValueError("age_scales must align with teacher_names")
        tau, sigma, observed, quality = (
            self.tau,
            self.sigma_quoted,
            self.observed,
            self.quality,
        )
        for what, value in (
            ("tau", tau),
            ("sigma_quoted", sigma),
            ("quality", quality),
        ):
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(f"{what} must be a floating torch.Tensor")
            if value.ndim != 2 or value.shape[0] != len(names):
                raise ValueError(f"{what} must be [n_teachers, n_rows]")
            if value.shape != tau.shape:
                raise ValueError(f"{what} must share the [teacher, row] shape")
        if observed.dtype is not torch.bool or observed.shape != tau.shape:
            raise ValueError("observed must be a boolean [n_teachers, n_rows] mask")
        if not torch.equal(observed, torch.isfinite(tau)):
            raise ValueError("observed must be exactly isfinite(tau)")
        if not torch.isfinite(tau[observed]).all():
            raise ValueError("observed tau must be finite")
        if not (torch.isfinite(sigma[observed]).all() and (sigma[observed] > 0).all()):
            raise ValueError("observed quoted errors must be finite and positive")
        if not ((quality >= 0) & (quality <= 1)).all():
            raise ValueError("quality must lie in [0, 1]")
        if not bool((quality[~observed] == 0).all()):
            raise ValueError("absent labels must carry zero quality")

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_rows(
        cls,
        rows: Sequence[Mapping[str, TeacherAgeLabel]],
        teacher_names: Sequence[str] | None = None,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> TeacherLabelBatch:
        """Stack per-star ``{teacher: label}`` mappings into one batch.

        ``teacher_names`` fixes the teacher axis (recommended: the head's own
        order). When omitted it is the first-appearance order over the rows.
        A teacher may not appear twice in one row, a label's ``teacher`` must
        match its key, and one teacher may not mix ``age_scale`` values.
        """
        materialized = list(rows)
        if teacher_names is None:
            order: list[str] = []
            for row in materialized:
                for key in row:
                    if key not in order:
                        order.append(key)
            if not order:
                raise ValueError(
                    "teacher_names is required when no row carries a label"
                )
            names = tuple(order)
        else:
            names = _check_names(teacher_names, "teacher_names")
        index = {name: i for i, name in enumerate(names)}
        n_rows = len(materialized)
        tau = torch.full((len(names), n_rows), float("nan"), dtype=dtype)
        sigma = torch.full((len(names), n_rows), float("nan"), dtype=dtype)
        quality = torch.zeros((len(names), n_rows), dtype=dtype)
        scales: list[str | None] = [None] * len(names)
        for row_index, row in enumerate(materialized):
            for key, label in row.items():
                if not isinstance(key, str):
                    raise TypeError("row keys must be teacher names")
                if key not in index:
                    raise ValueError(
                        f"unknown teacher {key!r}; expected one of {names}"
                    )
                if not isinstance(label, TeacherAgeLabel):
                    raise TypeError(f"row {row_index} teacher {key!r} is not a label")
                if label.teacher != key:
                    raise ValueError(
                        f"row {row_index}: label teacher {label.teacher!r} does not "
                        f"match its key {key!r}"
                    )
                column = index[key]
                if scales[column] is None:
                    scales[column] = label.age_scale
                elif scales[column] != label.age_scale:
                    raise ValueError(
                        f"teacher {key!r} mixes age scales "
                        f"{scales[column]!r} and {label.age_scale!r}"
                    )
                tau[column, row_index] = float(label.tau50)
                sigma[column, row_index] = float(label.sigma_quoted)
                quality[column, row_index] = float(label.quality)
        return cls(
            teacher_names=names,
            age_scales=tuple(scale or "" for scale in scales),
            tau=tau,
            sigma_quoted=sigma,
            observed=torch.isfinite(tau),
            quality=quality,
        )

    # -- accessors ---------------------------------------------------------- #
    @property
    def n_teachers(self) -> int:
        return len(self.teacher_names)

    @property
    def n_rows(self) -> int:
        return int(self.tau.shape[1])

    def index_of(self, teacher: str) -> int:
        try:
            return self.teacher_names.index(teacher)
        except ValueError as error:
            raise ValueError(
                f"unknown teacher {teacher!r}; have {self.teacher_names}"
            ) from error

    def to(
        self, *, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        """Move/cast the numeric payload (labels stay a batch, not a module)."""
        return TeacherLabelBatch(
            teacher_names=self.teacher_names,
            age_scales=self.age_scales,
            tau=self.tau.to(device=device, dtype=dtype),
            sigma_quoted=self.sigma_quoted.to(device=device, dtype=dtype),
            observed=self.observed.to(device=device),
            quality=self.quality.to(device=device, dtype=dtype),
        )

    def select_teachers(self, names: Sequence[str]) -> TeacherLabelBatch:
        """Reindex the teacher axis (absent teachers keep zero labels)."""
        indices = [self.index_of(name) for name in names]
        return TeacherLabelBatch(
            teacher_names=_check_names(names, "teacher_names"),
            age_scales=tuple(self.age_scales[i] for i in indices),
            tau=self.tau[indices],
            sigma_quoted=self.sigma_quoted[indices],
            observed=self.observed[indices],
            quality=self.quality[indices],
        )

    def describe(self) -> dict[str, object]:
        """Manifest-ready label inventory; counts per teacher, no values."""
        rows = []
        for column, name in enumerate(self.teacher_names):
            observed = self.observed[column]
            count = int(observed.sum())
            sigma = self.sigma_quoted[column][observed]
            rows.append(
                {
                    "teacher": name,
                    "age_scale": self.age_scales[column],
                    "n_labels": count,
                    "n_quality_zero": int((self.quality[column][observed] == 0).sum()),
                    "sigma_quoted_median": (float(sigma.median()) if count else None),
                    "sigma_quoted_min": float(sigma.min()) if count else None,
                    "sigma_quoted_max": float(sigma.max()) if count else None,
                }
            )
        return {
            "n_rows": self.n_rows,
            "n_rows_with_label": int(self.observed.any(0).sum()),
            "n_labels": int(self.observed.sum()),
            "teachers": rows,
            "partial_label_policy": "per_teacher_mask; no imputation",
        }


# --------------------------------------------------------------------------- #
# Teacher link a_s(x), b_s(x), sigma_s(x) and the quoted-error inflation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class TeacherMapping:
    """Per-row teacher link ``y = offset + slope * tau`` and its noise terms.

    ``effective_sigma(q) = sqrt(intrinsic_sigma^2 + (kappa * q)^2)``; the
    intrinsic scatter is fitted, ``kappa`` inflates the quoted error, and
    neither is floored.
    """

    teacher: str
    offset: Tensor
    slope: Tensor
    intrinsic_sigma: Tensor
    kappa: Tensor

    def mapped_mean(self, tau: Tensor) -> Tensor:
        """``a + b * tau``: the teacher scale mean for latent ``tau``."""
        return self.offset + self.slope * tau

    def mapped_scale(self, sigma: Tensor) -> Tensor:
        """``b * sigma``: the latent width carried onto the teacher scale."""
        return self.slope * sigma

    def effective_sigma(self, sigma_quoted: Tensor) -> Tensor:
        """Total teacher standard deviation in ``tau`` units."""
        if sigma_quoted.shape != self.offset.shape:
            raise ValueError("quoted errors must align with the teacher mapping")
        if not torch.isfinite(sigma_quoted).all():
            raise ValueError("quoted errors must be finite")
        if not (sigma_quoted >= 0).all():
            raise ValueError("quoted errors must be non-negative")
        variance = self.intrinsic_sigma.square() + (self.kappa * sigma_quoted).square()
        sigma = variance.sqrt()
        if not (torch.isfinite(sigma).all() and (sigma > 0).all()):
            raise ValueError("invalid teacher sigma; no floor or clipping")
        return sigma


class TeacherCalibration(nn.Module):
    """Teacher-specific ``a_s(x)``, ``b_s(x)``, ``sigma_s(x)``, ``kappa_s``.

    ``mode="affine"``
        Constant ``a`` and ``b = exp(log_b)``; ``sigma_s = exp(log sigma_s)``.
    ``mode="mlp"``
        A small MLP over the declared features producing ``a``, ``log b``,
        ``log sigma_s``; the last layer is zero-initialised so training starts
        from ``(initial_offset, initial_slope, initial_intrinsic_sigma)``.
    ``mode="stage_affine"``
        One affine triple per integer value of the ``stage`` column ("feature-
        binned affine"); the stage column must be integral and inside
        ``[0, n_stages)`` - no clipping of an out-of-range stage.

    The link is monotone in ``tau`` because ``b > 0`` is enforced by
    construction (``exp``) and rejected if it overflows. ``kappa`` is a free
    positive inflation initialised at ``initial_kappa`` (normally 1): the plan's
    ``kappa_s >= 1`` is *reported* (``inflation_at_least_one``), never clamped,
    because clamping would hide a teacher whose quoted errors are too large.

    ``freeze_identity`` pins ``a = 0`` and ``b = 1`` exactly for the reference
    teacher, which is what defines the latent age scale; its ``sigma_s`` and
    ``kappa_s`` stay trainable.
    """

    def __init__(
        self,
        teacher: str,
        *,
        feature_names: Sequence[str] = DEFAULT_TEACHER_FEATURE_NAMES,
        mode: str = "affine",
        hidden: int = 32,
        n_stages: int | None = None,
        initial_offset: float = 0.0,
        initial_slope: float = 1.0,
        initial_intrinsic_sigma: float = 0.1,
        initial_kappa: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.teacher = _require_name("teacher", teacher)
        self.feature_names = _check_names(feature_names, "feature_names")
        if mode not in TEACHER_CALIBRATION_MODES:
            raise ValueError(f"mode must be one of {TEACHER_CALIBRATION_MODES}")
        self.mode = mode
        self.initial_offset = _require_finite("initial_offset", initial_offset)
        self.initial_slope = _require_positive("initial_slope", initial_slope)
        self.initial_intrinsic_sigma = _require_positive(
            "initial_intrinsic_sigma", initial_intrinsic_sigma
        )
        self.initial_kappa = _require_positive("initial_kappa", initial_kappa)
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        options = {"dtype": dtype, "device": device}
        log_slope = math.log(self.initial_slope)
        log_sigma = math.log(self.initial_intrinsic_sigma)
        log_kappa = math.log(self.initial_kappa)
        if mode == "affine":
            self.offset = nn.Parameter(torch.tensor(self.initial_offset, **options))
            self.log_slope = nn.Parameter(torch.tensor(log_slope, **options))
            self.log_intrinsic_sigma = nn.Parameter(torch.tensor(log_sigma, **options))
        elif mode == "mlp":
            self.trunk = nn.Sequential(
                nn.Linear(len(self.feature_names), hidden, **options),
                nn.GELU(),
                nn.Linear(hidden, 3, **options),
            )
            nn.init.zeros_(self.trunk[-1].weight)
            nn.init.zeros_(self.trunk[-1].bias)
        else:
            if (
                n_stages is None
                or isinstance(n_stages, bool)
                or not isinstance(n_stages, int)
            ):
                raise ValueError("stage_affine mode requires an integer n_stages")
            if n_stages < 1:
                raise ValueError("n_stages must be positive")
            if STAGE_FEATURE_NAME not in self.feature_names:
                raise ValueError(
                    f"stage_affine mode requires a {STAGE_FEATURE_NAME!r} feature"
                )
            self.n_stages = n_stages
            self.stage_column = self.feature_names.index(STAGE_FEATURE_NAME)
            self.offset = nn.Parameter(
                torch.full((n_stages,), self.initial_offset, **options)
            )
            self.log_slope = nn.Parameter(torch.full((n_stages,), log_slope, **options))
            self.log_intrinsic_sigma = nn.Parameter(
                torch.full((n_stages,), log_sigma, **options)
            )
        self.log_kappa = nn.Parameter(torch.tensor(log_kappa, **options))
        self._frozen_identity = False

    # -- parameters / diagnostics ------------------------------------------ #
    @property
    def kappa(self) -> Tensor:
        """Positive quoted-error inflation ``kappa_s`` (no clamping)."""
        kappa = self.log_kappa.exp()
        if not (torch.isfinite(kappa) and kappa > 0):
            raise ValueError("kappa overflowed; no clipping")
        return kappa

    @property
    def intrinsic_sigma(self) -> Tensor:
        """Positive fitted intrinsic scatter ``sigma_s`` (last value if binned)."""
        sigma = self.log_intrinsic_sigma.exp()
        if not (torch.isfinite(sigma).all() and (sigma > 0).all()):
            raise ValueError("intrinsic sigma overflowed; no clipping")
        return sigma

    @property
    def inflation_at_least_one(self) -> bool:
        """Diagnostic: is the quoted error not shrunk (``kappa >= 1``)?"""
        return bool(self.kappa.detach() >= 1.0)

    def freeze_identity(self) -> TeacherCalibration:
        """Pin ``a = 0, b = 1`` (the reference teacher that defines the scale)."""
        if self.mode != "affine":
            raise ValueError(
                "only the affine parameterization can be frozen to an exact "
                "identity; use mode='affine' for the reference teacher"
            )
        with torch.no_grad():
            self.offset.zero_()
            self.log_slope.zero_()
        self.offset.requires_grad_(False)
        self.log_slope.requires_grad_(False)
        self._frozen_identity = True
        return self

    @property
    def is_identity(self) -> bool:
        """Exact identity link (``a = 0`` and ``b = 1`` bitwise)."""
        if self.mode != "affine":
            return False
        return bool(
            (self.offset.detach() == 0).all() and (self.log_slope.detach() == 0).all()
        )

    def stage_index(self, features: Tensor) -> Tensor:
        """Integer stage code per row; raises instead of clipping bad codes."""
        stage = features[:, self.stage_column]
        rounded = stage.round()
        if not torch.allclose(stage, rounded, rtol=0.0, atol=1e-6):
            raise ValueError("the stage feature must be integral for stage_affine")
        index = rounded.to(torch.int64)
        if not ((index >= 0) & (index < self.n_stages)).all():
            raise ValueError(
                f"stage codes must lie in [0, {self.n_stages}); no clipping"
            )
        return index

    def _link_parameters(self, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        rows = features.shape[0]
        if self.mode == "affine":
            return (
                self.offset.expand(rows),
                self.log_slope.expand(rows),
                self.log_intrinsic_sigma.expand(rows),
            )
        if self.mode == "mlp":
            raw = self.trunk(features)
            return (
                raw[:, 0] + self.initial_offset,
                raw[:, 1] + math.log(self.initial_slope),
                raw[:, 2] + math.log(self.initial_intrinsic_sigma),
            )
        index = self.stage_index(features)
        return (
            self.offset[index],
            self.log_slope[index],
            self.log_intrinsic_sigma[index],
        )

    def forward(self, features: Tensor) -> TeacherMapping:
        """Evaluate the link at ``features`` (``[row, len(feature_names)]``)."""
        _check_features(features, self.feature_names, what=f"{self.teacher} features")
        offset, log_slope, log_sigma = self._link_parameters(features)
        slope = log_slope.exp()
        sigma = log_sigma.exp()
        if not (
            torch.isfinite(offset).all()
            and torch.isfinite(slope).all()
            and torch.isfinite(sigma).all()
        ):
            raise ValueError("nonfinite teacher link; no clipping")
        if not (slope > 0).all():
            raise ValueError("teacher slope must be strictly positive")
        if not (sigma > 0).all():
            raise ValueError("teacher intrinsic sigma must be strictly positive")
        mapping = TeacherMapping(
            teacher=self.teacher,
            offset=offset,
            slope=slope,
            intrinsic_sigma=sigma,
            kappa=self.kappa.expand(features.shape[0]),
        )
        if mapping.offset.shape != (features.shape[0],):
            raise ValueError("teacher mapping must be one value per row")
        return mapping

    def transform(
        self,
        tau: Tensor,
        features: Tensor,
        *,
        mapping: TeacherMapping | None = None,
    ) -> Tensor:
        """Map latent ``tau`` onto the teacher scale: ``a + b * tau``."""
        if not isinstance(tau, Tensor) or tau.ndim != 1:
            raise ValueError("tau must be a one-dimensional [row] tensor")
        evaluated = self(features) if mapping is None else mapping
        if evaluated.offset.shape[0] not in (tau.shape[0], 1):
            raise ValueError(
                "tau and the teacher mapping must share the row axis "
                "(a one-row mapping broadcasts)"
            )
        if not torch.isfinite(tau).all():
            raise ValueError("tau must be finite")
        return evaluated.mapped_mean(tau)

    def inverse(
        self,
        mapped_tau: Tensor,
        features: Tensor,
        *,
        mapping: TeacherMapping | None = None,
    ) -> Tensor:
        """Map a teacher-scale value back to latent ``tau``: ``(y - a) / b``."""
        if not isinstance(mapped_tau, Tensor) or mapped_tau.ndim != 1:
            raise ValueError("mapped_tau must be a one-dimensional [row] tensor")
        evaluated = self(features) if mapping is None else mapping
        if evaluated.offset.shape[0] not in (mapped_tau.shape[0], 1):
            raise ValueError(
                "mapped_tau and the mapping must share the row axis "
                "(a one-row mapping broadcasts)"
            )
        if not torch.isfinite(mapped_tau).all():
            raise ValueError("mapped_tau must be finite")
        return (mapped_tau - evaluated.offset) / evaluated.slope

    def diagnostics(self, features: Tensor) -> dict[str, object]:
        """Reportable per-teacher link summary (no data values)."""
        mapping = self(features)
        offset = mapping.offset.detach()
        slope = mapping.slope.detach()
        intrinsic = mapping.intrinsic_sigma.detach()
        return {
            "teacher": self.teacher,
            "mode": self.mode,
            "offset_min": float(offset.min()) if offset.numel() else 0.0,
            "offset_mean": float(offset.mean()) if offset.numel() else 0.0,
            "offset_max": float(offset.max()) if offset.numel() else 0.0,
            "slope_min": float(slope.min()) if slope.numel() else 0.0,
            "slope_mean": float(slope.mean()) if slope.numel() else 0.0,
            "slope_max": float(slope.max()) if slope.numel() else 0.0,
            "intrinsic_sigma": float(intrinsic.mean()) if intrinsic.numel() else 0.0,
            "kappa": float(self.kappa.detach()),
            "kappa_at_least_one": self.inflation_at_least_one,
            "identity": self.is_identity,
            "n_parameters": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }


# --------------------------------------------------------------------------- #
# Latent integration: Gauss-Hermite (latent or residual space) or fixed QMC
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QuadratureDiagnostics:
    """Which rule integrated which rows, and the worst resolution of each.

    ``min_latent_resolution`` is the *worst* (largest) ``b sigma / sigma_eff``
    among the rows integrated in latent space: that rule is trustworthy while
    the value stays below about one. ``min_residual_resolution`` is the worst
    (largest) ``sigma_eff / (b sigma)`` among the residual-space rows, with the
    same reading. Both are ``inf`` when no row used that branch, and both must
    be reported with a run: they say how close the batch came to the crossover.
    """

    latent_rule_rows: int = 0
    residual_rule_rows: int = 0
    min_latent_resolution: float = float("inf")
    min_residual_resolution: float = float("inf")


def _log_marginal_latent(
    y: Tensor,
    mean: Tensor,
    scale: Tensor,
    sigma_eff: Tensor,
    node: Tensor,
    weight: Tensor,
) -> Tensor:
    """``log E_u[N(y | mean + scale u, sigma_eff)]``, nodes ``u`` in latent space."""
    residual = (y[:, None] - mean[:, None] - scale[:, None] * node[None, :]) / (
        sigma_eff[:, None]
    )
    log_normal = -0.5 * residual.square() - sigma_eff.log()[:, None] - 0.5 * _LOG_2PI
    return torch.logsumexp(weight.log()[None, :] + log_normal, dim=1) - 0.5 * _LOG_2PI


def _log_marginal_residual(
    y: Tensor,
    mean: Tensor,
    scale: Tensor,
    sigma_eff: Tensor,
    node: Tensor,
    weight: Tensor,
) -> Tensor:
    """Same integral in the standardized teacher residual, with the Jacobian.

    With ``u_k = (y - mean - sigma_eff r_k) / scale`` the teacher likelihood
    becomes ``N(r_k) / sigma_eff`` and ``du = (sigma_eff/scale) dr``, so the two
    ``sigma_eff`` cancel and the integrand is ``N(u_k) / scale``: the rule is
    exact in the narrow-likelihood limit ``sigma_eff -> 0`` and rapidly
    convergent when ``sigma_eff << scale``, where the latent-space rule stalls.
    """
    latent = (y[:, None] - mean[:, None] - sigma_eff[:, None] * node[None, :]) / scale[
        :, None
    ]
    log_normal = -0.5 * latent.square() - 0.5 * _LOG_2PI
    terms = weight.log()[None, :] + log_normal - scale.log()[:, None]
    return torch.logsumexp(terms, dim=1) - 0.5 * _LOG_2PI


def teacher_marginal_nll(
    y: Tensor,
    sigma_quoted: Tensor,
    mu: Tensor,
    log_sigma: Tensor,
    mapping: TeacherMapping,
    *,
    rule: str = "auto",
    order: int = DEFAULT_QUADRATURE_ORDER,
    nodes: str = "gh",
    qmc_count: int = DEFAULT_QMC_COUNT,
    qmc_seed: int = DEFAULT_QMC_SEED,
) -> tuple[Tensor, QuadratureDiagnostics]:
    """``-log int p_theta(tau|x) N(y | a + b tau, v) dtau`` per observed row.

    ``p_theta`` is the Gaussian latent ``N(mu, exp(log_sigma)^2)`` and the teacher
    likelihood is Gaussian with variance ``v = sigma_s^2 + kappa_s^2 sigma_quoted^2``.
    Both quadrature rules below are algebraically exact identities for any
    ``b sigma > 0``; they differ only in convergence speed:

    ``gh_latent``
        nodes ``tau_k = mu + sigma u_k``. Accurate when the teacher likelihood is
        broad compared with the latent (``b sigma / sqrt(v) <~ 1``), i.e. exactly
        where the label is noisy.
    ``gh_residual``
        nodes ``tau_k = (y - a - sqrt(v) r_k) / b``, with the exact Jacobian.
        Accurate when the latent is broad compared with the teacher error
        (``sqrt(v) <~ b sigma``), i.e. exactly where the label is precise; it
        needs ``b sigma > 0`` and raises otherwise.

    ``rule="auto"`` (default) selects per row at ``b sigma / sqrt(v) = 1``, well
    inside both rules' good domain, and the branch never sees an invalid
    argument (no ``inf``/NaN leaks through ``torch.where``). ``nodes="qmc"``
    swaps the fixed Gauss-Hermite nodes for the saved antithetic Sobol set with
    the same Gaussian expectation; that is the hook for a non-Gaussian latent
    (a flow) whose density can be evaluated at the transformed nodes.
    """
    _check_rule(rule, nodes)
    rows = y.shape[0]
    for what, value in (("y", y), ("sigma_quoted", sigma_quoted), ("mu", mu)):
        _check_1d(what, value, rows)
    _check_1d("log_sigma", log_sigma, rows)
    if y.device != mu.device or log_sigma.device != mu.device:
        raise ValueError("all quadrature arguments must share one device")
    if mapping.offset.shape != y.shape:
        raise ValueError("the teacher mapping must have one entry per row")
    sigma = log_sigma.exp()
    if not (torch.isfinite(sigma).all() and (sigma > 0).all()):
        raise ValueError("invalid latent scale; no clipping")
    mean = mapping.mapped_mean(mu)
    scale = mapping.mapped_scale(sigma)
    sigma_eff = mapping.effective_sigma(sigma_quoted)
    if not (torch.isfinite(mean).all() and torch.isfinite(scale).all()):
        raise ValueError("nonfinite mapped latent moments; no clipping")
    if rows == 0:
        return y[:0] * 0.0, QuadratureDiagnostics()
    ratio = scale / sigma_eff
    if rule == "auto":
        use_residual = ratio > 1.0
    else:
        use_residual = torch.full_like(ratio, float(rule == "gh_residual")).bool()
    if rule == "gh_residual" and not bool((scale > 0).all()):
        raise ValueError(
            "the residual rule needs a strictly positive latent scale; "
            "use rule='auto' or 'gh_latent'"
        )
    if nodes == "gh":
        node, weight = gauss_hermite(order, dtype=mu.dtype, device=mu.device)
    else:
        node, weight = fixed_qmc_nodes(
            qmc_count, qmc_seed, dtype=mu.dtype, device=mu.device
        )
    latent_index = (~use_residual).nonzero(as_tuple=False).reshape(-1)
    residual_index = use_residual.nonzero(as_tuple=False).reshape(-1)
    terms = torch.zeros(rows, dtype=mu.dtype, device=mu.device)
    if latent_index.numel():
        branch = _log_marginal_latent(
            y[latent_index],
            mean[latent_index],
            scale[latent_index],
            sigma_eff[latent_index],
            node,
            weight,
        )
        terms = terms.index_copy(0, latent_index, branch)
        latent_resolution = float(ratio[latent_index].detach().max())
    else:
        latent_resolution = float("inf")
    if residual_index.numel():
        branch = _log_marginal_residual(
            y[residual_index],
            mean[residual_index],
            scale[residual_index],
            sigma_eff[residual_index],
            node,
            weight,
        )
        terms = terms.index_copy(0, residual_index, branch)
        residual_resolution = float((1.0 / ratio[residual_index]).detach().max())
    else:
        residual_resolution = float("inf")
    nll = -terms
    if not torch.isfinite(nll).all():
        raise ValueError("nonfinite multi-teacher NLL; no clipping")
    return nll, QuadratureDiagnostics(
        latent_rule_rows=int(latent_index.numel()),
        residual_rule_rows=int(residual_index.numel()),
        min_latent_resolution=latent_resolution,
        min_residual_resolution=residual_resolution,
    )


def gaussian_convolution_nll(
    y: Tensor,
    sigma_quoted: Tensor,
    mu: Tensor,
    log_sigma: Tensor,
    mapping: TeacherMapping,
) -> Tensor:
    """Exact ``-log p(y | x)`` for a Gaussian latent and Gaussian teacher label.

    The convolution of ``N(a + b mu, (b sigma)^2)`` and
    ``N(0, sigma_s^2 + kappa^2 q^2)`` is ``N(a + b mu, (b sigma)^2 + v)``, so
    this is the numerical certificate of :func:`teacher_marginal_nll` (they must
    agree to quadrature error) and the exact objective whenever the latent is
    Gaussian. It is *not* available for a flow latent, which is why the
    quadrature path exists.
    """
    rows = y.shape[0]
    for what, value in (("y", y), ("sigma_quoted", sigma_quoted), ("mu", mu)):
        _check_1d(what, value, rows)
    _check_1d("log_sigma", log_sigma, rows)
    sigma = log_sigma.exp()
    mean = mapping.mapped_mean(mu)
    total_variance = (
        mapping.mapped_scale(sigma).square()
        + mapping.effective_sigma(sigma_quoted).square()
    )
    if not (torch.isfinite(total_variance).all() and (total_variance > 0).all()):
        raise ValueError("invalid marginal variance; no clipping")
    nll = (
        0.5 * (y - mean).square() / total_variance
        + 0.5 * total_variance.log()
        + 0.5 * _LOG_2PI
    )
    if not torch.isfinite(nll).all():
        raise ValueError("nonfinite marginal NLL; no clipping")
    return nll


# --------------------------------------------------------------------------- #
# The multi-teacher loss
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class TeacherNLL:
    """One teacher's contribution: weighted numerator, denominator, link stats.

    ``mean = numerator / weight_sum`` is that teacher's per-observation NLL
    *under its own denominator*; ``weight_sum`` is ``sum_i quality_i`` over its
    observed labels (equal to ``count`` when every quality is one).
    """

    name: str
    numerator: Tensor
    weight_sum: float
    count: int
    quality_zero: int
    quadrature: QuadratureDiagnostics
    offset_mean: float
    slope_mean: float
    intrinsic_sigma_mean: float
    kappa: float

    @property
    def mean(self) -> Tensor:
        if self.weight_sum <= 0.0:
            return self.numerator * 0.0
        return self.numerator / self.weight_sum

    @property
    def mean_value(self) -> float | None:
        return float(self.mean.detach()) if self.weight_sum > 0.0 else None

    @property
    def active(self) -> bool:
        return self.count > 0 and self.weight_sum > 0.0


@dataclass(frozen=True, eq=False)
class MultiTeacherAgeLoss:
    """Multi-teacher age objective plus its per-teacher diagnostics.

    ``numerator`` is ``sum_s w_s * (sum_i q_is nll_is / sum_i q_is)``: a sum of
    per-teacher mean NLLs. ``observation_count`` counts *active teachers*, not
    labels, so ``mean`` is the average per-teacher NLL and a teacher with many
    more labels cannot dominate by count. ``label_count`` is the raw number of
    used ``(teacher, star)`` labels and ``row_count`` the number of stars with at
    least one used label. An empty batch returns a differentiable zero.
    """

    numerator: Tensor
    observation_count: int
    row_count: int
    label_count: int
    teachers: tuple[TeacherNLL, ...]
    teacher_weights: tuple[float, ...]
    reference_teacher: str | None = None
    age_scale: str | None = None

    @property
    def mean(self) -> Tensor:
        return self.numerator / max(1, self.observation_count)

    @property
    def teacher_names(self) -> tuple[str, ...]:
        return tuple(teacher.name for teacher in self.teachers)

    def teacher(self, name: str) -> TeacherNLL:
        """Look up one teacher's contribution by name."""
        for entry in self.teachers:
            if entry.name == name:
                return entry
        raise ValueError(f"unknown teacher {name!r}; have {self.teacher_names}")

    @property
    def quadrature(self) -> QuadratureDiagnostics:
        """Aggregate quadrature diagnostics across teachers."""
        return QuadratureDiagnostics(
            latent_rule_rows=sum(t.quadrature.latent_rule_rows for t in self.teachers),
            residual_rule_rows=sum(
                t.quadrature.residual_rule_rows for t in self.teachers
            ),
            min_latent_resolution=min(
                (t.quadrature.min_latent_resolution for t in self.teachers),
                default=float("inf"),
            ),
            min_residual_resolution=min(
                (t.quadrature.min_residual_resolution for t in self.teachers),
                default=float("inf"),
            ),
        )

    def diagnostics(self) -> dict[str, object]:
        """Manifest-ready loss diagnostics (per-teacher denominators included)."""
        return {
            "evidence_status": MULTI_TEACHER_EVIDENCE_STATUS,
            "coordinate": AGE_COORDINATE,
            "loss_denominator": "per_teacher_observation_mean",
            "partial_label_policy": "per_teacher_mask; no imputation",
            "reference_teacher": self.reference_teacher,
            "age_scale": self.age_scale,
            "n_active_teachers": self.observation_count,
            "n_labels": self.label_count,
            "n_rows_with_label": self.row_count,
            "mean_nll": float(self.mean.detach()),
            "quadrature": {
                "latent_rule_rows": self.quadrature.latent_rule_rows,
                "residual_rule_rows": self.quadrature.residual_rule_rows,
                "min_latent_resolution": self.quadrature.min_latent_resolution,
                "min_residual_resolution": self.quadrature.min_residual_resolution,
            },
            "teachers": [
                {
                    "teacher": teacher.name,
                    "n_labels": teacher.count,
                    "n_quality_zero": teacher.quality_zero,
                    "weight_sum": teacher.weight_sum,
                    "mean_nll": teacher.mean_value,
                    "offset_mean": teacher.offset_mean,
                    "slope_mean": teacher.slope_mean,
                    "intrinsic_sigma": teacher.intrinsic_sigma_mean,
                    "kappa": teacher.kappa,
                    "latent_rule_rows": teacher.quadrature.latent_rule_rows,
                    "residual_rule_rows": teacher.quadrature.residual_rule_rows,
                }
                for teacher in self.teachers
            ],
        }


def multi_teacher_age_loss(
    mu: Tensor,
    log_sigma: Tensor,
    teacher_features: Tensor,
    labels: TeacherLabelBatch,
    calibrations: Mapping[str, TeacherCalibration],
    *,
    teacher_weights: Sequence[float] | None = None,
    rule: str = "auto",
    order: int = DEFAULT_QUADRATURE_ORDER,
    nodes: str = "gh",
    qmc_count: int = DEFAULT_QMC_COUNT,
    qmc_seed: int = DEFAULT_QMC_SEED,
    reference_teacher: str | None = None,
) -> MultiTeacherAgeLoss:
    """Multi-teacher latent-age likelihood with per-teacher denominators.

    ``mu``/``log_sigma`` (``[row]``) are the latent age distribution
    ``N(tau | mu, exp(log_sigma)^2)`` in ``tau``; ``teacher_features`` are the
    physical features ``a_s, b_s, sigma_s`` depend on; ``labels`` is the sparse
    teacher table; ``calibrations`` maps every teacher name to its
    :class:`TeacherCalibration`. Each observed ``(teacher, star)`` label enters
    its teacher's own Gaussian marginal likelihood (see
    :func:`teacher_marginal_nll`), the per-teacher means are summed with the
    declared ``teacher_weights``, and no teacher value is ever averaged with
    another: the joint term for a star with two teachers is the *product* of the
    two likelihoods given the shared latent.

    ``reference_teacher`` is bookkeeping for the diagnostics (the named scale);
    the gauge itself lives in the calibrations, whose reference link is frozen to
    ``a = 0, b = 1`` by :meth:`TeacherCalibration.freeze_identity`.

    ``mu`` is meant to be *amortized* (the network's ``p_theta(tau|x)``). Handing
    this functional loss an independent free latent per star instead makes the
    per-star latents absorb the label noise, which biases ``sigma_s`` and
    ``kappa_s`` low (the Neyman-Scott incidental-parameter effect) unless every
    star has many labels; the shared latent is what keeps those variance
    components identified.
    """
    if not isinstance(labels, TeacherLabelBatch):
        raise TypeError("labels must be a TeacherLabelBatch")
    rows = labels.n_rows
    _check_1d("mu", mu, rows)
    _check_1d("log_sigma", log_sigma, rows)
    feature_names = calibrations_feature_names(calibrations)
    _check_features(teacher_features, feature_names, what="teacher_features")
    if teacher_features.shape[0] != rows:
        raise ValueError("teacher_features must have one row per label row")
    for name in labels.teacher_names:
        if name not in calibrations:
            raise ValueError(f"no calibration supplied for teacher {name!r}")
    weights: tuple[float, ...]
    if teacher_weights is None:
        weights = tuple(1.0 for _ in labels.teacher_names)
    else:
        weights = tuple(float(value) for value in teacher_weights)
        if len(weights) != labels.n_teachers:
            raise ValueError("teacher_weights must align with teacher_names")
        if any(not math.isfinite(value) or value < 0.0 for value in weights):
            raise ValueError("teacher weights must be finite and non-negative")

    sigma = log_sigma.exp()
    if not (torch.isfinite(sigma).all() and (sigma > 0).all()):
        raise ValueError("invalid latent scale; no clipping")

    entries: list[TeacherNLL] = []
    numerator = _zero_like(mu, log_sigma)
    observation_count = 0
    label_count = 0
    row_has_label = torch.zeros(rows, dtype=torch.bool, device=mu.device)
    for column, name in enumerate(labels.teacher_names):
        calibration = calibrations[name]
        observed = labels.observed[column]
        quality = labels.quality[column]
        usable = observed & (quality > 0)
        count = int(observed.sum())
        quality_zero = int((observed & (quality <= 0)).sum())
        zero = _zero_like(mu, log_sigma)
        if not bool(usable.any()):
            entries.append(
                TeacherNLL(
                    name=name,
                    numerator=zero,
                    weight_sum=0.0,
                    count=count,
                    quality_zero=quality_zero,
                    quadrature=QuadratureDiagnostics(),
                    offset_mean=0.0,
                    slope_mean=0.0,
                    intrinsic_sigma_mean=0.0,
                    kappa=0.0,
                )
            )
            continue
        index = usable.nonzero(as_tuple=False).reshape(-1)
        y = labels.tau[column][usable]
        quoted = labels.sigma_quoted[column][usable]
        row_quality = quality[usable]
        features = teacher_features[usable]
        mapping = calibration(features)
        nll, quadrature = teacher_marginal_nll(
            y,
            quoted,
            mu[usable],
            log_sigma[usable],
            mapping,
            rule=rule,
            order=order,
            nodes=nodes,
            qmc_count=qmc_count,
            qmc_seed=qmc_seed,
        )
        weight_sum = float(row_quality.sum())
        weighted = (row_quality * nll).sum()
        entries.append(
            TeacherNLL(
                name=name,
                numerator=weighted,
                weight_sum=weight_sum,
                count=count,
                quality_zero=quality_zero,
                quadrature=quadrature,
                offset_mean=float(mapping.offset.detach().mean()),
                slope_mean=float(mapping.slope.detach().mean()),
                intrinsic_sigma_mean=float(mapping.intrinsic_sigma.detach().mean()),
                kappa=float(mapping.kappa.detach()[0]),
            )
        )
        numerator = numerator + weights[column] * (weighted / weight_sum)
        if weights[column] > 0.0:
            observation_count += 1
        label_count += int(index.numel())
        row_has_label = row_has_label | usable

    age_scale = None
    if reference_teacher is not None:
        age_scale = labels.age_scales[labels.index_of(reference_teacher)]
    return MultiTeacherAgeLoss(
        numerator=numerator,
        observation_count=observation_count,
        row_count=int(row_has_label.sum()),
        label_count=label_count,
        teachers=tuple(entries),
        teacher_weights=weights,
        reference_teacher=reference_teacher,
        age_scale=age_scale,
    )


def calibrations_feature_names(
    calibrations: Mapping[str, TeacherCalibration],
) -> tuple[str, ...]:
    """Feature names shared by a calibration mapping (validated to agree)."""
    names: tuple[str, ...] | None = None
    for calibration in calibrations.values():
        if not isinstance(calibration, TeacherCalibration):
            raise TypeError("calibrations must map names to TeacherCalibration")
        if names is None:
            names = calibration.feature_names
        elif calibration.feature_names != names:
            raise ValueError("all teacher calibrations must share feature_names")
    if names is None:
        raise ValueError("at least one teacher calibration is required")
    return names


class MultiTeacherAgeHead(nn.Module):
    """Latent Gaussian age head plus one calibration per teacher.

    The latent is :class:`...training.age_heads.AgeGaussianHead` (``mu``,
    ``log sigma`` in ``tau``); it stands in for the plan's ``p_pop(tau|x)`` and
    is fitted by maximizing the teacher-label marginal likelihood, not by
    regressing on one catalogue. Because the integral in
    :func:`teacher_marginal_nll` is a Gaussian convolution for this latent, the
    head's optimum is the exact marginal MLE; a flow latent can replace it at
    inference time through the fixed QMC nodes.

    ``reference_teacher`` (default: the first teacher) defines the named age
    scale and is frozen to ``a = 0, b = 1``; all other links, their intrinsic
    scatters and every ``kappa`` are learned jointly with the latent. The
    quantile head of ``age_heads`` remains the reporting head; this module is the
    training likelihood and correction layer.
    """

    def __init__(
        self,
        teacher_names: Sequence[str],
        input_dim: int = 256,
        *,
        reference_teacher: str | None = None,
        width: int = 128,
        feature_names: Sequence[str] = DEFAULT_TEACHER_FEATURE_NAMES,
        calibration_mode: str = "mlp",
        calibration_hidden: int = 32,
        n_stages: int | None = None,
        initial_intrinsic_sigma: float = 0.1,
        initial_kappa: float = 1.0,
        initial_latent_log_scale: float = 0.0,
        quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
        rule: str = "auto",
        nodes: str = "gh",
        qmc_count: int = DEFAULT_QMC_COUNT,
        qmc_seed: int = DEFAULT_QMC_SEED,
    ):
        super().__init__()
        self.teacher_names = _check_names(tuple(teacher_names), "teacher_names")
        if reference_teacher is None:
            reference_teacher = self.teacher_names[0]
        if reference_teacher not in self.teacher_names:
            raise ValueError(
                f"reference_teacher {reference_teacher!r} is not a declared teacher"
            )
        _check_rule(rule, nodes)
        self.reference_teacher = reference_teacher
        self.feature_names = _check_names(feature_names, "feature_names")
        self.input_dim = input_dim
        self.latent = AgeGaussianHead(
            input_dim=input_dim, width=width, initial_log_scale=initial_latent_log_scale
        )
        self.calibrations = nn.ModuleDict(
            {
                name: TeacherCalibration(
                    name,
                    feature_names=self.feature_names,
                    mode="affine" if name == reference_teacher else calibration_mode,
                    hidden=calibration_hidden,
                    n_stages=n_stages,
                    initial_intrinsic_sigma=initial_intrinsic_sigma,
                    initial_kappa=initial_kappa,
                )
                for name in self.teacher_names
            }
        )
        self.calibrations[reference_teacher].freeze_identity()
        self.quadrature_order = quadrature_order
        self.rule = rule
        self.nodes = nodes
        self.qmc_count = qmc_count
        self.qmc_seed = qmc_seed

    def forward(self, latent_features: Tensor) -> dict[str, Tensor]:
        """Latent predictive ``mean``/``log_scale``/``scale`` in ``tau``."""
        return self.latent(latent_features)

    def teacher_mappings(self, teacher_features: Tensor) -> dict[str, TeacherMapping]:
        """Evaluate every teacher link at the same physical features."""
        _check_features(teacher_features, self.feature_names, what="teacher_features")
        return {
            name: self.calibrations[name](teacher_features)
            for name in self.teacher_names
        }

    def loss(
        self,
        latent_features: Tensor,
        teacher_features: Tensor,
        labels: TeacherLabelBatch,
        *,
        teacher_weights: Sequence[float] | None = None,
    ) -> MultiTeacherAgeLoss:
        """Multi-teacher objective for one batch (see ``multi_teacher_age_loss``)."""
        unknown = [
            name for name in labels.teacher_names if name not in self.teacher_names
        ]
        if unknown:
            raise ValueError(
                f"labels carry teachers {unknown} not declared on this head "
                f"({self.teacher_names})"
            )
        outputs = self(latent_features)
        return multi_teacher_age_loss(
            outputs["mean"],
            outputs["log_scale"],
            teacher_features,
            labels,
            self.calibrations,
            teacher_weights=teacher_weights,
            rule=self.rule,
            order=self.quadrature_order,
            nodes=self.nodes,
            qmc_count=self.qmc_count,
            qmc_seed=self.qmc_seed,
            reference_teacher=self.reference_teacher,
        )

    def diagnostics(
        self,
        teacher_features: Tensor,
        labels: TeacherLabelBatch | None = None,
    ) -> dict[str, object]:
        """Reportable head manifest: gauge, scale, per-teacher links."""
        _check_features(teacher_features, self.feature_names, what="teacher_features")
        age_scale = None
        if labels is not None and self.reference_teacher in labels.teacher_names:
            age_scale = labels.age_scales[labels.index_of(self.reference_teacher)]
        return {
            "evidence_status": MULTI_TEACHER_EVIDENCE_STATUS,
            "coordinate": AGE_COORDINATE,
            "reference_teacher": self.reference_teacher,
            "age_scale": age_scale,
            "gauge": "reference_identity(a=0, b=1)",
            "quadrature": {
                "rule": self.rule,
                "nodes": self.nodes,
                "order": self.quadrature_order,
                "qmc_count": self.qmc_count if self.nodes == "qmc" else None,
                "qmc_seed": self.qmc_seed if self.nodes == "qmc" else None,
            },
            "latent": {
                "family": "gaussian",
                "initial_log_scale": float(self.latent.trunk.output.bias[1].detach()),
            },
            "teachers": [
                self.calibrations[name].diagnostics(teacher_features)
                for name in self.teacher_names
            ],
            "n_teacher_parameters": sum(
                parameter.numel()
                for parameter in self.calibrations.parameters()
                if parameter.requires_grad
            ),
        }

    def calibration_parameters(self) -> list[Tensor]:
        """Trainable teacher-link parameters (a separate optimizer group)."""
        return [
            parameter
            for parameter in self.calibrations.parameters()
            if parameter.requires_grad
        ]

    def latent_parameters(self) -> list[Tensor]:
        """Trainable latent parameters (a separate optimizer group)."""
        return list(self.latent.parameters())


# --------------------------------------------------------------------------- #
# Cluster coevality
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class ClusterCoevalLoss:
    """Cluster coeval objective with cluster-level denominators.

    ``numerator`` is the mean over present clusters of the per-cluster weighted
    mean squared residual, so every cluster contributes equally regardless of
    membership size and duplicating a member leaves the value unchanged.
    ``cluster_age`` holds the ages actually used (profiled or supplied),
    ``cluster_counts`` the used member counts, and ``profiled`` says whether the
    ages were solved for. ``row_count`` is the number of members used and
    ``observation_count`` the number of clusters.
    """

    numerator: Tensor
    observation_count: int
    row_count: int
    cluster_age: Tensor
    cluster_ids: tuple[int, ...]
    cluster_counts: tuple[int, ...]
    cluster_losses: tuple[float, ...]
    profiled: bool
    reduction: str

    @property
    def mean(self) -> Tensor:
        return self.numerator / max(1, self.observation_count)

    def diagnostics(self) -> dict[str, object]:
        """Manifest-ready cluster diagnostics (per-cluster counts and losses)."""
        return {
            "n_clusters": self.observation_count,
            "n_members": self.row_count,
            "profiled_cluster_age": self.profiled,
            "reduction": self.reduction,
            "cluster_losses": list(self.cluster_losses),
            "cluster_counts": list(self.cluster_counts),
            "residual_rms": float(self.mean.detach().sqrt())
            if self.observation_count
            else None,
        }


class LearnableClusterAges(nn.Module):
    """Learnable latent ages of ``n_clusters`` coeval populations.

    Initialise from the member means of a first pass
    (``from_members``, which detaches) or from a supplied vector. The parameter
    is a plain ``[n_clusters]`` tensor, so it can be optimized alone (cluster
    anchors, A3) or jointly with the age head.
    """

    def __init__(self, n_clusters: int, *, initial_tau: Tensor | None = None):
        super().__init__()
        if isinstance(n_clusters, bool) or not isinstance(n_clusters, int):
            raise ValueError("n_clusters must be an integer")
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        if initial_tau is None:
            initial = torch.zeros(n_clusters, dtype=torch.float64)
        else:
            if initial_tau.shape != (n_clusters,):
                raise ValueError("initial_tau must be [n_clusters]")
            if not torch.isfinite(initial_tau).all():
                raise ValueError("initial_tau must be finite")
            initial = initial_tau.detach().clone()
        self.ages = nn.Parameter(initial)

    @classmethod
    def from_members(
        cls, tau_pred: Tensor, cluster_ids: Tensor, n_clusters: int | None = None
    ) -> LearnableClusterAges:
        """Initialise each age at the (detached) mean of its members."""
        if cluster_ids.ndim != 1 or cluster_ids.shape != tau_pred.shape:
            raise ValueError("cluster_ids must be [row] and match tau_pred")
        valid = cluster_ids >= 0
        if not bool(valid.any()):
            raise ValueError("no member carries a cluster id")
        top = int(cluster_ids.max()) + 1
        if n_clusters is not None and n_clusters < top:
            raise ValueError("n_clusters is smaller than the largest cluster id + 1")
        total = torch.zeros(top, dtype=tau_pred.dtype, device=tau_pred.device)
        counts = torch.zeros(top, dtype=tau_pred.dtype, device=tau_pred.device)
        total = total.index_add(0, cluster_ids[valid], tau_pred[valid].detach())
        counts = counts.index_add(
            0, cluster_ids[valid], torch.ones_like(tau_pred[valid].detach())
        )
        if not bool((counts > 0).all()):
            raise ValueError("cluster ids must be contiguous from zero")
        return cls(
            n_clusters if n_clusters is not None else top, initial_tau=total / counts
        )

    def per_member(self, cluster_ids: Tensor) -> Tensor:
        """Ages broadcast to members; missing (negative) ids get a placeholder."""
        if cluster_ids.ndim != 1 or cluster_ids.dtype not in (
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("cluster_ids must be a one-dimensional integer tensor")
        index = cluster_ids.clamp(min=0)
        ages = self.ages[index]
        return torch.where(cluster_ids >= 0, ages, torch.zeros_like(ages))


def cluster_coeval_loss(
    tau_pred: Tensor,
    cluster_ids: Tensor,
    cluster_age: Tensor | None = None,
    sigma_cluster: Tensor | float | None = None,
    *,
    tau_sigma: Tensor | None = None,
    selection_weights: Tensor | None = None,
    member_mask: Tensor | None = None,
    reduction: str = "cluster_mean",
) -> ClusterCoevalLoss:
    """Coevality of cluster members around one shared age per cluster.

    ``tau_pred`` are the latent ages (``[row]``, ``tau``), ``cluster_ids`` the
    integer cluster of each row (negative = unassigned/missing, masked with
    ``member_mask``). The shared age is either **supplied** (``cluster_age``,
    ``[n_clusters]`` or a scalar; an ``nn.Parameter`` is differentiable, see
    :class:`LearnableClusterAges`) or **profiled out** when ``cluster_age`` is
    ``None``: the closed-form weighted mean of the members. Members share one
    age - the loss has no per-member age freedom.

    The member weight is ``selection_weight / (tau_sigma^2 + sigma_cluster^2)``
    (unit weight when neither is given), so a published cluster selection
    function and a cluster random effect enter as declared variances rather than
    as an invented floor. ``sigma_cluster`` is ``[n_clusters]`` or scalar; it is
    the *cluster* age uncertainty and is separate from each member's own
    ``tau_sigma``.

    ``reduction="cluster_mean"`` (default) returns the mean over clusters of the
    per-cluster weighted mean squared residual: each cluster counts once, and a
    duplicated member leaves the value unchanged. ``"member_mean"`` returns the
    pooled member mean instead (larger clusters dominate). With fixed member
    variances the per-cluster form is the cluster NLL up to a per-cluster
    constant, which is why it is a declared balanced objective rather than the
    literal likelihood. No clipping: non-finite predictions or non-positive
    weights raise.
    """
    if reduction not in ("cluster_mean", "member_mean"):
        raise ValueError("reduction must be 'cluster_mean' or 'member_mean'")
    if not isinstance(tau_pred, Tensor) or not tau_pred.is_floating_point():
        raise TypeError("tau_pred must be a floating tensor")
    if tau_pred.ndim != 1:
        raise ValueError("tau_pred must be one-dimensional [row]")
    rows = tau_pred.shape[0]
    if not torch.isfinite(tau_pred).all():
        raise ValueError("tau_pred must be finite; no clipping")
    if not isinstance(cluster_ids, Tensor) or cluster_ids.ndim != 1:
        raise ValueError("cluster_ids must be a one-dimensional integer tensor")
    if cluster_ids.shape[0] != rows:
        raise ValueError("cluster_ids must have one entry per row")
    if cluster_ids.is_floating_point():
        raise TypeError(
            "cluster_ids must be an integer tensor (missing = negative); "
            "floating ids (e.g. NaN) are not silently masked"
        )
    valid = cluster_ids >= 0
    if member_mask is not None:
        if (
            member_mask.dtype is not torch.bool
            or member_mask.shape != cluster_ids.shape
        ):
            raise ValueError("member_mask must be a boolean [row] mask")
        valid = valid & member_mask
    selection = None
    if selection_weights is not None:
        if (
            selection_weights.shape != tau_pred.shape
            or not selection_weights.is_floating_point()
        ):
            raise ValueError("selection_weights must be a floating [row] tensor")
        if not (
            torch.isfinite(selection_weights).all() and (selection_weights >= 0).all()
        ):
            raise ValueError("selection weights must be finite and non-negative")
        valid = valid & (selection_weights > 0)
        selection = selection_weights[valid]
    if not bool(valid.any()):
        return ClusterCoevalLoss(
            numerator=_zero_like(tau_pred),
            observation_count=0,
            row_count=0,
            cluster_age=torch.zeros(0, dtype=tau_pred.dtype, device=tau_pred.device),
            cluster_ids=(),
            cluster_counts=(),
            cluster_losses=(),
            profiled=cluster_age is None,
            reduction=reduction,
        )
    ids = cluster_ids[valid]
    n_clusters = int(ids.max()) + 1
    if cluster_age is not None:
        if not isinstance(cluster_age, Tensor) or not cluster_age.is_floating_point():
            raise TypeError("cluster_age must be a floating tensor")
        if cluster_age.ndim == 0:
            ages = cluster_age.expand(n_clusters)
        elif cluster_age.ndim == 1 and cluster_age.shape[0] >= n_clusters:
            ages = cluster_age
            n_clusters = int(cluster_age.shape[0])
        else:
            raise ValueError(
                "cluster_age must be a scalar or [n_clusters]; members of one "
                "cluster share one age"
            )
        if not torch.isfinite(ages).all():
            raise ValueError("cluster_age must be finite")
    else:
        ages = None
    member_tau = tau_pred[valid]
    cluster_variance = torch.zeros(
        n_clusters, dtype=tau_pred.dtype, device=tau_pred.device
    )
    if sigma_cluster is not None:
        sigma = torch.as_tensor(
            sigma_cluster, dtype=tau_pred.dtype, device=tau_pred.device
        )
        if sigma.ndim == 0:
            sigma = sigma.expand(n_clusters)
        elif sigma.ndim != 1 or sigma.shape[0] < n_clusters:
            raise ValueError("sigma_cluster must be a scalar or [n_clusters]")
        if not (torch.isfinite(sigma).all() and (sigma >= 0).all()):
            raise ValueError("sigma_cluster must be finite and non-negative")
        cluster_variance = sigma.square()
    member_variance = cluster_variance[ids]
    if tau_sigma is not None:
        if tau_sigma.shape != tau_pred.shape or not tau_sigma.is_floating_point():
            raise ValueError("tau_sigma must be a floating [row] tensor")
        member_sigma = tau_sigma[valid]
        if not (torch.isfinite(member_sigma).all() and (member_sigma > 0).all()):
            raise ValueError("tau_sigma must be finite and strictly positive")
        member_variance = member_variance + member_sigma.square()
    unit = torch.ones_like(member_tau) if selection is None else selection
    # A member with no declared variance enters with its selection weight alone;
    # the safe denominator keeps the unused branch free of inf/NaN gradients.
    safe_variance = torch.where(
        member_variance > 0, member_variance, torch.ones_like(member_variance)
    )
    weights = torch.where(member_variance > 0, unit / safe_variance, unit)
    if not bool((weights > 0).all()):
        raise ValueError("every used member needs a strictly positive weight")
    counts = torch.zeros(n_clusters, dtype=torch.float64, device=tau_pred.device)
    counts = counts.index_add(
        0, ids, torch.ones(ids.numel(), dtype=torch.float64, device=tau_pred.device)
    )
    present = counts > 0
    present_index = present.nonzero(as_tuple=False).reshape(-1)
    denominator = torch.zeros_like(counts, dtype=tau_pred.dtype).index_add(
        0, ids, weights
    )
    safe_denominator = torch.where(present, denominator, torch.ones_like(denominator))
    if ages is None:
        total = torch.zeros_like(denominator).index_add(0, ids, weights * member_tau)
        ages = torch.where(present, total / safe_denominator, torch.zeros_like(total))
    residual = member_tau - ages[ids]
    squared = weights * residual.square()
    numerator_per_cluster = torch.zeros_like(denominator).index_add(0, ids, squared)
    per_cluster = numerator_per_cluster[present_index] / safe_denominator[present_index]
    if reduction == "cluster_mean":
        numerator = per_cluster.mean()
    else:
        numerator = squared.sum() / weights.sum()
    if not torch.isfinite(numerator):
        raise ValueError("nonfinite cluster coeval loss; no clipping")
    return ClusterCoevalLoss(
        numerator=numerator,
        observation_count=int(present_index.numel()),
        row_count=int(weights.numel()),
        cluster_age=ages,
        cluster_ids=tuple(int(value) for value in present_index.tolist()),
        cluster_counts=tuple(int(counts[i].item()) for i in present_index.tolist()),
        cluster_losses=tuple(
            float(per_cluster[i].detach()) for i in range(per_cluster.numel())
        ),
        profiled=cluster_age is None,
        reduction=reduction,
    )


# --------------------------------------------------------------------------- #
# Predict-time monotone meta-correction
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class MetaCorrectionOutput:
    """A corrected age and the pieces of the monotone map that produced it."""

    tau_corrected: Tensor
    offset: Tensor
    slope: Tensor
    epsilon: Tensor
    center: Tensor
    width: Tensor
    target_scale: str


@dataclass(frozen=True, eq=False)
class MetaCorrectionFit:
    """Outcome of a few-shot meta-correction fit (anchor rows only)."""

    initial_loss: float
    final_loss: float
    steps: int
    anchor_rows_used: int
    anchor_rows_ignored: int
    target_scale: str
    residual_mean: float
    residual_rms: float
    max_abs_correction: float
    slope_min: float
    slope_max: float
    loss_kind: str
    improved: bool

    def diagnostics(self) -> dict[str, object]:
        """Manifest-ready fit report."""
        return {
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "steps": self.steps,
            "anchor_rows_used": self.anchor_rows_used,
            "anchor_rows_ignored": self.anchor_rows_ignored,
            "target_scale": self.target_scale,
            "residual_mean": self.residual_mean,
            "residual_rms": self.residual_rms,
            "max_abs_correction": self.max_abs_correction,
            "slope_min": self.slope_min,
            "slope_max": self.slope_max,
            "loss_kind": self.loss_kind,
            "improved": self.improved,
            "fit_scope": "meta_correction_parameters_only; anchor_rows_only",
        }


class MetaCorrection(nn.Module):
    """Monotone correction ``tau_corr = tau_base + g_phi(tau_base, x, scale)``.

    Family (monotone **by construction**, not by penalty)::

        u     = (tau - m(x)) / w
        tau_c = c(x) + s(x) [ (tau - m(x)) + w eps(x) log cosh(u) ]
        dtau_c/dtau = s(x) [ 1 + eps(x) tanh(u) ]

    with ``s = exp(.) > 0`` and ``eps = tanh(.)`` in ``(-1, 1)``, so the slope
    lies in ``s(1 - |eps|, s(1 + |eps|))`` and is strictly positive for every
    row. ``eps = 0`` is an exact affine map ``c + s(tau - m)`` and the
    initialization ``c = m = eps = 0, s = 1`` is the identity, so an untrained
    module never moves an age. The ``log cosh`` term is a smooth symmetric
    stretch around ``m(x)``: young and old ages are pushed apart (or together)
    by an amount that grows away from the pivot, which is the shape the measured
    teacher offsets show (small at 0-4 Gyr, large and age-growing above 8 Gyr).

    ``target_scale`` selects one parameter block, so a single module can hold
    the correction onto several named scales. ``fit`` optimizes **only these
    parameters**, on **only the rows it is given** (``anchor_mask`` filters
    first), which is the few-shot A2 hook: a new survey re-fits ``g_phi`` and
    nothing else. ``apply`` validates shapes and features and is the only entry
    point a prediction path needs.
    """

    def __init__(
        self,
        feature_dim: int,
        target_scales: Sequence[str] = (DEFAULT_TARGET_SCALE,),
        *,
        hidden: int = 32,
        mode: str = "mlp",
        initial_width: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int):
            raise ValueError("feature_dim must be an integer")
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        if mode not in ("affine", "mlp"):
            raise ValueError("mode must be 'affine' or 'mlp'")
        if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
            raise ValueError("hidden must be a positive integer")
        self.feature_dim = feature_dim
        self.target_scales = _check_names(tuple(target_scales), "target_scales")
        self.mode = mode
        self.initial_width = _require_positive("initial_width", initial_width)
        options = {"dtype": dtype, "device": device}
        n_scales = len(self.target_scales)
        self.log_width = nn.Parameter(
            torch.full((n_scales,), math.log(self.initial_width), **options)
        )
        if mode == "affine":
            self.offset = nn.Parameter(torch.zeros(n_scales, **options))
            self.log_slope = nn.Parameter(torch.zeros(n_scales, **options))
            self.center = nn.Parameter(torch.zeros(n_scales, **options))
            self.raw_epsilon = nn.Parameter(torch.zeros(n_scales, **options))
        else:
            self.trunks = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(feature_dim, hidden, **options),
                    nn.GELU(),
                    nn.Linear(hidden, 4, **options),
                )
                for _ in range(n_scales)
            )
            for trunk in self.trunks:
                nn.init.zeros_(trunk[-1].weight)
                nn.init.zeros_(trunk[-1].bias)

    # -- helpers ------------------------------------------------------------ #
    def scale_index(self, target_scale: str) -> int:
        """Index of a declared target scale; unknown names raise."""
        if target_scale not in self.target_scales:
            raise ValueError(
                f"unknown target_scale {target_scale!r}; declared {self.target_scales}"
            )
        return self.target_scales.index(target_scale)

    def _check_inputs(
        self, tau_base: Tensor, features: Tensor, target_scale: str
    ) -> int:
        index = self.scale_index(target_scale)
        if not isinstance(tau_base, Tensor) or not tau_base.is_floating_point():
            raise TypeError("tau_base must be a floating tensor")
        if tau_base.ndim != 1:
            raise ValueError("tau_base must be one-dimensional [row]")
        if not torch.isfinite(tau_base).all():
            raise ValueError("tau_base must be finite; no clipping")
        if not isinstance(features, Tensor) or not features.is_floating_point():
            raise TypeError("features must be a floating tensor")
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(f"features must be [row, {self.feature_dim}]")
        if features.shape[0] != tau_base.shape[0]:
            raise ValueError("features and tau_base must share the row axis")
        if not torch.isfinite(features).all():
            raise ValueError("features must be finite")
        return index

    def _raw(
        self, features: Tensor, index: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.mode == "affine":
            rows = features.shape[0]
            return (
                self.offset[index].expand(rows),
                self.log_slope[index].expand(rows),
                self.center[index].expand(rows),
                self.raw_epsilon[index].expand(rows),
            )
        raw = self.trunks[index](features)
        return raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]

    def forward(
        self, tau_base: Tensor, features: Tensor, target_scale: str
    ) -> MetaCorrectionOutput:
        """Evaluate the monotone correction (see the class docstring)."""
        index = self._check_inputs(tau_base, features, target_scale)
        log_offset, log_slope, center, raw_epsilon = self._raw(features, index)
        slope = log_slope.exp()
        width = self.log_width[index].exp()
        epsilon = torch.tanh(raw_epsilon)
        if not (
            torch.isfinite(slope).all()
            and torch.isfinite(width)
            and width > 0
            and torch.isfinite(epsilon).all()
        ):
            raise ValueError("nonfinite meta-correction parameters; no clipping")
        u = (tau_base - center) / width
        log_cosh = torch.logaddexp(u, -u) - math.log(2.0)
        corrected = log_offset + slope * (
            (tau_base - center) + width * epsilon * log_cosh
        )
        local_slope = slope * (1.0 + epsilon * torch.tanh(u))
        if not (torch.isfinite(corrected).all() and torch.isfinite(local_slope).all()):
            raise ValueError("nonfinite meta-corrected age; no clipping")
        if not (local_slope > 0).all():
            raise ValueError("the correction must stay strictly monotone in tau")
        return MetaCorrectionOutput(
            tau_corrected=corrected,
            offset=corrected - tau_base,
            slope=local_slope,
            epsilon=epsilon,
            center=center,
            width=width.expand(tau_base.shape[0]),
            target_scale=target_scale,
        )

    def apply(self, tau_base: Tensor, features: Tensor, target_scale: str) -> Tensor:
        """Corrected ``tau`` only, with the same validation as ``forward``."""
        return self(tau_base, features, target_scale).tau_corrected

    def diagnostics(
        self, tau_base: Tensor, features: Tensor, target_scale: str
    ) -> dict[str, object]:
        """Reportable correction summary: slope bounds and offset range."""
        output = self(tau_base, features, target_scale)
        offset = output.offset.detach()
        return {
            "target_scale": target_scale,
            "mode": self.mode,
            "n_rows": int(tau_base.shape[0]),
            "offset_min": float(offset.min()) if offset.numel() else 0.0,
            "offset_mean": float(offset.mean()) if offset.numel() else 0.0,
            "offset_max": float(offset.max()) if offset.numel() else 0.0,
            "slope_min": float(output.slope.detach().min()) if offset.numel() else 0.0,
            "slope_max": float(output.slope.detach().max()) if offset.numel() else 0.0,
            "monotone": bool((output.slope.detach() > 0).all()),
            "width": float(
                self.log_width[self.scale_index(target_scale)].exp().detach()
            ),
            "identity_max_abs": float(offset.abs().max()) if offset.numel() else 0.0,
        }

    def fit(
        self,
        tau_base: Tensor,
        features: Tensor,
        target_tau: Tensor,
        *,
        target_scale: str,
        anchor_mask: Tensor | None = None,
        steps: int = 400,
        lr: float = 0.05,
        loss: str = "mse",
        weights: Tensor | None = None,
    ) -> MetaCorrectionFit:
        """Few-shot fit of ``g_phi`` on anchor rows only (deterministic Adam).

        Rows are filtered by ``anchor_mask`` **before** any arithmetic, so a
        test row in the same tensor cannot influence the result: fitting with a
        mask over a full table and fitting on the pre-subsetted table give
        identical parameters. Only this module's parameters are optimized; the
        base model is never touched. ``tau_base`` is the frozen base prediction,
        ``target_tau`` the anchor's reference age (on the target scale) in
        ``tau`` - no catalogue is treated as truth, the anchor is simply the
        scale being mapped onto.
        """
        self._check_inputs(tau_base, features, target_scale)
        if not isinstance(target_tau, Tensor) or target_tau.ndim != 1:
            raise ValueError("target_tau must be a one-dimensional [row] tensor")
        if target_tau.shape[0] != tau_base.shape[0]:
            raise ValueError("target_tau must match tau_base rows")
        if not torch.isfinite(target_tau).all():
            raise ValueError("target_tau must be finite; no clipping")
        if loss not in ("mse", "huber"):
            raise ValueError("loss must be 'mse' or 'huber'")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive")
        available = int(tau_base.shape[0])
        if anchor_mask is None:
            used = torch.ones(available, dtype=torch.bool, device=tau_base.device)
        else:
            if (
                anchor_mask.dtype is not torch.bool
                or anchor_mask.shape != tau_base.shape
            ):
                raise ValueError("anchor_mask must be a boolean [row] mask")
            used = anchor_mask
        rows = int(used.sum())
        if rows == 0:
            raise ValueError("no anchor rows selected; refusing an empty fit")
        base = tau_base[used]
        target = target_tau[used]
        anchor_features = features[used]
        row_weights = None
        if weights is not None:
            if weights.shape != tau_base.shape or not weights.is_floating_point():
                raise ValueError("weights must be a floating [row] tensor")
            row_weights = weights[used]
            if not (torch.isfinite(row_weights).all() and (row_weights >= 0).all()):
                raise ValueError("weights must be finite and non-negative")

        def objective() -> Tensor:
            output = self(base, anchor_features, target_scale)
            residual = output.tau_corrected - target
            if loss == "mse":
                per_row = residual.square()
            else:
                per_row = torch.nn.functional.smooth_l1_loss(
                    output.tau_corrected, target, reduction="none"
                )
            if row_weights is None:
                return per_row.mean()
            total = row_weights.sum()
            if not bool(total > 0):
                raise ValueError("row weights must not all be zero")
            return (row_weights * per_row).sum() / total

        parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise ValueError("MetaCorrection has no trainable parameters")
        optimizer = torch.optim.Adam(parameters, lr=lr)
        initial = float(objective().detach())
        final = initial
        for _ in range(steps):
            optimizer.zero_grad()
            value = objective()
            value.backward()
            optimizer.step()
            final = float(value.detach())
        with torch.no_grad():
            output = self(base, anchor_features, target_scale)
            residual = output.tau_corrected - target
            offset = output.offset
        return MetaCorrectionFit(
            initial_loss=initial,
            final_loss=final,
            steps=steps,
            anchor_rows_used=rows,
            anchor_rows_ignored=available - rows,
            target_scale=target_scale,
            residual_mean=float(residual.mean()),
            residual_rms=float(residual.square().mean().sqrt()),
            max_abs_correction=float(offset.abs().max()) if offset.numel() else 0.0,
            slope_min=float(output.slope.min()) if offset.numel() else 0.0,
            slope_max=float(output.slope.max()) if offset.numel() else 0.0,
            loss_kind=loss,
            improved=final <= initial,
        )


# --------------------------------------------------------------------------- #
# Systematic-variance decomposition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class SystematicVariance:
    """Additive variance budget with the plan's reporting labels.

    ``total = model + within_teacher + between_teacher + scale`` in exactly that
    order, so the components add exactly (no residual term, no clipping).
    ``fractions`` are only defined when the total is positive; a zero total
    yields NaN fractions rather than a silent division guard. The between-
    teacher term is the **systematic envelope**, not coverage of truth.
    """

    labels: tuple[str, ...]
    components: tuple[Tensor, ...]
    total: Tensor
    total_sigma: Tensor
    fractions: tuple[Tensor, ...]

    @property
    def model(self) -> Tensor:
        return self.components[0]

    @property
    def within_teacher(self) -> Tensor:
        return self.components[1]

    @property
    def between_teacher(self) -> Tensor:
        return self.components[2]

    @property
    def scale(self) -> Tensor:
        return self.components[3]

    def as_dict(self) -> dict[str, Tensor]:
        """Component name -> variance, plus ``total``/``total_sigma``/fractions."""
        report = dict(zip(self.labels, self.components, strict=True))
        report["total"] = self.total
        report["total_sigma"] = self.total_sigma
        for label, fraction in zip(self.labels, self.fractions, strict=True):
            report[f"frac_{label}"] = fraction
        return report

    def as_report(self) -> list[dict[str, float | str | None]]:
        """Rows ready for ``age.csv``: label, variance, sigma, fraction.

        Requires a scalar decomposition (one star or already reduced); a
        per-star tensor must be reduced by the caller and read through
        :meth:`as_dict`. Nothing is silently averaged here.
        """
        if self.total.ndim != 0:
            raise ValueError(
                "as_report needs a scalar variance decomposition; reduce the "
                "per-star tensors first (as_dict exposes them unreduced)"
            )
        rows: list[dict[str, float | str | None]] = []
        for label, value, fraction in zip(
            self.labels, self.components, self.fractions, strict=True
        ):
            rows.append(
                {
                    "component": label,
                    "variance": float(value),
                    "sigma": float(value.sqrt()),
                    "fraction": None
                    if bool(torch.isnan(fraction))
                    else float(fraction),
                }
            )
        rows.append(
            {
                "component": "total",
                "variance": float(self.total),
                "sigma": float(self.total_sigma),
                "fraction": 1.0 if bool(self.total > 0) else None,
            }
        )
        return rows


def systematic_variance_decomposition(
    model_var: Tensor | float,
    within_teacher_var: Tensor | float,
    between_teacher_var: Tensor | float,
    scale_var: Tensor | float,
    *,
    labels: Sequence[str] = SYSTEMATIC_VARIANCE_LABELS,
) -> SystematicVariance:
    """``Var(age|x) = Var_model + Var_within + Var_between + Var_scale``.

    Each component must be finite and non-negative: a negative variance is a
    modelling error and raises instead of being clipped, and a NaN is never
    silently dropped. Components broadcast against each other; the returned
    ``total`` is their sum in the declared order, so ``total`` equals the sum of
    the returned components exactly. The between-teacher term is reported as a
    **systematic envelope**; nothing here claims coverage of a true age.
    """
    components = []
    names = tuple(labels)
    if len(names) != len(SYSTEMATIC_VARIANCE_LABELS):
        raise ValueError(
            f"labels must have {len(SYSTEMATIC_VARIANCE_LABELS)} entries "
            "(model, within-teacher, between-teacher, scale)"
        )
    for label, value in zip(
        names,
        (model_var, within_teacher_var, between_teacher_var, scale_var),
        strict=True,
    ):
        tensor = torch.as_tensor(value)
        if not tensor.is_floating_point():
            if tensor.is_complex():
                raise TypeError(f"{label} must be real")
            tensor = tensor.to(torch.float64)
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{label} must be finite; no clipping")
        if not (tensor >= 0).all():
            raise ValueError(f"{label} is negative; no clipping of variances")
        components.append(tensor)
    broadcast = torch.broadcast_tensors(*components)
    model, within, between, scale = broadcast
    total = model + within + between + scale
    total_sigma = total.sqrt()
    positive = total > 0
    safe_total = torch.where(positive, total, torch.ones_like(total))
    fractions = tuple(
        torch.where(
            positive, component / safe_total, torch.full_like(total, float("nan"))
        )
        for component in broadcast
    )
    for component in (*broadcast, total, total_sigma):
        if not torch.isfinite(component).all():
            raise ValueError("nonfinite variance decomposition; no clipping")
    return SystematicVariance(
        labels=names,
        components=tuple(broadcast),
        total=total,
        total_sigma=total_sigma,
        fractions=fractions,
    )
