"""Versioned reporting contract (C6): row schemas, ``null_reason``, certificate.

This module is the single source of truth for the emitted metric contract of
``docs/plan-20261022/05-heads-losses-metrics.md`` section 7.1.  Every number a
run publishes lives in one of four CSV tables (``metrics``, ``joint``,
``distance``, ``age``) plus one ``prediction-certificate.json``.  A cell is
either a real value or ``null`` *with an explicit reason*: a bare ``NaN``/
``Infinity`` is a contract violation, not a missing-value convention.

Design constraints:

* no torch/scipy import -- the schema is importable in any scoring environment;
* no implicit defaults for evidence: ``evidence_status`` must be stated;
* an old failed verdict is never erased (see :func:`merge_verdicts` below).

Terminology follows the plan: ``density_kind`` distinguishes an intrinsic
density from an observed-label density (they are never mixed), and
``loss_denominator`` distinguishes per-coordinate from per-row losses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

SCHEMA_VERSION = "1.1.0"
CONTRACT_ID = "C6"

# 1.1.0 (R1, 2026-10-22): additive *optional* columns for the Mondrian +
# theory-error conformal product -- ``calibration_kind``, ``pooled_strata``,
# ``theory_sigma``, ``gamma`` and ``theory_envelope_low/high``.  No required
# column changed, so a 1.0.0 producer still validates (new cells are null with an
# explicit ``null_reason``).

#: Frozen target order of the plan (``teff, logg, fe_h, s=log d, tau=log age, alpha``).
#: C2 owns the executable ``TargetSpec`` registry; this module only carries the
#: order/units/transforms so a certificate cannot silently reorder labels.
TARGET_ORDER: tuple[str, ...] = ("teff", "logg", "fe_h", "distance", "age", "alpha")
TARGET_UNITS: tuple[str, ...] = ("K", "dex", "dex", "kpc", "Gyr", "dex")
TARGET_TRANSFORMS: tuple[str, ...] = (
    "identity",
    "identity",
    "identity",
    "log_kpc",
    "log_gyr",
    "identity",
)

DENSITY_KINDS: tuple[str, ...] = ("intrinsic", "observed_label")
LOSS_DENOMINATORS: tuple[str, ...] = ("per_coordinate", "per_row")
NOISE_MODEL_KINDS: tuple[str, ...] = ("diag", "full", "none")
PARTIAL_LABEL_POLICIES: tuple[str, ...] = (
    "gaussian_exact_marginal",
    "complete_case",
    "subflow_sum",
)
EVIDENCE_STATUSES: tuple[str, ...] = (
    "verified",
    "producer_only",
    "strict_replay_failed",
)
REPLAY_STATUSES: tuple[str, ...] = EVIDENCE_STATUSES + ("not_replayed",)
BOOTSTRAP_KINDS: tuple[str, ...] = ("star", "block", "two_way", "none")
VERDICT_STATUSES: tuple[str, ...] = (
    "pass",
    "fail",
    "blocked",
    "not_run",
    "not_applicable",
)
CI_KINDS: tuple[str, ...] = ("profile", "predictive")
REF_TYPES: tuple[str, ...] = ("HIP", "cluster", "seismic", "none")
LEVELSET_METHODS: tuple[str, ...] = (
    "analytic_ellipsoid",
    "box",
    "mc_samples",
    "saved_membership",
    "none",
)
CONFORMAL_METHODS: tuple[str, ...] = ("split_rank", "cqr", "none")
#: Which calibration construction produced an interval/region.  ``split_rank``
#: is marginal split conformal, ``mondrian_split_rank`` is the stage/tier
#: Mondrian product and ``theory_scaled_*`` carries a cross-fitted theory-error
#: scale (see ``reporting.theory``).
CALIBRATION_KINDS: tuple[str, ...] = (
    "split_rank",
    "mondrian_split_rank",
    "theory_scaled_split_rank",
    "theory_scaled_mondrian",
    "cqr",
    "none",
)
JOINT_METHODS: tuple[str, ...] = (
    "rank_score_mahalanobis",
    "rank_score_neg_log_p",
    "bonferroni_box",
    "none",
)
DENSITY_INTEGRATION_METHODS: tuple[str, ...] = ("analytic", "gh", "qmc_fixed", "none")
LABEL_UNITS: tuple[str, ...] = ("K", "dex", "kpc", "Gyr", "mas", "dimensionless")
STRATUM_KINDS: tuple[str, ...] = (
    "bulk",
    "feh_lt_minus2",
    "feh_lt_minus3",
    "dwarf",
    "mst",
    "giant",
    "luminous_giant",
    "age_gt_10gyr",
    "xp_on",
    "xp_off",
    "healpix64",
    "survey",
    "custom",
    # R1 stage strata (reporting.strata.stage_stratum).
    "dwarf_ms",
    "turnoff_sgb",
    "rgb",
    "rc",
    "agb_other",
    # R1 tier strata (reporting.strata.tier_stratum).
    "xp_full",
    "xp_partial",
    "no_xp",
    "gaia_only",
)


class NullReason(StrEnum):
    """Why a contract cell is ``null``.  Exactly one reason per row.

    The enum is deliberately *causal*: each value names a distinct failure
    mode, so a null can never be mistaken for a zero, a perfect score, or a
    silently skipped computation.
    """

    #: Metric is undefined for this model family (e.g. CRPS of a Huber point head).
    NOT_APPLICABLE = "not_applicable"
    #: The arm/table version did not emit this metric (older run, partial replay).
    NOT_COMPUTED = "not_computed"
    #: Fewer than ``minimum_count`` rows: descriptive only, values suppressed.
    INSUFFICIENT_N = "insufficient_n"
    #: A stratum or split is empty.
    SPLIT_EMPTY = "split_empty"
    #: Head has no likelihood (point-only) or no calibrated distribution.
    NO_DENSITY = "no_density"
    #: A sample-based score was requested but no samples were saved.
    NO_SAMPLES = "no_samples"
    #: NLL/CRPS would mix estimator families; only same-family comparison is defined.
    FAMILY_MISMATCH = "family_mismatch"
    #: NLL/CRPS would mix ``intrinsic`` and ``observed_label`` densities.
    DENSITY_KIND_MISMATCH = "density_kind_mismatch"
    #: Estimand is not identified from the available labels (e.g. intrinsic
    #: density without a label-noise model).
    NOT_IDENTIFIED = "not_identified"
    #: Interval/region is unbounded (infinite conformal threshold or zero offset).
    UNBOUNDED = "unbounded"
    #: Distance/age product has no calibrated model at all in this round.
    CALIBRATION_MISSING = "calibration_missing"
    #: Score not comparable across unit conventions (variogram across K and dex).
    UNSUPPORTED_UNITS = "unsupported_units"
    #: A previously failed verdict is retained; values were deliberately not
    #: recomputed so the failure stays visible.
    FAILED_VERDICT = "failed_verdict"


NULL_REASONS: tuple[str, ...] = tuple(reason.value for reason in NullReason)


def null_reason_ok(value) -> bool:
    """``True`` when ``value`` is a valid :class:`NullReason` (or its string)."""
    if isinstance(value, NullReason):
        return True
    return isinstance(value, str) and value in NULL_REASONS


@dataclass(frozen=True)
class Field:
    """One column of a contract table.

    ``dtype`` is one of ``str``/``int``/``float``/``bool``.  ``valid_range`` is
    an inclusive ``(low, high)`` pair; ``None`` for either bound means
    unbounded.  ``enum`` restricts string values.  ``required`` means the key
    must be present in every row (it may still be ``null`` if ``nullable``).
    """

    name: str
    dtype: str
    unit: str | None = None
    required: bool = True
    nullable: bool = True
    valid_range: tuple[float | None, float | None] | None = None
    enum: tuple[str, ...] | None = None
    description: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "unit": self.unit,
            "required": self.required,
            "nullable": self.nullable,
            "valid_range": list(self.valid_range) if self.valid_range else None,
            "enum": list(self.enum) if self.enum else None,
            "description": self.description,
        }


def _f(
    name: str, unit: str | None = None, *, rng=None, required=False, desc=""
) -> Field:
    return Field(
        name, "float", unit, required=required, valid_range=rng, description=desc
    )


def _i(name: str, *, rng=None, required=True, desc="") -> Field:
    return Field(
        name, "int", None, required=required, valid_range=rng, description=desc
    )


def _s(
    name: str,
    *,
    enum: tuple[str, ...] | None = None,
    required=True,
    nullable: bool = False,
    desc="",
) -> Field:
    return Field(
        name,
        "str",
        None,
        required=required,
        nullable=nullable,
        enum=enum,
        description=desc,
    )


_PROVENANCE_FIELDS = (
    _s("run_id", desc="immutable run identifier"),
    _s(
        "failure_id",
        required=False,
        nullable=True,
        desc="back-reference to a retained failure",
    ),
    _s(
        "evidence_status",
        enum=EVIDENCE_STATUSES,
        desc="verified | producer_only | strict_replay_failed",
    ),
    Field(
        "verified",
        "bool",
        None,
        required=False,
        nullable=True,
        description="derived: evidence_status == 'verified'",
    ),
    _s(
        "null_reason",
        enum=NULL_REASONS,
        required=True,
        nullable=True,
        desc="why any cell is null",
    ),
    _s(
        "notes",
        required=False,
        nullable=True,
        desc="free text; never a substitute for a metric",
    ),
)

_UNIT_FIELD = _s(
    "unit", enum=LABEL_UNITS, desc="native unit of the label and of the error scores"
)
_STRATUM_FIELD = _s("stratum", desc="pre-registered slice name (see STRATUM_KINDS)")

# R1 optional calibration/theory columns.  They are additive: a producer that
# does not run the Mondrian/theory product leaves them null (and therefore must
# state a null_reason, as for every other optional metric cell).
_CALIBRATION_KIND_FIELD = _s(
    "calibration_kind",
    enum=CALIBRATION_KINDS,
    required=False,
    nullable=True,
    desc="marginal split | Mondrian | theory-scaled conformal construction",
)
_POOLED_STRATA_FIELD = _s(
    "pooled_strata",
    required=False,
    nullable=True,
    desc="comma-joined pooled Mondrian group (includes this stratum); empty if unpooled",
)
_THEORY_SIGMA_FIELD = _f(
    "theory_sigma",
    unit="native",
    rng=(0.0, None),
    required=False,
    desc="mean per-row model-family dispersion sigma_theo in native label units",
)
_GAMMA_FIELD = _f(
    "gamma",
    rng=(0.0, None),
    required=False,
    desc="cross-fitted theory-error scale in sigma_exp + gamma * sigma_theo",
)
_THEORY_ENVELOPE_LOW_FIELD = _f(
    "theory_envelope_low",
    unit="native",
    required=False,
    desc="mean lower bound of the model-family sensitivity envelope (not a CI)",
)
_THEORY_ENVELOPE_HIGH_FIELD = _f(
    "theory_envelope_high",
    unit="native",
    required=False,
    desc="mean upper bound of the model-family sensitivity envelope (not a CI)",
)

METRICS_FIELDS: tuple[Field, ...] = (
    _PROVENANCE_FIELDS[0],
    _s("model_id", desc="model/arm identity inside a run"),
    _s("label", desc="target name in the frozen TARGET_ORDER spelling"),
    _UNIT_FIELD,
    _STRATUM_FIELD,
    _i("n", rng=(0.0, None), desc="rows in this slice"),
    _f("n_eff", desc="effective sample size (n if unweighted)"),
    _f("mae", unit="native", rng=(0.0, None)),
    _f("rmse", unit="native", rng=(0.0, None)),
    _f("bias", unit="native", desc="mean(y_hat - y), signed"),
    _f("nmad", unit="native", rng=(0.0, None), desc="1.4826 * median|r - median(r)|"),
    _f("r2", desc="computed inside the slice; may be negative"),
    _f("atten_slope", desc="OLS slope of y on y_hat (dilution diagnostic)"),
    _f("outlier_frac", rng=(0.0, 1.0), desc="|r| > k * sd(y) in this slice"),
    _f(
        "tail_mae",
        unit="native",
        rng=(0.0, None),
        desc="mean |r| over the worst decile",
    ),
    _f("crps", unit="native", rng=(0.0, None)),
    _f("nll", desc="negative mean log density, natural log"),
    _s(
        "nll_space",
        enum=DENSITY_KINDS,
        required=False,
        nullable=True,
        desc="density_kind of the NLL row",
    ),
    _f("pit_ks", rng=(0.0, 1.0)),
    _f("pit_cvm", rng=(0.0, None)),
    _f(
        "pit_local_w1",
        rng=(0.0, 1.0),
        desc="local PIT W1 on (Teff, logg, FeH, G) features",
    ),
    _f("cov_68", rng=(0.0, 1.0)),
    _f("cov_90", rng=(0.0, 1.0)),
    _f("cov_95", rng=(0.0, 1.0)),
    _f("cov_nominal", rng=(0.0, 1.0), desc="nominal level matching cov_90 (0.9)"),
    _f("width_90", unit="native", rng=(0.0, None), desc="mean 90% interval width"),
    _f("interval_score_90", unit="native", rng=(0.0, None)),
    _CALIBRATION_KIND_FIELD,
    _POOLED_STRATA_FIELD,
    _THEORY_SIGMA_FIELD,
    _GAMMA_FIELD,
    _THEORY_ENVELOPE_LOW_FIELD,
    _THEORY_ENVELOPE_HIGH_FIELD,
    _f("calib_slope", desc="OLS slope of cover-indicator on log width"),
    _f("ci_lo", desc="grouped-bootstrap lower CI bound of the headline error"),
    _f("ci_hi", desc="grouped-bootstrap upper CI bound of the headline error"),
    _s("bootstrap_kind", enum=BOOTSTRAP_KINDS, desc="star | block | two_way | none"),
    _i("bootstrap_reps", rng=(0.0, None), required=False),
    _s(
        "multiplicity",
        desc="stated correction, e.g. 'bonferroni_k=16' or 'bh_fdr_q=0.05'",
    ),
    _PROVENANCE_FIELDS[1],
    _PROVENANCE_FIELDS[2],
    _PROVENANCE_FIELDS[3],
    _PROVENANCE_FIELDS[4],
    _PROVENANCE_FIELDS[5],
)

JOINT_FIELDS: tuple[Field, ...] = (
    _PROVENANCE_FIELDS[0],
    _s("model_id"),
    _s("label_set", desc="joint label tuple, e.g. 'teff+logg+fe_h'"),
    _STRATUM_FIELD,
    _i("n", rng=(0.0, None)),
    _f("joint_cov_90", rng=(0.0, 1.0), desc="all-labels-inside joint coverage at 0.90"),
    _f(
        "joint_cov_90_bonferroni",
        rng=(0.0, 1.0),
        desc="Bonferroni box (alpha/n_labels per label)",
    ),
    _f("joint_cov_90_rank_score", rng=(0.0, 1.0), desc="rank-score region (primary)"),
    _f("region_volume_native", rng=(0.0, None), desc="mean native-unit region volume"),
    _s("levelset_method", enum=LEVELSET_METHODS),
    _CALIBRATION_KIND_FIELD,
    _POOLED_STRATA_FIELD,
    _f("tarp_coverage", rng=(0.0, 1.0), required=False),
    _f("energy_score", rng=(0.0, None)),
    _f("variogram_score", rng=(0.0, None)),
    _s("replay_status", enum=REPLAY_STATUSES),
    _PROVENANCE_FIELDS[1],
    _PROVENANCE_FIELDS[2],
    _PROVENANCE_FIELDS[3],
    _PROVENANCE_FIELDS[4],
    _PROVENANCE_FIELDS[5],
)

DISTANCE_FIELDS: tuple[Field, ...] = (
    _PROVENANCE_FIELDS[0],
    _s("model_id", required=False, nullable=True),
    _s(
        "kind",
        enum=("mas", "kpc"),
        desc="parallax-space (mas) or distance-space (kpc) row",
    ),
    _STRATUM_FIELD,
    _i("n", rng=(0.0, None)),
    _f("bias_mas", unit="mas"),
    _f("mae_mas", unit="mas", rng=(0.0, None)),
    _f(
        "p68_mas",
        unit="mas",
        rng=(0.0, None),
        desc="68th percentile absolute parallax error",
    ),
    _f("mae_logd", unit="dex", rng=(0.0, None)),
    _f("bias_logd", unit="dex"),
    _f("frac_err_q50", rng=(0.0, None)),
    _f("frac_err_q90", rng=(0.0, None)),
    _f("cov90_mas", rng=(0.0, 1.0)),
    _f("cov90_kpc", rng=(0.0, 1.0)),
    _GAMMA_FIELD,
    _CALIBRATION_KIND_FIELD,
    _THEORY_SIGMA_FIELD,
    _THEORY_ENVELOPE_LOW_FIELD,
    _THEORY_ENVELOPE_HIGH_FIELD,
    _f(
        "frac_ci_empty",
        rng=(0.0, 1.0),
        desc="profile CI empty (never cosmetically capped)",
    ),
    _f("frac_ci_upper_unbounded", rng=(0.0, 1.0)),
    _s("ci_kind", enum=CI_KINDS),
    _s("ref_type", enum=REF_TYPES),
    _PROVENANCE_FIELDS[1],
    _PROVENANCE_FIELDS[2],
    _PROVENANCE_FIELDS[3],
    _PROVENANCE_FIELDS[4],
    _PROVENANCE_FIELDS[5],
)

AGE_FIELDS: tuple[Field, ...] = (
    _PROVENANCE_FIELDS[0],
    _s("model_id", required=False, nullable=True),
    _STRATUM_FIELD,
    _i("n", rng=(0.0, None)),
    _f("bias_Gyr", unit="Gyr"),
    _f("mae_Gyr", unit="Gyr", rng=(0.0, None)),
    _f("bias_logGyr", unit="dex"),
    _f("mae_logGyr", unit="dex", rng=(0.0, None)),
    _f("cov68", rng=(0.0, 1.0)),
    _f("cov90", rng=(0.0, 1.0)),
    _f("width90_log", unit="dex", rng=(0.0, None)),
    _GAMMA_FIELD,
    _CALIBRATION_KIND_FIELD,
    _THEORY_SIGMA_FIELD,
    _THEORY_ENVELOPE_LOW_FIELD,
    _THEORY_ENVELOPE_HIGH_FIELD,
    _i("gt10Gyr_n", rng=(0.0, None), required=False),
    _f("gt10Gyr_bias_log", unit="dex"),
    _s(
        "seismic_ref",
        required=False,
        nullable=True,
        desc="external seismic reference tag, never truth",
    ),
    _PROVENANCE_FIELDS[1],
    _PROVENANCE_FIELDS[2],
    _PROVENANCE_FIELDS[3],
    _PROVENANCE_FIELDS[4],
    _PROVENANCE_FIELDS[5],
)

TABLE_FIELDS: dict[str, tuple[Field, ...]] = {
    "metrics": METRICS_FIELDS,
    "joint": JOINT_FIELDS,
    "distance": DISTANCE_FIELDS,
    "age": AGE_FIELDS,
}

TABLE_FILENAMES: dict[str, str] = {
    "metrics": "metrics.csv",
    "joint": "joint.csv",
    "distance": "distance.csv",
    "age": "age.csv",
}

#: Semantic row rules enforced by ``reporting.io.validate_row``.  Kept here so
#: the emitted ``schema.json`` documents what a producer must satisfy.
TABLE_CHECKS: dict[str, tuple[str, ...]] = {
    "metrics": (
        "any_null_requires_null_reason",
        "no_bare_nan_inf",
        "nll_requires_nll_space",
        "cov90_requires_cov_nominal",
        "verified_matches_evidence_status",
        "failed_verdict_requires_failure_id",
    ),
    "joint": (
        "any_null_requires_null_reason",
        "no_bare_nan_inf",
        "verified_matches_evidence_status",
        "failed_verdict_requires_failure_id",
    ),
    "distance": (
        "any_null_requires_null_reason",
        "no_bare_nan_inf",
        "verified_matches_evidence_status",
        "failed_verdict_requires_failure_id",
    ),
    "age": (
        "any_null_requires_null_reason",
        "no_bare_nan_inf",
        "verified_matches_evidence_status",
        "failed_verdict_requires_failure_id",
    ),
}

METRIC_COLUMNS: tuple[str, ...] = tuple(f.name for f in METRICS_FIELDS)
JOINT_COLUMNS: tuple[str, ...] = tuple(f.name for f in JOINT_FIELDS)
DISTANCE_COLUMNS: tuple[str, ...] = tuple(f.name for f in DISTANCE_FIELDS)
AGE_COLUMNS: tuple[str, ...] = tuple(f.name for f in AGE_FIELDS)

#: Every row schema, by name, as ordered column tuples.
ROW_SCHEMAS: dict[str, tuple[str, ...]] = {
    "metrics": METRIC_COLUMNS,
    "joint": JOINT_COLUMNS,
    "distance": DISTANCE_COLUMNS,
    "age": AGE_COLUMNS,
}


# ---------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------

CERTIFICATE_REQUIRED: tuple[str, ...] = (
    "run_id",
    "source_sha",
    "library_commit",
    "checkpoint_sha256",
    "cohort_sha256",
    "scaler_sha256",
    "eval",
    "labels",
    "units",
    "target_transforms",
    "jacobian",
    "density_kind",
    "loss_denominator",
    "noise_model",
    "partial_label_policy",
    "n_observed_per_label",
    "conformal",
    "density_integration",
    "n",
    "gates",
    "verdicts",
    "supersedes",
)

CERTIFICATE_ENUMS: dict[str, tuple[str, ...]] = {
    "density_kind": DENSITY_KINDS,
    "loss_denominator": LOSS_DENOMINATORS,
    "partial_label_policy": PARTIAL_LABEL_POLICIES,
}

EVAL_KEYS: tuple[str, ...] = ("dtype", "device", "batch_size", "torch")


def certificate_template() -> dict[str, Any]:
    """A skeleton certificate with every required key present and empty.

    Values are placeholders (``None``/``{}``); callers must fill them and then
    call :func:`build_certificate`.  This exists so producers never omit a
    contract key by accident.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_ID,
        "run_id": None,
        "source_sha": None,
        "library_commit": None,
        "checkpoint_sha256": None,
        "cohort_sha256": None,
        "scaler_sha256": None,
        "eval": {},
        "labels": list(TARGET_ORDER),
        "units": list(TARGET_UNITS),
        "target_transforms": list(TARGET_TRANSFORMS),
        "jacobian": {
            "form": "per_row_per_coordinate",
            "applied": False,
            "terms": ["-log w", "-log d", "-log age"],
        },
        "density_kind": None,
        "loss_denominator": None,
        "noise_model": {"kind": "none", "source_tagged": False, "kappa": {}},
        "partial_label_policy": None,
        "n_observed_per_label": {},
        "conformal": {
            "method": "none",
            "alpha": None,
            "n_cal": None,
            "order_stat_index": None,
            "finite_sample_exact": False,
            "joint_method": "none",
            "strata": [],
        },
        "density_integration": {
            "method": "none",
            "nodes": None,
            "seed": None,
            "buffer_sha256": None,
            "convergence": {},
        },
        "n": {"fit": 0, "tune": 0, "cal": 0, "test": 0},
        "gates": {},
        "verdicts": {},
        "failed_verdicts": [],
        "supersedes": {},
    }


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"certificate field {name!r} must be a mapping")
    return value


def validate_certificate(doc: Mapping[str, Any]) -> None:
    """Validate a ``prediction-certificate.json`` document in place (read-only).

    Checks presence, types, enums and the provenance invariants: a failed
    verdict is retained in ``failed_verdicts`` and cannot be dropped by a later
    pass.  Raises :class:`ValueError` on the first violation.
    """
    if not isinstance(doc, Mapping):
        raise ValueError("certificate must be a mapping")
    missing = [key for key in CERTIFICATE_REQUIRED if key not in doc]
    if missing:
        raise ValueError(f"certificate missing required keys: {missing}")
    for key, enum in CERTIFICATE_ENUMS.items():
        value = doc[key]
        if value is None:
            continue
        if value not in enum:
            raise ValueError(f"certificate {key}={value!r} not in {enum}")

    labels = doc["labels"]
    units = doc["units"]
    transforms = doc["target_transforms"]
    if not isinstance(labels, list | tuple) or len(labels) != len(TARGET_ORDER):
        raise ValueError(
            f"certificate labels must list exactly {len(TARGET_ORDER)} targets"
        )
    if len(set(map(str, labels))) != len(labels):
        raise ValueError("certificate labels must be unique (no positional inversion)")
    if not isinstance(units, list | tuple) or len(units) != len(labels):
        raise ValueError("certificate units must match labels length")
    if not isinstance(transforms, list | tuple) or len(transforms) != len(labels):
        raise ValueError("certificate target_transforms must match labels length")
    bad = [t for t in transforms if t not in ("identity", "log_kpc", "log_gyr")]
    if bad:
        raise ValueError(f"certificate target_transforms not in the frozen set: {bad}")

    evaluation = _require_mapping(doc["eval"], "eval")
    for key in EVAL_KEYS:
        if key not in evaluation:
            raise ValueError(f"certificate eval missing {key!r}")

    jacobian = _require_mapping(doc["jacobian"], "jacobian")
    if jacobian.get("form") != "per_row_per_coordinate":
        raise ValueError("certificate jacobian.form must be 'per_row_per_coordinate'")
    if "applied" not in jacobian:
        raise ValueError("certificate jacobian.applied is required")
    _require_mapping(doc["noise_model"], "noise_model")

    conformal = _require_mapping(doc["conformal"], "conformal")
    if conformal.get("method") not in CONFORMAL_METHODS:
        raise ValueError(
            f"certificate conformal.method must be one of {CONFORMAL_METHODS}"
        )
    if conformal.get("joint_method") not in JOINT_METHODS:
        raise ValueError(
            f"certificate conformal.joint_method must be one of {JOINT_METHODS}"
        )
    if conformal.get("method") == "split_rank":
        if conformal.get("finite_sample_exact") is not True:
            raise ValueError(
                "split_rank conformal must declare finite_sample_exact=True "
                "(exact order statistic, not a plug-in quantile)"
            )
        if not isinstance(conformal.get("order_stat_index"), int):
            raise ValueError(
                "split_rank conformal must record the integer order_stat_index"
            )
        if not isinstance(conformal.get("n_cal"), int):
            raise ValueError("split_rank conformal must record n_cal")

    integration = _require_mapping(doc["density_integration"], "density_integration")
    if integration.get("method") not in DENSITY_INTEGRATION_METHODS:
        raise ValueError(
            f"certificate density_integration.method must be one of {DENSITY_INTEGRATION_METHODS}"
        )

    counts = _require_mapping(doc["n"], "n")
    for role in ("fit", "tune", "cal", "test"):
        if role not in counts:
            raise ValueError(f"certificate n missing role {role!r}")
        if not isinstance(counts[role], int) or counts[role] < 0:
            raise ValueError(f"certificate n[{role!r}] must be a non-negative integer")

    verdicts = _require_mapping(doc["verdicts"], "verdicts")
    failed = doc.get("failed_verdicts", [])
    if not isinstance(failed, list | tuple):
        raise ValueError("certificate failed_verdicts must be a list")
    for gate, node in verdicts.items():
        if not isinstance(node, Mapping) or "current" not in node:
            raise ValueError(
                f"certificate verdicts[{gate!r}] must be a {{'current','history'}} node"
            )
        current = node["current"]
        if current is not None and current.get("status") not in VERDICT_STATUSES:
            raise ValueError(
                f"certificate verdicts[{gate!r}].current.status is invalid"
            )
        history = node.get("history", [])
        if not isinstance(history, list | tuple):
            raise ValueError(f"certificate verdicts[{gate!r}].history must be a list")
        for entry in history:
            if (
                not isinstance(entry, Mapping)
                or entry.get("status") not in VERDICT_STATUSES
            ):
                raise ValueError(
                    f"certificate verdicts[{gate!r}] has an invalid history entry"
                )
        if current is not None and current.get("status") == "fail":
            if current.get("failure_id") is None:
                raise ValueError(
                    f"certificate verdicts[{gate!r}] failed without a failure_id back-reference"
                )
    for entry in failed:
        if not isinstance(entry, Mapping) or entry.get("status") != "fail":
            raise ValueError(
                "certificate failed_verdicts may only contain failed verdicts"
            )
        if entry.get("failure_id") is None:
            raise ValueError("certificate failed_verdicts entry missing failure_id")


def build_certificate(*, validate: bool = True, **fields: Any) -> dict[str, Any]:
    """Start from :func:`certificate_template`, apply ``fields``, validate.

    Unknown keys are rejected: a certificate is a contract, not a free-form
    bag.  ``failed_verdicts`` is preserved if supplied.
    """
    cert = certificate_template()
    unknown = sorted(set(fields) - set(cert) - {"schema_version", "contract"})
    if unknown:
        raise ValueError(f"certificate got unknown keys: {unknown}")
    cert.update(fields)
    if validate:
        validate_certificate(cert)
    return cert


def record_verdict(
    certificate: dict[str, Any],
    gate: str,
    status: str,
    *,
    evidence_status: str = "producer_only",
    detail: str | None = None,
    failure_id: str | None = None,
) -> dict[str, Any]:
    """Record one gate verdict without ever erasing an earlier failure.

    The previous ``current`` entry is pushed to ``history``; a ``fail`` entry
    is also appended to ``failed_verdicts`` (append-only).  Returns the mutated
    certificate for convenience.
    """
    if status not in VERDICT_STATUSES:
        raise ValueError(f"verdict status must be one of {VERDICT_STATUSES}")
    if evidence_status not in EVIDENCE_STATUSES:
        raise ValueError(f"evidence_status must be one of {EVIDENCE_STATUSES}")
    if status == "fail" and failure_id is None:
        raise ValueError("a failed verdict requires a failure_id back-reference")
    entry = {
        "status": status,
        "evidence_status": evidence_status,
        "detail": detail,
        "failure_id": failure_id,
    }
    node = certificate.setdefault("verdicts", {}).setdefault(
        gate, {"current": None, "history": []}
    )
    if node.get("current") is not None:
        node.setdefault("history", []).append(node["current"])
    node["current"] = entry
    if status == "fail":
        certificate.setdefault("failed_verdicts", []).append(entry)
    return certificate


def merge_verdicts(
    certificate: dict[str, Any], new: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge ``{gate: entry}`` into a certificate, preserving failure history.

    ``new`` maps gate name to either a status string or an entry mapping with
    ``status``/``evidence_status``/``detail``/``failure_id``.  Existing history
    is never dropped.
    """
    for gate, item in new.items():
        if isinstance(item, str):
            record_verdict(certificate, gate, item)
            continue
        payload = dict(item)
        status = payload.pop("status")
        record_verdict(certificate, gate, status, **payload)
    return certificate


def schema_document(*, provenance: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Machine-readable ``schema.json`` for the whole contract.

    Includes the versioned fields of every table, the ``null_reason`` enum, the
    certificate requirements, the gate list and provenance receipts.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_ID,
        "null_reason_enum": list(NULL_REASONS),
        "target_order": list(TARGET_ORDER),
        "target_units": list(TARGET_UNITS),
        "target_transforms": list(TARGET_TRANSFORMS),
        "tables": {
            name: {
                "filename": TABLE_FILENAMES[name],
                "columns": [field.to_json() for field in fields],
            }
            for name, fields in TABLE_FIELDS.items()
        },
        "certificate": {
            "required": list(CERTIFICATE_REQUIRED),
            "eval_keys": list(EVAL_KEYS),
            "enums": {key: list(value) for key, value in CERTIFICATE_ENUMS.items()},
            "verdict_statuses": list(VERDICT_STATUSES),
            "evidence_statuses": list(EVIDENCE_STATUSES),
            "replay_statuses": list(REPLAY_STATUSES),
        },
        "gates": list(VERDICT_STATUSES),
        "provenance": dict(provenance or {}),
    }
