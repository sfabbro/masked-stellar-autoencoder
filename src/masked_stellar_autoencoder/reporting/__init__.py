"""Reporting contract C6: metrics, exact conformal calibration, schema and strict I/O.

This package implements ``docs/plan-20261022/TRAINING_PLAN.md`` section 9 item C6
and ``05-heads-losses-metrics.md`` section 7.1:

* :mod:`~masked_stellar_autoencoder.reporting.schema` -- versioned row schemas
  for ``metrics.csv``/``joint.csv``/``distance.csv``/``age.csv``, the
  ``null_reason`` enum and the ``prediction-certificate.json`` contract;
* :mod:`~masked_stellar_autoencoder.reporting.metrics` -- point errors, proper
  scores (NLL/CRPS/energy/variogram/interval), PIT/rank diagnostics, marginal
  and Mondrian coverage, grouped two-way bootstrap and multiplicity helpers;
* :mod:`~masked_stellar_autoencoder.reporting.calibration` -- exact split
  conformal rank, joint rank-score regions (primary) and Bonferroni boxes
  (secondary), Mondrian wrapper and region-size helpers;
* :mod:`~masked_stellar_autoencoder.reporting.io` -- validating writers/readers
  that never emit a bare ``NaN`` and never erase a failed verdict;
* :mod:`~masked_stellar_autoencoder.reporting.strata` -- pre-registered
  Teff/logg stage strata, explicit-data tier strata, minimum-n Mondrian
  calibration with adjacent-stage pooling, and per-stratum coverage/width;
* :mod:`~masked_stellar_autoencoder.reporting.theory` -- model-family theory
  dispersion/envelope (a sensitivity bound, not coverage) and the
  theory-scaled conformal score with a cross-fitted ``gamma`` on calibration
  rows only.

All functions are NumPy/float64 and checkpoint-free.  ``density_kind`` is
mandatory on every proper-score call: intrinsic and observed-label densities are
different estimands and are never mixed in one aggregator.
"""

from __future__ import annotations

from . import calibration, io, metrics, schema, strata, theory
from .calibration import (
    InsufficientCalibrationError,
    bonferroni_box_region,
    calibrate_mondrian_intervals,
    ellipsoid_region_volume,
    exact_quantile,
    finite_sample_coverage_floor,
    finite_sample_rank,
    joint_rank_score_region,
    mahalanobis_scores,
    mondrian_conformal,
    mondrian_membership,
    neg_log_p_scores,
    region_volume,
    split_conformal_interval,
)
from .io import (
    ContractError,
    NonFiniteValueError,
    NullReasonError,
    build_row,
    read_certificate,
    read_csv,
    read_json,
    validate_row,
    validate_rows,
    write_certificate,
    write_contract_directory,
    write_csv,
    write_json,
    write_schema_json,
)
from .metrics import (
    MIN_COUNT,
    benjamini_hochberg,
    bonferroni,
    coverage,
    coverage_calibration_slope,
    crps_from_samples,
    crps_gaussian,
    energy_score,
    grouped_paired_bootstrap,
    interval_score,
    mondrian_coverage,
    multiplicity_note,
    nll_from_log_prob,
    nll_gaussian,
    pit_diagnostics,
    pit_from_samples,
    point_metrics,
    rank_histogram,
    variogram_score,
)
from .schema import (
    CALIBRATION_KINDS,
    CONTRACT_ID,
    NULL_REASONS,
    ROW_SCHEMAS,
    SCHEMA_VERSION,
    TARGET_ORDER,
    TARGET_TRANSFORMS,
    TARGET_UNITS,
    NullReason,
    build_certificate,
    certificate_template,
    merge_verdicts,
    record_verdict,
    schema_document,
)
from .strata import (
    STAGE_BOUNDARIES,
    STAGE_DESCRIPTIONS,
    STAGE_ORDER,
    STRATUM_SEPARATOR,
    TIER_ORDER,
    MondrianCalibrator,
    compose_strata,
    stage_stratum,
    stratum_metrics,
    tier_stratum,
)
from .theory import (
    CALIBRATION_KIND,
    THEORY_MODEL_ORDER,
    THEORY_SCORE_KIND,
    TheoryScaledScore,
    theory_dispersion,
    theory_envelope,
)

__all__ = [
    "CALIBRATION_KIND",
    "CALIBRATION_KINDS",
    "CONTRACT_ID",
    "ContractError",
    "InsufficientCalibrationError",
    "MIN_COUNT",
    "MondrianCalibrator",
    "NULL_REASONS",
    "NonFiniteValueError",
    "NullReason",
    "NullReasonError",
    "ROW_SCHEMAS",
    "SCHEMA_VERSION",
    "STAGE_BOUNDARIES",
    "STAGE_DESCRIPTIONS",
    "STAGE_ORDER",
    "STRATUM_SEPARATOR",
    "TARGET_ORDER",
    "TARGET_TRANSFORMS",
    "TARGET_UNITS",
    "THEORY_MODEL_ORDER",
    "THEORY_SCORE_KIND",
    "TIER_ORDER",
    "TheoryScaledScore",
    "benjamini_hochberg",
    "bonferroni",
    "bonferroni_box_region",
    "build_certificate",
    "build_row",
    "calibrate_mondrian_intervals",
    "calibration",
    "certificate_template",
    "compose_strata",
    "coverage",
    "coverage_calibration_slope",
    "crps_from_samples",
    "crps_gaussian",
    "ellipsoid_region_volume",
    "energy_score",
    "exact_quantile",
    "finite_sample_coverage_floor",
    "finite_sample_rank",
    "grouped_paired_bootstrap",
    "interval_score",
    "io",
    "joint_rank_score_region",
    "mahalanobis_scores",
    "merge_verdicts",
    "metrics",
    "mondrian_conformal",
    "mondrian_coverage",
    "mondrian_membership",
    "multiplicity_note",
    "neg_log_p_scores",
    "nll_from_log_prob",
    "nll_gaussian",
    "pit_diagnostics",
    "pit_from_samples",
    "point_metrics",
    "rank_histogram",
    "read_certificate",
    "read_csv",
    "read_json",
    "record_verdict",
    "region_volume",
    "schema",
    "schema_document",
    "split_conformal_interval",
    "stage_stratum",
    "strata",
    "stratum_metrics",
    "theory",
    "theory_dispersion",
    "theory_envelope",
    "tier_stratum",
    "validate_row",
    "validate_rows",
    "variogram_score",
    "write_certificate",
    "write_contract_directory",
    "write_csv",
    "write_json",
    "write_schema_json",
]
