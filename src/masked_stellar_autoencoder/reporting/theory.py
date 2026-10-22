"""Theory-error conformal scores and model-family envelopes (plan item R1).

``docs/plan-20261022/10-rfit-profile-likelihood-and-theoretical-conformal.md``
sections 3--5 recommend three things that this module keeps strictly separate:

1. a **theory dispersion** ``sigma_theo(x)`` from an independent model-family
   ensemble (:func:`theory_dispersion`);
2. a **theory-inflated nonconformity score**
   ``|y - median_prediction| / (sigma_exp + gamma * sigma_theo)`` whose ``gamma``
   is selected by K-fold cross-fitting **on calibration rows only**
   (:class:`TheoryScaledScore`);
3. a **theory envelope** across the tested model family
   (:func:`theory_envelope`) that is a *sensitivity bound*, never a conformal
   interval and never a coverage claim.

Model-family axis
-----------------
``model_predictions`` always has shape ``[n_theory_models, n_rows, ...]``: axis 0
is the model-family axis in the **declared** order
:data:`THEORY_MODEL_ORDER` (``MIST, PARSEC, GARSTEC, BASTI``).  The order must be
fixed before looking at widths; reordering families to shrink an envelope is a
silent analysis change, so duplicate or missing family names raise.

What is *not* claimed
---------------------
* The theory envelope of :func:`theory_envelope` contains every tested model by
  construction, but that is not ``1 - alpha`` coverage of the truth: synthetic
  and real data are not exchangeable and the perturbation family is not a random
  sample of model space (plan section 3, Method 3).
* ``gamma`` is selected by cross-fitting on calibration rows and the conformal
  threshold is then an exact finite-sample rank on the calibration split for
  that frozen ``gamma``; this is the plan's recommended compromise, but it is a
  calibration heuristic, not a conditional-coverage theorem.  Report the
  gamma-sensitivity table alongside the selected value.
* Coverage is evaluated empirically on the test rows against measured/predicted
  labels, never against physical truth.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .calibration import (
    InsufficientCalibrationError,
    exact_quantile,
    finite_sample_rank,
)
from .metrics import MIN_COUNT, descriptive_only, interval_score
from .schema import NullReason
from .strata import stratum_metrics

__all__ = [
    "CALIBRATION_KIND",
    "THEORY_MODEL_ORDER",
    "THEORY_SCORE_KIND",
    "TheoryScaledScore",
    "theory_dispersion",
    "theory_envelope",
]

#: Declared model-family axis order.  Axis 0 of ``model_predictions`` follows it.
THEORY_MODEL_ORDER: tuple[str, ...] = ("MIST", "PARSEC", "GARSTEC", "BASTI")

#: Value used in the optional ``calibration_kind`` contract column.
CALIBRATION_KIND = "theory_scaled_split_rank"

#: Human-readable nonconformity score definition.
THEORY_SCORE_KIND = "abs_residual_over_sigma_exp_plus_gamma_sigma_theo"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def _family_names(
    n_models: int, model_families: Sequence[str] | None
) -> tuple[str, ...]:
    if model_families is None:
        if n_models <= len(THEORY_MODEL_ORDER):
            return THEORY_MODEL_ORDER[:n_models]
        return tuple(f"model_{index}" for index in range(n_models))
    if isinstance(model_families, str):
        raise ValueError("model_families must be a sequence of names, not one string")
    names = tuple(str(name) for name in model_families)
    if len(names) != n_models:
        raise ValueError(
            f"model_families has {len(names)} name(s) for {n_models} theory models"
        )
    if any(not name for name in names):
        raise ValueError("model_families names must be non-empty")
    if len(set(names)) != len(names):
        raise ValueError(
            "model_families names must be unique (no reordered duplicates)"
        )
    return names


def _as_model_predictions(
    model_predictions, model_families: Sequence[str] | None = None
) -> tuple[np.ndarray, tuple[str, ...]]:
    array = np.asarray(model_predictions, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(
            "model_predictions must have shape [n_theory_models, n_rows, ...], "
            f"got {array.shape}"
        )
    n_models, n_rows = array.shape[0], array.shape[1]
    if n_models < 2:
        raise ValueError(
            "at least 2 theory models are required to define a dispersion; "
            f"got {n_models}"
        )
    if n_rows < 1:
        raise ValueError("model_predictions needs at least one row")
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise ValueError(
            f"model_predictions must be finite: {bad} non-finite value(s); a "
            "missing theory model is a failed envelope, not a row to average away"
        )
    return array, _family_names(n_models, model_families)


def _as_aligned_1d(*arrays: tuple[str, object]) -> list[np.ndarray]:
    out = [np.asarray(values, dtype=np.float64).reshape(-1) for _, values in arrays]
    sizes = {array.size for array in out}
    if len(sizes) != 1:
        raise ValueError(f"inputs must have equal length, got {sorted(sizes)}")
    for (name, _), array in zip(arrays, out, strict=False):
        if not np.isfinite(array).all():
            bad = int((~np.isfinite(array)).sum())
            raise ValueError(f"{name} must be finite ({bad} non-finite value(s))")
    return out


# ---------------------------------------------------------------------------
# theory dispersion and envelope
# ---------------------------------------------------------------------------
def theory_dispersion(model_predictions, *, model_families=None) -> dict:
    """Per-row dispersion across theory models.

    Parameters
    ----------
    model_predictions : array-like ``[n_theory_models, n_rows, ...]``.
        Axis 0 is the model-family axis in :data:`THEORY_MODEL_ORDER` (or the
        explicit ``model_families`` order); trailing axes are label coordinates.
    model_families : optional names, one per model, in axis-0 order.

    Returns
    -------
    dict
        ``std`` (population standard deviation over the family axis, ``ddof=0``),
        ``range`` (``max - min``), ``low``/``high``, ``median`` and the shape /
        family provenance.  ``None`` is never returned for a valid input: a
        non-finite prediction or a degenerate axis raises instead.

    Raises
    ------
    ValueError
        For fewer than 2 models, non-finite predictions, a 1-D input, or
        mismatched/duplicate ``model_families``.
    """
    array, families = _as_model_predictions(model_predictions, model_families)
    return {
        "std": np.std(array, axis=0, ddof=0),
        "range": np.max(array, axis=0) - np.min(array, axis=0),
        "low": np.min(array, axis=0),
        "high": np.max(array, axis=0),
        "median": np.median(array, axis=0),
        "n_models": int(array.shape[0]),
        "n_rows": int(array.shape[1]),
        "model_families": families,
        "axis": 0,
        "ddof": 0,
        "score_kind": "model_family_dispersion",
        "null_reason": "",
    }


def theory_envelope(
    model_predictions, level: float = 0.95, *, model_families=None
) -> dict:
    """Per-row sensitivity envelope across the tested theory models.

    ``lower``/``upper`` are the **full** range (min/max) over the model-family
    axis, so every tested model is inside the envelope by construction.  This is
    a *sensitivity bound over the tested family*, not a conformal interval and not
    a coverage statement about the truth.

    ``level`` only controls the additional central summary
    ``level_lower``/``level_upper`` (the ``level`` inter-model quantile range);
    it never narrows ``lower``/``upper``.  At ``level=1`` the central summary
    equals the full range.

    Parameters
    ----------
    model_predictions : array-like ``[n_theory_models, n_rows, ...]``.
    level : central model-family quantile in ``(0, 1]``.
    model_families : optional axis-0 family names.

    Returns
    -------
    dict
        ``lower``, ``upper``, ``width`` (full range), ``level_lower``,
        ``level_upper``, ``level_width``, ``median``, ``level``, model/shape
        provenance, ``contains_all_models=True``, ``kind`` and ``null_reason``.
    """
    array, families = _as_model_predictions(model_predictions, model_families)
    level = float(level)
    if not 0.0 < level <= 1.0:
        raise ValueError(f"level must be in (0, 1], got {level!r}")
    tail = (1.0 - level) / 2.0
    low = np.min(array, axis=0)
    high = np.max(array, axis=0)
    level_lower = np.quantile(array, tail, axis=0)
    level_upper = np.quantile(array, 1.0 - tail, axis=0)
    return {
        "lower": low,
        "upper": high,
        "width": high - low,
        "level_lower": level_lower,
        "level_upper": level_upper,
        "level_width": level_upper - level_lower,
        "median": np.median(array, axis=0),
        "level": level,
        "n_models": int(array.shape[0]),
        "n_rows": int(array.shape[1]),
        "model_families": families,
        "contains_all_models": True,
        "kind": "theory_sensitivity_envelope",
        "mean_width": float(np.mean(high - low)),
        "mean_level_width": float(np.mean(level_upper - level_lower)),
        "null_reason": "",
    }


# ---------------------------------------------------------------------------
# theory-scaled conformal score
# ---------------------------------------------------------------------------
@dataclass
class TheoryScaledScore:
    """Theory-inflated split conformal with a cross-fitted ``gamma``.

    The score is

    .. math::

        s_i = \\frac{|y_i - \\hat{m}_i|}{\\sigma_{exp,i} + \\gamma \\sigma_{theo,i}}

    with a floored denominator (``denominator_floor``; the number of floored rows
    is reported, never silently discarded).  ``gamma`` is chosen by K-fold
    cross-fitting on the **calibration rows only**: for every candidate ``gamma``
    each fold is scored with a threshold fitted on the other folds, and the
    candidate with the best out-of-fold interval score is selected.  The final
    threshold is then the exact finite-sample rank
    ``s_(ceil((n_cal+1)(1-alpha)))`` on all calibration rows at that frozen
    ``gamma``.

    ``fit`` has no test argument: test rows cannot change ``gamma``, the
    threshold or the folds.

    Parameters
    ----------
    alpha : target miscoverage in ``(0, 1)``.
    n_folds : number of cross-fitting folds (>= 2).
    gamma_grid : candidate non-negative ``gamma`` values.
    fixed_gamma : if given, skip selection and use this value (diagnostics for
        only this gamma; ``gamma_selection="fixed"``).
    fold_seed : seed of the deterministic balanced fold assignment.
    denominator_floor : strictly positive floor for the score denominator.
    min_count : below this test-row count :meth:`evaluate` is descriptive only.
    selection : ``"interval_score"`` (default; minimises the mean out-of-fold
        interval score, a proper interval score) or ``"width_at_nominal"``
        (minimises width among gammas whose out-of-fold coverage meets nominal).
    """

    alpha: float = 0.1
    n_folds: int = 5
    gamma_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    fixed_gamma: float | None = None
    fold_seed: int = 0
    denominator_floor: float = 1e-12
    min_count: int = MIN_COUNT
    selection: str = "interval_score"
    document_: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        self.alpha = alpha
        self.n_folds = int(self.n_folds)
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        grid = tuple(float(value) for value in self.gamma_grid)
        if not grid:
            raise ValueError("gamma_grid must not be empty")
        if any(not math.isfinite(value) or value < 0.0 for value in grid):
            raise ValueError("gamma_grid values must be finite and non-negative")
        if len(set(grid)) != len(grid):
            raise ValueError("gamma_grid values must be unique")
        self.gamma_grid = grid
        if self.fixed_gamma is not None:
            fixed = float(self.fixed_gamma)
            if not math.isfinite(fixed) or fixed < 0.0:
                raise ValueError("fixed_gamma must be finite and non-negative")
            self.fixed_gamma = fixed
        self.denominator_floor = float(self.denominator_floor)
        if not math.isfinite(self.denominator_floor) or self.denominator_floor <= 0.0:
            raise ValueError("denominator_floor must be finite and positive")
        self.min_count = int(self.min_count)
        if self.min_count < 1:
            raise ValueError("min_count must be a positive integer")
        if self.selection not in ("interval_score", "width_at_nominal"):
            raise ValueError("selection must be 'interval_score' or 'width_at_nominal'")

    # -- fitted state ------------------------------------------------------
    def _require_document(self) -> dict:
        if self.document_ is None:
            raise RuntimeError("TheoryScaledScore.fit has not been called")
        return self.document_

    @property
    def gamma(self) -> float:
        """Selected (or fixed) theory-error scale."""
        return float(self._require_document()["gamma"])

    @property
    def threshold(self) -> float:
        """Exact finite-sample score threshold at the selected ``gamma``."""
        return float(self._require_document()["threshold"])

    # -- score pieces ------------------------------------------------------
    def denominator(self, sigma_exp, sigma_theo, *, gamma: float | None = None):
        """Safe denominator and the number of floored rows.

        ``sigma_exp >= 0`` and ``sigma_theo >= 0`` are required; the raw sum is
        floored at ``denominator_floor`` so an exactly zero quoted error cannot
        produce ``inf`` or ``NaN``.  The returned count makes the floor visible.
        """
        sigma_exp_arr, sigma_theo_arr = _as_aligned_1d(
            ("sigma_exp", sigma_exp), ("sigma_theo", sigma_theo)
        )
        if np.any(sigma_exp_arr < 0.0):
            raise ValueError("sigma_exp must be non-negative")
        if np.any(sigma_theo_arr < 0.0):
            raise ValueError("sigma_theo must be non-negative")
        if gamma is None:
            if self.document_ is not None:
                gamma = self.gamma
            elif self.fixed_gamma is not None:
                gamma = self.fixed_gamma
            else:
                raise RuntimeError(
                    "call fit() first or pass gamma explicitly before calibration"
                )
        gamma = float(gamma)
        if not math.isfinite(gamma) or gamma < 0.0:
            raise ValueError("gamma must be finite and non-negative")
        raw = sigma_exp_arr + gamma * sigma_theo_arr
        floored = int(np.sum(raw < self.denominator_floor))
        return np.maximum(raw, self.denominator_floor), floored

    def score(
        self,
        y,
        median_prediction,
        sigma_exp,
        sigma_theo,
        *,
        gamma: float | None = None,
    ) -> np.ndarray:
        """Theory-scaled absolute-residual score (higher = worse)."""
        y_arr, median_arr = _as_aligned_1d(
            ("y", y), ("median_prediction", median_prediction)
        )
        denominator, _ = self.denominator(sigma_exp, sigma_theo, gamma=gamma)
        if denominator.size != y_arr.size:
            raise ValueError(
                "sigma_exp/sigma_theo must align with y, got "
                f"{denominator.size} and {y_arr.size} rows"
            )
        return np.abs(y_arr - median_arr) / denominator

    def interval(
        self,
        median_prediction,
        sigma_exp,
        sigma_theo,
        *,
        gamma: float | None = None,
        threshold: float | None = None,
    ):
        """Symmetric theory-scaled interval ``median +/- q * denominator``."""
        median_arr = np.asarray(median_prediction, dtype=np.float64).reshape(-1)
        if not np.isfinite(median_arr).all():
            raise ValueError("median_prediction must be finite")
        denominator, _ = self.denominator(sigma_exp, sigma_theo, gamma=gamma)
        if denominator.size != median_arr.size:
            raise ValueError("sigma_exp/sigma_theo must align with median_prediction")
        if threshold is None:
            threshold = self.threshold
        threshold = float(threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("threshold must be finite and non-negative")
        half_width = threshold * denominator
        return median_arr - half_width, median_arr + half_width

    # -- cross-fitting -----------------------------------------------------
    def _fold_assignment(self, n: int, fold_ids) -> tuple[np.ndarray, int]:
        if fold_ids is not None:
            raw = np.asarray(fold_ids)
            if raw.ndim != 1 or raw.size != n:
                raise ValueError("fold_ids must be one id per calibration row")
            unique = sorted(set(raw.tolist()))
            if len(unique) < 2:
                raise ValueError("fold_ids must define at least 2 folds")
            mapping = {value: index for index, value in enumerate(unique)}
            return np.asarray(
                [mapping[value] for value in raw.tolist()], dtype=np.int64
            ), len(unique)
        if self.n_folds > n:
            raise ValueError(
                f"n_folds={self.n_folds} exceeds n_cal={n}; cross-fitting would "
                "leave empty folds"
            )
        permutation = np.random.default_rng(int(self.fold_seed)).permutation(n)
        ids = np.empty(n, dtype=np.int64)
        ids[permutation] = np.arange(n) % self.n_folds
        return ids, self.n_folds

    def fit(
        self,
        y_cal,
        median_prediction,
        sigma_exp,
        sigma_theo,
        *,
        fold_ids=None,
    ) -> dict:
        """Select ``gamma`` on calibration rows and fit the exact threshold.

        Parameters
        ----------
        y_cal : (n_cal,) calibration labels.
        median_prediction : (n_cal,) median prediction across theory models.
        sigma_exp, sigma_theo : (n_cal,) quoted statistical error and model-family
            dispersion, in native label units.
        fold_ids : optional pre-registered fold id per calibration row.

        Returns
        -------
        dict
            ``method="theory_scaled_split_rank"``, ``gamma``, ``gamma_selection``,
            ``gamma_sensitivity`` (out-of-fold coverage/width/interval score for
            every candidate), ``threshold``, the exact ``order_stat_index``,
            ``n_cal``, ``achievable_coverage``, ``finite_sample_exact``,
            ``oof_scores``/``oof_membership``/``oof_coverage``, denominator-floor
            count and ``used_test_rows=False``.
        """
        y_arr, median_arr, sigma_exp_arr, sigma_theo_arr = _as_aligned_1d(
            ("y_cal", y_cal),
            ("median_prediction", median_prediction),
            ("sigma_exp", sigma_exp),
            ("sigma_theo", sigma_theo),
        )
        n = int(y_arr.size)
        if n < 2:
            raise ValueError("fit needs at least 2 calibration rows")
        ids, n_folds = self._fold_assignment(n, fold_ids)
        fold_sizes = [int(np.sum(ids == fold)) for fold in range(n_folds)]
        min_train = n - max(fold_sizes)
        if finite_sample_rank(min_train, self.alpha) > min_train:
            raise InsufficientCalibrationError(
                f"cross-fitting folds too small: n_train={min_train} cannot support "
                f"alpha={self.alpha} (rank={finite_sample_rank(min_train, self.alpha)})"
            )
        gammas = (
            (self.fixed_gamma,) if self.fixed_gamma is not None else self.gamma_grid
        )
        sensitivity: list[dict] = []
        for gamma in gammas:
            denominator, floored = self.denominator(
                sigma_exp_arr, sigma_theo_arr, gamma=gamma
            )
            scores = np.abs(y_arr - median_arr) / denominator
            covered_total = 0
            rows_total = 0
            interval_total = 0.0
            width_total = 0.0
            fold_records: list[dict] = []
            for fold in range(n_folds):
                train = ids != fold
                held = ~train
                if not held.any():
                    continue
                train_scores = scores[train]
                rank_train = finite_sample_rank(train_scores.size, self.alpha)
                if rank_train > train_scores.size:
                    raise InsufficientCalibrationError(
                        f"fold {fold}: finite-sample rank {rank_train} exceeds "
                        f"n_train={train_scores.size} at alpha={self.alpha}"
                    )
                threshold = exact_quantile(train_scores, self.alpha)
                held_scores = scores[held]
                covered = held_scores <= threshold
                half = threshold * denominator[held]
                fold_interval = interval_score(
                    y_arr[held],
                    median_arr[held] - half,
                    median_arr[held] + half,
                    alpha=self.alpha,
                )
                if fold_interval["interval_score"] is None:
                    raise ValueError(
                        "a cross-fitting fold produced no finite interval score"
                    )
                n_held = int(held.sum())
                rows_total += n_held
                covered_total += int(covered.sum())
                interval_total += float(fold_interval["interval_score"]) * n_held
                width_total += float(fold_interval["width"]) * n_held
                fold_records.append(
                    {
                        "fold": fold,
                        "n_train": int(train.sum()),
                        "n_held": n_held,
                        "threshold": float(threshold),
                        "coverage": float(np.mean(covered)),
                        "mean_width": float(fold_interval["width"]),
                        "interval_score": float(fold_interval["interval_score"]),
                    }
                )
            sensitivity.append(
                {
                    "gamma": float(gamma),
                    "oof_coverage": covered_total / float(rows_total),
                    "oof_mean_width": width_total / float(rows_total),
                    "oof_interval_score": interval_total / float(rows_total),
                    "n_oof": int(rows_total),
                    "n_folds": len(fold_records),
                    "n_denominator_floored": floored,
                    "folds": fold_records,
                }
            )

        selected, selection_kind = self._select_gamma(sensitivity)
        selected_record = next(
            record for record in sensitivity if record["gamma"] == float(selected)
        )
        denominator, floored = self.denominator(
            sigma_exp_arr, sigma_theo_arr, gamma=selected
        )
        scores = np.abs(y_arr - median_arr) / denominator
        threshold = exact_quantile(scores, self.alpha)
        rank = finite_sample_rank(n, self.alpha)
        fold_thresholds = {
            record["fold"]: record["threshold"] for record in selected_record["folds"]
        }
        oof_membership = np.zeros(n, dtype=bool)
        for fold in range(n_folds):
            held = ids == fold
            if held.any():
                oof_membership[held] = scores[held] <= fold_thresholds[fold]

        document = {
            "method": "theory_scaled_split_rank",
            "calibration_kind": CALIBRATION_KIND,
            "score_kind": THEORY_SCORE_KIND,
            "alpha": self.alpha,
            "gamma": float(selected),
            "gamma_selection": selection_kind,
            "gamma_grid": [float(value) for value in gammas],
            "gamma_sensitivity": sensitivity,
            "selected_gamma_record": selected_record,
            "threshold": float(threshold),
            "order_stat_index": int(rank),
            "n_cal": n,
            "achievable_coverage": rank / float(n + 1),
            "finite_sample_exact": True,
            "n_folds": n_folds,
            "fold_sizes": fold_sizes,
            "fold_thresholds": fold_thresholds,
            "oof_scores": scores,
            "oof_membership": oof_membership,
            "oof_coverage": float(np.mean(oof_membership)),
            "oof_mean_width": float(selected_record["oof_mean_width"]),
            "oof_interval_score": float(selected_record["oof_interval_score"]),
            "n_denominator_floored": int(floored),
            "denominator_floor": self.denominator_floor,
            "used_test_rows": False,
            "n_test_rows_used": 0,
            "null_reason": "",
        }
        self.document_ = document
        return document

    def _select_gamma(self, sensitivity: Sequence[dict]) -> tuple[float, str]:
        if self.fixed_gamma is not None:
            return float(self.fixed_gamma), "fixed"
        if self.selection == "interval_score":
            best = min(
                sensitivity,
                key=lambda record: (record["oof_interval_score"], record["gamma"]),
            )
            return float(best["gamma"]), "kfold_interval_score"
        nominal = 1.0 - self.alpha
        eligible = [
            record for record in sensitivity if record["oof_coverage"] >= nominal
        ]
        if eligible:
            best = min(
                eligible,
                key=lambda record: (record["oof_mean_width"], record["gamma"]),
            )
            return float(best["gamma"]), "kfold_width_at_nominal"
        # No candidate met the nominal out-of-fold coverage: pick the closest
        # (highest coverage, then narrowest) and keep that failure visible in the
        # documented selection kind.
        best = max(
            sensitivity,
            key=lambda record: (
                record["oof_coverage"],
                -record["oof_mean_width"],
                -record["gamma"],
            ),
        )
        return float(best["gamma"]), "kfold_width_at_nominal_nominal_unmet"

    # -- evaluation --------------------------------------------------------
    def evaluate(
        self,
        y,
        median_prediction,
        sigma_exp,
        sigma_theo,
        *,
        envelope_low=None,
        envelope_high=None,
        strata=None,
    ) -> dict:
        """Empirical test coverage and width, with the envelope reported apart.

        The conformal interval is ``median +/- threshold * denominator`` at the
        cross-fitted ``gamma``; its empirical coverage is the only coverage claim
        returned.  If ``envelope_low``/``envelope_high`` are supplied (from
        :func:`theory_envelope`), their mean width and the fraction of labels
        inside them are reported as **descriptive sensitivity diagnostics**;
        they are never called coverage.
        """
        self._require_document()
        y_arr, median_arr, sigma_exp_arr, sigma_theo_arr = _as_aligned_1d(
            ("y", y),
            ("median_prediction", median_prediction),
            ("sigma_exp", sigma_exp),
            ("sigma_theo", sigma_theo),
        )
        n = int(y_arr.size)
        if n == 0:
            return {
                "coverage": None,
                "n": 0,
                "n_inside": 0,
                "mean_width": None,
                "median_width": None,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "threshold": self.threshold,
                "calibration_kind": CALIBRATION_KIND,
                "coverage_kind": "marginal_conformal",
                "theory_sigma_mean": None,
                "n_denominator_floored": 0,
                "theory_envelope_width": None,
                "theory_envelope_low_mean": None,
                "theory_envelope_high_mean": None,
                "envelope_contains_y_fraction": None,
                "envelope_claim": "none",
                "per_stratum": [] if strata is not None else None,
                "per_stratum_coverage_claim": "none",
                "descriptive_only": True,
                "null_reason": NullReason.SPLIT_EMPTY.value,
            }
        threshold = self.threshold
        denominator, floored = self.denominator(
            sigma_exp_arr, sigma_theo_arr, gamma=self.gamma
        )
        scores = np.abs(y_arr - median_arr) / denominator
        membership = scores <= threshold
        half = threshold * denominator
        lo = median_arr - half
        hi = median_arr + half

        envelope_width = None
        envelope_low_mean = None
        envelope_high_mean = None
        envelope_contains = None
        if envelope_low is not None or envelope_high is not None:
            if envelope_low is None or envelope_high is None:
                raise ValueError(
                    "envelope_low and envelope_high must be supplied together"
                )
            low_arr, high_arr = _as_aligned_1d(
                ("envelope_low", envelope_low), ("envelope_high", envelope_high)
            )
            if low_arr.size != n:
                raise ValueError("envelope bounds must align with y")
            if np.any(high_arr < low_arr):
                raise ValueError("envelope_high must not be below envelope_low")
            width = high_arr - low_arr
            envelope_width = float(np.mean(width))
            envelope_low_mean = float(np.mean(low_arr))
            envelope_high_mean = float(np.mean(high_arr))
            envelope_contains = float(np.mean((y_arr >= low_arr) & (y_arr <= high_arr)))

        per_stratum = None
        if strata is not None:
            per_stratum = stratum_metrics(
                y_arr,
                lo,
                hi,
                strata,
                min_count=self.min_count,
            )

        return {
            "coverage": float(np.mean(membership)),
            "n": n,
            "n_inside": int(np.sum(membership)),
            "mean_width": float(np.mean(hi - lo)),
            "median_width": float(np.median(hi - lo)),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "threshold": threshold,
            "calibration_kind": CALIBRATION_KIND,
            "coverage_kind": "marginal_conformal",
            "theory_sigma_mean": float(np.mean(sigma_theo_arr)),
            "n_denominator_floored": int(floored),
            "theory_envelope_width": envelope_width,
            "theory_envelope_low_mean": envelope_low_mean,
            "theory_envelope_high_mean": envelope_high_mean,
            "envelope_contains_y_fraction": envelope_contains,
            "envelope_claim": "sensitivity_only"
            if envelope_width is not None
            else "none",
            "per_stratum": per_stratum,
            "per_stratum_coverage_claim": "descriptive" if per_stratum else "none",
            "descriptive_only": descriptive_only(n, min_count=self.min_count),
            "null_reason": "",
        }

    def metrics_fields(self, evaluation: Mapping | None = None, **overrides) -> dict:
        """Optional ``metrics.csv``/``age.csv``/``distance.csv`` column values.

        Produces the R1 optional fields (``gamma``, ``calibration_kind``,
        ``theory_sigma``, ``theory_envelope_low``, ``theory_envelope_high``).
        With an :meth:`evaluate` result, the envelope fields are the mean bounds
        of the slice; callers that aggregate differently may override them.
        """
        self._require_document()
        fields = {
            "gamma": self.gamma,
            "calibration_kind": CALIBRATION_KIND,
            "theory_sigma": None,
            "theory_envelope_low": None,
            "theory_envelope_high": None,
        }
        if evaluation is not None:
            fields["theory_sigma"] = evaluation.get("theory_sigma_mean")
            fields["theory_envelope_low"] = evaluation.get("theory_envelope_low_mean")
            fields["theory_envelope_high"] = evaluation.get("theory_envelope_high_mean")
        fields.update(overrides)
        return fields
