"""Pre-registered Mondrian strata and minimum-n pooling (plan item R1).

``docs/plan-20261022/10-rfit-profile-likelihood-and-theoretical-conformal.md``
section 3 (Method 2) asks for **pre-registered stage strata** calibrated with a
minimum number of stars and **pooling of adjacent stages** when a stratum is too
small.  This module implements that contract without looking at the labels being
scored:

* :func:`stage_stratum` assigns a stage from measured *or* predicted ``Teff`` (K)
  and ``logg`` (dex) only -- the function has no label/residual argument, so a
  stratum cannot be chosen by looking at the conformity scores;
* :func:`tier_stratum` assigns a data tier from explicit flags (``xp_full``,
  ``xp_partial``, ``no_xp``, ``gaia_only``), and :func:`compose_strata` appends
  an optional survey tag;
* :class:`MondrianCalibrator` fits one exact finite-sample threshold per stratum
  (``q_hat = s_(ceil((n+1)(1-alpha)))``), pools adjacent strata when a group is
  below ``min_count``, reports the pooling, and either raises or flags strata
  that still cannot be calibrated;
* :func:`stratum_metrics` reports ``n``, coverage and interval width per
  stratum, marking small slices descriptive-only and carrying the pooling note.

Stage boundaries (pre-registered; not re-tuned per run)
--------------------------------------------------------
The cuts are rounded working boundaries on the Kiel plane, ordered by the
evolutionary sequence used for adjacency.  They are a **reporting
stratification**, not a physical classification: ``Teff``/``logg`` alone cannot
separate the red clump from the red giant branch, nor the AGB from luminous RGB
stars.  The red clump is therefore a box *inside* the giant region, and the
low-gravity bucket is deliberately named ``agb_other``.

=============  ==============================================================
stratum        boundary (``logg`` dex, ``Teff`` K; lower bound inclusive)
=============  ==============================================================
dwarf_ms       ``logg >= 4.0`` (any Teff)
turnoff_sgb    ``3.5 <= logg < 4.0`` (any Teff)
rc             ``2.5 <= logg < 3.5`` and ``4600 <= Teff <= 5400``
rgb            ``2.5 <= logg < 3.5`` and ``Teff < 4600``
agb_other      ``logg < 2.5``, or ``2.5 <= logg < 3.5`` and ``Teff > 5400``
=============  ==============================================================

The adjacency chain for pooling follows the evolutionary order
``dwarf_ms -> turnoff_sgb -> rgb -> rc -> agb_other`` (``STAGE_ORDER``).  The
tier chain is ``xp_full -> xp_partial -> no_xp -> gaia_only`` (``TIER_ORDER``).
Pooling a group never silently changes the reported stratum label: each original
stratum still has a row, and the row carries the full pooled group.

Nothing here computes a coverage claim for a stratum that has no calibration
rows: :class:`~masked_stellar_autoencoder.reporting.calibration.InsufficientCalibrationError`
is raised (default) or the stratum is flagged with an explicit ``null_reason``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from .calibration import (
    InsufficientCalibrationError,
    exact_quantile,
    finite_sample_rank,
)
from .metrics import MIN_COUNT, coverage, descriptive_only
from .schema import NullReason

__all__ = [
    "STAGE_ORDER",
    "STAGE_DESCRIPTIONS",
    "STAGE_BOUNDARIES",
    "TIER_ORDER",
    "STRATUM_SEPARATOR",
    "MondrianCalibrator",
    "compose_strata",
    "stage_stratum",
    "stratum_metrics",
    "tier_stratum",
]

#: Evolutionary order used for adjacency and deterministic tie-breaking.
STAGE_ORDER: tuple[str, ...] = (
    "dwarf_ms",
    "turnoff_sgb",
    "rgb",
    "rc",
    "agb_other",
)

#: Plain-language description of each stage stratum.
STAGE_DESCRIPTIONS: dict[str, str] = {
    "dwarf_ms": "dwarf / main sequence (logg >= 4.0)",
    "turnoff_sgb": "turn-off / subgiant branch (3.5 <= logg < 4.0)",
    "rgb": "red giant branch (2.5 <= logg < 3.5, Teff < 4600 K)",
    "rc": "red clump (2.5 <= logg < 3.5, 4600 <= Teff <= 5400 K)",
    "agb_other": "AGB / other luminous giant (logg < 2.5, or hot low-gravity star)",
}

#: Machine-readable pre-registered boundaries.  ``*_min`` fields are inclusive,
#: ``*_max`` fields are exclusive, except the RC ``teff`` box which is inclusive
#: at both ends.  ``None`` means unbounded.
STAGE_BOUNDARIES: dict[str, dict[str, float | None]] = {
    "dwarf_ms": {
        "logg_min": 4.0,
        "logg_max": None,
        "teff_min": None,
        "teff_max": None,
    },
    "turnoff_sgb": {
        "logg_min": 3.5,
        "logg_max": 4.0,
        "teff_min": None,
        "teff_max": None,
    },
    "rc": {
        "logg_min": 2.5,
        "logg_max": 3.5,
        "teff_min": 4600.0,
        "teff_max": 5400.0,
    },
    "rgb": {
        "logg_min": 2.5,
        "logg_max": 3.5,
        "teff_min": None,
        "teff_max": 4600.0,
    },
    "agb_other": {
        "logg_min": None,
        "logg_max": 2.5,
        "teff_min": None,
        "teff_max": None,
    },
}

#: Tier order used for adjacency (best data first).
TIER_ORDER: tuple[str, ...] = ("xp_full", "xp_partial", "no_xp", "gaia_only")

#: Separator used by :func:`compose_strata`; stratum components may not contain it.
STRATUM_SEPARATOR = "|"

_RC_TEFF_MIN = 4600.0
_RC_TEFF_MAX = 5400.0
_RC_LOGG_MIN = 2.5
_RC_LOGG_MAX = 3.5
_SGB_LOGG_MIN = 3.5
_DWARF_LOGG_MIN = 4.0


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _as_finite_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.isfinite(array).all():
        bad = int((~np.isfinite(array)).sum())
        raise ValueError(
            f"{name} must be finite: {bad} non-finite value(s) cannot be assigned "
            "to a pre-registered stratum"
        )
    return array


def _as_label_vector(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D label per row")
    labels: list[str] = []
    for item in array.tolist():
        if not isinstance(item, str) or not item:
            raise ValueError(f"{name} must contain non-empty strings, got {item!r}")
        labels.append(item)
    return np.asarray(labels, dtype=str)


def _as_flag(values, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D boolean flag per row")
    if array.dtype == bool:
        return array.astype(bool, copy=False)
    if np.issubdtype(array.dtype, np.integer):
        unique = set(np.unique(array).tolist())
        if unique <= {0, 1}:
            return array.astype(bool)
    raise ValueError(f"{name} must be a boolean (or 0/1 integer) flag per row")


def _ordered_labels(labels: Sequence[str], order: Sequence[str] | None) -> list[str]:
    """Unique labels, ordered by ``order`` when given, else sorted."""
    observed = set(labels)
    if order is None:
        return sorted(observed)
    known = [label for label in order if label in observed]
    unknown = sorted(observed - set(known))
    return known + unknown


def _auto_order(labels: Sequence[str]) -> list[str]:
    """Pre-registered chain for known stage/tier labels; else alphabetical."""
    observed = set(labels)
    if observed and observed <= set(STAGE_ORDER):
        return [label for label in STAGE_ORDER if label in observed]
    if observed and observed <= set(TIER_ORDER):
        return [label for label in TIER_ORDER if label in observed]
    return sorted(observed)


# ---------------------------------------------------------------------------
# strata assignment
# ---------------------------------------------------------------------------
def stage_stratum(teff, logg) -> np.ndarray:
    """Assign the pre-registered evolutionary-stage stratum per row.

    Uses **only** ``Teff`` and ``logg`` (measured or predicted); there is no
    label, residual or density argument, so the stratification cannot be chosen
    to make the conformity scores look good.

    Parameters
    ----------
    teff : (n,) effective temperature in kelvin.
    logg : (n,) surface gravity in dex.

    Returns
    -------
    (n,) string array with one of :data:`STAGE_ORDER`.  The mapping is
    exhaustive and disjoint; see the module docstring and
    :data:`STAGE_BOUNDARIES` for the exact inclusive/exclusive cuts.

    Raises
    ------
    ValueError
        For non-finite ``Teff``/``logg`` (a row that cannot be assigned is not
        silently dropped or dumped into a catch-all) or mismatched lengths.
    """
    if np.asarray(teff).ndim > 1 or np.asarray(logg).ndim > 1:
        raise ValueError("teff and logg must be 1-D arrays (one value per row)")
    teff_arr = _as_finite_vector(teff, "teff")
    logg_arr = _as_finite_vector(logg, "logg")
    if teff_arr.size != logg_arr.size:
        raise ValueError(
            f"teff and logg must have equal length, got {teff_arr.size} and "
            f"{logg_arr.size}"
        )
    dwarf = logg_arr >= _DWARF_LOGG_MIN
    sgb = (logg_arr >= _SGB_LOGG_MIN) & (logg_arr < _DWARF_LOGG_MIN)
    giant = (logg_arr >= _RC_LOGG_MIN) & (logg_arr < _RC_LOGG_MAX)
    rc = giant & (teff_arr >= _RC_TEFF_MIN) & (teff_arr <= _RC_TEFF_MAX)
    rgb = giant & (teff_arr < _RC_TEFF_MIN)
    # Everything else is the AGB/other bucket: logg < 2.5 (any Teff) or a hot
    # low-gravity star (2.5 <= logg < 3.5, Teff > 5400 K).
    assignments = np.select(
        [dwarf, sgb, rc, rgb],
        ["dwarf_ms", "turnoff_sgb", "rc", "rgb"],
        default="agb_other",
    )
    return np.asarray(assignments, dtype=str)


def tier_stratum(
    *,
    xp_full=None,
    xp_partial=None,
    no_xp=None,
    gaia_only=None,
) -> np.ndarray:
    """Assign the data-tier stratum from explicit boolean flags.

    Exactly one flag must be true per row.  The flags describe *available data*,
    never a label value; a row with zero or several flags raises instead of being
    assigned silently.

    Parameters
    ----------
    xp_full, xp_partial, no_xp, gaia_only : (n,) bool arrays or ``None``.
        Omitted tiers are treated as "not this tier".

    Returns
    -------
    (n,) string array with one of :data:`TIER_ORDER`.

    Raises
    ------
    ValueError
        If no flag is supplied, lengths disagree, a flag is not boolean, or any
        row does not have exactly one true flag.
    """
    flags = {
        "xp_full": xp_full,
        "xp_partial": xp_partial,
        "no_xp": no_xp,
        "gaia_only": gaia_only,
    }
    present = {
        name: _as_flag(value, name)
        for name, value in flags.items()
        if value is not None
    }
    if not present:
        raise ValueError("at least one tier flag is required")
    sizes = {array.size for array in present.values()}
    if len(sizes) != 1:
        raise ValueError(f"tier flags must share one length, got {sorted(sizes)}")
    n = sizes.pop()
    if n == 0:
        raise ValueError("tier flags must contain at least one row")
    counts = np.zeros(n, dtype=np.int64)
    for array in present.values():
        counts += array.astype(np.int64)
    bad = int(np.sum(counts != 1))
    if bad:
        raise ValueError(
            f"{bad} row(s) do not have exactly one true tier flag "
            f"(0 true: {int(np.sum(counts == 0))}, >1 true: {int(np.sum(counts > 1))})"
        )
    ordered = [name for name in TIER_ORDER if name in present]
    conditions = [present[name] for name in ordered]
    return np.asarray(np.select(conditions, ordered, default="gaia_only"), dtype=str)


def compose_strata(
    stage, tier=None, survey=None, *, separator: str = STRATUM_SEPARATOR
) -> np.ndarray:
    """Compose ``stage`` / ``tier`` / optional ``survey`` into one label.

    The survey component is prefixed (``survey:<name>``) so a survey called
    ``"rc"`` can never be confused with the stage stratum.  Passing only a stage
    returns the stage labels unchanged.  Pooling adjacency for composite labels
    is not guessed: supply ``MondrianCalibrator(order=...)`` explicitly if the
    composite order matters.

    Raises
    ------
    ValueError
        On length mismatch, empty components, or a component containing the
        separator (ambiguous labels are refused).
    """
    if not isinstance(separator, str) or not separator:
        raise ValueError("separator must be a non-empty string")
    stage_arr = _as_label_vector(stage, "stage")
    n = stage_arr.size
    components = [stage_arr]
    if tier is not None:
        tier_arr = _as_label_vector(tier, "tier")
        if tier_arr.size != n:
            raise ValueError("tier must have one label per stage row")
        components.append(tier_arr)
    if survey is not None:
        survey_arr = _as_label_vector(survey, "survey")
        if survey_arr.size != n:
            raise ValueError("survey must have one label per stage row")
        components.append(
            np.asarray([f"survey:{value}" for value in survey_arr], dtype=str)
        )
    for component in components:
        bad = [value for value in component.tolist() if separator in value]
        if bad:
            raise ValueError(
                f"stratum component {bad[0]!r} contains the separator {separator!r}"
            )
    if len(components) == 1:
        return stage_arr.copy()
    labels = [
        separator.join(component[index] for component in components)
        for index in range(n)
    ]
    return np.asarray(labels, dtype=str)


# ---------------------------------------------------------------------------
# Mondrian calibration with minimum-n pooling
# ---------------------------------------------------------------------------
@dataclass
class MondrianCalibrator:
    """Exact finite-sample Mondrian thresholds with adjacent-stage pooling.

    The class is deliberately *two-phase*: :meth:`fit` sees calibration rows
    only and returns a plain document, and :meth:`apply` / :meth:`evaluate` see
    test rows.  There is no test argument in :meth:`fit`, so a test row cannot
    influence a threshold or a pooling decision.

    Parameters
    ----------
    alpha : target miscoverage in ``(0, 1)``.
    min_count : minimum calibration rows per pooled group (plan section 3
        recommends 100--200).  A group below it is flagged/raised even when the
        exact rank would formally exist.
    pooling : ``"adjacent"`` (default; merge a too-small group with its
        evolutionary/tier neighbour), ``"none"`` (never pool) or ``"global"``
        (one declared group for all strata).
    on_insufficient : ``"raise"`` (default) or ``"flag"``.
    order : optional pre-registered label order used for adjacency and sorting
        (required for meaningful composite strata).
    adjacency : optional explicit undirected adjacency map; overrides the chain
        implied by ``order``.  Labels without an edge cannot be pooled.

    Attributes set by :meth:`fit`
    -----------------------------
    ``document_`` : the calibration document returned by :meth:`fit`.
    """

    alpha: float = 0.1
    min_count: int = 100
    pooling: str = "adjacent"
    on_insufficient: str = "raise"
    order: Sequence[str] | None = None
    adjacency: Mapping[str, Sequence[str]] | None = None
    document_: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        alpha = float(self.alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")
        self.alpha = alpha
        self.min_count = int(self.min_count)
        if self.min_count < 1:
            raise ValueError("min_count must be a positive integer")
        if self.pooling not in ("adjacent", "none", "global"):
            raise ValueError("pooling must be 'adjacent', 'none' or 'global'")
        if self.on_insufficient not in ("raise", "flag"):
            raise ValueError("on_insufficient must be 'raise' or 'flag'")
        if self.order is not None:
            self.order = tuple(_as_label_vector(self.order, "order").tolist())
        if self.adjacency is not None:
            normalised: dict[str, tuple[str, ...]] = {}
            for label, neighbours in self.adjacency.items():
                key = _as_label_vector([label], "adjacency label")[0]
                values = _as_label_vector(neighbours, f"adjacency[{key!r}]")
                normalised[key] = tuple(values.tolist())
            self.adjacency = normalised

    # -- adjacency ---------------------------------------------------------
    def _ordered(self, labels: Sequence[str]) -> list[str]:
        if self.order is not None:
            return _ordered_labels(labels, self.order)
        return _auto_order(labels)

    def _neighbours(self, order: Sequence[str]) -> dict[str, set[str]]:
        if self.adjacency is not None:
            graph: dict[str, set[str]] = {label: set() for label in order}
            for label in order:
                for neighbour in self.adjacency.get(label, ()):
                    if neighbour in graph:
                        graph[label].add(neighbour)
                        graph[neighbour].add(label)
            return graph
        graph = {label: set() for label in order}
        for left, right in zip(order[:-1], order[1:], strict=False):
            graph[left].add(right)
            graph[right].add(left)
        return graph

    # -- pooling -----------------------------------------------------------
    def _pool_groups(self, counts: Mapping[str, int]) -> list[list[str]]:
        order = self._ordered(list(counts))
        if self.pooling == "none":
            return [[label] for label in order]
        if self.pooling == "global":
            return [list(order)]
        graph = self._neighbours(order)
        groups: list[list[str]] = [[label] for label in order]
        group_n = {label: int(counts[label]) for label in order}
        while True:
            sizes = [sum(group_n[label] for label in group) for group in groups]
            under = [index for index, size in enumerate(sizes) if size < self.min_count]
            if not under:
                break
            # Merge the smallest too-small group first; ties by chain position.
            index = min(under, key=lambda item: (sizes[item], item))
            group = groups[index]
            candidates = [
                other
                for other, candidate in enumerate(groups)
                if other != index
                and any(
                    neighbour in graph.get(label, ())
                    for label in group
                    for neighbour in candidate
                )
            ]
            if not candidates:
                break
            # Prefer the neighbour carrying the most rows; then the nearest in
            # the pre-registered order; then the earlier group.
            partner = max(
                candidates,
                key=lambda other: (sizes[other], -abs(other - index), -other),
            )
            left, right = sorted((index, partner))
            groups[left] = groups[left] + groups[right]
            del groups[right]
        return groups

    # -- fit ---------------------------------------------------------------
    def fit(self, scores, strata) -> dict:
        """Fit per-stratum thresholds on calibration rows only.

        Parameters
        ----------
        scores : (n_cal,) nonconformity scores (higher = worse).
        strata : (n_cal,) pre-registered stratum label per row.

        Returns
        -------
        dict
            ``method="mondrian_split_rank"``, ``alpha``, ``min_count``,
            ``pooling``, per-stratum ``thresholds``/``order_stat_index``/
            ``n_cal``/``achievable_coverage``/``groups``/``group_of``,
            ``group_thresholds``/``group_n_cal``, ``pooling_notes``,
            ``uncalibrated``, ``finite_sample_exact`` and ``null_reason``.
            Per-stratum ``n_cal`` is the **pooled group** size; ``groups`` names
            every member of that group.
        """
        values = _as_finite_vector(scores, "scores")
        labels = _as_label_vector(strata, "strata")
        if values.size != labels.size:
            raise ValueError(
                f"scores and strata must have equal length, got {values.size} and "
                f"{labels.size}"
            )
        counts = {
            label: int(np.sum(labels == label))
            for label in sorted(set(labels.tolist()))
        }
        groups = self._pool_groups(counts)

        group_records: dict[str, dict] = {}
        uncalibrated: list[str] = []
        for members in groups:
            key = "+".join(members)
            mask = np.isin(labels, members)
            n = int(mask.sum())
            rank = finite_sample_rank(n, self.alpha)
            if n < self.min_count:
                calibrated = False
                reason = "n_below_min_count"
            elif rank > n:
                calibrated = False
                reason = "rank_exceeds_n"
            else:
                calibrated = True
                reason = ""
            threshold = exact_quantile(values[mask], self.alpha) if calibrated else None
            group_records[key] = {
                "members": list(members),
                "n": n,
                "order_stat_index": int(rank),
                "achievable_coverage": rank / float(n + 1),
                "threshold": threshold,
                "calibrated": calibrated,
                "reason": reason,
            }
            if not calibrated:
                uncalibrated.extend(members)

        if uncalibrated and self.on_insufficient == "raise":
            details = [
                f"{record['members']}: n={record['n']} ({record['reason']})"
                for record in group_records.values()
                if not record["calibrated"]
            ]
            raise InsufficientCalibrationError(
                "Mondrian strata cannot be calibrated at alpha="
                f"{self.alpha} with min_count={self.min_count}: " + "; ".join(details)
            )

        thresholds: dict[str, float | None] = {}
        ranks: dict[str, int] = {}
        sizes: dict[str, int] = {}
        achievable: dict[str, float | None] = {}
        group_of: dict[str, str] = {}
        groups_of: dict[str, list[str]] = {}
        for key, record in group_records.items():
            for member in record["members"]:
                thresholds[member] = record["threshold"]
                ranks[member] = record["order_stat_index"]
                sizes[member] = record["n"]
                achievable[member] = (
                    record["achievable_coverage"] if record["calibrated"] else None
                )
                group_of[member] = key
                groups_of[member] = list(record["members"])

        pooling_notes: list[dict] = []
        singleton = {label: counts[label] for label in counts}
        for key, record in group_records.items():
            if len(record["members"]) < 2:
                continue
            if self.pooling == "global":
                reason = "global_pool"
            else:
                reason = "adjacent_pool_below_min_count"
            pooling_notes.append(
                {
                    "group": key,
                    "members": list(record["members"]),
                    "n": record["n"],
                    "min_count": self.min_count,
                    "reason": reason,
                    "below_min_count_members": [
                        member
                        for member in record["members"]
                        if singleton[member] < self.min_count
                    ],
                    "calibrated": record["calibrated"],
                }
            )

        calibrated_groups = [
            record for record in group_records.values() if record["calibrated"]
        ]
        document = {
            "method": "mondrian_split_rank",
            "alpha": self.alpha,
            "min_count": self.min_count,
            "pooling": self.pooling,
            "thresholds": thresholds,
            "order_stat_index": ranks,
            "n_cal": sizes,
            "achievable_coverage": achievable,
            "groups": groups_of,
            "group_of": group_of,
            "group_thresholds": {
                key: record["threshold"] for key, record in group_records.items()
            },
            "group_n_cal": {key: record["n"] for key, record in group_records.items()},
            "group_order_stat_index": {
                key: record["order_stat_index"] for key, record in group_records.items()
            },
            "group_calibrated": {
                key: record["calibrated"] for key, record in group_records.items()
            },
            "group_reasons": {
                key: record["reason"] for key, record in group_records.items()
            },
            "pooling_notes": pooling_notes,
            "pooled": bool(pooling_notes),
            "uncalibrated": sorted(uncalibrated),
            "n_strata": len(counts),
            "n_groups": len(group_records),
            "n_cal_total": int(values.size),
            "finite_sample_exact": len(calibrated_groups) == len(group_records),
            "null_reason": "" if not uncalibrated else NullReason.INSUFFICIENT_N.value,
        }
        self.document_ = document
        return document

    # -- application -------------------------------------------------------
    def _require_document(self) -> dict:
        if self.document_ is None:
            raise RuntimeError("MondrianCalibrator.fit has not been called")
        return self.document_

    @property
    def thresholds(self) -> dict[str, float | None]:
        """Per-stratum threshold (``None`` for a flagged stratum)."""
        return dict(self._require_document()["thresholds"])

    @property
    def groups(self) -> dict[str, list[str]]:
        """Per-stratum pooled group members (the stratum itself included)."""
        return {
            key: list(value)
            for key, value in self._require_document()["groups"].items()
        }

    def threshold_for(self, stratum: str) -> float:
        """Return the calibrated threshold for ``stratum`` or raise."""
        document = self._require_document()
        try:
            threshold = document["thresholds"][stratum]
        except KeyError as error:
            known = sorted(document["thresholds"])
            raise ValueError(
                f"no calibration threshold for stratum {stratum!r}; known strata={known}"
            ) from error
        if threshold is None:
            raise InsufficientCalibrationError(
                f"stratum {stratum!r} was not calibrated (n_cal="
                f"{document['n_cal'][stratum]}, min_count={self.min_count}, "
                f"alpha={self.alpha})"
            )
        return float(threshold)

    def apply(self, test_scores, strata) -> np.ndarray:
        """Membership ``score <= threshold[stratum]`` for test rows."""
        document = self._require_document()
        values = _as_finite_vector(test_scores, "test_scores")
        labels = _as_label_vector(strata, "strata")
        if values.size != labels.size:
            raise ValueError(
                f"test_scores and strata must have equal length, got {values.size} and "
                f"{labels.size}"
            )
        missing = sorted(set(labels.tolist()) - set(document["thresholds"]))
        if missing:
            raise ValueError(
                f"no calibration threshold for strata {missing}; a test stratum "
                "cannot borrow the marginal threshold silently"
            )
        limits = np.asarray([self.threshold_for(label) for label in labels])
        return values <= limits

    def evaluate(
        self,
        y,
        lo,
        hi,
        strata,
        *,
        min_count: int = MIN_COUNT,
    ) -> list[dict]:
        """Per-stratum coverage/width, carrying the fit-time pooling notes."""
        document = self._require_document()
        return stratum_metrics(
            y,
            lo,
            hi,
            strata,
            min_count=min_count,
            pooled=document["groups"],
        )


# ---------------------------------------------------------------------------
# per-stratum reporting
# ---------------------------------------------------------------------------
def stratum_metrics(
    y,
    lo,
    hi,
    strata,
    *,
    min_count: int = MIN_COUNT,
    pooled: Mapping[str, Sequence[str]] | None = None,
    order: Sequence[str] | None = None,
) -> list[dict]:
    """Coverage and width per pre-registered stratum.

    Parameters
    ----------
    y, lo, hi : (n,) labels and interval bounds, native units.
    strata : (n,) stratum label per row.
    min_count : below this ``n`` the row is descriptive only (default 20, the
        plan's reporting floor -- *not* the calibrator's ``min_count``).
    pooled : optional mapping ``stratum -> pooled group members`` from
        :meth:`MondrianCalibrator.fit`; written into ``pooled_strata`` and the
        notes so a pooled row is never mistaken for a clean per-stratum one.
    order : optional display order; defaults to sorted labels.

    Returns
    -------
    list[dict]
        One dict per stratum with ``stratum``, ``n``, ``n_inside``,
        ``n_inverted``, ``coverage``, ``width`` (mean), ``median_width``,
        ``descriptive_only``, ``pooled_strata``, ``notes`` and ``null_reason``.
        ``coverage``/``width`` are ``None`` for an empty or non-finite slice,
        with an explicit ``null_reason``.
    """
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    lo_arr = np.asarray(lo, dtype=np.float64).reshape(-1)
    hi_arr = np.asarray(hi, dtype=np.float64).reshape(-1)
    if not (y_arr.size == lo_arr.size == hi_arr.size):
        raise ValueError("y, lo and hi must have equal length")
    labels = _as_label_vector(strata, "strata")
    if labels.size != y_arr.size:
        raise ValueError(
            f"strata must have one label per row, got {labels.size} labels for "
            f"{y_arr.size} rows"
        )
    rows: list[dict] = []
    for stratum in _ordered_labels(labels.tolist(), order):
        mask = labels == stratum
        result = coverage(y_arr[mask], lo_arr[mask], hi_arr[mask], min_count=min_count)
        finite = (
            np.isfinite(y_arr[mask])
            & np.isfinite(lo_arr[mask])
            & np.isfinite(hi_arr[mask])
            & (lo_arr[mask] <= hi_arr[mask])
        )
        width_values = hi_arr[mask][finite] - lo_arr[mask][finite]
        n = int(result["n"])
        group = (
            list(pooled[stratum]) if pooled is not None and stratum in pooled else []
        )
        notes: list[str] = []
        if len(group) > 1:
            notes.append(f"pooled_group={','.join(group)} (n={n})")
        if n < min_count:
            notes.append(f"descriptive_only: n={n} < min_count={min_count}")
        if group:
            pooled_strata = group
        else:
            pooled_strata = []
        rows.append(
            {
                "stratum": stratum,
                "n": n,
                "n_inside": int(result["n_inside"]),
                "n_inverted": int(result["n_inverted"]),
                "coverage": result["coverage"],
                "width": float(np.mean(width_values)) if width_values.size else None,
                "median_width": (
                    float(np.median(width_values)) if width_values.size else None
                ),
                "descriptive_only": descriptive_only(n, min_count=min_count),
                "pooled_strata": pooled_strata,
                "notes": "; ".join(notes),
                "null_reason": result["null_reason"],
            }
        )
    return rows
