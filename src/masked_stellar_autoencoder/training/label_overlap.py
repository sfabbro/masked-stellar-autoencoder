"""Per-survey [Fe/H] overlap diagnostics against a pooled high-resolution reference.

For every label source (staged survey sidecars and master-label sources) this
measures, on stars shared with the high-resolution reference tier:

* binned offsets and robust scatter of survey − reference [Fe/H];
* the survey's apparent metallicity floor (lowest value and pile-up there);
* which survey→reference map complexity the overlap supports, by K-fold
  cross-validation: identity, offset, affine or piecewise-linear;
* how many of the survey's stars get a lower, metal-poor value from a new
  metal-poor or high-resolution source.

These are diagnostics. They fix no gauge, assign no roles and change no
training input. The reference is pooled literature (several papers per star),
not physical truth; each high-resolution source is compared against a
reference built without its own dependency group.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEH_BINS = (-np.inf, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, np.inf)
MODELS = ("identity", "offset", "affine", "piecewise_linear")
MP_SOURCE_TIERS = ("metal_poor_survey", "highres_reference")


def robust_sigma(values) -> float:
    values = np.asarray(values, dtype="f8")
    if len(values) == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def per_star(frame: pd.DataFrame, value: str = "fe_h") -> pd.Series:
    """Median value per Gaia star (rows with source_id 0 are dropped)."""
    frame = frame[(frame.source_id > 0) & np.isfinite(frame[value])]
    return frame.groupby("source_id")[value].median()


def usable(obs: pd.DataFrame) -> pd.DataFrame:
    """Quality-passing, grid-edge-free, Gaia-linked metallicity rows."""
    return obs[
        (obs.source_id > 0) & obs.quality & ~obs.fe_h_grid_edge & np.isfinite(obs.fe_h)
    ]


def reference_rows(obs: pd.DataFrame, *, exclude_group: str | None = None):
    """Usable high-res rows (source_id, fe_h, source, dependency_group)."""
    rows = usable(obs)
    rows = rows[rows.tier == "highres_reference"]
    if exclude_group is not None:
        rows = rows[rows.dependency_group != exclude_group]
    return rows[["source_id", "fe_h", "source", "dependency_group"]]


def reference(obs: pd.DataFrame, *, exclude_group: str | None = None) -> pd.Series:
    """Pooled high-res reference: per-star median over high-res rows."""
    return per_star(reference_rows(obs, exclude_group=exclude_group))


DUPLICATE_DEX = 0.011


def independent_reference(survey: pd.Series, rows: pd.DataFrame) -> pd.Series:
    """Reference per star without rows that repeat the survey's own value.

    Literature compilations (SAGA, JINAbase, PASTEL) republish the numbers of
    the papers they compile. A reference row equal to the tested value for
    the same star (within 0.011 dex, i.e. rounding) is treated as the same
    published measurement and dropped, so a paper is never scored against
    a copy of itself.
    """
    joined = rows.join(survey.rename("tested"), on="source_id", how="left")
    keep = ~(np.abs(joined.fe_h - joined.tested) <= DUPLICATE_DEX)
    return per_star(joined[keep])


def floor_stats(values, *, tolerance: float = 0.011) -> dict:
    """Lowest value and the share of stars piled up within tolerance of it."""
    values = np.asarray(values, dtype="f8")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return dict(n=0)
    low = float(values.min())
    at_floor = int((values <= low + tolerance).sum())
    return dict(
        n=int(len(values)),
        min=low,
        p001=float(np.quantile(values, 0.001)),
        at_min=at_floor,
        at_min_fraction=at_floor / len(values),
        pile_up=bool(at_floor >= 20 and at_floor / len(values) > 1e-3),
    )


def binned_offsets(x_survey, y_ref, bins=FEH_BINS) -> list[dict]:
    """Survey − reference statistics in bins of the reference metallicity."""
    x = np.asarray(x_survey, dtype="f8")
    y = np.asarray(y_ref, dtype="f8")
    out = []
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        keep = (y >= lo) & (y < hi)
        delta = x[keep] - y[keep]
        out.append(
            dict(
                ref_lo=None if np.isinf(lo) else lo,
                ref_hi=None if np.isinf(hi) else hi,
                n=int(keep.sum()),
                median_offset=float(np.median(delta)) if keep.any() else None,
                robust_sigma=robust_sigma(delta) if keep.sum() >= 3 else None,
            )
        )
    return out


def _design(x, model: str, knots=None) -> np.ndarray:
    if model == "offset":
        return np.ones((len(x), 1))
    if model == "affine":
        return np.column_stack([np.ones(len(x)), x])
    if model == "piecewise_linear":
        hinge = [np.maximum(0.0, x - k) for k in knots]
        return np.column_stack([np.ones(len(x)), x, *hinge])
    raise ValueError(model)


def fit_map(x, y, model: str, knots=None) -> np.ndarray | None:
    """Least-absolute-deviation-flavoured fit: IRLS with Huber weights."""
    if model == "identity":
        return None
    design = _design(x, model, knots)
    weights = np.ones(len(x))
    coef = np.linalg.lstsq(design, y - x, rcond=None)[0]
    for _ in range(20):
        resid = y - x - design @ coef
        scale = max(robust_sigma(resid), 1e-3)
        weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(resid), 1e-12))
        sw = np.sqrt(weights)
        coef = np.linalg.lstsq(design * sw[:, None], (y - x) * sw, rcond=None)[0]
    return coef


def predict(x, model: str, coef, knots=None) -> np.ndarray:
    if model == "identity":
        return np.asarray(x, dtype="f8")
    return x + _design(x, model, knots) @ coef


def model_selection(x_survey, y_ref, *, folds: int = 5, seed: int = 0) -> dict:
    """K-fold CV of survey→reference maps; pick the simplest adequate model.

    The map predicts the reference from the survey value as ``x + g(x)``.
    Scores are the median absolute held-out residual. The chosen model is the
    simplest whose score is within 0.005 dex or 2% of the best; maps that need
    more parameters than the overlap supports are not tried.
    """
    x = np.asarray(x_survey, dtype="f8")
    y = np.asarray(y_ref, dtype="f8")
    n = len(x)
    min_n = dict(identity=1, offset=5, affine=20, piecewise_linear=200)
    tried = [m for m in MODELS if n >= max(min_n[m], folds if m != "identity" else 1)]
    knots = np.quantile(x, [0.25, 0.5, 0.75]) if "piecewise_linear" in tried else None
    order = np.random.default_rng(seed).permutation(n)
    split = np.array_split(order, max(1, min(folds, n)))
    scores = {}
    for model in tried:
        resid = np.empty(n)
        for test in split:
            train = np.setdiff1d(order, test, assume_unique=True)
            coef = fit_map(x[train], y[train], model, knots) if len(train) else None
            resid[test] = y[test] - predict(x[test], model, coef, knots)
        scores[model] = float(np.median(np.abs(resid)))
    best = min(scores.values()) if scores else float("nan")
    chosen = next(
        (m for m in tried if scores[m] <= best + max(0.005, 0.02 * best)), None
    )
    coef = fit_map(x, y, chosen, knots) if chosen not in (None, "identity") else None
    return dict(
        n=n,
        cv_median_abs_residual=scores,
        chosen=chosen,
        coefficients=None if coef is None else [float(c) for c in coef],
        knots=None
        if knots is None or chosen != "piecewise_linear"
        else [float(k) for k in knots],
    )


def compare_to_reference(survey: pd.Series, ref_rows: pd.DataFrame) -> dict:
    """Overlap statistics of a per-star survey [Fe/H] series with the reference."""
    ref = independent_reference(survey, ref_rows)
    joined = pd.concat([survey.rename("x"), ref.rename("y")], axis=1, join="inner")
    x, y = joined.x.to_numpy(), joined.y.to_numpy()
    out = dict(overlap=int(len(joined)))
    if len(joined) == 0:
        return out
    delta = x - y
    out.update(
        median_offset=float(np.median(delta)),
        robust_sigma=robust_sigma(delta),
        bins=binned_offsets(x, y),
        maps=model_selection(x, y),
        survey_floor_on_overlap=floor_stats(x),
    )
    return out


def metal_poor_gain(existing: pd.Series, new_min: pd.Series) -> dict:
    """Existing-survey stars that a new metal-poor/high-res source puts lower."""
    joined = pd.concat(
        [existing.rename("old"), new_min.rename("new")], axis=1, join="inner"
    )
    old, new = joined.old.to_numpy(), joined.new.to_numpy()
    out = dict(overlap=int(len(joined)))
    for edge in (-2.0, -2.5, -3.0):
        out[f"crosses_below_{edge:+.1f}"] = int(((old >= edge) & (new < edge)).sum())
    out["lowered_by_0p5_and_below_-2"] = int(((new < old - 0.5) & (new < -2.0)).sum())
    return out


def read_sidecar_feh(path: str | Path, key: str) -> pd.Series:
    """Per-star observed [Fe/H] from a ``native_observations/<key>`` group."""
    import h5py

    with h5py.File(path, "r") as h5:
        group = h5["native_observations"][key]
        names = list(group.attrs.get("target_names", ("teff", "logg", "fe_h")))
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
        column = names.index("fe_h")
        ids = group["source_id"][:].astype("i8")
        value = group["value"][:, column].astype("f8")
        observed = group["observed"][:, column].astype(bool)
    frame = pd.DataFrame(dict(source_id=ids, fe_h=value))[observed & (ids > 0)]
    return per_star(frame)


def discover_sidecars(root: str | Path, log=print) -> list[dict]:
    """Every ``native_observations/<key>`` group under root (largest file per key)."""
    import h5py

    found: dict[str, list] = {}
    for path in sorted(Path(root).rglob("*.h5")):
        if path.stat().st_size > 50 * 2**30:
            continue
        try:
            with h5py.File(path, "r") as h5:
                if "native_observations" not in h5:
                    continue
                for key, group in h5["native_observations"].items():
                    if {"source_id", "value", "observed"} <= set(group):
                        found.setdefault(key, []).append(
                            (group["source_id"].shape[0], str(path))
                        )
        except OSError:
            continue
    chosen = []
    for key, options in sorted(found.items()):
        rows, path = max(options)
        chosen.append(dict(source=key, path=path, rows=rows, candidates=options))
        log(f"sidecar {key}: {rows} rows from {path}")
    return chosen


def run(obs: pd.DataFrame, existing: dict[str, pd.Series], log=print) -> dict:
    """Overlap report for staged surveys and every master-label source."""
    ref_rows_all = reference_rows(obs)
    ref_all = per_star(ref_rows_all)
    rows = usable(obs)
    new_min = rows[rows.tier.isin(MP_SOURCE_TIERS)].groupby("source_id").fe_h.min()
    report = dict(
        reference=dict(
            stars=int(len(ref_all)),
            metal_poor={f"lt_{e:+.1f}": int((ref_all < e).sum()) for e in (-2, -3)},
            sources=sorted(rows[rows.tier == "highres_reference"].source.unique()),
        ),
        staged_surveys={},
        master_sources={},
    )
    for name, series in existing.items():
        log(f"compare staged {name}: {len(series)} stars")
        entry = dict(stars=int(len(series)), floor=floor_stats(series.to_numpy()))
        entry.update(reference=compare_to_reference(series, ref_rows_all))
        entry.update(metal_poor_gain=metal_poor_gain(series, new_min))
        report["staged_surveys"][name] = entry
    for name, part in rows.groupby("source"):
        group = part.dependency_group.iloc[0]
        tier = part.tier.iloc[0]
        series = per_star(part)
        ref = (
            reference_rows(obs, exclude_group=group)
            if tier == "highres_reference"
            else ref_rows_all
        )
        log(f"compare master {name}: {len(series)} stars")
        report["master_sources"][name] = dict(
            tier=tier,
            dependency_group=group,
            stars=int(len(series)),
            floor=floor_stats(series.to_numpy()),
            reference=compare_to_reference(series, ref),
        )
    return report
