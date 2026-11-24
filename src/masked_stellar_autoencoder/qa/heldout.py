"""Held-out label checks per product and per pre-registered stratum (qa.md §5.2-5.3).

For each product with a reference: bias / NMAD overall and per stratum, the
68 % (and 95 %) coverage of the reference with Wilson bands per stratum, PIT
uniformity and pulls. Coverage is of the (noisy) reference label, not of the
true parameter (qa.md §0).
"""

from __future__ import annotations

import numpy as np

from .predictions import Predictions
from .stats import (
    MIN_N,
    STRATA,
    assign,
    central_interval,
    pit_from_quantiles,
    summary,
    wilson_by_stratum,
)

AXES = (
    "G",
    "bp_rp",
    "feh_ref",
    "feh_pred",
    "stage",
    "abs_b",
    "ebv_sfd",
    "input_class",
    "ruwe",
)


def strata_for(pred: Predictions, cov: dict, refs: dict) -> dict[str, np.ndarray]:
    """Stratum labels per axis for the rows of ``pred`` ('' = unknown)."""
    from .stats import stage

    n = len(pred)
    nan = np.full(n, np.nan)
    out = {}
    out["G"] = assign("G", cov.get("G", nan))
    out["bp_rp"] = assign("bp_rp", cov.get("bp_rp", nan))
    out["feh_ref"] = assign("feh_ref", refs.get("fe_h", nan))
    out["feh_pred"] = assign(
        "feh_pred", pred.median("fe_h") if "fe_h" in pred.q else nan
    )
    teff = refs.get("teff", nan)
    logg = refs.get("logg", nan)
    if "teff" in pred.q and "logg" in pred.q:  # stage of the prediction where no label
        teff = np.where(np.isfinite(teff), teff, pred.median("teff"))
        logg = np.where(np.isfinite(logg), logg, pred.median("logg"))
    out["stage"] = stage(teff, logg)
    out["abs_b"] = assign("abs_b", np.abs(np.asarray(cov.get("b", nan), np.float64)))
    out["ebv_sfd"] = assign("ebv_sfd", cov.get("ebv_sfd", nan))
    out["input_class"] = assign("input_class", cov.get("input_class", np.full(n, "")))
    out["ruwe"] = assign("ruwe", cov.get("ruwe", nan))
    return out


def _coverage(pred: Predictions, p: str, ref, mass: float):
    """Membership of the reference in the central ``mass`` set, or None."""
    own = pred.extra.get(f"{p}_cov{int(round(mass * 100))}")
    if own is not None:
        return np.asarray(own, bool), "model set"
    iv = central_interval(pred.levels, pred.q[p], mass) if p in pred.q else None
    if iv is None:
        return None, None
    lo, hi = iv
    return (ref >= lo) & (ref <= hi), "central quantile interval"


def evaluate_product(
    pred: Predictions, p: str, ref, strata: dict, full_label=None
) -> dict:
    """All held-out label metrics of one product."""
    from masked_stellar_autoencoder.reporting.metrics import pit_diagnostics

    ref = np.asarray(ref, np.float64)
    med = pred.median(p)
    ok = np.isfinite(ref) & np.isfinite(med)
    out = dict(
        n=int(ok.sum()), overall=summary(med[ok] - ref[ok]), by_stratum={}, coverage={}
    )
    if not ok.any():
        return out
    q = pred.q[p]
    iv = central_interval(pred.levels, q, 0.68)
    sd = 0.5 * (iv[1] - iv[0]) if iv is not None else np.full(len(q), np.nan)
    pull = (med - ref) / np.where(sd > 0, sd, np.nan)
    out["pull"] = dict(
        median=float(np.nanmedian(pull[ok])), nmad=summary(pull[ok])["nmad"]
    )
    if pred.kind != "A":  # confidence limits of a measurement are not a CDF
        pit = pit_from_quantiles(pred.levels, q, ref)
        if np.isfinite(pit[ok]).sum() >= MIN_N:
            diag = pit_diagnostics(pit[ok & np.isfinite(pit)])
            out["pit"] = {
                k: diag[k]
                for k in ("n", "pit_ks", "pit_ks_pvalue", "pit_mean", "pit_var")
            }
    for axis in AXES:
        lab = strata[axis]
        cells = {}
        for s in STRATA[axis][1]:
            m = ok & (lab == s)
            if m.any():
                cells[s] = summary(med[m] - ref[m])
        if cells:
            out["by_stratum"][axis] = cells
    for mass in (0.68, 0.95):
        covered, how = _coverage(pred, p, ref, mass)
        if covered is None:
            continue
        sel = ok
        if how == "model set" and full_label is not None:
            # the model's own sets are computed on the full (Teff, log g, [Fe/H])
            # label; rows without it carry no membership
            sel = ok & np.asarray(full_label, bool)
        cov = dict(
            method=how,
            overall=float(np.mean(covered[sel])) if sel.any() else float("nan"),
            n=int(sel.sum()),
            by_stratum={},
        )
        for axis in AXES:
            rows = wilson_by_stratum(np.where(sel, strata[axis], ""), covered, mass,
                                     order=list(STRATA[axis][1]))  # fmt: skip
            if rows:
                cov["by_stratum"][axis] = rows
        cov["failing_strata"] = [
            f"{axis}:{r['stratum']}"
            for axis, rows in cov["by_stratum"].items()
            for r in rows
            if not r["descriptive"] and not r["within_003"]
        ]
        out["coverage"][f"{int(round(mass * 100))}"] = cov
    return out


def evaluate_pit(pit, strata: dict, nominal: float = 0.68) -> dict:
    """Coverage from a supplied PIT (e.g. S3 parallax / E(B-V) likelihoods)."""
    pit = np.asarray(pit, np.float64)
    ok = np.isfinite(pit)
    lo, hi = 0.5 - nominal / 2, 0.5 + nominal / 2
    covered = (pit >= lo) & (pit <= hi)
    out = dict(n=int(ok.sum()), coverage=float(np.mean(covered[ok])) if ok.any() else np.nan,
               by_stratum={})  # fmt: skip
    for axis in ("G", "bp_rp", "abs_b", "input_class", "ruwe"):
        rows = wilson_by_stratum(np.where(ok, strata[axis], ""), covered, nominal,
                                 order=list(STRATA[axis][1]))  # fmt: skip
        if rows:
            out["by_stratum"][axis] = rows
    out["failing_strata"] = [
        f"{axis}:{r['stratum']}"
        for axis, rows in out["by_stratum"].items()
        for r in rows
        if not r["descriptive"] and not r["within_003"]
    ]
    return out


def evaluate_membership(member, strata: dict, nominal: float, keep=None) -> dict:
    """Coverage of the model's own (joint) set of the label, per stratum."""
    member = np.asarray(member, bool)
    ok = np.ones(len(member), bool) if keep is None else np.asarray(keep, bool)
    out = dict(n=int(ok.sum()), coverage=float(np.mean(member[ok])) if ok.any() else np.nan,
               by_stratum={})  # fmt: skip
    for axis in AXES:
        rows = wilson_by_stratum(np.where(ok, strata[axis], ""), member, nominal,
                                 order=list(STRATA[axis][1]))  # fmt: skip
        if rows:
            out["by_stratum"][axis] = rows
    out["failing_strata"] = [
        f"{axis}:{r['stratum']}"
        for axis, rows in out["by_stratum"].items()
        for r in rows
        if not r["descriptive"] and not r["within_003"]
    ]
    return out


def integrity(pred: Predictions) -> dict:
    """Catalogue integrity (qa.md §5.3, first row): duplicates, finiteness, monotone quantiles."""
    sid = pred.source_id
    out = dict(rows=int(len(sid)), duplicate_source_ids=int(len(sid) - len(np.unique(sid))),
               products={})  # fmt: skip
    for p, q in pred.q.items():
        fin = np.isfinite(q).all(axis=1)
        mono = np.all(np.diff(q, axis=1) >= 0, axis=1) | ~fin
        out["products"][p] = dict(
            finite=float(fin.mean()), non_monotone=int((~mono).sum())
        )
    return out
