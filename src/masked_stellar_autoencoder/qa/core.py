"""Frozen high-resolution [Fe/H] core: references, conditional bias, metal-poor selection.

Brought into the repo from the ARC-only scripts of the QA audit:

* ``rescore_analysis.py`` -> ``core_references`` (one reference per table row:
  original high-resolution papers first, then Gaia-ESO, then compilations, a
  fixed order independent of the model);
* ``estimator_table.py`` -> ``core_table`` (median offset and NMAD per
  reference [Fe/H] bin, crossed with G bins as in design 2026-11-20 §5);
* ``tail_prob.py`` -> ``tail_probability`` (Brier score, AUC and purity of
  P([Fe/H] < -2.5)); ``selection`` gives purity and completeness of a
  metal-poor selection rule per G bin (qa.md S1).
"""

from __future__ import annotations

import numpy as np

from .stats import CORE_G_BINS, FEH_BINS, MIN_N, summary

ORDER = (
    "bensby2014", "aoki2013_segue", "li2022_lamost_subaru", "smss_yong2021",
    "yong2013_program", "yong2013_literature", "rpa_hansen2018", "rpa_holmbeck2020",
    "sakari2018_rpa", "ezzeddine2020_rpa", "roederer2014", "kirby2010_highres",
    "gaia_eso_dr5_1", "jinabase_2021", "saga_mw", "pastel_2020",
)  # fmt: skip
G_NAMES = ("G<=13.5", "13.5-15", "15-16", "16-17.65")
MP_LIMIT = -2.5


def core_references(rec, core_prefixes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fe_h_ref, e_fe_h_ref, survey index) per table row from ``records.npz``."""
    names = [str(n) for n in rec["survey_names"]]
    fe = list(rec["parameters"]).index("fe_h")
    val, err = rec["value"][:, fe], rec["error"][:, fe]
    row, sv = rec["row"], rec["survey"]
    n = int(row.max()) + 1
    ref, eref = np.full(n, np.nan), np.full(n, np.nan)
    src = np.full(n, -1, np.int64)
    core = [i for i, s in enumerate(names) if s.startswith(tuple(core_prefixes))]
    order = [names.index(s) for s in ORDER if s in names]
    order += [i for i in core if i not in order]
    for s in reversed(order):  # earlier in ORDER overwrites later
        m = (sv == s) & np.isfinite(val)
        ref[row[m]], eref[row[m]], src[row[m]] = val[m], err[m], s
    return ref, eref, src


def core_table(pred, ref, G) -> dict:
    """Median offset (pred - ref) and NMAD per [Fe/H]_ref bin x G bin."""
    pred, ref, G = (np.asarray(a, np.float64) for a in (pred, ref, G))
    ok = np.isfinite(pred) & np.isfinite(ref) & np.isfinite(G)
    out = {}
    for lo, hi in zip(FEH_BINS[:-1], FEH_BINS[1:], strict=True):
        cells = {}
        for name, glo, ghi in zip(
            G_NAMES, CORE_G_BINS[:-1], CORE_G_BINS[1:], strict=True
        ):
            m = ok & (ref >= lo) & (ref < hi) & (glo < G) & (ghi >= G)
            cells[name] = summary(pred[m] - ref[m])
        m = ok & (ref >= lo) & (ref < hi) & (G <= 16.0)
        cells["G<=16"] = summary(pred[m] - ref[m])
        out[f"[{lo},{hi})"] = cells
    return out


def worst_bias(table: dict, column: str = "G<=16") -> dict:
    """Largest |bias| over the [Fe/H] bins with n >= MIN_N (S1 decision rule 1)."""
    best = dict(bin=None, bias=float("nan"), n=0)
    for b, cells in table.items():
        c = cells[column]
        if c["n"] >= MIN_N and not np.isnan(c["bias"]):
            if best["bin"] is None or abs(c["bias"]) > abs(best["bias"]):
                best = dict(bin=b, bias=c["bias"], n=c["n"])
    return best


def selection(ref, G, selected, limit: float = MP_LIMIT) -> dict:
    """Purity and completeness of ``selected`` for [Fe/H]_ref < limit, per G bin."""
    ref, G = np.asarray(ref, np.float64), np.asarray(G, np.float64)
    selected = np.asarray(selected, bool)
    ok = np.isfinite(ref) & np.isfinite(G)
    out = {}
    bins = (("G<=16", -np.inf, 16.0), ("G>16", 16.0, np.inf), ("all", -np.inf, np.inf))
    for name, lo, hi in bins:
        m = ok & (lo < G) & (hi >= G)
        truth = m & (ref < limit)
        sel = m & selected
        tp = int((truth & sel).sum())
        out[name] = dict(
            n_true=int(truth.sum()), n_selected=int(sel.sum()), true_positives=tp,
            purity=tp / sel.sum() if sel.sum() else float("nan"),
            completeness=tp / truth.sum() if truth.sum() else float("nan"),
        )  # fmt: skip
    return out


def tail_probability(ref, p, limit: float = MP_LIMIT) -> dict:
    """Brier score, AUC and calibration of P([Fe/H] < limit) on references."""
    ref, p = np.asarray(ref, np.float64), np.asarray(p, np.float64)
    ok = np.isfinite(ref) & np.isfinite(p)
    y, q = ref[ok] < limit, np.clip(p[ok], 0, 1)
    out = dict(n=int(ok.sum()), n_true=int(y.sum()))
    if not len(y):
        return out
    out["brier"] = float(np.mean((q - y) ** 2))
    if 0 < y.sum() < len(y):
        from sklearn.metrics import roc_auc_score

        out["auc"] = float(roc_auc_score(y, q))
    out["mean_p"] = float(q.mean())
    for t in (0.5, 0.9):
        s = q > t
        out[f"p_gt_{t}"] = dict(
            n=int(s.sum()), frac_true=float(y[s].mean()) if s.any() else np.nan
        )
    return out
