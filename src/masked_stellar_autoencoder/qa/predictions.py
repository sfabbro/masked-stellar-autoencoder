"""Prediction files for the evaluator: one canonical layout plus adapters.

Canonical layout (``.npz``; ``write_predictions`` writes it):

* ``source_id`` (N,) int64 Gaia DR3 ids; ``model`` (str), optional ``kind``
  (``A`` measurement, ``B`` predictive, ...).
* ``quantile_levels`` (K,) in (0, 1), sorted, and per product ``<p>_q`` (N, K)
  in physical units, or ``<p>_samples`` (N, S) (quantiles are then computed).
* optional per product: ``<p>_ref`` / ``<p>_ref_err`` (the reference the
  model is scored against), ``<p>_pit`` (PIT of that reference, when the
  predictive is not a set of quantiles), ``<p>_cov68`` / ``<p>_cov95``
  (membership of the reference in the model's own 68 / 95 % set, e.g. a
  conformal or Neyman set).
* optional covariates: ``G``, ``bp_rp``, ``ebv_sfd``, ``table_row``,
  ``table_split`` (row split of a legacy table: 0 train, 1 valid, 2 test);
  ``p_feh_lt_m2p5``.

Products: ``teff`` (K), ``logg`` (dex), ``fe_h`` (dex), ``ebv`` (mag),
``parallax`` (mas), ``dist_mod`` (mag). Adapters convert the v10-era files:
``posterior`` (``predict_fits_rows`` output: multi_*.npz, consistency_*.npz),
``s2`` (``s2_predictions_<family>.npz``), ``s3`` (``s3_predictions_<family>.npz``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PRODUCTS = ("teff", "logg", "fe_h", "ebv", "parallax", "dist_mod")
UNITS = dict(
    teff="K", logg="dex", fe_h="dex", ebv="mag", parallax="mas", dist_mod="mag"
)
LEVELS_68 = np.array([0.16, 0.5, 0.84])


@dataclass
class Predictions:
    model: str
    source_id: np.ndarray
    levels: np.ndarray
    q: dict[str, np.ndarray] = field(default_factory=dict)
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    kind: str = ""

    def __len__(self) -> int:
        return len(self.source_id)

    def median(self, p: str) -> np.ndarray:
        k = int(np.argmin(np.abs(self.levels - 0.5)))
        return self.q[p][:, k]

    def subset(self, mask) -> Predictions:
        return Predictions(
            self.model, self.source_id[mask], self.levels,
            {k: v[mask] for k, v in self.q.items()},
            {k: (v[mask] if np.ndim(v) and len(v) == len(self.source_id) else v)
             for k, v in self.extra.items()},
            self.kind,
        )  # fmt: skip


def write_predictions(path, pred: Predictions) -> None:
    out = dict(source_id=pred.source_id, quantile_levels=pred.levels, model=pred.model,
               kind=pred.kind)  # fmt: skip
    out.update({f"{p}_q": v for p, v in pred.q.items()})
    out.update(pred.extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)


def _canonical(z, model) -> Predictions:
    levels = np.asarray(z["quantile_levels"], np.float64)
    q, extra = {}, {}
    for k in z.files:
        if k.endswith("_q") and k[:-2] in PRODUCTS:
            q[k[:-2]] = np.asarray(z[k], np.float64)
        elif k.endswith("_samples") and k[:-8] in PRODUCTS:
            q[k[:-8]] = np.quantile(np.asarray(z[k], np.float64), levels, axis=1).T
        elif k not in ("source_id", "quantile_levels", "model", "kind"):
            extra[k] = z[k]
    name = model or (str(z["model"]) if "model" in z.files else "model")
    kind = str(z["kind"]) if "kind" in z.files else ""
    return Predictions(
        name, np.asarray(z["source_id"], np.int64), levels, q, extra, kind
    )


def _posterior(z, model) -> Predictions:
    """predict_fits_rows output: 16/50/84 % quantiles of the posterior head."""
    names = [str(n) for n in z["truth_names"]]
    quant = np.asarray(z["quantiles"], np.float64)  # (N, P, 3)
    rename = dict(ebv_star="ebv")
    q, extra = {}, {}
    for j, n in enumerate(names):
        p = rename.get(n, n)
        if p in PRODUCTS:
            q[p] = quant[:, j]
            extra[f"{p}_ref"] = np.asarray(z["truth"][:, j], np.float64)
    if quant.shape[1] > len(names):  # the last label is the (log10) parallax
        q["parallax"] = quant[:, len(names)]
    for k in (
        "G",
        "EBV",
        "table_row",
        "teff_irfm",
        "e_teff_irfm",
        "parallax_conditioned",
    ):
        if k in z.files:
            extra["ebv_sfd" if k == "EBV" else k] = z[k]
    if "p_feh_below_m2p5" in z.files:
        extra["p_feh_lt_m2p5"] = z["p_feh_below_m2p5"]
    return Predictions(model or "posterior", np.asarray(z["source_id"], np.int64),
                       LEVELS_68.copy(), q, extra, "B")  # fmt: skip


def _s2(z, model) -> Predictions:
    """s2_distributional_heads: median, q16, q84 of (Teff, log g, [Fe/H])."""
    q = {
        p: np.column_stack([z["q16"][:, j], z["median"][:, j], z["q84"][:, j]]).astype(
            "f8"
        )
        for j, p in enumerate(("teff", "logg", "fe_h"))
    }
    extra = {"table_row": z["table_row"], "table_split": z["split"]}
    if "p_mp" in z.files:
        extra["p_feh_lt_m2p5"] = z["p_mp"]
    return Predictions(model or "s2", np.asarray(z["source_id"], np.int64),
                       LEVELS_68.copy(), q, extra, "B0")  # fmt: skip


def _s3(z, model) -> Predictions:
    """s3_distance_extinction: s = ln(d / kpc) and E(B-V) quantiles, PITs."""
    s_q = np.asarray(z["s_q"], np.float64)
    q = {
        "dist_mod": 5.0 * s_q / np.log(10.0) + 10.0,  # 5 log10(d / 10 pc)
        "ebv": np.asarray(z["e_q"], np.float64),
    }
    extra = {
        "table_row": z["table_row"], "table_split": z["split"], "G": z["G"],
        "parallax_pit": z["pit_plx"], "ebv_pit": z["pit_e"],
    }  # fmt: skip
    return Predictions(model or "s3", np.asarray(z["source_id"], np.int64),
                       LEVELS_68.copy(), q, extra, "B")  # fmt: skip


ADAPTERS = dict(canonical=_canonical, posterior=_posterior, s2=_s2, s3=_s3)


def detect(files) -> str:
    files = set(files)
    if "quantile_levels" in files:
        return "canonical"
    if {"truth_names", "quantiles", "median"} <= files:
        return "posterior"
    if {"q16", "q84", "conformal_group"} <= files:
        return "s2"
    if {"s_q", "e_q", "pit_plx"} <= files:
        return "s3"
    raise ValueError(f"unrecognised prediction file layout: {sorted(files)[:12]}")


def load_predictions(
    path, adapter: str = "auto", model: str | None = None
) -> Predictions:
    with np.load(path, allow_pickle=False) as z:
        name = detect(z.files) if adapter == "auto" else adapter
        pred = ADAPTERS[name](z, model)
    if np.asarray(pred.source_id).dtype.kind != "i":
        raise ValueError(f"{path}: source_id must be integer")
    return pred
