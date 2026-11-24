"""Physical references: IRFM Teff, seismic log g, Gaia FGK benchmarks (GBS v3).

IRFM and seismic stars are training anchors of the v6+ fine-tunes (identity
maps), so held-out residuals test transfer, not the gauge (qa.md A1, A2).
GBS v3 is validation-only and QA-reserved in msa-split-v1; its exact DR3 ids
come from ``scripts/fix_gbs_dr3_ids.py`` (``training.anchor_labels``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .stats import summary

SEISMIC = (
    "anchor_apokasc3",
    "anchor_apok2",
    "anchor_apokasc_dwarfs",
    "anchor_kepler_legacy",
)


def load_physical(catalogues=None, gbs_path=None, log=print) -> pd.DataFrame:
    """Seismic log g (four catalogues) and GBS v3 labels, one row per (survey, star).

    Missing catalogues are skipped with a message (the evaluator reports them).
    """
    from masked_stellar_autoencoder.training import anchor_labels as A

    root = Path(catalogues) if catalogues is not None else A.CATALOGUES
    parts = []
    try:
        kic = A._kic_to_gaia(root)
    except (OSError, FileNotFoundError) as exc:
        log(f"physical: no Kepler id map ({exc}); dwarfs and LEGACY skipped")
        kic = None
    loaders = [("anchor_apokasc3", lambda: A.apokasc3(root)),
               ("anchor_apok2", lambda: A.apok2(root))]  # fmt: skip
    if kic is not None:
        loaders += [("anchor_apokasc_dwarfs", lambda: A.apokasc_dwarfs(root, kic)),
                    ("anchor_kepler_legacy", lambda: A.kepler_legacy(root, kic))]  # fmt: skip
    loaders.append(("gbs_v3", lambda: A.gaia_benchmarks(root, path=gbs_path)))
    for name, load in loaders:
        try:
            parts.append(load())
        except (OSError, FileNotFoundError, KeyError) as exc:
            log(f"physical: {name} not loaded ({exc})")
    if not parts:
        return pd.DataFrame(columns=["survey", "source_id", "teff", "logg", "fe_h"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["survey", "source_id"])


def score_physical(refs: pd.DataFrame, source_id, pred: dict[str, np.ndarray],
                   strata=None) -> dict:  # fmt: skip
    """bias / NMAD of prediction - reference per catalogue and parameter.

    ``pred`` maps teff / logg / fe_h to point predictions aligned with
    ``source_id``; ``strata`` (optional dict name -> labels aligned with
    ``source_id``) splits each cell, e.g. by evolutionary stage.
    """
    sid = np.asarray(source_id, np.int64)
    order = np.argsort(sid)
    ss = sid[order]
    out = {}
    for survey, g in refs.groupby("survey"):
        rid = g["source_id"].to_numpy(np.int64)
        pos = np.searchsorted(ss, rid).clip(max=max(len(ss) - 1, 0))
        hit = (ss[pos] == rid) if len(ss) else np.zeros(len(rid), bool)
        idx = order[pos[hit]]
        cell = dict(references=int(len(g)), predicted=int(hit.sum()))
        for p in ("teff", "logg", "fe_h"):
            if p not in pred or p not in g:
                continue
            r = g[p].to_numpy(np.float64)[hit]
            d = pred[p][idx] - r
            if np.isfinite(d).sum() == 0:
                continue
            cell[p] = summary(d)
            if strata:
                for name, lab in strata.items():
                    labs = np.asarray(lab)[idx]
                    cell[f"{p}_by_{name}"] = {
                        s: summary(d[labs == s]) for s in sorted(set(labs)) if s
                    }
        out[str(survey)] = cell
    return out


def score_irfm(teff_pred, teff_irfm) -> dict:
    """IRFM Teff (Casagrande+2021 via training.irfm_teff) against prediction."""
    return summary(
        np.asarray(teff_pred, np.float64) - np.asarray(teff_irfm, np.float64)
    )
