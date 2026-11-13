"""One label set per star from the master spectroscopic table.

Used by the stopgap fine-tune (the July pretrain + finetune_msa) after the
lineage of the multi-survey trainer was lost. There are no survey maps here:
each parameter comes from the best record that has it, ranked by tier
(high-resolution references, then APOGEE/GALAH, then metal-poor surveys,
then other surveys), then by the smaller quoted error. Records must pass
their survey's own ``fit_ok``. [M/H] (GSP-Spec, MaStar) is not used as
[Fe/H]; [alpha/Fe] and [alpha/M] are used as the alpha label.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARAMETERS = ("teff", "logg", "fe_h", "alpha")
TOP_SURVEYS = {"apogee_dr17", "galah_dr4"}
# Second pipelines on the same spectra, used only when nothing else exists.
DUPLICATE_PIPELINES = {"desi_dr1_sp", "lamost_dr12_mrs_cnn"}
ALPHA_KINDS = {"alpha_fe", "alpha_m"}


def record_priority(records: pd.DataFrame) -> np.ndarray:
    """Lower is better: 0 high-res, 1 APOGEE/GALAH, 2 metal-poor, 3 survey."""
    tier = records["tier"].astype(str).to_numpy()
    survey = records["survey"].astype(str).to_numpy()
    prio = np.full(len(records), 3, dtype="i4")
    prio[tier == "metal_poor_survey"] = 2
    prio[np.isin(survey, list(TOP_SURVEYS))] = 1
    prio[tier == "highres_reference"] = 0
    prio[survey == "gaia_dr3_gspspec"] = 4
    prio[np.isin(survey, list(DUPLICATE_PIPELINES))] = 5
    return prio


def _values(records: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray]:
    value = records[name].to_numpy("f8").copy()
    error = records["e_" + name].to_numpy("f8").copy()
    if name == "fe_h":
        kind = records["fe_h_kind"].astype(str).to_numpy()
        value[~np.char.startswith(kind.astype("U"), "fe_h")] = np.nan
        corrected = records["fe_h_corrected"].to_numpy("f8")
        use = np.isfinite(corrected)  # DESI published correction inside its box
        value[use] = corrected[use]
    if name == "alpha":
        kind = records["alpha_kind"].astype(str).to_numpy()
        value[~np.isin(kind, list(ALPHA_KINDS))] = np.nan
    error[~(np.isfinite(error) & (error > 0))] = np.nan
    return value, error


def select_labels(records: pd.DataFrame) -> pd.DataFrame:
    """Per Gaia star: best value, error and survey for each parameter."""
    ok = (records["source_id"].to_numpy("i8") > 0) & records["fit_ok"].fillna(
        False
    ).to_numpy(bool)
    r = records.loc[ok].reset_index(drop=True)
    prio = record_priority(r)
    sid = r["source_id"].to_numpy("i8")
    out = pd.DataFrame(index=pd.Index(np.unique(sid), name="source_id"))
    for name in PARAMETERS:
        value, error = _values(r, name)
        keep = np.isfinite(value)
        rank = np.where(np.isfinite(error), error, np.inf)
        frame = pd.DataFrame(
            dict(
                source_id=sid[keep],
                prio=prio[keep],
                rank=rank[keep],
                value=value[keep],
                error=error[keep],
                survey=r["survey"].astype(str).to_numpy()[keep],
            )
        )
        best = frame.sort_values(["source_id", "prio", "rank"], kind="stable")
        best = best.drop_duplicates("source_id").set_index("source_id")
        out[name] = best["value"]
        out["e_" + name] = best["error"]
        out["src_" + name] = best["survey"]
        out["prio_" + name] = best["prio"]
    out = out[np.isfinite(out[list(PARAMETERS)]).any(axis=1)]
    return out


def choose_stars(labels: pd.DataFrame, max_stars: int, seed: int = 42) -> pd.Index:
    """All high-priority or metal-poor stars, then a random fill to ``max_stars``."""
    prio = labels[[f"prio_{p}" for p in PARAMETERS]].min(axis=1)
    feh = labels["fe_h"].to_numpy("f8")
    keep = (prio.to_numpy() <= 2) | (np.isfinite(feh) & (feh < -1.0))
    chosen = labels.index[keep]
    rest = labels.index[~keep]
    room = max(0, max_stars - len(chosen))
    if room < len(rest):
        rest = pd.Index(
            np.sort(np.random.default_rng(seed).choice(rest.to_numpy(), room, False))
        )
    return chosen.append(rest).sort_values()
