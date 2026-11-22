#!/usr/bin/env python3
"""C0 age consensus: per-catalogue maps, noise, censoring and co-natal terms from
the audit's age records alone (no network). Design note section 7,
``docs/design-20261122-age-consensus.md``; model in
``masked_stellar_autoencoder.training.age_consensus``.

usage:
  age_consensus_c0.py run   --records OUT/records.parquet --targets consistency_targets.npz \
                            --work WORK --docs DOCS [--quick]
  age_consensus_c0.py plots --work WORK --docs DOCS --clusters clusters.fits

``run`` fits the APOKASC-3-anchored solution (gauge A), the Xiang 2024-anchored
solution (gauge C), the bridge variants and the leave-one-catalogue-out refits;
it writes tables to DOCS and the per-star consensus (Parquet, large) to WORK.
``plots`` draws the QA figures from WORK into DOCS.

No label is truth. Each source enters as follows: the gauge anchor through an
identity map plus fitted noise (quoted error x learned inflation, floor,
Student-t tails, shared effect with its input family); every other catalogue
through its fitted map and noise; clusters and wide binaries through fitted
group-age latents (flat prior, small intrinsic spread, membership mixture).
Hunt & Reffert (2023) and literature cluster ages are not in the fit; they are
shown next to the fitted cluster latents for comparison only.

User decisions applied (2026-11-22): APOKASC-3 identity anchor; second complete
solution anchored on Xiang 2024; FLAME excluded; Leung 2023 and Martig 2016 for
validation only; main-sequence records of the isochrone pipelines excluded (one
rule for StarHorse, GALAH DR4 and Xiang: the catalogue's own log g >= 4.1, the
audit's dwarf class; StarHorse "MSTO" stars reach log g 4.33); ages above
13.8 Gyr kept; catalogue caps (StarHorse 13.72, GALAH DR4 14.79, Willett 19.5,
Xiang 20 Gyr) right-censored within 1 % of the cap.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from masked_stellar_autoencoder.training.age_consensus import (
    FLEX,
    IDENTITY,
    LINEAR,
    OFFSET,
    T0,
    AgeMaps,
    Groups,
    MapSet,
    Records,
    clip_feh,
    fit_consensus,
    leave_key_out,
    map_values,
    record_residuals,
    solve_stars,
)

CLASSES = ("MS", "MSTO", "SGB", "RGB", "RC", "giant")
GIANT = {"RGB", "RC", "giant"}
STARHORSE = tuple(f"starhorse23_{s}" for s in (
    "apogee17", "galahdr3", "gesdr5", "lamostm7", "lamostl7", "sdssdr12", "ravedr6", "rvs"))  # fmt: skip
WILLETT = ("willett26_kepler", "willett26_k2_bhm", "willett26_k2_cor", "willett26_tess",
           "willett26_kepler_L", "willett26_k2_bhm_L", "willett26_k2_cor_L", "willett26_tess_L",
           "willett26_k2_gal_L")  # fmt: skip
LEGACY = tuple(
    f"legacy17_{p}" for p in ("BASTA", "AST", "C2kSMO", "GOE", "V&A", "YMCM")
)
#: Catalogues on the same inputs: one record per star and group enters the fit
#: (the first in the list the star has); the others are validation records.
SHARED_INPUT_GROUPS = {
    "apokasc": ("apokasc3", "apokasc2", "apok2_kepler"),
    "apok2": ("apok2",),
    "willett26": WILLETT,
    "legacy17": LEGACY,
    "starhorse23": STARHORSE,
    "chaplin14": ("chaplin14",),
    "xiang24": ("xiang24",),
    "galah_dr4": ("galah_dr4",),
}
FIT_SOURCES = tuple(s for g in SHARED_INPUT_GROUPS.values() for s in g)
VALIDATION_SOURCES = ("leung23", "martig16")
#: Shared age error per star for the isochrone pipelines (common Gaia
#: photometry/parallax, overlapping spectra); learned through independent
#: constraints (co-natal partners, seismic dwarfs). No seismic shared effect:
#: the audit found APOKASC-3 vs Willett consistent with independent errors, and
#: a seismic eta is not separable from the GALAH giant noise (its only
#: independent overlap), where it absorbed the cross-regime disagreement.
EFFECT_FAMILIES = ("isochrone",)
ISOCHRONE = set(STARHORSE) | {"xiang24", "galah_dr4"}
#: Upper caps (Gyr); values within 1 % are right-censored.
CAPS = {**{s: 13.725 for s in STARHORSE}, "galah_dr4": 14.794, "xiang24": 20.0,
        **{s: 19.5 for s in WILLETT}, "martig16": 13.0}  # fmt: skip
RANK = {**{s: 6 for s in LEGACY}, "apokasc3": 5, "apokasc2": 5, "apok2_kepler": 5,
        "chaplin14": 5, **{s: 4 for s in WILLETT}, "apok2": 4, "xiang24": 3,
        **{s: 2 for s in STARHORSE}, "galah_dr4": 2}  # fmt: skip
#: Reference of a component not joined to the anchor by >= DENSE shared stars.
REFERENCE_PREFERENCE = ("xiang24", "apokasc3", "starhorse23_lamostl7", "galah_dr4_sgb")
DENSE, MIN_OVERLAP, LINEAR_MIN, FLEX_MIN, SPAN_MIN = 1000, 30, 1000, 20000, 0.6
CLASS_MIN, NOISE_MIN = 300, 200
MS_LOGG = (
    4.1  # isochrone-pipeline records at or above are main sequence (prior-dominated)
)
CLUSTER_SPREAD, BINARY_SPREAD, BINARY_PRIOR = 0.01, 0.01, 0.9
#: Literature ages (Gyr, eclipsing binaries / lithium depletion; orientation
#: only, design note 2.3) shown beside the fitted cluster latents.
LITERATURE_GYR = {"NGC_6791": 8.0, "NGC_6819": 2.5, "NGC_2682": 4.0, "NGC_188": 6.5,
                  "NGC_6811": 1.0, "NGC_2632": 0.7, "Melotte_22": 0.12}  # fmt: skip
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f2328", "#5b6470", "#e6e6e3"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
#: one colour per evolutionary class in every panel
CLASS_COLOUR = {"SGB": SERIES[0], "RGB": SERIES[0], "MSTO": SERIES[1], "RC": SERIES[1],
                "giant": SERIES[2], "MS": SERIES[3]}  # fmt: skip


def nmad(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def record_class(source: str, stage, stage_class) -> np.ndarray:
    stage = np.asarray(stage, dtype=object).astype(str)
    sc = np.asarray(stage_class, dtype=object).astype(str)
    n = len(stage)
    if source in ("apokasc3", "apokasc2") or source.startswith("willett26_kepler"):
        return np.where(np.isin(stage, ["RGB", "RC"]), stage, "giant")
    if source.startswith(("apok2", "willett26", "martig16", "leung23")):
        return np.full(n, "giant", dtype=object)
    if source.startswith("legacy17"):
        return np.full(n, "MS", dtype=object)
    if source == "xiang24":
        return np.full(n, "SGB", dtype=object)
    if source in STARHORSE:
        return np.where(stage == "SGB", "SGB", "MSTO")
    # chaplin14, galah_dr4: Teff/log g class of the audit
    return np.select([sc == "dwarf", sc == "subgiant", sc == "giant"], ["MS", "SGB", "giant"],
                     "unknown").astype(object)  # fmt: skip


def prepare(records: Path, targets: Path) -> dict:
    """Harmonised fit/validation records and co-natal groups."""
    cols = [
        "source",
        "source_id",
        "age",
        "lage",
        "sig",
        "stage",
        "stage_class",
        "good",
        "feh",
        "logg",
    ]
    raw = pd.read_parquet(records, columns=cols)
    raw = raw[raw.good & (raw.source_id > 0) & np.isfinite(raw.lage)].drop(
        columns="good"
    )
    raw["source"] = raw.source.astype(str)
    all_ids = np.unique(raw.source_id.to_numpy())
    use = raw[raw.source.isin(FIT_SOURCES + VALIDATION_SOURCES)].copy()
    excluded = {}
    gd = use.source.isin(ISOCHRONE) & (use.logg >= MS_LOGG)
    for s, n in use[gd].source.value_counts().items():
        excluded[f"{s} main-sequence records (log g >= {MS_LOGG})"] = int(n)
    excluded["flame_dr3 + flame_spec_dr3 records"] = int(
        raw.source.str.startswith("flame").sum()
    )
    use = use[~gd]
    use["_s"] = use.sig.fillna(
        np.inf
    )  # one record per (source, star): smallest quoted error
    use = use.sort_values(["source", "source_id", "_s"], kind="stable")
    use = use.drop_duplicates(["source", "source_id"]).drop(columns="_s")
    use["cls"] = ""
    for s, idx in use.groupby("source").groups.items():
        g = use.loc[idx]
        use.loc[idx, "cls"] = record_class(s, g.stage, g.stage_class)
    bad = use.cls == "unknown"
    excluded["records without an evolutionary class"] = int(bad.sum())
    use = use[~bad]
    cap = use.source.map(CAPS).astype(float)
    use["censor"] = np.where(use.age >= 0.99 * cap, 1, 0).astype("i1")
    med = use.groupby("source_id").feh.median()  # [Fe/H]: own, else the star's median
    use["feh"] = use.feh.fillna(use.source_id.map(med))
    use["group"], use["prio"] = "", 0
    for gname, members in SHARED_INPUT_GROUPS.items():
        for p, s in enumerate(members):
            m = use.source == s
            use.loc[m, "group"] = gname
            use.loc[m, "prio"] = p
    fit = use[use.source.isin(FIT_SOURCES)].sort_values(["source_id", "group", "prio"])
    fit = fit.assign(
        representative=(~fit.duplicated(["source_id", "group"])).to_numpy()
    )
    z = np.load(targets)
    mem = pd.DataFrame({"source_id": z["member"], "cid": z["member_cluster"].astype("i8"),
                        "prob": z["member_prob"].astype(float)})  # fmt: skip
    mem = mem.sort_values("prob", ascending=False).drop_duplicates("source_id")
    pairs = pd.DataFrame({"s1": z["pair1"], "s2": z["pair2"]})
    return {"fit": fit.reset_index(drop=True),
            "val": use[use.source.isin(VALIDATION_SOURCES)].reset_index(drop=True),
            "all_ids": all_ids, "members": mem, "pairs": pairs, "excluded": excluded,
            "n_raw": int(len(raw))}  # fmt: skip


def conatal_groups(star_ids, members, pairs, use_groups=True):
    """Groups over the sorted ``star_ids``: clusters first, then binaries."""
    n = len(star_ids)
    gid, prior = np.full(n, -1, "i8"), np.zeros(n)
    spread, labels = [], []
    if not use_groups:
        return Groups(gid, prior, np.zeros(0)), labels
    sid = members.source_id.to_numpy()
    pos = np.minimum(np.searchsorted(star_ids, sid), n - 1)
    ok = star_ids[pos] == sid
    uc, inv = np.unique(members.cid.to_numpy()[ok], return_inverse=True)
    gid[pos[ok]] = inv
    prior[pos[ok]] = members.prob.to_numpy()[ok]
    spread += [CLUSTER_SPREAD] * len(uc)
    labels += [("cluster", int(c)) for c in uc]
    s1, s2 = pairs.s1.to_numpy(), pairs.s2.to_numpy()
    p1 = np.minimum(np.searchsorted(star_ids, s1), n - 1)
    p2 = np.minimum(np.searchsorted(star_ids, s2), n - 1)
    ok = (star_ids[p1] == s1) & (star_ids[p2] == s2)
    k = len(spread)
    for j in np.flatnonzero(ok):
        a, b = p1[j], p2[j]
        if gid[a] >= 0 or gid[b] >= 0:
            continue
        gid[a] = gid[b] = k
        prior[a] = prior[b] = BINARY_PRIOR
        spread.append(BINARY_SPREAD)
        labels.append(("binary", int(j)))
        k += 1
    return Groups(gid, prior, np.asarray(spread, float)), labels


# --------------------------------------------------------------------------- #
# Configuration -> maps, noise units, fit set
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    name: str
    anchor: str = "apokasc3"
    galah_continuity: bool = False  # one GALAH DR4 map for its giants and subgiants
    bridge_tier: int = OFFSET  # max tier of a component reference (thin bridges)
    mission_continuity: bool = True  # one Willett map per variant (Kepler, K2, TESS)
    conatal: bool = True
    identity_extra: tuple = ()  # further identity-mapped catalogues
    drop: tuple = ()  # held-out catalogues


def map_key(source: str, cls: str, cfg: Config) -> str:
    if cfg.mission_continuity and source.startswith("willett26"):
        return "willett26_L" if source.endswith("_L") else "willett26_dnu"
    if source == "galah_dr4" and not cfg.galah_continuity:
        return "galah_dr4_giant" if cls in GIANT else "galah_dr4_sgb"
    return source


def build_problem(data: dict, cfg: Config, log=print) -> dict:
    fit = data["fit"]
    fit = fit[~fit.source.isin(cfg.drop)].sort_values(["source_id", "group", "prio"])
    fit = fit[~fit.duplicated(["source_id", "group"])].copy()  # representatives
    fit["key"] = [map_key(s, c, cfg) for s, c in zip(fit.source, fit.cls)]
    star_ids = np.unique(fit.source_id.to_numpy())
    star = np.searchsorted(star_ids, fit.source_id.to_numpy())
    groups, _ = conatal_groups(star_ids, data["members"], data["pairs"], cfg.conatal)
    gid = groups.gid
    n_grp = fit.groupby("source_id").group.nunique().reindex(star_ids).to_numpy()
    gcount = np.bincount(gid[gid >= 0], minlength=max(len(groups.spread), 1))
    partner = (gid >= 0) & (gcount[np.maximum(gid, 0)] >= 2)
    fit["constrained"] = ((n_grp >= 2) | partner)[star]
    keys = sorted(fit.key.unique(), key=lambda k: (k != cfg.anchor, k))
    kidx = {k: i for i, k in enumerate(keys)}
    M, C = len(keys), len(CLASSES)
    tier = np.zeros(M, "i8")
    class_terms = np.zeros((M, C), bool)
    use_feh = np.zeros(M, bool)
    info = []
    for k in keys:
        g = fit[(fit.key == k) & fit.constrained]
        n = int(g.source_id.nunique())
        span = float(np.subtract(*np.percentile(g.lage, [95, 5]))) if n else 0.0
        fok = np.isfinite(g.feh)
        fspan = (
            float(np.subtract(*np.percentile(g.feh[fok], [95, 5])))
            if fok.sum() > 10
            else 0.0
        )
        i = kidx[k]
        if k == cfg.anchor or k in cfg.identity_extra:
            t = IDENTITY
        elif n < MIN_OVERLAP:
            t = -1
        elif n < LINEAR_MIN or span < SPAN_MIN:
            t = OFFSET
        else:
            t = LINEAR if n < FLEX_MIN else FLEX
        tier[i] = t
        use_feh[i] = bool(n) and fok.mean() > 0.5 and fspan >= 0.4
        cont = k in ("willett26_dnu", "willett26_L") or (k == "galah_dr4")
        if t >= LINEAR and not cont:
            cc = g.cls.value_counts()
            big = [c for c in cc.index if cc[c] >= CLASS_MIN]
            for c in big[1:]:  # the most populous class is the reference
                class_terms[i, CLASSES.index(c)] = True
        info.append({"key": k, "n_constrained": n, "age_span_dex": round(span, 3),
                     "feh_span_dex": round(fspan, 3), "tier_by_count": int(t)})  # fmt: skip
    # components joined by >= DENSE shared stars; references for anchor-less ones
    kk = fit.key.map(kidx).to_numpy()
    pc = _pair_counts(star, kk)
    comp = _components(M, [(a, b) for (a, b), n in pc.items() if n >= DENSE])
    sc = pd.DataFrame({"s": star, "c": comp[kk]}).drop_duplicates()
    for c in np.unique(comp):
        if c == comp[kidx[cfg.anchor]]:
            continue
        mk = [k for k in keys if comp[kidx[k]] == c and tier[kidx[k]] >= 0]
        if not mk:
            continue
        ref = next(
            (k for k in REFERENCE_PREFERENCE if k in mk),
            max(mk, key=lambda k: (fit.key == k).sum()),
        )
        width = _bridge_width(sc, gid, c)
        i = kidx[ref]
        if width < MIN_OVERLAP:
            for k in mk:
                tier[kidx[k]] = -1
            log(f"[{cfg.name}] component {mk} not connected (width {width}); dropped")
            continue
        if tier[i] != IDENTITY:
            tier[i] = OFFSET if width < CLASS_MIN else min(tier[i], cfg.bridge_tier)
            class_terms[i] = False
            if tier[i] == LINEAR:
                use_feh[i] = False
        for row in info:
            if row["key"] == ref:
                row["component_reference"] = True
                row["bridge_width"] = int(width)
    for row in info:
        i = kidx[row["key"]]
        row.update(tier=int(tier[i]), use_feh=bool(use_feh[i]), component=int(comp[i]),
                   class_terms=",".join(CLASSES[j] for j in np.flatnonzero(class_terms[i])))  # fmt: skip
    keep = [k for k in keys if tier[kidx[k]] >= 0]
    fit = fit[fit.key.isin(keep)].copy()
    sel = np.array([kidx[k] for k in keep])
    # data ranges per map (constrained records): no extrapolation of the [Fe/H]
    # and polynomial terms
    xr, tr = np.tile([-0.5, 0.3], (len(keep), 1)), np.tile([0.0, 1.0], (len(keep), 1))
    for i, k in enumerate(keep):
        g = fit[(fit.key == k) & fit.constrained]
        fe = g.feh[np.isfinite(g.feh)]
        if len(fe) > 10:
            xr[i] = np.percentile(fe, [1, 99])
        if len(g) > 10:
            tr[i] = np.percentile(g.lage, [1, 99])
    maps = MapSet(
        [
            AgeMaps(
                keep, tier[sel], class_terms[sel], use_feh[sel], x_range=xr, t_range=tr
            )
        ]
    )
    cnt = fit.groupby(["source", "cls"]).size()
    fit["unit"] = np.where([cnt[(s, c)] >= NOISE_MIN for s, c in zip(fit.source, fit.cls)],
                           fit.source + ":" + fit.cls, fit.source + ":all")  # fmt: skip
    units = sorted(fit.unit.unique())
    uidx = {u: i for i, u in enumerate(units)}
    star_ids = np.unique(fit.source_id.to_numpy())
    star = np.searchsorted(star_ids, fit.source_id.to_numpy())
    groups, glabels = conatal_groups(
        star_ids, data["members"], data["pairs"], cfg.conatal
    )
    gid = groups.gid
    nrec = np.bincount(star, minlength=len(star_ids))
    grec = np.bincount(gid[gid >= 0], minlength=max(len(groups.spread), 1))
    in_fit = (nrec >= 2) | ((gid >= 0) & (grec[np.maximum(gid, 0)] >= 2))
    effect = np.where(fit.source.isin(ISOCHRONE), 0, -1).astype("i8")
    kmap = {k: i for i, k in enumerate(keep)}
    rec = Records(
        star.astype("i8"), fit.key.map(kmap).to_numpy("i8"),
        np.array([CLASSES.index(c) for c in fit.cls], "i8"), fit.unit.map(uidx).to_numpy("i8"),
        fit.lage.to_numpy(float), fit.sig.to_numpy(float), fit.feh.to_numpy(float),
        fit.censor.to_numpy("i1"), len(star_ids), effect=effect,
    )  # fmt: skip
    return {"cfg": cfg, "fit_df": fit.reset_index(drop=True), "maps": maps, "units": units,
            "rec": rec, "star_ids": star_ids, "groups": groups, "group_labels": glabels,
            "in_fit": in_fit, "tiers": pd.DataFrame(info)}  # fmt: skip


def _pair_counts(star, kk):
    df = pd.DataFrame({"s": star, "k": kk}).drop_duplicates()
    m = df.merge(df, on="s")
    return m[m.k_x < m.k_y].groupby(["k_x", "k_y"]).size().to_dict()


def _components(n, edges):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        parent[find(a)] = find(b)
    return np.unique([find(i) for i in range(n)], return_inverse=True)[1]


def _bridge_width(sc: pd.DataFrame, gid: np.ndarray, c: int) -> int:
    """Stars of component ``c`` tied to another component by a shared star or a
    co-natal partner."""
    in_c = set(sc.s[sc.c == c])
    other = set(sc.s[sc.c != c])
    shared = {gid[s] for s in in_c if gid[s] >= 0} & {
        gid[s] for s in other if gid[s] >= 0
    }
    return len((in_c & other) | {s for s in in_c if gid[s] in shared})


# --------------------------------------------------------------------------- #
# Fitting helpers
# --------------------------------------------------------------------------- #
def fit_problem(prob, warm=None, covariance=True, log=print, **kw):
    rec, in_fit = prob["rec"], prob["in_fit"]
    fit_stars = np.flatnonzero(in_fit)
    remap = np.full(len(in_fit), -1, "i8")
    remap[fit_stars] = np.arange(len(fit_stars))
    r = rec.subset(in_fit[rec.star])
    r = replace(r, star=remap[r.star], n_star=len(fit_stars))
    g = prob["groups"]
    gsub = Groups(g.gid[fit_stars], g.prior[fit_stars], g.spread)
    units = prob["units"]
    lk0 = np.where([u.split(":")[0] in ISOCHRONE for u in units], np.log(2.0), 0.0)
    lw0 = np.where(
        [u.startswith("galah_dr4") for u in units], np.log(0.1), np.log(0.03)
    )
    le0 = np.full(len(EFFECT_FAMILIES), np.log(0.03))
    coefs0 = None
    maps = prob["maps"]
    if warm is not None:
        fam = maps.families[0]
        c0 = fam.init()
        for i, k in enumerate(fam.names):
            if k in warm["keys"]:
                c0[i] = warm["coef"][warm["keys"].index(k)] * fam.mask()[i]
        coefs0 = [c0]
        for i, u in enumerate(units):
            if u in warm["units"]:
                j = warm["units"].index(u)
                lk0[i], lw0[i] = warm["lk"][j], warm["lw"][j]
        le0 = warm["le"]
    t0 = time.time()
    res = fit_consensus(r, gsub, maps, len(units), len(EFFECT_FAMILIES), coefs0=coefs0, lk0=lk0,
                        lw0=lw0, le0=le0, covariance=covariance, log=log, **kw)  # fmt: skip
    rho = np.zeros(len(in_fit))
    rho[fit_stars] = res.rho
    sec = time.time() - t0
    log(f"[{prob['cfg'].name}] {r.n_star} stars, {len(r.y)} records, {len(res.history)} outer "
        f"iterations (last max|dtheta| {res.history[-1]['dtheta']:.1e}), {sec:.0f} s, "
        f"eta {np.round(np.exp(res.log_eta), 3).tolist()}")  # fmt: skip
    return {"res": res, "rho": rho, "n_fit_stars": int(r.n_star), "n_fit_rec": int(len(r.y)),
            "seconds": sec}  # fmt: skip


def warm_from(prob, out):
    res = out["res"]
    return {"keys": list(prob["maps"].names), "coef": res.coefs[0], "units": list(prob["units"]),
            "lk": res.log_kappa, "lw": res.log_omega, "le": res.log_eta}  # fmt: skip


def solve_all(prob, out):
    return solve_stars(prob["rec"], prob["groups"], out["res"], len(prob["units"]),
                       len(EFFECT_FAMILIES), rho=out["rho"])  # fmt: skip


def map_table(prob, out) -> pd.DataFrame:
    return map_table_from_res(out["res"])


def map_table_from_res(res) -> pd.DataFrame:
    """Coefficients and standard errors per map. Standard errors come from the
    finite-difference Hessian of the profiled objective at fixed noise and
    responsibilities; a non-positive variance (non-smooth or flat direction,
    e.g. clamped polynomial terms) is reported as NaN, not as zero."""
    fam = res.maps.families[0]
    mask = fam.mask()
    se = np.full(mask.shape, np.nan)
    if res.theta_cov is not None:
        d = np.diag(res.theta_cov)
        se[mask] = np.where(d > 0, np.sqrt(np.abs(d)), np.nan)
    rows = []
    for i, k in enumerate(fam.names):
        row = {
            "key": k,
            "tier": ["identity", "offset", "linear", "flex"][int(fam.tier[i])],
        }
        for j, nm in enumerate(fam.column_names()):
            if mask[i, j]:
                if nm.startswith(("db", "dA")):
                    nm = f"{nm[:2]}_{CLASSES[int(nm[2:])]}"
                row[nm], row[nm + "_se"] = res.coefs[0][i, j], se[i, j]
        rows.append(row)
    return pd.DataFrame(rows)


def noise_table(prob, out) -> pd.DataFrame:
    res, df = out["res"], prob["fit_df"]
    U = len(prob["units"])
    vc = res.var_cov
    rows = []
    for i, u in enumerate(prob["units"]):
        m = df.unit == u
        sig = df.sig[m]
        q = float(np.nanmedian(sig)) if np.isfinite(sig).any() else np.nan
        k, w = float(np.exp(res.log_kappa[i])), float(np.exp(res.log_omega[i]))
        kse = (
            k * np.sqrt(vc[i, i])
            if vc is not None and np.isfinite(vc[i, i])
            else np.nan
        )
        wse = w * np.sqrt(vc[U + i, U + i]) if vc is not None else np.nan
        scale = float(np.hypot(k * (q if np.isfinite(q) else 0.0), w))
        rows.append({"unit": u, "n_records": int(m.sum()), "median_quoted_sig": q,
                     "kappa": k if np.isfinite(q) else np.nan, "kappa_se": kse if np.isfinite(q) else np.nan,
                     "omega": w, "omega_se": wse, "scale_at_median_sig": scale,
                     "effective_inflation": scale / q if np.isfinite(q) else np.nan})  # fmt: skip
    for e, name in enumerate(EFFECT_FAMILIES):
        eta = float(np.exp(res.log_eta[e]))
        ese = eta * np.sqrt(vc[2 * U + e, 2 * U + e]) if vc is not None else np.nan
        rows.append({"unit": f"shared effect eta: {name}", "omega": eta, "omega_se": ese,
                     "scale_at_median_sig": eta})  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Held-out prediction (residuals against the fitted consensus of other sources)
# --------------------------------------------------------------------------- #
def predict_heldout(name, recs: pd.DataFrame, star_ids, sol, tier_hint=LINEAR, seed=0,
                    use_feh=True) -> list[dict]:  # fmt: skip
    """Fit the catalogue's map on half its stars against the consensus of the
    other sources (which enters with its own uncertainty, as a pseudo-record),
    then report residuals and z-scores on the other half."""
    sid = recs.source_id.to_numpy()
    pos = np.minimum(np.searchsorted(star_ids, sid), len(star_ids) - 1)
    hit = star_ids[pos] == sid
    t = np.where(hit, sol["t"][pos], np.nan)
    sd = np.where(hit, sol["sd"][pos], np.nan)
    ok = np.isfinite(t) & np.isfinite(sd) & (sd > 0) & (recs.censor.to_numpy() == 0)
    r, t, sd = recs[ok], t[ok], sd[ok]
    if len(r) < 40:
        return [{"catalogue": name, "class": "all", "n_test": int(len(r) // 2)}]
    half = np.random.default_rng(seed).random(len(r)) < 0.5
    n = len(r)
    cls_ids = np.array([CLASSES.index(c) for c in r.cls])
    big = [
        c for c in sorted(set(cls_ids[half])) if (cls_ids[half] == c).sum() >= CLASS_MIN
    ]
    ct = np.zeros((2, len(CLASSES)), bool)
    if tier_hint >= LINEAR and len(big) > 1:
        ref = max(big, key=lambda c: (cls_ids[half] == c).sum())
        ct[1, [c for c in big if c != ref]] = True
    tier = OFFSET if half.sum() < LINEAR_MIN // 2 else min(tier_hint, LINEAR)
    fehok = use_feh and np.isfinite(r.feh.to_numpy()).mean() > 0.5 and tier >= LINEAR
    maps = MapSet([AgeMaps(["consensus", name], [IDENTITY, tier], ct, [False, fehok])])
    U = 1 + len(CLASSES)
    star = np.r_[np.arange(n), np.arange(n)]
    rec = Records(star.astype("i8"), np.r_[np.zeros(n, "i8"), np.ones(n, "i8")],
                  np.r_[cls_ids, cls_ids].astype("i8"), np.r_[np.zeros(n, "i8"), 1 + cls_ids],
                  np.r_[t, r.lage.to_numpy(float)], np.r_[sd, r.sig.to_numpy(float)],
                  np.r_[r.feh.to_numpy(float), r.feh.to_numpy(float)], np.zeros(2 * n, "i1"), n)  # fmt: skip
    tr = np.flatnonzero(half)
    remap = np.full(n, -1, "i8")
    remap[tr] = np.arange(len(tr))
    rt = rec.subset(np.r_[half, half])
    rt = replace(rt, star=remap[rt.star], n_star=len(tr))
    lk0, lw0 = np.zeros(U), np.full(U, np.log(0.05))
    lw0[0] = np.log(1e-4)
    res = fit_consensus(rt, Groups.none(rt.n_star), maps, U, n_outer=25, covariance=False,
                        fix_noise=np.arange(U) == 0, lk0=lk0, lw0=lw0)  # fmt: skip
    test = ~half
    y = r.lage.to_numpy(float)[test]
    f, a = res.maps.families[0].evaluate(
        torch.as_tensor(res.coefs[0]), torch.ones(test.sum(), dtype=torch.long),
        torch.as_tensor(t[test]), torch.as_tensor(clip_feh(r.feh.to_numpy(float)[test])),
        torch.zeros((test.sum(), 0), dtype=torch.float64), torch.as_tensor(cls_ids[test]))  # fmt: skip
    f, a = f.numpy(), a.numpy()
    sig = np.nan_to_num(r.sig.to_numpy(float)[test])
    u = 1 + cls_ids[test]
    v = np.exp(2 * res.log_kappa[u]) * sig**2 + np.exp(2 * res.log_omega[u])
    raw, mapped = y - t[test], y - f
    z = mapped / np.sqrt(v + a * a * sd[test] ** 2)
    out = []
    cl = r.cls.to_numpy()[test]
    for c in ["all", *sorted(set(cl))]:
        m = np.ones(len(cl), bool) if c == "all" else cl == c
        if m.sum() < 10:
            continue
        out.append({"catalogue": name, "class": c, "n_test": int(m.sum()), "map_tier": int(tier),
                    "bias_raw": float(np.median(raw[m])), "nmad_raw": nmad(raw[m]),
                    "bias_mapped": float(np.median(mapped[m])), "nmad_mapped": nmad(mapped[m]),
                    "z_median": float(np.median(z[m])), "z_nmad": nmad(z[m]),
                    "consensus_sd_median": float(np.median(sd[test][m]))})  # fmt: skip
    return out


LOO_FAMILIES = {
    "apokasc2+apok2_kepler": ("apokasc2", "apok2_kepler"),
    "willett26": WILLETT,
    "apok2": ("apok2",),
    "legacy17": LEGACY,
    "chaplin14": ("chaplin14",),
    "xiang24": ("xiang24",),
    "galah_dr4": ("galah_dr4",),
    "starhorse23_lamostl7": ("starhorse23_lamostl7",),
    "starhorse23_rvs": ("starhorse23_rvs",),
    "starhorse23_apogee17": ("starhorse23_apogee17",),
    "starhorse23_galahdr3": ("starhorse23_galahdr3",),
    "starhorse23_lamostm7": ("starhorse23_lamostm7",),
    "starhorse23_ges+sdss+rave": ("starhorse23_gesdr5", "starhorse23_sdssdr12",
                                  "starhorse23_ravedr6"),
}  # fmt: skip


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(args) -> None:
    torch.set_num_threads(args.threads)
    work, docs = Path(args.work), Path(args.docs)
    work.mkdir(parents=True, exist_ok=True)
    docs.mkdir(parents=True, exist_ok=True)
    logf = open(work / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t_start = time.time()
    data = prepare(Path(args.records), Path(args.targets))
    log(f"records: {data['n_raw']} good with a DR3 ID; fit-source records {len(data['fit'])} "
        f"(representatives {int(data['fit'].representative.sum())}); validation {len(data['val'])}; "
        f"excluded {data['excluded']}")  # fmt: skip
    summary = {"excluded": data["excluded"], "n_records_good": data["n_raw"],
               "n_stars_all": int(len(data["all_ids"]))}  # fmt: skip
    kw = {"n_outer": args.n_outer, "verbose": args.verbose}

    def do(cfg, warm=None, covariance=True):
        prob = build_problem(data, cfg, log=log)
        return prob, fit_problem(prob, warm=warm, covariance=covariance, log=log, **kw)

    def record_fit(tag, prob, out):
        summary[f"fit_{tag}"] = {"stars": out["n_fit_stars"], "records": out["n_fit_rec"],
                                 "outer_iterations": len(out["res"].history),
                                 "last_max_dtheta": out["res"].history[-1]["dtheta"],
                                 "seconds": round(out["seconds"], 1)}  # fmt: skip

    if args.resume:  # reuse the saved main fits (fit_A.pkl, fit_C.pkl)
        results, sols = {}, {}
        for tag, cfg in (
            ("A", Config("A_apokasc3")),
            ("C", Config("C_xiang24", anchor="xiang24")),
        ):
            saved = pickle.load(open(work / f"fit_{tag}.pkl", "rb"))
            prob = build_problem(data, cfg, log=log)
            out = {"res": saved["res"], "rho": saved["rho"], "n_fit_stars": int(prob["in_fit"].sum()),
                   "n_fit_rec": int(prob["in_fit"][prob["rec"].star].sum()), "seconds": float("nan")}  # fmt: skip
            results[tag], sols[tag] = (prob, out), saved["sol"]
            record_fit(tag, prob, out)
        probA, outA = results["A"]
    else:
        probA, outA = do(Config("A_apokasc3"))
        probC, outC = do(Config("C_xiang24", anchor="xiang24"), warm=None)
        results = {"A": (probA, outA), "C": (probC, outC)}
        sols = {"A": solve_all(probA, outA), "C": solve_all(probC, outC)}
    for tag, (prob, out) in results.items():
        if args.resume:
            break
        map_table(prob, out).to_csv(docs / f"maps_{tag}.csv", index=False)
        noise_table(prob, out).to_csv(docs / f"noise_{tag}.csv", index=False)
        prob["tiers"].to_csv(docs / f"tiers_{tag}.csv", index=False)
        record_fit(tag, prob, out)
        with open(work / f"fit_{tag}.pkl", "wb") as fh:
            pickle.dump({"res": out["res"], "units": prob["units"], "tiers": prob["tiers"],
                         "rho": out["rho"], "star_ids": prob["star_ids"],
                         "group_labels": prob["group_labels"], "gid": prob["groups"].gid,
                         "prior": prob["groups"].prior, "spread": prob["groups"].spread,
                         "sol": sols[tag]}, fh)  # fmt: skip
    # ---- bridge variants (gauge A) -------------------------------------------
    variants = {
        "V_affine_bridge": Config("V_affine_bridge", bridge_tier=LINEAR),
        "V_galah_continuity": Config("V_galah_continuity", galah_continuity=True),
        "V_conatal_only": Config("V_conatal_only", mission_continuity=False),
        "V_continuity_only": Config(
            "V_continuity_only", galah_continuity=True, conatal=False
        ),
        "V_seismic_dwarfs": Config("V_seismic_dwarfs", identity_extra=("chaplin14",)),
    }
    if args.quick:
        variants = {}
    bridge = []
    grid_t = np.array([0.0, 0.3, 0.6, 0.9, 1.1])
    var_results = {"A_main": (probA, outA)}
    csv_only = {}
    for tag, cfg in variants.items():
        if args.resume and (docs / f"maps_{tag}.csv").exists():
            csv_only[tag] = pd.read_csv(
                docs / f"maps_{tag}.csv"
            )  # fitted before the resume
            continue
        # without co-natal terms the continuity-only problem converges slowly: capped
        extra = {"n_outer": 12} if not cfg.conatal else {}
        prob = build_problem(data, cfg, log=log)
        out = fit_problem(prob, warm=warm_from(probA, outA), covariance=cfg.conatal, log=log,
                          **{**kw, **extra})  # fmt: skip
        map_table(prob, out).to_csv(docs / f"maps_{tag}.csv", index=False)
        record_fit(tag, prob, out)
        var_results[tag] = (prob, out)
    for tag, mt in csv_only.items():  # bridge values from the saved coefficients
        row = mt.set_index("key").loc["xiang24"]
        for feh in (-0.5, 0.0, 0.3):
            u, x = grid_t - 0.6, feh
            d = (
                row.get("b", 0.0)
                + np.nan_to_num(row.get("A", 0.0)) * u
                + np.nan_to_num(row.get("beta", 0.0)) * x
            )
            for g, term in (
                ("g_uu", u * u),
                ("g_uuu", u**3),
                ("g_xu", x * u),
                ("g_xx", x * x + 0 * u),
            ):
                d = d + np.nan_to_num(row.get(g, 0.0)) * term
            se = row.get("b_se", np.nan) if row.tier == "offset" else np.nan
            for tt, dd in zip(grid_t, d):
                bridge.append({"variant": tag, "xiang_tier": ["identity", "offset", "linear", "flex"].index(row.tier),
                               "feh": feh, "t": tt, "age_gyr": round(10**tt, 2),
                               "xiang_minus_latent_dex": dd, "se": se})  # fmt: skip
    for tag, (prob, out) in var_results.items():
        if "xiang24" not in prob["maps"].names:
            continue
        tr = prob["tiers"].set_index("key")
        for feh in (-0.5, 0.0, 0.3):
            f, _, se = map_values(
                out["res"], "xiang24", grid_t, feh=feh, cls=CLASSES.index("SGB")
            )
            for tt, dd, ss in zip(grid_t, f - grid_t, se):
                bridge.append({"variant": tag, "xiang_tier": int(tr.tier["xiang24"]),
                               "bridge_width": tr.get("bridge_width", pd.Series()).get("xiang24", np.nan),
                               "feh": feh, "t": tt, "age_gyr": round(10**tt, 2),
                               "xiang_minus_latent_dex": dd, "se": ss})  # fmt: skip
    pd.DataFrame(bridge).to_csv(docs / "bridge_variants.csv", index=False)
    # ---- per-star consensus (both gauges) --------------------------------------
    stars = pd.DataFrame({"source_id": data["all_ids"]})
    for tag in ("A", "C"):
        prob = results[tag][0]
        sol = sols[tag]
        sid = stars.source_id.to_numpy()
        pos = np.minimum(
            np.searchsorted(prob["star_ids"], sid), len(prob["star_ids"]) - 1
        )
        hit = prob["star_ids"][pos] == sid
        for col in ("t", "sd", "t_rec", "sd_rec"):
            stars[f"{col}_{tag}"] = np.where(hit, sol[col][pos], np.nan).astype("f4")
        stars[f"limit_{tag}"] = np.where(hit, sol["limit"][pos], 0).astype("i1")
    fdf = probA["fit_df"]
    agg = fdf.groupby("source_id").agg(n_catalogues=("source", "size"),
                                       catalogues=("source", lambda s: ";".join(sorted(s))),
                                       n_families=("group", "nunique"))  # fmt: skip
    stage = (fdf.assign(r=fdf.source.map(RANK)).sort_values("r", ascending=False)
             .drop_duplicates("source_id").set_index("source_id").cls)  # fmt: skip
    stars = stars.join(agg, on="source_id").join(stage.rename("stage"), on="source_id")
    stars = stars.join(
        fdf.groupby("source_id").feh.median().rename("feh"), on="source_id"
    )
    stars["n_catalogues"] = stars.n_catalogues.fillna(0).astype("i2")
    stars["n_families"] = stars.n_families.fillna(0).astype("i2")
    sid = stars.source_id.to_numpy()
    pos = np.minimum(
        np.searchsorted(probA["star_ids"], sid), len(probA["star_ids"]) - 1
    )
    hit = probA["star_ids"][pos] == sid
    gidv = np.where(hit, probA["groups"].gid[pos], -1)
    lab = probA["group_labels"]
    stars["cluster_id"] = np.array(
        [lab[g][1] if g >= 0 and lab[g][0] == "cluster" else -1 for g in gidv], "i4"
    )
    stars["in_binary"] = np.array([g >= 0 and lab[g][0] == "binary" for g in gidv])
    stars["conatal_resp"] = np.where(hit, outA["rho"][pos], np.nan).astype("f4")
    stars["reason"] = np.where(stars.n_catalogues > 0, "",
                               "no record from a fitted catalogue (FLAME, Leung/Martig, "
                               "GALAH DR4 dwarf or uncalibrated)")  # fmt: skip
    stars["scale_diff_C_minus_A"] = (stars.t_C - stars.t_A).astype("f4")
    star_path = work / "consensus_stars.parquet"
    stars.to_parquet(star_path, index=False)
    summary["per_star_path"] = str(star_path)
    summary["n_stars_with_consensus_A"] = int(np.isfinite(stars.t_A).sum())
    summary["n_stars_with_consensus_C"] = int(np.isfinite(stars.t_C).sum())
    summary["n_lower_limits_A"] = int((stars.limit_A == 1).sum())
    summary["frac_consensus_A_above_13p8"] = float(
        np.mean(stars.t_A[np.isfinite(stars.t_A)] > np.log10(13.8))
    )
    # ---- record residuals before/after (consensus without the record's group) --
    for tag, (prob, out) in results.items():
        rec = prob["rec"]
        key = prob["fit_df"].group.astype("category").cat.codes.to_numpy()
        t_o, v_o = leave_key_out(rec, out["res"], len(prob["units"]), len(EFFECT_FAMILIES), key,
                                 sols[tag]["t"])  # fmt: skip
        raw, mapped, z = record_residuals(rec, out["res"], len(prob["units"]),
                                          len(EFFECT_FAMILIES), t_o, v_o)  # fmt: skip
        df = prob["fit_df"][
            ["source", "source_id", "cls", "key", "lage", "feh", "censor"]
        ].copy()
        df["t_other"], df["sd_other"] = t_o.astype("f4"), np.sqrt(v_o).astype("f4")
        df["raw"], df["mapped"], df["z"] = (
            raw.astype("f4"),
            mapped.astype("f4"),
            z.astype("f4"),
        )
        df[np.isfinite(df.t_other)].to_parquet(
            work / f"residuals_{tag}.parquet", index=False
        )
    # ---- validation-only and non-representative records vs the consensus ------
    val = pd.concat(
        [data["fit"][~data["fit"].representative], data["val"]], ignore_index=True
    )
    val_rows = []
    for s, g in val.groupby("source"):
        val_rows += predict_heldout(
            s,
            g,
            probA["star_ids"],
            {"t": sols["A"]["t_rec"], "sd": sols["A"]["sd_rec"]},
        )
    pd.DataFrame(val_rows).to_csv(docs / "validation_records.csv", index=False)
    # ---- leave-one-catalogue-out ---------------------------------------------
    loo = []
    warmA = warm_from(probA, outA)
    tiersA = probA["tiers"].set_index("key").tier
    for fname, srcs in ({} if args.quick else LOO_FAMILIES).items():
        prob, out = do(
            Config(f"LOO_{fname}", drop=tuple(srcs)), warm=warmA, covariance=False
        )
        sol = solve_all(prob, out)
        held = data["fit"][data["fit"].source.isin(srcs)]
        for s, g in held.groupby("source"):
            th = int(tiersA.get(map_key(s, "SGB", Config("A")), LINEAR))
            for row in predict_heldout(
                s, g, prob["star_ids"], sol, tier_hint=max(th, OFFSET)
            ):
                loo.append({"family_held_out": fname, **row})
        summary[f"fit_LOO_{fname}"] = {"stars": out["n_fit_stars"], "seconds": round(out["seconds"], 1),
                                       "outer_iterations": len(out["res"].history)}  # fmt: skip
        pd.DataFrame(loo).to_csv(docs / "loo_predictive.csv", index=False)
    summary["runtime_seconds"] = round(time.time() - t_start, 1)
    (docs / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"done in {time.time() - t_start:.0f} s")


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
def _style():
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 8, "axes.titlesize": 8.5, "legend.frameon": False, "lines.linewidth": 1.6,
        "axes.axisbelow": True,
    })  # fmt: skip


def _binned(x, y, edges, min_n=20):
    idx = np.digitize(x, edges) - 1
    out = []
    for k in range(len(edges) - 1):
        m = (idx == k) & np.isfinite(y)
        if m.sum() >= min_n:
            out.append((0.5 * (edges[k] + edges[k + 1]), np.median(y[m]),
                        np.percentile(y[m], 16), np.percentile(y[m], 84), m.sum()))  # fmt: skip
    return np.array(out).reshape(-1, 5)


def make_plots(work: Path, docs: Path, clusters: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style()
    fitA = pickle.load(open(work / "fit_A.pkl", "rb"))
    resA = fitA["res"]
    rA = pd.read_parquet(work / "residuals_A.parquet")
    stars = pd.read_parquet(work / "consensus_stars.parquet",
                            columns=["source_id", "t_A", "sd_A", "t_rec_A", "sd_rec_A", "t_C",
                                     "stage", "feh", "cluster_id", "conatal_resp", "n_catalogues"])  # fmt: skip
    # (1) maps: catalogue log age vs consensus (other groups), map curves per class
    keys = [k for k in resA.maps.names if (rA.key == k).sum() >= 30]
    ncol = 5
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(13, 2.5 * nrow), squeeze=False)
    tg = np.linspace(-1.0, 1.4, 120)
    for ax, k in zip(axs.flat, keys):
        g = rA[rA.key == k]
        ax.hexbin(g.t_other, g.lage, gridsize=40, extent=(-1, 1.4, -1, 1.4), bins="log",
                  cmap="Greys", mincnt=1, linewidths=0)  # fmt: skip
        ax.plot([-1, 1.4], [-1, 1.4], color=MUTED, lw=0.8, ls=":")
        for c, gc in g.groupby("cls"):
            if len(gc) < 30:
                continue
            lo, hi = np.percentile(gc.t_other, [1, 99])
            tt = tg[(tg >= lo) & (tg <= hi)]
            f, _, se = map_values(
                resA, k, tt, feh=float(np.nanmedian(gc.feh)), cls=CLASSES.index(c)
            )
            col = CLASS_COLOUR[c]
            ax.plot(tt, f, color=col, label=f"{c} (n={len(gc)})")
            if np.isfinite(se).any():
                ax.fill_between(tt, f - se, f + se, color=col, alpha=0.25, lw=0)
        ax.set_title(k, loc="left")
        ax.set_xlim(-1, 1.4)
        ax.set_ylim(-1, 1.4)
        ax.legend(loc="upper left", fontsize=6.5)
    for ax in axs[-1]:
        ax.set_xlabel("consensus log10(age/Gyr) [other groups]")
    for ax in axs[:, 0]:
        ax.set_ylabel("catalogue log10(age/Gyr)")
    for ax in list(axs.flat)[len(keys) :]:
        ax.set_visible(False)
    fig.suptitle("C0 maps (gauge A, APOKASC-3): catalogue vs consensus of the other "
                 "input groups; lines = fitted map per class at the class median [Fe/H] (±1σ)",
                 x=0.01, ha="left", fontsize=9)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(docs / "c0_maps.png", dpi=90)
    plt.close(fig)
    # (2) residuals before/after vs age and [Fe/H]
    top = rA.groupby("key").size().sort_values(ascending=False)
    top = [k for k in top.index if top[k] >= 300][:10]
    fig, axs = plt.subplots(
        len(top), 2, figsize=(9, 1.6 * len(top)), squeeze=False, sharex="col"
    )
    ea, ef = np.arange(-0.6, 1.31, 0.1), np.arange(-1.5, 0.61, 0.15)
    for row, k in zip(axs, top):
        g = rA[rA.key == k]
        for ax, x, e in (
            (row[0], g.t_other.to_numpy(), ea),
            (row[1], g.feh.to_numpy(), ef),
        ):
            for j, (col, lab) in enumerate(
                (("raw", "before map: y − t"), ("mapped", "after map: y − f(t)"))
            ):
                b = _binned(x, g[col].to_numpy(), e)
                if len(b):
                    ax.plot(
                        b[:, 0], b[:, 1], color=SERIES[j], label=lab, marker="o", ms=2.5
                    )
                    ax.fill_between(
                        b[:, 0], b[:, 2], b[:, 3], color=SERIES[j], alpha=0.12, lw=0
                    )
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_ylim(-0.45, 0.45)
        row[0].set_ylabel(f"{k}\nresidual (dex)", fontsize=7)
    axs[0, 1].legend(loc="lower left", fontsize=6.5)
    axs[-1, 0].set_xlabel("consensus log10(age/Gyr) [other groups]")
    axs[-1, 1].set_xlabel("[Fe/H] (dex)")
    fig.suptitle("Residuals against the consensus of the other input groups (median, 16–84 %)",
                 x=0.01, ha="left", fontsize=9)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(docs / "c0_residuals.png", dpi=90)
    plt.close(fig)
    # (3) clusters: fitted cluster latents, member spread before/after, z-scores
    from astropy.table import Table

    cl = Table.read(clusters, hdu=1)
    cl = cl[["Name", "ID", "Type", "logAge50", "CMDCl50"]].to_pandas()
    cl["Name"] = cl.Name.astype(str).str.strip()
    cl = cl.set_index("ID")
    ctab = pd.read_csv(docs / "clusters_A.csv")
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    ax = axs[0]
    ok = ctab.n_members >= 3
    c = ctab[ok]
    ax.errorbar(c.hr23_logage - 9, c.T_fit, yerr=c.T_sd, fmt="o", ms=2.5, color=SERIES[0],
                elinewidth=0.6, alpha=0.7, label="fitted cluster latent (consensus, gauge A)")  # fmt: skip
    ax.scatter(c.hr23_logage - 9, c.raw_median, s=6, color=SERIES[1], alpha=0.6,
               label="before: median raw member age (all catalogues)")  # fmt: skip
    lit = c[c.name.isin(LITERATURE_GYR)]
    for _, rw in lit.iterrows():
        ax.annotate(rw["name"].replace("_", " "), (rw.hr23_logage - 9, rw.T_fit), fontsize=6,
                    color=INK, xytext=(3, 3), textcoords="offset points")  # fmt: skip
    ax.plot([-1.5, 1.2], [-1.5, 1.2], color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Hunt & Reffert 2023 log10(age/Gyr) (comparison, not truth)")
    ax.set_ylabel("cluster log10(age/Gyr)")
    ax.set_title(
        "Cluster ages: fitted latent vs HR23 (clusters with ≥ 3 members)", loc="left"
    )
    ax.legend(loc="lower right", fontsize=6.5)
    ax = axs[1]
    sp = ctab[(ctab.n_members >= 5)]
    bins = np.linspace(0, 0.5, 26)
    ax.hist(sp.raw_spread.dropna(), bins=bins, color=SERIES[1], alpha=0.6,
            label=f"before: raw catalogue values (median {np.nanmedian(sp.raw_spread):.3f})")  # fmt: skip
    ax.hist(sp.rec_spread.dropna(), bins=bins, color=SERIES[0], alpha=0.6,
            label=f"after: mapped records-only consensus, all members (median {np.nanmedian(sp.rec_spread):.3f})")  # fmt: skip
    ax.set_xlabel("within-cluster NMAD of member log age (dex)")
    ax.set_ylabel("clusters (≥ 5 members)")
    ax.set_title("Co-natal spread before / after the maps", loc="left")
    ax.legend(fontsize=6.5)
    ax = axs[2]
    zz = pd.read_parquet(work / "cluster_members_A.parquet")
    zb = np.linspace(-6, 6, 49)
    xx = np.linspace(-6, 6, 200)
    for j, (lab, m) in enumerate(
        (
            ("open clusters", zz.kind == "cluster"),
            ("wide binaries", zz.kind == "binary"),
        )
    ):
        v = zz.z[m & np.isfinite(zz.z)]
        if len(v) > 20:
            ax.hist(v, bins=zb, density=True, histtype="step", color=SERIES[j], lw=1.4,
                    label=f"{lab}: n={len(v)}, NMAD {nmad(v):.2f}")  # fmt: skip
    ax.plot(
        xx,
        np.exp(-0.5 * xx**2) / np.sqrt(2 * np.pi),
        color=MUTED,
        ls=":",
        label="N(0, 1)",
    )
    ax.set_xlabel("z = (t_i − T_group,−i) / σ   (members, responsibility > 0.5)")
    ax.set_ylabel("density")
    ax.set_title("Co-natal consistency of the consensus", loc="left")
    ax.legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(docs / "c0_clusters.png", dpi=90)
    plt.close(fig)
    # (4) scale difference C - A vs age and [Fe/H] per stage
    s = stars[
        np.isfinite(stars.t_A) & np.isfinite(stars.t_C) & (stars.n_catalogues > 0)
    ]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    stages = [
        c
        for c in ("SGB", "MSTO", "RGB", "RC", "giant", "MS")
        if (s.stage == c).sum() >= 50
    ]
    for j, c in enumerate(stages):
        g = s[s.stage == c]
        col = SERIES[j % 4]
        ls = "-" if j < 4 else "--"
        for ax, x, e in (
            (axs[0], g.t_A.to_numpy(), ea),
            (axs[1], g.feh.to_numpy(), ef),
        ):
            b = _binned(x, (g.t_C - g.t_A).to_numpy(), e, min_n=30)
            if len(b):
                ax.plot(
                    b[:, 0],
                    b[:, 1],
                    color=col,
                    ls=ls,
                    marker="o",
                    ms=2.5,
                    label=f"{c} (n={len(g)})",
                )
                ax.fill_between(b[:, 0], b[:, 2], b[:, 3], color=col, alpha=0.1, lw=0)
    for ax in axs:
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_ylabel("t(Xiang gauge) − t(APOKASC-3 gauge) (dex)")
    axs[0].set_xlabel("consensus log10(age/Gyr), APOKASC-3 gauge")
    axs[1].set_xlabel("[Fe/H] (dex)")
    axs[0].legend(fontsize=6.5, loc="center right")
    fig.suptitle("Scale systematic: Xiang 2024-anchored minus APOKASC-3-anchored consensus "
                 "(median, 16–84 %)", x=0.01, ha="left", fontsize=9)  # fmt: skip
    fig.tight_layout()
    fig.savefig(docs / "c0_scale_difference.png", dpi=90)
    plt.close(fig)
    # (5) leave-one-catalogue-out predictive accuracy
    loo = pd.read_csv(docs / "loo_predictive.csv")
    loo = loo[(loo["class"] == "all") & (loo.n_test >= 20)].copy()
    if len(loo):
        loo = loo.sort_values("nmad_mapped")
        fig, axs = plt.subplots(1, 3, figsize=(13, 0.32 * len(loo) + 1.4), sharey=True)
        yy = np.arange(len(loo))
        for ax, (a, b, lab) in zip(
            axs[:2],
            (
                ("bias_raw", "bias_mapped", "median residual (dex)"),
                ("nmad_raw", "nmad_mapped", "NMAD of residuals (dex)"),
            ),
        ):
            ax.scatter(loo[a], yy, color=SERIES[1], s=14, label="before map (y − t)")
            ax.scatter(loo[b], yy, color=SERIES[0], s=14, label="after map (y − f(t))")
            ax.set_xlabel(lab)
            ax.axvline(0, color=MUTED, lw=0.8)
        axs[2].scatter(loo.z_nmad, yy, color=SERIES[2], s=14, label="z-score NMAD")
        axs[2].axvline(1, color=MUTED, lw=0.8, ls=":")
        axs[2].set_xlabel("NMAD of z = (y − f(t)) / σ_pred")
        axs[0].set_yticks(yy)
        axs[0].set_yticklabels(
            [f"{r.catalogue} (n={r.n_test})" for r in loo.itertuples()], fontsize=6.5
        )
        axs[0].legend(fontsize=6.5, loc="lower right")
        axs[2].legend(fontsize=6.5)
        fig.suptitle("Leave-one-catalogue-out: held-out catalogue vs the refitted consensus "
                     "(map refit on half its stars, tested on the other half)", x=0.01, ha="left",
                     fontsize=9)  # fmt: skip
        fig.tight_layout()
        fig.savefig(docs / "c0_loo.png", dpi=90)
        plt.close(fig)


def cluster_tables(
    work: Path, docs: Path, records: Path, clusters: Path, targets: Path
) -> None:
    """Fitted cluster latents (gauge A), raw vs consensus member spreads, member z-scores."""
    from astropy.table import Table

    fitA = pickle.load(open(work / "fit_A.pkl", "rb"))
    sol, star_ids = fitA["sol"], fitA["star_ids"]
    gid, prior, spread, labels = (
        fitA["gid"],
        fitA["prior"],
        fitA["spread"],
        fitA["group_labels"],
    )
    rho = fitA["rho"]
    t, sd = sol["t_rec"], sol["sd_rec"]
    ok = (gid >= 0) & np.isfinite(t) & np.isfinite(sd) & (sd > 0)
    s2 = spread[np.maximum(gid, 0)] ** 2
    r = np.clip(rho, 1e-6, 1)
    om = np.where(ok, r / (r * sd**2 + s2), 0.0)
    G = len(spread)
    W0 = np.bincount(gid[ok], om[ok], G)
    W1 = np.bincount(gid[ok], om[ok] * t[ok], G)
    rest = W0[np.maximum(gid, 0)] - om
    T_loo = np.where(
        ok & (rest > 0),
        (W1[np.maximum(gid, 0)] - om * t) / np.maximum(rest, 1e-300),
        np.nan,
    )
    z = (t - T_loo) / np.sqrt(sd**2 + s2 + 1.0 / np.maximum(rest, 1e-300))
    kind = np.array([labels[g][0] if g >= 0 else "" for g in gid])
    mem = pd.DataFrame({"source_id": star_ids, "gid": gid, "kind": kind, "t_rec": t, "sd_rec": sd,
                        "rho": rho, "prior": prior, "z": np.where(ok & (rho > 0.5), z, np.nan)})  # fmt: skip
    mem = mem[mem.gid >= 0]
    mem.to_parquet(work / "cluster_members_A.parquet", index=False)
    # raw member values (all fit-source records of members), as in the audit
    raw = pd.read_parquet(records, columns=["source", "source_id", "lage", "good"])
    raw = raw[raw.good & raw.source.astype(str).isin(FIT_SOURCES)]
    cm = mem[mem.kind == "cluster"].copy()
    cm["cid"] = [labels[g][1] for g in cm.gid]
    cat = Table.read(clusters, hdu=1)[
        ["Name", "ID", "Type", "logAge50", "CMDCl50"]
    ].to_pandas()
    cat["Name"] = cat.Name.astype(str).str.strip()
    cat = cat.set_index("ID")
    # raw values (all good records of fit catalogues) of the same member stars
    rj = raw.merge(cm.loc[np.isfinite(cm.t_rec), ["source_id", "cid"]], on="source_id")
    rows = []
    for cid, g in cm.groupby("cid"):
        gg = g[np.isfinite(g.t_rec)]
        if len(gg) < 2:
            continue
        w = gg.rho / (gg.rho * gg.sd_rec**2 + CLUSTER_SPREAD**2)
        T = float(np.sum(w * gg.t_rec) / np.sum(w))
        rr = rj[rj.cid == cid]
        per_cat = [nmad(x.lage) for _, x in rr.groupby("source") if len(x) >= 3]
        rows.append({"cid": int(cid), "name": cat.Name.get(cid, ""), "type": cat.Type.get(cid, ""),
                     "n_members": int(len(gg)), "n_members_resp_gt_0.5": int((gg.rho > 0.5).sum()),
                     "T_fit": T, "T_sd": float(1 / np.sqrt(np.sum(w))),
                     "age_fit_gyr": 10**T, "rec_spread": nmad(gg.t_rec) if len(gg) >= 3 else np.nan,
                     "rec_spread_resp_gt_0.5": nmad(gg.t_rec[gg.rho > 0.5]) if (gg.rho > 0.5).sum() >= 3 else np.nan,
                     "raw_median": float(np.median(rr.lage)) if len(rr) else np.nan,
                     "raw_spread": float(np.nanmedian(per_cat)) if per_cat else np.nan,
                     "hr23_logage": float(cat.logAge50.get(cid, np.nan)),
                     "hr23_cmd_class": float(cat.CMDCl50.get(cid, np.nan)),
                     "literature_gyr": LITERATURE_GYR.get(cat.Name.get(cid, ""), np.nan),
                     "z_nmad": nmad(gg.z)})  # fmt: skip
    pd.DataFrame(rows).sort_values("n_members", ascending=False).to_csv(
        docs / "clusters_A.csv", index=False
    )


def export_model(work: Path, docs: Path) -> None:
    """Frozen measurement model per gauge (JSON): what C1 needs to train age
    heads against the consensus likelihood instead of consensus values."""
    for tag in ("A", "C"):
        fit = pickle.load(open(work / f"fit_{tag}.pkl", "rb"))
        res = fit["res"]
        map_table_from_res(res).to_csv(docs / f"maps_{tag}.csv", index=False)
        fam = res.maps.families[0]
        mask = fam.mask()
        maps = {}
        for i, k in enumerate(fam.names):
            maps[k] = {"tier": ["identity", "offset", "linear", "flex"][int(fam.tier[i])],
                       "coef": {c: float(res.coefs[0][i, j]) for j, c in enumerate(fam.column_names()) if mask[i, j]},
                       "feh_range": fam.x_range[i].tolist(), "t_range": fam.t_range[i].tolist()}  # fmt: skip
        units = {u: {"kappa": float(np.exp(res.log_kappa[i])), "omega": float(np.exp(res.log_omega[i]))}
                 for i, u in enumerate(fit["units"])}  # fmt: skip
        out = {"gauge": "apokasc3" if tag == "A" else "xiang24", "latent": "log10(age/Gyr)",
               "pivot_T0": T0, "classes": list(CLASSES), "student_t_nu": res.nu,
               "map_form": "f = t + b + db_c + (A + dA_c)(t - T0) + beta x + g_uu u^2 + g_uuu u^3 "
                           "+ g_xu x u + g_xx x^2; x = [Fe/H] clamped to feh_range, u in the "
                           "polynomial terms clamped to t_range - T0; db_c/dA_c index CLASSES",
               "noise": "Student-t scale^2 = kappa^2 sigma_quoted^2 + omega^2 per unit "
                        "(source:class); records of isochrone catalogues share an age error "
                        "u ~ N(0, eta^2) per star (y = f(t + u))",
               "eta": {n: float(np.exp(v)) for n, v in zip(EFFECT_FAMILIES, res.log_eta)},
               "isochrone_sources": sorted(ISOCHRONE), "caps_gyr_right_censored_within_1pc": CAPS,
               "main_sequence_rule": f"isochrone records with catalogue log g >= {MS_LOGG} excluded",
               "shared_input_groups_priority": SHARED_INPUT_GROUPS, "maps": maps, "noise_units": units}  # fmt: skip
        (docs / f"measurement_model_{tag}.json").write_text(json.dumps(out, indent=1))


def export_records(work: Path, records: Path, targets: Path) -> None:
    """Per-record table of the gauge-A fit (representative records): the input
    C1 needs to train against the C0 measurement model (records, not consensus
    values). Large; written to WORK."""
    data = prepare(records, targets)
    prob = build_problem(data, Config("A_apokasc3"))
    fitA = pickle.load(open(work / "fit_A.pkl", "rb"))
    df = prob["fit_df"][["source_id", "source", "group", "key", "unit", "cls", "lage", "sig", "feh",
                         "censor"]].copy()  # fmt: skip
    df["effect"] = np.where(df.source.isin(ISOCHRONE), "isochrone", "")
    pos = np.searchsorted(prob["star_ids"], df.source_id.to_numpy())
    gid = prob["groups"].gid[pos]
    lab = prob["group_labels"]
    df["conatal_kind"] = [lab[g][0] if g >= 0 else "" for g in gid]
    df["conatal_id"] = np.array([lab[g][1] if g >= 0 else -1 for g in gid], "i8")
    df["conatal_prior"] = prob["groups"].prior[pos].astype("f4")
    df["conatal_resp_A"] = fitA["rho"][pos].astype("f4")
    df.to_parquet(work / "records_c0_A.parquet", index=False)


def plots(args) -> None:
    work, docs = Path(args.work), Path(args.docs)
    export_model(work, docs)
    export_records(work, Path(args.records), Path(args.targets))
    cluster_tables(
        work, docs, Path(args.records), Path(args.clusters), Path(args.targets)
    )
    make_plots(work, docs, Path(args.clusters))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("stage", choices=["run", "plots"])
    ap.add_argument("--records", default="records.parquet")
    ap.add_argument("--targets", default="consistency_targets_20261118.npz")
    ap.add_argument("--clusters", default="clusters.fits")
    ap.add_argument("--work", default="age_c0_work")
    ap.add_argument("--docs", default="docs/age-c0-20261122")
    ap.add_argument("--n-outer", type=int, default=30)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quick", action="store_true", help="main solutions only")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--resume", action="store_true", help="reuse saved main fits and variant tables"
    )
    args = ap.parse_args(argv)
    {"run": run, "plots": plots}[args.stage](args)


if __name__ == "__main__":
    main()
