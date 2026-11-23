#!/usr/bin/env python3
"""C0/C0b age consensus: per-catalogue maps, noise, censoring and co-natal
terms from the audit's age records and the C0b bridge labels (no network).
Design note section 7, ``docs/design-20261122-age-consensus.md``; model in
``masked_stellar_autoencoder.training.age_consensus``.

usage:
  age_consensus_c0.py run   --records OUT/records.parquet --targets consistency_targets.npz \
                            --labels LABELS --clusters clusters.fits --work WORK --docs DOCS [--quick]
  age_consensus_c0.py plots --work WORK --docs DOCS --clusters clusters.fits [--labels LABELS]

``run`` fits the APOKASC-3-anchored solution (gauge A), the Xiang 2024-anchored
solution (gauge C, from scratch and from the transformed gauge-A solution),
the arbitration fits (benchmark-cluster and white-dwarf ages validation-only,
both gauges), the variants and the leave-one-catalogue-out refits; it writes
tables to DOCS and the per-star consensus (Parquet, large) to WORK. ``plots``
draws the QA figures from WORK into DOCS.

No label is truth; every label is fitted. The gauge anchor enters through an
identity map plus fitted noise (quoted error x learned inflation, floor,
Student-t tails, shared effect with its input family); every other catalogue
through its fitted map and noise. Cluster ages (Hunt & Reffert 2023,
Cantat-Gaudin 2020, Dias 2021, and the benchmark LDB / eclipsing-binary / GC
ages of ``--labels``) are records of a cluster node, a pseudo-member of the
cluster group, each source with its own map and noise; cluster latents are
fitted (flat prior). Wide pairs (El-Badry 2021, R_chance < 0.1) are equality
constraints with a small spread; memberships are fixed external priors with
a Student-t member term (gauge-covariant). White-dwarf system ages are
records of the main-sequence companion.

User decisions applied (2026-11-22): APOKASC-3 identity anchor; second complete
solution anchored on Xiang 2024; FLAME excluded; Leung 2023 and Martig 2016 for
validation only; main-sequence records of the isochrone pipelines excluded (one
rule for StarHorse, GALAH DR4 and Xiang: the catalogue's own log g >= 4.1, the
audit's dwarf class; StarHorse "MSTO" stars reach log g 4.33); ages above
13.8 Gyr kept; catalogue caps (StarHorse 13.72, GALAH DR4 14.79, Willett 19.5,
Xiang 20 Gyr) right-censored within 1 % of the cap. C0b (2026-11-23): white-
dwarf ages validation-only in the main fit (fitted in a variant); benchmark
clusters fitted in the main fit, validation-only in the arbitration fits.
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
    noise_identified,
    record_residuals,
    solve_stars,
)

CLASSES = ("MS", "MSTO", "SGB", "RGB", "RC", "giant", "cluster", "system")
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
#: C0b sources (``--labels``): extra seismology joins the groups whose light
#: curves it shares; white-dwarf system ages are records of the main-sequence
#: companion; benchmark and catalogue cluster ages are records of a cluster node.
SEISMIC_EXTRA = ("zinn22_k2gap", "mackereth21_tess")
WD_SOURCES = ("wd_qiu21", "wd_raddi22_H_C18")
BENCH_SOURCES = ("bench_ldb", "bench_eb", "bench_gc")
CLUSTER_CATALOGUES = ("cl_hr23", "cl_cg20", "cl_dias21")
#: Independent-physics arbiters (benchmark clusters, white-dwarf systems): one
#: offset map each; with a handful of records (one per cluster) an offset and a
#: noise scale need FEW_MIN constrained records (two residual degrees of freedom).
OFFSET_ONLY = BENCH_SOURCES + WD_SOURCES
FEW_MIN = 3
SHARED_INPUT_GROUPS = {
    "apokasc": ("apokasc3", "apokasc2", "apok2_kepler"),
    "apok2": ("apok2", "zinn22_k2gap"),
    "willett26": (*WILLETT, "mackereth21_tess"),
    "legacy17": LEGACY,
    "starhorse23": STARHORSE,
    "chaplin14": ("chaplin14",),
    "xiang24": ("xiang24",),
    "galah_dr4": ("galah_dr4",),
    "wd": WD_SOURCES,
    **{s: (s,) for s in BENCH_SOURCES + CLUSTER_CATALOGUES},
}
FIT_SOURCES = tuple(s for g in SHARED_INPUT_GROUPS.values() for s in g)
VALIDATION_SOURCES = ("leung23", "martig16")
#: Shared age error per star for the isochrone pipelines (common Gaia
#: photometry/parallax, overlapping spectra); learned through independent
#: constraints (co-natal partners, seismic dwarfs). No seismic shared effect:
#: the audit found APOKASC-3 vs Willett consistent with independent errors, and
#: a seismic eta is not separable from the GALAH giant noise (its only
#: independent overlap), where it absorbed the cross-regime disagreement.
EFFECT_FAMILIES = ("isochrone", "cluster_cmd")
ISOCHRONE = set(STARHORSE) | {"xiang24", "galah_dr4"}
#: Nested sub-families of the isochrone family: pipelines on the same spectra.
SUB_FAMILIES = {
    "lamost_spectra": ("starhorse23_lamostl7", "starhorse23_lamostm7", "xiang24"),
    "galah_spectra": ("starhorse23_galahdr3", "galah_dr4"),
}
LDB_SYS_FRAC = 0.09  # Galindo-Guil+22 / Jeffries & Naylor 2001: 7-11 % LDB systematics
VB21_MIN_PROB = 0.5
NODE_ID0 = -1_000_000  # cluster nodes get source_id NODE_ID0 - cluster id
#: Upper caps (Gyr); values within 1 % are right-censored.
CAPS = {**{s: 13.725 for s in STARHORSE}, "galah_dr4": 14.794, "xiang24": 20.0,
        **{s: 19.5 for s in WILLETT}, "martig16": 13.0}  # fmt: skip
RANK = {**{s: 6 for s in LEGACY}, "apokasc3": 5, "apokasc2": 5, "apok2_kepler": 5,
        "chaplin14": 5, **{s: 4 for s in WILLETT}, "apok2": 4, "xiang24": 3,
        **{s: 4 for s in SEISMIC_EXTRA}, **{s: 2 for s in STARHORSE}, "galah_dr4": 2,
        **{s: 1 for s in WD_SOURCES}, **{s: 0 for s in BENCH_SOURCES + CLUSTER_CATALOGUES}}  # fmt: skip
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
YOUNG_CLUSTERS = {"Melotte_22": "Pleiades", "NGC_2632": "Praesepe", "Melotte_20": "alpha Per",
                  "Melotte_25": "Hyades", "NGC_2516": "NGC 2516", "IC_2602": "IC 2602",
                  "IC_2391": "IC 2391", "Blanco_1": "Blanco 1"}  # fmt: skip
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f2328", "#5b6470", "#e6e6e3"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
PREFIX = "c0b"  # QA figure file prefix
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


def _frame(source, sid, age, sig, cls, feh=np.nan, censor=0, lage2=np.nan):
    """Records in the common column layout (age in Gyr, sig in dex)."""
    sid = np.asarray(sid, "i8")
    n = len(sid)
    age = np.asarray(age, float)
    return pd.DataFrame({
        "source": source, "source_id": sid, "age": age,
        "lage": np.log10(np.where(age > 0, age, np.nan)), "sig": np.broadcast_to(sig, n).astype(float),
        "stage": cls, "stage_class": cls, "feh": np.broadcast_to(feh, n).astype(float),
        "logg": np.nan, "cls": cls, "censor": np.broadcast_to(censor, n).astype("i1"),
        "lage2": np.broadcast_to(lage2, n).astype(float),
    })  # fmt: skip


def _dex(err_lo, err_hi, value):
    e = np.nanmean(np.c_[np.asarray(err_lo, float), np.asarray(err_hi, float)], axis=1)
    return e / np.asarray(value, float) / np.log(10.0)


def load_bridge_labels(labels: Path, clusters: Path, members: pd.DataFrame):
    """C0b label families (README in ``labels``). Returns (records, extra
    members, wide pairs, cluster table). Nothing is taken as truth: every
    record gets a fitted map and noise; cluster ages are records of a cluster
    node (a pseudo-member of the cluster group), not values of the latent."""
    from astropy.table import Table

    frames = []
    # ---- extra seismology (P6): ages as given, giants
    z = pd.read_parquet(labels / "seis_zinn22_k2gap.parquet")
    z = z[(z.kind == "age_gyr") & (z.value > 0)]
    frames.append(_frame("zinn22_k2gap", z.source_id, z.value, _dex(z.err_lo, z.err_hi, z.value),
                         "giant", z.feh))  # fmt: skip
    m = pd.read_parquet(labels / "seis_mackereth21_tess.parquet")
    m = m[(m.kind == "age_gyr") & (m.value > 0)].drop_duplicates("source_id")
    frames.append(_frame("mackereth21_tess", m.source_id, m.value, _dex(m.err_lo, m.err_hi, m.value),
                         "giant", m.feh))  # fmt: skip
    # ---- white-dwarf system ages (P3) as records of the MS companion
    w = pd.read_parquet(labels / "wd_ages_published.parquet")
    pairs_wd = pd.read_parquet(labels / "wdms_pairs.parquet",
                               columns=["wd_source_id", "ms_source_id", "usable_H", "atm_hint"])  # fmt: skip
    usable = pairs_wd[pairs_wd.usable_H.astype(bool)].drop_duplicates("wd_source_id")
    for src, name in (
        ("qiu21_base9", "wd_qiu21"),
        ("raddi22_H_C18", "wd_raddi22_H_C18"),
    ):
        x = w[(w.source == src) & (w.value > 0) & np.isfinite(w.companion_source_id)]
        x = x.merge(usable, left_on="source_id", right_on="wd_source_id")
        if name.startswith("wd_raddi22_H"):
            x = x[x.atm_hint.astype(str) != "He"]
        cens = np.where(
            x["limit"].to_numpy() == -1, 1, 0
        )  # '>' progenitor age: lower limit
        frames.append(_frame(name, x.companion_source_id.astype("i8"), x.value,
                             _dex(x.err_lo, x.err_hi, x.value), "system", censor=cens))  # fmt: skip
    # ---- cluster nodes: HR23 table, CG20, Dias21, benchmarks (P1)
    cl = Table.read(clusters, hdu=1)[
        ["Name", "ID", "logAge16", "logAge50", "logAge84"]
    ].to_pandas()
    cl["Name"] = cl.Name.astype(str).str.strip()
    vb = pd.read_parquet(labels / "cluster_members_extra.parquet")
    vb = vb[vb.prob >= VB21_MIN_PROB]
    name2id = dict(zip(cl.Name, cl.ID.astype("i8")))
    extra_id = {c: 900_000 + j for j, c in enumerate(sorted(vb.cluster[vb.hr23_name.isna() |
                                                              (vb.hr23_name == "")].unique()))}  # fmt: skip
    vb["cid"] = [
        name2id.get(h, extra_id.get(c, -1)) for h, c in zip(vb.hr23_name, vb.cluster)
    ]
    vb = vb[vb.cid >= 0]
    # VB21 replaces HR23 membership for its globular clusters
    mem = pd.concat([members[~members.cid.isin(vb.cid.unique())],
                     pd.DataFrame({"source_id": vb.source_id, "cid": vb.cid.astype("i8"),
                                   "prob": vb.prob.astype(float)})], ignore_index=True)  # fmt: skip
    mem = mem.sort_values("prob", ascending=False).drop_duplicates("source_id")
    ok = np.isfinite(cl.logAge50)
    frames.append(_frame("cl_hr23", NODE_ID0 - cl.ID[ok].astype("i8"), 10 ** (cl.logAge50[ok] - 9),
                         0.5 * (cl.logAge84[ok] - cl.logAge16[ok]), "cluster"))  # fmt: skip
    msto = pd.read_parquet(labels / "cluster_msto_catalogues.parquet")
    msto = msto[(msto.hr23_id >= 0) & np.isfinite(msto.value)]
    for cat, name in (("cantatgaudin20", "cl_cg20"), ("dias21", "cl_dias21")):
        x = msto[msto.catalogue == cat].drop_duplicates("hr23_id")
        frames.append(_frame(name, NODE_ID0 - x.hr23_id.astype("i8"), 10 ** (x.value - 9),
                             x.err_lo.to_numpy(float), "cluster", x.feh))  # fmt: skip
    pr = pd.read_csv(labels / "cluster_priors.csv")
    pr["family"] = np.select([pr.method.str.startswith("LDB"), pr.cluster_type == "GC"],
                             ["bench_ldb", "bench_gc"], "bench_eb")  # fmt: skip
    pr["cid"] = [name2id.get(h, extra_id.get(c, -1)) if isinstance(h, str) else
                 next((v for k, v in extra_id.items() if k.split(" (")[0] in c), -1)
                 for h, c in zip(pr.hr23_name, pr.cluster)]  # fmt: skip
    pr = pr[pr.cid >= 0]
    # one record per cluster and method family: the primary row, else the first
    pr = pr.sort_values(["cid", "family", "primary"], ascending=[True, True, False])
    pr = pr.drop_duplicates(["cid", "family"])
    stat = _dex(pr.err_lo_gyr, pr.err_hi_gyr, pr.age_gyr)
    sysd = np.nan_to_num(pr.sys_err_gyr.to_numpy(float) / pr.age_gyr / np.log(10.0))
    model = np.where(np.isfinite(pr.model_range_lo_gyr) & np.isfinite(pr.model_range_hi_gyr),
                     0.5 * (np.log10(pr.model_range_hi_gyr) - np.log10(pr.model_range_lo_gyr)), 0.0)  # fmt: skip
    ldb = (pr.family == "bench_ldb").to_numpy()
    sig = np.sqrt(
        np.nan_to_num(stat) ** 2
        + sysd**2
        + np.where(ldb, model**2 + np.log10(1 + LDB_SYS_FRAC) ** 2, 0.0)
    )
    # published ranges (M67 "between about 3.5 and 4.0 Gyr", NGC 6253 3.80-4.25):
    # interval records with a flat-top likelihood, not Gaussians
    rng = pr.notes.fillna("").str.contains("range|between", case=False).to_numpy() & \
        pr.method.str.contains("EB").to_numpy()  # fmt: skip
    lo = np.where(rng, pr.age_gyr - pr.err_lo_gyr, pr.age_gyr)
    hi = np.where(rng, pr.age_gyr + pr.err_hi_gyr, np.nan)
    for fam in BENCH_SOURCES:
        sel = (pr.family == fam).to_numpy()
        frames.append(_frame(fam, NODE_ID0 - pr.cid[sel].astype("i8"), lo[sel], np.where(rng[sel], 0.0, sig[sel]),
                             "cluster", pr.feh[sel], censor=np.where(rng[sel], 2, 0),
                             lage2=np.log10(np.where(rng[sel], hi[sel], np.nan))))  # fmt: skip
    rec = pd.concat(frames, ignore_index=True)
    rec = rec[np.isfinite(rec.lage)]
    # cluster nodes are members (probability 1) of their own cluster group
    nodes = np.unique(rec.source_id[rec.source_id <= NODE_ID0])
    mem = pd.concat([mem, pd.DataFrame({"source_id": nodes, "cid": NODE_ID0 - nodes, "prob": 1.0})],
                    ignore_index=True)  # fmt: skip
    wp = pd.read_parquet(labels / "wide_pairs_eb21.parquet",
                         columns=["source_id1", "source_id2", "R_chance_align", "binary_type"])  # fmt: skip
    wp = wp[wp.R_chance_align < 0.1]
    pairs = pd.DataFrame({"s1": wp.source_id1.to_numpy("i8"), "s2": wp.source_id2.to_numpy("i8"),
                          "prior": 1.0 - wp.R_chance_align.to_numpy(float),
                          "type": wp.binary_type.astype(str).to_numpy()})  # fmt: skip
    clusters_tab = pd.concat([cl, pd.DataFrame({"Name": list(extra_id), "ID": list(extra_id.values())})],
                             ignore_index=True)  # fmt: skip
    clusters_tab["vb21"] = clusters_tab.ID.isin(vb.cid.unique())
    return rec, mem, pairs, clusters_tab


def prepare(records: Path, targets: Path, labels: Path | None = None,
            clusters: Path | None = None) -> dict:  # fmt: skip
    """Harmonised fit/validation records and co-natal groups (C0b label
    families when ``labels`` is given)."""
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
    use["lage2"] = np.nan
    z = np.load(targets)
    mem = pd.DataFrame({"source_id": z["member"], "cid": z["member_cluster"].astype("i8"),
                        "prob": z["member_prob"].astype(float)})  # fmt: skip
    mem = mem.sort_values("prob", ascending=False).drop_duplicates("source_id")
    pairs = pd.DataFrame(
        {"s1": z["pair1"], "s2": z["pair2"], "prior": BINARY_PRIOR, "type": "MSMS"}
    )
    clusters_tab = None
    if labels is not None:
        extra, mem, pairs, clusters_tab = load_bridge_labels(
            Path(labels), Path(clusters), mem
        )
        extra = extra[(extra.source_id <= NODE_ID0) | (extra.source_id > 0)]
        excluded["C0b label records"] = int(len(extra))
        use = pd.concat([use, extra], ignore_index=True)
    med = use.groupby("source_id").feh.median()  # [Fe/H]: own, else the star's median
    use["feh"] = use.feh.fillna(use.source_id.map(med))
    use["group"], use["prio"] = "", 0
    for gname, members in SHARED_INPUT_GROUPS.items():
        for p, s in enumerate(members):
            msk = use.source == s
            use.loc[msk, "group"] = gname
            use.loc[msk, "prio"] = p
    fit = use[use.source.isin(FIT_SOURCES)].sort_values(["source_id", "group", "prio"])
    fit = fit.assign(
        representative=(~fit.duplicated(["source_id", "group"])).to_numpy()
    )
    return {"fit": fit.reset_index(drop=True),
            "val": use[use.source.isin(VALIDATION_SOURCES)].reset_index(drop=True),
            "all_ids": all_ids, "members": mem, "pairs": pairs, "excluded": excluded,
            "n_raw": int(len(raw)), "clusters": clusters_tab}  # fmt: skip


def conatal_groups(star_ids, members, pairs, use_groups=True):
    """Groups over the sorted ``star_ids``: clusters first (membership
    probability as a fixed prior), then wide pairs (prior 1 - R_chance)."""
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
    pp = (
        pairs.prior.to_numpy(float)
        if "prior" in pairs
        else np.full(len(pairs), BINARY_PRIOR)
    )
    k = len(spread)
    for j in np.flatnonzero(ok):
        a, b = p1[j], p2[j]
        if gid[a] >= 0 or gid[b] >= 0:
            continue
        gid[a] = gid[b] = k
        prior[a] = prior[b] = pp[j]
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
    galah_giants: bool = False  # GALAH DR4 giant ages in the fit (else validation only)
    bridge_tier: int = LINEAR  # cap on the tier of a component reference
    mission_continuity: bool = True  # one Willett map per variant (Kepler, K2, TESS)
    conatal: bool = True
    identity_extra: tuple = ()  # further identity-mapped catalogues
    drop: tuple = ()  # held-out catalogues
    wd_fit: bool = False  # white-dwarf system ages in the fit (else validation only)
    bench_fit: bool = True  # benchmark cluster ages in the fit (else validation only)
    identifiability: bool = (
        True  # slope/[Fe/H] terms only for units with identified noise
    )


def map_key(source: str, cls: str, cfg: Config) -> str:
    if cfg.mission_continuity and source.startswith("willett26"):
        return "willett26_L" if source.endswith("_L") else "willett26_dnu"
    if source == "galah_dr4" and not cfg.galah_continuity:
        return "galah_dr4_giant" if cls in GIANT else "galah_dr4_sgb"
    return source


def build_problem(data: dict, cfg: Config, log=print) -> dict:
    fit = data["fit"]
    fit = fit[~fit.source.isin(cfg.drop)]
    if not cfg.wd_fit:
        fit = fit[~fit.source.isin(WD_SOURCES)]
    if not cfg.bench_fit:
        fit = fit[~fit.source.isin(BENCH_SOURCES)]
    if not (cfg.galah_giants or cfg.galah_continuity):
        # GALAH DR4 giant isochrone ages: validation only (their map, calibrated
        # on K2 seismic giants, extrapolates badly to old metal-poor cluster giants)
        fit = fit[~((fit.source == "galah_dr4") & fit.cls.isin(GIANT))]
    fit = fit.sort_values(["source_id", "group", "prio"])
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
    # noise units and their identifiability (triads, or pairs with an identified unit)
    cnt0 = fit.groupby(["source", "cls"]).size()
    fit["unit"] = np.where([cnt0[(s, c)] >= NOISE_MIN for s, c in zip(fit.source, fit.cls)],
                           fit.source + ":" + fit.cls, fit.source + ":all")  # fmt: skip
    ucodes, ucode = np.unique(fit.unit.to_numpy(), return_inverse=True)
    # independent measurements: shared-input groups, with pipelines on the same
    # spectra (a nested shared effect) counted as one
    indep = fit.group.to_numpy().astype(object)
    for sub, srcs in SUB_FAMILIES.items():
        indep[fit.source.isin(srcs).to_numpy()] = sub
    gcode = np.unique(indep.astype(str), return_inverse=True)[1]
    ident, triad, pair_id = noise_identified(star, gcode, ucode, len(ucodes), partner,
                                             min_count=MIN_OVERLAP)  # fmt: skip
    unit_ident = dict(zip(ucodes, ident))
    unit_triad = dict(zip(ucodes, triad))
    unit_pair = dict(zip(ucodes, pair_id))
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
        elif k in OFFSET_ONLY:
            t = OFFSET if n >= FEW_MIN else -1
        elif n < MIN_OVERLAP:
            t = -1
        elif n < LINEAR_MIN or span < SPAN_MIN:
            t = OFFSET
        else:
            t = LINEAR if n < FLEX_MIN else FLEX
        kunits = sorted(fit.unit[fit.key == k].unique())
        # the slope is identified by the constrained records whose unit noise
        # is identified; records of other units follow that slope (same count
        # rule as for an offset)
        n_id = int(g.unit.map(unit_ident).sum())
        k_ident = n_id >= MIN_OVERLAP
        if cfg.identifiability and t > OFFSET and not k_ident:
            t = OFFSET  # EIV slope needs the variance ratio: only an offset is identified
        tier[i] = t
        use_feh[i] = bool(n) and fok.mean() > 0.5 and fspan >= 0.4
        cont = k in ("willett26_dnu", "willett26_L") or (k == "galah_dr4")
        if t >= LINEAR and not cont:
            cc = g.cls.value_counts()
            big = [c for c in cc.index if cc[c] >= CLASS_MIN]
            for c in big[1:]:  # the most populous class is the reference
                class_terms[i, CLASSES.index(c)] = True
        info.append({"key": k, "n_constrained": n, "age_span_dex": round(span, 3),
                     "feh_span_dex": round(fspan, 3), "tier_by_count": int(t),
                     "noise_identified": bool(k_ident), "n_constrained_identified": n_id,
                     "triad_stars": int(min(unit_triad[u] for u in kunits)),
                     "pairs_with_identified": int(min(unit_pair[u] for u in kunits))})  # fmt: skip
    # components joined by >= DENSE shared stars; references for anchor-less ones
    kk = fit.key.map(kidx).to_numpy()
    pc = _pair_counts(star, kk)
    comp = _components(M, [(a, b) for (a, b), n in pc.items() if n >= DENSE])
    live = tier[kk] >= 0
    sc = pd.DataFrame({"s": star[live], "c": comp[kk[live]]}).drop_duplicates()

    def need(c):
        mk = [k for k in keys if comp[kidx[k]] == c and tier[kidx[k]] >= 0]
        return FEW_MIN if all(k in OFFSET_ONLY for k in mk) else MIN_OVERLAP

    for c, width in _attach_order(sc, gid, comp[kidx[cfg.anchor]], need):
        mk = [k for k in keys if comp[kidx[k]] == c and tier[kidx[k]] >= 0]
        ref = next(
            (k for k in REFERENCE_PREFERENCE if k in mk),
            max(mk, key=lambda k: (fit.key == k).sum()),
        )
        i = kidx[ref]
        if width < need(c):
            for k in mk:
                tier[kidx[k]] = -1
            log(f"[{cfg.name}] component {mk} not connected (width {width}); dropped")
            continue
        if tier[i] != IDENTITY:
            # the reference of a thinly bridged component carries only what the
            # bridge supports: an offset below LINEAR_MIN independent links
            tier[i] = OFFSET if width < LINEAR_MIN else min(tier[i], cfg.bridge_tier)
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
    # a map is defined on the range of its data: mapped records below it (e.g.
    # pre-main-sequence stars in a subgiant catalogue) become upper limits at the
    # domain edge instead of extrapolated values
    # (the anchor too: the censoring set must not depend on the gauge)
    for k in keep:
        msk = (fit.key == k).to_numpy()
        base = fit.lage[msk & fit.constrained.to_numpy()]
        if len(base) < 100:
            continue
        lo = float(np.percentile(base, 0.1))
        below = msk & (fit.lage.to_numpy() < lo) & (fit.censor.to_numpy() == 0)
        fit.loc[below, "lage"] = lo
        fit.loc[below, "censor"] = -1
        for row in info:
            if row["key"] == k:
                row["domain_lo"] = round(lo, 3)
                row["n_below_domain"] = int(below.sum())
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
    effect = np.select([fit.source.isin(ISOCHRONE), fit.source.isin(CLUSTER_CATALOGUES)], [0, 1],
                       -1).astype("i8")  # fmt: skip
    subeffect = np.full(len(fit), -1, "i8")
    for j, srcs in enumerate(SUB_FAMILIES.values()):
        subeffect[fit.source.isin(srcs).to_numpy()] = j
    kmap = {k: i for i, k in enumerate(keep)}
    y2 = np.where(
        fit.censor.to_numpy() == 2, fit.lage2.to_numpy(float), fit.lage.to_numpy(float)
    )
    rec = Records(
        star.astype("i8"), fit.key.map(kmap).to_numpy("i8"),
        np.array([CLASSES.index(c) for c in fit.cls], "i8"), fit.unit.map(uidx).to_numpy("i8"),
        fit.lage.to_numpy(float), fit.sig.to_numpy(float), fit.feh.to_numpy(float),
        fit.censor.to_numpy("i1"), len(star_ids), effect=effect, subeffect=subeffect, y2=y2,
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


def _attach_order(sc: pd.DataFrame, gid: np.ndarray, root: int, need) -> list:
    """Components joined to the anchor's along a maximum spanning tree (Prim):
    the edge set of the tree does not depend on the root, so each bridge gets
    the same width, and hence the same reference tier, whichever catalogue is
    the anchor (in gauge A the seismic-isochrone edge sets Xiang's reference
    tier, in gauge C APOKASC-3's). The width of an edge counts independent
    links between two components: shared stars, plus co-natal groups with
    members on both sides (a cluster is one link, whatever its number of
    members). A component joins only through an edge of at least ``need(c)``
    links. Returns [(component, width)] in joining order, then the components
    left out with their widest edge to the joined set."""
    stars = {c: set(g.s) for c, g in sc.groupby("c")}
    grps = {c: {int(gid[s]) for s in ss if gid[s] >= 0} for c, ss in stars.items()}
    if root not in stars:
        return []
    comps = sorted(stars)
    W = {
        (a, b): len(stars[a] & stars[b]) + len(grps[a] & grps[b])
        for a in comps
        for b in comps
        if a < b
    }

    def w(a, b):
        return W[(min(a, b), max(a, b))]

    joined, left, out = [root], set(comps) - {root}, []
    while left:
        best = {c: max(w(j, c) for j in joined) for c in left}
        ok = [c for c in left if best[c] >= need(c)]
        if not ok:
            out += [(int(c), int(best[c])) for c in sorted(left)]
            break
        c = max(sorted(ok), key=lambda k: best[k])
        out.append((int(c), int(best[c])))
        joined.append(c)
        left.remove(c)
    return out


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
    le0 = np.full(len(EFFECT_FAMILIES) + len(SUB_FAMILIES), np.log(0.03))
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
    res = fit_consensus(r, gsub, maps, len(units), len(EFFECT_FAMILIES), n_sub=len(SUB_FAMILIES),
                        coefs0=coefs0, lk0=lk0, lw0=lw0, le0=le0, covariance=covariance, log=log,
                        **kw)  # fmt: skip
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
    for e, name in enumerate((*EFFECT_FAMILIES, *SUB_FAMILIES)):
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
    "zinn22_k2gap": ("zinn22_k2gap",),
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
    "cluster_catalogues": CLUSTER_CATALOGUES,
    "bench_clusters": BENCH_SOURCES,
}  # fmt: skip
GRID_T = np.array([0.0, 0.3, 0.6, 0.9, 1.1])
ZP_T = 0.6  # bridge zero point: Xiang minus latent at 4 Gyr, [Fe/H] = 0


# --------------------------------------------------------------------------- #
# Gauge change, arbitration offsets, co-natal tables
# --------------------------------------------------------------------------- #
def reparam_warm(probA, outA, probC, ref="xiang24"):
    """Gauge A's solution re-expressed on catalogue ``ref``'s scale: with
    ``ref``'s gauge-A map f = t + b + A (t - T0) (s = 1 + A), the latent is
    t_C = T0 + s (t_A - T0) + b and every map transforms exactly (polynomial
    and class terms included); latent-space shared-error scales scale by s.
    Returns (warm dict, fit_consensus starting kwargs, largest coefficient that
    gauge C's map layout cannot hold)."""
    resA = outA["res"]
    famA, famC = resA.maps.families[0], probC["maps"].families[0]
    cr = resA.coef_of(ref)
    b, s = float(cr[0]), 1.0 + float(cr[1])
    c = famA.n_cls
    c0 = famC.init()
    lost = 0.0
    for i, k in enumerate(famC.names):
        if k not in famA.names:
            continue
        r = resA.coefs[0][famA.names.index(k)].astype(float)
        guu, gxu = r[3 + 2 * c], r[5 + 2 * c]
        o = r.copy()
        o[0] = r[0] - (1.0 + r[1]) * b / s + guu * b * b / s**2
        o[1] = (1.0 + r[1]) / s - 1.0 - 2.0 * guu * b / s**2
        o[2] = r[2] - gxu * b / s
        o[3 : 3 + c] = r[3 : 3 + c] - r[3 + c : 3 + 2 * c] * b / s
        o[3 + c : 3 + 2 * c] = r[3 + c : 3 + 2 * c] / s
        o[3 + 2 * c] = guu / s**2
        o[4 + 2 * c] = r[4 + 2 * c] / s**3
        o[5 + 2 * c] = gxu / s
        m = famC.mask()[i]
        lost = max(lost, float(np.max(np.abs(o[~m]), initial=0.0)))
        c0[i] = o * m
    if not (np.array_equal(probA["in_fit"], probC["in_fit"])
            and np.array_equal(probA["star_ids"], probC["star_ids"])):  # fmt: skip
        raise ValueError("gauges A and C must share the fit set")
    warm = {"keys": famC.names, "coef": c0, "units": list(probA["units"]),
            "lk": resA.log_kappa, "lw": resA.log_omega, "le": resA.log_eta + np.log(s)}  # fmt: skip
    start = {"t0": T0 + s * (resA.t - T0) + b, "w0": resA.weight, "rho0": resA.rho}
    return warm, start, lost


def offset_vs_consensus(name, recs: pd.DataFrame, star_ids, sol, col="t") -> dict:
    """Offset of a validation-only source from the gauge: an offset map with
    fitted noise (kappa, omega; Student-t) against the consensus, which enters
    as an identity record with its own uncertainty. Interval and limit records
    enter censored. Also the inverse-variance mean with quoted errors (Birge
    ratio applied) as a cross-check."""
    sid = recs.source_id.to_numpy()
    pos = np.minimum(np.searchsorted(star_ids, sid), len(star_ids) - 1)
    hit = star_ids[pos] == sid
    t = np.where(hit, sol[col][pos], np.nan)
    sd = np.where(hit, sol["sd" if col == "t" else "sd_rec"][pos], np.nan)
    ok = np.isfinite(t) & np.isfinite(sd) & (sd > 0)
    r, t, sd = recs[ok], t[ok], sd[ok]
    n = len(r)
    row = {"source": name, "n": int(n)}
    if n < FEW_MIN:
        return row
    y, sig = r.lage.to_numpy(float), r.sig.to_numpy(float)
    cens = r.censor.to_numpy("i1")
    y2 = np.where(cens == 2, r.lage2.to_numpy(float), y)
    maps = MapSet([AgeMaps(["consensus", name], [IDENTITY, OFFSET],
                           np.zeros((2, len(CLASSES)), bool), [False, False])])  # fmt: skip
    rec = Records(np.r_[np.arange(n), np.arange(n)].astype("i8"), np.r_[np.zeros(n), np.ones(n)].astype("i8"),
                  np.zeros(2 * n, "i8"), np.r_[np.zeros(n), np.ones(n)].astype("i8"), np.r_[t, y],
                  np.r_[sd, sig], np.zeros(2 * n), np.r_[np.zeros(n), cens].astype("i1"), n,
                  y2=np.r_[t, y2])  # fmt: skip
    res = fit_consensus(rec, Groups.none(n), maps, 2, n_outer=40, covariance=True,
                        fix_noise=np.array([True, False]), lk0=np.zeros(2),
                        lw0=np.log([1e-4, 0.05]))  # fmt: skip
    se = (
        float(np.sqrt(res.theta_cov[0, 0]))
        if res.theta_cov is not None and res.theta_cov[0, 0] > 0
        else np.nan
    )
    unc = cens == 0
    mid = np.where(cens == 2, 0.5 * (y + y2), y)
    var = np.where(cens == 2, (y2 - y) ** 2 / 12.0, np.nan_to_num(sig) ** 2) + sd**2
    use = (unc | (cens == 2)) & (var > 0)
    d = mid[use] - t[use]
    w = 1.0 / var[use]
    mean = float(np.sum(w * d) / np.sum(w)) if use.any() else np.nan
    chi2 = (
        float(np.sum(w * (d - mean) ** 2) / max(use.sum() - 1, 1))
        if use.sum() > 1
        else np.nan
    )
    se_q = (
        float(np.sqrt(1.0 / np.sum(w)) * max(1.0, np.sqrt(chi2)))
        if use.any()
        else np.nan
    )
    row.update(offset=float(res.coef_of(name)[0]), offset_se=se,
               kappa=float(np.exp(res.log_kappa[1])), omega=float(np.exp(res.log_omega[1])),
               offset_quoted=mean, offset_quoted_se=se_q, chi2_dof_quoted=chi2,
               raw_median=float(np.median(d)) if len(d) else np.nan, n_censored=int((cens != 0).sum()),
               consensus_sd_median=float(np.median(sd)))  # fmt: skip
    return row


def record_scales(prob, out, t_star) -> tuple[np.ndarray, np.ndarray]:
    """Per record of ``prob``: map slope a = df/dt at the star latent and the
    noise variance v (kappa^2 sigma^2 + omega^2); information a^2 / v."""
    rec, res = prob["rec"], out["res"]
    fam = res.maps.families[0]
    t = torch.as_tensor(np.nan_to_num(t_star[rec.star], nan=T0), dtype=torch.float64)
    with torch.no_grad():
        _, a = fam.evaluate(torch.as_tensor(res.coefs[0]), torch.as_tensor(rec.map_id), t,
                            torch.as_tensor(clip_feh(rec.feh)), torch.zeros((len(t), 0), dtype=torch.float64),
                            torch.as_tensor(rec.cls))  # fmt: skip
    sig = np.where(np.isfinite(rec.sig) & (rec.sig > 0), rec.sig, 0.0)
    v = np.exp(2 * res.log_kappa[rec.noise_id]) * sig**2 + np.exp(
        2 * res.log_omega[rec.noise_id]
    )
    return a.numpy(), v


def _stage_of(fit_df: pd.DataFrame) -> pd.Series:
    return (fit_df.assign(r=fit_df.source.map(RANK).fillna(0)).sort_values("r", ascending=False)
            .drop_duplicates("source_id").set_index("source_id").cls)  # fmt: skip


def conatal_tables(prob, out, sol, clusters_tab, raw=None, sol_ref=None):
    """Fitted cluster latents (the cluster node's consensus where the cluster
    has catalogue or benchmark records, else the member mean), the share of
    each information source in them, member z-scores and wide-pair links.
    ``sol_ref``: a solution without the benchmark records (arbitration
    variant) for the shift of each latent. Returns (clusters, members, pairs)."""
    star_ids, g = prob["star_ids"], prob["groups"]
    gid, spread, labels = g.gid, g.spread, prob["group_labels"]
    rho = out["rho"]
    t, sd = sol["t_rec"], sol["sd_rec"]
    is_node = star_ids <= NODE_ID0
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
    kind = np.array([labels[k][0] if k >= 0 else "" for k in gid])
    stage = _stage_of(prob["fit_df"]).reindex(star_ids).to_numpy()
    mem = pd.DataFrame({"source_id": star_ids, "gid": gid, "kind": kind, "t_rec": t, "sd_rec": sd,
                        "rho": rho, "prior": g.prior, "stage": stage, "node": is_node,
                        "z": np.where(ok & (rho > 0.5) & ~is_node, z, np.nan)})  # fmt: skip
    mem = mem[mem.gid >= 0]
    # node records: information a^2 / v per source family
    a, v = record_scales(prob, out, sol["t"])
    info = a * a / v
    df = prob["fit_df"]
    rec_node = is_node[prob["rec"].star]
    fam = np.select([df.source.isin(CLUSTER_CATALOGUES), df.source.isin(BENCH_SOURCES)],
                    ["catalogues", "benchmarks"], "other")  # fmt: skip
    nid = star_ids[prob["rec"].star]
    node_info = pd.DataFrame({"node": nid[rec_node], "fam": fam[rec_node], "fisher": info[rec_node],
                              "src": df.source.to_numpy()[rec_node],
                              "y": df.lage.to_numpy()[rec_node]})  # fmt: skip
    names = (
        dict(zip(clusters_tab.ID.astype("i8"), clusters_tab.Name.astype(str)))
        if clusters_tab is not None
        else {}
    )
    hr = (
        dict(zip(clusters_tab.ID.astype("i8"), clusters_tab.logAge50))
        if clusters_tab is not None
        else {}
    )
    ref_t = {}
    if sol_ref is not None:
        ref_t = {int(s): (tt, ss) for s, tt, ss in zip(sol_ref["star_ids"], sol_ref["t"], sol_ref["sd"])
                 if s <= NODE_ID0}  # fmt: skip
    pos_of = {int(s): i for i, s in enumerate(star_ids) if s <= NODE_ID0}
    rows = []
    cm = mem[(mem.kind == "cluster")]
    # raw catalogue values of the member stars (all good fit-catalogue records)
    raw_med, raw_spr = {}, {}
    if raw is not None:
        rj = raw.merge(
            cm.loc[~cm.node & np.isfinite(cm.t_rec), ["source_id", "gid"]],
            on="source_id",
        )
        raw_med = rj.groupby("gid").lage.median().to_dict()
        pc = rj.groupby(["gid", "source"]).lage.agg(["size", nmad])
        raw_spr = pc[pc["size"] >= 3]["nmad"].groupby(level=0).median().to_dict()
    node_by = dict(tuple(node_info.groupby("node")))
    empty = node_info.iloc[:0]
    for k, gg in cm.groupby("gid"):
        cid = labels[k][1]
        node = NODE_ID0 - cid
        sk2 = float(spread[k]) ** 2
        st = gg[~gg.node & np.isfinite(gg.t_rec) & (gg.sd_rec > 0)]
        p_mem = float(np.sum(st.rho / (st.rho * st.sd_rec**2 + sk2)))
        ni = node_by.get(node, empty)
        p_cat = float(ni.fisher[ni.fam == "catalogues"].sum())
        p_ben = float(ni.fisher[ni.fam == "benchmarks"].sum())
        if node in pos_of and np.isfinite(sol["t"][pos_of[node]]):
            T, Tsd = float(sol["t"][pos_of[node]]), float(sol["sd"][pos_of[node]])
        elif len(st) >= 2:
            w = st.rho / (st.rho * st.sd_rec**2 + sk2)
            T, Tsd = (
                float(np.sum(w * st.t_rec) / np.sum(w)),
                float(1 / np.sqrt(np.sum(w))),
            )
        else:
            continue
        tot = p_mem + p_cat + p_ben
        row = {"cid": int(cid), "name": names.get(cid, ""), "n_members": int(len(st)),
               "n_members_rho_gt_0.5": int((st.rho > 0.5).sum()), "T_fit": T, "T_sd": Tsd,
               "age_fit_gyr": 10**T, "info_share_members": p_mem / tot if tot > 0 else np.nan,
               "info_share_catalogues": p_cat / tot if tot > 0 else np.nan,
               "info_share_benchmarks": p_ben / tot if tot > 0 else np.nan,
               "rec_spread": nmad(st.t_rec) if len(st) >= 3 else np.nan,
               "raw_median": raw_med.get(k, np.nan), "raw_spread": raw_spr.get(k, np.nan),
               "hr23_logage": float(hr.get(cid, np.nan)),
               "literature_gyr": LITERATURE_GYR.get(names.get(cid, ""), np.nan),
               "young_benchmark": YOUNG_CLUSTERS.get(names.get(cid, ""), ""),
               "z_nmad": nmad(st.z)}  # fmt: skip
        for src in CLUSTER_CATALOGUES + BENCH_SOURCES:
            x = ni.y[ni.src == src]
            row[f"rec_{src}"] = float(x.iloc[0]) if len(x) else np.nan
        if node in ref_t:
            row["T_without_benchmarks"], row["T_sd_without_benchmarks"] = ref_t[node]
        rows.append(row)
    clusters = pd.DataFrame(rows)
    if len(clusters):
        clusters = clusters.sort_values("n_members", ascending=False)
    # wide pairs: both members with records-only latents
    bm = mem[(mem.kind == "binary") & np.isfinite(mem.t_rec) & (mem.sd_rec > 0)]
    two = (
        bm.groupby("gid")
        .filter(lambda x: len(x) == 2)
        .sort_values(["gid", "source_id"])
    )
    pairs = pd.DataFrame()
    if len(two):
        a1, a2 = (
            two.iloc[0::2].reset_index(drop=True),
            two.iloc[1::2].reset_index(drop=True),
        )
        zz = (a1.t_rec - a2.t_rec) / np.sqrt(
            a1.sd_rec**2 + a2.sd_rec**2 + 2 * BINARY_SPREAD**2
        )

        def cat(sg):
            return np.select([np.isin(sg, list(GIANT)), np.isin(sg, ["SGB", "MSTO"]), sg == "MS"],
                             [0, 1, 2], 3)  # fmt: skip

        c1, c2 = (
            cat(a1.stage.astype(str).to_numpy()),
            cat(a2.stage.astype(str).to_numpy()),
        )
        nm = np.array(["giant", "subgiant", "dwarf", "other"])
        lo, hi = np.minimum(c1, c2), np.maximum(c1, c2)
        link = np.char.add(np.char.add(nm[lo], "+"), nm[hi])
        link = np.where((hi == 2) & (lo <= 1), "evolved+dwarf", link)
        pairs = pd.DataFrame({"s1": a1.source_id, "s2": a2.source_id, "link": link, "z": zz,
                              "dt": a1.t_rec - a2.t_rec})  # fmt: skip
    return clusters, mem, pairs


def conatal_summary(mem, pairs) -> dict:
    out = {}
    cz = mem.z[(mem.kind == "cluster") & np.isfinite(mem.z)]
    out["cluster_members"] = {
        "n": int(len(cz)),
        "z_nmad": nmad(cz),
        "z_median": float(np.median(cz)) if len(cz) else np.nan,
    }
    if len(pairs):
        out["wide_pairs_all"] = {"n": int(len(pairs)), "z_nmad": nmad(pairs.z)}
        for lk, g in pairs.groupby("link"):
            out[f"wide_pairs_{lk}"] = {
                "n": int(len(g)),
                "z_nmad": nmad(g.z),
                "dt_nmad": nmad(g.dt),
            }
    return out


def bridge_rows(tag, prob, out, se=True) -> list[dict]:
    if "xiang24" not in prob["maps"].names:
        return []
    tr = prob["tiers"].set_index("key")
    rows = []
    for feh in (-0.5, 0.0, 0.3):
        f, _, s = map_values(
            out["res"], "xiang24", GRID_T, feh=feh, cls=CLASSES.index("SGB")
        )
        for tt, dd, ss in zip(GRID_T, f - GRID_T, s):
            rows.append({"variant": tag, "xiang_tier": int(tr.tier["xiang24"]),
                         "bridge_width": tr.get("bridge_width", pd.Series(dtype=float)).get("xiang24", np.nan),
                         "feh": feh, "t": tt, "age_gyr": round(10**tt, 2), "xiang_minus_latent_dex": dd,
                         "se": ss if se else np.nan})  # fmt: skip
    return rows


def bench_map_rows(tag, prob, out) -> list[dict]:
    """Fitted offset maps of the benchmark / cluster-catalogue / WD sources."""
    mt = map_table(prob, out).set_index("key")
    rows = []
    for k in (*BENCH_SOURCES, *WD_SOURCES, *CLUSTER_CATALOGUES):
        if k in mt.index:
            rw = mt.loc[k]
            rows.append({"variant": tag, "source": k, "role": "fitted map", "tier": rw.tier,
                         "offset": rw.get("b", np.nan), "offset_se": rw.get("b_se", np.nan),
                         "slope_A": rw.get("A", np.nan), "n": int((prob["fit_df"].key == k).sum())})  # fmt: skip
    return rows


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
    labels = Path(args.labels) if args.labels else None
    data = prepare(Path(args.records), Path(args.targets), labels,
                   Path(args.clusters) if labels else None)  # fmt: skip
    log(f"records: {data['n_raw']} good with a DR3 ID; fit-source records {len(data['fit'])} "
        f"(representatives {int(data['fit'].representative.sum())}); validation {len(data['val'])}; "
        f"excluded {data['excluded']}")  # fmt: skip
    summary = {"excluded": data["excluded"], "n_records_good": data["n_raw"],
               "n_stars_all": int(len(data["all_ids"])), "n_members": int(len(data["members"])),
               "n_pairs": int(len(data["pairs"]))}  # fmt: skip
    kw = {"n_outer": args.n_outer, "verbose": args.verbose}

    def do(cfg, warm=None, covariance=True, **extra):
        prob = build_problem(data, cfg, log=log)
        return prob, fit_problem(
            prob, warm=warm, covariance=covariance, log=log, **{**kw, **extra}
        )

    def record_fit(tag, prob, out):
        summary[f"fit_{tag}"] = {"stars": out["n_fit_stars"], "records": out["n_fit_rec"],
                                 "outer_iterations": len(out["res"].history),
                                 "last_max_dtheta": out["res"].history[-1]["dtheta"],
                                 "seconds": round(out["seconds"], 1)}  # fmt: skip

    def save(tag, prob, out, sol):
        map_table(prob, out).to_csv(docs / f"maps_{tag}.csv", index=False)
        noise_table(prob, out).to_csv(docs / f"noise_{tag}.csv", index=False)
        prob["tiers"].to_csv(docs / f"tiers_{tag}.csv", index=False)
        record_fit(tag, prob, out)
        with open(work / f"fit_{tag}.pkl", "wb") as fh:
            pickle.dump({"res": out["res"], "units": prob["units"], "tiers": prob["tiers"],
                         "rho": out["rho"], "star_ids": prob["star_ids"],
                         "group_labels": prob["group_labels"], "gid": prob["groups"].gid,
                         "prior": prob["groups"].prior, "spread": prob["groups"].spread,
                         "sol": sol}, fh)  # fmt: skip

    cfgA, cfgC = Config("A_apokasc3"), Config("C_xiang24", anchor="xiang24")
    results, sols = {}, {}
    for tag, cfg in (("A", cfgA), ("C", cfgC)):
        if args.resume and (work / f"fit_{tag}.pkl").exists():
            saved = pickle.load(open(work / f"fit_{tag}.pkl", "rb"))
            prob = build_problem(data, cfg, log=log)
            out = {"res": saved["res"], "rho": saved["rho"], "n_fit_stars": int(prob["in_fit"].sum()),
                   "n_fit_rec": int(prob["in_fit"][prob["rec"].star].sum()), "seconds": float("nan")}  # fmt: skip
            results[tag], sols[tag] = (prob, out), saved["sol"]
            record_fit(tag, prob, out)
            continue
        prob, out = do(cfg)
        results[tag], sols[tag] = (prob, out), solve_all(prob, out)
        save(tag, prob, out, sols[tag])
    probA, outA = results["A"]
    probC, outC = results["C"]
    # ---- gauge C started from the transformed gauge-A solution ----------------
    warm, start, lost = reparam_warm(probA, outA, probC)
    outCA = fit_problem(probC, warm=warm, covariance=False, log=log, **kw, **start)
    sols["C_fromA"] = solve_all(probC, outCA)
    map_table(probC, outCA).to_csv(docs / "maps_C_fromA.csv", index=False)
    record_fit("C_fromA", probC, outCA)
    famC = probC["maps"].families[0]
    msk = famC.mask()
    t_start0 = start["t0"]
    okt = np.isfinite(t_start0)
    dcoef_ca = np.abs(outCA["res"].coefs[0] - warm["coef"])[msk]
    dcoef_c = np.abs(outC["res"].coefs[0] - warm["coef"])[msk]
    summary["gauge_check"] = {
        "xiang_offset_in_A": float(outA["res"].coef_of("xiang24")[0]),
        "xiang_slope_in_A": float(outA["res"].coef_of("xiang24")[1]),
        "coefficient_not_representable_in_C": lost,
        "C_fromA_max_coef_move": float(dcoef_ca.max()),
        "C_fromA_median_abs_latent_move": float(np.median(np.abs(outCA["res"].t - t_start0)[okt])),
        "C_fromA_p99_abs_latent_move": float(np.percentile(np.abs(outCA["res"].t - t_start0)[okt], 99)),
        "C_scratch_max_coef_diff_from_transformed_A": float(dcoef_c.max()),
        "C_scratch_median_abs_latent_diff": float(np.median(np.abs(outC["res"].t - t_start0)[okt])),
        "C_scratch_p99_abs_latent_diff": float(np.percentile(np.abs(outC["res"].t - t_start0)[okt], 99)),
        "eta_A": np.exp(outA["res"].log_eta).tolist(), "eta_C_fromA": np.exp(outCA["res"].log_eta).tolist(),
        "eta_C": np.exp(outC["res"].log_eta).tolist(),
    }  # fmt: skip
    log(f"gauge check: {summary['gauge_check']}")
    # ---- arbitration: benchmark and WD sources validation-only, both gauges ---
    arb = {"A": do(Config("ARB_A", bench_fit=False), warm=warm_from(probA, outA), covariance=False),
           "C": do(Config("ARB_C", anchor="xiang24", bench_fit=False), warm=warm_from(probC, outC),
                   covariance=False)}  # fmt: skip
    arb_rows, bench_rows = [], []
    fitdf = data["fit"]
    for g, (prob, out) in arb.items():
        sol = solve_all(prob, out)
        sol["star_ids"] = prob["star_ids"]
        sols[f"ARB_{g}"] = sol
        record_fit(f"ARB_{g}", prob, out)
        map_table(prob, out).to_csv(docs / f"maps_ARB_{g}.csv", index=False)
        for src in (*BENCH_SOURCES, *WD_SOURCES):
            rr = fitdf[fitdf.source == src]
            arb_rows.append(
                {
                    "gauge": g,
                    "variant": f"ARB_{g}",
                    **offset_vs_consensus(src, rr, prob["star_ids"], sol),
                }
            )
    for g in ("A", "C"):  # main fits: WD validation-only offsets, fitted benchmark maps
        prob, out = results[g]
        for src in WD_SOURCES:
            rr = fitdf[fitdf.source == src]
            arb_rows.append({"gauge": g, "variant": f"main_{g}",
                             **offset_vs_consensus(src, rr, prob["star_ids"], sols[g])})  # fmt: skip
        bench_rows += [
            {"gauge": g, **r} for r in bench_map_rows(f"main_{g}", prob, out)
        ]
    pd.DataFrame(arb_rows).to_csv(docs / "arbitration_offsets.csv", index=False)
    # ---- variants (gauge A) ----------------------------------------------------
    variants = {
        "V_no_identifiability": Config("V_no_identifiability", identifiability=False),
        "V_wd_fit": Config("V_wd_fit", wd_fit=True),
        "V_conatal_only": Config("V_conatal_only", mission_continuity=False),
        "V_seismic_dwarfs": Config("V_seismic_dwarfs", identity_extra=("chaplin14",)),
        "V_galah_giants": Config("V_galah_giants", galah_giants=True),
        "V_no_conatal": Config("V_no_conatal", conatal=False),
    }
    if args.quick:
        variants = {}
    bridge = bridge_rows("A_main", probA, outA) + bridge_rows(
        "ARB_A", *arb["A"], se=False
    )
    for tag, cfg in variants.items():
        extra = {"n_outer": min(12, args.n_outer)} if not cfg.conatal else {}
        prob, out = do(
            cfg, warm=warm_from(probA, outA), covariance=cfg.conatal, **extra
        )
        map_table(prob, out).to_csv(docs / f"maps_{tag}.csv", index=False)
        prob["tiers"].to_csv(docs / f"tiers_{tag}.csv", index=False)
        record_fit(tag, prob, out)
        bridge += bridge_rows(tag, prob, out)
        bench_rows += [{"gauge": "A", **r} for r in bench_map_rows(tag, prob, out)]
        if cfg.wd_fit:
            noise_table(prob, out).to_csv(docs / f"noise_{tag}.csv", index=False)
        pd.DataFrame(bridge).to_csv(docs / "bridge_variants.csv", index=False)
    pd.DataFrame(bench_rows).to_csv(docs / "benchmark_maps.csv", index=False)
    pd.DataFrame(bridge).to_csv(docs / "bridge_variants.csv", index=False)
    # ---- per-star consensus (both gauges; cluster nodes are not stars) ---------
    stars = pd.DataFrame({"source_id": data["all_ids"]})
    for tag, ptag in (("A", "A"), ("C", "C"), ("C_fromA", "C")):
        prob = results[ptag][0]
        sol = sols[tag]
        sid = stars.source_id.to_numpy()
        pos = np.minimum(
            np.searchsorted(prob["star_ids"], sid), len(prob["star_ids"]) - 1
        )
        hit = prob["star_ids"][pos] == sid
        cols = ("t", "sd", "t_rec", "sd_rec") if tag != "C_fromA" else ("t",)
        for col in cols:
            stars[f"{col}_{tag}"] = np.where(hit, sol[col][pos], np.nan).astype("f4")
        if tag != "C_fromA":
            stars[f"limit_{tag}"] = np.where(hit, sol["limit"][pos], 0).astype("i1")
    fdf = probA["fit_df"]
    fdf = fdf[fdf.source_id > 0]
    agg = fdf.groupby("source_id").agg(n_catalogues=("source", "size"),
                                       catalogues=("source", lambda s: ";".join(sorted(s))),
                                       n_families=("group", "nunique"))  # fmt: skip
    stars = stars.join(agg, on="source_id").join(
        _stage_of(fdf).rename("stage"), on="source_id"
    )
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
    ok = np.isfinite(stars.t_A) & np.isfinite(stars.t_C_fromA) & np.isfinite(stars.t_C)
    summary["C_minus_C_fromA_median_abs"] = float(
        np.median(np.abs(stars.t_C - stars.t_C_fromA)[ok])
    )
    del stars
    # ---- clusters, members and wide pairs --------------------------------------
    raw = pd.read_parquet(args.records, columns=["source", "source_id", "lage", "good"])
    raw = raw[raw.good & raw.source.astype(str).isin(FIT_SOURCES)]
    for g in ("A", "C"):
        prob, out = results[g]
        ctab, mem, pairs = conatal_tables(
            prob, out, sols[g], data["clusters"], raw, sol_ref=sols[f"ARB_{g}"]
        )
        ctab.to_csv(docs / f"clusters_{g}.csv", index=False)
        mem.to_parquet(work / f"cluster_members_{g}.parquet", index=False)
        if len(pairs):
            pairs.to_parquet(work / f"pairs_{g}.parquet", index=False)
        summary[f"conatal_{g}"] = conatal_summary(mem, pairs)
    del raw
    # ---- record residuals before/after (consensus without the record's group) --
    for tag, (prob, out) in results.items():
        rec = prob["rec"]
        key = prob["fit_df"].group.astype("category").cat.codes.to_numpy()
        t_o, v_o = leave_key_out(rec, out["res"], len(prob["units"]), len(EFFECT_FAMILIES), key,
                                 sols[tag]["t"])  # fmt: skip
        raw_r, mapped, z = record_residuals(rec, out["res"], len(prob["units"]),
                                            len(EFFECT_FAMILIES), t_o, v_o)  # fmt: skip
        df = prob["fit_df"][
            ["source", "source_id", "cls", "key", "lage", "feh", "censor"]
        ].copy()
        df["t_other"], df["sd_other"] = t_o.astype("f4"), np.sqrt(v_o).astype("f4")
        df["raw"], df["mapped"], df["z"] = (
            raw_r.astype("f4"),
            mapped.astype("f4"),
            z.astype("f4"),
        )
        df[np.isfinite(df.t_other)].to_parquet(
            work / f"residuals_{tag}.parquet", index=False
        )
    # ---- validation-only and non-representative records vs the consensus ------
    fit_all = data["fit"]
    galah_giant = (fit_all.source == "galah_dr4") & fit_all.cls.isin(GIANT)
    val = pd.concat([fit_all[~fit_all.representative & ~galah_giant], data["val"],
                     fit_all[galah_giant].assign(source="galah_dr4_giant")], ignore_index=True)  # fmt: skip
    val_rows = []
    for s, g in val.groupby("source"):
        val_rows += predict_heldout(
            s,
            g,
            probA["star_ids"],
            {"t": sols["A"]["t_rec"], "sd": sols["A"]["sd_rec"]},
        )
    pd.DataFrame(val_rows).to_csv(docs / "validation_records.csv", index=False)
    # ---- leave-one-catalogue-out (with the bridge zero point of each fold) ----
    loo = []
    warmA = warm_from(probA, outA)
    tiersA = probA["tiers"].set_index("key").tier
    for fname, srcs in ({} if args.quick else LOO_FAMILIES).items():
        prob, out = do(
            Config(f"LOO_{fname}", drop=tuple(srcs)), warm=warmA, covariance=False
        )
        sol = solve_all(prob, out)
        held = fit_all[fit_all.source.isin(srcs) & fit_all.representative]
        for s, g in held.groupby("source"):
            th = int(tiersA.get(map_key(s, "SGB", Config("A")), LINEAR))
            for row in predict_heldout(
                s, g, prob["star_ids"], sol, tier_hint=max(th, OFFSET)
            ):
                loo.append({"family_held_out": fname, **row})
        bridge += bridge_rows(f"LOO_{fname}", prob, out, se=False)
        summary[f"fit_LOO_{fname}"] = {"stars": out["n_fit_stars"], "seconds": round(out["seconds"], 1),
                                       "outer_iterations": len(out["res"].history)}  # fmt: skip
        pd.DataFrame(loo).to_csv(docs / "loo_predictive.csv", index=False)
        pd.DataFrame(bridge).to_csv(docs / "bridge_variants.csv", index=False)
    br = pd.DataFrame(bridge)
    zp = br[(br.feh == 0.0) & np.isclose(br.t, ZP_T)].copy()
    main_zp = float(zp.xiang_minus_latent_dex[zp.variant == "A_main"].iloc[0])
    zp["shift_vs_main"] = zp.xiang_minus_latent_dex - main_zp
    zp.to_csv(docs / "bridge_summary.csv", index=False)
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
    import pyarrow.parquet as pq

    cols = ["source_id", "t_A", "sd_A", "t_rec_A", "sd_rec_A", "t_C", "stage", "feh",
            "cluster_id", "conatal_resp", "n_catalogues"]  # fmt: skip
    if "t_C_fromA" in pq.read_schema(work / "consensus_stars.parquet").names:
        cols.append("t_C_fromA")
    stars = pd.read_parquet(work / "consensus_stars.parquet", columns=cols)
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
    fig.suptitle("C0b maps (gauge A, APOKASC-3): catalogue vs consensus of the other "
                 "input groups; lines = fitted map per class at the class median [Fe/H] (±1σ)",
                 x=0.01, ha="left", fontsize=9)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(docs / f"{PREFIX}_maps.png", dpi=90)
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
    fig.savefig(docs / f"{PREFIX}_residuals.png", dpi=90)
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
    ok = (ctab.n_members >= 3) | (ctab.young_benchmark.fillna("") != "")
    c = ctab[ok]
    ax.errorbar(c.hr23_logage - 9, c.T_fit, yerr=c.T_sd, fmt="o", ms=2.5, color=SERIES[0],
                elinewidth=0.6, alpha=0.7, label="fitted cluster latent (consensus, gauge A)")  # fmt: skip
    ax.scatter(c.hr23_logage - 9, c.raw_median, s=6, color=SERIES[1], alpha=0.6,
               label="before: median raw member age (all catalogues)")  # fmt: skip
    lit = c[c.name.isin(LITERATURE_GYR) | c.name.isin(YOUNG_CLUSTERS)]
    for _, rw in lit.iterrows():
        ax.annotate(rw["name"].replace("_", " "), (rw.hr23_logage - 9, rw.T_fit), fontsize=6,
                    color=INK, xytext=(3, 3), textcoords="offset points")  # fmt: skip
    ax.plot([-2.0, 1.2], [-2.0, 1.2], color=MUTED, lw=0.8, ls=":")
    ax.set_xlabel("Hunt & Reffert 2023 log10(age/Gyr) (comparison, not truth)")
    ax.set_ylabel("cluster log10(age/Gyr)")
    ax.set_title(
        "Cluster latents vs HR23 (≥ 3 members, and the young benchmarks)", loc="left"
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
            ("cluster members", zz.kind == "cluster"),
            ("wide-pair members (EB21)", zz.kind == "binary"),
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
    fig.savefig(docs / f"{PREFIX}_clusters.png", dpi=90)
    plt.close(fig)
    # (4) scale difference C - A vs age and [Fe/H] per stage (both C solutions)
    variants_c = [
        ("t_C", "Xiang gauge fitted from scratch")
    ]  # and from the gauge-A solution
    if "t_C_fromA" in stars:
        variants_c.append(
            ("t_C_fromA", "Xiang gauge started from the APOKASC-3 solution")
        )
    fig, axs = plt.subplots(len(variants_c), 2, figsize=(11, 3.6 * len(variants_c)),
                            squeeze=False)  # fmt: skip
    for row, (ccol, clab) in zip(axs, variants_c):
        s = stars[
            np.isfinite(stars.t_A) & np.isfinite(stars[ccol]) & (stars.n_catalogues > 0)
        ]
        stages = [
            c
            for c in ("SGB", "MSTO", "RGB", "RC", "giant", "MS")
            if (s.stage == c).sum() >= 50
        ]
        for j, c in enumerate(stages):
            g = s[s.stage == c]
            col, ls = SERIES[j % 4], ("-" if j < 4 else "--")
            for ax, x, e in (
                (row[0], g.t_A.to_numpy(), ea),
                (row[1], g.feh.to_numpy(), ef),
            ):
                b = _binned(x, (g[ccol] - g.t_A).to_numpy(), e, min_n=30)
                if len(b):
                    ax.plot(b[:, 0], b[:, 1], color=col, ls=ls, marker="o", ms=2.5,
                            label=f"{c} (n={len(g)})")  # fmt: skip
                    ax.fill_between(
                        b[:, 0], b[:, 2], b[:, 3], color=col, alpha=0.1, lw=0
                    )
        for ax in row:
            ax.axhline(0, color=MUTED, lw=0.8)
            ax.set_ylabel("t(Xiang gauge) − t(APOKASC-3) (dex)")
            ax.set_title(clab, loc="left")
        row[0].legend(fontsize=6.5, loc="center right")
    axs[-1, 0].set_xlabel("consensus log10(age/Gyr), APOKASC-3 gauge")
    axs[-1, 1].set_xlabel("[Fe/H] (dex)")
    fig.suptitle("Scale systematic: Xiang 2024-anchored minus APOKASC-3-anchored consensus "
                 "(median, 16–84 %)", x=0.01, ha="left", fontsize=9)  # fmt: skip
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(docs / f"{PREFIX}_scale_difference.png", dpi=90)
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
        fig.savefig(docs / f"{PREFIX}_loo.png", dpi=90)
        plt.close(fig)
    # (6) gauge arbitration: independent-physics offsets in both gauges, bridge zero point
    arb_path, zp_path = docs / "arbitration_offsets.csv", docs / "bridge_summary.csv"
    if arb_path.exists() and zp_path.exists():
        arb = pd.read_csv(arb_path)
        bm = (
            pd.read_csv(docs / "benchmark_maps.csv")
            if (docs / "benchmark_maps.csv").exists()
            else None
        )
        zp = pd.read_csv(zp_path)
        fams = [*BENCH_SOURCES, *WD_SOURCES]
        label = {"bench_ldb": "LDB clusters", "bench_eb": "EB clusters", "bench_gc": "GC (Hess/MSTO)",
                 "wd_qiu21": "WD+MS Qiu+21", "wd_raddi22_H_C18": "WD+MS Raddi+22 (H, C18)"}  # fmt: skip
        fig, axs = plt.subplots(
            1, 2, figsize=(13, 0.42 * max(len(fams), len(zp)) + 1.6)
        )
        ax = axs[0]
        yy = np.arange(len(fams))
        for j, (g, lab) in enumerate((("A", "gauge A (APOKASC-3), validation-only"),
                                      ("C", "gauge C (Xiang 2024), validation-only"))):  # fmt: skip
            a = arb[(arb.gauge == g) & arb.variant.str.startswith("ARB")].set_index(
                "source"
            )
            a = a.reindex(fams)
            ax.errorbar(a.offset, yy + (j - 0.5) * 0.22, xerr=a.offset_se, fmt="o", ms=4, color=SERIES[j],
                        elinewidth=1, capsize=2, label=lab)  # fmt: skip
        if bm is not None:
            m = bm[bm.variant == "main_A"].set_index("source").reindex(fams)
            ax.errorbar(m.offset, yy + 0.33, xerr=m.offset_se, fmt="s", ms=3.5, mfc="none",
                        color=SERIES[2], elinewidth=0.8, capsize=2, label="fitted offset map, main fit (gauge A)")  # fmt: skip
        nn = arb[arb.variant == "ARB_A"].set_index("source").reindex(fams).n
        ax.set_yticks(yy)
        ax.set_yticklabels(
            [
                f"{label[f]} (n={int(n) if np.isfinite(n) else 0})"
                for f, n in zip(fams, nn)
            ]
        )
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_xlabel("offset: source − fitted latent (dex), ±1σ")
        ax.set_title("Independent-physics ages against each gauge", loc="left")
        ax.legend(fontsize=6.5, loc="lower right")
        ax.invert_yaxis()
        ax = axs[1]
        zp = zp.reset_index(drop=True)
        yy = np.arange(len(zp))
        col = np.where(zp.variant.str.startswith("LOO"), SERIES[1],
                       np.where(zp.variant == "A_main", SERIES[0], SERIES[2]))  # fmt: skip
        for kind, c, lab in ((SERIES[0], SERIES[0], "main fit (gauge A)"), (SERIES[2], SERIES[2], "variant"),
                             (SERIES[1], SERIES[1], "leave one catalogue family out")):  # fmt: skip
            m = col == kind
            ax.errorbar(zp.xiang_minus_latent_dex[m], yy[m], xerr=zp.se[m], fmt="o", ms=3.5, color=c,
                        elinewidth=1, capsize=2, label=lab)  # fmt: skip
        ax.axvline(
            float(zp.xiang_minus_latent_dex[zp.variant == "A_main"].iloc[0]),
            color=SERIES[0],
            lw=0.8,
            ls=":",
        )
        ax.set_yticks(yy)
        ax.set_yticklabels(zp.variant, fontsize=6.5)
        ax.set_xlabel("Xiang 2024 − APOKASC-3 latent at 4 Gyr, [Fe/H] = 0 (dex)")
        ax.set_title("Cross-regime bridge zero point per variant and fold", loc="left")
        ax.legend(fontsize=6.5, loc="lower right")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(docs / f"{PREFIX}_arbitration.png", dpi=90)
        plt.close(fig)


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
                        "(source:class); records of an effect family share an age error "
                        "u ~ N(0, eta^2) per star (y = f(t + u)); pipelines on the same spectra "
                        "add a nested u_k ~ N(0, eta_k^2) (y = f(t + u + u_k))",
               "eta": {n: float(np.exp(v)) for n, v in zip((*EFFECT_FAMILIES, *SUB_FAMILIES), res.log_eta)},
               "effect_families": {"isochrone": sorted(ISOCHRONE), "cluster_cmd": list(CLUSTER_CATALOGUES),
                                   **{k: list(v) for k, v in SUB_FAMILIES.items()}},
               "isochrone_sources": sorted(ISOCHRONE), "caps_gyr_right_censored_within_1pc": CAPS,
               "main_sequence_rule": f"isochrone records with catalogue log g >= {MS_LOGG} excluded",
               "shared_input_groups_priority": SHARED_INPUT_GROUPS, "maps": maps, "noise_units": units}  # fmt: skip
        (docs / f"measurement_model_{tag}.json").write_text(json.dumps(out, indent=1))


def export_records(work: Path, records: Path, targets: Path, labels: Path | None = None,
                   clusters: Path | None = None) -> None:  # fmt: skip
    """Per-record table of the gauge-A fit (representative records): the input
    C1 needs to train against the C0 measurement model (records, not consensus
    values). Large; written to WORK."""
    data = prepare(records, targets, labels, clusters)
    prob = build_problem(data, Config("A_apokasc3"))
    fitA = pickle.load(open(work / "fit_A.pkl", "rb"))
    df = prob["fit_df"][["source_id", "source", "group", "key", "unit", "cls", "lage", "sig", "feh",
                         "censor"]].copy()  # fmt: skip
    df["effect"] = np.select([df.source.isin(ISOCHRONE), df.source.isin(CLUSTER_CATALOGUES)],
                             ["isochrone", "cluster_cmd"], "")  # fmt: skip
    df["subeffect"] = ""
    for k, v in SUB_FAMILIES.items():
        df.loc[df.source.isin(v), "subeffect"] = k
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
    labels = Path(args.labels) if args.labels else None
    export_records(work, Path(args.records), Path(args.targets), labels,
                   Path(args.clusters) if labels else None)  # fmt: skip
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
    ap.add_argument("--docs", default="docs/age-c0b-20261123")
    ap.add_argument("--labels", default=None,
                    help="C0b label directory (benchmark clusters, wide pairs, WD+MS, seismology)")  # fmt: skip
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
