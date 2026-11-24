"""Co-natal [Fe/H] checks: wide binaries, held-out binary pool, open and globular clusters.

Brought into the repo from the ARC-only scripts of the QA audit
(``/arc/projects/astroai/msa/analysis/scripts/``):

* ``make_targets.py`` -> ``read_vizier_tsv``, ``targets_from_vizier`` (El-Badry,
  Rix & Heintz 2021 MSMS pairs, Hunt & Reffert 2023 members), now keeping the
  chance-alignment probability ``R`` and the membership probability;
* ``consistency_analysis.py`` -> ``wide_binaries``, ``clusters``,
  ``open_cluster_summary``, ``globular_cluster_summary`` (same statistics),
  plus the cuts qa.md §4 asked for: ``R_chance_align < 0.1`` for pairs,
  ``Prob >= 0.7`` for open-cluster members, and every statistic split by
  "neither star labelled" to rule out label leakage;
* ``binary_heldout.py`` -> ``binary_pool_heldout``.

Every function takes predictions as arrays sorted by ``source_id`` (see
``PredTable``) and returns plain dicts for ``metrics.json``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import nmad

PAIR_MEAN_BINS = ((-0.5, 9), (-1, -0.5), (-1.5, -1), (-2, -1.5), (-2.5, -2), (-9, -2.5))
DG_BINS = ((0, 1), (1, 2), (2, 3), (3, 9))
COMPANION_TEFF_BINS = (
    (3000, 3700),
    (3700, 4200),
    (4200, 4800),
    (4800, 5500),
    (5500, 7000),
)
HARRIS_BINS = ((-1, 0), (-1.5, -1), (-2, -1.5), (-3, -2))
R_CHANCE_MAX = 0.1
OC_PROB_MIN = 0.7
GIANT_LOGG, DWARF_LOGG = 3.5, 3.8


@dataclass
class PredTable:
    """Point predictions and 1-sigma widths per star, sorted by source_id."""

    source_id: np.ndarray
    teff: np.ndarray
    logg: np.ndarray
    feh: np.ndarray
    feh_sd: np.ndarray
    G: np.ndarray
    labelled: np.ndarray

    @classmethod
    def from_arrays(cls, source_id, teff, logg, feh, feh_sd, G, labelled_ids=None):
        sid = np.asarray(source_id, np.int64)
        o = np.argsort(sid)
        lab = np.zeros(len(sid), bool)
        if labelled_ids is not None:
            lab = np.isin(sid[o], np.asarray(labelled_ids, np.int64))
        f = lambda a: np.asarray(a, np.float64)[o]  # noqa: E731
        return cls(sid[o], f(teff), f(logg), f(feh), f(feh_sd), f(G), lab)

    def look(self, ids):
        ids = np.asarray(ids, np.int64)
        if not len(self.source_id):
            return np.zeros(len(ids), np.int64), np.zeros(len(ids), bool)
        pos = np.searchsorted(self.source_id, ids).clip(max=len(self.source_id) - 1)
        return pos, self.source_id[pos] == ids


def read_vizier_tsv(path) -> pd.DataFrame:
    """VizieR TSV (header, units, dashes) as stripped strings."""
    t = pd.read_csv(path, sep="\t", comment="#", dtype=str)
    t = t.iloc[2:].apply(lambda c: c.str.strip())
    return t.reset_index(drop=True)


def norm_name(name) -> str:
    return re.sub(r"[\s_]+", "", str(name)).upper()


def targets_from_vizier(data_dir) -> dict:
    """Pairs (with R) and cluster members (with Prob) from the VizieR dumps."""
    b = read_vizier_tsv(f"{data_dir}/elbadry.tsv")
    b = b[(b.Source1 != "") & ~b.Source1.str.startswith("-") & (b.BinType == "MSMS")]
    m = read_vizier_tsv(f"{data_dir}/hr23_members.tsv")
    m = m[(m.GaiaDR3 != "") & ~m.GaiaDR3.str.startswith("-")]
    return dict(
        pair1=b.Source1.astype("i8").to_numpy(),
        pair2=b.Source2.astype("i8").to_numpy(),
        pair_r=pd.to_numeric(b.get("R"), errors="coerce").to_numpy(np.float64),
        member=m.GaiaDR3.astype("i8").to_numpy(),
        member_cluster=m.ID.astype("i4").to_numpy(),
        member_prob=pd.to_numeric(m.Prob, errors="coerce").to_numpy(np.float64),
    )


def _cell(d, z=None):
    out = dict(n=int(np.isfinite(d).sum()), median=float(np.nanmedian(d)) if len(d) else np.nan,
               nmad=nmad(d))  # fmt: skip
    if z is not None:
        zz = z[np.isfinite(z)]
        out["abs_z_lt_1"] = float(np.mean(np.abs(zz) < 1)) if len(zz) else np.nan
        out["z_nmad"] = nmad(zz)
    return out


def wide_binaries(P: PredTable, pair1, pair2, pair_r=None, keep=None,
                  r_max: float = R_CHANCE_MAX) -> dict:  # fmt: skip
    """Delta = [Fe/H](brighter) - [Fe/H](fainter) of pairs with both predicted.

    ``keep`` (bool per pair) selects pairs, e.g. those whose group is in TEST.
    """
    pair1, pair2 = np.asarray(pair1, np.int64), np.asarray(pair2, np.int64)
    sel = np.ones(len(pair1), bool) if keep is None else np.asarray(keep, bool)
    n_all = int(sel.sum())
    if pair_r is not None:
        r = np.asarray(pair_r, np.float64)
        sel &= np.isfinite(r) & (r < r_max)
    p1, o1 = P.look(pair1)
    p2, o2 = P.look(pair2)
    sel &= o1 & o2
    p1, p2 = p1[sel], p2[sel]
    bright = np.where(P.G[p1] <= P.G[p2], p1, p2)
    faint = np.where(P.G[p1] <= P.G[p2], p2, p1)
    fb, ff = P.feh[bright], P.feh[faint]
    d = fb - ff
    z = d / np.hypot(P.feh_sd[bright], P.feh_sd[faint])
    unl = ~P.labelled[bright] & ~P.labelled[faint]
    mean = 0.5 * (fb + ff)
    dG = P.G[faint] - P.G[bright]
    out = dict(
        pairs_selected=n_all, pairs_scored=int(len(d)), pairs_neither_labelled=int(unl.sum()),
        r_chance_max=r_max if pair_r is not None else None,
        all=_cell(d, z), neither_labelled=_cell(d[unl], z[unl]),
        by_pair_mean={}, by_dG={}, by_faint_teff={},
    )  # fmt: skip
    for lo, hi in PAIR_MEAN_BINS:
        m = (mean >= lo) & (mean < hi)
        out["by_pair_mean"][f"[{lo},{hi})"] = dict(
            all=_cell(d[m], z[m]), neither_labelled=_cell(d[m & unl], z[m & unl])
        )
    for lo, hi in DG_BINS:
        m = (dG >= lo) & (dG < hi)
        out["by_dG"][f"[{lo},{hi})"] = dict(
            all=_cell(d[m], z[m]), neither_labelled=_cell(d[m & unl], z[m & unl])
        )
    tf = P.teff[faint]
    for lo, hi in COMPANION_TEFF_BINS:
        m = (tf >= lo) & (tf < hi)
        out["by_faint_teff"][f"[{lo},{hi})"] = dict(
            all=_cell(d[m], z[m]), neither_labelled=_cell(d[m & unl], z[m & unl])
        )
    gi = (P.logg[bright] < GIANT_LOGG) & (P.logg[faint] > DWARF_LOGG)
    out["giant_bright_dwarf_faint"] = _cell(d[gi])
    return out


def binary_pool_heldout(P: PredTable, companion, primary, teff_companion, g_companion,
                        keep=None) -> dict:  # fmt: skip
    """Median [Fe/H](primary) - [Fe/H](companion), held-out pool pairs.

    Companions are binned by the pool's (teacher) Teff, so every model is
    scored in the same bins (binary_heldout.py).
    """
    sel = np.ones(len(companion), bool) if keep is None else np.asarray(keep, bool)
    pc, oc = P.look(companion)
    pp, op = P.look(primary)
    ok = sel & oc & op
    d = np.where(ok, P.feh[pp] - P.feh[pc], np.nan)
    teff = np.asarray(teff_companion, np.float64)
    out = dict(
        pairs=int(sel.sum()), scored=int(np.isfinite(d).sum()), by_companion_teff={}
    )
    for lo, hi in COMPANION_TEFF_BINS:
        m = (teff >= lo) & (teff < hi) & np.isfinite(d)
        out["by_companion_teff"][f"{lo}-{hi}"] = _cell(d[m])
    faint = (np.asarray(g_companion, np.float64) > 16.5) & np.isfinite(d)
    out["companion_G_gt_16p5"] = _cell(d[faint])
    out["all"] = _cell(d[np.isfinite(d)])
    return out


def clusters(P: PredTable, member, member_cluster, member_prob=None, cluster_table=None,
             harris=None, keep_cluster=None, prob_min: float = OC_PROB_MIN,
             min_members: int = 5) -> pd.DataFrame:  # fmt: skip
    """Per-cluster [Fe/H] statistics of the predicted members.

    ``keep_cluster`` (callable id -> bool) restricts clusters (e.g. by the
    partition of their group); members with ``member_prob < prob_min`` are
    dropped for open clusters (globular members are kept: HR23 does not
    assign them separate probabilities).
    """
    member = np.asarray(member, np.int64)
    cid = np.asarray(member_cluster, np.int64)
    prob = None if member_prob is None else np.asarray(member_prob, np.float64)
    pos, ok = P.look(member)
    ctype = {}
    cname = {}
    if cluster_table is not None:
        ctype = dict(zip(cluster_table["ID"], cluster_table["Type"], strict=True))
        cname = dict(zip(cluster_table["ID"], cluster_table["Name"], strict=True))
    harris = harris or {}
    rows = []
    for c in np.unique(cid[ok]):
        if keep_cluster is not None and not keep_cluster(int(c)):
            continue
        typ = ctype.get(int(c), "?")
        m = ok & (cid == c)
        if prob is not None and typ != "g":
            m &= np.isfinite(prob) & (prob >= prob_min)
        i = pos[m]
        f = P.feh[i]
        good = np.isfinite(f)
        i, f = i[good], f[good]
        if len(i) < min_members:
            continue
        giant = P.logg[i] < GIANT_LOGG
        dwarf = P.logg[i] > DWARF_LOGG
        gm = P.G[i]
        med = np.median(f)
        sd = P.feh_sd[i]
        chi2 = (
            np.sum(((f - med) / sd) ** 2) / max(len(f) - 1, 1)
            if np.all(sd > 0)
            else np.nan
        )
        rows.append(dict(
            id=int(c), name=cname.get(int(c), str(c)), type=typ, n=len(i),
            n_labelled=int(P.labelled[i].sum()), feh=float(med), nmad=nmad(f),
            chi2_dof=float(chi2),
            giant_minus_dwarf=(float(np.median(f[giant]) - np.median(f[dwarf]))
                               if giant.sum() >= 3 and dwarf.sum() >= 3 else np.nan),
            slope_per_mag=float(np.polyfit(gm, f, 1)[0]) if np.ptp(gm) > 1 else np.nan,
            bright=float(np.median(f[gm < np.median(gm)])) if len(f) > 1 else np.nan,
            faint=float(np.median(f[gm >= np.median(gm)])),
            harris=harris.get(norm_name(cname.get(int(c), "")), np.nan),
        ))  # fmt: skip
    return pd.DataFrame(rows)


def open_cluster_summary(r: pd.DataFrame, min_members: int = 20) -> dict:
    if not len(r):
        return dict(clusters=0)
    oc = r[(r.type == "o") & (r.n >= min_members)]
    if not len(oc):
        return dict(clusters=0)
    gd = oc.giant_minus_dwarf.dropna()
    return dict(
        clusters=int(len(oc)),
        members=int(oc.n.sum()),
        median_nmad=float(np.median(oc.nmad)),
        median_chi2_dof=float(np.nanmedian(oc.chi2_dof)),
        faint_minus_bright=float(np.median(oc.faint - oc.bright)),
        slope_per_mag=float(np.nanmedian(oc.slope_per_mag)),
        giant_minus_dwarf=float(np.median(gd)) if len(gd) else np.nan,
        giant_minus_dwarf_clusters=int(len(gd)),
        clusters_neither_labelled=int((oc.n_labelled == 0).sum()),
    )


def globular_cluster_summary(r: pd.DataFrame, min_members: int = 10) -> dict:
    if not len(r):
        return dict(clusters=0)
    gc = r[(r.type == "g") & (r.n >= min_members)]
    hg = gc[gc.harris.notna()]
    out = dict(clusters=int(len(gc)), with_harris=int(len(hg)), by_harris={})
    for lo, hi in HARRIS_BINS:
        m = (hg.harris >= lo) & (hg.harris < hi)
        if m.any():
            out["by_harris"][f"[{lo},{hi})"] = dict(
                n=int(m.sum()),
                pred_minus_harris=float(np.median(hg.feh[m] - hg.harris[m])),
                within_nmad=float(np.median(hg.nmad[m])),
                faint_minus_bright=float(np.median(hg.faint[m] - hg.bright[m])),
            )
    return out


def harris_table(path) -> dict:
    h = read_vizier_tsv(path)
    return {
        norm_name(n): float(v)
        for n, v in zip(h.ID, h["[Fe/H]"], strict=True)
        if v not in ("", None)
    }


def cluster_table(path) -> pd.DataFrame:
    c = read_vizier_tsv(path)
    c["ID"] = c.ID.astype(int)
    return c
