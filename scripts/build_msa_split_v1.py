"""Freeze the ``msa-split-v1`` data-set assignment for every Gaia DR3 source.

Spec: ``docs/programme/data.md`` section 3 and ``docs/programme/datasets.yaml``.

* **Hash.** ``u = splitmix64(key64 XOR SALT64) >> 11 * 2**-53`` with
  ``SALT64`` = first 8 bytes (little-endian) of ``sha256(b"msa-split-v1")``.
  Integer keys are Gaia DR3 ``source_id``; string keys map to
  ``int.from_bytes(sha256(key)[:8], "little") | 2**63``.
* **Partition.** FIT if u < 0.75, DEV if u < 0.80, CAL if u < 0.88, else TEST.
* **Groups.** Hunt & Reffert 2023 members (``Prob >= 0.7``, key
  ``grp:hr23:<Name>``) and El-Badry, Rix & Heintz 2021 wide binaries
  (``R_chance_align < 0.1``, key ``grp:eb21:<min(source_id1, source_id2)>``).
  A member takes the partition of its group key. A star in several clusters
  takes the most probable one (then the smaller key). Cluster before binary:
  a binary with a cluster member inherits that cluster's key, so the companion
  follows the cluster and the pair is not split. A star in several binary
  groups takes an inherited cluster first (most probable, then smaller key),
  then the smaller binary key. Only pairs whose two stars sit in different
  clusters (or in different binary groups) can end in different partitions;
  they are counted in the manifest.
* **Reserved.** Gaia FGK benchmarks v3, Vasiliev & Baumgardt 2021 globular
  members (``memberprob >= 0.5``), members (``Prob >= 0.7``) of the HR23
  clusters in ``cluster_priors.csv``, Kepler LEGACY dwarfs and the white-dwarf
  pairs/ages get role ``QA_RESERVED``: never FIT, DEV, CAL or REC.
  ``wd_ages_published.parquet`` stores companion ids as float64, which is not
  exact for 19-digit ids: a companion is resolved exactly through the EB21/WDMS
  pair that holds its white dwarf; otherwise every Gaia source whose id rounds
  to that float64 is reserved (a few stars per companion, always including it).
  The GBS v3 table stores ``GaiaDR3`` rounded through float64 as well (every id
  float-exact, median 10 trailing zero bits against 7 for Gaia ids), so its
  benchmarks are reserved by the same rounding superset. Any other id list
  that looks float-rounded stops the build.

Input: ``source_id``, ``has_xp_continuous`` and ``phot_{g,bp,rp}_mean_mag`` of
every partition of the Gaia DR3 HATS tree, read column-wise, one partition at a
time (a process pool streams the 2,016 partitions).

Output (``--output``):

* ``assignment/Norder=*/Dir=*/Npix=*.parquet``: one file per HATS partition, rows
  in the same order as the Gaia file: ``source_id`` (int64), ``partition``
  (uint8: 0 FIT, 1 DEV, 2 CAL, 3 TEST, 4 QA_RESERVED), ``group_id`` (int32 into
  ``groups.parquet``, -1 none) and ``input_class`` (uint8: 0 X = XP present,
  1 P = no XP with G, BP and RP, 2 G = G with at most one of BP/RP, 3 N = no G;
  Gaia bands only, so external photometry can later raise G to P);
* ``overrides.parquet``: every group member and reserved object, with its own
  hash partition, group, final partition and reason;
* ``groups.parquet``: group keys, the key actually hashed, unit and partition;
* ``counts.json`` / ``counts.csv``: rows per partition and input class (hash
  only and final), fractions, verification;
* ``manifest.json``: spec, salt, code commit, input checksums, counts and the
  verification result (``status: msa_split_v1_complete`` only when every check
  passes).

A rerun into the same ``--output`` resumes: partitions whose stats file carries
the same override fingerprint are not recomputed. A complete freeze is never
overwritten.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = "msa-split-v1"
SALT = b"msa-split-v1"
REC_SALT = b"msa-rec-v1"
EDGES = (0.75, 0.80, 0.88)  # upper edges of FIT, DEV, CAL; TEST up to 1
NOMINAL = {"FIT": 0.75, "DEV": 0.05, "CAL": 0.08, "TEST": 0.12}
PARTITIONS = ("FIT", "DEV", "CAL", "TEST", "QA_RESERVED")
FIT, DEV, CAL, TEST, RESERVED = range(5)
CLASSES = ("X", "P", "G", "N")
CLS_X, CLS_P, CLS_G, CLS_N = range(4)
MISSING = 255
HR23_MIN_PROB = 0.7
EB21_MAX_R = 0.1
VB21_MIN_PROB = 0.5
FRACTION_TOLERANCE = 1e-3
# reserved by the float64 rounding superset (bit j of the per-float mask)
FLOAT_REASONS = ("gbs_v3", "wd_age_companion_float")
GAIA_COLUMNS = (
    "source_id",
    "has_xp_continuous",
    "phot_g_mean_mag",
    "phot_bp_mean_mag",
    "phot_rp_mean_mag",
)
TEST_VECTORS = (  # datasets.yaml hash.test_vectors
    (4185355646116022528, 0.662500593453, "FIT"),
    (4398046511104, 0.798229566688, "DEV"),
    (4295806720, 0.852028592613, "CAL"),
    (4295806753, 0.882257348434, "TEST"),
    ("grp:hr23:Melotte_22", 0.440677168076, "FIT"),
)
RESERVED_REASONS = (
    "gbs_v3",
    "vb21_gc",
    "age_benchmark_cluster",
    "kepler_legacy",
    "wdms_pair",
    "wd_age_published",
    "wd_age_companion_float",
)

KPOP = Path("/arc/projects/k-pop/catalogues")
AGE = Path("/arc/projects/astroai/msa/analysis/age_labels_20261122")


@dataclass
class Inputs:
    gaia: Path = Path("/arc/projects/hats/gaia/dr3")
    hr23_members: Path = KPOP / "OpenClusters/members.fits"
    hr23_clusters: Path = KPOP / "OpenClusters/clusters.fits"
    eb21: Path = AGE / "wide_pairs_eb21.parquet"
    gbs: Path = KPOP / "gaia_fgk_benchmarks.parquet"
    vb21_dir: Path = KPOP / "gc_gaia/catalogues"
    vb21_extra: Path = AGE / "cluster_members_extra.parquet"
    age_clusters: Path = AGE / "cluster_priors.csv"
    kepler_legacy: Path = KPOP / "kepler-legacy-dwarfs.fits"
    kepler_gaia: Path = KPOP / "kepler_dr3_good.fits"
    wdms: Path = AGE / "wdms_pairs.parquet"
    wd_ages: Path = AGE / "wd_ages_published.parquet"


# ── hash ────────────────────────────────────────────────────────────────────


def salt64(label: bytes) -> int:
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "little")


SALT64 = salt64(SALT)  # 0xb6f6e5353645106b


def key64(key: int | str) -> int:
    """Spec key -> unsigned 64-bit integer (string keys get the top bit)."""
    if isinstance(key, str):
        digest = hashlib.sha256(key.encode()).digest()[:8]
        return int.from_bytes(digest, "little") | (1 << 63)
    k = int(key)
    if not 0 <= k < 1 << 63:
        raise ValueError(f"integer key out of range: {k}")
    return k


def _as_u64(keys) -> np.ndarray:
    a = np.asarray(keys)
    if a.dtype == np.int64:
        if np.any(a < 0):
            raise ValueError("negative integer key")
        return a.view(np.uint64)
    if a.dtype == np.uint64:
        return a
    if a.dtype.kind in "iuO":
        return np.asarray([key64(int(k)) for k in a.ravel()], np.uint64).reshape(
            a.shape
        )
    raise TypeError(f"keys must be integers, not {a.dtype}")


def unit(keys, salt: int = SALT64) -> np.ndarray:
    """Vectorised salted splitmix64 finaliser: uint64 keys -> float64 in [0, 1)."""
    z = _as_u64(keys) ^ np.uint64(salt)
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * 2.0**-53


def unit_of_keys(keys) -> np.ndarray:
    """unit() of a sequence of spec keys (ints or strings)."""
    return unit(np.asarray([key64(k) for k in keys], np.uint64))


def partition_of(u) -> np.ndarray:
    return np.searchsorted(np.asarray(EDGES), np.asarray(u), side="right").astype(
        np.uint8
    )


def fit_fold(u) -> np.ndarray:
    """Cross-fitting fold 0-4 inside FIT (``floor(u / 0.15)``), -1 outside."""
    u = np.asarray(u)
    return np.where(u < EDGES[0], np.floor(u / 0.15), -1).astype(np.int8)


def in_rec_sample(keys, rate: float) -> np.ndarray:
    """Nested REC subsample: ``unit(key, salt="msa-rec-v1") < rate``."""
    return unit(keys, salt64(REC_SALT)) < rate


def check_test_vectors() -> list[dict]:
    out = []
    for key, u_spec, name in TEST_VECTORS:
        u = float(unit_of_keys([key])[0])
        got = PARTITIONS[int(partition_of(u))]
        ok = abs(u - u_spec) < 1e-12 and got == name
        out.append(dict(key=key, u=u, u_spec=u_spec, partition=got, ok=ok))
        if not ok:
            raise AssertionError(f"spec test vector failed: {out[-1]}")
    return out


def input_class(has_xp, g, bp, rp) -> np.ndarray:
    """X (XP), P (no XP; G, BP and RP), G (G, at most one of BP/RP), N (no G)."""
    ok = [np.isfinite(np.asarray(x, np.float64)) for x in (g, bp, rp)]
    cls = np.full(len(ok[0]), CLS_N, np.uint8)
    cls[ok[0]] = CLS_G
    cls[ok[0] & ok[1] & ok[2]] = CLS_P
    cls[np.asarray(has_xp, bool)] = CLS_X
    return cls


# ── groups and reserved objects ──────────────────────────────────────────────


def resolve_groups(clusters: pd.DataFrame, pairs: pd.DataFrame):
    """Assign each co-natal star one group; return (groups, members, pairs).

    ``clusters``: one row per (star, cluster) membership that passes the rule,
    columns ``source_id, name, prob`` (optional ``hr23_id, hr23_type``).
    ``pairs``: binaries that pass the rule, columns ``source_id1, source_id2``.
    Every choice is a total order on (probability, key), so the result does not
    depend on row order.
    """
    c = pd.DataFrame(
        {
            "source_id": clusters["source_id"].to_numpy(np.int64),
            "key": "grp:hr23:" + clusters["name"].astype(str).str.strip(),
            "prob": clusters["prob"].to_numpy(np.float64),
            "hr23_id": clusters.get("hr23_id", pd.Series(-1, clusters.index))
            .fillna(-1)
            .to_numpy(np.int64),
            "hr23_type": clusters.get("hr23_type", pd.Series("", clusters.index))
            .fillna("")
            .astype(str)
            .to_numpy(),
        }
    )
    c = c.sort_values(
        ["source_id", "prob", "key"], ascending=[True, False, True], kind="mergesort"
    )
    best = c.drop_duplicates("source_id")[["source_id", "key", "prob"]]
    s1 = pairs["source_id1"].to_numpy(np.int64)
    s2 = pairs["source_id2"].to_numpy(np.int64)
    pp = pd.DataFrame({"lo": np.minimum(s1, s2), "hi": np.maximum(s1, s2)})
    pp = pp[pp.lo != pp.hi].drop_duplicates().sort_values(["lo", "hi"])
    pp["key"] = "grp:eb21:" + pp.lo.astype(str)
    pp = pp.reset_index(drop=True)
    pm = pd.concat(
        [
            pp[["lo", "key"]].rename(columns={"lo": "source_id"}),
            pp[["hi", "key"]].rename(columns={"hi": "source_id"}),
        ]
    ).drop_duplicates()
    # binary groups inherit the most probable cluster of any of their members
    inh = pm.merge(
        best.rename(columns={"key": "ckey", "prob": "cprob"}), on="source_id"
    )
    inh = inh.sort_values(
        ["key", "cprob", "ckey"], ascending=[True, False, True], kind="mergesort"
    ).drop_duplicates("key")[["key", "ckey", "cprob"]]
    bgroups = pd.DataFrame({"key": pp.key.unique()}).merge(inh, on="key", how="left")
    bgroups["partition_key"] = bgroups.ckey.fillna(bgroups.key)
    # stars outside clusters take one binary group: inherited cluster first
    bm = pm[~pm.source_id.isin(best.source_id)].merge(bgroups, on="key")
    bm["inherited"] = bm.ckey.notna()
    bm = bm.sort_values(
        ["source_id", "inherited", "cprob", "partition_key", "key"],
        ascending=[True, False, False, True, True],
        kind="mergesort",
        na_position="last",
    ).drop_duplicates("source_id")
    cinfo = c.drop_duplicates("key")[["key", "hr23_id", "hr23_type"]]
    groups = pd.concat(
        [
            cinfo.assign(kind="hr23", partition_key=cinfo.key),
            bgroups[["key", "partition_key"]].assign(
                kind="eb21", hr23_id=-1, hr23_type=""
            ),
        ],
        ignore_index=True,
    )
    groups = groups.sort_values("key", kind="mergesort").reset_index(drop=True)
    groups["group_id"] = np.arange(len(groups), dtype=np.int32)
    groups["unit"] = unit_of_keys(groups.partition_key.tolist())
    groups["partition"] = partition_of(groups.unit.to_numpy())
    gid = dict(zip(groups.key, groups.group_id, strict=True))
    members = pd.concat(
        [best[["source_id", "key"]], bm[["source_id", "key"]]], ignore_index=True
    )
    members["group_id"] = members.key.map(gid).astype(np.int32)
    members = members.sort_values("source_id", kind="mergesort").reset_index(drop=True)
    n = np.bincount(members.group_id.to_numpy(), minlength=len(groups))
    groups["n_members"] = n.astype(np.int32)
    npairs = pp.key.value_counts()
    groups["n_pairs"] = groups.key.map(npairs).fillna(0).astype(np.int32)
    cols = ["group_id", "key", "kind", "partition_key", "unit", "partition"]
    cols += ["n_members", "n_pairs", "hr23_id", "hr23_type"]
    return groups[cols], members[["source_id", "group_id"]], pp


def build_overrides(
    members: pd.DataFrame, groups: pd.DataFrame, reserved: dict[str, np.ndarray]
) -> pd.DataFrame:
    """One row per group member or reserved object (sorted by source_id)."""
    parts = [members.source_id.to_numpy(np.int64)]
    parts += [np.asarray(v, np.int64) for v in reserved.values()]
    ids = np.unique(np.concatenate(parts))
    if np.any(ids <= 0):
        raise ValueError("non-positive source_id in groups or reserved lists")
    group = np.full(len(ids), -1, np.int32)
    pos = np.searchsorted(ids, members.source_id.to_numpy(np.int64))
    group[pos] = members.group_id.to_numpy(np.int32)
    bits = np.zeros(len(ids), np.int64)
    for j, name in enumerate(RESERVED_REASONS):
        if name in reserved:
            bits[np.isin(ids, reserved[name])] |= 1 << j
    unknown = set(reserved) - set(RESERVED_REASONS)
    if unknown:
        raise ValueError(f"unknown reserved reasons: {sorted(unknown)}")
    labels = {
        b: ";".join(n for j, n in enumerate(RESERVED_REASONS) if b >> j & 1)
        for b in np.unique(bits)
    }
    base = partition_of(unit(ids))
    gpart = groups.partition.to_numpy(np.uint8)
    hashed = np.where(group >= 0, gpart[np.maximum(group, 0)], base).astype(np.uint8)
    is_res = bits != 0
    return pd.DataFrame(
        {
            "source_id": ids,
            "group_id": group,
            "base_partition": base,
            "hash_partition": hashed,
            "partition": np.where(is_res, RESERVED, hashed).astype(np.uint8),
            "reserved": is_res,
            "reserved_reason": pd.Series(bits).map(labels).to_numpy(),
        }
    )


def _read_fits_columns(path: Path, names) -> dict[str, np.ndarray]:
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        have = {c.upper(): c for c in data.columns.names}
        missing = [c for c in names if c.upper() not in have]
        if missing:
            raise SystemExit(f"{path}: columns {missing} missing")
        return {c: np.asarray(data[have[c.upper()]]) for c in names}


def _int_ids(values, what: str) -> np.ndarray:
    a = np.asarray(values)
    if a.dtype.kind == "f":
        raise SystemExit(f"{what}: source ids stored as float lose precision")
    return a.astype(np.int64)


def float_exact_fraction(ids) -> float:
    """Share of ids equal to their float64 rounding (~0.2-0.5 for Gaia ids)."""
    a = np.asarray(ids, np.int64)
    if not len(a):
        return float("nan")
    return float(np.mean(a.astype(np.float64).astype(np.int64) == a))


def check_not_float_rounded(ids, what: str, min_n: int = 50) -> float:
    """Fail closed on an id list that was rounded through float64 upstream."""
    f = float_exact_fraction(ids)
    if len(ids) >= min_n and f == 1.0:
        raise SystemExit(
            f"{what}: every id is float64-exact; ids were rounded upstream"
        )
    return f


def load_hr23(inputs: Inputs) -> tuple[pd.DataFrame, dict]:
    m = _read_fits_columns(inputs.hr23_members, ["GaiaDR3", "ID", "Prob", "Name"])
    cl = _read_fits_columns(inputs.hr23_clusters, ["ID", "Name", "Type"])
    names = pd.Series(np.char.strip(m["Name"].astype(str)))
    t = pd.DataFrame(
        {
            "source_id": _int_ids(m["GaiaDR3"], "HR23 members"),
            "name": names,
            "prob": m["Prob"].astype(np.float64),
            "hr23_id": m["ID"].astype(np.int64),
        }
    )
    ctab = pd.DataFrame(
        {
            "hr23_id": cl["ID"].astype(np.int64),
            "cname": np.char.strip(cl["Name"].astype(str)),
            "hr23_type": np.char.strip(cl["Type"].astype(str)),
        }
    )
    t = t.merge(ctab, on="hr23_id", how="left")
    mismatch = int((t.cname.notna() & (t.cname != t.name)).sum())
    keep = np.isfinite(t.prob) & (t.prob >= HR23_MIN_PROB) & (t.source_id > 0)
    info = dict(
        rows=len(t),
        kept_prob_ge_0p7=int(keep.sum()),
        clusters_kept=int(t.name[keep].nunique()),
        clusters_in_table=len(ctab),
        name_mismatch_rows=mismatch,
    )
    return t[keep].drop(columns="cname").reset_index(drop=True), info


def load_eb21(inputs: Inputs) -> tuple[pd.DataFrame, dict]:
    p = pd.read_parquet(
        inputs.eb21, columns=["source_id1", "source_id2", "R_chance_align"]
    )
    keep = p.R_chance_align.to_numpy(np.float64) < EB21_MAX_R
    out = pd.DataFrame(
        {
            "source_id1": _int_ids(p.source_id1, "EB21")[keep],
            "source_id2": _int_ids(p.source_id2, "EB21")[keep],
        }
    )
    return out, dict(rows=len(p), kept_r_lt_0p1=int(keep.sum()))


def load_vb21(inputs: Inputs) -> tuple[np.ndarray, dict]:
    ids, files = [], sorted(inputs.vb21_dir.glob("*.txt"))
    if not files:
        raise SystemExit(f"no VB21 catalogues in {inputs.vb21_dir}")
    for path in files:
        with path.open() as fh:
            names = fh.readline().lstrip("#").split()
        t = pd.read_csv(
            path,
            sep=r"\s+",
            comment="#",
            names=names,
            usecols=["source_id", "memberprob"],
            dtype={"source_id": np.int64},
        )
        ids.append(t.source_id.to_numpy()[t.memberprob.to_numpy() >= VB21_MIN_PROB])
    x = pd.read_parquet(inputs.vb21_extra, columns=["source_id", "prob"])
    extra = _int_ids(x.source_id, "VB21 extra")[x.prob.to_numpy() >= VB21_MIN_PROB]
    out = np.unique(np.concatenate(ids + [extra]))
    return out, dict(
        gc_files=len(files),
        members_catalogues=int(sum(len(i) for i in ids)),
        members_extra=len(extra),
        unique=len(out),
    )


def load_reserved(inputs: Inputs, hr23: pd.DataFrame, eb21: pd.DataFrame):
    """(reserved ids per reason, how each list resolved, sorted float64 ids of
    the white-dwarf companions that did not resolve exactly)."""
    info, res = {}, {}
    g = pd.read_parquet(inputs.gbs, columns=["GaiaDR3"])
    gid = np.unique(_int_ids(pd.to_numeric(g.GaiaDR3, errors="coerce").dropna(), "GBS"))
    # GaiaDR3 went through float64 upstream: reserve its rounding superset
    gbs_floats = np.unique(gid.astype(np.float64))
    res["gbs_v3"] = np.zeros(0, np.int64)
    info["gbs_v3"] = dict(
        rows=len(g),
        unique=len(gid),
        float_exact_fraction=float_exact_fraction(gid),
        rule="float64 rounding superset",
        float_ids=len(gbs_floats),
    )
    res["vb21_gc"], info["vb21_gc"] = load_vb21(inputs)
    info["vb21_gc"]["float_exact_fraction"] = check_not_float_rounded(
        res["vb21_gc"], "VB21"
    )
    pr = pd.read_csv(inputs.age_clusters)
    want = sorted(set(pr.hr23_name.dropna().astype(str).str.strip()) - {""})
    sel = hr23.name.isin(want)
    res["age_benchmark_cluster"] = np.unique(hr23.source_id[sel].to_numpy(np.int64))
    info["age_benchmark_cluster"] = dict(
        clusters_listed=int(pr.cluster.nunique()),
        hr23_names=len(want),
        hr23_names_with_members=int(hr23.name[sel].nunique()),
        without_hr23_name=sorted(set(pr.cluster[pr.hr23_name.isna()].astype(str))),
        unique=len(res["age_benchmark_cluster"]),
    )
    leg = _read_fits_columns(inputs.kepler_legacy, ["KIC"])["KIC"].astype(np.int64)
    k = _read_fits_columns(
        inputs.kepler_gaia, ["kepid", "source_id", "kepler_gaia_ang_dist"]
    )
    kt = pd.DataFrame(
        {
            "kepid": k["kepid"].astype(np.int64),
            "source_id": _int_ids(k["source_id"], "kepler_dr3_good"),
            "d": k["kepler_gaia_ang_dist"].astype(np.float64),
        }
    ).sort_values(["kepid", "d", "source_id"], kind="mergesort")
    kt = kt.drop_duplicates("kepid")
    kic = np.unique(leg)
    hit = kt[kt.kepid.isin(kic)]
    res["kepler_legacy"] = np.unique(hit.source_id.to_numpy(np.int64))
    check_not_float_rounded(kt.source_id, "kepler_dr3_good")
    info["kepler_legacy"] = dict(
        kic=len(kic),
        resolved=len(hit),
        unresolved_kic=sorted(int(x) for x in set(kic) - set(hit.kepid)),
    )
    w = pd.read_parquet(inputs.wdms, columns=["wd_source_id", "ms_source_id"])
    res["wdms_pair"] = np.unique(
        np.concatenate(
            [_int_ids(w.wd_source_id, "WDMS"), _int_ids(w.ms_source_id, "WDMS")]
        )
    )
    info["wdms_pair"] = dict(
        pairs=len(w),
        unique=len(res["wdms_pair"]),
        float_exact_fraction=check_not_float_rounded(res["wdms_pair"], "WDMS"),
    )
    a = pd.read_parquet(inputs.wd_ages, columns=["source_id", "companion_source_id"])
    wd = _int_ids(a.source_id, "WD ages")
    check_not_float_rounded(np.unique(wd), "WD ages source_id")
    # companion ids are stored as float64 (not exact for 19-digit ids): resolve
    # them exactly through the pair lists that contain the white dwarf
    links = pd.concat(
        [
            pd.DataFrame({"a": eb21.source_id1, "b": eb21.source_id2}),
            pd.DataFrame({"a": eb21.source_id2, "b": eb21.source_id1}),
            pd.DataFrame(
                {
                    "a": w.wd_source_id.astype(np.int64),
                    "b": w.ms_source_id.astype(np.int64),
                }
            ),
            pd.DataFrame(
                {
                    "a": w.ms_source_id.astype(np.int64),
                    "b": w.wd_source_id.astype(np.int64),
                }
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    comp = a.companion_source_id.to_numpy(np.float64)
    want = pd.DataFrame({"a": wd, "c": comp})[np.isfinite(comp)].drop_duplicates()
    cand = want.merge(links, on="a")
    cand = cand[cand.b.to_numpy(np.float64) == cand.c.to_numpy()]
    cand = cand.drop_duplicates(["a", "c", "b"])
    nres = cand.groupby(["a", "c"]).b.nunique()
    unique = nres[nres == 1].index
    resolved = np.zeros(0, np.int64)
    if len(unique):
        resolved = cand.set_index(["a", "c"]).loc[unique, "b"].to_numpy(np.int64)
    res["wd_age_published"] = np.unique(np.concatenate([wd, resolved]))
    done = pd.MultiIndex.from_frame(want[["a", "c"]]).isin(unique)
    floats = np.unique(want.c.to_numpy()[~done])
    info["wd_age_published"] = dict(
        rows=len(a),
        white_dwarfs=len(np.unique(wd)),
        companions=len(want),
        companions_resolved=len(unique),
        companions_unresolved=len(want) - len(unique),
        unresolved_float_ids=len(floats),
        unique=len(res["wd_age_published"]),
    )
    ids = np.r_[gbs_floats, floats]
    bits = np.r_[
        np.full(len(gbs_floats), 1, np.uint8), np.full(len(floats), 2, np.uint8)
    ]
    order = np.argsort(ids, kind="stable")
    ids, bits = ids[order], bits[order]
    uniq, first = np.unique(ids, return_index=True)
    mask = np.bitwise_or.reduceat(bits, first) if len(ids) else bits
    return res, info, uniq, mask.astype(np.uint8)


# ── scan of the Gaia HATS partitions ─────────────────────────────────────────

_OVER_IDS = np.zeros(0, np.int64)
_OVER_PART = np.zeros(0, np.uint8)
_OVER_GROUP = np.zeros(0, np.int32)
_FLOAT_IDS = np.zeros(0, np.float64)
_FLOAT_BITS = np.zeros(0, np.uint8)
_FINGERPRINT = ""


def init_worker(ids, part, group, floats, float_bits, fingerprint) -> None:
    """Pool initializer: the override arrays every partition is joined against."""
    global _OVER_IDS, _OVER_PART, _OVER_GROUP, _FLOAT_IDS, _FLOAT_BITS, _FINGERPRINT
    _OVER_IDS, _OVER_PART, _OVER_GROUP = ids, part, group
    _FLOAT_IDS, _FLOAT_BITS, _FINGERPRINT = floats, float_bits, fingerprint


def assign(source_id: np.ndarray):
    """(hash partition, final partition, group id, override positions) of rows."""
    base = partition_of(unit(source_id))
    part = base.copy()
    group = np.full(len(source_id), -1, np.int32)
    hit_pos = np.zeros(0, np.int64)
    hit_row = np.zeros(0, np.int64)
    if len(_OVER_IDS):
        pos = np.searchsorted(_OVER_IDS, source_id)
        pos = np.minimum(pos, len(_OVER_IDS) - 1)
        hit = _OVER_IDS[pos] == source_id
        hit_row = np.flatnonzero(hit)
        hit_pos = pos[hit_row]
        part[hit_row] = _OVER_PART[hit_pos]
        group[hit_row] = _OVER_GROUP[hit_pos]
    float_row = np.zeros(0, np.int64)
    float_bits = np.zeros(0, np.uint8)
    if len(_FLOAT_IDS):
        f = source_id.astype(np.float64)
        pos = np.minimum(np.searchsorted(_FLOAT_IDS, f), len(_FLOAT_IDS) - 1)
        float_row = np.flatnonzero(_FLOAT_IDS[pos] == f)
        float_bits = _FLOAT_BITS[pos[float_row]]
        part[float_row] = RESERVED
    return base, part, group, hit_pos, hit_row, float_row, float_bits


def _read_partition(path: Path):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=list(GAIA_COLUMNS))
    sid = t["source_id"]
    if sid.null_count:
        raise ValueError(f"{path}: null source_id")
    xp = pc.fill_null(t["has_xp_continuous"], False).to_numpy(zero_copy_only=False)
    mags = [
        pc.fill_null(t[c].cast("float64"), np.nan).to_numpy(zero_copy_only=False)
        for c in GAIA_COLUMNS[2:]
    ]
    return sid.to_numpy().astype(np.int64, copy=False), input_class(xp, *mags)


def process_partition(task):
    order, npix, src, dst, stats = task
    import pyarrow as pa
    import pyarrow.parquet as pq

    if stats.exists() and dst.exists():
        with np.load(stats) as z:
            if str(z["fingerprint"]) == _FINGERPRINT:
                return {k: z[k] for k in z.files}
    sid, cls = _read_partition(src)
    base, part, group, hit_pos, hit_row, float_row, float_bits = assign(sid)
    table = pa.table(
        {
            "source_id": pa.array(sid, pa.int64()),
            "partition": pa.array(part, pa.uint8()),
            "group_id": pa.array(group, pa.int32()),
            "input_class": pa.array(cls, pa.uint8()),
        }
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    pq.write_table(
        table,
        tmp,
        compression="zstd",
        use_dictionary=["partition", "group_id", "input_class"],
        column_encoding={"source_id": "DELTA_BINARY_PACKED"},
    )
    os.replace(tmp, dst)
    out = dict(
        order=np.int64(order),
        npix=np.int64(npix),
        rows=np.int64(len(sid)),
        duplicates=np.int64(len(sid) - len(np.unique(sid))),
        base=np.bincount(base * 4 + cls, minlength=16).astype(np.int64),
        final=np.bincount(part * 4 + cls, minlength=20).astype(np.int64),
        hit_pos=hit_pos,
        hit_part=part[hit_row],
        hit_cls=cls[hit_row],
        float_sid=sid[float_row],
        float_cls=cls[float_row],
        float_bits=float_bits,
        fingerprint=np.str_(_FINGERPRINT),
    )
    stats.parent.mkdir(parents=True, exist_ok=True)
    tmp = stats.with_name(stats.name + ".tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, stats)
    return out


def hats_partitions(root: Path) -> list[tuple[int, int, Path]]:
    info = pd.read_csv(root / "partition_info.csv")
    out = []
    for o, p in zip(info.Norder.astype(int), info.Npix.astype(int), strict=True):
        d = (p // 10000) * 10000
        out.append((o, p, Path(f"Norder={o}/Dir={d}/Npix={p}.parquet")))
    return out


def hats_properties(root: Path) -> dict[str, str]:
    props = {}
    for line in (root / "properties").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            props[k.strip()] = v.strip()
    return props


def add_float_reserved(over: pd.DataFrame, sid, cls, bits, floats, float_bits):
    """Reserve the Gaia sources whose id rounds to a float64-stored id.

    The workers already wrote them as QA_RESERVED; this adds them to the
    override list (or adds the reason to their existing row).
    """
    order = np.argsort(sid, kind="stable")
    sid = np.asarray(sid, np.int64)[order]
    cls = np.asarray(cls, np.uint8)[order]
    bits = np.asarray(bits, np.uint8)[order]
    info = {}
    for j, name in enumerate(FLOAT_REASONS):
        fl = floats[(float_bits >> j) & 1 == 1]
        s = sid[(bits >> j) & 1 == 1]
        hits = np.unique(s.astype(np.float64), return_counts=True)[1]
        info[name] = dict(
            float_ids=len(fl),
            float_ids_with_gaia_sources=len(hits),
            float_ids_without_gaia_source=len(fl) - len(hits),
            gaia_sources_reserved=len(s),
            sources_per_float_max=int(hits.max()) if len(hits) else 0,
            sources_per_float_median=float(np.median(hits)) if len(hits) else 0.0,
        )
    if not len(sid):
        return over, info
    ids = over.source_id.to_numpy()
    pos = np.minimum(np.searchsorted(ids, sid), max(len(ids) - 1, 0))
    known = (ids[pos] == sid) if len(ids) else np.zeros(len(sid), bool)

    def reasons(old: str, b: int) -> str:
        have = set(filter(None, old.split(";"))) | {
            n for j, n in enumerate(FLOAT_REASONS) if b >> j & 1
        }
        return ";".join(n for n in RESERVED_REASONS if n in have)

    over = over.copy()
    j = over.index[pos[known]]
    old = over.loc[j, "reserved_reason"].fillna("").tolist()
    over.loc[j, "reserved_reason"] = [
        reasons(o, int(b)) for o, b in zip(old, bits[known], strict=True)
    ]
    over.loc[j, "reserved"] = True
    over.loc[j, "partition"] = np.uint8(RESERVED)
    new_ids = sid[~known]
    b = partition_of(unit(new_ids))
    new = pd.DataFrame(
        {
            "source_id": new_ids,
            "group_id": np.full(len(new_ids), -1, np.int32),
            "base_partition": b,
            "hash_partition": b,
            "partition": np.full(len(new_ids), RESERVED, np.uint8),
            "reserved": True,
            "reserved_reason": [reasons("", int(x)) for x in bits[~known]],
            "in_gaia_dr3": True,
            "written_partition": np.full(len(new_ids), RESERVED, np.uint8),
            "input_class": cls[~known],
        }
    )
    over = pd.concat([over, new], ignore_index=True)
    over = over.sort_values("source_id", kind="mergesort").reset_index(drop=True)
    info.update(already_listed=int(known.sum()), added=len(new_ids))
    return over.astype(_OVER_DTYPES), info


_OVER_DTYPES = {
    "group_id": np.int32,
    "base_partition": np.uint8,
    "hash_partition": np.uint8,
    "partition": np.uint8,
    "reserved": bool,
    "in_gaia_dr3": bool,
    "written_partition": np.uint8,
    "input_class": np.uint8,
}


# ── provenance and verification ──────────────────────────────────────────────


def sha256_file(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def describe_input(path: Path) -> dict:
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.is_file())
        h = hashlib.sha256()
        for p in files:
            h.update(p.name.encode() + b"\0" + sha256_file(p).encode() + b"\n")
        return dict(
            path=str(path), files=len(files), sha256_of_file_sha256s=h.hexdigest()
        )
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path))


def code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fractions(counts: np.ndarray) -> dict:
    """Shares of FIT/DEV/CAL/TEST among non-reserved rows and their deviations."""
    tot = counts[:4].sum()
    out = {}
    for j, name in enumerate(PARTITIONS[:4]):
        f = counts[j] / tot if tot else float("nan")
        nom = NOMINAL[name]
        out[name] = dict(
            rows=int(counts[j]),
            fraction=float(f),
            nominal=nom,
            abs_diff=float(f - nom),
            rel_diff=float((f - nom) / nom),
        )
    return out


def fractions_ok(fr: dict, tol: float = FRACTION_TOLERANCE) -> bool:
    return all(
        abs(v["abs_diff"]) < tol and abs(v["rel_diff"]) < tol for v in fr.values()
    )


def verify_groups(over: pd.DataFrame, groups: pd.DataFrame, pairs: pd.DataFrame):
    """Group and pair integrity from the partitions actually written."""
    found = over.in_gaia_dr3.to_numpy()
    res = over.reserved.to_numpy()
    gid = over.group_id.to_numpy()
    written = over.written_partition.to_numpy()
    sel = found & ~res & (gid >= 0)
    expected = groups.partition.to_numpy()[gid[sel]]
    bad = written[sel] != expected
    bad_groups = np.unique(gid[sel][bad])
    # every group: members found and not reserved share one partition
    g = pd.DataFrame({"g": gid[sel], "p": written[sel]})
    mixed = int((g.groupby("g").p.nunique() > 1).sum())
    reserved_wrong = int((found & res & (written != RESERVED)).sum())
    unreserved_wrong = int((found & ~res & (written == RESERVED)).sum())
    ids = over.source_id.to_numpy()
    lo = np.searchsorted(ids, pairs.lo.to_numpy())
    hi = np.searchsorted(ids, pairs.hi.to_numpy())
    if not (np.array_equal(ids[lo], pairs.lo) and np.array_equal(ids[hi], pairs.hi)):
        raise AssertionError("a binary member is missing from the override list")
    both = found[lo] & found[hi] & ~res[lo] & ~res[hi]
    one_reserved = found[lo] & found[hi] & (res[lo] != res[hi])
    split = both & (written[lo] != written[hi])
    expect_split = both & (
        over.hash_partition.to_numpy()[lo] != over.hash_partition.to_numpy()[hi]
    )
    diff_group = both & (gid[lo] != gid[hi])
    return dict(
        members_checked=int(sel.sum()),
        member_partition_mismatches=int(bad.sum()),
        groups_with_mismatch=len(bad_groups),
        groups_with_mixed_partitions=mixed,
        reserved_not_reserved=reserved_wrong,
        unreserved_marked_reserved=unreserved_wrong,
        pairs=len(pairs),
        pairs_checked=int(both.sum()),
        pairs_split=int(split.sum()),
        pairs_split_expected=int(expect_split.sum()),
        pairs_in_different_groups=int(diff_group.sum()),
        pairs_with_one_reserved=int(one_reserved.sum()),
        ok=bool(
            not bad.any()
            and mixed == 0
            and reserved_wrong == 0
            and unreserved_wrong == 0
            and int(split.sum()) == int(expect_split.sum())
        ),
    )


# ── driver ───────────────────────────────────────────────────────────────────


def _save_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def _counts_table(base: np.ndarray, final: np.ndarray) -> pd.DataFrame:
    rows = []
    for j, p in enumerate(PARTITIONS):
        for k, c in enumerate(CLASSES):
            rows.append(
                dict(
                    partition=p,
                    input_class=c,
                    rows=int(final[j * 4 + k]),
                    hash_only_rows=int(base[j * 4 + k]) if j < 4 else 0,
                )
            )
    return pd.DataFrame(rows)


def run(args) -> int:
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    mpath = out / "manifest.json"
    if mpath.exists():
        old = json.loads(mpath.read_text())
        if old.get("status") == "msa_split_v1_complete" and not args.force:
            print(f"{mpath}: freeze already complete; refusing to overwrite")
            return 0
    started = time.monotonic()
    inputs = Inputs(
        **{
            k: Path(v)
            for k, v in vars(args).items()
            if k in Inputs.__dataclass_fields__ and v
        }
    )
    m = dict(
        spec=SPEC,
        status="running",
        created_utc=dt.datetime.now(dt.UTC).isoformat(),
        code_commit=code_commit(),
        script=dict(
            path="scripts/build_msa_split_v1.py", sha256=sha256_file(Path(__file__))
        ),
        salt=SALT.decode(),
        salt64=f"0x{SALT64:016x}",
        hash="u = (splitmix64(key64 ^ SALT64) >> 11) * 2**-53",
        string_key="int.from_bytes(sha256(key)[:8], 'little') | 2**63",
        edges=dict(FIT=EDGES[0], DEV=EDGES[1], CAL=EDGES[2], TEST=1.0),
        partition_codes={str(j): p for j, p in enumerate(PARTITIONS)},
        input_class_codes={str(j): c for j, c in enumerate(CLASSES)},
        input_class_rule=(
            "X: has_xp_continuous; P: no XP and G, BP, RP all present (two colours); "
            "G: G with at most one of BP/RP; N: no G. Gaia DR3 bands only: external "
            "photometry (P1 master) can raise G to P."
        ),
        group_rules=dict(
            hr23=f"Hunt & Reffert 2023 members, Prob >= {HR23_MIN_PROB}, all cluster types, key grp:hr23:<Name>",
            eb21=f"El-Badry, Rix & Heintz 2021 pairs, R_chance_align < {EB21_MAX_R}, key grp:eb21:<min(source_id1, source_id2)>",
            precedence="cluster before binary; most probable cluster, then smaller key; a binary group "
            "inherits the most probable cluster of its members (companion follows the cluster)",
        ),
        reserved_rules=dict(
            gbs_v3="gaia_fgk_benchmarks.parquet GaiaDR3 (stored rounded through float64): every "
            "Gaia source whose id rounds to the same float64",
            vb21_gc=f"VB21 gc_gaia/catalogues/*.txt + cluster_members_extra.parquet, memberprob >= {VB21_MIN_PROB}",
            age_benchmark_cluster=f"HR23 members (Prob >= {HR23_MIN_PROB}) of the hr23_name clusters in cluster_priors.csv",
            kepler_legacy="kepler-legacy-dwarfs.fits KIC -> kepler_dr3_good.fits (nearest)",
            wdms_pair="wdms_pairs.parquet wd_source_id and ms_source_id",
            wd_age_published="wd_ages_published.parquet source_id, and companion_source_id resolved "
            "exactly through EB21/WDMS pairs (the stored float64 is not exact)",
        ),
        output=dict(
            assignment="assignment/Norder=*/Dir=*/Npix=*.parquet (source_id, partition, group_id, input_class; "
            "row order of the Gaia HATS partition)",
            overrides="overrides.parquet",
            groups="groups.parquet",
            counts=["counts.json", "counts.csv"],
            partition_stats="partition_stats.parquet",
        ),
    )
    _save_json(mpath, m)
    m["test_vectors"] = check_test_vectors()
    print("spec test vectors ok", flush=True)

    hr23, info_hr23 = load_hr23(inputs)
    eb21, info_eb21 = load_eb21(inputs)
    print(f"HR23 {info_hr23}; EB21 {info_eb21}", flush=True)
    groups, members, pairs = resolve_groups(hr23, eb21)
    reserved, info_res, floats, float_bits = load_reserved(inputs, hr23, eb21)
    print(f"reserved {info_res}", flush=True)
    over = build_overrides(members, groups, reserved)
    m["groups"] = dict(
        hr23=info_hr23,
        eb21=info_eb21,
        groups=len(groups),
        hr23_groups=int((groups.kind == "hr23").sum()),
        eb21_groups=int((groups.kind == "eb21").sum()),
        eb21_groups_inheriting_cluster=int(
            ((groups.kind == "eb21") & (groups.partition_key != groups.key)).sum()
        ),
        members=len(members),
        hr23_members=int((groups.kind.to_numpy()[members.group_id] == "hr23").sum()),
    )
    m["reserved"] = dict(
        per_reason=info_res,
        unique=int(over.reserved.sum()),
        reserved_group_members=int((over.reserved & (over.group_id >= 0)).sum()),
    )
    m["overrides"] = dict(
        rows=len(over),
        moved_by_group=int(
            ((over.group_id >= 0) & (over.hash_partition != over.base_partition)).sum()
        ),
    )
    shared = [
        over.source_id.to_numpy(np.int64),
        over.partition.to_numpy(np.uint8),
        over.group_id.to_numpy(np.int32),
        floats.astype(np.float64),
        float_bits.astype(np.uint8),
    ]
    h = hashlib.sha256()
    h.update(f"{SPEC}:{SALT64}".encode())
    for a in shared:
        h.update(a.tobytes())
    shared.append(h.hexdigest())
    m["override_fingerprint"] = shared[-1]
    m["inputs"] = {}
    for k in Inputs.__dataclass_fields__:
        if k != "gaia":
            m["inputs"][k] = describe_input(getattr(inputs, k))
    props = hats_properties(inputs.gaia)
    m["inputs"]["gaia"] = dict(
        path=str(inputs.gaia),
        hats_nrows=int(props["hats_nrows"]),
        properties_sha256=sha256_file(inputs.gaia / "properties"),
        partition_info_sha256=sha256_file(inputs.gaia / "partition_info.csv"),
        columns=list(GAIA_COLUMNS),
        hats_creation_date=props.get("hats_creation_date"),
    )
    m["status"] = "scanning_gaia"
    _save_json(mpath, m)

    plist = hats_partitions(inputs.gaia)
    if args.max_partitions:
        plist = plist[: args.max_partitions]
    tasks = []
    for o, p, rel in plist:
        src = inputs.gaia / "dataset" / rel
        if not src.exists():
            raise SystemExit(f"missing HATS partition {src}")
        dst = out / "assignment" / rel
        st = out / "work" / "stats" / rel.with_suffix(".npz")
        tasks.append((o, p, src, dst, st))
    n_over = len(over)
    found = np.zeros(n_over, np.int64)
    written = np.full(n_over, MISSING, np.uint8)
    hit_cls = np.full(n_over, MISSING, np.uint8)
    base = np.zeros(16, np.int64)
    final = np.zeros(20, np.int64)
    pstats, rows_done, t0 = [], 0, time.monotonic()
    float_sid, float_cls, float_hit_bits = [], [], []
    # spawn, not fork: the parent already runs Arrow threads (parquet reads)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=init_worker, initargs=shared) as pool:
        for i, r in enumerate(pool.imap_unordered(process_partition, tasks), 1):
            np.add.at(found, r["hit_pos"], 1)
            written[r["hit_pos"]] = r["hit_part"]
            hit_cls[r["hit_pos"]] = r["hit_cls"]
            float_sid.append(r["float_sid"])
            float_cls.append(r["float_cls"])
            float_hit_bits.append(r["float_bits"])
            base += r["base"]
            final += r["final"]
            rows_done += int(r["rows"])
            pstats.append(
                dict(
                    Norder=int(r["order"]),
                    Npix=int(r["npix"]),
                    rows=int(r["rows"]),
                    duplicates=int(r["duplicates"]),
                    **{
                        p: int(r["final"][4 * j : 4 * j + 4].sum())
                        for j, p in enumerate(PARTITIONS)
                    },
                )
            )
            if i % args.progress_every == 0 or i == len(tasks):
                el = time.monotonic() - t0
                eta = el / i * (len(tasks) - i)
                print(
                    f"{i}/{len(tasks)} partitions, {rows_done:,} rows, "
                    f"{rows_done / max(el, 1e-9):,.0f} rows/s, eta {eta / 60:.1f} min",
                    flush=True,
                )
                m.update(partitions_done=i, rows_done=rows_done, scan_elapsed_s=el)
                _save_json(mpath, m)
    scan_s = time.monotonic() - t0

    over["in_gaia_dr3"] = found > 0
    over["written_partition"] = written
    over["input_class"] = hit_cls
    over, info_float = add_float_reserved(
        over,
        np.concatenate(float_sid),
        np.concatenate(float_cls),
        np.concatenate(float_hit_bits),
        floats,
        float_bits,
    )
    m["reserved"]["float_rounding_superset"] = info_float
    m["reserved"]["unique_final"] = int(over.reserved.sum())
    over.to_parquet(out / "overrides.parquet", index=False)
    gfound = np.bincount(
        over.group_id[over.in_gaia_dr3 & (over.group_id >= 0)].to_numpy(),
        minlength=len(groups),
    )
    groups["n_in_gaia"] = gfound.astype(np.int32)
    groups.to_parquet(out / "groups.parquet", index=False)
    ps = pd.DataFrame(pstats).sort_values(["Norder", "Npix"])
    ps.to_parquet(out / "partition_stats.parquet", index=False)
    (out / "assignment").mkdir(exist_ok=True)
    (out / "assignment" / "partition_info.csv").write_text(
        (inputs.gaia / "partition_info.csv").read_text()
    )
    ct = _counts_table(base, final)
    ct.to_csv(out / "counts.csv", index=False)

    final_p = final.reshape(5, 4).sum(axis=1)
    base_p = base.reshape(4, 4).sum(axis=1)
    fr_hash = fractions(base_p)
    fr_final = fractions(final_p)
    expected_rows = int(props["hats_nrows"]) if not args.max_partitions else rows_done
    gcheck = verify_groups(over, groups, pairs)
    per_reason_found = {
        r: int(over.in_gaia_dr3[over.reserved_reason.fillna("").str.contains(r)].sum())
        for r in RESERVED_REASONS
    }
    verification = dict(
        test_vectors=all(t["ok"] for t in m["test_vectors"]),
        rows=rows_done,
        rows_expected=expected_rows,
        rows_ok=rows_done == expected_rows,
        duplicate_source_ids_within_partitions=int(ps.duplicates.sum()),
        override_ids_seen_twice=int((found > 1).sum()),
        hash_fractions_ok=fractions_ok(fr_hash, args.fraction_tolerance),
        final_fractions_ok=fractions_ok(fr_final, args.fraction_tolerance),
        fraction_tolerance=args.fraction_tolerance,
        groups=gcheck,
        overrides_not_in_gaia=int((~over.in_gaia_dr3).sum()),
        group_members_not_in_gaia=int((~over.in_gaia_dr3 & (over.group_id >= 0)).sum()),
        reserved_not_in_gaia=int((~over.in_gaia_dr3 & over.reserved).sum()),
        reserved_in_gaia_per_reason=per_reason_found,
    )
    verification["ok"] = bool(
        verification["test_vectors"]
        and verification["rows_ok"]
        and verification["duplicate_source_ids_within_partitions"] == 0
        and verification["override_ids_seen_twice"] == 0
        and verification["hash_fractions_ok"]
        and verification["final_fractions_ok"]
        and gcheck["ok"]
    )
    counts = dict(
        rows=rows_done,
        partitions=len(tasks),
        by_partition={p: int(final_p[j]) for j, p in enumerate(PARTITIONS)},
        by_input_class={
            c: int(final.reshape(5, 4)[:, k].sum()) for k, c in enumerate(CLASSES)
        },
        by_partition_and_class={
            p: {c: int(final[j * 4 + k]) for k, c in enumerate(CLASSES)}
            for j, p in enumerate(PARTITIONS)
        },
        hash_only_by_partition_and_class={
            p: {c: int(base[j * 4 + k]) for k, c in enumerate(CLASSES)}
            for j, p in enumerate(PARTITIONS[:4])
        },
        fractions_hash_only=fr_hash,
        fractions_final=fr_final,
    )
    _save_json(
        out / "counts.json", dict(spec=SPEC, counts=counts, verification=verification)
    )
    assign_files = sorted((out / "assignment").rglob("*.parquet"))
    m.update(
        counts=counts,
        verification=verification,
        outputs_written=dict(
            assignment_files=len(assign_files),
            assignment_bytes=int(sum(p.stat().st_size for p in assign_files)),
        ),
        scan_seconds=scan_s,
        elapsed_seconds=time.monotonic() - started,
        finished_utc=dt.datetime.now(dt.UTC).isoformat(),
        status="msa_split_v1_complete" if verification["ok"] else "verification_failed",
    )
    if args.max_partitions:
        m["status"] = "smoke_" + m["status"]
        m["max_partitions"] = args.max_partitions
    _save_json(mpath, m)
    print(
        json.dumps(
            dict(
                status=m["status"],
                counts=counts["by_partition"],
                verification=verification,
            ),
            default=str,
        )
    )
    return 0 if verification["ok"] else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--max-partitions", type=int, help="smoke test on the first N partitions"
    )
    p.add_argument("--progress-every", type=int, default=50)
    p.add_argument(
        "--force", action="store_true", help="rebuild over a complete freeze"
    )
    p.add_argument(
        "--fraction-tolerance",
        type=float,
        default=FRACTION_TOLERANCE,
        help="max |f - nominal| and |f / nominal - 1| per partition",
    )
    for k, f in Inputs.__dataclass_fields__.items():
        p.add_argument("--" + k.replace("_", "-"), dest=k, help=f"default {f.default}")
    return run(p.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
