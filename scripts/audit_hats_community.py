"""Read-only audit of every HATS catalogue under a community tree (/arc/projects/hats).

Nothing is written under ``--root``: the script only lists directories, reads small
metadata files and Parquet footers, and reads sampled partitions. Every output goes
to ``--output`` (refused when it lies inside ``--root``).

Stages (each writes its JSON before the next starts, so a timeout keeps the results):

1. ``discovery``: walk the tree (no symlink is followed), find catalogue roots
   (``properties``/``hats.properties``/``partition_info.csv``) and collections.
2. ``structure``: per catalogue, properties keys/values, ``partition_info.csv`` vs the
   partition files present (missing, extra, wrong ``Dir``, nested/overlapping pixels,
   stray files), every partition footer (rows, row groups, schema, per-column
   min/max/null statistics), ``_metadata`` (rows, referenced files, stale entries) and
   ``_common_metadata`` (schema vs partitions, unit metadata), margin caches.
3. ``samples``: a stratified sample of partitions per catalogue (plus the partitions
   whose footer statistics put coordinates out of range, and named targets): pixel
   membership of every row, coordinate units, ``_healpix_29`` consistency, duplicate
   ids, nearest-neighbour distances (same position twice), column profiles (types,
   numbers stored as strings, sentinels, NaN), and a Gaia DR3-anchored check (how many
   rows lie within 1.5" of each Gaia source) with per-survey diagnostics.
4. ``fullscan``: every row's coordinates (NaN, range, pixel membership) for small
   catalogues and for the named rebuild candidates.
5. ``compare``: per-partition row counts of rebuilt trees vs their backups.
6. ``followup`` (only when asked): tree-wide numbers for issues the samples raised
   (Legacy Surveys DR11 north/south twins, PS1 single detections, Pristine duplicate
   source_id, SDSS ``id`` vs THING_ID, S-PLUS field-overlap twins, GALEX zero points,
   Euclid wide detection flags).

Usage (CANFAR, /arc mounted)::

    python -u scripts/audit_hats_community.py --root /arc/projects/hats \\
        --output /arc/projects/astroai/msa/runs/hats_survey_audit_20261124/results
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

MARKERS = {"properties", "hats.properties", "partition_info.csv"}
SKIP_DIR = re.compile(r"^(Norder=\d+|Dir=\d+|Npix=\d+.*|dataset)$")
NPIX_FILE = re.compile(r"^Npix=(\d+)(\.parquet)?$")
ALLOWED_DATASET_FILES = {"_metadata", "_common_metadata", "data_thumbnail.parquet"}
REQUIRED_KEYS = [
    "obs_collection",
    "dataproduct_type",
    "hats_nrows",
    "hats_col_ra",
    "hats_col_dec",
    "hats_order",
    "hats_version",
    "hats_builder",
    "hats_creation_date",
]
RECOMMENDED_KEYS = [
    "hats_col_healpix",
    "hats_col_healpix_order",
    "hats_npix_suffix",
    "moc_sky_fraction",
    "hats_max_rows",
    "hats_estsize",
    "obs_title",
    "hats_frame",
    "hats_cols_survey_id",
]
SENTINELS = [
    -999999500.0,
    -9999999.0,
    -99999.0,
    -9999.0,
    -999.0,
    -99.0,
    -9.99,
    99.0,
    99.99,
    999.0,
    9999.0,
    99999.0,
    -1e30,
    1e30,
]
GAIA_TREE = "gaia/dr3"
GAIA_COLS = ["source_id", "ra", "dec", "pmra", "pmdec", "phot_g_mean_mag", "bp_rp"]
MATCH_RADIUS = 1.5  # arcsec, the xmatcher default

# Per-tree hints, matched by the longest prefix of the catalogue path. Columns that a
# tree does not have are ignored.
VISTA = dict(id=["sourceid"], disc=["priorsec", "framesetid", "mergedclass"])
HINTS: dict[str, dict] = {
    "gaia/dr3": dict(id=["source_id"], disc=["duplicated_source"]),
    "gaia/dr3_xp_continuous": dict(id=["source_id"]),
    "2mass/psc": dict(id=["designation"], disc=["use_src", "dup_src", "ph_qual"]),
    "2mass/xsc": dict(id=["designation"]),
    "allwise": dict(
        id=["source_id"], disc=["nb", "na", "ext_flg"], extra=["pmra", "pmdec"]
    ),
    "catwise": dict(
        id=["source_id"], disc=["nb", "na"], extra=["PMRA", "PMDec", "MeanObsMJD"]
    ),
    "atlas": dict(id=["id"], disc=["PRIORSEC", "PRIMARY_SOURCE", "FRAMESETID"]),
    "kids": dict(id=["id"], disc=["KIDS_TILE", "THELI_NAME", "Flag"]),
    "delve": dict(id=["COADD_OBJECT_ID"], disc=["TILENAME", "FLAGS_GOLD", "EXT_MASH"]),
    "des": dict(id=["COADD_OBJECT_ID"], disc=["TILENAME", "FLAGS_GOLD", "EXT_MASH"]),
    "erosita": dict(id=["UID"]),
    "euclid/q1": dict(id=["OBJECT_ID"]),
    "euclid/dr1": dict(
        id=["object_id"],
        disc=["tileid", "spurious_flag", "det_quality_flag", "vis_det"],
        extra=["ra", "dec", "point_like_prob"],
    ),
    "galex": dict(id=["objid"], disc=["tilenum", "primgid", "type"]),
    "hsc/pdr3/forced_nb": dict(
        id=["object_id"], disc=["field", "merge_peak"], extra=["flux_psf", "mag_psf"]
    ),
    "hsc/pdr3/forced_": dict(
        id=["id"],
        disc=["parent", "deblend_nChild"],
        extra=[
            "base_PsfFlux_instFlux",
            "modelfit_CModel_instFlux",
            "base_ClassificationExtendedness_value",
        ],
    ),
    "hubble": dict(id=["MatchID"]),
    "jplus": dict(
        id=["TILE_ID", "NUMBER"],
        disc=["TILE_ID"],
        extra=["MAG_AUTO", "MAG_PSFCOR", "MAG_APER_3_0", "CLASS_STAR", "FLAGS"],
        arrays=["MAG_AUTO", "MAG_PSFCOR", "MAG_APER_3_0"],
    ),
    "jpas": dict(
        id=["TILE_ID", "NUMBER"],
        disc=["TILE_ID"],
        extra=["MAG_AUTO", "MAG_APER_COR_3_0", "MAG_APER_3_0", "CLASS_STAR", "FLAGS"],
        arrays=["MAG_AUTO", "MAG_APER_COR_3_0", "MAG_APER_3_0"],
    ),
    "legacysurvey": dict(
        id=["RELEASE", "BRICKID", "OBJID"],
        disc=["RELEASE", "PHOTSYS", "BRICKID", "TYPE"],
        extra=["REF_CAT", "REF_ID", "FLUX_R", "MASKBITS"],
    ),
    "lotss": dict(id=["Source_Name"]),
    "pristine": dict(
        id=["source_id"], extra=["RA_Pr", "Dec_Pr", "G_0", "merged_CASU_flag"]
    ),
    "ps1": dict(
        id=["objID"],
        disc=["nDetections", "qualityFlag", "objInfoFlag"],
        extra=[
            "raMeanErr",
            "decMeanErr",
            "pmra",
            "pmdec",
            "epochMean",
            "nDetections",
            "rMeanPSFMag",
        ],
    ),
    "sdss": dict(
        id=["id"],
        id2=["RUN", "RERUN", "CAMCOL", "FIELD", "ID"],
        disc=["RUN", "CAMCOL", "FIELD", "OBJC_TYPE", "RESOLVE_STATUS"],
        extra=["ID", "RUN", "RERUN", "CAMCOL", "FIELD", "THING_ID", "PSFFLUX_r"],
    ),
    "skymapper": dict(id=["object_id"], disc=["flags"], extra=["mean_epoch"]),
    "splus": dict(id=["id"], disc=["field"]),
    "uhs": VISTA,
    "vhs": VISTA,
    "vvvx": VISTA,
    "vmc": VISTA,
    "viking": VISTA,
    "video": VISTA,
    "ukidss": VISTA,
}
# Partitions whose earlier Pensus QA reported out-of-range coordinates.
TARGETS = {
    "uhs/dr3": ["4/175"],
    "uhs/dr3_prera_backup": ["4/175"],
    "des/y6_gold": ["5/4351", "5/4864"],
    "des/y6_gold_prera_backup": ["5/4351", "5/4864"],
}
# Exhaustive coordinate scans beyond the small catalogues (rebuild candidates).
FULL_SCAN = [
    "uhs/dr3",
    "uhs/dr3_prera_backup",
    "des/y6_gold",
    "delve/dr3_gold",
    "ps1/otmo",
]
COMPARE = [
    ("uhs/dr3", "uhs/dr3_prera_backup"),
    ("des/y6_gold", "des/y6_gold_prera_backup"),
    ("gaia/dr3", "gaia/dr3_xp_continuous"),
]


def log(*a):
    print(f"[{dt.datetime.now(dt.UTC):%H:%M:%S}]", *a, flush=True)


def jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, bytes):
        return o.decode("utf-8", "replace")
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def clean(o):
    """Recursively replace non-finite floats so the JSON stays strict."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [clean(v) for v in o]
    return o


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(clean(json.loads(json.dumps(obj, default=jdefault))), indent=1)
        + "\n"
    )
    tmp.replace(path)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", name)


def hint(cat: str) -> dict:
    best = ""
    for k in HINTS:
        if cat.startswith(k) and len(k) > len(best):
            best = k
    return HINTS.get(best, {})


def parse_props(path: Path) -> dict | None:
    if not path.is_file():
        return None
    out = {}
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def read_partition_info(path: Path) -> tuple[list[str], list[dict]]:
    if not path.is_file():
        return [], []
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    header = list(rows[0].keys()) if rows else []
    return header, rows


# ----------------------------------------------------------------------------- discovery


def discover(root: Path, max_depth: int = 7) -> dict:
    cats, colls, links, denied, dirs = [], [], [], [], {}
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        rel = str(d.relative_to(root)) if d != root else ""
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError as exc:
            denied.append(dict(path=rel, error=repr(exc)))
            continue
        names = {e.name for e in entries}
        rec = []
        for e in entries:
            item = dict(name=e.name)
            try:
                st = e.stat(follow_symlinks=False)
                item["size"] = st.st_size
                item["mtime"] = dt.datetime.fromtimestamp(
                    st.st_mtime, dt.UTC
                ).isoformat()
                if e.is_symlink():
                    item["kind"] = "link"
                    item["target"] = os.readlink(e.path)
                    links.append(
                        dict(
                            path=(rel + "/" if rel else "") + e.name,
                            target=item["target"],
                        )
                    )
                elif e.is_dir(follow_symlinks=False):
                    item["kind"] = "dir"
                else:
                    item["kind"] = "file"
            except OSError as exc:
                item["error"] = repr(exc)
            rec.append(item)
        dirs[rel] = rec
        is_cat = bool(names & MARKERS)
        if "collection.properties" in names:
            colls.append(rel)
        if is_cat:
            cats.append(rel)
            continue  # a catalogue root is listed, not descended (its subdirs are data)
        if depth >= max_depth:
            continue
        for e in entries:
            try:
                if (
                    e.is_dir(follow_symlinks=False)
                    and not e.is_symlink()
                    and not SKIP_DIR.match(e.name)
                ):
                    stack.append((Path(e.path), depth + 1))
            except OSError:
                pass
    linked = []
    for ln in links:
        tgt = (root / ln["path"]).parent / ln["target"]
        info = dict(ln, resolved=str(Path(os.path.normpath(tgt))))
        try:
            p = Path(info["resolved"])
            info["exists"] = p.exists()
            if p.is_dir():
                info["entries"] = sorted(os.listdir(p))[:50]
                for nm in ("properties", "hats.properties", "collection.properties"):
                    if (p / nm).is_file():
                        info[nm] = parse_props(p / nm)
        except OSError as exc:
            info["error"] = repr(exc)
        linked.append(info)
    return dict(
        root=str(root),
        catalogues=sorted(cats),
        collections=sorted(colls),
        links=linked,
        denied=denied,
        dirs=dirs,
    )


# ----------------------------------------------------------------------------- structure


def list_partitions(dataset: Path) -> dict:
    """Every partition file under dataset/, its layout and stray entries."""
    parts: dict[tuple[int, int], dict] = {}
    stray, wrong_dir, empty_files, dup = [], [], [], []
    top = []
    if not dataset.is_dir():
        return dict(
            parts=parts,
            stray=stray,
            wrong_dir=wrong_dir,
            empty_files=empty_files,
            top=top,
            dup=dup,
            exists=False,
        )
    for e in sorted(os.scandir(dataset), key=lambda e: e.name):
        m = re.match(r"^Norder=(\d+)$", e.name)
        if not (m and e.is_dir()):
            top.append(
                dict(
                    name=e.name,
                    size=e.stat().st_size if e.is_file() else None,
                    dir=e.is_dir(),
                )
            )
            continue
        order = int(m.group(1))
        for d in sorted(os.scandir(e.path), key=lambda e: e.name):
            md = re.match(r"^Dir=(\d+)$", d.name)
            if not (md and d.is_dir()):
                stray.append(f"Norder={order}/{d.name}")
                continue
            dirv = int(md.group(1))
            for f in os.scandir(d.path):
                mf = NPIX_FILE.match(f.name)
                if not mf:
                    stray.append(f"Norder={order}/{d.name}/{f.name}")
                    continue
                pix = int(mf.group(1))
                if pix // 10000 * 10000 != dirv:
                    wrong_dir.append(f"Norder={order}/{d.name}/{f.name}")
                if f.is_dir():
                    files = []
                    for g in os.scandir(f.path):
                        if g.name.endswith(".parquet") and g.is_file():
                            files.append((g.path, g.stat().st_size))
                        else:
                            stray.append(f"Norder={order}/{d.name}/{f.name}/{g.name}")
                    layout = "dir"
                else:
                    if not mf.group(2):
                        stray.append(f"Norder={order}/{d.name}/{f.name}")
                        continue
                    files = [(f.path, f.stat().st_size)]
                    layout = "file"
                for p, s in files:
                    if s == 0:
                        empty_files.append(p)
                if (order, pix) in parts:
                    dup.append(f"{order}/{pix}")
                    parts[(order, pix)]["files"] += files
                else:
                    parts[(order, pix)] = dict(files=files, layout=layout)
    return dict(
        parts=parts,
        stray=stray,
        wrong_dir=wrong_dir,
        empty_files=empty_files,
        top=top,
        dup=dup,
        exists=True,
    )


def overlapping(pixels: set[tuple[int, int]]) -> list[str]:
    """Partitions that are nested inside another partition of the same catalogue."""
    out = []
    orders = sorted({o for o, _ in pixels})
    for o, p in pixels:
        for oo in orders:
            if oo >= o:
                break
            if (oo, p >> (2 * (o - oo))) in pixels:
                out.append(f"{o}/{p} inside {oo}/{p >> (2 * (o - oo))}")
    return out


def schema_sig(schema) -> str:
    return ";".join(f"{f.name}:{f.type}" for f in schema)


def footer(path: str) -> dict:
    import pyarrow.parquet as pq

    md = pq.read_metadata(path)
    schema = md.schema.to_arrow_schema()
    names = [md.schema.column(j).path.split(".")[0] for j in range(md.num_columns)]
    stats: dict[str, list] = {}
    for i in range(md.num_row_groups):
        rg = md.row_group(i)
        for j in range(rg.num_columns):
            c = rg.column(j)
            s = c.statistics
            name = names[j]
            cur = stats.setdefault(
                name, [None, None, 0, 0, 0]
            )  # min,max,nulls,rg_with_stats,rg_total
            cur[4] += 1
            if s is None:
                continue
            if s.has_null_count:
                cur[2] += s.null_count
            if s.has_min_max:
                try:
                    lo, hi = s.min, s.max
                except Exception:
                    continue
                if isinstance(lo, bool | int | float) and isinstance(
                    hi, bool | int | float
                ):
                    cur[3] += 1
                    cur[0] = lo if cur[0] is None else min(cur[0], lo)
                    cur[1] = hi if cur[1] is None else max(cur[1], hi)
    return dict(
        rows=md.num_rows,
        row_groups=md.num_row_groups,
        created_by=md.created_by,
        sig=schema_sig(schema),
        stats=stats,
    )


def merge_stats(acc: dict, stats: dict) -> None:
    for name, (lo, hi, nulls, rgs, rgt) in stats.items():
        a = acc.setdefault(name, [None, None, 0, 0, 0])
        if lo is not None:
            a[0] = lo if a[0] is None else min(a[0], lo)
            a[1] = hi if a[1] is None else max(a[1], hi)
        a[2] += nulls
        a[3] += rgs
        a[4] += rgt


def near(v, target) -> bool:
    return (
        v is not None
        and isinstance(v, int | float)
        and abs(v - target) <= 1e-6 * max(1.0, abs(target))
    )


def audit_structure(root: Path, cat: str, workers: int, out_dir: Path) -> dict:
    import pyarrow.parquet as pq

    t0 = time.time()
    base = root / cat
    res: dict = dict(catalogue=cat)
    props, hprops = (
        parse_props(base / "properties"),
        parse_props(base / "hats.properties"),
    )
    res["has_properties"], res["has_hats_properties"] = (
        props is not None,
        hprops is not None,
    )
    P = dict(props or {})
    for k, v in (hprops or {}).items():
        P.setdefault(k, v)
    res["props"] = P
    if props and hprops and props != hprops:
        res["props_diff"] = {
            k: [props.get(k), hprops.get(k)]
            for k in sorted(set(props) | set(hprops))
            if props.get(k) != hprops.get(k)
        }
    res["missing_required"] = [k for k in REQUIRED_KEYS if k not in P]
    res["missing_recommended"] = [k for k in RECOMMENDED_KEYS if k not in P]
    res["root_entries"] = sorted(os.listdir(base))
    header, pinfo = read_partition_info(base / "partition_info.csv")
    res["partition_info_header"] = header
    pi = {}
    for r in pinfo:
        try:
            o, p = int(r["Norder"]), int(r["Npix"])
        except (KeyError, ValueError):
            res.setdefault("partition_info_bad_rows", []).append(r)
            continue
        cnt = r.get("num_rows") or r.get("count") or r.get("num_row")
        pi[(o, p)] = int(cnt) if cnt not in (None, "") else None
    res["n_partition_info"] = len(pi)
    res["partition_info_duplicates"] = (
        len(pinfo) - len(pi) - len(res.get("partition_info_bad_rows", []))
    )
    lst = list_partitions(base / "dataset")
    parts = lst["parts"]
    res["dataset_exists"] = lst["exists"]
    res["dataset_top"] = lst["top"]
    res["stray"] = lst["stray"][:200]
    res["n_stray"] = len(lst["stray"])
    res["wrong_dir"] = lst["wrong_dir"][:50]
    res["empty_files"] = lst["empty_files"][:50]
    res["duplicate_partitions_on_disk"] = lst["dup"][:50]
    res["layout"] = dict(Counter(v["layout"] for v in parts.values()))
    res["n_partition_files"] = sum(len(v["files"]) for v in parts.values())
    res["n_partitions_on_disk"] = len(parts)
    res["missing_on_disk"] = sorted(f"{o}/{p}" for o, p in set(pi) - set(parts))[:200]
    res["n_missing_on_disk"] = len(set(pi) - set(parts))
    res["extra_on_disk"] = sorted(f"{o}/{p}" for o, p in set(parts) - set(pi))[:200]
    res["n_extra_on_disk"] = len(set(parts) - set(pi))
    pix = set(pi) | set(parts)
    res["overlapping_partitions"] = overlapping(pix)[:100]
    orders = Counter(o for o, _ in pix)
    res["orders"] = dict(sorted(orders.items()))
    res["order_range"] = [min(orders), max(orders)] if orders else None
    res["hats_order_matches_max"] = (
        str(max(orders)) == P.get("hats_order") if orders else None
    )
    res["dataset_bytes"] = sum(s for v in parts.values() for _, s in v["files"])

    # footers ---------------------------------------------------------------
    jobs = [(o, p, f) for (o, p), v in parts.items() for f, _ in v["files"]]
    acc: dict = {}
    sigs: Counter = Counter()
    sig_example: dict = {}
    per_part_rows: dict = defaultdict(int)
    per_part_rg: dict = defaultdict(int)
    coord_bad: list = []
    unreadable: list = []
    created_by: Counter = Counter()
    ra_c, dec_c = P.get("hats_col_ra"), P.get("hats_col_dec")
    part_rows_csv = []
    log(f"  {cat}: {len(jobs)} partition files, reading footers")
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(footer, f): (o, p, f) for o, p, f in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            o, p, f = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                unreadable.append(dict(file=f, error=repr(exc)[:300]))
                continue
            per_part_rows[(o, p)] += r["rows"]
            per_part_rg[(o, p)] += r["row_groups"]
            sigs[r["sig"]] += 1
            sig_example.setdefault(r["sig"], f)
            created_by[r["created_by"]] += 1
            merge_stats(acc, r["stats"])
            st = r["stats"]
            rmin = rmax = dmin = dmax = None
            if ra_c in st and dec_c in st:
                rmin, rmax = st[ra_c][0], st[ra_c][1]
                dmin, dmax = st[dec_c][0], st[dec_c][1]
                bad = []
                if rmin is not None and (rmin < 0 or rmax >= 360):
                    bad.append(f"ra [{rmin}, {rmax}]")
                if dmin is not None and (dmin < -90 or dmax > 90):
                    bad.append(f"dec [{dmin}, {dmax}]")
                if st[ra_c][2] or st[dec_c][2]:
                    bad.append(f"null coords {st[ra_c][2]}/{st[dec_c][2]}")
                if bad:
                    coord_bad.append(dict(partition=f"{o}/{p}", file=f, issues=bad))
            part_rows_csv.append(
                [o, p, f, r["rows"], r["row_groups"], rmin, rmax, dmin, dmax]
            )
            if n % 2000 == 0:
                log(f"    footers {n}/{len(jobs)}")
    res["unreadable_files"] = unreadable[:50]
    res["n_unreadable_files"] = len(unreadable)
    res["footer_rows_total"] = int(sum(per_part_rows.values()))
    try:
        res["hats_nrows"] = int(P.get("hats_nrows")) if P.get("hats_nrows") else None
    except ValueError:
        res["hats_nrows"] = None
    res["rows_match_hats_nrows"] = (
        res["hats_nrows"] == res["footer_rows_total"]
        if res["hats_nrows"] is not None
        else None
    )
    mism = [
        f"{o}/{p}: info {pi[(o, p)]} footer {per_part_rows.get((o, p))}"
        for (o, p) in pi
        if pi[(o, p)] is not None
        and (o, p) in per_part_rows
        and pi[(o, p)] != per_part_rows[(o, p)]
    ]
    res["partition_info_rowcount_mismatch"] = mism[:50]
    rows_arr = np.array(list(per_part_rows.values()) or [0])
    res["rows_per_partition"] = dict(
        min=int(rows_arr.min()),
        p01=float(np.percentile(rows_arr, 1)),
        median=float(np.median(rows_arr)),
        p99=float(np.percentile(rows_arr, 99)),
        max=int(rows_arr.max()),
    )
    tiny = sorted(((r, o, p) for (o, p), r in per_part_rows.items() if r < 1000))
    res["n_tiny_partitions_lt1000"] = len(tiny)
    res["n_partitions_lt100"] = sum(1 for r, _, _ in tiny if r < 100)
    res["tiny_partitions"] = [f"{o}/{p}:{r}" for r, o, p in tiny[:40]]
    res["empty_partitions"] = [
        f"{o}/{p}" for (o, p), r in per_part_rows.items() if r == 0
    ][:40]
    try:
        mx = int(P.get("hats_max_rows", 0))
    except ValueError:
        mx = 0
    if mx:
        big = sorted(
            ((r, o, p) for (o, p), r in per_part_rows.items() if r > 1.5 * mx),
            reverse=True,
        )
        res["n_partitions_over_1p5x_max_rows"] = len(big)
        res["largest_partitions"] = [f"{o}/{p}:{r}" for r, o, p in big[:10]]
    res["order0_partitions"] = sorted(
        f"0/{p}:{per_part_rows.get((0, p))}" for o, p in pix if o == 0
    )
    res["n_schema_variants"] = len(sigs)
    if len(sigs) > 1:
        ref = sigs.most_common(1)[0][0]
        refset = set(ref.split(";"))
        res["schema_variants"] = [
            dict(
                count=c,
                example=sig_example[s],
                added=sorted(set(s.split(";")) - refset)[:20],
                removed=sorted(refset - set(s.split(";")))[:20],
            )
            for s, c in sigs.most_common()[1:10]
        ]
    res["created_by"] = dict(created_by.most_common(5))
    res["coord_stats_out_of_range"] = coord_bad[:100]
    res["n_partitions_coord_stats_out_of_range"] = len(coord_bad)
    colstats = {}
    for name, (lo, hi, nulls, rgs, rgt) in acc.items():
        colstats[name] = dict(
            min=lo, max=hi, nulls=nulls, rg_with_stats=rgs, rg_total=rgt
        )
    res["column_stats"] = colstats
    tot = max(res["footer_rows_total"], 1)
    res["all_null_columns"] = sorted(
        n for n, s in colstats.items() if s["nulls"] >= tot
    )
    res["columns_without_stats"] = sorted(
        n for n, s in colstats.items() if s["rg_with_stats"] == 0
    )
    sent = {}
    for n, s in colstats.items():
        hits = [v for v in SENTINELS if near(s["min"], v) or near(s["max"], v)]
        if hits:
            sent[n] = dict(min=s["min"], max=s["max"])
    res["sentinel_like_min_max"] = sent

    # _common_metadata / _metadata ------------------------------------------
    ds = base / "dataset"
    cm = ds / "_common_metadata"
    if cm.is_file():
        try:
            sch = pq.read_schema(cm)
            res["common_metadata"] = dict(
                ncol=len(sch),
                fields_with_unit=sum(
                    1
                    for f in sch
                    if f.metadata
                    and any(k.lower() in (b"unit", b"units") for k in f.metadata)
                ),
                fields_with_description=sum(
                    1
                    for f in sch
                    if f.metadata
                    and any(k.lower() in (b"description", b"desc") for k in f.metadata)
                ),
                schema_meta_keys=[k.decode() for k in (sch.metadata or {})],
            )
            if sigs:
                ref = set(sigs.most_common(1)[0][0].split(";"))
                cmset = set(schema_sig(sch).split(";"))
                res["common_metadata"]["only_in_common_metadata"] = sorted(cmset - ref)[
                    :30
                ]
                res["common_metadata"]["only_in_partitions"] = sorted(ref - cmset)[:30]
            res["types"] = {f.name: str(f.type) for f in sch}
        except Exception as exc:
            res["common_metadata"] = dict(error=repr(exc)[:300])
    else:
        res["common_metadata"] = None
    meta = ds / "_metadata"
    if meta.is_file():
        res["metadata_bytes"] = meta.stat().st_size
        try:
            md = pq.read_metadata(meta)
            files = Counter()
            rg_rows = defaultdict(int)
            for i in range(md.num_row_groups):
                rg = md.row_group(i)
                fp = rg.column(0).file_path if rg.num_columns else ""
                files[fp] += 1
                rg_rows[fp] += rg.num_rows
            actual = {}
            for (o, p), v in parts.items():
                for f, _ in v["files"]:
                    actual[os.path.relpath(f, ds)] = (
                        per_part_rows.get((o, p)) if len(v["files"]) == 1 else None
                    )
            ref_files = set(files)
            res["metadata"] = dict(
                rows=md.num_rows,
                row_groups=md.num_row_groups,
                files_referenced=len(ref_files),
                rows_match_hats_nrows=(md.num_rows == res["hats_nrows"])
                if res["hats_nrows"] is not None
                else None,
                rows_match_footers=md.num_rows == res["footer_rows_total"],
                referenced_missing=sorted(ref_files - set(actual))[:20],
                n_referenced_missing=len(ref_files - set(actual)),
                not_referenced=sorted(set(actual) - ref_files)[:20],
                n_not_referenced=len(set(actual) - ref_files),
                stale_rowcounts=[
                    f"{f}: _metadata {rg_rows[f]} footer {actual[f]}"
                    for f in ref_files & set(actual)
                    if actual[f] is not None and rg_rows[f] != actual[f]
                ][:20],
            )
            del md
        except Exception as exc:
            res["metadata"] = dict(error=repr(exc)[:300])
    else:
        res["metadata"] = None
    # margins: sibling trees declaring this one as primary
    parent = base.parent
    margins = []
    try:
        for sib in os.listdir(parent):
            sp = parent / sib
            if sp == base or not sp.is_dir():
                continue
            sprops = (
                parse_props(sp / "hats.properties")
                or parse_props(sp / "properties")
                or {}
            )
            if (
                sprops.get("dataproduct_type") in ("margin", "index")
                and sprops.get("hats_primary_table_url", "").rstrip("/").split("/")[-1]
                == base.name
            ):
                margins.append(
                    dict(
                        name=sib,
                        type=sprops.get("dataproduct_type"),
                        threshold=sprops.get("hats_margin_threshold"),
                    )
                )
    except OSError:
        pass
    res["margins_and_indexes"] = margins
    res["seconds"] = round(time.time() - t0, 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / (safe(cat) + "_partitions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "norder",
                "npix",
                "file",
                "rows",
                "row_groups",
                "ra_min",
                "ra_max",
                "dec_min",
                "dec_max",
            ]
        )
        w.writerows(sorted(part_rows_csv))
    res["_parts"] = {
        f"{o}/{p}": dict(
            rows=per_part_rows.get((o, p)),
            files=[f for f, _ in v["files"]],
            bytes=sum(s for _, s in v["files"]),
        )
        for (o, p), v in parts.items()
    }
    return res


# ----------------------------------------------------------------------------- samples


def choose_samples(cat: str, struct: dict, n: int, budget: float) -> list[str]:
    parts = struct["_parts"]
    rng = np.random.default_rng(int(hashlib.sha256(cat.encode()).hexdigest()[:8], 16))
    chosen: list[str] = []
    for t in TARGETS.get(cat, []):
        if t in parts:
            chosen.append(t)
    for b in struct.get("coord_stats_out_of_range", [])[:4]:
        if b["partition"] not in chosen:
            chosen.append(b["partition"])
    items = [(k, v) for k, v in parts.items() if v["rows"]]
    if not items:
        return chosen
    by_order = defaultdict(list)
    for k, v in items:
        by_order[int(k.split("/")[0])].append((v["rows"], k))
    # one per order (median-size partition), then random by rows quantile
    for o in sorted(by_order):
        lst = sorted(by_order[o])
        k = lst[len(lst) // 2][1]
        if k not in chosen:
            chosen.append(k)
    keys = [k for _, k in sorted((v["rows"], k) for k, v in items)]
    for q in rng.permutation(len(keys)):
        if len(chosen) >= n + len(TARGETS.get(cat, [])):
            break
        if keys[q] not in chosen:
            chosen.append(keys[q])
    out, spent = [], 0.0
    for k in chosen:
        b = parts[k]["bytes"] / 1e9
        if out and spent + b > budget and k not in TARGETS.get(cat, []):
            continue
        out.append(k)
        spent += b
    return out


def unitvec(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    cd = np.cos(dec)
    return np.column_stack([cd * np.cos(ra), cd * np.sin(ra), np.sin(dec)])


def chord(arcsec):
    return 2 * np.sin(np.radians(arcsec / 3600.0) / 2)


def sep_arcsec(d):
    return np.degrees(2 * np.arcsin(np.clip(np.asarray(d) / 2, 0, 1))) * 3600


def to_np(col):
    import pyarrow as pa

    if pa.types.is_dictionary(col.type):
        col = col.cast(col.type.value_type)
    return col.to_numpy(zero_copy_only=False)


def parse_array_text(s):
    if s is None:
        return None
    t = str(s).strip().strip("[](){}")
    if not t:
        return []
    try:
        return [float(x) for x in re.split(r"[,\s;]+", t) if x]
    except ValueError:
        return None


def profile_column(col) -> dict:
    import pyarrow as pa
    import pyarrow.compute as pc

    t = col.type
    d = dict(type=str(t), nulls=int(col.null_count), n=len(col))
    try:
        if pa.types.is_floating(t) or pa.types.is_integer(t) or pa.types.is_boolean(t):
            a = to_np(col.drop_null() if col.null_count else col).astype(np.float64)
            fin = np.isfinite(a)
            d["nan"] = int(np.isnan(a).sum())
            d["inf"] = int(np.isinf(a).sum())
            if fin.any():
                d["min"], d["max"] = float(a[fin].min()), float(a[fin].max())
            s = {}
            for v in SENTINELS:
                c = int(np.isclose(a, v, rtol=0, atol=1e-6 * max(1, abs(v))).sum())
                if c:
                    s[str(v)] = c
            if s:
                d["sentinels"] = s
        elif (
            pa.types.is_string(t)
            or pa.types.is_large_string(t)
            or pa.types.is_binary(t)
            or pa.types.is_large_binary(t)
        ):
            v = col.drop_null()
            if pa.types.is_binary(t) or pa.types.is_large_binary(t):
                v = pc.cast(v, pa.string(), safe=False) if len(v) else v
            vals = v.slice(0, 20000).to_pylist()
            d["examples"] = [str(x)[:80] for x in vals[:3]]
            d["max_len"] = max((len(str(x)) for x in vals), default=0)
            uniq = list(dict.fromkeys(vals))[:500]
            num = arr = 0
            lens = Counter()
            for x in uniq:
                try:
                    float(x)
                    num += 1
                    continue
                except (TypeError, ValueError):
                    pass
                pa_ = parse_array_text(x)
                if pa_ is not None and len(pa_) > 1:
                    arr += 1
                    lens[len(pa_)] += 1
            d["distinct_sampled"] = len(uniq)
            d["numeric_scalar_frac"] = round(num / len(uniq), 3) if uniq else None
            d["numeric_array_frac"] = round(arr / len(uniq), 3) if uniq else None
            if lens:
                d["array_lengths"] = dict(lens.most_common(5))
        elif (
            pa.types.is_list(t)
            or pa.types.is_large_list(t)
            or pa.types.is_fixed_size_list(t)
        ):
            ln = (
                to_np(pc.list_value_length(col))
                if not pa.types.is_fixed_size_list(t)
                else np.full(len(col), t.list_size)
            )
            d["list_len_min"], d["list_len_max"] = (
                int(np.nanmin(ln)) if len(ln) else None,
                int(np.nanmax(ln)) if len(ln) else None,
            )
    except Exception as exc:
        d["error"] = repr(exc)[:200]
    return d


class GaiaReader:
    def __init__(self, root: Path):
        self.root = root / GAIA_TREE
        _, rows = read_partition_info(self.root / "partition_info.csv")
        self.parts = {(int(r["Norder"]), int(r["Npix"])) for r in rows}
        self.orders = sorted({o for o, _ in self.parts})

    def path(self, o, p):
        return (
            self.root
            / "dataset"
            / f"Norder={o}"
            / f"Dir={p // 10000 * 10000}"
            / f"Npix={p}.parquet"
        )

    def rows_in(self, order: int, pix: int):
        import pyarrow as pa
        import pyarrow.parquet as pq

        shift = 2 * (29 - order)
        lo, hi = pix << shift, (pix + 1) << shift
        tabs = []
        for go in self.orders:
            if go <= order:
                gp = pix >> (2 * (order - go))
                if (go, gp) in self.parts:
                    tabs.append(
                        pq.read_table(
                            self.path(go, gp),
                            columns=GAIA_COLS + ["_healpix_29"],
                            filters=[
                                ("_healpix_29", ">=", lo),
                                ("_healpix_29", "<", hi),
                            ],
                        )
                    )
            else:
                for gp in range(
                    pix << (2 * (go - order)), (pix + 1) << (2 * (go - order))
                ):
                    if (go, gp) in self.parts:
                        tabs.append(
                            pq.read_table(
                                self.path(go, gp), columns=GAIA_COLS + ["_healpix_29"]
                            )
                        )
        if not tabs:
            return None
        return pa.concat_tables(tabs)


def pick_cols(names: list[str], wanted: list[str]) -> list[str]:
    low = {n.lower(): n for n in names}
    out = []
    for w in wanted:
        if w in names:
            out.append(w)
        elif w.lower() in low and low[w.lower()] not in out and w != "id":
            out.append(low[w.lower()])
    return list(dict.fromkeys(out))


def tally_pairs(tab, i, j, cols) -> dict:
    out = {}
    for c in cols:
        a = to_np(tab.column(c))
        vi, vj = a[i], a[j]
        same = vi == vj
        d = dict(same=int(np.sum(same)), different=int(np.sum(~same)))
        if a.dtype.kind in "OUSb" or c.upper() in (
            "RELEASE",
            "PHOTSYS",
            "TYPE",
            "PRIORSEC",
            "PRIMARY_SOURCE",
            "MERGEDCLASS",
            "FLAGS_GOLD",
        ):
            pairs = Counter(
                tuple(sorted((str(x), str(y))))
                for x, y in zip(vi[:20000], vj[:20000], strict=False)
            )
            d["top_value_pairs"] = [[list(k), v] for k, v in pairs.most_common(6)]
        out[c] = d
    return out


def sample_partition(
    root: Path, cat: str, struct: dict, key: str, gaia: GaiaReader | None, profile: bool
) -> dict:
    import healpy as hp
    import pyarrow.parquet as pq
    from scipy.spatial import cKDTree

    P = struct["props"]
    o, p = map(int, key.split("/"))
    files = struct["_parts"][key]["files"]
    h = hint(cat)
    ra_c, dec_c = P.get("hats_col_ra"), P.get("hats_col_dec")
    names = pq.read_schema(files[0]).names
    idc = [c for c in (h.get("id") or []) if c in names]
    if not idc and P.get("hats_cols_survey_id") in names:
        idc = [P["hats_cols_survey_id"]]
    id2 = [c for c in (h.get("id2") or []) if c in names]
    disc = pick_cols(names, h.get("disc", []))
    extra = pick_cols(names, h.get("extra", []))
    cols = (
        None
        if profile
        else list(
            dict.fromkeys(
                [
                    c
                    for c in ["_healpix_29", ra_c, dec_c, *idc, *id2, *disc, *extra]
                    if c in names
                ]
            )
        )
    )
    t0 = time.time()
    tab = (
        pq.read_table(files, columns=cols)
        if len(files) > 1
        else pq.read_table(files[0], columns=cols)
    )
    r: dict = dict(
        partition=key,
        rows=tab.num_rows,
        files=len(files),
        read_s=round(time.time() - t0, 1),
        id_columns=idc,
    )
    if profile:
        r["columns"] = {c: profile_column(tab.column(c)) for c in tab.column_names}
    ra = to_np(tab.column(ra_c)).astype(np.float64)
    dec = to_np(tab.column(dec_c)).astype(np.float64)
    finite = np.isfinite(ra) & np.isfinite(dec)
    oor = finite & ((ra < 0) | (ra >= 360) | (dec < -90) | (dec > 90))
    r["coords"] = dict(
        nonfinite=int((~finite).sum()),
        out_of_range=int(oor.sum()),
        ra_min=float(np.nanmin(ra)) if finite.any() else None,
        ra_max=float(np.nanmax(ra)) if finite.any() else None,
        dec_min=float(np.nanmin(dec)) if finite.any() else None,
        dec_max=float(np.nanmax(dec)) if finite.any() else None,
    )

    def examples(mask, k=5):
        ix = np.flatnonzero(mask)[:k]
        ex = []
        for i in ix:
            e = {c: tab.column(c)[int(i)].as_py() for c in idc[:3]}
            e.update(ra=float(ra[i]), dec=float(dec[i]))
            ex.append(e)
        return ex

    if oor.any() or (~finite).any():
        r["coords"]["examples"] = examples(oor | ~finite)
    # pixel membership (degrees), and the radians hypothesis when degrees fail
    ok = finite & ~oor
    pixv = np.full(len(ra), -1, dtype=np.int64)
    if ok.any():
        pixv[ok] = hp.ang2pix(2**o, ra[ok], dec[ok], nest=True, lonlat=True)
    outside = ok & (pixv != p)
    r["pixel_membership"] = dict(outside=int(outside.sum()), checked=int(ok.sum()))
    if outside.any():
        r["pixel_membership"]["examples"] = examples(outside, 3)
        if (
            np.nanmax(np.abs(ra[ok])) <= 2 * np.pi + 1e-9
            and np.nanmax(np.abs(dec[ok])) <= np.pi / 2 + 1e-9
        ):
            pr = hp.ang2pix(
                2**o,
                np.degrees(ra[ok]) % 360,
                np.degrees(dec[ok]),
                nest=True,
                lonlat=True,
            )
            r["pixel_membership"]["outside_if_radians"] = int((pr != p).sum())
    if "_healpix_29" in tab.column_names and ok.any():
        h29 = to_np(tab.column("_healpix_29")).astype(np.int64)
        r["healpix29"] = dict(
            outside_partition=int(((h29 >> (2 * (29 - o))) != p).sum())
        )
        idx = np.flatnonzero(ok)[:20000]
        calc = hp.ang2pix(2**29, ra[idx], dec[idx], nest=True, lonlat=True)
        r["healpix29"]["mismatch_vs_radec_of_20k"] = int((calc != h29[idx]).sum())
        r["healpix29"]["mismatch_at_order10_of_20k"] = int(
            ((calc >> 38) != (h29[idx] >> 38)).sum()
        )
    # duplicate ids
    for key_cols, label in ((idc, "id"), (id2, "id2")):
        if not key_cols:
            continue
        if len(key_cols) == 1:
            arr = to_np(tab.column(key_cols[0]))
            keyv = arr
        else:
            keyv = np.array(
                [
                    "|".join(map(str, t))
                    for t in zip(
                        *[to_np(tab.column(c)) for c in key_cols], strict=False
                    )
                ]
            )
        u, inv, cnt = np.unique(
            keyv.astype(str) if keyv.dtype.kind == "O" else keyv,
            return_inverse=True,
            return_counts=True,
        )
        dupmask = cnt[inv] > 1
        r[f"duplicate_{label}"] = dict(
            columns=key_cols,
            rows_with_dup=int(dupmask.sum()),
            dup_keys=int((cnt > 1).sum()),
            examples=[str(x) for x in u[cnt > 1][:5]],
        )
        if label == "id" and keyv.dtype.kind in "iuf":
            r["id_range"] = (
                [float(np.nanmin(keyv)), float(np.nanmax(keyv))] if len(keyv) else None
            )
    # nearest neighbour within the survey (same position twice)
    xyz = unitvec(ra[ok], dec[ok])
    okidx = np.flatnonzero(ok)
    nn = {}
    if len(xyz) > 1:
        tree = cKDTree(xyz)
        d, ix = tree.query(xyz, k=2)
        s = sep_arcsec(d[:, 1])
        for thr in (0.001, 0.1, 0.5, 1.0, 1.5):
            nn[f"frac_nn_lt_{thr}"] = round(float((s < thr).mean()), 5)
        nn["n_exact_0p001"] = int((s < 0.001).sum())
        close = np.flatnonzero(s < 0.5)
        if close.size:
            i, j = okidx[close], okidx[ix[close, 1]]
            keep = i < j
            pi_, pj_ = i[keep], j[keep]
            if not pi_.size:
                pi_, pj_ = i, j
            nn["pairs_lt_0p5"] = int(pi_.size)
            nn["pair_tally"] = tally_pairs(tab, pi_, pj_, disc)
            nn["pair_diagnostics"] = pair_diag(cat, tab, pi_, pj_)
            nn["pair_examples"] = [
                dict(
                    a={c: tab.column(c)[int(a)].as_py() for c in (idc + disc)[:5]},
                    b={c: tab.column(c)[int(b)].as_py() for c in (idc + disc)[:5]},
                    sep=float(
                        sep_arcsec(
                            np.linalg.norm(
                                xyz[np.searchsorted(okidx, a)]
                                - xyz[np.searchsorted(okidx, b)]
                            )
                        )
                    ),
                )
                for a, b in list(zip(pi_, pj_, strict=False))[:3]
            ]
    else:
        tree = None
    r["nearest_neighbour"] = nn
    # tree-specific diagnostics on the sample itself
    r["diagnostics"] = diagnostics(cat, tab, idc)
    # Gaia-anchored check
    if gaia is not None and tree is not None and not cat.startswith("gaia/"):
        try:
            r["gaia"] = gaia_check(cat, tab, ra, dec, ok, okidx, tree, o, p, gaia)
        except Exception as exc:
            r["gaia"] = dict(error=repr(exc)[:300])
    elif cat == "gaia/dr3_xp_continuous" and gaia is not None:
        try:
            g = gaia.rows_in(o, p)
            gid = to_np(g.column("source_id"))
            sid = to_np(tab.column("source_id"))
            found = np.isin(sid, gid)
            r["gaia_join"] = dict(
                xp_rows=len(sid), found_in_gaia_partition=int(found.sum())
            )
            gi = {v: i for i, v in enumerate(gid)}
            m = [gi[v] for v in sid[found][:20000]]
            gra, gdec = to_np(g.column("ra"))[m], to_np(g.column("dec"))[m]
            sep = sep_arcsec(
                np.linalg.norm(
                    unitvec(ra[found][:20000], dec[found][:20000]) - unitvec(gra, gdec),
                    axis=1,
                )
            )
            r["gaia_join"]["median_sep_arcsec"] = (
                float(np.median(sep)) if len(sep) else None
            )
            r["gaia_join"]["max_sep_arcsec"] = float(np.max(sep)) if len(sep) else None
        except Exception as exc:
            r["gaia_join"] = dict(error=repr(exc)[:300])
    r["seconds"] = round(time.time() - t0, 1)
    return r


def pair_diag(cat: str, tab, i, j) -> dict:
    """Relation between two rows at (nearly) the same position."""
    out: dict = {}
    names = tab.column_names
    try:
        if cat.startswith("hsc/pdr3/forced_") and "parent" in names:
            idv = to_np(tab.column("id")).astype(np.int64)
            par = to_np(tab.column("parent")).astype(np.int64)
            nch = to_np(tab.column("deblend_nChild"))
            pc_ = (par[i] == idv[j]) | (par[j] == idv[i])
            leaf = (nch[i] == 0) & (nch[j] == 0) & ~pc_
            out["n_pairs"] = int(len(i))
            out["parent_child_frac"] = round(float(np.mean(pc_)), 4)
            out["both_leaf_frac"] = round(float(np.mean(leaf)), 4)
            if leaf.any():
                out["both_leaf_same_id_gt_gt_42_frac"] = round(
                    float(np.mean((idv[i] >> 42)[leaf] == (idv[j] >> 42)[leaf])), 4
                )
                out["both_leaf_same_parent_frac"] = round(
                    float(np.mean(par[i][leaf] == par[j][leaf])), 4
                )
    except Exception as exc:
        out["error"] = repr(exc)[:200]
    return out


def objid_decode(v):
    v = np.asarray(v, dtype=np.int64)
    return dict(
        sky_version=(v >> 59) & 0xF,
        rerun=(v >> 48) & 0x7FF,
        run=(v >> 32) & 0xFFFF,
        camcol=(v >> 29) & 0x7,
        field=(v >> 16) & 0xFFF,
        obj=v & 0xFFFF,
    )


def diagnostics(cat: str, tab, idc) -> dict:
    out: dict = {}
    names = tab.column_names
    try:
        if cat.startswith("sdss") and "id" in names:
            v = to_np(tab.column("id"))
            out["id_dtype"] = str(v.dtype)
            out["id_examples"] = [int(x) for x in v[:3]]
            if v.dtype.kind in "iu":
                dec_ = objid_decode(v)
                for c, k in (
                    ("RUN", "run"),
                    ("CAMCOL", "camcol"),
                    ("FIELD", "field"),
                    ("ID", "obj"),
                ):
                    if c in names:
                        out[f"objid_{k}_matches_{c}"] = round(
                            float(
                                np.mean(
                                    dec_[k] == to_np(tab.column(c)).astype(np.int64)
                                )
                            ),
                            5,
                        )
                out["objid_sky_version_values"] = dict(
                    Counter(map(str, dec_["sky_version"][:100000])).most_common(3)
                )
            if "ID" in names:
                a = to_np(tab.column("ID"))
                out["ID_range"] = [int(a.min()), int(a.max())]
                out["ID_distinct_frac"] = round(len(np.unique(a)) / len(a), 4)
            if "RERUN" in names:
                out["RERUN_values"] = dict(
                    Counter(
                        map(str, tab.column("RERUN").to_pylist()[:100000])
                    ).most_common(3)
                )
        if cat.startswith("ps1") and "nDetections" in names:
            nd = to_np(tab.column("nDetections"))
            out["nDetections_eq1_frac"] = round(float(np.mean(nd == 1)), 4)
            out["nDetections_le2_frac"] = round(float(np.mean(nd <= 2)), 4)
            out["nDetections_eq0_frac"] = round(float(np.mean(nd == 0)), 4)
            for c in ("pmra", "pmdec", "raMeanErr", "epochMean"):
                if c in names:
                    a = to_np(tab.column(c)).astype(float)
                    f = np.isfinite(a)
                    out[f"{c}_finite_frac"] = round(float(f.mean()), 4)
                    if f.any():
                        out[f"{c}_pcts"] = [
                            float(x) for x in np.percentile(a[f], [1, 50, 99])
                        ]
                        out[f"{c}_eq_minus999_frac"] = round(
                            float(np.mean(a == -999)), 4
                        )
        if cat.startswith("legacysurvey"):
            for c in ("RELEASE", "PHOTSYS"):
                if c in names:
                    out[f"{c}_counts"] = dict(
                        Counter(map(str, tab.column(c).to_pylist())).most_common(6)
                    )
        if cat.startswith("hsc/pdr3/forced_") and "parent" in names:
            par = to_np(tab.column("parent"))
            nch = to_np(tab.column("deblend_nChild"))
            out["frac_parent_with_children"] = round(float(np.mean(nch > 0)), 4)
            out["frac_children"] = round(float(np.mean(par != 0)), 4)
            out["frac_isolated"] = round(float(np.mean((par == 0) & (nch == 0))), 4)
            idv = to_np(tab.column("id")).astype(np.int64)
            out["id_tract_values_if_id_gt_gt_42"] = dict(
                Counter(map(str, (idv >> 42)[:200000])).most_common(5)
            )
        if cat.startswith("atlas") and "PRIMARY_SOURCE" in names:
            out["PRIMARY_SOURCE_counts"] = dict(
                Counter(map(str, tab.column("PRIMARY_SOURCE").to_pylist())).most_common(
                    5
                )
            )
        if cat.split("/")[0] in (
            "uhs",
            "vhs",
            "vvvx",
            "vmc",
            "viking",
            "video",
            "ukidss",
        ):
            for c in ("priorsec", "mergedclass"):
                if c in names:
                    a = to_np(tab.column(c))
                    out[f"{c}_counts"] = (
                        dict(Counter(map(str, a[:200000])).most_common(5))
                        if c == "mergedclass"
                        else dict(zero=int(np.sum(a == 0)), nonzero=int(np.sum(a != 0)))
                    )
            if "priorsec" in names and "framesetid" in names:
                a, b = to_np(tab.column("priorsec")), to_np(tab.column("framesetid"))
                out["priorsec_eq_framesetid"] = int(np.sum(a == b))
                out["priorsec_secondary"] = int(np.sum((a != 0) & (a != b)))
        if cat.startswith(("jplus", "jpas")):
            for c in ("MAG_AUTO", "CLASS_STAR", "FLAGS", "TILE_ID", "NUMBER"):
                if c in names:
                    vals = tab.column(c).slice(0, 5).to_pylist()
                    out[f"{c}_examples"] = [str(x)[:200] for x in vals[:2]]
            if "MAG_AUTO" in names:
                parsed = [
                    parse_array_text(x)
                    for x in tab.column("MAG_AUTO").slice(0, 5000).to_pylist()
                ]
                out["MAG_AUTO_lengths"] = dict(
                    Counter(
                        len(x) if x is not None else -1 for x in parsed
                    ).most_common(5)
                )
                vals = np.array([v for x in parsed if x for v in x])
                if vals.size:
                    out["MAG_AUTO_value_pcts"] = [
                        float(x) for x in np.percentile(vals, [1, 50, 99])
                    ]
                    out["MAG_AUTO_eq_99_frac"] = round(
                        float(np.mean(np.isclose(vals, 99))), 4
                    )
        if cat.startswith("splus") and "id" in names:
            out["id_examples"] = [
                str(x) for x in tab.column("id").slice(0, 3).to_pylist()
            ]
        if cat.startswith("euclid/dr1") and "ra" in names:
            a = to_np(tab.column("ra")).astype(float)
            b = to_np(tab.column("ra_from_object_id")).astype(float)
            out["ra_minus_ra_from_object_id_max_arcsec"] = float(
                np.nanmax(np.abs(a - b)) * 3600
            )
    except Exception as exc:
        out["error"] = repr(exc)[:300]
    return out


def robust_slope(x, y):
    f = np.isfinite(x) & np.isfinite(y)
    if f.sum() < 30:
        return None
    x, y = x[f], y[f]
    lo, hi = np.percentile(x, [10, 90])
    a, b = y[x <= lo], y[x >= hi]
    if not (len(a) and len(b)) or hi - lo <= 0:
        return None
    return float(
        (np.median(b) - np.median(a)) / (np.median(x[x >= hi]) - np.median(x[x <= lo]))
    )


def gaia_check(cat, tab, ra, dec, ok, okidx, tree, o, p, gaia: GaiaReader) -> dict:
    """Survey rows within 1.5" of each Gaia DR3 source in (a sub-pixel of) the partition."""
    import healpy as hp
    from scipy.spatial import cKDTree

    so, sp = o, p
    sel = ok
    if o < 6:  # restrict to one order-6 sub-pixel holding the median survey row
        so = 6
        sub = hp.ang2pix(2**6, ra[ok], dec[ok], nest=True, lonlat=True)
        vals, cnt = np.unique(sub, return_counts=True)
        sp = int(vals[np.argsort(cnt)[len(cnt) // 2]])
        mask = np.zeros(len(ra), bool)
        mask[okidx[sub == sp]] = True
        sel = mask
        sidx = np.flatnonzero(sel)
        tree = cKDTree(unitvec(ra[sidx], dec[sidx]))
    else:
        sidx = okidx
    g = gaia.rows_in(so, sp)
    if g is None or g.num_rows == 0:
        return dict(pixel=f"{so}/{sp}", n_gaia=0)
    gra, gdec = (
        to_np(g.column("ra")).astype(float),
        to_np(g.column("dec")).astype(float),
    )
    gmag = to_np(g.column("phot_g_mean_mag")).astype(float)
    gx = unitvec(gra, gdec)
    d, ix = tree.query(gx, k=3, distance_upper_bound=chord(MATCH_RADIUS))
    within = np.isfinite(d)
    n = within.sum(1)
    sep1 = np.where(
        within[:, 0], sep_arcsec(np.where(within[:, 0], d[:, 0], 0)), np.nan
    )
    res = dict(
        pixel=f"{so}/{sp}",
        n_gaia=int(len(gra)),
        n_survey=int(len(sidx)),
        frac_gaia_matched=round(float(np.mean(n >= 1)), 4),
        frac_gaia_ge2=round(float(np.mean(n >= 2)), 4),
        frac_matched_with_ge2=round(float(np.sum(n >= 2) / max(np.sum(n >= 1), 1)), 4),
        median_sep_arcsec=float(np.nanmedian(sep1))
        if np.isfinite(sep1).any()
        else None,
        p90_sep_arcsec=float(np.nanpercentile(sep1, 90))
        if np.isfinite(sep1).any()
        else None,
    )
    for lo, hi in ((0, 17), (17, 19), (19, 21)):
        m = (gmag >= lo) & (gmag < hi)
        if m.any():
            res[f"frac_matched_G{lo}_{hi}"] = round(float(np.mean(n[m] >= 1)), 4)
            res[f"frac_ge2_G{lo}_{hi}"] = round(float(np.mean(n[m] >= 2)), 4)
    names = tab.column_names
    best = np.flatnonzero(within[:, 0])
    rows1 = sidx[ix[best, 0]]
    multi = np.flatnonzero(n >= 2)
    r1, r2 = sidx[ix[multi, 0]], sidx[ix[multi, 1]]
    h = hint(cat)
    disc = pick_cols(names, h.get("disc", []))
    if multi.size:
        res["second_match_tally"] = tally_pairs(tab, r1, r2, disc)
        res["second_match_pair_diagnostics"] = pair_diag(cat, tab, r1, r2)
        res["second_match_sep_between_rows_arcsec_median"] = float(
            np.median(
                sep_arcsec(
                    np.linalg.norm(
                        unitvec(ra[r1], dec[r1]) - unitvec(ra[r2], dec[r2]), axis=1
                    )
                )
            )
        )
        idc = [c for c in (h.get("id") or []) if c in names][:3]
        res["second_match_examples"] = [
            dict(
                gaia_source_id=int(to_np(g.column("source_id"))[multi[k]]),
                a={c: tab.column(c)[int(r1[k])].as_py() for c in idc + disc[:3]},
                b={c: tab.column(c)[int(r2[k])].as_py() for c in idc + disc[:3]},
            )
            for k in range(min(3, multi.size))
        ]
    gm = gmag[best]
    gpm = to_np(g.column("pmra")).astype(float)[best]
    gbr = to_np(g.column("bp_rp")).astype(float)[best]
    try:
        if cat.startswith("hsc/pdr3/forced_") and "parent" in names:
            nch = to_np(tab.column("deblend_nChild"))[rows1]
            par = to_np(tab.column("parent"))[rows1]
            res["best_match_parent_frac"] = round(float(np.mean(nch > 0)), 4)
            res["best_match_child_frac"] = round(float(np.mean(par != 0)), 4)
            res["best_match_isolated_frac"] = round(
                float(np.mean((par == 0) & (nch == 0))), 4
            )
            fl = to_np(tab.column("base_PsfFlux_instFlux")).astype(float)[rows1]
            m = (gm > 17) & (gm < 20) & (fl > 0) & (nch == 0)
            if m.sum() > 10:
                res["zp_G_plus_2p5log10_instFlux_median"] = float(
                    np.median(gm[m] + 2.5 * np.log10(fl[m]))
                )
        if cat.startswith("hsc/pdr3/forced_nb") and "flux_psf" in names:
            fl = to_np(tab.column("flux_psf")).astype(float)
            mg = to_np(tab.column("mag_psf")).astype(float)
            m = (fl > 0) & np.isfinite(mg)
            if m.any():
                res["mag_psf_plus_2p5log10_flux_psf_median"] = float(
                    np.median(mg[m] + 2.5 * np.log10(fl[m]))
                )
        if cat.startswith("ps1"):
            pmr = (
                to_np(tab.column("pmra")).astype(float)[rows1]
                if "pmra" in names
                else None
            )
            if pmr is not None:
                m = (
                    np.isfinite(pmr)
                    & np.isfinite(gpm)
                    & (np.abs(gpm) > 30)
                    & (pmr != 0)
                    & (pmr > -900)
                )
                res["n_pm_pairs"] = int(m.sum())
                if m.sum() > 5:
                    res["pmra_ratio_ps1_over_gaia_median"] = float(
                        np.median(pmr[m] / gpm[m])
                    )
            if "raMeanErr" in names:
                e = to_np(tab.column("raMeanErr")).astype(float)[rows1]
                m = (gm < 19) & np.isfinite(e) & (e > 0)
                if m.sum() > 10:
                    res["raMeanErr_median_G_lt19"] = float(np.median(e[m]))
                    res["sep_median_G_lt19_arcsec"] = float(np.median(sep1[best][m]))
            if "nDetections" in names:
                nd = to_np(tab.column("nDetections"))[rows1]
                res["best_match_nDetections_eq1_frac"] = round(
                    float(np.mean(nd == 1)), 4
                )
                allnd = to_np(tab.column("nDetections"))[sidx]
                res["survey_rows_nDetections_eq1_frac"] = round(
                    float(np.mean(allnd == 1)), 4
                )
                unmatched = np.ones(len(ra), bool)
                unmatched[rows1] = False
                res["unmatched_rows_nDetections_eq1_frac"] = round(
                    float(np.mean(allnd[unmatched[sidx]] == 1)), 4
                )
        if cat.startswith("catwise") and "PMRA" in names:
            pmr = to_np(tab.column("PMRA")).astype(float)[rows1]
            m = np.isfinite(pmr) & np.isfinite(gpm) & (np.abs(gpm) > 100)
            if m.sum() > 5:
                res["PMRA_ratio_over_gaia_median"] = float(np.median(pmr[m] / gpm[m]))
        if cat.startswith("allwise") and "pmra" in names:
            pmr = to_np(tab.column("pmra")).astype(float)[rows1]
            m = np.isfinite(pmr) & np.isfinite(gpm) & (np.abs(gpm) > 100)
            if m.sum() > 5:
                res["pmra_ratio_over_gaia_median"] = float(np.median(pmr[m] / gpm[m]))
        if cat.startswith("pristine"):
            gid = to_np(g.column("source_id"))
            sid = to_np(tab.column("source_id"))[sidx]
            gi = {v: i for i, v in enumerate(gid)}
            pairs = [(k, gi[v]) for k, v in enumerate(sid) if v in gi]
            if pairs:
                a, b = np.array(pairs).T
                rr, dd = ra[sidx][a], dec[sidx][a]
                res["pristine_RADec_vs_gaia_same_source_id_sep_arcsec"] = [
                    float(x)
                    for x in np.percentile(
                        sep_arcsec(np.linalg.norm(unitvec(rr, dd) - gx[b], axis=1)),
                        [50, 90, 99],
                    )
                ]
                if "RA_Pr" in names:
                    rp = to_np(tab.column("RA_Pr")).astype(float)[sidx][a]
                    dp = to_np(tab.column("Dec_Pr")).astype(float)[sidx][a]
                    f = np.isfinite(rp) & np.isfinite(dp)
                    if f.any():
                        res["pristine_RA_Pr_vs_gaia_sep_arcsec"] = [
                            float(x)
                            for x in np.percentile(
                                sep_arcsec(
                                    np.linalg.norm(
                                        unitvec(rp[f], dp[f]) - gx[b][f], axis=1
                                    )
                                ),
                                [50, 90, 99],
                            )
                        ]
                res["source_id_found_in_gaia_pixel_frac"] = round(
                    len(pairs) / max(len(sid), 1), 4
                )
        if cat.startswith("legacysurvey") and multi.size and "RELEASE" in names:
            a1 = to_np(tab.column("RELEASE"))[r1]
            a2 = to_np(tab.column("RELEASE"))[r2]
            res["second_match_release_pairs"] = [
                [list(k), v]
                for k, v in Counter(
                    tuple(sorted((str(x), str(y))))
                    for x, y in zip(a1, a2, strict=False)
                ).most_common(5)
            ]
        if cat.startswith(("jplus", "jpas")):
            for col in hint(cat).get("arrays", []):
                if col not in names:
                    continue
                vals = [
                    parse_array_text(x) for x in tab.column(col).take(rows1).to_pylist()
                ]
                L = Counter(len(v) for v in vals if v).most_common(1)
                if not L:
                    continue
                L = L[0][0]
                mat = np.full((len(vals), L), np.nan)
                for k, v in enumerate(vals):
                    if v and len(v) == L:
                        mat[k] = v
                mat[(mat > 50) | (mat < -50)] = np.nan
                m = (gm > 14) & (gm < 18.5)
                slopes = [robust_slope(gbr[m], mat[m, k] - gm[m]) for k in range(L)]
                res[f"{col}_slope_vs_bp_rp_per_element"] = [
                    None if s is None else round(s, 3) for s in slopes
                ]
                res[f"{col}_median_minus_G_per_element"] = [
                    None
                    if not np.isfinite(mat[m, k]).any()
                    else round(float(np.nanmedian(mat[m, k] - gm[m])), 3)
                    for k in range(L)
                ]
                break
        if cat.startswith("euclid/dr1") and multi.size and "tileid" in names:
            t1 = to_np(tab.column("tileid"))[r1]
            t2 = to_np(tab.column("tileid"))[r2]
            res["second_match_same_tileid_frac"] = round(float(np.mean(t1 == t2)), 4)
    except Exception as exc:
        res["diag_error"] = repr(exc)[:300]
    return res


# ----------------------------------------------------------------------------- full scan


def scan_file(
    path: str, ra_c: str, dec_c: str, o: int, p: int, membership: bool
) -> dict:
    import healpy as hp
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    cols = [ra_c, dec_c] + (["_healpix_29"] if "_healpix_29" in names else [])
    t = pq.read_table(path, columns=cols)
    ra = to_np(t.column(ra_c)).astype(np.float64)
    dec = to_np(t.column(dec_c)).astype(np.float64)
    fin = np.isfinite(ra) & np.isfinite(dec)
    oor = fin & ((ra < 0) | (ra >= 360) | (dec < -90) | (dec > 90))
    r = dict(rows=len(ra), nonfinite=int((~fin).sum()), out_of_range=int(oor.sum()))
    if oor.any() or (~fin).any():
        ix = np.flatnonzero(oor | ~fin)[:3]
        r["examples"] = [dict(ra=float(ra[i]), dec=float(dec[i])) for i in ix]
    if "_healpix_29" in t.column_names:
        h29 = to_np(t.column("_healpix_29")).astype(np.int64)
        r["h29_outside"] = int(((h29 >> (2 * (29 - o))) != p).sum())
    if membership:
        ok = fin & ~oor
        r["radec_outside"] = (
            int((hp.ang2pix(2**o, ra[ok], dec[ok], nest=True, lonlat=True) != p).sum())
            if ok.any()
            else 0
        )
    return r


def full_scan(cat: str, struct: dict, workers: int, deadline: float) -> dict:
    P = struct["props"]
    ra_c, dec_c = P.get("hats_col_ra"), P.get("hats_col_dec")
    membership = (struct.get("hats_nrows") or 0) < 4_000_000_000
    tot = Counter()
    bad = []
    done = 0
    t0 = time.time()
    jobs = [(k, f) for k, v in struct["_parts"].items() for f in v["files"]]
    with ThreadPoolExecutor(workers) as ex:
        futs = {}
        it = iter(jobs)
        for k, f in it:
            o, p = map(int, k.split("/"))
            futs[ex.submit(scan_file, f, ra_c, dec_c, o, p, membership)] = k
            if len(futs) >= workers * 2:
                break
        while futs:
            for fut in as_completed(list(futs)):
                k = futs.pop(fut)
                try:
                    r = fut.result()
                except Exception as exc:
                    bad.append(dict(partition=k, error=repr(exc)[:200]))
                    r = {}
                done += 1
                for c in (
                    "rows",
                    "nonfinite",
                    "out_of_range",
                    "h29_outside",
                    "radec_outside",
                ):
                    tot[c] += r.get(c, 0)
                if any(
                    r.get(c)
                    for c in (
                        "nonfinite",
                        "out_of_range",
                        "h29_outside",
                        "radec_outside",
                    )
                ):
                    bad.append(
                        dict(
                            partition=k,
                            **{
                                c: r.get(c)
                                for c in (
                                    "nonfinite",
                                    "out_of_range",
                                    "h29_outside",
                                    "radec_outside",
                                    "examples",
                                )
                                if r.get(c)
                            },
                        )
                    )
                if time.time() < deadline:
                    nxt = next(it, None)
                    if nxt:
                        o, p = map(int, nxt[0].split("/"))
                        futs[
                            ex.submit(scan_file, nxt[1], ra_c, dec_c, o, p, membership)
                        ] = nxt[0]
                if done % 2000 == 0:
                    log(f"    scan {cat} {done}/{len(jobs)}")
                break
    return dict(
        files_scanned=done,
        files_total=len(jobs),
        complete=done == len(jobs),
        pixel_membership_from_radec=membership,
        totals=dict(tot),
        bad_partitions=bad[:200],
        n_bad_partitions=len(bad),
        seconds=round(time.time() - t0, 1),
    )


# ----------------------------------------------------------------------------- followup
# Tree-wide numbers for the issues the samples raised (stage ``followup``).


def _files(root: Path, cat: str) -> list[tuple[int, int, str]]:
    parts = list_partitions(root / cat / "dataset")["parts"]
    return sorted((o, p, f) for (o, p), v in parts.items() for f, _ in v["files"])


def _read_cols(path: str, cols: list[str]):
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    return pq.read_table(path, columns=[c for c in cols if c in names])


def _map_files(root: Path, cat: str, fn, workers: int) -> list:
    out = []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fn, o, p, f) for o, p, f in _files(root, cat)]
        for fut in as_completed(futs):
            try:
                out.append(fut.result())
            except Exception as exc:
                out.append(dict(error=repr(exc)[:200]))
    return out


def nn_other(tree, xyz) -> tuple[np.ndarray, np.ndarray]:
    """Separation (arcsec) and index of each point's nearest neighbour other than itself
    (with exact duplicates the KD-tree may return the twin first and the point second)."""
    d, ix = tree.query(xyz, k=3)
    self_ = np.arange(len(xyz))[:, None]
    col = np.argmax(ix != self_, axis=1)
    rows = np.arange(len(xyz))
    return sep_arcsec(d[rows, col]), ix[rows, col]


def ls_overlap(
    root: Path, gaia: GaiaReader | None, workers: int, cap: int = 400
) -> dict:
    """Legacy Surveys DR11 north/south overlap: rows at the same position twice."""
    import pyarrow.parquet as pq
    from scipy.spatial import cKDTree

    cat = "legacysurvey/dr11"

    def rel_range(o, p, f):
        md = pq.read_metadata(f)
        names = [md.schema.column(j).path for j in range(md.num_columns)]
        j = names.index("RELEASE")
        vals = set()
        for i in range(md.num_row_groups):
            s = md.row_group(i).column(j).statistics
            vals |= {s.min, s.max}
        return (o, p, f, sorted(vals))

    ranges = _map_files(root, cat, rel_range, workers)
    mixed = [r for r in ranges if isinstance(r, tuple) and len(r[3]) > 1]
    res = dict(n_files=len(ranges), n_mixed_release_partitions=len(mixed))
    rng = np.random.default_rng(11)
    pick = (
        mixed
        if len(mixed) <= cap
        else [mixed[i] for i in sorted(rng.choice(len(mixed), cap, replace=False))]
    )
    tot = Counter()
    per = []
    for o, p, f, _ in pick:
        t = _read_cols(f, ["RA", "DEC", "RELEASE", "REF_CAT", "REF_ID", "TYPE"])
        ra, dec = (
            to_np(t.column("RA")).astype(float),
            to_np(t.column("DEC")).astype(float),
        )
        rel = to_np(t.column("RELEASE"))
        xyz = unitvec(ra, dec)
        s, j = nn_other(cKDTree(xyz), xyz)
        cross = rel != rel[j]
        tw01 = (s < 0.1) & cross
        tw05 = (s < 0.5) & cross
        refc = to_np(t.column("REF_CAT")).astype(str)
        refi = to_np(t.column("REF_ID"))
        g2 = np.char.strip(refc) == "G2"
        same_gaia = tw01 & g2 & g2[j] & (refi == refi[j])
        n = len(ra)
        tot.update(
            rows=n,
            twin01=int(tw01.sum()),
            twin05=int(tw05.sum()),
            same_gaia_twin=int(same_gaia.sum()),
        )
        item = dict(
            partition=f"{o}/{p}",
            rows=n,
            frac_cross_release_twin_lt0p1=round(float(tw01.mean()), 4),
            frac_cross_release_twin_lt0p5=round(float(tw05.mean()), 4),
            release_counts=dict(Counter(map(str, rel)).most_common(3)),
        )
        if gaia is not None and tw01.any():
            import healpy as hp

            so = max(o, 6)
            sub = hp.ang2pix(2**so, ra[tw01], dec[tw01], nest=True, lonlat=True)
            vals, cnt = np.unique(sub, return_counts=True)
            sp = int(vals[np.argmax(cnt)])
            sel = hp.ang2pix(2**so, ra, dec, nest=True, lonlat=True) == sp
            g = gaia.rows_in(so, sp)
            if g is not None and g.num_rows and sel.sum() > 1:
                sidx = np.flatnonzero(sel)
                st = cKDTree(xyz[sidx])
                gd, gi = st.query(
                    unitvec(
                        to_np(g.column("ra")).astype(float),
                        to_np(g.column("dec")).astype(float),
                    ),
                    k=2,
                    distance_upper_bound=chord(MATCH_RADIUS),
                )
                ok = np.isfinite(gd)
                m2 = ok[:, 1]
                item["gaia_subpixel"] = f"{so}/{sp}"
                item["gaia_n"] = int(len(gd))
                item["gaia_frac_matched"] = round(float(ok[:, 0].mean()), 4)
                item["gaia_frac_ge2"] = round(float(m2.mean()), 4)
                if m2.any():
                    r1, r2 = sidx[gi[m2, 0]], sidx[gi[m2, 1]]
                    item["gaia_ge2_cross_release_frac"] = round(
                        float(np.mean(rel[r1] != rel[r2])), 4
                    )
                tot.update(gaia_n=int(len(gd)), gaia_ge2=int(m2.sum()))
        per.append(item)
    res.update(partitions_read=len(pick), totals=dict(tot), partitions=per[:60])
    if tot["rows"]:
        res["frac_rows_with_cross_release_twin_lt0p1_in_mixed"] = round(
            tot["twin01"] / tot["rows"], 4
        )
    res["mixed_partitions"] = [f"{o}/{p}" for o, p, _, _ in mixed][:500]
    return res


def ps1_ndetections(root: Path, workers: int) -> dict:
    def f(o, p, path):
        nd = to_np(_read_cols(path, ["nDetections"]).column("nDetections"))
        return dict(
            n=len(nd),
            eq0=int((nd == 0).sum()),
            eq1=int((nd == 1).sum()),
            eq2=int((nd == 2).sum()),
        )

    tot = Counter()
    for r in _map_files(root, "ps1/otmo", f, workers):
        if "error" not in r:
            tot.update(r)
    return dict(
        tot,
        frac_eq1=round(tot["eq1"] / max(tot["n"], 1), 4),
        frac_le1=round((tot["eq0"] + tot["eq1"]) / max(tot["n"], 1), 4),
    )


def pristine_dups(root: Path) -> dict:
    import pyarrow as pa

    tabs = [
        _read_cols(
            f,
            [
                "source_id",
                "RA",
                "Dec",
                "G_0",
                "CaHK_0",
                "FeH_Pristine",
                "merged_CASU_flag",
            ],
        )
        for _, _, f in _files(root, "pristine/dr2")
    ]
    t = pa.concat_tables(tabs)
    sid = to_np(t.column("source_id"))
    order = np.argsort(sid, kind="stable")
    s = sid[order]
    dup = np.flatnonzero(s[1:] == s[:-1])
    res = dict(
        rows=len(sid),
        duplicate_pairs=int(len(dup)),
        distinct_ids=int(len(np.unique(sid))),
    )
    if len(dup):
        a, b = order[dup], order[dup + 1]
        for c in ("G_0", "CaHK_0", "FeH_Pristine", "merged_CASU_flag", "RA"):
            v = to_np(t.column(c)).astype(float)
            res[f"dup_pairs_identical_{c}"] = int(
                np.sum((v[a] == v[b]) | (np.isnan(v[a]) & np.isnan(v[b])))
            )
        res["examples"] = [int(x) for x in s[dup][:5]]
    return res


def sdss_ids(root: Path, workers: int) -> dict:
    cols = ["id", "THING_ID", "RUN", "CAMCOL", "FIELD", "ID"]

    def f(o, p, path):
        t = _read_cols(path, cols)
        idv = to_np(t.column("id")).astype(np.int64)
        th = to_np(t.column("THING_ID")).astype(np.int64)
        key = (
            (to_np(t.column("RUN")).astype(np.int64) << 32)
            | (to_np(t.column("CAMCOL")).astype(np.int64) << 29)
            | (to_np(t.column("FIELD")).astype(np.int64) << 16)
            | to_np(t.column("ID")).astype(np.int64)
        )
        return dict(n=len(idv), id_eq_thing=int((idv == th).sum()), ids=idv, keys=key)

    rs = [
        r
        for r in _map_files(root, "sdss/photoprimary_dr13", f, workers)
        if "error" not in r
    ]
    ids = np.concatenate([r["ids"] for r in rs])
    keys = np.concatenate([r["keys"] for r in rs])
    n = len(ids)
    res = dict(rows=n, id_equals_THING_ID=int(sum(r["id_eq_thing"] for r in rs)))
    ids.sort()
    dup = ids[1:] == ids[:-1]
    res["duplicate_id_values"] = int(dup.sum())
    res["duplicate_id_examples"] = [int(x) for x in ids[1:][dup][:5]]
    res["negative_ids"] = int((ids < 0).sum())
    keys.sort()
    res["duplicate_run_camcol_field_ID"] = int((keys[1:] == keys[:-1]).sum())
    return res


def splus_overlap(root: Path, workers: int) -> dict:
    from scipy.spatial import cKDTree

    def f(o, p, path):
        t = _read_cols(path, ["ra", "dec", "field"])
        ra, dec = (
            to_np(t.column("ra")).astype(float),
            to_np(t.column("dec")).astype(float),
        )
        fld = to_np(t.column("field")).astype(str)
        xyz = unitvec(ra, dec)
        s, j = nn_other(cKDTree(xyz), xyz)
        other = fld != fld[j]
        return dict(
            n=len(ra),
            twin05_other_field=int(((s < 0.5) & other).sum()),
            twin10_other_field=int(((s < 1.0) & other).sum()),
            twin05_same_field=int(((s < 0.5) & ~other).sum()),
        )

    tot = Counter()
    for r in _map_files(root, "splus/dr4", f, workers):
        if "error" not in r:
            tot.update(r)
    return dict(
        tot,
        frac_rows_with_other_field_twin_lt0p5=round(
            tot["twin05_other_field"] / max(tot["n"], 1), 4
        ),
    )


def galex_zp(root: Path, workers: int) -> dict:
    def f(o, p, path):
        t = _read_cols(path, ["nuv_mag", "nuv_mag_auto", "fuv_mag", "fuv_mag_auto"])
        out = {}
        for b in ("nuv", "fuv"):
            m, a = (
                to_np(t.column(f"{b}_mag")).astype(float),
                to_np(t.column(f"{b}_mag_auto")).astype(float),
            )
            ok = (m > 5) & (m < 30) & (a > -30) & (a < 30)
            out[b] = (m[ok] - a[ok])[::50]
        return out

    res = {}
    rs = [
        r
        for r in _map_files(root, "galex/guvcat_ais_gr67", f, workers)
        if "error" not in r
    ]
    for b in ("nuv", "fuv"):
        dd = np.concatenate([r[b] for r in rs]) if rs else np.array([])
        if dd.size:
            res[f"{b}_mag_minus_mag_auto_pcts_1_50_99"] = [
                round(float(x), 4) for x in np.percentile(dd, [1, 50, 99])
            ]
            res[f"{b}_n_sampled"] = int(dd.size)
    return res


def euclid_flags(root: Path, workers: int) -> dict:
    cat = "euclid/dr1/mer_catalogue_wide_survey/euclid_dr1_reformatted_catalogs_mer_catalogue_wide_survey-hats"

    def f(o, p, path):
        t = _read_cols(path, ["vis_det", "spurious_flag", "det_quality_flag"])
        v = to_np(t.column("vis_det"))
        s = to_np(t.column("spurious_flag"))
        q = to_np(t.column("det_quality_flag"))
        return dict(
            n=len(v),
            vis_det0=int((v == 0).sum()),
            spurious1=int((s == 1).sum()),
            dqf_nonzero=int((q != 0).sum()),
            vis1_spur0_dqf0=int(((v == 1) & (s == 0) & (q == 0)).sum()),
        )

    tot = Counter()
    for r in _map_files(root, cat, f, workers):
        if "error" not in r:
            tot.update(r)
    return dict(tot)


def followup(root: Path, gaia: GaiaReader | None, workers: int) -> dict:
    out = {}
    for name, fn in (
        ("ls_dr11_overlap", lambda: ls_overlap(root, gaia, workers)),
        ("ps1_ndetections", lambda: ps1_ndetections(root, workers)),
        ("pristine_duplicates", lambda: pristine_dups(root)),
        ("sdss_ids", lambda: sdss_ids(root, workers)),
        ("splus_overlap", lambda: splus_overlap(root, workers)),
        ("galex_zero_point", lambda: galex_zp(root, workers)),
        ("euclid_wide_flags", lambda: euclid_flags(root, workers)),
    ):
        t0 = time.time()
        try:
            out[name] = fn()
        except Exception as exc:
            out[name] = dict(
                error=repr(exc)[:500], traceback=traceback.format_exc()[-1500:]
            )
        out[name]["seconds"] = round(time.time() - t0, 1)
        log(f"  followup {name} {out[name]['seconds']}s")
    return out


# ----------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="/arc/projects/hats")
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--samples",
        type=int,
        default=5,
        help="random partitions per catalogue (plus one per order)",
    )
    ap.add_argument(
        "--sample-gb",
        type=float,
        default=2.5,
        help="read budget per catalogue (GB on disk)",
    )
    ap.add_argument(
        "--profile-partitions",
        type=int,
        default=2,
        help="sampled partitions read with every column",
    )
    ap.add_argument(
        "--small-rows",
        type=float,
        default=2e8,
        help="full coordinate scan below this many rows",
    )
    ap.add_argument("--only", nargs="*", help="catalogue path substrings to audit")
    ap.add_argument("--stages", default="discovery,structure,samples,fullscan,compare")
    ap.add_argument("--max-hours", type=float, default=10.0)
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    out = Path(a.output).resolve()
    if out == root or root in out.parents:
        raise SystemExit(f"refusing to write under the audited root: {out}")
    out.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + a.max_hours * 3600
    stages = set(a.stages.split(","))
    log(f"audit root={root} output={out} stages={sorted(stages)}")
    disc = discover(root)
    dump(out / "discovery.json", disc)
    cats = [c for c in disc["catalogues"] if not a.only or any(s in c for s in a.only)]
    log(
        f"{len(disc['catalogues'])} catalogues, {len(disc['collections'])} collections, {len(disc['links'])} links, {len(disc['denied'])} denied; auditing {len(cats)}"
    )
    structs: dict[str, dict] = {}
    if stages & {"structure", "samples", "fullscan", "compare"}:
        for c in cats:
            try:
                s = audit_structure(root, c, a.workers, out / "partitions")
            except Exception as exc:
                s = dict(
                    catalogue=c,
                    error=repr(exc),
                    traceback=traceback.format_exc()[-2000:],
                )
            structs[c] = s
            dump(
                out / "structure" / (safe(c) + ".json"),
                {k: v for k, v in s.items() if k != "_parts"},
            )
            log(
                f"  structure {c}: rows={s.get('footer_rows_total')} hats_nrows={s.get('hats_nrows')} parts={s.get('n_partitions_on_disk')} {s.get('seconds')}s"
            )
    gaia = None
    if "samples" in stages and (root / GAIA_TREE / "partition_info.csv").is_file():
        gaia = GaiaReader(root)
    if "samples" in stages:
        for c in cats:
            s = structs.get(c, {})
            if (
                "_parts" not in s
                or time.time() > deadline
                or s["props"].get("dataproduct_type") in ("margin", "index")
            ):
                continue
            res = dict(catalogue=c, partitions=[])
            try:
                keys = choose_samples(c, s, a.samples, a.sample_gb)
                res["chosen"] = keys
                for n, k in enumerate(keys):
                    try:
                        res["partitions"].append(
                            sample_partition(
                                root, c, s, k, gaia, profile=n < a.profile_partitions
                            )
                        )
                    except Exception as exc:
                        res["partitions"].append(
                            dict(
                                partition=k,
                                error=repr(exc)[:500],
                                traceback=traceback.format_exc()[-1500:],
                            )
                        )
                    log(f"  sample {c} {k} done")
            except Exception as exc:
                res["error"] = repr(exc)
            dump(out / "samples" / (safe(c) + ".json"), res)
    if "fullscan" in stages:
        order = [
            c
            for c in cats
            if c not in FULL_SCAN
            and (structs.get(c, {}).get("footer_rows_total") or 0) < a.small_rows
        ]
        order += [c for c in FULL_SCAN if c in cats]
        for c in order:
            s = structs.get(c, {})
            if (
                "_parts" not in s
                or time.time() > deadline
                or s["props"].get("dataproduct_type") in ("margin", "index")
            ):
                continue
            log(f"  full scan {c}")
            try:
                r = full_scan(c, s, a.workers, deadline)
            except Exception as exc:
                r = dict(error=repr(exc))
            dump(out / "fullscan" / (safe(c) + ".json"), dict(catalogue=c, **r))
    if "followup" in stages:
        g = (
            GaiaReader(root)
            if (root / GAIA_TREE / "partition_info.csv").is_file()
            else None
        )
        dump(out / "followup.json", followup(root, g, a.workers))
    if "compare" in stages:
        comp = []
        for x, y in COMPARE:
            if (
                x in structs
                and y in structs
                and "_parts" in structs[x]
                and "_parts" in structs[y]
            ):
                px, py = structs[x]["_parts"], structs[y]["_parts"]
                diff = [
                    f"{k}: {px.get(k, {}).get('rows')} vs {py.get(k, {}).get('rows')}"
                    for k in sorted(set(px) | set(py))
                    if px.get(k, {}).get("rows") != py.get(k, {}).get("rows")
                ]
                comp.append(
                    dict(
                        a=x,
                        b=y,
                        same_partition_set=set(px) == set(py),
                        n_partitions=[len(px), len(py)],
                        rows=[
                            structs[x].get("footer_rows_total"),
                            structs[y].get("footer_rows_total"),
                        ],
                        n_row_differences=len(diff),
                        row_differences=diff[:50],
                    )
                )
        dump(out / "compare.json", comp)
    summary = []
    for c in cats:
        s = structs.get(c, {})
        summary.append(
            {
                k: s.get(k)
                for k in (
                    "catalogue",
                    "hats_nrows",
                    "footer_rows_total",
                    "rows_match_hats_nrows",
                    "n_partition_info",
                    "n_partitions_on_disk",
                    "n_missing_on_disk",
                    "n_extra_on_disk",
                    "order_range",
                    "layout",
                    "n_schema_variants",
                    "n_partitions_coord_stats_out_of_range",
                    "n_tiny_partitions_lt1000",
                    "missing_required",
                    "error",
                )
            }
        )
    dump(
        out / "summary.json",
        dict(
            utc=dt.datetime.now(dt.UTC).isoformat(), root=str(root), catalogues=summary
        ),
    )
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
