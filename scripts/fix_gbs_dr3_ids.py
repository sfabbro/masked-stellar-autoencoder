"""Recover the exact Gaia DR3 ids of the Gaia FGK benchmark stars (GBS v3).

``/arc/projects/k-pop/catalogues/gaia_fgk_benchmarks.parquet`` (Soubiran et
al. 2024) stores ``GaiaDR3`` as int64, but the values went through float64
upstream: every id equals its float64 rounding, so most are not Gaia DR3
source ids at all and the rest may point at another star. This script
recovers the exact id of each benchmark from the Gaia archive:

1. **Rounding superset.** The true id ``n`` satisfies ``float64(n) ==
   float64(stored)``: all Gaia DR3 sources with an id in
   ``[stored - ulp/2, stored + ulp/2]`` are candidates (one TAP query per star;
   a multi-range query or an uploaded-range join is unreliable on the archive).
2. **Identity checks.** A candidate is the benchmark when its position,
   propagated with its proper motion to 2016.0, 2000.0 or 1991.25 (the table's
   RA_ICRS/DE_ICRS epoch is not stated), lies within ``--max-sep`` arcsec of
   RA_ICRS/DE_ICRS, and its parallax agrees with the table's ``Plx`` within
   ``max(1 mas, 10 %)``. Exactly one candidate must pass. When the superset
   holds a single source with a 2-parameter solution (no parallax, no pm:
   very bright stars), it is accepted within ``--cone`` arcsec
   (``superset_2p``).
3. **Cone fallback.** Rows with no stored id, or with no candidate passing,
   take the brightest Gaia DR3 source within ``--cone`` arcsec (same epochs)
   whose parallax agrees with ``Plx``; rows without ``Plx`` are left
   unmatched rather than guessed.

Output (``--output``): the original columns, ``source_id`` (exact int64,
null when unmatched), ``GaiaDR3_float_rounded`` (the stored value),
``id_method`` (``superset``, ``superset_2p``, ``cone`` or ``unmatched``), ``n_candidates``,
``sep_arcsec``, ``gaia_g``, ``gaia_parallax`` and ``stored_id_is_exact``;
plus ``<output>.json`` with counts. ``training.anchor_labels.gaia_benchmarks``
reads this table and refuses float-rounded id columns.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

TAP = "https://gea.esac.esa.int/tap-server/tap/sync"
SOURCE = "/arc/projects/k-pop/catalogues/gaia_fgk_benchmarks.parquet"
OUTPUT = "/arc/projects/astroai/msa/catalogues/gaia_fgk_benchmarks_v3_dr3ids.parquet"
EPOCHS = (2016.0, 2000.0, 1991.25)
COLUMNS = "source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag"


def tap_query(adql: str, retries: int = 3) -> pd.DataFrame:
    import requests

    for attempt in range(retries):
        r = requests.post(
            TAP,
            data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", QUERY=adql),
            timeout=180,
        )
        if r.status_code == 200:
            return pd.read_csv(io.StringIO(r.text))
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"TAP query failed ({r.status_code}): {r.text[:300]}")


def rounding_interval(stored: int) -> tuple[int, int]:
    """Inclusive int64 range holding every id whose float64 rounding equals
    ``stored``'s (integer arithmetic: ``x -/+ ulp/2`` is not representable)."""
    x = np.float64(stored)
    centre, half = int(x), int(np.spacing(x)) // 2
    return centre - half, centre + half


def same_float(ids, stored: int) -> np.ndarray:
    return np.asarray(ids, np.int64).astype(np.float64) == np.float64(stored)


def separation_arcsec(ra1, de1, ra2, de2):
    ra1, de1, ra2, de2 = map(np.radians, (ra1, de1, ra2, de2))
    s = np.sin((de2 - de1) / 2) ** 2
    s = s + np.cos(de1) * np.cos(de2) * np.sin((ra2 - ra1) / 2) ** 2
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(s, 0, 1)))) * 3600


def min_epoch_separation(ra, de, cand: pd.DataFrame) -> np.ndarray:
    """Smallest separation over EPOCHS, Gaia positions moved by their pm."""
    pmra = np.nan_to_num(cand["pmra"].to_numpy(np.float64))
    pmde = np.nan_to_num(cand["pmdec"].to_numpy(np.float64))
    cra, cde = cand["ra"].to_numpy(np.float64), cand["dec"].to_numpy(np.float64)
    out = np.full(len(cand), np.inf)
    for epoch in EPOCHS:
        dt = epoch - 2016.0
        e_ra = cra + pmra * dt / 3.6e6 / np.cos(np.radians(cde))
        e_de = cde + pmde * dt / 3.6e6
        out = np.minimum(out, separation_arcsec(ra, de, e_ra, e_de))
    return out


def parallax_agrees(plx_gaia, plx_table) -> np.ndarray:
    plx_gaia = np.asarray(plx_gaia, np.float64)
    if not np.isfinite(plx_table):
        return np.ones(len(plx_gaia), bool)
    tol = max(1.0, 0.1 * abs(plx_table))
    return np.isfinite(plx_gaia) & (np.abs(plx_gaia - plx_table) <= tol)


def resolve_row(row, query=tap_query, max_sep=1.5, cone=10.0) -> dict:
    """Exact DR3 id of one benchmark row (see module docstring)."""
    ra, de = float(row["RA_ICRS"]), float(row["DE_ICRS"])
    plx = float(row["Plx"]) if pd.notna(row.get("Plx")) else np.nan
    stored = row.get("GaiaDR3")
    out = dict(
        source_id=None, id_method="unmatched", n_candidates=0, sep_arcsec=np.nan,
        gaia_g=np.nan, gaia_parallax=np.nan, stored_id_is_exact=False,
    )  # fmt: skip
    if pd.notna(stored):
        stored = int(stored)
        lo, hi = rounding_interval(stored)
        cand = query(
            f"SELECT {COLUMNS} FROM gaiadr3.gaia_source "
            f"WHERE source_id BETWEEN {lo} AND {hi}"
        )
        cand = cand[same_float(cand["source_id"], stored)]
        out["n_candidates"] = len(cand)
        if len(cand):
            sep = min_epoch_separation(ra, de, cand)
            ok = (sep <= max_sep) & parallax_agrees(cand["parallax"], plx)
            if ok.sum() == 1:
                i = int(np.flatnonzero(ok)[0])
                out.update(id_method="superset", sep_arcsec=float(sep[i]))
                out.update(_identity(cand, i, stored))
                return out
            two_param = ~np.isfinite(cand["parallax"].to_numpy(np.float64))
            if len(cand) == 1 and two_param[0] and sep[0] <= cone:
                # the only id with this rounding has a 2-parameter solution (no
                # parallax or pm: very bright stars), so neither the parallax
                # check nor pm propagation applies; accept within the cone
                out.update(id_method="superset_2p", sep_arcsec=float(sep[0]))
                out.update(_identity(cand, 0, stored))
                return out
    if not np.isfinite(plx):
        return out  # nothing independent to confirm a positional match with
    cand = query(
        f"SELECT {COLUMNS} FROM gaiadr3.gaia_source WHERE 1=CONTAINS("
        f"POINT(ra, dec), CIRCLE({ra}, {de}, {300.0 / 3600.0}))"
    )  # 300": a 10 arcsec/yr star moves < 260" between 1991.25 and 2016.0
    if len(cand):
        sep = min_epoch_separation(ra, de, cand)
        ok = (sep <= cone) & parallax_agrees(cand["parallax"], plx)
        if ok.any():
            g = np.where(ok, cand["phot_g_mean_mag"].to_numpy(np.float64), np.inf)
            i = int(np.argmin(g))  # brightest passing source
            out.update(id_method="cone", sep_arcsec=float(sep[i]))
            out.update(_identity(cand, i, stored if pd.notna(stored) else None))
    return out


def _identity(cand: pd.DataFrame, i: int, stored) -> dict:
    # read the id from its int64 column: a row Series would upcast it to float
    sid = int(cand["source_id"].to_numpy(np.int64)[i])
    return dict(
        source_id=sid,
        gaia_g=float(cand["phot_g_mean_mag"].to_numpy(np.float64)[i]),
        gaia_parallax=float(cand["parallax"].to_numpy(np.float64)[i]),
        stored_id_is_exact=stored is not None and sid == int(stored),
    )


def fix_table(t: pd.DataFrame, query=tap_query, **kw) -> pd.DataFrame:
    rows = [resolve_row(r, query=query, **kw) for _, r in t.iterrows()]
    # build the id column from the Python ints: a DataFrame of the row dicts
    # would hold None as NaN and round every id through float64 again
    exact = pd.array([r["source_id"] for r in rows], dtype="Int64")
    res = pd.DataFrame([{k: v for k, v in r.items() if k != "source_id"} for r in rows],
                       index=t.index)  # fmt: skip
    out = t.copy()
    out["GaiaDR3_float_rounded"] = t["GaiaDR3"]
    out = out.drop(columns=["GaiaDR3"])
    out.insert(0, "source_id", exact)
    for c in ("id_method", "n_candidates", "sep_arcsec", "gaia_g", "gaia_parallax",
              "stored_id_is_exact"):  # fmt: skip
        out[c] = res[c]
    return out


def summary(fixed: pd.DataFrame) -> dict:
    sid = fixed["source_id"]
    stored = fixed["GaiaDR3_float_rounded"]
    return dict(
        rows=int(len(fixed)),
        rows_with_stored_id=int(stored.notna().sum()),
        unique_stored_ids=int(stored.dropna().nunique()),
        recovered=int(sid.notna().sum()),
        unique_recovered=int(sid.dropna().nunique()),
        by_method={
            str(k): int(v) for k, v in fixed["id_method"].value_counts().items()
        },
        stored_id_was_exact=int(fixed["stored_id_is_exact"].sum()),
        stored_id_wrong_or_missing=int(
            (stored.notna() & ~fixed["stored_id_is_exact"].astype(bool)).sum()
        ),
        median_sep_arcsec=float(np.nanmedian(fixed["sep_arcsec"])),
        largest_sep_arcsec=float(np.nanmax(fixed["sep_arcsec"])),
        unmatched_names=[
            str(n).strip() or str(h)
            for n, h in zip(
                fixed.loc[sid.isna(), "Name"], fixed.loc[sid.isna(), "HD"], strict=True
            )
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--source", default=SOURCE)
    p.add_argument("--output", default=OUTPUT)
    p.add_argument("--max-sep", type=float, default=1.5)
    p.add_argument("--cone", type=float, default=10.0)
    a = p.parse_args()
    t = pd.read_parquet(a.source)
    fixed = fix_table(t, max_sep=a.max_sep, cone=a.cone)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fixed.to_parquet(out, index=False)
    info = dict(
        source=str(a.source),
        method=__doc__.split("\n\n")[1:4],
        max_sep_arcsec=a.max_sep,
        cone_arcsec=a.cone,
        **summary(fixed),
    )
    out.with_suffix(".json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "method"}, indent=2))


if __name__ == "__main__":
    main()
