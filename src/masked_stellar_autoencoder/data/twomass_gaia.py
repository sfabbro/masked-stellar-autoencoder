"""Gaia-keyed 2MASS J/H/Ks replacing the mismatched columns of the Andrae file.

The J, H and KS columns (and E_J, E_H, E_KS) of the Andrae 2023 XP input file
``sslset-realmags-full-052725.h5`` belong to other stars: Ks is uncorrelated
with G (correlation 0.016, found 2026-11-18, ``training/reddening_labels.py``),
yet they were encoder inputs and reconstruction targets in pretraining and in
every fine-tune table copied from that file.

**Root cause (2026-11-24).** The Andrae values are exactly those of the
"Gaia-keyed" ``/arc/projects/k-pop/catalogues/adql_matches/twomass.fits``
(469,001,402 rows; identical Ks for 4,086 of 4,086 stars compared), and that
file (like its source ``tmass_adql_match_files/tmass_*.fits``, 2023) is itself
mis-keyed: its ``tmass_oid`` column holds DR3
``tmass_psc_xsc_best_neighbour.clean_tmass_psc_xsc_oid``, but the photometry
was taken from the DR1 ``tmass_original_valid`` row with ``tmass_oid`` equal to
that number. Both are counters increasing with declination that coincide only
for the first rows near the south pole, so every later Gaia source got the
J/H/Ks of another 2MASS source at a similar declination (only the first 10 of
100 sampled rows are paired right). ``KNOWN_MISKEYED`` lists these files and
``lookup`` refuses them.

**Replacement.** The correct pairing goes through the 2MASS designation: DR3
``tmass_psc_xsc_best_neighbour.original_ext_source_id`` = DR1
``tmass_original_valid.designation``. ``archive_lookup`` queries it from the
Gaia archive for a list of ids (exact; ~40 stars/s per job, so for samples, QA
and validation sets; full tables need the bulk route of the full-wavelength
master, ``docs/gaia-master-data.md``). ``lookup`` reads any table keyed that
way (columns ``source_id, ph_qual, j_m, j_msigcom, h_m, h_msigcom, ks_m,
ks_msigcom``). Without a correct source, table builders mask the 2MASS columns
(``mask``): missing is better than wrong.

Per band, a magnitude is kept only for 2MASS photometric quality A-D (the
``ph_qual`` letter of the band) with a finite ``*_msigcom``; upper limits (U)
and unreliable fits (E, F, X) become NaN, i.e. missing, like a star without a
2MASS match. ``ks_g_check`` is the acceptance check: for a Gaia sample, Ks
must track G (Pearson correlation >= ``MIN_KS_G_CORRELATION``; G - Ks is a
colour of 0-4 mag against a G range of ~10 mag). On 2,000 random wide-binary
companions: Andrae Ks 0.010, archive Ks 0.948 (n = 1,679 usable).

Existing tables take the replacement as a row-order sidecar
(``scripts/build_twomass_sidecar.py``, passed as ``data.extra_columns``).
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np

# Mis-keyed (see module docstring): never a source of 2MASS photometry.
KNOWN_MISKEYED = (
    "/arc/projects/k-pop/catalogues/adql_matches/twomass.fits",
    "/arc/projects/k-pop/catalogues/adql_matches/tmass_adql_match_files",
)
TAP_ASYNC = "https://gea.esac.esa.int/tap-server/tap/async"
ARCHIVE_QUERY = (
    "SELECT u.source_id, b.original_ext_source_id AS designation, t.ph_qual, "
    "t.j_m, t.j_msigcom, t.h_m, t.h_msigcom, t.ks_m, t.ks_msigcom "
    "FROM tap_upload.ids AS u "
    "JOIN gaiadr3.tmass_psc_xsc_best_neighbour AS b ON b.source_id = u.source_id "
    "JOIN gaiadr1.tmass_original_valid AS t ON t.designation = b.original_ext_source_id"
)
# table column -> (2MASS column, ph_qual character index)
BANDS = {"J": ("j_m", 0), "H": ("h_m", 1), "KS": ("ks_m", 2)}
ERRORS = {"E_J": "j_msigcom", "E_H": "h_msigcom", "E_KS": "ks_msigcom"}
COLUMNS = ("J", "H", "KS", "E_J", "E_H", "E_KS")
GOOD_QUALITY = b"ABCD"
MIN_KS_G_CORRELATION = 0.8


def refuse_miskeyed(path) -> None:
    """Fail closed on the mis-keyed ADQL 2MASS files."""
    p = str(Path(path))
    if any(p == bad or p.startswith(bad + "/") for bad in KNOWN_MISKEYED) or (
        Path(p).name == "twomass.fits" and "adql_matches" in p
    ):
        raise ValueError(
            f"{p} is mis-keyed (clean_tmass_psc_xsc_oid joined to tmass_oid): its "
            "J/H/Ks belong to other stars; use archive_lookup or a designation-keyed "
            "table (data/twomass_gaia.py)"
        )


def _good(ph_qual: np.ndarray, index: int) -> np.ndarray:
    q = np.ascontiguousarray(np.asarray(ph_qual, dtype="S3"))
    letter = q.view("S1").reshape(len(q), 3)[:, index]
    return np.isin(letter, np.frombuffer(GOOD_QUALITY, dtype="S1"))


def twomass_columns(rows) -> dict[str, np.ndarray]:
    """J/H/KS and E_* (float32, NaN when not usable) from twomass.fits rows."""
    out = {}
    q = np.asarray(rows["ph_qual"])
    for name, (col, k) in BANDS.items():
        mag = np.asarray(rows[col], dtype="f4")
        err = np.asarray(rows[ERRORS["E_" + name]], dtype="f4")
        ok = _good(q, k) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
        out[name] = np.where(ok, mag, np.float32(np.nan))
        out["E_" + name] = np.where(ok, err, np.float32(np.nan))
    return out


class _open_rows:
    """Row-sliceable view of a source_id-keyed 2MASS table (FITS or parquet)."""

    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        if self.path.suffix == ".parquet" or self.path.is_dir():
            import pandas as pd

            t = pd.read_parquet(self.path).to_records(index=False)
            return t, len(t)
        from astropy.io import fits

        self._hdul = fits.open(self.path, memmap=True)
        data = self._hdul[1].data
        return data, len(data)

    def __exit__(self, *exc):
        if hasattr(self, "_hdul"):
            self._hdul.close()


def lookup(source_ids, path, chunk_rows: int = 5_000_000, log=None):
    """2MASS from a correctly keyed table for each requested ``source_id``.

    Returns ``(columns, info)``: ``columns`` maps J, H, KS, E_J, E_H, E_KS to
    float32 arrays aligned with ``source_ids`` (NaN without a usable match);
    ``info`` counts rows scanned, matched ids and duplicated ids (the first
    row wins).
    """
    refuse_miskeyed(path)
    sid = np.asarray(source_ids, dtype=np.int64)
    wanted = np.unique(sid[sid > 0])
    n = len(wanted)
    found = {c: np.full(n, np.nan, np.float32) for c in COLUMNS}
    seen = np.zeros(n, bool)
    duplicates = 0
    with _open_rows(path) as (data, total):
        for start in range(0, total, chunk_rows):
            block = data[start : start + chunk_rows]
            ids = np.asarray(block["source_id"], dtype=np.int64)
            if not n:
                break
            pos = np.searchsorted(wanted, ids).clip(max=n - 1)
            hit = wanted[pos] == ids
            if hit.any():
                p = pos[hit]
                rows = block[hit]
                first = np.zeros(len(p), bool)
                first[np.unique(p, return_index=True)[1]] = True
                keep = first & ~seen[p]  # the first row of an id wins
                duplicates += int((~keep).sum())
                cols = twomass_columns(rows)
                for c in COLUMNS:
                    found[c][p[keep]] = cols[c][keep]
                seen[p[keep]] = True
            if log is not None:
                log(f"twomass: {min(start + chunk_rows, total)}/{total} rows, "
                    f"{int(seen.sum())}/{n} ids matched")  # fmt: skip
    pos = np.searchsorted(wanted, sid).clip(max=max(n - 1, 0))
    present = (sid > 0) & (n > 0)
    if n:
        present &= wanted[pos] == sid
    out = {}
    for c in COLUMNS:
        col = np.full(len(sid), np.nan, np.float32)
        if n:
            col[present] = found[c][pos[present]]
        out[c] = col
    info = dict(
        path=str(path),
        rows_scanned=int(total),
        requested_ids=int(n),
        matched_ids=int(seen.sum()),
        duplicate_rows=int(duplicates),
        usable_ks=int(np.isfinite(out["KS"]).sum()),
    )
    return out, info


def replace_columns(fields, matrix: np.ndarray, columns: dict) -> list[str]:
    """Overwrite the J/H/KS/E_* columns of ``matrix`` (in ``fields`` order)."""
    fields = list(fields)
    replaced = []
    for c in COLUMNS:
        if c in fields:
            matrix[:, fields.index(c)] = columns[c]
            replaced.append(c)
    return replaced


def ks_g_check(g, ks_old=None, ks_new=None) -> dict:
    """Ks against G before and after the replacement (the acceptance check)."""

    def stats(ks):
        if ks is None:
            return None
        g_, k_ = np.asarray(g, np.float64), np.asarray(ks, np.float64)
        ok = np.isfinite(g_) & np.isfinite(k_)
        if ok.sum() < 3:
            return dict(n=int(ok.sum()), corr=float("nan"))
        c = g_[ok] - k_[ok]
        med = float(np.median(c))
        return dict(
            n=int(ok.sum()),
            corr=float(np.corrcoef(g_[ok], k_[ok])[0, 1]),
            g_minus_ks_median=med,
            g_minus_ks_nmad=float(1.4826 * np.median(np.abs(c - med))),
            frac_g_minus_ks_outside_m1_6=float(np.mean((c < -1) | (c > 6))),
        )

    old, new = stats(ks_old), stats(ks_new)
    out = dict(old=old, new=new, min_corr=MIN_KS_G_CORRELATION)
    if new is not None:
        out["passed"] = bool(
            np.isfinite(new["corr"]) and new["corr"] >= MIN_KS_G_CORRELATION
        )
    return out


def write_sidecar(path, source_ids, columns: dict, info: dict | None = None) -> None:
    """Row-order sidecar for ``data.extra_columns`` (source_id + J/H/KS/E_*)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        source_id=np.asarray(source_ids, np.int64),
        **{c: columns[c] for c in COLUMNS},
    )


def apply_sidecar(source_ids, fields, matrix: np.ndarray, sidecar) -> list[str]:
    """Replace J/H/KS/E_* in ``matrix`` from a sidecar, joining on source_id."""
    with np.load(sidecar, allow_pickle=False) as z:
        sid_s = np.asarray(z["source_id"], np.int64)
        cols = {c: np.asarray(z[c], np.float32) for c in COLUMNS if c in z.files}
    order = np.argsort(sid_s)
    sid_s = sid_s[order]
    sid = np.asarray(source_ids, np.int64)
    pos = np.searchsorted(sid_s, sid).clip(max=len(sid_s) - 1)
    hit = sid_s[pos] == sid
    if not hit.all():
        raise ValueError(
            f"{sidecar}: {int((~hit).sum())} source_ids missing from the sidecar"
        )
    aligned = {c: v[order][pos] for c, v in cols.items()}
    return replace_columns(fields, matrix, aligned)


def _votable_ids(ids) -> str:
    rows = "".join(f"<TR><TD>{int(i)}</TD></TR>" for i in ids)
    return (
        '<?xml version="1.0"?><VOTABLE version="1.3" '
        'xmlns="http://www.ivoa.net/xml/VOTable/v1.3"><RESOURCE><TABLE name="ids">'
        '<FIELD name="source_id" datatype="long"/><DATA><TABLEDATA>'
        f"{rows}</TABLEDATA></DATA></TABLE></RESOURCE></VOTABLE>"
    )


def _archive_job(ids, poll: float = 5.0, timeout: float = 3600.0):
    import pandas as pd
    import requests

    r = requests.post(
        TAP_ASYNC,
        data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", PHASE="RUN",
                  UPLOAD="ids,param:t1", QUERY=ARCHIVE_QUERY),
        files=dict(t1=("ids.vot", _votable_ids(ids))),
        allow_redirects=False, timeout=300,
    )  # fmt: skip
    job = r.headers.get("Location")
    if not job:
        raise RuntimeError(
            f"archive job not created: HTTP {r.status_code} {r.text[:300]}"
        )
    t0 = time.time()
    while True:
        phase = requests.get(job + "/phase", timeout=60).text.strip()
        if phase in ("COMPLETED", "ERROR", "ABORTED"):
            break
        if time.time() - t0 > timeout:
            raise TimeoutError(f"archive job {job} still {phase}")
        time.sleep(poll)
    if phase != "COMPLETED":
        raise RuntimeError(f"archive job {job}: {phase}")
    text = requests.get(job + "/results/result", timeout=600).text
    return pd.read_csv(io.StringIO(text), dtype={"designation": str, "ph_qual": str})


def archive_lookup(source_ids, chunk: int = 20_000, log=None, job=_archive_job):
    """Correct 2MASS per ``source_id`` from the Gaia archive (designation join).

    Same return contract as ``lookup``; ``info`` also counts archive jobs.
    """
    sid = np.asarray(source_ids, dtype=np.int64)
    wanted = np.unique(sid[sid > 0])
    parts = []
    for k in range(0, len(wanted), chunk):
        parts.append(job(wanted[k : k + chunk]))
        if log is not None:
            log(f"twomass archive: {min(k + chunk, len(wanted))}/{len(wanted)} ids")
    import pandas as pd

    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["source_id", "ph_qual", *"j_m", "j_msigcom", "h_m", "h_msigcom", "ks_m", "ks_msigcom"]
    )  # fmt: skip
    res = res.drop_duplicates("source_id")
    rid = res["source_id"].to_numpy(np.int64)
    order = np.argsort(rid)
    rid = rid[order]
    cols = twomass_columns(res.iloc[order].to_dict("list"))
    out = {c: np.full(len(sid), np.nan, np.float32) for c in COLUMNS}
    if len(rid):
        pos = np.searchsorted(rid, sid).clip(max=len(rid) - 1)
        hit = rid[pos] == sid
        for c in COLUMNS:
            out[c][hit] = cols[c][pos[hit]]
    info = dict(
        source="gaia archive: DR3 tmass_psc_xsc_best_neighbour.original_ext_source_id "
        "= DR1 tmass_original_valid.designation",
        requested_ids=int(len(wanted)),
        matched_ids=int(len(rid)),
        archive_jobs=len(parts),
        usable_ks=int(np.isfinite(out["KS"]).sum()),
    )
    return out, info


def masked_columns(n: int) -> dict:
    """All-NaN J/H/KS/E_* (no correct 2MASS source: missing, not wrong)."""
    return {c: np.full(n, np.nan, np.float32) for c in COLUMNS}


def resolve(source_ids, source, log=None):
    """Replacement columns from ``source``: 'archive', 'mask' or a table path."""
    if str(source) == "archive":
        return archive_lookup(source_ids, log=log)
    if str(source) == "mask":
        n = len(source_ids)
        return masked_columns(n), dict(source="mask", requested_ids=int(n))
    return lookup(source_ids, source, log=log)
