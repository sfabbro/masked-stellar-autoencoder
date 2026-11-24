"""Competing catalogues on ARC, read on exact Gaia DR3 ids (qa.md §3).

Each loader returns a standard frame: ``source_id`` plus whichever of
``teff``, ``logg``, ``fe_h`` ([M/H] where that is what the catalogue gives),
``dist_mod`` and ``ebv`` it provides, and ``in_training`` where the catalogue
flags its own training stars (Andrae 2023). ``stage_competitors`` writes the
rows of a given id list to slim parquet files, so the comparison can run
away from ARC; ``load_competitor`` reads either form.

On ARC: Andrae, Rix & Chandra 2023 (XGBoost), SHBoost (Khalatyan+2024),
StarHorse 2023 spectroscopic runs (Queiroz+2023), Gaia DR3 GSP-Phot (Gaia
HATS tree, or the archive). Not on ARC (reported as missing): Zhang, Green &
Rix 2023; Survey of Surveys DR3; Bailer-Jones+2021; Bayestar19 / Edenhofer
2024; Ye+2025 FERRE-XP; Gaia Net.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd

KPOP = Path("/arc/projects/k-pop/catalogues")
SOURCES = {
    "andrae2023": KPOP / "table-1.fits.gz",
    "shboost": KPOP / "shboost",
    "starhorse2023": KPOP / "starhorse_ages",
    "gspphot": Path("/arc/projects/hats/gaia/dr3"),
}
# Caveats printed with the comparison (qa.md §3 comparator contract)
NOTES = {
    "andrae2023": "own training stars (in_training_sample) removed from the comparison",
    "starhorse2023": "spectroscopic StarHorse runs take the survey Teff, log g and [M/H] as "
    "inputs: agreement with survey references is partly by construction",
    "shboost": "trained on spectroscopic (StarHorse) labels of the same surveys; no "
    "training flag, so overlap with its training set is not removed",
    "gspphot": "Gaia DR3 GSP-Phot (Aeneas), independent of the survey labels",
}
MISSING = {
    "zhang_green_rix2023": "not on ARC (schema verified only)",
    "survey_of_surveys_dr3": "not on ARC",
    "bailer_jones2021": "no local copy (remote LSDB HATS only)",
    "bayestar19_edenhofer2024": "no 3-D dust map on ARC",
    "ye2025_ferre_xp": "not on ARC",
    "gaia_net": "Zenodo copy no longer on ARC",
}
STARHORSE_RUNS = ("apogee17", "galahdr3", "gesdr5", "lamostl7", "lamostm7", "ravedr6",
                  "rvs", "sdssdr12")  # fmt: skip
STANDARD = ("teff", "logg", "fe_h", "dist_mod", "ebv", "in_training")
R_V = 3.1  # A_V -> E(B-V) for SHBoost / StarHorse (descriptive comparison only)


def _dist_mod_kpc(d_kpc):
    d = np.asarray(d_kpc, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0, 5.0 * np.log10(d * 1000.0) - 5.0, np.nan)


def _dist_mod_pc(d_pc):
    d = np.asarray(d_pc, np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0, 5.0 * np.log10(d) - 5.0, np.nan)


def _as_bool(values) -> np.ndarray:
    """Booleans from a logical, integer or text column ('True' / 'False', 'T' / 'F').

    Andrae 2023 stores ``in_training_sample`` as 5-character text: casting the
    strings with ``bool`` would make every row True.
    """
    v = np.asarray(values)
    if v.dtype.kind in "SUO":
        t = np.char.lower(np.char.strip(v.astype(str)))
        return np.isin(t, ("true", "t", "1", "yes"))
    return v.astype(bool)


def _pick(cols, *names):
    low = {c.lower(): c for c in cols}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return None


def _filter_ids(frame: pd.DataFrame, ids) -> pd.DataFrame:
    if ids is None:
        return frame
    return frame[frame["source_id"].isin(np.asarray(ids, np.int64))]


def read_andrae(path, ids=None, chunk: int = 10_000_000) -> pd.DataFrame:
    from astropy.io import fits

    parts = []
    want = None if ids is None else np.unique(np.asarray(ids, np.int64))
    with fits.open(path, memmap=False) as h:
        d = h[1].data
        for s in range(0, len(d), chunk):
            b = d[s : s + chunk]
            sid = np.asarray(b["source_id"], np.int64)
            m = np.ones(len(sid), bool) if want is None else np.isin(sid, want)
            if m.any():
                parts.append(pd.DataFrame(dict(
                    source_id=sid[m],
                    teff=np.asarray(b["teff_xgboost"][m], np.float64),
                    logg=np.asarray(b["logg_xgboost"][m], np.float64),
                    fe_h=np.asarray(b["mh_xgboost"][m], np.float64),
                    in_training=_as_bool(b["in_training_sample"][m]),
                )))  # fmt: skip
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["source_id"])
    )


def read_shboost(path, ids=None) -> pd.DataFrame:
    """SHBoost 2024 parts, one file at a time (each file's own schema).

    Columns: ``xgb_logteff`` (log10 K), ``xgb_logg``, ``xgb_met``, ``xgb_av``,
    ``dist`` (kpc); ``source_id`` may be stored as the pandas index.
    """
    import pyarrow.parquet as pq

    files = (
        sorted(Path(path).glob("*.parquet")) if Path(path).is_dir() else [Path(path)]
    )
    want = None if ids is None else np.unique(np.asarray(ids, np.int64))
    wanted = ("source_id", "xgb_logteff", "xgb_logg", "xgb_met", "xgb_av", "dist")
    parts = []
    for f in files:
        names = pq.read_schema(f).names
        t = pq.read_table(f, columns=[c for c in wanted if c in names]).to_pandas()
        if "source_id" not in t.columns:
            t = t.reset_index()
        if "source_id" not in t.columns:
            raise KeyError(f"{f}: no source_id (columns {names[:12]})")
        sid = t["source_id"].to_numpy(np.int64)
        if want is not None:
            t = t[np.isin(sid, want)]
        if not len(t):
            continue
        out = pd.DataFrame({"source_id": t["source_id"].to_numpy(np.int64)})
        if "xgb_logteff" in t:
            out["teff"] = 10.0 ** t["xgb_logteff"].to_numpy(np.float64)
        if "xgb_logg" in t:
            out["logg"] = t["xgb_logg"].to_numpy(np.float64)
        if "xgb_met" in t:
            out["fe_h"] = t["xgb_met"].to_numpy(np.float64)
        if "dist" in t:
            out["dist_mod"] = _dist_mod_kpc(t["dist"])
        if "xgb_av" in t:
            out["ebv"] = t["xgb_av"].to_numpy(np.float64) / R_V
        parts.append(out)
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["source_id"])
    )


def read_starhorse(path, ids=None) -> pd.DataFrame:
    """StarHorse 2023 spectroscopic runs; a star in several runs keeps the first
    in ``STARHORSE_RUNS`` order."""
    from astropy.table import Table

    files = (
        [Path(path) / f"{r}.fits" for r in STARHORSE_RUNS]
        if Path(path).is_dir()
        else [Path(path)]
    )
    parts = []
    for f in files:
        if not f.exists():
            continue
        t = Table.read(f)
        sid_col = _pick(t.colnames, "GaiaEDR3", "source_id", "dr3_source_id")
        sid = np.asarray(t[sid_col], np.int64)
        m = (
            np.ones(len(sid), bool)
            if ids is None
            else np.isin(sid, np.asarray(ids, np.int64))
        )
        if not m.any():
            continue
        g = lambda n: np.asarray(t[n][m], np.float64) if n in t.colnames else np.nan  # noqa: E731
        parts.append(pd.DataFrame(dict(
            source_id=sid[m], teff=g("Teff50"), logg=g("logg50"), fe_h=g("Met50"),
            dist_mod=_dist_mod_kpc(g("Dist50")), ebv=g("AV50") / R_V, run=f.stem,
        )))  # fmt: skip
    if not parts:
        return pd.DataFrame(columns=["source_id"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("source_id")


GSPPHOT_COLUMNS = ("teff_gspphot", "logg_gspphot", "mh_gspphot", "distance_gspphot")


def _gspphot_frame(t: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(dict(
        source_id=t["source_id"].astype("int64"),
        teff=t["teff_gspphot"].astype("float64"), logg=t["logg_gspphot"].astype("float64"),
        fe_h=t["mh_gspphot"].astype("float64"), dist_mod=_dist_mod_pc(t["distance_gspphot"]),
    ))  # fmt: skip


def read_gspphot_hats(path, ids) -> pd.DataFrame:
    """GSP-Phot from the Gaia DR3 HATS tree, reading only the pixels of ``ids``."""
    import pyarrow.parquet as pq

    ids = np.unique(np.asarray(ids, np.int64))
    hp12 = ids >> 35
    info = pd.read_csv(Path(path) / "partition_info.csv")
    parts = []
    for order, npix in zip(info["Norder"], info["Npix"], strict=True):
        lo, hi = npix << (2 * (12 - order)), (npix + 1) << (2 * (12 - order))
        mine = ids[(hp12 >= lo) & (hp12 < hi)]
        if not len(mine):
            continue
        f = (
            Path(path)
            / "dataset"
            / f"Norder={order}"
            / f"Dir={npix // 10000 * 10000}"
            / f"Npix={npix}.parquet"
        )
        t = pq.read_table(f, columns=["source_id", *GSPPHOT_COLUMNS]).to_pandas()
        parts.append(_gspphot_frame(t[t.source_id.isin(mine)]))
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["source_id"])
    )


def read_gspphot_tap(ids, chunk: int = 50_000) -> pd.DataFrame:
    """GSP-Phot from the Gaia archive (sync uploads of ``chunk`` ids)."""
    import requests

    from masked_stellar_autoencoder.data.twomass_gaia import _votable_ids

    q = ("SELECT g.source_id, " + ", ".join(f"g.{c}" for c in GSPPHOT_COLUMNS)
         + " FROM tap_upload.ids AS u JOIN gaiadr3.gaia_source AS g ON g.source_id = u.source_id")  # fmt: skip
    ids = np.unique(np.asarray(ids, np.int64))
    parts = []
    for k in range(0, len(ids), chunk):
        r = requests.post(
            "https://gea.esac.esa.int/tap-server/tap/sync",
            data=dict(REQUEST="doQuery", LANG="ADQL", FORMAT="csv", UPLOAD="ids,param:t1", QUERY=q),
            files=dict(t1=("ids.vot", _votable_ids(ids[k : k + chunk]))), timeout=900,
        )  # fmt: skip
        r.raise_for_status()
        parts.append(_gspphot_frame(pd.read_csv(io.StringIO(r.text))))
    return (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=["source_id"])
    )


READERS = dict(andrae2023=read_andrae, shboost=read_shboost, starhorse2023=read_starhorse,
               gspphot=read_gspphot_hats)  # fmt: skip


def load_competitor(name: str, path, ids=None) -> pd.DataFrame:
    """A competitor on ``ids``: a staged slim parquet (``source_id`` + standard
    columns, from ``stage_competitors``) or the full catalogue at ``path``."""
    p = Path(path)
    if p.is_file() and p.suffix == ".parquet":
        import pyarrow.parquet as pq

        if set(pq.read_schema(p).names) <= {"source_id", "run", *STANDARD}:
            return _filter_ids(pd.read_parquet(p), ids)
    return READERS[name](path, ids)


def stage_competitors(ids, out_dir, sources=None, log=print) -> dict:
    """Write ``<name>.parquet`` (standard columns, rows of ``ids``) per source."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    for name, path in (sources or SOURCES).items():
        try:
            t = READERS[name](path, ids)
        except (OSError, FileNotFoundError, KeyError, ValueError) as exc:
            log(f"stage {name}: failed ({exc})")
            done[name] = dict(error=str(exc))
            continue
        t.to_parquet(out_dir / f"{name}.parquet", index=False)
        done[name] = dict(rows=int(len(t)), path=str(out_dir / f"{name}.parquet"))
        log(f"stage {name}: {len(t)} rows")
    return done


def compare(source_id, model: dict, ref: dict, comp: pd.DataFrame, keep=None) -> dict:
    """Model and competitor against a common reference, on exact-id overlap.

    ``model``/``ref`` map product -> arrays aligned with ``source_id``. Per
    product: bias / NMAD of model - ref and competitor - ref on the same stars,
    and the paired difference of median absolute errors (bootstrap 95 %).
    Products without a reference (distance, E) get the model - competitor
    difference only (descriptive: no catalogue is truth).
    """
    from .stats import paired_bootstrap, summary

    sid = np.asarray(source_id, np.int64)
    sel = np.ones(len(sid), bool) if keep is None else np.asarray(keep, bool)
    c = comp.drop_duplicates("source_id").set_index("source_id")
    hit = sel & np.isin(sid, c.index.to_numpy())
    out = dict(overlap=int(hit.sum()))
    if "in_training" in c.columns:
        it = np.zeros(len(sid), bool)
        it[hit] = c.loc[sid[hit], "in_training"].to_numpy(bool)
        out["overlap_in_competitor_training"] = int(it.sum())
        hit &= ~it  # competitor training stars are excluded from the comparison
    for p, m in model.items():
        if p not in c.columns:
            continue
        cv = np.full(len(sid), np.nan)
        cv[hit] = c.loc[sid[hit], p].to_numpy(np.float64)
        r = ref.get(p)
        if r is not None:
            ok = hit & np.isfinite(r) & np.isfinite(m) & np.isfinite(cv)
            out[p] = dict(
                model=summary(m[ok] - r[ok]), competitor=summary(cv[ok] - r[ok]),
                median_abs_err_model_minus_competitor=paired_bootstrap(m[ok] - r[ok], cv[ok] - r[ok]),
            )  # fmt: skip
        else:
            ok = hit & np.isfinite(m) & np.isfinite(cv)
            out[p] = dict(model_minus_competitor=summary(m[ok] - cv[ok]))
    return out
