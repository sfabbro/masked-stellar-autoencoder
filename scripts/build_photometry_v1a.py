"""Build ``photometry_v1a``: the Gaia-keyed photometry (MSA-IN v1a, ``docs/programme/
data.md`` section 2.7) of every Gaia DR3 source, in the Gaia DR3 HATS partition layout.

One output partition per Gaia DR3 partition (``dataset/Norder=k/Dir=d/Npix=n.parquet``),
same rows in the same order as ``/arc/projects/hats/gaia/dr3``, with exactly the columns
the v11 column contract ``gaia_keyed_v1a`` (``src/masked_stellar_autoencoder/v11/
contract.yaml``) reads, in their native names: Gaia G/BP/RP magnitudes and flux-over-error,
parallax and proper motions with errors, ``sfd_ebv``, SDSS, SkyMapper, PS1, 2MASS and
CatWISE magnitudes, errors and quality flags, plus ``source_id`` and ``_healpix_29``.
A survey a star has no counterpart in is null (never a fill value); values that are
sentinels in the sources (SDSS -9999, PS1 -999, NaN) are null too. Quality rules, bright
limits and error caps are *not* applied here: the contract applies them when reading.

**2MASS is masked** (its columns are all null) unless ``--twomass`` names a correctly
keyed table: ``adql_matches/twomass.fits`` pairs every Gaia source with another 2MASS
star (``masked_stellar_autoencoder.data.twomass_gaia``; refused). Correct 2MASS comes
with the full-wavelength master's designation join.

**Pairing validation.** The other match tables come from the same 2023 ADQL work, so
before any is used its pairing is tested on a whole-sky sample (one random Gaia
partition per HEALPix order-1 pixel): residuals of G - x about a robust cubic in
BP - RP (x = PS1 r, SDSS r, SkyMapper r, CatWISE W1; a second band and declination
bands reported), against the same rows with x permuted (``pairing_metrics``). A table
that fails is masked whole; in the build every partition repeats the test and a
partition that fails is masked for that survey. Numbers in ``validation.json`` and the
manifest.

Sources (read only):

* Gaia DR3 HATS ``/arc/projects/hats/gaia/dr3`` (the anchor: rows, order, partitions);
* ``adql_matches`` (Gaia-archive best-neighbour joins keyed by ``source_id``):
  ``sdss.fits``, ``catwise.fits``
  (Gaia ``source_id`` is ``id2``; its ``source_id`` is the CatWISE name), and PS1 as
  ``pans_gaia_ids.fits`` (``source_id -> original_ext_source_id``) joined to the
  photometry files ``panstarrsdr1_matches/ids_pansdr1_<k>.fits.gz`` (plain FITS; file k
  holds the PS1 rows of id-table rows [k * 2,000,000, (k + 1) * 2,000,000), matched on
  ``obj_id``; ids missing from their chunk file are looked up in every file);
* SkyMapper DR4 HATS ``/arc/projects/hats/skymapper/dr4``: rows with ``gaia_dr3_id1``
  and ``gaia_dr3_dist1`` < ``--skymapper-max-sep`` (2 arcsec). Its ``gaia_dr3_id1`` went
  through float64 upstream (every id is float-exact; 38 % of them are no DR3 id, others
  belong to a neighbouring id), so the key is the rounded id: a Gaia source takes,
  among the rows with its rounded id, the one within 1" of its position propagated to
  the SkyMapper mean epoch (one-to-one, nearest first; ``skymapper_pairs``);
* SFD98 E(B-V) maps (``SFD_dust_4096_{ngp,sgp}.fits``, dustmaps' files) evaluated at the
  Gaia (l, b) exactly as ``dustmaps.sfd.SFDQuery`` does (ZEA WCS, bilinear, nearest at
  the edge); no 0.86 rescaling.

Method. The match tables are not sorted by ``source_id``, so each is first staged as a
``source_id``-sorted, de-duplicated column store (``.npy`` per column; a ``source_id``
that appears twice with different values is ambiguous and dropped, identical repeats
are kept once; SkyMapper keeps the nearest). The Gaia partitions are then streamed by a
process pool: each partition reads its Gaia columns, finds its rows in every store by
binary search (the store block of the partition's HEALPix range, plus individual
look-ups for the few sources whose ``source_id`` HEALPix index lies outside their
partition), evaluates SFD and writes one parquet file. Stages and partitions are
resumable: a store with ``sorted/meta.json`` and a partition with its stats file (same
fingerprint) are not rebuilt.

Checks (``manifest.json``; ``status: photometry_v1a_complete`` only when all pass): row
count = Gaia ``hats_nrows`` (1,811,709,771) and per partition; no duplicate
``source_id`` (per partition, and across partitions for the out-of-range sources); every
staged survey row lands on a Gaia source; every survey used passed the pairing
validation (partitions failing the gate counted); 2MASS Ks-G correlation when 2MASS is
used; SFD against dustmaps on a sample (and, informative, SkyMapper's ``ebmv_sfd``).
Per band: rows with a value, rows usable under the contract (quality rules applied by
``ContractReader``), against the data.md row counts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BUILD_VERSION = "photometry-v1a-3"
CONTRACT = "gaia_keyed_v1a"
GAIA_NROWS_DR3 = 1_811_709_771
HEALPIX_SHIFT = 35  # Gaia source_id // 2**35 = HEALPix level-12 nested index
LEVEL = 12
PS1_CHUNK_ROWS = 2_000_000
MAG_RANGE = (-10.0, 40.0)  # magnitudes outside (sentinels such as -9999, -999) -> null
ERR_RANGE = (0.0, 10.0)  # errors must be in (0, 10) mag, else null
ROW_GROUP = 131_072
KSG_EDGES = np.round(np.arange(-3.0, 10.0001, 0.05), 3)  # G - Ks histogram
SFD_EDGES = np.concatenate([[0.0], np.logspace(-3, 2.5, 56)])
DSFD_EDGES = np.array([-1e9, -0.1, -0.01, -0.001, -1e-4, 1e-4, 0.001, 0.01, 0.1, 1e9])
RATIO_EDGES = np.round(np.arange(0.5, 1.5001, 0.01), 2)  # ours / SkyMapper, E > 0.05

# pairing validation (pairing_metrics): survey band tested against Gaia G, BP - RP and
# the magnitude range used (unsaturated, S/N high enough for a colour relation)
PAIRING_BAND = {
    "sdss": ("r", (14.5, 20.5)),
    "skymapper": ("r_psf", (10.0, 19.0)),
    "ps1": ("r_mean_psf_mag", (14.5, 20.5)),
    "twomass": ("ks_m", (5.0, 15.0)),
    "catwise": ("w1mpro", (8.0, 16.5)),
}
PAIR_MIN_N = 50
PAIR_MAX_RATIO = 0.5
PAIR_MAX_OUTLIER = 0.3  # outlier share relative to the random-pairing null
PAIR_OUTLIER_MAG = 1.0
VALIDATE_ORDER = 1  # validation sample: one random Gaia partition per order-1 pixel

ADQL = Path("/arc/projects/k-pop/catalogues/adql_matches")
EXPECTED_ROWS = {  # data.md section 2.2 / 2.6 (rows of the Gaia-keyed match tables)
    "twomass": 469_001_402,
    "ps1": 951_450_474,
    "sdss": 119_663_765,
    "catwise": 678_990_741,
}

GAIA_READ = [
    "_healpix_29",
    "source_id",
    "ra",
    "dec",
    "l",
    "b",
    "phot_g_mean_mag",
    "phot_g_mean_flux_over_error",
    "phot_bp_mean_mag",
    "phot_bp_mean_flux_over_error",
    "phot_rp_mean_mag",
    "phot_rp_mean_flux_over_error",
    "parallax",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
]
GAIA_OUT = [c for c in GAIA_READ if c not in ("ra", "dec", "l", "b")]
GAIA_EPOCH_MJD = 57388.5  # J2016.0 (TCB); SkyMapper DR4 mean_epoch is an MJD
SKYMAPPER_POS_SEP = 1.0  # arcsec: Gaia (propagated to the SkyMapper epoch) vs SkyMapper


@dataclass(frozen=True)
class Col:
    name: str  # output (contract) column
    kind: str  # mag | err | flag | text | extra (staged for checks, not written)
    dtype: str  # staged / output numpy dtype
    src: str  # input column
    pair: str | None = None  # err: its magnitude column


@dataclass(frozen=True)
class Survey:
    name: str
    columns: tuple[Col, ...]
    dedup: str = "drop_ambiguous"  # drop_ambiguous | nearest (_sep column) | none
    # exact: the key is the Gaia DR3 source_id. float64: the key is the source_id
    # rounded through float64 upstream (SkyMapper DR4 HATS); a Gaia source takes the
    # positionally consistent row among those with its rounded id (skymapper_pairs).
    id_mode: str = "exact"
    spec: str = "1"  # staged-store layout version (a change re-stages the store)


def _band_cols(spec):
    """[(mag, err)] -> Col list (float32 magnitude and error)."""
    out = []
    for mag, err in spec:
        out += [
            Col(mag[0], "mag", "f4", mag[1]),
            Col(err[0], "err", "f4", err[1], mag[0]),
        ]
    return out


SURVEYS = {
    "sdss": Survey(
        "sdss",
        tuple(
            [Col(f"flags_{b}", "flag", "i8", f"flags_{b}") for b in "ugriz"]
            + _band_cols([((b, b), (f"err_{b}", f"err_{b}")) for b in "ugriz"])
        ),
    ),
    "skymapper": Survey(
        "skymapper",
        tuple(
            [Col("flags", "flag", "i4", "flags")]
            + _band_cols(
                [
                    ((f"{b}_psf", f"{b}_psf"), (f"e_{b}_psf", f"e_{b}_psf"))
                    for b in "uvgriz"
                ]
            )
            + [
                Col("_sep_arcsec", "extra", "f4", "gaia_dr3_dist1"),
                Col("_ebmv_sfd", "extra", "f4", "ebmv_sfd"),
                Col("_ra", "extra", "f8", "raj2000"),
                Col("_dec", "extra", "f8", "dej2000"),
                Col("_epoch", "extra", "f8", "mean_epoch"),
            ]
        ),
        dedup="none",
        id_mode="float64",
        spec="2",
    ),
    "ps1": Survey(
        "ps1",
        tuple(
            [Col("quality_flag", "flag", "i2", "quality_flag")]
            + _band_cols(
                [
                    ((f"{b}_mean_psf_mag",) * 2, (f"{b}_mean_psf_mag_error",) * 2)
                    for b in "grizy"
                ]
            )
        ),
    ),
    "twomass": Survey(
        "twomass",
        tuple(
            [Col("ph_qual", "text", "S3", "ph_qual")]
            + _band_cols(
                [((f"{b}_m",) * 2, (f"{b}_msigcom",) * 2) for b in ("j", "h", "ks")]
            )
        ),
    ),
    "catwise": Survey(
        "catwise",
        tuple(
            _band_cols(
                [
                    (("w1mpro",) * 2, ("w1sigmpro",) * 2),
                    (("w2mpro",) * 2, ("w2sigmpro",) * 2),
                ]
            )
        ),
    ),
}
SURVEY_ORDER = ("sdss", "skymapper", "ps1", "twomass", "catwise")


def output_columns() -> list[str]:
    cols = list(GAIA_OUT) + ["sfd_ebv"]
    for s in SURVEY_ORDER:
        cols += [c.name for c in SURVEYS[s].columns if c.kind != "extra"]
    return cols


@dataclass
class Inputs:
    gaia: Path = Path("/arc/projects/hats/gaia/dr3")
    # 2MASS: "mask" (default) or a correctly keyed table. adql_matches/twomass.fits is
    # mis-keyed (masked_stellar_autoencoder.data.twomass_gaia) and refused.
    twomass: str = "mask"
    sdss: Path = ADQL / "sdss.fits"
    catwise: Path = ADQL / "catwise.fits"
    ps1_ids: Path = ADQL / "pans_gaia_ids.fits"
    ps1_phot: Path = ADQL / "panstarrsdr1_matches"
    skymapper: Path = Path("/arc/projects/hats/skymapper/dr4")
    sfd: Path = Path("/arc/projects/astroai/msa/catalogues/sfd")


@dataclass
class Options:
    skymapper_max_sep: float = 2.0
    ps1_chunk_rows: int = PS1_CHUNK_ROWS
    fits_chunk_rows: int = 4_000_000
    extra: dict = field(default_factory=dict)


# ── small utilities ──────────────────────────────────────────────────────────


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default) + "\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def _md5(path: Path, block: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log(msg: str) -> None:
    print(f"[{dt.datetime.now(dt.UTC):%H:%M:%S}] {msg}", flush=True)


# ── HATS layout ─────────────────────────────────────────────────────────────


def hats_partitions(root: Path) -> list[tuple[int, int]]:
    lines = (Path(root) / "partition_info.csv").read_text().split()
    head = lines[0].split(",")
    io, ip = head.index("Norder"), head.index("Npix")
    return [(int(f[io]), int(f[ip])) for f in (ln.split(",") for ln in lines[1:])]


def partition_rel(order: int, npix: int) -> Path:
    return (
        Path(f"Norder={order}")
        / f"Dir={npix // 10000 * 10000}"
        / f"Npix={npix}.parquet"
    )


def partition_files(root: Path, order: int, npix: int) -> list[Path]:
    rel = partition_rel(order, npix)
    for base in (Path(root) / "dataset", Path(root)):
        f = base / rel
        if f.is_file():
            return [f]
        d = f.with_suffix("")
        if d.is_dir():
            return sorted(d.glob("*.parquet"))
    return []


def pixel_range(order: int, npix: int) -> tuple[int, int]:
    """[lo, hi) of HEALPix level-12 nested indices covered by a partition."""
    s = 2 * (LEVEL - order)
    return npix << s, (npix + 1) << s


def id_range(order: int, npix: int) -> tuple[int, int]:
    lo, hi = pixel_range(order, npix)
    return lo << HEALPIX_SHIFT, hi << HEALPIX_SHIFT


def hats_properties(root: Path) -> dict[str, str]:
    p = Path(root) / "properties"
    if not p.exists():
        p = Path(root) / "hats.properties"
    out = {}
    for line in p.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


# ── FITS binary tables (fixed-width rows, read directly) ─────────────────────

_FITS_TYPES = {"K": ">i8", "J": ">i4", "I": ">i2", "E": ">f4", "D": ">f8", "B": "u1"}


@dataclass(frozen=True)
class FitsLayout:
    offset: int
    nrows: int
    dtype: np.dtype


def fits_layout(path: Path) -> FitsLayout:
    """Byte layout of the first binary-table extension (no scaling, no heap)."""
    from astropy.io import fits

    with open(path, "rb") as fh:
        magic = fh.read(6)
    if magic != b"SIMPLE":
        raise ValueError(f"{path}: not an uncompressed FITS file")
    with fits.open(path, memmap=True, lazy_load_hdus=True) as hdul:
        h = hdul[1].header
        offset = hdul[1].fileinfo()["datLoc"]
    if h.get("XTENSION") != "BINTABLE" or h.get("PCOUNT", 0):
        raise ValueError(f"{path}: expected a binary table without heap")
    fields = []
    for i in range(1, h["TFIELDS"] + 1):
        if h.get(f"TSCAL{i}", 1) != 1 or h.get(f"TZERO{i}", 0) != 0:
            raise ValueError(f"{path}: scaled column {h[f'TTYPE{i}']}")
        form = h[f"TFORM{i}"].strip()
        rep = int(form[:-1] or 1)
        code = form[-1]
        name = h[f"TTYPE{i}"].strip()
        if code == "A":
            fields.append((name, f"S{rep}"))
        elif code in _FITS_TYPES:
            fields.append(
                (name, _FITS_TYPES[code])
                if rep == 1
                else (name, _FITS_TYPES[code], (rep,))
            )
        else:
            raise ValueError(f"{path}: unsupported TFORM {form}")
    dtype = np.dtype(fields)
    if dtype.itemsize != h["NAXIS1"]:
        raise ValueError(f"{path}: row size {dtype.itemsize} != NAXIS1 {h['NAXIS1']}")
    n = int(h["NAXIS2"])
    if os.path.getsize(path) < offset + n * dtype.itemsize:
        raise ValueError(f"{path}: truncated ({n} rows declared)")
    return FitsLayout(int(offset), n, dtype)


def read_fits_rows(path: Path, lay: FitsLayout, lo: int, hi: int) -> np.ndarray:
    buf = np.empty(hi - lo, lay.dtype)
    mv = memoryview(buf.view(np.uint8).reshape(-1))
    with open(path, "rb", buffering=0) as fh:
        fh.seek(lay.offset + lo * lay.dtype.itemsize)
        got = 0
        while got < len(mv):
            k = fh.readinto(mv[got:])
            if not k:
                raise OSError(f"{path}: short read")
            got += k
    return buf


# ── staged column stores ─────────────────────────────────────────────────────


def npy_create(path: Path, dtype, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.dtype(dtype), shape=(n,))
    del mm


def npy_write(path: Path, start: int, arr: np.ndarray) -> None:
    """Write ``arr`` at row ``start`` of an existing 1-d .npy file (positional write)."""
    mm = np.load(path, mmap_mode="r")
    off, dtype = mm.offset, mm.dtype
    del mm
    data = np.ascontiguousarray(arr, dtype=dtype).tobytes()
    fd = os.open(path, os.O_WRONLY)
    try:
        pos, view = off + start * dtype.itemsize, memoryview(data)
        while view:
            k = os.pwrite(fd, view, pos)
            view, pos = view[k:], pos + k
    finally:
        os.close(fd)


def store_columns(survey: Survey) -> list[tuple[str, str]]:
    return [("source_id", "i8")] + [(c.name, c.dtype) for c in survey.columns]


def unsorted_dir(staging: Path, name: str) -> Path:
    return staging / name / "unsorted"


def sorted_dir(staging: Path, name: str) -> Path:
    return staging / name / "sorted"


def store_meta(staging: Path, name: str) -> dict | None:
    p = sorted_dir(staging, name) / "meta.json"
    if p.exists():
        m = json.loads(p.read_text())
        if m.get("status") == "complete" and m.get("spec", "1") == SURVEYS[name].spec:
            return m
    return None


# ── stage 1: fill unsorted stores (pool tasks) ───────────────────────────────


def task_fits_chunk(t: dict) -> dict:
    """Rows [lo, hi) of a Gaia-keyed FITS match table -> unsorted store rows [lo, hi)."""
    lay = fits_layout(t["path"])
    rec = read_fits_rows(t["path"], lay, t["lo"], t["hi"])
    out = Path(t["out"])
    sid = rec[t["sid"]].astype(np.int64)
    bad = int((sid <= 0).sum())  # no Gaia id: dropped when sorted (counted there)
    npy_write(out / "source_id.npy", t["lo"], np.where(sid > 0, sid, -1))
    for name, src, dtype in t["cols"]:
        npy_write(out / f"{name}.npy", t["lo"], rec[src].astype(dtype))
    (out / "done" / t["key"]).touch()
    return dict(survey=t["survey"], key=t["key"], rows=t["hi"] - t["lo"], bad_ids=bad)


def _ps1_lookup(obj: np.ndarray, ext: np.ndarray):
    if len(obj) > 1 and np.any(obj[1:] < obj[:-1]):
        order = np.argsort(obj, kind="stable")
    else:
        order = None
    s = obj if order is None else obj[order]
    j = np.searchsorted(s, ext).clip(max=max(len(s) - 1, 0))
    hit = (s[j] == ext) if len(s) else np.zeros(len(ext), bool)
    return (j if order is None else order[j]), hit


def task_ps1_chunk(t: dict) -> dict:
    """Id-table rows of chunk k joined to photometry file k (on obj_id)."""
    ids_lay = fits_layout(t["ids"])
    ids = read_fits_rows(t["ids"], ids_lay, t["lo"], t["hi"])
    sid = ids["source_id"].astype(np.int64)
    ext = ids["original_ext_source_id"].astype(np.int64)
    out = Path(t["out"])
    if t["phot"] is not None:
        lay = fits_layout(t["phot"])
        ph = read_fits_rows(t["phot"], lay, 0, lay.nrows)
        j, hit = _ps1_lookup(ph["obj_id"].astype(np.int64), ext)
    else:
        ph, j, hit = None, np.zeros(len(ext), np.int64), np.zeros(len(ext), bool)
    npy_write(out / "source_id.npy", t["lo"], np.where(hit, sid, -1))
    for name, src, dtype in t["cols"]:
        if ph is None:
            v = np.zeros(len(ext), dtype)
        else:
            v = ph[src][j].astype(dtype)
        npy_write(out / f"{name}.npy", t["lo"], v)
    miss = np.flatnonzero(~hit)
    um = out / "unmatched" / f"{t['key']}.npz"
    um.parent.mkdir(parents=True, exist_ok=True)
    tmp = um.with_name(um.name + ".tmp.npz")
    np.savez(tmp, row=t["lo"] + miss, source_id=sid[miss], ext=ext[miss])
    os.replace(tmp, um)
    (out / "done" / t["key"]).touch()
    return dict(survey="ps1", key=t["key"], rows=t["hi"] - t["lo"], unmatched=len(miss))


def task_ps1_search(t: dict) -> dict:
    """Look the chunk-unmatched PS1 ids up in one photometry file."""
    want = np.load(t["want"])
    lay = fits_layout(t["phot"])
    ph = read_fits_rows(t["phot"], lay, 0, lay.nrows)
    obj = ph["obj_id"].astype(np.int64)
    pos = np.searchsorted(want, obj).clip(max=len(want) - 1)
    hit = np.flatnonzero(want[pos] == obj)
    return dict(
        ext=obj[hit],
        **{name: ph[src][hit].astype(dtype) for name, src, dtype in t["cols"]},
    )


def task_skymapper_file(t: dict) -> dict:
    """One SkyMapper DR4 parquet file -> unsorted store rows (source id 0 = no match)."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    tab = pq.read_table(
        t["path"],
        columns=["gaia_dr3_id1", "gaia_dr3_dist1"]
        + [s for _, s, _ in t["cols"] if s not in ("gaia_dr3_dist1",)],
    )
    if tab.num_rows != t["hi"] - t["lo"]:
        raise ValueError(
            f"{t['path']}: {tab.num_rows} rows, expected {t['hi'] - t['lo']}"
        )
    gid = (
        pc.fill_null(tab["gaia_dr3_id1"], 0)
        .to_numpy(zero_copy_only=False)
        .astype(np.int64)
    )
    dist = tab["gaia_dr3_dist1"].to_numpy(zero_copy_only=False).astype(np.float64)
    keep = (gid > 0) & np.isfinite(dist) & (dist < t["max_sep"])
    out = Path(t["out"])
    npy_write(out / "source_id.npy", t["lo"], np.where(keep, gid, 0))
    for name, src, dtype in t["cols"]:
        a = tab[src]
        if np.dtype(dtype).kind == "f":
            v = a.to_numpy(zero_copy_only=False).astype(dtype)
        else:
            v = pc.fill_null(a, 0).to_numpy(zero_copy_only=False).astype(dtype)
        npy_write(out / f"{name}.npy", t["lo"], v)
    (out / "done" / t["key"]).touch()
    return dict(
        survey="skymapper", key=t["key"], rows=tab.num_rows, kept=int(keep.sum())
    )


# ── stage 1b: sort + de-duplicate one store ──────────────────────────────────


def _differs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a != b
    if a.dtype.kind == "f":
        d &= ~(np.isnan(a) & np.isnan(b))
    return d


def task_sort_store(t: dict) -> dict:
    """Unsorted store -> ``sorted/`` (ascending unique source_id) + ``meta.json``."""
    name, staging = t["survey"], Path(t["staging"])
    survey = SURVEYS[name]
    u, s_dir = unsorted_dir(staging, name), sorted_dir(staging, name)
    t0 = time.monotonic()
    sid = np.load(u / "source_id.npy")
    n_in = len(sid)
    valid = np.flatnonzero(sid > 0)
    n_drop = n_in - len(valid)
    if survey.dedup == "nearest":
        sep = np.load(u / "_sep_arcsec.npy")[valid]
        valid = valid[np.argsort(sep, kind="stable")]
        del sep
    sv = sid[valid]
    o = np.argsort(sv, kind="stable")
    del sv
    order = valid[o]
    del valid, o
    s = sid[order]
    del sid
    first = np.ones(len(s), bool)
    first[1:] = s[1:] != s[:-1]
    n_dup_rows = int((~first).sum())
    n_amb_ids = 0
    n_identical = 0
    if n_dup_rows and survey.dedup == "none":
        keep = np.ones(len(s), bool)
    elif n_dup_rows and survey.dedup == "nearest":
        keep = first
    elif n_dup_rows:
        rep = np.flatnonzero(~first)  # rows equal (in source_id) to the previous row
        run = np.cumsum(first) - 1
        differ = np.zeros(len(s), bool)
        for c in survey.columns:
            col = np.load(u / f"{c.name}.npy", mmap_mode="r")
            a = np.asarray(col[order[rep]])
            b = np.asarray(col[order[rep - 1]])
            differ[rep] |= _differs(a, b)
        amb = np.zeros(int(run[-1]) + 1, bool)
        amb[run[differ]] = True
        keep = first & ~amb[run]
        n_amb_ids = int(amb.sum())
        n_identical = int((~first & ~amb[run]).sum())
        del run, differ, amb, rep
    else:
        keep = first
    del first
    sel = order[keep]
    del order
    tmp = s_dir.with_name("sorted.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    np.save(tmp / "source_id.npy", s[keep])
    n_out = int(keep.sum())
    del s, keep
    counts = {}
    for c in survey.columns:
        col = np.load(u / f"{c.name}.npy")[sel]
        np.save(tmp / f"{c.name}.npy", col)
        if c.kind == "mag":
            counts[c.name] = int(_valid_mag(col).sum())
        del col
    meta = dict(t.get("info", {}))
    meta.update(
        status="complete",
        survey=name,
        rows=n_out,
        input_rows=n_in,
        dropped_no_match=n_drop,
        duplicate_rows=n_dup_rows,
        identical_repeats_dropped=n_identical,
        ambiguous_ids_dropped=n_amb_ids,
        dedup=survey.dedup,
        id_mode=survey.id_mode,
        spec=survey.spec,
        valid_mag_counts=counts,
        columns=[n for n, _ in store_columns(survey)],
        sort_seconds=time.monotonic() - t0,
    )
    _save_json(tmp / "meta.json", meta)
    if s_dir.exists():
        shutil.rmtree(s_dir)
    os.replace(tmp, s_dir)
    if t.get("clean_unsorted", True):
        shutil.rmtree(u, ignore_errors=True)
    return meta


# ── validity rules ───────────────────────────────────────────────────────────


def _valid_mag(v: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.isfinite(v) & (v > MAG_RANGE[0]) & (v < MAG_RANGE[1])


def _valid_err(e: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.isfinite(e) & (e > ERR_RANGE[0]) & (e < ERR_RANGE[1])


# ── SFD ─────────────────────────────────────────────────────────────────────


class SFD:
    """SFD98 E(B-V) at galactic (l, b), as ``dustmaps.sfd.SFDQuery.query(order=1)``."""

    def __init__(self, map_dir: Path):
        from astropy.io import fits
        from astropy.wcs import WCS

        self.maps = {}
        for pole in ("ngp", "sgp"):
            with fits.open(Path(map_dir) / f"SFD_dust_4096_{pole}.fits") as h:
                self.maps[pole] = (
                    np.asarray(h[0].data, dtype=np.float32),
                    WCS(h[0].header),
                )

    def __call__(self, l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates

        out = np.full(len(l_deg), np.nan, np.float32)
        for pole in ("ngp", "sgp"):
            m = (b_deg >= 0) if pole == "ngp" else (b_deg < 0)
            if np.any(m):
                data, w = self.maps[pole]
                x, y = w.wcs_world2pix(l_deg[m], b_deg[m], 0)
                out[m] = map_coordinates(data, [y, x], order=1, mode="nearest")
        return out


# ── stage 2: one Gaia partition ──────────────────────────────────────────────

_W: dict = {}  # per-worker cache: stores, SFD maps, contract reader


def _worker_state(cfg: dict) -> dict:
    if _W.get("key") != cfg["fingerprint"]:
        _W.clear()
        _W["key"] = cfg["fingerprint"]
        _W["sfd"] = SFD(Path(cfg["sfd"]))
        stores = {}
        for name in cfg["active"]:
            d = sorted_dir(Path(cfg["staging"]), name)
            stores[name] = {
                c: np.load(d / f"{c}.npy", mmap_mode="r")
                for c, _ in store_columns(SURVEYS[name])
            }
        _W["stores"] = stores
        from masked_stellar_autoencoder.v11.schema import ContractReader, load_registry

        reg = load_registry()
        _W["reader"] = ContractReader(reg, CONTRACT)
        _W["band_names"] = [reg.bands[i].name for i, *_ in _W["reader"].plan]
    return _W


def lookup(store_sid: np.ndarray, query: np.ndarray, lo_id: int, hi_id: int):
    """Store positions of ``query`` ids (-1 absent): the store block [a, b) of the id
    range [lo_id, hi_id) is read once; ids outside it are searched individually."""
    a = int(np.searchsorted(store_sid, lo_id, side="left"))
    b = int(np.searchsorted(store_sid, hi_id, side="left"))
    block = np.asarray(store_sid[a:b])
    pos = np.full(len(query), -1, np.int64)
    inside = (query >= lo_id) & (query < hi_id)
    q = query[inside]
    if len(block) and len(q):
        j = np.searchsorted(block, q).clip(max=len(block) - 1)
        pos[inside] = np.where(block[j] == q, a + j, -1)
    out = np.flatnonzero(~inside)
    n = len(store_sid)
    if len(out) and n:
        q = query[out]
        j = np.searchsorted(store_sid, q).clip(max=n - 1)
        pos[out] = np.where(np.asarray(store_sid[j]) == q, j, -1)
    return pos, a, b


def gather(col: np.ndarray, pos: np.ndarray, a: int, b: int) -> np.ndarray:
    out = np.zeros(len(pos), col.dtype)
    m_in = (pos >= a) & (pos < b)
    if m_in.any():
        out[m_in] = np.asarray(col[a:b])[pos[m_in] - a]
    m_out = np.flatnonzero((pos >= 0) & ~m_in)
    if len(m_out):
        out[m_out] = np.asarray(col[pos[m_out]])
    return out


def _arrow_text(values: np.ndarray, valid: np.ndarray):
    import pyarrow as pa
    import pyarrow.compute as pc

    width = values.dtype.itemsize
    fsb = pa.FixedSizeBinaryArray.from_buffers(
        pa.binary(width),
        len(values),
        [None, pa.py_buffer(np.ascontiguousarray(values).tobytes())],
    )
    s = fsb.cast(pa.binary()).cast(pa.string())
    return pc.if_else(pa.array(valid), s, pa.scalar(None, pa.string()))


def float64_key(source_id: np.ndarray) -> np.ndarray:
    """The source_id as stored by a table that passed it through float64."""
    return np.asarray(source_id, np.int64).astype(np.float64).astype(np.int64)


def lookup_multi(store_key: np.ndarray, qkey: np.ndarray, lo: int, hi: int):
    """All (query index, store position) pairs with equal keys; the store may repeat
    keys. The store block [lo, hi) is read once, other keys are searched one by one."""
    a = int(np.searchsorted(store_key, lo, side="left"))
    b = int(np.searchsorted(store_key, hi, side="left"))
    block = np.asarray(store_key[a:b])
    inside = (qkey >= lo) & (qkey < hi)
    qs, ps = [], []
    for sel, arr, off in (
        (np.flatnonzero(inside), block, a),
        (np.flatnonzero(~inside), store_key, 0),
    ):
        if not len(sel) or not len(arr):
            continue
        q = qkey[sel]
        left = np.searchsorted(arr, q, side="left")
        right = np.searchsorted(arr, q, side="right")
        cnt = right - left
        tot = int(cnt.sum())
        starts = np.repeat(left, cnt)
        offs = np.arange(tot) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        qs.append(np.repeat(sel, cnt))
        ps.append(off + starts + offs)
    if not qs:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(qs), np.concatenate(ps).astype(np.int64)


def _sep_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    r1, d1, r2, d2 = (
        np.radians(np.asarray(x, np.float64)) for x in (ra1, dec1, ra2, dec2)
    )
    h = (
        np.sin((d2 - d1) / 2) ** 2
        + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    )
    return np.degrees(2 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))) * 3600.0


def skymapper_pairs(
    store, sid, ra, dec, pmra, pmdec, lo_id, hi_id, max_sep=SKYMAPPER_POS_SEP
):
    """Gaia row -> SkyMapper store position (-1 none) for a store keyed by the
    float64-rounded Gaia id: among the rows sharing a Gaia source's rounded id (several
    Gaia sources can round to one key), accept pairs whose SkyMapper position lies
    within ``max_sep`` of the Gaia position propagated (proper motion) to the SkyMapper
    mean epoch; one-to-one, nearest first."""
    key = float64_key(sid)
    pad = 1 << 11  # float64 rounding moves an id by < 2**10
    qi, sp = lookup_multi(store["source_id"], key, lo_id - pad, hi_id + pad)
    pos = np.full(len(sid), -1, np.int64)
    st = dict(candidates=len(qi), gaia_with_candidate=int(len(np.unique(qi))))
    if not len(qi):
        st.update(accepted=0)
        return pos, st
    order = np.argsort(sp, kind="stable")
    sp_sorted = sp[order]
    a, b = int(sp_sorted[0]), int(sp_sorted[-1]) + 1
    sra = gather(store["_ra"], sp, a, b)
    sdec = gather(store["_dec"], sp, a, b)
    sep_ep = gather(store["_epoch"], sp, a, b)
    dt = np.where(np.isfinite(sep_ep), (sep_ep - GAIA_EPOCH_MJD) / 365.25, 0.0)
    pr = np.nan_to_num(np.asarray(pmra, np.float64)[qi])
    pd_ = np.nan_to_num(np.asarray(pmdec, np.float64)[qi])
    d0 = np.asarray(dec, np.float64)[qi]
    gdec = d0 + pd_ * dt / 3.6e6
    gra = np.asarray(ra, np.float64)[qi] + pr * dt / 3.6e6 / np.maximum(
        np.cos(np.radians(d0)), 1e-6
    )
    sep = _sep_arcsec(gra, gdec, sra, sdec)
    ok = np.isfinite(sep) & (sep < max_sep)
    st.update(within_sep=int(ok.sum()), beyond_sep=int((~ok).sum()))
    qi, sp, sep = qi[ok], sp[ok], sep[ok]
    o = np.argsort(sep, kind="stable")
    qi, sp, sep = qi[o], sp[o], sep[o]
    _, first_s = np.unique(sp, return_index=True)  # each SkyMapper row: nearest Gaia
    keep = np.zeros(len(sp), bool)
    keep[first_s] = True
    qi2, sp2, sep2 = qi[keep], sp[keep], sep[keep]
    o = np.argsort(sep2, kind="stable")
    qi2, sp2, sep2 = qi2[o], sp2[o], sep2[o]
    _, first_q = np.unique(qi2, return_index=True)  # each Gaia source: nearest row
    pos[qi2[first_q]] = sp2[first_q]
    acc = sep2[first_q]
    st.update(
        accepted=int(len(first_q)),
        conflicts=int(len(sp) - len(first_q)),  # pairs within max_sep not taken
        sep_hist=np.histogram(acc, np.linspace(0, max_sep, 21))[0].tolist(),
    )
    return pos, st


def _nmad(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def _poly_resid(c: np.ndarray, y: np.ndarray, deg: int = 3, iters: int = 3):
    """Residuals of y about a robust (5 NMAD clipped) polynomial in c."""
    m = np.ones(len(c), bool)
    res = y - np.median(y)
    for _ in range(iters):
        if m.sum() <= deg + 1:
            break
        coef = np.polyfit(c[m], y[m], deg)
        res = y - np.polyval(coef, c)
        cut = 5 * max(_nmad(res[m]), 1e-3)
        m = np.abs(res - np.median(res[m])) < cut
    return res


def pairing_metrics(g, bp, rp, x, seed: int = 0) -> dict:
    """Is ``x`` (a survey magnitude) the photometry of the Gaia source on its row?

    Residuals of G - x about a robust cubic in BP - RP: for true pairs their NMAD
    is the survey's colour scatter (a few 0.01 mag in the optical, ~0.1-0.3 mag in
    the mid-IR with reddening); for pairs with another star's photometry it is the
    spread of G - x itself. ``ratio`` divides the NMAD by that of the same rows with
    x permuted (the random-pairing null, from the same sky): ~1 when the table pairs
    rows with other stars. ``outlier_frac`` is the share of |residual| >
    PAIR_OUTLIER_MAG and ``outlier_ratio`` divides it by the null's share: about the
    fraction of wrong pairs in a mixture (crowding and variability add a little).
    Pass: ratio < PAIR_MAX_RATIO and outlier_ratio < PAIR_MAX_OUTLIER, with at least
    PAIR_MIN_N usable rows; a table or partition with more than ~30 % wrong pairs fails.
    """
    c = np.asarray(bp, np.float64) - np.asarray(rp, np.float64)
    g = np.asarray(g, np.float64)
    x = np.asarray(x, np.float64)
    ok = np.isfinite(c) & np.isfinite(g) & np.isfinite(x) & (c > -0.5) & (c < 4.0)
    n = int(ok.sum())
    out = dict(n=n)
    if n < PAIR_MIN_N:
        return out
    c, g, x = c[ok], g[ok], x[ok]
    res = _poly_resid(c, g - x)
    rng = np.random.default_rng(seed)
    res0 = _poly_resid(c, g - x[rng.permutation(n)])
    nm, nm0 = _nmad(res), _nmad(res0)
    ratio = nm / max(nm0, 1e-6)
    frac = float(np.mean(np.abs(res - np.median(res)) > PAIR_OUTLIER_MAG))
    frac0 = float(np.mean(np.abs(res0 - np.median(res0)) > PAIR_OUTLIER_MAG))
    oratio = frac / max(frac0, 1e-3)
    out.update(
        nmad=nm,
        nmad_null=nm0,
        ratio=ratio,
        outlier_frac=frac,
        outlier_frac_null=frac0,
        outlier_ratio=oratio,
        corr_g_x=float(np.corrcoef(g, x)[0, 1]),
        passed=bool(ratio < PAIR_MAX_RATIO and oratio < PAIR_MAX_OUTLIER),
    )
    return out


def _null_array(c: Col, n: int):
    import pyarrow as pa

    if c.kind == "text":
        return pa.nulls(n, pa.string())
    if c.kind in ("mag", "err"):
        return pa.nulls(n, pa.float32())
    return pa.nulls(n, pa.from_numpy_dtype(np.dtype(c.dtype)))


def partition_table(order: int, npix: int, gaia_files: list[Path], cfg: dict):
    """(Arrow table in Gaia row order, stats dict) of one partition."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    W = _worker_state(cfg)
    g = pa.concat_tables([pq.read_table(f, columns=GAIA_READ) for f in gaia_files])
    n = g.num_rows
    if g["source_id"].null_count:
        raise ValueError(f"Norder={order} Npix={npix}: null source_id")
    sid = g["source_id"].to_numpy().astype(np.int64, copy=False)
    lo_id, hi_id = id_range(order, npix)
    outside = (sid < lo_id) | (sid >= hi_id)
    st = dict(
        Norder=order,
        Npix=npix,
        rows=n,
        unique=bool(len(np.unique(sid)) == n),
        outside_ids=sid[outside].tolist(),
        surveys={},
        columns={},
    )
    cols: dict = {}
    for c in GAIA_OUT:
        a = g[c]
        if pa.types.is_floating(a.type):
            v = a.to_numpy(zero_copy_only=False)
            ok = np.isfinite(v)
            cols[c] = pa.array(v, type=a.type, mask=~ok)
        else:
            cols[c] = a.combine_chunks()
    lv = g["l"].to_numpy(zero_copy_only=False).astype(np.float64)
    bv = g["b"].to_numpy(zero_copy_only=False).astype(np.float64)
    sfd = W["sfd"](lv, bv)
    cols["sfd_ebv"] = pa.array(sfd, type=pa.float32(), mask=~np.isfinite(sfd))
    st["sfd"] = dict(
        null=int((~np.isfinite(sfd)).sum()),
        hist=np.histogram(sfd[np.isfinite(sfd)], SFD_EDGES)[0].tolist(),
        max=float(np.nanmax(sfd)) if np.isfinite(sfd).any() else 0.0,
        sum=float(np.nansum(sfd.astype(np.float64))),
    )
    for c in GAIA_OUT:
        st["columns"][c] = int(n - cols[c].null_count)
    st["columns"]["sfd_ebv"] = int(n - cols["sfd_ebv"].null_count)
    extras = {}
    g_mag = g["phot_g_mean_mag"].to_numpy(zero_copy_only=False).astype(np.float64)
    bp_mag = g["phot_bp_mean_mag"].to_numpy(zero_copy_only=False).astype(np.float64)
    rp_mag = g["phot_rp_mean_mag"].to_numpy(zero_copy_only=False).astype(np.float64)
    ra_deg, dec_deg, pm_ra, pm_dec = (
        g[c].to_numpy(zero_copy_only=False).astype(np.float64)
        for c in ("ra", "dec", "pmra", "pmdec")
    )
    for name in SURVEY_ORDER:
        survey = SURVEYS[name]
        if name not in cfg["active"]:  # no trusted source, or failed validation
            for c in survey.columns:
                if c.kind == "extra":
                    extras[(name, c.name)] = np.full(n, np.nan)
                else:
                    cols[c.name] = _null_array(c, n)
                    st["columns"][c.name] = 0
                    st["columns"][c.name + ":sentinel"] = 0
            st["surveys"][name] = dict(masked=cfg["masked"].get(name, "masked"))
            continue
        store = W["stores"][name]
        if survey.id_mode == "float64":
            pos, xst = skymapper_pairs(
                store, sid, ra_deg, dec_deg, pm_ra, pm_dec, lo_id, hi_id
            )
            good = pos[pos >= 0]
            a, b = (int(good.min()), int(good.max()) + 1) if len(good) else (0, 0)
        else:
            pos, a, b = lookup(store["source_id"], sid, lo_id, hi_id)
            xst = None
        hit = pos >= 0
        matched_in_block = int(((pos >= a) & (pos < b)).sum())
        vals = {c.name: gather(store[c.name], pos, a, b) for c in survey.columns}
        # pairing gate: this partition's matches must be the photometry of their row
        xb, (xlo, xhi) = PAIRING_BAND[name]
        xv = vals[xb]
        with np.errstate(invalid="ignore"):
            xok = hit & _valid_mag(xv) & (xv >= xlo) & (xv <= xhi)
        met = pairing_metrics(
            g_mag, bp_mag, rp_mag, np.where(xok, xv, np.nan), seed=order * 100003 + npix
        )
        gate = met.get("passed", True)  # too few rows to judge: keep
        keep = hit & gate
        sst = dict(
            matched=int(hit.sum()),
            kept=int(keep.sum()),
            masked_by_pairing_gate=int((hit & ~keep).sum()),
            block_rows=b - a,
            block_unmatched=(b - a) - matched_in_block,
            matched_outside=int((hit & outside).sum()),
            any_band=0,
            pairing=met,
        )
        if xst is not None:
            sst["position_match"] = xst
        mags = {}
        for c in survey.columns:
            v = vals[c.name]
            if c.kind == "extra":
                extras[(name, c.name)] = np.where(keep, v, np.nan)
                continue
            if c.kind == "mag":
                ok = keep & _valid_mag(v)
                mags[c.name] = (v, ok)
                sentinel = int((keep & np.isfinite(v) & ~ok).sum())
                cols[c.name] = pa.array(v, type=pa.float32(), mask=~ok)
            elif c.kind == "err":
                ok = keep & _valid_err(v) & mags[c.pair][1]
                sentinel = int(
                    (keep & mags[c.pair][1] & np.isfinite(v) & ~_valid_err(v)).sum()
                )
                cols[c.name] = pa.array(v, type=pa.float32(), mask=~ok)
            elif c.kind == "flag":
                ok, sentinel = keep, 0
                cols[c.name] = pa.array(v, mask=~ok)
            else:  # text
                ok, sentinel = keep, 0
                cols[c.name] = _arrow_text(v, ok)
            st["columns"][c.name] = int(ok.sum())
            st["columns"][c.name + ":sentinel"] = sentinel
        anyb = np.zeros(n, bool)
        for _, ok in mags.values():
            anyb |= ok
        sst["any_band"] = int(anyb.sum())
        st["surveys"][name] = sst
        if name == "twomass":
            ks, ks_ok = mags["ks_m"]
            q = cols["ph_qual"]
            good = np.asarray(
                pc.fill_null(
                    pc.is_in(
                        pc.utf8_slice_codeunits(q, 2, 3),
                        value_set=pa.array(list("ABC")),
                    ),
                    False,
                ).to_numpy(zero_copy_only=False),
                bool,
            )
            m = ks_ok & good & np.isfinite(g_mag)
            x, y = g_mag[m], ks[m].astype(np.float64)
            st["ks_g"] = dict(
                n=int(m.sum()),
                sx=float(x.sum()),
                sy=float(y.sum()),
                sxx=float((x * x).sum()),
                syy=float((y * y).sum()),
                sxy=float((x * y).sum()),
                hist=np.histogram(x - y, KSG_EDGES)[0].tolist(),
                below=int(((x - y) < KSG_EDGES[0]).sum()),
                above=int(((x - y) >= KSG_EDGES[-1]).sum()),
            )
    ebv_sm = extras[("skymapper", "_ebmv_sfd")]
    m = np.isfinite(ebv_sm) & np.isfinite(sfd)
    st["sfd_vs_skymapper"] = dict(
        n=int(m.sum()),
        hist=np.histogram((sfd[m] - ebv_sm[m]).astype(np.float64), DSFD_EDGES)[
            0
        ].tolist(),
        abs_sum=float(np.abs(sfd[m] - ebv_sm[m]).astype(np.float64).sum()),
        ratio_hist=np.histogram(
            (sfd[m & (ebv_sm > 0.05)] / ebv_sm[m & (ebv_sm > 0.05)]).astype(np.float64),
            RATIO_EDGES,
        )[0].tolist(),
    )
    names = output_columns()
    table = pa.table({c: cols[c] for c in names})
    # what the v11 contract reads (quality rules, bright limits, error caps applied)
    usable = dict.fromkeys(W["band_names"], 0)
    for i in range(0, n, ROW_GROUP):
        arr = W["reader"].read(table.slice(i, ROW_GROUP))
        for k, v in arr.present_counts.items():
            usable[k] += v
    st["contract_usable"] = usable
    return table, st


def task_home_check(t) -> int:
    """How many of ``ids`` (out-of-range ids found elsewhere) also sit in ``path``."""
    import pyarrow.parquet as pq

    path, ids = t
    home = pq.read_table(path, columns=["source_id"])["source_id"].to_numpy()
    return int(np.isin(ids, home).sum())


def task_partition(t: dict) -> dict:
    import pyarrow.parquet as pq

    stats_path, dst = Path(t["stats"]), Path(t["dst"])
    if stats_path.exists() and dst.exists():
        old = json.loads(stats_path.read_text())
        if old.get("fingerprint") == t["cfg"]["fingerprint"]:
            return old
    t0 = time.monotonic()
    table, st = partition_table(
        t["order"], t["npix"], [Path(f) for f in t["gaia"]], t["cfg"]
    )
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    flags = [
        c.name
        for s in SURVEYS.values()
        for c in s.columns
        if c.kind in ("flag", "text")
    ]
    pq.write_table(
        table,
        tmp,
        compression="zstd",
        row_group_size=ROW_GROUP,
        use_dictionary=flags,
        column_encoding={
            "source_id": "DELTA_BINARY_PACKED",
            "_healpix_29": "DELTA_BINARY_PACKED",
        },
    )
    os.replace(tmp, dst)
    st.update(
        fingerprint=t["cfg"]["fingerprint"],
        seconds=time.monotonic() - t0,
        bytes=dst.stat().st_size,
    )
    _save_json(stats_path, st)
    return st


# ── driver ──────────────────────────────────────────────────────────────────


def _pool(workers: int):
    # spawn, not fork: the parent runs Arrow threads
    return mp.get_context("spawn").Pool(workers, maxtasksperchild=64)


def _run_tasks(pool, fn, tasks, what: str, every: int = 50):
    out, t0 = [], time.monotonic()
    for i, r in enumerate(pool.imap_unordered(fn, tasks), 1):
        out.append(r)
        if i % every == 0 or i == len(tasks):
            el = time.monotonic() - t0
            log(
                f"{what}: {i}/{len(tasks)} ({el / 60:.1f} min, eta {el / i * (len(tasks) - i) / 60:.1f} min)"
            )
    return out


def ps1_phot_file(inputs: Inputs, k: int) -> Path | None:
    for name in (f"ids_pansdr1_{k}.fits.gz", f"ids_pansdr1_{k}.fits"):
        p = Path(inputs.ps1_phot) / name
        if p.exists():
            return p
    return None


def enabled_surveys(inputs: Inputs) -> tuple[list[str], dict[str, str]]:
    """Surveys with a trusted source, and why the others are masked."""
    masked = {}
    if str(inputs.twomass) == "mask":
        masked["twomass"] = (
            "no trusted Gaia-keyed 2MASS table: adql_matches/twomass.fits is mis-keyed "
            "(DR3 clean_tmass_psc_xsc_oid joined to DR1 tmass_oid; corr(G, Ks) 0.002); "
            "2MASS comes with the full-wavelength master's designation join"
        )
    else:
        from masked_stellar_autoencoder.data.twomass_gaia import refuse_miskeyed

        refuse_miskeyed(inputs.twomass)
    return [s for s in SURVEY_ORDER if s not in masked], masked


def prepare_stores(
    inputs: Inputs,
    opt: Options,
    staging: Path,
    workers: int,
    sort_workers: int,
    surveys: list[str],
) -> dict:
    """Stage every match table as a sorted, de-duplicated column store."""
    import pyarrow.parquet as pq

    tasks_by: dict[str, list] = {}
    info: dict[str, dict] = {}
    todo = [s for s in surveys if store_meta(staging, s) is None]
    for name in todo:
        survey = SURVEYS[name]
        u = unsorted_dir(staging, name)
        (u / "done").mkdir(parents=True, exist_ok=True)
        cols = [(c.name, c.src, c.dtype) for c in survey.columns]
        tasks = []
        if name in ("twomass", "sdss", "catwise"):
            path = Path(getattr(inputs, name))
            lay = fits_layout(path)
            n = lay.nrows
            sid_col = "id2" if name == "catwise" else "source_id"
            if lay.dtype[sid_col].kind != "i":
                raise ValueError(f"{path}: {sid_col} is not an integer column")
            step = opt.fits_chunk_rows
            for lo in range(0, n, step):
                tasks.append(
                    dict(
                        survey=name,
                        path=str(path),
                        lo=lo,
                        hi=min(lo + step, n),
                        out=str(u),
                        sid=sid_col,
                        cols=cols,
                        key=f"{lo}",
                    )
                )
            info[name] = dict(
                input=str(path),
                input_rows=n,
                bytes=os.path.getsize(path),
                source_id_column=sid_col,
            )
        elif name == "ps1":
            lay = fits_layout(inputs.ps1_ids)
            n = lay.nrows
            step = opt.ps1_chunk_rows
            missing = []
            for k, lo in enumerate(range(0, n, step)):
                f = ps1_phot_file(inputs, k)
                if f is None:
                    missing.append(k)
                tasks.append(
                    dict(
                        survey=name,
                        ids=str(inputs.ps1_ids),
                        phot=None if f is None else str(f),
                        lo=lo,
                        hi=min(lo + step, n),
                        out=str(u),
                        cols=cols,
                        key=f"{k}",
                    )
                )
            info[name] = dict(
                input=str(inputs.ps1_ids),
                input_rows=n,
                photometry_dir=str(inputs.ps1_phot),
                chunks=len(tasks),
                chunk_rows=step,
                missing_chunk_files=missing,
            )
        else:  # skymapper
            lo = 0
            for o, p in hats_partitions(inputs.skymapper):
                for f in partition_files(inputs.skymapper, o, p):
                    nr = pq.ParquetFile(f).metadata.num_rows
                    tasks.append(
                        dict(
                            survey=name,
                            path=str(f),
                            lo=lo,
                            hi=lo + nr,
                            out=str(u),
                            cols=cols,
                            key=f"{o}_{p}_{f.stem}",
                            max_sep=opt.skymapper_max_sep,
                        )
                    )
                    lo += nr
            n = lo
            props = hats_properties(inputs.skymapper)
            if "hats_nrows" in props and int(props["hats_nrows"]) != n:
                raise ValueError(
                    f"SkyMapper: {n} rows in files, properties say {props['hats_nrows']}"
                )
            info[name] = dict(
                input=str(inputs.skymapper),
                input_rows=n,
                files=len(tasks),
                max_sep_arcsec=opt.skymapper_max_sep,
                rule="gaia_dr3_id1 with gaia_dr3_dist1 < max_sep; nearest SkyMapper object per Gaia source",
            )
        layout = u / "layout.json"
        if not layout.exists():
            for c, dtype in store_columns(survey):
                npy_create(u / f"{c}.npy", dtype, n)
            _save_json(layout, dict(rows=n, columns=store_columns(survey)))
        elif json.loads(layout.read_text())["rows"] != n:
            raise ValueError(
                f"{layout}: row count changed; remove the staging directory"
            )
        done = {p.name for p in (u / "done").iterdir()}
        tasks_by[name] = [t for t in tasks if t["key"] not in done]
        log(
            f"store {name}: {n:,} input rows, {len(tasks)} tasks, {len(tasks_by[name])} to do"
        )
    fns = dict(
        twomass=task_fits_chunk,
        sdss=task_fits_chunk,
        catwise=task_fits_chunk,
        ps1=task_ps1_chunk,
        skymapper=task_skymapper_file,
    )
    allt = [(fns[s], t) for s in todo for t in tasks_by[s]]
    if allt:
        with _pool(workers) as pool:
            _run_tasks(pool, _apply, allt, "fill stores")
    if "ps1" in todo:
        info["ps1"].update(ps1_fallback(inputs, opt, staging, workers))
    metas = {}
    sort_tasks = [
        dict(survey=s, staging=str(staging), info=info.get(s, {})) for s in todo
    ]
    if sort_tasks:
        with _pool(max(1, min(sort_workers, len(sort_tasks)))) as pool:
            for m in _run_tasks(
                pool, task_sort_store, sort_tasks, "sort stores", every=1
            ):
                log(
                    f"store {m['survey']}: {m['rows']:,} rows sorted ({m['sort_seconds']:.0f} s)"
                )
    for s in surveys:
        metas[s] = store_meta(staging, s)
        if metas[s] is None:
            raise RuntimeError(f"store {s} incomplete")
    return metas


def validation_sample(plist, seed: int = 20261124) -> list[tuple[int, int]]:
    """One random Gaia partition per HEALPix order-VALIDATE_ORDER pixel (whole sky)."""
    rng = np.random.default_rng(seed)
    by: dict[int, list] = {}
    for o, p in plist:
        lo, _ = pixel_range(o, p)
        by.setdefault(lo >> 2 * (LEVEL - VALIDATE_ORDER), []).append((o, p))
    return [v[int(rng.integers(len(v)))] for _, v in sorted(by.items())]


SECOND_BAND = {
    "sdss": "i",
    "skymapper": "i_psf",
    "ps1": "i_mean_psf_mag",
    "twomass": "j_m",
    "catwise": "w2mpro",
}
DEC_BANDS = (-90, -60, -30, 0, 30, 60, 90)


def task_validate(t: dict) -> dict:
    """Pairing metrics of every store on one Gaia partition (+ a row subsample)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = [
        "source_id",
        "dec",
        "phot_g_mean_mag",
        "phot_bp_mean_mag",
        "phot_rp_mean_mag",
        "ra",
        "pmra",
        "pmdec",
    ]
    g = pa.concat_tables([pq.read_table(f, columns=cols) for f in t["gaia"]])
    sid = g["source_id"].to_numpy().astype(np.int64, copy=False)
    arr = {c: g[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in cols[1:]}
    lo_id, hi_id = id_range(t["order"], t["npix"])
    rng = np.random.default_rng(t["order"] * 100003 + t["npix"])
    out = dict(Norder=t["order"], Npix=t["npix"], rows=len(sid), surveys={})
    for name in t["surveys"]:
        d = sorted_dir(Path(t["staging"]), name)
        ssid = np.load(d / "source_id.npy", mmap_mode="r")
        res = {}
        if SURVEYS[name].id_mode == "float64":
            store = {
                c: np.load(d / f"{c}.npy", mmap_mode="r")
                for c in ("source_id", "_ra", "_dec", "_epoch")
            }
            pos, res["position_match"] = skymapper_pairs(
                store,
                sid,
                arr["ra"],
                arr["dec"],
                arr["pmra"],
                arr["pmdec"],
                lo_id,
                hi_id,
            )
            good = pos[pos >= 0]
            a, b = (int(good.min()), int(good.max()) + 1) if len(good) else (0, 0)
        else:
            pos, a, b = lookup(ssid, sid, lo_id, hi_id)
        hit = pos >= 0
        res["matched"] = int(hit.sum())
        # ids rounded through float64 upstream are all float-exact (Gaia ids ~0.2-0.5)
        k = np.asarray(ssid[a:b][:200_000]) if b > a else np.zeros(0, np.int64)
        res["float_exact"] = [int((float64_key(k) == k).sum()), int(len(k))]
        sub = {}
        for key, band in (
            ("primary", PAIRING_BAND[name][0]),
            ("second", SECOND_BAND[name]),
        ):
            x = gather(np.load(d / f"{band}.npy", mmap_mode="r"), pos, a, b).astype(
                np.float64
            )
            xlo, xhi = PAIRING_BAND[name][1]
            with np.errstate(invalid="ignore"):
                ok = hit & _valid_mag(x) & (x >= xlo - 1) & (x <= xhi + 1)
                if key == "primary":
                    ok &= (x >= xlo) & (x <= xhi)
            x = np.where(ok, x, np.nan)
            res[key] = dict(
                band=band,
                **pairing_metrics(
                    arr["phot_g_mean_mag"],
                    arr["phot_bp_mean_mag"],
                    arr["phot_rp_mean_mag"],
                    x,
                    seed=t["npix"],
                ),
            )
            sub[key] = x
        use = np.flatnonzero(np.isfinite(sub["primary"]) | np.isfinite(sub["second"]))
        if len(use) > t["sample_rows"]:
            use = np.sort(rng.choice(use, t["sample_rows"], replace=False))
        res["sample"] = {
            "g": arr["phot_g_mean_mag"][use],
            "bp": arr["phot_bp_mean_mag"][use],
            "rp": arr["phot_rp_mean_mag"][use],
            "dec": arr["dec"][use],
            "x": sub["primary"][use],
            "x2": sub["second"][use],
        }
        out["surveys"][name] = res
    return out


def validate_pairing(
    plist,
    gaia: Path,
    staging: Path,
    surveys: list[str],
    workers: int,
    sample_rows: int = 20000,
) -> dict:
    """Pairing validation of each staged survey on a whole-sky random sample of Gaia
    partitions, before any is used: pooled and per-partition ``pairing_metrics``
    (primary band decides; a second band and declination bands are reported)."""
    parts = validation_sample(plist)
    tasks = [
        dict(
            order=o,
            npix=p,
            gaia=[str(f) for f in partition_files(gaia, o, p)],
            staging=str(staging),
            surveys=surveys,
            sample_rows=sample_rows,
        )
        for o, p in parts
    ]
    with _pool(workers) as pool:
        results = _run_tasks(pool, task_validate, tasks, "validate pairing", every=12)
    report = dict(
        rule=(
            f"residual NMAD of G - x about a robust cubic in BP - RP, divided by the NMAD "
            f"with x permuted (random-pairing null), < {PAIR_MAX_RATIO}; share of "
            f"|residual| > {PAIR_OUTLIER_MAG} mag over the null's share < {PAIR_MAX_OUTLIER}; "
            f">= {PAIR_MIN_N} rows; "
            "a survey is used only if the pooled sample passes and >= 80 % of the judged "
            "sampled partitions pass"
        ),
        sample_partitions=[(r["Norder"], r["Npix"]) for r in results],
        surveys={},
    )
    for name in surveys:
        rs = [r["surveys"][name] for r in results]
        pooled = {
            k: np.concatenate([r["sample"][k] for r in rs])
            for k in ("g", "bp", "rp", "dec", "x", "x2")
        }
        prim = pairing_metrics(
            pooled["g"], pooled["bp"], pooled["rp"], pooled["x"], seed=1
        )
        sec = pairing_metrics(
            pooled["g"], pooled["bp"], pooled["rp"], pooled["x2"], seed=2
        )
        by_dec = {}
        for lo, hi in zip(DEC_BANDS[:-1], DEC_BANDS[1:], strict=True):
            m = (pooled["dec"] >= lo) & (pooled["dec"] < hi)
            by_dec[f"{lo}..{hi}"] = pairing_metrics(
                pooled["g"][m], pooled["bp"][m], pooled["rp"][m], pooled["x"][m], seed=3
            )
        judged = [r["primary"] for r in rs if "passed" in r["primary"]]
        frac = float(np.mean([j["passed"] for j in judged])) if judged else 0.0
        extra: dict = {}
        if name == "catwise":
            w12 = pooled["x"] - pooled["x2"]
            w12 = w12[np.isfinite(w12)]
            extra["w1_minus_w2"] = dict(
                n=len(w12),
                median=float(np.median(w12)) if len(w12) else None,
                nmad=_nmad(w12),
            )
        fe = np.sum([r["float_exact"] for r in rs], axis=0)
        fe_frac = float(fe[0] / fe[1]) if fe[1] else None
        id_ok = SURVEYS[name].id_mode == "float64" or fe[1] < 1000 or fe_frac < 0.99
        if SURVEYS[name].id_mode == "float64":
            extra["position_match"] = {
                k: int(sum(r["position_match"].get(k, 0) for r in rs))
                for k in (
                    "candidates",
                    "gaia_with_candidate",
                    "within_sep",
                    "beyond_sep",
                    "accepted",
                    "conflicts",
                )
            }
        report["surveys"][name] = dict(
            id_float_exact_fraction=fe_frac,
            id_mode=SURVEYS[name].id_mode,
            ids_ok=bool(id_ok),
            pooled_primary=prim,
            pooled_second=sec,
            by_dec=by_dec,
            partitions_judged=len(judged),
            partitions_passed_frac=frac,
            partitions=[
                dict(
                    Norder=r["Norder"],
                    Npix=r["Npix"],
                    matched=r["surveys"][name]["matched"],
                    **{k: v for k, v in r["surveys"][name]["primary"].items()},
                )
                for r in results
            ],
            passed=bool(prim.get("passed", False) and frac >= 0.8 and id_ok),
            **extra,
        )
        log(
            f"validate {name}: pooled {prim}; partitions passed {frac:.2f} of {len(judged)}"
        )
    return report


def _apply(ft):
    fn, t = ft
    return fn(t)


def ps1_fallback(inputs: Inputs, opt: Options, staging: Path, workers: int) -> dict:
    """Fill PS1 rows whose obj_id was not in its chunk's photometry file from any file."""
    u = unsorted_dir(staging, "ps1")
    parts = sorted((u / "unmatched").glob("*.npz"))
    rows, sids, exts = [], [], []
    for p in parts:
        with np.load(p) as z:
            rows.append(z["row"])
            sids.append(z["source_id"])
            exts.append(z["ext"])
    row = np.concatenate(rows) if rows else np.zeros(0, np.int64)
    sid = np.concatenate(sids) if sids else np.zeros(0, np.int64)
    ext = np.concatenate(exts) if exts else np.zeros(0, np.int64)
    res = dict(chunk_unmatched=len(row), fallback_found=0)
    if not len(row):
        return res
    want = np.unique(ext)
    wpath = u / "unmatched_ext.npy"
    np.save(wpath, want)
    cols = [(c.name, c.src, c.dtype) for c in SURVEYS["ps1"].columns]
    files = sorted(Path(inputs.ps1_phot).glob("ids_pansdr1_*.fits*"))
    tasks = [dict(want=str(wpath), phot=str(f), cols=cols) for f in files]
    found_ext, found = [], {c[0]: [] for c in cols}
    with _pool(workers) as pool:
        for r in _run_tasks(pool, task_ps1_search, tasks, "ps1 fallback search"):
            found_ext.append(r["ext"])
            for c in found:
                found[c].append(r[c])
    fe = np.concatenate(found_ext)
    fe, first = np.unique(fe, return_index=True)
    j = np.searchsorted(fe, ext).clip(max=max(len(fe) - 1, 0))
    hit = (fe[j] == ext) if len(fe) else np.zeros(len(ext), bool)
    order = np.argsort(row[hit])
    r_hit = row[hit][order]
    for c in found:
        v = np.concatenate(found[c])[first][j[hit]][order]
        _scatter_write(u / f"{c}.npy", r_hit, v)
    _scatter_write(u / "source_id.npy", r_hit, sid[hit][order])
    res["fallback_found"] = int(hit.sum())
    res["no_photometry"] = int((~hit).sum())
    return res


def _scatter_write(path: Path, rows: np.ndarray, values: np.ndarray) -> None:
    mm = np.load(path, mmap_mode="r+")
    mm[rows] = values
    mm.flush()
    del mm


def build(inputs: Inputs, opt: Options, args) -> int:
    out = Path(args.output)
    work = Path(args.work or out.parent / (out.name + "_work"))
    staging = Path(args.staging or work / "staging")
    for p in (out, work):
        p.mkdir(parents=True, exist_ok=True)
    if str(out.resolve()).startswith("/arc/projects/hats"):
        raise SystemExit("never write under /arc/projects/hats")
    mpath = out / "manifest.json"
    if mpath.exists() and not args.force:
        old = json.loads(mpath.read_text())
        if old.get("status") == "photometry_v1a_complete" and not args.partitions:
            log(f"{mpath}: already complete; refusing to overwrite (use --force)")
            return 0
    t_start = time.monotonic()
    from masked_stellar_autoencoder.v11.schema import ContractReader, load_registry

    reader = ContractReader(load_registry(), CONTRACT)
    need = set(reader.columns()) - set(
        v
        for k in ("coefficients", "errors", "correlations")
        for v in reader.contract["xp"][k].values()
    )
    have = set(output_columns()) - {"source_id", "_healpix_29"}
    if need != have:
        raise SystemExit(
            f"output columns differ from {CONTRACT}: missing {sorted(need - have)}, extra {sorted(have - need)}"
        )
    gprops = hats_properties(inputs.gaia)
    m = dict(
        product="photometry_v1a",
        build_version=BUILD_VERSION,
        status="running",
        created_utc=dt.datetime.now(dt.UTC).isoformat(),
        code_commit=code_commit(),
        script="scripts/build_photometry_v1a.py",
        contract=CONTRACT,
        output=str(out),
        work=str(work),
        staging=str(staging),
        columns=output_columns(),
        rules=dict(
            absent="null when the source has no counterpart; never a fill value",
            magnitude_valid=f"finite and {MAG_RANGE[0]} < m < {MAG_RANGE[1]} (else null: SDSS -9999, PS1 -999, NaN)",
            error_valid=f"finite and {ERR_RANGE[0]} < e < {ERR_RANGE[1]} mag and its magnitude valid (else null)",
            quality="not applied here; contract quality rules, bright limits and max_error_mag apply when read",
            duplicates="per match table: identical repeats kept once, a source_id with differing rows is dropped (ambiguous); SkyMapper: nearest within max_sep",
            pairing="whole-sky validation of each table (validation.json) and a per-partition gate: G - x residual NMAD about a cubic in BP - RP vs the permuted-pairing null, and the outlier share vs the null's; a failing table is masked whole, a failing partition for that survey",
            twomass="masked unless --twomass names a correctly keyed table (adql_matches/twomass.fits is mis-keyed and refused)",
            sfd="SFD98 E(B-V) at Gaia (l, b): dustmaps SFDQuery equivalent (ZEA WCS, order-1 map_coordinates, mode nearest); no 0.86 rescaling",
        ),
        inputs={k: str(getattr(inputs, k)) for k in Inputs.__dataclass_fields__},
        gaia=dict(
            hats_nrows=int(gprops["hats_nrows"]),
            hats_creation_date=gprops.get("hats_creation_date"),
        ),
        options=dict(
            skymapper_max_sep_arcsec=opt.skymapper_max_sep,
            ps1_chunk_rows=opt.ps1_chunk_rows,
        ),
        sfd_md5={
            p: _md5(Path(inputs.sfd) / f"SFD_dust_4096_{p}.fits")
            for p in ("ngp", "sgp")
        },
    )
    _save_json(mpath, m)
    stages = set(args.stage.split(","))
    enabled, masked = enabled_surveys(inputs)
    m["masked_surveys"] = dict(masked)
    log(f"photometry_v1a {BUILD_VERSION}; output {out}; staging {staging}")
    log(f"surveys with a source: {enabled}; masked: {sorted(masked)}")
    if stages & {"prepare", "all"}:
        m["status"] = "staging_stores"
        _save_json(mpath, m)
        t0 = time.monotonic()
        metas = prepare_stores(
            inputs, opt, staging, args.workers, args.sort_workers, enabled
        )
        m["stores"] = metas
        m["prepare_seconds"] = time.monotonic() - t0
        _save_json(mpath, m)
    if not stages & {"build", "all"}:
        return 0
    metas = {s: store_meta(staging, s) for s in enabled}
    if any(v is None for v in metas.values()):
        raise SystemExit(
            f"stores incomplete: {[s for s, v in metas.items() if v is None]}"
        )
    m["stores"] = metas
    store_fp = {
        s: {k: metas[s][k] for k in ("rows", "input_rows", "duplicate_rows")}
        for s in metas
    }
    plist_all = hats_partitions(inputs.gaia)
    # pairing validation on a whole-sky sample before any survey is used
    vpath = work / "validation.json"
    val = json.loads(vpath.read_text()) if vpath.exists() else None
    if (
        val is None
        or val.get("stores") != store_fp
        or val.get("version") != BUILD_VERSION
    ):
        m["status"] = "validating_pairing"
        _save_json(mpath, m)
        val = validate_pairing(plist_all, inputs.gaia, staging, enabled, args.workers)
        val.update(stores=store_fp, version=BUILD_VERSION)
        _save_json(vpath, val)
    m["pairing_validation"] = {
        k: {kk: vv for kk, vv in v.items() if kk != "partitions"}
        for k, v in val["surveys"].items()
    }
    m["pairing_validation_file"] = str(vpath)
    for name in enabled:
        if not val["surveys"][name]["passed"]:
            masked[name] = "failed the pairing validation (validation.json)"
    active = [s for s in enabled if s not in masked]
    m["masked_surveys"] = dict(masked)
    m["active_surveys"] = active
    log(f"surveys used: {active}; masked: {masked}")
    fp = hashlib.sha256(
        json.dumps(
            dict(
                v=BUILD_VERSION,
                stores=store_fp,
                active=active,
                sfd=m["sfd_md5"],
                opt=m["options"],
                gate=[PAIR_MIN_N, PAIR_MAX_RATIO, PAIR_MAX_OUTLIER, PAIR_OUTLIER_MAG],
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    m["fingerprint"] = fp
    cfg = dict(
        fingerprint=fp,
        staging=str(staging),
        sfd=str(inputs.sfd),
        active=active,
        masked=masked,
    )
    plist = plist_all
    if args.partitions:
        want = {tuple(int(x) for x in s.split("/")) for s in args.partitions.split(",")}
        plist = [p for p in plist if p in want]
        if len(plist) != len(want):
            raise SystemExit(
                f"partitions not in the Gaia tree: {sorted(want - set(plist))}"
            )
    tasks = []
    for o, p in plist:
        files = partition_files(inputs.gaia, o, p)
        if not files:
            raise SystemExit(f"missing Gaia partition Norder={o} Npix={p}")
        rel = partition_rel(o, p)
        tasks.append(
            dict(
                order=o,
                npix=p,
                gaia=[str(f) for f in files],
                dst=str(out / "dataset" / rel),
                stats=str(work / "stats" / rel.with_suffix(".json")),
                cfg=cfg,
            )
        )
    m["status"] = "building_partitions"
    _save_json(mpath, m)
    t0 = time.monotonic()
    stats, rows = [], 0
    with _pool(args.workers) as pool:
        for i, st in enumerate(pool.imap_unordered(task_partition, tasks), 1):
            stats.append(st)
            rows += st["rows"]
            if i % args.progress_every == 0 or i == len(tasks):
                el = time.monotonic() - t0
                log(
                    f"partitions {i}/{len(tasks)}, {rows:,} rows, {rows / max(el, 1e-9):,.0f} rows/s, eta {el / i * (len(tasks) - i) / 60:.1f} min"
                )
                m.update(partitions_done=i, rows_done=rows, build_elapsed_s=el)
                _save_json(mpath, m)
    m["build_seconds"] = time.monotonic() - t0
    finalize(m, stats, plist, inputs, out, work, staging, args)
    m["elapsed_seconds"] = time.monotonic() - t_start
    m["finished_utc"] = dt.datetime.now(dt.UTC).isoformat()
    _save_json(mpath, m)
    _save_json(work / "manifest.json", m)
    log(f"status {m['status']}")
    if (
        m["status"] == "photometry_v1a_complete"
        and args.clean_staging
        and not args.partitions
    ):
        shutil.rmtree(staging, ignore_errors=True)
        log(f"removed {staging}")
    return (
        0
        if m["status"] in ("photometry_v1a_complete", "photometry_v1a_subset_complete")
        else 1
    )


def finalize(
    m: dict,
    stats: list,
    plist,
    inputs: Inputs,
    out: Path,
    work: Path,
    staging: Path,
    args,
) -> None:
    import pyarrow.parquet as pq

    subset = bool(args.partitions)
    stats = sorted(stats, key=lambda s: (s["Norder"], s["Npix"]))
    rows = sum(s["rows"] for s in stats)
    checks = {}
    # rows
    checks["rows_total"] = dict(
        value=rows,
        expected=None if subset else m["gaia"]["hats_nrows"],
        gaia_dr3_rows=GAIA_NROWS_DR3,
        ok=subset or rows == m["gaia"]["hats_nrows"],
    )
    gaia_rows = {}
    for o, p in plist:
        md = sum(
            pq.ParquetFile(f).metadata.num_rows
            for f in partition_files(inputs.gaia, o, p)
        )
        gaia_rows[(o, p)] = md
    bad_rows = [
        (s["Norder"], s["Npix"])
        for s in stats
        if s["rows"] != gaia_rows[(s["Norder"], s["Npix"])]
    ]
    checks["rows_per_partition"] = dict(mismatched=bad_rows, ok=not bad_rows)
    # duplicates: within partitions, then the out-of-range ids against their home partitions
    dup_parts = [(s["Norder"], s["Npix"]) for s in stats if not s["unique"]]
    starts = np.array([pixel_range(o, p)[0] for o, p in plist])
    srt = np.argsort(starts)
    out_ids = np.array([sid for s in stats for sid in s["outside_ids"]], np.int64)
    cross_dups = len(out_ids) - len(np.unique(out_ids))
    home_missing = 0
    home_tasks = []
    if len(out_ids):
        pix = out_ids >> HEALPIX_SHIFT
        i = np.searchsorted(starts[srt], pix, side="right") - 1
        k = srt[np.maximum(i, 0)]
        for j in np.unique(k):
            o, p = plist[j]
            lo, hi = pixel_range(o, p)
            sel = k == j
            inr = sel & (i >= 0) & (pix >= lo) & (pix < hi)
            home_missing += int((sel & ~inr).sum())
            f = out / "dataset" / partition_rel(o, p)
            if inr.any() and f.exists():
                home_tasks.append((str(f), out_ids[inr]))
    if home_tasks:
        with _pool(args.workers) as pool:
            cross_dups += sum(pool.imap_unordered(task_home_check, home_tasks))
    checks["duplicate_source_id"] = dict(
        partitions_with_duplicates=dup_parts,
        out_of_range_ids=len(out_ids),
        cross_partition_duplicates=cross_dups,
        out_of_range_without_home=home_missing,
        ok=not dup_parts and cross_dups == 0,
    )
    # surveys
    surveys = {}
    active = m["active_surveys"]
    for name in SURVEY_ORDER:
        if name not in active:
            surveys[name] = dict(masked=m["masked_surveys"][name], rows_with_any_band=0)
            continue
        meta = m["stores"][name]
        matched = sum(s["surveys"][name]["matched"] for s in stats)
        pm = [s["surveys"][name]["pairing"] for s in stats]
        judged = [x for x in pm if "passed" in x]
        failed = [
            dict(Norder=s["Norder"], Npix=s["Npix"], **s["surveys"][name]["pairing"])
            for s in stats
            if s["surveys"][name]["pairing"].get("passed") is False
        ]
        surveys[name] = dict(
            input_rows=meta["input_rows"],
            expected_rows=EXPECTED_ROWS.get(name),
            staged_rows=meta["rows"],
            dropped_no_match=meta["dropped_no_match"],
            duplicate_rows=meta["duplicate_rows"],
            ambiguous_ids_dropped=meta["ambiguous_ids_dropped"],
            identical_repeats_dropped=meta["identical_repeats_dropped"],
            matched_gaia_rows=matched,
            staged_rows_not_in_gaia=meta["rows"] - matched,
            kept_rows=sum(s["surveys"][name]["kept"] for s in stats),
            pairing_gate=dict(
                partitions_judged=len(judged),
                partitions_failed=len(failed),
                rows_masked=sum(
                    s["surveys"][name]["masked_by_pairing_gate"] for s in stats
                ),
                nmad_median=float(np.median([x["nmad"] for x in judged]))
                if judged
                else None,
                nmad_p90=float(np.percentile([x["nmad"] for x in judged], 90))
                if judged
                else None,
                ratio_median=float(np.median([x["ratio"] for x in judged]))
                if judged
                else None,
                ratio_p99=float(np.percentile([x["ratio"] for x in judged], 99))
                if judged
                else None,
                corr_g_x_median=float(np.median([x["corr_g_x"] for x in judged]))
                if judged
                else None,
                failed_partitions=failed[:200],
            ),
            rows_with_any_band=sum(s["surveys"][name]["any_band"] for s in stats),
            matched_outside_healpix_range=sum(
                s["surveys"][name]["matched_outside"] for s in stats
            ),
            bands={
                c.name: dict(
                    rows_with_value=sum(s["columns"][c.name] for s in stats),
                    sentinel_to_null=sum(
                        s["columns"][c.name + ":sentinel"] for s in stats
                    ),
                )
                for c in SURVEYS[name].columns
                if c.kind in ("mag", "err")
            },
        )
        for k in (
            "chunk_unmatched",
            "fallback_found",
            "no_photometry",
            "missing_chunk_files",
        ):
            if k in meta:
                surveys[name][k] = meta[k]
    usable = {}
    for s in stats:
        for k, v in s["contract_usable"].items():
            usable[k] = usable.get(k, 0) + v
    m["surveys"] = surveys
    m["contract_usable_rows"] = usable
    m["columns_non_null"] = {
        c: sum(s["columns"].get(c, 0) for s in stats) for c in output_columns()
    }
    exact = [k for k in active if SURVEYS[k].id_mode == "exact"]
    for k in active:
        if SURVEYS[k].id_mode != "exact":
            pm = [s["surveys"][k].get("position_match", {}) for s in stats]
            surveys[k]["position_match"] = {
                f: int(sum(x.get(f, 0) for x in pm))
                for f in (
                    "candidates",
                    "gaia_with_candidate",
                    "within_sep",
                    "beyond_sep",
                    "accepted",
                    "conflicts",
                )
            }
            hs = [x["sep_hist"] for x in pm if "sep_hist" in x]
            surveys[k]["position_match"]["sep_hist_0_to_1arcsec_20bins"] = (
                np.sum(hs, axis=0).tolist() if hs else None
            )
            surveys[k]["staged_rows_not_in_gaia"] = None  # keys are rounded ids
    checks["staged_rows_on_gaia"] = dict(
        per_survey={k: surveys[k]["staged_rows_not_in_gaia"] for k in exact},
        ok=subset or all(surveys[k]["staged_rows_not_in_gaia"] == 0 for k in exact),
    )
    checks["pairing"] = dict(
        validation={k: v["passed"] for k, v in m["pairing_validation"].items()},
        partitions_failed={
            k: surveys[k]["pairing_gate"]["partitions_failed"] for k in active
        },
        rule="every survey used passed the whole-sky validation; partitions failing the "
        "gate are masked for that survey (counted)",
        ok=all(m["pairing_validation"][k]["passed"] for k in active),
    )
    if "twomass" in active:
        checks["twomass_ks_g"] = twomass_ks_g(stats, subset)
    else:
        checks["twomass_ks_g"] = dict(skipped="2MASS masked", ok=True)
    _finalize_sfd(m, checks, stats, inputs, plist, rows)
    _finalize_write(m, checks, stats, plist, inputs, out, work, subset, rows)


def twomass_ks_g(stats, subset) -> dict:
    a = {
        k: sum(s["ks_g"][k] for s in stats)
        for k in ("n", "sx", "sy", "sxx", "syy", "sxy", "below", "above")
    }
    hist = np.sum([s["ks_g"]["hist"] for s in stats], axis=0)
    n = a["n"]
    if n > 2:
        cov = a["sxy"] / n - a["sx"] * a["sy"] / n**2
        vx = a["sxx"] / n - (a["sx"] / n) ** 2
        vy = a["syy"] / n - (a["sy"] / n) ** 2
        r = cov / np.sqrt(vx * vy)
    else:
        r = float("nan")
    cdf = np.cumsum(hist) + a["below"]
    tot = cdf[-1] + a["above"] if len(cdf) else 0

    def q(f):
        i = int(np.searchsorted(cdf, f * tot))
        return float(KSG_EDGES[min(i + 1, len(KSG_EDGES) - 1)])

    gk = dict(
        n=n,
        pearson_r_ks_g=float(r),
        g_minus_ks_median=q(0.5) if tot else None,
        g_minus_ks_p01=q(0.01) if tot else None,
        g_minus_ks_p99=q(0.99) if tot else None,
        frac_outside_m1_7=float(
            (
                (hist[(KSG_EDGES[:-1] < -1) | (KSG_EDGES[1:] > 7)]).sum()
                + a["below"]
                + a["above"]
            )
            / max(tot, 1)
        ),
    )
    return dict(
        **gk,
        rule="Pearson r(G, Ks) >= 0.85 and median G-Ks in [0.5, 4] for ph_qual(Ks) in ABC",
        ok=bool(n == 0 and subset)
        or bool(
            r >= 0.85
            and gk["g_minus_ks_median"] is not None
            and 0.5 <= gk["g_minus_ks_median"] <= 4
        ),
    )


def _finalize_sfd(m, checks, stats, inputs, plist, rows) -> None:
    sfd_null = sum(s["sfd"]["null"] for s in stats)
    dh = np.sum([s["sfd_vs_skymapper"]["hist"] for s in stats], axis=0)
    dn = sum(s["sfd_vs_skymapper"]["n"] for s in stats)
    sm = dict(
        n=dn,
        hist_edges=DSFD_EDGES.tolist(),
        hist=dh.tolist(),
        mean_abs_diff=sum(s["sfd_vs_skymapper"]["abs_sum"] for s in stats) / max(dn, 1),
        frac_within_0p01=float(dh[2:7].sum() / max(dn, 1)) if dn else None,
        note="SkyMapper DR4 ebmv_sfd at the SkyMapper position, its own interpolation "
        "and scaling: informative, not a gate",
    )
    rh = [s["sfd_vs_skymapper"].get("ratio_hist") for s in stats]
    if all(r is not None for r in rh) and rh:
        rh = np.sum(rh, axis=0)
        if rh.sum():
            c = np.cumsum(rh) / rh.sum()
            sm["ratio_ours_over_skymapper_e_gt_0p05"] = dict(
                n=int(rh.sum()),
                median=float(RATIO_EDGES[int(np.searchsorted(c, 0.5)) + 1]),
                p16=float(RATIO_EDGES[int(np.searchsorted(c, 0.16)) + 1]),
                p84=float(RATIO_EDGES[int(np.searchsorted(c, 0.84)) + 1]),
            )
    checks["sfd"] = dict(
        null=sfd_null,
        max=max(s["sfd"]["max"] for s in stats),
        mean=sum(s["sfd"]["sum"] for s in stats) / max(rows, 1),
        vs_skymapper_ebmv_sfd=sm,
        dustmaps_parity=sfd_dustmaps_parity(inputs, plist),
        ok=sfd_null == 0,
    )
    checks["sfd"]["ok"] = bool(
        checks["sfd"]["ok"] and checks["sfd"]["dustmaps_parity"].get("ok", True)
    )


def _finalize_write(m, checks, stats, plist, inputs, out, work, subset, rows) -> None:
    import pyarrow.parquet as pq

    m["checks"] = checks
    m["rows"] = rows
    m["partitions"] = len(stats)
    m["bytes"] = sum(s["bytes"] for s in stats)
    ok = all(c["ok"] for c in checks.values())
    m["status"] = (
        ("photometry_v1a_subset_complete" if subset else "photometry_v1a_complete")
        if ok
        else "photometry_v1a_checks_failed"
    )
    # HATS metadata (partition list, properties, schema)
    if not subset or not (out / "partition_info.csv").exists():
        with open(out / "partition_info.csv", "w") as fh:
            fh.write("Norder,Npix\n")  # the Gaia tree's order
            for o, p in plist:
                fh.write(f"{o},{p}\n")
    first = out / "dataset" / partition_rel(stats[0]["Norder"], stats[0]["Npix"])
    pq.write_metadata(
        pq.ParquetFile(first).schema_arrow, out / "dataset" / "_common_metadata"
    )
    (out / "properties").write_text(
        "\n".join(
            [
                "#HATS catalog (sidecar in the Gaia DR3 HATS layout; no ra/dec: join on source_id)",
                "obs_collection=msa_photometry_v1a",
                "dataproduct_type=object",
                f"hats_nrows={rows}",
                "hats_col_healpix=_healpix_29",
                "hats_col_healpix_order=29",
                "hats_npix_suffix=.parquet",
                "hats_cols_survey_id=source_id",
                f"hats_order={max(s['Norder'] for s in stats)}",
                f"hats_creation_date={dt.datetime.now(dt.UTC):%Y-%m-%dT%H:%MUTC}",
                f"obs_description=MSA-IN v1a Gaia-keyed photometry ({CONTRACT}): Gaia DR3 G/BP/RP, astrometry, SFD E(B-V) and {', '.join(m['active_surveys'])}; masked (columns all null): {', '.join(sorted(m['masked_surveys'])) or 'none'}; rows and order of gaia/dr3",
                f"prov_progenitor={inputs.gaia}; {ADQL}; {inputs.skymapper}; {inputs.sfd}",
                f"msa_build_version={BUILD_VERSION}",
                f"msa_code_commit={m['code_commit']}",
            ]
        )
        + "\n"
    )
    pstats = [
        dict(
            Norder=s["Norder"],
            Npix=s["Npix"],
            rows=s["rows"],
            seconds=s["seconds"],
            bytes=s["bytes"],
            **{f"{k}_kept": v["kept"] for k, v in s["surveys"].items() if "kept" in v},
        )
        for s in stats
    ]
    _save_json(work / "partition_stats.json", pstats)


def sfd_dustmaps_parity(inputs: Inputs, plist, n: int = 200_000) -> dict:
    """Our SFD evaluation against dustmaps.sfd.SFDQuery on random sky positions."""
    try:
        from astropy.coordinates import SkyCoord
        from dustmaps.sfd import SFDQuery
    except ImportError as exc:
        return dict(skipped=str(exc), ok=True)
    rng = np.random.default_rng(20261124)
    lon = rng.uniform(0, 360, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    ours = SFD(Path(inputs.sfd))(lon, lat)
    ref = SFDQuery(map_dir=str(inputs.sfd))(
        SkyCoord(lon, lat, unit="deg", frame="galactic")
    )
    d = np.abs(ours.astype(np.float64) - np.asarray(ref, np.float64))
    return dict(n=n, max_abs_diff=float(d.max()), ok=bool(d.max() <= 1e-6))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--output", required=True, type=Path, help="the photometry_v1a HATS tree root"
    )
    p.add_argument(
        "--work", type=Path, help="stats, logs, manifest copy (default <output>_work)"
    )
    p.add_argument(
        "--staging",
        type=Path,
        help="sorted match-table stores (default <work>/staging)",
    )
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--sort-workers",
        type=int,
        default=2,
        help="stores sorted concurrently (memory)",
    )
    p.add_argument("--stage", default="all", help="prepare, build or all (comma list)")
    p.add_argument(
        "--partitions", help="only these Gaia partitions, 'order/npix,...' (testing)"
    )
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument(
        "--clean-staging",
        action="store_true",
        help="remove the stores after a complete build",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="rebuild even if the manifest says complete",
    )
    p.add_argument("--skymapper-max-sep", type=float, default=2.0)
    p.add_argument("--ps1-chunk-rows", type=int, default=PS1_CHUNK_ROWS)
    p.add_argument("--fits-chunk-rows", type=int, default=4_000_000)
    for k in Inputs.__dataclass_fields__:
        p.add_argument(
            "--" + k.replace("_", "-"),
            type=str if k == "twomass" else Path,
            default=getattr(Inputs(), k),
            help="'mask' or a correctly keyed 2MASS table" if k == "twomass" else None,
        )
    a = p.parse_args(argv)
    inputs = Inputs(**{k: getattr(a, k) for k in Inputs.__dataclass_fields__})
    opt = Options(
        skymapper_max_sep=a.skymapper_max_sep,
        ps1_chunk_rows=a.ps1_chunk_rows,
        fits_chunk_rows=a.fits_chunk_rows,
    )
    return build(inputs, opt, a)


if __name__ == "__main__":
    sys.exit(main())
