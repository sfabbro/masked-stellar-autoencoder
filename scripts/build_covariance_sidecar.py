"""Per-star Gaia DR3 XP coefficient and astrometric covariance sidecar from HATS.

For each named input list of source_ids (a FITS/npz/npy/parquet column) this
writes ``sidecar_<name>.h5``: one row per input star in the input order, with
the XP continuous-spectrum coefficients, errors, packed correlations and
solution metadata from the HATS ``xp_continuous_mean_spectrum`` catalogue, the
astrometric/photometric columns from the HATS ``gaia_source`` catalogue, the
Lindegren+2021 parallax zero point, and ``found_xp`` / ``found_astrometry``
flags. Schema, units and covariance conventions:
``masked_stellar_autoencoder.data.covariance_sidecar``. One ``manifest.json``
records sources, counts, checks and the code version.

Lookup: Gaia source_id encodes the level-12 nested HEALPix pixel
(``source_id >> 35``); its HATS partition is the ``(Norder, Npix)`` of
``partition_info.csv`` with ``Npix = pix12 >> 2(12 - Norder)``. HATS places
stars by their J2016 position (``_healpix_29``), which for ~0.5 % of stars is a
different level-12 pixel than the source_id one and, for ~1e-4 of stars, a
different partition; stars not found in their source_id partition are searched
in the partitions of the 8 neighbouring level-12 pixels. All inputs are looked
up in one pass, so each partition is read once: worker processes read only
the parquet files of partitions holding targets (column projection, row groups
pruned by source_id statistics, record batches filtered to the targets), and
the main process writes the HDF5 files (gzip-1 + shuffle, chunked by rows).

Example (CANFAR; the launcher step ``covariance-sidecar``)::

    python -u scripts/build_covariance_sidecar.py \
        --input table=/arc/projects/astroai/msa/runs/multisurvey_table_20261117_v5/finetune.fits \
        --input unlabelled=/arc/projects/astroai/msa/runs/multisurvey_table_20261117_v5/unlabelled.npz \
        --sample unlabelled=1000000 --output "$experiment_output" --workers 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
try:
    from masked_stellar_autoencoder.data import covariance_sidecar as cs
except ImportError:  # running from a checkout without the editable install
    sys.path.insert(0, str(HERE.parent / "src"))
    from masked_stellar_autoencoder.data import covariance_sidecar as cs

XP_HATS = "/arc/projects/hats/gaia/dr3_xp_continuous"
GAIA_HATS = "/arc/projects/hats/gaia/dr3"
DEFAULT_SEED = 20261120
NSIDE12 = 4096
XP_READ = ["source_id", "_healpix_29"] + [c.name for c in cs.XP_COLUMNS]
GAIA_READ = ["source_id", "_healpix_29"] + [c.name for c in cs.GAIA_COLUMNS]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ selection


def splitmix64(x: np.ndarray, seed: int) -> np.ndarray:
    """SplitMix64 finaliser of ``x XOR seed`` (uint64, wrapping arithmetic)."""
    z = np.asarray(x).astype(np.uint64) ^ np.uint64(seed & 0xFFFFFFFFFFFFFFFF)
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def hash_subset(source_id: np.ndarray, n: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """Indices (ascending) of the ``n`` stars with the smallest source_id hash.

    Depends only on the source_ids and the seed, not on their order or count.
    """
    if n >= len(source_id):
        return np.arange(len(source_id))
    h = splitmix64(source_id, seed)
    return np.sort(np.argpartition(h, n - 1)[:n])


def read_source_ids(path: Path, key: str = "source_id") -> np.ndarray:
    name = path.name.lower()
    if name.endswith((".fits", ".fit", ".fits.gz")):
        from astropy.io import fits

        with fits.open(path, memmap=True) as hdul:
            hdu = next(h for h in hdul if getattr(h, "columns", None) is not None)
            return np.asarray(hdu.data[key], dtype=np.int64)
    if name.endswith(".npz"):
        with np.load(path) as z:
            return np.asarray(z[key], dtype=np.int64)
    if name.endswith(".npy"):
        return np.asarray(np.load(path, mmap_mode="r"), dtype=np.int64)
    if name.endswith(".parquet"):
        import pyarrow.parquet as pq

        return pq.read_table(path, columns=[key]).column(0).to_numpy().astype(np.int64)
    raise ValueError(f"unsupported input file {path}")


# ----------------------------------------------------------------- partitions


def load_partitions(root: Path) -> dict[int, np.ndarray]:
    """``{Norder: sorted Npix}`` from a HATS ``partition_info.csv``."""
    lines = (root / "partition_info.csv").read_text().split()
    header = lines[0].split(",")
    io, ip = header.index("Norder"), header.index("Npix")
    rows = np.array(
        [[int(x) for x in ln.split(",")] for ln in lines[1:]], dtype=np.int64
    )
    return {int(k): np.sort(rows[rows[:, io] == k, ip]) for k in np.unique(rows[:, io])}


def assign_pix12(pix12: np.ndarray, partitions: dict[int, np.ndarray]):
    """Partition ``(norder, npix)`` holding each level-12 pixel; -1 where none."""
    pix12 = np.asarray(pix12, np.int64)
    norder = np.full(pix12.shape, -1, np.int8)
    npix = np.full(pix12.shape, -1, np.int64)
    for k in sorted(partitions):
        if k > 12:
            raise ValueError("partition orders above 12 are not supported")
        cand = pix12 >> (2 * (12 - k))
        pix = partitions[k]
        pos = np.searchsorted(pix, cand).clip(max=len(pix) - 1)
        hit = (pix[pos] == cand) & (pix12 >= 0)
        if np.any(hit & (norder >= 0)):
            raise ValueError("overlapping HATS partitions")
        norder[hit], npix[hit] = k, cand[hit]
    return norder, npix


def assign_partitions(source_id: np.ndarray, partitions: dict[int, np.ndarray]):
    return assign_pix12(np.asarray(source_id, np.int64) >> 35, partitions)


def neighbour_partitions(source_id, partitions) -> list[tuple[int, int, int]]:
    """``(star index, norder, npix)`` for partitions of the 8 neighbouring level-12
    pixels that differ from the star's own source_id partition."""
    import healpy

    source_id = np.asarray(source_id, np.int64)
    if not len(source_id):
        return []
    pix12 = source_id >> 35
    own = assign_pix12(pix12, partitions)
    nb = healpy.get_all_neighbours(NSIDE12, pix12, nest=True)  # [8, n], -1 = none
    star = np.broadcast_to(np.arange(len(pix12)), nb.shape)
    k, p = assign_pix12(nb.ravel(), partitions)
    keep = (k >= 0) & ~((k == np.repeat(own[0][None], 8, 0).ravel()) &
                        (p == np.repeat(own[1][None], 8, 0).ravel()))  # fmt: skip
    trip = np.stack([star.ravel()[keep], k[keep], p[keep]], axis=1)
    return [tuple(int(v) for v in t) for t in np.unique(trip, axis=0)]


def partition_files(root: Path, norder: int, npix: int) -> list[Path]:
    """Parquet files of one partition (``Npix=n/`` directory or ``Npix=n.parquet``)."""
    base = root / "dataset" / f"Norder={norder}" / f"Dir={npix // 10000 * 10000}"
    d = base / f"Npix={npix}"
    if d.is_dir():
        return sorted(d.glob("*.parquet"))
    f = base / f"Npix={npix}.parquet"
    return [f] if f.is_file() else []


# ------------------------------------------------------------------- reading


def _isin_sorted(sorted_ids: np.ndarray, x: np.ndarray) -> np.ndarray:
    if not len(sorted_ids):
        return np.zeros(len(x), bool)
    pos = np.searchsorted(sorted_ids, x).clip(max=len(sorted_ids) - 1)
    return sorted_ids[pos] == x


def _list_matrix(arr, width: int, dtype: str):
    """Arrow list column -> ``[n, width]`` numpy; wrong-length/null lists -> NaN."""
    import pyarrow as pa
    import pyarrow.compute as pc

    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    n = len(arr)
    lengths = pc.fill_null(pc.list_value_length(arr), -1).to_numpy(zero_copy_only=False)
    ok = lengths == width
    out = np.full((n, width), np.nan, dtype)
    if ok.all():
        out[:] = arr.flatten().to_numpy(zero_copy_only=False).reshape(n, width)
    elif ok.any():
        good = arr.filter(pa.array(ok)).flatten().to_numpy(zero_copy_only=False)
        out[ok] = good.reshape(-1, width)
    return out, int((~ok).sum())


def _scalar(arr, dtype: str) -> np.ndarray:
    import pyarrow.compute as pc

    fill = cs.fill_value(dtype)
    if np.dtype(dtype).kind in "iu":
        arr = pc.fill_null(arr, fill)
    return arr.to_numpy(zero_copy_only=False).astype(dtype)


def read_matches(files, targets, columns, specs, batch_size=4096):
    """Rows of ``files`` whose source_id is in sorted ``targets``.

    Returns ``(arrays, stats)``; ``arrays`` maps each column to numpy (lists as
    ``[n, width]``), duplicated source_ids keep their first row.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    parts = {c: [] for c in columns}
    stats = dict(files_read=0, row_groups_read=0, rows_scanned=0, bad_list_rows=0)
    for path in files:
        pf = pq.ParquetFile(path, buffer_size=8 << 20, pre_buffer=False)
        missing = [c for c in columns if pf.schema_arrow.get_field_index(c) < 0]
        if missing:
            raise KeyError(f"{path}: missing columns {missing}")
        md = pf.metadata
        isid = pf.schema_arrow.get_field_index("source_id")
        rgs = []
        for i in range(md.num_row_groups):
            st = md.row_group(i).column(isid).statistics
            if st is not None and st.has_min_max:
                lo = np.searchsorted(targets, st.min, "left")
                if lo >= np.searchsorted(targets, st.max, "right"):
                    continue
            rgs.append(i)
        if not rgs:
            continue
        sid = pf.read_row_groups(rgs, columns=["source_id"]).column(0).to_numpy()
        mask = _isin_sorted(targets, sid)
        stats["files_read"] += 1
        stats["rows_scanned"] += len(sid)
        if not mask.any():
            continue
        keep = [i for i, n0, n1 in _rg_spans(md, rgs) if mask[n0:n1].any()]
        stats["row_groups_read"] += len(keep)
        mask = np.concatenate(
            [mask[n0:n1] for i, n0, n1 in _rg_spans(md, rgs) if i in keep]
        )
        offset = 0
        for batch in pf.iter_batches(
            batch_size=batch_size, row_groups=keep, columns=columns
        ):
            m = mask[offset : offset + batch.num_rows]
            offset += batch.num_rows
            if not m.any():
                continue
            sub = batch.filter(pa.array(m))
            for c in columns:
                width, dtype = specs[c]
                if width:
                    values, bad = _list_matrix(sub.column(c), width, dtype)
                    stats["bad_list_rows"] += bad
                else:
                    values = _scalar(sub.column(c), dtype)
                parts[c].append(values)
        if offset != len(mask):
            raise RuntimeError(f"{path}: batch rows {offset} != mask rows {len(mask)}")
    out = {}
    for c in columns:
        width, dtype = specs[c]
        empty = np.empty((0, width) if width else (0,), dtype)
        out[c] = np.concatenate(parts[c]) if parts[c] else empty
    _, first = np.unique(out["source_id"], return_index=True)
    stats["duplicate_rows"] = int(len(out["source_id"]) - len(first))
    if stats["duplicate_rows"]:
        first.sort()
        out = {c: v[first] for c, v in out.items()}
    return out, stats


def _rg_spans(md, rgs):
    n0 = 0
    for i in rgs:
        n1 = n0 + md.row_group(i).num_rows
        yield i, n0, n1
        n0 = n1


SPECS = {
    "source_id": (0, "i8"),
    "_healpix_29": (0, "i8"),
    **{c.name: (c.width, c.dtype) for c in cs.XP_COLUMNS + cs.GAIA_COLUMNS},
}


def process_partition(task: dict) -> dict:
    """Worker: look up ``task['xp_ids']`` / ``task['gaia_ids']`` (sorted) in one
    partition of both catalogues."""
    import pyarrow as pa

    pa.set_cpu_count(task.get("threads", 1))
    t0 = time.perf_counter()
    k, npix = task["norder"], task["npix"]
    result = dict(norder=k, npix=npix, stats={})
    for cat, root, ids, columns in (
        ("xp", task["xp_root"], task["xp_ids"], XP_READ),
        ("gaia", task["gaia_root"], task["gaia_ids"], GAIA_READ),
    ):
        if not len(ids):
            result[cat] = None
            continue
        files = partition_files(Path(root), k, npix)
        arrays, stats = read_matches(
            files, ids, columns, SPECS, task.get("batch_size", 4096)
        )
        stats["partition_files"] = len(files)
        sid, hp = arrays["source_id"], arrays.pop("_healpix_29")
        stats["matched"] = len(sid)
        stats["level12_mismatch"] = int(np.sum((hp >> 34) != (sid >> 35)))
        stats["outside_partition"] = int(np.sum((hp >> (2 * (29 - k))) != npix))
        if cat == "gaia" and len(sid):
            zpt, _ = cs.parallax_zero_point_l21(
                arrays["phot_g_mean_mag"], arrays["nu_eff_used_in_astrometry"],
                arrays["pseudocolour"], arrays["ecl_lat"], arrays["astrometric_params_solved"],
            )  # fmt: skip
            arrays["parallax_zpt_l21"] = zpt.astype(np.float32)
        result[cat] = arrays
        result["stats"][cat] = stats
    result["seconds"] = time.perf_counter() - t0
    return result


# ------------------------------------------------------------------- writing


class SidecarWriter:
    """HDF5 file with every :data:`cs.SCHEMA` column for ``n`` rows."""

    def __init__(
        self, path: Path, source_id: np.ndarray, row_index: np.ndarray, attrs: dict
    ):
        import h5py

        self.path = path
        self.n = len(source_id)
        self.f = h5py.File(path, "w", rdcc_nbytes=16 << 20, rdcc_nslots=10007)
        for c in cs.SCHEMA:
            shape = (self.n, c.width) if c.width else (self.n,)
            kw = {}
            if self.n:
                # small chunks: partitions finish in any order, so a chunk shared by
                # two partitions is re-compressed; keep that cheap
                rows = 64 if c.width > 256 else 1024 if c.width else 8192
                kw = dict(chunks=(min(rows, self.n),) + shape[1:], compression="gzip",
                          compression_opts=1, shuffle=True)  # fmt: skip
            ds = self.f.create_dataset(
                c.name,
                shape,
                dtype=np.dtype(c.dtype),
                fillvalue=cs.fill_value(c.dtype),
                **kw,
            )
            ds.attrs["unit"] = c.unit
            ds.attrs["description"] = c.description
        self.f["source_id"][...] = source_id
        self.f["row_index"][...] = row_index
        for key, value in attrs.items():
            self.f.attrs[key] = value

    def write(self, positions: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
        if not len(positions):
            return
        order = np.argsort(positions, kind="stable")
        pos = positions[order]
        if np.any(np.diff(pos) == 0):
            raise ValueError("duplicate positions in one write")
        breaks = np.flatnonzero(np.diff(pos) != 1) + 1
        runs = np.split(np.arange(len(pos)), breaks)
        for name, values in arrays.items():
            values = np.asarray(values)[order]
            ds = self.f[name]
            for r in runs:
                ds[int(pos[r[0]]) : int(pos[r[-1]]) + 1] = values[r[0] : r[-1] + 1]

    def close(self):
        self.f.close()


# ----------------------------------------------------------------------- main


def _properties(root: Path) -> dict:
    for name in ("properties", "hats.properties"):
        p = root / name
        if p.is_file():
            out = {}
            for line in p.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    out[key.strip()] = value.strip()
            return {
                k: out[k]
                for k in sorted(out)
                if k.startswith(("hats_", "obs_", "prov_"))
            }
    return {}


def _code_version() -> dict:
    out = {
        "script": "scripts/build_covariance_sidecar.py",
        "format_version": cs.FORMAT_VERSION,
    }
    for key, path in (
        ("script_sha256", Path(__file__)),
        ("module_sha256", Path(cs.__file__)),
    ):
        out[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL
        ).strip()
        out["git_dirty"] = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=HERE,
                text=True,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return out


def _parse_pairs(items, kind=str) -> dict:
    out = {}
    for item in items or []:
        name, _, value = item.partition("=")
        if not name or not value:
            raise SystemExit(f"expected NAME=VALUE, got {item!r}")
        out[name] = kind(value)
    return out


def _run_tasks(tasks, workers, handle):
    if workers <= 1:
        for t in tasks:
            handle(process_partition(t))
        return
    it = iter(tasks)
    pending = set()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
        while True:
            while len(pending) < 2 * workers:
                t = next(it, None)
                if t is None:
                    break
                pending.add(pool.submit(process_partition, t))
            if not pending:
                return
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                handle(fut.result())


def build(args) -> dict:
    t_start = time.perf_counter()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        old = json.loads(manifest_path.read_text())
        if old.get("status") == "complete":
            raise SystemExit(
                f"{manifest_path} is complete; pass --overwrite to rebuild"
            )
    xp_root, gaia_root = Path(args.xp_hats), Path(args.gaia_hats)
    partitions = load_partitions(xp_root)
    gaia_partitions = load_partitions(gaia_root)
    same = all(
        np.array_equal(partitions.get(k, []), gaia_partitions.get(k, []))
        for k in set(partitions) | set(gaia_partitions)
    )
    if not same:
        raise SystemExit("XP and gaia_source HATS partitions differ")

    inputs = _parse_pairs(args.input, Path)
    samples = _parse_pairs(args.sample, int)
    if unknown := set(samples) - set(inputs):
        raise SystemExit(f"--sample for unknown inputs {sorted(unknown)}")
    manifest = dict(
        status="running",
        format_version=cs.FORMAT_VERSION,
        created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        code=_code_version(),
        sources=dict(
            xp_hats=str(xp_root),
            xp_properties=_properties(xp_root),
            gaia_hats=str(gaia_root),
            gaia_properties=_properties(gaia_root),
            partitions=int(sum(len(v) for v in partitions.values())),
            partition_orders=sorted(int(k) for k in partitions),
        ),  # fmt: skip
        conventions=dict(
            xp_packing=cs.XP_PACKING,
            xp_covariance=cs.XP_COVARIANCE_CONVENTION,
            table_normalisation=cs.TABLE_NORMALISATION,
            astrometric_order=list(cs.ASTROMETRY),
            astrometric_units=list(cs.ASTROMETRY_UNITS),
            parallax_zero_point="Lindegren et al. 2021 (A&A 649, A4) Tables 9/10, mas, "
            "subtract from parallax; NaN for 2-parameter solutions",
            missing_values="NaN (float), -1 (int), False (bool)",
        ),
        columns={
            c.name: dict(
                dtype=np.dtype(c.dtype).name,
                unit=c.unit,
                shape=["N", c.width] if c.width else ["N"],
                description=c.description,
            )
            for c in cs.SCHEMA
        },  # fmt: skip
        inputs={},
        workers=args.workers,
    )

    def save():
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Inputs -> per-input id arrays and one global set of unique targets.
    ids_per_input, rows_per_input = {}, {}
    for name, path in inputs.items():
        ids = read_source_ids(path, args.key)
        info = dict(path=str(path), key=args.key, rows_in_file=int(len(ids)))
        if name in samples:
            rows = hash_subset(ids, samples[name], args.seed)
            info["selection"] = (
                f"{len(rows)} smallest splitmix64(source_id ^ seed) hashes, seed={args.seed}; "
                "file order"
            )
        else:
            rows = np.arange(len(ids))
            info["selection"] = "all rows, file order"
        ids_per_input[name], rows_per_input[name] = ids[rows], rows
        info.update(selected=int(len(rows)), output=f"sidecar_{name}.h5")
        manifest["inputs"][name] = info
        log(f"input {name}: {len(rows)} of {len(ids)} rows from {path}")
    names = list(inputs)
    all_ids = (
        np.concatenate([ids_per_input[n] for n in names])
        if names
        else np.empty(0, "i8")
    )
    owner = np.concatenate(
        [np.full(len(ids_per_input[n]), i) for i, n in enumerate(names)]
    )
    position = np.concatenate([np.arange(len(ids_per_input[n])) for n in names])
    uids, inverse = np.unique(all_ids, return_inverse=True)
    occ_order = np.argsort(inverse, kind="stable")
    occ_start = np.concatenate(
        [[0], np.cumsum(np.bincount(inverse, minlength=len(uids)))]
    )
    found = {"xp": np.zeros(len(uids), bool), "gaia": np.zeros(len(uids), bool)}
    has_xp_flag = np.ones(len(uids), bool)  # gaia_source has_xp_continuous, once known
    stats_total = {
        c: dict(files_read=0, row_groups_read=0, rows_scanned=0, bad_list_rows=0,
                duplicate_rows=0, matched=0, level12_mismatch=0, outside_partition=0,
                partitions_without_files=0)
        for c in ("xp", "gaia")
    }  # fmt: skip
    missing_partition_files = set()

    writers = {}
    for i, name in enumerate(names):
        path = out_dir / f"sidecar_{name}.h5.partial"
        attrs = dict(
            format_version=cs.FORMAT_VERSION, input_name=name,
            input_path=manifest["inputs"][name]["path"],
            selection=manifest["inputs"][name]["selection"],
            xp_packing=cs.XP_PACKING, xp_covariance=cs.XP_COVARIANCE_CONVENTION,
            table_normalisation=cs.TABLE_NORMALISATION,
            xp_hats=str(xp_root), gaia_hats=str(gaia_root),
            code=json.dumps(manifest["code"]),
        )  # fmt: skip
        writers[name] = SidecarWriter(
            path, ids_per_input[name], rows_per_input[name], attrs
        )

    xp_fields = [c.name for c in cs.XP_COLUMNS]
    gaia_fields = [c.name for c in cs.GAIA_COLUMNS] + ["parallax_zpt_l21"]
    done_partitions = [0]
    t_loop = time.perf_counter()

    def handle(res):
        k, npix = res["norder"], res["npix"]
        for cat, fields, flag in (("xp", xp_fields, "found_xp"),
                                  ("gaia", gaia_fields, "found_astrometry")):  # fmt: skip
            arrays = res.get(cat)
            st = res["stats"].get(cat)
            if st is None:
                continue
            for key in stats_total[cat]:
                stats_total[cat][key] += st.get(key, 0)
            if st["partition_files"] == 0:
                stats_total[cat]["partitions_without_files"] += 1
                missing_partition_files.add(f"{cat}:{k}/{npix}")
            if arrays is None or not len(arrays["source_id"]):
                continue
            u = np.searchsorted(uids, arrays["source_id"])
            new = ~found[cat][u]  # neighbour retries never overwrite a match
            u = u[new]
            arrays = {f: v[new] for f, v in arrays.items()}
            found[cat][u] = True
            if cat == "gaia":
                has_xp_flag[u] = arrays["has_xp_continuous"]
            # every occurrence of each matched star in the inputs (CSR expansion)
            counts = occ_start[u + 1] - occ_start[u]
            src = np.repeat(np.arange(len(u)), counts)
            first = np.repeat(occ_start[u] - (np.cumsum(counts) - counts), counts)
            occ = occ_order[first + np.arange(len(src))]
            for i, name in enumerate(names):
                sel = owner[occ] == i
                if not sel.any():
                    continue
                rows = src[sel]
                payload = {f: arrays[f][rows] for f in fields}
                payload[flag] = np.ones(len(rows), bool)
                payload["hats_norder"] = np.full(len(rows), k, np.int8)
                payload["hats_npix"] = np.full(len(rows), npix, np.int32)
                writers[name].write(position[occ][sel], payload)
        done_partitions[0] += 1
        n = done_partitions[0]
        if n % args.log_every == 0 or n == n_tasks[0]:
            el = time.perf_counter() - t_loop
            log(
                f"{pass_name[0]}: {n}/{n_tasks[0]} partitions, xp found {found['xp'].sum()}, "
                f"gaia found {found['gaia'].sum()} of {len(uids)}, {el:.0f}s, "
                f"eta {el / n * (n_tasks[0] - n):.0f}s"
            )
            manifest["progress"] = dict(
                pass_=pass_name[0], partitions_done=n, partitions=n_tasks[0],
                xp_found=int(found["xp"].sum()), gaia_found=int(found["gaia"].sum()),
                targets=int(len(uids)), elapsed_seconds=round(time.perf_counter() - t_start, 1),
            )  # fmt: skip
            save()

    common = dict(
        xp_root=str(xp_root), gaia_root=str(gaia_root), batch_size=args.batch_size
    )
    pass_name, n_tasks = ["primary"], [0]

    # Pass 1: the source_id partition of every target.
    norder, npix = assign_partitions(uids, partitions)
    unassigned = int(np.sum(norder < 0))
    key = norder.astype(np.int64) * (1 << 40) + npix
    valid = np.flatnonzero(norder >= 0)
    order = valid[np.argsort(key[valid], kind="stable")]
    bounds = np.flatnonzero(np.diff(key[order])) + 1
    tasks = []
    for grp in np.split(order, bounds) if len(order) else []:
        ids = uids[grp]
        tasks.append(dict(norder=int(norder[grp[0]]), npix=int(npix[grp[0]]),
                          xp_ids=ids, gaia_ids=ids, **common))  # fmt: skip
    tasks.sort(key=lambda t: -len(t["xp_ids"]))
    if args.max_partitions:
        tasks = tasks[: args.max_partitions]
    n_tasks[0] = len(tasks)
    log(
        f"{len(uids)} unique targets in {len(tasks)} partitions ({unassigned} unassigned)"
    )
    save()
    _run_tasks(tasks, args.workers, handle)
    primary = {c: int(found[c].sum()) for c in found}

    # Pass 2: stars missing from their source_id partition -> neighbour partitions.
    retry = {"xp": 0, "gaia": 0}
    if not args.no_neighbours and not args.max_partitions:
        want_xp = ~found["xp"] & has_xp_flag
        want_gaia = ~found["gaia"]
        cand = np.flatnonzero(want_xp | want_gaia)
        trips = neighbour_partitions(uids[cand], partitions)
        groups = {}
        for star, k, p in trips:
            groups.setdefault((k, p), []).append(cand[star])
        tasks = []
        for (k, p), members in groups.items():
            members = np.array(sorted(members))
            tasks.append(dict(norder=k, npix=p, xp_ids=uids[members[want_xp[members]]],
                              gaia_ids=uids[members[want_gaia[members]]], **common))  # fmt: skip
        pass_name[0], n_tasks[0], done_partitions[0] = "neighbours", len(tasks), 0
        log(f"neighbour pass: {len(cand)} stars, {len(tasks)} partitions")
        t_loop = time.perf_counter()
        _run_tasks(tasks, args.workers, handle)
        retry = {c: int(found[c].sum()) - primary[c] for c in found}

    for w in writers.values():
        w.close()
    for i, name in enumerate(names):
        u = inverse[owner == i]
        info = manifest["inputs"][name]
        info.update(
            found_xp=int(found["xp"][u].sum()),
            missing_xp=int((~found["xp"][u]).sum()),
            missing_xp_with_has_xp_flag=int((~found["xp"][u] & found["gaia"][u] & has_xp_flag[u]).sum()),
            found_astrometry=int(found["gaia"][u].sum()),
            missing_astrometry=int((~found["gaia"][u]).sum()),
        )  # fmt: skip
        final = out_dir / f"sidecar_{name}.h5"
        os.replace(writers[name].path, final)
        info["bytes"] = final.stat().st_size
        info["bytes_per_star"] = round(info["bytes"] / max(1, info["selected"]), 1)
    manifest.update(
        status="complete" if not args.max_partitions else "partial (max_partitions)",
        runtime_seconds=round(time.perf_counter() - t_start, 1),
        targets=dict(
            unique=int(len(uids)),
            unassigned=unassigned,
            found_primary=primary,
            found_in_neighbour_partition=retry,
        ),  # fmt: skip
        reads=stats_total,
        missing_partition_files=sorted(missing_partition_files),
    )
    manifest.pop("progress", None)
    save()
    log(f"done in {manifest['runtime_seconds']}s: " + json.dumps(manifest["inputs"]))
    return manifest


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="source_id list (.fits/.npz/.npy/.parquet); writes sidecar_NAME.h5",
    )
    p.add_argument(
        "--sample",
        action="append",
        metavar="NAME=N",
        help="keep N stars of input NAME by source_id hash",
    )
    p.add_argument("--key", default="source_id", help="source_id column/array name")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--xp-hats", default=XP_HATS)
    p.add_argument("--gaia-hats", default=GAIA_HATS)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument(
        "--batch-size", type=int, default=4096, help="parquet record batch rows"
    )
    p.add_argument(
        "--max-partitions",
        type=int,
        default=0,
        help="smoke test: process only the N largest partitions",
    )
    p.add_argument(
        "--no-neighbours", action="store_true", help="skip the neighbour-partition pass"
    )
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args(argv)  # fmt: skip


if __name__ == "__main__":
    build(parse_args())
