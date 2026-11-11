"""Unlabelled full-Gaia reconstruction pool of stars without XP spectra.

Scans every Gaia DR3 GaiaSource HDF5 shard on ARC (column arrays at root) and
keeps a deterministic, reproducible hash sample of ``has_xp_continuous=False``
stars that have G. The XP side of the pool is the existing 150M-row
reconstruction cache (a sample of the ~220M XP pretraining stars, with all
photometry); this pool supplies the ~1.6B Gaia stars without XP.

* sampling: splitmix64(source_id) below ``rate`` x 2^64, so any star's
  membership is reproducible and independent of shard order;
* held-out training stars (roles other than FIT in ``--exclude-source``) are
  excluded, so reconstruction never sees TUNE/CAL/DEV/benchmark inputs;
* columns kept in raw units: G, BP, RP, PARALLAX, pmra, pmdec, RA, DEC (the
  encoder slots Gaia provides), stored as one float32 [N, 8] memmap plus
  int64 source_id, with NaN for missing values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np

GAIA_ROOT = Path("/arc/projects/k-pop/gaia/GaiaSource")
ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
COLUMNS = (
    ("phot_g_mean_mag", "G"),
    ("phot_bp_mean_mag", "BP"),
    ("phot_rp_mean_mag", "RP"),
    ("parallax", "PARALLAX"),
    ("pmra", "pmra"),
    ("pmdec", "pmdec"),
    ("ra", "RA"),
    ("dec", "DEC"),
)
MASK64 = (1 << 64) - 1


def splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 finaliser on uint64 (wrapping arithmetic)."""
    z = x.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def in_sample(source_id: np.ndarray, rate: float) -> np.ndarray:
    if not 0 < rate <= 1:
        raise ValueError("Sampling rate must lie in (0, 1]")
    threshold = np.uint64(min(int(rate * 2.0**64), MASK64))
    return splitmix64(np.asarray(source_id, dtype=np.int64).view(np.uint64)) < threshold


def select_rows(source_id, has_xp, g_mag, rate, excluded):
    """Indices of no-XP stars with finite G in the hash sample, not excluded."""
    sid = np.asarray(source_id, dtype=np.int64)
    keep = ~np.asarray(has_xp, dtype=bool) & np.isfinite(np.asarray(g_mag, "f8"))
    keep &= in_sample(sid, rate)
    if excluded is not None and len(excluded):
        keep &= ~np.isin(sid, excluded)
    return np.flatnonzero(keep)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rate", type=float, default=0.125)
    p.add_argument("--gaia-root", type=Path, default=GAIA_ROOT)
    p.add_argument(
        "--exclude-source",
        help="source run whose non-FIT stars are excluded (e.g. expanded_six_source_20261111_v7)",
    )
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--max-shards", type=int, help="limit shards (smoke tests)")
    a = p.parse_args()
    parts = a.output / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    m = dict(status="scanning_gaia", training_performed=False, rate=a.rate)

    def save():
        tmp = a.output / "manifest.tmp"
        tmp.write_text(json.dumps(m, indent=2, default=str) + "\n")
        tmp.replace(a.output / "manifest.json")

    save()
    try:
        excluded = None
        if a.exclude_source:
            sm = json.loads((a.root / a.exclude_source / "manifest.json").read_text())
            with h5py.File(sm["prepared_path"]) as f:
                ids = f["source_id"][:].astype("i8")
                roles = f["role"][:].astype("S8")
            excluded = np.sort(ids[roles != b"fit"])
            m.update(
                exclude_source=a.exclude_source, excluded_heldout_stars=len(excluded)
            )
        shards = sorted(a.gaia_root.glob("*.hdf5")) + sorted(a.gaia_root.glob("*.h5"))
        if a.max_shards:
            shards = shards[: a.max_shards]
        if not shards:
            raise RuntimeError("No GaiaSource shards found")
        m.update(gaia_shards=len(shards), scanned=0, no_xp_seen=0, kept=0)
        save()
        started = time.monotonic()
        for i, path in enumerate(shards):
            part = parts / (path.stem + ".npz")
            if part.exists():
                with np.load(part) as z:
                    m["kept"] += len(z["source_id"])
                    m["scanned"] += int(z["scanned"])
                    m["no_xp_seen"] += int(z["no_xp_seen"])
                continue
            with h5py.File(path) as f:
                sid = f["source_id"][:]
                has_xp = f["has_xp_continuous"][:].astype(bool)
                g = f["phot_g_mean_mag"][:]
                rows = select_rows(sid, has_xp, g, a.rate, excluded)
                values = np.full((len(rows), len(COLUMNS)), np.nan, dtype="f4")
                for j, (column, _) in enumerate(COLUMNS):
                    if column in f:
                        values[:, j] = f[column][:][rows]
            np.savez(
                part.with_suffix(".tmp.npz"),
                source_id=np.asarray(sid, dtype=np.int64)[rows],
                values=values,
                scanned=len(sid),
                no_xp_seen=int((~has_xp).sum()),
            )
            part.with_suffix(".tmp.npz").replace(part)
            m["scanned"] += len(sid)
            m["no_xp_seen"] += int((~has_xp).sum())
            m["kept"] += len(rows)
            if i % 50 == 0:
                m.update(shards_done=i + 1, elapsed_s=time.monotonic() - started)
                save()
                print(f"{i + 1}/{len(shards)} kept={m['kept']}", flush=True)
        files = sorted(parts.glob("*.npz"))
        n = m["kept"]
        ids_out = np.lib.format.open_memmap(
            a.output / "source_id.npy", "w+", "i8", (n,)
        )
        val_out = np.lib.format.open_memmap(
            a.output / "values.npy", "w+", "f4", (n, len(COLUMNS))
        )
        start = 0
        for path in files:
            with np.load(path) as z:
                k = len(z["source_id"])
                ids_out[start : start + k] = z["source_id"]
                val_out[start : start + k] = z["values"]
                start += k
        if start != n:
            raise ValueError("Part totals differ from the scan count")
        ids_out.flush()
        val_out.flush()
        v = np.asarray(val_out)
        m.update(
            status="complete",
            stars=n,
            columns=[feature for _, feature in COLUMNS],
            visible_fraction={
                feature: float(np.isfinite(v[:, j]).mean()) if n else 0.0
                for j, (_, feature) in enumerate(COLUMNS)
            },
            g_quantiles=dict(
                zip(
                    ("p05", "p50", "p95"),
                    np.nanquantile(v[:, 0], [0.05, 0.5, 0.95]).tolist() if n else [],
                    strict=False,
                )
            ),
            files={
                name: hashlib.sha256((a.output / name).read_bytes()).hexdigest()
                if (a.output / name).stat().st_size < 2**31
                else "large; see size"
                for name in ("source_id.npy",)
            },
            bytes={
                name: (a.output / name).stat().st_size
                for name in ("source_id.npy", "values.npy")
            },
        )
        save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
