"""Stream staged candidate inputs; count unique-star coverage without XP selection.

Availability and known-error counts are not native quality admission. New stars
remain role-unassigned here. No normalization or model fitting is performed.
"""

import argparse
import datetime as dt
import json
from pathlib import Path
import resource
import time

import h5py
import numpy as np
import pyarrow.parquet as pq


GAIA_VALUES = {
    "G": "phot_g_mean_mag", "BP": "phot_bp_mean_mag", "RP": "phot_rp_mean_mag",
    "PARALLAX": "parallax", "pmra": "pmra", "pmdec": "pmdec",
    "RA": "ra", "DEC": "dec",
}
GAIA_ERRORS = {"PARALLAX": "parallax_error", "pmra": "pmra_error", "pmdec": "pmdec_error"}
ROLES = ("unassigned", "fit", "tune", "cal", "dev", "devtest", "external", "excluded")


def valid(values):
    return np.isfinite(values) & (values != -9999)


def positions(batch, ids):
    source = batch["source_id"].to_numpy(zero_copy_only=False)
    row = batch["candidate_row"].to_numpy(zero_copy_only=False)
    if source.dtype.kind not in "iu" or row.dtype.kind not in "iu":
        raise ValueError("Exact integer source_id and candidate_row required")
    if ((row < 0) | (row >= len(ids))).any() or not np.array_equal(ids[row], source):
        raise ValueError("Staged candidate_row does not identify its native Gaia ID")
    return row


def add_bits(bits, rows, masks):
    """OR repeated observation records, counting each star at most once per column."""
    for start in range(0, masks.shape[1], 64):
        width = min(64, masks.shape[1] - start)
        powers = np.left_shift(np.uint64(1), np.arange(width, dtype="u8"))
        encoded = (masks[:, start:start + width] * powers).sum(axis=1, dtype="u8")
        np.bitwise_or.at(bits[:, start // 64], rows, encoded)


def census(root, contract, progress):
    names = contract["feature_names"]
    errors = dict(zip(names, contract["error_names"], strict=True))
    gaia = root / "full_gaia_gaia_stage_20261108_v2"
    photo = root / "full_gaia_photometry_stage_20261108_v1"
    ids = np.load(gaia / "label-candidate-source-id.npy", mmap_mode="r")
    if ids.dtype.kind not in "iu" or not np.all(ids[1:] > ids[:-1]):
        raise ValueError("Unique sorted integer candidate IDs required")
    other = np.load(photo / "label-candidate-source-id.npy", mmap_mode="r")
    if other.dtype.kind not in "iu" or not np.array_equal(ids, other):
        raise ValueError("Gaia and photometry candidate parents differ")
    roles = np.zeros(len(ids), dtype="u1")
    with h5py.File(root / "expanded_six_source_20261107_v5/prepared.h5") as source:
        oldids, oldroles = source["source_id"][:], source["role"][:]
    if oldids.dtype.kind not in "iu" or not np.all(oldids[1:] > oldids[:-1]):
        raise ValueError("Unique sorted integer inherited role IDs required")
    row = np.searchsorted(ids, oldids)
    hit = row < len(ids)
    hit[hit] &= ids[row[hit]] == oldids[hit]
    if not hit.all():
        raise ValueError("Candidate parent omitted stars from the inherited role registry")
    for value in np.unique(oldroles):
        text = value.decode() if isinstance(value, bytes) else str(value)
        if text not in ROLES:
            raise ValueError(f"Unknown inherited role {text}")
        roles[row[oldroles == value]] = ROLES.index(text)
    del oldids, oldroles, row, hit
    bits = np.zeros((len(ids), (len(names) + 63) // 64), dtype="u8")
    error_bits = np.zeros_like(bits)
    # 0: unknown; 1: native no-XP; 2: native XP present.
    xp = np.zeros(len(ids), dtype="u1")
    for kind, folder in [("gaia", gaia / "labelled-gaia"), ("photometry", photo / "raw-observations")]:
        files = sorted(folder.glob("*.parquet"))
        if not files:
            raise ValueError(f"No staged {kind} partitions")
        for index, path in enumerate(files):
            parquet = pq.ParquetFile(path)
            available = set(parquet.schema_arrow.names)
            lookup = GAIA_VALUES if kind == "gaia" else {n: n for n in names}
            errlookup = GAIA_ERRORS if kind == "gaia" else errors
            requested = {"source_id", "candidate_row"}
            requested.update(c for c in lookup.values() if c in available)
            requested.update(c for c in errlookup.values() if c in available)
            if kind == "gaia":
                requested.add("has_xp_continuous")
                for band in ["g", "bp", "rp"]:
                    requested.update([f"phot_{band}_mean_flux", f"phot_{band}_mean_flux_error"])
            for batch in parquet.iter_batches(batch_size=32768, columns=sorted(requested & available)):
                row = positions(batch, ids)
                values = np.zeros((len(row), len(names)), dtype=bool)
                known_errors = np.zeros_like(values)
                for j, name in enumerate(names):
                    column = lookup.get(name)
                    if column not in batch.schema.names:
                        continue
                    values[:, j] = valid(batch[column].to_numpy(zero_copy_only=False))
                    error = errlookup.get(name)
                    if error in batch.schema.names:
                        e = batch[error].to_numpy(zero_copy_only=False)
                        known_errors[:, j] = values[:, j] & valid(e) & (e > 0)
                    if kind == "gaia" and name in ["G", "BP", "RP"]:
                        flux, sigma = f"phot_{name.lower()}_mean_flux", f"phot_{name.lower()}_mean_flux_error"
                        if flux in batch.schema.names and sigma in batch.schema.names:
                            f, e = batch[flux].to_numpy(zero_copy_only=False), batch[sigma].to_numpy(zero_copy_only=False)
                            known_errors[:, j] = values[:, j] & valid(f) & (f > 0) & valid(e) & (e > 0)
                add_bits(bits, row, values)
                add_bits(error_bits, row, known_errors)
                if kind == "gaia" and "has_xp_continuous" in batch.schema.names:
                    flag = batch["has_xp_continuous"]
                    if flag.null_count:
                        raise ValueError("Unknown native XP flags must not become False")
                    value = 1 + flag.to_numpy(zero_copy_only=False).astype("u1")
                    previous = xp[row]
                    if ((previous != 0) & (previous != value)).any():
                        raise ValueError("Conflicting native Gaia XP flags")
                    xp[row] = value
            if index % 100 == 0 or index == len(files) - 1:
                progress(kind, index + 1, len(files))
    group = roles * 3 + xp

    def counts(mask):
        n = np.bincount(group[mask], minlength=len(ROLES) * 3).reshape(len(ROLES), 3)
        return {role: {state: int(n[i, j]) for j, state in enumerate(["unknown_xp", "no_xp", "xp"])} for i, role in enumerate(ROLES)}

    channels = {}
    eligible = np.zeros(len(ids), dtype=bool)
    for j, name in enumerate(names):
        seen = (bits[:, j // 64] & (np.uint64(1) << np.uint64(j % 64))) != 0
        known = (error_bits[:, j // 64] & (np.uint64(1) << np.uint64(j % 64))) != 0
        channels[name] = {"visible_unique_stars": int(seen.sum()), "known_positive_error_unique_stars": int(known.sum()), "visible_by_role_and_native_xp": counts(seen)}
        if name not in ["PARALLAX", "EBV", "pmra", "pmdec", "RA", "DEC"]:
            eligible |= seen
    return {
        "candidate_stars": len(ids), "input_eligible_before_native_quality": counts(eligible),
        "parent_by_role_and_native_xp": counts(np.ones(len(ids), dtype=bool)),
        "channels": channels, "training_admitted": False,
        "scope": "Staged candidate union, not a population-uniform Gaia sample; finite native values only, before quality/domain admission",
        "pending": ["Native photometric flags and bandpass/error semantics", "New-star protected dependency-group role assignment", "Published age/alpha quality admission", "FIT-only normalization"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "running", "training_performed": False, "started_utc": dt.datetime.now(dt.UTC).isoformat()}
    def save():
        temporary = args.output / "manifest.tmp"
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output / "manifest.json")
    def progress(kind, done, total):
        report.update(current_stage=kind, completed_partitions=done, total_partitions=total)
        save()
    save()
    try:
        report.update(census(args.root, json.loads(args.contract.read_text()), progress), status="input_census_complete")
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save()


if __name__ == "__main__":
    main()
