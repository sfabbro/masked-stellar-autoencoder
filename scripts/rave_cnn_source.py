"""Raw-preserving, source-only RAVE DR6 CNN / Gaia EDR3 identity adapter.

The output records source facts and provisional per-target eligibility. It is not a
training set, calibration, or an independent validation of the published match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from astropy.io import fits

TARGETS = ("teff", "logg", "m_h", "fe_h", "alpha_m", "si_fe", "al_fe", "mg_fe", "ni_fe")
ID_NULL = np.iinfo(np.int64).max
MISSING_VALUE = -9999.0  # conservative candidate exclusion; raw value retained


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _stat(path: Path) -> tuple[int, ...]:
    s = path.stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def _checked_digest(path: Path, expected: str) -> tuple[str, tuple[int, ...]]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("an explicit lowercase SHA-256 is required for each source")
    before = _stat(path)
    actual = sha256_file(path)
    after = _stat(path)
    if before != after:
        raise ValueError(f"source changed while hashing: {path}")
    if actual != expected:
        raise ValueError(f"source SHA-256 mismatch: {path}")
    return actual, after


def _source_id(hdu: fits.BinTableHDU) -> np.ndarray:
    if "source_id" not in hdu.columns.names:
        raise ValueError("missing required source_id")
    column = hdu.columns["source_id"]
    if str(column.format).strip().upper() != "K" or column.null != ID_NULL:
        raise ValueError("source_id must be signed FITS K with positive int64 TNULL")
    raw = np.asarray(hdu.data["source_id"])
    if raw.ndim != 1 or raw.dtype.kind != "i" or raw.dtype.itemsize != 8:
        raise ValueError(
            "source_id must be a one-dimensional signed int64, never float"
        )
    return np.array(raw, dtype=np.int64, copy=True)


def _obs_column(hdu: fits.BinTableHDU) -> np.ndarray:
    if (
        "rave_obs_id" not in hdu.columns.names
        or str(hdu.columns["rave_obs_id"].format).strip().upper() != "32A"
    ):
        raise ValueError("rave_obs_id must use the documented FITS 32A schema")
    return _obs_id(hdu.data["rave_obs_id"])


def _float_column(hdu: fits.BinTableHDU, name: str) -> np.ndarray:
    col = hdu.columns[name]
    raw = np.asarray(hdu.data[name])
    if (
        str(col.format).strip().upper() != "E"
        or raw.dtype.kind != "f"
        or raw.dtype.itemsize != 4
    ):
        raise ValueError(f"{name} must use the native FITS float32 E schema")
    return np.array(raw, dtype=np.float32, copy=True)


def _obs_id(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw)
    if raw.ndim != 1 or raw.dtype.kind not in "SU":
        raise ValueError("rave_obs_id must be a one-dimensional string field")
    try:
        b = np.char.encode(raw, "ascii") if raw.dtype.kind == "U" else raw
    except UnicodeEncodeError as exc:
        raise ValueError("rave_obs_id must be ASCII") from exc
    if np.any(np.char.str_len(b) > 32):
        raise ValueError("rave_obs_id exceeds the documented 32-byte field")
    # Remove FITS padding only; never change case, leading spaces or ID content.
    return np.char.rstrip(np.array(b, dtype="S32", copy=True), b" ")


def _known_id(ids: np.ndarray) -> np.ndarray:
    return (ids > 0) & (ids != ID_NULL)


def _read_cnn(path: Path, expected_rows: int | None) -> dict[str, np.ndarray]:
    with fits.open(path, memmap=True) as hdus:
        hdu = hdus[1]
        names = set(hdu.columns.names)
        needed = {"rave_obs_id", "source_id", "snr_cnn"}
        needed.update(
            f"{t}_{kind}cnn" for t in TARGETS for kind in ("", "error_", "flag_")
        )
        missing = needed - names
        if missing:
            raise ValueError(f"CNN missing columns: {sorted(missing)}")
        n = len(hdu.data)
        if expected_rows is not None and n != expected_rows:
            raise ValueError(f"unexpected CNN row count: {n}")
        result = {
            "rave_obs_id": _obs_column(hdu),
            "gaia_dr2_source_id_raw": _source_id(hdu),
            "snr_cnn": _float_column(hdu, "snr_cnn"),
        }
        for key, suffix in (
            ("value", "cnn"),
            ("error", "error_cnn"),
            ("flag", "flag_cnn"),
        ):
            result[key] = np.stack(
                [_float_column(hdu, f"{t}_{suffix}") for t in TARGETS], axis=1
            )
        return result


def _read_xmatch(path: Path, expected_rows: int | None) -> dict[str, np.ndarray]:
    with fits.open(path, memmap=True) as hdus:
        hdu = hdus[1]
        if "rave_obs_id" not in hdu.columns.names:
            raise ValueError("crossmatch missing rave_obs_id")
        n = len(hdu.data)
        if expected_rows is not None and n != expected_rows:
            raise ValueError(f"unexpected crossmatch row count: {n}")
        return {
            "rave_obs_id": _obs_column(hdu),
            "gaia_edr3_source_id_raw": _source_id(hdu),
        }


def _key_multiplicity(keys: np.ndarray) -> np.ndarray:
    if len(keys) == 0:
        return np.zeros(0, dtype=np.int64)
    _, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    result = counts[inv].astype(np.int64)
    result[keys == b""] = 0
    return result


def _join(
    cnn: dict[str, np.ndarray], xmatch: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    keys = cnn["rave_obs_id"]
    xkeys = xmatch["rave_obs_id"]
    order = np.argsort(xkeys, kind="stable")
    ordered = xkeys[order]
    left = np.searchsorted(ordered, keys, side="left")
    right = np.searchsorted(ordered, keys, side="right")
    count = (right - left).astype(np.int64)
    count[keys == b""] = 0
    unique = count == 1
    cross_row = np.full(len(keys), -1, dtype=np.int64)
    cross_row[unique] = order[left[unique]]
    mapped = np.full(len(keys), -1, dtype=np.int64)
    mapped[unique] = xmatch["gaia_edr3_source_id_raw"][cross_row[unique]]
    mapped_known = unique & _known_id(mapped)
    # Preserve the raw EDR3 id on a single null row via the crossmatch group.
    mapped[~mapped_known] = -1
    obs_count = _key_multiplicity(keys)
    # Count distinct observations mapping to the same star, accounting for duplicate
    # reductions of an observation only once. These are dependence annotations.
    star_count = np.zeros(len(keys), dtype=np.int64)
    good = mapped_known & (keys != b"")
    if np.any(good):
        pairs = np.rec.fromarrays([mapped[good], keys[good]], names="id,obs")
        unique_pairs = np.unique(pairs)
        stars, nobs = np.unique(unique_pairs["id"], return_counts=True)
        star_count[good] = nobs[np.searchsorted(stars, mapped[good])]
    identity = mapped_known & (obs_count == 1)
    return {
        "crossmatch_count": count,
        "crossmatch_row": cross_row,
        "gaia_edr3_dr3_source_id": mapped,
        "gaia_edr3_dr3_id_known": mapped_known,
        "cnn_observation_count": obs_count,
        "mapped_distinct_observations_for_star": star_count,
        "identity_candidate": identity,
    }


def _counts(
    cnn: dict[str, np.ndarray],
    xmatch: dict[str, np.ndarray],
    join: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    per_target = {}
    for j, target in enumerate(TARGETS):
        per_target[target] = {k: int(v[:, j].sum()) for k, v in masks.items()}
    xknown = _known_id(xmatch["gaia_edr3_source_id_raw"])
    return {
        "cnn_rows": len(cnn["rave_obs_id"]),
        "crossmatch_rows": len(xmatch["rave_obs_id"]),
        "cnn_dr2_known_rows": int(_known_id(cnn["gaia_dr2_source_id_raw"]).sum()),
        "crossmatch_edr3_known_rows": int(xknown.sum()),
        "crossmatch_none_or_blank_rows": int((join["crossmatch_count"] == 0).sum()),
        "crossmatch_single_rows": int((join["crossmatch_count"] == 1).sum()),
        "crossmatch_ambiguous_rows": int((join["crossmatch_count"] > 1).sum()),
        "cnn_duplicate_observation_rows": int(
            (join["cnn_observation_count"] > 1).sum()
        ),
        "identity_candidate_rows": int(join["identity_candidate"].sum()),
        "identity_candidate_distinct_stars": int(
            np.unique(join["gaia_edr3_dr3_source_id"][join["identity_candidate"]]).size
        ),
        "mapped_rows_with_repeated_star_observations": int(
            (join["mapped_distinct_observations_for_star"] > 1).sum()
        ),
        "per_target": per_target,
    }


def _capture_raw_fits_table(group: h5py.Group, path: Path) -> None:
    """Preserve every HDU1 row (and FITS heap, if any) byte for byte."""
    with fits.open(path, memmap=True) as hdus:
        hdu = hdus[1]
        if hdu.header.get("XTENSION") != "BINTABLE":
            raise ValueError("HDU1 must be a FITS binary table")
        size = int(hdu.header["NAXIS1"]) * int(hdu.header["NAXIS2"]) + int(
            hdu.header.get("PCOUNT", 0)
        )
        offset = int(hdu.fileinfo()["datLoc"])
        if size < 0 or offset + size > path.stat().st_size:
            raise ValueError("invalid FITS table extent")
        group.attrs["raw_fits_header"] = hdu.header.tostring(
            sep="", endcard=True, padding=True
        )
        group.attrs["raw_fits_data_offset"] = offset
        group.attrs["raw_fits_row_bytes"] = int(hdu.header["NAXIS1"])
        group.attrs["raw_fits_heap_bytes"] = int(hdu.header.get("PCOUNT", 0))
    data = group.create_dataset(
        "raw_fits_table_and_heap",
        shape=(size,),
        dtype="u1",
        chunks=(min(1048576, size),) if size else None,
    )
    with path.open("rb") as source:
        source.seek(offset)
        for start in range(0, size, 1048576):
            length = min(1048576, size - start)
            raw = source.read(length)
            if len(raw) != length:
                raise ValueError("source truncated during FITS table capture")
            data[start : start + length] = np.frombuffer(raw, dtype=np.uint8)


def build_rave_cnn_source(
    cnn_path: str | Path,
    crossmatch_path: str | Path,
    output: str | Path,
    *,
    expected_cnn_sha256: str,
    expected_xmatch_sha256: str,
    expected_cnn_rows: int | None = None,
    expected_xmatch_rows: int | None = None,
) -> dict[str, Any]:
    cnn_path, crossmatch_path, output = map(Path, (cnn_path, crossmatch_path, output))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not output.parent.is_dir():
        raise ValueError("output parent directory must exist")
    cnn_hash, cnn_stat = _checked_digest(cnn_path, expected_cnn_sha256)
    x_hash, x_stat = _checked_digest(crossmatch_path, expected_xmatch_sha256)
    cnn = _read_cnn(cnn_path, expected_cnn_rows)
    xmatch = _read_xmatch(crossmatch_path, expected_xmatch_rows)
    if _stat(cnn_path) != cnn_stat or _stat(crossmatch_path) != x_stat:
        raise ValueError("source changed while reading")
    joined = _join(cnn, xmatch)
    available = np.isfinite(cnn["value"]) & (cnn["value"] != MISSING_VALUE)
    available[:, 0] &= cnn["value"][:, 0] > 0
    domain = np.isfinite(cnn["flag"]) & (cnn["flag"] == 0)
    error_known = (
        np.isfinite(cnn["error"]) & (cnn["error"] > 0) & (cnn["error"] != MISSING_VALUE)
    )
    masks = {
        "raw_value_available": available,
        "boundary_flag_in_domain": domain,
        "positive_finite_error_known": error_known,
        "provisional_target_candidate": available
        & domain
        & joined["identity_candidate"][:, None],
    }
    counts = _counts(cnn, xmatch, joined, masks)
    metadata = {
        "adapter": "rave_dr6_cnn_edr3_raw_union_v1",
        "target_order": TARGETS,
        "units": {"teff": "K", "logg": "dex(cm/s^2)", "abundances": "dex"},
        "sources": {
            "cnn": {
                "path": str(cnn_path),
                "sha256": cnn_hash,
                "bytes": cnn_stat[2],
                "rows": len(cnn["rave_obs_id"]),
                "gaia_identifier_release": "DR2",
            },
            "crossmatch": {
                "path": str(crossmatch_path),
                "sha256": x_hash,
                "bytes": x_stat[2],
                "rows": len(xmatch["rave_obs_id"]),
                "gaia_identifier_release": "EDR3",
            },
        },
        "source_documentation": [
            "https://www.rave-survey.org/metadata/ravedr6/dr6_cnn/",
            "https://www.rave-survey.org/metadata/ravedr6/dr6_x_gaiaedr3/",
            "https://arxiv.org/pdf/2004.12666",
        ],
        "identity": "Exact RAVE observation ID with trailing FITS padding removed; published RAVE positional best-neighbour EDR3 match, not revalidated. Duplicate observations and multiple matching catalogue rows excluded from candidate status. EDR3 identifier is used in DR3 source identity system, not reidentified by DR2 ID.",
        "dependency": "CNN trained on APOGEE DR16, using RAVE spectra, Gaia DR2 astrometry and photometry, 2MASS and WISE. Model-derived labels are not independent APOGEE or Gaia truth. Group repeated spectra and stars downstream.",
        "error_semantics": "Quoted errors are internal CNN training dispersion, not necessarily calibrated measurement uncertainties. Unknown/non-positive errors retained raw, marked unknown, and never replaced with zero; a downstream positive noise-floor policy is required.",
        "selection_semantics": "Target candidate = unique observation and valid unambiguous published crossmatch, finite native value, non-sentinel value and CNN boundary flag 0 (Teff also >0). Error availability is separate. No global target intersection, transformation, quality certification, training or role assignment is performed.",
        "provenance_caveat": "Local FITS headers and official metadata row counts differ for the EDR3 table; its exact publication version and identity require reconciliation against the source digest.",
        "counts": counts,
    }
    # Write an exclusive temporary file and atomically publish without overwriting.
    tmp = output.with_name(output.name + f".partial-{os.getpid()}")
    try:
        with h5py.File(tmp, "x") as file:
            file.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
            group = file.create_group("cnn")
            _capture_raw_fits_table(group, cnn_path)
            group.create_dataset(
                "source_row", data=np.arange(len(cnn["rave_obs_id"]), dtype=np.int64)
            )
            for key, data in cnn.items():
                group.create_dataset(key, data=data)
            group.create_dataset(
                "gaia_dr2_id_known", data=_known_id(cnn["gaia_dr2_source_id_raw"])
            )
            for key, data in joined.items():
                group.create_dataset(key, data=data)
            for key, data in masks.items():
                group.create_dataset(key, data=data)
            xgroup = file.create_group("crossmatch")
            _capture_raw_fits_table(xgroup, crossmatch_path)
            xgroup.create_dataset(
                "source_row", data=np.arange(len(xmatch["rave_obs_id"]), dtype=np.int64)
            )
            for key, data in xmatch.items():
                xgroup.create_dataset(key, data=data)
            xgroup.create_dataset(
                "gaia_edr3_id_known", data=_known_id(xmatch["gaia_edr3_source_id_raw"])
            )
        cnn_final, cnn_final_stat = _checked_digest(cnn_path, expected_cnn_sha256)
        cross_final, cross_final_stat = _checked_digest(
            crossmatch_path, expected_xmatch_sha256
        )
        if (
            cnn_final != cnn_hash
            or cross_final != x_hash
            or cnn_final_stat != cnn_stat
            or cross_final_stat != x_stat
        ):
            raise ValueError("source changed while producing output")
        os.link(tmp, output)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "output_bytes": output.stat().st_size,
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn", required=True)
    parser.add_argument("--crossmatch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cnn-sha256", required=True)
    parser.add_argument("--crossmatch-sha256", required=True)
    parser.add_argument("--expected-cnn-rows", type=int)
    parser.add_argument("--expected-crossmatch-rows", type=int)
    args = parser.parse_args()
    report = build_rave_cnn_source(
        args.cnn,
        args.crossmatch,
        args.output,
        expected_cnn_sha256=args.cnn_sha256,
        expected_xmatch_sha256=args.crossmatch_sha256,
        expected_cnn_rows=args.expected_cnn_rows,
        expected_xmatch_rows=args.expected_crossmatch_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
