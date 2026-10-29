"""Exact-ID source-only join for Gaia DR3 GSP-Spec M/H and Fe I/M.

One derived coordinate per raw Fe I/M row; it is not an independent third
measurement. Unknown source covariance and role/ancestry are not invented.
"""

from __future__ import annotations

import hashlib
import os
import re
import zipfile
from pathlib import Path

import h5py
import numpy as np

FIELDS = (
    "source_id",
    "mh_gspspec",
    "fem_gspspec",
    "fem_gspspec_lower",
    "fem_gspspec_upper",
    "fem_gspspec_nlines",
    "fem_gspspec_linescatter",
    "flags_gspspec",
)
GROUP = "native_observations/gaia_dr3_gspspec"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def state(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def validate_ids(ids, *, sorted_required=False):
    a = np.asarray(ids)
    if (
        a.ndim != 1
        or a.dtype.kind not in "iu"
        or a.dtype.itemsize != 8
        or np.any(a <= 0)
    ):
        raise ValueError("Gaia IDs must be exact positive 64-bit integers")
    if a.dtype.kind == "u" and np.any(a > np.iinfo("i8").max):
        raise ValueError("Gaia source ID exceeds int64")
    a = a.astype("i8", copy=False)
    if sorted_required:
        if np.any(a[1:] <= a[:-1]):
            raise ValueError("Fe/M IDs must be sorted and unique")
    elif np.unique(a).size != a.size:
        raise ValueError("Gaia sidecar IDs must be unique")
    return a


def flags_ascii(values):
    a = np.asarray(values)
    if (
        a.ndim == 2
        and a.shape[1] == 41
        and a.dtype.kind == "u"
        and a.dtype.itemsize == 1
    ):
        return np.ascontiguousarray(a).view("S41").reshape(-1)
    if a.ndim != 1 or a.dtype.kind not in "SU" or np.any(np.char.str_len(a) > 41):
        raise ValueError("Gaia flags must be aligned ASCII strings of length <= 41")
    return np.asarray(a, dtype="S41")


def build(
    sidecar,
    archive,
    output,
    *,
    expected_sidecar_sha,
    expected_archive_sha,
    expected_sidecar_rows=None,
    expected_archive_rows=None,
):
    sidecar, archive, output = map(Path, (sidecar, archive, output))
    if output.exists():
        raise FileExistsError(output)
    if not output.parent.is_dir():
        raise ValueError("Output directory must exist")
    for h in (expected_sidecar_sha, expected_archive_sha):
        if not re.fullmatch("[0-9a-f]{64}", h):
            raise ValueError("Explicit SHA-256 required")
    side_state, arc_state = state(sidecar), state(archive)
    if sha(sidecar) != expected_sidecar_sha or sha(archive) != expected_archive_sha:
        raise ValueError("Input SHA differs")
    if (state(sidecar), state(archive)) != (side_state, arc_state):
        raise ValueError("Input changed while hashing")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError("Fe/M archive CRC mismatch")
    with np.load(archive, allow_pickle=False) as z:
        if not set(FIELDS).issubset(z.files):
            raise ValueError("Fe/M source fields missing")
        arrays = {name: z[name].copy() for name in FIELDS}
    ids = validate_ids(arrays["source_id"], sorted_required=True)
    n = len(ids)
    if expected_archive_rows is not None and n != expected_archive_rows:
        raise ValueError("Fe/M row count differs")
    if any(a.shape != (n,) for a in arrays.values()):
        raise ValueError("Fe/M arrays misaligned")
    flags = flags_ascii(arrays["flags_gspspec"])
    with h5py.File(sidecar, "r") as h5:
        group = h5[GROUP]
        sid = validate_ids(group["source_id"][:])
        if expected_sidecar_rows is not None and len(sid) != expected_sidecar_rows:
            raise ValueError("Sidecar row count differs")
        order = np.argsort(sid, kind="stable")
        ordered = sid[order]
        pos = np.searchsorted(ordered, ids)
        matched = pos < len(ordered)
        ix = np.flatnonzero(matched)
        matched[ix] &= ordered[pos[ix]] == ids[ix]
        ix = np.flatnonzero(matched)
        rows = np.full(n, -1, dtype="i8")
        rows[ix] = order[pos[ix]]
        # h5py requires sorted row indices; load source arrays once to preserve ID order.
        native_mh = np.asarray(group["mh_gspspec"][:], dtype="f8")
        native_flags = flags_ascii(group["flags_ascii"][:])
        selected_mh = np.asarray(group["selected_observed_mh"][:], dtype=bool)
        if any(x.shape != (len(sid),) for x in (native_mh, native_flags, selected_mh)):
            raise ValueError("Gaia sidecar columns misaligned")
        if not np.array_equal(flags[ix], native_flags[rows[ix]]):
            raise ValueError("Common exact-ID Gaia GSP-Spec flags differ")
        tap_mh = np.asarray(arrays["mh_gspspec"], dtype="f8")
        if not np.isclose(
            tap_mh[ix], native_mh[rows[ix]], atol=2e-5, rtol=0, equal_nan=True
        ).all():
            raise ValueError("Common exact-ID Gaia M/H differs by more than 2e-5 dex")
        mh_selected = np.zeros(n, dtype=bool)
        mh_selected[ix] = selected_mh[rows[ix]]
        alpha = np.full(n, np.nan, dtype="f8")
        alpha[ix] = np.asarray(group["alphafe_gspspec"][:], dtype="f8")[rows[ix]]
    fem = np.asarray(arrays["fem_gspspec"], dtype="f8")
    raw = np.isfinite(tap_mh) & np.isfinite(fem) & (tap_mh != -9999) & (fem != -9999)
    derived = np.where(raw, tap_mh + fem, np.nan)
    digits = np.ascontiguousarray(flags).view("u1").reshape(-1, 41)
    # Gaia published positions 28 (FeUpLim) and 29 (FeUncer), zero-based 27 and 28.
    detected = digits[:, 27] == ord("0")
    candidate = matched & raw & mh_selected & detected
    if (state(sidecar), state(archive)) != (side_state, arc_state):
        raise ValueError("Input changed during join")
    partial = output.with_name(output.name + f".partial-{os.getpid()}")
    try:
        with h5py.File(partial, "x") as out:
            out.attrs["schema"] = "gaia-gspspec-fem-exact-join-v1"
            out.attrs["source_only"] = True
            out.attrs["dependency"] = (
                "Fe I/H derived from the same M/H and Fe I/M, not independent; covariance unknown"
            )
            out.attrs["match_semantics"] = (
                "exact Gaia DR3 int64 source ID; unmatched records retained; no role assignment"
            )
            out.attrs["sidecar_sha256"] = expected_sidecar_sha
            out.attrs["archive_sha256"] = expected_archive_sha
            out.attrs["mh_match_absolute_tolerance_dex"] = 2e-5
            for name, value in arrays.items():
                out.create_dataset(name, data=value)
            for name, value in {
                "sidecar_row": rows,
                "sidecar_matched": matched,
                "sidecar_selected_mh": mh_selected,
                "sidecar_alphafe": alpha,
                "derived_feh_native": derived,
                "derived_feh_raw_available": raw,
                "fe_upper_limit_flag": digits[:, 27],
                "fe_uncertainty_flag": digits[:, 28],
                "fe_detection_flag_zero": detected,
                "provisional_derived_feh_candidate": candidate,
            }.items():
                out.create_dataset(name, data=value)
        with h5py.File(partial, "r") as h5:
            if not np.array_equal(h5["source_id"][:], ids) or not np.array_equal(
                h5["sidecar_matched"][:], matched
            ):
                raise ValueError("Gaia output roundtrip mismatch")
            if not np.array_equal(
                h5["provisional_derived_feh_candidate"][:], candidate
            ):
                raise ValueError("Gaia selection roundtrip mismatch")
        os.link(partial, output)
    finally:
        if partial.exists():
            partial.unlink()
    return {
        "output": str(output),
        "output_sha256": sha(output),
        "output_bytes": output.stat().st_size,
        "counts": {
            "archive_rows": n,
            "sidecar_rows": len(sid),
            "matched_rows": int(matched.sum()),
            "unmatched_rows": int((~matched).sum()),
            "raw_derived_available": int(raw.sum()),
            "matched_detection_candidates": int(candidate.sum()),
        },
        "scope": "Exact-ID source join only; not independent labels, calibrated uncertainty, assigned roles, feature eligibility or model training.",
    }
