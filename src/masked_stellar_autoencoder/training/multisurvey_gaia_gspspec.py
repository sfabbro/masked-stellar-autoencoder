"""Stage Gaia DR3 GSP-Spec as a sparse, separately flagged survey source.

This adapter keeps raw RVS quantities and 16th/84th percentiles, but it does
not convert those percentiles to independent Gaussian measurement errors. The
current three-target consensus interface receives Teff and logg only: Gaia's
[M/H] and alpha remain separately named diagnostics, not invented Fe/H.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

from .multisurvey_overlay import file_sha256, write_observation_sidecar

SOURCE = "gaia_dr3_gspspec"
REGISTRY = {
    "release": "Gaia DR3",
    "pipeline": "GSP-Spec MatisseGauguin from RVS spectra; published uncalibrated scale",
    "dependency_id": "gaia-dr3-gspspec-rvs",
    "targets": {
        "teff": {"quantity": "teff", "unit": "K", "field": "teff_gspspec"},
        "logg": {"quantity": "logg", "unit": "dex(cm/s2)", "field": "logg_gspspec"},
        "fe_h": None,
    },
    "quality_policy": "provisional per-target GSP-Spec flag tier and broad sanity bounds; all raw rows retained",
    "uncertainty": "16th/84th Monte Carlo percentiles retained; Gaussian random errors not asserted",
    "other_quantities": "mh_gspspec is global M/H and alphafe_gspspec is source-specific alpha/Fe, not direct Fe/H",
    "physical_truth_claim": False,
}


def quality_masks(teff, logg, flags, *, mh=None, alpha=None):
    """Return raw and provisional minimal per-target masks without intersections.

    Zero-based GSP-Spec flag positions: 0/3 temperature broadening/RV,
    1/4 gravity broadening/RV, 6 noise, 7 extrapolation, 8 negative flux,
    9 NaN flux, 10 emission, and 11 null flux errors. The provisional tier
    allows broadening/RV values 0 or 1 for the affected target, noise 0--3,
    extrapolation 0--2, negative flux 0--1 and requires the last three to be 0.
    Broad sanity bounds are 2500--10000 K and -1--6.5 dex. If temperature is
    present but outside its broad bound, gravity is also rejected because the
    joint atmospheric fit is suspect; a missing temperature alone does not
    make a valid gravity missing. These cuts are not physical-scale calibration.
    """
    t = np.asarray(teff, dtype=np.float64)
    g = np.asarray(logg, dtype=np.float64)
    source_flags = np.asarray(flags)
    if t.ndim != 1 or g.shape != t.shape or source_flags.shape != t.shape:
        raise ValueError("Gaia targets and flags must be aligned vectors")
    if source_flags.dtype.kind not in "SU":
        raise ValueError("GSP-Spec flags must be text or ASCII bytes")
    if np.any(np.char.str_len(source_flags) > 41):
        raise ValueError("GSP-Spec flags cannot exceed 41 characters")
    try:
        text = np.ascontiguousarray(source_flags, dtype="S41")
    except UnicodeEncodeError as error:
        raise ValueError("GSP-Spec flags must be ASCII") from error
    digits = text.view(np.uint8).reshape(-1, 41)
    is_digit = (digits >= ord("0")) & (digits <= ord("9"))
    shared = (
        is_digit[:, 6:12].all(axis=1)
        & (digits[:, 6] <= ord("3"))
        & (digits[:, 7] <= ord("2"))
        & (digits[:, 8] <= ord("1"))
        & (digits[:, 9] == ord("0"))
        & (digits[:, 10] == ord("0"))
        & (digits[:, 11] == ord("0"))
    )
    teff_flags = (
        is_digit[:, [0, 3]].all(axis=1)
        & (digits[:, 0] <= ord("1"))
        & (digits[:, 3] <= ord("1"))
    )
    logg_flags = (
        is_digit[:, [1, 4]].all(axis=1)
        & (digits[:, 1] <= ord("1"))
        & (digits[:, 4] <= ord("1"))
    )
    raw_t = np.isfinite(t) & (t > 0) & (t != -9999)
    raw_g = np.isfinite(g) & (g != -9999)
    domain_t = (t >= 2500) & (t <= 10000)
    domain_g = (g >= -1) & (g <= 6.5)
    temperature_context = np.isnan(t) | (t == -9999) | domain_t
    selected_t = raw_t & domain_t & shared & teff_flags
    selected_g = raw_g & domain_g & temperature_context & shared & logg_flags
    raw = np.column_stack((raw_t, raw_g, np.zeros(len(t), dtype=bool)))
    selected = np.column_stack((selected_t, selected_g, np.zeros(len(t), dtype=bool)))
    result = {"raw": raw, "selected": selected, "flags_ascii": digits.copy()}
    for name, values in (("mh", mh), ("alpha", alpha)):
        if values is None:
            continue
        values = np.asarray(values, dtype=np.float64)
        if values.shape != t.shape:
            raise ValueError("Gaia abundance values must align with source IDs")
        observed = np.isfinite(values) & (values != -9999)
        metallicity_flags = (
            is_digit[:, [2, 5]].all(axis=1)
            & (digits[:, 2] <= ord("1"))
            & (digits[:, 5] <= ord("1"))
        )
        candidate = observed & shared & temperature_context & metallicity_flags
        if name == "alpha":
            # Gaia's KMgiantPar warning marks alpha as unreliable, even when
            # an atmospheric parameter remains usable. Keep the raw value.
            candidate &= (
                teff_flags & logg_flags & is_digit[:, 12] & (digits[:, 12] == ord("0"))
            )
        result["raw_" + name] = observed
        result["selected_" + name] = candidate
    return result


def _stat(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def build_gaia_sidecar(source, expected_sha256, output, *, expected_rows=None):
    """Create a new raw-preserving sidecar from an immutable staged FITS file."""
    source = Path(source)
    before = _stat(source)
    if file_sha256(source) != expected_sha256 or _stat(source) != before:
        raise ValueError("Gaia source content or file identity differs")
    fields = (
        "source_id",
        "teff_gspspec",
        "teff_gspspec_lower",
        "teff_gspspec_upper",
        "logg_gspspec",
        "logg_gspspec_lower",
        "logg_gspspec_upper",
        "mh_gspspec",
        "mh_gspspec_lower",
        "mh_gspspec_upper",
        "alphafe_gspspec",
        "alphafe_gspspec_lower",
        "alphafe_gspspec_upper",
        "flags_gspspec",
    )
    with fits.open(source, memmap=True) as hdus:
        data = hdus[1].data
        if expected_rows is not None and len(data) != expected_rows:
            raise ValueError("Gaia source row count differs")
        if not set(fields).issubset(data.dtype.names or ()):
            raise ValueError("Gaia source lacks required GSP-Spec fields")
        ids = np.asarray(data["source_id"])
        if ids.dtype.kind not in "iu" or np.any(ids <= 0):
            raise ValueError("Gaia source identities must be positive exact integers")
        if ids.dtype.kind == "u" and np.any(ids > np.iinfo("i8").max):
            raise ValueError("Gaia source identity exceeds int64")
        ids = ids.astype("i8", copy=False)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Gaia source identities must be unique")
        teff = np.asarray(data["teff_gspspec"], dtype="f8")
        logg = np.asarray(data["logg_gspspec"], dtype="f8")
        masks = quality_masks(
            teff,
            logg,
            data["flags_gspspec"],
            mh=data["mh_gspspec"],
            alpha=data["alphafe_gspspec"],
        )
        n = len(ids)
        missing = np.full(n, np.nan, dtype="f8")
        block = {
            "source_id": ids,
            "source_row": np.arange(n, dtype="i8"),
            "value": np.column_stack((teff, logg, missing)),
            "error": np.full((n, 3), np.nan, dtype="f8"),
            "observed": masks["selected"],
            "raw_observed": masks["raw"],
            "flags_ascii": masks["flags_ascii"],
            "raw_observed_mh": masks["raw_mh"],
            "selected_observed_mh": masks["selected_mh"],
            "raw_observed_alpha": masks["raw_alpha"],
            "selected_observed_alpha": masks["selected_alpha"],
            "percentile16": np.column_stack(
                (data["teff_gspspec_lower"], data["logg_gspspec_lower"], missing)
            ),
            "percentile84": np.column_stack(
                (data["teff_gspspec_upper"], data["logg_gspspec_upper"], missing)
            ),
        }
        for target in ("mh", "alphafe"):
            for suffix in ("", "_lower", "_upper"):
                field = target + "_gspspec" + suffix
                block[field] = np.asarray(data[field], dtype="f8")
        provenance = {
            "source_path": str(source.resolve()),
            "source_sha256": expected_sha256,
            "source_rows": n,
            "source_identity_scope": "project-staged FITS hash; three rows matched public Gaia DR3, not a publisher checksum",
            "selection": REGISTRY["quality_policy"],
            "raw_fields": list(fields),
            "scope": "No survey calibration, physical-truth certification, Fe/H derivation or model training",
        }
        receipt = write_observation_sidecar(
            output, {SOURCE: block}, {SOURCE: REGISTRY}, provenance=provenance
        )
        counts = {
            "rows": n,
            "raw_teff": int(masks["raw"][:, 0].sum()),
            "raw_logg": int(masks["raw"][:, 1].sum()),
            "selected_teff": int(masks["selected"][:, 0].sum()),
            "selected_logg": int(masks["selected"][:, 1].sum()),
            "selected_both": int(masks["selected"][:, :2].all(axis=1).sum()),
            "selected_fe_h": 0,
            "raw_mh": int(masks["raw_mh"].sum()),
            "selected_mh": int(masks["selected_mh"].sum()),
            "raw_alpha": int(masks["raw_alpha"].sum()),
            "selected_alpha": int(masks["selected_alpha"].sum()),
        }
    if _stat(source) != before:
        raise ValueError("Gaia source changed during sidecar build")
    return {**receipt, "source_sha256": expected_sha256, "counts": counts}


def summarize_gaia_roles(base_path, sidecar_path):
    """Count Gaia values by existing star role; never assign outside-base roles."""
    with h5py.File(base_path, "r") as base, h5py.File(sidecar_path, "r") as gaia:
        ids = np.asarray(base["source_id"][:])
        if ids.dtype.kind not in "iu" or np.any(np.diff(ids) <= 0):
            raise ValueError("Base Gaia IDs must be sorted exact unique integers")
        group = gaia["native_observations/" + SOURCE]
        source_ids = np.asarray(group["source_id"][:])
        if source_ids.dtype.kind not in "iu" or len(np.unique(source_ids)) != len(
            source_ids
        ):
            raise ValueError("Gaia source identities must be exact and unique")
        pos = np.searchsorted(ids, source_ids)
        matched = pos < len(ids)
        indices = np.flatnonzero(matched)
        matched[indices] &= ids[pos[indices]] == source_ids[indices]
        role = np.asarray(base["role"][:]).astype("U")
        primary = np.asarray(base["primary_input_eligible"][:], dtype=bool)
        if role.shape != ids.shape or primary.shape != ids.shape:
            raise ValueError("Base roles and input eligibility must align")
        observed = group["observed"][:]
        mh = group["selected_observed_mh"][:]
        alpha = group["selected_observed_alpha"][:]
        result = {}
        for name in ("fit", "tune", "cal", "devtest", "external", "excluded"):
            mask = matched.copy()
            index = np.flatnonzero(mask)
            mask[index] &= (role[pos[index]] == name) & primary[pos[index]]
            result[name] = {
                "gaia_id_matches_with_primary_input": int(mask.sum()),
                "teff": int((mask & observed[:, 0]).sum()),
                "logg": int((mask & observed[:, 1]).sum()),
                "mh_source_specific": int((mask & mh).sum()),
                "alpha_source_specific": int((mask & alpha).sum()),
                "fe_h_direct": 0,
            }
        return {
            "roles": result,
            "gaia_source_rows": len(source_ids),
            "matched_base_ids": int(matched.sum()),
            "outside_base_ids": int((~matched).sum()),
            "scope": "Per-source reference availability by existing role, not a validated physical scale; outside-base roles and feature availability remain unknown.",
        }
