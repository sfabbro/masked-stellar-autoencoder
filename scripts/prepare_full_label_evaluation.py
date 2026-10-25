"""Uncapped source-only evaluation assembly; no model, optimizer or latent pass.

Every current FIT/TUNE/CAL identity is excluded. Existing TEST inputs and
references remain separately recoverable. Survey scales and posterior targets
are never averaged or silently renamed as physical truth.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from build_age_anchor_labels import APOKASC3_COLUMNS
from prepare_five_job_source import join_catalogue

from masked_stellar_autoencoder.training.evaluation_feature_lookup import (
    lookup_evaluation_features,
)

NAMES = ("teff", "logg", "fe_h", "alpha", "age", "distance")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(spec):
    path = Path(spec["path"])
    if sha(path) != spec["sha256"]:
        raise ValueError("Pinned source changed: " + str(path))
    return path


def exact_ids(values):
    values = np.asarray(values)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError("Exact integer Gaia IDs required; float identities forbidden")
    if values.dtype.kind == "u" and np.any(values > np.iinfo("i8").max):
        raise ValueError("Gaia ID outside int64")
    return values.astype("i8", copy=False)


def first_unique(ids):
    """First original quality-eligible source row, independent of residual/error."""
    ids = exact_ids(ids)
    good = np.flatnonzero(ids > 0)
    _, first = np.unique(ids[good], return_index=True)
    return good[first]


def ordered_bounds(bounds):
    bounds = np.asarray(bounds)
    return (
        np.isfinite(bounds).all(1)
        & (bounds > 0).all(1)
        & (np.diff(bounds, axis=1) > 0).all(1)
    )


def posterior_quality(age, distance, ageflag, inputflag, outputflag, ageinout):
    """Same conservative source-flag/precision policy as the existing cohorts."""
    ageflag, inputflag, outputflag, ageinout = [
        np.char.strip(np.asarray(v).astype("S128"))
        for v in (ageflag, inputflag, outputflag, ageinout)
    ]
    clean = (outputflag == b"") & (ageinout == b"")
    clean &= (np.char.find(inputflag, b"WARN") < 0) & (
        np.char.find(inputflag, b"BAD") < 0
    )

    def valid(value, precision):
        result = (
            np.isfinite(value).all(1)
            & (value > 0).all(1)
            & (np.diff(value, axis=1) >= 0).all(1)
        )
        relative = np.full(len(value), np.inf)
        relative[result] = (value[result, 3] - value[result, 1]) / (
            2 * value[result, 2]
        )
        return result & (relative <= precision) & ordered_bounds(value[:, [1, 2, 3]])

    return clean & np.isin(ageflag, [b"MSTO", b"SGB"]) & valid(
        age, 0.30
    ), clean & valid(distance, 0.20)


def native_block(config):
    path = checked(config["native"])
    qa = json.loads(Path(config["native"]["source_qa"]).read_text())
    if (
        qa.get("status") != "passed"
        or qa.get("sidecar_sha256") != config["native"]["sha256"]
        or qa.get("cohort_sha256") != config["feature_lookup"]["legacy_sha256"]
        or not all(
            qa.get(key) is True
            for key in (
                "native_labels_verified",
                "all_original_source_target_values_errors_flags_rows_verified",
                "all_target_quality_masks_verified",
                "all_source_id_roles_and_reference_bans_verified",
            )
        )
    ):
        raise ValueError("Original native all-row source proof missing")
    with h5py.File(path) as f:
        ids = exact_ids(f["source_id"][:])
        value = f["targets/value"][:]
        error = f["targets/uncertainty"][:]
        valid = f["targets/observed"][:].astype(bool) & f["targets/quality_pass"][
            :
        ].astype(bool)
        valid &= np.isfinite(value) & np.isfinite(error) & (error > 0)
        sources = f["targets/survey"][:].astype("U64")
    if value.shape != (len(ids), 3) or len(np.unique(ids)) != len(ids):
        raise ValueError("Native source identity/layout differs")
    return dict(
        ids=ids,
        columns=(0, 1, 2),
        value=value,
        error=error,
        valid=valid,
        primary=valid.copy(),
        sources=sources,
        name="native_three",
    )


def alpha_block(config):
    spec = config["alpha"]
    path = checked(spec)
    audit = json.loads(checked(spec["source_audit"]).read_text())
    if (
        audit.get("status") != "all_original_projection_rows_independently_verified"
        or audit.get("sidecar_sha256") != spec["sha256"]
        or audit.get("source_sha256") != spec["raw_sha256"]
    ):
        raise ValueError("Reused all-row DESI projection proof differs")
    ids, value, error, feh, row = [], [], [], [], []
    with h5py.File(path) as f:
        data = f["measurements"]
        for start in range(0, len(data), 65536):
            block = data[start : start + 65536]
            valid = block["eligible"].astype(bool)
            valid &= (
                np.isfinite(block["alpha_fe"])
                & np.isfinite(block["alpha_fe_error"])
                & (block["alpha_fe_error"] > 0)
            )
            ids.append(exact_ids(block["source_id"])[valid])
            value.append(block["alpha_fe"][valid])
            error.append(block["alpha_fe_error"][valid])
            feh.append(block["feh_raw"][valid])
            row.append(block["source_row"][valid])
    ids = np.concatenate(ids)
    count = len(ids)
    keep = first_unique(ids)
    value, error, feh, row = [np.concatenate(v)[keep] for v in (value, error, feh, row)]
    ids = ids[keep]
    primary = np.isfinite(feh) & (feh > -2)
    return dict(
        ids=ids,
        columns=(3,),
        value=value[:, None],
        error=error[:, None],
        valid=np.ones((len(ids), 1), bool),
        primary=primary[:, None],
        sources=np.full(len(ids), "DESI_RVSpecFit_alpha_fe"),
        feh=feh,
        row=row,
        name="DESI_alpha",
        raw_eligible_measurement_rows=count,
        duplicate_excess=count - len(ids),
    )


def posterior_blocks(config):
    blocks = []
    with h5py.File(checked(config["posterior"])) as f:
        for survey in ("apogee17", "galahdr3"):
            g = f[survey]
            ids = exact_ids(g["source_id"][:])
            age = g["Age_percentiles"][:]
            distance = g["Dist_percentiles"][:]
            for key, unit in (("Age_percentiles", "Gyr"), ("Dist_percentiles", "kpc")):
                if g[key].attrs.get("unit") != unit:
                    raise ValueError("Posterior native unit differs")
            valid_age, valid_distance = posterior_quality(
                age,
                distance,
                *[
                    g["flags/" + key][:]
                    for key in ("AgeFlag", "InputFlag", "OutputFlag", "AgeInOut")
                ],
            )
            keep = first_unique(ids)
            ids = ids[keep]
            bounds = np.stack(
                (age[keep][:, [1, 2, 3]], distance[keep][:, [1, 2, 3]]), 1
            )
            valid = np.column_stack((valid_age[keep], valid_distance[keep]))
            valid &= ids[:, None] > 0
            blocks.append(
                dict(
                    ids=ids,
                    columns=(4, 5),
                    value=bounds[:, :, 1],
                    error=np.full((len(ids), 2), np.nan),
                    valid=valid,
                    primary=valid.copy(),
                    bounds=bounds,
                    sources=np.full(
                        len(ids), "StarHorse_" + survey + "_posterior_median"
                    ),
                    name="StarHorse_" + survey,
                )
            )
    return blocks


def apokasc_panel(config):
    """Parse Gaia identifiers as integers directly, never through float64."""
    spec = config["external_apokasc3"]
    if not spec.get("enabled", True):
        return dict(
            ids=np.empty(0, dtype="i8"),
            value=np.empty(0),
            bounds=np.empty((0, 3)),
            raw_rows=0,
        )
    raw = pd.read_csv(
        checked(spec),
        sep=r"\s+",
        names=APOKASC3_COLUMNS,
        dtype=str,
        engine="python",
        compression="infer",
    )
    ids = np.array(
        [
            int(v)
            if isinstance(v, str) and v.isdecimal() and 0 < int(v) <= np.iinfo("i8").max
            else 0
            for v in raw.GaiaDR3
        ],
        dtype="i8",
    )
    numeric = {
        key: pd.to_numeric(raw[key], errors="coerce")
        .replace(-9999, np.nan)
        .to_numpy(float)
        for key in ("AgeRGB", "AgeRC", "e_AgeRGB", "E_AgeRGB", "e_AgeRC", "E_AgeRC")
    }
    rgb = raw.AgeCat.isin(["RGB", "RGB_AGB"]).to_numpy()
    rc = raw.AgeCat.isin(["RC", "RC/RGB"]).to_numpy()
    mid = np.where(rgb, numeric["AgeRGB"], np.where(rc, numeric["AgeRC"], np.nan))
    lo = mid + np.where(
        rgb, numeric["e_AgeRGB"], np.where(rc, numeric["e_AgeRC"], np.nan)
    )
    hi = mid + np.where(
        rgb, numeric["E_AgeRGB"], np.where(rc, numeric["E_AgeRC"], np.nan)
    )
    bounds = np.column_stack((lo, mid, hi))
    valid = (ids > 0) & ordered_bounds(bounds)
    take = np.flatnonzero(valid)
    take = take[first_unique(ids[take])]
    return dict(ids=ids[take], value=mid[take], bounds=bounds[take], raw_rows=len(raw))


def desi_photometry_fallback(config, ids):
    """New source quantities: stream real Gaia magnitudes, verify every ID row.

    Missing publication/release metadata stays diagnostic, even when usable
    photometry exists. No Gaia-release mapping or XP availability is invented.
    """
    spec = config["desi_photometry"]
    if not spec.get("enabled", True):
        return None, {"status": "disabled"}
    path = Path(spec["path"])
    if sha(path) != spec["sha256"]:
        raise ValueError("New DESI Gaia photometry source SHA differs")
    n = len(ids)
    values = np.full((n, 3), np.nan, dtype="f4")
    errors = np.full((n, 3), np.nan, dtype="f4")
    verified = np.zeros(n, bool)
    conflict = np.zeros(n, bool)
    seen = np.zeros(n, bool)
    proof = dict(
        source_path=str(path),
        source_sha256=spec["sha256"],
        release="unknown",
        unknown_release_diagnostic_only=True,
        new_photometry_projection=True,
    )
    with (
        fits.open(path, memmap=True) as hdus,
        h5py.File(config["alpha"]["path"]) as side,
    ):
        if "GAIA" not in hdus:
            return None, dict(proof, status="gaia_extension_unavailable")
        hdu = hdus["GAIA"]
        columns = {name.upper(): name for name in hdu.columns.names}
        id_name = columns.get("SOURCE_ID")
        magnitude_names = []
        for band in ("G", "BP", "RP"):
            aliases = (f"PHOT_{band}_MEAN_MAG", f"GAIA_PHOT_{band}_MEAN_MAG")
            magnitude_names.append(
                next((columns[a] for a in aliases if a in columns), None)
            )
        proof["actual_columns"] = magnitude_names
        proof["actual_units"] = [
            hdu.columns[name].unit if name else None for name in magnitude_names
        ]
        proof["unit_evidence"] = (
            "Only explicitly named Gaia mean-magnitude columns; native TUNIT values recorded, blank units not falsely called calibrated. Release unresolved, diagnostic only."
        )
        if (
            id_name is None
            or magnitude_names[0] is None
            or sum(v is not None for v in magnitude_names) < 2
        ):
            return None, dict(
                proof, status="required_gaia_magnitude_columns_unavailable"
            )
        saved = side["measurements"]
        if len(saved) != len(hdu.data):
            raise ValueError("DESI Gaia projection row count differs")
        for start in range(0, len(saved), 65536):
            stop = min(start + 65536, len(saved))
            source = hdu.data[start:stop]
            source_ids = exact_ids(source[id_name])
            np.testing.assert_array_equal(source_ids, saved[start:stop]["source_id"])
            pos = np.searchsorted(ids, source_ids)
            match = (pos < n) & (source_ids > 0)
            match[match] &= ids[pos[match]] == source_ids[match]
            if not match.any():
                continue
            rows = np.flatnonzero(match)
            dest = pos[match]
            magnitude = np.full((len(rows), 3), np.nan)
            uncertainty = np.full_like(magnitude, np.nan)
            for column, name in enumerate(magnitude_names):
                if name is None:
                    continue
                unit = hdu.columns[name].unit
                if unit is not None and str(unit).strip().lower() != "mag":
                    raise ValueError(
                        "Explicit Gaia magnitude column has conflicting units"
                    )
                magnitude[:, column] = np.asarray(source[name][rows], dtype="f8")
                band = ("G", "BP", "RP")[column]
                flux_names = (f"PHOT_{band}_MEAN_FLUX", f"GAIA_PHOT_{band}_MEAN_FLUX")
                error_names = (
                    f"PHOT_{band}_MEAN_FLUX_ERROR",
                    f"GAIA_PHOT_{band}_MEAN_FLUX_ERROR",
                )
                flux_name = next((columns[a] for a in flux_names if a in columns), None)
                error_name = next(
                    (columns[a] for a in error_names if a in columns), None
                )
                if flux_name and error_name:
                    flux = np.asarray(source[flux_name][rows], dtype="f8")
                    error = np.asarray(source[error_name][rows], dtype="f8")
                    valid = (
                        np.isfinite(flux)
                        & np.isfinite(error)
                        & (flux > 0)
                        & (error > 0)
                    )
                    uncertainty[valid, column] = (
                        (2.5 / np.log(10)) * error[valid] / flux[valid]
                    )
            magnitude[
                (~np.isfinite(magnitude)) | (magnitude <= -20) | (magnitude >= 40)
            ] = np.nan
            uncertainty[
                (~np.isfinite(uncertainty)) | (uncertainty <= 0) | (uncertainty > 5)
            ] = np.nan
            # Repeated source IDs may not quietly select a favorable magnitude.
            for offset, destination in enumerate(dest):
                if seen[destination]:
                    if not np.allclose(
                        values[destination],
                        magnitude[offset],
                        atol=1e-6,
                        rtol=0,
                        equal_nan=True,
                    ):
                        conflict[destination] = True
                else:
                    values[destination] = magnitude[offset]
                    errors[destination] = uncertainty[offset]
                    seen[destination] = True
                    verified[destination] = True
    verified &= ~conflict
    proof.update(
        status="source_id_associated_diagnostic_photometry",
        all_original_gaia_ids_verified=True,
        candidate_photometry_rows=int(seen.sum()),
        conflicting_candidate_ids=int(conflict.sum()),
    )
    return dict(
        source_id=ids,
        names=["G", "BP", "RP"],
        values=values,
        errors=errors,
        identity_verified=verified,
        release="unknown",
        provenance=proof,
    ), proof


def assemble_labels(blocks, base, ancestor_ids, external):
    """Join all source labels before any feature selection or old split sampling."""
    roles = np.asarray(base["role"]).astype(str)
    old = np.flatnonzero(roles == "test")
    protected = exact_ids(base["source_id"])[np.isin(roles, ["fit", "tune", "cal"])]
    original_ids = exact_ids(base["source_id"])[old]
    union = np.unique(
        np.concatenate(
            [
                original_ids,
                external["ids"],
                *[b["ids"][b["valid"].any(1)] for b in blocks],
            ]
        )
    )
    protected_hit = np.isin(union, protected)
    ancestor_hit = np.isin(union, ancestor_ids)
    original_hit = np.isin(union, original_ids)
    exclude = protected_hit | (ancestor_hit & ~original_hit)
    counts = dict(
        union_label_ids_before_exclusions=len(union),
        current_fit_tune_cal_excluded=int(protected_hit.sum()),
        new_ancestor_feh_ids_excluded=int(
            (ancestor_hit & ~original_hit & ~protected_hit).sum()
        ),
    )
    ids = union[~exclude]
    n = len(ids)
    result = dict(
        source_id=ids,
        value=np.full((n, 6), np.nan),
        error=np.full((n, 6), np.nan),
        observed=np.zeros((n, 6), bool),
        quality=np.zeros((n, 6), bool),
        diagnostic_eligible=np.zeros((n, 6), bool),
        error_observed=np.zeros((n, 6), bool),
        posterior_quantiles=np.full((n, 6, 3), np.nan),
        label_source_code=np.zeros((n, 6), dtype="u1"),
        raw_alpha_feh=np.full(n, np.nan),
    )
    dictionary = [""]
    source_counts = {}
    for block in blocks:
        pos = np.searchsorted(ids, block["ids"])
        match = pos < len(ids)
        match[match] &= ids[pos[match]] == block["ids"][match]
        selected = np.flatnonzero(match)
        dest = pos[match]
        for source in np.unique(block["sources"]):
            if source not in dictionary:
                dictionary.append(str(source))
        codes = np.array(
            [dictionary.index(str(v)) for v in block["sources"]], dtype="u1"
        )
        for k, column in enumerate(block["columns"]):
            good = block["valid"][selected, k] & ~result["observed"][dest, column]
            rr = dest[good]
            ss = selected[good]
            result["value"][rr, column] = block["value"][ss, k]
            result["error"][rr, column] = block["error"][ss, k]
            result["observed"][rr, column] = True
            result["quality"][rr, column] = block["primary"][ss, k]
            result["diagnostic_eligible"][rr, column] = True
            result["error_observed"][rr, column] = np.isfinite(
                block["error"][ss, k]
            ) & (block["error"][ss, k] > 0)
            result["label_source_code"][rr, column] = codes[ss]
            if "bounds" in block:
                result["posterior_quantiles"][rr, column] = block["bounds"][ss, k]
        if "feh" in block:
            result["raw_alpha_feh"][dest] = block["feh"][selected]
        source_counts[block["name"]] = {
            "source_unique_rows": len(block["ids"]),
            "source_valid_per_target": block["valid"].sum(0).tolist(),
            "source_primary_per_target": block["primary"].sum(0).tolist(),
            "rows_after_id_exclusions": len(dest),
        }
        if "raw_eligible_measurement_rows" in block:
            source_counts[block["name"]]["raw_eligible_measurement_rows"] = block[
                "raw_eligible_measurement_rows"
            ]
    original_rows = np.searchsorted(ids, original_ids)
    if not np.array_equal(ids[original_rows], original_ids):
        raise ValueError("An original TEST identity was lost")
    for key in ("value", "error", "observed", "quality", "error_observed"):
        default = np.nan if key in ("value", "error") else False
        result["original_reference_" + key] = np.full(
            (n, 6), default, dtype=base[key].dtype
        )
        result["original_reference_" + key][original_rows] = base[key][old]
    original_sources = np.full((n, 6), 0, dtype="u1")
    for source in np.unique(base["label_source"][old].astype(str)):
        if source not in dictionary:
            dictionary.append(str(source))
    original_sources[original_rows] = np.array(
        [
            [dictionary.index(str(v)) for v in row]
            for row in base["label_source"][old].astype(str)
        ],
        dtype="u1",
    )
    result["original_reference_label_source_code"] = original_sources
    old_valid = base["observed"][old].astype(bool) & base["quality"][old].astype(bool)
    for column in range(6):
        rr = original_rows[old_valid[:, column]]
        ss = old[old_valid[:, column]]
        result["value"][rr, column] = base["value"][ss, column]
        result["error"][rr, column] = base["error"][ss, column]
        result["error_observed"][rr, column] = base["error_observed"][ss, column]
        result["observed"][rr, column] = True
        result["quality"][rr, column] = True
        result["diagnostic_eligible"][rr, column] = True
        result["label_source_code"][rr, column] = original_sources[rr, column]
    result["original_test_member"] = np.isin(ids, original_ids)
    result["ancestor_feh_supervised_fit_exposed"] = np.isin(ids, ancestor_ids)
    result["strict_unexposed_star"] = ~result["ancestor_feh_supervised_fit_exposed"]
    # The old benchmark is preserved separately; known exposed original stars
    # receive no new primary/reference backfills, and never count as strict stars.
    exposed = result["ancestor_feh_supervised_fit_exposed"]
    result["observed"][exposed] = result["original_reference_observed"][exposed]
    result["quality"][exposed] = False
    result["diagnostic_eligible"][exposed] = False
    # DESI alpha below-2 remains a diagnostic measured reference, not primary.
    has_desi = np.isfinite(result["raw_alpha_feh"])
    result["quality"][has_desi, 3] &= result["raw_alpha_feh"][has_desi] > -2
    result["primary_score_eligible"] = result["quality"] & result["observed"]
    result["expanded_reference"] = result["diagnostic_eligible"] & ~(
        result["original_reference_observed"] & result["original_reference_quality"]
    )
    ext_pos = np.searchsorted(ids, external["ids"])
    m = ext_pos < len(ids)
    m[m] &= ids[ext_pos[m]] == external["ids"][m]
    result["external_age_value"] = np.full(n, np.nan)
    result["external_age_bounds"] = np.full((n, 3), np.nan)
    result["external_age_observed"] = np.zeros(n, bool)
    result["external_age_value"][ext_pos[m]] = external["value"][m]
    result["external_age_bounds"][ext_pos[m]] = external["bounds"][m]
    result["external_age_observed"][ext_pos[m]] = True
    result["external_age_observed"][exposed] = False
    counts.update(
        source_counts=source_counts,
        remaining_union_ids=n,
        external_apokasc_valid_source_rows=len(external["ids"]),
        external_apokasc_after_exclusions=int(result["external_age_observed"].sum()),
    )
    return result, dictionary, old, original_rows, counts


def _write_rows(group, name, array, indices, chunk_rows=32768):
    shape = (len(indices), *array.shape[1:])
    dtype = (
        np.dtype(f"S{array.dtype.itemsize // 4}")
        if array.dtype.kind == "U"
        else array.dtype
    )
    if not len(indices):
        group.create_dataset(name, shape=shape, dtype=dtype)
        return
    node = group.create_dataset(
        name,
        shape=shape,
        dtype=dtype,
        chunks=(min(chunk_rows, max(1, len(indices))), *array.shape[1:]),
        compression="gzip",
        compression_opts=1,
    )
    for start in range(0, len(indices), chunk_rows):
        node[start : start + chunk_rows] = array[
            indices[start : start + chunk_rows]
        ].astype(dtype, copy=False)


def corner_selection(labels, features, keep, feature_names, fit_g):
    """Model-blind common stars: FeH bin, FIT G split and genuine XP modality."""
    keep = np.asarray(keep, dtype="i8")
    g = features["raw"][keep, list(feature_names).index("G")]
    finite_fit = np.asarray(fit_g)[np.isfinite(fit_g) & (np.asarray(fit_g) != -9999)]
    split = float(np.median(finite_fit)) if len(finite_fit) else 15.0
    feh = labels["value"][keep, 2]
    eligible = labels["primary_score_eligible"][keep, 2].astype(bool)
    eligible &= labels["strict_unexposed_star"][keep].astype(bool) & np.isfinite(g)
    eligible &= g != -9999
    chosen, cells = [], []
    for lo, hi in [(-np.inf, -3.0), (-3.0, -2.5), (-2.5, -2.0)]:
        for bright in (True, False):
            for modality in ("real_xp_present", "real_no_xp"):
                mask = eligible & (feh >= lo) & (feh < hi)
                mask &= (g < split) == bright
                mask &= features[modality][keep].astype(bool)
                rows = np.flatnonzero(mask)
                # Ordering uses immutable identity only, never predictions/residuals.
                if len(rows):
                    row = rows[np.argmin(labels["source_id"][keep[rows]])]
                    chosen.append(int(row))
                cells.append(
                    dict(
                        feh_lower=None if not np.isfinite(lo) else lo,
                        feh_upper=hi,
                        bright=bright,
                        modality=modality,
                        eligible=int(len(rows)),
                    )
                )
    rows = np.array(chosen, dtype="i8")
    axes = np.array(
        [
            [3000, 8500],
            [-0.5, 5.5],
            [-5, 0.5],
            [-1, 1.2],
            [np.log(0.05), np.log(18)],
            [np.log(0.01), np.log(100)],
        ]
    )
    return (
        rows,
        np.broadcast_to(axes, (len(rows), 6, 2)).copy(),
        dict(
            cells=cells,
            fit_g_median=split,
            maximum=12,
            model_output_used=False,
            axes_scope="Common fixed display axes; age/distance in natural log, not physical support limits",
        ),
    )


def run(config, output):
    if os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1":
        raise RuntimeError("Run source preparation on CANFAR headless CPU only")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Never overwrite an attempted source preparation")
    output.mkdir(parents=True)
    manifest = dict(
        status="preparing",
        run_id=config["run_id"],
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_provenance=config,
        no_model_construction=True,
        training_performed=False,
    )

    def save():
        tmp = output / "manifest.tmp"
        tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        tmp.replace(output / "manifest.json")

    save()
    try:
        with np.load(checked(config["frozen"]), allow_pickle=False) as d:
            base = {k: d[k] for k in d.files}
        if tuple(base["target_names"]) != NAMES:
            raise ValueError("Target order changed")
        with np.load(checked(config["feh_ancestry"]), allow_pickle=False) as d:
            ancestor = exact_ids(d["historical_fit_source_id"])[
                d["historical_fit_sample_count"] > 0
            ]
        blocks = [native_block(config), alpha_block(config), *posterior_blocks(config)]
        external = apokasc_panel(config)
        labels, dictionary, old, old_rows, census = assemble_labels(
            blocks, base, ancestor, external
        )
        del blocks
        manifest.update(status="materializing_features", source_census=census)
        save()
        look = config["feature_lookup"]
        photometry, photometry_proof = desi_photometry_fallback(
            config, labels["source_id"]
        )
        with np.load(checked(config["andrae"]), allow_pickle=False) as archive:
            labels.update(
                join_catalogue(
                    labels["source_id"], {k: archive[k] for k in archive.files}
                )
            )
        labels["andrae_projection_queried"] = labels["andrae_matched"].copy()
        features, proof = lookup_evaluation_features(
            labels["source_id"],
            base["feature_names"].tolist(),
            base["scaler_center"],
            base["scaler_scale"],
            legacy_path=look["legacy_path"],
            expected_legacy_sha256=look["legacy_sha256"],
            cache_path=look["cache_path"],
            original_path=look["original_path"],
            gaia_vac=look["gaia_vac"],
            chunk_rows=look.get("chunk_rows", 65536),
            output_directory=output / "feature-cache",
            photometry_fallback=photometry,
            allow_unknown_photometry=True,
        )
        # Same-object previous predictions may be reused only for byte-identical
        # model inputs. Legacy lookup alternatives never replace those inputs.
        features["input"][old_rows] = base["input"][old]
        features["visible"][old_rows] = base["visible"][old]
        if (
            features["input"][old_rows].dtype != base["input"][old].dtype
            or features["input"][old_rows].tobytes() != base["input"][old].tobytes()
            or features["visible"][old_rows].tobytes() != base["visible"][old].tobytes()
        ):
            raise ValueError("Original TEST model inputs changed")
        for key in (
            "raw",
            "feature_errors",
            "real_xp_present",
            "real_no_xp",
            "xp_status",
            "official_has_xp_known",
            "official_has_xp_continuous",
            "input_has_xp",
        ):
            if key in features and key in base:
                features[key][old_rows] = base[key][old]
        features["feature_recoverable"][old_rows] = True
        diagnostic_input = np.isin(
            features["xp_status"],
            ["photometry_only_publication_unknown", "xp_published_not_staged"],
        ) | features.get(
            "photometry_is_domain_transfer", np.zeros(len(labels["source_id"]), bool)
        )
        diagnostic_input[old_rows] = False
        labels["input_diagnostic_only"] = diagnostic_input
        labels["primary_score_eligible"][diagnostic_input] = False
        keep = np.flatnonzero(features["feature_recoverable"])
        bad = np.flatnonzero(~features["feature_recoverable"])
        if len(keep) < len(old):
            raise ValueError("Original benchmark row count lost")
        manifest.update(status="writing_prepared", feature_eligible_rows=len(keep))
        save()
        path = output / "prepared.h5"
        staged = output / "prepared.h5.building"
        with h5py.File(staged, "x") as h:
            h.attrs.update(
                schema="expanded-sparse-six-target-evaluation-v1",
                source_commit=manifest["source_commit"],
                original_benchmark_input_sha256=config["frozen"]["sha256"],
                label_source_dictionary_json=json.dumps(dictionary),
                physical_truth_claim=False,
            )
            for key, value in labels.items():
                if key.endswith("_code"):
                    continue
                _write_rows(h, key, value, keep)
            for name, codes in [
                ("label_source", labels["label_source_code"]),
                (
                    "original_reference_label_source",
                    labels["original_reference_label_source_code"],
                ),
            ]:
                ds = h.create_dataset(
                    name,
                    shape=(len(keep), 6),
                    dtype="S64",
                    chunks=(min(32768, len(keep)), 6),
                    compression="gzip",
                    compression_opts=1,
                )
                lookup = np.asarray(dictionary, dtype="S64")
                for start in range(0, len(keep), 32768):
                    ds[start : start + 32768] = lookup[
                        codes[keep[start : start + 32768]]
                    ]
            for key, value in features.items():
                if (
                    not isinstance(value, np.ndarray)
                    or value.ndim == 0
                    or value.shape[0] != len(labels["source_id"])
                    or key in h
                ):
                    continue
                _write_rows(h, key, value, keep)
            for key in ("feature_names", "target_names", "target_units"):
                if key in base:
                    h.create_dataset(key, data=np.asarray(base[key]).astype("S64"))
            corner_rows, corner_limits, corner_proof = corner_selection(
                labels,
                features,
                keep,
                base["feature_names"].tolist(),
                base["raw"][
                    base["role"] == "fit", base["feature_names"].tolist().index("G")
                ],
            )
            h.create_dataset("corner_rows", data=corner_rows)
            h.create_dataset("corner_plot_limits", data=corner_limits)
            manifest["corner_selection"] = corner_proof
            for key in ("scaler_center", "scaler_scale"):
                h.create_dataset(key, data=base[key])
            h.create_dataset(
                "role", data=np.full(len(keep), b"test", dtype="S5"), compression="gzip"
            )
            h.create_dataset(
                "input_eligible", data=np.ones(len(keep), bool), compression="gzip"
            )
            h.create_dataset(
                "external_age_source",
                data=np.where(
                    labels["external_age_observed"][keep],
                    b"APOKASC3_seismic_model",
                    b"",
                ).astype("S32"),
                compression="gzip",
            )
            h.create_dataset(
                "unrecoverable/source_id",
                data=labels["source_id"][bad],
                compression="gzip",
            )
            h.create_dataset(
                "unrecoverable/reason",
                data=np.full(len(bad), b"no_recoverable_verified_input", dtype="S40"),
                compression="gzip",
            )
            h.attrs["complete"] = True
        staged.replace(path)
        primary = labels["primary_score_eligible"][keep]
        diagnostic = labels["diagnostic_eligible"][keep]
        manifest.update(
            status="full_label_evaluation_source_complete",
            model_ready=True,
            prepared_path=str(path),
            input_sha256=sha(path),
            feature_eligible_rows=len(keep),
            source_candidate_rows=len(labels["source_id"]),
            unrecoverable_rows=len(bad),
            original_benchmark_inputs_preserved=True,
            original_benchmark_input_sha256=config["frozen"]["sha256"],
            original_benchmark_rows=len(old),
            primary_reference_counts=primary.sum(0).tolist(),
            diagnostic_reference_counts=diagnostic.sum(0).tolist(),
            external_age_reference_count=int(
                labels["external_age_observed"][keep].sum()
            ),
            modality_counts={
                str(v): int((features["xp_status"][keep] == v).sum())
                for v in np.unique(features["xp_status"][keep])
            },
            feature_lookup_proof=proof,
            desi_photometry_proof=photometry_proof,
            andrae_projection_scope="Archived partial paired projection only; absent IDs are unqueried or unmatched, not proof of absence from released Andrae catalogue",
            completed_at=dt.datetime.now(dt.UTC).isoformat(),
            scope="Uncapped source-qualified reference union; known FIT/TUNE/CAL and large-FeH ancestry exclusions; parent86 all-history proof not claimed; posterior ages/distances and source-scale abundances are not physical truth",
            pending_sources=[
                "Willett Gaia identity recovery beyond already verified Kepler crossmatch",
                "DESI native3 reference corrections; native3 primary remains independently verified sidecar",
            ],
        )
        save()
        print(json.dumps(manifest))
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        save()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(yaml.safe_load(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
