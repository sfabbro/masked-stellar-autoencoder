"""Headless source-only broad TEST preparation; never model inference or fitting.

Preserve v3 FIT/TUNE/CAL byte exactly. Genuine no-XP requires a staged Gaia DR3
VAC false publication flag, zero raw XP coefficients and actual photometry.
Missing inputs alone never establish physical XP absence. No acquisitions.
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
import yaml
from astropy.io import fits

NAMES = ("teff", "logg", "fe_h", "alpha", "age", "distance")
ROW_KEYS = (
    "source_id",
    "role",
    "value",
    "observed",
    "quality",
    "error",
    "error_observed",
    "label_source",
    "posterior_quantiles",
    "label_source_role",
    "original_role_eligible",
    "reference_holdout",
    "input",
    "visible",
    "raw",
    "feature_errors",
    "benchmark_partition",
    "feature_row",
)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_npz(spec):
    if sha(spec["path"]) != spec["sha256"]:
        raise ValueError(f"Pinned input differs: {spec['path']}")
    with np.load(spec["path"], allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def publication_flag(values):
    raw = np.asanyarray(values)
    known = ~np.ma.getmaskarray(raw)
    data = np.asarray(np.ma.getdata(raw))
    if data.dtype.kind == "b":
        return data, known
    if data.dtype.kind in "iu":
        return data == 1, known & np.isin(data, [0, 1])
    text = np.char.lower(np.char.strip(data.astype("U16")))
    return np.isin(text, ["t", "true", "1"]), known & np.isin(
        text, ["t", "f", "true", "false", "0", "1"]
    )


def xp_evidence(raw, names, published, known):
    xp = [
        j
        for j, n in enumerate(names)
        if n in {f"{arm}_{i}" for arm in ("bp", "rp") for i in range(1, 56)}
    ]
    if len(xp) != 110:
        raise ValueError("Require all110 original XP columns")
    present = np.isfinite(raw[:, xp]) & (raw[:, xp] != -9999)
    complete, missing = present.all(1), ~present.any(1)
    no_xp = missing & known & ~published
    status = np.full(len(raw), "unknown", dtype="U32")
    status[present.any(1) & ~complete] = "partial_xp"
    status[missing & known & published] = "xp_published_not_staged"
    status[complete] = "complete_xp"
    status[no_xp] = "official_no_xp"
    return complete, no_xp, status


def unchanged_training(base, expanded):
    rows = np.flatnonzero(np.isin(base["role"], ["fit", "tune", "cal"]))
    proof = {}
    for key in ROW_KEYS:
        a, b = base[key][rows], expanded[key][rows]
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise ValueError(f"FIT/TUNE/CAL changed: {key}")
        proof[key] = hashlib.sha256(a.tobytes()).hexdigest()
    return proof


def join_catalogue(ids, source):
    cids = source["source_id"]
    if len(np.unique(cids)) != len(cids):
        raise ValueError("Duplicate Andrae IDs")
    order = np.argsort(cids)
    pos = np.searchsorted(cids[order], ids)
    match = pos < len(cids)
    match[match] &= cids[order[pos[match]]] == ids[match]
    predictions = np.full((len(ids), 3), np.nan)
    predictions[match] = source["other"][order[pos[match]]]
    known = match & bool(source["heldout_available"])
    training = np.zeros(len(ids), bool)
    training[known] = ~source["heldout"][order[pos[known]]].astype(bool)
    return dict(
        andrae_matched=match,
        andrae_prediction_native=predictions,
        andrae_training_membership_known=known,
        andrae_in_training=training,
    )


def ancestry_proof(data, actual_fit_ids):
    exposed = np.isin(data["source_id"], actual_fit_ids)
    eligible = data["quality"] & data["observed"]
    for role in ("tune", "cal", "test"):
        if np.any(exposed & (data["role"] == role) & eligible[:, 2]):
            raise ValueError(
                "Direct FeH supervised ancestry overlaps held-out FeH labels"
            )
    counts = {
        role: (eligible[(data["role"] == role) & exposed]).sum(0).tolist()
        for role in ("fit", "tune", "cal", "test")
    }
    return exposed, counts


def cache_membership(ids, directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("status") != "complete"
        or not manifest.get("sample_source_ids_unique")
        or not manifest.get("original_validation_ids_excluded")
    ):
        raise ValueError("Original150M sample proof incomplete")
    path = directory / "source_id.npy"
    source = np.load(path, mmap_mode="r", allow_pickle=False)
    if (
        source.shape != (150_000_000,)
        or source.dtype != np.dtype("<i8")
        or not source.flags.c_contiguous
    ):
        raise ValueError("Original150M ID schema changed")
    spec = manifest["files"]["source_id.npy"]
    if path.stat().st_size != spec["bytes"]:
        raise ValueError("Original150M ID byte count changed")
    order = np.argsort(ids)
    sorted_ids = ids[order]
    found = np.zeros(len(ids), bool)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        digest.update(stream.read(source.offset))
    for start in range(0, len(source), 1_000_000):
        block = np.asarray(source[start : start + 1_000_000])
        digest.update(block.tobytes())
        positions = np.searchsorted(sorted_ids, block)
        valid = positions < len(ids)
        valid[valid] &= sorted_ids[positions[valid]] == block[valid]
        found[order[positions[valid]]] = True
    if digest.hexdigest() != spec["sha256"]:
        raise ValueError("Original150M ID content changed")
    return found, dict(
        manifest_sha256=sha(directory / "manifest.json"),
        source_id_sha256=digest.hexdigest(),
        feature_cache_read=False,
    )


def vac_projection(spec, wanted_ids, raw, errors, names, evidence=None):
    path = Path(spec["path"])
    if path.stat().st_size != spec["bytes"]:
        raise ValueError("Staged Gaia VAC size differs")
    first_sha = sha(path)
    published, known = np.zeros(len(raw), bool), np.zeros(len(raw), bool)
    source_rows = np.full(len(raw), -1, dtype="i8")
    with fits.open(path, memmap=True) as hdus:
        table = hdus[1].data
        if len(table) != spec["rows"]:
            raise ValueError("Staged Gaia VAC row count differs")
        ids = np.asarray(table["source_id"], dtype="i8")
        valid = np.flatnonzero(ids > 0)
        unique, first, inverse = np.unique(
            ids[valid], return_index=True, return_inverse=True
        )
        first = valid[first]
        pos = np.searchsorted(unique, wanted_ids)
        match = pos < len(unique)
        match[match] &= unique[pos[match]] == wanted_ids[match]
        source_rows[match] = first[pos[match]]
        all_flag, all_known = publication_flag(table["has_xp_continuous"])
        true_count = np.bincount(
            inverse,
            weights=(all_flag[valid] & all_known[valid]).astype("i8"),
            minlength=len(unique),
        )
        false_count = np.bincount(
            inverse,
            weights=(~all_flag[valid] & all_known[valid]).astype("i8"),
            minlength=len(unique),
        )
        unknown_count = np.bincount(
            inverse, weights=(~all_known[valid]).astype("i8"), minlength=len(unique)
        )
        group_known = (unknown_count == 0) & ((true_count == 0) | (false_count == 0))
        requested_group = np.isin(unique, wanted_ids)
        ambiguous_photometry = np.zeros(len(unique), bool)
        duplicate_count = np.bincount(inverse, minlength=len(unique))
        photometry_fields = [
            "phot_g_mean_mag",
            "phot_bp_mean_mag",
            "phot_rp_mean_mag",
            "W1mag",
            "W2mag",
        ]
        photometry_fields += [
            prefix + suffix
            for prefix in ("phot_g", "phot_bp", "phot_rp")
            for suffix in ("_mean_flux", "_mean_flux_error")
        ]
        for field in photometry_fields:
            values = np.asarray(table[field], dtype="f8").copy()
            values[~np.isfinite(values) | (values == -9999)] = np.nan
            if field in (
                "phot_g_mean_mag",
                "phot_bp_mean_mag",
                "phot_rp_mean_mag",
                "W1mag",
                "W2mag",
            ):
                values[(values <= -20) | (values >= 40)] = np.nan
            different = ~np.isclose(
                values[valid],
                values[first][inverse],
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            )
            np.logical_or.at(
                ambiguous_photometry, inverse, different & requested_group[inverse]
            )
        group_known &= ~ambiguous_photometry
        published[match] = true_count[pos[match]] > 0
        known[match] = group_known[pos[match]]
        if evidence is not None:
            evidence.update(
                duplicate_positive_source_ids=int((duplicate_count > 1).sum()),
                requested_ambiguous_photometry_ids=int(
                    (ambiguous_photometry & requested_group).sum()
                ),
                requested_unknown_or_conflicting_flag_ids=int(
                    (~group_known & requested_group).sum()
                ),
                canonical_photometry_row="first original positive GaiaID row, only after repeated-ID flag/flux/magnitude consistency",
                magnitude_valid_range=[-20, 40],
                gaia_error_conversion="2.5/ln10 * positive_flux_error/positive_flux; magnitude-error columns with inconsistent units not used",
            )
        _, no_xp, _ = xp_evidence(raw, names, published, known)
        dest = np.flatnonzero(no_xp)
        source = source_rows[dest]
        mapping = {
            "G": "phot_g_mean_mag",
            "BP": "phot_bp_mean_mag",
            "RP": "phot_rp_mean_mag",
            "W1": "W1mag",
            "W2": "W2mag",
        }
        for name, column in mapping.items():
            if name not in names:
                continue
            j = names.index(name)
            value = np.asarray(table[column][source], dtype="f8")
            value[~np.isfinite(value) | (value <= -20) | (value >= 40)] = np.nan
            raw[dest, j] = value
            if name in ("W1", "W2"):
                uncertainty = np.asarray(table["e_" + column][source], dtype="f8")
            else:
                prefix = {"G": "phot_g", "BP": "phot_bp", "RP": "phot_rp"}[name]
                flux = np.asarray(table[prefix + "_mean_flux"][source], dtype="f8")
                error = np.asarray(
                    table[prefix + "_mean_flux_error"][source], dtype="f8"
                )
                uncertainty = np.full(len(source), np.nan)
                usable = (
                    np.isfinite(flux) & np.isfinite(error) & (flux > 0) & (error > 0)
                )
                uncertainty[usable] = (2.5 / np.log(10)) * error[usable] / flux[usable]
            uncertainty[
                ~np.isfinite(uncertainty) | (uncertainty <= 0) | (uncertainty > 5)
            ] = np.nan
            errors[dest, j] = uncertainty
    return published, known, source_rows, first_sha


def fixed_corners(data):
    quality = data["quality"] & data["observed"]
    test = data["role"] == "test"
    tail = quality[:, 2] & (data["value"][:, 2] < -2)
    groups = dict(
        xp_metalpoor=test & tail & data["real_xp_present"],
        no_xp_metalpoor=test & tail & data["real_no_xp"],
        no_xp_global=test & ~tail & data["real_no_xp"],
        complete_six=test & quality.all(1),
    )
    selected, cells = [], []
    for name, mask in groups.items():
        candidates = [int(i) for i in np.flatnonzero(mask) if int(i) not in selected]
        candidates.sort(
            key=lambda i: hashlib.sha256(
                f"five-job-corners:{name}:{int(data['source_id'][i])}".encode()
            ).digest()
        )
        take = candidates[:2]
        selected.extend(take)
        cells.append(
            dict(
                group=name,
                eligible_count=len(candidates),
                source_id=data["source_id"][take].tolist(),
                empty=not candidates,
            )
        )
    selected = np.asarray(selected, dtype="i8")
    coordinate = data["value"].copy()
    coordinate[:, 4:] = np.log(
        np.where(coordinate[:, 4:] > 0, coordinate[:, 4:], np.nan)
    )
    limits = np.empty((len(selected), 6, 2))
    for j in range(6):
        values = coordinate[(data["role"] == "fit") & quality[:, j], j]
        lo, median, hi = np.quantile(values, [0.05, 0.5, 0.95])
        half = max((hi - lo) / 4, 1e-5)
        for slot, row in enumerate(selected):
            center = coordinate[row, j] if quality[row, j] else median
            limits[slot, j] = [center - half, center + half]
    return selected, limits, cells


def run(config, output):
    if os.environ.get("MSA_EXPERIMENT_CPU_ONLY") != "1":
        raise RuntimeError("Source preparation must execute on CANFAR CPU headless")
    if output.exists():
        raise FileExistsError(output)
    base = checked_npz(config["base"])
    if tuple(base["target_names"]) != NAMES:
        raise ValueError("Original target order differs")
    for key in ("features", "native_sidecar"):
        if sha(config[key]["path"]) != config[key]["sha256"]:
            raise ValueError(f"Pinned {key} differs")
    qa = json.loads(Path(config["native_source_qa"]).read_text())
    required_qa = (
        "native_labels_verified",
        "all_original_source_target_values_errors_flags_rows_verified",
        "all_source_id_roles_and_reference_bans_verified",
        "all_target_quality_masks_verified",
    )
    if (
        qa.get("status") != "passed"
        or qa.get("sidecar_sha256") != config["native_sidecar"]["sha256"]
        or qa.get("cohort_sha256") != config["features"]["sha256"]
        or not all(qa.get(name) is True for name in required_qa)
    ):
        raise ValueError("Native source/quality/roles proof incomplete or changed")
    with (
        h5py.File(config["features"]["path"]) as features,
        h5py.File(config["native_sidecar"]["path"]) as side,
    ):
        ids = features["features/source_id"][:]
        np.testing.assert_array_equal(ids, side["source_id"][:])
        names = json.loads(features.attrs["feature_columns_json"])
        if names != list(base["feature_names"]):
            raise ValueError("Feature order changed")
        values = side["targets/value"][:]
        errors = side["targets/uncertainty"][:]
        mask = side["targets/observed"][:].astype(bool) & side["targets/quality_pass"][
            :
        ].astype(bool)
        mask &= np.isfinite(values) & np.isfinite(errors) & (errors > 0)
        candidate = (
            (side["roles/identity_split"][:] == 2)
            & mask.any(1)
            & ~np.isin(ids, base["source_id"])
            & ~side["roles/reference_holdout"][:].astype(bool)
        )
        rows = np.flatnonzero(candidate)
        raw = features["features/values"][rows]
        feature_error = features["features/reported_feature_errors"][rows]
        partition = features["features/benchmark_partition"][:][rows]
        survey = side["targets/survey"][:][rows].astype("U64")
    vac_evidence = {}
    published, known, vac_rows, vac_sha = vac_projection(
        config["gaia_vac"], ids[rows], raw, feature_error, names, evidence=vac_evidence
    )
    complete, no_xp, status = xp_evidence(raw, names, published, known)
    photo = [
        names.index(name) for name in ("W1", "W2", "G", "BP", "RP") if name in names
    ]
    photo_present = np.isfinite(raw[:, photo]) & (raw[:, photo] != -9999)
    accept = complete | (
        no_xp
        & (photo_present.sum(1) >= 2)
        & np.isfinite(raw[:, names.index("G")])
        & (raw[:, names.index("G")] != -9999)
    )
    kept = rows[accept]
    raw, feature_error = raw[accept], feature_error[accept]
    n, old_n = len(kept), len(base["source_id"])
    extra = {}
    for key in ROW_KEYS:
        template = np.asarray(base[key])
        shape = (n, *template.shape[1:])
        fill = (
            np.nan
            if template.dtype.kind in "fc"
            else ""
            if template.dtype.kind in "US"
            else 0
        )
        extra[key] = np.full(shape, fill, dtype=template.dtype)
    extra["source_id"] = ids[kept]
    extra["role"][:] = "test"
    for key, source in (
        ("value", values),
        ("error", errors),
        ("observed", mask),
        ("quality", mask),
        ("original_role_eligible", mask),
        ("error_observed", mask),
    ):
        extra[key][:, :3] = source[kept]
    extra["label_source"][:, :3] = survey[accept, None]
    extra["label_source_role"][:, :3] = "test"
    extra["feature_row"] = kept
    extra["benchmark_partition"] = partition[accept]
    extra["raw"], extra["feature_errors"] = raw, feature_error
    allowed = np.array(
        [
            name in ("W1", "W2", "G", "BP", "RP")
            or name in {f"{arm}_{i}" for arm in ("bp", "rp") for i in range(1, 56)}
            for name in names
        ]
    )
    scaled = (raw - base["scaler_center"]) / base["scaler_scale"]
    visible = allowed[None] & np.isfinite(raw) & (raw != -9999) & np.isfinite(scaled)
    extra["input"] = np.where(visible, scaled, -9999).astype(base["input"].dtype)
    extra["visible"] = visible
    data = {
        key: np.concatenate([value, extra[key]]) if key in ROW_KEYS else value
        for key, value in base.items()
    }
    protected = unchanged_training(base, data)
    original_complete, _, _ = xp_evidence(
        base["raw"], names, np.zeros(old_n, bool), np.zeros(old_n, bool)
    )
    if not original_complete.all():
        raise ValueError("Original v3 modality was not actual complete XP")
    if len(np.unique(data["source_id"])) != len(data["source_id"]):
        raise ValueError("Expanded source IDs duplicate")
    data.update(
        input_has_xp=np.r_[np.ones(old_n, bool), complete[accept]],
        real_xp_present=np.r_[np.ones(old_n, bool), complete[accept]],
        real_no_xp=np.r_[np.zeros(old_n, bool), no_xp[accept]],
        xp_status=np.r_[np.full(old_n, "complete_xp", dtype="U32"), status[accept]],
        official_has_xp_known=np.r_[np.zeros(old_n, bool), known[accept]],
        official_has_xp_continuous=np.r_[np.zeros(old_n, bool), published[accept]],
        gaia_vac_source_row=np.r_[np.full(old_n, -1, dtype="i8"), vac_rows[accept]],
        test_extended=np.r_[np.zeros(old_n, bool), np.ones(n, bool)],
    )
    data.update(join_catalogue(data["source_id"], checked_npz(config["andrae"])))
    ancestry = checked_npz(config["feh_ancestry"])
    actual_fit_ids = ancestry["historical_fit_source_id"][
        ancestry["historical_fit_sample_count"] > 0
    ]
    data["ancestor_feh_supervised_fit_exposed"], ancestry_counts = ancestry_proof(
        data, actual_fit_ids
    )
    data["ancestor_150m_reconstruction_exposed"], sample_proof = cache_membership(
        data["source_id"], config["reconstruction_cache"]
    )
    corner_rows, limits, cells = fixed_corners(data)
    data.update(
        corner_rows=corner_rows,
        corner_source_id=data["source_id"][corner_rows],
        corner_plot_limits=limits,
    )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "prepared.npz", **data)
    report = dict(
        status="five_job_source_preparation_complete",
        run_id=config["run_id"],
        input_sha256=sha(output / "prepared.npz"),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_provenance=config,
        completed_at=dt.datetime.now(dt.UTC).isoformat(),
        gaia_vac_first_mounted_sha256=vac_sha,
        gaia_vac_publisher_sha_known=False,
        gaia_vac_identity_and_photometry_evidence=vac_evidence,
        protected_fit_tune_cal_field_sha256=protected,
        original_roles_byte_identical=True,
        role_counts={
            role: int((data["role"] == role).sum())
            for role in ("fit", "tune", "cal", "test")
        },
        observed_per_role_target={
            role: (
                data["quality"][data["role"] == role]
                & data["observed"][data["role"] == role]
            )
            .sum(0)
            .tolist()
            for role in ("fit", "tune", "cal", "test")
        },
        added_test_count=n,
        modality_counts={
            name: int((data["xp_status"] == name).sum())
            for name in ("complete_xp", "official_no_xp")
        },
        genuine_no_xp_observed_target_counts=data["quality"][data["real_no_xp"]]
        .sum(0)
        .tolist(),
        rejected_candidate_status_counts={
            str(name): int((status[~accept] == name).sum())
            for name in np.unique(status[~accept])
        },
        test_andrae_overlap=int(
            (data["andrae_matched"] & (data["role"] == "test")).sum()
        ),
        test_andrae_known_not_training=int(
            (
                data["andrae_matched"]
                & data["andrae_training_membership_known"]
                & ~data["andrae_in_training"]
                & (data["role"] == "test")
            ).sum()
        ),
        sample_id_membership_proof=sample_proof,
        feh_aux_ancestry_exposed_observed_counts=ancestry_counts,
        feh_direct_holdout_overlap=False,
        ancestry_scope="Exact completed4.5B run FeH auxiliary stream and actual150M reconstruction IDs only; earlier parent86 ancestry and all other training history not certified",
        genuine_no_xp_minimum_per_target=32,
        genuine_no_xp_sufficient_per_target=(
            data["quality"][data["real_no_xp"]].sum(0) >= 32
        ).tolist(),
        corner_selection=cells,
        corner_limit_policy="FIT-only robust coordinate span about eligible reference or FIT median; all4models same fixed axes; preserve samples and report outside fractions",
        no_native_model_execution=True,
        no_acquisition=True,
        target_caveat="AndraeM/Hproxy notnativeFeH; StarHorseage/distanceposteriors notphysicaltruth; no newalpha/age/distlabels invented",
        no_xp_caveat="Official false Gaia XP flag plus no actual coefficients and realphotometry; model trained onXP-richinputs, so noXP evaluation is modality-shifted",
        physical_truth_claim=False,
    )
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(yaml.safe_load(args.config.read_text()), args.output)


if __name__ == "__main__":
    main()
