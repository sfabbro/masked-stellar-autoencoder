"""Project already acquired original native rows into an audited joint sidecar.

This command does no network access, downloads, fitting or CANFAR submission.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

from masked_stellar_autoencoder.training.joint_native_labels import (
    SOURCE_COLUMNS,
    TARGET_NAMES,
    TARGET_UNITS,
    assign_roles,
    project_catalogue,
)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def materialize(
    cohort,
    sources,
    source_sha,
    reference_ids,
    output,
    *,
    cohort_sha256="321f27234bc306f64ef1a942fef981663f3c50df84990359b9943a1979262120",
    reference_sha256="5336a296bc6e540596fe797dbc5371350c976901f80beb6cd99b557f9c4ef25c",
):
    """Write once, validating original file hashes and all native associations."""
    output = Path(output)
    if output.exists() or output.with_suffix(".building").exists():
        raise FileExistsError(
            "Refusing to overwrite an existing sidecar or incomplete predecessor"
        )
    if sha(cohort) != cohort_sha256:
        raise ValueError("Feature cohort SHA256 differs from the audited source pin")
    for name in SOURCE_COLUMNS:
        if name not in sources or name not in source_sha:
            raise ValueError(
                "Explicit original source paths and SHA256 pins are required"
            )
        if sha(sources[name]) != source_sha[name]:
            raise ValueError(f"Original {name} SHA256 differs from source pin")
    reference_ids = np.asarray(reference_ids)
    if (
        reference_ids.ndim != 1
        or not np.issubdtype(reference_ids.dtype, np.integer)
        or np.any(reference_ids <= 0)
        or len(np.unique(reference_ids)) != len(reference_ids)
    ):
        raise ValueError("Reference union must contain distinct positive integer IDs")
    if len(reference_ids) != 99574:
        raise ValueError("Expected complete declared Hipparcos+seismic reference union")
    reference_digest = hashlib.sha256(
        np.asarray(reference_ids, dtype="<i8").tobytes()
    ).hexdigest()
    if reference_digest != reference_sha256:
        raise ValueError("Reference-ID union differs from the audited holdout pin")
    with h5py.File(cohort) as f:
        if not bool(f.attrs.get("complete", False)):
            raise ValueError("Input feature cohort is incomplete")
        ids = f["features/source_id"][:]
        partition = f["features/benchmark_partition"][:]
        available = f["features/feature_available"][:]
        native = f["raw_layers/native/measurements"][:]
        native_row = f["layer_index/native/feature_row"][:]
        if not np.array_equal(ids[native_row], native["source_id"]):
            raise ValueError("Native measurement and feature-ID association differs")
        roles = assign_roles(ids, partition, available, reference_ids)
        np.testing.assert_array_equal(
            roles["identity_split"], f["features/label_split"][:]
        )
        names = json.loads(f.attrs["feature_columns_json"])
        j = names.index("PARALLAX")
        values = f["features/values"][:, j]
        errors = f["features/reported_feature_errors"][:, j]
        errors_observed = f["features/error_observed"][:, j]
        value_observed = f["features/feature_observed"][:, j]
        parallax_observed = (
            value_observed
            & errors_observed
            & np.isfinite(values)
            & np.isfinite(errors)
            & (errors > 0)
        )
        cohort_metadata = dict(f.attrs)
    target = np.full((len(ids), 3), np.nan)
    uncertainty = np.full_like(target, np.nan)
    observed = np.zeros_like(target, dtype=bool)
    target_quality = np.zeros_like(observed)
    dependent_logg = np.zeros(len(ids), dtype=bool)
    original_paths = np.full(len(ids), b"", dtype="S512")
    original_rows = np.full(len(ids), -1, dtype="i8")
    survey = np.full(len(ids), b"", dtype="S32")
    raw = {}
    metadata = {}
    for source, spec in SOURCE_COLUMNS.items():
        selected = native["survey"] == source.encode()
        requested = native[selected]
        if len(np.unique(requested["input_file"])) != 1:
            raise ValueError("Expected one fixed original catalogue per native source")
        if Path(requested["input_file"][0].decode()).name != Path(sources[source]).name:
            raise ValueError(
                "Original source basename differs from accepted provenance"
            )
        feature_rows = native_row[selected]
        with fits.open(sources[source], memmap=True) as hdul:
            symbols = None
            if source == "apogee":
                symbols = (
                    np.asarray(hdul[2].data["PARAM_SYMBOL"][0]).reshape(-1).tolist()
                )
            projected = project_catalogue(
                hdul[1].data, source, requested, parameter_symbols=symbols
            )
            raw[source] = projected["raw"]
            raw[source]["feature_row"] = feature_rows
            raw[source]["source_row"] = requested["source_row"]
            metadata[source] = {
                "source_path": str(sources[source]),
                "accepted_original_path": requested["input_file"][0].decode(),
                "sha256": source_sha[source],
                "rows": len(hdul[1].data),
                "selected_rows": len(requested),
                "parameter_symbols": [str(s) for s in symbols]
                if symbols is not None
                else None,
                "fits_units": {
                    c.name: c.unit
                    for c in hdul[1].columns
                    if c.name in projected["raw"]
                },
            }
        for name, dest in [
            ("value", target),
            ("uncertainty", uncertainty),
            ("observed", observed),
            ("quality_pass", target_quality),
        ]:
            dest[feature_rows] = projected[name]
        dependent_logg[feature_rows] = projected["logg_parallax_conditioned"]
        original_paths[feature_rows] = requested["input_file"]
        original_rows[feature_rows] = requested["source_row"]
        survey[feature_rows] = source.encode()
    roles["fit_complete_targets"] = roles["fit"] & observed.all(1)
    # Raw Gaia errors do not establish five/six-parameter quality or a zero-point model.
    counts = {
        name: int(mask.sum()) for name, mask in roles.items() if mask.dtype == bool
    }
    report = {
        "schema": "joint-native-labels-v1",
        "cohort_sha256": sha(cohort),
        "cohort_content_id": str(cohort_metadata["content_id"]),
        "target_names": TARGET_NAMES,
        "target_units": TARGET_UNITS,
        "reference_holdout_count": len(np.unique(reference_ids)),
        "reference_holdout_ordered_id_sha256": reference_digest,
        "counts": counts,
        "sources": metadata,
        "target_valid_counts": dict(
            zip(TARGET_NAMES, observed.sum(0).tolist(), strict=True)
        ),
        "publication_gate_complete": True,
        "training_launch_ready": False,
        "alpha_enabled": False,
        "age_enabled": False,
        "limitations": [
            "No new survey acquisition; original raw source rows, units metadata and target flags preserved.",
            "GALAH DR4 logg is parallax-adjusted; do not treat its error as independent of the Gaia parallax likelihood.",
            "APOGEE alpha/M is not alpha/Fe; alpha not materialized as a universal target.",
            "Native quoted errors are not a certified total calibrated covariance; cross-target covariance remains unknown.",
            "Finite raw Gaia parallax plus formal error is not an astrometric quality/zero-point certificate; distance_fit_eligible remains false.",
            "Existing Li-VMPS source rows are retained in native provenance but are not promoted to joint high-resolution targets.",
            "Identity test contains already inspected stars and does not certify unseen unsupervised pretraining.",
        ],
        "documentation": [
            "https://www.galah-survey.org/dr4/table_schema/",
            "https://www.galah-survey.org/dr4/using_the_data/",
            "https://www.sdss4.org/dr17/irspec/aspcap/",
            "https://www.sdss4.org/dr17/irspec/apogee-bitmasks/",
        ],
    }
    staged = output.with_suffix(".building")
    with h5py.File(staged, "x") as f:
        f.attrs["schema"] = report["schema"]
        f.attrs["complete"] = False
        f.attrs["provenance_json"] = json.dumps(report, allow_nan=False)
        f.create_dataset("source_id", data=ids)
        f.create_dataset("native_measurements", data=native, compression="gzip")
        f.create_dataset("native_feature_row", data=native_row)
        for name, data in [
            ("value", target),
            ("uncertainty", uncertainty),
            ("observed", observed),
            ("quality_pass", target_quality),
        ]:
            f.create_dataset("targets/" + name, data=data, compression="gzip")
        f.create_dataset("targets/logg_parallax_conditioned", data=dependent_logg)
        f.create_dataset("targets/source_file", data=original_paths, compression="gzip")
        f.create_dataset("targets/source_row", data=original_rows)
        f.create_dataset("targets/survey", data=survey)
        for name, data in roles.items():
            f.create_dataset("roles/" + name, data=data)
        for name, data in [
            ("parallax_mas", values),
            ("parallax_error_mas", errors),
            ("observed", parallax_observed),
        ]:
            f.create_dataset("astrometry/" + name, data=data)
        f.create_dataset(
            "astrometry/distance_fit_eligible", data=np.zeros(len(ids), bool)
        )
        f.create_dataset("reference_holdout_source_id", data=np.unique(reference_ids))
        for source, fields in raw.items():
            for name, data in fields.items():
                f.create_dataset(
                    "raw/" + source + "/" + name, data=data, compression="gzip"
                )
        f.attrs["complete"] = True
        f.flush()
    staged.replace(output)
    report["output_sha256"] = sha(output)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--galah", type=Path, required=True)
    p.add_argument("--apogee", type=Path, required=True)
    p.add_argument("--source-sha-json", type=Path, required=True)
    p.add_argument(
        "--reference-ids",
        type=Path,
        required=True,
        help="NPY complete declared holdout-ID union",
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = materialize(
        a.cohort,
        {"galah": a.galah, "apogee": a.apogee},
        json.loads(a.source_sha_json.read_text()),
        np.load(a.reference_ids, allow_pickle=False),
        a.output,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
