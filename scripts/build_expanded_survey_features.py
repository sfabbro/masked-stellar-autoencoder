"""Source-only feature recovery outside the current base; no fitting or inference."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

from masked_stellar_autoencoder.training.evaluation_feature_lookup import (
    lookup_evaluation_features,
)


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    c = json.loads(args.config.read_text())
    if (
        c["training_enabled"]
        or c["assign_roles"]
        or c["scan_original"]
        or c["allow_unknown_photometry"]
    ):
        raise ValueError(
            "This bounded source-only recovery does not enable training, assign roles or scan the full original source"
        )
    output = Path(c["output"])
    if args.output is not None and args.output != output:
        raise ValueError("Output override differs from the source contract")
    output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(output.parent).free < 50_000_000_000:
        raise RuntimeError("At least 50GB filesystem reserve is required")
    manifest = dict(
        status="admitting_sources",
        training_performed=False,
        model_loaded=False,
        role_assignment_performed=False,
        config_sha256=sha(args.config),
        config=c,
    )

    def save(**updates):
        manifest.update(updates)
        temp = output / "manifest.tmp"
        temp.write_text(json.dumps(manifest, indent=2) + "\n")
        temp.replace(output / "manifest.json")
        print(json.dumps(dict(status=manifest["status"])), flush=True)

    save()
    candidates = []
    for path, expected, name in [
        (c["gaia_sidecar"], c["gaia_sha256"], "gaia_dr3_gspspec"),
        (
            c["lamost_sidecar"],
            c["lamost_sha256"],
            c.get("lamost_source_name", "lamost_dr8v2_lrs_lasp"),
        ),
    ]:
        if sha(path) != expected:
            raise ValueError("Source sidecar digest differs: " + name)
        with h5py.File(path) as f:
            g = f["native_observations/" + name]
            sid = g["source_id"][:]
            observed = g["observed"][:].any(1)
            candidates.append(sid[observed])
    union = np.unique(np.concatenate(candidates)).astype("i8")
    with h5py.File(c["base"]) as base:
        old = base["source_id"][:]
    new = union[~np.isin(union, old)]
    np.save(output / "source_id.npy", new)
    binding = json.loads((Path(c["cache"]) / "manifest.json").read_text())["request"]
    names = binding["features"]
    center = np.asarray(binding["center"])
    scale = np.asarray(binding["scale"])
    save(
        status="recovering_features",
        candidate_union=len(union),
        new_candidate_stars=len(new),
        new_ids_sha256=hashlib.sha256(new.astype("<i8").tobytes()).hexdigest(),
    )
    arrays, proof = lookup_evaluation_features(
        new,
        names,
        center,
        scale,
        legacy_path=c["legacy"],
        expected_legacy_sha256=c["legacy_sha256"],
        cache_path=c["cache"],
        original_path=c["original"],
        output_directory=output / "feature-cache",
        expected_cache_request_sha256=c["cache_request_sha256"],
        scan_original=False,
        allow_unknown_photometry=False,
        association_rows=128,
    )
    (output / "feature-proof.json").write_text(
        json.dumps(proof, indent=2, allow_nan=False) + "\n"
    )
    save(
        status="writing_feature_table",
        recovered_stars=int(arrays["feature_recoverable"].sum()),
    )
    with h5py.File(output / "new-star-features.h5", "x") as f:
        f.attrs["scope"] = (
            "New source candidates only; no roles, labels, model training or physical-scale certificate"
        )
        f.attrs["normalization_binding"] = (
            "validated 150M cache request; must be checked again against any training parent"
        )
        f.create_dataset("source_id", data=new)
        f.create_dataset(
            "feature_names", data=np.array(names, dtype=h5py.string_dtype())
        )
        f.create_dataset("scaler_center", data=center)
        f.create_dataset("scaler_scale", data=scale)
        for key in [
            "input",
            "visible",
            "feature_recoverable",
            "input_has_xp",
            "real_xp_present",
            "real_no_xp",
            "cache_row",
            "original_row",
            "original_shard",
            "input_xp_missing_publication_unknown",
        ]:
            data = arrays[key]
            dataset = f.create_dataset(
                key, shape=data.shape, dtype=data.dtype, chunks=True, compression="lzf"
            )
            for start in range(0, len(new), 32768):
                dataset[start : start + 32768] = data[start : start + 32768]
        f.create_dataset("xp_status", data=np.asarray(arrays["xp_status"], dtype="S48"))
    # Reopen and verify identity and a bounded byte-for-byte feature sample.
    with h5py.File(output / "new-star-features.h5") as f:
        np.testing.assert_array_equal(f["source_id"][:], new)
        for row in np.linspace(0, len(new) - 1, min(128, len(new)), dtype=int):
            np.testing.assert_array_equal(f["input"][row], arrays["input"][row])
    save(
        status="feature_union_candidate_complete",
        counts=proof["counts"],
        recovered_stars=int(arrays["feature_recoverable"].sum()),
        input_xp_present=int(arrays["input_has_xp"].sum()),
        feature_table_sha256=sha(output / "new-star-features.h5"),
        feature_proof_sha256=sha(output / "feature-proof.json"),
        source_scope="Existing base not modified or copied; new candidates remain unassigned; only available cached/legacy inputs recovered, no full original-source scan",
    )


if __name__ == "__main__":
    main()
