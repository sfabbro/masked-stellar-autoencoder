"""Recover admitted survey stars outside the current input union from full raw Gaia.

Source staging only: no roles, source activation, training or modification of the
already admitted large-run input bundle. Cache absence is never labelled no-XP.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from masked_stellar_autoencoder.training.evaluation_feature_lookup import (
    lookup_evaluation_features,
)

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--feature-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--parent-run",
        default="expanded_six_source_20261107_v2",
        help="complete source run whose stars are already covered",
    )
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    c = json.loads(a.config.read_text())
    feature = json.loads(a.feature_config.read_text())
    source = json.loads((ROOT / a.parent_run / "manifest.json").read_text())
    if source.get("status") != "expanded_six_source_complete":
        raise ValueError("Current admitted input union is not ready")
    with h5py.File(source["prepared_path"]) as f:
        old = f["source_id"][:]
    candidates = []
    for spec in c["additional_sources"]:
        if sha(spec["path"]) != spec["sha256"]:
            raise ValueError("Native source changed")
        with h5py.File(spec["path"]) as f:
            g = f["native_observations/" + spec["name"]]
            sid = g["source_id"][:]
            obs = g["observed"][:].astype(bool).any(1)
            candidates.append(sid[obs])
    union = np.unique(np.concatenate(candidates))
    ids = union[~np.isin(union, old)]
    m = dict(
        status="recovering_full_raw_inputs",
        training_performed=False,
        role_assignment_performed=False,
        training_source_ready=False,
        current_source_sha256=source["prepared_sha256"],
        parent_run=a.parent_run,
        additional_candidate_stars=len(ids),
        scope="All admitted additional-survey candidates outside current 9.2M-star input union; full raw lookup including cache misses. Source-only, no automatic change to the active training job or its roles.",
    )

    def save():
        path = a.output / "manifest.tmp"
        path.write_text(json.dumps(m, indent=2) + "\n")
        path.replace(a.output / "manifest.json")

    save()
    cache_manifest = json.loads((Path(feature["cache"]) / "manifest.json").read_text())
    binding = cache_manifest["request"]
    names = binding["features"]
    center = np.asarray(binding["center"])
    scale = np.asarray(binding["scale"])
    arrays, proof = lookup_evaluation_features(
        ids,
        names,
        center,
        scale,
        legacy_path=feature["legacy"],
        expected_legacy_sha256=feature["legacy_sha256"],
        cache_path=feature["cache"],
        original_path=feature["original"],
        output_directory=a.output / "feature-cache",
        expected_cache_request_sha256=feature["cache_request_sha256"],
        scan_original=True,
        allow_unknown_photometry=False,
        association_rows=128,
    )
    (a.output / "feature-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    raw_keys = binding["train_keys"] + binding["validation_keys"]
    shards = np.asarray(arrays["original_shard"])
    rows = np.asarray(arrays["original_row"])
    value = np.full(len(ids), np.nan)
    error = value.copy()
    found = np.zeros(len(ids), bool)
    with h5py.File(feature["original"]) as raw:
        for shard, key in enumerate(raw_keys):
            take = np.flatnonzero((shards == shard) & (rows >= 0))
            order = np.argsort(rows[take])
            take = take[order]
            rr = rows[take]
            for cell in np.unique(rr // 65536):
                left, right = np.searchsorted(
                    rr, [int(cell) * 65536, (int(cell) + 1) * 65536]
                )
                pos = take[left:right]
                start = int(cell) * 65536
                block = raw[key].fields(["source_id", "PARALLAX", "e_parallax"])[
                    start : min(start + 65536, len(raw[key]))
                ]
                records = block[rr[left:right] - start]
                np.testing.assert_array_equal(records["source_id"], ids[pos])
                value[pos] = records["PARALLAX"]
                error[pos] = records["e_parallax"]
                found[pos] = True
    observed = (
        found & np.isfinite(value) & (value != -9999) & np.isfinite(error) & (error > 0)
    )
    output = a.output / "additional-star-features.h5"
    with h5py.File(output, "x") as f:
        f["source_id"] = ids
        f["feature_names"] = np.asarray(names, dtype=h5py.string_dtype())
        f["scaler_center"] = center
        f["scaler_scale"] = scale
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
            dest = f.create_dataset(
                key, shape=data.shape, dtype=data.dtype, chunks=True, compression="lzf"
            )
            for start in range(0, len(ids), 16384):
                dest[start : start + 16384] = data[start : start + 16384]
        f["xp_status"] = np.asarray(arrays["xp_status"], dtype="S48")
        f["measured_parallax/parallax_mas"] = value
        f["measured_parallax/parallax_error_mas"] = error
        f["measured_parallax/observed"] = observed
        f.attrs["scope"] = m["scope"]
        f.attrs["raw_shards_json"] = json.dumps(raw_keys)
    m.update(
        status="additional_full_raw_features_complete",
        input_recovered=int(arrays["feature_recoverable"].sum()),
        input_with_xp=int(arrays["input_has_xp"].sum()),
        raw_parallax_with_formal_error=int(observed.sum()),
        output_sha256=sha(output),
        remaining="Join identities to original supervised/benchmark/dependency roles and fresh protected groups before any future activation. The running large candidate keeps its pinned existing source.",
    )
    save()


if __name__ == "__main__":
    main()
