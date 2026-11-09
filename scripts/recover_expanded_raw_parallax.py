"""Recover exact-ID native parallax/error records for expanded cache members."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
CACHE = ROOT / "residual-huber-e86-150m-cache-v1"
FEATURE = ROOT / "multisurvey_feature_union_dr12_20261106_v1/new-star-features.h5"
RAW = Path("/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5")


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    result = dict(status="admitting", training_performed=False, profile_activated=False)

    def save():
        temp = a.output / "manifest.tmp"
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(a.output / "manifest.json")

    save()
    try:
        if (
            sha(FEATURE)
            != "d4d8ea8e972625cb040c0fdcf4cc60675c03823b43826cef4b57d6d725e14f23"
        ):
            raise ValueError("Expanded feature cohort changed")
        receipt = json.loads((CACHE / "manifest.json").read_text())
        if (
            receipt["request_sha256"]
            != "8d14aa3f741946e8fa220fd0d9f8d74bb44f5949e2f9d2ff0449d21cfa5e5b9c"
        ):
            raise ValueError("Cache normalization binding changed")
        for name in ["source_id.npy", "provenance.npy"]:
            if sha(CACHE / name) != receipt["files"][name]["sha256"]:
                raise ValueError("Cache identity/provenance changed: " + name)
        census = json.loads(
            (
                ROOT / "andrae_full_footprint_census_20261106_v2/manifest.json"
            ).read_text()
        )
        before = RAW.stat()
        identity = census["raw_identity"]
        if (
            identity["path"] != str(RAW)
            or identity["bytes"] != before.st_size
            or identity["mtime_ns"] != before.st_mtime_ns
        ):
            raise ValueError("Raw input differs from full-footprint census")
        with h5py.File(FEATURE) as f:
            ids, rows = f["source_id"][:], f["cache_row"][:]
            names = f["feature_names"][:].astype("U").tolist()
            center, scale = f["scaler_center"][:], f["scaler_scale"][:]
        cached_ids = np.load(CACHE / "source_id.npy", mmap_mode="r")
        provenance = np.load(CACHE / "provenance.npy", mmap_mode="r")
        cached_features = np.load(CACHE / "features.npy", mmap_mode="r")
        hit = np.flatnonzero(rows >= 0)
        np.testing.assert_array_equal(cached_ids[rows[hit]], ids[hit])
        locations = provenance[rows[hit]]
        value, error = np.full(len(ids), np.nan), np.full(len(ids), np.nan)
        shard_out, row_out = (
            np.full(len(ids), -1, dtype="i4"),
            np.full(len(ids), -1, dtype="i8"),
        )
        max_cache_difference = 0.0
        j = names.index("PARALLAX")
        result.update(
            status="recovering_native_records",
            candidate_stars=len(ids),
            cache_members=len(hit),
            verified_records=0,
            raw_identity=identity,
        )
        save()
        with h5py.File(RAW) as f:
            # Cache source-shard order is part of the immutable request.
            keys = receipt["request"].get("train_keys")
            if not isinstance(keys, list) or not all(
                isinstance(k, str) and k in f for k in keys
            ):
                raise ValueError(
                    "Explicit cache raw-shard ordering is required; no guessed key order"
                )
            result["raw_shards"] = keys
            result["raw_attributes"] = {k: str(v) for k, v in f.attrs.items()}
            for shard, key in enumerate(keys):
                select = np.flatnonzero(locations["shard"] == shard)
                raw_rows = locations["row"][select]
                d = f[key]
                if not {"source_id", "PARALLAX", "e_parallax"}.issubset(
                    d.dtype.names or ()
                ):
                    raise ValueError("Native parallax schema unavailable: " + key)
                if len(raw_rows) and (
                    (raw_rows < 0).any() or (raw_rows >= len(d)).any()
                ):
                    raise ValueError("Raw row outside immutable shard")
                order = np.argsort(raw_rows)
                select, raw_rows = select[order], raw_rows[order]
                for cell in np.unique(raw_rows // 65536):
                    left, right = np.searchsorted(
                        raw_rows, [int(cell) * 65536, (int(cell) + 1) * 65536]
                    )
                    pos = hit[select[left:right]]
                    rr = raw_rows[left:right]
                    start = int(cell) * 65536
                    block = d.fields(["source_id", "PARALLAX", "e_parallax"])[
                        start : min(start + 65536, len(d))
                    ]
                    selected = block[rr - start]
                    np.testing.assert_array_equal(selected["source_id"], ids[pos])
                    value[pos], error[pos] = (
                        selected["PARALLAX"],
                        selected["e_parallax"],
                    )
                    shard_out[pos], row_out[pos] = shard, rr
                    cached = cached_features[rows[pos], j]
                    native_good = (
                        np.isfinite(value[pos])
                        & (value[pos] != -9999)
                        & np.isfinite(cached)
                        & (cached != -9999)
                    )
                    if native_good.any():
                        delta = abs(
                            (value[pos][native_good] - center[j]) / scale[j]
                            - cached[native_good]
                        )
                        max_cache_difference = max(
                            max_cache_difference, float(delta.max())
                        )
                        np.testing.assert_allclose(
                            (value[pos][native_good] - center[j]) / scale[j],
                            cached[native_good],
                            rtol=2e-5,
                            atol=2e-5,
                        )
                result["verified_records"] += len(select)
                result["last_shard"] = key
                save()
        after = RAW.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Raw source changed during recovery")
        observed = (
            (row_out >= 0)
            & np.isfinite(value)
            & (value != -9999)
            & np.isfinite(error)
            & (error > 0)
        )
        with h5py.File(a.output / "native-parallax.h5", "x") as f:
            for name, data in dict(
                source_id=ids,
                parallax_mas=value,
                parallax_error_mas=error,
                observed=observed,
                raw_shard=shard_out,
                raw_row=row_out,
            ).items():
                f.create_dataset(name, data=data, compression="lzf")
            f.attrs["scope"] = (
                "Raw signed Gaia parallax and formal error in the repository native-unit contract (mas); no zero-point or astrometric quality correction; exact-ID verified, no distance inversion, not activated as a training profile."
            )
        result.update(
            status="complete",
            usable_measurements=int(observed.sum()),
            negative_parallaxes_retained=int((observed & (value < 0)).sum()),
            outside_cache_unrecovered=int((rows < 0).sum()),
            max_normalized_cache_difference=max_cache_difference,
            output_sha256=sha(a.output / "native-parallax.h5"),
        )
        save()
    except Exception as exc:
        result.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
