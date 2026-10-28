import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

root = Path("/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3")
manifest = json.loads((root / "benchmark.h5.manifest.json").read_text())
assert manifest["status"] == "complete"
result = {
    "manifest": manifest,
    "size_bytes": (root / "benchmark.h5").stat().st_size,
    "checks": {},
}
with (
    h5py.File(root / "benchmark.h5") as f,
    h5py.File(
        "/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261011/benchmark.h5.building"
    ) as old,
):
    ids = []
    for split, n in [("train", 9000000), ("validation", 1000000)]:
        g = f[split]
        assert (
            len(g["records"]) == n and g["records"].dtype == old[split]["records"].dtype
        )
        ids.append(g["records"]["source_id"])
        assert g["xp_covariance_available"][:].all()
        for begin in range(0, n, 8192):
            end = min(begin + 8192, n)
            corr = g["xp_correlations"][begin:end]
            assert np.isfinite(corr).all() and np.all(abs(corr) <= 1.00001)
            assert np.isfinite(g["xp_band_metadata"][begin:end]).all()
            for field in ["records", "source_shard", "source_row"]:
                # Compound records contain NaNs; compare their retained bytes exactly.
                assert (
                    g[field][begin:end].tobytes()
                    == old[split][field][begin:end].tobytes()
                ), (split, field, begin)
        result["checks"][split] = {
            "rows": n,
            "all_records_provenance_equal": True,
            "full_covariance_coverage": True,
            "all_correlations_finite_in_range": True,
        }
        print("SPLIT_VERIFIED", split, flush=True)
    joined = np.concatenate(ids)
    assert np.unique(joined).size == 10000000
    digest = hashlib.sha256(joined.tobytes()).hexdigest()
    assert digest == "41c459bd0db60ca2312eb8b2f0e8c3366387ae2d98276f16a87bfb2972baabc2"
    result["checks"].update(source_id_sha256=digest, unique_ids_no_split_overlap=True)
    scaler = f["reference_scaler"]
    assert (
        scaler.attrs["checkpoint_sha256"]
        == "c60dfd816cade1b2ee255063840d4bee3e918f58e8bce3a907baa63e352f6d86"
    )
    for key in ["center", "scale"]:
        np.testing.assert_array_equal(scaler[key][:], old["reference_scaler"][key][:])
    with h5py.File(
        "/arc/projects/k-pop/msa_runs/xp-hdf5-comparison-20261012/XpContinuousMeanSpectrum_000000-003111.hdf5"
    ) as raw:
        ri = raw["source_id"][:]
        order = np.argsort(joined)
        sorted_ids = joined[order]
        p = np.searchsorted(sorted_ids, ri)
        keep = p < len(sorted_ids)
        keep[keep] &= sorted_ids[p[keep]] == ri[keep]
        selected = np.flatnonzero(keep)[:128]
        assert len(selected) > 0
        global_rows = order[p[selected]]
        records = []
        for row in global_rows:
            records.append(
                f["train/records"][int(row)]
                if row < 9000000
                else f["validation/records"][int(row - 9000000)]
            )
        records = np.array(records, dtype=f["train/records"].dtype)
        audit = {"matched_rows": len(records), "columns": list(records.dtype.names)}
        assert "G" in records.dtype.names
        flux = 10 ** ((8.5 - records["G"].astype("f8")) / 2.5)
        for band in ["bp", "rp"]:
            c = raw[band + "_coefficients"][selected]
            e = raw[band + "_coefficient_errors"][selected]
            stored_c = np.column_stack(
                [records[band + "_" + str(i)] for i in range(1, 56)]
            )
            stored_e = np.column_stack(
                [records[band + "e_" + str(i)] for i in range(1, 56)]
            )
            valid = np.isfinite(c) & (abs(c) > 1e-20) & np.isfinite(e) & (e > 0)
            audit[band] = {
                "coefficient_normalized_ratio": float(
                    np.median((stored_c * flux[:, None] / c)[valid])
                ),
                "error_normalized_ratio": float(
                    np.median((stored_e * flux[:, None] / e)[valid])
                ),
                "error_raw_ratio": float(np.median((stored_e / e)[valid])),
            }
        result["normalization_audit"] = audit
result["status"] = "verified_structure_and_coverage"
(root / "verification.json").write_text(json.dumps(result, indent=2))
print("RESULT " + json.dumps(result), flush=True)
