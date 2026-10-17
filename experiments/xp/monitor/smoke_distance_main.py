"""Full distance runner on actual inputs plus an explicit sparse schema fixture.

The fixture exercises the ORIGINAL nested-record layout and unit-reader code.
It is NOT the immutable 121GB benchmark and cannot certify its XP associations.
The accepted GPU must still perform the real original benchmark unit audit.
"""

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import run_distance_value as runner
import run_xp_representation_probes as probe
from stellar_dataset_views import EXPECTED_IDS

root = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
config = yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text())
features = config["data"]["feature_cols"]
x, _, ids, _, _ = probe.read_fits(
    "/tmp/msa-labels-complete-20261015.fits", features, ["fe_h"], "g_normalized"
)
fields = ["G"] + [n for n in features if n.startswith(("bp_", "rp_"))]
valid = np.isfinite(x[:, [features.index(n) for n in fields]]).all(1)
take = np.flatnonzero(valid)[:128]
dtype = np.dtype([("source_id", "<i8")] + [(n, "<f4") for n in fields])
rows = np.zeros(128, dtype=dtype)
rows["source_id"] = ids[take]
for n in fields:
    rows[n] = x[take, features.index(n)]
fixture = Path("/tmp/msa-distance-original-layout-schema-fixture.h5")
with h5py.File(fixture, "w") as f:
    f.attrs["provenance"] = json.dumps(
        dict(
            source_id_sha256=EXPECTED_IDS, LOCAL_SCHEMA_FIXTURE_NOT_REAL_BENCHMARK=True
        )
    )
    for split, count in [("train", 9000000), ("validation", 1000000)]:
        ds = f.create_group(split).create_dataset(
            "records", shape=(count,), dtype=dtype, chunks=(256,)
        )
        if split == "train":
            ds[:128] = rows
runner.ROOT = root
runner.PARENT = "/tmp/msa-standard-checkpoint-57.pth"
runner.LEGACY = "/tmp/msa-labels-complete-20261015.fits"
runner.BENCH = fixture
out = root / "distance-main-local-smoke-v1"
sys.argv = [
    "run_distance_value.py",
    "--run-id",
    "LOCAL_SMOKE_NOT_SCIENTIFIC_RESULT",
    "--output",
    str(out),
]
started = time.monotonic()
runner.main()
(out / "LOCAL_SMOKE_LIMITATIONS.json").write_text(
    json.dumps(
        dict(
            actual_native_and_reference_inputs=True,
            actual_parent=True,
            all_31550_rows_per_mode=True,
            full_ridge_and_encoder_reload_workflow=True,
            original_benchmark_schema_fixture_only=True,
            actual_original_benchmark_XP_associations_NOT_verified_by_this_smoke=True,
            scientific_result=False,
            runtime_seconds=time.monotonic() - started,
        ),
        indent=2,
    )
    + "\n"
)
print(
    "Full actual-input runner smoke passed; original benchmark schema fixture only, not a scientific result"
)
