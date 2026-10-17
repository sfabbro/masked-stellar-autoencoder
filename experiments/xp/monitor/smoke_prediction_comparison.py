"""Actual epoch72/cohort integration, explicitly surrogate benchmark schema.

This never certifies the remote original128-star XP unit check or 10M budget.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
base = (
    ROOT
    / "experiments/xp/results-stellar-improvement/prediction-e72-local-integration-v2"
)
base.mkdir(exist_ok=False)
cohort_path = (
    ROOT
    / "experiments/xp/results-stellar-improvement/enriched_feature_cohort_v1/enriched-feature-cohort.h5"
)
config = yaml.safe_load((ROOT / "configs/pretrain.canfar.yaml").read_text())
features = config["data"]["feature_cols"]
fixture = base / "actual-values-benchmark-SCHEMA-FIXTURE.h5"
with h5py.File(cohort_path) as source, h5py.File(fixture, "w") as dest:
    raw = source["features/values"][:640]
    for group, values in (("train", raw[:512]), ("validation", raw[512:])):
        records = np.empty(len(values), dtype=[(name, "f8") for name in features])
        for j, name in enumerate(features):
            records[name] = values[:, j]
        g = dest.create_group(group)
        g.create_dataset("records", data=records)
        if group == "validation":
            g.attrs["source_shard_names"] = json.dumps(config["data"]["valid_keys"])
    scaler = dest.create_group("reference_scaler")
    for name in ("center", "scale"):
        scaler.create_dataset(name, data=source["reference_scaler/" + name][:])
    scaler.attrs["feature_cols"] = json.dumps(features)
    dest.attrs["LIMITATION"] = (
        "Actual published cohort values in a surrogate schema. Not the immutable original benchmark."
    )
suite = yaml.safe_load((ROOT / "configs/prediction-comparison-10m.yaml").read_text())
for arm in suite["arms"]:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_prediction_comparison.py"),
            "--run-id",
            arm["run_id"],
            "--output",
            str(base / arm["run_id"]),
            "--checkpoint",
            "/tmp/msa_pretrain_checkpoint_72.pth",
            "--benchmark",
            str(fixture),
            "--cohort",
            str(cohort_path),
            "--smoke",
        ],
        cwd=ROOT,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        check=True,
    )
reports = [
    json.loads((base / a["run_id"] / "manifest.json").read_text())
    for a in suite["arms"]
]
for field in ("parent_sha256", "row_sha256", "mask_sha256", "supervised_stream_sha256"):
    assert len({r[field] for r in reports}) == 1, field
assert all(
    r["status"] == "complete" and r["smoke"] and r["presentations"] == 512
    for r in reports
)
report = {
    "status": "passed",
    "arms": len(reports),
    "actual_epoch72_parent_and_full_head_decoder_reload": True,
    "row_mask_supervised_stream_hashes_paired": True,
    "limitations": "128 actual native fit/test stars each;512 replay presentations from actual published values in a benchmark SCHEMA FIXTURE; not original128unit certification, whole-cohort production inference,10M training or science.",
}
(base / "integration-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
