import csv
import json
from pathlib import Path

import numpy as np

root = Path("experiments/xp/results-stellar-improvement")
load = lambda p: json.loads(p.read_text())
base = root / "mae_b32768_lr1_s42"
bm = load(base / "manifest.json")
bp = load(base / base.name / "training_pairing.json")
with np.load(base / base.name / "residual_init.npz") as a:
    initial = {k: a[k].copy() for k in a.files}
rows = []
columns = []
fields = [
    "checkpoint_sha256",
    "source_code_sha256",
    "runner_sha256",
    "xp_helpers_sha256",
    "experiment_environment_lock_sha256",
    "validation_id",
    "benchmark_source_id_sha256",
    "benchmark_manifest_sha256",
    "config_sha256",
    "git_diff_sha256",
    "sample",
    "spectral_projection",
]
fields = [k for k in fields if k in bm]
for p in sorted(root.glob("mae_b*_s42")):
    if not (p / "manifest.json").exists():
        continue
    m = load(p / "manifest.json")
    pair = load(p / p.name / "training_pairing.json")
    policy = load(p / p.name / "optimizer_policy.json")
    reload = load(p / p.name / "checkpoint_verification.json")
    assert m["status"] == "complete", p
    for k in fields:
        assert m[k] == bm[k], (p, k)
    assert (
        m["checkpoint_sha256"]
        == "9b033ae63a3b87c93dcfeffd989172f7888d029f77143e6cf315cf7a032ab2ca"
    )
    assert pair["rows"] == 20000000
    for k in [
        "training_cache_row_index_sha256",
        "training_mask_sha256",
        "micro_batch_size",
        "mask_chunk_rows",
    ]:
        assert pair[k] == bp[k], (p, k)
    assert (
        pair["optimizer_steps"]
        == {32768: 612, 8192: 2443, 2048: 9768}[pair["optimizer_batch_size"]]
    )
    assert (
        reload["reload_predictions_equal"]
        and reload["parent_sha256"] == m["checkpoint_sha256"]
    )
    with np.load(p / p.name / "residual_init.npz") as a:
        assert set(a.files) == set(initial)
        for k in a.files:
            np.testing.assert_array_equal(a[k], initial[k], err_msg=str((p, k)))
    metrics = [
        json.loads(s) for s in (p / p.name / "metrics.jsonl").read_text().splitlines()
    ]
    final = metrics[-1]
    assert final["rows_seen_total"] == 20000000
    assert not policy["reset_moments"] and policy["warmup_steps"] == 0
    assert len(set(policy["learning_rates"])) == 1
    assert all(r["learning_rate"] == policy["learning_rates"][0] for r in metrics)
    rows.append(
        dict(
            run_id=p.name,
            batch=pair["optimizer_batch_size"],
            lr=final["learning_rate"],
            optimizer_steps=pair["optimizer_steps"],
            final_XP_MAE=final["xp_only_mae"],
            final_all_column_MAE=final["all_columns_mae"],
            train_seconds=final["train_seconds"],
            last_validation_seconds=final["validation_seconds"],
        )
    )
    for col, mae in final["per_column_mae"].items():
        columns.append(
            dict(
                run_id=p.name, column=col, mae=mae, count=final["per_column_count"][col]
            )
        )
assert len(rows) == 13
report = dict(
    scope="All13completed fixed-budget LR/batch continuations; seed42 only",
    checks=dict(
        complete=True,
        actual_rows_masks_equal=True,
        exact_initial_arrays_equal=True,
        parent_and_reload_verified=True,
        shared_manifest_fields=fields,
        optimizer_fixed_inherited_moments=True,
    ),
    row_sha256=bp["training_cache_row_index_sha256"],
    mask_sha256=bp["training_mask_sha256"],
    runs=rows,
    caveats=[
        "Single seed and bounded20M presentations; no downstream representation inference.",
        "train_seconds cumulative training only; last_validation_seconds is one evaluation, not total evaluation wall time.",
        "Compare by presentations and updates; batch size changes the number of optimizer updates.",
    ],
)
(root / "lr-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
for name, data in [("lr-final-metrics.csv", rows), ("lr-per-column.csv", columns)]:
    with (root / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=data[0].keys())
        w.writeheader()
        w.writerows(data)
for r in rows:
    print(r["run_id"], r["final_XP_MAE"])
