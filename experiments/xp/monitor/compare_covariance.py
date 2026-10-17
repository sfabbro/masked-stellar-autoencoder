import csv
import json
from pathlib import Path

root = Path("experiments/xp/results-covariance")
names = ["unchanged_control", "corrected_diagonal_xp", "full_covariance_xp"]
data = {}
for name in names:
    folder = root / name
    m = json.loads((folder / "manifest.json").read_text())
    q = json.loads((folder / "residual_latest.json").read_text())
    v = json.loads((folder / "checkpoint_verification.json").read_text())
    metrics = [
        json.loads(line) for line in (folder / "metrics.jsonl").read_text().splitlines()
    ]
    final = metrics[-1]
    assert m["status"] == "complete" and final["phase"] == "complete"
    assert final["rows_seen_total"] == 20000000 and final["optimizer_steps"] == 612
    assert (
        v["reload_predictions_equal"] and v["parent_sha256"] == m["checkpoint_sha256"]
    )
    assert q["validation_id"] == m["validation_id"] == final["validation_id"]
    data[name] = (m, q, final, metrics[0])
control = data[names[0]]
for name, (m, q, final, initial) in data.items():
    for field in [
        "checkpoint_sha256",
        "checkpoint_epoch",
        "benchmark_source_id_sha256",
        "benchmark_manifest_sha256",
        "validation_id",
        "seed",
        "validation_seed",
        "independent_rng",
        "mask_policy",
        "runner_sha256",
        "xp_helpers_sha256",
        "experiment_environment_lock_sha256",
    ]:
        assert m[field] == control[0][field], (name, field)
    assert (
        m["sample"]["train_rows"] == 9000000
        and m["sample"]["validation_rows"] == 1000000
    )
    assert m["config"]["data"] == control[0]["config"]["data"]
    assert m["config"]["training"] == control[0]["config"]["training"]
    for field in ["all_columns_mae", "xp_only_mae"]:
        assert abs(initial[field] - control[3][field]) < 1e-7, (name, "initial", field)
rows = []
for regime in control[1]["regimes"]:
    baseline = {
        item["name"]: item for item in control[1]["regimes"][regime]["features"]
    }
    for name, (_, q, _, _) in data.items():
        features = q["regimes"][regime]["features"]
        assert {item["name"] for item in features} == set(baseline)
        for item in features:
            ref = baseline[item["name"]]
            assert item["count"] == ref["count"]
            delta = (
                item["mae"] - ref["mae"]
                if item["mae"] is not None and ref["mae"] is not None
                else None
            )
            rows.append(
                {
                    "experiment": name,
                    "regime": regime,
                    "column": item["name"],
                    "count": item["count"],
                    "mae": item["mae"],
                    "control_mae": ref["mae"],
                    "delta_mae": delta,
                    "delta_percent": 100 * delta / ref["mae"]
                    if delta is not None and ref["mae"]
                    else None,
                }
            )
with (root / "per-column-mae.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
summary = {
    "identity_checks": "passed",
    "parent_checkpoint_sha256": control[0]["checkpoint_sha256"],
    "benchmark_source_id_sha256": control[0]["benchmark_source_id_sha256"],
    "validation_id": control[0]["validation_id"],
    "validation_stars": 1000000,
    "training_stars": 9000000,
    "presentations": 20000000,
    "optimizer_steps": 612,
    "experiments": [],
}
for name, (m, q, final, initial) in data.items():
    spectral = q["regimes"]["common"]["spectral"]
    entry = {
        "experiment": name,
        "all_reconstructed_mae": final["all_columns_mae"],
        "xp_only_mae": final["xp_only_mae"],
        "all_mae_delta_percent": 100
        * (final["all_columns_mae"] / control[2]["all_columns_mae"] - 1),
        "xp_mae_delta_percent": 100
        * (final["xp_only_mae"] / control[2]["xp_only_mae"] - 1),
        "spectral_rows": spectral["rows"],
        "spectral_integrated_absolute_fraction_mean": spectral[
            "integrated_absolute_fraction_mean"
        ],
        "spectral_integrated_absolute_fraction_median": spectral[
            "integrated_absolute_fraction_median"
        ],
        "spectral_mean_delta_percent": 100
        * (
            spectral["integrated_absolute_fraction_mean"]
            / control[1]["regimes"]["common"]["spectral"][
                "integrated_absolute_fraction_mean"
            ]
            - 1
        ),
    }
    summary["experiments"].append(entry)
(root / "comparison.json").write_text(json.dumps(summary, indent=2))
report = [
    "# Corrected XP covariance comparison",
    "",
    "All three completed the matched 20M-presentation continuation (612 steps). Parent checkpoint, epoch-27 scaler validation, benchmark identity, validation data/masks, seeds, initial predictions, training configuration and checkpoint reload probes match. Validation MAE uses all 1M stars and hidden finite reconstructed entries. Spectral diagnostics use 4,996 common-mask stars from the deterministic 10,000-star diagnostic sample.",
    "",
]
for entry in summary["experiments"]:
    report.append(
        f"- {entry['experiment']}: all reconstructed-column MAE {entry['all_reconstructed_mae']:.9f}; XP-only MAE {entry['xp_only_mae']:.9f} ({entry['xp_mae_delta_percent']:+.3f}% versus control). Spectral integrated absolute fraction mean {entry['spectral_integrated_absolute_fraction_mean']:.9f} ({entry['spectral_mean_delta_percent']:+.3f}%), median {entry['spectral_integrated_absolute_fraction_median']:.9f}."
    )
report += [
    "",
    "Neither corrected uncertainty objective improves XP coefficient MAE over the unchanged MAE control in this bounded continuation. These are single-seed continuations with inherited AdamW moments and a fixed 0.1 predictive-variance floor; they do not establish that covariance-aware objectives cannot help under another floor, optimizer policy or longer training. Full covariance retains BP/RP correlations and scores hidden principal submatrices; both uncertainty experiments use corrected raw errors divided by the G flux factor and RobustScaler scale. Other columns retain MAE.",
    "",
    "Per-column MAE, counts and deltas for every validation mask regime are saved in per-column-mae.csv. Raw manifests, diagnostics, metrics and checkpoint verification JSON are retained in each experiment directory. Existing ARC checkpoints remain persistent.",
]
(root / "comparison.md").write_text("\n".join(report) + "\n")
print(json.dumps(summary, indent=2))
