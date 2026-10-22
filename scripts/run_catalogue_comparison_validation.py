#!/usr/bin/env python3
"""Paired saved-catalogue metrics and object bootstrap; evaluation only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage(csv_path, destination, columns):
    frame = pd.read_csv(csv_path, dtype={"source_id": "int64"})
    ids = frame.source_id.to_numpy()
    if (ids <= 0).any() or len(np.unique(ids)) != len(ids):
        raise ValueError("paired CSV requires unique positive exact Gaia identities")
    y = frame[[x[1] for x in columns]].to_numpy(dtype=float)
    ours = frame[[x[2] for x in columns]].to_numpy(dtype=float)
    other = frame[[x[3] for x in columns]].to_numpy(dtype=float)
    heldout = np.zeros(len(frame), dtype=bool)
    if "andrae_training_sample" in frame:
        raw = frame["andrae_training_sample"].astype(str).str.lower()
        if not raw.isin(["true", "false"]).all():
            raise ValueError("unrecognized catalogue training membership")
        heldout = raw.eq("false").to_numpy()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    np.savez_compressed(
        destination,
        source_id=ids,
        y=y,
        ours=ours,
        other=other,
        feh_true=frame["feh_true"].to_numpy(dtype=float),
        heldout=heldout,
        heldout_available=np.array("andrae_training_sample" in frame),
        csv_sha256=np.array(digest(csv_path)),
    )
    return {
        "input_npz_sha256": digest(destination),
        "source_csv_sha256": digest(csv_path),
        "source_csv": str(csv_path),
        "n_rows": len(ids),
    }


def metrics(errors):
    return np.array(
        [
            np.mean(np.abs(errors)),
            np.mean(errors),
            np.median(np.abs(errors - np.median(errors))),
        ]
    )


def paired_bootstrap(ours, other, *, seed=20261022, repetitions=200):
    if len(ours) != len(other) or not len(ours):
        raise ValueError("nonempty paired residuals required")
    if not np.isfinite(ours).all() or not np.isfinite(other).all():
        raise ValueError("bootstrap requires same finite paired rows")
    rng = np.random.default_rng(seed)
    draws = np.empty((repetitions, 3))
    for i in range(repetitions):
        indices = rng.integers(len(ours), size=len(ours))
        draws[i] = metrics(ours[indices]) - metrics(other[indices])
    return np.quantile(draws, [0.025, 0.975], axis=0)


def run(case, input_path, output, *, repetitions=200):
    if digest(input_path) != case["input_npz_sha256"]:
        raise ValueError("NPZ SHA binding mismatch")
    with np.load(input_path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    if data["csv_sha256"].item() != case["source_csv_sha256"]:
        raise ValueError("source CSV SHA mismatch")
    ids, y, ours, other = (data[k] for k in ("source_id", "y", "ours", "other"))
    if (
        ids.dtype.kind not in "iu"
        or len(np.unique(ids)) != len(ids)
        or (ids <= 0).any()
    ):
        raise ValueError("invalid paired object identities")
    if (
        y.shape != ours.shape
        or y.shape != other.shape
        or y.shape != (len(ids), len(case["targets"]))
    ):
        raise ValueError("paired prediction shape mismatch")
    subsets = {
        "all": np.ones(len(ids), dtype=bool),
        "FeH_below_minus2": data["feh_true"] < -2,
    }
    if data["heldout_available"].item():
        subsets["catalogue_training_heldout"] = data["heldout"]
        subsets["catalogue_training_heldout_FeH_below_minus2"] = data["heldout"] & (
            data["feh_true"] < -2
        )
    rows = []
    for subset, mask in subsets.items():
        for j, target in enumerate(case["targets"]):
            valid = (
                mask
                & np.isfinite(y[:, j])
                & np.isfinite(ours[:, j])
                & np.isfinite(other[:, j])
            )
            a, b = ours[valid, j] - y[valid, j], other[valid, j] - y[valid, j]
            if len(a):
                ma, mb = metrics(a), metrics(b)
                ci = paired_bootstrap(a, b, repetitions=repetitions)
            for k, name in enumerate(("MAE", "bias", "MedAD")):
                rows.append(
                    {
                        "subset": subset,
                        "target": target,
                        "metric": name,
                        "n": len(a),
                        "excluded_nonfinite": int(mask.sum() - valid.sum()),
                        "ours": float(ma[k]) if len(a) else None,
                        "catalogue": float(mb[k]) if len(a) else None,
                        "difference_ours_minus_catalogue": float(ma[k] - mb[k])
                        if len(a)
                        else None,
                        "paired_bootstrap_95_low": float(ci[0, k]) if len(a) else None,
                        "paired_bootstrap_95_high": float(ci[1, k]) if len(a) else None,
                    }
                )
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "run_id": case["run_id"],
        "status": "paired_evaluation_only_complete",
        "input_npz_sha256": digest(input_path),
        "source_csv_sha256": case["source_csv_sha256"],
        "n_unique_paired_rows": len(ids),
        "source_id_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "bootstrap": {
            "unit": "paired_object_same_indices_both_models",
            "repetitions": repetitions,
            "seed": 20261022,
            "interval": "percentile_95",
            "cluster_survey_dependence_modelled": False,
        },
        "MedAD_definition": "median(abs(residual - median(residual)))",
        "bias_definition": "mean(prediction - native_reported_label)",
        "target_caveat": case["target_caveat"],
        "warnings": [
            "Native reported labels are not physical truth",
            "Same finite paired subset for both models per target and metric",
            "Bootstrap uncertainty is conditional on this overlap; point MAEs alone do not establish statistical indistinguishability",
            "No acquisition, fitting or production-model modification",
        ],
        "metrics": rows,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(rows).to_csv(output / "paired-metrics.csv", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/catalogue-comparison-wave-20261022.yaml"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    case = next(
        c
        for c in yaml.safe_load(args.config.read_text())["cases"]
        if c["run_id"] == args.run_id
    )
    run(case, args.input or Path(case["input"]), args.output)


if __name__ == "__main__":
    main()
