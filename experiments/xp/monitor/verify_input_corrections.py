"""Independent NumPy correction algebra plus complete checkpoint CPU replay.

Never trains. Uses original raw HDF5 fields and fit-only moments, rather than
production calibration objects. SIMEX requires every contributing checkpoint.
"""

import argparse
import gc
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from verify_prediction_comparison import ROOT, identity_split, sha, verify


def correction_reference(fit, sigma, method, values, own_sigma):
    """Univariate measurement-error conditional means, from fit-only moments."""
    fit = np.asarray(fit, dtype="f8")
    mean = fit.mean(axis=0)
    variance = np.var(fit, axis=0, ddof=1)
    signal = np.maximum(variance - np.asarray(sigma) ** 2, 1e-6 * variance)
    if method == "rc_shared":
        noise = np.asarray(sigma) ** 2
    elif method == "rc_per_star":
        # torchregress6311's numerical floor for posterior standard deviations;
        # this is a relative algebra regularizer, not a physical error floor.
        noise = np.maximum(own_sigma, 1e-6 * np.sqrt(signal)) ** 2
    else:
        raise ValueError("Unknown regression calibration method")
    gain = signal / (signal + noise)
    return mean + gain * (np.asarray(values, dtype="f8") - mean)


def independent_extrapolate(lambdas, values):
    """Two-column least squares, independent of production's centered formula."""
    lam = np.asarray(lambdas, dtype="f8")
    y = np.asarray(values, dtype="f8")
    result = np.zeros(y.shape[1])
    best = np.full(y.shape[1], np.inf)
    anchor = y[0]
    shifted = y - anchor
    for c in np.linspace(1.05, 8.0, 400):
        design = np.column_stack([np.ones(len(lam)), 1 / (c + lam)])
        coefficients = np.linalg.lstsq(design, shifted, rcond=None)[0]
        residual = np.sum((design @ coefficients - shifted) ** 2, axis=0)
        take = residual < best
        best[take] = residual[take]
        result[take] = coefficients[0, take] + coefficients[1, take] / (c - 1)
    return result + anchor


class InputReference:
    def __init__(self, cohort, paired):
        with h5py.File(cohort) as f:
            ids = f["features/source_id"][:]
            split = identity_split(ids)
            np.testing.assert_array_equal(split, f["features/label_split"][:])
            partition = f["features/benchmark_partition"][:]
            canonical_rows = f["canonical/feature_row"][:]
            fit_rows = canonical_rows[
                (split[canonical_rows] == 0) & (partition[canonical_rows] == 0)
            ]
            tests = []
            available = f["features/feature_available"][:]
            for layer in ["native", "PIGS", "Li2025_CaT_silver"]:
                rows = f["layer_index/" + layer + "/feature_row"][:]
                tests.append(rows[available[rows] & (split[rows] == 2)])
            self.used = np.unique(np.concatenate([fit_rows, *tests]))
            assert len(self.used) == 121098
            names = json.loads(f.attrs["feature_columns_json"])
            self.columns = np.array(
                [
                    i
                    for i, n in enumerate(names)
                    if n.startswith(("bp_", "rp_")) and n.split("_")[-1].isdigit()
                ]
            )
            assert len(self.columns) == 110
            values = f["features/values"][:][self.used]
            errors = f["features/reported_feature_errors"][:][self.used]
            scale = f["reference_scaler/scale"][:]
            center = f["reference_scaler/center"][:]
            self.xp = ((values - center) / scale).astype("f4")[:, self.columns]
            factor = 10.0 ** ((8.5 - values[:, names.index("G")]) / 2.5)
            self.sigma = errors[:, self.columns] / factor[:, None] / scale[self.columns]
            assert np.isfinite(self.sigma).all() and (self.sigma > 0).all()
            order = np.argsort(ids)
            fit_global = order[np.searchsorted(ids[order], paired["fit_source_id"])]
            np.testing.assert_array_equal(ids[fit_global], paired["fit_source_id"])
            self.fit = np.searchsorted(self.used, fit_global)
            np.testing.assert_array_equal(self.used[self.fit], fit_global)
            self.rms = np.sqrt(np.mean(self.sigma[self.fit] ** 2, axis=0))
        self.method = None
        self.noise = None

    def select_replica(self, method, record=None):
        self.method = method
        self.noise = None
        if method == "simex" and record["lambda_value"] > 0:
            lam = record["lambda_value"]
            index = [0.0, 0.5, 1.0, 1.5, 2.0].index(lam)
            seed = 20000 + index * 100 + record["simulation"]
            assert seed == record["perturbation_seed"]
            # Same CPU PRNG distribution is needed for byte-associated draws;
            # noise scaling/membership and all correction algebra are rederived.
            rng = torch.Generator().manual_seed(seed)
            noise = torch.randn(
                (len(self.used), 110), generator=rng, dtype=torch.float64
            )
            factor = torch.linalg.cholesky(torch.diag(torch.as_tensor(self.rms) ** 2))
            self.noise = (noise @ factor.T).numpy() * np.sqrt(lam)

    def __call__(self, kind, rows, safe):
        positions = np.searchsorted(self.used, rows)
        np.testing.assert_array_equal(self.used[positions], rows)
        out = safe.copy()
        if self.method == "simex":
            if self.noise is not None:
                out[:, self.columns] = (
                    self.xp[positions].astype("f8") + self.noise[positions]
                )
        else:
            out[:, self.columns] = correction_reference(
                self.xp[self.fit],
                self.rms,
                self.method,
                self.xp[positions],
                self.sigma[positions],
            )
        other = np.setdiff1d(np.arange(safe.shape[1]), self.columns)
        np.testing.assert_array_equal(out[:, other], safe[:, other])
        return out


def run(directory, cohort, parent):
    root = json.loads((directory / "manifest.json").read_text())
    if root["status"] != "complete" or root["smoke"]:
        raise ValueError("Require completed production outputs")
    assert root["xp_error_source_audit"]["status"] == "verified"
    with np.load(directory / "paired-cohort.npz", allow_pickle=False) as paired:
        reference = InputReference(cohort, paired)
    method = root["arm"]["input_correction"]["method"]
    if method != "simex":
        reference.select_replica(method)
        return verify(directory, cohort, parent, input_reference=reference)
    index = json.loads((directory / "replicas.json").read_text())
    assert len(index["replicas"]) == root["replica_count"] == 17
    records = index["replicas"]
    assert [(r["lambda_value"], r["simulation"]) for r in records] == [
        (0.0, 0),
        *[(level, sim) for level in [0.5, 1.0, 1.5, 2.0] for sim in range(4)],
    ]
    assert (
        sum(r["presentations"] for r in records) == root["presentations"] == 170000000
    )
    assert (
        sum(r["optimizer_updates"] for r in records)
        == root["optimizer_updates"]
        == 5202
    )
    for field in [
        "parent_sha256",
        "source_sha",
        "row_sha256",
        "mask_sha256",
        "supervised_stream_sha256",
    ]:
        assert len({r[field] for r in records}) == 1
    reports, points, lambdas = [], [], []
    for number, record in enumerate(index["replicas"]):
        path = directory / record["directory"]
        assert path.parent == directory and record["index"] == number
        assert sha(path / "replica-manifest.json") == record["manifest_sha256"]
        for name, expected in record["files"].items():
            file = path / name
            assert file.parent == path and file.stat().st_size == expected["bytes"]
            assert sha(file) == expected["sha256"]
        reference.select_replica(method, record)
        per = {**root, **record, "run_id": root["run_id"] + "/" + record["directory"]}
        report_path = path / "independent-verification.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            assert (
                report["status"] == "passed"
                and report["corrected_inputs_independently_derived"]
            )
            assert report["final_checkpoint_sha256"] == sha(path / "checkpoint.pth")
        else:
            report = verify(
                path, cohort, parent, input_reference=reference, manifest_override=per
            )
        reports.append(report)
        with np.load(path / "predictions.npz", allow_pickle=False) as actual:
            points.append(actual["prediction"])
        lambdas.append(record["lambda_value"])
        gc.collect()
    order = [0.0, 0.5, 1.0, 1.5, 2.0]
    means = np.stack(
        [
            np.mean(
                [
                    p.astype("f8")
                    for p, level_value in zip(points, lambdas, strict=True)
                    if level_value == level
                ],
                axis=0,
            )
            for level in order
        ]
    )
    result = independent_extrapolate(order, means)
    with np.load(directory / "predictions.npz", allow_pickle=False) as assembled:
        np.testing.assert_array_equal(
            np.stack(points), assembled["per_replica_prediction"]
        )
        np.testing.assert_array_equal(means, assembled["lambda_mean_prediction"])
        np.testing.assert_allclose(
            result, assembled["prediction"], rtol=1e-8, atol=1e-8
        )
        maximum = float(np.max(np.abs(result - assembled["prediction"])))
    report = dict(
        status="passed",
        run_id=root["run_id"],
        source_sha=root["source_sha"],
        replicas=17,
        all_contributing_checkpoints_cpu_replayed=True,
        independent_numpy_extrapolation=True,
        extrapolation_max_absolute_difference=maximum,
        cpu_max_absolute_member_output_difference=max(
            r["cpu_max_absolute_member_output_difference"] for r in reports
        ),
        checkpoint_sha256s=[r["final_checkpoint_sha256"] for r in reports],
    )
    (directory / "independent-verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument(
        "--cohort",
        type=Path,
        default=ROOT
        / "experiments/xp/results-stellar-improvement/enriched_feature_cohort_v1/enriched-feature-cohort.h5",
    )
    p.add_argument(
        "--parent",
        type=Path,
        default=ROOT
        / "experiments/xp/results-stellar-improvement/prediction-comparison-e72/parent/checkpoint-72.pth",
    )
    args = p.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(run(args.directory, args.cohort, args.parent)))
