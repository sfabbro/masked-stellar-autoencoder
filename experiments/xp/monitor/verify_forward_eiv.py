"""Independent full-cohort source/error and NumPy likelihood-grid replay."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from stellar_catalog_jobs import identity_split  # noqa: E402

from masked_stellar_autoencoder.models.checkpoint_load import (
    torch_load_trusted,  # noqa: E402
)
from masked_stellar_autoencoder.training.forward_eiv import (  # noqa: E402
    IntegratedForward,
    decoder,
)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(directory, cohort, *, allow_smoke=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert not manifest["smoke"] or allow_smoke
    for name, record in json.loads(
        (directory / "artifact-sha256.json").read_text()
    ).items():
        assert sha(directory / name) == record["sha256"], name
    paired = np.load(directory / "paired-cohort.npz", allow_pickle=False)
    saved = np.load(directory / "predictions.npz", allow_pickle=False)
    center, scale = float(paired["label_center"]), float(paired["label_scale"])
    with h5py.File(cohort) as source:
        raw = source["raw_layers/native/measurements"][:]
        feature_row = source["layer_index/native/feature_row"][:]
        ids = source["features/source_id"][:]
        np.testing.assert_array_equal(raw["source_id"], ids[feature_row])
        available = source["features/feature_available"][:][feature_row]
        partition = source["features/benchmark_partition"][:][feature_row]
        split = identity_split(raw["source_id"])
        fit = raw[available & (partition == 0) & (split == 0)]
        test = raw[available & (split == 2)]
        assert len(fit) == 19859 and len(test) == 100620
        for role, measurements in [("fit", fit), ("test", test)]:
            if manifest["smoke"]:
                measurements = measurements[
                    np.isin(
                        measurements["source_id"],
                        paired[role + "_measurements"]["source_id"],
                    )
                ]
            expected, stored = (
                a[np.argsort(a["source_id"])]
                for a in [measurements, paired[role + "_measurements"]]
            )
            for field in expected.dtype.names:
                assert expected[field].tobytes() == stored[field].tobytes(), field
            ordered = np.argsort(ids)
            rows = ordered[
                np.searchsorted(
                    ids[ordered], paired[role + "_measurements"]["source_id"]
                )
            ]
            columns = paired["xp_columns"]
            values = source["features/values"][:][rows].astype("f8")
            errors = source["features/reported_feature_errors"][:][rows].astype("f8")
            labels = json.loads(source.attrs["feature_columns_json"])
            g = values[:, labels.index("G")]
            expected_sigma = (
                errors[:, columns]
                / (10 ** ((8.5 - g[:, None]) / 2.5))
                / paired["scaler_scale"][columns]
            )
            expected_x = (
                values[:, columns] - paired["scaler_center"][columns]
            ) / paired["scaler_scale"][columns]
            np.testing.assert_allclose(
                expected_sigma, paired[role + "_sigma"], rtol=1e-12, atol=1e-12
            )
            np.testing.assert_allclose(
                expected_x, paired[role + "_x"], rtol=5e-6, atol=5e-6
            )
        np.testing.assert_allclose(
            paired["fit_label_standardized"],
            (paired["fit_measurements"]["value"] - center) / scale,
        )
        np.testing.assert_allclose(
            paired["fit_label_sigma_standardized"],
            paired["fit_measurements"]["uncertainty"] / scale,
        )
    x, sx = paired["test_x"].astype("f8"), paired["test_sigma"].astype("f8")
    grid = paired["grid_standardized"].astype("f8")
    report, replay = {}, {}
    torch.set_num_threads(1)
    for method in manifest["policy"]["methods"]:
        state = torch_load_trusted(
            directory / (method + ".pth"), map_location="cpu", weights_only=False
        )
        width = x.shape[1]
        if method == "direct_mlp":
            model = nn.Sequential(
                nn.Linear(width, 64),
                nn.SiLU(),
                nn.Linear(64, 64),
                nn.SiLU(),
                nn.Linear(64, 1),
            )
            model.load_state_dict(state["state"], strict=True)
            with torch.no_grad():
                values = torch.cat(
                    [
                        model(torch.tensor(b, dtype=torch.float32))
                        for b in np.array_split(x, max(1, len(x) // 256))
                    ]
                ).numpy()
            np.testing.assert_allclose(
                values, saved[method + "_raw"], rtol=5e-5, atol=5e-5
            )
            report[method] = dict(
                max_abs=float(np.max(np.abs(values - saved[method + "_raw"])))
            )
            replay[method] = values[:, 0] * scale + center
            continue
        if method == "forward_plugin":
            model = decoder(width)
            model.load_state_dict(state["decoder"], strict=True)
            variance = state["intrinsic_variance"].numpy().astype("f8")
            log_prior = np.zeros_like(grid)
        else:
            whole = IntegratedForward(width, flow=method == "flow_eb")
            whole.load_state_dict(state["state"], strict=True)
            model = whole.decoder
            variance = whole.intrinsic_variance.detach().numpy().astype("f8")
            with torch.no_grad():
                log_prior = (
                    np.zeros_like(grid)
                    if whole.prior is None
                    else whole.prior()
                    .log_prob(torch.tensor(grid[:, None], dtype=torch.float32))
                    .numpy()
                    .astype("f8")
                )
        with torch.no_grad():
            mean = (
                model(torch.tensor(grid[:, None], dtype=torch.float32))
                .numpy()
                .astype("f8")
            )
        weights = np.zeros_like(grid)
        weights[[0, -1]] = -np.log(2)
        results = []
        for start in range(0, len(x), 128):
            # Explicit NumPy residual Gaussian: independent of production's matrix expansion.
            v = sx[start : start + 128] ** 2 + variance
            loglike = -0.5 * (
                (
                    (x[start : start + 128, None, :] - mean[None, :, :]) ** 2
                    / v[:, None, :]
                )
                + np.log(2 * np.pi * v[:, None, :])
            ).sum(-1)
            logs = loglike + log_prior + weights
            probability = np.exp(logs - logs.max(-1, keepdims=True))
            probability /= probability.sum(-1, keepdims=True)
            expectation = probability @ grid
            std = np.sqrt(
                (probability * (grid[None, :] - expectation[:, None]) ** 2).sum(-1)
            )
            boundary = probability[:, :4].sum(-1) + probability[:, -4:].sum(-1)
            results.append(
                np.stack([grid[loglike.argmax(-1)], expectation, std, boundary], -1)
            )
        values = np.concatenate(results)
        # Continuous posterior moments must pass the unchanged numerical gate.
        np.testing.assert_allclose(
            values[:, 1:], saved[method + "_raw"][:, 1:], rtol=5e-5, atol=5e-5
        )
        mismatch = np.flatnonzero(values[:, 0] != saved[method + "_raw"][:, 0])
        report[method] = dict(
            moment_max_abs=float(
                np.max(np.abs(values[:, 1:] - saved[method + "_raw"][:, 1:]))
            ),
            grid_mle_disagree_rows=mismatch.tolist(),
            exact_grid_mle_verified=len(mismatch) == 0,
        )
        replay[method + "_grid_mle"] = values[:, 0] * scale + center
        replay[method + "_posterior_mean"] = values[:, 1] * scale + center
        if len(mismatch):
            raise ValueError(
                method + " grid MLE differs; preserve failure, do not relax tolerance"
            )
    np.savez_compressed(directory / "independent-cpu-predictions.npz", **replay)
    report = dict(
        status="smoke_passed" if manifest["smoke"] else "passed",
        rows=len(x),
        source_and_error_fields_verified=True,
        methods=report,
    )
    (directory / "independent-verification.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    progress = args.directory / "independent-replay-progress.json"
    progress.write_text(json.dumps(dict(status="running")) + "\n")
    try:
        report = verify(args.directory, args.cohort, allow_smoke=args.allow_smoke)
        progress.write_text(json.dumps(dict(status=report["status"])) + "\n")
        print(json.dumps(report, indent=2))
    except Exception as error:
        progress.write_text(json.dumps(dict(status="failed", error=repr(error))) + "\n")
        (args.directory / "independent-verification-failure.json").write_text(
            json.dumps(dict(status="failed", error=repr(error)), indent=2) + "\n"
        )
        raise
