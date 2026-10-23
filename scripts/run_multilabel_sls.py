"""Native SLS/SLSConformal screen using the audited matched reconstruction loop.

SLS estimates joint regions, not normalized densities. The diagnostic point is
its geometric frontier center, not a conditional mean. TTT is disabled.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import run_multilabel_screen as native
import torch
import yaml

from masked_stellar_autoencoder.training.sls_experiments import (
    SLSHead,
    finite_sample_threshold,
    region_log_volume,
)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="configs/multilabel-sls-20261023.yaml")
    known, _ = parser.parse_known_args()
    if "--config" not in sys.argv and not any(
        a.startswith("--config=") for a in sys.argv
    ):
        sys.argv.extend(["--config", known.config])
    cfg = yaml.safe_load(Path(known.config).read_text())
    if cfg.get("target_names") != ["teff", "logg", "fe_h"]:
        raise ValueError(
            "Native SLS screen requires declared Teff/logg/FeH target order"
        )
    arm = next(a for a in cfg["arms"] if a["run_id"] == known.run_id)
    if arm.get("ttt") or arm.get("noise_aware") or cfg.get("density_conformal"):
        raise ValueError(
            "SLS region screen does not implement TTT or measurement convolution"
        )
    original_prediction = native.predictions
    original_calibrate = native.calibrate_and_save
    role_results = {}
    calibration_context = {}
    head_reference = {}

    def factory(kind, **kwargs):
        if kind != "sls":
            raise ValueError("SLS entrypoint accepts only SLS arms")
        head = SLSHead(mode=arm["frontier_mode"], **kwargs)
        head_reference["head"] = head
        return head

    def predictions(model, head, data, rows, device, batch=128):
        result = original_prediction(model, head, data, rows, device, batch)
        encoder = (
            copy.deepcopy(model.encoder).to(device=device, dtype=torch.float64).eval()
        )
        evaluation = copy.deepcopy(head).to(device=device, dtype=torch.float64).eval()
        values = []
        latent_values = []
        is_cal = np.isin(rows, data["roles"]["calibration"]).all()
        with torch.no_grad():
            for start in range(0, len(rows), batch):
                selected = rows[start : start + batch]
                x = torch.as_tensor(
                    data["input"][selected], dtype=torch.float64, device=device
                )
                y = torch.as_tensor(
                    (data["value"][selected] - data["label_center"])
                    / data["label_scale"],
                    dtype=torch.float64,
                    device=device,
                )
                latent = encoder(x)
                if is_cal:
                    latent_values.append(latent.cpu())
                g, q, logp = evaluation.region_values(latent, y)
                values.append(torch.stack((g, q, logp), -1).cpu().numpy())
        a = np.concatenate(values)
        if not np.isfinite(a).all() or (a[:, :2] < 0).any():
            raise FloatingPointError("Nonfinite or negative SLS frontier/quantile")
        result.update(
            frontier=a[:, 0],
            frontier_quantile=a[:, 1],
            log_precision_root=a[:, 2],
            sls_score=a[:, 0] / a[:, 1],
        )
        role_results[tuple(rows)] = result
        if is_cal:
            calibration_context[tuple(rows)] = torch.cat(latent_values)
        return result

    def calibrate(output, data, test, cal, means_test, means_cal, prefix=""):
        residual = original_calibrate(
            output, data, test, cal, means_test, means_cal, prefix
        )
        a, c = role_results[tuple(test)], role_results[tuple(cal)]
        if np.intersect1d(data["source_id"][test], data["source_id"][cal]).size:
            raise ValueError("SLS calibration/test source overlap")
        arrays = {
            "test_source_id": data["source_id"][test],
            "calibration_source_id": data["source_id"][cal],
            "test_score": a["sls_score"],
            "calibration_score": c["sls_score"],
        }
        summary = {
            "estimand": "joint SLS region, not normalized PDF",
            "raw_tau": 0.9,
            "raw_joint_coverage": float(np.mean(a["sls_score"] <= 1)),
            "frontier_center_is_not_conditional_mean": True,
            "schedule": "explicit persisted supervised criterion calls, warmup50",
            "residual_cp": residual,
        }
        from torchregress.losses import SLSConformal

        sls = copy.deepcopy(head_reference["head"].sls).cpu().double().eval()
        calibration_y = torch.as_tensor(
            (data["value"][cal] - data["label_center"]) / data["label_scale"],
            dtype=torch.float64,
        )
        for alpha in (0.1, 0.05):
            threshold = finite_sample_threshold(c["sls_score"], alpha)
            library = SLSConformal(sls, alpha=alpha)
            library.calibrate(calibration_context[tuple(cal)], calibration_y)
            np.testing.assert_allclose(
                float(library.q_hat), threshold, rtol=1e-9, atol=1e-9
            )
            membership = a["sls_score"] <= threshold
            logvolume = region_log_volume(
                a["frontier_quantile"],
                a["log_precision_root"],
                threshold,
                data["label_scale"],
            )
            name = str(round(100 * (1 - alpha)))
            arrays.update(
                {
                    "threshold_" + name: np.array(threshold),
                    "membership_" + name: membership,
                    "native_log_volume_" + name: logvolume,
                }
            )
            summary[name] = {
                "threshold": float(threshold) if np.isfinite(threshold) else None,
                "unbounded_finite_sample_region": not np.isfinite(threshold),
                "joint_coverage": float(membership.mean()),
                "mean_native_log_volume": float(logvolume.mean())
                if np.isfinite(logvolume).all()
                else None,
                "volume_unit": "K*dex*dex",
                "method": "SLSConformal G/q finite-sample calibration",
            }
        np.savez_compressed(output / "sls-conformal.npz", **arrays)
        native.atomic(output / "sls-conformal.json", summary)
        return summary

    original_atomic = native.atomic

    def atomic(path, value):
        if path.name == "manifest.json":
            value["scope"] = (
                "Native3 joint SLS level sets; geometric frontier center diagnostic; no normalized PDF"
            )
            value["density_available"] = False
            value["noise_likelihood_evaluation"] = (
                "No normalized PDF or quoted-label-error convolution; SLS joint observed-label regions only"
            )
            value["density_contract"] = {
                "available": False,
                "reason": "SLS is a region estimator, not a normalized density",
            }
            value["sls_frontier_mode"] = arm["frontier_mode"]
            value["sls_schedule_unit"] = (
                "supervised minibatch criterion calls including head warmup"
            )
        original_atomic(path, value)

    native.atomic = atomic
    native.MultiLabelHead = factory
    native.predictions = predictions
    native.calibrate_and_save = calibrate
    native._point_estimator = lambda _: (
        "inverse-flow geometric frontier center; NOT conditional mean"
    )
    native.main()


if __name__ == "__main__":
    main()
