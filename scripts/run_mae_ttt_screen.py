"""Evaluation-only MAE-style tabular TTT on a fixed selected native checkpoint.

Policies are fixed before using the disjoint TEST metal-poor evaluation stratum.
Labels never enter adaptation, masking, optimizer choice or parameter updates.
This is not a reproduction of image MAE pretraining or its frozen linear probe.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import run_multilabel_screen as native
import torch
import yaml

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted
from masked_stellar_autoencoder.training.input_measurement import xp_indices
from masked_stellar_autoencoder.training.mae_test_time import adapt_masked_mse
from masked_stellar_autoencoder.training.multilabel_experiments import (
    MultiLabelHead,
    hash_selected,
)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def select_evaluation_rows(data, predictions, cfg, smoke=False):
    """Hash-global plus separately flagged metal-poor evaluation union.

    Stratum labels determine only the explicitly separate evaluation sample;
    the policy and all optimizer/mask choices were frozen in the input config.
    Never treat the unweighted union as a representative global estimate.
    """
    xp = xp_indices(cfg["feature_columns"])
    test = predictions["test_rows"]
    cal = predictions["calibration_rows"]
    eligible = np.asarray(data["visible"])[:, xp].sum(1) >= 2
    test = test[eligible[test]]
    cal = cal[eligible[cal]]
    global_rows = test[
        hash_selected(data["source_id"][test], 1 if smoke else cfg["global_objects"])
    ]
    metal_rows = test[data["value"][test, 2] < -2]
    if smoke:
        metal_rows = metal_rows[:1]
    calibration_rows = cal[
        hash_selected(
            data["source_id"][cal],
            2 if smoke else cfg["calibration_objects"],
            seed=1043,
        )
    ]
    union = np.unique(np.r_[global_rows, metal_rows])
    if np.intersect1d(
        data["source_id"][union], data["source_id"][calibration_rows]
    ).size:
        raise ValueError("TTT calibration/test source overlap")
    return union, global_rows, metal_rows, calibration_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", default="configs/mae-ttt-screen-20261023.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--paired")
    parser.add_argument("--predictions")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    arm = next(a for a in cfg["arms"] if a["run_id"] == args.run_id)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "preparing",
        "run_id": args.run_id,
        "policy": cfg,
        "arm": arm,
        "smoke": args.smoke,
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "training_presentations": 0,
        "training_optimizer_updates": 0,
        "evaluation_only": True,
        "paper": "https://arxiv.org/abs/2209.07522",
        "limitations": [
            "MSA tabular XP masking, not image patches",
            "jointly trained Gaussian head, not separately fitted frozen linear probe",
            "native reported labels not physical-truth validation",
            "test metal-poor labels used only for separately reported fixed-policy evaluation stratum",
        ],
    }
    native.atomic(args.output / "manifest.json", manifest)
    try:
        torch.set_num_threads(4)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda" and not args.smoke:
            raise ValueError("Production TTT requires actual CUDA")
        direct = json.loads(
            importlib.metadata.distribution("torchregress").read_text("direct_url.json")
            or "{}"
        )
        if (
            not args.smoke
            and direct.get("vcs_info", {}).get("commit_id") != cfg["library_revision"]
        ):
            raise ValueError("TTT library revision differs from published pin")
        paths = {
            k: Path(getattr(args, k) or cfg[k])
            for k in ("checkpoint", "paired", "predictions")
        }
        for k, p in paths.items():
            if sha(p) != cfg[k + "_sha256"]:
                raise ValueError(f"Immutable {k} checksum differs")
        disk = torch_load_trusted(
            paths["checkpoint"], map_location="cpu", weights_only=False
        )
        if (
            disk["parent_sha256"] != cfg["parent_sha256"]
            or disk["source_sha"] != cfg["checkpoint_source_sha"]
        ):
            raise ValueError("TTT checkpoint training/parent identity differs")
        if (
            disk["completed_presentations"] != 99000000
            or disk["completed_updates"] != 3025
        ):
            raise ValueError(
                "Expected actual tune-selected99M state, not final270M state"
            )
        with np.load(paths["paired"], allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        with np.load(paths["predictions"], allow_pickle=False) as f:
            predictions = {k: f[k] for k in f.files}
        cfg["feature_columns"] = disk["feature_columns"]
        for role in ("test", "calibration"):
            np.testing.assert_array_equal(
                data["source_id"][predictions[role + "_rows"]],
                predictions[role + "_source_id"],
            )
        np.testing.assert_array_equal(data["label_center"], disk["label_center"])
        np.testing.assert_array_equal(data["label_scale"], disk["label_scale"])
        model = native.probe.load_model(
            yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text()),
            disk,
            device,
        )
        head = MultiLabelHead(
            "gaussian", target_dim=3, noise_aware=True, mc_samples=32
        ).to(device)
        head.load_state_dict(disk["head_state_dict"], strict=True)
        rows, global_rows, metal_rows, cal = select_evaluation_rows(
            data, predictions, cfg, args.smoke
        )
        manifest.update(
            status="running",
            device=str(device),
            gpu_name=torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
            checkpoint_sha256=cfg["checkpoint_sha256"],
            paired_sha256=cfg["paired_sha256"],
            predictions_sha256=cfg["predictions_sha256"],
            selection={
                "global_hash_count": len(global_rows),
                "separate_metal_poor_count": len(metal_rows),
                "deduplicated_test_count": len(rows),
                "calibration_count": len(cal),
                "pooled_population_claim_forbidden": True,
            },
            original_density_failure_retained=True,
            adaptation_labels=False,
            head_frozen=True,
            optimizer_state_reset_per_object=True,
        )
        native.atomic(args.output / "manifest.json", manifest)
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        hb = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        xp = xp_indices(cfg["feature_columns"]).tolist()
        output = {}
        diagnostics = {}
        for role, selected in [("test", rows), ("calibration", cal)]:
            points = []
            variances = []
            traces = []
            for i, row in enumerate(selected):
                p, v, trace = adapt_masked_mse(
                    model,
                    head,
                    torch.tensor(data["input"][row : row + 1], device=device),
                    torch.tensor(data["visible"][row : row + 1], device=device),
                    xp,
                    int(data["source_id"][row]),
                    scope=arm["scope"],
                    steps=arm["steps"],
                    lr=arm["lr"],
                    momentum=0.9,
                    mask_fraction=arm["mask_fraction"],
                )
                points.append(p.cpu().numpy()[0])
                variances.append(v.cpu().numpy()[0])
                traces.append(trace)
                if (i + 1) % 16 == 0 or i + 1 == len(selected):
                    native.atomic(
                        args.output / "progress.json",
                        {
                            "role": role,
                            "objects": i + 1,
                            "total": len(selected),
                            "source_id": int(data["source_id"][row]),
                        },
                    )
            output[role + "_rows"] = selected
            output[role + "_source_id"] = data["source_id"][selected]
            output[role + "_mean"] = np.stack(points)
            output[role + "_variance"] = np.stack(variances)
            positions = {
                int(row): i for i, row in enumerate(predictions[role + "_rows"])
            }
            output[role + "_clean_mean"] = predictions[role + "_mean"][
                [positions[int(row)] for row in selected]
            ]
            diagnostics[role] = traces
        for k, v in model.state_dict().items():
            if not torch.equal(v.detach().cpu(), before[k]):
                raise AssertionError(f"Model reset failed {k}")
        for k, v in head.state_dict().items():
            if not torch.equal(v.detach().cpu(), hb[k]):
                raise AssertionError(f"Fixed head changed {k}")
        output.update(
            test_global_hash_membership=np.isin(rows, global_rows),
            test_metal_poor_membership=np.isin(rows, metal_rows),
            label_center=data["label_center"],
            label_scale=data["label_scale"],
            test_target=data["value"][rows],
            calibration_target=data["value"][cal],
        )
        np.savez_compressed(args.output / "ttt-predictions.npz", **output)
        cp = native.calibrate_and_save(
            args.output,
            data,
            rows,
            cal,
            output["test_mean"],
            output["calibration_mean"],
            prefix="adapted-",
        )
        baseline_cp = native.calibrate_and_save(
            args.output,
            data,
            rows,
            cal,
            output["test_clean_mean"],
            output["calibration_clean_mean"],
            prefix="clean-",
        )
        native.atomic(args.output / "diagnostics.json", diagnostics)
        strata = {}
        with np.load(args.output / "adapted-conformal.npz") as intervals:
            for name, membership in (
                ("global_hash", output["test_global_hash_membership"]),
                ("metal_poor_evaluation_stratum", output["test_metal_poor_membership"]),
            ):
                y = (
                    np.asarray(data["value"][rows], dtype="f8") - data["label_center"]
                ) / data["label_scale"]
                error = (output["test_mean"] - y) * data["label_scale"]
                clean_error = (output["test_clean_mean"] - y) * data["label_scale"]
                covered = (y >= intervals["marginal_lower"]) & (
                    y <= intervals["marginal_upper"]
                )
                strata[name] = {
                    "count": int(membership.sum()),
                    "mae_native": np.abs(error[membership]).mean(0).tolist()
                    if membership.any()
                    else None,
                    "clean_mae_native": np.abs(clean_error[membership]).mean(0).tolist()
                    if membership.any()
                    else None,
                    "marginal_coverage": covered[membership].mean(0).tolist()
                    if membership.any()
                    else None,
                }
        manifest.update(
            status="complete",
            evaluation_strata=strata,
            all_model_state_reset_exact=True,
            all_head_state_unchanged_exact=True,
            union_conformal_diagnostic_not_population=cp,
            matched_clean_conformal=baseline_cp,
            baseline_reused_no_model_forward=True,
            adaptation_optimizer_steps=arm["steps"] * (len(rows) + len(cal)),
            output_sha256=sha(args.output / "ttt-predictions.npz"),
        )
        native.atomic(args.output / "manifest.json", manifest)
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        native.atomic(args.output / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
