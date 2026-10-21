"""Independent finite-sample, grouping and joint-density adapter invariants."""

import importlib
import json

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.density_calibration import (
    calibrate_density_scores,
    evaluate_density_scores,
    fit_magnitude_tertiles,
    magnitude_groups,
    numpy_threshold,
    save_density_calibration,
)


def _new_api():
    module = importlib.import_module("torchregress.losses.conformal")
    if not hasattr(module, "ContrastiveFlowConformal"):
        pytest.skip("requires the new pinned torchregress density conformal APIs")


def _inputs():
    return {
        "fit_g": np.arange(90, dtype=float),
        "calibration_g": np.tile([10.0, 40.0, 70.0], 40),
        "test_g": np.tile([10.0, 40.0, 70.0], 4),
        "fit_ids": np.arange(90),
        "calibration_ids": np.arange(100, 220),
        "test_ids": np.arange(300, 312),
        "calibration_scores": np.arange(120, dtype=float) / 10,
        "test_scores": np.linspace(0, 13, 12),
        "provenance": {
            key: "a" * 64
            for key in ("checkpoint_sha256", "source_sha256", "scaler_sha256")
        },
    }


def test_finite_sample_rank_ties_and_small_calibration():
    assert numpy_threshold([3, 1, 1, 7], 0.4) == 3
    assert numpy_threshold([3, 1, 1, 7], 0.1) == np.inf
    with pytest.raises(ValueError, match="nonfinite"):
        numpy_threshold([1, np.nan], 0.1)


def test_fit_only_tertiles_and_explicit_boundary_rule():
    boundaries = fit_magnitude_tertiles([0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(boundaries, [2, 4])
    np.testing.assert_array_equal(
        magnitude_groups([-100, 2, 3, 4, 100], boundaries), [0, 1, 1, 2, 2]
    )
    with pytest.raises(ValueError, match="degenerate"):
        fit_magnitude_tertiles(np.ones(8))


def test_adapter_joint_score_and_native_constant_jacobian():
    target = torch.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, 1.0]], dtype=torch.float64)
    context = torch.zeros((2, 1))
    score = evaluate_density_scores(
        lambda c, y: -0.5 * (y * y).sum(dim=1), context, target, batch_size=1
    )
    np.testing.assert_array_equal(score, [7, 2.5])
    offset = np.log([1200, 1.2, 0.7]).sum()
    cal = np.arange(50, dtype=float)
    np.testing.assert_array_equal(
        score <= numpy_threshold(cal, 0.1),
        score + offset <= numpy_threshold(cal + offset, 0.1),
    )
    with pytest.raises(ValueError, match="joint density"):
        evaluate_density_scores(lambda c, y: y, context, target)


@pytest.mark.parametrize("api", ["density", "contrastive"])
def test_library_and_numpy_joint_memberships_and_saved_receipts(api, tmp_path):
    _new_api()
    summary, arrays = calibrate_density_scores(**_inputs(), api=api)
    assert summary["status"] == "density_level_set_arithmetic_passed"
    assert summary["counts"]["calibration"] == [40, 40, 40]
    assert len(summary["calibrations"]) == 4
    np.testing.assert_array_equal(arrays["calibration_weights"], np.ones(120))
    # Explicitly check Mondrian ranks independently of module quantile helper.
    for alpha in (0.1, 0.05):
        key = f"mondrian_alpha_{alpha:g}"
        expected = [
            np.sort(arrays["calibration_scores"][arrays["calibration_groups"] == g])[
                int(np.ceil(41 * (1 - alpha))) - 1
            ]
            for g in range(3)
        ]
        np.testing.assert_array_equal(arrays[f"{key}_thresholds"], expected)
    save_density_calibration(tmp_path, summary, arrays)
    stored = json.loads((tmp_path / "density-calibration-summary.json").read_text())
    assert stored["array_sha256"] == summary["array_sha256"]
    with np.load(
        tmp_path / "density-calibration-arrays.npz", allow_pickle=False
    ) as data:
        np.testing.assert_array_equal(data["fit_ids"], arrays["fit_ids"])
        assert "global_alpha_0.1_membership" in data


def test_disjoint_roles_and_missing_calibration_group_rejected():
    arguments = _inputs()
    arguments["test_ids"][0] = arguments["calibration_ids"][0]
    with pytest.raises(ValueError, match="disjoint"):
        calibrate_density_scores(**arguments)
    arguments = _inputs()
    arguments["calibration_g"][:] = 10
    with pytest.raises(ValueError, match="each FIT-defined"):
        calibrate_density_scores(**arguments)


def test_malformed_provenance_and_row_alignment_rejected():
    arguments = _inputs()
    arguments["provenance"]["source_sha256"] = "missing"
    with pytest.raises(ValueError, match="SHA256"):
        calibrate_density_scores(**arguments)
    arguments = _inputs()
    arguments["calibration_scores"] = arguments["calibration_scores"][:-1]
    with pytest.raises(ValueError, match="row aligned"):
        calibrate_density_scores(**arguments)


def test_small_groups_are_unbounded_and_test_scores_do_not_change_thresholds():
    _new_api()
    args = _inputs()
    for key in ("calibration_g", "calibration_ids", "calibration_scores"):
        args[key] = args[key][:6]
    summary, arrays = calibrate_density_scores(**args)
    assert all(summary["calibrations"][1]["unbounded"])
    assert arrays["mondrian_alpha_0.05_membership"].all()
    args["test_scores"] = np.full(12, 1e8)
    args["test_g"] = np.full(12, 1e6)
    _, alternate = calibrate_density_scores(**args)
    np.testing.assert_array_equal(arrays["g_boundaries"], alternate["g_boundaries"])
    np.testing.assert_array_equal(
        arrays["mondrian_alpha_0.1_thresholds"],
        alternate["mondrian_alpha_0.1_thresholds"],
    )
