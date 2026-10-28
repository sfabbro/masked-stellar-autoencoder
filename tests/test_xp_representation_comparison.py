"""Fail closed when paired representation evaluations cease to be comparable."""

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "representation_comparison",
    Path(__file__).parents[1]
    / "experiments/xp/monitor/compare_representation_followup.py",
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def paired_group():
    truth = np.arange(20, dtype=np.float32)
    prediction = truth + 0.5
    metric = {"count": 20, "mae": 0.5}
    result = {
        "split_counts": [100000, 20000, 20],
        "split_source_id_sha256": ["train", "tune", "test"],
        "duplicate_source_ids_removed": 0,
        "fits_units_audit": {
            "matched_stars": 128,
            "chosen_units": "g_normalized",
            "candidates": {"g_normalized": {"passed": True}},
        },
        "targets": {},
    }
    arrays = {}
    for target in comparison.TARGETS:
        scaler = {"train_target_center": 4.0, "train_target_scale": 1.0}
        report = {"ridge": metric.copy(), "ridge_policy": scaler.copy(), "budgets": {}}
        arrays[target + "_truth"] = truth.copy()
        arrays[target + "_source_ids"] = np.arange(1, 21)
        arrays[target + "_ridge"] = prediction.copy()
        for budget in ("1000", "10000", "100000"):
            report["budgets"][budget] = {
                "ridge": metric.copy(),
                "head": metric.copy(),
                "finetuned_encoder_head": metric.copy(),
                "ridge_policy": scaler.copy(),
                "head_steps": 1000,
                "head_lr": 0.001,
                "encoder_frozen": True,
                "finetune_policy": {"steps": 200, "encoder_lr": 1e-6},
            }
            for method in ("head", "finetuned"):
                arrays[f"{target}_{method}_{budget}"] = prediction.copy()
        result["targets"][target] = report
    return {
        "manifest": {
            "arguments": dict.fromkeys(comparison.PROBE_ARGUMENTS, "matched"),
            "script_sha256": "same",
        },
        "results": result,
        "arrays": arrays,
    }


def test_changed_test_truth_cannot_be_reported_as_representation_gain():
    baseline = paired_group()
    comparison.verify_probe(baseline, copy.deepcopy(baseline))
    candidate = copy.deepcopy(baseline)
    candidate["arrays"]["teff_truth"][0] += 1
    with pytest.raises(AssertionError):
        comparison.verify_probe(candidate, baseline)


def test_changed_adaptation_budget_cannot_be_compared():
    baseline = paired_group()
    candidate = copy.deepcopy(baseline)
    candidate["results"]["targets"]["logg"]["budgets"]["1000"]["finetune_policy"][
        "steps"
    ] = 201
    with pytest.raises(ValueError, match="Equal supervised adaptation budget"):
        comparison.verify_probe(candidate, baseline)


def test_declared_metric_must_match_actual_paired_prediction_payload():
    baseline = paired_group()
    candidate = copy.deepcopy(baseline)
    candidate["results"]["targets"]["fe_h"]["ridge"]["mae"] = 0.25
    with pytest.raises(AssertionError):
        comparison.verify_probe(candidate, baseline)


def test_paired_improvement_has_correct_sign_and_win_count():
    truth = np.array([1.0, 2.0, 3.0])
    result = comparison.score_pairs(truth + 0.5, truth, truth + 1.0)
    assert result["mae_change_percent"] == -50.0
    assert result["paired_mae_difference"] == -0.5
    assert result["fraction_smaller_absolute_error"] == 1.0
