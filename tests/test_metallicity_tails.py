"""Falsifiable units, thresholds, signs and paired bootstrap checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "metallicity_tails",
    Path(__file__).parents[1] / "scripts/analysis/analyze_metallicity_tails.py",
)
tails = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tails)


def test_catalogue_threshold_is_strict_and_bins_partition_every_finite_star_once():
    y = np.array([-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5])
    masks = tails.strata(y)
    assert masks["FeH<-2"].sum() == 2
    assert masks["FeH<-2.5"].sum() == 1
    bins = [value for key, value in masks.items() if key.startswith("[")]
    np.testing.assert_array_equal(np.sum(bins, axis=0), np.ones(len(y)))
    assert masks["[-2.5,-2.0)"][1]


def test_bias_and_catastrophic_fraction_use_signed_dex_and_strict_point5_cut():
    result = tails.error_metrics(
        np.array([-1.5, -2.5, -2.6]), np.array([-2.0, -2.0, -2.0])
    )
    assert result["bias_dex"] == pytest.approx(-0.2)
    assert result["mae_dex"] == pytest.approx(1.6 / 3)
    assert result["catastrophic_fraction_abs_gt_0p5dex"] == pytest.approx(1 / 3)
    assert result["rmse_dex"] == pytest.approx(np.sqrt((0.25 + 0.25 + 0.36) / 3))


def test_paired_bootstrap_preserves_exact_common_error_cancellation():
    y = np.arange(100, dtype=np.float64)
    baseline = y + np.linspace(1, 3, len(y))
    candidate = baseline - 0.25
    result = tails.paired_bootstrap(candidate, y, baseline, seed=42)
    assert result["mae_difference_dex"] == pytest.approx(-0.25)
    np.testing.assert_allclose(result["ci95_dex"], [-0.25, -0.25])
    again = tails.paired_bootstrap(candidate, y, baseline, seed=42)
    assert result == again


def test_tiny_tail_cannot_produce_reassuring_bootstrap_interval():
    result = tails.paired_bootstrap(
        np.array([-2.0] * 5), np.array([-3.0] * 5), np.array([-1.0] * 5), seed=42
    )
    assert result["count"] == 5
    assert result["ci95_dex"] is None
    assert not result["adequate_count_for_interval"]


def test_selection_completeness_and_purity_have_distinct_denominators():
    result = tails.selection_metrics(
        np.array([-2.2, -1.5, -2.2, -1.0]), np.array([-3.0, -2.4, -1.0, 0.0]), -2.0
    )
    assert (
        result["true_positive"],
        result["false_positive"],
        result["false_negative"],
        result["true_negative"],
    ) == (1, 1, 1, 1)
    assert result["completeness"] == result["purity"] == 0.5
    assert not result["adequate_counts"]
    empty = tails.selection_metrics(np.array([-1.0]), np.array([-1.5]), -2.0)
    assert empty["completeness"] is None and empty["purity"] is None


def test_nonfinite_predictions_fail_instead_of_dropping_stars():
    with pytest.raises(ValueError, match="Nonfinite"):
        tails.error_metrics(np.array([np.nan]), np.array([-2.0]))
