"""Small checks for paired references and deliberate policy differences."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "compare_followup", Path(__file__).with_name("compare_followup.py")
)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


def test_seed_pairing_and_count_weighted_groups():
    assert compare.reference_for("spectral_s43", {"seed": 43}) == "control_s43"
    assert (
        compare.reference_for("photometry_spectral_s44", {"seed": 44}) == "control_s44"
    )
    assert compare.reference_for("spectral_lr05_s42", {"seed": 42}) == "spectral_s42"
    assert (
        compare.weighted([{"count": 1, "mae": 4}, {"count": 3, "mae": 0}])["mae"] == 1
    )
    assert compare.delta(9, 10) == pytest.approx(-10)


def test_zero_row_spectra_have_no_statistic_and_nonempty_must_supply_it():
    result = compare.compare_spectral({"rows": 0}, {"rows": 0})
    assert result["integrated_absolute_fraction_mean"] is None
    assert result["integrated_absolute_fraction_median_delta_percent"] is None
    with pytest.raises(AssertionError):
        compare.compare_spectral({"rows": 0}, {"rows": 1})
    with pytest.raises(KeyError):
        compare.compare_spectral({"rows": 1}, {"rows": 1})


def test_policy_verifier_allows_declared_reset_lr_warmup_only():
    baseline = {"source_learning_rates": [0.004]}
    policy = {
        "seed": 43,
        "independent_rng": True,
        "reset_moments": True,
        "warmup_steps": 32,
        "source_learning_rates": [0.004],
        "learning_rates": [0.002],
    }
    settings = {
        "seed": 43,
        "reset_optimizer": True,
        "warmup_steps": 32,
        "lr_factor": 0.5,
    }
    compare.verify_policy(policy, settings, baseline)
    with pytest.raises(AssertionError):
        compare.verify_policy(policy, {**settings, "seed": 42}, baseline)
    with pytest.raises(AssertionError):
        compare.verify_policy(policy, {**settings, "warmup_steps": 0}, baseline)


def test_initial_probe_verifier_checks_predictions_and_actual_masks(tmp_path):
    ref, actual = tmp_path / "reference.npz", tmp_path / "actual.npz"
    np.savez(
        ref, common_prediction=np.ones((2, 3)), common_mask=np.zeros((2, 3), dtype=bool)
    )
    np.savez(
        actual,
        common_prediction=np.ones((2, 3)),
        common_mask=np.zeros((2, 3), dtype=bool),
    )
    compare.verify_initial(actual, ref)
    np.savez(
        actual,
        common_prediction=np.ones((2, 3)),
        common_mask=np.ones((2, 3), dtype=bool),
    )
    with pytest.raises(AssertionError):
        compare.verify_initial(actual, ref)


def test_dropout_policy_verifier_retains_encoder_for_decoder_only_override():
    baseline = {"encoder.block.dp": 0.1, "decoder.block.dp": 0.1}
    compare.verify_dropout(
        {"encoder.block.dp": 0.1, "decoder.block.dp": 0},
        baseline,
        {"decoder_dropout": 0},
    )
    with pytest.raises(AssertionError):
        compare.verify_dropout(
            {"encoder.block.dp": 0, "decoder.block.dp": 0},
            baseline,
            {"decoder_dropout": 0},
        )
