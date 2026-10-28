"""Analytical tests for the independent point/CP and stream/decoder references."""

import importlib.util
from pathlib import Path

import numpy as np


def load_reference(name):
    path = (
        Path(__file__).resolve().parents[1] / "experiments/xp/monitor" / (name + ".py")
    )
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_point_head_and_finite_sample_cp_analytical_reference():
    load_reference("verify_multilabel_points").self_test()


def test_short_cohort_boundary_counters_are_not_update_times_batch():
    reference = load_reference("verify_multilabel_decoder_streams")
    assert reference.counters((5, 3), 4) == [4, 5, 8]
    assert reference.counters()[274:277] == [9000000, 9032768, 9065536]
    assert len(reference.counters()) == 306


def test_independent_validation_mask_keeps_whole_xp_blocks():
    reference = load_reference("verify_multilabel_decoder_streams")
    mask = reference.reference_mask((100, 3), ["g", "bp_1", "rp_1"])
    assert mask.dtype == bool
    np.testing.assert_array_equal(mask[:, 1], mask[:, 2])
    assert mask[:, 1].sum() == 50
    np.testing.assert_array_equal(
        mask, reference.reference_mask((100, 3), ["g", "bp_1", "rp_1"])
    )
