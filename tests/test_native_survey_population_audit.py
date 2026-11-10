import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "native_population_audit", Path(__file__).parents[1] / "scripts/run_native_survey_population_audit.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_native_union_deduplicates_repeats_and_survey_overlap_without_parent_filtering():
    ids = np.arange(100000000000000001, 100000000000000006, dtype="i8")
    counts, missing, expanded = m.compare_sets({
        "desi": ids[[0, 0, 1]], "lrs": ids[[1, 2, 3]], "mrs": ids[[2, 4]],
    }, ids[:2])
    assert counts["unique_gaia_ids_by_source"] == {"desi": 2, "lrs": 3, "mrs": 2}
    assert counts["native_desi_lamost_union_unique_gaia_stars"] == 5
    assert counts["pairwise_unique_star_overlap"]["desi__lrs"] == 1
    assert counts["native_union_absent_from_candidate_parent"] == 3
    np.testing.assert_array_equal(missing, ids[2:])
    np.testing.assert_array_equal(expanded, ids)


def test_native_population_audit_rejects_rounded_float_ids_and_removes_invalid_ids():
    ids = np.array([100000000000000001, 100000000000000002], dtype="i8")
    with pytest.raises(ValueError, match="never float"):
        m.positive_unique(ids.astype(float))
    np.testing.assert_array_equal(m.positive_unique(np.r_[ids, -1, 0, ids]), ids)
