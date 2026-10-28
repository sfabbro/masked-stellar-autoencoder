import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_posterior_target_wave import role_indices
from run_posterior_target_wave import fit_normalization


def test_family_holdout_roles_are_disjoint_and_exact_int64():
    ids = np.arange(2**60, 2**60 + 400, dtype="i8")
    masks = {
        "strict_fit": np.arange(400) < 100,
        "tune": (np.arange(400) >= 100) & (np.arange(400) < 200),
        "identity_test": np.arange(400) >= 200,
    }
    excluded = [int(ids[2]), int(ids[104])]
    roles = role_indices(ids, masks, excluded_ids=excluded, caps=(30, 30, 30, 30))
    assert roles[2] == roles[104] == ""
    assert all((roles == r).sum() == 30 for r in ("fit", "tune", "cal", "test"))
    np.testing.assert_array_equal(
        roles,
        role_indices(ids.copy(), masks, excluded_ids=excluded, caps=(30, 30, 30, 30)),
    )


def test_normalization_never_reads_test_targets():
    target = np.arange(80, dtype=float).reshape(40, 2)
    roles = np.array(["fit"] * 20 + ["tune"] * 10 + ["test"] * 10)
    center, scale = fit_normalization(target, roles)
    changed = target.copy()
    changed[roles == "test"] *= 1e9
    other = fit_normalization(changed, roles)
    np.testing.assert_array_equal(center, other[0])
    np.testing.assert_array_equal(scale, other[1])
