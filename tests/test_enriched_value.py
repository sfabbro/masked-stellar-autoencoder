import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from run_enriched_value import fit_heads
from run_pristine_value import replay


def test_common_scaling_initial_states_no_test_tuning_and_independent_lstsq():
    rng = np.random.default_rng(4)
    z = rng.normal(size=(120, 7))
    y = rng.normal(size=120)
    native = np.arange(60)
    augmented = np.arange(85)
    states = fit_heads(z, y, native, augmented)
    a, b = states.values()
    np.testing.assert_array_equal(a["center"], b["center"])
    np.testing.assert_array_equal(a["scale"], b["scale"])
    assert a["ycenter"] == b["ycenter"] and a["yscale"] == b["yscale"]
    changed_z = z.copy()
    changed_z[85:] += 1000
    changed_y = y.copy()
    changed_y[85:] = 1e9
    counterfactual = fit_heads(changed_z, changed_y, native, augmented)
    for name, rows in [("native_only", native), ("canonical_augmented", augmented)]:
        state = states[name]
        for key in state:
            np.testing.assert_array_equal(state[key], counterfactual[name][key])
        x = np.column_stack([np.ones(len(z)), (z - state["center"]) / state["scale"]])
        regularizer = np.eye(x.shape[1]) * np.sqrt(10.0)
        regularizer[0, 0] = 0
        coeff = np.linalg.lstsq(
            np.vstack([x[rows], regularizer]),
            np.r_[(y[rows] - state["ycenter"]) / state["yscale"], np.zeros(x.shape[1])],
            rcond=None,
        )[0]
        np.testing.assert_allclose(coeff, state["coef"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            replay(z, state),
            (x @ coeff) * state["yscale"] + state["ycenter"],
            rtol=1e-12,
            atol=1e-12,
        )
    with pytest.raises(ValueError):
        fit_heads(z, y, native, np.r_[augmented, 0])
