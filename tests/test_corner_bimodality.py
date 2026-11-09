import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "modes", Path(__file__).parents[1] / "scripts/audit_corner_bimodality.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_peak_diagnostic_distinguishes_clear_separated_modes_from_one_gaussian():
    rng = np.random.default_rng(21)
    assert not module.gravity_modes(rng.normal(3.0, 0.4, 8192))[
        "clear_separated_gravity_modes"
    ]
    x = np.r_[rng.normal(2.0, 0.1, 3000), rng.normal(4.0, 0.15, 5192)]
    assert module.gravity_modes(x)["clear_separated_gravity_modes"]
