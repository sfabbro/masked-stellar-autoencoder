import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "corner_panel",
    Path(__file__).parents[1] / "scripts/run_representative_corner_worker.py",
)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_random_panel_is_reproducible_and_stress_additions_are_separate():
    ids = np.arange(100, dtype="i8") + 2**60
    roles = np.array(["fit"] * 20 + ["devtest"] * 80)
    eligible = np.ones(100, bool)
    eligible[20] = False
    panel, random, stress = worker.select_panel(
        ids, roles, eligible, [ids[25], ids[0], ids[20]], size=15
    )
    again = worker.select_panel(ids, roles, eligible, [ids[25]], size=15)
    np.testing.assert_array_equal(panel, again[0])
    assert random.sum() == 15 and stress.sum() == 1
    assert np.all(roles[panel] == "devtest") and eligible[panel].all()
    np.testing.assert_array_equal(ids[panel[stress]], [ids[25]])
