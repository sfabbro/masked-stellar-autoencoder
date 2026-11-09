import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "tess_report",
    Path(__file__).resolve().parents[1] / "scripts/report_tess_gravity_qa.py",
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_reference_convolution_does_not_change_point_error():
    truth = np.array([2.0, 3.0])
    draws = np.broadcast_to(truth + 0.1, (1024, 2)).copy()
    tight = report.gravity_stats(draws, truth, np.array([0.001, 0.001]))
    broad = report.gravity_stats(draws, truth, np.array([0.3, 0.3]))
    assert abs(tight["mae"] - 0.1) < 1e-10
    assert tight["mae"] == broad["mae"]
    assert tight["reference_coverage90"] == 0
    assert broad["reference_coverage90"] == 1
    assert report.gravity_stats(
        np.empty((1024, 0)), np.array([]), np.array([])
    ) == dict(n=0)
