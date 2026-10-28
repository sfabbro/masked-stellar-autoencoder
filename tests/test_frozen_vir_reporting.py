"""NIG marginal scoring must distinguish Student-t scale from standard deviation."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t

MONITOR = Path(__file__).resolve().parents[1] / "experiments/xp/monitor"
sys.path.insert(0, str(MONITOR))
spec = importlib.util.spec_from_file_location(
    "frozen_vir_reports", MONITOR / "report_frozen_vir.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_nig_student_marginal_density_and_exact_interval():
    posterior = np.array([[0.5, 2.0, 3.0, 0.8], [-0.7, 0.3, 1.2, 2.0]])
    y = np.array([0.4, -1.5])
    center, scale = -0.2, 0.31
    actual = report.student_arrays(posterior, y, center, scale)
    gamma, nu, alpha, beta = posterior.T
    location = gamma * scale + center
    t_scale = np.sqrt(beta * (1 + nu) / (alpha * nu)) * scale
    expected = t(df=2 * alpha, loc=location, scale=t_scale)
    np.testing.assert_allclose(actual["nll"], -expected.logpdf(y), rtol=1e-14)
    np.testing.assert_allclose(actual["std"], expected.std(), rtol=1e-14)
    np.testing.assert_allclose(
        actual["width_95"], expected.ppf(0.975) - expected.ppf(0.025), rtol=1e-14
    )
    assert not np.allclose(actual["std"], t_scale)


def test_student_density_unit_conversion_and_label_origin():
    posterior = np.array([[0.5, 2.0, 3.0, 0.8], [-0.7, 0.3, 1.2, 2.0]])
    truth = np.array([0.2, -0.8])
    base = report.student_arrays(posterior, truth, 0.0, 1.0)
    transformed = report.student_arrays(posterior, 7 * truth - 4, -4.0, 7.0)
    np.testing.assert_allclose(transformed["nll"], base["nll"] + np.log(7), atol=2e-14)
    np.testing.assert_array_equal(transformed["covered_95"], base["covered_95"])
    np.testing.assert_allclose(transformed["width_95"], 7 * base["width_95"])
