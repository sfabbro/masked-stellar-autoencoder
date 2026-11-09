import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "covariance_audit",
    Path(__file__).resolve().parents[1] / "scripts/audit_native_label_covariance.py",
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_gaia_structured_low_bits_do_not_collapse_crossfit_fold():
    ids = np.arange(10000, dtype="i8") * 128 + 4185355646000000000
    fold = audit.crossfit_fold(ids)
    assert 0.45 < fold.mean() < 0.55
    np.testing.assert_array_equal(fold, audit.crossfit_fold(ids))


def test_correlated_score_recovers_expected_heldout_preference():
    covariance = np.array([[1.0, 0.7, 0.3], [0.7, 1.0, 0.2], [0.3, 0.2, 1.0]])
    residual = np.random.default_rng(14).multivariate_normal(
        np.zeros(3), covariance, size=20000
    )
    assert (
        audit.logscore(residual, covariance).mean()
        < audit.logscore(residual, np.eye(3)).mean() - 0.2
    )
    assert np.isfinite(audit.logscore(residual, covariance, student=True)).all()
