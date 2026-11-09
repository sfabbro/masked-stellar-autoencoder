import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "comparison",
    Path(__file__).resolve().parents[1] / "scripts/compare_matched_desi_logg.py",
)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def test_paired_bootstrap_uses_differences_not_unpaired_panels():
    result = comparison.paired_mean_ci([-0.04] * 30)
    assert abs(result["mean"] + 0.04) < 1e-10
    assert all(abs(x + 0.04) < 1e-10 for x in result["ci95"])
