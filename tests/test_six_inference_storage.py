import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "inference_bench",
    Path(__file__).resolve().parents[1] / "scripts/benchmark_six_nsf_inference.py",
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def test_storage_counts_all_six_float32_axes():
    sizes = benchmark.storage_bytes(10, 128, 33)
    assert sizes["latent_float32"] == 5120
    assert sizes["six_marginal_quantile_grids_float32"] == 7920
    assert sizes["joint_1024_draws_float32"] == 245760
    with pytest.raises(ValueError):
        benchmark.storage_bytes(0, 128)
