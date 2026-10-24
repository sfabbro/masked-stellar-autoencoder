"""Synthetic distribution identities only; no native checkpoints or inputs."""

from types import SimpleNamespace

import pytest
import torch
from torch.distributions import AffineTransform

from masked_stellar_autoencoder.training.distribution_diagnostics import (
    marginal_cdf,
    native_coordinates,
    predictive_draws,
    predictive_quantile_metrics,
    sample_metrics,
)


class Gaussian:
    kind = "gaussian"

    def head(self, latent):
        return torch.cat([latent, torch.zeros_like(latent)], -1)


class Mixture:
    kind = "mdn"

    def __init__(self):
        self.head = self

    def __call__(self, latent):
        return latent

    def _mixture_parameters(self, latent):
        means = torch.stack([latent - 2, latent + 2], 1)
        return (
            latent.new_full((len(latent), 2), -torch.log(latent.new_tensor(2))),
            means,
            torch.full_like(means, -4),
        )


class Scalar:
    def __init__(self):
        p = (torch.arange(32, dtype=torch.float64) + 0.5) / 32
        self.measurement_nodes = torch.special.ndtri(p)[:, None]

    def distribution(self, latent):
        return SimpleNamespace(
            transform=AffineTransform(-latent[:, :1] / 2, latent.new_tensor(0.5))
        )


def test_exact_normal_cdf_and_observed_convolution():
    z = torch.zeros((3, 6), dtype=torch.float64)
    y = torch.linspace(-2, 2, 18).reshape(3, 6).double()
    sigma = torch.full_like(y, 0.4)
    assert torch.allclose(marginal_cdf(Gaussian(), z, y), torch.special.ndtr(y))
    assert torch.allclose(
        marginal_cdf(Gaussian(), z, y, sigma), torch.special.ndtr(y / (1.16**0.5))
    )
    with pytest.raises(ValueError):
        marginal_cdf(Gaussian(), z, y, -sigma)


def test_mixture_keeps_shared_component_not_independent_marginals():
    head = Mixture()
    latent = torch.zeros((1, 6), dtype=torch.float64)
    y = torch.zeros_like(latent)
    assert torch.allclose(marginal_cdf(head, latent, y), torch.full_like(y, 0.5))
    samples = predictive_draws(head, latent, 4096, torch.Generator().manual_seed(12))[
        :, 0
    ]
    assert float(torch.corrcoef(samples.T)[0, 1]) > 0.99


def test_scalar_cdf_direction_and_independent_base_dimensions():
    latent = torch.full((2, 6), 1.0, dtype=torch.float64)
    head = SimpleNamespace(kind="factorized_nsf", heads=[Scalar() for _ in range(6)])
    target = torch.full_like(latent, 1.5)
    assert torch.allclose(
        marginal_cdf(head, latent, target), torch.special.ndtr((target - 1) / 2)
    )
    noisy = marginal_cdf(head, latent, target, torch.full_like(latent, 0.3))
    exact = torch.special.ndtr((target - 1) / (4.09**0.5))
    assert torch.allclose(noisy, exact, atol=0.001)
    sample = predictive_draws(
        head, latent[:1], 4096, torch.Generator().manual_seed(31)
    )[:, 0]
    assert float((torch.corrcoef(sample.T) - torch.eye(6)).abs().max()) < 0.08


def test_library_proper_scores_against_explicit_pairwise_formulas():
    g = torch.Generator().manual_seed(111)
    coordinate = torch.randn((7, 2, 6), generator=g, dtype=torch.float64)
    target = torch.randn((2, 6), generator=g, dtype=torch.float64)
    native = coordinate.clone()
    native[..., 4:] = native[..., 4:].exp()
    native_y = target.clone()
    native_y[:, 4:] = native_y[:, 4:].exp()
    result = sample_metrics(
        native,
        native_y,
        torch.zeros(6),
        torch.ones(6),
        torch.ones_like(native_y, dtype=torch.bool),
    )
    delta = (native[:, None] - native[None]).abs().sum((0, 1)) / (7 * 6)
    crps = (native - native_y).abs().mean(0) - 0.5 * delta
    assert torch.allclose(result["crps_native"], crps, atol=1e-7, rtol=1e-7)
    assert torch.allclose(result["vario_rho1_native"], -crps, atol=1e-7, rtol=1e-7)
    energy = torch.linalg.vector_norm(coordinate - target, dim=-1).mean(
        0
    ) - 0.5 * torch.linalg.vector_norm(
        coordinate[:, None] - coordinate[None], dim=-1
    ).sum((0, 1)) / (7 * 6)
    assert torch.allclose(
        result["energy_full6_fit_standardized"], energy, atol=1e-7, rtol=1e-7
    )
    vario = sum(
        (
            (target[:, i] - target[:, j]).abs().sqrt()
            - (coordinate[:, :, i] - coordinate[:, :, j]).abs().sqrt().mean(0)
        ).square()
        for i in range(6)
        for j in range(i + 1, 6)
    )
    assert torch.allclose(
        result["variogram_full6_fit_standardized"], vario, atol=1e-7, rtol=1e-7
    )


def test_missing_joint_blocks_are_not_imputed_and_native_support_is_positive():
    standardized = torch.randn(
        (64, 2, 6), generator=torch.Generator().manual_seed(11), dtype=torch.float64
    )
    draws = native_coordinates(standardized, torch.zeros(6), torch.ones(6))
    assert (draws[:, :, 4:] > 0).all()
    target = draws.mean(0)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[:, 3:] = False
    score = sample_metrics(draws, target, torch.zeros(6), torch.ones(6), mask)
    assert score["energy_full6_fit_standardized"].isnan().all()
    assert score["crps_native"][:, 3:].isnan().all()
    q = predictive_quantile_metrics(draws, target, mask)
    assert (q["quantile_0.025_native"] <= q["quantile_0.975_native"]).all()
    assert (q["predictive_95_width_native"][:, :3] >= 0).all()
    assert q["predictive_95_interval_score_native"][:, 3:].isnan().all()


def test_complete_metric_batch_and_pilot_use_observed_labels_only():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_final_six_distribution_metrics import calculate_batch, pilot_contracts

    z = torch.zeros((3, 6), dtype=torch.float64)
    y = torch.zeros_like(z)
    target = y.clone()
    target[:, 4:] = 1
    mask = torch.ones_like(y, dtype=torch.bool)
    # Missing first-column references must not make the Vario contract NaN.
    mask[:, 0] = False
    target[:, 0] = float("nan")
    result, samples = calculate_batch(
        Gaussian(), z, y, z, target, mask, z[0], torch.ones(6), 12
    )
    assert result["intrinsic_pit"][:, 0].isnan().all()
    report = pilot_contracts(
        Gaussian(), z, y, mask, target, z, z[0], torch.ones(6), samples
    )
    assert report["status"] == "passed"
    assert report["empirical_cdf_max_gap"] < 0.12


def test_failure_recording_does_not_erase_terminal_or_live_locked_outputs(tmp_path):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_final_six_distribution_metrics as runner

    config = tmp_path / "config.yaml"
    config.write_text("test: true\n")
    output = tmp_path / "out"
    output.mkdir()
    args = SimpleNamespace(output=output, config=config)
    path = output / "manifest.json"
    for status in ["complete", "failed", "running"]:
        path.write_text(json.dumps({"status": status}))
        original = path.read_bytes()
        runner.record_failure(args, BlockingIOError("busy"))
        assert path.read_bytes() == original
        if status != "running":
            runner.record_failure(args, ValueError("terminal output"))
            assert path.read_bytes() == original
    path.write_text(
        json.dumps(
            {
                "status": "running",
                "bindings": {
                    "runner": runner.sampling.sha(runner.__file__),
                    "config": runner.sampling.sha(config),
                },
            }
        )
    )
    runner.record_failure(args, ValueError("actual new metric failure"))
    assert json.loads(path.read_text())["status"] == "failed"


def test_new_metric_cache_requires_own_hash_and_exact_source_ids(tmp_path):
    import json
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import run_final_six_distribution_metrics as runner

    p = tmp_path / "test-000000.npz"
    np.savez_compressed(p, source_id=np.array([123, 456]), pit=np.array([0.2, 0.5]))
    with pytest.raises(ValueError):
        runner.load_bound_cache(p, np.array([123, 456]))
    receipt = p.with_suffix(".sha256.json")
    receipt.write_text(json.dumps({"sha256": runner.sampling.sha(p)}))
    assert np.array_equal(
        runner.load_bound_cache(p, np.array([123, 456]))["source_id"], [123, 456]
    )
    with pytest.raises(ValueError):
        runner.load_bound_cache(p, np.array([123, 457]))
    np.savez_compressed(p, source_id=np.array([123, 456]), pit=np.array([0.4, 0.5]))
    with pytest.raises(ValueError):
        runner.load_bound_cache(p, np.array([123, 456]))
