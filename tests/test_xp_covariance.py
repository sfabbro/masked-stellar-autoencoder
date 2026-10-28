import importlib.util
import math
from pathlib import Path

import h5py
import numpy as np
import torch

from masked_stellar_autoencoder.training.xp_experiments import experiment_objective


def test_full_covariance_equals_diagonal_and_microbatch_gradients():
    torch.manual_seed(3)
    prediction = torch.randn(3, 111, requires_grad=True)
    target = torch.randn(3, 111)
    mask = torch.rand(3, 111) > 0.4
    target[~mask] = float("nan")
    sigma = torch.full_like(target, 0.2)
    xp = list(range(1, 111))
    corr = torch.zeros(3, 2, 1485)
    diagonal = experiment_objective(
        prediction,
        target,
        mask,
        xp,
        {"loss": "torchregress_noise"},
        mask.sum(),
        sigma=sigma,
    )
    full = experiment_objective(
        prediction,
        target,
        mask,
        xp,
        {"loss": "torchregress_covariance"},
        mask.sum(),
        sigma=sigma,
        correlations=corr,
    )
    torch.testing.assert_close(full.float(), diagonal, atol=1e-6, rtol=1e-6)
    parts = sum(
        experiment_objective(
            prediction[i : i + 1],
            target[i : i + 1],
            mask[i : i + 1],
            xp,
            {"loss": "torchregress_covariance"},
            mask.sum(),
            sigma=sigma[i : i + 1],
            correlations=corr[i : i + 1],
        )
        for i in range(3)
    )
    torch.testing.assert_close(full, parts)
    full_grad = torch.autograd.grad(full, prediction, retain_graph=True)[0]
    diag_grad = torch.autograd.grad(diagonal, prediction)[0]
    torch.testing.assert_close(full_grad, diag_grad)
    assert torch.isfinite(full_grad).all() and (full_grad[~mask] == 0).all()


def test_covariance_scores_masked_principal_submatrix_against_analytic_solution():
    prediction = torch.zeros(1, 110, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[0, :2] = torch.tensor([1.0, 2.0])
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0, :2] = True
    sigma = torch.ones_like(target)
    corr = torch.zeros(1, 2, 1485)
    corr[0, 0, 0] = 0.5  # (1,0) in numpy lower-triangle order
    value = experiment_objective(
        prediction,
        target,
        mask,
        list(range(110)),
        {"loss": "torchregress_covariance"},
        mask.sum(),
        sigma=sigma,
        correlations=corr,
    )
    covariance = np.array([[1.1, 0.5], [0.5, 1.1]])
    residual = np.array([1.0, 2.0])
    expected = (
        0.5
        * (
            np.linalg.slogdet(covariance)[1]
            + residual @ np.linalg.solve(covariance, residual)
            + 2 * math.log(2 * math.pi)
        )
        / 2
    )
    np.testing.assert_allclose(value.detach().numpy(), expected, rtol=1e-7)


def test_benchmark_errors_are_flux_normalized_before_scaler(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "runner", Path(__file__).parents[1] / "scripts/run_pretrain_experiments.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with h5py.File(tmp_path / "test.h5", "w") as source:
        source["train/records"] = np.array(
            [(11.0, 100.0, 2.0), (8.5, 50.0, 3.0)],
            dtype=[("G", "f8"), ("bp_1", "f8"), ("bpe_1", "f8")],
        )
        values, sigma, _ = runner.read_prefixes(
            source,
            ["train/records"],
            ["G", "bp_1"],
            [None, "bpe_1"],
            2,
            np.array([0.0, 0.0]),
            np.array([1.0, 2.0]),
            tmp_path / "cache.npy",
            correct_xp_errors=True,
        )
        np.testing.assert_allclose(values[:, 1], [50.0, 25.0])
        np.testing.assert_allclose(sigma[:, 1], [10.0, 1.5])
