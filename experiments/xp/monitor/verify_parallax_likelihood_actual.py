"""Numerical validation with preserved signed raw references; no fitting/science."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from masked_stellar_autoencoder.training.parallax_likelihood import signed_parallax_nll

root = Path(__file__).resolve().parents[3]
source = (
    root
    / "experiments/xp/results-stellar-improvement/distance_value_frozen_v3/paired-cohort.npz"
)
with np.load(source, allow_pickle=False) as data:
    measured = data["reference_hipparcos_parallax_mas"].copy()
    sigma = data["reference_hipparcos_error_mas"].copy()
    ids = data["reference_source_id"].copy()
assert len(measured) == len(sigma) == len(np.unique(ids)) == 4166
assert np.isfinite(measured).all() and (np.isfinite(sigma) & (sigma > 0)).all()
# A declared constant test value, never a fitted or scientific model prediction.
mu = torch.full((len(ids),), 10.0, dtype=torch.float64, requires_grad=True)
loss, count = signed_parallax_nll(
    mu, torch.from_numpy(measured), torch.from_numpy(sigma)
)
assert count == len(ids)
reference = np.mean(
    0.5 * ((measured - 1.0) / sigma) ** 2 + np.log(sigma * np.sqrt(2 * np.pi))
)
np.testing.assert_allclose(loss.detach().numpy(), reference, rtol=1e-12, atol=1e-12)
loss.backward()
derivative = (measured - 1.0) * np.log(10) / (5 * sigma**2 * len(ids))
np.testing.assert_allclose(mu.grad.numpy(), derivative, rtol=1e-12, atol=1e-12)
with source.open("rb") as stream:
    source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
report = dict(
    status="passed",
    purpose="actual signed-reference numerical validation only; no fitting or scientific performance",
    input_path=str(source),
    input_sha256=source_sha,
    rows=count,
    nonpositive_measured_parallaxes_retained=int((measured <= 0).sum()),
    all_raw_errors_retained_without_floor=True,
    all_finite_model_gradients=bool(torch.isfinite(mu.grad).all()),
    independent_numpy_objective_and_analytic_gradient_passed=True,
    maximum_absolute_gradient_difference=float(
        np.max(np.abs(mu.grad.numpy() - derivative))
    ),
    limitations="Constant test mu=10; no encoder, optimizer, Gaia zero-point, covariance calibration, or joint runner exercised.",
)
output = (
    root
    / "experiments/xp/results-stellar-improvement/joint-continuation-preparation-v1"
)
output.mkdir(parents=True, exist_ok=True)
(output / "signed-parallax-actual-input-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report))
