import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
spec = importlib.util.spec_from_file_location(
    "stellar_labels",
    Path(__file__).parents[1] / "scripts/run_stellar_label_experiments.py",
)
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


def test_sampler_keeps_support_and_exact_importance_identity():
    y = np.array([0.0, 0.0, 0.0, -3.0])
    q = job.sample_probabilities(y, 0.5)
    assert np.all(q > 0) and np.isclose(q.sum(), 1)
    loss = np.array([1.0, 2.0, 3.0, 20.0])
    w = 1 / (len(y) * q)
    np.testing.assert_allclose(np.sum(q * w * loss), loss.mean())


def test_rare_rows_survive_cap_and_splits_are_disjoint():
    ids = np.arange(1, 10001)
    y = np.zeros(10000)
    y[::7] = -3
    selected, weights = job.select_rows(
        ids, y, np.ones(10000, dtype=bool), np.zeros(10000), "identity", cap=100
    )
    partitions = job.probe.identity_split(ids)
    for k, (rows, w) in enumerate(zip(selected, weights, strict=True)):
        assert set(np.flatnonzero((partitions == k) & (y < -2))) <= set(rows)
        assert np.all(w >= 1)
    assert not set(selected[0]) & set(selected[2])


def test_noisy_label_variance_is_added_in_scaled_units():
    mean = torch.tensor([0.0])
    logvar = torch.tensor([0.0])
    target = torch.tensor([0.0])
    a = job.supervised_loss(mean, logvar, target, torch.tensor([1.0]), "noisy_nll")
    b = job.supervised_loss(mean, logvar, target, torch.tensor([2.0]), "noisy_nll")
    np.testing.assert_allclose(float(b - a), 0.5 * np.log(5 / 2), rtol=1e-6)


def test_refinement_cannot_see_unobserved_features_or_change_weights():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(3, 2)
            self.decoder = torch.nn.Linear(2, 3)
            self.reconstruction_layer = torch.nn.Linear(3, 3)

    torch.manual_seed(42)
    model = Model()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(4, 3)
    visible = torch.ones_like(x, dtype=torch.bool)
    visible[:, 2] = False
    a = job.refine(model, x, 3, visible=visible)
    changed = x.clone()
    changed[:, 2] = 10000
    b = job.refine(model, changed, 3, visible=visible)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())
