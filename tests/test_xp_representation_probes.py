import importlib.util
from pathlib import Path

import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "xp_probes", Path(__file__).parents[1] / "scripts/run_xp_representation_probes.py"
)
probes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probes)


def test_identity_split_duplicate_stars_cannot_cross_splits():
    ids = np.repeat(np.arange(1, 5000), 2)
    split = probes.identity_split(ids)
    np.testing.assert_array_equal(split[::2], split[1::2])
    assert set(split) == {0, 1, 2}
    np.testing.assert_array_equal(probes.identity_split(ids[::-1])[::-1], split)


def test_target_aliases_are_masked_without_modifying_original():
    raw = np.ones((3, 6), dtype=np.float32)
    result = probes.safe_inputs(
        raw,
        ["teff", "e_teff", "teff_gspphot", "bp_1", "RA", "DEC"],
        ["teff"],
        ["RA", "DEC"],
    )
    np.testing.assert_array_equal(result[:, [0, 1, 2, 4, 5]], -9999)
    np.testing.assert_array_equal(result[:, 3], 1)
    np.testing.assert_array_equal(raw, 1)


def test_heldout_values_and_predictions_have_no_objective_gradient():
    prediction = torch.tensor([[2.0, 40.0]], requires_grad=True)
    z = torch.tensor([[1.0]], requires_grad=True)
    visible = torch.tensor([[True, False]])
    kwargs = dict(
        visible=visible,
        z=z,
        z0=torch.zeros_like(z),
        zscale=torch.ones_like(z),
        prior=0.1,
    )
    first = probes.observed_objective(
        prediction, torch.tensor([[3.0, 100.0]]), **kwargs
    )
    second = probes.observed_objective(
        prediction, torch.tensor([[3.0, -1000.0]]), **kwargs
    )
    torch.testing.assert_close(first, second)
    first.backward()
    assert prediction.grad[0, 1] == 0
    assert prediction.grad[0, 0] != 0


def test_train_only_ridge_can_recover_an_independent_linear_target():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(500, 4))
    y = x @ np.array([2.0, -3.0, 0.5, 4.0]) + 7.0
    prediction, settings = probes.fit_ridge(
        x[:300], y[:300], x[300:400], y[300:400], x[400:]
    )
    assert probes.score(prediction, y[400:])["mae"] < 0.001
    assert abs(settings["train_target_center"] - y[:300].mean()) < 1e-12


def test_raw_fits_unit_audit_uses_matched_ids_and_rejects_wrong_choice(tmp_path):
    import h5py
    import pytest

    features = ["G", "bp_1", "rp_1"]
    ids = np.arange(1, 150, dtype=np.int64)
    normalized = np.column_stack(
        [np.full(len(ids), 14.0), np.ones(len(ids)), np.full(len(ids), 2.0)]
    )
    raw = normalized.copy()
    raw[:, 1:] *= 10 ** ((8.5 - 14.0) / 2.5)
    records = np.zeros(
        len(ids),
        dtype=[("source_id", "i8"), ("G", "f4"), ("bp_1", "f4"), ("rp_1", "f4")],
    )
    records["source_id"] = ids
    for col, name in enumerate(features):
        records[name] = normalized[:, col]
    path = tmp_path / "benchmark.h5"
    with h5py.File(path, "w") as source:
        source.create_dataset("train/records", data=records)
        source.create_dataset("validation/records", data=records[:0])
    with pytest.raises(ValueError, match="fail independent audit"):
        probes.audit_fits_xp_units(path, features, raw.copy(), ids, "g_normalized")
    audit = probes.audit_fits_xp_units(path, features, raw, ids, "auto")
    assert audit["chosen_units"] == "raw"
    assert audit["matched_stars"] == 128
    np.testing.assert_allclose(raw, normalized)


def test_latent_refinement_never_uses_heldout_truth():
    class TinyAutoencoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(2, 1, bias=False)
            self.decoder = torch.nn.Identity()
            self.reconstruction_layer = torch.nn.Linear(1, 2, bias=False)
            with torch.no_grad():
                self.encoder.weight[:] = torch.tensor([[0.5, 0.0]])
                self.reconstruction_layer.weight[:] = torch.tensor([[1.0], [2.0]])

        def forward(self, inputs):
            latent = self.encoder(inputs)
            return self.reconstruction_layer(latent), latent

    model = TinyAutoencoder()
    hidden = torch.tensor([[False, True]])
    _, first = probes.refine(
        model, torch.tensor([[2.0, 8.0]]), hidden, torch.ones(1), steps=5
    )
    _, second = probes.refine(
        model, torch.tensor([[2.0, -1000.0]]), hidden, torch.ones(1), steps=5
    )
    torch.testing.assert_close(first, second)
    assert first[0, 0] > 1.0
