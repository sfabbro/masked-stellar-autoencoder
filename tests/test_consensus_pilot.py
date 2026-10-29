import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torchregress.calibration import RelativeSurveySplineHead

from masked_stellar_autoencoder.training import multisurvey_consensus_pilot as pilot
from masked_stellar_autoencoder.training.multisurvey_data import SparseSurveyDataset
from masked_stellar_autoencoder.training.multisurvey_model import (
    CanonicalAtmosphereModel,
    CanonicalGaussianHead,
)


@pytest.mark.parametrize("frozen", [False, True])
def test_real_loop_synthetic_three_roles(tmp_path, frozen):
    path = tmp_path / "source.h5"
    ids = np.arange(1, 13, dtype="i8")
    targets = {
        k: {"field": k, "quantity": k, "unit": u}
        for k, u in [("teff", "K"), ("logg", "dex(cm/s2)"), ("fe_h", "dex")]
    }
    registry = {
        s: {"release": "r", "pipeline": "p", "dependency_id": s, "targets": targets}
        for s in ["a", "b"]
    }
    with h5py.File(path, "w") as h:
        h["source_id"] = ids
        h["input"] = np.tile([1.0, 2.0, 3.0, 4.0], (12, 1))
        h["role"] = np.array([b"fit"] * 4 + [b"tune"] * 4 + [b"devtest"] * 4)
        h["primary_input_eligible"] = np.ones(12, dtype=bool)
        root = h.create_group("native_observations")
        for name, sub in [("a", ids), ("b", ids[::2])]:
            g = root.create_group(name)
            g.attrs["source_fields"] = ["teff", "logg", "fe_h"]
            g["source_id"] = sub
            g["source_row"] = np.arange(len(sub))
            g["value"] = np.ones((len(sub), 3))
            g["error"] = np.full((len(sub), 3), 0.1)
            g["observed"] = np.ones((len(sub), 3), bool)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    datasets = [
        SparseSurveyDataset(path, registry, role=r, expected_source_sha256=digest)
        for r in ["fit", "tune", "devtest"]
    ]

    class MAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(4, 8), nn.Tanh())
            self.decoder = nn.Linear(8, 4)

        def forward(self, x):
            return self.decoder(self.encoder(torch.where(x == -9999, 0, x)))

    model = CanonicalAtmosphereModel(
        MAE(),
        CanonicalGaussianHead(
            8,
            torch.ones(3, dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
            width=8,
        ).double(),
    )
    if frozen:
        model.mae.encoder.requires_grad_(False)
    initial_encoder = [v.detach().clone() for v in model.mae.encoder.parameters()]
    knots = torch.tensor(
        [-5.0, -5.0, -5.0, -5.0, 0.0, 5.0, 5.0, 5.0, 5.0], dtype=torch.float64
    )
    relative = RelativeSurveySplineHead(
        [knots] * 3, torch.ones((2, 3), dtype=torch.float64)
    ).double()
    out = tmp_path / "output"
    out.mkdir()
    result = pilot.run_pilot(
        model,
        relative,
        *datasets,
        torch.full((2, 3), 0.1, dtype=torch.float64),
        output=out,
        parent_sha256="a" * 64,
        normalization={"center": [1.0, 1.0, 1.0], "scale": [1.0, 1.0, 1.0]},
        normalization_sha256="b" * 64,
        library_revision="c" * 40,
        msa_revision="d" * 40,
        reconstruction_width=4,
        seed=57,
        encoder_lr=0.001,
        head_lr=0.001,
        updates=2,
        batch_size=3,
        tune_every=1,
        panel_size=4,
    )
    assert (
        all(
            torch.equal(a, b)
            for a, b in zip(
                initial_encoder, model.mae.encoder.parameters(), strict=True
            )
        )
        == frozen
    )
    assert result["completed_updates"] == 2 and result["selected_step"] in (1, 2)
    assert Path(result["selected_checkpoint"]).exists()
    with (
        np.load(out / "tune-selected.npz") as tune,
        np.load(out / "devtest-selected.npz") as dev,
    ):
        assert set(tune["star_source_id"]) == set(range(5, 9))
        assert set(dev["star_source_id"]) == set(range(9, 13))
        assert not set(tune["star_source_id"]) & set(dev["star_source_id"])
        assert np.isfinite(dev["record_prediction"][dev["record_observed"]]).all()
    for ds in datasets:
        ds.close()
