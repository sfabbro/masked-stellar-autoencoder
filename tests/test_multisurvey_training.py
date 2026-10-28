"""Synthetic FIT-only consensus training and checkpoint contracts."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip("torchregress.calibration.multisource")

from torchregress.calibration import RelativeSurveySplineHead

from masked_stellar_autoencoder.training.multisurvey_data import (
    SparseSurveyDataset,
    collate_multisurvey,
)
from masked_stellar_autoencoder.training.multisurvey_model import (
    CanonicalAtmosphereModel,
    CanonicalGaussianHead,
)
from masked_stellar_autoencoder.training.multisurvey_training import (
    ConsensusTrainer,
    FitIdentityGate,
)

spec = importlib.util.spec_from_file_location(
    "synthetic_survey_data", Path(__file__).with_name("test_multisurvey_data.py")
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def setup_trainer(tmp_path):
    path = tmp_path / "source.h5"
    fixture.make_source(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = SparseSurveyDataset(
        path, fixture.registry(), role="fit", expected_source_sha256=sha
    )
    ids_sha = hashlib.sha256(
        np.asarray(dataset.ids[dataset.star_rows], dtype="<i8").tobytes()
    ).hexdigest()
    gate = FitIdentityGate(dataset, source_binding=sha, fit_ids_sha256=ids_sha)

    class MissingAwareEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 4)

        def forward(self, x):
            # The real MAE embeds missing sentinels; a raw linear layer would
            # amplify -9999 and is not a faithful synthetic encoder.
            return self.linear(torch.where(x == -9999, 0, x))

    class MAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = MissingAwareEncoder()
            self.decoder = nn.Linear(4, 2)

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = CanonicalAtmosphereModel(
        MAE(),
        CanonicalGaussianHead(4, torch.tensor([1.0, 0.0, 0.0]), torch.ones(3), width=8),
    )
    knots = torch.tensor([-5.0, -5.0, -5.0, -5.0, 0.0, 5.0, 5.0, 5.0, 5.0])
    relative = RelativeSurveySplineHead([knots] * 3, torch.ones(2, 3))
    optimizer = torch.optim.Adam(
        [*model.parameters(), *relative.parameters()], lr=0.001
    )
    trainer = ConsensusTrainer(
        model,
        relative,
        optimizer,
        gate,
        torch.full((2, 3), 0.1),
        parent_sha256="a" * 64,
        normalization_sha256="b" * 64,
        library_revision="c" * 40,
        msa_revision="d" * 40,
        assume_diagonal_catalog_covariance=True,
        reconstruction_weight=0.03,
        reconstruction_width=2,
    )
    return trainer, dataset


def test_training_uses_natural_inputs_and_rejects_forged_or_heldout_stars(tmp_path):
    trainer, dataset = setup_trainer(tmp_path)
    batch = collate_multisurvey([dataset[0], dataset[1]])
    seen = []
    hook = trainer.model.mae.encoder.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone())
    )
    metrics = trainer.step(
        batch, reconstruction_mask=torch.tensor([[True, False], [True, False]])
    )
    hook.remove()
    torch.testing.assert_close(seen[0], batch["input"])
    assert seen[1][0, 0] == -9999 and batch["input"][0, 0] != -9999
    assert np.isfinite(metrics["total"]) and trainer.steps == 1
    forged = dict(batch)
    forged["input"] = batch["input"].clone()
    forged["input"][0, 0] = -9999
    with pytest.raises(ValueError, match="differs from source"):
        trainer.step(
            forged, reconstruction_mask=torch.tensor([[True, False], [True, False]])
        )
    forged = dict(batch)
    forged["source_id"] = torch.tensor([11, 44], dtype=torch.int64)
    with pytest.raises(ValueError, match="outside the pinned FIT"):
        trainer.step(
            forged, reconstruction_mask=torch.tensor([[True, False], [True, False]])
        )
    dataset.close()


def test_checkpoint_contract_restores_state_and_refuses_tampering(tmp_path):
    trainer, dataset = setup_trainer(tmp_path)
    batch = collate_multisurvey([dataset[0], dataset[1]])
    trainer.step(
        batch, reconstruction_mask=torch.tensor([[True, False], [True, False]])
    )
    path = tmp_path / "checkpoint.pth"
    digest = trainer.save(path, sampler_state={"epoch": 1, "offset": 2})
    old = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    with torch.no_grad():
        next(trainer.model.parameters()).add_(5)
    assert trainer.load(path, expected_sha256=digest) == {"epoch": 1, "offset": 2}
    assert trainer.steps == 1
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, old[key])
    # Repeating the same next step after restore must reproduce model updates.
    trainer.step(
        batch, reconstruction_mask=torch.tensor([[True, False], [True, False]])
    )
    continued = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.load(path, expected_sha256=digest)
    trainer.step(
        batch, reconstruction_mask=torch.tensor([[True, False], [True, False]])
    )
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, continued[key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        trainer.save(path, sampler_state={})
    with pytest.raises(ValueError, match="SHA256 differs"):
        trainer.load(path, expected_sha256="0" * 64)
    dataset.close()


def test_mutated_fixed_gauge_cannot_be_trained_or_saved(tmp_path):
    trainer, dataset = setup_trainer(tmp_path)
    batch = collate_multisurvey([dataset[0], dataset[1]])
    with torch.no_grad():
        trainer.relative_head.weights[0, 0] += 1
    with pytest.raises(ValueError, match="buffers changed"):
        trainer.step(
            batch, reconstruction_mask=torch.tensor([[True, False], [True, False]])
        )
    with pytest.raises(ValueError, match="buffers changed"):
        trainer.save(tmp_path / "bad.pth", sampler_state={})
    dataset.close()


def test_changed_source_after_construction_rejects_fit_gate(tmp_path):
    import h5py

    path = tmp_path / "source.h5"
    fixture.make_source(path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    dataset = SparseSurveyDataset(
        path, fixture.registry(), role="fit", expected_source_sha256=sha
    )
    ids_sha = hashlib.sha256(
        np.asarray(dataset.ids[dataset.star_rows], dtype="<i8").tobytes()
    ).hexdigest()
    with h5py.File(path, "r+") as h:
        h["input"][0, 0] = 42.0
    with pytest.raises(ValueError, match="changed after SHA256 verification"):
        FitIdentityGate(dataset, source_binding=sha, fit_ids_sha256=ids_sha)
    dataset.close()


def test_changed_source_floor_is_rejected(tmp_path):
    trainer, dataset = setup_trainer(tmp_path)
    batch = collate_multisurvey([dataset[0], dataset[1]])
    trainer.source_floor[0, 0] += 0.5
    with pytest.raises(ValueError, match="error floors changed"):
        trainer.step(
            batch, reconstruction_mask=torch.tensor([[True, False], [True, False]])
        )
    dataset.close()


def test_reordered_optimizer_parameters_cannot_save(tmp_path):
    trainer, dataset = setup_trainer(tmp_path)
    trainer.optimizer.param_groups[0]["params"].reverse()
    with pytest.raises(ValueError, match="parameter order changed"):
        trainer.save(tmp_path / "bad.pth", sampler_state={})
    dataset.close()


def test_changed_head_settings_cannot_save(tmp_path):
    trainer, dataset = setup_trainer(tmp_path)
    trainer.relative_head.boundary = "linear"
    with pytest.raises(ValueError, match="configuration changed"):
        trainer.save(tmp_path / "bad.pth", sampler_state={})
    dataset.close()


def test_resume_rejects_a_different_noise_floor_contract(tmp_path):
    trainer, dataset = setup_trainer(tmp_path)
    path = tmp_path / "checkpoint.pth"
    digest = trainer.save(path, sampler_state={"epoch": 0})
    changed = ConsensusTrainer(
        trainer.model,
        trainer.relative_head,
        trainer.optimizer,
        trainer.gate,
        torch.full((2, 3), 0.2),
        parent_sha256="a" * 64,
        normalization_sha256="b" * 64,
        library_revision="c" * 40,
        msa_revision="d" * 40,
        assume_diagonal_catalog_covariance=True,
        reconstruction_weight=0.03,
        reconstruction_width=2,
    )
    with pytest.raises(ValueError, match="training contract differs"):
        changed.load(path, expected_sha256=digest)
    dataset.close()
