"""Synthetic source fidelity, role exclusion and joint-path gradient checks."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.multisurvey_data import collate_multisurvey
from masked_stellar_autoencoder.training.multisurvey_flow import MixtureResidualHead
from masked_stellar_autoencoder.training.multisurvey_joint import (
    IndexedRows,
    RoleMemoryDataset,
    mixture_terms,
    physical_components,
)

spec = importlib.util.spec_from_file_location(
    "joint_fixture", Path(__file__).with_name("test_multisurvey_training.py")
)
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


def test_role_memory_preserves_original_records_and_rejects_other_rows(tmp_path):
    trainer, source = fixture.setup_trainer(tmp_path)
    memory = RoleMemoryDataset(source)
    for i in range(len(source)):
        batch = collate_multisurvey([memory[i]])
        trainer.gate.verify_batch(batch)
    with pytest.raises(ValueError, match="outside cached role"):
        memory._handle()["input"][
            next(i for i in range(len(source.ids)) if i not in source.star_rows)
        ]
    with pytest.raises(ValueError):
        memory._memory["input"].values[0, 0] = 99
    source.close()


def test_mixture_supervision_and_reconstruction_reach_encoder_decoder_and_survey_heads(
    tmp_path,
):
    trainer, source = fixture.setup_trainer(tmp_path)
    model, relative = trainer.model, trainer.relative_head
    for p in model.head.parameters():
        p.requires_grad_(False)
    model.head.double()
    relative.double()
    mixture = MixtureResidualHead(4, width=8).double()
    batch = collate_multisurvey([source[0], source[1]])
    center = torch.zeros(4, dtype=torch.float64)
    scale = torch.ones_like(center)
    mask = torch.tensor([[True, False], [True, False]])
    terms = mixture_terms(
        model,
        mixture,
        relative,
        batch,
        trainer.source_floor,
        center,
        scale,
        mask=mask,
        reconstruction_width=2,
    )
    supervised = torch.autograd.grad(
        terms["nll_sum"], list(model.mae.encoder.parameters()), retain_graph=True
    )
    assert sum(float(g.square().sum()) for g in supervised) > 0
    loss = (
        terms["nll_sum"] / terms["active"]
        + 0.03 * terms["reconstruction_sum"] / terms["masked_count"]
        + 0.0001 * relative.regularization()
    )
    loss.backward()
    for module in (model.mae.encoder, model.mae.decoder, mixture, relative):
        gs = [p.grad for p in module.parameters() if p.grad is not None]
        assert gs and all(torch.isfinite(g).all() for g in gs)
        assert sum(float(g.abs().sum()) for g in gs) > 0
    w, mu, lower, mean, cov = physical_components(
        mixture, terms["context"], terms["reference_mean"], terms["reference_lower"]
    )
    torch.testing.assert_close(w.sum(1), torch.ones(2, dtype=w.dtype))
    torch.testing.assert_close(mean, (w[..., None] * mu).sum(1))
    assert (torch.linalg.eigvalsh(cov) > 0).all()
    with pytest.raises(ValueError, match="available"):
        mixture_terms(
            model,
            mixture,
            relative,
            batch,
            trainer.source_floor,
            center,
            scale,
            mask=torch.ones(2, 2, dtype=torch.bool),
            reconstruction_width=2,
        )
    source.close()


def test_global_row_cache_handles_empty_and_vector_lookups():
    rows = IndexedRows([2, 5], np.array([[1, 2], [3, 4]]))
    np.testing.assert_array_equal(rows[[5, 2]], [[3, 4], [1, 2]])
    with pytest.raises(ValueError):
        rows[4]


def test_fullfit_launch_binds_complete_passes_reconstruction_and_all_role_exports():
    import yaml

    scripts = Path(__file__).parents[1] / "scripts"
    spec = importlib.util.spec_from_file_location(
        "joint_policy", scripts / "multisurvey_joint_launch_policy.py"
    )
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    plan = yaml.safe_load(
        Path("configs/multisurvey-joint-launch-20261105.yaml").read_text()
    )["jobs"][0]
    env = policy.bind_joint_policy(
        plan, {"MSA_EXPERIMENT_OUTPUT": policy.ROOT + plan["run_id"]}
    )
    config = yaml.safe_load(policy.CONFIG.read_text())
    assert env["MSA_XP_EXPERIMENTS"] == "0"
    assert "fit_stars" not in config and config["epochs"] == 3
    assert config["reconstruction_weight"] > 0
    assert {"fit", "devtest"} <= set(config["export_roles"])
    plan["resources"]["gpu"] = 2
    with pytest.raises(ValueError):
        policy.bind_joint_policy(plan, env)
