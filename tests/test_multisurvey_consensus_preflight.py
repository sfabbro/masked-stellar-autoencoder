import importlib.util
from pathlib import Path

import pytest
import torch

from masked_stellar_autoencoder.training.multisurvey_consensus_preflight import (
    run_two_steps,
)


def test_two_step_smoke_uses_float64_and_checks_both_gradients(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "trainer_fixture",
        Path(__file__).with_name("test_multisurvey_training.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trainer, dataset = module.setup_trainer(tmp_path)
    trainer.model.head.double()
    trainer.relative_head.double()
    result = run_two_steps(
        trainer.model,
        trainer.relative_head,
        dataset,
        torch.full((2, 3), 0.1, dtype=torch.float64),
        parent_sha256="a" * 64,
        normalization_sha256="b" * 64,
        library_revision="c" * 40,
        msa_revision="d" * 40,
        reconstruction_width=2,
    )
    assert result["updates"] == 2 and not result["supervised_xp_masking"]
    assert result["likelihood_precision"] == "float64"
    assert all(
        step["supervised_encoder_gradient_l2"] > 0
        and step["reconstruction_decoder_gradient_l2"] > 0
        for step in result["steps"]
    )
    dataset.close()


def test_preflight_rejects_fp32_head(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "trainer_fixture",
        Path(__file__).with_name("test_multisurvey_training.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    trainer, dataset = module.setup_trainer(tmp_path)
    with pytest.raises(ValueError, match="Float64"):
        run_two_steps(
            trainer.model,
            trainer.relative_head,
            dataset,
            torch.full((2, 3), 0.1),
            parent_sha256="a" * 64,
            normalization_sha256="b" * 64,
            library_revision="c" * 40,
            msa_revision="d" * 40,
            reconstruction_width=2,
        )
    dataset.close()
