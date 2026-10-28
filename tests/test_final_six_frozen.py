"""Synthetic frozen-control schedule/cache/state invariants only."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_final_six_frozen as frozen


def test_matched_frozen_head_schedule_has_zero_encoder_and_reconstruction():
    cfg = yaml.safe_load(Path("configs/final-six-frozen-20261024.yaml").read_text())
    budget = frozen.schedule(cfg)
    assert budget["warmup_updates"] == 250
    assert budget["head_optimizer_updates"] == 306
    assert budget["supervised_minibatch_calls"] == 250 + 306 * 8 == 2698
    assert budget["total_head_adam_steps"] == 556
    assert (
        budget["encoder_updates"]
        == budget["decoder_updates"]
        == budget["reconstruction_presentations"]
        == 0
    )
    assert len(cfg["arms"]) == 8 and len({a["run_id"] for a in cfg["arms"]}) == 8
    assert all("10m" not in a["run_id"] for a in cfg["arms"])
    assert frozen.schedule(cfg, smoke=True)["total_head_adam_steps"] == 4
    with pytest.raises(ValueError, match="Positive"):
        frozen.schedule(dict(cfg, warmup_updates=0))


def test_frozen_cache_requires_exact_provenance_and_bytes(tmp_path):
    cache = tmp_path / "fit.npz"
    bindings = dict(input_sha256="input", parent_sha256="parent", dtype="float32")
    assert not frozen.bound_cache(cache, bindings)
    np.savez_compressed(cache, embedding=np.ones((4, 2), dtype="f4"))
    with pytest.raises(ValueError, match="Incomplete"):
        frozen.bound_cache(cache, bindings)
    receipt = cache.with_suffix(".bindings.json")
    receipt.write_text(
        json.dumps(dict(bindings=bindings, cache_sha256=frozen.joint.sha(cache)))
    )
    assert frozen.bound_cache(cache, bindings)
    with pytest.raises(ValueError, match="Changed"):
        frozen.bound_cache(cache, dict(bindings, parent_sha256="other"))
    np.savez_compressed(cache, embedding=np.zeros((4, 2), dtype="f4"))
    with pytest.raises(ValueError, match="Changed"):
        frozen.bound_cache(cache, bindings)


def test_frozen_toy_encoder_values_gradients_and_head_adam_counts():
    torch.manual_seed(37)
    encoder = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.5)).eval()
    encoder.requires_grad_(False)
    before = frozen.frozen_parameter_digest(encoder)
    head = torch.nn.Linear(4, 2)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.01)
    x = torch.randn(8, 3)
    with torch.no_grad():
        latent = encoder(x)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        head(latent).square().mean().backward()
        optimizer.step()
    assert frozen.frozen_parameter_digest(encoder) == before
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    assert all(float(v["step"]) == 4 for v in optimizer.state.values())
    with torch.no_grad():
        encoder[0].weight.add_(0.01)
    assert frozen.frozen_parameter_digest(encoder) != before


def test_smoke_roles_are_heldout_and_have_every_label():
    roles = np.array(["fit"] * 40 + ["tune"] * 40 + ["cal"] * 40 + ["test"] * 40)
    mask = np.ones((len(roles), 6), dtype=bool)
    mask[:, 1::2] = np.arange(len(roles))[:, None] % 2 == 0
    data = dict(role=roles)
    for role in ["tune", "cal", "test"]:
        rows = frozen.role_rows(data, mask, role, True)
        assert (roles[rows] == role).all()
        assert (mask[rows].sum(0) >= 16).all()
    assert np.array_equal(
        frozen.role_rows(data, mask, "test", False), np.flatnonzero(roles == "test")
    )


def test_native_runner_rejects_local_without_reading_source(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CANFAR"):
        frozen.run(
            SimpleNamespace(
                config=tmp_path / "missing.yaml",
                run_id="forbidden",
                output=tmp_path / "no-output",
            )
        )
    assert not (tmp_path / "no-output").exists()
