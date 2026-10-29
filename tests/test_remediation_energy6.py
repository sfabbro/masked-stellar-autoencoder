"""Synthetic integration checks only; no stellar model/checkpoint is loaded."""

import numpy as np
import pytest
import torch

from masked_stellar_autoencoder.training.remediation_energy6 import (
    EnergyJointSixHead,
    backward_energy_chunks,
    capture_pilot_rng_state,
    energy_device_preflight,
    enforce_energy_encoder_gradient,
    restore_pilot_rng_state,
    validate_energy_config,
)


def config():
    return {
        "energy_training_draws": 16,
        "energy_micro_batch_size": 4,
        "energy_sample_seed": 101,
        "energy_noise_seed": 202,
        "unknown_error_floor_normalized": [0.1] * 6,
        "energy_noise_assumption": "independent_gaussian",
        "energy_floor_scope": "illustrative_not_measured",
    }


def toy():
    torch.manual_seed(34)
    head = EnergyJointSixHead(
        latent_dim=3,
        width=16,
        particles=16,
        seed=42,
        components=4,
        energy_training_draws=16,
        unknown_error_floor_normalized=[0.1] * 6,
    ).double()
    head.configure_generators("cpu", 101, 202)
    context = torch.randn((4, 3), dtype=torch.float64)
    target = torch.randn((4, 6), dtype=torch.float64)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [1, 0, 1, 0, 1, 0],
        ],
        dtype=torch.bool,
    )
    known = mask.clone()
    known[0, 3] = False
    sigma = torch.where(known, 0.05, 0.0).to(dtype=torch.float64)
    return head, context, target, sigma, mask, known


def test_config_requires_explicit_scoped_positive_floors_and_separate_seeds():
    np.testing.assert_allclose(validate_energy_config(config()), np.full(6, 0.1))
    for field, bad in (
        ("energy_training_draws", 1),
        ("energy_micro_batch_size", 0),
        ("energy_noise_seed", 101),
        ("unknown_error_floor_normalized", [0.0] * 6),
        ("unknown_error_floor_normalized", [True] * 6),
        ("energy_noise_assumption", "unknown"),
        ("energy_floor_scope", "measured"),
    ):
        with pytest.raises(ValueError):
            validate_energy_config({**config(), field: bad})


def test_generators_advance_and_restore_exactly_and_proposal_is_frozen():
    head, context, target, sigma, mask, known = toy()
    assert all(not parameter.requires_grad for parameter in head.gaussian.parameters())
    before = head.generator_state()
    first = head.criterion(context, target, sigma, mask, known)
    after = head.generator_state()
    assert all(not torch.equal(before[k], after[k]) for k in before)
    head.restore_generator_state(before, device="cpu")
    replay = head.criterion(context, target, sigma, mask, known)
    torch.testing.assert_close(first, replay, rtol=0, atol=0)
    assert set(head.last_loss_components) == {"masked_energy_score"}
    first.backward()
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad)
        for p in head.flow.parameters()
    )
    assert all(p.grad is None for p in head.gaussian.parameters())
    with pytest.raises(ValueError, match="Both energy generator states"):
        head.restore_generator_state({"sample": before["sample"]}, device="cpu")


def test_copy_preflight_preserves_original_parameters_and_rngs():
    head, context, target, sigma, mask, known = toy()
    before = {key: value.detach().clone() for key, value in head.state_dict().items()}
    rng = head.generator_state()
    global_rng = torch.get_rng_state().clone()
    result = energy_device_preflight(head, context, target, sigma, mask, known)
    assert result["copy_only"] and result["updates"] == 2
    assert result["steps"][1]["context_gradient_l2"] > 0
    assert all(
        torch.equal(value, head.state_dict()[key]) for key, value in before.items()
    )
    assert all(
        torch.equal(value, head.generator_state()[key]) for key, value in rng.items()
    )
    assert torch.equal(global_rng, torch.get_rng_state())


def test_checkpoint_rng_state_replays_sampling_and_independent_noise():
    head, context, target, sigma, mask, known = toy()
    mask_generator = torch.Generator().manual_seed(59)
    sampler = np.random.default_rng(60)
    saved = capture_pilot_rng_state(head, mask_generator, sampler)
    first = head.criterion(context, target, sigma, mask, known)
    first_mask = torch.rand(3, generator=mask_generator)
    first_rows = sampler.integers(100, size=4)
    first_global = torch.rand(3)
    restore_pilot_rng_state(head, mask_generator, sampler, saved, device="cpu")
    second = head.criterion(context, target, sigma, mask, known)
    second_mask = torch.rand(3, generator=mask_generator)
    second_rows = sampler.integers(100, size=4)
    second_global = torch.rand(3)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_mask, second_mask, rtol=0, atol=0)
    np.testing.assert_array_equal(first_rows, second_rows)
    torch.testing.assert_close(first_global, second_global, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Complete pilot RNG state"):
        restore_pilot_rng_state(head, mask_generator, sampler, {}, device="cpu")


def test_gradient_guard_allows_only_flow_projection_warmup_then_supervised_encoder():
    head, _, target, sigma, mask, known = toy()
    torch.manual_seed(53)
    encoder = torch.nn.Linear(3, 3).double()
    features = torch.randn((4, 3), dtype=torch.float64)
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *head.flow.parameters()], lr=0.002
    )
    for update in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        loss = head.criterion(encoder(features), target, sigma, mask, known)
        separate = torch.autograd.grad(loss, encoder.parameters(), retain_graph=True)
        supervised = sum(float(g.detach().square().sum()) for g in separate) ** 0.5
        loss.backward()
        result = enforce_energy_encoder_gradient(encoder, head, update, supervised)
        if update == 1:
            assert result == "zero_initialized_flow_projection_first_step"
        else:
            assert result == "supervised_nonzero" and supervised > 0
        optimizer.step()
    with pytest.raises(FloatingPointError, match="no supervised gradient"):
        enforce_energy_encoder_gradient(encoder, head, 2, 0.0)


def test_microbatch_energy_weights_unique_stars_not_observed_coordinates():
    parameter = torch.nn.Parameter(torch.tensor(1.0))

    class MockHead:
        def criterion(self, context, target, sigma, mask, known):
            assert mask.dtype == known.dtype == torch.bool
            return context.mean()

    def latent(rows):
        return parameter * torch.as_tensor(rows + 1, dtype=torch.float32)

    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(value, dtype=dtype)

    mask = np.array(
        [[1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]], dtype=bool
    )
    rows = np.array([0, 1, 2], dtype="i8")
    targets = np.zeros((3, 6), dtype="f4")
    value, stats = backward_energy_chunks(
        MockHead(),
        latent,
        targets,
        targets,
        mask,
        mask,
        rows,
        chunk_rows=1,
        tensor=tensor,
    )
    torch.testing.assert_close(value, torch.tensor(2.0))
    torch.testing.assert_close(parameter.grad, torch.tensor(2.0))
    assert stats == {"masked_energy_score": 2.0}
    with pytest.raises(ValueError, match="Sorted unique"):
        backward_energy_chunks(
            MockHead(),
            latent,
            targets,
            targets,
            mask,
            mask,
            np.array([0, 0]),
            chunk_rows=1,
            tensor=tensor,
        )
    mask[1] = False
    with pytest.raises(ValueError, match="observed target"):
        backward_energy_chunks(
            MockHead(),
            latent,
            targets,
            targets,
            mask,
            mask,
            rows,
            chunk_rows=1,
            tensor=tensor,
        )


def test_chunked_two_update_training_finetunes_encoder_with_separate_reconstruction():
    import importlib.util
    from pathlib import Path

    path = Path("scripts/run_remediation_energy_pilot.py")
    spec = importlib.util.spec_from_file_location("energy_runner_test", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    torch.manual_seed(16)
    head, context, targets, sigma, mask, known = toy()
    encoder = torch.nn.Linear(3, 3).double()
    decoder = torch.nn.Linear(3, 3).double()
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), *decoder.parameters(), *head.flow.parameters()],
        lr=0.002,
    )
    target = targets.numpy()
    noise = sigma.numpy()
    mask_np = mask.numpy()
    known_np = known.numpy()

    def tensor(value, dtype=torch.float32):
        return torch.as_tensor(
            value, dtype=torch.float64 if dtype == torch.float32 else dtype
        )

    def latent(rows):
        return encoder(context[rows])

    norms = []
    for update in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        score, _ = backward_energy_chunks(
            head,
            latent,
            target,
            noise,
            mask_np,
            known_np,
            np.arange(4),
            chunk_rows=2,
            tensor=tensor,
        )
        reconstruction = (decoder(encoder(context)) - context).square().mean()
        contract = runner.accumulated_encoder_gradient_contract(
            reconstruction, encoder.parameters()
        )
        (0.03 * reconstruction).backward()
        enforce_energy_encoder_gradient(
            encoder, head, update, contract["supervised_l2"]
        )
        assert all(torch.isfinite(p.grad).all() for p in decoder.parameters())
        assert torch.isfinite(score)
        norms.append(contract["supervised_l2"])
        optimizer.step()
    assert norms[0] == 0 and norms[1] > 0


def test_evaluation_and_corner_samples_are_genuine_joint_six_vectors():
    head, context, targets, sigma, mask, known = toy()
    head.eval()
    with torch.no_grad():
        gen = torch.Generator(device=context.device).manual_seed(133)
        draw = head.sample(context, 8192, gen)
        pred = torch.quantile(draw, 0.5, dim=0)
        pit = (draw <= targets.unsqueeze(0)).double().mean(dim=0)
        cov = torch.cov(draw[:, 0, :].T)
    assert draw.shape == (8192, 4, 6)
    assert pred.shape == pit.shape == (4, 6)
    assert cov.shape == (6, 6)
    assert torch.isfinite(draw).all() and torch.isfinite(cov).all()


def test_runner_evaluate_path_for_nsf_energy_preserves_xp_and_saves_pits(tmp_path):
    import ast
    from pathlib import Path

    source = Path("scripts/run_remediation_energy_pilot.py").read_text()
    module = ast.parse(source)
    run = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    evaluate = next(
        node
        for node in ast.walk(run)
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate"
    )
    extracted = ast.fix_missing_locations(ast.Module(body=[evaluate], type_ignores=[]))
    head, context, targets, sigma, mask, known = toy()

    class Model:
        def eval(self):
            return self

    y = targets.numpy()
    observed = mask.numpy()
    errors = sigma.numpy()
    error_known = known.numpy()
    data = {
        "real_xp_present": np.array([True] * 4),
        "real_no_xp": np.array([False] * 4),
        "measured_parallax/observed": np.array([True] * 4),
    }
    namespace = {
        "model": Model(),
        "head": head,
        "cfg": {"evaluation_batch": 2, "seed": 3, "evaluation_draws": 64},
        "arm": {"kind": "nsf_energy"},
        "torch": torch,
        "np": np,
        "latent": lambda rows: context[rows],
        "device": torch.device("cpu"),
        "y": y,
        "sigma": errors,
        "error_known": error_known,
        "scale": np.ones(6),
        "center": np.zeros(6),
        "values": np.column_stack((y[:, :4], np.exp(y[:, 4:]))),
        "mask": observed,
        "native_scale": np.ones(6),
        "ids": np.arange(1, 5),
        "errors": errors,
        "label_sources": np.full((4, 6), "toy"),
        "roles": np.full(4, "tune"),
        "data": data,
    }
    exec(compile(extracted, "<runner-evaluate>", "exec"), namespace)
    out = tmp_path / "evaluated.npz"
    result = namespace["evaluate"](np.arange(4), out)
    assert np.isfinite(result[0])
    with np.load(out) as saved:
        assert saved["native_median"].shape == (4, 6)
        assert saved["empirical_intrinsic_pit"].shape == (4, 6)
        assert np.isnan(saved["empirical_observed_pit"][0, 3])
        assert saved["real_xp_present"].all() and not saved["real_no_xp"].any()
