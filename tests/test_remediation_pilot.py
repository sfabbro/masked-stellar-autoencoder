"""Model-free prerequisite, sparse-pool and paired-selection pilot regressions."""

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

spec = importlib.util.spec_from_file_location(
    "pilot", Path(__file__).parents[1] / "scripts/run_remediation_pilot.py"
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def targets():
    values = np.column_stack([np.linspace(j + 1, j + 4, 40) for j in range(6)])
    mask = np.ones(values.shape, bool)
    errors = np.full(values.shape, np.nan)
    known = np.zeros_like(mask)
    fit = np.arange(40) < 30
    return values, mask, errors, known, fit


def test_fit_only_log_normalization_unknown_errors():
    values, mask, errors, known, fit = targets()
    values[~fit, :] = 1e9
    y, sigma, valid, center, scale, error_known = pilot.normalize_targets(
        values, mask, errors, known, fit
    )
    np.testing.assert_allclose(center[:4], values[fit, :4].mean(0))
    np.testing.assert_allclose(center[4:], np.log(values[fit, 4:]).mean(0))
    np.testing.assert_allclose(y[fit].mean(0), 0, atol=1e-14)
    assert not sigma.any() and not error_known.any() and valid.all()
    assert (scale > 0).all()


def test_sparse_targets_are_not_complete_case():
    values, mask, errors, known, fit = targets()
    mask[:20, 4] = False
    mask[20:, 5] = False
    y, _, valid, _, _, _ = pilot.normalize_targets(
        values, mask, errors, known, np.ones(40, bool)
    )
    assert valid[:, 4].sum() == 20 and valid[:, 5].sum() == 20
    assert not valid[:, 4:].all(1).any()
    assert np.all(y[~valid] == 0)


def test_teacher_widths_and_bad_known_errors_rejected():
    values, mask, errors, known, fit = targets()
    known[0, 4] = True
    errors[0, 4] = 0.2
    with pytest.raises(ValueError, match="teacher widths"):
        pilot.normalize_targets(values, mask, errors, known, fit)
    known[:] = False
    known[0, 0] = True
    with pytest.raises(ValueError, match="finite positive"):
        pilot.normalize_targets(values, mask, errors, known, fit)


def test_round_robin_gravity_pools_include_every_fit_identity():
    size = 72
    mask = np.ones((size, 6), bool)
    mask[-1, 5] = False
    fit = np.arange(size) < 60
    sources = np.full((size, 6), "galah", dtype="U12")
    sources[30:, :] = "apogee"
    gravity = np.tile([0.5, 1.5, 2.5, 3.5, 4.25, 4.8], 12)
    pools = pilot.balanced_pools(fit, mask, sources, gravity)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([p[2] for p in pools[1]])), np.flatnonzero(fit)
    )
    assert {s[1] for s in pools[1]} == set(range(6))
    assert len(pools[0]) == 2
    a, b = np.random.default_rng(12), np.random.default_rng(12)
    for step in range(12):
        left = pilot.draw_rows(a, pools, 48, target_column=step % 6)
        right = pilot.draw_rows(b, pools, 48, target_column=step % 6)
        np.testing.assert_array_equal(left, right)
        assert fit[left].all()


def test_fixed_tune_panel_deterministic_and_role_pure():
    ids = np.arange(1, 81, dtype="i8")
    roles = np.where(ids % 2, "fit", "tune")
    mask = np.ones((len(ids), 6), bool)
    left = pilot.fixed_role_rows(ids, roles, mask, "tune", 8)
    right = pilot.fixed_role_rows(ids, roles, mask, "tune", 8)
    np.testing.assert_array_equal(left, right)
    assert len(left) == 8 and np.all(roles[left] == "tune")


def test_measured_profile_changes_only_verified_parallax(tmp_path):
    path = tmp_path / "source.h5"
    with h5py.File(path, "w") as f:
        f["input"] = np.array([[2, 3, -9999], [4, 5, -9999]], dtype="f4")
        f["feature_names"] = np.array(["G", "bp_1", "PARALLAX"], dtype="S12")
        f["measured/input"] = [0.5, -9999]
        f["measured/observed"] = [True, False]
    source = {
        "feature_profiles": {
            "phot_xp_measured_parallax": dict(
                feature_index=2,
                override_dataset="measured/input",
                availability_dataset="measured/observed",
            )
        }
    }
    with h5py.File(path, "r") as f:
        x = pilot.profile_batch(f, [0, 1], "phot_xp_measured_parallax", source)
        np.testing.assert_array_equal(x, [[2, 3, 0.5], [4, 5, -9999]])
        with pytest.raises(ValueError, match="sorted and unique"):
            pilot.profile_batch(f, [1, 0], "phot_xp", source)


def prerequisites(tmp_path):
    source = dict(
        status="remediation_source_complete",
        training_source_ready=True,
        feature_profiles={"phot_xp": dict(ready=True)},
    )
    audit = dict(
        readiness=dict(
            phot_xp_input_pass=True,
            native_label_association_pass=True,
            astrometry_physical_likelihood_ready=False,
        )
    )
    preflight = dict(
        status="remediation_joint6_preflight_pass",
        run_id="preflight",
        library_revision="6bd",
        checks={k: dict(passed=True) for k in ("gaussian", "mdn", "nsf")},
    )
    cfg = dict(joint_preflight_run_id="preflight", library_revision="6bd")
    for key, value in [
        ("source_manifest", source),
        ("input_audit_manifest", audit),
        ("joint_preflight_manifest", preflight),
    ]:
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps(value))
        cfg[key] = str(path)
    return cfg, audit


def test_admission_requires_actual_success_and_scoped_readiness(tmp_path):
    cfg, audit = prerequisites(tmp_path)
    arm = dict(kind="nsf", encoder="finetune", reconstruction_weight=0)
    pilot.admission(cfg, arm)
    audit["readiness"]["phot_xp_input_pass"] = False
    Path(cfg["input_audit_manifest"]).write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="associations"):
        pilot.admission(cfg, arm)


def test_failed_nsf_preflight_never_admitted(tmp_path):
    cfg, _ = prerequisites(tmp_path)
    preflight = json.loads(Path(cfg["joint_preflight_manifest"]).read_text())
    preflight["checks"]["nsf"]["passed"] = False
    Path(cfg["joint_preflight_manifest"]).write_text(json.dumps(preflight))
    with pytest.raises(ValueError, match="nsf"):
        pilot.admission(
            cfg, dict(kind="huber", encoder="frozen", reconstruction_weight=0)
        )


def test_strict_real_sparse_nsf_importance_gate():
    cfg = dict(
        median_relative_ess_min=0.1, minimum_relative_ess_min=0.01, gap_p95_max=0.1
    )
    diag = dict(
        relative_ess=np.full(128, 0.5),
        independent_replica_log_prob_gap=np.full(128, 0.01),
        refinement_log_prob_gap=np.full(128, 0.01),
    )
    assert pilot.integration_gate(diag, cfg)["relative_ess_median"] == 0.5
    diag["relative_ess"][0] = 0.001
    with pytest.raises(ValueError, match="integration gate failed"):
        pilot.integration_gate(diag, cfg)


def test_declared_fourteen_arms_include_coupled_nsf_controls():
    config = yaml.safe_load(
        (
            Path(__file__).parents[1] / "configs/remediation-pilots-20261026.yaml"
        ).read_text()
    )
    assert config["updates"] == 1000 and len(config["arms"]) == 14
    assert len({arm["run_id"] for arm in config["arms"]}) == 14
    assert {arm["kind"] for arm in config["arms"]} == {
        "huber",
        "gaussian",
        "mdn",
        "nsf",
    }
    nsf = [arm for arm in config["arms"] if arm["kind"] == "nsf"]
    assert len(nsf) == 4 and any(
        arm.get("input_profile") == "phot_xp_measured_parallax" for arm in nsf
    )
    assert all(
        arm["reconstruction_weight"] == 0
        for arm in config["arms"]
        if arm["encoder"] == "frozen"
    )


def test_corner_gate_precedes_training_and_rejects_fit_or_missing():
    ids = np.array([7, 9, 11], dtype="i8")
    np.testing.assert_array_equal(
        pilot.heldout_corner_rows(ids, np.array(["fit", "tune", "devtest"]), [11, 9]),
        [1, 2],
    )
    with pytest.raises(ValueError, match="heldout"):
        pilot.heldout_corner_rows(ids, np.array(["fit", "tune", "devtest"]), [7])
    with pytest.raises(ValueError, match="missing"):
        pilot.heldout_corner_rows(ids, np.array(["fit", "tune", "devtest"]), [12])


def test_independent_encoder_gradient_angle_without_native_model():
    import torch

    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    result = pilot.encoder_gradient_contract(
        (p * torch.tensor([1.0, 0.0])).sum(), (p * torch.tensor([0.0, 2.0])).sum(), [p]
    )
    assert result["supervised_l2"] == 1 and result["reconstruction_unweighted_l2"] == 2
    assert result["cosine"] == 0


@pytest.mark.parametrize("kind", ["gaussian", "mdn"])
def test_zero_projection_warmup_must_reach_encoder_on_second_update(kind):
    """A real joint head initially has zero context gradient; it must recover."""
    import torch

    torch.manual_seed(47)
    encoder = torch.nn.Linear(3, 4).double()
    head = pilot.build_head(
        kind, 4, dict(head_width=8, particles=16, seed=11, mixture_components=3)
    ).double()
    optimizer = torch.optim.Adam([*encoder.parameters(), *head.parameters()], lr=1e-2)
    x = torch.randn(8, 3, dtype=torch.float64)
    y = torch.randn(8, 6, dtype=torch.float64)
    mask = torch.ones_like(y, dtype=torch.bool)
    sigma = torch.full_like(y, 0.1)
    for update in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        supervised = head.criterion(encoder(x), y, sigma, mask)
        contract = pilot.encoder_gradient_contract(
            supervised, supervised.new_zeros(()), encoder.parameters()
        )
        supervised.backward()
        state = pilot.enforce_encoder_gradient(
            encoder, head, kind, update, contract["supervised_l2"]
        )
        if update == 1:
            assert state == "zero_initialized_output_first_step"
            assert contract["supervised_l2"] == 0
        else:
            assert state == "nonzero" and contract["supervised_l2"] > 0
        optimizer.step()


def test_encoder_guard_rejects_disconnected_or_permanently_zero_encoder():
    import torch

    encoder = torch.nn.Linear(2, 2)
    head = pilot.build_head(
        "gaussian", 2, dict(head_width=4, particles=8, seed=11, mixture_components=2)
    )
    (
        head.criterion(
            torch.ones(2, 2),
            torch.ones(2, 6),
            torch.zeros(2, 6),
            torch.ones(2, 6, dtype=torch.bool),
        )
    ).backward()
    with pytest.raises(FloatingPointError, match="disconnected"):
        pilot.enforce_encoder_gradient(encoder, head, "gaussian", 1, 0.0)
    encoder.weight.grad = torch.ones_like(encoder.weight)
    with pytest.raises(FloatingPointError, match="no supervised gradient"):
        pilot.enforce_encoder_gradient(encoder, head, "gaussian", 2, 0.0)


def test_qmc_diagnostic_row_chunks_preserve_same_joint_flow_estimate():
    import torch

    from masked_stellar_autoencoder.training.remediation_joint6 import JointSixHead

    torch.manual_seed(32)
    head = JointSixHead("nsf", latent_dim=3, width=8, particles=16).double()
    context = torch.randn(5, 3, dtype=torch.double)
    y = torch.randn(5, 6, dtype=torch.double)
    sigma = torch.full_like(y, 0.1)
    mask = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    with torch.no_grad():
        full = head.diagnostics(context, y, sigma, mask, particles=16)
        chunked = pilot.diagnostics_in_row_chunks(
            head, context, y, sigma, mask, particles=16, chunk_rows=2
        )
    for key in full:
        if isinstance(full[key], torch.Tensor):
            torch.testing.assert_close(full[key], chunked[key], rtol=1e-11, atol=1e-11)
        else:
            assert full[key] == chunked[key]


def test_nsf_probe_requires_fixed_no_training_policy():
    cfg = {
        "integration_probe_particles": [128, 256, 512, 1024],
        "integration_probe_chunk_rows": 8,
    }
    arm = {
        "kind": "nsf",
        "encoder": "frozen",
        "reconstruction_weight": 0.0,
        "integration_probe": True,
    }
    pilot.validate_integration_probe(cfg, arm)
    with pytest.raises(ValueError, match="policy"):
        pilot.validate_integration_probe(cfg, {**arm, "encoder": "finetune"})
    with pytest.raises(ValueError, match="policy"):
        pilot.validate_integration_probe(
            {**cfg, "integration_probe_particles": [64]}, arm
        )


def test_nsf_chunked_backward_matches_full_joint_objective_and_gradients():
    import copy

    import torch

    from masked_stellar_autoencoder.training.remediation_joint6 import JointSixHead

    torch.manual_seed(401)
    encoder = torch.nn.Linear(3, 3).double()
    head = JointSixHead("nsf", latent_dim=3, width=8, particles=16).double()
    encoder2, head2 = copy.deepcopy(encoder), copy.deepcopy(head)
    x = torch.randn(5, 3, dtype=torch.double)
    y = torch.randn(5, 6, dtype=torch.double)
    sigma = torch.full_like(y, 0.15)
    mask = torch.tensor(
        [
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 1],
        ],
        dtype=torch.bool,
    )
    full = head.criterion(encoder(x), y, sigma, mask)
    full.backward()
    rows = np.arange(len(x))
    to_tensor = lambda value, dtype=torch.float64: torch.as_tensor(value, dtype=dtype)
    chunked, _ = pilot.backward_supervised_chunks(
        head2,
        lambda rr: encoder2(x[rr]),
        y.numpy(),
        sigma.numpy(),
        mask.numpy(),
        rows,
        chunk_rows=2,
        tensor=to_tensor,
    )
    torch.testing.assert_close(full.detach(), chunked, atol=1e-12, rtol=1e-12)
    for a, b in zip(
        [*encoder.parameters(), *head.parameters()],
        [*encoder2.parameters(), *head2.parameters()],
        strict=True,
    ):
        if a.grad is None:
            assert b.grad is None
        else:
            torch.testing.assert_close(a.grad, b.grad, atol=2e-10, rtol=2e-9)


def test_nsf_recovery_requires_real_probe_receipt(tmp_path):
    probe = {
        "status": "remediation_nsf_qmc_probe_complete",
        "run_id": "remediation_nsf6_qmc_probe_v1",
        "source_commit": "a" * 40,
        "first_passing_particles": 256,
        "training_performed": False,
        "library_revision": "b" * 40,
        "parent_sha256": "c" * 64,
        "integration_probe_outcomes": [{"particles": 256, "passed": True}],
    }
    path = tmp_path / "probe.json"
    path.write_text(json.dumps(probe))
    cfg = {
        "particles": 256,
        "integration_gate_particles": 256,
        "nsf_micro_batch_size": 16,
        "integration_chunk_rows": 8,
        "integration_probe_manifest": str(path),
        "integration_probe_manifest_sha256": pilot.sha(path),
        "integration_probe_source_commit": "a" * 40,
        "library_revision": "b" * 40,
        "checkpoint_sha256": "c" * 64,
    }
    assert pilot.validate_nsf_recovery(cfg, {"kind": "nsf"}) == probe
    with pytest.raises(ValueError, match="particle or microbatch"):
        pilot.validate_nsf_recovery({**cfg, "particles": 64}, {"kind": "nsf"})
    probe["integration_probe_outcomes"][0]["passed"] = False
    path.write_text(json.dumps(probe))
    cfg["integration_probe_manifest_sha256"] = pilot.sha(path)
    with pytest.raises(ValueError, match="did not pass"):
        pilot.validate_nsf_recovery(cfg, {"kind": "nsf"})


def test_microbatch_finetuning_rejects_batch_normalization():
    import torch

    encoder = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.BatchNorm1d(4))
    with pytest.raises(ValueError, match="batch normalization"):
        pilot.validate_microbatch_encoder(encoder, finetune=True)
    pilot.validate_microbatch_encoder(encoder, finetune=False)
    pilot.validate_microbatch_encoder(torch.nn.LayerNorm(4), finetune=True)
