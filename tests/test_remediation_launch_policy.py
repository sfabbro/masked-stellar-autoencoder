import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import remediation_launch_policy as policy


@pytest.mark.parametrize(
    "task,gpu",
    [
        ("remediation-input-audit", 0),
        ("remediation-source", 0),
        ("remediation-joint-preflight", 1),
        ("remediation-pilot", 1),
    ],
)
def test_remediation_explicit_config_resources_and_pin(
    tmp_path, monkeypatch, task, gpu
):
    f = tmp_path / "policy.yaml"
    f.write_text(
        yaml.safe_dump(
            dict(
                run_id="test",
                arms=[dict(run_id="test")],
                library_revision=policy.LIBRARY,
            )
        )
    )
    monkeypatch.setitem(policy.CONFIGS, task, f)
    plan = dict(
        run_id="test",
        task=task,
        resources=dict(gpu=gpu),
        image="images.canfar.net/astroai/terminal:latest",
    )
    env = policy.bind_remediation_policy(plan, {})
    assert env["MSA_EXPERIMENT_PRESENTATIONS"] == "0"
    assert env["MSA_REMEDIATION_POLICY"] == str(f)
    for bad in [
        dict(run_id="unknown"),
        dict(resources=dict(gpu=1 - gpu)),
        dict(image="private/image"),
        dict(env=dict(MSA_EXPERIMENT_PRESENTATIONS="450000000")),
    ]:
        with pytest.raises(ValueError):
            policy.bind_remediation_policy(dict(plan, **bad), {})


def test_project_storage_gate_blocks_actual_over_quota_and_missing_properties():
    props = {
        "ivo://ivoa.net/vospace/core#quota": "20000000000000",
        "ivo://ivoa.net/vospace/core#length": "20000005095420",
    }
    with pytest.raises(ValueError, match="no job attempted"):
        policy.assert_project_storage_budget(dict(properties=str(props)))
    props["ivo://ivoa.net/vospace/core#length"] = "19950000000000"
    from types import MappingProxyType

    receipt = policy.assert_project_storage_budget(
        dict(properties=MappingProxyType(props))
    )
    assert receipt["available_bytes"] == 50_000_000_000
    with pytest.raises(KeyError):
        policy.assert_project_storage_budget(dict(properties={}))


def test_post_cleanup_recovery_plan_is_distinct_and_matches_bound_configs():
    root = Path(__file__).resolve().parents[1]
    jobs = yaml.safe_load(
        (root / "configs/remediation-recovery-launch-20261026-v3.yaml").read_text()
    )["jobs"]
    assert len(jobs) == 3
    for job in jobs:
        assert job["run_id"].endswith("_v3")
        if job["task"] != "remediation-input-audit":
            assert (
                yaml.safe_load((root / policy.CONFIGS[job["task"]]).read_text())[
                    "run_id"
                ]
                == job["run_id"]
            )
            policy.bind_remediation_policy(job, {})


def test_gradient_recovery_uses_distinct_run_ids_and_original_inputs():
    root = Path(__file__).parents[1]
    original = yaml.safe_load(
        (root / "configs/remediation-pilots-20261026.yaml").read_text()
    )
    recovery = yaml.safe_load(
        (
            root / "configs/remediation-pilots-gradient-recovery-20261026.yaml"
        ).read_text()
    )
    launch = yaml.safe_load(
        (
            root / "configs/remediation-pilot-gradient-recovery-launch-20261026.yaml"
        ).read_text()
    )
    assert len(recovery["arms"]) == len(launch["jobs"]) == 3
    assert {x["run_id"] for x in recovery["arms"]} == {
        x["run_id"] for x in launch["jobs"]
    }
    assert not {x["run_id"] for x in recovery["arms"]} & {
        x["run_id"] for x in original["arms"]
    }
    for key in (
        "checkpoint_sha256",
        "source_manifest",
        "library_revision",
        "seed",
        "updates",
    ):
        assert recovery[key] == original[key]
    for job in launch["jobs"]:
        env = policy.bind_remediation_policy(job, {})
        assert env["MSA_REMEDIATION_POLICY"].endswith(
            "remediation-pilots-gradient-recovery-20261026.yaml"
        )


def test_nsf_probe_launch_uses_distinct_inference_only_config():
    root = Path(__file__).parents[1]
    cfg = yaml.safe_load(
        (root / "configs/remediation-nsf-qmc-probe-20261026.yaml").read_text()
    )
    launch = yaml.safe_load(
        (root / "configs/remediation-nsf-qmc-probe-launch-20261026.yaml").read_text()
    )
    assert len(cfg["arms"]) == len(launch["jobs"]) == 1
    arm = cfg["arms"][0]
    assert (
        arm["integration_probe"]
        and arm["encoder"] == "frozen"
        and arm["reconstruction_weight"] == 0
    )
    assert (
        arm["run_id"] == launch["jobs"][0]["run_id"] == "remediation_nsf6_qmc_probe_v1"
    )
    env = policy.bind_remediation_policy(launch["jobs"][0], {})
    assert env["MSA_REMEDIATION_POLICY"].endswith(
        "remediation-nsf-qmc-probe-20261026.yaml"
    )


def test_nsf_qmc256_recoveries_bind_passing_probe_and_new_runs():
    root = Path(__file__).parents[1]
    original = yaml.safe_load(
        (root / "configs/remediation-pilots-20261026.yaml").read_text()
    )
    cfg = yaml.safe_load(
        (root / "configs/remediation-nsf-qmc256-recovery-20261026.yaml").read_text()
    )
    plan = yaml.safe_load(
        (
            root / "configs/remediation-nsf-qmc256-recovery-launch-20261026.yaml"
        ).read_text()
    )
    assert cfg["particles"] == cfg["integration_gate_particles"] == 256
    assert cfg["nsf_micro_batch_size"] == 16 and cfg["integration_chunk_rows"] == 8
    assert cfg["integration_gate"] == original["integration_gate"]
    assert len(cfg["arms"]) == len(plan["jobs"]) == 4
    assert not {x["run_id"] for x in cfg["arms"]} & {
        x["run_id"] for x in original["arms"]
    }
    for job in plan["jobs"]:
        assert "remediation_nsf6_qmc_probe_v1" in job["depends_on"]
        assert policy.bind_remediation_policy(job, {})[
            "MSA_REMEDIATION_POLICY"
        ].endswith("remediation-nsf-qmc256-recovery-20261026.yaml")


def test_nsf_startupfix_v3_recovers_exactly_two_failed_arms():
    import copy
    import hashlib

    root = Path(__file__).parents[1]
    cfg = yaml.safe_load((root / policy.NSF_STARTUP_CONFIG).read_text())
    original = yaml.safe_load((root / policy.NSF_QMC256_CONFIG).read_text())
    plan = yaml.safe_load(
        (
            root / "configs/remediation-nsf-qmc256-startupfix-launch-20261027.yaml"
        ).read_text()
    )
    assert len(cfg["arms"]) == len(plan["jobs"]) == 2
    assert {a["run_id"] for a in cfg["arms"]} == {j["run_id"] for j in plan["jobs"]}
    assert {a["recovery_from"]["session_id"] for a in cfg["arms"]} == {
        "yv16nrs1",
        "q6lyfrhv",
    }
    assert not any(
        "frozen" in a["run_id"] or "recon003" in a["run_id"] for a in cfg["arms"]
    )
    assert {k: v for k, v in cfg.items() if k != "arms"} == {
        k: v for k, v in original.items() if k != "arms"
    }
    for arm in cfg["arms"]:
        assert arm["run_id"].endswith("startupfix_v3")
        prior = arm["recovery_from"]
        assert prior["source_commit"] == "4b8d0b5b780366517baecf10e00165a9c9414b21"
        assert (
            hashlib.sha256((root / prior["failed_manifest"]).read_bytes()).hexdigest()
            == prior["failed_manifest_sha256"]
        )
        policy.validate_nsf_startup_failure(arm)
    for job in plan["jobs"]:
        assert policy.bind_remediation_policy(job, {})["MSA_REMEDIATION_POLICY"] == str(
            policy.NSF_STARTUP_CONFIG
        )
        bad = copy.deepcopy(job)
        bad["env"] = {
            "MSA_EXPERIMENT_OUTPUT": "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/remediation_nsf6_ft_qmc256_v2"
        }
        with pytest.raises(ValueError, match="output must be distinct"):
            policy.bind_remediation_policy(bad, {})
        bad = copy.deepcopy(job)
        bad["depends_on"] = []
        with pytest.raises(ValueError, match="dependencies"):
            policy.bind_remediation_policy(bad, {})
    changed = copy.deepcopy(cfg)
    changed["particles"] = 64
    with pytest.raises(ValueError, match="scientific settings"):
        policy.validate_nsf_startup_config(changed)
    changed = copy.deepcopy(cfg)
    changed["arms"][0]["encoder"] = "frozen"
    with pytest.raises(ValueError, match="arm settings"):
        policy.validate_nsf_startup_config(changed)


def test_nsf_startupfix_preserved_failure_sha_and_host_session_binding(tmp_path):
    import copy
    import json

    root = Path(__file__).parents[1]
    cfg = yaml.safe_load((root / policy.NSF_STARTUP_CONFIG).read_text())
    arm = copy.deepcopy(cfg["arms"][0])
    prior = arm["recovery_from"]
    evidence = tmp_path / prior["failed_manifest"]
    evidence.parent.mkdir(parents=True)
    evidence.write_bytes((root / prior["failed_manifest"]).read_bytes())
    accepted = tmp_path / prior["accepted_record"]
    accepted.parent.mkdir(parents=True, exist_ok=True)
    accepted.write_bytes((root / prior["accepted_record"]).read_bytes())
    policy.validate_nsf_startup_failure(arm, evidence_root=tmp_path)
    evidence.write_bytes(evidence.read_bytes() + b" ")
    with pytest.raises(ValueError, match="evidence SHA"):
        policy.validate_nsf_startup_failure(arm, evidence_root=tmp_path)
    evidence.write_bytes((root / prior["failed_manifest"]).read_bytes())
    attempts = json.loads(accepted.read_text())
    next(j for j in attempts["jobs"] if j["run_id"] == prior["run_id"])[
        "session_id"
    ] = "wrong-id"
    accepted.write_text(json.dumps(attempts))
    with pytest.raises(ValueError, match="session ID"):
        policy.validate_nsf_startup_failure(arm, evidence_root=tmp_path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "running"),
        ("run_id", "wrong-run"),
        ("source_commit", "0" * 40),
        ("error", "a different error"),
        ("error_type", "RuntimeError"),
        ("completed_updates", 1),
        ("training_performed", True),
        (
            "encoder_gradient_contract",
            {"update": 2, "supervised_l2": 0.0, "reconstruction_unweighted_l2": 0.0},
        ),
    ],
)
def test_nsf_startupfix_rejects_other_terminal_content_after_digest_validation(
    tmp_path, monkeypatch, field, value
):
    import copy
    import hashlib
    import json

    root = Path(__file__).parents[1]
    cfg = yaml.safe_load((root / policy.NSF_STARTUP_CONFIG).read_text())
    arm = copy.deepcopy(cfg["arms"][0])
    prior = arm["recovery_from"]
    failed = json.loads((root / prior["failed_manifest"]).read_text())
    failed[field] = value
    evidence = tmp_path / prior["failed_manifest"]
    evidence.parent.mkdir(parents=True)
    evidence.write_text(json.dumps(failed))
    prior["failed_manifest_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    monkeypatch.setitem(
        policy.NSF_STARTUP_PREVIOUS, arm["run_id"], copy.deepcopy(prior)
    )
    accepted = tmp_path / prior["accepted_record"]
    accepted.parent.mkdir(parents=True, exist_ok=True)
    accepted.write_bytes((root / prior["accepted_record"]).read_bytes())
    with pytest.raises(ValueError, match="exact terminal zero-update"):
        policy.validate_nsf_startup_failure(arm, evidence_root=tmp_path)


def test_copula_spline_policy_binds_new_library_and_preserved_gate_failure():
    import copy

    cfg = yaml.safe_load(policy.COPULA_CONFIG.read_text())
    policy.validate_copula_config(cfg)
    plan = yaml.safe_load(
        Path("configs/remediation-copula6-launch-20261027.yaml").read_text()
    )["jobs"][0]
    output = (
        "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/" + plan["run_id"]
    )
    env = policy.bind_remediation_policy(plan, {"MSA_EXPERIMENT_OUTPUT": output})
    assert env["MSA_REMEDIATION_POLICY"] == str(policy.COPULA_CONFIG)
    with pytest.raises(ValueError, match="output must be distinct"):
        policy.bind_remediation_policy(plan, {"MSA_EXPERIMENT_OUTPUT": "/wrong"})
    changed = copy.deepcopy(cfg)
    changed["integration_gate"]["gap_p95_max"] = 10.0
    with pytest.raises(ValueError, match="scientific settings"):
        policy.validate_copula_config(changed)
    changed = copy.deepcopy(cfg)
    changed["library_revision"] = policy.LIBRARY
    with pytest.raises(ValueError, match="binding"):
        policy.validate_copula_config(changed)
