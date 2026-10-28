"""Keep the matched validation wave bounded, supported and distinct from old fits."""

from pathlib import Path

import pytest
import yaml

from masked_stellar_autoencoder.training.multilabel_experiments import (
    TARGET_NAMES,
    MultiLabelHead,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/multilabel-validation-wave-20261022.yaml"


def policy():
    return yaml.safe_load(CONFIG.read_text())


def identity(arm):
    return (
        arm["kind"],
        arm.get("flow_type"),
        arm.get("mdn_components"),
        bool(arm.get("noise_aware", False)),
        arm["seed"],
        float(arm["reconstruction_weight"]),
        arm.get("flow_objective", "nll"),
        float(arm.get("contrastive_weight", 0.1)),
    )


def test_validation_wave_exact_matched_budget_and_native_targets():
    cfg = policy()
    arms = cfg["arms"]
    assert len(arms) == len({arm["run_id"] for arm in arms}) == 24
    assert tuple(cfg["target_names"]) == TARGET_NAMES == ("teff", "logg", "fe_h")
    assert cfg["library_revision"] == "1d735c6a94456e5edbb4fd53f404121598c85830"
    assert cfg["checkpoint"].endswith("msa_pretrain_checkpoint_86.pth")
    assert (
        cfg["checkpoint_sha256"]
        == "d1ed550e3650817df1e1337b4e405fd50fefa43010866b828faa1aedf5040fb3"
    )
    assert cfg["reconstruction_presentations"] * len(arms) == 240000000
    assert cfg["budget"]["total_nominal_presentations"] == 240000000
    assert cfg["postfit_ensemble_policy"]["new_training_jobs"] == 0
    assert all(not arm["ttt"] and not arm["library_tta"] for arm in arms)
    for kind in ("gaussian", "mdn", "flow"):
        for noise in (False, True):
            group = [
                arm
                for arm in arms
                if arm["kind"] == kind
                and arm["noise_aware"] == noise
                and arm.get("flow_objective", "nll") == "nll"
            ]
            assert {(arm["seed"], arm["reconstruction_weight"]) for arm in group} == {
                (45, 0.01),
                (45, 0.03),
                (46, 0.03),
            }


def test_validation_does_not_duplicate_old_ids_or_same_seed_settings():
    cfg = policy()
    old = []
    for filename in (
        "multilabel-screen-20261020.yaml",
        "multilabel-w1-20261022.yaml",
        "multilabel-w2-20261022.yaml",
        "multilabel-density-20261021.yaml",
    ):
        old.extend(yaml.safe_load((ROOT / "configs" / filename).read_text())["arms"])
    assert not ({a["run_id"] for a in cfg["arms"]} & {a["run_id"] for a in old})
    assert not ({identity(a) for a in cfg["arms"]} & {identity(a) for a in old})
    assert len({identity(a) for a in cfg["arms"]}) == 24


@pytest.mark.parametrize(
    "kind,noise",
    [(kind, noise) for kind in ("gaussian", "mdn", "flow") for noise in (False, True)],
)
def test_all_six_head_schemas_construct_without_fabricated_target_columns(kind, noise):
    head = MultiLabelHead(
        kind, target_dim=3, flow_type="nsf", noise_aware=noise, mdn_components=4
    )
    assert head.target_dim == 3
    assert head.kind == kind


def test_clean_contrastive_matrix_and_density_conformal_policy():
    cfg = policy()
    contrastive = [a for a in cfg["arms"] if a.get("flow_objective", "nll") != "nll"]
    assert len(contrastive) == 6
    assert {
        (a["flow_objective"], a["contrastive_weight"], a["seed"]) for a in contrastive
    } == {
        (objective, weight, seed)
        for objective, weight in [
            ("contrastive", 0.1),
            ("nll_contrastive", 0.1),
            ("nll_contrastive", 1.0),
        ]
        for seed in (45, 46)
    }
    assert all(
        a["kind"] == "flow"
        and a["flow_type"] == "nsf"
        and not a["noise_aware"]
        and a["reconstruction_weight"] == 0.03
        and a["density_conformal"]
        for a in contrastive
    )
    assert cfg["density_conformal"] is False
    assert cfg["contrastive_negative_policy"]["negatives_per_positive"] == 1
    assert cfg["contrastive_negative_policy"]["duplicate_source_ids"] == "reject"


@pytest.mark.parametrize(
    "objective,weight",
    [("contrastive", 0.1), ("nll_contrastive", 0.1), ("nll_contrastive", 1.0)],
)
def test_contrastive_head_schemas_and_noise_guard(objective, weight):
    head = MultiLabelHead(
        "flow",
        target_dim=3,
        flow_type="nsf",
        noise_aware=False,
        flow_objective=objective,
        contrastive_weight=weight,
    )
    assert head.flow_objective == objective
    assert head.contrastive_weight == weight
    with pytest.raises(ValueError):
        MultiLabelHead(
            "flow",
            target_dim=3,
            flow_type="nsf",
            noise_aware=True,
            flow_objective=objective,
            contrastive_weight=weight,
        )
