"""Historical rescoring is one fixed checkpoint per bounded-memory zero-SGD job."""

import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_density_rescoring_wave as wave


def config():
    return yaml.safe_load(
        (
            Path(__file__).resolve().parents[1]
            / "configs/density-rescoring-wave-20261022.yaml"
        ).read_text()
    )


def test_coverage_and_bound_policy():
    cfg = config()
    assert len(cfg["cases"]) == 15
    original = [c["artifact_policy"]["run_id"] for c in cfg["cases"]]
    assert len(set(original)) == len(original)
    assert "multi_gaussian_noise_recon_30e_v1" in original
    assert "multi_bpf_recon_s42_v1" in original
    assert cfg["training_updates"] == 0 and cfg["evaluation_batch"] == 128
    for case in cfg["cases"]:
        assert wave.selected_case(cfg, case["run_id"]) == case
        p = wave.make_policy(case, cfg)
        assert p["head_kwargs"]["target_dim"] == 3
        assert p["checkpoint_sha256"] == case["artifact_policy"]["checkpoint_sha256"]
        assert len(p["scoring_code_sha256"]) == 64
        assert "original_library_revision" in p


@pytest.mark.parametrize("fault", ["duplicate", "unknown", "memory", "sgd"])
def test_invalid_selection_fails(fault):
    cfg = copy.deepcopy(config())
    run_id = cfg["cases"][0]["run_id"]
    if fault == "duplicate":
        cfg["cases"].append(cfg["cases"][0])
    elif fault == "unknown":
        run_id = "missing"
    elif fault == "memory":
        cfg["evaluation_batch"] = 97695
    else:
        cfg["training_updates"] = 1
    with pytest.raises(ValueError):
        wave.selected_case(cfg, run_id)
