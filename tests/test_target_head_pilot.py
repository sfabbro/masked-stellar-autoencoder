"""Synthetic end-to-end integration; no native source or production claims."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "target_pilot", Path(__file__).parents[1] / "scripts/run_target_head_pilot.py"
)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def prepared(tmp_path, target):
    arrays = {
        "source_id": np.arange(6, dtype=np.int64) + 100,
        "role": np.array(["fit", "fit", "tune", "tune", "test", "test"]),
        "embeddings": np.arange(18, dtype=float).reshape(6, 3) / 18,
    }
    metadata = {
        "schema_version": 1,
        "target": target,
        "encoder_checkpoint_sha256": "a" * 64,
        "source_provenance": "synthetic known rows",
        "split_provenance": "synthetic disjoint IDs",
        "frozen_embeddings": True,
    }
    if target == "age":
        arrays.update(
            teacher_tau=np.array(
                [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.2, 0.3, np.nan, 0.5, 0.6, 0.7]]
            ),
            teacher_sigma=np.full((2, 6), 0.2),
            teacher_quality=np.array([[1] * 6, [1, 1, 0, 1, 1, 1]]),
            teacher_features=np.zeros((6, 5)),
        )
        metadata.update(
            units={"age": "ln(age/Gyr)"},
            teacher_names=["a", "b"],
            age_scales=["a_scale", "b_scale"],
            reference_teacher="a",
            teacher_feature_policy="unused_affine",
        )
    else:
        arrays.update(
            parallax_mas=np.array([0.8, -0.1, 1.0, 0.4, 0.2, -0.2]),
            sigma_eff_mas=np.full(6, 0.4),
            zero_point_mas=np.full(6, 0.01),
        )
        metadata.update(
            units={"parallax": "mas", "latent": "ln(distance/kpc)"},
            zero_point_metadata="synthetic .01 mas",
            effective_sigma_metadata="synthetic .4 mas",
        )
    path, manifest = tmp_path / f"{target}.npz", tmp_path / f"{target}.json"
    np.savez(path, **arrays)
    metadata["input_sha256"] = pilot.digest(path)
    manifest.write_text(json.dumps(metadata))
    return path, manifest, arrays, metadata


@pytest.mark.parametrize("target", ["age", "distance"])
def test_actual_head_pilot_and_test_target_isolation(tmp_path, target):
    path, manifest, arrays, metadata = prepared(tmp_path, target)
    first = tmp_path / "first"
    result = pilot.run_pilot(path, manifest, first, target=target, steps=2, width=4)
    assert result["completed_steps"] == 2
    assert result["role_counts"] == {"fit": 2, "tune": 2, "test": 2}
    assert result["scope"].endswith("production certificate")
    if target == "age":
        arrays["teacher_tau"][:, -2:] += 100
    else:
        arrays["parallax_mas"][-2:] -= 100
    np.savez(path, **arrays)
    metadata["input_sha256"] = pilot.digest(path)
    manifest.write_text(json.dumps(metadata))
    second = tmp_path / "second"
    again = pilot.run_pilot(path, manifest, second, target=target, steps=2, width=4)
    assert result["history"] == again["history"]
    with (
        np.load(first / "predictions.npz") as a,
        np.load(second / "predictions.npz") as b,
    ):
        for key in a.files:
            assert np.array_equal(a[key], b[key])
    state = torch.load(first / "checkpoint.pth", weights_only=True)
    assert state["step"] == result["selected_step"]
    assert (
        result["selected_step"]
        == min(result["history"], key=lambda row: row["tune_nll"])["step"]
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("sha", "SHA"),
        ("units", "units"),
        ("ids", "source IDs"),
        ("role", "roles"),
        ("provenance", "metadata"),
        ("zero_point", "zero-point"),
    ],
)
def test_preparation_fails_closed(tmp_path, mutation, match):
    path, manifest, arrays, metadata = prepared(tmp_path, "distance")
    if mutation == "sha":
        metadata["input_sha256"] = "bad"
    elif mutation == "units":
        metadata["units"]["parallax"] = "arcsec"
    elif mutation == "ids":
        arrays["source_id"][-1] = arrays["source_id"][0]
    elif mutation == "role":
        arrays["role"][-2:] = "tune"
    elif mutation == "provenance":
        metadata.pop("split_provenance")
    elif mutation == "zero_point":
        metadata.pop("zero_point_metadata")
    np.savez(path, **arrays)
    if mutation != "sha":
        metadata["input_sha256"] = pilot.digest(path)
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=match):
        pilot.run_pilot(path, manifest, tmp_path / "fail", target="distance", steps=1)
    assert not (tmp_path / "fail").exists()
