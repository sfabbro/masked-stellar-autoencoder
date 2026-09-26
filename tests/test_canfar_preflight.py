from pathlib import Path

import h5py
import numpy as np
import yaml
from astropy.table import Table

from masked_stellar_autoencoder.training.canfar_preflight import (
    _candidate_error_columns,
    _has_blockers,
    _missing_columns,
    build_report,
)


def test_error_column_candidates_follow_known_catalogue_names():
    assert _candidate_error_columns("bp_12") == ["bpe_12"]
    assert _candidate_error_columns("rp_4") == ["rpe_4"]
    assert "E_G_SMSS" in _candidate_error_columns("G_SMSS")
    assert _candidate_error_columns("G") == ["e_G"]
    assert "e_parallax" in _candidate_error_columns("PARALLAX")


def test_schema_report_only_suggests_columns_that_exist():
    missing = _missing_columns(
        ["bp_1", "PARALLAX", "EBV"],
        [],
        {"bpe_1", "e_parallax", "EBV"},
    )

    assert missing == [
        {"feature": "bp_1", "configured_error": None, "candidates": ["bpe_1"]},
        {
            "feature": "PARALLAX",
            "configured_error": None,
            "candidates": ["e_parallax"],
        },
        {"feature": "EBV", "configured_error": None, "candidates": []},
    ]


def test_explicit_null_uncertainties_are_not_schema_blockers():
    missing = _missing_columns(
        ["W1", "G", "EBV"],
        [None, "e_G", None],
        {"e_G"},
    )

    assert missing == []


def test_preflight_blockers_accept_stage_specific_schema_fields():
    report = {
        "runtime": {"cuda_available": True},
        "pretrain": {
            "exists": True,
            "missing_valid_keys": [],
            "missing_features": [],
            "missing_recon_cols": [],
            "missing_configured_errors": [],
            "unmapped_errors": [],
            "error_count_matches_features": True,
            "error_list_matches_features": False,
        },
        "finetune": {
            "exists": True,
            "missing_features": [],
            "missing_recon_cols": [],
            "missing_classes": [],
            "missing_configured_errors": [],
            "unmapped_errors": [],
            "error_count_matches_features": True,
            "error_list_matches_features": False,
        },
        "finetune_checkpoint": {"exists": True},
        "compatibility": {
            "feature_cols_match": True,
            "recon_cols_match": True,
            "layer_dims_match": True,
            "decoder_dims_match": True,
            "rtdl_embed_match": True,
            "pt_activ_func_match": True,
            "norm_match": True,
            "encoder_type_match": True,
            "growth_rate_match": True,
            "num_dense_layers_match": True,
            "cosine_latent_match": True,
            "heteroscedastic_match": True,
        },
    }

    assert not _has_blockers(report, require_cuda=True)


def test_preflight_accepts_matching_synthetic_pipeline_inputs(tmp_path):
    pretrain_path = tmp_path / "pretrain.h5"
    dtype = np.dtype([("feature", "f4"), ("error", "f4")])
    sample = np.array([(1.0, 0.1)], dtype=dtype)
    with h5py.File(pretrain_path, "w") as h5:
        h5.create_dataset("train", data=sample)
        h5.create_dataset("valid", data=sample)

    finetune_path = tmp_path / "finetune.fits"
    Table(
        {
            "feature": np.array([1.0]),
            "error": np.array([0.1]),
            "label": np.array([2.0]),
        }
    ).write(finetune_path)
    checkpoint_path = tmp_path / "pretrain.pth"
    checkpoint_path.touch()
    model = {"layer_dims": [8, 4], "decoder_dims": [4], "rtdl_embed": 2}
    pretrain_config = {
        "data": {
            "datafile": str(pretrain_path),
            "valid_keys": ["valid"],
            "feature_cols": ["feature"],
            "error_cols": ["error"],
            "recon_cols": ["feature"],
        },
        "model": model,
    }
    finetune_config = {
        "data": {
            "ft_datafile": str(finetune_path),
            "feature_cols": ["feature"],
            "error_cols": ["error"],
            "recon_cols": ["feature"],
            "classes": ["label"],
        },
        "model": {**model, "saved_weights": str(checkpoint_path)},
    }

    report = build_report(pretrain_config, finetune_config)

    assert not _has_blockers(report, require_cuda=False)


def test_all_training_configs_match_pretrain_checkpoint_architecture():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    for suffix in ("", ".canfar", ".narval.example"):
        with (config_dir / f"pretrain{suffix}.yaml").open() as stream:
            pretrain = yaml.safe_load(stream)
        with (config_dir / f"finetune{suffix}.yaml").open() as stream:
            finetune = yaml.safe_load(stream)

        report = build_report(pretrain, finetune)
        assert all(report["compatibility"].values())
        assert pretrain["data"]["feature_cols"] == finetune["data"]["feature_cols"]
        assert pretrain["data"]["recon_cols"] == finetune["data"]["recon_cols"]
        assert len(pretrain["data"]["error_cols"]) == len(
            pretrain["data"]["feature_cols"]
        )
        assert pretrain["data"]["error_cols"] == finetune["data"]["error_cols"]
        assert pretrain["data"]["error_cols"][:5] == [
            None,
            None,
            "e_G",
            "e_BP",
            "e_RP",
        ]
