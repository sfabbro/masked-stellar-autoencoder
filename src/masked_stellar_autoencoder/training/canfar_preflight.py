"""Inspect a CANFAR training environment and report catalogue schema gaps."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import h5py
import torch
import yaml
from astropy.io import fits

from .config_paths import expand_config_paths


def _candidate_error_columns(feature: str) -> list[str]:
    """Return conventional names to check, without treating them as a mapping."""
    if match := re.fullmatch(r"bp_(\d+)", feature):
        return [f"bpe_{match.group(1)}"]
    if match := re.fullmatch(r"rp_(\d+)", feature):
        return [f"rpe_{match.group(1)}"]
    special = {
        "G": ["e_G"],
        "BP": ["e_BP"],
        "RP": ["e_RP"],
        "PARALLAX": ["e_parallax", "parallax_error"],
        "pmra": ["e_pmra", "pmra_error"],
        "pmdec": ["e_pmdec", "pmdec_error"],
        "RA": ["ra_error"],
        "DEC": ["dec_error"],
    }
    if feature in {"G", "BP", "RP"}:
        return special[feature]
    return list(
        dict.fromkeys(
            [
                *special.get(feature, []),
                f"E_{feature}",
                f"e_{feature}",
                f"{feature}_error",
                f"{feature}_err",
            ]
        )
    )


def _missing_columns(
    features: list[str], errors: list[str | None], available: set[str]
) -> list[dict[str, Any]]:
    missing = []
    for index, feature in enumerate(features):
        error = errors[index] if index < len(errors) else None
        if error is None and len(errors) == len(features):
            continue
        if error in available:
            continue
        candidates = [
            name for name in _candidate_error_columns(feature) if name in available
        ]
        missing.append(
            {"feature": feature, "configured_error": error, "candidates": candidates}
        )
    return missing


def _load_config(path: str) -> dict[str, Any]:
    with Path(path).open() as stream:
        config = yaml.safe_load(stream)
    expand_config_paths(config)
    return config


def _inspect_hdf5(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config["data"]
    path = Path(data_config["datafile"])
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "valid_keys": data_config.get("valid_keys", []),
        "missing_valid_keys": [],
        "fields": [],
        "missing_features": [],
        "missing_recon_cols": [],
        "missing_configured_errors": [],
        "unavailable_errors": [],
        "unmapped_errors": [],
        "error_count_matches_features": False,
        "error_list_matches_features": False,
    }
    if not path.is_file():
        return result

    with h5py.File(path, "r") as h5:
        keys = list(h5.keys())
        result["missing_valid_keys"] = [
            key for key in result["valid_keys"] if key not in h5
        ]
        key = next((key for key in keys if key not in result["valid_keys"]), None)
        key = key or next((key for key in keys if key in result["valid_keys"]), None)
        if key is None:
            result["empty_hdf5"] = True
            return result

        result["sample_key"] = key
        fields = set(h5[key].dtype.names or ())
        result["fields"] = sorted(fields)
        features = data_config.get("feature_cols", [])
        recon_cols = data_config.get("recon_cols", [])
        errors = data_config.get("error_cols", []) or []
        result["missing_features"] = sorted(set(features) - fields)
        result["missing_recon_cols"] = sorted(set(recon_cols) - fields)
        result["missing_configured_errors"] = sorted(
            {error for error in errors if error is not None} - fields
        )
        result["unavailable_errors"] = [
            feature
            for feature, error in zip(features, errors, strict=False)
            if error is None
        ]
        result["unmapped_errors"] = _missing_columns(features, errors, fields)
        result["error_count_matches_features"] = len(errors) == len(features)
        result["error_list_matches_features"] = any(
            error == feature
            for feature, error in zip(features, errors, strict=False)
            if error is not None
        )
    return result


def _inspect_fits(config: dict[str, Any]) -> dict[str, Any]:
    data_config = config["data"]
    path = Path(data_config["ft_datafile"])
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "fields": [],
        "missing_features": [],
        "missing_recon_cols": [],
        "missing_classes": [],
        "missing_configured_errors": [],
        "unavailable_errors": [],
        "unmapped_errors": [],
        "error_count_matches_features": False,
        "error_list_matches_features": False,
    }
    if not path.is_file():
        return result

    with fits.open(path, memmap=True) as hdul:
        table_hdu = next(
            (hdu for hdu in hdul if getattr(hdu, "columns", None) is not None), None
        )
        if table_hdu is None:
            result["has_table_hdu"] = False
            return result
        fields = set(table_hdu.columns.names)

    features = data_config.get("feature_cols", [])
    recon_cols = data_config.get("recon_cols", [])
    errors = data_config.get("error_cols", []) or []
    classes = data_config.get("classes", [])
    result["fields"] = sorted(fields)
    result["missing_features"] = sorted(set(features) - fields)
    result["missing_recon_cols"] = sorted(set(recon_cols) - fields)
    result["missing_classes"] = sorted(set(classes) - fields)
    result["missing_configured_errors"] = sorted(
        {error for error in errors if error is not None} - fields
    )
    result["unavailable_errors"] = [
        feature
        for feature, error in zip(features, errors, strict=False)
        if error is None
    ]
    result["unmapped_errors"] = _missing_columns(features, errors, fields)
    result["error_count_matches_features"] = len(errors) == len(features)
    result["error_list_matches_features"] = any(
        error == feature
        for feature, error in zip(features, errors, strict=False)
        if error is not None
    )
    return result


def build_report(
    pretrain_config: dict[str, Any], finetune_config: dict[str, Any] | None
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "runtime": {
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "pretrain": _inspect_hdf5(pretrain_config),
    }
    if finetune_config is None:
        return report

    report["finetune"] = _inspect_fits(finetune_config)
    checkpoint_path = Path(finetune_config["model"]["saved_weights"])
    report["finetune_checkpoint"] = {
        "path": str(checkpoint_path),
        "exists": checkpoint_path.is_file(),
    }
    pre_data = pretrain_config["data"]
    ft_data = finetune_config["data"]
    pre_model = pretrain_config.get("model", {})
    ft_model = finetune_config.get("model", {})
    compatibility = {
        "feature_cols_match": pre_data.get("feature_cols")
        == ft_data.get("feature_cols"),
        "recon_cols_match": pre_data.get("recon_cols") == ft_data.get("recon_cols"),
    }
    for key in (
        "layer_dims",
        "decoder_dims",
        "rtdl_embed",
        "pt_activ_func",
        "norm",
        "encoder_type",
        "growth_rate",
        "num_dense_layers",
        "cosine_latent",
    ):
        compatibility[f"{key}_match"] = pre_model.get(key) == ft_model.get(key)
    compatibility["heteroscedastic_match"] = pretrain_config.get("training", {}).get(
        "heteroscedastic", False
    ) == finetune_config.get("training", {}).get("heteroscedastic", False)
    report["compatibility"] = compatibility
    return report


def _has_blockers(report: dict[str, Any], require_cuda: bool) -> bool:
    if require_cuda and not report["runtime"]["cuda_available"]:
        return True
    for stage in ("pretrain", "finetune"):
        data = report.get(stage)
        if data is None:
            continue
        if not data["exists"] or data.get("empty_hdf5"):
            return True
        if data.get("has_table_hdu") is False:
            return True
        if (
            data.get("missing_valid_keys")
            or data.get("missing_features")
            or data.get("missing_recon_cols")
            or data.get("missing_classes")
            or data.get("missing_configured_errors")
            or data.get("unmapped_errors")
            or not data.get("error_count_matches_features", True)
            or data.get("error_list_matches_features", False)
        ):
            return True
    if "finetune" in report and not report["finetune_checkpoint"]["exists"]:
        return True
    compatibility = report.get("compatibility", {})
    return not all(compatibility.values()) if compatibility else False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check CANFAR CUDA, training data headers, and stage compatibility"
    )
    parser.add_argument("--pretrain-config", default="configs/pretrain.canfar.yaml")
    parser.add_argument("--finetune-config", default="configs/finetune.canfar.yaml")
    parser.add_argument(
        "--skip-finetune", action="store_true", help="Check only pretraining inputs"
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail unless this process can see a CUDA GPU",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Report headers and uncertainty candidates without failing for known config gaps",
    )
    parser.add_argument("--output", type=Path, help="Also write the JSON report here")
    args = parser.parse_args()

    pretrain_config = _load_config(args.pretrain_config)
    finetune_path = Path(args.finetune_config)
    finetune_config = (
        _load_config(str(finetune_path))
        if finetune_path.is_file() and not args.skip_finetune
        else None
    )
    report = build_report(pretrain_config, finetune_config)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")

    if _has_blockers(report, args.require_cuda) and not args.schema_only:
        sys.exit(1)


if __name__ == "__main__":
    main()
