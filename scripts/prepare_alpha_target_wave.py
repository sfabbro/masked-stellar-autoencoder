"""Independently audit every projected DESI alpha row, then prepare real targets.

CPU-only preparation of existing data. No source acquisitions, alpha imputation,
error floors, cross-survey recalibration or fitting. Every raw projected field is
checked against the SHA-pinned original FITS before the feature/identity join.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import run_xp_representation_probes as probe
import torch
import yaml
from astropy.io import fits
from prepare_posterior_target_wave import PARENT_SHA, role_indices, sha

from masked_stellar_autoencoder.models.checkpoint_load import torch_load_trusted

RAW_SHA = "eae3b31807c58ac340c257e06f66935d01bb81698a08108e6ca57a372cab76b5"
FEATURE_SHA = "321f27234bc306f64ef1a942fef981663f3c50df84990359b9943a1979262120"
REFERENCE_SHA = "fe377385e9a5a26a1d3f14ca15ea6c56904d73c34233fb9e17da1b944b76b1ad"


def validate_unit_policy(policy, columns):
    required = {
        "source_sha256": RAW_SHA,
        "native_extension": "RVTAB",
        "native_quantity_column": "ALPHAFE",
        "native_error_column": "ALPHAFE_ERR",
        "unit": "dex",
        "definition_is_apogee_alpha_m": False,
    }
    if any(policy.get(k) != v for k, v in required.items()):
        raise ValueError("Alpha native definition/unit/source policy differs")
    if not str(policy.get("paper_url", "")).startswith(
        "https://arxiv.org/html/2505.14787v4"
    ) or not policy.get("unit_evidence"):
        raise ValueError("Published alpha unit evidence is required")
    for key in ("ALPHAFE", "ALPHAFE_ERR"):
        native_unit = columns[key].get("unit")
        if native_unit in (None, ""):
            if (
                policy.get("native_header_blank_allowed_with_this_explicit_policy")
                is not True
            ):
                raise ValueError(
                    "Blank native alpha unit lacks explicit published policy"
                )
        elif str(native_unit).strip().lower() != "dex":
            raise ValueError(
                "Native alpha unit conflicts with published dex coordinate"
            )
    return {key: columns[key].get("unit") for key in ("ALPHAFE", "ALPHAFE_ERR")}


def independently_audit_projection(raw, sidecar, *, policy, chunk_rows=65536):
    if sha(raw) != RAW_SHA:
        raise ValueError("Original DESI source SHA changed")
    counts = {
        "all_rows_checked": 0,
        "eligible_rows": 0,
        "eligible_raw_feh_below_minus2": 0,
    }
    with fits.open(raw, memmap=True) as hdul, h5py.File(sidecar) as h5:
        columns = {c.name: {"unit": c.unit} for c in hdul["RVTAB"].columns}
        native_units = validate_unit_policy(policy, columns)
        rv, gaia, fiber, sp = [
            hdul[n].data for n in ("RVTAB", "GAIA", "FIBERMAP", "SPTAB")
        ]
        saved = h5["measurements"]
        if any(len(v) != len(rv) for v in (gaia, fiber, sp, saved)):
            raise ValueError("Original/projected extension row count mismatch")
        for start in range(0, len(rv), chunk_rows):
            end = min(start + chunk_rows, len(rv))
            r, g, expected = rv[start:end], gaia[start:end], saved[start:end]
            for other in (fiber[start:end], sp[start:end]):
                np.testing.assert_array_equal(r["TARGETID"], other["TARGETID"])
            direct = {
                "source_id": g["SOURCE_ID"],
                "target_id": r["TARGETID"],
                "source_row": np.arange(start, end),
                "alpha_fe": r["ALPHAFE"],
                "alpha_fe_error": r["ALPHAFE_ERR"],
                "feh_raw": r["FEH"],
                "feh_error": r["FEH_ERR"],
                "teff": r["TEFF"],
                "logg": r["LOGG"],
                "rvs_warn": r["RVS_WARN"],
                "success": r["SUCCESS"],
                "primary": r["PRIMARY"],
                "snr_r": r["SN_R"],
            }
            for key, value in direct.items():
                np.testing.assert_array_equal(
                    expected[key], value, err_msg=f"Raw alpha projection changed: {key}"
                )
            # Independent unit-vector separation (producer uses haversine).
            ra1, dec1, ra2, dec2 = map(
                np.deg2rad, (r["TARGET_RA"], r["TARGET_DEC"], g["RA"], g["DEC"])
            )
            u = np.column_stack(
                (np.cos(dec1) * np.cos(ra1), np.cos(dec1) * np.sin(ra1), np.sin(dec1))
            )
            v = np.column_stack(
                (np.cos(dec2) * np.cos(ra2), np.cos(dec2) * np.sin(ra2), np.sin(dec2))
            )
            separation = (
                np.rad2deg(
                    np.arctan2(
                        np.linalg.norm(np.cross(u, v), axis=1), np.sum(u * v, axis=1)
                    )
                )
                * 3600
            )
            np.testing.assert_allclose(
                expected["match_sep_arcsec"],
                separation,
                rtol=1e-8,
                atol=1e-8,
                equal_nan=True,
            )
            star = np.char.strip(np.asarray(r["RR_SPECTYPE"]).astype(str)) == "STAR"
            np.testing.assert_array_equal(expected["stellar_class"], star)
            valid = (
                (g["SOURCE_ID"] > 0)
                & np.isfinite(r["ALPHAFE"])
                & np.isfinite(r["ALPHAFE_ERR"])
                & (r["ALPHAFE_ERR"] > 0)
            )
            valid &= (
                np.isfinite(separation)
                & (expected["match_sep_arcsec"] <= 1)
                & (r["RVS_WARN"] == 0)
                & r["SUCCESS"].astype(bool)
                & r["PRIMARY"].astype(bool)
            )
            valid &= np.isfinite(r["SN_R"]) & (r["SN_R"] >= 10) & star
            np.testing.assert_array_equal(expected["eligible"], valid)
            counts["all_rows_checked"] += end - start
            counts["eligible_rows"] += int(valid.sum())
            counts["eligible_raw_feh_below_minus2"] += int(
                (valid & (r["FEH"] < -2)).sum()
            )
    return dict(
        status="all_original_projection_rows_independently_verified",
        source_sha256=RAW_SHA,
        sidecar_sha256=sha(sidecar),
        independent_verifier_sha256=sha(Path(__file__)),
        counts=counts,
        native_header_units=native_units,
        effective_coordinate="DESI RVSpecFit native [alpha/Fe] in dex via explicit published policy",
    )


def prepare(projection_directory, base, output, unit_evidence, checkpoint_path):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    source_revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    (output / "manifest.json").write_text(
        json.dumps(
            dict(
                status="preparing",
                source_revision=source_revision,
                checkpoint_path=str(checkpoint_path),
            )
        )
    )
    source_manifest = json.loads((projection_directory / "manifest.json").read_text())
    policy = json.loads(unit_evidence.read_text())
    if (
        source_manifest.get("status")
        != "native_source_projection_complete_not_training_certificate"
        or source_manifest.get("source_sha256") != RAW_SHA
    ):
        raise ValueError("Actual alpha source projection is not complete/pinned")
    raw = base / "desi_collection_v1/mwsall-pix-iron.fits"
    sidecar = projection_directory / "desi-alpha-native.h5"
    if sha(sidecar) != source_manifest["sidecar_sha256"]:
        raise ValueError("Actual alpha projection checksum changed")
    feature = base / "enriched_feature_cohort_v1/enriched-feature-cohort.h5"
    reference = base / "distance_feature_cohort_v1/distance-feature-cohort.h5"
    parent = Path(checkpoint_path)
    if (
        sha(feature) != FEATURE_SHA
        or sha(reference) != REFERENCE_SHA
        or sha(parent) != PARENT_SHA
    ):
        raise ValueError("Alpha feature/reference/parent source pin changed")
    audit = independently_audit_projection(raw, sidecar, policy=policy)
    (output / "independent-source-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    with h5py.File(feature) as h5:
        fids = h5["features/source_id"][:]
        if fids.dtype.kind not in "iu" or len(np.unique(fids)) != len(fids):
            raise ValueError("Feature identities are not unique integers")
        index = {int(s): i for i, s in enumerate(fids)}
        available = h5["features/feature_available"][:].astype(bool)
        partitions = h5["features/benchmark_partition"][:]
        with h5py.File(sidecar) as s:
            data = s["measurements"][:]
        eligible = data["eligible"] & np.isfinite(data["feh_raw"])
        rows = np.flatnonzero(eligible & np.isin(data["source_id"], fids))
        # Canonical first original source row for repeated Gaia IDs; never choose
        # by favorable label, residual or error.
        _, first = np.unique(data["source_id"][rows], return_index=True)
        rows = rows[np.sort(first)]
        feature_rows = np.array([index[int(s)] for s in data["source_id"][rows]])
        rows, feature_rows = (
            rows[available[feature_rows]],
            feature_rows[available[feature_rows]],
        )
        ids, partition = data["source_id"][rows], partitions[feature_rows]
        bucket = np.array(
            [
                int.from_bytes(
                    hashlib.sha256(str(int(s)).encode()).digest()[:4], "little"
                )
                % 10
                for s in ids
            ]
        )
        recommended = data["feh_raw"][rows] > -2
        masks = dict(
            strict_fit=(bucket < 6) & (partition != 1) & recommended,
            tune=(bucket >= 6) & (bucket < 8) & (partition != 1) & recommended,
            identity_test=bucket >= 8,
        )
        with h5py.File(reference) as ref:
            exclusions = ref["ALL_reference_holdout_source_ids"][:]
        # Reference union excluded all roles to keep any future independent
        # reference comparison distinct from this native-label prediction split.
        excluded = np.isin(ids, exclusions)
        for key in masks:
            masks[key] &= ~excluded
        roles = role_indices(ids, masks, caps=(8192, 2048, 2048, 4096))
        selected = roles != ""
        ids, roles, rows, feature_rows, partition = (
            ids[selected],
            roles[selected],
            rows[selected],
            feature_rows[selected],
            partition[selected],
        )
        order = np.argsort(feature_rows)
        raw_features = np.empty((len(ids), h5["features/values"].shape[1]))
        raw_features[order] = h5["features/values"][feature_rows[order]]
        names = json.loads(h5.attrs["feature_columns_json"])
    checkpoint = torch_load_trusted(parent, map_location="cpu", weights_only=False)
    signature = checkpoint["run_signature"]
    if names != signature["feature_cols"]:
        raise ValueError("Parent86 alpha feature names changed")
    values = (raw_features - np.asarray(signature["scaler_center"])) / np.asarray(
        signature["scaler_scale"]
    )
    allowed = np.array(
        [
            n in ("W1", "W2", "G", "BP", "RP") or n.startswith(("bp_", "rp_"))
            for n in names
        ]
    )
    values[:, ~allowed] = -9999
    values[(~np.isfinite(values)) | (raw_features == -9999)] = -9999
    model = probe.load_model(
        yaml.safe_load(Path("configs/pretrain.canfar.yaml").read_text()),
        checkpoint,
        torch.device("cpu"),
    )
    model.eval()
    torch.set_num_threads(4)
    with torch.no_grad():
        embeddings = torch.cat(
            [
                model.encoder(torch.tensor(values[i : i + 128], dtype=torch.float32))
                for i in range(0, len(values), 128)
            ]
        ).numpy()
    np.savez_compressed(
        output / "prepared.npz",
        source_id=ids,
        role=roles,
        embeddings=embeddings,
        target=data["alpha_fe"][rows, None],
        quoted_error=data["alpha_fe_error"][rows, None],
        source_row=data["source_row"][rows],
        raw_feh=data["feh_raw"][rows],
        benchmark_partition=partition,
    )
    (output / "independent-source-audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    manifest = dict(
        status="source_audited_alpha_features_complete",
        source_revision=source_revision,
        checkpoint_path=str(parent),
        input_sha256=sha(output / "prepared.npz"),
        encoder_checkpoint_sha256=PARENT_SHA,
        source_audit_sha256=sha(output / "independent-source-audit.json"),
        source_provenance={
            str(p): sha(p) for p in (sidecar, feature, reference, unit_evidence)
        },
        role_counts={
            r: int((roles == r).sum()) for r in ("fit", "tune", "cal", "test")
        },
        target_names=["desi_rvs_alpha_fe"],
        target_units=["dex"],
        coordinate="linear_native_bracket_abundance",
        unit_evidence=policy,
        feature_policy="Parent86 photometry+XP only; no astrometry/EBV or source labels enter encoder",
        target_estimand="Single DESI RVSpecFit native alpha/Fe template-fit measurement scale, not APOGEEalpha/M or physical truth",
        family_holdout="ExactGaiaID hashes across repeated DESItargets; canonicalfirstsource row;99574reference union excluded all roles; benchmarkvalidation excludedFIT/TUNE",
        primary_fit_tune_minimum_raw_feh_exclusive=-2,
        metal_poor_test_labels_recommended=False,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--projection-directory", type=Path, required=True)
    p.add_argument(
        "--base",
        type=Path,
        default=Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--unit-evidence",
        type=Path,
        default=Path("configs/desi-alpha-source-policy-20261023.json"),
    )
    a = p.parse_args()
    try:
        prepare(a.projection_directory, a.base, a.output, a.unit_evidence, a.checkpoint)
    except Exception as error:
        if a.output.is_dir():
            path = a.output / "manifest.json"
            state = json.loads(path.read_text()) if path.exists() else {}
            state.update(
                status="failed", error=str(error), exception=type(error).__name__
            )
            path.write_text(json.dumps(state, indent=2))
        raise


if __name__ == "__main__":
    main()
