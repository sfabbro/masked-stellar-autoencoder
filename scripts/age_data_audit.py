#!/usr/bin/env python3
"""A0 age-data audit: catalogue identity, coverage, teacher overlap, SHA provenance.

No training, no model inference, no label imputation. Reads the existing APOKASC-3
table, the Willett et al. 2026 Zenodo catalogues, the current StarHorse age cohort,
and the fixed 10M benchmark ID table, then writes a JSON/CSV audit report.

The Willett release currently has a constant ``GAIADR3_ID`` column; the audit detects
and reports it instead of trusting it. Crossmatch is therefore proposed via KIC/TESS/
APOGEE identifiers, not the Gaia column.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

WILLETT_FILES = {
    "cat_Kepler_APG.csv": {
        "bytes": 9741569,
        "sha256": "5e6f04e3dc4f5e929eef20863f3a7edda18af49d4ea58f53199f0e9d85186c74",
    },
    "cat_K2_APG_BHM.csv": {
        "bytes": 14282359,
        "sha256": "d546118ddd73378d9f06abbaea5f80f02b9709110970a770eb514f68ad5c9110",
    },
    "cat_K2_APG_COR.csv": {
        "bytes": 10092059,
        "sha256": "54c5edc1dee6de4cfb68c5f39a4a12af30388f8bf32508b0ddda03727d34ec4a",
    },
    "cat_K2_GAL_BHM.csv": {
        "bytes": 8109954,
        "sha256": "6535cb2f377ef73e65299451ec5cffdb8051f6a8d7311bfb04fa61c33f2698ee",
    },
    "cat_TESS_APG.csv": {
        "bytes": 1818486,
        "sha256": "61170dbac20c9c391d51f478dfb9c4a605e31b361afb50f28f130e1d72d641af",
    },
}
WILLETT_BASE = "https://zenodo.org/records/20428851/files"

APOKASC3_COLUMNS = [
    "KIC",
    "EvolSt",
    "EvolSrc",
    "CatTab",
    "SeisSrc",
    "SpecSrc",
    "NNumax",
    "NDNu",
    "Quar",
    "Numax",
    "e_Numax",
    "DNu",
    "e_DNu",
    "fDNu",
    "e_fDNu",
    "fNumax",
    "Mass",
    "e_Mass",
    "Radius",
    "e_Radius",
    "loggSeis",
    "e_loggSeis",
    "Teff",
    "e_Teff",
    "loggSpec",
    "e_loggSpec",
    "FeH",
    "e_FeH",
    "aFe",  # codespell:ignore afe
    "e_aFe",
    "CFe",
    "e_CFe",
    "NFe",
    "e_NFe",
    "InvRGaia",
    "e_InvRGaia",
    "AgeCat",
    "AgeRGB",
    "E_AgeRGB",
    "e_AgeRGB",
    "AgeRC",
    "E_AgeRC",
    "e_AgeRC",
    "vsini",
    "alphaCat",
    "GaiaDR3",
    "dummy",
    "2MASS",
]


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def median_nmad(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype="f8")
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "median": float("nan"), "nmad": float("nan")}
    med = float(np.median(values))
    return {
        "n": int(len(values)),
        "median": med,
        "nmad": float(1.4826 * np.median(np.abs(values - med))),
    }


def grouped_offsets(frame: pd.DataFrame, column: str, group: str) -> list[dict]:
    out = []
    classified = frame[frame[group].notna()]
    for key, part in classified.groupby(group):
        stats = median_nmad((part["age_apokasc"] - part[column]).to_numpy())
        stats["group"] = str(key)
        out.append(stats)
    return out


def load_apokasc3(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep=r"\s+",
        names=APOKASC3_COLUMNS,
        engine="python",
        dtype=str,
        compression="infer",
    )
    for name in [
        "KIC",
        "AgeRGB",
        "AgeRC",
        "FeH",
        "Teff",
        "loggSeis",
        "GaiaDR3",
        "E_AgeRGB",
        "e_AgeRGB",
        "E_AgeRC",
        "e_AgeRC",
    ]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["GaiaDR3"] = frame["GaiaDR3"].astype("Int64")
    rgb = frame["AgeCat"].isin(["RGB", "RGB_AGB"])
    rc = frame["AgeCat"].isin(["RC", "RC/RGB"])
    frame["age_apokasc"] = np.where(
        rgb, frame["AgeRGB"], np.where(rc, frame["AgeRC"], np.nan)
    )
    frame["age_apokasc_upper"] = np.where(
        rgb, frame["E_AgeRGB"], np.where(rc, frame["E_AgeRC"], np.nan)
    )
    frame["age_apokasc_lower"] = np.where(
        rgb, frame["e_AgeRGB"], np.where(rc, frame["e_AgeRC"], np.nan)
    )
    return frame


def download_willett(directory: Path, timeout: int = 120) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    for name, expected in WILLETT_FILES.items():
        destination = directory / name
        if not destination.is_file():
            url = f"{WILLETT_BASE}/{name}?download=1"
            urllib.request.urlretrieve(url, destination)  # noqa: S310 — pinned HTTPS URL
        actual = sha256(destination)
        if (
            destination.stat().st_size != expected["bytes"]
            or actual != expected["sha256"]
        ):
            raise ValueError(f"Willett file {name} differs from pinned size/SHA256")
        digests[name] = actual
    return digests


def load_willett(directory: Path) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    frames: dict[str, pd.DataFrame] = {}
    provenance: list[dict] = []
    for name in WILLETT_FILES:
        path = directory / name
        frame = pd.read_csv(path, dtype=str)
        frames[name] = frame
        provenance.append(
            {
                "file": name,
                "bytes": int(path.stat().st_size),
                "sha256": sha256(path),
                "rows": int(len(frame)),
                "columns": int(len(frame.columns)),
            }
        )
    return frames, provenance


def willett_coverage(
    frames: dict[str, pd.DataFrame], willett_dir: Path
) -> tuple[dict, pd.DataFrame]:
    """Per-file counts, constant-column detection, and age-column completeness."""
    coverage: dict[str, dict] = {}
    for name, frame in frames.items():
        age_columns = [c for c in frame.columns if c.startswith("age_")]
        id_columns = [
            c
            for c in frame.columns
            if c.endswith("_ID")
            or c in {"KIC_ID", "TESS_ID", "GAIADR3_ID", "APOGEE_ID", "GALAH_ID"}
        ]
        constant_ids = {}
        for column in id_columns:
            values = frame[column].dropna().astype(str)
            if len(values) and values.nunique() == 1:
                constant_ids[column] = str(values.iloc[0])
        age_complete = {}
        for column in age_columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            age_complete[column] = int(numeric.notna().sum())
        coverage[name] = {
            "rows": int(len(frame)),
            "age_columns": age_columns,
            "age_non_null": age_complete,
            "constant_identifier_columns": constant_ids,
            "unique_kic": int(
                pd.to_numeric(frame.get("KIC_ID"), errors="coerce").nunique()
            )
            if "KIC_ID" in frame
            else 0,
            "unique_tess": int(
                pd.to_numeric(frame.get("TESS_ID"), errors="coerce").nunique()
            )
            if "TESS_ID" in frame
            else 0,
            "unique_apogee_id": int(frame["APOGEE_ID"].nunique())
            if "APOGEE_ID" in frame
            else 0,
        }
    # APOKASC3 vs Willett Kepler teacher comparison
    kepler = frames.get("cat_Kepler_APG.csv")
    comparison = pd.DataFrame()
    if kepler is not None:
        kepler = kepler.copy()
        for column in ["KIC_ID", "age_Dnu", "age_L", "Fe_H"]:
            if column in kepler:
                kepler[column] = pd.to_numeric(kepler[column], errors="coerce")
        comparison = kepler
    return coverage, comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apokasc3", type=Path, required=True)
    parser.add_argument("--willett-dir", type=Path, required=True)
    parser.add_argument("--age-cohort", type=Path, required=True)
    parser.add_argument("--dataset-view", type=Path, required=True)
    parser.add_argument("--posterior-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--download", action="store_true", help="Download Willett files if absent"
    )
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=False)
    if args.download:
        download_willett(args.willett_dir)

    apokasc3 = load_apokasc3(args.apokasc3)
    wil_frames, wil_provenance = load_willett(args.willett_dir)
    wil_coverage, kepler = willett_coverage(wil_frames, args.willett_dir)

    with h5py.File(args.dataset_view) as f:
        train_ids = set(np.asarray(f["train/source_id"][:], dtype="i8").tolist())
        val_ids = set(np.asarray(f["validation/source_id"][:], dtype="i8").tolist())
    with h5py.File(args.age_cohort) as f:
        selected_ids = np.asarray(f["selected/source_id"][:], dtype="i8")
        selected_survey = np.asarray(f["selected/survey"][:]).astype(str)
        selected_age = np.asarray(f["selected/age_percentiles"][:], dtype="f8")
        strict = np.asarray(f["selected/strict_fit"][:], dtype=bool)
    with h5py.File(args.posterior_labels) as f:
        starhorse_apo = np.asarray(f["apogee17/source_id"][:], dtype="i8")
        starhorse_gal = np.asarray(f["galahdr3/source_id"][:], dtype="i8")
        starhorse_age_flags = (
            np.unique(f["apogee17/flags/AgeFlag"][:]).astype(str).tolist()
        )

    apo_ids = set(apokasc3["GaiaDR3"].dropna().astype("int64").tolist())
    report: dict = {
        "status": "complete",
        "scope": "A0 age-data audit only; no training, no imputation, no truth claim",
        "apokasc3": {
            "path": str(args.apokasc3),
            "sha256": sha256(args.apokasc3),
            "rows": int(len(apokasc3)),
            "unique_gaia": int(len(apo_ids)),
            "recommended_age_rows": int(apokasc3["age_apokasc"].notna().sum()),
            "age_class_counts": apokasc3["AgeCat"].value_counts().to_dict(),
            "evol_state_counts": apokasc3["EvolSt"].value_counts().to_dict(),
            "in_10m_train": int(len(apo_ids & train_ids)),
            "in_10m_validation": int(len(apo_ids & val_ids)),
            "in_current_age_cohort": int(len(apo_ids & set(selected_ids.tolist()))),
            "in_raw_apogee17_source_id": int(
                len(apo_ids & set(starhorse_apo.tolist()))
            ),
        },
        "current_age_cohort": {
            "rows": int(len(selected_ids)),
            "strict_fit": int(strict.sum()),
            "surveys": {
                str(k): int(v)
                for k, v in pd.Series(selected_survey).value_counts().items()
            },
            "median_age_gyr": float(np.nanmedian(selected_age[:, 2])),
        },
        "starhorse": {
            "age_flags": starhorse_age_flags,
            "raw_apogee_rows": int(len(starhorse_apo)),
            "raw_galah_rows": int(len(starhorse_gal)),
        },
        "willett": {
            "provenance": wil_provenance,
            "coverage": wil_coverage,
        },
    }

    # Teacher comparison APOKASC3 vs Willett Kepler by KIC
    if len(kepler):
        merged = apokasc3.merge(kepler, left_on="KIC", right_on="KIC_ID", how="inner")
        teacher: dict = {"matched_kic": int(len(merged))}
        for column in ["age_Dnu", "age_L"]:
            valid = merged[
                np.isfinite(merged["age_apokasc"])
                & np.isfinite(merged[column])
                & (merged["age_apokasc"] > 0)
                & (merged[column] > 0)
            ].copy()
            values = valid["age_apokasc"] - valid[column]
            teacher[column] = {
                "overall": median_nmad(values),
                "by_age": grouped_offsets(
                    valid.assign(
                        age_group=pd.cut(valid["age_apokasc"], [0, 4, 8, 12, 20, 100])
                    ),
                    column,
                    "age_group",
                ),
                "by_feh": grouped_offsets(
                    valid.assign(
                        feh_group=pd.cut(valid["Fe_H"], [-3, -1, -0.5, 0, 0.6])
                    ),
                    column,
                    "feh_group",
                ),
            }
        report["teacher_comparison_apokasc3_willett"] = teacher

    (args.output / "age-data-audit.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    pd.DataFrame(wil_provenance).to_csv(
        args.output / "willett-provenance.csv", index=False
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "apokasc3_rows": int(len(apokasc3)),
            }
        )
    )


if __name__ == "__main__":
    main()
