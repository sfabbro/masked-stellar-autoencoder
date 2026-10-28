#!/usr/bin/env python3
"""Build the A1/A2/A3 anchor label layer from seismic catalogues.

Outputs two parquet tables plus a manifest:
  apokasc3.parquet   Gaia DR3 IDs, recommended ages, 16/84 bounds, stage/flags
  willett.parquet    Kepler/K2/TESS ages, 16/84 bounds, survey identifiers;
                     the released GAIADR3_ID column is NOT trusted (constant bug)
  manifest.json      counts, SHA256s, provenance and caveats

No crossmatch/imputation is done here. A later step joins Willett via KIC/TESS/
APOGEE identifiers and clusters/seismic anchors via Gaia DR3 source_id.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
WILLETT_FILES = {
    "cat_Kepler_APG.csv": "kepler",
    "cat_K2_APG_BHM.csv": "k2_bhm",
    "cat_K2_APG_COR.csv": "k2_cor",
    "cat_K2_GAL_BHM.csv": "k2_gal_bhm",
    "cat_TESS_APG.csv": "tess",
}


def sha256(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def build_apokasc3(path: Path) -> pd.DataFrame:
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
        "E_AgeRGB",
        "e_AgeRGB",
        "E_AgeRC",
        "e_AgeRC",
        "GaiaDR3",
        "FeH",
        "Teff",
        "loggSeis",
    ]:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["GaiaDR3"] = frame["GaiaDR3"].astype("Int64")
    age_columns = ["AgeRGB", "AgeRC", "E_AgeRGB", "e_AgeRGB", "E_AgeRC", "e_AgeRC"]
    frame[age_columns] = frame[age_columns].replace(-9999, np.nan)
    rgb = frame["AgeCat"].isin(["RGB", "RGB_AGB"])
    rc = frame["AgeCat"].isin(["RC", "RC/RGB"])
    out = pd.DataFrame(
        {
            "source_id": frame["GaiaDR3"],
            "teacher": "apokasc3",
            "age_scale": "apokasc3",
            "age50": np.where(
                rgb, frame["AgeRGB"], np.where(rc, frame["AgeRC"], np.nan)
            ),
            "age16": np.where(
                rgb,
                frame["AgeRGB"] + frame["e_AgeRGB"],
                np.where(rc, frame["AgeRC"] + frame["e_AgeRC"], np.nan),
            ),
            "age84": np.where(
                rgb,
                frame["AgeRGB"] + frame["E_AgeRGB"],
                np.where(rc, frame["AgeRC"] + frame["E_AgeRC"], np.nan),
            ),
            "stage": frame["AgeCat"],
            "evol_state": frame["EvolSt"],
            "cat_tab": frame["CatTab"],
            "kic": frame["KIC"],
            "feh": frame["FeH"],
            "teff": frame["Teff"],
            "logg": frame["loggSeis"],
            "source_flags": frame["Quar"].fillna(""),
        }
    )
    valid = (
        (out["age16"] > 0)
        & (out["age16"] <= out["age50"])
        & (out["age50"] <= out["age84"])
    )
    out["age_bounds_valid"] = valid
    out.loc[~valid, ["age16", "age84"]] = np.nan
    return out.dropna(subset=["source_id"]).reset_index(drop=True)


def build_willett(directory: Path) -> pd.DataFrame:
    rows = []
    for name, survey in WILLETT_FILES.items():
        path = directory / name
        if not path.is_file():
            continue
        frame = pd.read_csv(path, dtype=str)
        age50 = pd.to_numeric(frame.get("age_Dnu"), errors="coerce")
        age16 = pd.to_numeric(frame.get("age_16_Dnu"), errors="coerce")
        age84 = pd.to_numeric(frame.get("age_84_Dnu"), errors="coerce")
        fallback = pd.to_numeric(frame.get("age_L"), errors="coerce")
        fb16 = pd.to_numeric(frame.get("age_16_L"), errors="coerce")
        fb84 = pd.to_numeric(frame.get("age_84_L"), errors="coerce")
        use_l = ~np.isfinite(age50) & np.isfinite(fallback)
        age50 = np.where(use_l, fallback, age50)
        age16 = np.where(use_l, fb16, age16)
        age84 = np.where(use_l, fb84, age84)
        out = pd.DataFrame(
            {
                "teacher": "willett",
                "age_scale": f"willett_{survey}",
                "survey": survey,
                "age50": age50,
                "age16": age16,
                "age84": age84,
                "kic": frame.get("KIC_ID"),
                "tess_id": frame.get("TESS_ID"),
                "k2_id": frame.get("K2_ID"),
                "apogee_id": frame.get("APOGEE_ID"),
                "galah_id": frame.get("GALAH_ID"),
                "gaia_id_raw": frame.get("GAIADR3_ID"),
                "stage": frame.get("EvoState"),
                "flags": frame.get("warn_low_numax", "").astype(str)
                + ";"
                + frame.get("warn_high_Dnu", "").astype(str),
                "source_file": name,
            }
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apokasc3", type=Path, required=True)
    parser.add_argument("--willett-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    apokasc3 = build_apokasc3(args.apokasc3)
    willett = build_willett(args.willett_dir)
    if len(willett):
        raw_unique = willett["gaia_id_raw"].dropna().astype(str).nunique()
        willett["gaia_id_usable"] = raw_unique > 1
    else:
        willett["gaia_id_usable"] = False
    apokasc3.to_parquet(args.output / "apokasc3.parquet", index=False)
    willett.to_parquet(args.output / "willett.parquet", index=False)
    manifest = {
        "status": "complete",
        "scope": "anchor label layer; no crossmatch, no training, no truth claim",
        "apokasc3": {
            "path": str(args.apokasc3),
            "sha256": sha256(args.apokasc3),
            "rows": int(len(apokasc3)),
            "recommended_age": int(apokasc3["age50"].notna().sum()),
            "stage_counts": apokasc3["stage"].value_counts(dropna=False).to_dict(),
            "age_scale": "apokasc3",
            "bound_conversion": "age16=age50+signed_lower_error; age84=age50+upper_error",
            "valid_positive_ordered_bounds": int(apokasc3["age_bounds_valid"].sum()),
        },
        "willett": {
            "directory": str(args.willett_dir),
            "rows": int(len(willett)),
            "per_survey": willett["survey"].value_counts().to_dict()
            if len(willett)
            else {},
            "recommended_age": int(willett["age50"].notna().sum())
            if len(willett)
            else 0,
            "gaia_id_valid": bool(willett["gaia_id_usable"].any())
            if len(willett)
            else False,
            "gaia_id_caveat": "released GAIADR3_ID is constant; crossmatch via KIC/TESS/APOGEE IDs before Gaia join",
            "age_scale": "willett_{kepler,k2_*,tess}",
        },
        "caveats": [
            "APOKASC3 and Willett are seismic/model-dependent scales, not independent truth",
            "Willett Gaia IDs unusable in the current release; anchor join is ID-crossmatch TODO",
            "cluster/globular-cluster anchors are not included yet",
        ],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output),
                "apokasc3_rows": len(apokasc3),
                "willett_rows": len(willett),
            }
        )
    )


if __name__ == "__main__":
    main()
