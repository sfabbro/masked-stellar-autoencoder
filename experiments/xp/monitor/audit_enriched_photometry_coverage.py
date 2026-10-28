"""Measure actual feature/error availability, not proposed survey acquisitions."""

import json
from pathlib import Path

import h5py
import numpy as np

base = Path(__file__).resolve().parents[1] / "results-stellar-improvement"
with h5py.File(base / "enriched_feature_cohort_v1/enriched-feature-cohort.h5") as f:
    columns = json.loads(f.attrs["feature_columns_json"])
    errors = json.loads(f.attrs["error_columns_json"])
    observed = f["features/feature_observed"][:]
    eobserved = f["features/error_observed"][:]
    available = f["features/feature_available"][:]
    source = f["features/input_source"][:]
    groups = {
        "WISE": ["W1", "W2"],
        "Gaia": ["G", "BP", "RP"],
        "SMSS": ["U_SMSS", "V_SMSS", "G_SMSS", "R_SMSS", "I_SMSS", "Z_SMSS"],
        "SDSS": ["U_SDSS", "G_SDSS", "R_SDSS", "I_SDSS", "Z_SDSS"],
        "PS1": ["G_PS1", "R_PS1", "I_PS1", "Z_PS1", "Y_PS1"],
        "2MASS": ["J", "H", "KS"],
    }
    report = dict(
        cohort_sha256="321f27234bc306f64ef1a942fef981663f3c50df84990359b9943a1979262120",
        definition="All requested bands finite in actual materialized input; not survey quality or all-sky footprint. Configured error mapping only, absent error columns do not mean zero physical uncertainty.",
        by_source={},
    )
    masks = {"unique_complete_XP": available}
    for layer in ["native", "PIGS", "Li2025_CaT_silver"]:
        rows = f["layer_index/" + layer + "/feature_row"][:]
        mask = np.zeros(len(available), bool)
        mask[rows] = True
        masks[layer] = mask & available
    for layer, mask in masks.items():
        d = {
            "unique_ids": int(mask.sum()),
            "input_provenance_counts": {
                str(code): int((mask & (source == code)).sum()) for code in range(4)
            },
            "groups": {},
        }
        allphot = np.ones(len(available), bool)
        for name, bands in groups.items():
            indices = [columns.index(b) for b in bands]
            complete = observed[:, indices].all(1)
            allphot &= complete
            d["groups"][name] = {
                "all_bands_finite": int((mask & complete).sum()),
                "per_band": {
                    band: {
                        "finite_values": int((mask & observed[:, j]).sum()),
                        "configured_error_column": errors[j],
                        "finite_positive_configured_errors": int(
                            (mask & eobserved[:, j]).sum()
                        ),
                    }
                    for band, j in zip(bands, indices)
                },
            }
        d["common_all_six_photometry_groups"] = int((mask & allphot).sum())
        report["by_source"][layer] = d
    report["not_materialized_as_named_survey_inputs"] = [
        "GALEX",
        "DESILS",
        "VHS",
        "UHS",
        "NSC",
        "DELVE",
        "SPLUS",
        "PHAST",
        "ZTF",
        "KiDS",
        "HSCSSP",
        "IPHAS",
        "VVV",
        "APASS",
        "Pristine_CaHK",
    ]
    report["limitations"] = (
        "This only audits the current enriched artifact. Separate Pristine catalogue/pilot exists; missing named inputs here do not imply catalogue non-existence or zero overlap. Raw survey errors/crossmatch quality for prospective imports remain unfinished. Cross-source layer counts share stars."
    )
(base / "enriched_feature_cohort_v1/actual-photometry-coverage.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(
    json.dumps(
        {
            name: {
                "unique_ids": v["unique_ids"],
                "common_all_six": v["common_all_six_photometry_groups"],
            }
            for name, v in report["by_source"].items()
        }
    )
)
