#!/usr/bin/env python
"""Read-only identity, label-distribution, and overlap audit of saved sources.

This script does not import or execute a stellar model. It only reads three
previously preserved, SHA-verified numerical source files and writes one JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "census": (
        ROOT
        / "experiments/xp/results-stellar-improvement/remediation-wave-20261026/desi-cache-census-preservation-20261027/census/identities.npz",
        "df3250519acf3280a50ad6227a2be17c53cceaa3ed811fc5729ea39940e41a24",
    ),
    "desi_raw_fields": (
        ROOT
        / "experiments/xp/results-stellar-improvement/desi_labels_v1/desi-labels.h5",
        "e3f45f89220e5d403dfd06569e5c3ce4a1adc1ef8ab2bc1af8b701b9cf37693f",
    ),
    "native_sidecar": (
        ROOT
        / "experiments/xp/results-stellar-improvement/joint-native-preparation-v2/joint-native-labels.h5",
        "3998fe216ed89b23dc450fbeec3c04462d2d8def35dafe9e4ec2679b7deba85b",
    ),
}


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantiles(values, levels):
    return [float(v) for v in np.percentile(values, levels)]


def residual_stats(native, desi, nerr, derr, separation, mask):
    diff = native[mask] - desi[mask]
    if len(diff) == 0:
        return {"n": 0}
    med = float(np.median(diff))
    err_ok = (
        np.isfinite(nerr[mask])
        & (nerr[mask] > 0)
        & np.isfinite(derr[mask])
        & (derr[mask] > 0)
    )
    nsigma = np.sqrt(nerr[mask][err_ok] ** 2 + derr[mask][err_ok] ** 2)
    return {
        "n": len(diff),
        "median_native_minus_desi_dex": med,
        "mean_native_minus_desi_dex": float(np.mean(diff)),
        "robust_1p4826_MAD_dex": float(1.4826 * np.median(np.abs(diff - med))),
        "rmse_dex": float(np.sqrt(np.mean(diff**2))),
        "p90_absolute_difference_dex": float(np.quantile(np.abs(diff), 0.9)),
        "median_desi_gaia_separation_arcsec": float(np.median(separation[mask])),
        "positive_finite_quoted_pair_errors": int(err_ok.sum()),
        "median_quoted_independence_pair_sigma_dex": float(np.median(nsigma))
        if len(nsigma)
        else None,
        "fraction_over_3_quoted_independence_pair_sigma": float(
            np.mean(np.abs(diff[err_ok]) > 3 * nsigma)
        )
        if len(nsigma)
        else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = {}
    for name, (path, expected) in FILES.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"{name}: SHA mismatch")
        inputs[name] = {
            "path": str(path),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    with zipfile.ZipFile(FILES["census"][0]) as zf:
        if zf.testzip() is not None:
            raise ValueError("Census NPZ CRC check failed")

    with np.load(FILES["census"][0], allow_pickle=False) as z:
        keep = (
            (z["quality_rows"] == 1)
            & (z["base_role"] == "fit")
            & z["base_primary_input"]
            & z["observed"][:, 1]
        )
        ids = np.sort(z["source_id"][keep].copy())
    if len(np.unique(ids)) != len(ids) or np.any(ids <= 0):
        raise ValueError("Census subset identities are not positive and unique")

    fields = [
        "source_row",
        "teff",
        "logg",
        "feh_raw",
        "teff_err_native",
        "logg_err_native",
        "feh_err_native",
        "snr_r",
        "correction_valid",
        "eligible",
        "match_sep_arcsec",
        "rvs_warn",
        "success",
        "primary",
    ]
    counts = np.zeros(len(ids), dtype=np.uint32)
    vals = {
        k: np.full(len(ids), -1, np.int64)
        if k == "source_row"
        else np.full(len(ids), np.nan)
        for k in fields
    }
    source_row_failures = 0
    with h5py.File(FILES["desi_raw_fields"][0], "r") as f:
        d = f["measurements"]
        for start in range(0, len(d), 65536):
            stop = min(start + 65536, len(d))
            chunk = d.fields(["source_id"] + fields)[start:stop]
            positions = np.searchsorted(ids, chunk["source_id"])
            hit = positions < len(ids)
            j = np.flatnonzero(hit)
            hit[j] &= ids[positions[j]] == chunk["source_id"][j]
            chunk_indices = np.arange(start, stop, dtype=np.int64)[hit]
            chunk, positions = chunk[hit], positions[hit]
            source_row_failures += int(
                np.count_nonzero(chunk["source_row"] != chunk_indices)
            )
            np.add.at(counts, positions, 1)
            for k in fields:
                vals[k][positions] = chunk[k]
    if source_row_failures:
        raise ValueError("Original DESI source_row differs from saved HDF table row")
    only = counts == 1
    if not np.isfinite(vals["logg"][only]).all():
        raise ValueError("Census observed-logg mask includes nonfinite saved label")
    if (
        np.any(vals["rvs_warn"][only] != 0)
        or np.any(vals["success"][only] != 1)
        or np.any(vals["primary"][only] != 1)
        or np.any(vals["snr_r"][only] < 10)
        or np.any(vals["match_sep_arcsec"][only] > 1)
    ):
        raise ValueError("Saved quality fields contradict the selected cohort")

    g = vals["logg"][only]
    ge = vals["logg_err_native"][only]
    err_ok = np.isfinite(ge) & (ge > 0)
    summary = {
        "census_subset_ids": len(ids),
        "exactly_one_row_in_entire_raw_field_table": int(only.sum()),
        "more_than_one_row_in_entire_raw_field_table": int((counts > 1).sum()),
        "no_row_in_entire_raw_field_table": int((counts == 0).sum()),
        "source_row_identity_mismatches": source_row_failures,
        "logg_bins_dex": {
            "lt_0": int((g < 0).sum()),
            "0_to_2": int(((g >= 0) & (g < 2)).sum()),
            "2_to_3": int(((g >= 2) & (g < 3)).sum()),
            "3_to_4": int(((g >= 3) & (g < 4)).sum()),
            "4_to_5": int(((g >= 4) & (g < 5)).sum()),
            "ge_5": int((g >= 5).sum()),
        },
        "logg_percentiles_0_1_5_50_95_99_100_dex": quantiles(
            g, [0, 1, 5, 50, 95, 99, 100]
        ),
        "known_positive_finite_logg_random_errors": int(err_ok.sum()),
        "unknown_or_nonpositive_logg_random_errors": int((~err_ok).sum()),
        "positive_logg_error_percentiles_1_5_50_95_99_dex": quantiles(
            ge[err_ok], [1, 5, 50, 95, 99]
        ),
        "older_corrected_feh_validity_box": int(
            (vals["correction_valid"][only] > 0).sum()
        ),
        "older_strict_feh_policy_eligible": int((vals["eligible"][only] > 0).sum()),
    }

    with h5py.File(FILES["native_sidecar"][0], "r") as f:
        native_id = f["source_id"][:]
        value = f["targets/value"][:]
        uncertainty = f["targets/uncertainty"][:]
        observed = f["targets/observed"][:]
        quality = f["targets/quality_pass"][:]
        survey = f["targets/survey"][:].astype("U")
        source_row = f["targets/source_row"][:]
        pos = np.searchsorted(ids, native_id)
        hit = pos < len(ids)
        j = np.flatnonzero(hit)
        hit[j] &= ids[pos[j]] == native_id[j]
        j = np.flatnonzero(hit)
        hit[j] &= counts[pos[j]] == 1
        hit &= observed[:, 1] & quality[:, 1] & np.isfinite(value[:, 1])
        ni = np.flatnonzero(hit)
        di = pos[ni]
        native_logg, desi_logg = value[ni, 1], vals["logg"][di]
        native_err, desi_err = uncertainty[ni, 1], vals["logg_err_native"][di]
        sep, paired_survey = vals["match_sep_arcsec"][di], survey[ni]
        source_checks = {}
        for name, id_field, gravity_field in (
            ("galah", "gaiadr3_source_id", "logg"),
            ("apogee", "GAIAEDR3_SOURCE_ID", "LOGG"),
        ):
            ii = ni[paired_survey == name]
            group = f["raw/" + name]
            sr = group["source_row"][:]
            if len(np.unique(sr)) != len(sr):
                raise ValueError(f"{name} has duplicate raw source rows")
            order = np.argsort(sr)
            match = np.searchsorted(sr[order], source_row[ii])
            exists = match < len(sr)
            jj = np.flatnonzero(exists)
            exists[jj] &= sr[order[match[jj]]] == source_row[ii[jj]]
            if not exists.all():
                raise ValueError(f"{name} original row missing")
            loc = order[match]
            same_ids = group[id_field][:][loc] == native_id[ii]
            same_logg = group[gravity_field][:][loc] == value[ii, 1]
            if not same_ids.all() or not same_logg.all():
                raise ValueError(
                    f"{name} identity or native gravity association changed"
                )
            source_checks[name] = {
                "paired_rows": len(ii),
                "original_row_found": int(exists.sum()),
                "exact_native_gaia_id": int(same_ids.sum()),
                "exact_native_raw_logg": int(same_logg.sum()),
            }

    if len(ni) != sum(v["paired_rows"] for v in source_checks.values()):
        raise ValueError("Unexpected native survey in paired cohort")
    bins = [
        ("lt_0", -np.inf, 0),
        ("0_to_2", 0, 2),
        ("2_to_3", 2, 3),
        ("3_to_4", 3, 4),
        ("4_to_5", 4, 5),
        ("ge_5", 5, np.inf),
    ]
    strata = {}
    for name in ("galah", "apogee"):
        source = paired_survey == name
        strata[name] = {
            "all": residual_stats(
                native_logg, desi_logg, native_err, desi_err, sep, source
            )
        }
        for label, gravity in (("native_logg", native_logg), ("desi_logg", desi_logg)):
            strata[name][label] = {}
            for bin_name, lo, hi in bins:
                mask = source & (gravity >= lo) & (gravity < hi)
                strata[name][label][bin_name] = residual_stats(
                    native_logg, desi_logg, native_err, desi_err, sep, mask
                )

    result = {
        "scope": "Read-only audit of previously saved source arrays, not model inference, training, or a held-out performance test.",
        "inputs": inputs,
        "census_npz_crc": "pass",
        "subset_definition": "Census quality_rows==1, base_role=='fit', base_primary_input, census observed[:,1], then exactly one occurrence of that ID among every row in saved DESI raw-field table. Thus the sole raw row is the sole quality-selected DESI row. This is not precisely the 1,192,466 additions: it may include old-label overlaps and excludes IDs with extra rejected raw records.",
        "input_caveat": "The earlier DESI raw-field table and census carry the same publisher FITS SHA provenance; exact selected-sidecar content and original remote FITS were not re-read here. HDF row numbers and native saved source associations were checked; this is not a new full raw-source verification.",
        "comparison_definition": "Exact Gaia source ID, native observed and quality-passing logg; delta = native minus raw DESI in dex. Native labels are an older selected joint-native sidecar, not necessarily identical to every later remediation-source native record. FIT is not an independent test set.",
        "scientific_limitations": [
            "The overlapping stars are selected and are not representative of all FIT stars or of the full Galaxy.",
            "GALAH logg is parallax-conditioned. Survey labels, random errors, and sky matching are not independent physical truth.",
            "Binning on either noisy label can bias a conditional comparison; both directions are reported.",
            "Quoted pair-sigma summaries assume independent random errors and exclude cross-survey and model systematics; they cannot identify which survey is wrong or calibrate a unique error floor.",
            "Uncertainty source, label-domain validity, atmospheric grids, pipelines, shared priors and cross-survey correlations require further validation.",
        ],
        "subset_distribution": summary,
        "paired_cohort": {
            "total": len(ni),
            "saved_native_row_checks": source_checks,
            "strata": strata,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": sha256(args.output),
                "subset": summary["exactly_one_row_in_entire_raw_field_table"],
                "pairs": len(ni),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
