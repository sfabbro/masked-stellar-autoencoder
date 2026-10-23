"""Recover omitted native DESI alpha measurements from the existing pinned FITS.

CPU-only, no acquisitions or fitting. Saves every raw measurement/error and the
actual quality columns. Eligibility is a declared diagnostic policy, not a
cross-survey alpha calibration or evidence for physical-truth coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from desi_stellar_labels import RAW_SHA, ROOT, check_target_alignment, separation_arcsec


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def alpha_eligibility(
    source_id, alpha, sigma, separation, warn, success, primary, snr, star
):
    """Keep signed abundances; reject missing errors without inventing a floor."""
    return (
        (np.asarray(source_id) > 0)
        & np.isfinite(alpha)
        & np.isfinite(sigma)
        & (np.asarray(sigma) > 0)
        & np.isfinite(separation)
        & (np.asarray(separation) <= 1.0)
        & (np.asarray(warn) == 0)
        & np.asarray(success, bool)
        & np.asarray(primary, bool)
        & np.isfinite(snr)
        & (np.asarray(snr) >= 10)
        & np.asarray(star, bool)
    )


def prepare(raw, output, *, expected_sha=RAW_SHA, chunk_rows=65536):
    if output.exists():
        raise FileExistsError(output)
    if file_sha(raw) != expected_sha:
        raise ValueError("Original DESI FITS SHA differs from immutable publisher pin")
    output.mkdir(parents=True, exist_ok=False)
    counts = {"raw_rows": 0, "eligible_rows": 0, "eligible_feh_below_minus2": 0}
    dtype = np.dtype(
        [
            ("source_id", "<i8"),
            ("target_id", "<i8"),
            ("source_row", "<i8"),
            ("alpha_fe", "<f8"),
            ("alpha_fe_error", "<f8"),
            ("feh_raw", "<f8"),
            ("feh_error", "<f8"),
            ("teff", "<f8"),
            ("logg", "<f8"),
            ("match_sep_arcsec", "<f8"),
            ("rvs_warn", "<i8"),
            ("success", "?"),
            ("primary", "?"),
            ("snr_r", "<f8"),
            ("stellar_class", "?"),
            ("eligible", "?"),
        ]
    )
    destination = output / "desi-alpha-native.h5"
    with fits.open(raw, memmap=True) as hdul, h5py.File(destination, "x") as out:
        rv, gaia, fiber, sp = [
            hdul[n].data for n in ("RVTAB", "GAIA", "FIBERMAP", "SPTAB")
        ]
        if len(rv) != len(gaia):
            raise ValueError("DESI Gaia row count mismatch")
        required = {
            "ALPHAFE",
            "ALPHAFE_ERR",
            "FEH",
            "FEH_ERR",
            "TEFF",
            "LOGG",
            "TARGETID",
            "RVS_WARN",
            "SUCCESS",
            "PRIMARY",
            "SN_R",
            "RR_SPECTYPE",
            "TARGET_RA",
            "TARGET_DEC",
        }
        if required - set(rv.names):
            raise ValueError(
                f"Native alpha/schema fields missing: {required - set(rv.names)}"
            )
        columns = {
            c.name: {"unit": c.unit, "format": c.format} for c in hdul["RVTAB"].columns
        }
        dataset = out.create_dataset(
            "measurements",
            (len(rv),),
            dtype=dtype,
            chunks=(min(chunk_rows, len(rv)),),
            compression="gzip",
            compression_opts=1,
        )
        for start in range(0, len(rv), chunk_rows):
            stop = min(start + chunk_rows, len(rv))
            r, g = rv[start:stop], gaia[start:stop]
            check_target_alignment(r, fiber[start:stop])
            check_target_alignment(r, sp[start:stop])
            ids = np.asarray(g["SOURCE_ID"], dtype="i8")
            alpha, sigma = (
                np.asarray(r["ALPHAFE"], float),
                np.asarray(r["ALPHAFE_ERR"], float),
            )
            sep = separation_arcsec(r["TARGET_RA"], r["TARGET_DEC"], g["RA"], g["DEC"])
            star = np.char.strip(np.asarray(r["RR_SPECTYPE"]).astype(str)) == "STAR"
            eligible = alpha_eligibility(
                ids,
                alpha,
                sigma,
                sep,
                r["RVS_WARN"],
                r["SUCCESS"],
                r["PRIMARY"],
                r["SN_R"],
                star,
            )
            records = np.empty(stop - start, dtype=dtype)
            values = dict(
                source_id=ids,
                target_id=r["TARGETID"],
                source_row=np.arange(start, stop),
                alpha_fe=alpha,
                alpha_fe_error=sigma,
                feh_raw=r["FEH"],
                feh_error=r["FEH_ERR"],
                teff=r["TEFF"],
                logg=r["LOGG"],
                match_sep_arcsec=sep,
                rvs_warn=r["RVS_WARN"],
                success=r["SUCCESS"],
                primary=r["PRIMARY"],
                snr_r=r["SN_R"],
                stellar_class=star,
                eligible=eligible,
            )
            for key, value in values.items():
                records[key] = value
            dataset[start:stop] = records
            counts["raw_rows"] += len(r)
            counts["eligible_rows"] += int(eligible.sum())
            counts["eligible_feh_below_minus2"] += int(
                (eligible & (r["FEH"] < -2)).sum()
            )
        out.attrs["source_sha256"] = expected_sha
        out.attrs["alpha_definition"] = "DESI RVS native ALPHAFE; not APOGEE alpha/M"
        out.attrs["original_rvtab_columns_json"] = json.dumps(columns)
    report = dict(
        status="native_source_projection_complete_not_training_certificate",
        source_path=str(raw),
        source_sha256=expected_sha,
        sidecar_sha256=file_sha(destination),
        counts=counts,
        source_columns=columns,
        scope="Existing source only; exact TARGETID extension alignment; Gaia SOURCE_ID retained as int64; no cross-survey offset/error-floor/coverage invented",
        unresolved=[
            "Published ALPHAFE units/definition and quality semantics must be checked against actual FITS metadata/documentation",
            "All source IDs and every selected raw row require independent source projection verification",
            "Object-disjoint feature overlap and FIT/TUNE/CAL/TEST assignment",
            "Measured native-source overlap offset/scatter and source-tagged heldout comparison",
            "Marginal/joint FeH-alpha calibration",
        ],
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "mwsall-pix-iron.fits")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw, args.output)))


if __name__ == "__main__":
    main()
