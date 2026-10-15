"""DESI DR1 RVS labels with bounded published FeH correction and audited IDs.

Raw spectroscopic errors are not total uncertainties of calibrated abundances.
The immutable original FITS remains the authority for every field and flag.
"""

import hashlib
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

REFERENCE = "https://github.com/desimilkyway/dr1_vac_code/blob/10b6ab6f409397dde646184c42bc9dddfb452971/feh_correct.py"
RAW_SHA = "eae3b31807c58ac340c257e06f66935d01bb81698a08108e6ca57a372cab76b5"
ROOT = Path(
    "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/desi_collection_v1"
)


def corrected_feh(raw, teff, logg):
    """Published RVS DR1 polynomial; never extrapolate beyond its validity box."""
    raw, teff, logg = np.broadcast_arrays(
        np.asarray(raw, float), np.asarray(teff, float), np.asarray(logg, float)
    )
    valid = np.isfinite(raw) & np.isfinite(teff) & np.isfinite(logg)
    valid &= (teff >= 4200) & (teff < 6600) & (logg >= 1.8) & (logg < 5.5)
    t = np.log10(np.where(valid, teff, 5000) / 5000) / 0.1
    low = logg < 4.3
    a = np.where(low, 0.197, 0.039)
    b = np.where(low, -0.809, -0.284)
    c = np.where(low, 0.441, 0.079)
    corrected = np.where(valid, raw - (a + b * t + c * t * t), np.nan)
    derivative = np.where(
        valid,
        -(b + 2 * c * t) / (0.1 * np.log(10) * np.where(valid, teff, 5000)),
        np.nan,
    )
    return corrected, valid, derivative


def separation_arcsec(ra1, dec1, ra2, dec2):
    """Stable spherical separation, including RA wrap; positions are degrees."""
    r1, d1, r2, d2 = np.deg2rad([ra1, dec1, ra2, dec2])
    haversine = (
        np.sin((d2 - d1) / 2) ** 2
        + np.cos(d1) * np.cos(d2) * np.sin((r2 - r1) / 2) ** 2
    )
    return np.rad2deg(2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1)))) * 3600


def check_target_alignment(rv, other):
    if len(rv) != len(other):
        raise ValueError("DESI extensions have different row counts")
    if not np.array_equal(rv["TARGETID"], other["TARGETID"]):
        raise ValueError("DESI TARGETID row alignment mismatch")


def build_labels(output, scratch, root=ROOT, expected_sha=RAW_SHA):
    from stellar_catalog_jobs import identity_split
    from stellar_dataset_views import publish

    raw = Path(root) / "mwsall-pix-iron.fits"
    publisher = (Path(root) / "desi-publisher.sha256sum").read_text().split()[0]
    if publisher != expected_sha:
        raise ValueError("Unexpected publisher checksum")
    local = scratch / raw.name
    digest = hashlib.sha256()
    with raw.open("rb") as src, local.open("xb") as dest:
        while block := src.read(8 * 1024 * 1024):
            digest.update(block)
            dest.write(block)
    if digest.hexdigest() != expected_sha:
        raise ValueError("DESI raw FITS differs from publisher checksum")
    dtype = np.dtype(
        [
            ("source_id", "<i8"),
            ("target_id", "<i8"),
            ("source_row", "<i8"),
            ("ra", "<f8"),
            ("dec", "<f8"),
            ("teff", "<f8"),
            ("logg", "<f8"),
            ("feh_raw", "<f8"),
            ("feh_corrected", "<f8"),
            ("feh_err_native", "<f8"),
            ("teff_err_native", "<f8"),
            ("logg_err_native", "<f8"),
            ("temperature_correction_err_independent", "<f8"),
            ("match_sep_arcsec", "<f8"),
            ("correction_valid", "?"),
            ("rvs_warn", "<i8"),
            ("success", "?"),
            ("primary", "?"),
            ("snr_r", "<f8"),
            ("eligible", "?"),
            ("split", "u1"),
        ]
    )
    staged = scratch / "desi-labels.h5.building"
    counts = {
        "rows": 0,
        "positive_DR3_ID": 0,
        "correction_valid": 0,
        "eligible": 0,
        "eligible_below_minus2": 0,
        "eligible_below_minus3": 0,
    }
    with fits.open(local, memmap=True) as hdul, h5py.File(staged, "x") as out:
        rv, gaia, fiber, sp = [
            hdul[n].data for n in ["RVTAB", "GAIA", "FIBERMAP", "SPTAB"]
        ]
        if not (len(rv) == len(gaia) == len(fiber) == len(sp)):
            raise ValueError("DESI extension lengths disagree")
        ds = out.create_dataset(
            "measurements",
            (len(rv),),
            dtype=dtype,
            chunks=(min(65536, len(rv)),),
            compression="gzip",
            compression_opts=1,
        )
        for start in range(0, len(rv), 65536):
            end = min(start + 65536, len(rv))
            r, g = rv[start:end], gaia[start:end]
            check_target_alignment(r, fiber[start:end])
            check_target_alignment(r, sp[start:end])
            ids = np.asarray(g["SOURCE_ID"], dtype="i8")  # REF_ID is not a DR3 ID.
            corrected, valid, deriv = corrected_feh(r["FEH"], r["TEFF"], r["LOGG"])
            sep = separation_arcsec(r["TARGET_RA"], r["TARGET_DEC"], g["RA"], g["DEC"])
            error = np.asarray(r["FEH_ERR"], float)
            teff_error = np.asarray(r["TEFF_ERR"], float)
            logg_error = np.asarray(r["LOGG_ERR"], float)
            parameter_errors_valid = np.isfinite(teff_error) & (teff_error > 0)
            parameter_errors_valid &= np.isfinite(logg_error) & (logg_error > 0)
            eligible = (ids > 0) & np.isfinite(sep) & (sep <= 1.0) & valid
            eligible &= np.asarray(r["SUCCESS"], bool) & np.asarray(r["PRIMARY"], bool)
            eligible &= (
                (r["RVS_WARN"] == 0) & np.isfinite(error) & (error > 0) & (error < 0.5)
            )
            eligible &= (
                np.char.strip(np.asarray(r["RR_SPECTYPE"]).astype(str)) == "STAR"
            )
            eligible &= parameter_errors_valid
            eligible &= np.isfinite(r["SN_R"]) & (r["SN_R"] >= 10)
            records = np.zeros(end - start, dtype=dtype)
            values = dict(
                source_id=ids,
                target_id=r["TARGETID"],
                source_row=np.arange(start, end),
                ra=g["RA"],
                dec=g["DEC"],
                teff=r["TEFF"],
                logg=r["LOGG"],
                feh_raw=r["FEH"],
                feh_corrected=corrected,
                feh_err_native=error,
                teff_err_native=r["TEFF_ERR"],
                logg_err_native=r["LOGG_ERR"],
                temperature_correction_err_independent=np.where(
                    parameter_errors_valid, np.abs(deriv * teff_error), np.nan
                ),
                match_sep_arcsec=sep,
                correction_valid=valid,
                rvs_warn=r["RVS_WARN"],
                success=r["SUCCESS"],
                primary=r["PRIMARY"],
                snr_r=r["SN_R"],
                eligible=eligible,
                split=identity_split(ids),
            )
            for name, value in values.items():
                records[name] = value
            ds[start:end] = records
            counts["rows"] += len(r)
            counts["positive_DR3_ID"] += int((ids > 0).sum())
            counts["correction_valid"] += int(valid.sum())
            counts["eligible"] += int(eligible.sum())
            for threshold in [2, 3]:
                counts[f"eligible_below_minus{threshold}"] += int(
                    (eligible & (corrected < -threshold)).sum()
                )
        # Verify Gaia row correspondence independently using positions, then retain strict1arcsec subset.
        sep = ds.fields("match_sep_arcsec")[:]
        ids = ds.fields("source_id")[:]
        finite = (ids > 0) & np.isfinite(sep)
        if not finite.any() or float(np.median(sep[finite])) > 0.5:
            raise ValueError("DESI Gaia crossmatch position audit failed")
        mask = ds.fields("eligible")[:]
        good_rows = np.flatnonzero(mask)
        selected = ds[:][mask]
        order = np.lexsort((selected["feh_err_native"], selected["source_id"]))
        selected, good_rows = selected[order], good_rows[order]
        keep = (
            np.r_[True, np.diff(selected["source_id"]) != 0]
            if len(selected)
            else np.array([], bool)
        )
        out.create_dataset(
            "selected/measurements", data=selected[keep], compression="gzip"
        )
        out.create_dataset(
            "selected/raw_measurement_row", data=good_rows[keep], compression="gzip"
        )
        counts["selected_unique_stars"] = int(keep.sum())
        counts["median_match_sep_arcsec"] = float(np.median(sep[finite]))
        counts["positive_id_beyond_1arcsec"] = int((finite & (sep > 1)).sum())
        out.attrs.update(
            complete=True,
            content_id=expected_sha,
            source_file=str(raw),
            raw_sha256=expected_sha,
            correction_reference=REFERENCE,
            pipeline="RVS only",
            uncertainty_definition="feh_err_native is the reported raw random error. Temperature-correction error assumes independent Teff and FeH. Logg branch switching, parameter covariance and correction systematic uncertainty are NOT included; no total calibrated sigma asserted.",
            quality_policy="Exploratory conservative subset: native SUCCESS, PRIMARY, RVS_WARN=0, RR_SPECTYPE=STAR, SN_R>=10, 0<FEH_ERR<0.5dex, published correction bounds, positive Gaia DR3 ID and <=1arcsec sky agreement. Full rejected/raw rows retained.",
        )
    destination = output / "desi-labels.h5"
    publish(staged, destination)
    local.unlink()
    return dict(
        file=str(destination),
        bytes=destination.stat().st_size,
        raw_sha256=expected_sha,
        correction_reference=REFERENCE,
        counts=counts,
        publication_roundtrip=True,
        uncertainties="Native random errors retained; total calibrated uncertainty not established",
        original_all_fields_readonly_file=str(raw),
        pipeline="RVS only",
    )
