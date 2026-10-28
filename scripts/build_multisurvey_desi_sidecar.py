"""Build an observation-only DESI DR1 RVS raw-label sidecar, without downloads.

The original publisher-bound FITS remains immutable. FITS processing is chunked;
selected columns are spooled to disk-backed arrays before the shared sidecar
writer validates them. That writer currently has O(N) row-uniqueness workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits
from desi_stellar_labels import RAW_SHA, ROOT, separation_arcsec

from masked_stellar_autoencoder.training.multisurvey_overlay import (
    file_sha256,
    write_observation_sidecar,
)

SOURCE_NAME = "desi_dr1_rvs_raw"
REGISTRY = {
    "release": "DESI DR1 mwsall-pix-iron",
    "pipeline": "RVS raw spectroscopic labels; no published bounded FeH correction applied",
    "dependency_id": "desi-dr1-rvs-common-spectrum",
    "targets": {
        "teff": {"quantity": "teff", "unit": "K", "field": "TEFF"},
        "logg": {"quantity": "logg", "unit": "dex(cm/s2)", "field": "LOGG"},
        "fe_h": {"quantity": "fe_h", "unit": "dex", "field": "FEH"},
    },
    "error_fields": ["TEFF_ERR", "LOGG_ERR", "FEH_ERR"],
    "uncertainty": "Reported raw random errors only; parameter covariances/systematics unknown, no calibrated total error asserted",
    "unknown_error": "Nonpositive or nonfinite native errors retained in raw_error and represented as NaN in error; a positive declared floor is required downstream",
    "dependency": "All three parameters share an RVS spectrum; independence across targets or repeated Gaia identities is not asserted",
    "gaia_conditioning": "not certified by this raw-field adapter",
}


def _integer_column(table, name):
    values = np.asarray(table[name])
    if values.dtype.kind not in "iu":
        raise ValueError(
            f"{name} must be exact integer identity, never float-converted"
        )
    if values.dtype.kind == "u" and np.any(values > np.iinfo("i8").max):
        raise ValueError(f"{name} exceeds signed int64")
    return values.astype("i8", copy=False)


def _quality(rv, gaia, fiber, sp, snr_min):
    target = _integer_column(rv, "TARGETID")
    for name, other in (("FIBERMAP", fiber), ("SPTAB", sp)):
        if not np.array_equal(target, _integer_column(other, "TARGETID")):
            raise ValueError(f"DESI TARGETID alignment mismatch: {name}")
    gaia_targetid = "TARGETID" in gaia.dtype.names
    if gaia_targetid and not np.array_equal(target, _integer_column(gaia, "TARGETID")):
        raise ValueError("DESI TARGETID alignment mismatch: GAIA")
    source = _integer_column(gaia, "SOURCE_ID")
    sep = separation_arcsec(rv["TARGET_RA"], rv["TARGET_DEC"], gaia["RA"], gaia["DEC"])
    star = np.char.strip(np.asarray(rv["RR_SPECTYPE"]).astype(str)) == "STAR"
    # Native FITS logical columns arrive as booleans, not truthy strings.
    success, primary = np.asarray(rv["SUCCESS"]), np.asarray(rv["PRIMARY"])
    if success.dtype.kind not in "biu" or primary.dtype.kind not in "biu":
        raise ValueError("SUCCESS/PRIMARY must be native logical or integer flags")
    good = (source > 0) & (target > 0) & np.isfinite(sep) & (sep <= 1.0)
    good &= (success == 1) & (primary == 1) & (rv["RVS_WARN"] == 0) & star
    good &= np.isfinite(rv["SN_R"]) & (rv["SN_R"] >= snr_min)
    good &= np.isfinite(gaia["DEC"]) & (np.abs(gaia["DEC"]) <= 90)
    good &= np.isfinite(rv["TARGET_DEC"]) & (np.abs(rv["TARGET_DEC"]) <= 90)
    return good, sep, source, target, star, gaia_targetid


def build_desi_sidecar(
    output,
    *,
    raw_path=ROOT / "mwsall-pix-iron.fits",
    publisher_path=ROOT / "desi-publisher.sha256sum",
    expected_sha=RAW_SHA,
    chunk_rows=65536,
    snr_min=10.0,
    scratch_dir=None,
):
    """Retain all quality-passing rows, including duplicates and partial labels.

    ``source_row`` is the zero-based immutable RVTAB row. GAIA SOURCE_ID is
    retained as int64; RVTAB/FIBERMAP/SPTAB TARGETIDs must match exactly. When
    GAIA has no TARGETID (as in the existing staged schema), its association is
    the publisher-bound row correspondence plus <=1 arcsec sky agreement,
    explicitly recorded rather than claimed to be an independent ID join.
    No split assignment, deduplication, acquisition, or feature/model use occurs.
    """
    if (
        isinstance(chunk_rows, bool)
        or not isinstance(chunk_rows, int)
        or chunk_rows < 1
    ):
        raise ValueError("chunk_rows must be a positive integer")
    if not np.isfinite(snr_min) or snr_min < 0:
        raise ValueError("snr_min must be finite and nonnegative")
    raw_path, publisher_path, output = (
        Path(raw_path),
        Path(publisher_path),
        Path(output),
    )
    if output.exists():
        raise FileExistsError(output)
    publisher = publisher_path.read_text().split()
    if not publisher or publisher[0].lower() != expected_sha.lower():
        raise ValueError("Publisher checksum differs from expected raw SHA256")
    before = raw_path.stat()
    if file_sha256(raw_path) != expected_sha.lower():
        raise ValueError("Original DESI FITS differs from publisher checksum")
    after_hash = raw_path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
        after_hash.st_size,
        after_hash.st_mtime_ns,
        after_hash.st_ctime_ns,
        after_hash.st_ino,
    ):
        raise ValueError("Source changed during checksum verification")
    with fits.open(raw_path, memmap=True) as hdus:
        rv, gaia, fiber, sp = [
            hdus[name].data for name in ("RVTAB", "GAIA", "FIBERMAP", "SPTAB")
        ]
        if not len(rv) == len(gaia) == len(fiber) == len(sp):
            raise ValueError("DESI extension row counts disagree")
        # Access names before creating output; missing FEH is never replaced by MH.
        fields = ("TEFF", "LOGG", "FEH")
        errors = ("TEFF_ERR", "LOGG_ERR", "FEH_ERR")
        for field in (*fields, *errors):
            if field not in rv.dtype.names:
                raise ValueError(f"Missing native RVS field: {field}")
        count = 0
        for start in range(0, len(rv), chunk_rows):
            stop = start + chunk_rows
            good, *_ = _quality(
                rv[start:stop],
                gaia[start:stop],
                fiber[start:stop],
                sp[start:stop],
                snr_min,
            )
            count += int(good.sum())
        spectype_width = np.asarray(rv[:1]["RR_SPECTYPE"]).astype("S").dtype.itemsize
        layout = {
            "source_id": ("i8", ()),
            "source_row": ("i8", ()),
            "target_id": ("i8", ()),
            "value": ("f8", (3,)),
            "error": ("f8", (3,)),
            "raw_error": ("f8", (3,)),
            "observed": ("?", (3,)),
            "ra": ("f8", ()),
            "dec": ("f8", ()),
            "target_ra": ("f8", ()),
            "target_dec": ("f8", ()),
            "match_sep_arcsec": ("f8", ()),
            "rvs_warn": ("i8", ()),
            "success": ("?", ()),
            "primary": ("?", ()),
            "spectype_star": ("?", ()),
            "rr_spectype_ascii": ("u1", (spectype_width,)),
            "snr_r": ("f8", ()),
        }
        with tempfile.TemporaryDirectory(
            prefix="desi-sidecar-columns-", dir=scratch_dir
        ) as temp:
            block = {
                key: np.lib.format.open_memmap(
                    Path(temp) / (key + ".npy"),
                    mode="w+",
                    dtype=dtype,
                    shape=(count, *tail),
                )
                if count
                else np.empty((0, *tail), dtype=dtype)
                for key, (dtype, tail) in layout.items()
            }
            offset = 0
            masks = np.zeros(3, dtype="i8")
            for start in range(0, len(rv), chunk_rows):
                stop = start + chunk_rows
                r, g = rv[start:stop], gaia[start:stop]
                good, sep, source, target, star, _ = _quality(
                    r, g, fiber[start:stop], sp[start:stop], snr_min
                )
                n = int(good.sum())
                rows = slice(offset, offset + n)
                values = np.column_stack([r[field][good] for field in fields]).astype(
                    "f8"
                )
                sigma = np.column_stack([r[field][good] for field in errors]).astype(
                    "f8"
                )
                # Preserve usable values even when no uncertainty is reported.
                # -9999 is an explicit missing-value sentinel, not a label.
                observed = np.isfinite(values) & (values != -9999)
                observed[:, 0] &= values[:, 0] > 0
                error = np.where(np.isfinite(sigma) & (sigma > 0), sigma, np.nan)
                mapping = {
                    "source_id": source[good],
                    "source_row": np.arange(start, min(stop, len(rv)), dtype="i8")[
                        good
                    ],
                    "target_id": target[good],
                    "value": values,
                    "error": error,
                    "raw_error": sigma,
                    "observed": observed,
                    "ra": g["RA"][good],
                    "dec": g["DEC"][good],
                    "target_ra": r["TARGET_RA"][good],
                    "target_dec": r["TARGET_DEC"][good],
                    "match_sep_arcsec": sep[good],
                    "rvs_warn": r["RVS_WARN"][good],
                    "success": r["SUCCESS"][good],
                    "primary": r["PRIMARY"][good],
                    "spectype_star": star[good],
                    "rr_spectype_ascii": np.ascontiguousarray(
                        np.asarray(r["RR_SPECTYPE"])[good].astype(f"S{spectype_width}")
                    )
                    .view("u1")
                    .reshape(n, spectype_width),
                    "snr_r": r["SN_R"][good],
                }
                for key, array in mapping.items():
                    block[key][rows] = array
                masks += observed.sum(0)
                offset += n
            if offset != count:
                raise ValueError("Source row selection changed during streaming passes")
            after = raw_path.stat()
            if (
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
                before.st_ino,
            ) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
                after.st_ino,
            ):
                raise ValueError("Source changed during ingestion")
            provenance = {
                "source_file": str(raw_path.resolve()),
                "raw_sha256": expected_sha.lower(),
                "publisher_checksum_file": str(publisher_path.resolve()),
                "source_rows": len(rv),
                "quality_rows": count,
                "chunk_rows": chunk_rows,
                "quality": f"positive Gaia SOURCE_ID/TARGETID; <=1arcsec; RR_SPECTYPE=STAR; SUCCESS=PRIMARY=True; RVS_WARN=0; SN_R>={snr_min}",
                "alignment": "Exact RVTAB/FIBERMAP/SPTAB TARGETID; "
                + (
                    "exact GAIA TARGETID and SOURCE_ID retained"
                    if "TARGETID" in gaia.dtype.names
                    else "GAIA has no TARGETID: immutable row correspondence plus sky audit; SOURCE_ID retained exactly"
                ),
                "duplicates": "All quality-passing duplicate Gaia identities retained; explicit downstream dependency resolution required",
                "labels": "Independent per-target masks require finite native value (excluding -9999; Teff must be positive). Missing or nonpositive random errors remain NaN, with original values in raw_error; no FeH correction or imputation",
                "selection_scope": "Only quality-passing raw rows retained here; rejected raw rows remain in original SHA-bound FITS",
                "memory_scope": "FITS bounded chunks and disk-backed column spools; shared writer row-uniqueness check has O(N) workspace",
            }
            with tempfile.TemporaryDirectory(
                prefix=".desi-sidecar-publish-", dir=output.parent
            ) as staging:
                path = Path(staging) / "sidecar.h5"
                receipt = write_observation_sidecar(
                    path,
                    {SOURCE_NAME: block},
                    {SOURCE_NAME: REGISTRY},
                    provenance=provenance,
                )
                # Same-filesystem exclusive publication; never replace an existing output.
                os.link(path, output)
            return {
                "path": str(output.resolve()),
                "sha256": receipt["sha256"],
                "quality_rows": count,
                "observed_counts": masks.tolist(),
                "raw_sha256": expected_sha.lower(),
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--raw", type=Path, default=ROOT / "mwsall-pix-iron.fits")
    parser.add_argument(
        "--publisher", type=Path, default=ROOT / "desi-publisher.sha256sum"
    )
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--chunk-rows", type=int, default=65536)
    args = parser.parse_args()
    print(
        json.dumps(
            build_desi_sidecar(
                args.output,
                raw_path=args.raw,
                publisher_path=args.publisher,
                scratch_dir=args.scratch,
                chunk_rows=args.chunk_rows,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
