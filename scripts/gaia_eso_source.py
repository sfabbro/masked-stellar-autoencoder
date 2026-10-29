"""Raw Gaia-ESO DR5.1 source projection; no Gaia identity inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits
from rave_cnn_source import _checked_digest, _stat, sha256_file

REQUIRED = (
    "OBJECT",
    "RA",
    "DECLINATION",
    "TEFF",
    "E_TEFF",
    "LOGG",
    "E_LOGG",
    "FEH",
    "E_FEH",
    "MG1",
    "E_MG1",
    "SI1",
    "E_SI1",
    "CA1",
    "E_CA1",
    "TI1",
    "E_TI1",
)


def produce(source, output, source_sha, expected_rows=114916):
    source, output = map(Path, (source, output))
    _, before = _checked_digest(source, source_sha)
    counts = {
        "rows": 0,
        "finite_values": {},
        "position_epoch": "unknown_per_row",
        "gaia_id": "not_supplied_in_checked_schema",
    }
    with fits.open(source, memmap=True) as hdus:
        h = hdus[1]
        if len(h.data) != expected_rows or not set(REQUIRED).issubset(h.columns.names):
            raise ValueError("Gaia-ESO row count or required columns differ")
        with h5py.File(output, "x") as f:
            f.attrs["source_sha256"] = source_sha
            f.attrs["scope"] = (
                "Gaia-ESO DR5.1 raw columns; source astrometric epoch not supplied; no inferred Gaia IDs, transformations or training roles."
            )
            f.attrs["publisher"] = "ESO Gaia-ESO DR5.1"
            f.create_dataset(
                "source_row",
                data=np.arange(len(h.data), dtype=np.int64),
                compression="gzip",
                shuffle=True,
            )
            g = f.create_group("raw")
            for col in h.columns:
                arr = np.asarray(h.data[col.name])
                if arr.dtype.kind == "U":
                    arr = np.char.encode(arr, "ascii")
                if arr.dtype.kind not in "Sifub":
                    raise ValueError("Unsupported FITS dtype: " + col.name)
                d = g.create_dataset(
                    col.name, data=arr, compression="gzip", shuffle=True
                )
                d.attrs["fits_format"] = str(col.format)
                d.attrs["fits_unit"] = str(col.unit or "")
                d.attrs["fits_null"] = str(col.null if col.null is not None else "")
            for name in ("TEFF", "LOGG", "FEH", "MG1", "SI1", "CA1", "TI1"):
                counts["finite_values"][name] = int(np.isfinite(h.data[name]).sum())
            ra = np.asarray(h.data["RA"])
            dec = np.asarray(h.data["DECLINATION"])
            counts["valid_coordinate_rows"] = int(
                (
                    np.isfinite(ra)
                    & np.isfinite(dec)
                    & (ra >= 0)
                    & (ra < 360)
                    & (dec >= -90)
                    & (dec <= 90)
                ).sum()
            )
            counts["rows"] = len(h.data)
            f.attrs["counts_json"] = json.dumps(counts, sort_keys=True)
    if _stat(source) != before:
        raise ValueError("Gaia-ESO source changed during read")
    return {
        "status": "gaia_eso_raw_source_complete",
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "output_bytes": output.stat().st_size,
        "source_sha256": source_sha,
        "counts": counts,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--source-sha256", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--receipt", required=True)
    a = p.parse_args()
    if Path(a.receipt).exists():
        raise FileExistsError(a.receipt)
    result = produce(a.source, a.output, a.source_sha256)
    with open(a.receipt, "x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
