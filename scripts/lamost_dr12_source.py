"""Source-only DR12 catalogue download and exact Gaia/base/cache crossmatch."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tempfile
import urllib.request
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

from masked_stellar_autoencoder.training.evaluation_feature_lookup import (
    CACHE_REQUEST_SHA,
    _cache_maps,
)
from masked_stellar_autoencoder.training.multisurvey_census import (
    _stat,
    _verify,
    join_base,
    scan_cache,
    verify_cache_associations,
)

PRODUCTS = {
    "lrs": {
        "url": "https://www.lamost.org/dr12/v2.0/catdl?name=dr12_v2.0_LRS_stellar.fits.gz",
        "bytes": 1221786815,
        "rows": 8376135,
        "groups": {
            "lasp": {
                "values": ["teff", "logg", "feh", "alpha_m"],
                "errors": ["teff_err", "logg_err", "feh_err", "alpha_m_err"],
            }
        },
    },
    "mrs": {
        "url": "https://www.lamost.org/dr12/v2.0/medcas/catdl?name=dr12_v2.0_MRS_stellar.fits.gz",
        "bytes": 725541580,
        "rows": 3224901,
        "groups": {
            "lasp": {
                "values": ["teff_lasp", "logg_lasp", "feh_lasp", "alpha_m_lasp"],
                "errors": [
                    "teff_lasp_err",
                    "logg_lasp_err",
                    "feh_lasp_err",
                    "alpha_m_lasp_err",
                ],
            },
            "cnn": {
                "values": [
                    "teff_cnn",
                    "logg_cnn",
                    "feh_cnn",
                    "m_h_cnn",
                    "alpha_m_cnn",
                    "c_fe",
                    "n_fe",
                    "o_fe",
                    "mg_fe",
                    "al_fe",
                    "si_fe",
                    "s_fe",
                    "ca_fe",
                    "ti_fe",
                    "cr_fe",
                    "ni_fe",
                    "cu_fe",
                ],
                "errors": [None] * 17,
            },
        },
    },
}


def exact_gaia_ids(raw):
    """Parse published text IDs without floating point; zero explicitly means missing/invalid."""
    a = np.asarray(raw)
    if a.ndim != 1 or a.dtype.kind not in "SU":
        raise ValueError("Gaia identifiers must be text")

    def one(value):
        v = value.strip()
        if isinstance(v, bytes):
            if not v.isdigit() or len(v) > 19:
                return 0
        elif not v.isascii() or not v.isdigit() or len(v) > 19:
            return 0
        x = int(v)
        return x if 0 < x <= np.iinfo("i8").max else 0

    return np.fromiter((one(v) for v in a), dtype="<i8", count=len(a))


def native_values(table, names):
    values = np.column_stack([np.asarray(table[n], dtype="f8") for n in names])
    available = np.isfinite(values) & (values != -9999)
    return values, available


def download(url, target, expected_bytes, *, direct=False):
    opener = (
        urllib.request.build_opener(urllib.request.ProxyHandler({}))
        if direct
        else urllib.request.build_opener()
    )
    with opener.open(
        urllib.request.Request(url, headers={"User-Agent": "msa-dr12-source/1.0"}),
        timeout=90,
    ) as src:
        if (
            src.status != 200
            or int(src.headers.get("Content-Length", "-1")) != expected_bytes
        ):
            raise ValueError("Publisher response size differs")
        digest = hashlib.sha256()
        total = 0
        with Path(target).open("xb") as out:
            while block := src.read(4 * 1024 * 1024):
                out.write(block)
                digest.update(block)
                total += len(block)
        if total != expected_bytes:
            raise ValueError("Publisher download truncated")
    return digest.hexdigest()


def produce(
    *,
    product,
    archive,
    base,
    cache,
    original,
    base_sha,
    output,
    expected_cache_rows=150000000,
    chunk_rows=65536,
    progress=None,
):
    spec = PRODUCTS[product]
    archive, base, cache, original, output = map(
        Path, (archive, base, cache, original, output)
    )
    if output.exists():
        raise FileExistsError(output)
    before = _verify(base, base_sha)
    base_stat = _stat(archive)
    with tempfile.TemporaryDirectory(prefix="msa-lamost-dr12-", dir="/scratch") as tmp:
        unpacked = Path(tmp) / "stellar.fits"
        with gzip.open(archive, "rb") as src, unpacked.open("xb") as dst:
            shutil.copyfileobj(src, dst, 4 * 1024 * 1024)
        with fits.open(unpacked, memmap=True) as hdus:
            table = hdus[1].data
            if len(table) != spec["rows"]:
                raise ValueError("Publisher FITS row count differs")
            fields = set(table.dtype.names or ())
            needed = {"gaia_source_id", "obsid"}
            for g in spec["groups"].values():
                needed.update(g["values"])
                needed.update(x for x in g["errors"] if x)
            if not needed.issubset(fields):
                raise ValueError("Publisher FITS required schema differs")
            row_ids = exact_gaia_ids(table["gaia_source_id"])
            valid = row_ids > 0
            ids = np.unique(row_ids[valid])
            if not len(ids):
                raise ValueError("No valid Gaia identifiers")
            with h5py.File(base, "r") as b:
                matched, roles, primary, exposed, reserved = join_base(
                    ids,
                    b["source_id"][:],
                    b["role"][:],
                    b["primary_input_eligible"][:],
                    b["known_supervised_ancestor_exposed"][:],
                    b["reserved_reference"][:],
                )
                names = b["feature_names"][:].astype("U").tolist()
                center, scale = b["scaler_center"][:], b["scaler_scale"][:]
            watched = [
                cache / name
                for name in (
                    "manifest.json",
                    "features.npy",
                    "source_id.npy",
                    "provenance.npy",
                )
            ]
            stats = [_stat(x) for x in watched]
            orig_stat = _stat(original)
            features, cache_ids, provenance, cache_manifest = _cache_maps(
                cache, names, center, scale, CACHE_REQUEST_SHA, expected_cache_rows
            )
            xp = [
                names.index(f"{arm}_{j}") for arm in ("bp", "rp") for j in range(1, 56)
            ]
            with (cache / "source_id.npy").open("rb") as stream:
                prefix = stream.read(cache_ids.offset)
            cache_rows, complete, cache_hash, invalid = scan_cache(
                ids,
                cache_ids,
                features,
                xp,
                chunk_rows=chunk_rows,
                hash_prefix=prefix,
                progress=progress,
            )
            if cache_hash != cache_manifest["files"]["source_id.npy"]["sha256"]:
                raise ValueError("Cache ID content hash differs")
            assoc = verify_cache_associations(
                original,
                ids,
                cache_rows,
                features,
                provenance,
                names,
                center,
                scale,
                cache_manifest["request"],
            )
            if (
                _stat(base) != before
                or _stat(archive) != base_stat
                or _stat(original) != orig_stat
                or [_stat(x) for x in watched] != stats
            ):
                raise ValueError("Bound input changed")
            with h5py.File(output, "x") as h:
                h.attrs["schema"] = "lamost-dr12-gaia-cache-source-v1"
                h.attrs["product"] = product
                h.attrs["scope"] = (
                    "Raw source-specific quantities and exact source IDs. Not training-ready; M/H and alpha scales and dependencies require modeling."
                )
                star = h.create_group("stars")
                star.create_dataset("source_id", data=ids, compression="gzip")
                star.create_dataset("cache_row", data=cache_rows, compression="gzip")
                star.create_dataset(
                    "cache_xp_state",
                    data=np.where(cache_rows >= 0, complete.astype("i1"), -1),
                    compression="gzip",
                )
                star.create_dataset(
                    "base_role", data=roles.astype("S16"), compression="gzip"
                )
                star.create_dataset("base_matched", data=matched, compression="gzip")
                for name, value in [
                    ("primary", primary),
                    ("ancestor_listed", exposed),
                    ("reserved", reserved),
                ]:
                    star.create_dataset(
                        name + "_state",
                        data=np.where(matched, value.astype("i1"), -1),
                        compression="gzip",
                    )
                rows = h.create_group("observations")
                rows.create_dataset("source_id", data=row_ids, compression="gzip")
                rows.create_dataset(
                    "source_row",
                    data=np.arange(len(row_ids), dtype="<i8"),
                    compression="gzip",
                )
                rows.create_dataset(
                    "obsid",
                    data=np.asarray(table["obsid"], dtype="<i8"),
                    compression="gzip",
                )
                counts = {}
                for name, g in spec["groups"].items():
                    sub = rows.create_group(name)
                    vals, observed = native_values(table, g["values"])
                    errs = np.column_stack(
                        [
                            np.asarray(table[x], dtype="f8")
                            if x
                            else np.full(len(table), np.nan)
                            for x in g["errors"]
                        ]
                    )
                    known = observed & np.isfinite(errs) & (errs > 0) & (errs != -9999)
                    sub.create_dataset("value", data=vals, compression="gzip")
                    sub.create_dataset(
                        "raw_observed", data=observed, compression="gzip"
                    )
                    sub.create_dataset("error", data=errs, compression="gzip")
                    sub.create_dataset("error_known", data=known, compression="gzip")
                    sub.attrs["fields"] = json.dumps(g["values"])
                    counts[name] = {
                        k: int(v)
                        for k, v in zip(g["values"], observed.sum(axis=0), strict=True)
                    }
            return {
                "product": product,
                "source_rows": len(row_ids),
                "rows_with_valid_gaia_ids": int(valid.sum()),
                "unique_gaia_ids": len(ids),
                "matched_base": int(matched.sum()),
                "matched_cache": int((cache_rows >= 0).sum()),
                "complete_xp_in_cache": int(complete.sum()),
                "outside_base_ancestry_unknown": int((~matched).sum()),
                "cache_absence_not_no_xp": True,
                "cache_source_id_sha256": cache_hash,
                "cache_invalid_ids": invalid,
                "cache_associations_verified": assoc,
                "raw_label_counts_by_measurement": counts,
                "output": str(output.resolve()),
                "output_bytes": output.stat().st_size,
                "output_sha256": hashlib.file_digest(
                    output.open("rb"), "sha256"
                ).hexdigest(),
            }
