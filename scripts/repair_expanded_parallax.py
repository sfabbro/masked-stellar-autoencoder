"""Version the prepared source with exact-ID raw Gaia astrometry for every cohort.

Only measured_parallax changes. Labels, splits, legacy inputs and scalers retain
byte-identical dataset values. An independent ESA release/unit audit is required.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
RAW = Path("/arc/projects/k-pop/catalogues/andrae2023/sslset-realmags-full-052725.h5")


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def match_block(ids, block):
    """Return source positions and exact matched raw records; no float IDs."""
    raw_ids = block["source_id"]
    if raw_ids.dtype.kind not in "iu" or ids.dtype.kind not in "iu":
        raise ValueError("Integer identities required")
    if np.any(raw_ids < 0):
        raise ValueError("Negative raw identity")
    pos = np.searchsorted(ids, raw_ids)
    valid = pos < len(ids)
    valid[valid] &= ids[pos[valid]] == raw_ids[valid]
    selected = pos[valid]
    if len(np.unique(selected)) != len(selected):
        raise ValueError("Repeated raw identities in block")
    return selected, block[valid]


def verify_unchanged(parent, child):
    checked = []

    def check(name, original):
        if not isinstance(original, h5py.Dataset) or name.startswith(
            "measured_parallax/"
        ):
            return
        other = child[name]
        if other.shape != original.shape or other.dtype != original.dtype:
            raise ValueError("Dataset schema changed: " + name)
        if original.ndim == 0:
            np.testing.assert_array_equal(original[()], other[()])
        else:
            for start in range(0, len(original), 65536):
                np.testing.assert_array_equal(
                    original[start : start + 65536], other[start : start + 65536]
                )
        checked.append(name)

    parent.visititems(check)
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="recovering_full_cohort_astrometry",
        training_source_ready=False,
        training_performed=False,
    )

    def save():
        path = a.output / "manifest.tmp"
        path.write_text(json.dumps(m, indent=2) + "\n")
        path.replace(a.output / "manifest.json")

    save()
    try:
        pm = json.loads(
            (ROOT / "expanded_six_source_20261107_v2/manifest.json").read_text()
        )
        audit_path = ROOT / "gaia_parallax_release_unit_audit_20261107_v1/manifest.json"
        audit = json.loads(audit_path.read_text())
        if (
            not audit.get("pass_gate")
            or audit["source_sha256"] != pm["prepared_sha256"]
        ):
            raise ValueError(
                "Independent Gaia release/unit audit did not pass for parent"
            )
        parent = Path(pm["prepared_path"])
        if sha(parent) != pm["prepared_sha256"]:
            raise ValueError("Parent source changed")
        census = json.loads(
            (
                ROOT / "andrae_full_footprint_census_20261106_v2/manifest.json"
            ).read_text()
        )
        identity = census["raw_identity"]
        before = RAW.stat()
        if identity["path"] != str(RAW) or (
            identity["bytes"],
            identity["mtime_ns"],
        ) != (before.st_size, before.st_mtime_ns):
            raise ValueError("Raw catalogue identity changed")
        with h5py.File(parent) as f:
            ids, roles = f["source_id"][:], f["role"][:]
            old_seen = f["measured_parallax/observed"][:].astype(bool)
            values = f["measured_parallax/parallax_mas"][:].astype("f8")
            errors = f["measured_parallax/parallax_error_mas"][:].astype("f8")
            names = f["feature_names"][:].astype("U").tolist()
            j = names.index("PARALLAX")
            center, scale = float(f["scaler_center"][j]), float(f["scaler_scale"][j])
        if np.any(np.diff(ids) <= 0) or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid source identity/scaler")
        found = np.zeros(len(ids), bool)
        with h5py.File(RAW) as f:
            keys = [
                k
                for k in f
                if isinstance(f[k], h5py.Dataset)
                and f[k].dtype.names
                and "source_id" in f[k].dtype.names
            ]
            if sum(len(f[k]) for k in keys) != census["raw_rows_total"]:
                raise ValueError("Raw row census differs")
            scanned = 0
            for key in keys:
                d = f[key]
                for start in range(0, len(d), 262144):
                    block = d.fields(["source_id", "PARALLAX", "e_parallax"])[
                        start : start + 262144
                    ]
                    pos, records = match_block(ids, block)
                    if found[pos].any():
                        raise ValueError("Repeated raw identity across blocks")
                    found[pos] = True
                    values[pos], errors[pos] = (
                        records["PARALLAX"],
                        records["e_parallax"],
                    )
                    scanned += len(block)
                m.update(
                    raw_rows_scanned=scanned,
                    exact_id_matches=int(found.sum()),
                    last_shard=key,
                )
                save()
        after = RAW.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Raw catalogue changed during recovery")
        # Unmatched formerly observed inputs keep independently checked release convention.
        seen = (
            (found | old_seen)
            & np.isfinite(values)
            & (values != -9999)
            & np.isfinite(errors)
            & (errors > 0)
        )
        output = a.output / "prepared.h5"
        m["status"] = "copying_and_verifying_source"
        save()
        shutil.copyfile(parent, output)
        with h5py.File(output, "r+") as f:
            group = f["measured_parallax"]
            for key, v in dict(
                parallax_mas=values,
                parallax_error_mas=errors,
                observed=seen,
                input_value=np.where(seen, (values - center) / scale, -9999),
            ).items():
                group[key][:] = v
            group.attrs["provenance"] = (
                "Exact integer-ID full raw Gaia join; signed mas and positive formal errors. No parallax inversion or zeropoint correction."
            )
            group.create_dataset("raw_matched", data=found, compression="lzf")
        with h5py.File(parent) as first, h5py.File(output) as second:
            checked = verify_unchanged(first, second)
            np.testing.assert_array_equal(
                second["measured_parallax/observed"][:].astype(bool), seen
            )
        with h5py.File(a.output / "raw-join-provenance.h5", "x") as f:
            f.create_dataset("source_id", data=ids, compression="lzf")
            f.create_dataset("raw_matched", data=found, compression="lzf")
        counts = {}
        for role in np.unique(roles):
            take = roles == role
            counts[role.decode()] = dict(
                stars=int(take.sum()),
                before=int((take & old_seen).sum()),
                after=int((take & seen).sum()),
                newly_available=int((take & seen & ~old_seen).sum()),
                negative=int((take & seen & (values < 0)).sum()),
            )
        result = dict(pm)
        result.update(
            status="expanded_six_source_complete",
            training_source_ready=True,
            prepared_path=str(output),
            prepared_sha256=sha(output),
            parent_source_sha256=pm["prepared_sha256"],
            parallax_repair=dict(
                raw_identity=identity,
                independent_esa_audit_sha256=sha(audit_path),
                role_availability=counts,
                raw_matched=int(found.sum()),
                unmatched_old_observed_retained=int((~found & old_seen & seen).sum()),
                unchanged_datasets=checked,
                scope="Only measured parallax values/errors/availability change. All labels, roles, features, scalers verified unchanged. Input error is stored, not yet an encoder precision context.",
            ),
        )
        m.clear()
        m.update(result)
        save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc), training_source_ready=False)
        save()
        raise


if __name__ == "__main__":
    main()
