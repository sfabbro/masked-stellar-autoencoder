"""CANFAR CPU: stage LAMOST LRS candidate and audit benchmark feature coverage."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from astropy.io import fits

from masked_stellar_autoencoder.training.multisurvey_overlay import (
    write_observation_sidecar,
)

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
SOURCE = "/arc/projects/k-pop/catalogues/lamost-param/dr8_v2_LRS_stellar.fits"
RELEASE_ROWS = {"dr8v2": 6684413, "dr12v2": 8376135}
BASE = ROOT + "remediation_source_v3/prepared.h5"
REGISTRY = dict(
    release="LAMOST DR8 v2.0 LRS AFGK native stellar catalogue",
    pipeline="LASP LRS",
    dependency_id="lamost-dr8v2-lrs-lasp",
    targets={
        key: dict(quantity=key, unit=unit, field=field)
        for key, unit, field in [
            ("teff", "K", "teff"),
            ("logg", "dex(cm/s2)", "logg"),
            ("fe_h", "dex", "feh"),
        ]
    },
    quality_policy="provisional SN g>=30; per-target finite value/domain and positive quoted error; highest-SNR repeat per native Gaia DR3 ID",
    identity_policy="DR8v2 Gaia DR3 association per publisher; raw mounted-file provenance, no independently repeated astrometric crossmatch",
    training_enabled=False,
)


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def integer_gaia_ids(raw):
    a = np.asarray(raw)
    if a.dtype.kind in "iu":
        return a.astype("i8"), a > 0
    if a.dtype.kind not in "SU":
        raise ValueError("Gaia IDs must be strings or exact integers")
    text = np.char.strip(a.astype("U"))
    valid = np.char.isdecimal(text)
    # Reject overflow and signed/decimal strings before int64 conversion.
    valid &= np.char.str_len(text) <= 19
    text[~valid] = "0"
    ids = np.array(
        [int(v) if int(v) <= np.iinfo("i8").max else 0 for v in text], dtype="i8"
    )
    return ids, valid & (ids > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmarks", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--release", choices=["dr8v2", "dr12v2"], required=True)
    p.add_argument("--source", type=Path)
    args = p.parse_args()
    if args.release == "dr12v2" and args.source is None:
        p.error("DR12 requires an explicit acquired native source")
    source = str(args.source or SOURCE)
    expected_rows = RELEASE_ROWS[args.release]
    source_name = "lamost_" + args.release + "_lrs_lasp"
    registry = dict(REGISTRY)
    registry.update(
        release="LAMOST "
        + args.release.replace("v2", " v2.0").upper()
        + " LRS AFGK native stellar catalogue",
        dependency_id="lamost-" + args.release + "-lrs-lasp",
        identity_policy=args.release
        + " Gaia DR3 association per publisher; exact native IDs, no independently repeated astrometric crossmatch",
    )
    if not Path(BASE).is_file():
        raise RuntimeError("CANFAR source mounts required")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    benchmarks = np.asarray(json.loads(args.benchmarks.read_text()), dtype="i8")
    with h5py.File(BASE) as base:
        ids = base["source_id"][:]
        roles = base["role"][:].astype("U")
        eligible = base["primary_input_eligible"][:].astype(bool)
        if ids.dtype.kind not in "iu" or (np.diff(ids) <= 0).any():
            raise ValueError("Base source IDs must be sorted exact unique integers")
        matched = np.searchsorted(ids, benchmarks)
        ok = matched < len(ids)
        ok[ok] &= ids[matched[ok]] == benchmarks[ok]
        audit = []
        for sid, row in zip(benchmarks[ok], matched[ok]):
            values = base["input"][row]
            entry = dict(
                source_id=int(sid),
                role=str(roles[row]),
                primary_input_eligible=bool(eligible[row]),
                input_width=len(values),
                available_inputs=int((np.isfinite(values) & (values != -9999)).sum()),
            )
            for key in ["input_has_xp", "real_xp_present", "real_no_xp"]:
                if key in base:
                    entry[key] = bool(base[key][row])
            audit.append(entry)
        (out / "benchmark-input-audit.json").write_text(
            json.dumps(
                dict(base=BASE, stars=audit, unmatched_ids=benchmarks[~ok].tolist()),
                indent=2,
            )
            + "\n"
        )
    print(
        json.dumps(dict(phase="benchmark_input_audit", matched=len(audit))), flush=True
    )
    stat = Path(source).stat()
    identity = (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    digest = sha(source)
    chunks = []
    raw_valid_id = 0
    with fits.open(source, memmap=True) as f:
        a = f[1].data
        if len(a) != expected_rows:
            raise ValueError("Native LRS row count differs")
        for start in range(0, len(a), 100000):
            block = a[start : start + 100000]
            sid, valid = integer_gaia_ids(block["gaia_source_id"])
            raw_valid_id += int(valid.sum())
            snr = np.asarray(block["snrg"], dtype="f8")
            value = np.column_stack([block[k] for k in ["teff", "logg", "feh"]]).astype(
                "f8"
            )
            error = np.column_stack(
                [block[k] for k in ["teff_err", "logg_err", "feh_err"]]
            ).astype("f8")
            observed = (
                np.isfinite(value) & np.isfinite(error) & (error > 0) & (value != -9999)
            )
            observed[:, 0] &= (value[:, 0] >= 2500) & (value[:, 0] <= 10000)
            observed[:, 1] &= (value[:, 1] >= -1) & (value[:, 1] <= 6.5)
            observed[:, 2] &= (value[:, 2] > -5) & (value[:, 2] < 1)
            keep = valid & np.isfinite(snr) & (snr >= 30) & observed.any(1)
            chunks.append(
                dict(
                    source_id=sid[keep],
                    source_row=np.arange(start, start + len(block), dtype="i8")[keep],
                    value=value[keep],
                    error=error[keep],
                    observed=observed[keep],
                    snr=snr[keep],
                )
            )
    after = Path(source).stat()
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != identity:
        raise ValueError("Source changed during admission")
    data = {key: np.concatenate([block[key] for block in chunks]) for key in chunks[0]}
    before = len(data["source_id"])
    order = np.lexsort((data["source_row"], -data["snr"], data["source_id"]))
    sorted_ids = data["source_id"][order]
    order = order[np.r_[True, np.diff(sorted_ids) != 0]]
    data = {key: value[order] for key, value in data.items()}
    data["benchmark_reserved"] = np.isin(data["source_id"], benchmarks)
    receipt = write_observation_sidecar(
        out / ("lamost-" + args.release + "-lrs-candidate.h5"),
        {source_name: data},
        {source_name: registry},
        provenance=dict(
            source_path=source,
            source_sha256=digest,
            source_rows=expected_rows,
            training_enabled=False,
            metadata=registry,
        ),
    )
    role_counts = {}
    pos = np.searchsorted(ids, data["source_id"])
    matched = pos < len(ids)
    matched[matched] &= ids[pos[matched]] == data["source_id"][matched]
    for role in ["fit", "tune", "cal", "devtest", "external", "excluded"]:
        ii = np.flatnonzero(matched)
        ii = ii[(roles[pos[ii]] == role) & eligible[pos[ii]]]
        role_counts[role] = dict(
            stars=len(ii),
            observed_per_target=data["observed"][ii].sum(0).tolist(),
            benchmark_reserved=int(data["benchmark_reserved"][ii].sum()),
        )
    result = dict(
        status="source_candidate_complete",
        training_performed=False,
        model_loaded=False,
        source_path=source,
        source_sha256=digest,
        raw_rows=expected_rows,
        valid_gaia_id_rows=raw_valid_id,
        quality_selected_spectrum_rows=before,
        selected_unique_stars=len(data["source_id"]),
        selected_per_target=data["observed"].sum(0).tolist(),
        benchmark_reserved=int(data["benchmark_reserved"].sum()),
        role_counts=role_counts,
        outside_existing_base=int((~matched).sum()),
        outside_base_scope="feature availability and roles unresolved; not promoted to FIT",
        receipt=receipt,
        benchmark_input_audit_sha256=sha(out / "benchmark-input-audit.json"),
        publisher_checksum_verified=False,
    )
    (out / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
