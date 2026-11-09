"""Source-only provisional DR20 BOSSNet and DR12 MRS LASP admission."""

import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from astropy.io import fits

from masked_stellar_autoencoder.training.multisurvey_overlay import (
    write_observation_sidecar,
)

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
SOURCES = [
    dict(
        name="sdss_dr20_bossnet_081",
        path=ROOT
        + "decision_dr20_legacy_source_audit_20261106_v1/dr20_bossnet.fits.gz",
        columns=["teff", "logg", "fe_h"],
        errors=["e_teff", "e_logg", "e_fe_h"],
        id="gaia_dr3_source_id",
        release="SDSS DR20 Astra 0.8.1 BOSSNet",
        pipeline="BOSSNet SDSS-V BOSS",
        snr="snr",
        flag="flag_bad",
    ),
    dict(
        name="lamost_dr12v2_mrs_lasp",
        path=ROOT
        + "multisurvey_lamost_dr12_mrs_source_direct_v2/dr12_mrs_stellar.fits.gz",
        columns=["teff_lasp", "logg_lasp", "feh_lasp"],
        errors=["teff_lasp_err", "logg_lasp_err", "feh_lasp_err"],
        id="gaia_source_id",
        release="LAMOST DR12 v2.0 MRS",
        pipeline="LASP MRS; CNN and alpha/M retained only in raw source, not aliased",
        snr="snr",
    ),
]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmarks", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    benchmarks = np.array(json.loads(a.benchmarks.read_text()), dtype="i8")
    result = dict(
        status="running",
        training_performed=False,
        role_assignment_performed=False,
        sources={},
    )

    def save():
        (a.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    for spec in SOURCES:
        source = Path(spec["path"])
        # Preceding source acquisition is independent; wait only on CPU.
        import time

        deadline = time.monotonic() + 6 * 3600
        while not source.exists() and time.monotonic() < deadline:
            time.sleep(30)
        if not source.exists():
            raise RuntimeError("Required source not acquired by deadline")
        if spec["name"].startswith("sdss"):
            manifest = json.loads(
                Path(
                    ROOT + "decision_dr20_legacy_source_audit_20261106_v1/manifest.json"
                ).read_text()
            )
            if sha(source) != manifest["sources"]["dr20_bossnet"]["sha256"]:
                raise ValueError("DR20 source SHA differs")
        elif (
            sha(source)
            != "e10551be296cddcacf0ceb9c9b2827884d10e550b14850adb01348435be12369"
        ):
            raise ValueError("MRS archive differs")
        raw = Path("/scratch") / (spec["name"] + ".fits")
        with gzip.open(source, "rb") as x, raw.open("xb") as y:
            shutil.copyfileobj(x, y, 8 * 1024 * 1024)
        pieces = []
        with fits.open(raw, memmap=True) as f:
            table = f[1].data
            names = {n.lower(): n for n in f[1].columns.names}
            for start in range(0, len(table), 100000):
                b = table[start : start + 100000]
                raw_ids = np.asarray(b[names[spec["id"]]])
                if raw_ids.dtype.kind in "SU":
                    text = np.char.strip(raw_ids.astype("U"))
                    valid = np.char.isdecimal(text) & (np.char.str_len(text) <= 19)
                    ids = np.array(
                        [
                            int(t) if ok and int(t) <= np.iinfo("i8").max else 0
                            for t, ok in zip(text, valid)
                        ],
                        dtype="i8",
                    )
                elif raw_ids.dtype.kind in "iu":
                    ids = raw_ids.astype("i8")
                else:
                    raise ValueError("Native IDs must be exact")
                val = np.column_stack([b[names[k]] for k in spec["columns"]]).astype(
                    "f8"
                )
                err = np.column_stack([b[names[k]] for k in spec["errors"]]).astype(
                    "f8"
                )
                snr = np.asarray(b[names[spec["snr"]]], dtype="f8")
                observed = (
                    np.isfinite(val) & np.isfinite(err) & (err > 0) & (val != -9999)
                )
                observed[:, 0] &= (val[:, 0] >= 2500) & (val[:, 0] <= 10000)
                observed[:, 1] &= (val[:, 1] >= -1) & (val[:, 1] <= 6.5)
                observed[:, 2] &= (val[:, 2] > -5) & (val[:, 2] < 1)
                keep = (ids > 0) & np.isfinite(snr) & (snr >= 30) & observed.any(1)
                if spec.get("flag"):
                    keep &= ~np.asarray(b[names[spec["flag"]]], dtype=bool)
                pieces.append(
                    dict(
                        source_id=ids[keep],
                        source_row=np.arange(start, start + len(b), dtype="i8")[keep],
                        value=val[keep],
                        error=err[keep],
                        observed=observed[keep],
                        snr=snr[keep],
                    )
                )
            rows = len(table)
        block = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        order = np.lexsort((block["source_row"], -block["snr"], block["source_id"]))
        sorted_ids = block["source_id"][order]
        order = order[np.r_[True, np.diff(sorted_ids) != 0]]
        block = {k: v[order] for k, v in block.items()}
        block["benchmark_reserved"] = np.isin(block["source_id"], benchmarks)
        metadata = dict(
            release=spec["release"],
            pipeline=spec["pipeline"],
            training_enabled=False,
            dependency_id=spec["name"],
            quality_policy="Provisional SNR>=30, finite positive target errors, broad FGKM target domains; BOSSNet flag_bad false; highest-SNR native Gaia-ID repeat; not a published final quality certificate",
            targets={
                k: dict(quantity=k, unit=u, field=field)
                for k, u, field in zip(
                    ["teff", "logg", "fe_h"],
                    ["K", "dex(cm/s2)", "dex"],
                    spec["columns"],
                )
            },
        )
        receipt = write_observation_sidecar(
            a.output / (spec["name"] + "-candidate.h5"),
            {spec["name"]: block},
            {spec["name"]: metadata},
            provenance=dict(
                source_path=str(source),
                source_sha256=sha(source),
                training_enabled=False,
            ),
        )
        result["sources"][spec["name"]] = dict(
            status="source_candidate_complete",
            raw_rows=rows,
            selected_unique_stars=len(block["source_id"]),
            observed_per_target=block["observed"].sum(0).tolist(),
            benchmark_reserved=int(block["benchmark_reserved"].sum()),
            receipt=receipt,
        )
        save()
        print(json.dumps(result["sources"][spec["name"]]), flush=True)
    result["status"] = "additional_native_source_candidates_complete"
    save()


if __name__ == "__main__":
    main()
