"""Stage corrected benchmark references and a disabled RAVE label candidate."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from masked_stellar_autoencoder.training.survey_expansion_sources import (
    assemble_benchmarks,
    write_rave_candidate,
)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def stage(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    p = args.parameters
    a = args.abundances
    b = assemble_benchmarks(p, a)
    ids = np.sort(b.loc[b.source_id.notna(), "source_id"].to_numpy(dtype="i8"))
    roles = {}
    if sha(args.fit_ids) != args.fit_ids_sha256:
        raise ValueError("FIT identity file hash differs")
    roles["current_fit"] = np.load(args.fit_ids, allow_pickle=False)
    for role, path, expected in args.prediction:
        if sha(path) != expected:
            raise ValueError("Prediction content differs: " + role)
        with h5py.File(path) as f:
            roles[role] = f["source_id"][:]
    for role, members in roles.items():
        b[role] = b.source_id.isin(members)
    # Nullable IDs are written as integer decimal CSV text, not floating point.
    b.to_csv(out / "benchmarks.csv", index=False)
    np.save(out / "benchmark-fit-exclusions.npy", ids)
    manifest = dict(
        status="source_candidates_staged",
        training_performed=False,
        benchmark_sources={str(p): sha(p), str(a): sha(a)},
        benchmark_rows=len(b),
        direct_diameters=int(b.direct_diameter.sum()),
        indirect_diameters=int((~b.direct_diameter).sum()),
        gaia_ids=len(ids),
        spectral_warning_stars=int(b.spectral_warning.sum()),
        reserved_for_evaluation=True,
        future_fit_exclusions=len(ids),
        benchmark_role_counts={role: int(b[role].sum()) for role in roles},
        benchmarks_not_in_checked_roles=int((~b[list(roles)].any(axis=1)).sum()),
        checked_roles=list(roles),
        remaining_role_scope="TUNE/CAL and input-ineligible/not-in-cohort unresolved",
        reference_dependencies=[
            "Teff bolometric flux may use Gaia XP/photometry also supplied to the model",
            "gravity uses parallax and evolutionary masses; not model-independent dynamical gravity",
            "FeI/H and FeII/H are separate spectroscopy-based references using fixed fundamental atmosphere parameters",
            "survey calibration ancestry remains to audit; new release does not guarantee independent standards",
        ],
    )
    block, receipt = write_rave_candidate(
        args.madera, args.iron, args.rave_match, out / "rave-candidate.h5", ids
    )
    manifest["rave"] = dict(
        receipt=receipt,
        selected_unique_stars=len(block["source_id"]),
        observed_per_target=block["observed"].sum(0).tolist(),
        known_error_per_target=(block["observed"] & block["error_known"])
        .sum(0)
        .tolist(),
        benchmark_reserved=int(block["benchmark_reserved"].sum()),
        training_enabled=False,
        role_counts={
            role: int(np.isin(block["source_id"], members).sum())
            for role, members in roles.items()
        },
        future_fit_candidates_after_benchmark_exclusion=int(
            (
                np.isin(block["source_id"], roles["current_fit"])
                & ~block["benchmark_reserved"]
                & block["observed"].any(1)
            ).sum()
        ),
    )
    manifest["artifacts"] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in [
        "parameters",
        "abundances",
        "fit-ids",
        "madera",
        "iron",
        "rave-match",
        "output",
    ]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--fit-ids-sha256", required=True)
    p.add_argument(
        "--prediction",
        nargs=3,
        action="append",
        default=[],
        metavar=("ROLE", "HDF5", "SHA256"),
    )
    stage(p.parse_args())


if __name__ == "__main__":
    main()
