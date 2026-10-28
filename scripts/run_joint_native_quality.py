"""Bounded independent QA of an existing native sidecar and original ARC files.

No projection, source acquisition, model fitting, output replacement or submission.
"""

import argparse
import json
import signal
import subprocess
from pathlib import Path

import h5py
import yaml
from prepare_joint_native_labels import immutable_json, preflight
from verify_joint_native_sidecar import digest, verify


def verify_existing(config, sidecar_directory, output_directory, *, source_sha):
    """Verify the existing immutable sidecar, writing a distinct strengthened report."""
    sidecar_directory = Path(sidecar_directory)
    output_directory = Path(output_directory)
    if sidecar_directory.resolve() == output_directory.resolve():
        raise ValueError("Quality output must differ from the preparation output")
    preflight(config)
    if digest(config["cohort"]["path"]) != config["cohort"]["sha256"]:
        raise ValueError("Actual cohort differs from accepted source pin")
    sidecar = sidecar_directory / config["sidecar_filename"]
    pins = json.loads((sidecar_directory / "source-pins.json").read_text())
    if set(pins) != set(config["sources"]):
        raise ValueError("Recorded source pins do not match configured sources")
    for source, spec in config["sources"].items():
        if spec["sha256"] is not None and pins[source] != spec["sha256"]:
            raise ValueError("Source pin differs from accepted acquisition")
    with h5py.File(sidecar) as f:
        if not bool(f.attrs.get("complete", False)):
            raise ValueError("Existing native sidecar is incomplete")
        provenance = json.loads(f.attrs["provenance_json"])
        if provenance["cohort_sha256"] != config["cohort"]["sha256"]:
            raise ValueError("Sidecar cohort differs from accepted source pin")
        if (
            provenance["reference_holdout_ordered_id_sha256"]
            != config["references"]["ordered_id_sha256"]
        ):
            raise ValueError("Sidecar reference union differs from accepted pin")
    # Do not overwrite the original, weaker verifier/preparation JSON. The verifier
    # rebuilds all target masks, values and source associations before publication.
    output_directory.mkdir(parents=True, exist_ok=True)
    report = verify(
        sidecar,
        config["cohort"]["path"],
        {name: Path(spec["path"]) for name, spec in config["sources"].items()},
        pins,
        output_directory / "native-label-quality-independent-verification.json",
    )
    if not report["all_target_quality_masks_verified"]:
        raise ValueError("Independent target-quality mask gate did not pass")
    with h5py.File(sidecar) as f:
        observed = f["targets/observed"][:]
        quality = f["targets/quality_pass"][:]
        complete = (observed & quality).all(1)
        role_counts = {}
        for name in ("fit", "tune", "calibration", "test", "external_reference"):
            role = f["roles/" + name][:]
            role_counts[name] = {
                "rows": int(role.sum()),
                "complete_targets": int((role & complete).sum()),
                "observed_targets": (observed & quality)[role].sum(0).tolist(),
            }
    manifest = {
        "status": "complete",
        "mode": "joint-native-quality-verification",
        "source_sha": source_sha,
        "cpu_only": True,
        "sidecar_path": str(sidecar),
        "sidecar_sha256": report["sidecar_sha256"],
        "cohort_sha256": report["cohort_sha256"],
        "native_labels_verified": True,
        "all_target_quality_masks_verified": True,
        "source_pins": pins,
        "role_target_counts": role_counts,
        "training_launch_ready": False,
        "scope": "Teff/logg/FeH native-source QA; no distance or full pipeline certification",
    }
    immutable_json(output_directory / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/joint-native-preparation.yaml")
    )
    parser.add_argument("--sidecar-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    resources = config["requested_resources"]
    if config["cpu_only"] is not True or resources["gpus"] != 0:
        raise ValueError("Quality verification requires explicit CPU-only resources")
    limit = int(resources["maximum_runtime_seconds"])
    if not 0 < limit <= 3600:
        raise ValueError("Quality verification requires an at-most-one-hour bound")

    def timeout(_signal, _frame):
        raise TimeoutError("Native quality verification exceeded configured wall time")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(limit)
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        report = verify_existing(
            config, args.sidecar_directory, args.output, source_sha=revision
        )
        print(json.dumps(report, indent=2, allow_nan=False))
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
