"""Submit the per-survey [Fe/H] overlap diagnostics as one CANFAR CPU job.

Uses the master-label census observation table already on ARC and every
staged ``native_observations`` sidecar under the stellar-improvement root.
Credentials are prepared exactly as in ``launch_master_label_census``.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

FILES = {
    "scripts/run_master_label_overlap.py": REPO / "scripts/run_master_label_overlap.py",
    "src/masked_stellar_autoencoder/training/label_overlap.py": REPO
    / "src/masked_stellar_autoencoder/training/label_overlap.py",
}
EMPTY = (
    "src/masked_stellar_autoencoder/__init__.py",
    "src/masked_stellar_autoencoder/training/__init__.py",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", default="master_label_overlap_20261110_v1")
    p.add_argument("--census-run", default="master_label_census_20261110_v3")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    from launch_decision_wave import ROOT
    from launch_master_label_census import prepare_credentials

    files = {name: path.read_text() for name, path in FILES.items()}
    files.update({name: "" for name in EMPTY})
    argv = [
        "scripts/run_master_label_overlap.py",
        "--observations",
        ROOT + a.census_run + "/observations.parquet",
        "--existing-root",
        ROOT.rstrip("/"),
        "--output",
        ROOT + a.run_id,
    ]
    command = (
        "import runpy,sys;sys.argv="
        + repr(argv)
        + ";runpy.run_path('scripts/run_master_label_overlap.py',run_name='__main__')"
    )
    print(dict(run_id=a.run_id, argv=argv))
    if a.dry_run:
        return
    print("credential:", prepare_credentials())
    from launch_decision_cpu import submit_cpu

    submit_cpu(a.run_id, files, command, cpu=4, ram=32)


if __name__ == "__main__":
    main()
