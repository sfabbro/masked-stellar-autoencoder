"""Launch the master-label training chain on CANFAR, one step at a time.

Steps, each a headless job that clones this repository at the pushed commit:

1. ``features``: ``recover_remaining_survey_inputs.py``. Input features for
   master-label stars not yet in the parent source (``expanded_six_source_20261107_v5``).
2. ``source``: ``extend_full_raw_source.py``. Appends those stars with
   inherited/hash roles (benchmarks and globular members external) and adds one
   ``native_observations`` group per new master-label source.
3. ``train``: ``run_expanded_six_training.py`` with
   ``configs/master-label-six-nsf-fullfit-20261111.yaml`` (GPU). It waits for
   the source run to finish and applies the per-source map families.

Each step refuses to submit unless HEAD is pushed to the tracking branch and
the previous step's manifest on ARC is complete (checked by the job scripts
themselves). Credentials are prepared as in ``launch_master_label_census``.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ROOT = "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
SOURCE_CONFIG = "configs/master-label-source-20261111.json"
FEATURE_CONFIG = "configs/multisurvey-feature-union-dr12-20261106.json"
TRAIN_CONFIG = "configs/master-label-six-nsf-fullfit-20261111.yaml"
PARENT = "expanded_six_source_20261107_v5"
# v1 (XP-only lookup) was cancelled; v2 keeps photometry-only (no-XP) stars.
FEATURES = "master_label_raw_features_20261111_v2"
SOURCE = "expanded_six_source_20261111_v6"
TRAIN = "master_label_six_nsf_fullfit_20261111_v1"

STEPS = {
    "features": dict(
        run_id=FEATURES,
        gpu=0,
        ram=32,
        command=(
            f"python -u scripts/recover_remaining_survey_inputs.py --config {SOURCE_CONFIG}"
            f" --feature-config {FEATURE_CONFIG} --parent-run {PARENT}"
            " --allow-unknown-photometry"
            f" --output {ROOT}{FEATURES}"
        ),
        files=[SOURCE_CONFIG, FEATURE_CONFIG],
    ),
    "source": dict(
        run_id=SOURCE,
        gpu=0,
        ram=64,
        command=(
            f"python -u scripts/extend_full_raw_source.py --config {SOURCE_CONFIG}"
            f" --parent-run {PARENT} --feature-run {FEATURES} --include-no-xp"
            f" --output {ROOT}{SOURCE}"
        ),
        files=[SOURCE_CONFIG],
    ),
    "train": dict(
        run_id=TRAIN,
        gpu=1,
        ram=32,
        command=(
            f"python -u scripts/run_expanded_six_training.py --config {TRAIN_CONFIG}"
            ' --output "$experiment_output" --reuse-waiting-output'
        ),
        files=[TRAIN_CONFIG],
    ),
}


def pushed_head() -> tuple[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    ).strip()
    remote = subprocess.check_output(
        ["git", "ls-remote", "origin", "refs/heads/" + branch], text=True
    ).split()
    if not remote or remote[0] != head:
        raise SystemExit("Push HEAD to origin/" + branch + " before launching")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise SystemExit("Working tree is not clean")
    return branch, head


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", choices=sorted(STEPS), required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    step = STEPS[a.step]
    files = {name: (HERE.parent / name).read_text() for name in step["files"]}
    print(dict(step=a.step, run_id=step["run_id"], command=step["command"]))
    if a.dry_run:
        return
    branch, head = pushed_head()
    from launch_master_label_census import prepare_credentials

    print("credential:", prepare_credentials(), "commit:", head)
    from launch_decision_wave import submit

    submit(
        step["run_id"],
        files,
        step["command"],
        gpu=step["gpu"],
        cpu=4,
        ram=step["ram"],
        branch=branch,
        commit=head,
    )


if __name__ == "__main__":
    main()
