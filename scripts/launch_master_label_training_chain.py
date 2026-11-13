"""Launch the master-label training chain on CANFAR, one step at a time.

Steps, each a headless job that clones this repository at the pushed commit:

1. ``features``: ``recover_remaining_survey_inputs.py``. Input features for
   master-label stars not yet in the parent source (``expanded_six_source_20261107_v5``).
2. ``source``: ``extend_full_raw_source.py``. Appends those stars with
   inherited/hash roles (benchmarks and globular members external) and adds one
   ``native_observations`` group per new master-label source.
3. ``fullinputs``: ``build_full_input_source.py``. All 140 encoder slots from
   the staged catalogues (NaN + visible) and head metadata context
   (``expanded_six_source_20261111_v7``).
4. ``train``: ``run_expanded_six_training.py`` with
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
# v2 recovered only XP stars (the pretraining files hold only XP stars); v3
# adds the remaining labelled candidates from Gaia DR3 as no-XP stars.
GAIA_FEATURES = "master_label_raw_features_20261111_v3"
SOURCE = "expanded_six_source_20261111_v6"
FULL = "expanded_six_source_20261111_v7"
POOL = "gaia_noxp_unlabelled_pool_20261111_v1"
LOOSE = "loose_survey_labels_20261111_v1"
CHEM = "chemistry_labels_20261111_v1"
# v1 wrote its admission decision before failing the library pin (no
# training); its output cannot be reused, so training restarts as v2.
# v2 (20261111) ran 16-star chunks; v2b is the same run with 256-star chunks.
TRAIN = "master_label_six_nsf_fullfit_20261112_v2b"

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
    "gaiaonly": dict(
        run_id=GAIA_FEATURES,
        gpu=0,
        ram=32,
        command=(
            f"python -u scripts/add_gaia_only_stars.py --feature-run {FEATURES}"
            f" --output {ROOT}{GAIA_FEATURES}"
        ),
        files=[],
    ),
    "source": dict(
        run_id=SOURCE,
        gpu=0,
        ram=64,
        command=(
            f"python -u scripts/extend_full_raw_source.py --config {SOURCE_CONFIG}"
            f" --parent-run {PARENT} --feature-run {GAIA_FEATURES} --include-no-xp"
            f" --output {ROOT}{SOURCE}"
        ),
        files=[SOURCE_CONFIG],
    ),
    "fullinputs": dict(
        run_id=FULL,
        gpu=0,
        ram=96,
        command=(
            f"python -u scripts/build_full_input_source.py --parent-run {SOURCE}"
            f" --feature-run {GAIA_FEATURES}"
            f" --extra-gaia-dir {GAIA_FEATURES}/gaia-only --output {ROOT}{FULL}"
        ),
        files=[],
    ),
    "pool": dict(
        run_id=POOL,
        gpu=0,
        ram=32,
        command=(
            "python -u scripts/build_gaia_unlabelled_pool.py --rate 0.125"
            f" --exclude-source {FULL} --output {ROOT}{POOL}"
        ),
        files=[],
    ),
    "loose": dict(
        run_id=LOOSE,
        gpu=0,
        ram=64,
        command=(
            "python -u scripts/build_loose_survey_sidecars.py --snr-min 10"
            f" --parent-run {SOURCE} --output {ROOT}{LOOSE}"
        ),
        files=[],
    ),
    "chemistry": dict(
        run_id=CHEM,
        gpu=0,
        ram=64,
        command=f"python -u scripts/build_chemistry_sidecars.py --output {ROOT}{CHEM}",
        files=[],
    ),
    "train": dict(
        run_id=TRAIN,
        gpu=1,
        ram=32,
        command=(
            f"python -u scripts/run_expanded_six_training.py --config {TRAIN_CONFIG}"
            ' --output "$experiment_output" --reuse-waiting-output'
            " --retry-pre-admission-failure"
        ),
        files=[TRAIN_CONFIG],
    ),
}


# The no-XP v3 chain (gaia-only stars, full-Gaia pool, SSL phase) was put on
# hold for an XP-only run; its configs stay in configs/*-v3.* for later.


# Consolidated master spectroscopic label table (all records, traceable).
MASTER = "master_spectroscopic_labels_20261111_v1"
STEPS["master"] = dict(
    run_id=MASTER,
    gpu=0,
    ram=128,
    command=(
        "python -u scripts/build_master_spectroscopic_labels.py"
        f" --output {ROOT}{MASTER}"
    ),
    files=[],
)


# v2 adds the extra census sources (census v4) and the DESI SP / LAMOST MRS
# CNN pipelines; unchanged staged surveys are copied from v1's work/stage.
MASTER_V2 = "master_spectroscopic_labels_20261111_v2"
CENSUS_V4 = "master_label_census_20261111_v4"
STEPS["master-v2"] = dict(
    run_id=MASTER_V2,
    gpu=0,
    ram=160,
    command=(
        "python -u scripts/build_master_spectroscopic_labels.py"
        f" --census {ROOT}{CENSUS_V4}/observations.parquet"
        f" --reuse-stage {ROOT}{MASTER}/work/stage"
        f" --output {ROOT}{MASTER_V2}"
    ),
    files=[],
)


# v3: census v6 fixes the Bensby+2014 position epoch (VizieR J2000 positions)
# and resolves AMBRE target names through SIMBAD.
MASTER_V3 = "master_spectroscopic_labels_20261111_v3"
CENSUS_V5 = "master_label_census_20261111_v6"
STEPS["master-v3"] = dict(
    run_id=MASTER_V3,
    gpu=0,
    ram=160,
    command=(
        "python -u scripts/build_master_spectroscopic_labels.py"
        f" --census {ROOT}{CENSUS_V5}/observations.parquet"
        f" --reuse-stage {ROOT}{MASTER_V2}/work/stage"
        f" --output {ROOT}{MASTER_V3}"
    ),
    files=[],
)


# Rebuild after the msa_runs loss (2026-11-13): re-download the staged surveys,
# rerun the census (master_label_census_20261113_v1, own launcher) and rebuild
# the master table without training-source annotations (those runs are gone).
STAGE_R = "master_survey_stage_20261113_v1"
CENSUS_R = "master_label_census_20261113_v2"  # v1 lacked SAGA (http-only host)
MASTER_R = "master_spectroscopic_labels_20261113_v1"
STEPS["stage-surveys"] = dict(
    run_id=STAGE_R,
    gpu=0,
    ram=16,
    command=f"python -u scripts/stage_master_survey_files.py --output {ROOT}{STAGE_R}",
    files=[],
)
STEPS["master-r"] = dict(
    run_id=MASTER_R,
    gpu=0,
    ram=160,
    command=(
        "python -u scripts/build_master_spectroscopic_labels.py"
        f" --census {ROOT}{CENSUS_R}/observations.parquet"
        f" --stage {ROOT}{STAGE_R} --training-source ''"
        f" --output {ROOT}{MASTER_R}"
    ),
    files=[],
)


# Stopgap: fine-tune table from the rebuilt master (labels + raw MSA inputs),
# then finetune_msa from the July pretrain checkpoint.
STOPGAP_TABLE = "stopgap_finetune_table_20261113_v1"
STOPGAP_FT = "stopgap_finetune_20261113_v1"
STOPGAP_CONFIG = "configs/finetune-stopgap-20261113.yaml"
STEPS["stopgap-table"] = dict(
    run_id=STOPGAP_TABLE,
    gpu=0,
    ram=96,
    cpu=8,
    command=(
        "python -u scripts/build_stopgap_finetune_table.py"
        f" --master {ROOT}{MASTER_R} --config {STOPGAP_CONFIG}"
        f" --output {ROOT}{STOPGAP_TABLE} --workers 8"
    ),
    files=[STOPGAP_CONFIG],
)
STEPS["stopgap-finetune"] = dict(
    run_id=STOPGAP_FT,
    gpu=1,
    ram=96,
    command=(
        "python -u -m masked_stellar_autoencoder.training.finetune_msa"
        f" --config {STOPGAP_CONFIG}"
    ),
    files=[STOPGAP_CONFIG],
)


# XP-only v3 run. New label groups from census v6 (master sidecar v2:
# highres_literature_extra, kirby2010_dsph, mastar_params; chemistry v2:
# ambre_feros_chem), feature recovery against v6 for every config source,
# source v8 without no-XP stars, full inputs v9, training with xp_only.
CENSUS_XP = "master_label_census_20261111_v6"
SIDECARS_XP = "master_label_sidecars_20261111_v2"
CHEM_XP = "chemistry_labels_20261111_v2"
SOURCE_CONFIG_XP = "configs/master-label-source-20261111-xp.json"
TRAIN_CONFIG_XP = "configs/master-label-six-nsf-fullfit-20261111-xp.yaml"
FEATURES_XP = "master_label_raw_features_20261111_v6"
SOURCE_XP = "expanded_six_source_20261111_v8"
FULL_XP = "expanded_six_source_20261111_v9"
# v3 (20261111) ran 16-star chunks at 2.5 s/update; v4 is the same run with
# 256-star chunks.
TRAIN_XP = "master_label_six_nsf_fullfit_20261112_v4"
STEPS.update(
    {
        "sidecars-xp": dict(
            run_id=SIDECARS_XP,
            gpu=0,
            ram=16,
            command=(
                "python -u scripts/write_master_label_sidecars.py"
                f" --observations {ROOT}{CENSUS_XP}/observations.parquet"
                " --sources highres_literature_extra kirby2010_dsph mastar_params"
                f" --output {ROOT}{SIDECARS_XP}"
                f" --arc-path {ROOT}{SIDECARS_XP}/master-label-sidecar.h5"
            ),
            files=[],
        ),
        "chem-xp": dict(
            run_id=CHEM_XP,
            gpu=0,
            ram=16,
            command=(
                "python -u scripts/build_chemistry_sidecars.py"
                f" --observations {ROOT}{CENSUS_XP}/observations.parquet"
                f" --only ambre_feros_chem --output {ROOT}{CHEM_XP}"
            ),
            files=[],
        ),
        "features-xp": dict(
            run_id=FEATURES_XP,
            gpu=0,
            ram=48,
            command=(
                "python -u scripts/recover_remaining_survey_inputs.py"
                f" --config {SOURCE_CONFIG_XP} --feature-config {FEATURE_CONFIG}"
                f" --parent-run {SOURCE} --output {ROOT}{FEATURES_XP}"
            ),
            files=[SOURCE_CONFIG_XP, FEATURE_CONFIG],
        ),
        "source-xp": dict(
            run_id=SOURCE_XP,
            gpu=0,
            ram=96,
            command=(
                "python -u scripts/extend_full_raw_source.py"
                f" --config {SOURCE_CONFIG_XP} --parent-run {SOURCE}"
                f" --feature-run {FEATURES_XP} --output {ROOT}{SOURCE_XP}"
            ),
            files=[SOURCE_CONFIG_XP],
        ),
        "fullinputs-xp": dict(
            run_id=FULL_XP,
            gpu=0,
            ram=128,
            command=(
                f"python -u scripts/build_full_input_source.py --parent-run {SOURCE_XP}"
                f" --feature-run {GAIA_FEATURES} --feature-run {FEATURES_XP}"
                f" --extra-gaia-dir {GAIA_FEATURES}/gaia-only"
                f" --output {ROOT}{FULL_XP}"
            ),
            files=[],
        ),
        "train-xp": dict(
            run_id=TRAIN_XP,
            gpu=1,
            ram=64,
            command=(
                "python -u scripts/run_expanded_six_training.py"
                f" --config {TRAIN_CONFIG_XP}"
                ' --output "$experiment_output" --reuse-waiting-output'
                " --retry-pre-admission-failure"
            ),
            files=[TRAIN_CONFIG_XP],
        ),
    }
)


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
    p.add_argument(
        "--slices",
        type=int,
        default=1,
        help="run a GPU step data-parallel over this many GPU slices "
        "(one rank per slice, scripts/run_multi_slice.py)",
    )
    p.add_argument(
        "--attempt",
        type=int,
        help="retry attempt number (>=2) after reconciling a failed receipt",
    )
    p.add_argument("--backend", choices=["gloo", "nccl"], default="nccl")
    a = p.parse_args()
    step = dict(STEPS[a.step])
    if a.slices > 1:
        if step["gpu"] != 1:
            raise SystemExit("--slices applies to single-GPU training steps")
        step.update(
            gpu=a.slices,
            cpu=2 * a.slices,
            ram=max(step["ram"], 24 * a.slices),
            command=f"python -u scripts/run_multi_slice.py --slices {a.slices}"
            f" --threads-per-rank 2 --backend {a.backend} -- {step['command']}",
        )
    files = {name: (HERE.parent / name).read_text() for name in step["files"]}
    print(
        dict(
            step=a.step,
            run_id=step["run_id"],
            gpu=step["gpu"],
            cpu=step.get("cpu", 4),
            ram=step["ram"],
            command=step["command"],
        )
    )
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
        cpu=step.get("cpu", 4),
        ram=step["ram"],
        branch=branch,
        commit=head,
        attempt=a.attempt,
    )


if __name__ == "__main__":
    main()
