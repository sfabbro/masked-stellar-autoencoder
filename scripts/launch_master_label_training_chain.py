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

ROOT = "/arc/projects/astroai/msa/runs/"
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
# v2 drops the duplicate-pipeline labels (DESI SP grid-floor artefact in v1).
STOPGAP_TABLE = "stopgap_finetune_table_20261113_v2"
STOPGAP_FT = "stopgap_finetune_20261113_v2"
STOPGAP_CONFIG = "configs/finetune-stopgap-20261113-v2.yaml"
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

STOPGAP_EVAL = "stopgap_eval_20261113_v2"
# Best-validation epoch of the fine-tune (v1: 28); set after the run.
STOPGAP_EPOCH = 26
STEPS["stopgap-eval"] = dict(
    run_id=STOPGAP_EVAL,
    gpu=1,
    ram=96,
    command=(
        "python -u -m masked_stellar_autoencoder.training.eval_ensemble"
        f" --config {STOPGAP_CONFIG}"
        f" --checkpoints {ROOT}{STOPGAP_FT}/finetune_checkpoint_{STOPGAP_EPOCH}.pth"
        ' --out "$experiment_output" --save-predictions'
    ),
    files=[STOPGAP_CONFIG],
)


# Stopgap versus multi-survey on one table (same stars, inputs and split).
MS_TABLE = "multisurvey_table_20261113_v1"
MS_PAIR = {
    "ms-stopgap": (
        "stopgap_finetune_20261113_v3",
        "configs/finetune-stopgap-20261113-v3.yaml",
    ),
    "ms-multi": (
        "multisurvey_finetune_20261113_v1",
        "configs/finetune-multisurvey-20261113.yaml",
    ),
    "ms-multi-v2": (
        "multisurvey_finetune_20261115_v2",
        "configs/finetune-multisurvey-20261115-v2.yaml",
    ),
    "ms-multi-v3": (
        "multisurvey_finetune_20261115_v3",
        "configs/finetune-multisurvey-20261115-v3.yaml",
    ),
    "ms-multi-v4": (
        "multisurvey_finetune_20261115_v4",
        "configs/finetune-multisurvey-20261115-v4.yaml",
    ),
}
# Best-validation epoch of each fine-tune; set after the runs.
MS_EPOCH = {
    "ms-stopgap": 23,
    "ms-multi": 23,
    "ms-multi-v2": 23,
    "ms-multi-v3": 23,
    "ms-multi-v4": 23,
}
STEPS["probe-image"] = dict(
    run_id="probe_image_base_20261113_v1",
    gpu=0,
    ram=8,
    command=(
        'python -u -c "import sys,subprocess;print(sys.version);'
        "print(subprocess.run(['git','--version'],capture_output=True,text=True).stdout)\""
    ),
    files=[],
)
STEPS["ms-table"] = dict(
    run_id=MS_TABLE,
    gpu=0,
    ram=128,
    cpu=8,
    command=(
        "python -u scripts/build_stopgap_finetune_table.py"
        f" --master {ROOT}{MASTER_R} --config {STOPGAP_CONFIG}"
        " --references /arc/projects/k-pop/catalogues"
        f" --output {ROOT}{MS_TABLE} --workers 8"
    ),
    files=[STOPGAP_CONFIG],
)
# v5: semi-supervised on a larger table with IRFM Teff anchors; the v1 table's
# test stars are left out so every model is scored on them (ms-v5-eval).
MS_TABLE_V5 = "multisurvey_table_20261117_v5"
MS_V5 = (
    "multisurvey_finetune_20261117_v5",
    "configs/finetune-multisurvey-20261117-v5.yaml",
)
MS_V5_EPOCH = 11
STEPS["ms-table-v5"] = dict(
    run_id=MS_TABLE_V5,
    gpu=0,
    ram=192,
    cpu=8,
    command=(
        "python -u scripts/build_stopgap_finetune_table.py"
        f" --master {ROOT}{MASTER_R} --config {STOPGAP_CONFIG}"
        " --references /arc/projects/k-pop/catalogues --irfm-anchors"
        f" --exclude-test-of {ROOT}{MS_TABLE} --unlabelled-fraction 0.03"
        f" --max-stars 8000000 --output {ROOT}{MS_TABLE_V5} --workers 8"
    ),
    files=[STOPGAP_CONFIG],
)
STEPS["ms-multi-v5"] = dict(
    run_id=MS_V5[0],
    gpu=1,
    ram=192,
    cpu=8,
    command=(
        "python -u -m masked_stellar_autoencoder.training.finetune_msa"
        f" --config {MS_V5[1]}"
    ),
    files=[MS_V5[1]],
)
_V5_EVAL = [
    ("stopgap", MS_PAIR["ms-stopgap"], 23),
    ("multi_v4", MS_PAIR["ms-multi-v4"], 23),
    ("multi_v5", MS_V5, MS_V5_EPOCH),
]
# CPU: prediction on ~130k test stars needs no GPU, and the GPU node
# keel-prod-g0123 fails with an NVML driver/library mismatch.
STEPS["ms-v5-eval"] = dict(
    run_id="multisurvey_eval_20261117_v5",
    gpu=0,
    cpu=8,
    ram=64,
    # The entrypoint wraps only the first command in pixi run; wrap the rest.
    command=' && "${pixi_run[@]}" '.join(
        "python -u scripts/predict_fits_rows.py"
        f" --config {_cfg} --checkpoint {ROOT}{_run}/finetune_checkpoint_{_ep}.pth"
        f" --fits {ROOT}{MS_TABLE}/finetune.fits"
        f" --test-split {MS_PAIR['ms-multi-v4'][1]} --irfm"
        f' --out "$experiment_output/{_tag}_test.npz"'
        for _tag, (_run, _cfg), _ep in _V5_EVAL
    ),
    files=[MS_V5[1], MS_PAIR["ms-multi-v4"][1], MS_PAIR["ms-stopgap"][1]],
)
# v6: anchored formulation (see its config). Reddening labels first (CPU):
# intrinsic colours fitted on the v5 table's training split, applied to the
# v5 table (training) and the v1 table (evaluation on its test stars).
MS_V6 = (
    "multisurvey_finetune_20261118_v6",
    "configs/finetune-multisurvey-20261118-v6.yaml",
)
MS_V6_EPOCH = 10  # set from anchor_metrics.jsonl (lowest score) before eval
REDDENING_V6 = "reddening_labels_20261118_v6"
STEPS["ms-reddening-v6"] = dict(
    run_id=REDDENING_V6,
    gpu=0,
    cpu=8,
    ram=128,
    command=(
        "python -u scripts/build_reddening_labels.py"
        f" --fit-table {ROOT}{MS_TABLE_V5}/finetune.fits --split-config {MS_V6[1]}"
        f" --apply v5={ROOT}{MS_TABLE_V5}/finetune.fits v1={ROOT}{MS_TABLE}/finetune.fits"
        ' --out "$experiment_output"'
    ),
    files=[MS_V6[1]],
)
STEPS["ms-multi-v6"] = dict(
    run_id=MS_V6[0],
    gpu=1,
    ram=192,
    cpu=8,
    command=(
        "python -u -m masked_stellar_autoencoder.training.finetune_msa"
        f" --config {MS_V6[1]}"
    ),
    files=[MS_V6[1]],
)
STEPS["ms-v6-eval"] = dict(
    run_id="multisurvey_eval_20261118_v6",
    gpu=0,
    cpu=8,
    ram=64,
    command=(
        "python -u scripts/predict_fits_rows.py"
        f" --config {MS_V6[1]}"
        f" --checkpoint {ROOT}{MS_V6[0]}/finetune_checkpoint_{MS_V6_EPOCH}.pth"
        f" --fits {ROOT}{MS_TABLE}/finetune.fits"
        f" --test-split {MS_PAIR['ms-multi-v4'][1]} --irfm"
        f" --extra-labels {ROOT}{REDDENING_V6}/v1_reddening.npz"
        ' --out "$experiment_output/multi_v6_test.npz"'
    ),
    files=[MS_V6[1], MS_PAIR["ms-multi-v4"][1]],
)
# v7: v6 + clean label input, censored survey [Fe/H] floors, SLAM domain cut,
# no proper motions / RA / Dec in the encoder (scripts/make_v7_configs.py).
# Evaluation conditions the head on the measured Gaia parallax.
MS_V7 = (
    "multisurvey_finetune_20261118_v7",
    "configs/finetune-multisurvey-20261118-v7.yaml",
)
MS_V7_EPOCH = 9  # set from anchor_metrics.jsonl (lowest score) before eval
STEPS["ms-multi-v7"] = dict(
    run_id=MS_V7[0],
    gpu=1,
    ram=192,
    cpu=8,
    command=(
        "python -u -m masked_stellar_autoencoder.training.finetune_msa"
        f" --config {MS_V7[1]}"
    ),
    files=[MS_V7[1]],
)
STEPS["ms-v7-eval"] = dict(
    run_id="multisurvey_eval_20261118_v7",
    gpu=0,
    cpu=8,
    ram=64,
    command=(
        "python -u scripts/predict_fits_rows.py"
        f" --config {MS_V7[1]}"
        f" --checkpoint {ROOT}{MS_V7[0]}/finetune_checkpoint_{MS_V7_EPOCH}.pth"
        f" --fits {ROOT}{MS_TABLE}/finetune.fits"
        f" --test-split {MS_PAIR['ms-multi-v4'][1]} --irfm --condition-parallax 5"
        f" --extra-labels {ROOT}{REDDENING_V6}/v1_reddening.npz"
        ' --out "$experiment_output/multi_v7_test.npz"'
    ),
    files=[MS_V7[1], MS_PAIR["ms-multi-v4"][1]],
)
# Label-free [Fe/H] consistency diagnostic: wide-binary (El-Badry+2021, MSMS,
# R_chance < 0.1, both G < 17.65) and cluster members (Hunt & Reffert 2023,
# p > 0.5, incl. globulars); targets in ROOT/consistency_targets_20261118.npz.
CONSIST = "consistency_table_20261118"
for _tag, (_run, _cfg), _ep in (("v6", MS_V6, MS_V6_EPOCH), ("v7", MS_V7, MS_V7_EPOCH)):
    STEPS[f"ms-consist-{_tag}"] = dict(
        run_id=f"consistency_pred_20261118_{_tag}",
        gpu=0,
        cpu=8,
        ram=64,
        command=(
            "python -u scripts/build_consistency_table.py"
            f" --targets {ROOT}consistency_targets_20261118.npz --config {MS_V6[1]}"
            f" --output {ROOT}{CONSIST} --workers 8"
            f" --labelled-table {ROOT}{MS_TABLE_V5}/finetune.fits"
            ' && "${pixi_run[@]}" python -u scripts/predict_fits_rows.py'
            f" --config {_cfg} --checkpoint {ROOT}{_run}/finetune_checkpoint_{_ep}.pth"
            f" --fits {ROOT}{CONSIST}/stars.fits --condition-parallax 5"
            f' --out "$experiment_output/consistency_{_tag}.npz"'
        ),
        files=[MS_V6[1], _cfg],
    )
# Frozen re-scoring: every checkpoint predicts the same stars (all table rows
# with an [Fe/H] core record, any split), with parallax conditioning, so
# variants that exclude or deduplicate sources are compared on one set.
RESCORE = "rescore_20261119"
_ABL_BEST = {
    "a01_v6": 6, "a02_clean": 5, "a03_censor": 6, "a04_nopm": 6, "a05_nosky": 6,
    "a06_small_noncore": 6, "a07_small_all": 6, "b01_v7": 6, "b02_v7_small": 6,
    "b03_v7_ctx_ebv": 6, "b04_v7_ctx_plx": 6, "b05_v7_posenc": 6, "b06_v7_dust": 6,
    "b07_v7_noperturb": 5, "c01_v7_apogee_lean": 6, "c02_v7_litdedupe": 6,
    "c03_v7_lean_lit_noastronn": 6,
}  # fmt: skip
_LATENT = ("multi_v7", "c03_v7_lean_lit_noastronn", "a01_v6")


def _rescore(models):
    rows = f"{ROOT}{RESCORE}/rows.npy"
    cmds = [
        f"python -u scripts/core_eval_rows.py --config {MS_V7[1]}"
        f" --output {ROOT}{RESCORE}"
    ]
    for name, (cfg, ckpt) in models.items():
        latent = " --save-latent" if name in _LATENT else ""
        cmds.append(
            f"python -u scripts/predict_fits_rows.py --config {cfg}"
            f" --checkpoint {ckpt} --fits {ROOT}{MS_TABLE_V5}/finetune.fits"
            f" --rows {rows} --condition-parallax 5{latent}"
            f' --out "$experiment_output/{name}.npz" || echo RESCORE_FAILED {name}'
        )
    return ' && "${pixi_run[@]}" '.join(cmds)


_models = {
    "multi_v6": (MS_V6[1], f"{ROOT}{MS_V6[0]}/finetune_checkpoint_{MS_V6_EPOCH}.pth"),
    "multi_v7": (MS_V7[1], f"{ROOT}{MS_V7[0]}/finetune_checkpoint_{MS_V7_EPOCH}.pth"),
}
for _n, _ep in _ABL_BEST.items():
    _models[_n] = (
        f"configs/ablation-20261118-v7/{_n}.yaml",
        f"{ROOT}multisurvey_ablation_20261118_v7/{_n}/finetune_checkpoint_{_ep}.pth",
    )
STEPS["ms-rescore"] = dict(
    run_id="rescore_pred_20261119",
    gpu=0,
    cpu=8,
    ram=64,
    command=_rescore(_models),
    files=sorted({c for c, _ in _models.values()}),
)
# Second wave (c03 setup): metal-poor/scale experiments and mixture heads.
# Best epochs by anchor score (set from each variant's anchor_metrics.jsonl).
_ABL_BEST_2 = {
    "d00_c03_seed43": 6, "d01_c03_censorband": 6, "d03_c03_binary": 4,
}  # fmt: skip
_models_2 = {
    _n: (
        f"configs/ablation-20261118-v7/{_n}.yaml",
        f"{ROOT}multisurvey_ablation_20261118_v7/{_n}/finetune_checkpoint_{_ep}.pth",
    )
    for _n, _ep in _ABL_BEST_2.items()
}
STEPS["ms-rescore-2"] = dict(
    run_id="rescore_pred_20261119_2",
    gpu=0,
    cpu=8,
    ram=64,
    command=_rescore(_models_2).replace(
        " --condition-parallax 5",
        " --condition-parallax 5"
        " --feh-prior configs/feh_prior_multisurvey_table_20261117_v5.npz",
    ),
    files=sorted({c for c, _ in _models_2.values()}),
)


def _rescore_step(run_id, best):
    models = {
        n: (
            f"configs/ablation-20261118-v7/{n}.yaml",
            f"{ROOT}multisurvey_ablation_20261118_v7/{n}/finetune_checkpoint_{ep}.pth",
        )
        for n, ep in best.items()
    }
    return dict(
        run_id=run_id,
        gpu=0,
        cpu=8,
        ram=64,
        command=_rescore(models).replace(
            " --condition-parallax 5",
            " --condition-parallax 5"
            " --feh-prior configs/feh_prior_multisurvey_table_20261117_v5.npz",
        ),
        files=sorted({c for c, _ in models.values()}),
    )


STEPS["ms-rescore-3"] = _rescore_step("rescore_pred_20261119_3", {"d02_c03_balance": 6})
STEPS["ms-rescore-4"] = _rescore_step(
    "rescore_pred_20261119_4",
    {"e01_c03_mix3": 6, "e02_c03_mix3_gauss": 6, "e03_c03_mix3_gauss_diag": 5},
)
# Binary/cluster consistency predictions of selected ablation checkpoints.
_CONSIST_ABL = ("d00_c03_seed43", "d03_c03_binary")
STEPS["ms-consist-abl"] = dict(
    run_id="consistency_pred_20261119_abl",
    gpu=0,
    cpu=8,
    ram=64,
    command=' && "${pixi_run[@]}" '.join(
        f"python -u scripts/predict_fits_rows.py --config {_models_2[n][0]}"
        f" --checkpoint {_models_2[n][1]} --fits {ROOT}{CONSIST}/stars.fits"
        f' --condition-parallax 5 --out "$experiment_output/consistency_{n}.npz"'
        f" || echo CONSIST_FAILED {n}"
        for n in _CONSIST_ABL
    ),
    files=sorted({_models_2[n][0] for n in _CONSIST_ABL}),
)
# v8: v7 + c03 sources + cool binary pool (+ the head / sampling chosen from
# the ablations; scripts/make_v8_config.py). The pool is built first.
MS_V8 = (
    "multisurvey_finetune_20261119_v8",
    "configs/finetune-multisurvey-20261119-v8.yaml",
)
MS_V8_EPOCH = 12  # set from anchor_metrics.jsonl (lowest score) before eval
STEPS["ms-multi-v8"] = dict(
    run_id=MS_V8[0],
    gpu=1,
    ram=192,
    cpu=8,
    command=(
        "python -u scripts/build_binary_pool.py"
        f" --stars {ROOT}{CONSIST}/stars.fits"
        f" --teacher {ROOT}consistency_pred_20261118_v7/consistency_v7.npz"
        f" --targets {ROOT}consistency_targets_20261118.npz --config {MS_V6[1]}"
        f" --output {ROOT}binary_pool_20261119_cool/pool.npz --max-companion-teff 5000"
        ' && "${pixi_run[@]}" python -u -m masked_stellar_autoencoder.training.finetune_msa'
        f" --config {MS_V8[1]}"
    ),
    files=[MS_V8[1], MS_V6[1]],
)
_V8_CKPT = f"{ROOT}{MS_V8[0]}/finetune_checkpoint_{MS_V8_EPOCH}.pth"
_PRIOR = " --feh-prior configs/feh_prior_multisurvey_table_20261117_v5.npz"
STEPS["ms-v8-eval"] = dict(
    run_id="multisurvey_eval_20261119_v8",
    gpu=0,
    cpu=8,
    ram=64,
    command=' && "${pixi_run[@]}" '.join(
        [
            "python -u scripts/predict_fits_rows.py"
            f" --config {MS_V8[1]} --checkpoint {_V8_CKPT}"
            f" --fits {ROOT}{MS_TABLE}/finetune.fits"
            f" --test-split {MS_PAIR['ms-multi-v4'][1]} --irfm --condition-parallax 5"
            f" --extra-labels {ROOT}{REDDENING_V6}/v1_reddening.npz{_PRIOR}"
            ' --out "$experiment_output/multi_v8_test.npz"',
            "python -u scripts/predict_fits_rows.py"
            f" --config {MS_V8[1]} --checkpoint {_V8_CKPT}"
            f" --fits {ROOT}{MS_TABLE_V5}/finetune.fits"
            f" --rows {ROOT}{RESCORE}/rows.npy --condition-parallax 5{_PRIOR}"
            ' --out "$experiment_output/multi_v8_core.npz"',
            "python -u scripts/predict_fits_rows.py"
            f" --config {MS_V8[1]} --checkpoint {_V8_CKPT}"
            f" --fits {ROOT}{CONSIST}/stars.fits --condition-parallax 5{_PRIOR}"
            ' --out "$experiment_output/consistency_v8.npz"',
        ]
    ),
    files=[MS_V8[1], MS_PAIR["ms-multi-v4"][1]],
)
# Full-data v8 ablations (scripts/make_v8_config.py --variant): v8 without the
# cool binary pool, and v8 without balanced sampling. Each trains 12 epochs,
# then predicts the frozen [Fe/H] core and the consistency stars at epoch 12.
for _v in ("nopool", "nobalance"):
    _run = f"{MS_V8[0]}_{_v}"
    _cfg = f"configs/finetune-multisurvey-20261119-v8-{_v}.yaml"
    _ck = f"{ROOT}{_run}/finetune_checkpoint_12.pth"
    STEPS[f"ms-multi-v8-{_v}"] = dict(
        run_id=_run,
        gpu=1,
        ram=192,
        cpu=8,
        command=' && "${pixi_run[@]}" '.join(
            [
                f"python -u -m masked_stellar_autoencoder.training.finetune_msa --config {_cfg}",
                f"python -u scripts/predict_fits_rows.py --config {_cfg} --checkpoint {_ck}"
                f" --fits {ROOT}{MS_TABLE_V5}/finetune.fits"
                f" --rows {ROOT}{RESCORE}/rows.npy --condition-parallax 5{_PRIOR}"
                f' --out "$experiment_output/multi_v8_{_v}_core.npz"',
                f"python -u scripts/predict_fits_rows.py --config {_cfg} --checkpoint {_ck}"
                f" --fits {ROOT}{CONSIST}/stars.fits --condition-parallax 5{_PRIOR}"
                f' --out "$experiment_output/consistency_v8_{_v}.npz"',
            ]
        ),
        files=[_cfg],
    )
# Test-split evaluation of the current best model (v8 without the pool).
_NP = (f"{MS_V8[0]}_nopool", "configs/finetune-multisurvey-20261119-v8-nopool.yaml")
STEPS["ms-v8-nopool-eval"] = dict(
    run_id="multisurvey_eval_20261120_v8_nopool",
    gpu=0,
    cpu=8,
    ram=64,
    command=(
        f"python -u scripts/predict_fits_rows.py --config {_NP[1]}"
        f" --checkpoint {ROOT}{_NP[0]}/finetune_checkpoint_12.pth"
        f" --fits {ROOT}{MS_TABLE}/finetune.fits"
        f" --test-split {MS_PAIR['ms-multi-v4'][1]} --irfm --condition-parallax 5"
        f" --extra-labels {ROOT}{REDDENING_V6}/v1_reddening.npz{_PRIOR}"
        ' --out "$experiment_output/multi_v8_nopool_test.npz"'
    ),
    files=[_NP[1], MS_PAIR["ms-multi-v4"][1]],
)
# v9: v8 without the fixed pool + wide-binary self-consistency (pairs pool:
# companions < 5000 K tied to the model's own primary prediction). The pair
# pool is built first (primaries' inputs included); predictions at epoch 12.
MS_V9 = (
    "multisurvey_finetune_20261120_v9",
    "configs/finetune-multisurvey-20261120-v9.yaml",
)
_V9_CK = f"{ROOT}{MS_V9[0]}/finetune_checkpoint_12.pth"
STEPS["ms-multi-v9"] = dict(
    run_id=MS_V9[0],
    gpu=1,
    ram=192,
    cpu=8,
    command=' && "${pixi_run[@]}" '.join(
        [
            "python -u scripts/build_binary_pool.py"
            f" --stars {ROOT}{CONSIST}/stars.fits"
            f" --teacher {ROOT}consistency_pred_20261118_v7/consistency_v7.npz"
            f" --targets {ROOT}consistency_targets_20261118.npz --config {MS_V6[1]}"
            f" --output {ROOT}binary_pool_20261120_pairs/pool.npz"
            " --max-companion-teff 5000",
            f"python -u -m masked_stellar_autoencoder.training.finetune_msa --config {MS_V9[1]}",
            f"python -u scripts/predict_fits_rows.py --config {MS_V9[1]} --checkpoint {_V9_CK}"
            f" --fits {ROOT}{MS_TABLE}/finetune.fits"
            f" --test-split {MS_PAIR['ms-multi-v4'][1]} --irfm --condition-parallax 5"
            f" --extra-labels {ROOT}{REDDENING_V6}/v1_reddening.npz{_PRIOR}"
            ' --out "$experiment_output/multi_v9_test.npz"',
            f"python -u scripts/predict_fits_rows.py --config {MS_V9[1]} --checkpoint {_V9_CK}"
            f" --fits {ROOT}{MS_TABLE_V5}/finetune.fits"
            f" --rows {ROOT}{RESCORE}/rows.npy --condition-parallax 5{_PRIOR}"
            ' --out "$experiment_output/multi_v9_core.npz"',
            f"python -u scripts/predict_fits_rows.py --config {MS_V9[1]} --checkpoint {_V9_CK}"
            f" --fits {ROOT}{CONSIST}/stars.fits --condition-parallax 5{_PRIOR}"
            ' --out "$experiment_output/consistency_v9.npz"',
        ]
    ),
    files=[MS_V9[1], MS_V6[1], MS_PAIR["ms-multi-v4"][1]],
)
# Likelihood inputs (docs/design-20261120-likelihood-inference.md, sec. 4.1):
# per-star Gaia DR3 XP coefficient covariances (coefficients, errors, packed
# correlations) and astrometric covariances + L21 parallax zero point, from the
# HATS xp_continuous_mean_spectrum and gaia_source catalogues, for every v5
# table row (table order) and a 1M source_id-hash subset of its unlabelled
# stars; one pass over the partitions, then validation (packing, PD,
# table-vs-HATS normalisation). Outputs sidecar_table.h5,
# sidecar_unlabelled.h5, manifest.json, validation.json.
_V5_DIR = f"{ROOT}{MS_TABLE_V5}"
STEPS["covariance-sidecar"] = dict(
    run_id="covariance_sidecar_20261120",
    gpu=0,
    cpu=16,
    ram=64,
    command=' && "${pixi_run[@]}" '.join(
        [
            "python -u scripts/build_covariance_sidecar.py"
            f" --input table={_V5_DIR}/finetune.fits"
            f" --input unlabelled={_V5_DIR}/unlabelled.npz"
            " --sample unlabelled=1000000 --workers 16"
            ' --output "$experiment_output"',
            "python -u scripts/validate_covariance_sidecar.py"
            ' --sidecar table="$experiment_output/sidecar_table.h5"'
            ' --sidecar unlabelled="$experiment_output/sidecar_unlabelled.h5"'
            f" --table-fits {_V5_DIR}/finetune.fits"
            ' --out "$experiment_output/validation.json"',
        ]
    ),
    files=[],
)
# S1 (docs/design-20261120-likelihood-inference.md): likelihood (ratio) heads on
# the frozen latents of v8 without the pool (all v5 table rows), common-scale
# labels from the checkpoint's survey maps, MLE / profile / Neyman inference
# and the fixed validation table against the posterior head.
STEPS["ms-s1-likelihood-head"] = dict(
    run_id="likelihood_head_s1_20261120",
    gpu=1,
    ram=128,
    cpu=8,
    command=(
        f"python -u scripts/s1_likelihood_head.py --config {_NP[1]}"
        f" --checkpoint {ROOT}{_NP[0]}/finetune_checkpoint_12.pth"
        f" --fits {ROOT}{MS_TABLE_V5}/finetune.fits"
        f" --records {ROOT}{MS_TABLE_V5}/records.npz"
        f" --core-split {ROOT}{RESCORE}/rows_split.npz"
        " --device cuda --threads 8 --chunk-size 262144"
        ' --output "$experiment_output"'
    ),
    files=[_NP[1]],
)
# v10: v8 without the pool + [Fe/H] core references on the identity map
# (scripts/make_v10_config.py). Predictions at epoch 12: test split (IRFM,
# extra labels), frozen core rows, consistency stars.
MS_V10 = (
    "multisurvey_finetune_20261121_v10",
    "configs/finetune-multisurvey-20261121-v10.yaml",
)
_V10_CK = f"{ROOT}{MS_V10[0]}/finetune_checkpoint_12.pth"
STEPS["ms-multi-v10"] = dict(
    run_id=MS_V10[0],
    gpu=1,
    ram=192,
    cpu=8,
    command=' && "${pixi_run[@]}" '.join(
        [
            f"python -u -m masked_stellar_autoencoder.training.finetune_msa --config {MS_V10[1]}",
            f"python -u scripts/predict_fits_rows.py --config {MS_V10[1]} --checkpoint {_V10_CK}"
            f" --fits {ROOT}{MS_TABLE}/finetune.fits"
            f" --test-split {MS_PAIR['ms-multi-v4'][1]} --irfm --condition-parallax 5"
            f" --extra-labels {ROOT}{REDDENING_V6}/v1_reddening.npz{_PRIOR}"
            ' --out "$experiment_output/multi_v10_test.npz"',
            f"python -u scripts/predict_fits_rows.py --config {MS_V10[1]} --checkpoint {_V10_CK}"
            f" --fits {ROOT}{MS_TABLE_V5}/finetune.fits"
            f" --rows {ROOT}{RESCORE}/rows.npy --condition-parallax 5{_PRIOR}"
            ' --out "$experiment_output/multi_v10_core.npz"',
            f"python -u scripts/predict_fits_rows.py --config {MS_V10[1]} --checkpoint {_V10_CK}"
            f" --fits {ROOT}{CONSIST}/stars.fits --condition-parallax 5{_PRIOR}"
            ' --out "$experiment_output/consistency_v10.npz"',
        ]
    ),
    files=[MS_V10[1], MS_PAIR["ms-multi-v4"][1]],
)
# S1 on v10 (identity [Fe/H] core): likelihood heads on the frozen v10
# latents, labels mapped by v10's maps, MLE/profiles restricted to the label
# support, Neyman sets, same validation table and decision rule as S1.
STEPS["ms-s1-v10"] = dict(
    run_id="likelihood_head_s1_v10_20261121",
    gpu=1,
    ram=128,
    cpu=8,
    command=(
        f"python -u scripts/s1_likelihood_head.py --config {MS_V10[1]}"
        f" --checkpoint {_V10_CK}"
        f" --fits {ROOT}{MS_TABLE_V5}/finetune.fits"
        f" --records {ROOT}{MS_TABLE_V5}/records.npz"
        f" --core-split {ROOT}{RESCORE}/rows_split.npz"
        " --device cuda --threads 8 --chunk-size 262144"
        ' --output "$experiment_output"'
    ),
    files=[MS_V10[1]],
)
# S1 diagnostics: separate the S1 failure causes on the frozen latents
# (latents vs label mapping vs dimension; MLE restricted to label support).
STEPS["ms-s1-diagnostics"] = dict(
    run_id="likelihood_head_s1_diagnostics_20261121",
    gpu=1,
    ram=64,
    cpu=8,
    command=(
        "python -u scripts/s1_diagnostics.py"
        f" --s1-dir {ROOT}likelihood_head_s1_20261120"
        f" --fits {ROOT}{MS_TABLE_V5}/finetune.fits --device cuda --seeds 3"
        ' --output "$experiment_output"'
    ),
    files=[],
)
# Move the MSA data from k-pop to /arc/projects/astroai/msa (copy + verify
# first; delete only after the verify report shows no mismatch).
for _mode in ("copy", "delete"):
    STEPS[f"arc-migrate-{_mode}"] = dict(
        run_id=f"arc_migration_{_mode}_20261120",
        gpu=0,
        cpu=4,
        ram=16,
        command=f"bash scripts/migrate_arc_to_astroai.sh {_mode}",
        files=[],
    )
# v7 ablations: 25 % training subsample, 6 epochs, one change each; four GPU
# sessions run their variants in sequence (a failure does not stop the rest).
from make_v7_configs import ABL_RUN, ABL_TRAIN_ROWS, GROUPS  # noqa: E402

# d03: the wide-binary reference pool is built (once) before training.
_POOL = (
    "python -u scripts/build_binary_pool.py"
    f" --stars {ROOT}{CONSIST}/stars.fits"
    f" --teacher {ROOT}consistency_pred_20261118_v7/consistency_v7.npz"
    f" --targets {ROOT}consistency_targets_20261118.npz --config {MS_V6[1]}"
    f" --output {ROOT}binary_pool_20261119/pool.npz || echo POOL_FAILED"
)
for _g, _names in GROUPS.items():
    _cfgs = [f"configs/ablation-20261118-v7/{n}.yaml" for n in _names]
    _pre = [_POOL] if any(n.startswith("d03") for n in _names) else []
    STEPS[f"ms-ablate-v7-{_g}"] = dict(
        run_id=f"{ABL_RUN}_g{_g}",
        gpu=1,
        ram=192,
        cpu=8,
        command=' && "${pixi_run[@]}" '.join(
            ["mkdir -p " + " ".join(f"{ROOT}{ABL_RUN}/{n}" for n in _names)]
            + _pre
            + [
                "python -u -m masked_stellar_autoencoder.training.finetune_msa"
                f" --config {c} --max-train-rows {ABL_TRAIN_ROWS}"
                f" > {ROOT}{ABL_RUN}/{n}/stdout.log 2>&1"
                f" || echo ABLATION_FAILED {n}"
                for c, n in zip(_cfgs, _names, strict=True)
            ]
        ),
        files=_cfgs,
    )
for _name, (_run, _config) in MS_PAIR.items():
    STEPS[_name] = dict(
        run_id=_run,
        gpu=1,
        ram=96,
        command=(
            "python -u -m masked_stellar_autoencoder.training.finetune_msa"
            f" --config {_config}"
        ),
        files=[_config],
    )
    STEPS[_name + "-eval"] = dict(
        run_id=_run.replace("finetune", "eval"),
        gpu=1,
        ram=96,
        command=(
            "python -u -m masked_stellar_autoencoder.training.eval_ensemble"
            f" --config {_config}"
            f" --checkpoints {ROOT}{_run}/finetune_checkpoint_{MS_EPOCH[_name]}.pth"
            ' --out "$experiment_output" --save-predictions'
        ),
        files=[_config],
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


# F2: data-driven XP forward model (docs/design-20261120-likelihood-inference.md, section 3):
# emulator of XP coefficients + absolute magnitudes from (Teff, log g, [Fe/H], E(B-V)),
# likelihood with the per-star Gaia XP covariances (sidecar) plus learned fractional
# scatter, parallax term with the Lindegren+2021 zero point; then MLE and profile
# intervals for held-out test rows and the frozen-core rows.
# Run 2 (20261122): E(B-V) anchored in training by the empirical reddening labels
# (training-only; they derive from the labels and BP-RP), per-star (E, distance)
# fitted by Gauss-Newton every batch, train-split core rows in training (held-out
# core rows stay out), v10 labels (identity [Fe/H] core), 2MASS dropped (mismatched in
# the tables), fixed per-star XP factor and float32 at inference. Run 1
# (xp_forward_model_20261120) used the S1 20261120 labels and excluded every core row.
XPFM_RUN = "xp_forward_model_20261122"
XPFM_SIDECAR = "covariance_sidecar_20261120/sidecar_table.h5"
XPFM_LABELS = f"{ROOT}likelihood_head_s1_v10_20261121/labels.npz"  # v10 labels
XPFM_REDDENING = f"{ROOT}{REDDENING_V6}/v5_reddening.npz"  # ebv_star, e_ebv_star
STEPS["ms-xp-forward-model"] = dict(
    run_id=XPFM_RUN,
    gpu=1,
    ram=128,
    cpu=8,
    command=(
        "python -u scripts/run_xp_forward_model.py"
        f" --table {ROOT}{MS_TABLE_V5}/finetune.fits --labels {XPFM_LABELS}"
        f" --reddening-labels {XPFM_REDDENING}"
        f" --covariance-sidecar {ROOT}{XPFM_SIDECAR}"
        f" --core-rows {ROOT}{RESCORE}/rows.npy"
        " --max-train-stars 1000000 --max-heldout-stars 50000 --epochs 20"
        " --nuisance-steps 3 --infer-dtype float32"
        ' --output "$experiment_output"'
    ),
    files=[],
)
# Diagnosis of run 2's inference (attempt 3, s81kdn7v): MLE chi^2/dof 2-5 against ~1
# at the labels, half the stars at the lower E bound, Teff uncorrelated with the
# labels; synthetic stars drawn from the same model are recovered. Exact NLL at the
# labels vs the MLE, start quality, oracle (label) starts and a wide search on 2000
# held-out stars.
STEPS["ms-xp-forward-diagnose"] = dict(
    run_id="xp_forward_diagnose_20261122",
    gpu=1,
    ram=64,
    cpu=8,
    command=(
        "python -u scripts/diagnose_xp_forward_inference.py"
        f" --table {ROOT}{MS_TABLE_V5}/finetune.fits --labels {XPFM_LABELS}"
        f" --checkpoint {ROOT}{XPFM_RUN}/model.pt"
        f" --covariance-sidecar {ROOT}{XPFM_SIDECAR}"
        f" --core-rows {ROOT}{RESCORE}/rows.npy --n 2000"
        ' --output "$experiment_output"'
    ),
    files=[],
)
# P1 (S1 precision): parallax as a parameter with its measurement likelihood
# (latents without the PARALLAX input; heads on theta' = (Teff, log g, [Fe/H],
# log10 parallax) trained on parallax/error >= 10; score + log N(parallax -
# L21 zero point | parallax, error^2), parallax profiled inside the 4-D label
# support) and a 5-head ensemble (mean log-ratio). Reuses the S1 v10 labels;
# the S1 v10 'all' arm joins every report table. Profile intervals on the
# core rows only (coverage uses the label statistics of every test row); 512
# MLE candidates (the 4-D support slice at a measured parallax is smaller than
# the 3-D box). Expected ~2.5 h (2-4 h) on a 12 GB GPU slice, peak ~7-9 GiB.
S1_V10 = "likelihood_head_s1_v10_20261121"
STEPS["ms-s1-v10-plx-ens"] = dict(
    run_id="likelihood_head_s1_v10_plx_ens_20261122",
    gpu=1,
    ram=128,
    cpu=8,
    command=(
        f"python -u scripts/s1_likelihood_head.py --config {MS_V10[1]}"
        f" --checkpoint {_V10_CK}"
        f" --fits {ROOT}{MS_TABLE_V5}/finetune.fits"
        f" --records {ROOT}{MS_TABLE_V5}/records.npz"
        f" --labels {ROOT}{S1_V10}/labels.npz --baseline {ROOT}{S1_V10}"
        f" --sidecar {ROOT}{XPFM_SIDECAR}"
        " --parallax-nuisance --ensemble 5 --profile-rows core --n-candidates 512"
        " --device cuda --threads 8 --chunk-size 32768"
        ' --output "$experiment_output"'
    ),
    files=[MS_V10[1]],
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
