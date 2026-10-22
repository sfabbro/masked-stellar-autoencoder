#!/usr/bin/env bash
set -euo pipefail

stage="${1:?Usage: canfar_entrypoint.sh install|schema|source-index|fetch-dustmaps|preprocess|combine|pretrain-pilot|pretrain-batch-pilot|pretrain-experiments|pretrain|finetune-pilot|finetune|preflight}"
project_root="${CANFAR_PROJECT_ROOT:-/arc/projects/k-pop}"
work_root="${WORK:-/scratch/${USER:?USER is unset}}"
script_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
msa_source="${MSA_SOURCE:-$script_root}"
if [[ ! -d "$msa_source" ]]; then
  echo "MSA source snapshot is missing: $msa_source" >&2
  exit 2
fi
msa_source="$(cd -- "$msa_source" && pwd -P)"
repo_dir="$script_root"
if [[ "$msa_source" != "$script_root" ]]; then
  repo_dir="$work_root/msa/masked-stellar-autoencoder"
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH
# The AstroAI image keeps Pixi's global home under /usr/local/share, which is
# read-only in sessions. Keep Pixi state and both package caches in $WORK.
export PIXI_HOME="$work_root/.pixi"
export PIXI_CACHE_DIR="$work_root/.cache/pixi"
export PIXI_CACHE_PYPI_WHEELS_DIR="$work_root/.cache/uv"
export UV_CACHE_DIR="$work_root/.cache/uv"
export DUSTMAPS_CONFIG_FNAME="${DUSTMAPS_CONFIG_FNAME:-$work_root/.dustmapsrc}"
export MSA_OUTPUT_ROOT="${MSA_OUTPUT_ROOT:-$project_root/msa_runs}"
export DUSTMAPS_DATA_DIR="${DUSTMAPS_DATA_DIR:-$MSA_OUTPUT_ROOT/dustmaps}"
export MSA_PREPROCESS_DIR="${MSA_PREPROCESS_DIR:-$MSA_OUTPUT_ROOT/preprocess}"
export MSA_PREPROCESS_CACHE_DIR="${MSA_PREPROCESS_CACHE_DIR:-$work_root/msa-crossmatch-cache-${HOSTNAME:-session}}"
export MSA_SOURCE_IDS_FILE="${MSA_SOURCE_IDS_FILE:-$MSA_PREPROCESS_DIR/source_ids_x_file_names.h5}"
export MSA_CATWISE_FILE="${MSA_CATWISE_FILE:-$project_root/catalogues/andrae2023/table_1_catwise.fits.gz}"
export MSA_GAIA_XP_DIR="${MSA_GAIA_XP_DIR:-/arc/projects/k-pop/spectra/gaia/dr3/xp_continuous_mean_spectrum}"
export MSA_GAIA_SOURCE_DIR="${MSA_GAIA_SOURCE_DIR:-$project_root/gaia/GaiaSource}"
export MSA_ADQL_MATCH_DIR="${MSA_ADQL_MATCH_DIR:-$project_root/catalogues/adql_matches}"
export MSA_PRETRAIN_DATA="${MSA_PRETRAIN_DATA:-$project_root/catalogues/andrae2023/sslset-realmags-full-052725.h5}"
export MSA_PRETRAIN_RESUME="${MSA_PRETRAIN_RESUME:-}"
export MSA_FINETUNE_DATA="${MSA_FINETUNE_DATA:-$project_root/catalogues/andrae2023/ftset_spec_ga_0602_realmags.fits}"
export MSA_FINETUNE_RESUME="${MSA_FINETUNE_RESUME:-}"
export MSA_PRETRAIN_OUTPUT_DIR="${MSA_PRETRAIN_OUTPUT_DIR:-$MSA_OUTPUT_ROOT/pretrain}"
export MSA_PRETRAIN_CHECKPOINT_DIR="${MSA_PRETRAIN_CHECKPOINT_DIR:-$MSA_PRETRAIN_OUTPUT_DIR/checkpoints}"
export MSA_PRETRAIN_CHECKPOINT="${MSA_PRETRAIN_CHECKPOINT:-$MSA_PRETRAIN_CHECKPOINT_DIR/msa_pretrain.pth}"
export MSA_FINETUNE_OUTPUT_DIR="${MSA_FINETUNE_OUTPUT_DIR:-$MSA_OUTPUT_ROOT/finetune}"
export MSA_FINETUNE_CHECKPOINT="${MSA_FINETUNE_CHECKPOINT:-$MSA_FINETUNE_OUTPUT_DIR/checkpoints/msa_finetune.pth}"
export MSA_PRETRAIN_HDF5_OUT="${MSA_PRETRAIN_HDF5_OUT:-$MSA_PREPROCESS_DIR/pretrain_dataset_incomplete.h5}"
export MSA_PREPROCESS_WORKERS="${MSA_PREPROCESS_WORKERS:-4}"
if [[ "$stage" == pretrain-experiments ]]; then
  experiment_output="${MSA_EXPERIMENT_OUTPUT:-$MSA_OUTPUT_ROOT/experiments-$(date -u +%Y%m%dT%H%M%SZ)}"
  mkdir -p "$(dirname -- "$experiment_output")"
  exec > >(tee "$experiment_output.session.log") 2>&1
fi

pixi_platform=linux-64-cuda
case "$stage" in
  schema|source-index|fetch-dustmaps|preprocess|combine) pixi_platform=linux-64-cpu ;;
esac
pixi_manifest="$repo_dir/pyproject.toml"
pixi_environment=gpu
if [[ "$stage" == pretrain-experiments && "${MSA_XP_EXPERIMENTS:-0}" == 1 ]]; then
  pixi_manifest="$repo_dir/experiments/xp/pixi.toml"
  pixi_environment=default
fi

if ! command -v pixi >/dev/null 2>&1; then
  echo "Pixi is required in the CANFAR image but was not found on PATH." >&2
  exit 2
fi

mkdir -p "$work_root/.cache/pixi" "$work_root/.cache/uv"
if [[ "$msa_source" != "$script_root" ]]; then
  mkdir -p "$repo_dir"
  tar -C "$msa_source" \
    --exclude=.git --exclude=.pixi --exclude=.venv --exclude=__pycache__ \
    --exclude=.pytest_cache --exclude=.ruff_cache --exclude=.ipynb_checkpoints \
    -cf - . | tar -C "$repo_dir" -xf -
fi
cd "$repo_dir"

pixi lock --manifest-path "$pixi_manifest" --check
pixi install --manifest-path "$pixi_manifest" --frozen --environment "$pixi_environment" --platform "$pixi_platform"
pixi_run=(pixi run --manifest-path "$pixi_manifest" --environment "$pixi_environment" --platform "$pixi_platform")

case "$stage" in
  install)
    "${pixi_run[@]}" python -c 'import masked_stellar_autoencoder, torch; assert torch.version.cuda == "13.0", torch.version.cuda; print(f"MSA import OK; torch={torch.__version__}; CUDA={torch.version.cuda}")'
    ;;
  schema)
    "${pixi_run[@]}" python -m masked_stellar_autoencoder.training.canfar_preflight \
      --schema-only --output "$MSA_PRETRAIN_OUTPUT_DIR/schema-preflight.json"
    ;;
  source-index)
    "${pixi_run[@]}" python -u data/source_ids_x_file_names.py
    ;;
  fetch-dustmaps)
    mkdir -p "$DUSTMAPS_DATA_DIR"
    "${pixi_run[@]}" python -c 'import os; from dustmaps.config import config; config["data_dir"] = os.environ["DUSTMAPS_DATA_DIR"]; from dustmaps.sfd import fetch; fetch()'
    ;;
  preprocess)
    "${pixi_run[@]}" python -u data/pretraining-partial-table-maker.py
    ;;
  combine)
    "${pixi_run[@]}" python -u data/combine-partial-tables.py
    ;;
  preflight)
    "${pixi_run[@]}" canfar-preflight
    ;;
  pretrain-pilot)
    "${pixi_run[@]}" canfar-preflight-pretrain
    "${pixi_run[@]}" pretrain-canfar-pilot
    ;;
  pretrain-batch-pilot)
    "${pixi_run[@]}" canfar-preflight-pretrain
    "${pixi_run[@]}" pretrain-canfar-batch-pilot
    ;;
  pretrain)
    "${pixi_run[@]}" canfar-preflight-pretrain
    "${pixi_run[@]}" pretrain-canfar
    ;;
  pretrain-experiments)
    : "${MSA_PRETRAIN_RESUME:?Set MSA_PRETRAIN_RESUME to the immutable full checkpoint}"
    if [[ "${MSA_EXPERIMENT_PHOTOMETRY_PREFLIGHT:-0}" != 1 && "${MSA_EXPERIMENT_CPU_ONLY:-0}" != 1 ]]; then
      "${pixi_run[@]}" python -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == age-data-audit ]]; then
      : "${MSA_AGE_APOKASC3:?Set MSA_AGE_APOKASC3}" "${MSA_AGE_COHORT:?}" "${MSA_AGE_DATASET_VIEW:?}" "${MSA_AGE_POSTERIOR:?}"
      "${pixi_run[@]}" python -u scripts/age_data_audit.py \
        --apokasc3 "$MSA_AGE_APOKASC3" \
        --willett-dir "/scratch/willett-age-${MSA_EXPERIMENT_RUNS}" \
        --age-cohort "$MSA_AGE_COHORT" \
        --dataset-view "$MSA_AGE_DATASET_VIEW" \
        --posterior-labels "$MSA_AGE_POSTERIOR" \
        --output "$experiment_output" --download
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == density-precision-certificate ]]; then
      "${pixi_run[@]}" python -u scripts/run_density_precision_certificate.py \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multilabel-screen ]]; then
      "${pixi_run[@]}" python -u scripts/run_multilabel_screen.py \
        --config configs/multilabel-screen-20261020.yaml \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" \
        --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multilabel-w2 ]]; then
      "${pixi_run[@]}" python -u scripts/run_multilabel_screen.py \
        --config configs/multilabel-w2-20261022.yaml \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" \
        --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multilabel-w1 ]]; then
      "${pixi_run[@]}" python -u scripts/run_multilabel_screen.py \
        --config configs/multilabel-w1-20261022.yaml \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" \
        --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multilabel-density-screen ]]; then
      "${pixi_run[@]}" python -u scripts/run_multilabel_screen.py \
        --config configs/multilabel-density-20261021.yaml \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" \
        --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multilabel-long ]]; then
      "${pixi_run[@]}" python -u scripts/run_multilabel_screen.py \
        --config configs/multilabel-long-20261020.yaml \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" \
        --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == catalog && "${MSA_CATALOG_MODE:-}" == joint-native-quality-verification ]]; then
      "${pixi_run[@]}" python -u scripts/run_joint_native_quality.py \
        --config configs/joint-native-preparation.yaml \
        --sidecar-directory "${MSA_NATIVE_SIDECAR_DIRECTORY:?Set existing audited native sidecar directory}" \
        --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == catalog && "${MSA_CATALOG_MODE:-}" == joint-native-preparation ]]; then
      "${pixi_run[@]}" python -u scripts/prepare_joint_native_labels.py \
        --config configs/joint-native-preparation.yaml --record-source-pins --output-directory "$experiment_output"
      "${pixi_run[@]}" python -u scripts/prepare_joint_native_labels.py \
        --config configs/joint-native-preparation.yaml --execute --output-directory "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == catalog ]]; then
      "${pixi_run[@]}" python -u scripts/stellar_catalog_jobs.py \
        --mode "$MSA_CATALOG_MODE" --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == test-time-screen ]]; then
      "${pixi_run[@]}" python -u scripts/run_test_time_screen.py \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" \
        --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == forward-eiv-screen ]]; then
      "${pixi_run[@]}" python -u scripts/run_forward_eiv_screen.py \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" \
        --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == frozen-head-screen ]]; then
      "${pixi_run[@]}" python -u scripts/run_frozen_head_screen.py \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" \
        --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == large-huber ]]; then
      "${pixi_run[@]}" python -u scripts/run_prediction_comparison.py \
        --config configs/residual-huber-150m.yaml --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" \
        --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == prediction-continuation ]]; then
      "${pixi_run[@]}" python -u scripts/run_prediction_comparison.py \
        --config configs/residual-huber-long.yaml --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" \
        --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == targeted-prediction ]]; then
      "${pixi_run[@]}" python -u scripts/run_prediction_comparison.py \
        --config configs/targeted-10m-20261019.yaml \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" \
        --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == prediction-comparison ]]; then
      "${pixi_run[@]}" python -u scripts/run_prediction_comparison.py \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" \
        --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == enriched-value ]]; then
      "${pixi_run[@]}" python -u scripts/run_enriched_value.py \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == distance-value ]]; then
      "${pixi_run[@]}" python -u scripts/run_distance_value.py \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == native-age ]]; then
      "${pixi_run[@]}" python -u scripts/run_native_age.py \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == pristine-value ]]; then
      "${pixi_run[@]}" python -u scripts/run_pristine_value.py \
        --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == stellar-labels ]]; then
      "${pixi_run[@]}" python -u scripts/run_stellar_label_experiments.py \
        --config "$MSA_LABEL_CONFIG" --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == analysis ]]; then
      : "${MSA_ANALYSIS_SPEC:?Set the pinned analysis specification path}"
      "${pixi_run[@]}" python -u scripts/run_xp_analysis.py \
        --spec "$MSA_ANALYSIS_SPEC" --run-id "$MSA_EXPERIMENT_RUNS" \
        --output "$experiment_output"
      exit 0
    fi
    experiment_args=(
      --experiments "${MSA_EXPERIMENTS_CONFIG:-configs/pretrain.experiments.yaml}"
      --checkpoint "$MSA_PRETRAIN_RESUME"
      --output "$experiment_output"
      --cache-dir "${MSA_EXPERIMENT_CACHE:-$work_root/msa-experiment-cache}"
      --train-rows "${MSA_EXPERIMENT_TRAIN_ROWS:-3000000}"
      --validation-rows "${MSA_EXPERIMENT_VALIDATION_ROWS:-10000}"
      --presentations "${MSA_EXPERIMENT_PRESENTATIONS:-5000000}"
      --log-rows "${MSA_EXPERIMENT_LOG_ROWS:-1000000}"
      --diagnostic-rows "${MSA_EXPERIMENT_DIAGNOSTIC_ROWS:-100000}"
      --seed "${MSA_EXPERIMENT_SEED:-42}"
      --validation-seed "${MSA_EXPERIMENT_VALIDATION_SEED:-42}"
    )
    if [[ -n "${MSA_EXPERIMENT_ARMS:-}" ]]; then
      read -r -a experiment_arms <<< "$MSA_EXPERIMENT_ARMS"
      experiment_args+=(--arms "${experiment_arms[@]}")
    fi
    if [[ -n "${MSA_EXPERIMENT_RUNS:-}" ]]; then
      read -r -a experiment_runs <<< "$MSA_EXPERIMENT_RUNS"
      experiment_args+=(--runs "${experiment_runs[@]}")
    fi
    if [[ "${MSA_XP_EXPERIMENTS:-0}" == 1 ]]; then
      experiment_args+=(--xp-diagnostics)
    fi
    if [[ "${MSA_EXPERIMENT_INDEPENDENT_RNG:-0}" == 1 ]]; then
      experiment_args+=(--independent-rng)
    fi
    if [[ "${MSA_EXPERIMENT_PHOTOMETRY_PREFLIGHT:-0}" == 1 ]]; then
      experiment_args+=(--photometry-preflight)
    fi
    if [[ "${MSA_EXPERIMENT_EVALUATION_ONLY:-0}" == 1 ]]; then
      experiment_args+=(--evaluation-only)
    fi
    if [[ -n "${MSA_EXPERIMENT_BENCHMARK:-}" ]]; then
      experiment_args+=(--benchmark "$MSA_EXPERIMENT_BENCHMARK")
    fi
    if [[ -n "${MSA_EXPERIMENT_MICRO_BATCH:-}" ]]; then
      experiment_args+=(--micro-batch-size "$MSA_EXPERIMENT_MICRO_BATCH")
    fi
    "${pixi_run[@]}" python -u scripts/run_pretrain_experiments.py "${experiment_args[@]}"
    ;;
  finetune-pilot)
    "${pixi_run[@]}" canfar-preflight
    "${pixi_run[@]}" finetune-canfar-pilot
    ;;
  finetune)
    "${pixi_run[@]}" canfar-preflight
    "${pixi_run[@]}" finetune-canfar
    ;;
  *)
    echo "Unknown CANFAR stage: $stage" >&2
    exit 2
    ;;
esac
