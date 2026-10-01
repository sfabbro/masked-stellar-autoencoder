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

pixi_platform=linux-64-cuda
case "$stage" in
  schema|source-index|fetch-dustmaps|preprocess|combine) pixi_platform=linux-64-cpu ;;
esac

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

pixi lock --check
pixi install --frozen --environment gpu --platform "$pixi_platform"
pixi_run=(pixi run --environment gpu --platform "$pixi_platform")

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
    "${pixi_run[@]}" check-canfar-gpu
    experiment_output="${MSA_EXPERIMENT_OUTPUT:-$MSA_OUTPUT_ROOT/experiments-$(date -u +%Y%m%dT%H%M%SZ)}"
    experiment_args=(
      --checkpoint "$MSA_PRETRAIN_RESUME"
      --output "$experiment_output"
      --cache-dir "${MSA_EXPERIMENT_CACHE:-$work_root/msa-experiment-cache}"
      --train-rows "${MSA_EXPERIMENT_TRAIN_ROWS:-3000000}"
      --validation-rows "${MSA_EXPERIMENT_VALIDATION_ROWS:-10000}"
      --presentations "${MSA_EXPERIMENT_PRESENTATIONS:-5000000}"
      --log-rows "${MSA_EXPERIMENT_LOG_ROWS:-1000000}"
    )
    if [[ -n "${MSA_EXPERIMENT_ARMS:-}" ]]; then
      read -r -a experiment_arms <<< "$MSA_EXPERIMENT_ARMS"
      experiment_args+=(--arms "${experiment_arms[@]}")
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
