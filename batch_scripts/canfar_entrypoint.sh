#!/usr/bin/env bash
set -euo pipefail

stage="${1:?Usage: canfar_entrypoint.sh schema|source-index|fetch-dustmaps|preprocess|combine|pretrain-pilot|pretrain|finetune-pilot|finetune|preflight}"
project_root="${CANFAR_PROJECT_ROOT:-/arc/projects/k-pop}"
work_root="${WORK:-/scratch/${USER:?USER is unset}}"
msa_source="${MSA_SOURCE:-$project_root/software/masked-stellar-autoencoder}"
repo_dir="$work_root/src/masked-stellar-autoencoder"

export PYTHONNOUSERSITE=1
unset PYTHONPATH
# The AstroAI image keeps Pixi's global home under /usr/local/share, which is
# read-only in sessions. Keep Pixi state and both package caches in $WORK.
export PIXI_HOME="$work_root/.pixi"
export PIXI_CACHE_DIR="$work_root/.cache/pixi"
export PIXI_CACHE_PYPI_WHEELS_DIR="$work_root/.cache/uv"
export UV_CACHE_DIR="$work_root/.cache/uv"
export DUSTMAPS_DATA_DIR="${DUSTMAPS_DATA_DIR:-$project_root/catalogues/dustmaps}"
export MSA_PREPROCESS_DIR="${MSA_PREPROCESS_DIR:-$project_root/catalogues/andrae2023/preprocess}"
export MSA_PREPROCESS_CACHE_DIR="${MSA_PREPROCESS_CACHE_DIR:-$work_root/msa-crossmatch-cache-${HOSTNAME:-session}}"
export MSA_SOURCE_IDS_FILE="${MSA_SOURCE_IDS_FILE:-$project_root/catalogues/andrae2023/source_ids_x_file_names.h5}"
export MSA_CATWISE_FILE="${MSA_CATWISE_FILE:-$project_root/catalogues/andrae2023/table_1_catwise.fits.gz}"
export MSA_GAIA_XP_DIR="${MSA_GAIA_XP_DIR:-/gaia/dr3/xp_continuous_mean_spectrum}"
export MSA_GAIA_SOURCE_DIR="${MSA_GAIA_SOURCE_DIR:-$project_root/gaia/GaiaSource}"
export MSA_ADQL_MATCH_DIR="${MSA_ADQL_MATCH_DIR:-$project_root/catalogues/adql_matches}"
export MSA_PRETRAIN_DATA="${MSA_PRETRAIN_DATA:-$project_root/catalogues/andrae2023/sslset-realmags-full-052725.h5}"
export MSA_PRETRAIN_RESUME="${MSA_PRETRAIN_RESUME:-}"
export MSA_FINETUNE_DATA="${MSA_FINETUNE_DATA:-$project_root/catalogues/andrae2023/ftset_spec_ga_0602_realmags.fits}"
export MSA_FINETUNE_RESUME="${MSA_FINETUNE_RESUME:-}"
export MSA_OUTPUT_ROOT="${MSA_OUTPUT_ROOT:-$project_root/msa_runs}"
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

if [[ ! -d "$msa_source" ]]; then
  echo "MSA source snapshot is missing: $msa_source" >&2
  echo "Sync the repository there or set MSA_SOURCE to its persistent location." >&2
  exit 2
fi
if ! command -v pixi >/dev/null 2>&1; then
  echo "Pixi is required in the CANFAR image but was not found on PATH." >&2
  exit 2
fi

mkdir -p "$repo_dir" "$work_root/.cache/pixi" "$work_root/.cache/uv"
tar -C "$msa_source" \
  --exclude=.git --exclude=.pixi --exclude=.venv --exclude=__pycache__ \
  --exclude=.pytest_cache --exclude=.ruff_cache --exclude=.ipynb_checkpoints \
  -cf - . | tar -C "$repo_dir" -xf -
cd "$repo_dir"

pixi lock --check
pixi install --frozen --environment gpu --platform "$pixi_platform"
pixi_run=(pixi run --environment gpu --platform "$pixi_platform")

case "$stage" in
  schema)
    "${pixi_run[@]}" python -m masked_stellar_autoencoder.training.canfar_preflight \
      --schema-only --output "$MSA_PRETRAIN_OUTPUT_DIR/schema-preflight.json"
    ;;
  source-index)
    "${pixi_run[@]}" python -u data/source_ids_x_file_names.py
    ;;
  fetch-dustmaps)
    mkdir -p "$DUSTMAPS_DATA_DIR"
    "${pixi_run[@]}" python -c 'from dustmaps.sfd import fetch; fetch()'
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
  pretrain)
    "${pixi_run[@]}" canfar-preflight-pretrain
    "${pixi_run[@]}" pretrain-canfar
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
