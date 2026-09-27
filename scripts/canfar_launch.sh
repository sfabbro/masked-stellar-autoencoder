#!/usr/bin/env bash
set -euo pipefail

stage="${1:?Usage: canfar_launch.sh <stage> (see batch_scripts/README.md)}"
shift
image="${CANFAR_IMAGE:-astroai/base:latest}"
name="${CANFAR_SESSION_NAME:-msa-${stage}}"
ref="${CANFAR_GIT_REF:-main}"

case "$stage" in
  install|schema|source-index|fetch-dustmaps|preprocess|combine|preflight|pretrain-pilot|pretrain|finetune-pilot|finetune) ;;
  *)
    echo "Unknown CANFAR stage: $stage" >&2
    exit 2
    ;;
esac

create_args=(
  --name "$name"
  --cpu "${CANFAR_CPU:-4}"
  --memory "${CANFAR_MEMORY:-16}"
  --env "CANFAR_GIT_REF=$ref"
  --env "CANFAR_STAGE=$stage"
  --env PYTHONNOUSERSITE=1
)
case "$stage" in
  preflight|pretrain-pilot|pretrain|finetune-pilot|finetune) create_args+=(--gpu 1) ;;
esac
for env_assignment in "$@"; do
  if [[ ! "$env_assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    echo "Expected KEY=VALUE stage override, got: $env_assignment" >&2
    exit 2
  fi
  create_args+=(--env "$env_assignment")
done
if [[ "${CANFAR_DRY_RUN:-0}" == 1 ]]; then create_args+=(--dry-run); fi

remote_command='set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export WORK="${WORK:-/scratch/src}"
repo_dir="$WORK/sfabbro/masked-stellar-autoencoder"
mkdir -p "$WORK/sfabbro"
git clone --depth 50 --branch "$CANFAR_GIT_REF" https://github.com/sfabbro/masked-stellar-autoencoder.git "$repo_dir"
cd "$repo_dir"
printf "MSA commit: "
git rev-parse --short HEAD
bash batch_scripts/canfar_entrypoint.sh "$CANFAR_STAGE"'

canfar create headless "$image" "${create_args[@]}" -- bash -lc "$remote_command"
