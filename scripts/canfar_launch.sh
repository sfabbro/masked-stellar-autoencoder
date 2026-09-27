#!/usr/bin/env bash
set -euo pipefail

stage="${1:?Usage: canfar_launch.sh <stage> (see batch_scripts/README.md)}"
shift
image="${CANFAR_IMAGE:-astroai/base:latest}"
name="${CANFAR_SESSION_NAME:-msa-${stage}}"
ref="${CANFAR_GIT_REF:-main}"
work_root="${CANFAR_WORK_ROOT:-/scratch/src}"

case "$stage" in
  install|schema|source-index|fetch-dustmaps|preprocess|combine|preflight|pretrain-pilot|pretrain|finetune-pilot|finetune) ;;
  *)
    echo "Unknown CANFAR stage: $stage" >&2
    exit 2
    ;;
esac

if [[ ! "$ref" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ ]] || ! git check-ref-format --branch "$ref" >/dev/null; then
  echo "Invalid CANFAR_GIT_REF: $ref" >&2
  exit 2
fi
if [[ ! "$work_root" =~ ^/[A-Za-z0-9._/-]+$ ]]; then
  echo "CANFAR_WORK_ROOT must be an absolute path without shell metacharacters: $work_root" >&2
  exit 2
fi

repo_dir="$work_root/sfabbro/masked-stellar-autoencoder"
printf -v quoted_work_root '%q' "$work_root"
printf -v quoted_repo_dir '%q' "$repo_dir"
printf -v quoted_ref '%q' "$ref"

create_args=(
  --name "$name"
  --cpu "${CANFAR_CPU:-4}"
  --memory "${CANFAR_MEMORY:-16}"
  --env "WORK=$work_root"
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

remote_command="set -euo pipefail
export PYTHONNOUSERSITE=1
unset PYTHONPATH
mkdir -p $quoted_work_root/sfabbro
git clone --depth 50 --branch $quoted_ref https://github.com/sfabbro/masked-stellar-autoencoder.git $quoted_repo_dir
cd $quoted_repo_dir
printf 'MSA commit: '
git rev-parse --short HEAD
bash batch_scripts/canfar_entrypoint.sh $stage"

# Skaha treats dollar references in the command as regex replacement groups.
if [[ "$remote_command" == *'$'* ]]; then
  echo "CANFAR launch command contains a dollar reference unsupported by Skaha." >&2
  exit 2
fi

canfar create headless "$image" "${create_args[@]}" -- bash -lc "$remote_command"
