#!/usr/bin/env bash
set -euo pipefail

stage="${1:?Usage: canfar_launch.sh <stage> (see batch_scripts/README.md)}"
shift
image="${CANFAR_IMAGE:-astroai/base:latest}"
name="${CANFAR_SESSION_NAME:-msa-${stage}}"
ref="${CANFAR_GIT_REF:-main}"
work_root="${CANFAR_WORK_ROOT:-/scratch/src}"

case "$stage" in
  install|schema|source-index|fetch-dustmaps|preprocess|combine|preflight|pretrain-pilot|pretrain-batch-pilot|pretrain|finetune-pilot|finetune) ;;
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

create_args=(
  --name "$name"
  --cpu "${CANFAR_CPU:-4}"
  --memory "${CANFAR_MEMORY:-16}"
  --env "WORK=$work_root"
  --env PYTHONNOUSERSITE=1
)
case "$stage" in
  preflight|pretrain-pilot|pretrain-batch-pilot|pretrain|finetune-pilot|finetune) create_args+=(--gpu 1) ;;
esac
for env_assignment in "$@"; do
  if [[ ! "$env_assignment" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]]; then
    echo "Expected KEY=VALUE stage override, got: $env_assignment" >&2
    exit 2
  fi
  create_args+=(--env "$env_assignment")
done
if [[ "${CANFAR_DRY_RUN:-0}" == 1 ]]; then create_args+=(--dry-run); fi

bootstrap="o=__import__('os');s=__import__('subprocess');w=o.environ['WORK'];r=w+'/sfabbro/masked-stellar-autoencoder';o.makedirs(w+'/sfabbro',exist_ok=True);s.run(['git','clone','--depth','50','--branch','$ref','https://github.com/sfabbro/masked-stellar-autoencoder.git',r],check=True);s.run(['bash',r+'/batch_scripts/canfar_entrypoint.sh','$stage'],check=True)"

# Skaha parses args as a command-line string; keep its Python -c payload one token.
if [[ "$bootstrap" =~ [[:space:]] || "$bootstrap" == *'$'* || "$bootstrap" == *'"'* ]]; then
  echo "CANFAR bootstrap must be a whitespace-free, dollar-free Python token." >&2
  exit 2
fi

canfar create headless "$image" "${create_args[@]}" -- python -u -c "$bootstrap"
