#!/usr/bin/env bash
set -euo pipefail

stage="${1:?Usage: canfar_launch.sh <stage> (see batch_scripts/README.md)}"
source_root="${MSA_CANFAR_REPO_PATH:-/arc/projects/k-pop/software/masked-stellar-autoencoder}"
image="${CANFAR_IMAGE:-images.canfar.net/astroai/base:latest}"
name="${CANFAR_SESSION_NAME:-msa-${stage}}"

case "$stage" in
  schema|source-index|fetch-dustmaps|preprocess|combine|preflight|pretrain-pilot|pretrain|finetune-pilot|finetune) ;;
  *)
    echo "Unknown CANFAR stage: $stage" >&2
    exit 2
    ;;
esac

create_args=(--name "$name")
case "$stage" in
  preflight|pretrain-pilot|pretrain|finetune-pilot|finetune) create_args+=(--gpu 1) ;;
esac

canfar create "${create_args[@]}" headless "$image" -- \
  bash "$source_root/batch_scripts/canfar_entrypoint.sh" "$stage"
