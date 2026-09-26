#!/usr/bin/env bash
set -euo pipefail

stage="${1:?Usage: canfar_launch.sh <stage> (see batch_scripts/README.md)}"
shift
image="${CANFAR_IMAGE:-astroai/base:latest}"
name="${CANFAR_SESSION_NAME:-msa-${stage}}"
ref="${CANFAR_GIT_REF:-main}"

case "$stage" in
  schema|source-index|fetch-dustmaps|preprocess|combine|preflight|pretrain-pilot|pretrain|finetune-pilot|finetune) ;;
  *)
    echo "Unknown CANFAR stage: $stage" >&2
    exit 2
    ;;
esac

job_args=(--repo sfabbro/masked-stellar-autoencoder --branch "$ref" --image "$image" --name "$name")
if [[ -n "${CANFAR_CPU:-}" ]]; then job_args+=(--cpu "$CANFAR_CPU"); fi
if [[ -n "${CANFAR_MEMORY:-}" ]]; then job_args+=(--memory "$CANFAR_MEMORY"); fi
case "$stage" in
  preflight|pretrain-pilot|pretrain|finetune-pilot|finetune) job_args+=(--gpu 1) ;;
esac

cmd=(bash batch_scripts/canfar_entrypoint.sh "$stage")
if (($#)); then cmd=(env "$@" "${cmd[@]}"); fi

canfar-job run "${job_args[@]}" -- "${cmd[@]}"
