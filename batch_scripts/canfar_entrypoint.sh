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
    if command -v nvidia-smi >/dev/null 2>&1; then
      echo "=== GPU allocation (node/profile):"
      nvidia-smi -L || true
      nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
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
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multilabel-sls ]]; then
      "${pixi_run[@]}" python -u scripts/run_multilabel_sls.py --config configs/multilabel-sls-20261023.yaml --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multilabel-noise-contrastive ]]; then
      "${pixi_run[@]}" python -u scripts/run_multilabel_screen.py --config configs/multilabel-noise-contrastive-20261023.yaml --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == floweb-quadrature-recovery ]]; then
      recovery_args=(--reuse-prior "$MSA_FLOWEB_PRIOR" --expected-prior-sha "$MSA_FLOWEB_PRIOR_SHA" --adaptive-prior --prior-node-cap "$MSA_FLOWEB_PRIOR_NODE_CAP")
      if [[ -n "${MSA_FLOWEB_CONVERGENCE_REPORT:-}" ]]; then
        recovery_args+=(--resume-prior-convergence "$MSA_FLOWEB_CONVERGENCE_REPORT" --expected-convergence-sha "$MSA_FLOWEB_CONVERGENCE_SHA")
      fi
      "${pixi_run[@]}" python -u scripts/run_forward_optimizer_wave.py --original "$MSA_FORWARD_ORIGINAL" --output "$experiment_output" --mode "$MSA_FORWARD_MODE" --seed "$MSA_FORWARD_SEED" --test-count "$MSA_FORWARD_TEST_COUNT" --cal-count "$MSA_FORWARD_CAL_COUNT" "${recovery_args[@]}"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == forward-floweb ]]; then
      "${pixi_run[@]}" python -u scripts/run_forward_optimizer_wave.py --original "$MSA_FORWARD_ORIGINAL" --output "$experiment_output" --mode "$MSA_FORWARD_MODE" --seed "$MSA_FORWARD_SEED" --test-count "$MSA_FORWARD_TEST_COUNT" --cal-count "$MSA_FORWARD_CAL_COUNT"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == expanded-source-contract-audit ]]; then
      "${pixi_run[@]}" python -u scripts/audit_expanded_source_contract.py --config configs/full-label-source-preparation-20261024.yaml --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == full-label-evaluation-preparation ]]; then
      "${pixi_run[@]}" python -u scripts/prepare_full_label_evaluation.py --config "${MSA_EXPANDED_SOURCE_POLICY:-configs/full-label-source-preparation-20261024.yaml}" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == remediation-input-audit ]]; then
      "${pixi_run[@]}" python -u scripts/audit_remediation_inputs.py --config "$MSA_REMEDIATION_POLICY" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == remediation-source ]]; then
      "${pixi_run[@]}" python -u scripts/prepare_remediation_source.py --config "$MSA_REMEDIATION_POLICY" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multisurvey-gaia-gspspec-cache-census ]]; then
      "${pixi_run[@]}" python -u scripts/run_multisurvey_gaia_cache_census.py --config "$MSA_MULTISURVEY_POLICY" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multisurvey-gaia-gspspec-overlay ]]; then
      "${pixi_run[@]}" python -u scripts/run_multisurvey_gaia_overlay.py --config "$MSA_MULTISURVEY_POLICY" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multisurvey-fit-collation-census ]]; then
      "${pixi_run[@]}" python -u scripts/run_multisurvey_fit_collation_census.py --config "$MSA_MULTISURVEY_POLICY" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multisurvey-consensus-preflight ]]; then
      "${pixi_run[@]}" python -u scripts/run_multisurvey_consensus_preflight.py --config "$MSA_MULTISURVEY_POLICY" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multisurvey-desi-overlay ]]; then
      "${pixi_run[@]}" python -u scripts/run_multisurvey_desi_overlay.py --config "$MSA_MULTISURVEY_POLICY" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multisurvey-desi-census ]]; then
      "${pixi_run[@]}" python -u scripts/run_multisurvey_desi_census.py --config "$MSA_MULTISURVEY_POLICY" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multisurvey-desi-source ]]; then
      "${pixi_run[@]}" python -u scripts/run_multisurvey_desi_source.py --config "$MSA_MULTISURVEY_POLICY" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == remediation-joint-preflight ]]; then
      "${pixi_run[@]}" python -u scripts/run_remediation_joint_preflight.py --config "$MSA_REMEDIATION_POLICY" --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == remediation-pilot ]]; then
      "${pixi_run[@]}" python -u scripts/run_remediation_pilot.py --config "$MSA_REMEDIATION_POLICY" --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == expanded-report-recovery ]]; then
      "${pixi_run[@]}" python -u scripts/recover_expanded_evaluation_report.py --config "$MSA_EXPANDED_REPORT_POLICY" --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == expanded-checkpoint-evaluation ]]; then
      "${pixi_run[@]}" python -u scripts/run_expanded_checkpoint_evaluation.py --config "${MSA_EXPANDED_EVALUATION_POLICY:-configs/expanded-checkpoint-evaluation-20261024.yaml}" --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == five-job-source-preparation ]]; then
      "${pixi_run[@]}" python -u scripts/prepare_five_job_source.py --config configs/five-job-source-20261024.yaml --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == five-wave-training ]]; then
      "${pixi_run[@]}" python -u scripts/run_five_wave_training.py --config configs/five-wave-training-20261024.yaml --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit "$?"
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == final-six-normalization-export ]]; then
      "${pixi_run[@]}" python -u scripts/export_final_six_normalization.py --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == final-six-distribution-metrics ]]; then
      "${pixi_run[@]}" python -u scripts/run_final_six_distribution_metrics.py --config "$MSA_DISTRIBUTION_METRICS_CONFIG" --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == metalpoor-corner-sampling ]]; then
      "${pixi_run[@]}" python -u scripts/sample_metalpoor_corners.py --config "$MSA_CORNER_SAMPLING_CONFIG" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == final-six-independent ]]; then
      "${pixi_run[@]}" python -u experiments/xp/monitor/verify_final_six_heads.py --run-id "$MSA_FINAL_SIX_PRODUCER_RUN" --expected-source-sha "$MSA_FINAL_SIX_PRODUCER_SHA" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == final-six-frozen ]]; then
      "${pixi_run[@]}" python -u scripts/run_final_six_frozen.py --config configs/final-six-frozen-20261024.yaml --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == final-six-ancestry ]]; then
      "${pixi_run[@]}" python -u scripts/audit_final_six_ancestry.py --config configs/final-six-ancestry-20261024.yaml --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == metalpoor-corner-selection ]]; then
      "${pixi_run[@]}" python -u scripts/select_metalpoor_corners.py --config configs/metalpoor-corner-selection-20261024.yaml --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == final-six-recipes ]]; then
      "${pixi_run[@]}" python -u scripts/run_final_six_recipes.py --config configs/final-six-recipes-20261024.yaml --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == final-six-smoke ]]; then
      "${pixi_run[@]}" python -u scripts/run_final_six_smoke.py --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == six-target-preparation ]]; then
      "${pixi_run[@]}" python -u scripts/prepare_six_target_wave.py --config "$MSA_SIX_SOURCE_CONFIG" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == guarded-ttt ]]; then
      "${pixi_run[@]}" python -u scripts/run_guarded_ttt_screen.py --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == alpha-head-verification ]]; then
      "${pixi_run[@]}" python -u experiments/xp/monitor/verify_alpha_frozen_heads.py --config configs/alpha-independent-heads-20261023.yaml --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == alpha-feature-preparation ]]; then
      "${pixi_run[@]}" python -u scripts/prepare_alpha_target_wave.py --checkpoint "$MSA_PRETRAIN_RESUME" --projection-directory /arc/projects/k-pop/msa_runs/stellar-improvement-20261015/alpha_desi_source_projection_v1 --base /arc/projects/k-pop/msa_runs/stellar-improvement-20261015 --output "$experiment_output" --unit-evidence configs/desi-alpha-source-policy-20261023.json
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == alpha-target-predictions ]]; then
      "${pixi_run[@]}" python -u scripts/run_alpha_target_wave.py --prepared "$MSA_TARGET_PREPARED" --preparation-manifest "$MSA_TARGET_PREPARATION_MANIFEST" --output "$experiment_output" --kind "$MSA_TARGET_KIND" --seed "$MSA_TARGET_SEED" --steps "$MSA_TARGET_STEPS" --library-revision "$MSA_TARGET_LIBRARY_REV"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == mae-ttt ]]; then
      "${pixi_run[@]}" python -u scripts/run_mae_ttt_screen.py --config configs/mae-ttt-screen-20261023.yaml --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == target-source-evidence ]]; then
      "${pixi_run[@]}" python -u scripts/prepare_alpha_source_evidence.py --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == target-predictions ]]; then
      "${pixi_run[@]}" python -u scripts/run_posterior_target_wave.py --prepared "$MSA_TARGET_PREPARED" --preparation-manifest "$MSA_TARGET_PREPARATION_MANIFEST" --output "$experiment_output" --kind "$MSA_TARGET_KIND" --seed "$MSA_TARGET_SEED" --steps "$MSA_TARGET_STEPS"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == multilabel-validation ]]; then
      "${pixi_run[@]}" python -u scripts/run_multilabel_screen.py --config configs/multilabel-validation-wave-20261022.yaml --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output" --checkpoint "$MSA_PRETRAIN_RESUME" --benchmark "$MSA_EXPERIMENT_BENCHMARK"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == density-rescore ]]; then
      "${pixi_run[@]}" python -u scripts/run_density_rescoring_wave.py --config configs/density-rescoring-wave-20261022.yaml --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == conformal-validation ]]; then
      "${pixi_run[@]}" python -u scripts/run_conformal_validation.py --config configs/conformal-validation-wave-20261022.yaml --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == target-head-pilot ]]; then
      "${pixi_run[@]}" python -u scripts/run_target_pilot_wave.py --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
      exit 0
    fi
    if [[ "${MSA_EXPERIMENT_TASK:-training}" == catalogue-comparison ]]; then
      "${pixi_run[@]}" python -u scripts/run_catalogue_comparison_validation.py --config configs/catalogue-comparison-wave-20261022.yaml --run-id "$MSA_EXPERIMENT_RUNS" --output "$experiment_output"
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
