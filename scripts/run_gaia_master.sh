#!/usr/bin/env bash
# Build the Gaia-anchored master catalogue on CANFAR or a large workstation.
#
# The anchor is Gaia DR3 (LSDB HATS by default); every survey is matched
# independently and the output keeps exactly one row per Gaia source.
#
# Examples:
#   bash scripts/run_gaia_master.sh --preflight
#   OUT=/arc/projects/hats/xmatcher/gaia_master.hats \
#     CACHE=/arc/projects/hats \
#     bash scripts/run_gaia_master.sh
#
# Local CANFAR products are picked up only when their environment variable is
# set; no fallback sky is substituted for an unset local path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANCHOR="${ANCHOR:-gaia_lsdb_hats}"
RADIUS="${RADIUS:-1.5}"
TARGET_EPOCH="${TARGET_EPOCH:-2016.0}"
CACHE="${CACHE:-${XMATCHER_CACHE_ROOT:-$HOME/.cache/xmatcher}}"
OUT="${OUT:-$PWD/gaia_master.hats}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "$arg" in
    --preflight) PREFLIGHT_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

SURVEYS=(
  twomass_hats
  allwise_hats
  catwise_hats
  skymapper_hats
  sdss_hats
  ps1_hats
  vhs_hats
  galex_hats
  splus_hats
)

# Prefer local CANFAR copies when present.
if [[ -n "${XMATCHER_SPLUS_DR5_HATS:-}" ]]; then
  SURVEYS=("${SURVEYS[@]/splus_hats/splus5_hats}")
fi
if [[ -n "${XMATCHER_UHS_DR3_HATS:-}" ]]; then SURVEYS+=(uhs_hats); fi
if [[ -n "${XMATCHER_PRISTINE_DR2_HATS:-}" ]]; then SURVEYS+=(pristine_hats); fi
if [[ -n "${XMATCHER_HSC_PDR3_HATS:-}" ]]; then SURVEYS+=(hsc_hats); fi
if [[ -n "${XMATCHER_DP2_HATS:-}" ]]; then SURVEYS+=(dp2); fi

if [[ -n "${XMATCHER_SURVEYS:-}" ]]; then
  # shellcheck disable=SC2206
  SURVEYS=(${XMATCHER_SURVEYS})
fi

echo "anchor:  ${ANCHOR}"
echo "surveys: ${SURVEYS[*]}"
echo "radius:  ${RADIUS} arcsec"
echo "epoch:   ${TARGET_EPOCH}"
echo "cache:   ${CACHE}"
echo "output:  ${OUT}"
echo
echo "disk headroom:"
df -h "$(dirname "${OUT}")" 2>/dev/null || true
echo

if [[ "${PREFLIGHT_ONLY}" == "1" ]]; then
  echo "preflight only; no match started"
  exit 0
fi

if [[ -e "${OUT}" ]]; then
  echo "output already exists: ${OUT}" >&2
  echo "remove it or choose a new OUT; the engine resumes only a matching fingerprint" >&2
  exit 1
fi

CMD=(
  xmatcher match
  "${ANCHOR}"
  "${SURVEYS[@]}"
  --engine ray-anchor
  --radius "${RADIUS}"
  --target-epoch "${TARGET_EPOCH}"
  --cache-root "${CACHE}"
  --no-color
  -o "${OUT}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'dry run: '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
echo
python "${SCRIPT_DIR}/validate_anchor_output.py" "${OUT}" --progress-every 100
