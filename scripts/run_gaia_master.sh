#!/usr/bin/env bash
# Build the Gaia-anchored master catalogue from a CANFAR session with xmatcher
# (sfabbro/xmatcher >= 665f484) on PATH. Headless runs go through
# scripts/run_gaia_master_pilot.sh (launcher steps gaia-master-*), which also
# installs xmatcher and records the write-protection checks.
#
# The anchor is the local Gaia DR3 tree (gaia_lsdb_hats -> /arc/projects/hats/gaia/dr3);
# every survey is a community tree under /arc/projects/hats read in place
# (--no-sync: nothing is mirrored, nothing is written under /arc/projects/hats).
# The output keeps exactly one row per Gaia source.
#
# Examples:
#   bash scripts/run_gaia_master.sh --preflight
#   bash scripts/run_gaia_master.sh --dry-run
#   ANCHOR_PARTITIONS=2/69,6/42837 OUT=/arc/projects/astroai/msa/xmatcher/pilot_x.hats \
#     bash scripts/run_gaia_master.sh
set -euo pipefail

ANCHOR="${ANCHOR:-gaia_lsdb_hats}"
RADIUS="${RADIUS:-1.5}"
TARGET_EPOCH="${TARGET_EPOCH:-2016.0}"
CACHE="${CACHE:-${XMATCHER_CACHE_ROOT:-/arc/projects/astroai/hats}}"
OUT="${OUT:-/arc/projects/astroai/msa/xmatcher/gaia_master.hats}"
ANCHOR_PARTITIONS="${ANCHOR_PARTITIONS:-}"
XMATCHER_SRC="${XMATCHER_SRC:-}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "$arg" in
    --preflight) PREFLIGHT_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

# Community trees verified by the P1 pilot (docs/gaia-master-data.md). Left out:
# PS1 OTMO, DES Y6, DELVE DR3 and UHS DR3 (Pensus rebuilds) and J-PLUS/J-PAS
# (string magnitudes). XMATCHER_SURVEYS replaces the list.
SURVEYS=(
  twomass_psc_hats allwise_hats catwise2020_catalog skymapper_dr4_hats
  sdss_dr16_photoprimary ls_dr11_sweep galex_guvcat_ais vhs_dr7
  ukidss_dr11_las ukidss_dr11_gcs viking_dr5 vvvx_dr1 vmc_dr7
  splus_dr4_dual pristine_dr2
  hsc_pdr3_forced_g hsc_pdr3_forced_r hsc_pdr3_forced_i hsc_pdr3_forced_z hsc_pdr3_forced_y
  euclid_dr1_mer_wide
)
if [[ -n "${XMATCHER_SURVEYS:-}" ]]; then
  # shellcheck disable=SC2206
  SURVEYS=(${XMATCHER_SURVEYS})
fi

for path in "$OUT" "$CACHE"; do
  case "$(realpath -m "$path")/" in
    /arc/projects/hats/*)
      echo "refusing to write under /arc/projects/hats: $path" >&2
      exit 2
      ;;
  esac
done

echo "anchor:  ${ANCHOR}${ANCHOR_PARTITIONS:+ (partitions ${ANCHOR_PARTITIONS})}"
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

if [[ -e "${OUT}" && "${DRY_RUN}" != "1" ]]; then
  echo "output already exists: ${OUT}" >&2
  echo "the engine resumes only a matching fingerprint; remove it or choose a new OUT" >&2
  exit 1
fi

CMD=(
  xmatcher -v match
  "${ANCHOR}"
  "${SURVEYS[@]}"
  --engine ray-anchor
  --radius "${RADIUS}"
  --target-epoch "${TARGET_EPOCH}"
  --no-sync
  --cache-root "${CACHE}"
  --task-rows 4000000000
  --chunk-memory-gb 4096
  --no-color
  -o "${OUT}"
)
if [[ -n "${ANCHOR_PARTITIONS}" ]]; then
  CMD+=(--anchor-partitions "${ANCHOR_PARTITIONS}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'dry run: '
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

XMATCHER_CACHE_ROOT="${CACHE}" "${CMD[@]}"
echo
if [[ -n "${XMATCHER_SRC}" ]]; then
  expected=()
  [[ -z "${ANCHOR_PARTITIONS}" ]] && expected=(--expected-rows 1811709771)
  python "${XMATCHER_SRC}/scripts/validate_anchor_output.py" "${OUT}" \
    --progress-every 100 "${expected[@]}"
else
  echo "set XMATCHER_SRC to an xmatcher checkout to run scripts/validate_anchor_output.py"
fi
