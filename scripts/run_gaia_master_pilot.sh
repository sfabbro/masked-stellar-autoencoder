#!/usr/bin/env bash
# P1 pilot of the Gaia-anchored full-wavelength master (docs/gaia-master-data.md).
#
# Runs `xmatcher match --engine ray-anchor` on a few Gaia DR3 anchor partitions
# against every community tree read in place, then checks and reports:
#   - each input is read in place (xmatcher's "in place" log lines);
#   - /arc/projects/hats is unchanged (snapshots before/after);
#   - what the cache root gained (only the community index is expected);
#   - validator, per-survey match/crowded/ambiguous/invalid, per-task time and
#     peak memory, and per-partition candidate estimates for the full run.
#
# Usage (CANFAR headless, /arc mounted; xmatcher is a private repo, so its
# source comes as a tarball on ARC):
#   bash scripts/run_gaia_master_pilot.sh --report DIR \
#     --xmatcher-tarball /arc/projects/astroai/msa/launcher/xmatcher_src/xmatcher-<sha>.tar.gz \
#     --xmatcher-sha256 <sha256> [--anchor-partitions 2/69,6/42837] [--out PATH]
set -euo pipefail

REPORT=""
TARBALL=""
TARBALL_SHA=""
OUT="/arc/projects/astroai/msa/xmatcher/pilot_gaia_master.hats"
CACHE="/arc/projects/astroai/hats"
ANCHOR_PARTITIONS="2/69,6/42837"
COMMUNITY="/arc/projects/hats"
ANCHOR="gaia_lsdb_hats"
# Every community entry with local_paths (or a sync record) that is not being
# rebuilt: PS1 OTMO, DES Y6, DELVE DR3 and UHS DR3 are left out (Pensus
# rebuilds), J-PLUS/J-PAS too (string magnitudes).
SURVEYS=(
  twomass_psc_hats allwise_hats catwise2020_catalog skymapper_dr4_hats
  sdss_dr16_photoprimary ls_dr11_sweep galex_guvcat_ais vhs_dr7
  ukidss_dr11_las ukidss_dr11_gcs viking_dr5 vvvx_dr1 vmc_dr7
  splus_dr4_dual pristine_dr2
  hsc_pdr3_forced_g hsc_pdr3_forced_r hsc_pdr3_forced_i hsc_pdr3_forced_z hsc_pdr3_forced_y
  euclid_dr1_mer_wide
)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --report) REPORT="$2"; shift 2 ;;
    --xmatcher-tarball) TARBALL="$2"; shift 2 ;;
    --xmatcher-sha256) TARBALL_SHA="$2"; shift 2 ;;
    --anchor-partitions) ANCHOR_PARTITIONS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --cache) CACHE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$REPORT" && -n "$TARBALL" && -n "$TARBALL_SHA" ]] || {
  echo "need --report, --xmatcher-tarball and --xmatcher-sha256" >&2; exit 2; }
for path in "$OUT" "$CACHE" "$REPORT"; do
  case "$(realpath -m "$path")/" in
    "$COMMUNITY"/*) echo "refusing to write under $COMMUNITY: $path" >&2; exit 2 ;;
  esac
done
if [[ -e "$OUT" ]]; then
  echo "pilot output already exists: $OUT (choose a new --out)" >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
unset PYTHONPATH RAY_ADDRESS
mkdir -p "$REPORT"
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
WORK="${WORK:-/scratch/src}/xmatcher-pilot"
XM="$WORK/xmatcher"
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ---------------------------------------------------------------- xmatcher
echo "$TARBALL_SHA  $TARBALL" | sha256sum -c -
rm -rf "$XM" && mkdir -p "$WORK"
tar -xzf "$TARBALL" -C "$WORK"
[[ -f "$XM/pixi.toml" ]] || { echo "tarball has no xmatcher/pixi.toml" >&2; exit 2; }
# Leave the MSA environment's activation behind for xmatcher's own env.
xm_env() {
  env -u PIXI_PROJECT_ROOT -u PIXI_PROJECT_NAME -u PIXI_PROJECT_MANIFEST \
    -u PIXI_PROJECT_VERSION -u PIXI_ENVIRONMENT_NAME -u PIXI_ENVIRONMENT_PLATFORMS \
    -u PIXI_PROMPT -u PIXI_IN_SHELL -u PIXI_EXE -u CONDA_PREFIX -u CONDA_DEFAULT_ENV \
    -u CONDA_SHLVL "$@"
}
log "installing xmatcher env"
xm_env pixi install --manifest-path "$XM/pixi.toml" --frozen
# No network is needed past this point; Ray's local gRPC must not go through
# the session's CONNECT proxy.
xm_py() {
  xm_env env -u HTTPS_PROXY -u HTTP_PROXY -u https_proxy -u http_proxy \
    XMATCHER_CACHE_ROOT="$CACHE" NO_COLOR=1 \
    pixi run --manifest-path "$XM/pixi.toml" --frozen "$@"
}
xm_py python -c 'import xmatcher, torchsky, ray, polars; print("xmatcher", xmatcher.__version__, "torchsky", torchsky.__version__, "ray", ray.__version__, "polars", polars.__version__)' \
  | tee "$REPORT/versions.txt"
cat "$XM/XMATCHER_COMMIT" 2>/dev/null | tee -a "$REPORT/versions.txt" || true

# ---------------------------------------------------------------- snapshots
mem_monitor() {
  while true; do
    {
      printf '%s ' "$(date -u +%s)"
      cat /sys/fs/cgroup/memory.current 2>/dev/null \
        || cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo NA
    } >> "$REPORT/memory.log"
    sleep 15
  done
}
mem_monitor &
MON=$!
trap 'kill $MON 2>/dev/null || true' EXIT

log "snapshot before: $COMMUNITY and $CACHE"
xm_py python "$HERE/gaia_master_pilot.py" snapshot "$COMMUNITY" "$REPORT/hats_before.tsv"
mkdir -p "$CACHE"
xm_py python "$HERE/gaia_master_pilot.py" snapshot "$CACHE" "$REPORT/cache_before.tsv"
df -h "$(dirname "$OUT")" "$CACHE" | tee "$REPORT/df_before.txt" || true

# ---------------------------------------------------------------- match
log "pilot match: anchor $ANCHOR partitions $ANCHOR_PARTITIONS, ${#SURVEYS[@]} surveys"
CMD=(xmatcher -v match "$ANCHOR" "${SURVEYS[@]}"
  --engine ray-anchor -r 1.5 --target-epoch 2016.0
  --anchor-partitions "$ANCHOR_PARTITIONS"
  --no-sync --cache-root "$CACHE"
  --task-rows 4000000000 --chunk-memory-gb 4096
  --no-color -o "$OUT")
printf '%q ' "${CMD[@]}" > "$REPORT/command.txt"; echo >> "$REPORT/command.txt"
cat > "$WORK/timed.py" <<'PY'
import json, resource, subprocess, sys, time
t = time.time()
code = subprocess.call(sys.argv[1:])
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print("MATCH_RUSAGE " + json.dumps({"exit": code, "wall_s": round(time.time() - t, 1),
      "children_maxrss_gib": round(ru.ru_maxrss / 1024**2, 3),
      "user_s": round(ru.ru_utime, 1), "sys_s": round(ru.ru_stime, 1)}), flush=True)
sys.exit(code)
PY
started=$(date +%s)
set +e
xm_py python "$WORK/timed.py" "${CMD[@]}" > "$REPORT/match.log" 2>&1
code=$?
set -e
echo "match exit=$code wall=$(( $(date +%s) - started ))s" | tee "$REPORT/match_exit.txt"
grep -E "in place|not usable|mirror|anchor task|anchor plan|dropped|Error|MATCH_RUSAGE" \
  "$REPORT/match.log" | tee "$REPORT/match_key_lines.txt" || true
cat /sys/fs/cgroup/memory.peak 2>/dev/null > "$REPORT/cgroup_memory_peak.txt" || true

# ---------------------------------------------------------------- checks
log "snapshot after"
xm_py python "$HERE/gaia_master_pilot.py" snapshot "$COMMUNITY" "$REPORT/hats_after.tsv"
xm_py python "$HERE/gaia_master_pilot.py" snapshot "$CACHE" "$REPORT/cache_after.tsv"
xm_py python "$HERE/gaia_master_pilot.py" diff "$REPORT/hats_before.tsv" "$REPORT/hats_after.tsv" \
  --json "$REPORT/hats_diff.json" --depth 2 > /dev/null
xm_py python "$HERE/gaia_master_pilot.py" diff "$REPORT/cache_before.tsv" "$REPORT/cache_after.tsv" \
  --json "$REPORT/cache_diff.json" --depth 2 > /dev/null
gzip -f "$REPORT"/hats_before.tsv "$REPORT"/hats_after.tsv
df -h "$(dirname "$OUT")" "$CACHE" | tee "$REPORT/df_after.txt" || true

if [[ "$code" == 0 ]]; then
  log "validate"
  set +e
  xm_py python "$XM/scripts/validate_anchor_output.py" "$OUT" \
    --json "$REPORT/validation.json" --progress-every 1 > "$REPORT/validate.log" 2>&1
  vcode=$?
  set -e
  echo "validator exit=$vcode" | tee -a "$REPORT/match_exit.txt"
  xm_py python "$HERE/gaia_master_pilot.py" summary --validation "$REPORT/validation.json" \
    --hats "$OUT" --out "$REPORT/pilot_summary.json"
  cp "$OUT/properties" "$OUT/resume.state" "$OUT/partition_info.csv" "$REPORT/" || true
  du -sb "$OUT" > "$REPORT/output_du.txt" || true
fi

log "full-run task estimates (footers of every input partition)"
xm_py python "$HERE/gaia_master_pilot.py" estimate --anchor "$ANCHOR" \
  --surveys "${SURVEYS[@]}" --cache-root "$CACHE" --radius 1.5 --target-epoch 2016.0 \
  --threads 16 --out "$REPORT/task_estimates.parquet" > "$REPORT/estimate.log" 2>&1 \
  || echo "estimate failed (see estimate.log)"
log "done"
exit "$code"
