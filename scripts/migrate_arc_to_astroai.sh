#!/usr/bin/env bash
# Move the MSA data from the k-pop project to /arc/projects/astroai/msa/.
#
# usage: migrate_arc_to_astroai.sh copy|verify|delete
#
# Layout of the new tree:
#   runs/                    every run, table, label and evaluation directory of
#                            k-pop/msa_runs/stellar-improvement-20261015 (same
#                            names; session logs stay next to their runs)
#   launcher/source_bundles/ the launcher's per-attempt source bundles
#   pretrain/                k-pop/msa_pretrain (July 2026 pretraining)
#   analysis/                analysis products copied from working sessions
#
# copy   rsync both source trees into place (resumable).
# verify per top-level entry, compare file counts and apparent sizes between
#        source and copy; writes launcher/migration_20261120_verify.tsv and
#        fails on any mismatch.
# delete remove the source trees, only if the verify report exists and has no
#        mismatch (and re-checks counts first).
set -euo pipefail
SRC=/arc/projects/k-pop/msa_runs/stellar-improvement-20261015
PRE=/arc/projects/k-pop/msa_pretrain
DST=/arc/projects/astroai/msa
BUNDLES=decision_wave_source_bundles_20261106_v1
REPORT=$DST/launcher/migration_20261120_verify.tsv

pairs() {  # source<TAB>destination for every top-level entry
  for e in "$SRC"/*; do
    n=$(basename "$e")
    case "$n" in
      "$BUNDLES") printf '%s\t%s\n' "$e" "$DST/launcher/source_bundles" ;;
      puttest.txt) ;;  # upload test, not data
      *) printf '%s\t%s\n' "$e" "$DST/runs/$n" ;;
    esac
  done
  printf '%s\t%s\n' "$PRE" "$DST/pretrain"
}

stats() {  # files<TAB>bytes of a file or directory
  if [ -d "$1" ]; then
    printf '%s\t%s\n' "$(find "$1" -type f | wc -l)" "$(du -sb --apparent-size "$1" | cut -f1)"
  else
    printf '1\t%s\n' "$(stat -c %s "$1")"
  fi
}

sync() {  # sync SRC/ DST/ [exclude...]: rsync, or cp -a if the image has no rsync
  local src=$1 dst=$2; shift 2
  if command -v rsync >/dev/null; then
    local ex=(); for e in "$@"; do ex+=(--exclude "/$e"); done
    rsync -a "${ex[@]}" "$src" "$dst"
  else
    shopt -s nullglob dotglob
    for e in "$src"*; do
      n=$(basename "$e")
      for x in "$@"; do [ "$n" = "$x" ] && continue 2; done
      cp -a "$e" "$dst"
    done
    shopt -u nullglob dotglob
  fi
}

case "${1:-}" in
  copy)
    mkdir -p "$DST/runs" "$DST/launcher/source_bundles" "$DST/pretrain" "$DST/analysis"
    sync "$SRC/" "$DST/runs/" "$BUNDLES" puttest.txt
    sync "$SRC/$BUNDLES/" "$DST/launcher/source_bundles/"
    sync "$PRE/" "$DST/pretrain/"
    "$0" verify
    ;;
  verify)
    bad=0
    printf 'source\tdestination\tsrc_files\tsrc_bytes\tdst_files\tdst_bytes\tstatus\n' > "$REPORT"
    while IFS=$'\t' read -r s d; do
      a=$(stats "$s")
      if [ -e "$d" ]; then b=$(stats "$d"); else b=$'0\t0'; fi
      if [ "$(basename "$s")" = "$BUNDLES" ]; then
        # destination may also hold bundles of later launches: require >=
        sf=${a%%$'\t'*}; sb=${a##*$'\t'}; df=${b%%$'\t'*}; db=${b##*$'\t'}
        if [ "$df" -ge "$sf" ] && [ "$db" -ge "$sb" ]; then st=ok; else st=MISMATCH; fi
      elif [ "$a" = "$b" ]; then st=ok; else st=MISMATCH; fi
      [ "$st" = ok ] || bad=$((bad + 1))
      printf '%s\t%s\t%s\t%s\t%s\n' "$s" "$d" "$a" "$b" "$st" >> "$REPORT"
    done < <(pairs)
    echo "verify: $bad mismatch(es); report $REPORT"
    [ "$bad" -eq 0 ]
    ;;
  delete)
    [ -f "$REPORT" ] || { echo "no verify report"; exit 1; }
    if grep -q MISMATCH "$REPORT"; then echo "report has mismatches"; exit 1; fi
    "$0" verify
    rm -rf -- "$SRC" "$PRE"
    rmdir /arc/projects/k-pop/msa_runs 2>/dev/null || true
    echo "deleted $SRC and $PRE"
    ;;
  *) echo "usage: $0 copy|verify|delete"; exit 2 ;;
esac
