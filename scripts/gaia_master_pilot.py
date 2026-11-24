"""Helpers for the P1 pilot of the Gaia-anchored master (docs/gaia-master-data.md).

Subcommands (run by ``scripts/run_gaia_master_pilot.sh``):

``snapshot ROOT OUT``
    Write one ``type, mtime_ns, size, path`` line per entry under ROOT (no
    symlinks followed). Reads only: used to prove that nothing is written under
    ``/arc/projects/hats`` and to see what the cache root gained.
``diff BEFORE AFTER --json OUT``
    Compare two snapshots; group added, removed and modified entries by tree
    (the first ``--depth`` path components under the root).
``estimate``
    For every anchor partition, the candidate survey partitions and rows (from
    Parquet footers) a ``ray-anchor`` task would read, plus the compressed bytes
    of the selected columns. Needs ``xmatcher`` (run it in xmatcher's env). The
    inputs are resolved exactly as ``xmatcher match --no-sync`` does, so a tree
    that is not read in place is an error, never a mirror.
``summary``
    Per-survey pilot table from the validator report, the output properties and
    the per-task statistics in ``resume.state``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


# --------------------------------------------------------------------------- #
# snapshot / diff
# --------------------------------------------------------------------------- #
def snapshot(root: Path, out: Path) -> dict:
    started = time.time()
    n = 0
    errors = 0
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w") as fh:
        stack = [str(root)]
        while stack:
            directory = stack.pop()
            try:
                entries = list(os.scandir(directory))
            except OSError:
                errors += 1
                continue
            for entry in entries:
                try:
                    st = entry.stat(follow_symlinks=False)
                except OSError:
                    errors += 1
                    continue
                if entry.is_symlink():
                    kind = "l"
                elif entry.is_dir(follow_symlinks=False):
                    kind = "d"
                    stack.append(entry.path)
                else:
                    kind = "f"
                rel = os.path.relpath(entry.path, root)
                fh.write(f"{kind}\t{st.st_mtime_ns}\t{st.st_size}\t{rel}\n")
                n += 1
    tmp.replace(out)
    info = dict(
        root=str(root),
        entries=n,
        errors=errors,
        seconds=round(time.time() - started, 1),
    )
    print(json.dumps(info), flush=True)
    return info


def _load(path: Path) -> dict[str, tuple[str, int, int]]:
    out = {}
    for line in path.read_text().splitlines():
        kind, mtime, size, rel = line.split("\t", 3)
        out[rel] = (kind, int(mtime), int(size))
    return out


def diff(before: Path, after: Path, depth: int) -> dict:
    a, b = _load(before), _load(after)

    def tree(rel: str) -> str:
        return "/".join(rel.split("/")[:depth])

    groups: dict[str, Counter] = defaultdict(Counter)
    examples: dict[str, list[str]] = defaultdict(list)
    for rel in b.keys() - a.keys():
        groups[tree(rel)]["added"] += 1
        if len(examples[tree(rel)]) < 5:
            examples[tree(rel)].append("+" + rel)
    for rel in a.keys() - b.keys():
        groups[tree(rel)]["removed"] += 1
        if len(examples[tree(rel)]) < 5:
            examples[tree(rel)].append("-" + rel)
    for rel in a.keys() & b.keys():
        if a[rel] != b[rel]:
            groups[tree(rel)]["modified"] += 1
            if len(examples[tree(rel)]) < 5:
                examples[tree(rel)].append("~" + rel)
    return {
        "before_entries": len(a),
        "after_entries": len(b),
        "unchanged": not groups,
        "trees": {
            k: {**dict(v), "examples": examples[k]} for k, v in sorted(groups.items())
        },
    }


# --------------------------------------------------------------------------- #
# estimate
# --------------------------------------------------------------------------- #
def _footer(path: Path, columns: list[str]) -> tuple[int, int]:
    import pyarrow.parquet as pq

    md = pq.read_metadata(path)
    wanted = set(columns)
    rows = 0
    nbytes = 0
    for r in range(md.num_row_groups):
        rg = md.row_group(r)
        rows += rg.num_rows
        for c in range(rg.num_columns):
            col = rg.column(c)
            if col.path_in_schema in wanted:
                nbytes += int(col.total_compressed_size)
    return rows, nbytes


def estimate(args: argparse.Namespace) -> dict:
    import polars as pl
    from xmatcher.crossmatch import CrossMatch
    from xmatcher.mirror import _mirrored_source, locate_mirrored
    from xmatcher.ray_anchor import (
        DEFAULT_MOTION_HALO_ARCSEC,
        _build_catalogue_plan,
        _candidate_partitions,
        _safe_tag,
        anchor_cone_radius_deg,
    )
    from xmatcher.storage import community_roots, open_storage

    cm = CrossMatch()
    cache_cfg = cm._cache_cfg()
    read_roots = community_roots(cache_cfg)
    names = [args.anchor, *args.surveys]
    sources = []
    resolved = {}
    for name in names:
        src = cm.resolve_source(name, {})
        root, rel = locate_mirrored(
            src, cache_roots=[args.cache_root], read_roots=read_roots
        )
        resolved[src.name] = str(Path(root) / rel) if rel else root
        sources.append(_mirrored_source(src, open_storage(root), rel, root))

    used: set[str] = set()
    plans = []
    for i, src in enumerate(sources):
        tag = "anchor" if i == 0 else _safe_tag(src.name, used)
        plans.append(
            (
                tag,
                _build_catalogue_plan(
                    src,
                    tag=tag,
                    is_anchor=i == 0,
                    anchor_id_name="source_id",
                    cache_root=args.cache_root,
                    hats_threshold=100_000,
                ),
            )
        )

    # Footers (rows + compressed bytes of the selected columns) for every partition.
    jobs = []
    for tag, cat in plans:
        for part in cat.partitions:
            jobs.append((tag, cat, part))
    started = time.time()
    footers: dict[tuple[str, str], tuple[int, int]] = {}

    def read(job):
        tag, cat, part = job
        return (tag, part.rel), _footer(Path(cat.root) / part.rel, list(cat.cols))

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for done, (key, value) in enumerate(pool.map(read, jobs), start=1):
            footers[key] = value
            if done % 5000 == 0:
                print(
                    f"footers {done}/{len(jobs)} ({time.time() - started:.0f}s)",
                    flush=True,
                )
    footer_s = time.time() - started

    halo = (
        (
            args.motion_halo_arcsec
            if args.motion_halo_arcsec is not None
            else DEFAULT_MOTION_HALO_ARCSEC
        )
        if args.target_epoch is not None
        else 0.0
    ) / 3600.0
    anchor_tag, anchor = plans[0]
    records = []
    for part in anchor.partitions:
        a_rows, a_bytes = footers[(anchor_tag, part.rel)]
        rec = {
            "order": part.order,
            "pix": part.pix,
            "anchor_rows": a_rows,
            "anchor_bytes": a_bytes,
            "cone_deg": anchor_cone_radius_deg(part, args.radius, halo),
        }
        for tag, cat in plans[1:]:
            idx = _candidate_partitions(part, cat, cone_radius_deg=rec["cone_deg"])
            rows = sum(footers[(tag, cat.partitions[i].rel)][0] for i in idx)
            nbytes = sum(footers[(tag, cat.partitions[i].rel)][1] for i in idx)
            rec[f"{tag}__parts"] = len(idx)
            rec[f"{tag}__rows"] = rows
            rec[f"{tag}__bytes"] = nbytes
        records.append(rec)
    frame = pl.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.out)
    totals = {
        tag: {
            "partitions": len(cat.partitions),
            "rows": sum(footers[(tag, p.rel)][0] for p in cat.partitions),
            "selected_bytes": sum(footers[(tag, p.rel)][1] for p in cat.partitions),
            "columns": len(cat.cols),
        }
        for tag, cat in plans
    }
    info = {
        "resolved": resolved,
        "footer_seconds": round(footer_s, 1),
        "anchor_partitions": len(anchor.partitions),
        "totals": totals,
        "out": str(args.out),
    }
    print(json.dumps(info, indent=1), flush=True)
    return info


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def _properties(root: Path) -> dict[str, str]:
    out = {}
    for line in (root / "properties").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def summary(args: argparse.Namespace) -> dict:
    report = json.loads(args.validation.read_text())
    props = _properties(args.hats)
    state = json.loads((args.hats / "resume.state").read_text())
    invalid = {}
    for item in props.get("xmatcher_invalid_survey_rows", "").split():
        tag, _, n = item.rpartition(":")
        invalid[tag] = int(n)
    tasks = state.get("task_stats") or {}
    rows = []
    for survey, block in sorted(report["surveys"].items()):
        n = block.get("rows", 0) or 1
        quality = block.get("quality", {})
        ambiguous = quality.get("ambiguous", 0) + quality.get("ambiguous+pm_warn", 0)
        per_task = {
            rel: {
                k: t["surveys"].get(survey, {}).get(k)
                for k in (
                    "partitions",
                    "rows",
                    "matched",
                    "read_s",
                    "match_s",
                    "invalid",
                )
            }
            for rel, t in tasks.items()
        }
        rows.append(
            {
                "survey": survey,
                "anchor_rows": block.get("rows", 0),
                "matched": block.get("matched", 0),
                "match_fraction": round(block.get("matched", 0) / n, 4),
                "crowded_fraction": round(block.get("crowded", 0) / n, 4),
                "crowded_of_matched": round(
                    block.get("crowded", 0) / max(block.get("matched", 0), 1), 4
                ),
                "ambiguous_fraction": round(ambiguous / n, 4),
                "pm_warn_fraction": round(block.get("pm_warn", 0) / n, 4),
                "invalid_rows_dropped": invalid.get(survey),
                "quality": quality,
                "tasks": per_task,
            }
        )
    out = {
        "hats": str(args.hats),
        "rows": report.get("rows"),
        "advertised_rows": report.get("advertised_rows"),
        "validator_error": report.get("error"),
        "anchor_partitions": props.get("xmatcher_anchor_partitions"),
        "tasks": {
            rel: {k: v for k, v in t.items() if k != "surveys"}
            for rel, t in tasks.items()
        },
        "surveys": rows,
    }
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    lines = [
        "| survey | matched | match frac | crowded frac | ambiguous frac | pm_warn frac | invalid dropped | rows read (tasks) |",
        "|---|--:|--:|--:|--:|--:|--:|---|",
    ]
    for r in rows:
        read = ", ".join(f"{(v.get('rows') or 0):,}" for v in r["tasks"].values())
        lines.append(
            f"| {r['survey']} | {r['matched']:,} | {r['match_fraction']:.3f} | "
            f"{r['crowded_fraction']:.3f} | {r['ambiguous_fraction']:.3f} | "
            f"{r['pm_warn_fraction']:.3f} | {r['invalid_rows_dropped']} | {read} |"
        )
    lines.append("")
    lines.append("| task | anchor rows | seconds | peak RSS GiB |")
    lines.append("|---|--:|--:|--:|")
    for rel, t in sorted(tasks.items()):
        lines.append(
            f"| {rel} | {t.get('anchor_rows', 0):,} | {t.get('seconds')} | {t.get('peak_rss_gib')} |"
        )
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("snapshot")
    s.add_argument("root", type=Path)
    s.add_argument("out", type=Path)
    d = sub.add_parser("diff")
    d.add_argument("before", type=Path)
    d.add_argument("after", type=Path)
    d.add_argument("--json", type=Path, required=True)
    d.add_argument("--depth", type=int, default=2)
    e = sub.add_parser("estimate")
    e.add_argument("--anchor", required=True)
    e.add_argument("--surveys", nargs="+", required=True)
    e.add_argument("--cache-root", required=True)
    e.add_argument("--radius", type=float, default=1.5)
    e.add_argument("--target-epoch", type=float)
    e.add_argument("--motion-halo-arcsec", type=float)
    e.add_argument("--threads", type=int, default=16)
    e.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("summary")
    m.add_argument("--validation", type=Path, required=True)
    m.add_argument("--hats", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "snapshot":
        snapshot(a.root, a.out)
    elif a.cmd == "diff":
        result = diff(a.before, a.after, a.depth)
        a.json.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=1), flush=True)
    elif a.cmd == "estimate":
        estimate(a)
    else:
        summary(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
