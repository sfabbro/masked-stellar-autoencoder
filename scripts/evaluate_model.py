"""The single evaluator of the MSA programme (qa.md §5; design 2026-11-20 §5).

usage: evaluate_model.py --predictions NAME=PATH[:ADAPTER] [...] --out DIR [options]

Reads saved predictions only (``qa/predictions.py``: canonical npz layout, or
the v10-era ``posterior`` / ``s2`` / ``s3`` files) and the frozen
``msa-split-v1`` assignment (hash + ``overrides.parquet``, ``data/msa_split.py``).
Every check runs on TEST rows only (``--partition dev --allow-dev`` for DEV);
QA-reserved objects (GBS, globular clusters, age benchmarks) are scored
report-only. Rows a legacy model has seen can be dropped with
``--exclude-table-splits`` (data.md §3.10: v5-v10 models are compared only on
TEST minus v5 train and valid).

Checks (each skipped, and listed under ``missing``, when its inputs are absent):

* integrity: duplicates, finite and monotone quantiles;
* held-out labels per product: bias / NMAD and 68 / 95 % coverage with
  Wilson-Bonferroni bands per stratum (G, BP-RP, [Fe/H] ref and predicted,
  stage, |b|, SFD E(B-V), input class X/P/G, RUWE), PIT, pulls;
  supplied PITs (S3 parallax and E(B-V)) per stratum;
* frozen metal-poor core: bias per [Fe/H]_ref x G cell, metal-poor selection
  purity / completeness, P([Fe/H] < -2.5) Brier and AUC;
* co-natal: wide binaries (El-Badry+2021, R_chance < 0.1, TEST pair groups,
  neither-labelled split), held-out binary pool, open clusters (HR23,
  Prob >= 0.7, TEST clusters), globular clusters vs Harris (reserved);
* physical references: IRFM Teff, seismic log g (APOKASC-3, APO-K2, APOKASC
  dwarfs, Kepler LEGACY), GBS v3 (exact DR3 ids);
* competitors on exact-id overlap: Andrae 2023 (own training stars removed),
  SHBoost, StarHorse 2023, GSP-Phot; the others are listed as missing.

Outputs in ``--out``: ``metrics.json``, ``report.md``, ``plots/*.png``.
``--stage-competitors DIR`` instead writes the competitor rows of the
evaluated ids to slim parquet files (run where the catalogues are mounted).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from masked_stellar_autoencoder.data.msa_split import FREEZE, PARTITIONS, MsaSplit
from masked_stellar_autoencoder.qa import competitors as C
from masked_stellar_autoencoder.qa import conatal as K
from masked_stellar_autoencoder.qa import core as Q
from masked_stellar_autoencoder.qa import physical as Ph
from masked_stellar_autoencoder.qa.heldout import (
    evaluate_membership,
    evaluate_pit,
    evaluate_product,
    integrity,
    strata_for,
)
from masked_stellar_autoencoder.qa.predictions import load_predictions
from masked_stellar_autoencoder.qa.stats import pit_from_quantiles

PART_CODE = {"test": 3, "dev": 1}
REF_PRODUCTS = ("teff", "logg", "fe_h", "ebv")


def log(msg):
    print(msg, flush=True)


def _kv(items):
    out = {}
    for it in items or []:
        name, _, rest = it.partition("=")
        if not rest:
            raise SystemExit(f"expected NAME=PATH, got {it!r}")
        out[name] = rest
    return out


def load_covariates(path) -> pd.DataFrame | None:
    """source_id-keyed covariates: G, bp_rp, l, b, ebv_sfd, ruwe, input_class."""
    if not path:
        return None
    t = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.DataFrame(
        dict(np.load(path, allow_pickle=False))
    )  # fmt: skip
    ren = dict(phot_g_mean_mag="G", has_xp_continuous="has_xp")
    t = t.rename(columns={k: v for k, v in ren.items() if k in t.columns})
    if "input_class" not in t.columns and "has_xp" in t.columns:
        # Gaia-only rule of the freeze: X with XP, P with G, BP and RP, else G
        t["input_class"] = np.where(t["has_xp"].astype(bool), "X",
                                    np.where(np.isfinite(t.get("bp_rp", np.nan)), "P", "G"))  # fmt: skip
    return t.drop_duplicates("source_id").set_index("source_id")


def covariates_for(pred, cov: pd.DataFrame | None) -> dict:
    n = len(pred)
    out = {}
    if cov is not None:
        sid = pred.source_id
        hit = np.isin(sid, cov.index.to_numpy())
        for c in ("G", "bp_rp", "l", "b", "ebv_sfd", "ruwe", "input_class"):
            if c in cov.columns:
                v = np.full(n, "" if c == "input_class" else np.nan, dtype=object)
                v[hit] = cov.loc[sid[hit], c].to_numpy()
                out[c] = v if c == "input_class" else v.astype(np.float64)
    for c in ("G", "ebv_sfd", "bp_rp"):  # fall back on the prediction file's own
        if c in pred.extra and (c not in out or not np.isfinite(out[c]).any()):
            out[c] = np.asarray(pred.extra[c], np.float64)
    if "input_class" not in out and pred.extra.get("input_class") is not None:
        out["input_class"] = np.asarray(pred.extra["input_class"]).astype(str)
    return out


def load_labels(path) -> pd.DataFrame | None:
    if not path:
        return None
    t = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.DataFrame(
        dict(np.load(path, allow_pickle=False))
    )  # fmt: skip
    return t.drop_duplicates("source_id").set_index("source_id")


def references_for(pred, labels: pd.DataFrame | None) -> dict:
    """Reference per product: the file's own ``<p>_ref``, else ``--labels``."""
    refs = {}
    for p in REF_PRODUCTS:
        if f"{p}_ref" in pred.extra:
            refs[p] = np.asarray(pred.extra[f"{p}_ref"], np.float64)
    if labels is not None:
        sid = pred.source_id
        hit = np.isin(sid, labels.index.to_numpy())
        for p in REF_PRODUCTS:
            if p in labels.columns:
                v = refs.get(p, np.full(len(sid), np.nan)).copy()
                fill = hit & ~np.isfinite(v)
                v[fill] = labels.loc[sid[fill], p].to_numpy(np.float64)
                refs[p] = v
    return refs


def select_rows(pred, split, partition: int, exclude_splits) -> tuple:
    part = split.partition(pred.source_id)
    seen = np.zeros(len(pred), bool)
    if exclude_splits and "table_split" in pred.extra:
        seen = np.isin(np.asarray(pred.extra["table_split"]), exclude_splits)
    keep = (part == partition) & ~seen
    counts = {PARTITIONS[k]: int((part == k).sum()) for k in range(5)}
    counts["excluded_seen_in_partition"] = int(((part == partition) & seen).sum())
    counts["evaluated"] = int(keep.sum())
    return keep, part == 4, counts


def evaluate_set(name, pred, split, a, cov, labels, physical, comp_frames) -> dict:
    log(f"[{name}] {len(pred)} rows, products {sorted(pred.q)}")
    keep, reserved, counts = select_rows(
        pred, split, PART_CODE[a.partition], a.exclude_table_splits
    )
    out = dict(model=pred.model, kind=pred.kind, rows=counts, integrity=integrity(pred))
    sub = pred.subset(keep)
    c = covariates_for(sub, cov)
    refs = references_for(sub, labels)
    strata = strata_for(sub, c, refs)
    out["reference_source"] = {
        p: ("prediction file (<p>_ref)" if f"{p}_ref" in sub.extra else "--labels")
        for p in refs
    }
    out["strata_known"] = {k: float(np.mean(v != "")) for k, v in strata.items()}
    out["heldout"] = {}
    for p in REF_PRODUCTS:
        if p in sub.q and p in refs and np.isfinite(refs[p]).any():
            out["heldout"][p] = evaluate_product(sub, p, refs[p], strata)
    out["supplied_pit"] = {}
    for key in ("parallax_pit", "ebv_pit"):
        if key in sub.extra:
            out["supplied_pit"][key] = evaluate_pit(sub.extra[key], strata)
    out["joint_sets"] = {}
    for mass in ("68", "95"):  # the model's own joint set (Neyman, conformal)
        key = f"joint_member{mass}"
        if key in sub.extra:
            has_ref = np.ones(len(sub), bool)
            if refs:
                has_ref = np.any([np.isfinite(v) for v in refs.values()], axis=0)
            out["joint_sets"][mass] = evaluate_membership(
                sub.extra[key], strata, int(mass) / 100, has_ref
            )
    if "teff" in sub.q and "teff_irfm" in sub.extra:
        out["irfm"] = Ph.score_irfm(sub.median("teff"), sub.extra["teff_irfm"])
    if physical is not None and len(physical):
        pts = {p: pred.median(p) for p in ("teff", "logg", "fe_h") if p in pred.q}
        seen = np.zeros(len(pred), bool)
        if a.exclude_table_splits and "table_split" in pred.extra:
            seen = np.isin(
                np.asarray(pred.extra["table_split"]), a.exclude_table_splits
            )
        # reserved benchmarks are report-only; rows the model has seen are not scored
        test_or_res = (keep | reserved) & ~seen
        out["physical_rows_seen_dropped"] = int(((keep | reserved) & seen).sum())
        allp = pred.subset(test_or_res)
        out["physical"] = Ph.score_physical(
            physical, allp.source_id, {k: v[test_or_res] for k, v in pts.items()},
            strata=dict(stage=strata_for(allp, covariates_for(allp, cov), {})["stage"]),
        )  # fmt: skip
    if comp_frames:
        model = {
            p: sub.median(p)
            for p in sub.q
            if p in ("teff", "logg", "fe_h", "ebv", "dist_mod")
        }
        out["competitors"] = {
            cname: dict(
                C.compare(sub.source_id, model, refs, frame),
                note=C.NOTES.get(cname, ""),
            )
            for cname, frame in comp_frames.items()
        }
    return out, sub, refs, strata


def run_core(a, pred, split, cov=None) -> dict:
    """Frozen core: references from the records of the table the rows index."""
    import yaml

    rec = np.load(a.core_records, allow_pickle=False)
    cfg = yaml.safe_load(Path(a.core_config).read_text())
    ref_all, eref_all, _ = Q.core_references(rec, cfg["multisurvey"]["fe_h_core"])
    rows = np.asarray(pred.extra["table_row"], np.int64)
    ref = ref_all[rows]
    keep, _, counts = select_rows(
        pred, split, PART_CODE[a.partition], a.exclude_table_splits
    )
    feh = pred.median("fe_h")
    G = np.asarray(pred.extra.get("G", np.full(len(pred), np.nan)), np.float64)
    if not np.isfinite(G).any():
        G = np.asarray(covariates_for(pred, cov).get("G", G), np.float64)
    m = keep & np.isfinite(ref)
    table = Q.core_table(feh[m], ref[m], G[m])
    out = dict(
        rows=counts,
        with_reference=int(m.sum()),
        table=table,
        worst_bias_G_le_16=Q.worst_bias(table),
    )
    p = pred.extra.get("p_feh_lt_m2p5")
    if p is None:
        p = pit_from_quantiles(
            pred.levels, pred.q["fe_h"], np.full(len(pred), Q.MP_LIMIT)
        )
    p = np.asarray(p, np.float64)
    out["selection_p_gt_0.9"] = Q.selection(ref[m], G[m], p[m] > 0.9)
    out["tail_probability"] = Q.tail_probability(ref[m], p[m])
    if "fe_h" in pred.q:
        from masked_stellar_autoencoder.qa.stats import central_interval

        iv = central_interval(pred.levels, pred.q["fe_h"], 0.68)
        if iv is not None:
            mp = m & (ref < -2)
            out["coverage68_feh_ref_lt_m2"] = dict(
                n=int(mp.sum()),
                coverage=float(np.mean((ref[mp] >= iv[0][mp]) & (ref[mp] <= iv[1][mp]))) if mp.any() else np.nan,
            )  # fmt: skip
    return out


def run_conatal(a, pred, split) -> dict:
    """Wide binaries, binary pool and clusters on the consistency predictions."""
    feh_q = pred.q["fe_h"]
    sd = 0.5 * (feh_q[:, -1] - feh_q[:, 0])
    lab = np.load(a.labelled)["source_id"] if a.labelled else None
    G = pred.extra.get("G", np.full(len(pred), np.nan))
    P = K.PredTable.from_arrays(pred.source_id, pred.median("teff"), pred.median("logg"),
                                pred.median("fe_h"), sd, G, lab)  # fmt: skip
    target = PART_CODE[a.partition]
    T = K.targets_from_vizier(a.vizier_dir)
    out = {}
    pair_part = split.partition(T["pair1"])
    out["wide_binaries"] = K.wide_binaries(
        P, T["pair1"], T["pair2"], T["pair_r"], keep=pair_part == target
    )
    if a.binary_pool:
        pool = np.load(a.binary_pool)
        hold = pool["heldout"].astype(bool)
        comp = pool["source_id"][hold]
        keep = split.partition(comp) == target
        out["binary_pool_heldout"] = K.binary_pool_heldout(
            P, comp, pool["primary_id"][hold], pool["teff_companion"][hold],
            pool["g_companion"][hold], keep=keep,
        )  # fmt: skip
    ct = K.cluster_table(f"{a.vizier_dir}/hr23_clusters.tsv")
    harris = K.harris_table(f"{a.vizier_dir}/harris.tsv")
    member, mcl, prob = T["member"], T["member_cluster"], T["member_prob"]
    mpart = split.partition(member)
    cpart = pd.Series(mpart).groupby(mcl).agg(lambda s: s.mode().iloc[0]).to_dict()
    ctype = dict(zip(ct["ID"], ct["Type"], strict=True))

    def keep_oc(c):
        return ctype.get(c) != "g" and cpart.get(c) == target

    def keep_gc(c):
        return ctype.get(c) == "g"  # VB21 globular members are QA-reserved: report-only

    r_oc = K.clusters(P, member, mcl, prob, ct, harris, keep_cluster=keep_oc)
    r_gc = K.clusters(P, member, mcl, prob, ct, harris, keep_cluster=keep_gc)
    out["open_clusters"] = K.open_cluster_summary(r_oc)
    out["globular_clusters"] = K.globular_cluster_summary(r_gc)
    out["_tables"] = dict(open=r_oc, globular=r_gc)
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--predictions",
        action="append",
        default=[],
        help="NAME=PATH[:ADAPTER] (adapter: auto, canonical, posterior, s2, s3)",
    )
    p.add_argument(
        "--core",
        action="append",
        default=[],
        help="NAME of a --predictions set whose table_row indexes --core-records (repeatable)",
    )
    p.add_argument("--core-records")
    p.add_argument(
        "--core-config", help="fine-tune config naming multisurvey.fe_h_core"
    )
    p.add_argument(
        "--conatal", help="PATH[:ADAPTER] of predictions for the consistency stars"
    )
    p.add_argument(
        "--vizier-dir",
        help="elbadry.tsv, hr23_members.tsv, hr23_clusters.tsv, harris.tsv",
    )
    p.add_argument(
        "--labelled", help="npz: source_id of stars whose labels a model has seen"
    )
    p.add_argument("--binary-pool")
    p.add_argument(
        "--labels",
        help="parquet/npz reference labels (source_id, teff, logg, fe_h, ebv)",
    )
    p.add_argument(
        "--covariates",
        help="parquet: source_id, G, bp_rp, l, b, ebv_sfd, ruwe, input_class|has_xp",
    )
    p.add_argument(
        "--physical-catalogues", help="folder of the seismic catalogues (k-pop)"
    )
    p.add_argument("--gbs", help="GBS v3 table with exact DR3 ids")
    p.add_argument(
        "--competitor", action="append", default=[], help="NAME=PATH (full or staged)"
    )
    p.add_argument(
        "--stage-competitors",
        help="write competitor rows of the evaluated ids here, then exit",
    )
    p.add_argument("--split-dir", default=str(FREEZE))
    p.add_argument("--partition", choices=sorted(PART_CODE), default="test")
    p.add_argument("--allow-dev", action="store_true")
    p.add_argument(
        "--exclude-table-splits",
        type=lambda s: [int(x) for x in s.split(",")],
        default=[],
        help="drop rows whose table_split is listed (e.g. 0,1)",
    )
    p.add_argument(
        "--table-splits",
        action="append",
        default=[],
        help="NAME=npz (table_row, split): legacy row splits of the table that set "
        "NAME's table_row indexes (only for sets without table_split)",
    )
    p.add_argument("--title", default="MSA QA")
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    if a.partition == "dev" and not a.allow_dev:
        raise SystemExit("DEV is for decisions: pass --allow-dev to evaluate it")
    split = MsaSplit.from_freeze(a.split_dir)
    preds = {}
    for name, spec in _kv(a.predictions).items():
        path, _, adapter = spec.partition(":")
        preds[name] = load_predictions(path, adapter or "auto", model=name)
    for name, path in _kv(a.table_splits).items():
        pr = preds[name]
        if "table_split" in pr.extra:
            raise SystemExit(f"{name} already carries table_split")
        with np.load(path) as z:
            split_of = dict(
                zip(z["table_row"].tolist(), z["split"].tolist(), strict=True)
            )
        rows = np.asarray(pr.extra["table_row"]).tolist()
        pr.extra["table_split"] = np.array([split_of.get(r, -1) for r in rows])
    if a.stage_competitors:
        ids = np.unique(np.concatenate([
            pr.source_id[split.partition(pr.source_id) == PART_CODE[a.partition]] for pr in preds.values()
        ]))  # fmt: skip
        sources = {k: Path(v) for k, v in _kv(a.competitor).items()} or None
        info = C.stage_competitors(ids, a.stage_competitors, sources, log=log)
        Path(a.stage_competitors, "staged.json").write_text(
            json.dumps(info, indent=2) + "\n"
        )
        return
    if a.out is None:
        raise SystemExit("--out is required")
    out_dir = a.out
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    cov = load_covariates(a.covariates)
    labels = load_labels(a.labels)
    physical = None
    missing = {}
    if a.physical_catalogues or a.gbs:
        physical = Ph.load_physical(a.physical_catalogues, a.gbs, log=log)
    else:
        missing["physical"] = "no --physical-catalogues / --gbs"
    comp_frames = {}
    all_ids = (
        np.unique(np.concatenate([pr.source_id for pr in preds.values()]))
        if preds
        else []
    )
    for cname, path in _kv(a.competitor).items():
        try:
            comp_frames[cname] = C.load_competitor(cname, path, all_ids)
        except (OSError, FileNotFoundError, KeyError) as exc:
            missing[f"competitor:{cname}"] = str(exc)
    for cname in C.SOURCES:
        if cname not in comp_frames and f"competitor:{cname}" not in missing:
            missing[f"competitor:{cname}"] = (
                "not given (on ARC: " + str(C.SOURCES[cname]) + ")"
            )
    for cname, why in C.MISSING.items():
        missing[f"competitor:{cname}"] = why
    metrics = dict(
        title=a.title, spec="msa-split-v1", split_dir=str(a.split_dir), partition=a.partition.upper(),
        exclude_table_splits=a.exclude_table_splits, created_utc=dt.datetime.now(dt.UTC).isoformat(),
        code_commit=git_commit(), inputs=dict(predictions=_kv(a.predictions), conatal=a.conatal,
        covariates=a.covariates, labels=a.labels, core_records=a.core_records,
        competitors=_kv(a.competitor)), sets={},
    )  # fmt: skip
    keep_for_plots = {}
    for name, pred in preds.items():
        res, sub, refs, strata = evaluate_set(
            name, pred, split, a, cov, labels, physical, comp_frames
        )
        metrics["sets"][name] = res
        keep_for_plots[name] = (sub, refs, strata)
    if a.core:
        if not (a.core_records and a.core_config):
            missing["core"] = "--core needs --core-records and --core-config"
        else:
            metrics["core"] = {
                name: run_core(a, preds[name], split, cov) for name in a.core
            }
    else:
        missing["core"] = "no --core set"
    tables = {}
    if a.conatal and a.vizier_dir:
        path, _, adapter = a.conatal.partition(":")
        cpred = load_predictions(path, adapter or "auto", model="conatal")
        res = run_conatal(a, cpred, split)
        tables = res.pop("_tables")
        metrics["conatal"] = res
    else:
        missing["conatal"] = "no --conatal / --vizier-dir"
    metrics["missing"] = missing
    from masked_stellar_autoencoder.qa.report import write_outputs

    write_outputs(out_dir, metrics, keep_for_plots, tables)
    log(f"wrote {out_dir}/metrics.json, report.md, plots/")


if __name__ == "__main__":
    sys.exit(main())
