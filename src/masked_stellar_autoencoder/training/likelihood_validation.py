"""Fixed validation protocol of the S1 likelihood head (design note 2026-11-20, §5, §7).

All tables compare the likelihood estimates (MLE, profile and Neyman sets) and
the posterior head on the same stars:

* **Frozen core**: median offset and NMAD against the frozen high-resolution
  [Fe/H] reference (``common_scale_labels.frozen_core_reference``), held-out
  (valid + test) core stars, per reference [Fe/H] bin x G bin.
* **Anchors**: test stars' IRFM Teff and seismic log g anchor records.
* **Coverage**: joint Neyman sets (lambda(theta_label) <= C(theta_label, G)),
  per-parameter Neyman intervals, raw Wilks sets (joint and per parameter,
  from the statistics of the label), per-parameter profile intervals and the
  head's 16-84 % interval, test stars, binned per Teff, log g, [Fe/H] and G
  (``binned_conditional_coverage``).
  The labels are noisy measurements, so coverage of the label is approximate
  (conservative when the critical values were calibrated on equally noisy
  labels).
* **Selection**: purity and completeness of [Fe/H] < -2.5 against the frozen
  reference (held-out core): a likelihood-ratio test (upper end of the 95 %
  profile or Neyman interval below -2.5) and the head's P > 0.9.

Decision rule (§7), fixed before any run; an arm is adopted only if all hold:

1. **bias**: frozen held-out core, G <= 16: |median offset| <= 0.10 dex in
   every reference [Fe/H] bin (bins with >= ``MIN_N`` stars, pooled over the
   G bins up to 16);
2. **coverage**: test stars with G <= 16: the joint 68 % Neyman set covers the
   label in 0.68 +- 0.08 of the stars in every bin of Teff, log g, [Fe/H]
   and G (bins with >= ``MIN_N`` stars);
3. **bulk**: frozen held-out core, reference [Fe/H] >= -1, G <= 16:
   NMAD(MLE) <= NMAD(head) + 0.01 dex;
4. **anchors**: test stars, "no worse than the head": NMAD <= 1.05 NMAD(head)
   and |median offset| <= |offset(head)| + 10 K (Teff, IRFM anchors) or
   + 0.01 dex (log g, seismic anchors).
"""

from __future__ import annotations

import math

import numpy as np

PARAMS = ("teff", "logg", "fe_h")
FEH_BINS = (-9.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 9.0)
G_BINS = (-math.inf, 13.5, 15.0, 16.0, 17.65)
G_NAMES = ("G<=13.5", "13.5-15", "15-16", "16-17.65")
TEFF_BINS = (3000.0, 4000.0, 4500.0, 5000.0, 5500.0, 6000.0, 6500.0, 8000.0)
LOGG_BINS = (-0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.5)
COVERAGE_BINS = dict(teff=TEFF_BINS, logg=LOGG_BINS, fe_h=FEH_BINS,
                     G=(0.0, 13.5, 15.0, 16.0, 17.65))  # fmt: skip
MIN_N = 20
G_MAX = 16.0
BIAS_LIMIT = 0.10
COVERAGE_TOL = 0.08
BULK_TOL = 0.01
ANCHOR_TOL = dict(teff=10.0, logg=0.01)
ANCHOR_NMAD_FACTOR = 1.05
MP_LIMIT = -2.5


def nmad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else math.nan


def _summary(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return dict(n=0, bias=math.nan, nmad=math.nan)
    return dict(n=int(len(d)), bias=float(np.median(d)), nmad=nmad(d))


def offset_table(pred, ref, gmag):
    """Offsets per reference [Fe/H] bin: per G bin, G <= 16 pooled, all G."""
    out = []
    groups = [(n, lo, hi) for n, lo, hi in zip(G_NAMES, G_BINS[:-1], G_BINS[1:],
                                               strict=True)]  # fmt: skip
    groups += [("G<=16", -math.inf, G_MAX), ("all", -math.inf, math.inf)]
    for lo, hi in zip(FEH_BINS[:-1], FEH_BINS[1:], strict=True):
        inbin = (ref >= lo) & (ref < hi)
        for name, glo, ghi in groups:
            m = inbin & (gmag > glo) & (gmag <= ghi)
            out.append(dict(feh_lo=lo, feh_hi=hi, g=name, **_summary(pred[m] - ref[m])))
    return out


def anchor_table(pred, anchors):
    """Offsets against anchor records.

    ``pred``: estimator name -> (n_rows, 3) prediction per table row.
    ``anchors``: list of (name, parameter index, table rows, values).
    """
    out = {}
    for name, k, rows, values in anchors:
        out[name] = {est: _summary(p[rows, k] - values) for est, p in pred.items()}
    return out


def anchor_records(rec, rows_mask):
    """Anchor records of the selected table rows: [(name, k, rows, values)]."""
    names = [str(n) for n in rec["survey_names"]]
    params = [str(p) for p in rec["parameters"]]
    is_anchor = np.asarray(rec["survey_is_anchor"], bool)
    row = np.asarray(rec["row"], "i8")
    survey = np.asarray(rec["survey"], "i8")
    value = np.asarray(rec["value"], "f8")
    ok = row < len(rows_mask)
    ok[ok] &= rows_mask[row[ok]]
    out = []
    for p, label in (("teff", "teff_anchor"), ("logg", "logg_anchor")):
        k = params.index(p)
        m = ok & is_anchor[survey] & np.isfinite(value[:, k])
        if m.any():
            out.append((label, PARAMS.index(p), row[m], value[m, k]))
        for s in np.unique(survey[m]):
            ms = m & (survey == s)
            out.append((f"{label}:{names[s]}", PARAMS.index(p), row[ms], value[ms, k]))
    return out


def coverage_table(theta, G, covered, nominal):
    """Binned coverage per Teff, log g, [Fe/H] and G (Wilson intervals)."""
    import torch
    from torchregress.inference import binned_conditional_coverage

    feats = np.column_stack([theta, G])
    ok = np.isfinite(feats).all(1)
    if ok.sum() < 2:
        return {}
    edges = [torch.tensor(COVERAGE_BINS[k], dtype=torch.float64)
             for k in ("teff", "logg", "fe_h", "G")]  # fmt: skip
    res = binned_conditional_coverage(
        torch.as_tensor(feats[ok], dtype=torch.float64),
        torch.as_tensor(np.asarray(covered)[ok], dtype=torch.float64),
        nominal=nominal, bins=edges,
    )  # fmt: skip
    out = {}
    for name, r in zip(("teff", "logg", "fe_h", "G"), res, strict=True):
        out[name] = dict(
            edges=r.edges.tolist(), count=r.count.tolist(),
            coverage=[None if math.isnan(c) else c for c in r.coverage.tolist()],
            lower=[None if math.isnan(c) else c for c in r.lower.tolist()],
            upper=[None if math.isnan(c) else c for c in r.upper.tolist()],
        )  # fmt: skip
    out["all"] = dict(n=int(ok.sum()), coverage=float(np.mean(np.asarray(covered)[ok])))
    return out


def selection_table(ref, G, selected):
    """Purity / completeness of [Fe/H] < -2.5 (all G, G <= 16, G > 16)."""
    out = {}
    truth = ref < MP_LIMIT
    for gname, gm in (("all", np.ones(len(G), bool)), ("G<=16", G <= G_MAX),
                      ("G>16", G > G_MAX)):  # fmt: skip
        out[gname] = {}
        for name, sel in selected.items():
            sel = np.asarray(sel, bool) & gm
            tp = int((sel & truth).sum())
            n_sel, n_true = int(sel.sum()), int((truth & gm).sum())
            out[gname][name] = dict(
                n_selected=n_sel, n_true=n_true, true_positive=tp,
                purity=tp / n_sel if n_sel else math.nan,
                completeness=tp / n_true if n_true else math.nan,
            )  # fmt: skip
    return out


def decide(core_offsets, head_offsets, coverage68, anchors, est, head="posterior"):
    """Apply the §7 decision rule to one likelihood estimator; see module doc."""
    crit = {}
    fails = [
        r for r in core_offsets
        if r["g"] == "G<=16" and r["n"] >= MIN_N and abs(r["bias"]) > BIAS_LIMIT
    ]  # fmt: skip
    crit["bias"] = dict(
        passed=not fails,
        worst=max((abs(r["bias"]) for r in core_offsets
                   if r["g"] == "G<=16" and r["n"] >= MIN_N), default=math.nan),
        failing_bins=[(r["feh_lo"], r["feh_hi"], r["bias"], r["n"]) for r in fails],
    )  # fmt: skip
    bad = []
    for name, tab in coverage68.items():
        if name == "all":
            continue
        for lo, hi, n, c in zip(tab["edges"][:-1], tab["edges"][1:], tab["count"],
                                tab["coverage"], strict=True):  # fmt: skip
            if n >= MIN_N and c is not None and abs(c - 0.68) > COVERAGE_TOL:
                bad.append((name, lo, hi, n, c))
    crit["coverage"] = dict(passed=bool(coverage68) and not bad, failing_bins=bad)
    bulk = [r for r in core_offsets if r["g"] == "G<=16" and r["feh_lo"] >= -1.0]
    hbulk = [r for r in head_offsets if r["g"] == "G<=16" and r["feh_lo"] >= -1.0]
    crit["bulk"] = dict(
        passed=bool(bulk) and bulk[0]["nmad"] <= hbulk[0]["nmad"] + BULK_TOL,
        nmad=bulk[0]["nmad"] if bulk else math.nan,
        nmad_head=hbulk[0]["nmad"] if hbulk else math.nan,
    )  # fmt: skip
    ok_anchor, detail = True, {}
    for key, p in (("teff_anchor", "teff"), ("logg_anchor", "logg")):
        if key not in anchors:
            continue
        a, h = anchors[key][est], anchors[key][head]
        good = (a["nmad"] <= ANCHOR_NMAD_FACTOR * h["nmad"]
                and abs(a["bias"]) <= abs(h["bias"]) + ANCHOR_TOL[p])  # fmt: skip
        ok_anchor &= bool(good)
        detail[key] = dict(passed=bool(good), mle=a, head=h)
    crit["anchors"] = dict(passed=bool(ok_anchor and detail), detail=detail)
    return dict(criteria=crit, adopt=all(c["passed"] for c in crit.values()))


def _fmt(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def render_report(m):
    """Markdown report from the metrics dict written by ``s1_likelihood_head``."""
    L = []
    L.append("# S1: likelihood head on frozen latents\n")
    L.append(m.get("summary", "") + "\n")
    L.append("## Decision (design note §7)\n")
    for arm, d in m["decision"].items():
        verdict = "ADOPT" if d["adopt"] else "DO NOT ADOPT"
        L.append(f"**Arm `{arm}`: {verdict}** the likelihood product.\n")
        c = d["criteria"]
        L.append("| criterion | passed | detail |\n|---|---|---|")
        L.append(f"| bias, every [Fe/H] bin, G <= 16 | {c['bias']['passed']} | worst "
                 f"\\|bias\\| {_fmt(c['bias']['worst'])}; failing "
                 f"{c['bias']['failing_bins']} |")  # fmt: skip
        L.append(f"| Neyman 68 % coverage 0.68 +- 0.08, every bin, G <= 16 | "
                 f"{c['coverage']['passed']} | failing "
                 f"{[(b[0], b[1], b[2], b[3], round(b[4], 3)) for b in c['coverage']['failing_bins']]} |")  # fmt: skip
        L.append(
            f"| bulk NMAD <= head + 0.01 | {c['bulk']['passed']} | "
            f"{_fmt(c['bulk']['nmad'])} vs head {_fmt(c['bulk']['nmad_head'])} |"
        )
        det = "; ".join(
            f"{k}: MLE {_fmt(v['mle']['bias'], 3, True)}/{_fmt(v['mle']['nmad'])}, head "
            f"{_fmt(v['head']['bias'], 3, True)}/{_fmt(v['head']['nmad'])}"
            for k, v in c["anchors"]["detail"].items()
        )
        L.append(f"| anchors no worse than head | {c['anchors']['passed']} | {det} |\n")
    L.append(
        "## Frozen held-out core: median offset / NMAD (n) per reference [Fe/H] bin\n"
    )
    cols = list(G_NAMES) + ["G<=16", "all"]
    for est, tab in m["frozen_core"].items():
        L.append(f"**{est}**\n")
        L.append("| [Fe/H]ref | " + " | ".join(cols) + " |")
        L.append("|---" * (len(cols) + 1) + "|")
        for lo, hi in zip(FEH_BINS[:-1], FEH_BINS[1:], strict=True):
            cells = []
            for g in cols:
                r = next(r for r in tab if r["feh_lo"] == lo and r["g"] == g)
                cells.append(f"{_fmt(r['bias'], 2, True)}/{_fmt(r['nmad'], 2)} ({r['n']})"
                             if r["n"] else "-")  # fmt: skip
            L.append(f"| [{lo:g}, {hi:g}) | " + " | ".join(cells) + " |")
        L.append("")
    if m.get("precision"):
        L.append("## Precision summary (MLE scatter and raw Wilks coverage)\n")
        L.append("Frozen held-out core NMAD per reference [Fe/H] bin (G <= 16), anchor "
                 "and test-label NMAD, and coverage of the test labels (G <= 16). "
                 "`mle_<arm>_single`: member 0 of the arm's ensemble alone.\n")  # fmt: skip
        ests = list(m["precision"])
        names = list(dict.fromkeys(k for v in m["precision"].values() for k in v))
        L.append("| metric | " + " | ".join(ests) + " |")
        L.append("|---" * (len(ests) + 1) + "|")
        for name in names:
            nd = 0 if name.startswith(("Teff", "test teff")) else 3
            cells = [_fmt(m["precision"][e].get(name), nd) for e in ests]
            L.append(f"| {name} | " + " | ".join(cells) + " |")
        L.append("")
    if m.get("edges"):
        L.append("## MLE at the edges and left out (fractions of rows)\n")
        L.append("left out: no finite score inside the label support; box: MLE on a "
                 "box bound (any of Teff, log g, [Fe/H]); support edge: MLE in a "
                 "support cell next to its boundary; window edge: parallax at its "
                 "search-window edge; pull > 3: |parallax MLE - measured| > 3 "
                 "errors.\n")  # fmt: skip
        cols = [("left_out", "left out"), ("at_box_bound", "box"),
                ("at_bound_teff", "box Teff"), ("at_bound_logg", "box log g"),
                ("at_bound_fe_h", "box [Fe/H]"), ("support_edge", "support edge"),
                ("parallax_window_edge", "window edge"),
                ("parallax_pull_gt3", "pull > 3"),
                ("parallax_measured", "parallax measured")]  # fmt: skip
        L.append("| arm | rows | n | " + " | ".join(t for _, t in cols) + " |")
        L.append("|---" * (len(cols) + 3) + "|")
        for arm, groups in m["edges"].items():
            for g, d in groups.items():
                cells = [_fmt(d.get(k), 4) for k, _ in cols]
                L.append(f"| {arm} | {g} | {d['n']} | " + " | ".join(cells) + " |")
        L.append("")
    L.append("## Anchors (test stars): median offset / NMAD (n)\n")
    ests = sorted({e for v in m["anchors"].values() for e in v})
    L.append("| anchor | " + " | ".join(ests) + " |")
    L.append("|---" * (len(ests) + 1) + "|")
    for name, v in m["anchors"].items():
        nd = 0 if name.startswith("teff") else 3
        cells = [f"{_fmt(v[e]['bias'], nd, True)}/{_fmt(v[e]['nmad'], nd)} ({v[e]['n']})"
                 if e in v else "-" for e in ests]  # fmt: skip
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("## Coverage (test stars, labels of each arm)\n")
    L.append(
        "'Neyman 68/95': the joint (Teff, log g, [Fe/H]) set; 'Neyman 68 <p>': "
        "the per-parameter Neyman interval; 'Wilks 68 joint': raw joint set "
        "(lambda <= 1.763, chi2_3); 'Wilks 68 <p>': raw profile set (lambda_j <= "
        "0.5); 'profile 68': the Wilks (0.5) profile interval (profiled rows).\n"
    )
    L.append("Labels are noisy measurements: coverage of the label is approximate. "
             "Critical values were calibrated on the valid split's labels, which carry "
             "the same noise, so the Neyman sets include it (conservative for the "
             "true parameters).\n")  # fmt: skip
    kinds = [("neyman68", "Neyman 68"), ("neyman95", "Neyman 95"),
             ("neyman68_fe_h", "Neyman 68 [Fe/H]"), ("neyman68_teff", "Neyman 68 Teff"),
             ("neyman68_logg", "Neyman 68 log g"), ("wilks68_joint", "Wilks 68 joint"),
             ("wilks68_fe_h", "Wilks 68 [Fe/H]"), ("wilks68_teff", "Wilks 68 Teff"),
             ("wilks68_logg", "Wilks 68 log g"),
             ("profile68_fe_h", "profile 68 [Fe/H]"), ("posterior68_fe_h", "head 16-84 [Fe/H]"),
             ("profile68_teff", "profile 68 Teff"), ("posterior68_teff", "head 16-84 Teff"),
             ("profile68_logg", "profile 68 log g"), ("posterior68_logg", "head 16-84 log g")]  # fmt: skip
    for arm, cov in m["coverage"].items():
        have = [(k, t) for k, t in kinds if cov.get(k)]
        if not have:
            continue
        L.append(f"**Arm `{arm}`**, G <= 16, n = {cov[have[0][0]]['all']['n']}; "
                 "all: " + ", ".join(f"{t} {_fmt(cov[k]['all']['coverage'], 2)}"
                                      for k, t in have) + "\n")  # fmt: skip
        for by in ("fe_h", "G", "teff", "logg"):
            ref = cov[have[0][0]][by]
            L.append(f"| {by} bin | n | " + " | ".join(t for _, t in have) + " |")
            L.append("|---" * (len(have) + 2) + "|")
            for i, (lo, hi) in enumerate(zip(ref["edges"][:-1], ref["edges"][1:],
                                             strict=True)):  # fmt: skip
                if not ref["count"][i]:
                    continue
                cells = [_fmt(cov[k][by]["coverage"][i], 2) for k, _ in have]
                L.append(f"| [{lo:g}, {hi:g}) | {ref['count'][i]} | " + " | ".join(cells)
                         + " |")  # fmt: skip
            L.append("")
    L.append("## Metal-poor selection ([Fe/H]ref < -2.5, held-out core)\n")
    for g, tab in m["selection"].items():
        L.append(f"**{g}**\n")
        L.append(
            "| method | selected | true MP | purity | completeness |\n|---|---|---|---|---|"
        )
        for name, r in tab.items():
            L.append(f"| {name} | {r['n_selected']} | {r['n_true']} | {_fmt(r['purity'], 2)}"
                     f" | {_fmt(r['completeness'], 2)} |")  # fmt: skip
        L.append("")
    if m.get("test_labels"):
        L.append("## Test split vs mapped labels: median offset / NMAD (n)\n")
        for est, tab in m["test_labels"].items():
            L.append(f"**{est}**: " + "; ".join(
                f"{p} {g}: {_fmt(r['bias'], 0 if p == 'teff' else 3, True)}/"
                f"{_fmt(r['nmad'], 0 if p == 'teff' else 3)} ({r['n']})"
                for p, rows in tab.items() for g, r in rows.items()
            ))  # fmt: skip
        L.append("")
    if m.get("notes"):
        L.append("## Notes\n")
        L.extend(f"- {n}" for n in m["notes"])
        L.append("")
    return "\n".join(L) + "\n"
