"""metrics.json, report.md and plots of ``scripts/evaluate_model.py``."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .predictions import UNITS
from .stats import STRATA


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _f(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _nd(p):
    return 0 if p == "teff" else 3


def _cell(s, p):
    if not s or not s.get("n"):
        return "–"
    star = "*" if s.get("descriptive") else ""
    return f"{_f(s['bias'], _nd(p), True)} / {_f(s['nmad'], _nd(p))} ({s['n']}){star}"


def _fails(cov) -> str:
    """Strata (n >= 20) outside nominal ± 0.03, counted per axis."""
    fails = cov["failing_strata"]
    if not fails:
        return "every stratum with n ≥ 20 within ±0.03"
    total = sum(
        sum(not r["descriptive"] for r in rows) for rows in cov["by_stratum"].values()
    )
    axes = {}
    for f in fails:
        axes.setdefault(f.split(":")[0], []).append(f.split(":", 1)[1])
    return f"{len(fails)} of {total} strata (n ≥ 20) outside ±0.03 (" + "; ".join(
        f"{a}: {', '.join(v)}" for a, v in axes.items()) + ")"  # fmt: skip


def _headline(m) -> list[str]:
    L = ["## Headline", "",
         "| set | product | n | bias | NMAD | 68 % cov | 95 % cov | strata failing 68 % |",
         "|---|---|--:|--:|--:|--:|--:|--:|"]  # fmt: skip
    for name, res in m["sets"].items():
        for p, h in res.get("heldout", {}).items():
            c68 = h.get("coverage", {}).get("68", {})
            c95 = h.get("coverage", {}).get("95", {})
            o = h["overall"]
            L.append(f"| {name} | {p} | {o['n']} | {_f(o['bias'], _nd(p), True)} | {_f(o['nmad'], _nd(p))} "
                     f"| {_f(c68.get('overall'), 3)} | {_f(c95.get('overall'), 3)} | {len(c68.get('failing_strata', []))} |")  # fmt: skip
        for key, sp in res.get("supplied_pit", {}).items():
            L.append(
                f"| {name} | {key} | {sp['n']} | | | {_f(sp['coverage'], 3)} | | {len(sp['failing_strata'])} |"
            )
    L.append("")
    for name, core in m.get("core", {}).items():
        w = core["worst_bias_G_le_16"]
        L.append(f"- core `{name}`: worst |bias| (G ≤ 16, n ≥ 20) {_f(w['bias'], 3, True)} in {w['bin']} "
                 f"({core['with_reference']} TEST rows with a core reference)")  # fmt: skip
    c = m.get("conatal")
    if c:
        wb = c["wide_binaries"]
        L.append(f"- wide binaries: Δ[Fe/H] {_f(wb['all']['median'], 3, True)} / NMAD {_f(wb['all']['nmad'])} "
                 f"({wb['pairs_scored']} pairs), |z| < 1 {_f(wb['all'].get('abs_z_lt_1'), 2)}")  # fmt: skip
        oc = c["open_clusters"]
        if oc.get("clusters"):
            L.append(f"- open clusters: giant − dwarf {_f(oc['giant_minus_dwarf'], 3, True)} "
                     f"({oc['giant_minus_dwarf_clusters']} clusters), median χ²/dof {_f(oc['median_chi2_dof'], 2)}")  # fmt: skip
        if "binary_pool_heldout" in c:
            bp = c["binary_pool_heldout"]["by_companion_teff"]
            L.append("- binary pool, primary − companion: " + "; ".join(
                f"{k} K {_f(v['median'], 3, True)}" for k, v in bp.items() if v["n"]))  # fmt: skip
    return L + [""]


def _md_set(name, res) -> list[str]:
    L = [f"### {name} (`{res['model']}`, kind {res['kind'] or '?'})", ""]
    r = res["rows"]
    L.append(
        "Rows: " + ", ".join(f"{k} {v}" for k, v in r.items()) + ". Integrity: "
        f"{res['integrity']['duplicate_source_ids']} duplicate ids; non-monotone quantiles "
        + ", ".join(
            f"{p} {v['non_monotone']}" for p, v in res["integrity"]["products"].items()
        )
        + "."
    )
    if res.get("reference_source"):
        L.append(
            "References: "
            + ", ".join(f"{p} from {v}" for p, v in res["reference_source"].items())
            + "."
        )
    L.append("")
    for p, h in res.get("heldout", {}).items():
        L.append(f"**{p}** ({UNITS.get(p, '')}) vs reference: bias / NMAD (n) overall "
                 f"{_cell(h['overall'], p)}; pull NMAD {_f(h.get('pull', {}).get('nmad'), 2)}"
                 + (f"; PIT KS {_f(h['pit']['pit_ks'], 3)}" if h.get("pit") else ""))  # fmt: skip
        for mass, cov in h.get("coverage", {}).items():
            L.append(f"- {mass} % coverage ({cov['method']}): {_f(cov['overall'], 3)}; "
                     f"{_fails(cov)}")  # fmt: skip
        for axis in ("G", "feh_ref", "stage", "input_class", "ruwe", "abs_b"):
            cells = h["by_stratum"].get(axis)
            if not cells:
                continue
            row = "; ".join(f"{s}: {_cell(c, p)}" for s, c in cells.items())
            L.append(f"- by {axis}: {row}")
        L.append("")
    for key, sp in res.get("supplied_pit", {}).items():
        L.append(
            f"**{key}**: 68 % coverage {_f(sp['coverage'], 3)} (n {sp['n']}); {_fails(sp)}"
        )
        g = sp["by_stratum"].get("G", [])
        if g:
            L.append("- by G: " + "; ".join(f"{c['stratum']}: {_f(c['coverage'], 3)} "
                                             f"[{_f(c['lower'], 3)}, {_f(c['upper'], 3)}] ({c['n']})"
                                             for c in g))  # fmt: skip
        L.append("")
    if "irfm" in res:
        L.append(
            f"**IRFM Teff**: {_cell(res['irfm'], 'teff')} (training anchor: tests transfer)"
        )
        L.append("")
    if res.get("physical"):
        L.append("**Physical references** (bias / NMAD (n); TEST and reserved stars):")
        for survey, cell in res["physical"].items():
            parts = [
                f"{p} {_cell(cell[p], p)}"
                for p in ("teff", "logg", "fe_h")
                if p in cell
            ]
            L.append(f"- {survey}: {cell['predicted']} of {cell['references']} predicted; "
                     + ("; ".join(parts) if parts else "no overlap"))  # fmt: skip
        L.append("")
    if res.get("competitors"):
        L.append(
            "**Competitors** (exact-id overlap; model vs competitor against the same reference):"
        )
        L.append("")
        L.append(
            "| catalogue | overlap | product | model bias / NMAD | competitor bias / NMAD | Δ median abs err (95 %) |"
        )
        L.append("|---|--:|---|---|---|---|")
        for cname, cr in res["competitors"].items():
            for p in ("teff", "logg", "fe_h", "ebv", "dist_mod"):
                if p not in cr:
                    continue
                c = cr[p]
                if "model" in c:
                    d = c["median_abs_err_model_minus_competitor"]
                    L.append(f"| {cname} | {cr['overlap']} | {p} | {_cell(c['model'], p)} | "
                             f"{_cell(c['competitor'], p)} | {_f(d['delta'], _nd(p), True)} "
                             f"[{_f(d['lo'], _nd(p), True)}, {_f(d['hi'], _nd(p), True)}] |")  # fmt: skip
                else:
                    L.append(f"| {cname} | {cr['overlap']} | {p} | model − competitor "
                             f"{_cell(c['model_minus_competitor'], p)} | | |")  # fmt: skip
        L.append("")
    return L


def _md_core(name, core) -> list[str]:
    L = [f"### Core: set `{name}`", "",
         f"Rows with a core reference: {core['with_reference']}; rows: "
         + ", ".join(f"{k} {v}" for k, v in core["rows"].items()) + ".", "",
         "Median offset / NMAD (n) of [Fe/H] − core reference, per [Fe/H]_ref × G "
         "(* = n < 20, descriptive):", "",
         "| [Fe/H]_ref | G ≤ 13.5 | 13.5–15 | 15–16 | 16–17.65 | G ≤ 16 |", "|---|---|---|---|---|---|"]  # fmt: skip
    for b, cells in core["table"].items():
        L.append(f"| {b} | " + " | ".join(_cell(cells[g], "fe_h") for g in
                                       ("G<=13.5", "13.5-15", "15-16", "16-17.65", "G<=16")) + " |")  # fmt: skip
    w = core["worst_bias_G_le_16"]
    L += ["", f"Worst |bias| at G ≤ 16 (n ≥ 20): {_f(w['bias'], 3, True)} in {w['bin']} "
          f"(acceptance ≤ 0.10, qa.md §5.3).", ""]  # fmt: skip
    s = core["selection_p_gt_0.9"]
    L.append("Metal-poor selection P([Fe/H] < −2.5) > 0.9: " + "; ".join(
        f"{k}: purity {_f(v['purity'], 2)}, completeness {_f(v['completeness'], 2)} "
        f"({v['true_positives']}/{v['n_true']})" for k, v in s.items()))  # fmt: skip
    t = core["tail_probability"]
    L.append(f"Tail probability: n {t['n']} ({t['n_true']} true), Brier {_f(t.get('brier'), 4)}, "
             f"AUC {_f(t.get('auc'), 3)}.")  # fmt: skip
    if "coverage68_feh_ref_lt_m2" in core:
        c = core["coverage68_feh_ref_lt_m2"]
        L.append(
            f"68 % coverage at [Fe/H]_ref < −2: {_f(c['coverage'], 3)} (n {c['n']})."
        )
    return L + [""]


def _md_conatal(c) -> list[str]:
    L = ["## Co-natal checks", ""]
    wb = c["wide_binaries"]
    L += [f"**Wide binaries** (El-Badry+2021 MSMS, R_chance < {wb['r_chance_max']}, pair group in "
          f"the evaluated partition): {wb['pairs_scored']} pairs scored "
          f"({wb['pairs_neither_labelled']} with neither star labelled). Δ = [Fe/H](brighter) − "
          "[Fe/H](fainter): median / NMAD (n); |z| < 1 fraction (ideal 0.68).", "",
          "| subset | all pairs | |z|<1 | neither labelled | |z|<1 |", "|---|---|--:|---|--:|"]  # fmt: skip

    def row(label, d):
        a, u = d["all"], d.get("neither_labelled", {})
        return (f"| {label} | {_f(a.get('median'), 3, True)} / {_f(a.get('nmad'))} ({a.get('n')}) | "
                f"{_f(a.get('abs_z_lt_1'), 2)} | {_f(u.get('median'), 3, True)} / {_f(u.get('nmad'))} "
                f"({u.get('n')}) | {_f(u.get('abs_z_lt_1'), 2)} |")  # fmt: skip

    L.append(row("all", dict(all=wb["all"], neither_labelled=wb["neither_labelled"])))
    for k, v in wb["by_pair_mean"].items():
        L.append(row(f"pair mean {k}", v))
    for k, v in wb["by_faint_teff"].items():
        L.append(row(f"fainter Teff {k}", v))
    for k, v in wb["by_dG"].items():
        L.append(row(f"ΔG {k}", v))
    L.append("")
    if "binary_pool_heldout" in c:
        bp = c["binary_pool_heldout"]
        L.append(f"**Held-out binary pool** ({bp['scored']} of {bp['pairs']} pairs scored): median "
                 "[Fe/H](primary) − [Fe/H](companion) by companion Teff: " + "; ".join(
                     f"{k} K {_f(v['median'], 3, True)} ({v['n']})" for k, v in bp["by_companion_teff"].items())
                 + f"; companion G > 16.5: {_f(bp['companion_G_gt_16p5']['median'], 3, True)} "
                 f"({bp['companion_G_gt_16p5']['n']}).")  # fmt: skip
        L.append("")
    oc = c["open_clusters"]
    if oc.get("clusters"):
        L.append(f"**Open clusters** (HR23, Prob ≥ 0.7, ≥ 20 members, cluster in the evaluated "
                 f"partition): {oc['clusters']} clusters ({oc['members']} members). Median within-cluster "
                 f"NMAD {_f(oc['median_nmad'])}, median χ²/dof {_f(oc['median_chi2_dof'], 2)}, giant − dwarf "
                 f"{_f(oc['giant_minus_dwarf'], 3, True)} ({oc['giant_minus_dwarf_clusters']} clusters), "
                 f"faint − bright {_f(oc['faint_minus_bright'], 3, True)}, slope "
                 f"{_f(oc['slope_per_mag'], 3, True)} dex/mag.")  # fmt: skip
    else:
        L.append(
            "**Open clusters**: none in the evaluated partition with ≥ 20 predicted members."
        )
    gc = c["globular_clusters"]
    L.append("")
    L.append(f"**Globular clusters** (QA-reserved, report-only; ≥ 10 members): {gc['clusters']} "
             f"clusters, {gc.get('with_harris', 0)} with Harris [Fe/H]. Median(pred − Harris): " + "; ".join(
                 f"{k}: {_f(v['pred_minus_harris'], 3, True)} ({v['n']} GCs, within NMAD "
                 f"{_f(v['within_nmad'])})" for k, v in gc.get("by_harris", {}).items()))  # fmt: skip
    return L + [""]


def write_markdown(path, m) -> None:
    L = [f"# {m['title']}", "",
         f"Evaluator `scripts/evaluate_model.py` (commit `{m['code_commit'][:10]}`), "
         f"{m['created_utc'][:19]} UTC. Split `{m['spec']}` ({m['split_dir']}); partition "
         f"**{m['partition']}** only (QA-reserved objects report-only)"
         + (f"; rows with table split in {m['exclude_table_splits']} dropped (seen by the model)"
            if m["exclude_table_splits"] else "") + ".", "",
         "Conventions (qa.md §0): bias = median(prediction − reference), NMAD = 1.4826 MAD, "
         "coverage of the (noisy) reference with Wilson 95 % Bonferroni bands, cells with n < 20 "
         "descriptive (*). Teff in K, others in dex or mag.", ""]  # fmt: skip
    L += _headline(m)
    L += ["## Held-out labels and calibration", ""]
    for name, res in m["sets"].items():
        L += _md_set(name, res)
    if "core" in m:
        L += ["## Frozen metal-poor core", ""]
        for name, core in m["core"].items():
            L += _md_core(name, core)
    if "conatal" in m:
        L += _md_conatal(m["conatal"])
    L += ["## Not run / missing inputs", ""]
    L += [f"- {k}: {v}" for k, v in m["missing"].items()]
    Path(path).write_text("\n".join(L) + "\n")


def write_plots(out_dir, m, sets, tables) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir) / "plots"
    made = []
    labels = STRATA["G"][1]
    for name, res in m["sets"].items():
        hp = [p for p in ("teff", "logg", "fe_h", "ebv") if p in res.get("heldout", {})]
        if not hp:
            continue
        fig, axes = plt.subplots(2, len(hp), figsize=(3.6 * len(hp), 6), squeeze=False)
        for j, p in enumerate(hp):
            h = res["heldout"][p]
            cells = h["by_stratum"].get("G", {})
            xs = [k for k, s in enumerate(labels) if s in cells]
            b = [cells[labels[k]]["bias"] for k in xs]
            nm = [cells[labels[k]]["nmad"] for k in xs]
            ax = axes[0, j]
            ax.errorbar(xs, b, yerr=nm, fmt="o", capsize=3)
            ax.axhline(0, color="0.6", lw=0.8)
            ax.set_xticks(range(len(labels)), labels, rotation=45, fontsize=7)
            ax.set_title(f"{p}: bias ± NMAD by G", fontsize=9)
            ax = axes[1, j]
            cov = h.get("coverage", {}).get("68", {}).get("by_stratum", {}).get("G", [])
            if cov:
                x = [labels.index(c["stratum"]) for c in cov]
                c = [c_["coverage"] for c_ in cov]
                lo = [c_["coverage"] - c_["lower"] for c_ in cov]
                hi = [c_["upper"] - c_["coverage"] for c_ in cov]
                ax.errorbar(x, c, yerr=[lo, hi], fmt="s", capsize=3)
                ax.axhspan(0.65, 0.71, color="0.9")
                ax.axhline(0.68, color="0.5", lw=0.8)
                ax.set_ylim(0, 1)
                ax.set_xticks(range(len(labels)), labels, rotation=45, fontsize=7)
                ax.set_title(f"{p}: 68 % coverage (Wilson)", fontsize=9)
        fig.suptitle(name, fontsize=10)
        fig.tight_layout()
        f = out_dir / f"qa_{name}_by_G.png"
        fig.savefig(f, dpi=110)
        plt.close(fig)
        made.append(f.name)
        sub, refs, _ = sets[name]
        from .stats import pit_from_quantiles

        if sub.kind == "A":
            continue
        fig, axes = plt.subplots(
            1, len(hp), figsize=(3.2 * len(hp), 2.8), squeeze=False
        )
        for j, p in enumerate(hp):
            pit = pit_from_quantiles(sub.levels, sub.q[p], refs[p])
            axes[0, j].hist(pit[np.isfinite(pit)], bins=20, range=(0, 1), color="C0")
            axes[0, j].set_title(f"{p} PIT", fontsize=9)
        fig.tight_layout()
        f = out_dir / f"qa_{name}_pit.png"
        fig.savefig(f, dpi=110)
        plt.close(fig)
        made.append(f.name)
    for name, core in m.get("core", {}).items():
        t = core["table"]
        gn = ("G<=13.5", "13.5-15", "15-16", "16-17.65")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for k, g in enumerate(gn):
            xs = [i for i, b in enumerate(t) if t[b][g]["n"] >= 5]
            ax.errorbar(np.array(xs) + 0.08 * k, [t[list(t)[i]][g]["bias"] for i in xs],
                        yerr=[t[list(t)[i]][g]["nmad"] for i in xs], fmt="o", label=g, capsize=2)  # fmt: skip
        ax.axhspan(-0.1, 0.1, color="0.92")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(range(len(t)), list(t), rotation=30, fontsize=7)
        ax.set_ylabel("[Fe/H] − core ref")
        ax.set_title(f"frozen core: {name} (cells with n ≥ 5)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        f = out_dir / f"qa_core_{name}.png"
        fig.savefig(f, dpi=110)
        plt.close(fig)
        made.append(f.name)
    if "conatal" in m:
        wb = m["conatal"]["wide_binaries"]["by_faint_teff"]
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        ks = list(wb)
        for key, lab in (
            ("all", "all pairs"),
            ("neither_labelled", "neither labelled"),
        ):
            y = [wb[k][key].get("median", np.nan) for k in ks]
            axes[0].plot(range(len(ks)), y, "o-", label=lab)
        axes[0].axhline(0, color="0.5", lw=0.8)
        axes[0].set_xticks(range(len(ks)), ks, rotation=30, fontsize=7)
        axes[0].set_title("wide binaries: Δ[Fe/H] by fainter Teff", fontsize=9)
        axes[0].legend(fontsize=7)
        oc = tables.get("open")
        if oc is not None and len(oc):
            gd = oc.loc[oc.n >= 20, "giant_minus_dwarf"].dropna()
            axes[1].hist(gd, bins=30, range=(-1, 1))
            axes[1].axvline(0, color="0.5")
            axes[1].set_title(f"open clusters: giant − dwarf ({len(gd)})", fontsize=9)
        fig.tight_layout()
        f = out_dir / "qa_conatal.png"
        fig.savefig(f, dpi=110)
        plt.close(fig)
        made.append(f.name)
    return made


def write_outputs(out_dir, metrics, sets, tables) -> None:
    out_dir = Path(out_dir)
    metrics["plots"] = write_plots(out_dir, metrics, sets, tables)
    for key, t in (tables or {}).items():
        if t is not None and len(t):
            t.to_csv(out_dir / f"clusters_{key}.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(_clean(metrics), indent=1) + "\n")
    write_markdown(out_dir / "report.md", _clean(metrics))
