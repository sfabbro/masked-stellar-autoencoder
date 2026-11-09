"""Frozen export diagnostics: survey references, mixtures and astrometric GC members.

No torch import, checkpoint execution, fitting, head selection or recalibration.
"""

import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/msa-validation-mpl")
import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from masked_stellar_autoencoder.training.export_mixture_diagnostics import (
    marginal_scores,
    summarize_residuals,
    summarize_scores,
)

TARGETS = ["Teff", "logg", "FeH"]
SURVEYS = ["APOGEE", "DESI", "GALAH"]


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def nmad(x):
    return float(1.4826 * np.median(abs(x - np.median(x))))


def join(ids, other):
    """Exact integer ID join; caller owns reference deduplication."""
    ids, other = np.asarray(ids), np.asarray(other)
    if ids.dtype.kind not in "iu" or other.dtype.kind not in "iu":
        raise ValueError("Source IDs must remain integers")
    idx = np.searchsorted(ids, other)
    ok = (idx < len(ids)) & (ids[np.minimum(idx, len(ids) - 1)] == other)
    return idx[ok], np.flatnonzero(ok)


def compare(data, rows, reference, errors, mask, name, output, *, score=True):
    rows = np.asarray(rows)
    result = {}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for t in range(3):
        keep = mask[:, t] & np.isfinite(reference[:, t])
        if not keep.any():
            axes[0, t].set_title(TARGETS[t] + ": unavailable")
            continue
        rr = rows[keep]
        y = reference[keep, t]
        pred = data["mean"][rr, t]
        result[TARGETS[t]] = summarize_residuals(pred, y)
        # Population diagnostics defined by reference quantities, not predictions.
        groups = {
            "cool": reference[:, 0] < 4500,
            "hot": reference[:, 0] > 6500,
            "giants": reference[:, 1] < 3,
            "dwarfs": reference[:, 1] > 3.8,
            "metalpoor": reference[:, 2] < -1.5,
        }
        result[TARGETS[t]]["subgroups"] = {}
        for label, group in groups.items():
            k = group[keep]
            if k.sum() >= 20:
                result[TARGETS[t]]["subgroups"][label] = summarize_residuals(
                    pred[k], y[k]
                )
        ax = axes[0, t]
        ax.hexbin(y, pred, gridsize=70, bins="log", mincnt=1)
        lo, hi = np.quantile(y, [0.005, 0.995])
        ax.plot([lo, hi], [lo, hi], color="tomato", lw=1)
        ax.set(
            xlabel=f"Reference {TARGETS[t]}",
            ylabel="Common predictive mean",
            title=f"{name}: {len(y):,} stars",
        )
        ax.text(
            0.02,
            0.97,
            f"bias {np.median(pred - y):.3g}\nNMAD {nmad(pred - y):.3g}",
            va="top",
            transform=ax.transAxes,
        )
        if score:
            v = (data["component_lower"][rr, :, t, :] ** 2).sum(-1)
            s = marginal_scores(
                data["weights"][rr], data["component_mean"][rr, :, t], v, y
            )
            result[TARGETS[t]]["intrinsic_reference_comparison"] = summarize_scores(s)
            axes[1, t].hist(
                s["pit"],
                bins=np.linspace(0, 1, 21),
                density=True,
                histtype="step",
                label="Intrinsic distribution",
            )
            e = errors[keep, t]
            usable = np.isfinite(e) & (e >= 0)
            if usable.any():
                se = marginal_scores(
                    data["weights"][rr[usable]],
                    data["component_mean"][rr[usable], :, t],
                    v[usable],
                    y[usable],
                    e[usable],
                )
                result[TARGETS[t]]["gaussian_reference_error_proxy"] = summarize_scores(
                    se
                )
                axes[1, t].hist(
                    se["pit"],
                    bins=np.linspace(0, 1, 21),
                    density=True,
                    histtype="step",
                    label="+ reference error proxy",
                )
            axes[1, t].axhline(1, color="grey", ls="--")
            axes[1, t].set(xlabel="Full-mixture marginal PIT", ylabel="Density")
            axes[1, t].legend(fontsize=7)
        else:
            axes[1, t].hexbin(y, pred - y, gridsize=70, bins="log", mincnt=1)
            axes[1, t].axhline(0, color="tomato")
            axes[1, t].set(
                xlabel=f"Reference {TARGETS[t]}", ylabel="Common mean − native label"
            )
    fig.suptitle(
        name + "\nReference-scale comparison; catalogue values are not absolute truth"
    )
    fig.savefig(output / (name + ".png"), dpi=150)
    plt.close(fig)
    return result


def conditional_width(values, g):
    # Shared independent photometric bins; never bin by predicted parameters.
    residuals = []
    for b in np.unique(np.floor(g * 2)):
        k = np.floor(g * 2) == b
        if k.sum() >= 5:
            residuals.extend(values[k] - np.median(values[k]))
    return dict(
        n=len(residuals), nmad=nmad(np.asarray(residuals)) if residuals else None
    )


def clusters(data, fit_ids, archive, output):
    if digest(archive, "md5") != "3a07093be9a1fcbfb0e3c545ca175c99":
        raise ValueError("Cluster archive differs from published version 2")
    records, panels = [], []
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            if not name.startswith("catalogues/") or not name.endswith(".txt"):
                continue
            text = z.read(name).decode()
            columns = text.splitlines()[0].lstrip("#").split()
            tab = pd.read_csv(
                io.StringIO(text),
                sep=r"\s+",
                comment="#",
                names=columns,
                dtype={"source_id": np.int64},
            )
            ids = tab.source_id.to_numpy()
            quality = (tab.qflag.to_numpy(dtype=int) & 2) != 0
            member = tab.memberprob.to_numpy()
            selected = quality & (member > 0.9)
            fit_count = int(np.isin(ids[selected], fit_ids, assume_unique=False).sum())
            rr, jj = join(data["source_id"], ids)
            k = selected[jj]
            rr, jj = rr[k], jj[k]
            rec = dict(
                cluster=Path(name).stem,
                total_members=int(selected.sum()),
                fit_members=fit_count,
                devtest_members=len(rr),
                no_selected_members_in_current_fit=fit_count == 0,
                sensitivity=[],
            )
            r_all, j_all = join(data["source_id"], ids)
            for threshold in [0.5, 0.9, 0.99]:
                sel = quality[j_all] & (member[j_all] > threshold)
                r = r_all[sel]
                rec["sensitivity"].append(
                    dict(
                        probability=threshold,
                        devtest_members=len(r),
                        mean_feh_nmad=nmad(data["mean"][r, 2]) if len(r) >= 5 else None,
                    )
                )
            if len(rr) >= 5:
                rec.update(
                    feh_median=float(np.median(data["mean"][rr, 2])),
                    mean_feh_nmad=nmad(data["mean"][rr, 2]),
                    median_individual_feh_sigma=float(
                        np.median(np.sqrt(data["covariance"][rr, 2, 2]))
                    ),
                )
                rec["native_matched"] = {}
                for s, survey in enumerate(SURVEYS):
                    observed = data["native_observed"][rr, s, 2].astype(bool)
                    r = rr[observed]
                    if len(r) >= 5:
                        native = data["native_value"][r, s, 2]
                        common = data["mean"][r, 2]
                        rec["native_matched"][survey] = dict(
                            n=len(r),
                            native_feh_median=float(np.median(native)),
                            common_feh_median=float(np.median(common)),
                            native_feh_nmad=nmad(native),
                            common_feh_nmad=nmad(common),
                        )
                        if len(r) >= 20:
                            rng = np.random.default_rng(42)
                            ratios = []
                            for _ in range(500):
                                ii = rng.integers(len(r), size=len(r))
                                denom = nmad(native[ii])
                                if denom > 0:
                                    ratios.append(nmad(common[ii]) / denom)
                            if ratios:
                                rec["native_matched"][survey][
                                    "scatter_ratio_bootstrap95"
                                ] = np.quantile(ratios, [0.025, 0.975]).tolist()
                    both = data["native_observed"][rr, s, :2].all(1)
                    # Red photometric locus only; a descriptive RGB/AGB proxy.
                    both &= (tab.bp_rp.to_numpy()[jj] > 0.8) & (
                        tab.g_mag.to_numpy()[jj] < 18
                    )
                    if both.sum() >= 20:
                        r = rr[both]
                        g = tab.g_mag.to_numpy()[jj[both]]
                        widths = {}
                        for t in (0, 1):
                            widths[TARGETS[t]] = dict(
                                native=conditional_width(
                                    data["native_value"][r, s, t], g
                                ),
                                common=conditional_width(data["mean"][r, t], g),
                            )
                        rec.setdefault("conditional_sequence_width", {})[survey] = (
                            widths
                        )
                panels.append((len(rr), rec["cluster"], rr, tab.iloc[jj], fit_count))
            records.append(rec)
    pd.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if k
                not in ["sensitivity", "native_matched", "conditional_sequence_width"]
            }
            for r in records
        ]
    ).to_csv(output / "clusters.csv", index=False)
    panels.sort(reverse=True, key=lambda x: x[0])
    if panels:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
        for ax, (_, name, rr, tab, fit_count) in zip(axes.flat, panels[:6]):
            points = ax.scatter(
                data["mean"][rr, 0],
                data["mean"][rr, 1],
                c=data["mean"][rr, 2],
                vmin=-2.5,
                vmax=0.3,
                cmap="viridis",
                s=18,
            )
            ax.set(
                xlim=(7500, 3500),
                ylim=(5.5, -0.5),
                xlabel="Teff [K]",
                ylabel="log g",
                title=f"{name}: {len(rr)} test stars\n{fit_count} selected FIT members in cluster",
            )
        fig.colorbar(points, ax=list(axes.flat), label="Common predictive mean [Fe/H]")
        fig.suptitle(
            "All high-confidence devtest members: common-reference Kiel diagrams"
        )
        fig.savefig(output / "cluster-common-summary.png", dpi=150)
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
        matched = []
        for rec in records:
            for survey, stats in rec.get("native_matched", {}).items():
                matched.append((rec["cluster"], survey, stats))
        labels = [
            f"{name} / {survey} (N={stats['n']})" for name, survey, stats in matched
        ]
        xx = np.arange(len(matched))
        axes[0].barh(
            xx - 0.17,
            [stats["native_feh_nmad"] for _, _, stats in matched],
            height=0.34,
            label="Native reference",
        )
        axes[0].barh(
            xx + 0.17,
            [stats["common_feh_nmad"] for _, _, stats in matched],
            height=0.34,
            label="Common mean",
        )
        axes[0].set(
            yticks=xx, yticklabels=labels, xlabel="Matched-star [Fe/H] NMAD [dex]"
        )
        axes[0].tick_params(axis="y", labelsize=7)
        axes[0].legend(fontsize=8)
        for name, survey, stats in matched:
            axes[1].scatter(
                stats["native_feh_median"],
                stats["common_feh_median"],
                label=f"{name} {survey}",
            )
        axes[1].plot([-2.5, 0], [-2.5, 0], color="grey", ls="--")
        axes[1].set(
            xlabel="Median native [Fe/H]", ylabel="Median common predicted [Fe/H]"
        )
        fig.suptitle(
            "Cluster metallicity checks: same members, no metallicity-based selection; small N"
        )
        fig.savefig(output / "cluster-metallicity-summary.png", dpi=170)
        plt.close(fig)
    for _, name, rr, tab, fit_count in panels[:12]:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
        for s, survey in enumerate(SURVEYS):
            k = data["native_observed"][rr, s, :2].all(1)
            native = data["native_value"][rr[k], s]
            common = data["mean"][rr[k]]
            for ax, a, label in [
                (axes[0, s], native, "Native"),
                (axes[1, s], common, "Common mean"),
            ]:
                ax.scatter(
                    a[:, 0],
                    a[:, 1],
                    c=a[:, 2],
                    cmap="viridis",
                    vmin=-2.5,
                    vmax=0.3,
                    s=12,
                )
                ax.set(
                    xlim=(8000, 3500),
                    ylim=(5.5, -0.5),
                    title=f"{survey} {label}: {len(a)} stars",
                    xlabel="Teff [K]",
                    ylabel="log g",
                )
        fig.suptitle(
            f"{name}: devtest astrometric members p>0.9, quality bit set\n{len(rr)} test members; {fit_count} FIT members in this cluster. Colour: Fe/H. Same stars in each column."
        )
        fig.savefig(output / f"cluster-{name}-kiel.png", dpi=150)
        plt.close(fig)
    if panels:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        for _, name, rr, tab, fit_count in panels[:8]:
            axes[0, 0].scatter(
                tab.g_mag, data["mean"][rr, 2], s=5, label=name, alpha=0.5
            )
            axes[0, 1].scatter(
                np.hypot(tab.x, tab.y) * 60, data["mean"][rr, 2], s=5, alpha=0.5
            )
            axes[1, 0].scatter(data["mean"][rr, 0], data["mean"][rr, 2], s=5, alpha=0.5)
            axes[1, 1].scatter(data["mean"][rr, 1], data["mean"][rr, 2], s=5, alpha=0.5)
        for ax, label in zip(
            axes.flat,
            [
                "G magnitude",
                "Cluster-centric radius [arcmin]",
                "Predicted Teff [K]",
                "Predicted log g",
            ],
        ):
            ax.set(xlabel=label, ylabel="Predicted [Fe/H]")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            "Devtest cluster means: magnitude/radius/parameter trends (diagnostic, no fits)"
        )
        fig.savefig(output / "cluster-trends.png", dpi=150)
        plt.close(fig)
    return dict(source_sha256=digest(archive), clusters=records)


def population_audit(data, p):
    from scipy.special import ndtr

    w, m, lower, mean, cov, ids, nv, obs = [
        data[k]
        for k in [
            "weights",
            "component_mean",
            "component_lower",
            "mean",
            "covariance",
            "source_id",
            "native_value",
            "native_observed",
        ]
    ]
    v = (lower * lower).sum(-1)
    sd = np.sqrt(v)
    entropy = -np.sum(w * np.log(np.maximum(w, 1e-300)), 1)
    between = np.sum(w[:, :, None] * (m - mean[:, None, :]) ** 2, 1)
    r = {
        "effective_components_quantiles": np.quantile(
            np.exp(entropy), [0.05, 0.5, 0.95]
        ).tolist(),
        "fraction_2_or_more_weights_over_005": float(np.mean((w > 0.05).sum(1) >= 2)),
        "between_component_variance_fraction_median": np.median(
            between / np.diagonal(cov, axis1=1, axis2=2), axis=0
        ).tolist(),
        "bounds": {},
    }
    for j, (name, lo, hi) in enumerate(
        [("Teff", 2500, 10000), ("logg", -1, 6.5), ("FeH", -5, 1)]
    ):
        mass = (
            w
            * (
                ndtr((lo - m[:, :, j]) / sd[:, :, j])
                + ndtr((m[:, :, j] - hi) / sd[:, :, j])
            )
        ).sum(1)
        r["bounds"][name] = {
            "limits": [lo, hi],
            "mean_mass_outside": float(np.mean(mass)),
            "stars_mass_over_001": int((mass > 0.01).sum()),
            "max_mass": float(mass.max()),
            "means_outside": int(((mean[:, j] < lo) | (mean[:, j] > hi)).sum()),
        }
    rows = []
    for s, survey in enumerate(["APOGEE", "DESI", "GALAH"]):
        for t, name in enumerate(["Teff", "logg", "FeH"]):
            ix = np.flatnonzero(obs[:, s, t])
            delta = mean[ix, t] - nv[ix, s, t]
            order = np.argsort(abs(delta))[-30:][::-1]
            for k in order:
                a = ix[k]
                rows.append(
                    dict(
                        source_id=int(ids[a]),
                        survey=survey,
                        target=name,
                        reference=float(nv[a, s, t]),
                        prediction=float(mean[a, t]),
                        residual=float(delta[k]),
                        sigma=float(np.sqrt(cov[a, t, t])),
                    )
                )
    pd.DataFrame(rows).to_csv(p / "largest-native-disagreements.csv", index=False)
    (p / "population-audit.json").write_text(json.dumps(r, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--fit-ids", type=Path, required=True)
    p.add_argument("--clusters", type=Path)
    p.add_argument("--gaia", type=Path)
    p.add_argument("--rave", type=Path)
    p.add_argument("--rave-fe", type=Path)
    p.add_argument("--rave-match", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    expected = "b73bd23c939ec8e36a51abcec29a9a306033e07025940c5cfd5702c644c2ccd9"
    if digest(args.predictions) != expected:
        raise ValueError("This audit is bound to the completed devtest export")
    with h5py.File(args.predictions) as f:
        data = {k: f[k][:] for k in f}
        attrs = {k: str(v) for k, v in f.attrs.items()}
    fit_ids = np.load(args.fit_ids)
    if (
        hashlib.sha256(np.sort(fit_ids).astype("<i8").tobytes()).hexdigest()
        != "870027a5393713aa5e507534750b048900428b8f9f839475181b7cbb0c010ed8"
    ):
        raise ValueError("FIT identity set differs")
    assert np.intersect1d(fit_ids, data["source_id"]).size == 0
    result = dict(
        predictions_sha256=expected,
        attributes=attrs,
        devtest_stars=len(data["source_id"]),
        fit_overlap=0,
        refitting=False,
        native_common_scale_comparisons={},
    )
    for s, name in enumerate(SURVEYS):
        result["native_common_scale_comparisons"][name] = compare(
            data,
            np.arange(len(data["source_id"])),
            data["native_value"][:, s],
            data["native_error"][:, s],
            data["native_observed"][:, s].astype(bool),
            name + "-native-reference",
            args.output,
            score=False,
        )
    population_audit(data, args.output)
    if args.clusters:
        result["globular_clusters"] = clusters(
            data, fit_ids, args.clusters, args.output
        )
    if args.gaia:
        from astropy.io import fits

        from masked_stellar_autoencoder.training.multisurvey_gaia_gspspec import (
            quality_masks,
        )

        if (
            digest(args.gaia)
            != "e165c6f1fc4375389f24db86ff093e0c5d6e7c5da7a49b5564998a625f55dd83"
        ):
            raise ValueError("Gaia reference differs")
        with fits.open(args.gaia, memmap=True) as f:
            a = f[1].data
            rr, jj = join(data["source_id"], a["source_id"])
            y = np.full((len(jj), 3), np.nan)
            e = y.copy()
            for t, key in enumerate(["teff", "logg"]):
                y[:, t] = a[key + "_gspspec"][jj]
                e[:, t] = (
                    a[key + "_gspspec_upper"][jj] - a[key + "_gspspec_lower"][jj]
                ) / 2
            flags = a["flags_gspspec"][jj]
            masks = quality_masks(y[:, 0], y[:, 1], flags)
        result["gaia"] = dict(
            matched=len(rr),
            source_sha256=digest(args.gaia),
            quality_policy="existing exploratory per-target flag tier; percentile widths only Gaussian proxies",
        )
        for tier in ["raw", "selected"]:
            result["gaia"][tier] = compare(
                data, rr, y, e, masks[tier], "Gaia-GSP-Spec-" + tier, args.output
            )
    if args.rave:
        from astropy.table import Table

        m = pd.read_csv(args.rave)
        fe = pd.read_csv(args.rave_fe)
        x = Table.read(args.rave_match, format="votable").to_pandas()
        # Reject TAP truncation; repeats chosen by metadata, not agreement.
        if len(x) != 517357:
            raise ValueError("RAVE crossmatch differs from the pinned 2026-11-05 query")
        a = x.merge(m, on="rave_obs_id", validate="one_to_one").merge(
            fe, on="rave_obs_id", how="left", validate="one_to_one"
        )
        a = a[
            np.isfinite(a.dist)
            & (a.dist <= 1)
            & (a.algo_conv_madera == 0)
            & (a.snr_madera >= 40)
        ]
        a = a.sort_values(
            ["source_id", "snr_madera", "dist", "rave_obs_id"],
            ascending=[True, False, True, True],
        ).drop_duplicates("source_id")
        rr, jj = join(data["source_id"], a.source_id.to_numpy(dtype=np.int64))
        a = a.iloc[jj]
        y = a[["teff_cal_madera", "logg_cal_madera", "fe_h_gauguin"]].to_numpy()
        e = a[
            ["teff_error_madera", "logg_error_madera", "fe_h_error_gauguin"]
        ].to_numpy()
        mask = np.isfinite(y) & (y != -9999)
        mask[:, 0] &= (y[:, 0] > 2500) & (y[:, 0] < 10000)
        mask[:, 1] &= (y[:, 1] > -1) & (y[:, 1] < 6.5)
        mask[:, 2] &= (
            (a.fe_h_nl_gauguin.to_numpy() > 0) & (y[:, 2] > -5) & (y[:, 2] < 1)
        )
        result["rave"] = dict(
            matched=len(rr),
            source_sha256={
                str(p): digest(p) for p in [args.rave, args.rave_fe, args.rave_match]
            },
            selection="MADERA convergence0; SNR>=40; EDR3 separation<=1arcsec; highest SNR repeat then smallest separation then observation ID",
            metrics=compare(
                data, rr, y, e, mask, "RAVE-spectroscopic-reference", args.output
            ),
        )
    (args.output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            dict(
                output=str(args.output),
                devtest_stars=result["devtest_stars"],
                complete=True,
            )
        )
    )


if __name__ == "__main__":
    main()
