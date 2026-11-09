"""Full-count, saved-array common-reference plots; no neural model is loaded."""

import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from masked_stellar_autoencoder.training.survey_map_arrays import inverse_map_axis

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
NAMES = [
    "Teff [K]",
    "log g [dex]",
    "[Fe/H] [dex]",
    "[alpha/Fe] [dex]",
    "age [Gyr; teacher scale]",
    "distance [kpc; teacher scale]",
]
KEYS = ["teff", "logg", "fe_h", "alpha", "age", "distance"]
ROLES = [b"fit", b"tune", b"cal", b"devtest"]
LIMITS = [(2300, 13000), (-1, 6.5), (-4, 1), (-1, 1.5), (0, 22), (0, 30)]
RESID = [2000, 2, 2, 1, 12, 12]


def sha(path, kind="sha256"):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, kind).hexdigest()


def stats(residual):
    r = np.asarray(residual)
    r = r[np.isfinite(r)]
    if not len(r):
        return dict(n=0)
    q = np.quantile(r, [0.025, 0.16, 0.5, 0.84, 0.975])
    return dict(
        n=len(r),
        bias_mean=float(r.mean()),
        bias_median=float(q[2]),
        mae=float(abs(r).mean()),
        rmse=float(np.sqrt(np.mean(r * r))),
        nmad=float(1.4826 * np.median(abs(r - q[2]))),
        p025=float(q[0]),
        p16=float(q[1]),
        p84=float(q[3]),
        p975=float(q[4]),
    )


def sample(n, count=1500):
    # Uniform without replacement; no ordering or residual-dependent selection.
    return np.sort(
        np.random.default_rng(2026110788).choice(n, min(n, count), replace=False)
    )


def density_line(ax, values, bins, color, *, style="-", label=None, width=1):
    h, edges = np.histogram(values, bins)
    if len(values):
        # Normalize to the full sample: plotted tails are not silently renormalized.
        ax.plot(
            (edges[1:] + edges[:-1]) / 2,
            h / (len(values) * np.diff(edges)),
            color=color,
            ls=style,
            lw=width,
            label=label,
        )


def kiel_contour(ax, teff, logg, color):
    h, xe, ye = np.histogram2d(
        teff, logg, bins=(160, 120), range=(LIMITS[0], LIMITS[1])
    )
    positive = h[h > 0]
    if len(positive) and positive.max() > 1:
        levels = np.unique(np.quantile(positive, [0.5, 0.85, 0.97]))
        levels = levels[(levels > h.min()) & (levels < h.max())]
        if len(levels):
            ax.contour(
                (xe[1:] + xe[:-1]) / 2,
                (ye[1:] + ye[:-1]) / 2,
                h.T,
                levels=levels,
                colors=[color],
                linewidths=0.65,
                alpha=0.7,
            )
    ix = sample(len(teff), 1500)
    ax.scatter(teff[ix], logg[ix], s=2, color=color, alpha=0.18, rasterized=True)
    return dict(
        n=len(teff),
        outside_display=int(
            (
                (teff < LIMITS[0][0])
                | (teff > LIMITS[0][1])
                | (logg < LIMITS[1][0])
                | (logg > LIMITS[1][1])
            ).sum()
        ),
    )


def kiel_axes(ax):
    ax.set(xlim=LIMITS[0][::-1], ylim=LIMITS[1][::-1], xlabel=NAMES[0], ylabel=NAMES[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--producer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="waiting_for_full_union_predictions",
        training_performed=False,
        model_loaded=False,
        plots=[],
        residual_statistics=[],
        common_label_counts={},
        clusters=[],
        scope="Complete usable labelled-source registry, immutable first large checkpoint. All histogram counts and residual statistics use all eligible measurements; scatter overlays use fixed uniform samples. Inverse-mapped noisy survey observations are references, not physical truth. No averaging of overlapping surveys. CAL is diagnostic only here; no new correction is fitted.",
    )

    def save():
        tmp = a.output / "manifest.tmp"
        tmp.write_text(json.dumps(m, indent=2, allow_nan=False) + "\n")
        tmp.replace(a.output / "manifest.json")

    def figure(fig, name):
        fig.savefig(a.output / name, dpi=150)
        plt.close(fig)
        m["plots"].append(name)

    save()
    try:
        deadline = time.monotonic() + 96 * 3600
        while time.monotonic() < deadline:
            try:
                pm = json.loads((a.producer / "manifest.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                pm = {}
            if pm.get("status") == "failed":
                raise RuntimeError("Inference failed: " + str(pm.get("error")))
            if pm.get("status") == "full_label_union_inference_complete":
                break
            time.sleep(30)
        else:
            raise TimeoutError("No full-union predictions in 96h")
        smpath = Path(pm["source_manifest_path"])
        source = json.loads(smpath.read_text())
        if (
            sha(source["prepared_path"]) != pm["source_sha256"]
            or source["prepared_sha256"] != pm["source_sha256"]
        ):
            raise ValueError("Source binding differs")
        if (
            sha(a.producer / "predictions.h5") != pm["predictions_sha256"]
            or sha(a.producer / "map-coefficients.npz") != pm["map_sha256"]
        ):
            raise ValueError("Prediction/map artifact differs")
        with np.load(a.producer / "map-coefficients.npz") as c:
            coefficients = {k: c[k] for k in c.files}
        registry = source["source_registry"]
        if list(coefficients["source_registry"]) != registry:
            raise ValueError("Map registry differs")
        colors = {s: plt.get_cmap("tab20")(i) for i, s in enumerate(registry)}
        spectro = [s for s in registry if not s.startswith("StarHorse")]
        with (
            h5py.File(a.producer / "predictions.h5") as pf,
            h5py.File(source["prepared_path"]) as sf,
        ):
            ids, role, eligible = (
                pf["source_id"][:],
                pf["role"][:],
                pf["eligible"][:].astype(bool),
            )
            np.testing.assert_array_equal(ids, sf["source_id"][:])
            np.testing.assert_array_equal(role, sf["role"][:].astype("S8"))
            predicted = pf["median"][:]
            if not np.isfinite(predicted[eligible]).all():
                raise ValueError("Incomplete finite prediction grid")
            m.update(
                status="plotting",
                source_sha256=pm["source_sha256"],
                checkpoint_sha256=pm["checkpoint_sha256"],
                checkpoint_update=pm["checkpoint_update"],
                total_prediction_stars=int(eligible.sum()),
                source_registry=registry,
                spectroscopic_sources=spectro,
                marginal_draws=pm["draws"],
            )
            # Keep the complete label outer join in a row-bound HDF5, source by source.
            refpath = a.output / "mapped-survey-observations.h5"
            labelsets = {}
            native_labelsets = {}
            with h5py.File(refpath, "w") as out:
                out.attrs["source_sha256"] = pm["source_sha256"]
                out.attrs["checkpoint_sha256"] = pm["checkpoint_sha256"]
                out.attrs["interpretation"] = (
                    "Exact inverse observation-map coordinates; measurement noise remains. No source averaging."
                )
                for si, name in enumerate(registry):
                    g = sf["native_observations/" + name]
                    rows = g["star_row"][:]
                    observed = g["observed"][:].astype(bool)
                    native = g["value"][:]
                    common = np.full(native.shape, np.nan)
                    for j in range(6):
                        valid = observed[:, j] & np.isfinite(native[:, j])
                        if j >= 4:
                            valid &= native[:, j] > 0
                        observed[:, j] &= valid
                        common[valid, j] = inverse_map_axis(
                            native[valid, j], j, si, coefficients
                        )
                    if not np.isfinite(common[observed]).all():
                        raise ValueError("Nonfinite inverse-mapped observation")
                    og = out.create_group(name)
                    for key, val in [
                        ("star_row", rows),
                        ("source_id", ids[rows]),
                        ("role", role[rows]),
                        ("observed", observed),
                        ("common_value", common),
                        ("native_value", native),
                    ]:
                        og.create_dataset(
                            key, data=val, compression="gzip", compression_opts=1
                        )
                    labelsets[name] = (rows, observed, common)
                    native_labelsets[name] = native
                    m["common_label_counts"][name] = dict(
                        records=len(rows),
                        observed_by_target=observed.sum(0).tolist(),
                        eligible_records=int(eligible[rows].sum()),
                    )
            save()
            # Survey-colored Kiel diagrams use exactly the same jointly observed T/g stars.
            union = np.zeros(len(ids), bool)
            for name in spectro:
                rows, obs, _ = labelsets[name]
                union[rows[obs.any(1)]] = True
            m["spectroscopic_union_stars"] = int((union & eligible).sum())
            m["spectroscopic_label_union_stars_including_no_prediction"] = int(
                union.sum()
            )
            m["kiel_counts"] = {}
            for r in [b"all"] + ROLES:
                fig, axes = plt.subplots(
                    1,
                    3,
                    figsize=(18, 5.8),
                    layout="constrained",
                    sharex=True,
                    sharey=True,
                )
                handles = []
                for name in spectro:
                    rows, obs, common = labelsets[name]
                    labelled = obs[:, 0] & obs[:, 1]
                    if r != b"all":
                        labelled &= role[rows] == r
                    take = labelled & eligible[rows]
                    all_y, y, pred = (
                        common[labelled],
                        common[take],
                        predicted[rows[take]],
                    )
                    if not len(all_y):
                        continue
                    allcnt = kiel_contour(
                        axes[0], all_y[:, 0], all_y[:, 1], colors[name]
                    )
                    cnt = kiel_contour(axes[1], y[:, 0], y[:, 1], colors[name])
                    predcnt = kiel_contour(
                        axes[2], pred[:, 0], pred[:, 1], colors[name]
                    )
                    m["kiel_counts"][r.decode() + "/" + name] = dict(
                        all_reference=allcnt, matched_reference=cnt, prediction=predcnt
                    )
                    handles.append(
                        Line2D(
                            [],
                            [],
                            color=colors[name],
                            label=f"{name}: {len(all_y):,} labels / {len(y):,} predicted",
                        )
                    )
                for ax in axes:
                    kiel_axes(ax)
                axes[0].set_title("All observed labels mapped to common reference")
                axes[1].set_title("Mapped labels with usable MSA inputs")
                axes[2].set_title("MSA common predictions: same stars as middle")
                fig.legend(
                    handles=handles, loc="outside lower center", ncol=2, fontsize=7
                )
                fig.suptitle(
                    f"{r.decode().upper()} — all-count contours; uniform scatter overlay; overlapping stars retained per survey",
                    fontsize=10,
                )
                figure(fig, "kiel-survey-colors-" + r.decode() + ".png")
            fig, axes = plt.subplots(
                1, 2, figsize=(12, 5), layout="constrained", sharex=True, sharey=True
            )
            for ax, r in zip(axes, [b"fit", b"devtest"], strict=True):
                rows = np.flatnonzero(eligible & union & (role == r))
                v = predicted[rows]
                h = ax.hist2d(
                    v[:, 0],
                    v[:, 1],
                    bins=(250, 180),
                    range=(LIMITS[0], LIMITS[1]),
                    norm=matplotlib.colors.LogNorm(),
                    cmap="magma",
                )
                kiel_axes(ax)
                ax.set_title(
                    f"{r.decode().upper()}: {len(rows):,} unique spectroscopic-union stars"
                )
                fig.colorbar(h[3], ax=ax, label="Stars per bin")
            figure(fig, "kiel-unique-union-density.png")
            # One residual sheet per target: source colors and four role columns.
            for j, target in enumerate(KEYS):
                # One streamed target read avoids pathological million-row h5py fancy selections.
                target_quantiles = pf["quantiles"][:, :, j]
                fig, axes = plt.subplots(3, 4, figsize=(17, 11), layout="constrained")
                distfig, dax = plt.subplots(
                    1, 4, figsize=(17, 4.5), layout="constrained"
                )
                bins = np.linspace(*LIMITS[j], 151)
                rbins = np.linspace(-RESID[j], RESID[j], 181)
                handles = []
                for name in registry:
                    rows, obs, common = labelsets[name]
                    if not obs[:, j].any():
                        continue
                    handles.append(Line2D([], [], color=colors[name], label=name))
                    for c, r in enumerate(ROLES):
                        take = obs[:, j] & eligible[rows] & (role[rows] == r)
                        rr = rows[take]
                        y, v = common[take, j], predicted[rr, j]
                        residual = v - y
                        record = dict(
                            source=name,
                            role=r.decode(),
                            target=target,
                            **stats(residual),
                            residual_outside_display=int(
                                (abs(residual) > RESID[j]).sum()
                            ),
                            reference_outside_display=int(
                                ((y < LIMITS[j][0]) | (y > LIMITS[j][1])).sum()
                            ),
                        )
                        if len(rr):
                            q = target_quantiles[rr]
                            record.update(
                                raw_latent_interval90_reference_inclusion=float(
                                    ((y >= q[:, 1]) & (y <= q[:, 7])).mean()
                                ),
                                scope="No observation-noise convolution; reference inclusion is not calibrated latent coverage",
                            )
                            if j >= 4:
                                record["fractional_residual"] = stats(residual / y)
                            ix = sample(len(y))
                            axes[0, c].scatter(
                                y[ix],
                                v[ix],
                                s=2,
                                alpha=0.25,
                                color=colors[name],
                                rasterized=True,
                            )
                            axes[1, c].scatter(
                                y[ix],
                                residual[ix],
                                s=2,
                                alpha=0.25,
                                color=colors[name],
                                rasterized=True,
                            )
                            # Full-count median residual trend, independent of scatter sample.
                            trend = []
                            for low, high in zip(bins[:-1:5], bins[5::5], strict=False):
                                b = (y >= low) & (y < high)
                                if b.sum() >= 30:
                                    trend.append(
                                        ((low + high) / 2, np.median(residual[b]))
                                    )
                            if trend:
                                t = np.asarray(trend)
                                axes[1, c].plot(
                                    t[:, 0], t[:, 1], color=colors[name], lw=1
                                )
                            density_line(axes[2, c], residual, rbins, colors[name])
                            density_line(dax[c], y, bins, colors[name], style="--")
                            density_line(dax[c], v, bins, colors[name])
                        m["residual_statistics"].append(record)
                for c, r in enumerate(ROLES):
                    axes[0, c].plot(LIMITS[j], LIMITS[j], color="black", lw=0.6)
                    axes[0, c].set(
                        xlim=LIMITS[j],
                        ylim=LIMITS[j],
                        title=r.decode().upper(),
                        xlabel="Mapped observed " + NAMES[j],
                        ylabel="MSA common median",
                    )
                    axes[1, c].axhline(0, color="black", lw=0.6)
                    axes[1, c].set(
                        xlim=LIMITS[j],
                        ylim=(-RESID[j], RESID[j]),
                        xlabel="Mapped observed " + NAMES[j],
                        ylabel="Prediction − mapped observation",
                    )
                    axes[2, c].set(
                        xlim=(-RESID[j], RESID[j]),
                        xlabel="Prediction − mapped observation",
                        ylabel="Residual density (all records)",
                    )
                    dax[c].set(
                        xlim=LIMITS[j],
                        title=r.decode().upper(),
                        xlabel=NAMES[j],
                        ylabel="Population density",
                    )
                fig.legend(
                    handles=handles, loc="outside lower center", ncol=3, fontsize=7
                )
                fig.suptitle(
                    target
                    + ": all-count residual statistics/histograms; uniformly sampled scatter; noisy mapped references",
                    fontsize=11,
                )
                figure(fig, "residuals-all-splits-" + target + ".png")
                distfig.legend(
                    handles=handles, loc="outside lower center", ncol=3, fontsize=7
                )
                distfig.suptitle(
                    "Matched populations: mapped observations dashed, MSA medians solid; "
                    + target,
                    fontsize=11,
                )
                figure(distfig, "population-distributions-" + target + ".png")
                save()
            del target_quantiles
            overlap_agreement(
                labelsets,
                native_labelsets,
                spectro,
                role,
                eligible,
                a.output,
                m,
                figure,
            )
            save()
            # Actual whole-vector cluster draws: distinguish median-star spread and uncertainty mixtures.
            archive = ROOT / "expanded_six_source_20261107_v2/clusters.zip"
            if (
                sha(archive)
                != "9eafa2e10afc031790cfe36ed8545aa6336cc87c317fab167e9cea9dacbd0748"
            ):
                raise ValueError("Cluster membership archive differs")
            clusterrows = (
                pf["cluster_row"][:] if "cluster_row" in pf else np.empty(0, dtype="i8")
            )
            clusterindex = {int(row): i for i, row in enumerate(clusterrows)}
            idsort = np.argsort(ids)
            sortedids = ids[idsort]
            clusters = []
            with zipfile.ZipFile(archive) as z:
                for member in sorted(z.namelist()):
                    if not member.startswith("catalogues/") or not member.endswith(
                        ".txt"
                    ):
                        continue
                    text = z.read(member).decode()
                    cols = text.splitlines()[0].lstrip("#").split()
                    tab = pd.read_csv(
                        io.StringIO(text),
                        sep=r"\s+",
                        comment="#",
                        names=cols,
                        dtype={"source_id": np.int64},
                    )
                    tab = tab[
                        ((tab.qflag.to_numpy(dtype=int) & 2) != 0)
                        & (tab.memberprob.to_numpy() > 0.9)
                    ]
                    ii = np.searchsorted(sortedids, tab.source_id.to_numpy())
                    good = ii < len(ids)
                    good[good] &= sortedids[ii[good]] == tab.source_id.to_numpy()[good]
                    rr = np.unique(idsort[ii[good]])
                    rr = rr[eligible[rr]]
                    if len(rr) >= 10:
                        clusters.append((len(rr), Path(member).stem, rr))
            for n, name, rr in sorted(clusters, reverse=True)[:8]:
                if (role[rr] == b"fit").any():
                    raise ValueError("Reserved cluster members unexpectedly in FIT")
                fig, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
                record = dict(
                    cluster=name,
                    n=n,
                    role_counts={
                        r.decode(): int((role[rr] == r).sum())
                        for r in np.unique(role[rr])
                    },
                    targets={},
                )
                di = np.asarray(
                    [clusterindex[int(r)] for r in rr if int(r) in clusterindex],
                    dtype="i8",
                )
                # Sorted cluster HDF5 indices for h5py; one star receives equal draw weight.
                draws = pf["cluster_draws"][np.sort(di)] if len(di) else None
                for j, ax in enumerate(axes.flat):
                    bins = np.linspace(*LIMITS[j], 151)
                    density_line(
                        ax,
                        predicted[rr, j],
                        bins,
                        "black",
                        label=f"MSA star medians (n={n})",
                        width=1.8,
                    )
                    if draws is not None:
                        density_line(
                            ax,
                            draws[:, :, j].ravel(),
                            bins,
                            "grey",
                            style=":",
                            label="Equal-star posterior mixture",
                            width=1.3,
                        )
                    rec = dict(
                        median_population=stats(predicted[rr, j]),
                        mapped_observations={},
                    )
                    for source_name in registry:
                        rows, obs, common = labelsets[source_name]
                        take = np.isin(rows, rr) & obs[:, j]
                        if not take.any():
                            continue
                        y = common[take, j]
                        v = predicted[rows[take], j]
                        density_line(
                            ax,
                            y,
                            bins,
                            colors[source_name],
                            style="--",
                            label=f"{source_name} labels ({len(y)})",
                        )
                        density_line(
                            ax,
                            v,
                            bins,
                            colors[source_name],
                            style="-",
                            label=f"{source_name} matched MSA",
                        )
                        rec["mapped_observations"][source_name] = dict(
                            label_distribution=stats(y),
                            matched_predicted_distribution=stats(v),
                            residual=stats(v - y),
                        )
                    record["targets"][KEYS[j]] = rec
                    ax.set(xlim=LIMITS[j], xlabel=NAMES[j], ylabel="Density")
                    ax.legend(fontsize=5)
                fig.suptitle(
                    name
                    + ": mapped labels dashed; matched medians solid; uncertainty mixture dotted",
                    fontsize=10,
                )
                figure(fig, "cluster-" + name + "-distributions.png")
                m["clusters"].append(record)
            # Keep comparator joins separate from survey observation maps.
            compare_catalogues(
                ids, role, eligible & union, predicted, a.output, m, figure, save
            )
        pd.json_normalize(m["residual_statistics"], sep="_").to_csv(
            a.output / "residual-statistics.csv", index=False
        )
        m.update(
            status="full_label_union_report_complete",
            mapped_observations_sha256=sha(refpath),
        )
        save()
        lines = [
            "# Full label-union common-reference QA",
            "",
            m["scope"],
            "",
            f"Prediction stars: {m['total_prediction_stars']:,}; unique usable spectroscopic-union stars: {m['spectroscopic_union_stars']:,}; checkpoint update: {m['checkpoint_update']}.",
            "",
            "An outer join retains all available labels; it does not define a unique true label. Each source measurement is transformed with the exact inverse of its learned monotone observation map. Overlap remains multiple measurements. Individual source maps do not establish an absolute physical reference.",
            "",
            "Kiel panels require Teff and logg actually observed in the same survey. Predictions are shown on identical stars. Density contours use every in-range record; scatter is a fixed uniform overlay. Tail counts are recorded in the manifest. Teacher age/distance channels are shown separately in residual statistics and are not spectroscopic age/distance truth.",
            "",
            "FIT residuals include optimization exposure; TUNE has selection exposure; DEV has development exposure. CAL is not used to adjust this model or intervals. Raw interval-reference inclusion omits measurement noise and is not a calibrated coverage claim. Full statistics: [CSV](residual-statistics.csv).",
            "",
            "Cluster median distributions and equal-star posterior mixtures are different quantities. Narrower distributions alone do not establish better physical calibration.",
            "",
        ]
        for name in m["plots"]:
            lines += [f"![{name}]({name})", ""]
        (a.output / "report.md").write_text("\n".join(lines))
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


def overlap_agreement(labelsets, native, spectro, roles, eligible, output, m, figure):
    """Same-star survey differences, before and after mapping, split by star role."""
    records = []
    m["overlap_duplicate_sources_skipped"] = []
    for ai, a in enumerate(spectro):
        ar, ao, av = labelsets[a]
        if len(np.unique(ar)) != len(ar):
            m["overlap_duplicate_sources_skipped"].append(a)
            continue
        for b in spectro[ai + 1 :]:
            br, bo, bv = labelsets[b]
            if len(np.unique(br)) != len(br):
                continue
            row, ia, ib = np.intersect1d(
                ar, br, assume_unique=True, return_indices=True
            )
            for j in range(3):
                both = ao[ia, j] & bo[ib, j] & eligible[row]
                for role in ROLES:
                    take = both & (roles[row] == role)
                    if not take.any():
                        continue
                    raw = native[a][ia[take], j] - native[b][ib[take], j]
                    mapped = av[ia[take], j] - bv[ib[take], j]
                    records.append(
                        dict(
                            source_a=a,
                            source_b=b,
                            role=role.decode(),
                            target=KEYS[j],
                            native=stats(raw),
                            common=stats(mapped),
                        )
                    )
    m["overlap_agreement"] = records
    pd.json_normalize(records, sep="_").to_csv(
        output / "same-star-survey-agreement.csv", index=False
    )
    for j in range(3):
        fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
        matrices = []
        for coordinate in ["native", "common"]:
            matrix = np.full((len(spectro), len(spectro)), np.nan)
            np.fill_diagonal(matrix, 0)
            for rec in records:
                if rec["role"] != "devtest" or rec["target"] != KEYS[j]:
                    continue
                ai, bi = spectro.index(rec["source_a"]), spectro.index(rec["source_b"])
                matrix[ai, bi] = matrix[bi, ai] = rec[coordinate]["nmad"]
            matrices.append(matrix)
        finite = np.concatenate([v[np.isfinite(v)] for v in matrices])
        upper = float(finite.max()) if len(finite) and finite.max() > 0 else 1
        for ax, matrix, title in zip(
            axes,
            matrices,
            ["Native survey differences", "Mapped common differences"],
            strict=True,
        ):
            artist = ax.imshow(matrix, vmin=0, vmax=upper, cmap="magma")
            ax.set_xticks(range(len(spectro)), spectro, rotation=90, fontsize=6)
            ax.set_yticks(range(len(spectro)), spectro, fontsize=6)
            ax.set_title(title)
            fig.colorbar(artist, ax=ax, label="NMAD of same-star survey difference")
        fig.suptitle(
            "DEV overlaps: "
            + NAMES[j]
            + "; matched stars, not proof of absolute accuracy",
            fontsize=10,
        )
        figure(fig, "overlap-agreement-" + KEYS[j] + ".png")


def compare_catalogues(ids, role, eligible, predicted, output, m, figure, save):
    """Join full releases, with exact int64 IDs and matched-star split comparisons."""
    import pyarrow.parquet as pq

    index = (
        ROOT / "andrae_full_footprint_census_20261106_v2/andrae-full-inference-index.h5"
    )
    im = json.loads((index.parent / "manifest.json").read_text())
    if im.get("status") != "complete" or sha(index) != im["index_sha256"]:
        raise ValueError("Andrae index differs")
    order = np.argsort(ids)
    sortedids = ids[order]
    catalogues = {}
    with h5py.File(index) as f:
        allids = f["source_id"][:]
        pos = np.searchsorted(allids, sortedids)
        good = pos < len(allids)
        good[good] &= allids[pos[good]] == sortedids[good]
        rr = pos[good]
        y = np.full((len(ids), 3), np.nan, dtype="f4")
        y[order[good]] = np.column_stack(
            [f[k][rr] for k in ["teff_xgboost", "logg_xgboost", "mh_xgboost"]]
        )
        train = np.zeros(len(ids), bool)
        train[order[good]] = f["in_training_sample"][rr]
    del allids
    catalogues["Andrae_MH_proxy"] = (y, train)
    path = ROOT / "decision_xp_catalogue_qa_20261106_v1/gaia-xp.parquet"
    if sha(path, "md5") != "4cc09524fc1b4fe774dbeeb9dc0d50c2":
        raise ValueError("Gaia Net checksum differs")
    y = np.full((len(ids), 3), np.nan, dtype="f4")
    seen = np.zeros(len(ids), bool)
    scanned = 0
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=500000, columns=["source_id", "logTeff", "logg", "FeH"]
    ):
        sid = batch.column(0).to_numpy(zero_copy_only=False)
        if sid.dtype.kind not in "iu" or (
            sid.dtype.kind == "u" and (sid > np.iinfo("i8").max).any()
        ):
            raise ValueError("Exact Gaia IDs required")
        sid = sid.astype("i8", copy=False)
        pos = np.searchsorted(sortedids, sid)
        good = pos < len(ids)
        good[good] &= sortedids[pos[good]] == sid[good]
        ii = np.flatnonzero(good)
        rr = order[pos[good]]
        if seen[rr].any() or len(np.unique(rr)) != len(rr):
            raise ValueError("Duplicate comparator IDs")
        y[rr] = np.column_stack(
            [
                10 ** batch.column(1).to_numpy(zero_copy_only=False)[ii],
                batch.column(2).to_numpy(zero_copy_only=False)[ii],
                batch.column(3).to_numpy(zero_copy_only=False)[ii],
            ]
        )
        seen[rr] = True
        scanned += len(sid)
    if scanned != 219196764:
        raise ValueError("Gaia Net full release row count differs")
    catalogues["GaiaNet"] = (y, None)
    m["comparator_statistics"] = {}
    with h5py.File(output / "matched-xp-catalogues.h5", "w") as out:
        out.create_dataset("source_id", data=ids)
        for name, (y, train) in catalogues.items():
            g = out.create_group(name)
            g.create_dataset("value", data=y, compression="gzip", compression_opts=1)
            if train is not None:
                g.create_dataset("in_training_sample", data=train)
            m["comparator_statistics"][name] = {}
            for j, target in enumerate(KEYS[:3]):
                fig, axes = plt.subplots(2, 4, figsize=(16, 7), layout="constrained")
                for c, r in enumerate(ROLES):
                    take = eligible & (role == r) & np.isfinite(y[:, j])
                    yy, v = y[take, j], predicted[take, j]
                    rec = stats(v - yy)
                    if train is not None:
                        rec["outside_comparator_training"] = stats(
                            predicted[take & ~train, j] - y[take & ~train, j]
                        )
                    m["comparator_statistics"][name][target + "/" + r.decode()] = rec
                    ix = sample(len(v), 12000)
                    axes[0, c].scatter(yy[ix], v[ix], s=1, alpha=0.15, rasterized=True)
                    axes[0, c].plot(LIMITS[j], LIMITS[j], color="black", lw=0.6)
                    axes[0, c].set(
                        xlim=LIMITS[j],
                        ylim=LIMITS[j],
                        xlabel=name + " " + NAMES[j],
                        ylabel="MSA common median",
                        title=r.decode().upper() + f" n={len(v):,}",
                    )
                    density_line(
                        axes[1, c],
                        v - yy,
                        np.linspace(-RESID[j], RESID[j], 181),
                        "black",
                    )
                    axes[1, c].set(
                        xlabel="MSA − catalogue", ylabel="All-match residual density"
                    )
                fig.suptitle(
                    name
                    + ": matched spectroscopic-union stars; catalogue agreement is not physical truth. Andrae M/H ≠ Fe/H.",
                    fontsize=10,
                )
                figure(fig, "comparator-" + name + "-" + target + ".png")
            take = eligible & np.isfinite(y).all(1)
            rr = np.flatnonzero(take)
            fig, axes = plt.subplots(
                1, 2, figsize=(12, 5), layout="constrained", sharex=True, sharey=True
            )
            for ax, v, title in [
                (axes[0], y[rr], name),
                (axes[1], predicted[rr], "MSA common predictions"),
            ]:
                h = ax.hist2d(
                    v[:, 0],
                    v[:, 1],
                    bins=(200, 150),
                    range=(LIMITS[0], LIMITS[1]),
                    norm=matplotlib.colors.LogNorm(),
                    cmap="magma",
                )
                kiel_axes(ax)
                ax.set_title(title + f": {len(rr):,} matched stars")
                fig.colorbar(h[3], ax=ax, label="Stars per bin")
            figure(fig, "comparator-" + name + "-matched-kiel.png")
            save()


if __name__ == "__main__":
    main()
