"""Saved-array benchmark, seismic, cluster, native-reference and corner QA."""

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

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
NAMES = ["teff", "logg", "fe_h", "alpha", "age", "distance"]


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def metrics(pred, truth):
    r = np.asarray(pred) - np.asarray(truth)
    return (
        dict(
            n=len(r),
            bias=float(np.median(r)),
            mae=float(np.mean(abs(r))),
            nmad=float(1.4826 * np.median(abs(r - np.median(r)))),
        )
        if len(r)
        else dict(n=0)
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qa-run", default="expanded_first_checkpoint_qa_20261107_v1")
    p.add_argument("--benchmarks", type=Path, required=True)
    p.add_argument("--reuse-waiting-output", action="store_true")
    a = p.parse_args()
    from saved_array_source_binding import (
        load_bound_source,
        population_panel_masks,
        prepare_report_output,
    )

    prepare_report_output(a.output, "waiting_for_cuda_arrays", a.reuse_waiting_output)
    from plot_decision_six_corners import corner, display_coordinates

    from masked_stellar_autoencoder.training.survey_map_arrays import map_axis

    qa = ROOT / a.qa_run
    deadline = time.monotonic() + 48 * 3600
    m = dict(
        status="waiting_for_cuda_arrays",
        model_loaded=False,
        training_performed=False,
        plots={},
        models={},
        native_reference={},
        clusters=[],
        scope="Preliminary first-checkpoint QA, not a selected final model. Catalogue/reference errors, calibration/teacher dependence, shared Gaia inputs and ancestor exposure remain. No DEV/benchmark outcome feeds training selection.",
    )

    def save():
        path = a.output / "manifest.tmp"
        path.write_text(json.dumps(m, indent=2) + "\n")
        path.replace(a.output / "manifest.json")

    save()
    while time.monotonic() < deadline:
        try:
            producer = json.loads((qa / "manifest.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            producer = {}
        if producer.get("status") in ["upstream_failed", "failed"]:
            m.update(status="upstream_failed", error=producer.get("error"))
            save()
            return
        if producer.get("status") in ["first_checkpoint_common_qa_complete", "selected_checkpoint_common_qa_complete"]:
            break
        time.sleep(30)
    else:
        raise TimeoutError("CUDA common-space QA did not finish in 48h")
    with np.load(qa / "panel.npz") as f:
        panel = {k: f[k] for k in f.files}
    models = {}
    for name, item in producer["models"].items():
        path = qa / (name + "-common.npz")
        if sha(path) != item["artifacts"][path.name]["sha256"]:
            raise ValueError("Saved producer SHA differs")
        with np.load(path) as f:
            models[name] = {k: f[k] for k in f.files}
        np.testing.assert_array_equal(models[name]["source_id"], panel["source_id"])
        m["models"][name] = dict(
            checkpoint_sha256=item["checkpoint_sha256"],
            training_update=item["training_update"],
            epoch=item.get("epoch"),
        )
    with np.load(qa / "snapshot-map-coefficients.npz") as f:
        coefficients = {k: f[k] for k in f.files}
    m["status"] = "rendering_saved_arrays"
    save()
    large_name = producer.get("large_model_key", "large_first_checkpoint")
    large = models[large_name]
    if producer.get("snapshot_kind") == "final_selected":
        m["scope"] = producer["scope"]
    m["snapshot_kind"] = producer.get("snapshot_kind", "intermediate")
    m["large_model_key"] = large_name
    source = load_bound_source(producer, sha)
    panel.update(population_panel_masks(source, panel["row"]))
    m["population_panel_selection"] = dict(
        seed=2026110742,
        fit=2048,
        devtest=2048,
        scope="Uniform model-blind panels, excluding extra benchmark/cluster/seismic/TESS QA augmentation",
    )
    benchmarks = pd.read_csv(
        a.benchmarks, dtype={"source_id": "Int64", "GaiaDR3": "Int64"}
    )
    benchmarks = benchmarks[benchmarks.source_id.notna()].copy()
    # Exact positive integer identifiers; never match through floating point.
    index = {int(v): i for i, v in enumerate(panel["source_id"])}
    keep = benchmarks.source_id.map(lambda v: int(v) in index)
    b = benchmarks[keep].copy()
    ii = np.array([index[int(v)] for v in b.source_id], dtype="i8")
    b["input_eligible"] = panel["input_eligible"][ii]
    truth_fields = [
        ("teff", "Teff", "e_Teff", 0),
        ("logg", "logg", "e_logg", 1),
        ("fe_i_h", "[FeI/H]", "e_[FeI/H]", 2),
        ("fe_ii_h", "[FeII/H]", "e_[FeII/H]", 2),
    ]
    m["benchmark_reference_sha256"] = sha(a.benchmarks)
    m["benchmark_counts"] = dict(
        available_ids=len(benchmarks),
        matched=len(b),
        input_eligible=int(b.input_eligible.sum()),
        scope="Fundamental Teff uses flux/diameter; logg uses parallax and evolutionary masses; Fe I and Fe II references stay separate. Parent/survey calibration ancestry is not certified independent.",
    )
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    for label, column, error, j in truth_fields:
        y = pd.to_numeric(b[column], errors="coerce").to_numpy()
        valid = np.isfinite(y) & b.input_eligible.to_numpy()
        for name, model in models.items():
            stats = metrics(model["median"][ii[valid], j], y[valid])
            m["models"][name].setdefault("benchmarks", {})[label] = stats
            axes[truth_fields.index((label, column, error, j))].scatter(
                y[valid], model["median"][ii[valid], j], s=13, label=name, alpha=0.6
            )
        ax = axes[truth_fields.index((label, column, error, j))]
        if valid.any():
            lo, hi = y[valid].min(), y[valid].max()
            ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
        ax.set(
            xlabel="Reference " + label,
            ylabel="Common prediction",
            title=f"n={valid.sum()} eligible stars",
        )
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Benchmark reference compatibility: first large checkpoint versus earlier NSF"
    )
    fig.savefig(a.output / "benchmarks-true-vs-predicted.png", dpi=160)
    plt.close(fig)
    refs = pd.read_parquet(
        ROOT / "seismic_gravity_qa_20261106_v1/apokasc3-gravity-qa-reference.parquet"
    )
    ri = {int(v): i for i, v in enumerate(refs.source_id)}
    seismic = np.flatnonzero(panel["seismic_devtest"])
    rr = np.array([ri[int(v)] for v in panel["source_id"][seismic]])
    y = refs.loggSeis.to_numpy()[rr]
    error = refs.e_loggSeis.to_numpy()[rr]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for name, model in models.items():
        draw = model["logg_draws"][:, seismic]
        noisy = (
            draw
            + np.random.default_rng(2026110799).standard_normal(draw.shape)
            * error[None]
        )
        pit = (noisy <= y[None]).mean(0)
        stats = metrics(model["median"][seismic, 1], y)
        stats.update(
            quoted_noise_coverage90=float(((pit >= 0.05) & (pit <= 0.95)).mean()),
            median_width90=float(
                np.median(
                    np.quantile(noisy, 0.95, axis=0) - np.quantile(noisy, 0.05, axis=0)
                )
            ),
        )
        m["models"][name]["seismic_devtest"] = stats
        axes[0].scatter(y, model["median"][seismic, 1], s=5, label=name, alpha=0.4)
        axes[1].hist(pit, bins=20, histtype="step", density=True, label=name)
    axes[0].plot([0, 5], [0, 5], color="black", lw=0.8)
    axes[0].set(xlabel="APOKASC loggSeis", ylabel="Common predicted logg")
    axes[1].set(xlabel="Quoted-error-convolved marginal PIT", ylabel="Density")
    axes[1].legend(fontsize=7)
    fig.suptitle(
        f"{len(seismic)} fixed development seismology stars; teacher/calibration dependencies remain"
    )
    fig.savefig(a.output / "seismic-gravity.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for row, (name, model) in enumerate(models.items()):
        for col, role in enumerate([b"fit", b"devtest"]):
            take = panel["population_" + role.decode()] & panel["input_eligible"]
            v = model["median"][take]
            metallicity_artist = axes[row, col].scatter(
                v[:, 0],
                v[:, 1],
                c=v[:, 2],
                vmin=-2.5,
                vmax=0.5,
                cmap="viridis",
                s=3,
                rasterized=True,
            )
            axes[row, col].set(
                xlim=(8000, 3500),
                ylim=(5.5, -0.5),
                xlabel="Teff [K]",
                ylabel="log g",
                title=f"{name}: {role.decode()}, {take.sum()} stars",
            )
    fig.colorbar(
        metallicity_artist,
        ax=axes.ravel().tolist(),
        label="Common predicted [Fe/H] [dex]",
        extend="both",
        shrink=0.85,
    )
    fig.suptitle("Common-space Kiel diagrams: model-blind FIT and DEV panels")
    fig.savefig(a.output / "common-kiel-training-test.png", dpi=160)
    plt.close(fig)
    corners = {}
    for name, item in producer["models"].items():
        path = qa / (name + "-single-star.npz")
        if sha(path) != item["artifacts"][path.name]["sha256"]:
            raise ValueError("Single-star artifact differs")
        with np.load(path) as f:
            corners[name] = {k: f[k] for k in f.files}
    for sid in corners[large_name]["source_id"]:
        values = {
            name: data["native_draws"][:, np.flatnonzero(data["source_id"] == sid)[0]]
            for name, data in corners.items()
        }
        limits = np.quantile(
            np.concatenate([display_coordinates(v) for v in values.values()]),
            [0.002, 0.998],
            axis=0,
        ).T
        for name, v in values.items():
            path = a.output / f"{name}-{int(sid)}-single-star.png"
            m["plots"][path.name] = corner(
                v,
                f"{name}: one-star conditional samples, {len(v)} coupled six-vectors\nGaia {int(sid)}; predeclared stress example, not prevalence; teacher-scale age/distance",
                path,
                limits=limits,
            )
    for name, model in models.items():
        take = panel["population_devtest"] & panel["input_eligible"]
        path = a.output / (name + "-population-corner.png")
        m["plots"][path.name] = corner(
            model["median"][take],
            f"{name}: population relations between different stars\nSix medians per star; this plot does not show within-star uncertainty",
            path,
        )
    native_iron = {}
    with h5py.File(source["prepared_path"]) as f:
        for si, name in enumerate(source["source_registry"]):
            g = f["native_observations/" + name]
            rows = g["star_row"][:]
            pos = np.searchsorted(panel["row"], rows)
            hit = pos < len(panel["row"])
            hit[hit] &= panel["row"][pos[hit]] == rows[hit]
            gr = np.flatnonzero(hit)
            pi = pos[hit]
            # Prior native exposures are QA here, never a selection criterion.
            value = g["value"][:][gr]
            observed = g["observed"][:][gr].astype(bool)
            native_iron[name] = (pi, value[:, 2], observed[:, 2])
            role = panel["role"][pi]
            record = {}
            available = np.flatnonzero(observed.any(0))
            fig, axes = plt.subplots(
                1,
                max(1, len(available)),
                figsize=(4 * max(1, len(available)), 4),
                squeeze=False,
                layout="constrained",
            )
            for ax, j in zip(axes[0], available, strict=False):
                keep = (
                    observed[:, j] & (role == b"devtest") & panel["input_eligible"][pi]
                )
                predicted = map_axis(large["median"][pi[keep], j], j, si, coefficients)
                record[NAMES[j]] = metrics(predicted, value[keep, j])
                ax.scatter(value[keep, j], predicted, s=4, alpha=0.5)
                if keep.any():
                    lo, hi = np.quantile(value[keep, j], [0.005, 0.995])
                    ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
                ax.set(
                    xlabel="Native " + NAMES[j],
                    ylabel="Predicted survey value",
                    title=f"{keep.sum()} DEV reference stars",
                )
            fig.suptitle(
                name + ": actual observation map; age/distance are posterior teachers"
            )
            fig.savefig(a.output / (name + "-true-vs-predicted.png"), dpi=140)
            plt.close(fig)
            m["native_reference"][name] = record
        archive = ROOT / "expanded_six_source_20261107_v2/clusters.zip"
        if (
            sha(archive)
            != "9eafa2e10afc031790cfe36ed8545aa6336cc87c317fab167e9cea9dacbd0748"
        ):
            raise ValueError("Cluster archive differs")
        with zipfile.ZipFile(archive) as z:
            panels = []
            for member in sorted(z.namelist()):
                if not member.startswith("catalogues/") or not member.endswith(".txt"):
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
                take = ((tab.qflag.to_numpy(dtype=int) & 2) != 0) & (
                    tab.memberprob.to_numpy() > 0.9
                )
                use = tab.source_id.isin(panel["source_id"])
                tab = tab[take & use]
                pi = np.array([index[int(s)] for s in tab.source_id], dtype="i8")
                valid = panel["input_eligible"][pi]
                pi = pi[valid]
                if len(pi) < 10:
                    continue
                rec = dict(
                    cluster=Path(member).stem,
                    n=len(pi),
                    new_stage_fit_members=int((panel["role"][pi] == b"fit").sum()),
                    common_iron={
                        name: dict(
                            median=float(np.median(model["median"][pi, 2])),
                            nmad=metrics(
                                model["median"][pi, 2],
                                np.full(len(pi), np.median(model["median"][pi, 2])),
                            )["nmad"],
                        )
                        for name, model in models.items()
                    },
                )
                rec["same_star_native_comparison"] = {}
                for si, (source_name, (native_pi, reference, observed)) in enumerate(
                    native_iron.items()
                ):
                    take = np.isin(native_pi, pi) & observed
                    if take.sum() < 5:
                        continue
                    native_values = reference[take]
                    common = large["median"][native_pi[take], 2]
                    projected = map_axis(common, 2, si, coefficients)
                    nmad = lambda v: float(1.4826 * np.median(abs(v - np.median(v))))
                    rec["same_star_native_comparison"][source_name] = dict(
                        n=int(take.sum()),
                        native_iron_nmad=nmad(native_values),
                        common_iron_nmad=nmad(common),
                        survey_projected_nmad=nmad(projected),
                        common_minus_native_median=float(
                            np.median(common - native_values)
                        ),
                    )
                if rec["new_stage_fit_members"]:
                    raise ValueError("Reserved globular group entered FIT")
                m["clusters"].append(rec)
                panels.append((len(pi), Path(member).stem, pi))
            for _, name, pi in sorted(panels, reverse=True)[:8]:
                fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
                for ax, (modelname, model) in zip(axes, models.items(), strict=True):
                    v = model["median"][pi]
                    metallicity_artist = ax.scatter(
                        v[:, 0],
                        v[:, 1],
                        c=v[:, 2],
                        s=10,
                        vmin=-2.5,
                        vmax=0.5,
                        cmap="viridis",
                    )
                    ax.set(
                        xlim=(8000, 3500),
                        ylim=(5.5, -0.5),
                        xlabel="Teff [K]",
                        ylabel="log g",
                        title=modelname,
                    )
                fig.colorbar(
                    metallicity_artist,
                    ax=axes.ravel().tolist(),
                    label="Common predicted [Fe/H] [dex]",
                    extend="both",
                    shrink=0.85,
                )
                fig.suptitle(
                    f"{name}: {len(pi)} reserved input-eligible members; parent exposure not certified absent"
                )
                fig.savefig(a.output / (name + "-common-kiel.png"), dpi=150)
                plt.close(fig)
    m.update(
        status="selected_checkpoint_array_report_complete" if m["snapshot_kind"] == "final_selected" else "first_checkpoint_array_report_complete",
        producer_manifest_sha256=sha(qa / "manifest.json"),
        benchmark_scope="Common predictions compared separately with Fe I and Fe II. Native/source maps used only for catalogue compatibility. Lower cluster dispersion alone does not prove calibrated accuracy.",
    )
    save()
    text = [
        "# Selected comparison-checkpoint QA" if m["snapshot_kind"] == "final_selected" else "# First large-checkpoint QA",
        "",
        m["scope"],
        "",
        f"Training snapshot: update {m['models'][large_name]['training_update']}; epoch {m['models'][large_name].get('epoch')}; snapshot kind {m['snapshot_kind']}.",
        "",
        "| Model | Seismic stars | logg MAE | noisy-reference C90 |",
        "|---|---:|---:|---:|",
    ]
    for name, item in m["models"].items():
        q = item["seismic_devtest"]
        text.append(
            f"| {name} | {q['n']} | {q['mae']:.4f} | {q['quoted_noise_coverage90']:.3f} |"
        )
    text += [
        "",
        "![Common Kiel diagrams](common-kiel-training-test.png)",
        "",
        "![Seismic gravity](seismic-gravity.png)",
        "",
        "![Benchmark comparisons](benchmarks-true-vs-predicted.png)",
        "",
        "Full single-star and population corner plots and source-native true-versus-predicted figures are saved beside this report. Age and distance remain posterior-teacher scales.",
    ]
    (a.output / "report.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
