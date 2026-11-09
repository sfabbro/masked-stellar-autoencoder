"""Saved-array common Kiel and genuine per-star corners for mapping controls."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plot_decision_six_corners import corner, display_coordinates

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
KINDS = [
    "equalweight_affine",
    "apogee_affine",
    "seismic_affine",
    "seismic_nodesig",
    "seismic_offset",
    "seismic_spline",
]


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    m = dict(
        status="waiting_for_maps",
        training_performed=False,
        model_loaded=False,
        models={},
    )

    def save():
        path = a.output / "manifest.tmp"
        path.write_text(json.dumps(m, indent=2) + "\n")
        path.replace(a.output / "manifest.json")

    save()
    try:
        deadline = time.monotonic() + 24 * 3600
        while time.monotonic() < deadline:
            states = {
                kind: json.loads(
                    (ROOT / f"sixmap_{kind}_20261107_v1/manifest.json").read_text()
                )
                for kind in KINDS
            }
            if any(v.get("status") == "failed" for v in states.values()):
                raise RuntimeError("Mapping control failed")
            if all(v.get("status") == "complete" for v in states.values()):
                break
            time.sleep(30)
        else:
            raise TimeoutError("Mapping controls unfinished")
        saved = {}
        identities = {}
        for kind, state in states.items():
            directory = ROOT / f"sixmap_{kind}_20261107_v1"
            if (
                sha(directory / "selected-checkpoint.pth")
                != state["selected_checkpoint_sha256"]
            ):
                raise ValueError("Selected state changed")
            data = {}
            hashes = {}
            for label, filename in [
                ("fit", "training-predictions.npz"),
                ("devtest", "development-predictions.npz"),
                ("corner", "whole-vector-corners.npz"),
            ]:
                path = directory / filename
                hashes[filename] = sha(path)
                with np.load(path, allow_pickle=False) as f:
                    data[label] = {k: f[k] for k in f.files}
                if label in identities:
                    np.testing.assert_array_equal(
                        data[label]["source_id"], identities[label]
                    )
                else:
                    identities[label] = data[label]["source_id"]
            g = {
                role: next(
                    x
                    for x in state["evaluation"][role]["sources"]
                    if x["source"] == "APOKASC3_loggSeis"
                )["targets"]["logg"]
                for role in ["tune", "devtest"]
            }
            m["models"][kind] = dict(
                selected_update=state["selected_update"],
                checkpoint_sha256=state["selected_checkpoint_sha256"],
                artifact_sha256=hashes,
                tune_macro_crps=state["tune_score"],
                seismic=g,
            )
            saved[kind] = data
        for role in ["fit", "devtest"]:
            fig, axes = plt.subplots(
                2, 3, figsize=(14, 9), constrained_layout=True, sharex=True, sharey=True
            )
            for ax, kind in zip(axes.flat, KINDS, strict=True):
                value = saved[kind][role]["native_median"]
                if not np.isfinite(value).all():
                    raise ValueError("Invalid saved common predictions")
                metallicity_artist = ax.scatter(
                    value[:, 0],
                    value[:, 1],
                    c=value[:, 2],
                    s=5,
                    cmap="viridis",
                    vmin=-2.5,
                    vmax=0.5,
                    rasterized=True,
                )
                ax.set(title=kind, xlabel="Common Teff [K]", ylabel="Common logg [dex]")
            fig.colorbar(
                metallicity_artist,
                ax=axes.ravel().tolist(),
                label="Common predicted [Fe/H] [dex]",
                extend="both",
                shrink=0.85,
            )
            axes.flat[0].invert_xaxis()
            axes.flat[0].invert_yaxis()
            fig.suptitle(
                f"{role}: same {len(identities[role]):,} stars, common predictive medians; selected 4,200-update pilots"
            )
            fig.savefig(a.output / f"common-kiel-{role}.png", dpi=170)
            plt.close(fig)
        fig, axes = plt.subplots(
            1, 2, figsize=(11, 4.6), constrained_layout=True, sharex=True, sharey=True
        )
        for ax, role in zip(axes, ["fit", "devtest"], strict=True):
            value = saved["seismic_spline"][role]["native_median"]
            metallicity_artist = ax.scatter(
                value[:, 0],
                value[:, 1],
                c=value[:, 2],
                s=6,
                cmap="viridis",
                vmin=-2.5,
                vmax=0.5,
                rasterized=True,
            )
            ax.set(
                xlabel="Common Teff [K]",
                ylabel="Common logg [dex]",
                title=f"{role.upper()} role: {len(value):,} plotted stars",
            )
        axes[0].invert_xaxis()
        axes[0].invert_yaxis()
        fig.colorbar(
            metallicity_artist,
            ax=axes.ravel().tolist(),
            label="Common predicted [Fe/H] [dex]",
            extend="both",
            shrink=0.85,
        )
        fig.suptitle(
            "Completed seismic-anchored spline mapping pilot; large fit is separate"
        )
        fig.savefig(a.output / "seismic-spline-kiel-fit-dev-metallicity.png", dpi=170)
        plt.close(fig)
        ids = identities["corner"]
        plots = []
        for row in range(min(2, len(ids))):
            # Identical plot limits across models, preserving genuine within-star draws.
            pooled = np.concatenate(
                [
                    display_coordinates(saved[k]["corner"]["native_draws"][:, row])
                    for k in KINDS
                ]
            )
            limits = np.quantile(pooled, [0.002, 0.998], axis=0).T
            for kind in KINDS:
                filename = f"{kind}-{int(ids[row])}-single-star.png"
                corner(
                    saved[kind]["corner"]["native_draws"][:, row],
                    f"{kind}: single-star coupled six-target flow draws\nGaia DR3 {int(ids[row])}; selected stress star, not representative prevalence",
                    a.output / filename,
                    limits=limits,
                )
                plots.append(filename)
        lines = [
            "# Mapping control readout",
            "",
            "Frozen common encoder; source maps and NSF head trained. These are the historical smaller-source pilots, not the expanded large run. Labels and posterior teachers keep their source definitions.",
            "",
            "| Map/gauge | TUNE macro CRPS | Seismic DEV MAE | Seismic DEV bias | DEV C90 |",
            "|---|---:|---:|---:|---:|",
        ]
        for kind, item in m["models"].items():
            g = item["seismic"]["devtest"]
            lines.append(
                f"| {kind} | {item['tune_macro_crps']:.6f} | {g['mae']:.5f} | {g['bias']:.5f} | {g['coverage90']:.3f} |"
            )
        lines += [
            "",
            "TUNE scores select the candidate; DEV is descriptive. The three seismic map families are close in score. No common-space truth is established by an equal-weight gauge. The original no-DESI-g arm also changed gravity-stratified sampling; the separate matched on/off experiment is required to isolate supervision.",
            "",
            "![Seismic-anchored pilot FIT/DEV, colored by common Fe/H](seismic-spline-kiel-fit-dev-metallicity.png)",
            "",
            "All Kiel panels share the same common-prediction metallicity scale; colour-bar arrows denote values outside the displayed -2.5 to +0.5 dex colour range.",
            "",
            "![Same FIT stars](common-kiel-fit.png)",
            "",
            "![Same DEV stars](common-kiel-devtest.png)",
            "",
            "The following are single-star corner plots from genuine coupled draws, with identical limits across models for each identity. They are selected stress stars. Age and distance are posterior-teacher-scale predictions; six-dimensional physical dependence is not fully identified by sparse labels. No source-native label is overlaid as common-reference truth.",
            "",
        ]
        for name in plots:
            lines += [f"![{name}]({name})", ""]
        (a.output / "report.md").write_text("\n".join(lines))
        m.update(
            status="complete",
            plots=plots,
            fit_stars=len(identities["fit"]),
            dev_stars=len(identities["devtest"]),
            scope="Pure saved-array plotting; no model execution. Same identities across every control. Single-star draws and population medians are distinct objects. Native-error-convolved seismic coverage, not physical joint certification.",
        )
        save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
