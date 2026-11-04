"""Morning report from immutable non-Gaussian predictions; no neural replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kurtosis, skew

NAMES = dict(
    gaussian="Gaussian control",
    mixture="Full-covariance mixture",
    copula="Copula spline flow",
    nsf="Autoregressive NSF",
)
LABELS = ("Teff [K]", "logg [dex]", "[Fe/H] [dex]")
SOURCES = ("APOGEE", "DESI", "GALAH")


def metrics(data, selected):
    y = data["reference"][selected]
    median = data["observed_median"][selected]
    low, high = data["observed_low"][selected], data["observed_high"][selected]
    pit = data["observed_pit"][selected]
    assert np.isfinite(np.stack([y, median, low, high, pit])).all()
    assert (low <= high).all() and ((pit >= 0) & (pit <= 1)).all()
    interval_score = (
        high - low + 20 * np.maximum(low - y, 0) + 20 * np.maximum(y - high, 0)
    )
    result = dict(
        count=len(y),
        mae=float(np.abs(median - y).mean()),
        bias=float((median - y).mean()),
        coverage90=float(((y >= low) & (y <= high)).mean()),
        width90=float((high - low).mean()),
        interval_score90=float(interval_score.mean()),
    )
    result["crps"] = (
        float(data["observed_crps"][selected].mean())
        if "observed_crps" in data
        else None
    )
    return result


def check_gate(gate):
    if gate.get("exact"):
        assert gate["pass_gate"]
        return
    raw = gate["raw"]
    ids = np.asarray(raw["source_id"], dtype="<i8")
    assert hashlib.sha256(ids.tobytes()).hexdigest() == gate["panel_sha256"]
    ess = np.asarray(raw["relative_ess"])
    gaps = np.abs(np.asarray(raw["log_prob"]) - np.asarray(raw["replica_log_prob"]))
    refinement = np.abs(
        np.asarray(raw["log_prob"]) - np.asarray(raw["refined_log_prob"])
    )
    for key, actual in (
        ("relative_ess_median", np.median(ess)),
        ("relative_ess_min", ess.min()),
        ("replica_gap_p95", np.quantile(gaps, 0.95)),
        ("replica_gap_max", gaps.max()),
        ("refinement_gap_p95", np.quantile(refinement, 0.95)),
        ("refinement_gap_max", refinement.max()),
    ):
        np.testing.assert_allclose(actual, gate[key], atol=1e-12)
    passed = (
        np.median(ess) >= 0.1
        and ess.min() >= 0.01
        and np.quantile(gaps, 0.95) <= 0.1
        and np.quantile(refinement, 0.95) <= 0.1
    )
    assert bool(passed) == gate["pass_gate"]


def report(root, output):
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    role = manifest.get("evaluation_role", "tune")
    assert role in {"tune", "cal"}
    states = json.loads((root / "head-status.json").read_text())
    shutil.copyfile(root / "manifest.json", output / "producer-manifest.json")
    shutil.copyfile(root / "head-status.json", output / "head-status.json")
    if (root / "collection.json").exists():
        shutil.copyfile(root / "collection.json", output / "collection.json")
    data, rows, ranking, shapes = {}, [], [], {}
    for kind, state in states.items():
        check_gate(state["initial_gate"])
        for snap in state.get("snapshots", []):
            check_gate(snap["gate"])
        if "selected_gate" in state:
            check_gate(state["selected_gate"])
        if state["status"] != "complete":
            continue
        assert state["selected_gate"]["pass_gate"]
        path = root / (kind + "-" + role + ".npz")
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == state["evaluation"]["prediction_sha256"]
        )
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
        with np.load(path, allow_pickle=False) as archive:
            item = dict(archive)
        assert (
            item["source_id"].dtype == np.int64
            and item["corner_source_id"].dtype == np.int64
        )
        active = item["mask"].any(1)
        nll = item["per_star_nll"][active].mean()
        np.testing.assert_allclose(nll, state["evaluation"]["nll"], atol=1e-11)
        data[kind] = item
        if "score_refinement" in state:
            audit = state["score_refinement"]
            np.testing.assert_array_equal(audit["source_id"], item["source_id"][active])
            np.testing.assert_array_equal(
                audit["first_nll"], item["per_star_nll"][active]
            )
            difference = np.asarray(audit["refined_nll"]) - audit["first_nll"]
            for key, value in (
                ("mean_signed_difference", difference.mean()),
                ("absolute_difference_p95", np.quantile(abs(difference), 0.95)),
                ("absolute_difference_max", abs(difference).max()),
            ):
                np.testing.assert_allclose(audit[key], value, atol=1e-12)
        ranking.append(
            dict(
                head=kind,
                name=NAMES[kind],
                nll=float(nll),
                selected_step=state["best_step"],
                parameters=state["parameters"],
                training_error=state.get("error"),
                score_refinement={
                    k: v
                    for k, v in state.get("score_refinement", {}).items()
                    if k not in {"source_id", "first_nll", "refined_nll"}
                },
                checkpoint_sha256=state["checkpoint_sha256"],
                checkpoint_path=state.get("checkpoint_path")
                or (
                    "/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/"
                    + manifest["run_id"]
                    + "/"
                    + kind
                    + "-selected.pth"
                ),
            )
        )
        target = np.broadcast_to(np.arange(9) % 3, item["mask"].shape)
        for s, source in enumerate(SOURCES):
            for d, label in enumerate(LABELS):
                keep = item["mask"] & (item["source_index"] == s) & (target == d)
                if keep.any():
                    rows.append(
                        dict(
                            head=kind,
                            source=source,
                            target=label,
                            subset="all",
                            **metrics(item, keep),
                        )
                    )
        for label, keep in (
            (
                "hot_native_teff_ge_8000K",
                item["mask"] & (target == 0) & (item["reference"] >= 8000),
            ),
            (
                "native_feh_lt_minus1p5",
                item["mask"] & (target == 2) & (item["reference"] < -1.5),
            ),
        ):
            if keep.any():
                rows.append(
                    dict(
                        head=kind,
                        source="pooled source records",
                        target=label,
                        subset="tail",
                        **metrics(item, keep),
                    )
                )
        for label, star_mask in (
            ("all_110_XP_inputs", item["xp_available_count"] == 110),
            ("no_staged_XP_inputs", item["xp_available_count"] == 0),
        ):
            for d, target_label in enumerate(LABELS):
                keep = item["mask"] & star_mask[:, None] & (target == d)
                if keep.any():
                    rows.append(
                        dict(
                            head=kind,
                            source=label,
                            target=target_label,
                            subset="input availability",
                            **metrics(item, keep),
                        )
                    )
        samples = item["joint_draws"]
        assert np.isfinite(samples).all()
        shapes[kind] = dict(
            marginal_skew=skew(samples, axis=1, bias=False).tolist(),
            marginal_excess_kurtosis=kurtosis(samples, axis=1, bias=False).tolist(),
            nonpositive_teff_fraction=(samples[:, :, 0] <= 0).mean(1).tolist(),
            canonical_teff_low_nonpositive_count=int(
                (item["canonical_low"][:, 0] <= 0).sum()
            ),
            scope="Ten stored illustrations; sample shape diagnostics, not physical covariance or a population normality test",
        )
    paired = {}
    if data:
        reference = next(iter(data.values()))
        for item in data.values():
            for key in (
                "source_id",
                "reference",
                "mask",
                "source_index",
                "xp_available_count",
                "corner_source_id",
            ):
                np.testing.assert_array_equal(reference[key], item[key])
    if "gaussian" in data:
        active = data["gaussian"]["mask"].any(1)
        rng = np.random.default_rng(20261105)
        indices = rng.integers(0, int(active.sum()), size=(2000, int(active.sum())))
        for kind, item in data.items():
            delta = (
                item["per_star_nll"][active] - data["gaussian"]["per_star_nll"][active]
            )
            paired[kind] = dict(
                nll_difference=float(delta.mean()),
                star_bootstrap95=np.quantile(
                    delta[indices].mean(1), [0.025, 0.975]
                ).tolist(),
            )
    ranking.sort(key=lambda x: x["nll"])
    summary = dict(
        run_id=manifest["run_id"],
        producer_status=manifest["status"],
        ranking=ranking,
        paired_vs_gaussian=paired,
        metrics=rows,
        sample_shapes=shapes,
        evaluation_role=role,
        scope=f"Previously assigned {role.upper()}; "
        + (
            "fixed TUNE-selected heads, no CAL fitting or selection; "
            if role == "cal"
            else "snapshot selection on reused TUNE; "
        )
        + "finite-QMC observation integrals for flows, Monte Carlo predictive quantiles/PIT, illustrative source floors and fixed affine survey map; not a physical-calibration certificate",
        checks=dict(
            prediction_sha256="pass",
            npz_crc="pass",
            exact_id_pairing="pass",
            gate_statistics="pass",
            score_reductions="pass",
        ),
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    if rows:
        with (output / "metrics.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    figures = []
    if data:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for d, label in enumerate(LABELS):
            for kind, item in data.items():
                target = np.broadcast_to(np.arange(9) % 3, item["mask"].shape)
                pit = item["observed_pit"][item["mask"] & (target == d)]
                axes[0, d].hist(
                    pit,
                    bins=np.linspace(0, 1, 11),
                    density=True,
                    histtype="step",
                    label=NAMES[kind],
                )
                levels = np.linspace(0.1, 0.98, 23)
                coverage = [
                    ((pit >= (1 - p) / 2) & (pit <= (1 + p) / 2)).mean() for p in levels
                ]
                axes[1, d].plot(levels, coverage, label=NAMES[kind])
            axes[0, d].axhline(1, color="gray", ls="--")
            axes[0, d].set(title=label, xlabel="Observed-label PIT", ylabel="Density")
            axes[1, d].plot([0, 1], [0, 1], color="gray", ls="--")
            axes[1, d].set(
                xlabel="Nominal probability", ylabel="Observed-label coverage"
            )
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(
            f"Joint non-Gaussian heads — matched multi-survey {role.upper()}, illustrative noise floors"
        )
        fig.tight_layout()
        fig.savefig(output / "calibration.png", dpi=160)
        plt.close(fig)
        figures.append("calibration.png")
        candidate = next(
            (r["head"] for r in ranking if r["head"] != "gaussian"), ranking[0]["head"]
        )
        selected = {
            k: data[k]
            for k in dict.fromkeys(["gaussian", candidate, "copula", "nsf"])
            if k in data
        }
        for i, source_id in enumerate(
            next(iter(selected.values()))["corner_source_id"][:3]
        ):
            fig, axes = plt.subplots(3, 3, figsize=(8, 8))
            combined = np.concatenate([v["joint_draws"][i] for v in selected.values()])
            for row in range(3):
                for col in range(3):
                    ax = axes[row, col]
                    if col > row:
                        ax.axis("off")
                        continue
                    for kind, item in selected.items():
                        samples = item["joint_draws"][i]
                        if row == col:
                            ax.hist(
                                samples[:, row],
                                bins=np.linspace(
                                    combined[:, row].min(), combined[:, row].max(), 60
                                ),
                                density=True,
                                histtype="step",
                                label=NAMES[kind],
                            )
                        else:
                            ax.scatter(
                                samples[::4, col],
                                samples[::4, row],
                                s=1,
                                alpha=0.12,
                                label=NAMES[kind],
                            )
                    ax.set_xlim(combined[:, col].min(), combined[:, col].max())
                    if row != col:
                        ax.set_ylim(combined[:, row].min(), combined[:, row].max())
                    if row == 2:
                        ax.set_xlabel(LABELS[col])
                    if col == 0 and row:
                        ax.set_ylabel(LABELS[row])
            axes[0, 0].legend(fontsize=7)
            fig.suptitle(
                f"Gaia {int(source_id)} — stored joint predictive samples\nFixed illustration; relative survey scale, not independent physical truth"
            )
            fig.tight_layout()
            name = f"corner-{int(source_id)}.png"
            fig.savefig(output / name, dpi=150)
            plt.close(fig)
            figures.append(name)
    text = (
        "# Morning report: joint non-Gaussian multi-survey predictions\n\n"
        + summary["scope"]
        + ".\n\n"
    )
    text += "| Head | Joint observation NLL | Selected update | Parameters |\n|---|---:|---:|---:|\n"
    for row in ranking:
        text += f"| {row['name']} | {row['nll']:.5f} | {row['selected_step']} | {row['parameters']} |\n"
    text += f"\nLower NLL is better on this {role.upper()} panel. Gaussian/mixture likelihoods are analytic; flow scores use checked importance integration. These values are not comparable directly to the earlier 1024-star panel.\n\n"
    for kind, state in states.items():
        if state["status"] != "complete" or state.get("error"):
            text += f"- {NAMES[kind]}: {state['status']}; {state.get('error', 'no eligible selected state')}.\n"
    for kind, value in paired.items():
        if kind != "gaussian":
            text += f"\n{NAMES[kind]} minus Gaussian NLL: {value['nll_difference']:.5f}; paired star-bootstrap 95% interval {value['star_bootstrap95']}. This excludes training-seed and source-systematic uncertainty.\n"
    text += "\nDensity families alone do not certify calibration or repair missing information, source-scale errors or weak tail supervision. See the per-source, tail and input-availability rows; narrow intervals are not automatically better.\n\n"
    for name in figures:
        text += f"![Saved prediction diagnostic]({name})\n\n"
    text += "[Detailed metrics](metrics.csv) · [Summary, checkpoints and sample shapes](summary.json) · [Producer manifest](producer-manifest.json) · [All numerical gates and failures](head-status.json).\n"
    (output / "index.md").write_text(text)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report(args.root, args.output)


if __name__ == "__main__":
    main()
