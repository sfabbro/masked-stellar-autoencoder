"""Incremental paired saved-array report for matched seismic supervision arms."""

import argparse
import hashlib
import io
import zipfile
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
SUITES = [
    "seismic_supervision_joint_s1_20261107_v1",
    "seismic_supervision_joint_s2_20261107_v1",
    "seismic_supervision_frozen_s1_20261107_v1",
]


def digest(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def summary(residual):
    r = np.asarray(residual, dtype="f8")
    if not len(r):
        return dict(n=0)
    return dict(
        n=len(r),
        mae=float(abs(r).mean()),
        median_bias=float(np.median(r)),
        nmad=float(1.4826 * np.median(abs(r - np.median(r)))),
    )


def paired_ci(delta, seed=2026110799):
    d = np.asarray(delta, dtype="f8")
    if not len(d) or not np.isfinite(d).all():
        raise ValueError("Finite nonempty paired residual differences required")
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(2000)])
    return dict(
        mean=float(d.mean()),
        ci95=np.quantile(boots, [0.025, 0.975]).tolist(),
        scope="Paired-star bootstrap conditional on these fixed training seeds; not a confidence interval over training randomness or source systematic uncertainty",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--benchmarks", type=Path, required=True)
    p.add_argument("--reuse-waiting-output", action="store_true")
    a = p.parse_args()
    if a.output.exists() and a.reuse_waiting_output:
        previous = json.loads((a.output / "manifest.json").read_text())
        if (
            previous.get("status") != "waiting_for_matched_experiments"
            or previous.get("suites")
            or list(a.output.glob("*.png"))
            or list(a.output.glob("*.csv"))
        ):
            raise ValueError("Cannot replace a report after readout processing")
        attempt = 1
        while (a.output / f"pre-readout-attempt-{attempt}.json").exists():
            attempt += 1
        (a.output / f"pre-readout-attempt-{attempt}.json").write_bytes(
            (a.output / "manifest.json").read_bytes()
        )
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    state = dict(
        status="waiting_for_matched_experiments",
        training_performed=False,
        model_loaded=False,
        suites={},
        comparisons={},
        scope="Fixed APOGEE gravity convention for all controlled arms, not the production seismic gauge. Neutral start before explicit APOKASC supervision. FIT only for gradients; terminal-budget TUNE and DEV diagnostics. CAL is not opened. No automatic production replacement.",
    )
    cache = {}

    def save():
        tmp = a.output / "manifest.tmp"
        tmp.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
        tmp.replace(a.output / "manifest.json")

    def write_report():
        lines = [
            "# Matched seismic supervision experiments",
            "",
            state["scope"],
            "",
            "Rows appear as terminal readouts finish. Positive paired ΔMAE means worse than seismic-off. Confidence intervals resample stars and do not capture uncertainty in instrument calibration or training variance.",
            "",
            "| Suite | Arm | TESS N | TESS logg MAE | TESS median bias | APOKASC common MAE | Native nonseismic TUNE macro CRPS |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for suite, result in state["suites"].items():
            for arm, v in result.get("readouts", {}).items():
                if arm == "baseline":
                    continue
                t = v["tess"]
                s = v["seismic_common"]
                lines.append(
                    f"| {suite} | {arm} | {t['n']} | {t.get('mae', float('nan')):.5f} | {t.get('median_bias', float('nan')):+.5f} | {s.get('mae', float('nan')):.5f} | {v['tune_nonseismic_macro_crps']:.6f} |"
                )
        lines += [
            "",
            "TESS scaling-reference comparisons are diagnostic, not independent absolute truth. APOKASC common errors are shown separately from source-projected scores: a learned calibration can alter source agreement without certifying common physical gravity. Noise-floor arms add 0.02 dex independent per-star scatter; this does not represent a shared instrument zero-point uncertainty. Calibration offset/slope/context coefficients are regularized point estimates.",
            "",
            "![TESS paired residual distributions](tess-residuals.png)",
            "",
            "![Gravity-dependent residuals](tess-gravity-trends.png)",
            "",
            "[Paired comparisons](paired-comparisons.csv) · [Detailed results](manifest.json)",
        ]
        (a.output / "report.md").write_text("\n".join(lines) + "\n")

    save()
    write_report()
    benchmark = pd.read_csv(a.benchmarks, dtype={"source_id": "Int64"})
    state["benchmark_rows_without_gaia_id"] = int(benchmark.source_id.isna().sum())
    benchmark = benchmark.loc[benchmark.source_id.notna()].copy()
    clusters = {}
    archive = ROOT / "expanded_six_source_20261107_v2/clusters.zip"
    if archive.exists():
        if (
            digest(archive)
            != "9eafa2e10afc031790cfe36ed8545aa6336cc87c317fab167e9cea9dacbd0748"
        ):
            raise ValueError("Cluster membership digest differs")
        with zipfile.ZipFile(archive) as z:
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
                keep = ((tab.qflag.to_numpy(dtype=int) & 2) != 0) & (
                    tab.memberprob.to_numpy() > 0.9
                )
                clusters[Path(member).stem] = tab.source_id.to_numpy()[keep]
    # Rows retain independently listed benchmark references; duplicates are not silently averaged.
    deadline = time.monotonic() + 72 * 3600
    while time.monotonic() < deadline:
        changed = False
        terminal = True
        for suite in SUITES:
            d = ROOT / suite
            try:
                upstream = json.loads((d / "manifest.json").read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                upstream = {}
            state["suites"].setdefault(suite, dict(readouts={}))
            result = state["suites"][suite]
            result.update(
                status=upstream.get("status", "not_yet_admitted"),
                error=upstream.get("error"),
                completed_updates={
                    k: v.get("completed_updates", 0)
                    for k, v in upstream.get("arms", {}).items()
                },
            )
            if upstream.get("status") not in ["seismic_experiments_complete", "failed"]:
                terminal = False
            if not (d / "panel.npz").exists():
                continue
            if not upstream.get("panel_sha256"):
                continue
            if digest(d / "panel.npz") != upstream["panel_sha256"]:
                raise ValueError("Diagnostic panel digest differs")
            with np.load(d / "panel.npz") as f:
                panel = {k: f[k] for k in f.files}
            targets = dict(
                tess=(
                    panel["tess_source_id"],
                    panel["tess_reference"],
                    panel["tess_error"],
                )
            )
            for arm, item in [
                (
                    "baseline",
                    dict(
                        status="complete",
                        readout_sha256=upstream.get("baseline_readout_sha256"),
                    ),
                )
            ] + list(upstream.get("arms", {}).items()):
                if not item.get("readout_sha256") or arm in result["readouts"]:
                    continue
                path = d / arm / "common.npz"
                if digest(path) != item["readout_sha256"]:
                    raise ValueError("Readout digest differs")
                with np.load(path) as f:
                    common = {k: f[k] for k in f.files}
                np.testing.assert_array_equal(common["source_id"], panel["source_id"])
                index = {int(v): i for i, v in enumerate(common["source_id"])}
                ti = np.array([index[int(s)] for s in targets["tess"][0]])
                if not (common["role"][ti] == b"devtest").all():
                    raise ValueError("Non-DEV TESS reference entered diagnostic")
                tr = common["median"][ti, 1] - targets["tess"][1]
                gpath = d / arm / "devtest-APOKASC3_loggSeis.npz"
                with np.load(gpath) as f:
                    gr = {k: f[k] for k in f.files}
                gi = np.array([index[int(s)] for s in gr["source_id"]])
                truth = gr["reference"][:, 1] * gr["scale"][1] + gr["center"][1]
                sr = common["median"][gi, 1] - truth
                noise = (
                    np.random.default_rng(2026110797).normal(
                        size=common["logg_draws"][:, ti].shape
                    )
                    * targets["tess"][2][None]
                )
                obs = common["logg_draws"][:, ti] + noise
                lo, hi = np.quantile(obs, [0.05, 0.95], axis=0)
                native = json.loads((d / arm / "readout.json").read_text())
                tune = [
                    v
                    for rec in native["source_native_scores"]
                    if rec["role"] == "tune"
                    and rec["source"] != len(upstream["source_registry"]) - 1
                    for v in rec["per_target_crps"]
                    if v is not None
                ]
                info = dict(
                    tess={
                        **summary(tr),
                        "quoted_noise_reference_inclusion90": float(
                            (
                                (targets["tess"][1] >= lo) & (targets["tess"][1] <= hi)
                            ).mean()
                        ),
                        "median_width90": float(np.median(hi - lo)),
                    },
                    seismic_common=summary(sr),
                    tune_nonseismic_macro_crps=float(np.mean(tune)),
                    source_native_scores=native["source_native_scores"],
                    seismic_coefficients=native["seismic_coefficients"],
                    benchmark={},
                )
                for label, j in [
                    ("Teff", 0),
                    ("logg", 1),
                    ("[FeI/H]", 2),
                    ("[FeII/H]", 2),
                ]:
                    selected = benchmark.source_id.isin(
                        common["source_id"]
                    ) & np.isfinite(pd.to_numeric(benchmark[label], errors="coerce"))
                    rows = benchmark[selected]
                    bi = np.array([index[int(s)] for s in rows.source_id])
                    y = pd.to_numeric(rows[label]).to_numpy()
                    info["benchmark"][label] = summary(common["median"][bi, j] - y)
                info["clusters"] = {}
                for name, members in clusters.items():
                    keep = np.isin(common["source_id"], members)
                    if keep.sum() < 10:
                        continue
                    if (common["role"][keep] == b"fit").any():
                        raise ValueError(
                            "Reserved globular members unexpectedly in FIT"
                        )
                    feh = common["median"][keep, 2]
                    info["clusters"][name] = dict(
                        n=int(keep.sum()),
                        median_feh=float(np.median(feh)),
                        nmad_feh=float(1.4826 * np.median(abs(feh - np.median(feh)))),
                        scope="Population of predicted stellar medians; narrower scatter alone is not an accuracy criterion",
                    )
                result["readouts"][arm] = info
                cache[(suite, arm)] = dict(
                    tess_residual=tr,
                    tess_reference=targets["tess"][1],
                    seismic_residual=sr,
                    tess_ids=targets["tess"][0],
                    seismic_ids=gr["source_id"],
                    median=common["median"],
                    ids=common["source_id"],
                )
                changed = True
            arms = upstream.get("arms", {})
            finished = [v for v in arms.values() if v.get("status") == "complete"]
            if finished and len({v["batch_identity_sha256"] for v in finished}) != 1:
                raise ValueError("Paired batch identity hash mismatch")
            if (
                finished
                and len({v["reconstruction_sequence_sha256"] for v in finished}) != 1
            ):
                raise ValueError("Paired reconstruction identity hash mismatch")
        if changed:
            state["status"] = "reporting_partial_results"
            comparisons = []
            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout="constrained")
            trendfig, taxes = plt.subplots(
                1, 3, figsize=(16, 4.5), layout="constrained"
            )
            for col, suite in enumerate(SUITES):
                for (s, arm), v in cache.items():
                    if s != suite:
                        continue
                    bins = np.linspace(-0.6, 0.6, 181)
                    axes[col].hist(
                        v["tess_residual"],
                        bins=bins,
                        density=True,
                        histtype="step",
                        label=arm,
                    )
                    trend = []
                    for low, high in zip(
                        [0, 1.5, 2, 2.5, 3], [1.5, 2, 2.5, 3, 5], strict=True
                    ):
                        take = (v["tess_reference"] >= low) & (
                            v["tess_reference"] < high
                        )
                        if take.sum() >= 20:
                            trend.append(
                                ((low + high) / 2, np.median(v["tess_residual"][take]))
                            )
                    if trend:
                        tt = np.array(trend)
                        taxes[col].plot(tt[:, 0], tt[:, 1], marker="o", label=arm)
                    off = cache.get((suite, "off"))
                    if off is not None and arm not in ["baseline", "off"]:
                        for ref in ["tess", "seismic"]:
                            np.testing.assert_array_equal(
                                v[ref + "_ids"], off[ref + "_ids"]
                            )
                            ci = paired_ci(
                                abs(v[ref + "_residual"]) - abs(off[ref + "_residual"])
                            )
                            comparisons.append(
                                dict(suite=suite, arm=arm, reference=ref, **ci)
                            )
                axes[col].set(
                    title=suite.replace("seismic_supervision_", ""),
                    xlabel="Common logg − TESS scaling reference [dex]",
                    ylabel="Density in displayed range",
                )
                taxes[col].set(
                    title=suite.replace("seismic_supervision_", ""),
                    xlabel="TESS reference logg",
                    ylabel="Median residual [dex]",
                )
                axes[col].legend(fontsize=6)
                if taxes[col].get_legend_handles_labels()[0]:
                    taxes[col].legend(fontsize=6)
                taxes[col].axhline(0, color="black", lw=0.5)
            fig.savefig(a.output / "tess-residuals.png", dpi=150)
            plt.close(fig)
            trendfig.savefig(a.output / "tess-gravity-trends.png", dpi=150)
            plt.close(trendfig)
            state["comparisons"] = comparisons
            pd.json_normalize(comparisons).to_csv(
                a.output / "paired-comparisons.csv", index=False
            )
            save()
            write_report()
        if terminal:
            state["status"] = (
                "seismic_comparison_complete"
                if all(
                    v["status"] == "seismic_experiments_complete"
                    for v in state["suites"].values()
                )
                else "partial_experiments_failed"
            )
            save()
            write_report()
            return
        save()
        time.sleep(45)
    state["status"] = "timed_out_waiting_for_experiments"
    save()
    write_report()


if __name__ == "__main__":
    main()
