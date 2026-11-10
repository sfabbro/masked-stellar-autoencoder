"""Saved-array TESS nu-max gravity QA, split by parallax precision and quality."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def gravity_stats(draws, truth, error):
    if not len(truth):
        return dict(n=0)
    residual = np.median(draws, axis=0) - truth
    noisy = (
        draws
        + np.random.default_rng(2026110788).standard_normal(draws.shape) * error[None]
    )
    lo, hi = np.quantile(noisy, [0.05, 0.95], axis=0)
    return dict(
        n=len(truth),
        median_bias=float(np.median(residual)),
        mae=float(abs(residual).mean()),
        nmad=float(1.4826 * np.median(abs(residual - np.median(residual)))),
        reference_coverage90=float(((truth >= lo) & (truth <= hi)).mean()),
        median_width90=float(np.median(hi - lo)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--qa-run", default="expanded_first_checkpoint_qa_20261107_v1")
    p.add_argument("--reuse-waiting-output", action="store_true")
    a = p.parse_args()
    from saved_array_source_binding import prepare_report_output

    prepare_report_output(
        a.output, ("waiting_for_cuda_arrays", "failed"), a.reuse_waiting_output
    )
    m = dict(
        status="waiting_for_cuda_arrays",
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
        qa = ROOT / a.qa_run
        deadline = time.monotonic() + 48 * 3600
        while time.monotonic() < deadline:
            path = qa / "manifest.json"
            producer = json.loads(path.read_text()) if path.exists() else {}
            if producer.get("status") in ["failed", "upstream_failed"]:
                raise RuntimeError("Checkpoint QA failed")
            if producer.get("status") in ["first_checkpoint_common_qa_complete", "selected_checkpoint_common_qa_complete"]:
                break
            time.sleep(30)
        else:
            raise TimeoutError("Checkpoint arrays unavailable")
        ref_dir = ROOT / "tess_k2_local_gaia_candidates_20261107_v1"
        reference_path = ref_dir / "seismic-gaia-qa-candidates.parquet"
        reference_sha = json.loads((ref_dir / "manifest.json").read_text())[
            "candidate_sha256"
        ]
        if (
            sha(reference_path) != reference_sha
            or producer["tess_candidate_sha256"] != reference_sha
            or sha(qa / "panel.npz") != producer["panel_sha256"]
        ):
            raise ValueError("Identity, reference or panel changed")
        with np.load(qa / "panel.npz") as f:
            panel = {k: f[k] for k in f.files}
        ref = pd.read_parquet(reference_path)
        ref = ref.loc[
            (ref.catalogue == "tess_hon_2021")
            & np.isfinite(ref.logg_nu_scaling)
            & (ref.e_logg_stat > 0)
        ].set_index("source_id")
        selected = np.flatnonzero(panel["tess_devtest"])
        sid = panel["source_id"][selected]
        if len(np.unique(sid)) != len(sid) or not np.all(
            np.isin(sid, ref.index.to_numpy(dtype="i8"))
        ):
            raise ValueError("Exact TESS DEV identities required")
        r = ref.loc[sid]
        truth, error = r.logg_nu_scaling.to_numpy(), r.e_logg_stat.to_numpy()
        parallax = panel["parallax_mas"][selected]
        ep = panel["parallax_error_mas"][selected]
        measured = panel["parallax_observed"][selected].astype(bool)
        snr = np.divide(
            parallax, ep, out=np.full(len(sid), np.nan), where=measured & (ep > 0)
        )
        masks = {
            "all": np.ones(len(sid), bool),
            "parallax_missing": ~measured,
            "parallax_snr_negative": measured & (snr < 0),
            "parallax_snr_0_5": measured & (snr >= 0) & (snr < 5),
            "parallax_snr_5_20": measured & (snr >= 5) & (snr < 20),
            "parallax_snr_ge20": measured & (snr >= 20),
            "ruwe_lt1p4": r.ruwe.to_numpy() < 1.4,
            "ruwe_ge1p4": r.ruwe.to_numpy() >= 1.4,
            "mass_flag_0": r.mass_flag.to_numpy() == 0,
            "mass_flag_1": r.mass_flag.to_numpy() == 1,
        }
        for lo, hi in [(0, 2), (2, 2.5), (2.5, 3), (3, 5)]:
            masks[f"gravity_{lo}_{hi}"] = (truth >= lo) & (truth < hi)
        fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
        for column, (name, binding) in enumerate(producer["models"].items()):
            path = qa / (name + "-common.npz")
            if sha(path) != binding["artifacts"][path.name]["sha256"]:
                raise ValueError("Saved model draws changed")
            with np.load(path) as f:
                np.testing.assert_array_equal(f["source_id"], panel["source_id"])
                draw = f["logg_draws"][:, selected]
            stats = {
                label: gravity_stats(draw[:, take], truth[take], error[take])
                for label, take in masks.items()
            }
            m["models"][name] = dict(
                checkpoint_sha256=binding["checkpoint_sha256"],
                training_update=binding["training_update"],
                slices=stats,
            )
            median = np.median(draw, axis=0)
            axes[0, column].hexbin(truth, median, gridsize=65, mincnt=1, bins="log")
            axes[0, column].plot([0, 5], [0, 5], "k--", lw=1)
            axes[0, column].set(
                xlabel="TESS nu-max scaling logg",
                ylabel="Common predicted logg",
                title=name,
            )
            good = np.isfinite(snr)
            axes[1, column].scatter(snr[good], (median - truth)[good], s=2, alpha=0.2)
            axes[1, column].set(
                xscale="symlog",
                xlabel="Signed parallax / formal error",
                ylabel="Predicted minus TESS logg",
                ylim=(-1, 1),
            )
            np.savez_compressed(
                a.output / (name + "-tess-dev.npz"),
                source_id=sid,
                reference=truth,
                reference_stat_error=error,
                prediction=median,
                parallax_snr=snr,
            )
        fig.savefig(a.output / "tess-gravity-vs-parallax.png", dpi=170)
        plt.close(fig)
        m.update(
            status="complete",
            reference_sha256=reference_sha,
            dev_stars=len(sid),
            scope="Primary QA uses only current-stage DEV stars; neither TESS reference nor quality flag used to fit/select this run. Conservative local positional QA candidates, not authoritative identifier promotion. Historical parent/teacher exposure and shared photometric Teff remain unknown. Raw intervals convolve per-star nu-max/Teff statistical errors; solar and scaling systematics are separate, not independent errors. The Gaia-dependent mass flag is only a secondary slice. K2 gravity excluded pending pipeline conventions. First checkpoint is preliminary, not final.",
        )
        m["snapshot_kind"] = producer.get("snapshot_kind", "intermediate")
        m["producer_manifest_sha256"] = sha(qa / "manifest.json")
        if m["snapshot_kind"] == "final_selected":
            m["scope"] = m["scope"].replace("First checkpoint is preliminary, not final.", "Selected final checkpoint of the older comparison architecture; not the corrected full-Gaia model.")
        save()
    except Exception as exc:
        m.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
