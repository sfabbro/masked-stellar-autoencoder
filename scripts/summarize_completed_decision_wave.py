"""Summarize completed immutable manifests and paired saved seismic scores."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    r = a.directory
    pdf = json.loads((r / "decision6_pdf_qa_worker_20261106_v1.json").read_text())
    seis = json.loads((r / "decision_seismic_pdf_qa_20261106_v1.json").read_text())
    point = json.loads(
        (r / "decision_wave_array_qa_report_20261106_v1.json").read_text()
    )
    mode = json.loads(
        (r / "representative_corner_analysis_20261106_v1.json").read_text()
    )
    rows = []
    for rid, v in pdf["results"].items():
        s = seis["results"][rid]
        m = mode["results"][rid]
        pt = point["results"][rid]
        row = dict(
            run_id=rid,
            tune_score=pt["best_tune_score"],
            energy=v["mean_masked_standardized_energy"],
            seismic_logg_mae=s["mae"],
            seismic_coverage90=s["convolved_coverage90"],
            seismic_crps=s["convolved_crps"],
            random_logg_mode_fraction=m["random_mode_fraction"],
            crps={t["target"]: t["mean_crps"] for t in v["per_target"]},
        )
        rows.append(row)
    pairs = {}
    arrays = {}
    for family in ["gaussian", "mdn", "nsf_energy"]:
        rid = f"decision6_{family}_parallax_20261106_v1"
        with np.load(r / (rid + "-seismic-pdf.npz")) as f:
            arrays[family] = {k: f[k] for k in f.files}
    np.testing.assert_array_equal(
        arrays["nsf_energy"]["source_id"], arrays["mdn"]["source_id"]
    )
    np.testing.assert_array_equal(
        arrays["nsf_energy"]["source_id"], arrays["gaussian"]["source_id"]
    )
    for family in ["mdn", "gaussian"]:
        np.testing.assert_array_equal(
            arrays["nsf_energy"]["logg_seismic"], arrays[family]["logg_seismic"]
        )
        np.testing.assert_array_equal(
            arrays["nsf_energy"]["reference_error"], arrays[family]["reference_error"]
        )
    rng = np.random.default_rng(2026110650)
    n = len(arrays["mdn"]["source_id"])
    idx = rng.integers(n, size=(5000, n))
    for control in ["mdn", "gaussian"]:

        def get(x):
            d = arrays[x]
            error = abs(np.median(d["logg_draws"], axis=0) - d["logg_seismic"])
            return dict(mae=error, crps=d["convolved_crps"].reshape(-1))

        nf, other = get("nsf_energy"), get(control)
        result = {}
        for metric in nf:
            delta = nf[metric] - other[metric]
            ci = np.quantile(delta[idx].mean(1), [0.025, 0.975])
            result[metric] = dict(
                mean_difference=float(delta.mean()),
                paired_star_bootstrap_95=ci.tolist(),
            )
        pairs["nsf_minus_" + control] = result
    result = dict(
        status="complete_saved_array_summary",
        runs=rows,
        paired_seismic_comparisons=pairs,
        bootstrap_scope="Conditional on fitted checkpoints and this development panel; not retraining/seed uncertainty, no independent physical-accuracy guarantee",
        model_loaded=False,
        training_performed=False,
    )
    (r / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
