import csv
import json
from pathlib import Path

import numpy as np

root = Path("experiments/xp/results-stellar-improvement")
rows = []
names = [
    "feh_clean_frozen_huber",
    "feh_clean_sft_huber",
    "feh_clean_recon001",
    "feh_clean_balanced",
    "feh_clean_balanced_corrected",
    "feh_clean_sft_nll",
    "feh_clean_sky_nll",
]
refs = {}
for name in names:
    p = root / name
    m = json.loads((p / "manifest.json").read_text())
    assert m["status"] == "complete"
    verify = json.loads((p / "checkpoint_verification.json").read_text())
    assert verify["encoder_and_head_reload_exact"] and verify["parent_unchanged"]
    with np.load(p / "predictions.npz") as a:
        d = {k: a[k].copy() for k in a.files}
    group = "sky" if name == "feh_clean_sky_nll" else "identity"
    if group not in refs:
        refs[group] = d
    for k in d:
        if k != "test_prediction" and k != "tune_prediction":
            np.testing.assert_array_equal(d[k], refs[group][k])
    y = d["test_truth"]
    pred = d["test_prediction"]
    err = pred - y
    w = d["test_population_weight"]
    survey = d["test_survey"]
    masks = {
        "all": np.ones(len(y), dtype=bool),
        "FeH_lt_minus2": y < -2,
        "FeH_lt_minus3": y < -3,
        "FeH_ge_minus2": y >= -2,
    }
    for s in np.unique(survey):
        masks["survey_" + s] = survey == s
        masks["survey_" + s + "_FeH_lt_minus2"] = (survey == s) & (y < -2)
    for label, sel in masks.items():
        n = int(sel.sum())
        for weighting in ["unweighted", "inverse_cap_inclusion"]:
            ww = np.ones(n) if weighting == "unweighted" else w[sel]
            rows.append(
                dict(
                    run_id=name,
                    holdout=group,
                    subset=label,
                    weighting=weighting,
                    n=n,
                    mae_dex=float(np.average(abs(err[sel]), weights=ww)) if n else None,
                    bias_dex=float(np.average(err[sel], weights=ww)) if n else None,
                    rmse_dex=float(np.sqrt(np.average(err[sel] ** 2, weights=ww)))
                    if n
                    else None,
                )
            )
    for threshold in [-2.0, -3.0]:
        truth = y < threshold
        candidate = pred < threshold
        for weighting, ww in [
            ("unweighted", np.ones(len(y))),
            ("inverse_cap_inclusion", w),
        ]:
            tp = float(ww[truth & candidate].sum())
            nt = float(ww[truth].sum())
            nc = float(ww[candidate].sum())
            rows.append(
                dict(
                    run_id=name,
                    holdout=group,
                    subset=f"selection_FeH_lt_{threshold:g}",
                    weighting=weighting,
                    n=len(y),
                    true_count=int(truth.sum()),
                    predicted_count=int(candidate.sum()),
                    true_positive_count=int((truth & candidate).sum()),
                    completeness=tp / nt if nt else None,
                    purity=tp / nc if nc else None,
                )
            )
fields = list(dict.fromkeys(k for r in rows for k in r))
with (root / "native-conditional-metrics.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
report = dict(
    metrics=rows,
    comparison_scope="Six matched source-ID holdout runs; sky-NLL has a separate holdout and is reported separately",
    pairing_verification="native-pairing-verification.json",
    caveats=[
        "One seed;184source-ID test stars haveFeH<-2.",
        "No independent confirmation of absence from unsupervised pretraining.",
        "Native pilot featurepool holdouts differ from strict10Mbenchmark validation.",
        "Inverse cap-inclusion weights restore the uncapped featurepool, not the entire Galactic population.",
        "Survey scales may differ; metallicity errors are reported against selected catalogue labels, not independent absolute truth.",
        "NLL uncertainty calibration cannot be established from mean predictions alone.",
    ],
)
(root / "native-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
lines = [
    "# Native metallicity pilot comparison",
    "",
    "Paired source-ID holdouts; all seven runs completed, with the sky split reported separately. Errors are in dex. One seed and only 184 source-ID test stars below [Fe/H] = -2; these results do not establish SOTA.",
    "",
    "At [Fe/H] < -2:",
]
for r in rows:
    if r["subset"] == "FeH_lt_minus2" and r["weighting"] == "unweighted":
        lines.append(
            f"- {r['run_id']}: n={r['n']}, MAE={r['mae_dex']:.4f}, bias={r['bias_dex']:+.4f}."
        )
lines += [
    "",
    "Population effects on the matched source-ID test set (inverse cap-inclusion weighting):",
]
for r in rows:
    if r["subset"] == "all" and r["weighting"] == "inverse_cap_inclusion":
        lines.append(
            f"- {r['run_id']}: MAE={r['mae_dex']:.4f} dex ({r['holdout']} holdout)."
        )
lines += [
    "",
    "Full per-survey, metal-poor conditional and threshold purity/completeness metrics are in native-conditional-metrics.csv. Compare recon/balancing against clean_sft_huber; the noisy NLL objective is different.",
    "",
] + report["caveats"]
(root / "native-comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
