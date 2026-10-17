import csv
import json
from pathlib import Path

import numpy as np

root = Path("experiments/xp/results-stellar-improvement")
load = lambda p: json.loads(p.read_text())


def arrays(p):
    with np.load(p) as a:
        return {k: a[k].copy() for k in a.files}


def validate(name, baseline, initial_equal):
    p = root / name
    b = root / baseline
    m = load(p / "manifest.json")
    bm = load(b / "manifest.json")
    assert m["status"] == "complete"
    v = load(p / "checkpoint_verification.json")
    assert v["encoder_and_head_reload_exact"] and v["parent_unchanged"]
    for k in [
        "parent_sha256",
        "label_file_sha256",
        "scaler_sha256",
        "benchmark_source_id_sha256",
        "split_id_sha256",
        "split_counts",
        "supervised_steps",
        "warmup_steps",
        "feature_columns",
        "label_version",
        "target",
    ]:
        assert m[k] == bm[k], (name, k)
    assert (
        m["parent_sha256"]
        == "9b033ae63a3b87c93dcfeffd989172f7888d029f77143e6cf315cf7a032ab2ca"
    )
    d = arrays(p / "predictions.npz")
    bd = arrays(b / "predictions.npz")
    for k in d:
        if not k.endswith("_prediction"):
            np.testing.assert_array_equal(d[k], bd[k], err_msg=f"{name}/{k}")
    a = arrays(p / "initial_predictions.npz")
    ba = arrays(b / "initial_predictions.npz")
    np.testing.assert_array_equal(a["source_ids"], ba["source_ids"])
    if initial_equal:
        for k in a:
            np.testing.assert_array_equal(a[k], ba[k])
    return m, d, bd


rows = []
base = "feh_frozen_huber"
bm = load(root / base / "manifest.json")
bd = arrays(root / base / "predictions.npz")
features = bm["feature_columns"]
names = [
    "photo_value_no_" + s for s in ["wise", "2mass", "skymapper", "sdss", "ps1", "xp"]
]
groups = {n: load(root / n / "manifest.json")["settings"]["blocked"] for n in names}
common = np.ones(len(bd["test_truth"]), bool)
for g in groups.values():
    common &= bd["test_input_available"][:, [features.index(k) for k in g]].all(axis=1)
for name in names:
    m, d, b = validate(name, base, False)
    cols = [features.index(k) for k in groups[name]]
    av = d["test_input_available"][:, cols]
    y = d["test_truth"]
    w = d["test_population_weight"]
    masks = {
        "all": np.ones(len(y), bool),
        "group_any_available": av.any(axis=1),
        "group_all_available": av.all(axis=1),
        "all_six_groups_complete": common,
        "metalpoor": y < -2,
        "metalpoor_group_all_available": (y < -2) & av.all(axis=1),
    }
    for subset, sel in masks.items():
        n = int(sel.sum())
        for weighting in ["unweighted", "inverse_cap_inclusion"]:
            ww = np.ones(n) if weighting == "unweighted" else w[sel]
            before = (
                float(np.average(abs(b["test_prediction"][sel] - y[sel]), weights=ww))
                if n
                else None
            )
            after = (
                float(np.average(abs(d["test_prediction"][sel] - y[sel]), weights=ww))
                if n
                else None
            )
            rows.append(
                dict(
                    run_id=name,
                    subset=subset,
                    weighting=weighting,
                    n=n,
                    baseline_mae_dex=before,
                    ablated_mae_dex=after,
                    delta_mae_dex=after - before if n else None,
                )
            )
age = []
m, d, b = validate("age_sft_nll", "age_frozen_nll", True)
for name, a in [("age_frozen_nll", b), ("age_sft_nll", d)]:
    for subset, sel in {
        "all": np.ones(len(a["test_truth"]), bool),
        "proxy_age_lt2": a["test_truth"] < 2,
        "proxy_age_2to8": (a["test_truth"] >= 2) & (a["test_truth"] < 8),
        "proxy_age_ge8": a["test_truth"] >= 8,
    }.items():
        err = a["test_prediction"][sel] - a["test_truth"][sel]
        age.append(
            dict(
                run_id=name,
                subset=subset,
                n=int(sel.sum()),
                mae_Gyr=float(abs(err).mean()) if len(err) else None,
                bias_Gyr=float(err.mean()) if len(err) else None,
            )
        )
report = {
    "photometry": rows,
    "age_proxy": age,
    "verification": {
        "all8complete": True,
        "reloads_exact": True,
        "paired_ids_truth_errors_survey_availability_weights": True,
        "paired_parent_scaler_benchmark_label_file_budgets": True,
        "age_initial_predictions_exact": True,
        "photometry_initial_latents_deliberately_different": True,
    },
    "common_complete_footprint_n": int(common.sum()),
    "caveats": [
        "Provisional legacy catalogue labels; native GALAH quality flags unavailable in feature table.",
        "Photometry tests refit a head on a frozen encoder after masking an existing group; they do not retrain the representation or import new surveys.",
        "Each group-complete subset has a different footprint; use all_six_groups_complete for a common footprint comparison, noting its sample size.",
        "Original observed availability is preserved; blocked columns are actually zeroed in runner safe_inputs.",
        "Age targets are APOGEE astroNN-derived proxy ages; no independent age accuracy orSOTA claim.",
        "One seed, fixed500warmup+2000steps. Source-ID label holdouts do not certify unseen unsupervised pretraining.",
        "Inverse cap-inclusion weights restore the uncapped labelled featurepool, not allGalacticstars.",
    ],
}
(root / "delayed-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
for name, data in [
    ("photometry-ablation.csv", rows),
    ("age-proxy-comparison.csv", age),
]:
    with (root / name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=data[0])
        writer.writeheader()
        writer.writerows(data)
lines = [
    "# Existing-photometry and age-proxy pilots",
    "",
    "All eight delayed jobs completed with paired labels, source-ID splits, scaler/parent/benchmark and budgets, plus exact checkpoint reloads. The age pair also has bit-exact initial latent/head arrays. Ablation initial latents differ intentionally because input groups are removed.",
    "",
    "Frozen-encoder metallicity head: all20K test stars, inverse cap-inclusion weights; positive delta means removing the group hurts.",
]
for r in rows:
    if r["subset"] == "all" and r["weighting"] == "inverse_cap_inclusion":
        lines.append(
            f"- {r['run_id']}: {r['baseline_mae_dex']:.5f} -> {r['ablated_mae_dex']:.5f} dex; delta {r['delta_mae_dex']:+.5f}."
        )
lines += [
    "",
    f"All-six-groups-complete common footprint: {common.sum()} test stars. Group-availability and metal-poor subsets are retained in the CSV; do not compare different footprints as survey rankings.",
    "",
    "APOGEE astroNN proxy-age agreement (not independent true-age performance):",
]
for r in age:
    if r["subset"] == "all":
        lines.append(
            f"- {r['run_id']}: MAE {r['mae_Gyr']:.4f} Gyr, bias {r['bias_Gyr']:+.4f} Gyr, n={r['n']}."
        )
lines += [""] + report["caveats"]
(root / "delayed-comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
