import json
from pathlib import Path

import numpy as np

root = Path("experiments/xp/results-stellar-improvement")
d = json.loads((root / "survey_overlap_v1/overlap-summary.json").read_text())
a = np.load(root / "survey_overlap_v1/overlap-pairs.npz")
checks = []
for name, summary in d["comparisons"].items():
    get = lambda k: a[name + "__" + k]
    ids = get("source_id")
    split = get("split")
    layer = get("left_feh")
    r = get("right_feh")
    delta = layer - r
    sigma = np.sqrt(get("left_sigma") ** 2 + get("right_sigma") ** 2)
    assert len(np.unique(ids)) == len(ids) == summary["pairs"]
    assert np.all(np.isfinite(delta))
    assert np.all(np.isfinite(sigma) & (sigma > 0))
    assert np.all(np.diff(ids) > 0)
    train = split == 0
    off = float(np.quantile(delta[train], 0.5)) if train.any() else None
    assert off == summary["train_only_median_offset_dex"]
    for si, sl in [(0, "train"), (1, "tune"), (2, "test")]:
        for label, mask in [
            ("all", np.ones(len(ids), bool)),
            ("left_FeH_lt_minus2", layer < -2),
            ("right_FeH_lt_minus2", r < -2),
            ("both_FeH_lt_minus2", (layer < -2) & (r < -2)),
        ]:
            select = (split == si) & mask
            reported = summary["bins"][sl + "_" + label]
            assert int(select.sum()) == reported["n"]
            for shifted in [False, True]:
                target = reported["after_train_offset"] if shifted else reported
                if not select.any() or target is None:
                    continue
                v = delta[select] - (off if shifted else 0)
                med = float(np.quantile(v, 0.5))
                mad = float(np.quantile(abs(v - med), 0.5)) * 1.4826
                for expected, actual in [
                    (med, target["median_delta_dex"]),
                    (mad, target["robust_scatter_dex"]),
                    (float(np.sum(abs(v)) / len(v)), target["mae_delta_dex"]),
                    (
                        float(np.count_nonzero(abs(v) > 3 * sigma[select]) / len(v)),
                        target["fraction_abs_delta_gt_3_reported_sigma"],
                    ),
                ]:
                    np.testing.assert_allclose(expected, actual, atol=1e-12, rtol=1e-12)
    checks.append(dict(pair=name, n=len(ids), all_bins_verified=True))
report = dict(
    array_checks=checks,
    train_only_offset_recomputed=True,
    all_conditional_statistics_recomputed=True,
    source_measurement_checks_pending=True,
)
(root / "survey_overlap_v1/independent-verification.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report))
