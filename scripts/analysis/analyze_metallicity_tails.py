"""Paired metallicity-tail audit of immutable saved representation predictions.

Catalogue FeH labels define the fixed strata; they are not noiseless intrinsic
truth. Star bootstrap intervals are conditional evaluation-sample uncertainty,
not training-seed, label-systematic, or survey-shift uncertainty.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
MONITOR = REPO / "experiments/xp/monitor"
BINS = (-np.inf, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, np.inf)
THRESHOLDS = (-1.0, -2.0, -2.5)
METHODS = ("ridge", "frozen_head", "finetuned_encoder_head_exploratory")
SELECTED = (
    "control_s42",
    "control_s43",
    "control_s44",
    "spectral_s42",
    "spectral_s43",
    "spectral_s44",
    "photometry_spectral_s42",
    "photometry_spectral_s43",
    "photometry_spectral_s44",
    "spectral_lr05_s42",
    "spectral_latent_views_s42",
    "multiscale_spectral_s42",
)


def load_helper(name):
    spec = importlib.util.spec_from_file_location(
        "metallicity_" + name, MONITOR / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


paired = load_helper("compare_representation_followup")
new = load_helper("compare_new_representations")
latest_helpers = load_helper("compare_latest_standard")


def error_metrics(prediction, target):
    residual = np.asarray(prediction, dtype=np.float64) - np.asarray(
        target, dtype=np.float64
    )
    if not np.isfinite(residual).all():
        raise ValueError("Nonfinite metallicity truth/predictions")
    if not len(residual):
        return {
            "count": 0,
            "mae_dex": None,
            "rmse_dex": None,
            "bias_dex": None,
            "scatter_std_dex": None,
            "scatter_mad_dex": None,
            "catastrophic_fraction_abs_gt_0p5dex": None,
        }
    return {
        "count": len(residual),
        "mae_dex": float(np.abs(residual).mean()),
        "rmse_dex": float(np.sqrt(np.mean(residual**2))),
        "bias_dex": float(residual.mean()),
        "scatter_std_dex": float(residual.std(ddof=1)) if len(residual) > 1 else None,
        "scatter_mad_dex": float(
            1.4826 * np.median(np.abs(residual - np.median(residual)))
        ),
        "catastrophic_fraction_abs_gt_0p5dex": float((np.abs(residual) > 0.5).mean()),
    }


def paired_bootstrap(
    prediction, target, baseline, *, seed, replicates=1000, minimum_count=20
):
    truth = np.asarray(target, dtype=np.float64)
    delta = np.abs(np.asarray(prediction, dtype=np.float64) - truth) - np.abs(
        np.asarray(baseline, dtype=np.float64) - truth
    )
    if not np.isfinite(delta).all():
        raise ValueError("Nonfinite paired bootstrap errors")
    result = {
        "count": len(delta),
        "mae_difference_dex": float(delta.mean()) if len(delta) else None,
        "ci95_dex": None,
        "bootstrap_replicates": replicates,
        "adequate_count_for_interval": len(delta) >= minimum_count,
    }
    if len(delta) < minimum_count:
        result["interval_caveat"] = (
            f"Fewer than {minimum_count} stars; interval deliberately not reported"
        )
        return result
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, replicates, 64):
        rows = rng.integers(len(delta), size=(min(64, replicates - start), len(delta)))
        estimates.append(delta[rows].mean(axis=1))
    result["ci95_dex"] = np.quantile(np.concatenate(estimates), [0.025, 0.975]).tolist()
    result["interval_caveat"] = (
        "Pointwise paired individual-star bootstrap, conditional on fitted models and catalogue labels; no multiple-comparison correction, spatial blocks or training/label-systematic uncertainty"
    )
    return result


def selection_metrics(prediction, target, threshold, minimum_count=20):
    prediction, target = np.asarray(prediction), np.asarray(target)
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite metallicity selection payload")
    actual, selected = target < threshold, prediction < threshold
    tp, fp = int((actual & selected).sum()), int((~actual & selected).sum())
    fn, tn = int((actual & ~selected).sum()), int((~actual & ~selected).sum())
    return {
        "threshold_dex": threshold,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "true_metal_poor_count": tp + fn,
        "selected_count": tp + fp,
        "completeness": tp / (tp + fn) if tp + fn else None,
        "purity": tp / (tp + fp) if tp + fp else None,
        "adequate_counts": tp + fn >= minimum_count and tp + fp >= minimum_count,
        "counts_caveat": "Completeness/purity are descriptive when either denominator is below the minimum count",
    }


def strata(target):
    result = {
        f"[{lo},{hi})": (target >= lo) & (target < hi)
        for lo, hi in zip(BINS[:-1], BINS[1:], strict=True)
    }
    result.update(
        {f"FeH<{threshold:g}": target < threshold for threshold in THRESHOLDS}
    )
    result["all"] = np.ones(len(target), dtype=bool)
    return result


def predictions(group):
    budgets = group["results"]["targets"]["fe_h"]["budgets"]
    largest = max(map(int, budgets))
    arrays = group["arrays"]
    return {
        "ridge": arrays["fe_h_ridge"],
        "frozen_head": arrays[f"fe_h_head_{largest}"],
        "finetuned_encoder_head_exploratory": arrays[f"fe_h_finetuned_{largest}"],
    }, largest


def verified_groups(args, split):
    entries = yaml.safe_load(args.analysis_spec.read_text())["analyses"]
    settings = {item["run_id"]: item for item in entries}
    groups = {}
    previous = [
        item["run_id"]
        for item in entries
        if item.get("mode") == "probes" and not item.get("depends_on")
    ]
    for name in previous:
        group = paired.load_group(args.root, settings[name], split)
        if group is None:
            raise ValueError("Missing completed earlier probe " + name + "/" + split)
        groups[name] = group
    for name in SELECTED:
        key = "probe_new_" + name
        group = new.load_dependent(args.root, settings[key], split)
        if group is None:
            raise ValueError(
                "Missing independently verified dependent probe " + key + "/" + split
            )
        groups[key] = group
    latest_settings = latest_helpers.latest_settings(args.latest_spec)
    latest = latest_helpers.load_latest(args.latest_root, split, latest_settings)
    if latest is None:
        raise ValueError("Missing latest standard completed probe " + split)
    groups["latest_standard57"] = latest
    baseline = groups["probe_new_control_s42"]
    for group in groups.values():
        paired.verify_probe(group, baseline)
        np.testing.assert_array_equal(
            group["arrays"]["fe_h_source_ids"], baseline["arrays"]["fe_h_source_ids"]
        )
        np.testing.assert_array_equal(
            group["arrays"]["fe_h_truth"], baseline["arrays"]["fe_h_truth"]
        )
    return groups


def seed_for(seed, split, region):
    offset = int.from_bytes(
        hashlib.sha256((split + ":" + region).encode()).digest()[:4], "little"
    )
    return (seed + offset) % (2**32)


def audit_split(args, split):
    groups = verified_groups(args, split)
    reference = groups["probe_new_control_s42"]
    truth = reference["arrays"]["fe_h_truth"]
    ids = reference["arrays"]["fe_h_source_ids"]
    masks = strata(truth)
    result = {
        "test_source_id_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "catalogue_target_sha256": hashlib.sha256(truth.tobytes()).hexdigest(),
        "test_stars": len(truth),
        "stratum_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "models": {},
    }
    all_predictions = {name: predictions(group)[0] for name, group in groups.items()}
    rows = []
    for name, group in groups.items():
        payload, budget = predictions(group)
        training_seed = group.get("training_proof", {}).get("training_seed", 42)
        control_name = (
            f"probe_new_control_s{training_seed}"
            if name.startswith("probe_new_") or name == "latest_standard57"
            else paired.CONTROL
        )
        references = {"matched_control": control_name, "parent47": paired.ANCESTOR}
        if any(fragment in name for fragment in ("lr05", "latent_views", "multiscale")):
            references["spectral_s42"] = "probe_new_spectral_s42"
        model = {
            "checkpoint_sha256": group["manifest"]["checkpoint_sha256"],
            "unequal_pretraining_budget_relative_to_continuations": name
            == "latest_standard57",
            "largest_finite_label_budget": budget,
            "baseline_names": references,
            "methods": {},
        }
        for method, prediction in payload.items():
            metrics = {"strata": {}, "selection": {}}
            for region, mask in masks.items():
                point = error_metrics(prediction[mask], truth[mask])
                comparisons = {}
                for role, baseline_name in references.items():
                    comparisons[role] = paired_bootstrap(
                        prediction[mask],
                        truth[mask],
                        all_predictions[baseline_name][method][mask],
                        seed=seed_for(args.seed, split, region),
                        replicates=args.bootstrap_replicates,
                        minimum_count=args.minimum_count,
                    )
                metrics["strata"][region] = {**point, "paired_comparisons": comparisons}
                flat = {
                    "split": split,
                    "model": name,
                    "method": method,
                    "stratum": region,
                    **point,
                }
                for role, comparison in comparisons.items():
                    flat[role + "_mae_difference_dex"] = comparison[
                        "mae_difference_dex"
                    ]
                    interval = comparison["ci95_dex"]
                    flat[role + "_ci95_lower_dex"] = interval[0] if interval else None
                    flat[role + "_ci95_upper_dex"] = interval[1] if interval else None
                rows.append(flat)
            for threshold in (-2.0, -2.5):
                metrics["selection"][str(threshold)] = selection_metrics(
                    prediction, truth, threshold, args.minimum_count
                )
            model["methods"][method] = metrics
        result["models"][name] = model
    return result, rows


def mean_seed_summaries(result):
    output = {}
    for family in ("control", "spectral", "photometry_spectral"):
        models = [
            result["models"][f"probe_new_{family}_s{seed}"] for seed in (42, 43, 44)
        ]
        summary = {}
        for method in METHODS:
            summary[method] = {}
            for region in result["stratum_counts"]:
                metrics = [
                    model["methods"][method]["strata"][region] for model in models
                ]
                values = [item["mae_dex"] for item in metrics]
                changes = [
                    item["paired_comparisons"]["matched_control"]["mae_difference_dex"]
                    for item in metrics
                ]
                if values[0] is None:
                    continue
                summary[method][region] = {
                    "seeds": [42, 43, 44],
                    "mae_values_dex": values,
                    "mae_mean_dex": float(np.mean(values)),
                    "mae_sample_std_dex": float(np.std(values, ddof=1)),
                    "seed_matched_control_difference_values_dex": changes,
                    "seed_matched_control_difference_mean_dex": float(np.mean(changes)),
                    "seed_matched_control_difference_sample_std_dex": float(
                        np.std(changes, ddof=1)
                    ),
                    "policy": "Descriptive continuation-seed variation, no combined significance assertion",
                }
        output[family] = summary
    return output


def render(report):
    lines = [
        "# Metallicity-tail audit",
        "",
        "All errors are in dex. Strata use the saved catalogue labels, which have measurement and survey systematics. "
        "The builder mixes APOGEE M_H renamed fe_h with GALAH/Li iron-abundance fields, so these are nominal fe_h strata rather than a demonstrated homogeneous chemical scale. "
        "Prediction/source-ID/label pairing, checkpoint identity and evaluation policy were verified before this audit. "
        "Negative paired MAE differences mean lower error than the named baseline. "
        "Pointwise bootstrap 95% intervals resample individual heldout stars 1000 times with a fixed seed; they are not corrected for multiple comparisons and do not include spatial covariance, training-seed uncertainty or label systematics. "
        "The later epoch57 standard model has a larger pretraining budget. No SOTA claim follows from this comparison.",
        "",
    ]
    for split, data in report["splits"].items():
        lines += [
            f"## {split} holdout",
            "",
            "Tail counts: "
            + ", ".join(
                f"{region}: {data['stratum_counts'][region]}"
                for region in ("FeH<-1", "FeH<-2", "FeH<-2.5")
            )
            + ".",
            "",
        ]
        for name in (
            paired.ANCESTOR,
            paired.CONTROL,
            "probe_spectral_auxiliary_mae",
            "probe_observed_photometry_mae",
            "probe_photometry_spectral_mae",
            "probe_anchored_decoder_consistency_mae",
            "probe_corrected_signal_weighted_mae",
            "probe_adam_reset_mae",
            "probe_adam_reset_lr2_mae",
            "probe_new_control_s42",
            "probe_new_spectral_s42",
            "probe_new_spectral_lr05_s42",
            "probe_new_spectral_latent_views_s42",
            "probe_new_multiscale_spectral_s42",
            "latest_standard57",
        ):
            for method in ("ridge", "frozen_head"):
                item = data["models"][name]["methods"][method]
                tail = item["strata"]["FeH<-2"]
                change = tail["paired_comparisons"]["matched_control"]
                interval = change["ci95_dex"]
                ci = (
                    "not reported (small sample)"
                    if interval is None
                    else f"[{interval[0]:+.3f},{interval[1]:+.3f}]"
                )
                selection = item["selection"]["-2.0"]
                completeness = (
                    "undefined"
                    if selection["completeness"] is None
                    else f"{100 * selection['completeness']:.1f}%"
                )
                purity = (
                    "undefined"
                    if selection["purity"] is None
                    else f"{100 * selection['purity']:.1f}%"
                )
                lines.append(
                    f"- **{name} / {method}**, labels <-2: MAE {tail['mae_dex']:.3f}, bias {tail['bias_dex']:+.3f}, catastrophic fraction {100 * tail['catastrophic_fraction_abs_gt_0p5dex']:.1f}%; paired MAE difference versus matched control {change['mae_difference_dex']:+.3f},95%CI {ci}. Selection completeness {completeness}, purity {purity} ({selection['selected_count']} selected; adequate denominator counts: {selection['adequate_counts']})."
                )
        lines += [
            "",
            "All bins, thresholds, three continuation seeds and exploratory200-step encoder+head adaptation results are retained in the JSON/CSV. "
            "Low-count tails are descriptive; final test strata and head budgets were fixed before this analysis.",
            "",
        ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=REPO / "experiments/xp/results-followup-e47"
    )
    parser.add_argument(
        "--analysis-spec", type=Path, default=REPO / "configs/xp-followup-analysis.yaml"
    )
    parser.add_argument(
        "--latest-root",
        type=Path,
        default=REPO
        / "experiments/xp/results-latest-standard-e57/probe_latest_standard57",
    )
    parser.add_argument(
        "--latest-spec",
        type=Path,
        default=REPO / "configs/xp-latest-standard-e57-analysis.yaml",
    )
    parser.add_argument(
        "--output", type=Path, default=REPO / "experiments/xp/science-review-20261015"
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--minimum-count", type=int, default=20)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100 or args.minimum_count < 2:
        parser.error("Require at least100 bootstrap replicates and minimum count>=2")
    report = {
        "status": "complete",
        "benchmark_source_id_sha256": paired.BENCHMARK,
        "feature_scaler_sha256": paired.SCALER,
        "continuation_parent_checkpoint_sha256": paired.PARENT,
        "fixed_catalogue_label_bin_edges": ["-inf", -2.5, -2, -1.5, -1, -0.5, 0, "inf"],
        "fixed_catalogue_label_tail_thresholds": list(THRESHOLDS),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.seed,
        "bootstrap_interval_scope": "pointwise conditional evaluation-star intervals, no multiple-comparison correction or training-seed/label-systematic uncertainty",
        "minimum_count_for_bootstrap_interval": args.minimum_count,
        "scatter_definition": "sample standard deviation and1.4826median absolute deviation of signed residual about median",
        "bias_convention": "prediction minus catalogue label; positive means overestimating metallicity",
        "sota_claim": False,
        "label_definition_caveat": "Builder renames APOGEE M_H/M_H_ERR to fe_h/e_fe_h while GALAH and Li fields denote iron abundance; mixed metallicity definitions need source-aware validation. Code reads stored FITS fe_h, not an XP-to-label lookup.",
        "splits": {},
    }
    rows = []
    for split in ("identity", "sky"):
        data, flat = audit_split(args, split)
        data["three_seed_descriptive"] = mean_seed_summaries(data)
        report["splits"][split] = data
        rows.extend(flat)
        print(
            json.dumps(
                {
                    "split": split,
                    "models": len(data["models"]),
                    "tail_counts": {
                        name: data["stratum_counts"][name]
                        for name in ("FeH<-1", "FeH<-2", "FeH<-2.5")
                    },
                }
            ),
            flush=True,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metallicity-tails.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (args.output / "metallicity-tails.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "metallicity-tails.md").write_text(render(report))


if __name__ == "__main__":
    main()
