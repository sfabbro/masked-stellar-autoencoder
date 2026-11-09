"""Maintain a durable morning report while the laptop is offline."""

import argparse
import datetime as dt
import json
import time
from pathlib import Path

ROOT = Path("/arc/projects/k-pop/msa_runs/stellar-improvement-20261015")
PUBLIC = "https://ws-uv.canfar.net/arc/files/projects/k-pop/msa_runs/stellar-improvement-20261015/"


def read(run):
    try:
        return json.loads((ROOT / run / "manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-live-output", action="store_true")
    p.add_argument("--formulation", type=Path)
    a = p.parse_args()
    if a.output.exists() and a.reuse_live_output:
        prior = json.loads((a.output / "manifest.json").read_text())
        if prior.get("status") != "morning_report_live":
            raise ValueError("Only a reconciled live report may be refreshed")
        generation = 1
        while (a.output / f"report-generation-{generation}").exists():
            generation += 1
        archive = a.output / f"report-generation-{generation}"
        archive.mkdir(exist_ok=False)
        for name in ["manifest.json", "morning-report.md"]:
            (archive / name).write_bytes((a.output / name).read_bytes())
    else:
        a.output.mkdir(parents=True, exist_ok=False)
    if a.formulation:
        (a.output / "common-map-formulation.md").write_bytes(a.formulation.read_bytes())
    deadline = time.monotonic() + 72 * 3600
    while time.monotonic() < deadline:
        now = dt.datetime.now(dt.UTC)
        perth = now.astimezone(dt.timezone(dt.timedelta(hours=8)))
        large = read("expanded_six_nsf_fullfit_20261107_v1")
        recovery = read("expanded_six_source_20261107_v4")
        extension = read("expanded_six_source_20261107_v5")
        source = (
            extension
            if extension.get("training_source_ready")
            else (
                recovery
                if recovery.get("training_source_ready")
                else read("expanded_six_source_20261107_v3")
            )
        )
        census = read("expanded_six_source_census_20261107_v1")
        if source.get("source_extension") and source.get("census"):
            census = dict(source["census"], status="complete")
        status = large.get("status", "not_yet_started")
        lines = [
            "# MSA morning report",
            "",
            f"Updated {perth:%Y-%m-%d %H:%M} Perth ({now:%H:%M} UTC).",
            "",
        ]
        if large.get("training_performed"):
            lines += [
                f"**Large training is running or has run:** {status}; {large.get('completed_updates', 0):,} updates, {large.get('completed_full_fit_epochs', 0)} complete FIT record epochs.",
                "",
            ]
        elif status == "failed":
            lines += [
                f"**Large training stopped before admission or completion:** {large.get('error', 'inspect manifest')}",
                "",
            ]
        else:
            lines += [
                f"**Large training is armed but has not started optimizing:** {status}. The CANFAR job chooses from TUNE results, checks its source, runs CUDA encoder/decoder/head gradient checks, warms new survey maps, then makes full FIT record passes.",
                "",
            ]
        if extension and not extension.get("training_source_ready"):
            lines += [
                f"Final full-raw cohort assembly: {extension.get('status')}. The armed fit waits for source v5, including the recovered additional XP stars and their protected star-level role checks. Counts below describe the last completed parent cohort until that larger source passes.",
                "",
            ]
        if large.get("training_scope"):
            lines += [large["training_scope"], ""]
        audit = read("gaia_parallax_release_unit_audit_20261107_v1")
        repair = source.get("parallax_repair", {})
        if audit.get("pass_gate"):
            lines += [
                "The independent ESA audit matched stored values and formal errors to Gaia DR3 mas in the sampled old/new/negative-parallax groups. All 128 sampled old FIT stars with unavailable stored parallax had measurements at ESA. This is not a Gaia zeropoint or quality certificate.",
                "",
            ]
        if repair:
            fit_astrometry = repair["role_availability"]["fit"]
            lines += [
                f"Parent-cohort parallax inputs repaired by exact ID: FIT availability {fit_astrometry['before']:,} → {fit_astrometry['after']:,}; {fit_astrometry['newly_available']:,} newly available. Every non-astrometry dataset was checked unchanged. Signed negative parallaxes are retained. Formal input errors are stored for QA; the encoder currently has no explicit parallax-precision context.",
                "",
            ]
        elif source:
            lines += [
                f"Full-cohort astrometry recovery: {source.get('status')}; {source.get('raw_rows_scanned', 0):,} raw rows scanned. Training waits for this source revision.",
                "",
            ]
        if recovery and not recovery.get("training_source_ready"):
            lines += [
                f"Gaia archive recovery for remaining raw-file gaps: {recovery.get('status')}; {recovery.get('usable_recovered', 0):,} recovered from {recovery.get('requested_stars', 0):,} requested eligible stars. The armed training candidate waits for source v4.",
                "",
            ]
        elif source.get("esa_parallax_recovery"):
            recovery_summary = source["esa_parallax_recovery"]
            lines += [
                f"Archive recovery added {recovery_summary['recovered_stars']:,} measurements beyond the raw-file recovery. {recovery_summary['still_missing_eligible']:,} usable-input stars still lack an available measurement. Response hashes and exact integer-ID joins are retained.",
                "",
            ]
        lines += [
            "The source-wise loss directly constrains each observed label block. Cross-block dependence, especially atmosphere versus age/distance, is inherited/indirectly constrained; genuine coupled corner draws do not certify calibrated physical joint correlations. [Detailed common-map formulation and evidence](common-map-formulation.md).",
            "",
        ]
        tess = read("tess_k2_local_gaia_candidates_20261107_v1")
        if tess.get("status") == "conservative_seismic_gaia_candidates_complete":
            lines += [
                f"Local motion-aware seismic matching found {tess['matched_by_catalogue']['tess_hon_2021']:,} TESS and {tess['matched_by_catalogue']['k2_gap_dr3']:,} conservative K2 counterparts in the full raw input. {tess['tess_with_nu_scaling_gravity']:,} TESS references have usable νmax scaling gravity. These are catalogue counts, not the number in our model DEV panel. TESS DEV gravity QA is queued by signed parallax S/N, RUWE and mass flag; K2 gravity awaits pipeline-specific solar convention verification. These QA candidates are not training identity promotion.",
                "",
            ]
        matched = read("matched_desi_logg_comparison_20261107_v1")
        if matched.get("status") == "complete":
            result = matched["results"]["devtest"]
            lines += [
                f"Matched fixed-terminal DESI logg on/off comparison on {result['stars']:,} seismic DEV stars: MAE(on) {result['on']['mae']:.5f}, MAE(off) {result['off']['mae']:.5f}; off−on paired 95% interval {result['off_minus_on_mae']['ci95']}. The exact source/star batch hashes agree. This is one seed with the frozen old encoder and historical parallax availability, not proof the catalogue is wrong.",
                "",
            ]
        decision = large.get("decision")
        if decision:
            lines += [
                f"Chosen objective: {decision['objective']}. Mapping: {decision['mapping_run']}. Noise assumption: {decision.get('noise_choice')}",
                "",
                decision["noise"],
                "",
            ]
        if source.get("training_source_ready"):
            lines += [
                f"The source registry contains {source['total_stars']:,} catalogue rows, of which **{sum(source['eligible_role_counts'].values()):,} have usable inputs** and **{source['eligible_role_counts']['fit']:,} are labelled FIT stars**. Registry rows include excluded/no-input objects and must not all be counted as usable stars. Base roles are inherited; new-star roles are stable ID splits. Benchmarks and selected globular members are reserved from this stage. Ancestor exposure is not erased.",
                "",
            ]
        if census.get("status") == "complete":
            fit = census["roles"]["fit"]
            lines += [
                f"These are {fit['labelled_unique_stars']:,} distinct labelled FIT stars, {fit['source_records']:,} source records per epoch, and {fit['complete_six_unique_stars']:,} stars with all six observed targets across sources. These counts are different from repeated training presentations.",
                "",
                "| Target | Distinct FIT stars |",
                "|---|---:|",
            ]
            lines += [
                f"| {name} | {n:,} |"
                for name, n in zip(
                    census["target_order"], fit["unique_stars_per_target"], strict=True
                )
            ]
            lines += ["", "| Source | FIT source records |", "|---|---:|"]
            lines += [
                f"| {name} | {item['fit_records']:,} |"
                for name, item in census["sources"].items()
            ]
            lines += [
                "",
                "The six targets are Teff, logg, Fe/H, alpha/Fe, age and distance. Age and distance remain source-specific posterior-teacher estimands, not independent physical measurement likelihoods. The alpha reference is DESI alpha/Fe; M/H and alpha/M are not silently treated as Fe/H or alpha/Fe.",
                "",
            ]
        lines += [
            "The architecture is the pretrained masked autoencoder, a conditional six-dimensional neural spline flow, and source-native forward maps on supported target subsets. Production fine-tuning keeps masked **MAE reconstruction at weight 0.03** and uses natural Gaia inputs for supervised prediction. The flow supplies flexible marginal distributions and genuine coupled single-star samples.",
            "",
            "The common scale uses an explicit convention for temperature, iron, alpha and posterior teachers, plus FIT-only seismic gravity anchoring. Offset, positive affine and monotone own-target spline maps are compared; nonlinear maps act on samples exactly, without a Taylor approximation. A shared physical reference is not established by averaging surveys.",
            "",
            "| Experiment | State | Updates | TUNE score |",
            "|---|---|---:|---:|",
        ]
        runs = [
            f"nsfconfirm_{n}_20261106_v1"
            for n in [
                "energy_s1",
                "energy_s2",
                "hybrid_s1",
                "hybrid_s2",
                "crps_s1",
                "hybrid_floor003",
                "hybrid_corrpos",
                "hybrid_corrneg",
                "hybrid_studentt",
            ]
        ] + [
            f"sixmap_{n}_20261107_v1"
            for n in [
                "equalweight_affine",
                "apogee_affine",
                "seismic_affine",
                "seismic_nodesig",
                "seismic_offset",
                "seismic_spline",
            ]
        ]
        for run in runs:
            m = read(run)
            score = m.get("tune_score")
            lines.append(
                f"| {run} | {m.get('status', 'waiting')} | {m.get('completed_updates', 0):,} | {score if score is not None else '—'} |"
            )
        covariance = read("native_label_covariance_audit_20261107_v2")
        if covariance.get("status") == "complete":
            lines += [
                "",
                "The paired-survey residual audit finds correlations among atmospheric labels. Relative TUNE negative log score improves with full versus diagonal residual covariance, and improves further with a covariance-matched t4 kernel:",
                "",
                "| Survey pair | Full − diagonal TUNE NLL | t4 − full Gaussian |",
                "|---|---:|---:|",
            ]
            for name, item in covariance["pairs"].items():
                if item.get("status") == "complete":
                    q = item["tune_mean_negative_log_score"]
                    lines.append(
                        f"| {name} | {q['fitted_full'] - q['fitted_diagonal']:.3f} | {q['fitted_full_t4'] - q['fitted_full']:.3f} |"
                    )
            lines += [
                "",
                "These are combined residuals of two surveys, shared errors and remaining bias; they do not identify either survey’s measurement covariance. Correlation/heavy-tail controls test operational noise assumptions. They do not change the reconstruction loss.",
                "",
            ]
        extra = read("remaining_survey_raw_features_20261107_v1")
        lines += [
            f"Additional full-raw Gaia input recovery: **{extra.get('status', 'queued')}**. This addresses admitted label stars outside the current input union, including cache misses; it does not alter the pinned running candidate. Input absence from a cache is not absence of Gaia XP.",
            "",
            "All 174,922,161 Andrae release stars were found in the 218,131,884-row raw input. Catalogue-wide prediction is separate from the smaller exported QA panels. A new first-checkpoint comparison covers available predicted stars; it does not imply the other raw stars cannot be compared.",
            "",
            "Gaia-ESO and MaSTAR remain deferred pending validated identity or source-quality/dependency admission. TESS/K2 identities are not silently promoted. Existing APOKASC and benchmark calibration/teacher dependencies are retained.",
            "",
        ]
        lines += [
            "```mermaid",
            "flowchart LR",
            '  X["Gaia XP + photometry + signed parallax"] --> E["Pretrained masked encoder"]',
            '  E --> F["Conditional six-target NSF"]',
            '  F --> C["Common marginals + coupled per-star draws"]',
            '  C --> H["Survey maps on supported target subsets"]',
            '  H --> Y["Noisy native survey/teacher labels"]',
            '  E --> D["Masked decoder: MAE × 0.03"]',
            "```",
            "",
        ]
        outputs = [
            (
                "Completed map/gauge Kiel and single-star corner report",
                "six_survey_map_report_20261107_v1",
                "report.md",
            ),
            (
                "All-sky NSF inference/storage benchmark",
                "six_nsf_inference_benchmark_20261107_v1",
                "manifest.json",
            ),
            (
                "Independent repaired-parallax audit",
                "gaia_parallax_release_unit_audit_20261107_v2",
                "manifest.json",
            ),
            (
                "TESS DEV gravity QA",
                "tess_first_checkpoint_gravity_qa_20261107_v1",
                "manifest.json",
            ),
            (
                "Matched DESI gravity ablation",
                "matched_desi_logg_comparison_20261107_v1",
                "manifest.json",
            ),
            (
                "Random per-star mode/corner QA",
                "nsf_confirmation_random_modes_20261107_v1",
                "manifest.json",
            ),
            (
                "Corrected CAL interval QA",
                "nsf_confirmation_intervals_20261107_v2",
                "manifest.json",
            ),
            (
                "First-checkpoint benchmark, cluster, Kiel, seismic and corner report",
                "expanded_checkpoint_report_20261107_v1",
                "report.md",
            ),
            (
                "First-checkpoint common PDF exports",
                "expanded_first_checkpoint_qa_20261107_v1",
                "manifest.json",
            ),
            (
                "Large training manifest",
                "expanded_six_nsf_fullfit_20261107_v1",
                "manifest.json",
            ),
        ]
        for title, run, filename in outputs:
            m = read(run)
            lines += [
                f"- [{title}]({PUBLIC}{run}/{filename}) — {m.get('status', 'waiting')}."
            ]
        lines += [
            "",
            "CAL adjusts interval thresholds independently of the fitted density. It does not by itself make the latent six-target PDF physically calibrated. First-checkpoint QAs are development diagnostics and do not select production hyperparameters. No catalogue is automatically published.",
            "",
        ]
        target = a.output / "morning-report.md"
        temp = a.output / "morning-report.tmp"
        temp.write_text("\n".join(lines))
        temp.replace(target)
        state = dict(
            status="morning_report_live",
            updated_utc=now.isoformat(),
            large_training_status=status,
            training_performed=large.get("training_performed", False),
            completed_updates=large.get("completed_updates", 0),
            report="morning-report.md",
        )
        (a.output / "manifest.json").write_text(json.dumps(state, indent=2) + "\n")
        if status == "large_six_candidate_complete":
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
