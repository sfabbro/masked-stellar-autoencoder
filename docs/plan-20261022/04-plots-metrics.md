# 04 — End-product measurement & visualisation suite (plots-audit findings)

Read-only investigation, 2026-10-22. Branch `codex/xp-reconstruction-experiments`, HEAD `1c51c85`
(`docs/plan-20261022/` untracked). All numbers below are read from local files; anything not read
locally is marked **needs verification**. Only this file written. Goal measured: best point + distribution
metrics for Teff, logg, FeH, alpha, distance, age; best on oldest/most-metal-poor/logg; tightest Kiel
diagram; all Gaia `source_id` incl. no-XP.

## 1. Inventory of existing diagnostics

### 1.1 What exists

| Artifact | Content | Schema / convention | Reusable for release? |
|---|---|---|---|
| `docs/reports/msa-programme-review-20261019.html` (21,923 B, tracked) | Narrative review + 10 sibling PNGs | **Hand-maintained**: inline `const pointRows=[…23]`, `const families=[…19]`, `const tests=[…8]`; no data load, no SHA binding | Copy/prose only; numbers drift by hand |
| `docs/reports/msa-density-review-20261022.html` (463,095 B) | Point/distribution/interval/capability review, filters by cohort/label/subset | **Template + injection**: `augment_catalog.py:54` regex-replaces `const DATA=` with JSON; template is tracked, data is not | **Yes** — the pattern to clone |
| `experiments/xp/results-stellar-improvement/density-review-20261022/` | `point.csv` 136×23, `distribution.csv` 182×18, `interval.csv` 486×15, `inventory.csv` 43×10, `capability.csv` 87×11, `metrics.json` (573,574 B), `validation.json`, `reproduce.py` (21,003 B), `augment_catalog.py` | `reproduce.py:1` "No checkpoints loaded, model inference, fitting or jobs"; every row carries `status` + `verified`; `validation.json` = `{status:"passed", checks:[7], visual_check:"passed…"}` | **Yes** — this is the reference implementation of the metric layer |
| `*-verified-comparison/` dirs (5: `imbalance`, `mechanism-v2`, `targeted-{conditional-flow,mse-seeds,parent}`, plus per-run `…/verified-point-cp-comparison/`) | `summary.csv` 30×11 (`method,subset,label,unit,n,mae,bias,rmse,marginal_90_coverage,marginal_mean_width,bonferroni_joint_90_coverage`), `conditional-survey.csv` 195×7, `report.json` (`status,rows,below_minus2_rows,training_presentations,selected_presentations,selection,summary[30],matched_ttt[6],bindings{run:{file:sha256}},limits[8]}`), `paired-test-predictions.csv` (36 MB), `visual-review.json`, `reproduce.py`, PNG+PDF | **Yes** — the row schema for a scorecard |
| `distance_value_frozen_v3/verified-comparison/` | `comparison.md`, `conditional-metrics.csv` (`kind,group,variant,count,mae,rmse,bias,formal_nll`), `independent-verification.json`, `all-reference-diagnostics.csv` (406,725 B), `input-range-diagnostics.json`, `distance-value.png/pdf` | Units split by `kind`: `reference_mas` vs `native_kpc` | **Yes** — the distance panel + the "diagnostics kept unfiltered" habit |
| `native-age-comparison/` | `comparison.md`, `conditional-metrics.csv` (`run_id,subset,n,mae_Gyr,bias_Gyr`), `independent-verification.json`, `native-age.png/pdf` | Gyr; 50-star LEGACY reference split by 6 pipelines | **Yes** — the age panel |
| `scripts/analysis/plot_loss_lr_review.py` | Loss/LR curves, writes PNG+PDF+CSV to `experiments/xp/science-review-20261015` | Docstring: "no fitted/smoothed losses"; **asserts** the recorded LR schedule (`np.testing.assert_allclose(lr, expected, rtol=1e-10)`) | Yes, style |
| `scripts/analysis/plot_metallicity_review.py` | FeH-bin MAE, 7 hardcoded bins, 3 models, identity+sky splits, reads `metallicity-tails.json` | Hardcoded bins/models/labels; asserts `audit["status"]=="complete"` | Partly — binner is reusable |
| `scripts/analysis/analyze_metallicity_tails.py` | `error_metrics`, `paired_bootstrap` (:82, 1000 reps, `minimum_count=20`, explicit caveat string), `selection_metrics` (:115 completeness/purity at FeH threshold), `strata` (:137) | Caveat: "Pointwise paired individual-star bootstrap … no multiple-comparison correction, spatial blocks or training/label-systematic uncertainty" | **Yes** — adopt as-is |
| `experiments/xp/monitor/` (~90 files) | ≈10 one-off `compare_*.py` (`compare_next`, `compare_followup`, …), 7 `report_*.py`, ~28 `verify_*.py`, `plot_*.py` (6) | Each pinned to one experiment wave; `report_prediction_comparison.py:143-153` defines the slice vocabulary (`below_-3`, `below_-2`, `apogee_galah_only`, `survey:<name>`) | Slice vocabulary yes; scripts not |
| `scripts/run_prediction_comparison.py` (1,451+ lines), `scripts/run_multilabel_screen.py` | Training+eval drivers: `manifest.json`, `comparison.json`, `checkpoint_verification.json`, `replicas.json`, `metrics.jsonl`, `atomic_json()` (:276), run-dir per arm | One-directory-per-run, atomic JSON, SHA of every input | Not for release plots (they train) |
| `docs/reports/*.png` (10 files) | All bar/line-score figures (`multilabel-long-point-comparison`, `native-maf-comparison`, `native-three-seed-ensemble`, `parent/continuation/tta/mask-comparison`); **zero** physics-plane figures | PNG 2×–3× scale bar charts | Re-frame as dashboard tiles |
| Kiel / HR / CMD | **Nothing.** `grep -i "kiel\|color-magnitude\|CMD"` over repo code/docs (excl. `.pixi`) → no hit | — | Greenfield |
| Metric library | `torchregress.metrics`: 130 public names incl. `mae, rmse, bias, nmad, r2_score, attenuation_factor, trimmed_mean_squared_error, OutlierFraction, NormalizedRMSE, HuberMetric`, `GaussianNLLEnsemble, crps_gaussian, crps_from_samples, dss_score, pinball_metric, IntervalScore, prediction_interval_coverage_probability, MeanPredictionIntervalWidth, kolmogorov_ks_uniform, pit_cvm, pit_local_w1, probability_integral_transform, sbc_rank_histogram, tarp_coverage, highest_posterior_density_coverage, conditional_density_estimation_loss, entropy_score, risk_coverage_curve, calibration_metrics_report, expected/marginal_calibration_error, tail_mae, paired_bootstrap_ci` | `paired_bootstrap_ci(a,b,*,n_boot,alpha,seed)` — **no group/cluster argument** | Reuse; add grouping + Mondrian layer |
| Units / evidence culture | `point.csv.units` ∈ {`K`,`dex`}; distance `mas`/`kpc`; age `Gyr`; density review `limits[7]` + `coverage_note`; `report.json.limits[8]` | Every claim either `verified` (independently replayed) or explicitly producer-only; failures retained verbatim | **Non-negotiable to preserve** |

### 1.2 Where end-product code should live

- Metric/plot library: `src/masked_stellar_autoencoder/reporting/` (`schema.py`, `metrics.py`, `slices.py`,
  `kiel.py`, `figures.py`) — importable, testable under `tests/`, no checkpoint access.
- Single entry point: **`scripts/make_release_report.py`** with `--metrics-dir … --cohorts … --out docs/reports
  --figures all,kiel,pit,qa --formats png,pdf --fail-on-gate`. Mirrors `reproduce.py` (saved arrays only) +
  `augment_catalog.py` (template injection). Do **not** extend `experiments/xp/monitor/` (one-off by design).
- `scripts/analysis/` stays for per-question one-offs that feed CSVs into `--metrics-dir`. Artifacts:
  `docs/reports/msa-release-report-<YYYYMMDD>.html` + `docs/reports/figures/<ID>.{png,pdf}`; numbers in
  `experiments/xp/results-stellar-improvement/release-report-<YYYYMMDD>/` (layout as `density-review-20261022/`).

## 2. Metric definitions

Per label, native units, evaluated on a frozen role split (`test_rows`), with paired CI vs incumbent.

### 2.1 Point (already implemented in `density-review-20261022/reproduce.py:28-31`)

| Metric | Definition | Note |
|---|---|---|
| n, MAE, RMSE, bias | usual; bias = mean(ŷ−y) signed | Teff K, logg/FeH dex |
| NMAD | 1.4826·median\|ŷ−y−median(ŷ−y)\| | robust; report with MAD |
| R², Pearson | computed **inside the subset** | tail R² can be negative (limit 4) |
| attenuation slope | OLS β of y on ŷ | β<1 ⇒ shrinkage; must be shown, never used alone |
| outlier fraction | \|r\| > 3·subset target sd (`OutlierFraction(mode="absolute")`) | threshold stated |
| trimmed MSE | 10% two-sided trim | guards rare catastrophic outliers |
| NRMSE(std/IQR), Huber(δ=1) | as logged | comparability across labels |
| **add** tail MAE | mean \|r\| over worst-decile \|r\| (`tail_mae`) | "oldest/most-metal-poor" reporting |
| **add** fractional error | for distance only: (ŷ−y)/y | kpc |

### 2.2 Distribution

| Metric | Formula / call | Units | Caveat |
|---|---|---|---|
| NLL | −mean log p̂(y\|x); joint 3-label = per-object **sum** of natural logs | K,dex,dex (joint) | Only comparable within the same estimator family; observed-label vs intrinsic (noise-added) must be labelled (`distribution.csv.distribution`) |
| CRPS | `crps_gaussian(μ,σ,y)` or `crps_from_samples` | same as label | not defined for failed-flow arms |
| DSS | mean(z²)+log σ² style (`dss_score`) | dimensionless | proper, sensitive to σ misscale |
| PIT | u=F̂(y\|x); `kolmogorov_ks_uniform`, `pit_cvm`, χ² on 20 bins (descriptive only) | [0,1] | depends on label noise; not a physical-truth test |
| Rank histogram / SBC | `sbc_rank_histogram`, `sbc_uniformity_pvalue` | counts | needs ≥ thousands objects per stratum |
| Coverage | PICP at 0.68/0.90/0.95; **marginal**, **Bonferroni joint** (α/3 per label, already in `summary.csv`), **Mondrian** (per pre-registered stratum) | fraction | binomial CI required; n<20 → descriptive only |
| Interval score / MPIW | `IntervalScore`, `MeanPredictionIntervalWidth` | units, units | width is a cost, never a score alone |
| Calibration slope | OLS of cover-indicator on log width, and of u on binned score | slope | detects width–difficulty mismatch |
| Joint level set | Gaussian: fraction with (y−μ)ᵀΣ⁻¹(y−μ) ≤ χ²₃(1−α); flow: saved level-set membership (replay-checked, `validation.json`) | fraction | only for joint models with saved Σ/sets |
| Local PIT | `pit_local_w1` on (Teff, logg, FeH, G) features | [0,1] | detects conditional miscalibration that marginal coverage hides |

### 2.3 Population slices (pre-registered, all reported with n)

| Slice | Definition | n available (verified locally) |
|---|---|---|
| Bulk | all test rows | 97,695 (3-label native) / 100,620 (scalar FeH cohort) |
| Metal-poor | FeH < −2.0 | **141** test; 1,016 in full measured cohort; **14** test below −3 (3 cohort-wide) |
| Very metal-poor | FeH < −3.0 | 14 test |
| Old | age > 10 Gyr | 22,830 age identity-test rows exist; strict-fit 4,440 (**needs verification** of >10 Gyr count) |
| Dwarf | 3.8 ≤ logg ≤ 4.3 | 187,922 cohort |
| MSTO | 3.4 ≤ logg < 4.0 (report separately from dwarf) | derived |
| Giant | logg < 3.5 | 383,751 cohort |
| Luminous giant | logg < 2.0 | 110,141 cohort |
| XP tier | all 110 XP coefficients visible | 673,345 cohort rows |
| **No-XP tier** | no XP coefficient visible | **83,491** cohort rows |
| Distance | >2 kpc, >5 kpc | distance pilot only (frozen head; **needs verification** of per-bin counts) |
| Brightness / extinction | G bins, EBV quartiles | from 140-dim input (`visible` mask) |
| Sky | HEALPix nside=64 regions + `survey` tags (GALAH 531,918 / APOGEE 185,859 in cohort) | derived |
| Per-survey | `survey:<name>`, `apogee_galah_only` | existing vocabulary |

### 2.4 Uncertainty and repeated stars

- Paired bootstrap on **per-object** deltas (1,000 reps, `minimum_count=20`), seed recorded.
- **Must add**: grouped resampling — resample by unique `source_id` and by HEALPix block, because the cohort
  has duplicate stars (age cohort records `cross_survey_duplicate_candidates=4,860`,
  `feature_duplicate_excess=43,288`). `torchregress.metrics.paired_bootstrap_ci` has **no** group argument →
  implement `grouped_paired_bootstrap` in `reporting/metrics.py`.
- Multiplicity: Bonferroni across the ~16 slices above (or a stated FDR); every table states the correction.
- Cross-survey systematic is *not* in the CI; report the measured offsets instead
  (GALAH−APOGEE −0.0045±0.048 dex; DESI−APOGEE +0.0024, scatter 0.0746 dex).

### 2.5 What cannot be claimed

1. Coverage/NLL are against **measured native labels**, never physical truth; label errors are inside the
   likelihood (density review limit 1, programme review "MAE … not physical truth").
2. CP intervals are calibrated **mean-residual absolute residuals**, not predictive densities/credible regions
   (programme review, MAF section).
3. Alpha: **no native alpha model**; APOGEE α/M ≠ α/Fe and cross-survey scale unaudited → no α claim until retrained.
4. Age: StarHorse posteriors are model-conditioned; 37 quality-labelled seismic external stars is not a truth set.
5. Distance: no production head; both frozen pilots are **worse than raw Gaia on 4,166 Hipparcos** (free: MAE
   42.30 mas; assisted: 3.23 mas; raw Gaia: 0.96 mas). No "beyond-Gaia" distance claim.
6. Flow-density rankings stay **blocked** (strict replay failures retained) — no density leaderboard until the
   float64 certificate exists; and Kiel tightness alone is **not** an accuracy claim (§3 pitfall 1).

## 3. Kiel-diagram tightness — rigorous definitions

Notation: measured label vector y=(Teff, logg, FeH); prediction ŷ; reference sample R = measured-label stars
**excluded from test** (available: 717,777 complete-case measured rows − 97,695 test = **620,082**), with
quality masks and asymmetric-free errors (median e_Teff **68.7 K**, e_logg **0.1011 dex**, e_FeH **0.0580 dex**).

**(a) Ridge-residual scatter σ_K (primary, pre-registered).**
Fit cross-fitted ridge r(Teff, FeH)→logg on R: 2-D binned local-linear on a fixed grid (ΔTeff = 125 K over
3,500–6,500 K, ΔFeH = 0.25 dex over −2.5…+0.5, min 200 stars/bin, emptier bins excluded and counted) with
2-fold cross-fitting so reference residuals carry no fit bias; σ_ref = 1.4826·MAD(logg − r(Teff,FeH)) on the
held-out half, σ_noise = √(median e_logg² + (∂r/∂Teff·e_Teff)² + (∂r/∂FeH·e_FeH)²). Model statistic:
σ_K = 1.4826·MAD(logg_pred − r(Teff_meas, FeH_meas)) on test stars — **measured** Teff/FeH, never predicted
ones (avoids errors-in-variables). Report R_K = σ_K/σ_ref, R_N = σ_K/σ_noise, grouped bootstrap 95% CI, per bin.
Pitfalls: a constant predictor scores σ_K = 0 → **gate**: no "tighter" claim unless R_K<1 with CI excluding 1
**and** logg MAE does not increase (CI on paired ΔMAE includes 0 or is better); cross-survey ridge offsets;
empty tail bins; selection bias from quality cuts (they correlate with logg).
Bins for reporting: (Teff 125 K × logg 0.25 dex × FeH 0.5 dex) plus pre-registered evolutionary bins from logg.

**(b) Distribution concentration vs a covariate-matched reference.**
S = mean log p̂_ref(Teff_pred, logg_pred) under a 2-D histogram/KDE of a reference subsample **quantile-matched**
to the predicted Teff marginal (match by stratified resampling, so an input-driven Teff shift cannot earn density);
complement with σ_W1 = sliced-Wasserstein/energy distance between predicted and matched reference Kiel clouds,
and HDR overlap O = ½[P_ref(pred in R's 90% HDR) + P_pred(ref in pred's 90% HDR)].
Pitfalls: collapse onto the ridge line maximises S and destroys the RGB/HeB bifurcation → always co-report
Teff-conditional logg entropy, **branch ratio** (fraction logg>3.5) against the reference value, and the energy
distance (which penalises collapse). Bin/branch definitions are frozen before the test read.

**(c) Isochrone / independent-logg residual.**
σ_iso (clusters): for members of clusters with literature [Fe/H], age and (m−M), compare logg_pred with
isochrone logg at Teff_pred (PARSEC/BaSTI, solar and α-enhanced); NMAD per cluster, cluster-level bootstrap.
**Cheap local substitute available now**: `apokasc3_collection_v1/table4.dat` (15,808 stars; `loggSeis` cgs,
`e_loggSeis`; 10,036 "Seis Gold"; 4,232 unambiguous gold RGB) → σ_seis = NMAD(logg_pred − loggSeis) on gold RGB.
Pitfalls: isochrone physics/systematics are not truth; seismic scaling has radius/Teff dependence; cluster
membership errors; do not mix with (a) in one headline number.

Reporting: one headline per definition, three panels, never a composite "tightness index".

**Figure design (F01).** Axes x=Teff [K, reversed, 3,500–7,000], y=logg [dex, reversed, 0–5].
A: log-density hexbin/contours (levels enclosing 50/68/90/95% of predicted mass) for the **all-Gaia** tier;
B: same for XP tier vs no-XP tier side by side, identical colour scale, counts annotated (cohort now shows
673,345 vs 83,491 rows — the tier figure is already demonstrable);
C: metal-poor (FeH<−2, n=1,016 cohort / 141 test) and old (>10 Gyr) zooms, per-point FeH colour;
D: residual map median(logg_pred − logg_meas) on the (a) grid, diverging ±0.3 dex, bins with n<20 hatched;
E: overlaid reference loci: measured cohort (GALAH/APOGEE), APOKASC3 `loggSeis` gold, and later
GALAH DR4/APOGEE DR17/LAMOST DR12/DESI/Gaia RVS GSP-Spec; isochrone curves at representative (FeH, age).
**Data needed**: the 1.8B-row all-Gaia inference product (source_id, 6 params + σ, tier flags, G, distance,
quality flags) as versioned parquet — not yet produced; reference layers above need DR ingestion
(LAMOST DR12, DESI SP, Gaia RVS, GALAH DR5 are **not** ingested; SDSS DR20 planned).

## 4. End-product figure / dashboard list

| ID | Purpose | Inputs | Panels, axes, units | Gate / threshold | Audience |
|---|---|---|---|---|---|
| F01 | **Kiel density + tightness** (§3) | cohort+preds, APOKASC3 | 5 panels, Teff K vs logg dex | R_K<1 (CI excl. 1) & no logg-MAE regression | Science lead, referees |
| F02 | Kiel residual vs covariates | preds + measured | median±IQR residual vs Teff, logg, FeH, G, EBV; `logg` dex | no bin with \|bias\|>0.1 dex for n≥50 | Science lead |
| F03 | HR / CMD with distance | preds + distance + G/BP/RP | M_G vs BP−RP (mag), Teff vs M_G; colour = log d; separate >2/>5 kpc | parallax-SNR cuts stated | Science lead |
| F04 | Age–metallicity & age–Teff (old/metal-poor) | age cohort + FeH/Teff | age [Gyr, log] vs FeH; age vs Teff colour-coded FeH | show only n≥20 bins; asymmetric errors | Science lead |
| F05 | [α/Fe] vs [Fe/H] | **blocked** until α trained | dex vs dex, reference overlay | explicitly "NOT RUN" tile | Programme |
| F06 | PIT + coverage per label | `distribution.csv`, `interval.csv` | PIT hist + KS/CvM; coverage vs nominal (0.68/0.90/0.95); marginal + Bonferroni + metal-poor | \|cov−nominal\|≤0.02 marginal; strata reported | Stats reviewer |
| F07 | Width vs precision | CP arrays | MPIW vs \|residual\| decile; width vs G/SNR; sharpness–coverage curve | width must grow with difficulty (positive slope) | Stats reviewer |
| F08 | Distance validation | `distance_value_frozen_v3/*` | Δϖ [mas] vs ϖ_ref; fractional distance error vs ref distance & ref type; sky map; extinction/latitude | must not regress vs raw-Gaia 0.96 mas baseline | Science lead |
| F09 | Age vs seismic refs | `native-age-comparison/*`, APOKASC3/LEGACY | predicted vs reference age [Gyr], asymmetric errors, 6-pipeline spread | bias and MAE reported, never "true age" | Science lead |
| F10 | Metal-poor tail zoom | 141 test / 1,016 cohort | per-star residual table + plot vs FeH; completeness/purity at FeH<−2, <−3 | n<20 ⇒ descriptive only, stated inline | Programme |
| F11 | Catalogue QA | all-Gaia output + tier flags | sky counts/HEALPix map, tier fractions, parameter distributions vs reference, missingness, duplicate `source_id`, per-healpix sanity | no cell >5σ from expectation; 0 duplicate source_id | Data engineer |
| F12 | Model/arm comparison matrix | all `*-verified-comparison` + density CSVs | paired ΔMAE/Δcoverage with CI vs incumbent; "beats incumbent" column | decision rule pre-registered (§5) | Programme |
| F13 | Label/provenance panel | registry v2, survey overlap | source mix, cross-survey offsets, label-error distributions | offsets quoted with n | Programme |
| F14 | Ledger footer | all | versions/SHAs, gate status, unverified rows listed separately | every failure preserved | Everyone |

## 5. Acceptance, reproducibility, gates

**Release scorecard** (frozen before the final test read; `scorecard.json`):
1. Primary endpoint: paired ΔMAE vs incumbent per label (Teff K, logg dex, FeH dex) on `test_rows` with grouped
   bootstrap 95% CI; "best" = CI upper bound < 0 for every label claimed.
2. Secondary: metal-poor (FeH<−2, n=141) and old slices reported but **not** allowed to override the primary;
   a tail-only win must be labelled as such (existing precedent: BMC/DRO).
3. Distribution arm is admitted only if its strict replay verification passes; otherwise `producer-only` tab.
4. Ties on MAE broken by (i) fewer parameters/compute, (ii) better marginal calibration, (iii) better tail.
5. No metric is reported without n, units and its slice definition.

**Gates** (all machine-checked, `validation.json`): **G1** units round-trip (`pred_std*label_scale+label_center`
reproduces saved native preds; Teff K, logg/FeH dex, ϖ mas, distance kpc, age Gyr); **G2** schema/counts
(`rmse²=mse` rtol 1e-12; MAE recomputed from absolute errors); **G3** regeneration (same CSVs → byte-identical
figures, no checkpoint load — `reproduce.py` convention); **G4** independent-verification boolean per row, with
unverified rows visually separated; **G5** visual QA per figure (`visual-review.json`, e.g.
`{"f01_kiel_png_visually_checked": true, "scope": "…", "no_tightness_index": true}`); **G6** no failure erasure
(failed density arms keep their status strings verbatim); **G7** multiplicity correction stated.

**Regeneration**: `python scripts/make_release_report.py --metrics-dir <release-report-dir> --out docs/reports
--figures all --formats png,pdf` reads only versioned CSV/JSON (+ `input_sha256` assertions) and rewrites
`msa-release-report-<date>.html` via the tracked template — same split as `reproduce.py` (computes) +
`augment_catalog.py:54` (injects). Target runtime < 2 min on CPU.

## 6. Minimum viable dashboard now vs final

**MVP (~2 days, no model recompute, no new training).** Uses only saved arrays:
cohort `multi_gaussian_{noise_recon_30e,gaussian_recon_s42,huber_recon_s42,…}_v1/paired-cohort.npz`
(756,836 rows: `value` (756,836×3 Teff/logg/FeH), `error`, `observed`, `quality`, `survey`, `input`
(756,836×140), role index arrays) + `predictions.npz`/`conformal.npz`; density-review CSVs; distance and age
pilot reports.

| MVP item | Exact input files |
|---|---|
| F01 Kiel **cohort/reference** version (all-Gaia tier deferred; label the tier honestly) | `multi_gaussian_noise_recon_30e_v1/{paired-cohort.npz,predictions.npz}`; `apokasc3_collection_v1/table4.dat` |
| F02 Kiel residual maps + σ_K/R_K for the 4–6 verified arms | same; `multi_{gaussian,huber,maf,cinn,nsf,bpf}_*_v1/predictions.npz` |
| F06 PIT/coverage incl. metal-poor | `density-review-20261022/{distribution.csv,interval.csv,point.csv}`; `…/verified-point-cp-comparison/{summary.csv,report.json}` |
| F07 width vs precision | `multi_*_v1/conformal.npz`; `verified-point-cp-comparison/paired-test-predictions.csv` |
| F10 metal-poor tail (141) | `summary.csv` subset `FeH below -2`; `point.csv` |
| F12 comparison matrix | `point.csv` (136 rows) + `report.json.summary` |
| F11 partial QA (counts, tier fractions, missingness) | `paired-cohort.npz` (`visible`, `survey`) |
| F03/F08/F09 tiles | re-frame existing `docs/reports/*.png` + `distance_value_frozen_v3/verified-comparison/*`, `native-age-comparison/*` (no regeneration) |

MVP limits on the dashboard: cohort ≠ all-Gaia; single seed; reused test; 141-star tail; no α; no production distance head; Kiel tightness cohort-only, not yet claimable for the delivered catalogue.

**Final version** (after new labels/model): all-Gaia inference product → F01 all-Gaia + XP/no-XP tiers, F03/F04
population planes, F05 α, F08 production distance, F09 cluster+seismic ages, F11 full-catalogue QA (1.8B rows,
HEALPix nside=64), F12 pre-registered decision table, F14 ledger. Requires the ingest list from BRIEF
(SDSS DR20, DESI SP/later, GALAH DR5, Gaia RVS, LAMOST DR12, ESO) plus HATS-scale streaming so the QA pass
never holds 1.8B×140 features in RAM (legacy 140-feature cache would need ~117 GB per 10M rows).

## 7. What this implies for the plan

1. **No end-product figure exists today** for the human's central artefact (Kiel/HR/CMD): zero repo hits.
   F01+F02 are the highest-value new work, and the reference layer is already local (620,082 non-test measured
   stars + 15,808 APOKASC3 seismic logg).
2. The **metric layer is 80% built**: `point/distribution/interval.csv` schema, `validation.json`, visual-QA JSON
   and `reproduce.py`-style saved-array reporting are reusable; real gaps are grouped/cluster-aware bootstrap,
   Mondrian coverage by slice, joint level-set per arm, and all Kiel metrics.
3. **Tightness must be gated against shrinkage** (R_K plus ΔMAE), else a flatter logg predictor "wins" F01 while
   failing the accuracy objective.
4. Report the **141-star metal-poor test tail** honestly; the Kiel metal-poor zoom can show 1,016 reference stars
   but predictive claims there are Poisson-limited (14 below −3). New labels, not plotting, are the constraint.
5. The density leaderboard is replay-blocked, so F06/F12 must render `producer-only` and `strict-replay-failed`
   rows in a visibly separate block — the evidence culture forbids a cleaner story.
