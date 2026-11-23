# QA catalogue: MSA stellar-parameter programme (2026-11-23)

This catalogue lists every quality-assessment (QA) check in the programme. For each check it gives the script and function, the reference data, the metrics, where the results live and the latest result. It then maps checks to products, lists the comparisons with competing catalogues, and proposes the QA suite for the full-Gaia run.

This is a read-only audit: no code was changed and no CANFAR job was run. One exception, marked **(this audit)**: the existing ARC scripts `consistency_analysis.py` and `binary_heldout.py` were run locally on the saved v10 predictions (`consistency_v10.npz`), because the last scored co-natal run was v7.

## 0. Conventions

**Current lineage** (13–23 Nov 2026). Everything below refers to this lineage unless marked historical.

| piece | run on ARC (`/arc/projects/astroai/msa/runs/`) | role |
|---|---|---|
| labelled table | `multisurvey_table_20261117_v5/finetune.fits` (5,872,237 rows, `records.npz`) | inputs and survey records |
| anchored multi-survey fine-tune v10 | `multisurvey_finetune_20261121_v10` (epoch-12 checkpoint) | encoder, survey maps, posterior Student-t head (product B, interim prior) |
| S1 likelihood heads | `likelihood_head_s1_v10_20261121`, `likelihood_head_s1_v10_ens_20261122` (arm `all`, K = 5), `likelihood_head_s1_v10_plx_ens_20261122` (arm `plx`, K = 5) | product A (measurement) |
| S2 distributional heads | `distributional_heads_s2_v10_20261122` | product B0 (flow, t-mixture, Gaussian mixture, t) |
| S3 distance and E(B−V) | `distance_extinction_s3_v10_20261122`, `distance_extinction_s3_v2_20261123`, `distance_extinction_s3_v21_20261123` (latest) | p(s, E \| z) trained on Gaia parallaxes |
| XP forward model | `xp_forward_model_20261122`, `xp_forward_infer_20261122`, `xp_forward_scan_20261122` | product A, alternative route |
| age consensus | `docs/age-c0-20261122`, `docs/age-c0b-20261123`, `docs/age-c0c-20261124` (repo); inputs in `/arc/projects/astroai/msa/analysis/age_audit_20261122`, `.../age_labels_20261122` | label-level age scale (no per-star age head yet) |

**Metrics.**
- **bias**: median(prediction − reference).
- **NMAD**: 1.4826 × median absolute deviation.
- **Coverage**: fraction of references inside the nominal set. Per-bin tests use Wilson 95 % intervals with a Bonferroni split (`torchregress.inference.binned_conditional_coverage`). Bins with n < 20 are descriptive only.
- **PIT**: the predictive CDF at the observation. **Pull**: residual / total sd.
- **χ²/dof**: of co-natal members about their group mean. **z-NMAD**: NMAD of co-natal pull scores, which should be 1.
- **NLL / log score**, **energy**, **variogram**, **CRPS**: proper scores.
- **Purity / completeness**: of a selection, for example [Fe/H] < −2.5.

**References are noisy.** Survey labels, anchors and literature values are measurements, not truth. Coverage is coverage of the label unless errors-in-variables (EIV) is stated.

**Where results live.**
- New runs: `/arc/projects/astroai/msa/runs/<run>/` (`report.md`, `metrics.json`, `qa_*.png`) and `/arc/projects/astroai/msa/analysis/` (`frozen_core/`, `consistency/`, `scripts/`, `age_*`).
- Age reports are in the repo (`docs/age-*`).
- Runs before 13 Nov lived in `/arc/projects/k-pop/msa_runs/`. That tree no longer exists (404), so their numbers survive only in `docs/reports/*.md`.

```bash
R=https://ws-uv.canfar.net/arc
curl -sSL --cert ~/.ssl/cadcproxy.pem "$R/files/projects/astroai/msa/runs/<run>/report.md"
curl -sSL --cert ~/.ssl/cadcproxy.pem "$R/nodes/projects/astroai/msa/runs?limit=500"   # list runs
```

**Status legend** used in the tables:
- **I**: implemented and run on the current lineage.
- **P**: partial: older checkpoint only, a subset only, not in a decision rule, or computed only in this audit.
- **H**: historical: code exists, ran on pre-13-Nov models, outputs lost.
- **M**: missing.
- **–**: not applicable.

## 1. Catalogue of QA checks

Paths are relative to the repo root unless they start with `/arc`. Several analysis scripts exist **only on ARC**, under `/arc/projects/astroai/msa/analysis/scripts/`: `consistency_analysis.py`, `binary_heldout.py`, `make_targets.py`, `rescore_analysis.py`, `estimator_table.py` and `tail_prob.py`.

### 1.1 Held-out label-scale checks

| ID | check | products | script:function | inputs | metrics | plots / outputs | latest run → result |
|---|---|---|---|---|---|---|---|
| L1 | Test split vs common-scale mapped labels, per G | Teff, log g, [Fe/H] (S1, S2, posterior) | `scripts/s1_likelihood_head.py:767` `stage_report` (`test_labels`); `scripts/s2_distributional_heads.py:682` | `labels.npz` (`common_scale_labels`, survey records through the v10 maps); split from `rescore_20261119/rows_split.npz` | bias, NMAD for G ≤ 16 and G > 16 | `report.md` | `s1_v10_plx_ens`, posterior head, G ≤ 16: Teff −3/67 K, log g −0.013/0.122, [Fe/H] +0.009/0.082 (n ≈ 2–3 × 10⁵). G > 16: [Fe/H] −0.024/0.196. S1 MLE (`all`, single head): [Fe/H] +0.005/0.094 (G ≤ 16), −0.025/0.250 (G > 16). |
| L2 | **Frozen high-resolution [Fe/H] core**: bias conditional on the truth, per [Fe/H]ref × G cell | [Fe/H] (all estimators); distance (S3) | `src/masked_stellar_autoencoder/training/likelihood_validation.py:75` `offset_table`; reference `training/common_scale_labels.py:50` `frozen_core_reference` (priority `CORE_PRIORITY`, line 36); rows `scripts/core_eval_rows.py`; S3 `scripts/s3_distance_extinction.py:824` `core_metrics` | 39,073 table rows with a high-res [Fe/H]: Bensby 2014, Aoki 2013, Li 2022 LAMOST-Subaru, SMSS Yong 2021, Yong 2013, the R-Process Alliance samples, Roederer 2014, Kirby 2010, Gaia-ESO DR5.1, JINAbase, SAGA, PASTEL 2020. 7,740 held out (valid + test), 646 below −2. `/arc/projects/astroai/msa/analysis/frozen_core/` | median offset, NMAD per [Fe/H] bin (−9, −3, −2.5, −2, −1.5, −1, −0.5, 9) × G (≤ 13.5, 13.5–15, 15–16, 16–17.65) | `report.md` tables; `analysis/frozen_core/rescore_wave{1,2}/` | Posterior head (balanced-sampling prior), G ≤ 16: +0.34 ([−9, −3), n = 106), +0.14 ([−3, −2.5), 192), +0.01 ([−2.5, −2), 264); at G 16–17.65 +0.80 / +0.34. S1 `all` 5-head ensemble: worst \|bias\| 0.051 (G ≤ 16), but the NMAD is larger (0.40–0.48 below −2.5 for the single head). S1 `plx`: +0.40 / +0.27 (fails). S2 flow (median): +0.50 at [−9, −3) (G ≤ 16). S3 v2.1 distance modulus: −0.02 to +0.07 mag per bin. |
| L3 | Per-epoch held-out anchor metrics of the anchored fine-tune; they select the epoch | Teff, log g, [Fe/H], E(B−V), parallax | `training/anchored_multisurvey.py:599` `AnchoredMultiSurveyWrapper.anchor_metrics`; table `scripts/summarize_ablation.py` | valid split: IRFM anchors, seismic anchors, [Fe/H] core, E labels, parallax | bias, NMAD, 16–84 % coverage (`cov`); `fe_h_core_mp` (< −2), `_vmp` (< −2.5); `logg_anchor_plx`; `map_slopes`; `score` | `<run>/finetune_anchor_metrics.jsonl` | v10, epoch 12: Teff vs IRFM −4.7/35.1 K. Seismic log g −0.002/0.102 (0.080 with parallax conditioning), cov 0.68. [Fe/H] core −0.019/0.171, cov 0.47. Below −2: +0.112/0.269, cov 0.47. Below −2.5: +0.206/0.296. E(B−V) +0.003/0.0246, cov 0.51. Parallax NMAD 0.049, cov 0.60. |
| L4 | Pairwise model comparison on one test split: seismic, high-res [Fe/H] by bin, IRFM, per-survey offsets from the common prediction, optional GBS | Teff, log g, [Fe/H] | `scripts/compare_multisurvey_pair.py:51` | `<eval>_predictions.npz` or `predict_fits_rows.py --test-split` outputs; `<table>_records.npz` | bias, NMAD, 16–84 % coverage | stdout only | Used for v1–v8 decisions. No archived output. |
| L5 | Survey-map validity (slopes per survey and parameter) | Teff, log g, [Fe/H] | `anchor_metrics` (`map_slopes`); historical `scripts/audit_survey_mapping_geometry.py` | checkpoint maps | slope of each survey map | `finetune_anchor_metrics.jsonl` | v10, epoch 12: BOSSNet log g slope 0.69, SSPP log g 0.86, GSP-Spec Teff 0.87. No pass/fail rule. |
| L6 | v7 ablation grid (25 % runs) | all | `scripts/summarize_ablation.py` | `multisurvey_ablation_20261118_v7/*/finetune_anchor_metrics.jsonl`; `/arc/projects/astroai/msa/analysis/frozen_core/ablation_v7_anchor_metrics/` | as L3 | stdout | v7 era (18 Nov). |

### 1.2 Physical anchors and independent references

| ID | check | products | script:function | inputs | metrics | latest run → result |
|---|---|---|---|---|---|---|
| A1 | **IRFM Teff** (Casagrande et al. 2021, vendored `colte`) | Teff | `training/irfm_teff.py:61` `photometric_teff`; S1/S2 anchors via `likelihood_validation.py:89` `anchor_table`, `:101` `anchor_records` (survey `anchor_irfm_colte`); `scripts/predict_fits_rows.py --irfm` | G, BP, RP, 2MASS (NIR colours kept only when they agree with BP−RP), low-EBV stars | bias, NMAD on 59,363 test stars | S1 plx ensemble report: posterior −4/36 K, S1 `all` (single) −4/34, S1 `all` (K = 5) −3/32, S1 `plx` +1/39. S2: flow −1/33, flow without EIV −1/31, t-mixture 0/33. **Caveats:** IRFM is also a training anchor (an identity map), so this tests transfer to held-out stars, not the gauge. The tables' 2MASS columns are mismatched for many stars (`training/reddening_labels.py` docstring), so the anchor is effectively BP−RP-based and shares that input with the encoder. |
| A2 | **Seismic log g**: APOKASC-3, APO-K2, APOKASC dwarfs (Serenelli+2017), Kepler LEGACY | log g | `training/anchor_labels.py:73` `apokasc3`, `:82` `apok2`, `:93` `apokasc_dwarfs`, `:103` `kepler_legacy`, `:127` `load_reference_labels` | `/arc/projects/k-pop/catalogues/{apokasc-3-2024, apo_k2_2024, apokasc-1-dwarfs, kepler-legacy-dwarfs, kepler_dr3_good}.fits` | bias, NMAD per catalogue | 1,132 test stars (APOKASC-3 859, APO-K2 258, dwarfs 15, LEGACY 0). Posterior −0.007/0.078. S1 `plx` 0.000/0.088. S1 `all` (single) +0.022/0.124. S1 `all` (K = 5) +0.029/0.117. S2 flow +0.007/0.096. Like A1, these are training anchors. |
| A3 | TESS (Hon+2021 νmax) and K2 GAP DR3 seismic log g, by parallax S/N, RUWE and mass flag | log g | `scripts/run_seismic_gravity_qa.py:92` (`acquire_vizier:49`); `scripts/report_tess_gravity_qa.py:24` `gravity_stats`, `:43`; `scripts/stage_tess_k2_gaia_candidates.py` | VizieR J/ApJ/919/131, J/ApJ/926/191; 151,316 TESS references matched to the raw input (7 Nov) | bias, NMAD, coverage of draws | **H**: first-checkpoint QA of the old fullfit model. The staged candidates were lost with `k-pop/msa_runs`. Not run on v10. |
| A4 | **Gaia FGK benchmark stars, GBS v3** (Soubiran+2024); the only validation-only physical set | Teff, log g, [Fe/H] | `training/anchor_labels.py:114` `gaia_benchmarks` (role `validation`); `scripts/build_stopgap_finetune_table.py` writes `validation.fits`; `scripts/predict_fits_rows.py`; `scripts/compare_multisurvey_pair.py --gbs`; historical `scripts/qa_updated_benchmarks.py` | `/arc/projects/k-pop/catalogues/gaia_fgk_benchmarks.parquet` | bias, NMAD | **P**: not scored for v10, S1, S2 or S3. Historical (HANDOFF): benchmark log g MAE 0.4305 (epoch-1 expanded model) vs 0.3903 (c59). |
| A5 | **Empirical E(B−V) labels** (intrinsic colours learned from low-dust stars) and the high-latitude zero point | E(B−V) | `training/reddening_labels.py:125` `fit_intrinsic_colours`, `:181` `reddening_labels`; `scripts/build_reddening_labels.py` | `reddening_labels_20261118_v6/{v5,v1}_reddening.npz`, `diagnostics.json`; SFD from the table's `EBV` column | low-map bias and NMAD, fraction > 3σ above the map, ratio to SFD per map bin | v5 labels: low-map bias +0.0001, NMAD 0.031, 2.8 % above the map at 3σ. Ratio to SFD: 1.04 (SFD 0.05–0.1), 0.95 (0.1–0.3), 0.83 (0.3–1), 0.46 (1–5 mag), as expected for a 2-D map at high extinction. |
| A6 | Hipparcos distances vs raw Gaia; parallax unit checks | distance | `scripts/run_distance_value.py`, `scripts/hipparcos_reference.py`, `scripts/stellar_hipparcos_inputs.py`; `scripts/audit_parallax_against_esa.py:22` | Hipparcos reference set (4,166 stars) | parallax MAE | **H** (Oct): frozen pilots were worse than raw Gaia (MAE 42.30 and 3.23 mas vs 0.96 mas; `plan-20261022/04` §2.5). Not run for S3. |
| A7 | External RAVE DR6 reference (MADERA Teff and log g, GAUGUIN [Fe/H]) | Teff, log g, [Fe/H] | `scripts/validate_multisurvey_predictions.py:55` `compare` | `/arc/projects/k-pop/catalogues/rave_dr6_{madera,cnn,bdasp,x_gaiaedr3}.fits` | bias, NMAD, 90 % coverage | **H** (5 Nov, old model): +60.6/115 K, +0.169/0.330, +0.130/0.161; coverage 59/54/74 %. RAVE is not a v10 training survey, so it is again an independent external set (§4). |

### 1.3 Co-natal and label-free checks

| ID | check | products | script:function | inputs | metrics | latest run → result |
|---|---|---|---|---|---|---|
| C1 | **Wide binaries, [Fe/H] consistency** (El-Badry+2021, MSMS) | [Fe/H] (posterior head only) | target list: ARC `make_targets.py`; inputs `scripts/build_consistency_table.py`; predictions `scripts/predict_fits_rows.py --condition-parallax 5` → `consistency_vX.npz`; scoring: ARC `consistency_analysis.py:67` | `consistency_table_20261118/stars.fits` (1,209,544 stars); `analysis/consistency/consistency_targets.npz`; `diag_data_vizier_cfa.tar.gz` (`elbadry.tsv`, `hr23_*.tsv`, `harris.tsv`); `labelled.npz` (107,026 targets in the labelled table) | Δ = [Fe/H](brighter) − [Fe/H](fainter): median, NMAD, \|z\| < 1 fraction per pair-mean bin, by ΔG, giant + dwarf pairs | **v10 (this audit)**: 280,045 pairs. Δ = +0.088/0.187 at pair mean > −0.5 (v7: +0.129/0.211). +0.036/0.235 at [−1, −0.5). By ΔG: +0.018 (0–1 mag) rising to +0.236 (> 3 mag). The fainter star reads more metal-poor. \|z\| < 1 = 0.56 (should be 0.68). **Caveats**: no R_chance cut and no exclusion of labelled stars from the statistics (223,945 pairs have neither star labelled, but all pairs are scored). |
| C2 | **Held-out binary pool**: companions inherit the primary's [Fe/H] (1/5 of pairs held out) | [Fe/H] | `scripts/build_binary_pool.py:26` `heldout`, `:32`; scoring: ARC `binary_heldout.py` | `binary_pool_20261120_pairs/pool.npz` (90,123 pairs, 18,201 held out); primaries are bright FGK dwarfs | median primary − companion [Fe/H] by companion Teff; companions at G > 16.5 | **v10 (this audit)**: +0.316 (3000–3700 K, n = 5,616), +0.260 (3700–4200 K), +0.095 (4200–4800 K), +0.021 (4800–5500 K); G > 16.5: +0.299. Cool dwarfs read 0.26–0.32 dex too metal-poor: the gauge freedom below ~4500 K (design 2026-11-20 §2, item 3). |
| C3 | **Open clusters, [Fe/H]** (Hunt & Reffert 2023 members) | [Fe/H] | ARC `consistency_analysis.py:99` (clusters), `:148` (OCs) | `hr23_members.tsv`, `hr23_clusters.tsv` (no membership-probability cut) | within-cluster NMAD, faint − bright, slope per mag, giant − dwarf | **v10 (this audit)**: 5,033 OCs with ≥ 20 members. NMAD 0.295 (v7 0.249), faint − bright −0.159, **giant − dwarf +0.343** (v7 +0.179; 2,024 clusters), slope −0.017 dex/mag. Worse than v7. |
| C4 | **Globular clusters, [Fe/H] vs Harris** | [Fe/H] | ARC `consistency_analysis.py:134`; historical `scripts/validate_multisurvey_predictions.py:161` `clusters` (Vasiliev & Baumgardt 2021 EDR3 members, `/arc/projects/k-pop/catalogues/gc_gaia/`) | HR23 type `g` members; `harris.tsv` | median(pred − Harris) per Harris bin; within-cluster NMAD; faint − bright | **v10 (this audit)**: 118 GCs, 106 with Harris values. Harris [−3, −2): −0.019 (v7 +0.228). [−2, −1.5): −0.088 (v7 +0.077). [−1.5, −1): +0.006. [−1, 0): +0.028. Within-cluster NMAD 0.24–0.34. v10 (identity [Fe/H] core, no binary pool) no longer shows the GC metal-poor bias. |
| C5 | **Open clusters, distance and E(B−V)**: members share s and E | distance, E(B−V) | `scripts/s3_distance_extinction.py:391` `stage_clusters`, `:896` `cluster_metrics` | `/arc/projects/k-pop/catalogues/OpenClusters/{members,clusters}.fits` (Hunt & Reffert; Prob ≥ 0.7, type `o`, ≥ 5 held-out members) | χ²/dof of members' s and E about the cluster mean; pull NMAD; Δμ vs the cluster's geometric distance (members' parallaxes, plx/err > 20) | S3 v2.1 gmix: 96 clusters (972 members). χ²/dof 0.84 (s) and 1.24 (E); pull NMAD 0.88 / 1.05; Δμ +0.029/0.161 mag (95 clusters). `qa_clusters.png`. |
| C6 | **Age co-natal terms**: cluster members, wide pairs | age | `scripts/age_consensus_c0.py:1433` `conatal_tables`, `:1578` `conatal_summary` | HR23 members; El-Badry pairs (R_chance < 0.1) | z-NMAD of member and pair pulls | C0c: cluster members z-NMAD 0.67 (3,301). Dwarf–dwarf pairs 0.15 (99), subgiant pairs 0.58 (349), all pairs 0.49 (510). Below 1, so the σ are conservative. |
| C7 | Age audit co-natal tests, per catalogue | age | `scripts/age_label_audit.py:1076` `cluster_consistency`, `:1125` `binary_consistency` | audit `records.parquet`; HR23; El-Badry MSMS pairs | within-cluster NMAD / quoted σ; per-star pair scatter | `docs/age-audit-20261122/`: isochrone pipelines' quoted σ are 1.5–5× too small. Pair scatter: FLAME 0.30 dex (101,294 pairs), StarHorse-LAMOST 0.084 dex. |

### 1.4 Calibration and proper scores

| ID | check | products | script:function | metrics / acceptance | latest run → result |
|---|---|---|---|---|---|
| K1 | S1 Neyman, Wilks and profile coverage per Teff, log g, [Fe/H] and G bin | Teff, log g, [Fe/H] (A) | `likelihood_validation.py:123` `coverage_table`; `s1_likelihood_head.py:639` `_coverage`; Neyman critical values fitted per (θ, G) on the valid split | joint 68 % Neyman set 0.68 ± 0.08 in every bin with n ≥ 20 at G ≤ 16 | Neyman 68 = 0.68 overall and in every bin (`all` and `plx`). Raw Wilks 0.54–0.63 (under-covers). Posterior head 16–84 %: 0.59–0.61; 0.16 at [Fe/H] < −3. |
| K2 | **S1 decision rule** (fixed before the runs) | A | `likelihood_validation.py:170` `decide` | (1) \|bias\| ≤ 0.10 in every core [Fe/H] bin (G ≤ 16); (2) coverage as in K1; (3) bulk NMAD ≤ head + 0.01; (4) anchors: NMAD ≤ 1.05 × head and \|bias\| ≤ \|head\| + 10 K or + 0.01 dex | `s1_v10_ens` arm `all` (K = 5): passes bias (0.051), coverage and bulk (0.158 vs 0.153); **fails log g anchors** (0.117 vs 0.078). Arm `plx`: fails bias (0.399 at < −3), bulk (0.182) and anchors. Neither adopted. |
| K3 | S2 raw HPD and conformal coverage: marginal, Mondrian by G, Mondrian by G × predicted [Fe/H] | B0 | `s2_distributional_heads.py:524` `stage_calibrate`, `:578` `_coverage_strata`, `:611` `_coverage_fails` | within ±0.03 of nominal at 68 and 95 % in every G and label-[Fe/H] stratum (Wilson–Bonferroni) | Mondrian sets cover in the G strata (0.674–0.681). They **fail in label [Fe/H] < −1**: flow 0.257 / 0.36 / 0.585 at [−9, −3) / [−3, −2.5) / [−2.5, −2) (68 %). Conformal calibration cannot undo shrinkage in label space. |
| K4 | S2 proper scores | B0 | `s2_distributional_heads.py:459` `stage_evaluate` (`_evaluate_one:315`) | NLL of the label predictive, energy, variogram, CRPS, PIT KS, per G | Flow without EIV is best: NLL −2.080, energy 0.215, CRPS 52.2 K / 0.101 / 0.082. Posterior: −1.501, 0.222, 57.6 K / 0.098 / 0.085. |
| K5 | **S2 decision rule** | B0 | `s2_distributional_heads.py:812` `_decide`, `:629` `_cell_fails` | better log and energy scores than the posterior; Mondrian coverage as K3; no [Fe/H]ref × G cell worse (bias ≤ \|bias_post\| + max(0.02, 2 SE), NMAD ≤ NMAD_post + max(0.01, 2 SE)) | **No head competitive.** All heads pass log and energy, and all fail conformal coverage (7 failing strata) and per-cell bias/NMAD (14–20 cells). |
| K6 | **S3 parallax calibration**: PIT coverage per G and per predicted-parallax S/N bin; mean pull per G and [Fe/H] | distance | `s3_distance_extinction.py:750` `distance_metrics`, `:1036` `_decide` | coverage within ±0.03 per bin; \|mean pull\| ≤ max(0.05, 3 SE) per G bin | v2.1 gmix: 68 % coverage 0.721; fails 4 G and 4 S/N bins (over-covers); pull −0.019 ± sd 0.904. v2 flow: 1 G and 1 S/N fail. v10 gmix: passed (0.706). Precise-parallax Δμ +0.009/0.209 mag (296,169 stars), rising to +0.074/0.309 at G 17–18. |
| K7 | S3 E(B−V) calibration per G | E(B−V) | `s3_distance_extinction.py:852` `ebv_metrics` | as K6 (reported; not part of `calibrated`) | v2.1: bias +0.001, NMAD 0.031, **68 % coverage 0.899** (over-covers in every G bin), pull sd 1.255. Fitted κ_E 0.26–0.52 with an 8.8 % outlier fraction: the quoted label errors are 2–4× too large for most stars. |
| K8 | S3 fitted error models | distance, E(B−V) | `s3_distance_extinction.py` (`--parallax-inflation fit`, `--fit-zp-sigma`, `--e-noise fit`) | κ_g per G bin, σ_zp, κ_E | κ_g 2.3 (G ≤ 13) to 1.2 (G 17–18): bright-star parallax errors are inflated about 2×. σ_zp fitted at 0. |
| K9 | S3 log scores of the measurements | distance, E(B−V) | `s3_distance_extinction.py:882` `score_metrics` | best held-out NLL selects the family | v2.1 gmix −2.282; without XP −1.222. v2 flow −2.304 (best). |
| K10 | Interval coverage in the per-epoch anchor metrics | all | `anchored_multisurvey.py:599` | 16–84 % coverage | v10: core [Fe/H] 0.47 and metal-poor 0.47 (under-cover); log g 0.68; E 0.51; parallax 0.60. |
| K11 | Reporting contract (metrics, conformal, strata, schema) | all | `src/masked_stellar_autoencoder/reporting/` (`metrics.py:115` `point_metrics`, `:462` `pit_diagnostics`, `:561` `energy_score`, `:721` `mondrian_coverage`; `strata.py:212` `stage_stratum`, `:260` `tier_stratum`, `:374` `MondrianCalibrator`); historical driver `scripts/run_conformal_validation.py` | pre-registered stage and tier strata; minimum-n pooling | **P**: implemented and tested (`tests/test_reporting_metrics.py`, `test_theory_conformal.py`), **not used** by S1/S2/S3. |
| K12 | Pipeline-path CQR and joint-coverage gate | Teff, log g, [Fe/H] | `src/masked_stellar_autoencoder/pipeline/steps.py:584` `fit_cqr`, `:623` `coverage_after_inverse`, `:684` `gaussian_joint_coverage`, `:705` `choose_joint` | a flow ships only with joint coverage ≥ 0.9 | **P**: separate catalogue path (`docs/pipeline.md`), not run on real data. |

### 1.5 Selection QA

| ID | check | products | script:function | latest run → result |
|---|---|---|---|---|
| S1 | Metal-poor selection: purity and completeness of [Fe/H]ref < −2.5 on the held-out core, per G | [Fe/H] | `likelihood_validation.py:151` `selection_table`; `s1_likelihood_head.py:600`; S2 `stage_report`; ARC `tail_prob.py` (Brier, AUC) | Posterior P > 0.9: 129 selected, purity 0.93, completeness 0.36. S1 likelihood-ratio test (95 % upper bound < −2.5): 0.66 / 0.18. S2: flow 0.95 / 0.05, t 0.97 / 0.17. G > 16: at most 2 of 35 true metal-poor stars recovered (posterior). |
| S2 | Saved-prediction tail audit with paired bootstrap | [Fe/H] | `scripts/analysis/analyze_metallicity_tails.py:115` `selection_metrics`, `:82` `paired_bootstrap` | **H** (October saved predictions). |
| S3 | Non-Gaussianity: fraction of stars with a secondary mode holding ≥ 10 % of the mass | Teff, log g, [Fe/H] | `s2_distributional_heads.py` (`distributional_heads.secondary_mode`) | flow 0.7 %, Gaussian mixture 2.4 %; 1.5–3.5 % at G 16–17.65. `qa_examples.png`. |

### 1.6 Model-internal and numerical QA

| ID | check | script:function | latest run → result |
|---|---|---|---|
| N1 | S1 MLE at box, support or window edges, left-out rows, parallax pull > 3 | `s1_likelihood_head.py:676` `_edges`, `:510` `_on_support_edge` | `plx` arm: support edge 18 % (test G ≤ 16) and 57 % (G > 16); parallax pull > 3 for 7 % at G > 16. |
| N2 | S1 failure diagnostics D1–D5 (latents vs label map vs dimension) | `scripts/s1_diagnostics.py:121` → `likelihood_head_s1_diagnostics_20261121/diagnostics.md` | 1-D raw-label MLE: +0.03/0.35 below −3 (G ≤ 16). Mapped 3-D: +0.32. The label mapping, not the latent, carried the metal-poor bias. |
| N3 | S3 quadrature convergence | `quadrature_<family>.json` (`stage_predict`) | v2.1: \|ΔlogL\| median 1e-6, p99 0.165, max 1.09. |
| N4 | XP forward model: χ²/dof at the labels per G; MLE recovery and coverage | `scripts/run_xp_forward_model.py:124`; `training/xp_forward_model.py` | χ²/dof at the labels 1.02 / 1.01 / 1.28 / 1.15 by G bin. Inference fails: held-out Teff −122/232 K, 68 % coverage 0.18. E at the MLE: slope 0.19 against the labels. Core: Teff NMAD 328 K. |
| N5 | XP forward-model likelihood scans; inference diagnosis | `scripts/scan_xp_forward_likelihood.py:163` (`_summarise:143`); `scripts/diagnose_xp_forward_inference.py:80` → `xp_forward_scan_20261122/scan.md` | Exact likelihood: Teff argmin within 2 steps for 54 % of stars, multimodal for 51 %. (Teff, E) grid: within 100 K for 65 %. The "known" variant runs to the lower E bound for 94 %. |
| N6 | Throughput | S2 `_speed:403` (`speed.json`); S3 `speed.json` | S2 on GPU: posterior 181k, gmix 118k, t-mixture 11.7k, flow 2.2k stars/s (summaries). S3 gmix: 2,032 stars/s with all summaries and scores. |
| N7 | Covariance sidecar validation (XP packing, Cholesky, astrometry) | `scripts/validate_covariance_sidecar.py:58` `check_xp`, `:114`, `:143` | `covariance_sidecar_20261120`. |
| N8 | Age-consensus internals: identifiability gate, convergence, gauge-mode check, residual null curve | `training/age_consensus.py:535` `noise_identified`, `:1357` `record_residuals`; `docs/age-c0-20261122/gauge_mode_check.json` | C0c staged fits stop at 15 outer iterations with coefficient moves of 0.7–8, so function-space convergence is not shown. The null curve explains the "old-age ceiling" as regression to the mean. |

### 1.7 Age-specific QA (consensus programme)

| ID | check | script:function | inputs | latest result |
|---|---|---|---|---|
| G1 | Inventory, crossmatch and pairwise inter-catalogue systematics (Bland–Altman, per class, [Fe/H] and age) | `scripts/age_label_audit.py:977` `inventory`, `:1012` `pairwise`, `:1232` `plots` | `/arc/projects/k-pop/catalogues/` (StarHorse 2023, APOKASC-2/3, APO-K2, LEGACY, Chaplin 2014, Martig 2016, Leung 2023, GALAH DR4, Xiang 2024, AMBRE, FLAME via Gaia TAP); Willett 2026 | `docs/age-audit-20261122/`. APOKASC-3 − Willett: −0.021/0.065 dex. FLAME − StarHorse RVS at [Fe/H] < −1: −0.27. APOKASC-3 − FLAME: +0.89, so FLAME giant ages are unusable. |
| G2 | Leave-one-catalogue-out predictive | `scripts/age_consensus_c0.py:1211` `predict_heldout`; `training/age_consensus.py:1330` `leave_key_out` | C0b fit | `docs/age-c0b-20261123/loo_predictive.csv`, class `all` (mapped bias / NMAD / z-NMAD): Willett-Kepler −0.010/0.065/0.30 (3,807). Xiang −0.000/0.086/0.49. GALAH DR4 −0.002/0.049. APO-K2 −0.001/0.120/1.00. LEGACY BASTA +0.004/0.145 (31). Chaplin +0.026/0.119 (146). Meets the ≤ 0.03 dex bias gate in class `all` except APOKASC-2 (+0.034, n = 45). |
| G3 | **Benchmark arbitration**: lithium depletion boundary (LDB), eclipsing binaries (EB), WD+MS, GC | `age_consensus_c0.py:1614` `bench_map_rows`, `:1344` `offset_vs_consensus` | `--labels` bridge labels (`analysis/age_labels_20261122/`: `wd_ages_published`, `wdms_pairs`, `cluster_priors.csv`) | C0c `arbitration_offsets.csv`. Gauge A: EB +0.077 ± 0.017 (15), LDB −0.157 ± 0.055 (9), WD+MS −0.008 ± 0.111 (6) and −0.072 ± 0.098 (5). Gauge C: EB −0.134, LDB −0.285, WD −0.35 / −0.42. **GC: 0 records.** |
| G4 | Cluster latent ages vs literature | `age_consensus_c0.py:1433`; `make_plots:2124` (`c0c_clusters.png`) | HR23, Cantat-Gaudin 2020 and Dias 2021 ages as records of a cluster node | Gauge A, Gyr (literature EB / LDB in brackets): Praesepe 0.68 (lit. ~0.7); Hyades 1.15 (0.90 / 0.70); Pleiades 0.222 (0.127); M67 2.90 (3.50); NGC 188 3.75 (6.2); young clusters biased by rotation and lithium. |
| G5 | **Chemistry check**: giants (seismic) vs subgiants/MSTO (isochrone) at fixed [Fe/H] and [α/M] | `scripts/age_metal_poor_check.py` (`med_se:35`) | consensus parquet; APOGEE α | `metal_poor_check.csv`, `c0c_metal_poor_check.png`. High-α, [Fe/H] −1 to +0.1: gauge A puts subgiants −0.21 dex younger; gauge C agrees to +0.02. [Fe/H] < −1, high-α: A −0.005 ± 0.022, C +0.198 ± 0.024. |
| G6 | Rotation and lithium cluster ladder (fitted monotone maps) | `age_consensus_c0.py:2051` `_plot_monotone`; `training/age_consensus.py:407` `prefit_monotone` | `rot_*`, `li_*` parquet files in `analysis/age_labels_20261122/` | Gyro maps follow Mamajek & Hillenbrand 2008: the Sun gets 28 d (true 25.4 d), Praesepe K dwarfs 10.7 d. Little age information below 300 Myr. |
| G7 | Fraction of consensus ages above 13.8 Gyr | `summary.json` | – | 6.0 % (gauge A). |

### 1.8 Input and data QA (product-adjacent, listed for completeness)

`scripts/audit_parallax_against_esa.py`, `audit_survey_fits_headers.py`, `audit_survey_inventory.py`, `audit_native_label_covariance.py`, `audit_mastar_gaia_join.py`, `audit_full_gaia_sources.py`, `audit_target_integration.py`, `verify_*`, `validate_covariance_sidecar.py`. These check source admission, units, identity joins and covariance packing, not product accuracy.

### 1.9 Tests covering QA code

`tests/test_s1_pipeline.py`, `test_likelihood_head.py`, `test_s2_distributional_heads.py`, `test_s3_distance_extinction.py`, `test_age_consensus_c0.py`, `test_age_label_audit.py`, `test_catalogue_comparison_validation.py`, `test_seismic_gravity_qa.py`, `test_tess_gravity_stats.py`, `test_reporting_metrics.py`, `test_metallicity_tails.py`, `test_xp_forward_model.py`, `test_covariance_sidecar.py`, `test_conformal*.py`, `test_theory_conformal.py`. There are no tests for the ARC-only scripts (§1).

## 2. Matrix: QA check × product

| QA check | Teff | log g | [Fe/H] | [α/Fe] | distance | E(B−V) | age |
|---|---|---|---|---|---|---|---|
| Held-out residuals vs labels, per G (L1, K6, K7) | I | I | I | M | I (parallax PIT and pull) | I | P (catalogue LOO, G2; no per-star head) |
| Frozen high-res [Fe/H] core per [Fe/H] × G (L2) | – | – | I | M | I | M | – |
| Per-epoch anchor metrics (L3) | I | I | I | M | I (parallax) | I | – |
| IRFM anchors (A1) | I | – | – | – | – | – | – |
| Seismic anchors: APOKASC-3, APO-K2, dwarfs, LEGACY (A2) | – | I | – | – | M | – | I (gauge A, LOO) |
| TESS/K2 seismic (A3) | – | H | – | – | – | – | P (Mackereth 2021, Zinn 2022 as C0b records) |
| Gaia benchmark stars GBS v3 (A4) | P | P | P | M | – | – | – |
| Empirical E labels and SFD zero point (A5) | – | – | – | – | – | I | – |
| Hipparcos / bright geometric distances (A6) | – | – | – | – | H | – | – |
| RAVE DR6 external reference (A7) | H | H | H | M | – | – | – |
| Open clusters (C3, C5, C6) | M | M | P (posterior only; v10 this audit) | M | I | I | I |
| Globular clusters (C4) | M | M | P (posterior only; v10 this audit) | M | M | M | M (0 GC benchmark records) |
| Wide binaries / co-moving pairs (C1, C2, C6) | – | – | P (posterior only; v10 this audit) | M | M | M | I |
| LDB / EB / WD+MS benchmarks (G3) | – | – | – | – | – | – | I |
| Chemistry-cell consistency (G5) | – | – | – | – | – | – | I |
| Calibration per stratum (K1, K3, K6, K7) | I | I | I | M | I | I | M |
| Proper scores (K4, K9) | I | I | I | M | I | I | M |
| Metal-poor selection (S1) | – | – | I | – | – | – | – |
| Without-XP latents (S3 `noxp`) | M | M | M | M | I | I | M |
| Faint-end S/N degradation (design 2026-11-20, step S2) | M | M | M | M | P (G bins to > 18) | P | M |
| Sky / extinction / crowding strata | M | M | M | M | P (\|b\| > 60 only) | P | M |
| Evolutionary-stage strata (`reporting.strata`) | M | M | M | M | M | M | I (classes) |
| Kiel tightness, plan 04 F01 | – | M | – | – | – | – | – |
| Catalogue integrity, plan 04 F11 (duplicates, HEALPix) | M | M | M | M | M | M | M |
| Competing catalogues (§3) | H | H | H | M | M | P (SFD sanity only) | P (age audit) |

**Strata reported by the current checks.**

| check | G | colour / Teff | [Fe/H] | stage | sky / extinction | XP / no-XP |
|---|---|---|---|---|---|---|
| S1 (K1, K2, L2) | 4 bins to 17.65 | Teff bins (coverage) | ref bins | log g bins only | no | no |
| S2 (K3–K5) | 4 bins | no | label and predicted bins | no | no | no |
| S3 (K6–K9, C5) | 6 bins to > 18; predicted-parallax S/N | no | label bins; core | no | \|b\| > 60 vs SFD | yes |
| anchor metrics (L3) | no | no | below −2 and below −2.5 | no | no | no |
| co-natal (C1–C4) | ΔG, G > 16.5 | companion Teff (C2) | pair-mean, Harris bins | giant/dwarf | no | no |
| age (G1–G6) | no | Teff bins (rotation, Li) | yes | MS/MSTO/SGB/RGB/RC | no | – |

## 3. Comparisons with competing catalogues

| catalogue | products | on ARC | script | implemented? | numbers (model, date) |
|---|---|---|---|---|---|
| **Andrae, Rix & Chandra 2023** XGBoost (175M; keep `in_training_sample`) | Teff, log g, [M/H] | yes: `/arc/projects/k-pop/catalogues/andrae2023/table_1_catwise.fits.gz` (3.6 GB), `table_2_catwise.fits.gz` (vetted giants), slim `/arc/projects/k-pop/catalogues/table-1.fits.gz` (124,060,658 rows) | `scripts/compare_released_andrae.py:100` (`scan_fits:58`); `run_andrae_full_footprint_census.py:34`; `run_decision_xp_catalogues.py:109,180`; `compare_remediation_andrae_logg.py`; `compare_expanded_xp_catalogues.py:37` | **H** | **22 Oct** (95,245 test stars): MAE Teff 57.7 vs 62.8 K; log g 0.118 vs 0.071; [Fe/H] 0.074 vs 0.070 ([M/H] proxy). Tail < −2 (n = 141): 0.393 vs 0.209 (Andrae better). **6 Nov** (46,640 GALAH devtest stars outside Andrae training): log g 0.104 vs 0.073 (Andrae better), Teff 53.6 vs 73.4 K and [Fe/H] 0.064 vs 0.071 (MSA better). The full ID census found all 174,922,161 Andrae stars in the raw input. Not run on v10, S1 or S2. |
| **Gaia DR3 GSP-Phot** | Teff, log g, [M/H], distance, A0/E(BP−RP) | yes: HATS `/arc/projects/hats/gaia/dr3` (gaia_source, 1,811,709,771 rows, includes `*_gspphot`) | `scripts/compare_released_gspphot.py:68,118` (Gaia TAP); `compare_uncertainties.py:36,63` | **H** for Teff, log g, [M/H]; **M** for distance and extinction | 22 Oct (67,667 stars), MAE GSP-Phot vs MSA: Teff 255 vs 59 K, log g 0.216 vs 0.118, [M/H] 0.280 vs 0.074. 16/84 % interval comparison in `uncertainty-comparison-20261022.md`. |
| Gaia DR3 GSP-Spec | Teff, log g, [M/H], [α/Fe] | yes (training source `gaia_dr3_gspspec`) | – | not independent: it is a v10 training survey (map slopes 0.87 / 1.03 / 1.00) | – |
| Srivastava+2026 XP [Fe/H] (5M) | [Fe/H] | Zenodo 20645375 | `scripts/compare_released_srivastava.py:55` | **H** | 1,773 stars: MAE 0.0629 (MSA) vs 0.0634; no metal-poor stars in the overlap. |
| Gaia Net (Huson+2025) | Teff, log g, [Fe/H] | no (Zenodo 15048995 pinned; the downloaded copy in `decision_xp_catalogue_qa_20261106_v1` is no longer on ARC) | `run_decision_xp_catalogues.py` | **H** | 6 Nov: 2,497,991 matches; MSA had lower native-reference MAE for all three targets. |
| **Zhang, Green & Rix 2023** (220M) | Teff, log g, [Fe/H], E, distance | no | – | **M** (schema verified only) | – |
| SHBoost 2024 (217M) | Teff, log g, [M/H], distance, A_V, mass, age | **yes**: `/arc/projects/k-pop/catalogues/shboost/part.*.parquet` | – | **M** (data on ARC, unused) | – |
| Ye et al. 2025 FERRE-XP (68.4M) | Teff, log g, [M/H] | no | – | **M** | – |
| **Survey of Surveys DR3** (arXiv:2611.07936); SoS II (Turchi+2025, 23M) | Teff, log g, [Fe/H] on a homogenised spectroscopic scale | no | – | **M**. Cited only as a design precedent (`training/anchor_labels.py` docstring; `research_notes/…/gaia_scale_and_combination.md`) | – |
| **StarHorse 2023** | distance, A_V, age (and mass) | yes: `/arc/projects/k-pop/catalogues/starhorse_ages/{apogee17,galahdr3,gesdr5,lamostl7,lamostm7,ravedr6,rvs,sdssdr12,combined_starhorse}.fits` | `age_label_audit.py` (ages) | ages **I** (audit and C0 source); distance and A_V **M** | Age audit: StarHorse APOGEE − RVS +0.072 / 0.118 dex (input-scale leakage); LEGACY − StarHorse −0.10 to −0.21. |
| **Bailer-Jones+2021** EDR3 distances | distance | no local copy (remote LSDB HATS listed in `/arc/projects/hats/registry/hats_registry.json`) | – | **M** | – |
| **SFD** (Schlegel+1998) | E(B−V) | table column `EBV` (also an encoder input) | S3 `ebv_metrics` (\|b\| > 60) | **P** (zero-point sanity only; circular) | v2.1: prediction − SFD +0.000/0.026 (30,940 stars); labels − SFD −0.002/0.031. |
| **Bayestar19 / Edenhofer 2024 / other 3-D maps** | E(B−V) along the line of sight | no (only `dustmaps` with `SFDQuery` pinned) | – | **M** | – |
| Hipparcos | distance | reference set (historical) | `run_distance_value.py` | **H** | Raw Gaia 0.96 mas MAE was the baseline to beat. |
| RAVE DR6 | Teff, log g, [Fe/H] | yes | `validate_multisurvey_predictions.py:55` | **H** | §1.2 A7. |
| Age catalogues: APOKASC-2/3, Willett 2026, Xiang 2024, GALAH DR4, StarHorse, LEGACY, Chaplin 2014, Leung 2023, Martig 2016, FLAME, AMBRE | age | yes (`k-pop/catalogues/`, audit records under `/arc/projects/astroai/msa/analysis/age_audit_20261122`) | `age_label_audit.py:1012` | **I** | §1.7 G1 (`pairwise_summary.csv`). |

Every number from Andrae, GSP-Phot, Srivastava and Gaia Net is from pre-13-Nov models, before the anchored multi-survey lineage. The comparison contract for the current model is `configs/gaia-xp-comparator-qa-20261106.json` and `docs/reports/msa-expanded-survey-qa-plan-20261105.md` §17. It requires exact Gaia-ID joins, preserved FIT/TUNE/CAL/DEV roles, competitor-training membership, paired errors against a common spectroscopic reference, and spatial or cluster block bootstrap. It has not been executed for v10.

## 4. Gaps

Ordered by impact on a full-Gaia release.

1. **No single evaluator.** Design 2026-11-20 §5 promises `scripts/evaluate_model.py`. It does not exist. QA is spread across the S1, S2 and S3 reports, the fine-tune `jsonl` files and the ARC-only scripts, none of which is under version control or tested.
2. **Co-natal [Fe/H] checks score the posterior head only.** They do not score the S1 (A) or S2 (B0) products. Design 2026-11-22 §9 step 7, "co-moving pairs for A, B0, B1", is not done. The v10 numbers in this catalogue exist only because this audit ran them; they show two regressions against v7:
   - OC giant − dwarf: +0.34 dex (v7 +0.18);
   - held-out cool companions read 0.26–0.32 dex too metal-poor.

   The inputs need membership-probability and R_chance cuts. Statistics must also be split by "neither star labelled", to rule out leakage.
3. **No validation-only physical reference is scored on v10.** IRFM and seismic stars are training anchors, so held-out anchor residuals test transfer, not the gauge. GBS v3, the one validation-only set, is not scored. TESS/K2 seismic QA was lost with the old tree. RAVE DR6 is again independent, because it is not a v10 survey, but it is unused.
4. **No current competitor comparison.** Andrae, GSP-Phot, Srivastava and Gaia Net results are from October and early November models. Zhang, Green & Rix, SHBoost (on ARC), Ye et al., Survey of Surveys DR3 and the StarHorse and Bailer-Jones distances were never compared.
5. **Distance QA rests on Gaia parallaxes and 96 OCs.** Missing:
   - globular-cluster distances (Baumgardt & Vasiliev 2021);
   - wide-pair Δs (shared distance; label-free, and it grows with N);
   - halo standard candles (RR Lyrae, BHB) beyond the parallax horizon;
   - comparisons with Bailer-Jones, StarHorse and GSP-Phot.

   The S3 test set has only 439 stars at G > 18.
6. **E(B−V) uncertainties are mis-calibrated, and there is no independent dust check.**
   - 68 % coverage is 0.90: the fitted κ_E ≈ 0.3 says the quoted label errors are too large.
   - SFD is both an encoder input and the only map comparison.
   - No 3-D dust map, GSP-Phot E(BP−RP) or Harris GC E(B−V) is compared.
7. **[α/Fe] has no QA.** The v6+ anchored models carry no α label. Native per-pipeline α quantities, GC α-enhancement, thick/thin sequences and GBS abundances are all unused.
8. **There is no per-star age product yet** (stage C1). Missing:
   - coverage per evolutionary stage;
   - held-out clusters by cluster ID for a head;
   - the seismic precision ceiling (0.045 dex);
   - young main-sequence informativeness;
   - GC ages (bench_gc has 0 records; VandenBerg 2013 not acquired).

   The gauge choice (A vs C) is unresolved: the EB and WD+MS benchmarks favour A, the chemistry check favours C for old stars.
9. **No-XP and faint-end tests cover S3 only.**
   - S1 and S2 have no no-XP evaluation. Full Gaia is ~2 × 10⁹ stars, of which ~2.2 × 10⁸ have DR3 XP.
   - The S/N-degradation test (design 2026-11-20, step S2) has not been run.
   - The labelled table stops at G = 17.65.
10. **No sky, extinction, crowding or stage strata** for Teff, log g and [Fe/H]. `reporting.strata` (stage and tier strata, minimum-n pooling) exists but is unused.
11. **No catalogue-level QA**: duplicates, HEALPix counts, units round-trip, monotone quantiles (plan 04 F11, gates G1–G7). No Kiel-tightness statistic (F01). No propensity or positivity diagnostics, and no PPI vs NPMLE agreement (design 2026-11-22 §5, §9).
12. **Coverage is of noisy labels.** It is not EIV-corrected except in the S2 EIV heads. The S1 Neyman sets are calibrated on equally noisy valid labels. The anchor-metric `cov` of 0.47 for core [Fe/H] is untreated.
13. **Lost history.** Everything under `/arc/projects/k-pop/msa_runs/` is gone. Any number quoted from `docs/reports/` before 13 Nov must be recomputed before it is compared with the current lineage.

## 5. Proposed QA suite for the full-Gaia run

### 5.1 Rules, from the existing documents

- Freeze a fresh, group-aware final panel before scoring. Protect clusters, binaries, benchmarks and calibrator families from FIT, TUNE and CAL. Fix thresholds before the panel is read (`msa-expanded-survey-qa-plan-20261105.md`, Evaluation governance).
- Bootstrap by the independent unit: star, cluster, pair or HEALPix block. Bins with n < 20 are descriptive. Per-bin tests use Wilson 95 % intervals with a Bonferroni split.
- Keep products A, B and C separate (design 2026-11-22 §2). Report coverage of the label and, where an EIV model exists, of θ.
- One driver, `scripts/evaluate_model.py`, writes `/arc/projects/astroai/msa/runs/qa_<model>_<date>/` with `metrics.json` (schema `reporting.schema`), `report.md` and `qa_*.png`. It reads saved predictions only. Bring the ARC-only scripts into the repo, with tests.

### 5.2 Pre-registered strata, applied to every check

| axis | bins |
|---|---|
| G | ≤ 13.5, 13.5–15, 15–16, 16–17, 17–18, 18–19, > 19 (beyond 17.65 only label-free checks and competitor overlaps exist) |
| colour | BP−RP < 0.5, 0.5–0.8, 0.8–1.2, 1.2–1.8, 1.8–2.5, > 2.5; or Teff bins (`likelihood_validation.TEFF_BINS`) |
| [Fe/H] | −9, −3, −2.5, −2, −1.5, −1, −0.5, +9, by reference and by prediction |
| stage | `reporting.strata.stage_stratum`: dwarf_ms, turnoff_sgb, rc, rgb, agb_other |
| sky / extinction | \|b\| < 5, 5–15, 15–30, 30–60, > 60; SFD E(B−V) < 0.05, 0.05–0.3, 0.3–1, > 1; source density (crowding); HEALPix nside 16 blocks |
| data tier | `reporting.strata.tier_stratum`: xp_full, xp_partial, no_xp, gaia_only; predicted-parallax S/N < 1, 1–3, 3–10, > 10; RUWE < 1.4 / ≥ 1.4 |

### 5.3 Checks and acceptance criteria per product

The Source column is **doc** when the criterion is already written in a design document or in code, and **proposed** when this catalogue suggests it.

| product | check | reference | metric | acceptance | source |
|---|---|---|---|---|---|
| all | catalogue integrity | Gaia DR3 HATS (`/arc/projects/hats/gaia/dr3`) | duplicates, counts per HEALPix, finite and flag fractions, units round-trip, monotone quantiles, support-edge fraction | 0 duplicate `source_id`; no HEALPix cell > 5σ from expectation; units round-trip exact | doc (plan 04 F11, G1–G2) |
| all | throughput | – | stars/s per GPU for summaries | ≥ 10⁵ | doc (design 2026-11-22 §6) |
| Teff | IRFM held-out anchors, per stratum | `anchor_irfm_colte` | bias, NMAD | NMAD ≤ 1.05 × incumbent; \|bias\| ≤ \|incumbent\| + 10 K | doc (`likelihood_validation.decide`) |
| Teff | GBS v3 and interferometric Teff (validation only) | `gaia_fgk_benchmarks.parquet` | bias, NMAD with n | descriptive (n ≈ 200); report every star | proposed |
| Teff, log g, [Fe/H] | interval calibration per stratum (B) | test labels; EIV for θ | conformal HPD coverage at 68 / 95 % | within ±0.03 in every stratum with n ≥ 20, including G > 16 and [Fe/H] < −2.5 | doc (design 2026-11-22 §6; S2 `_decide`) |
| Teff, log g, [Fe/H] | Neyman coverage (A) | test labels | joint 68 % set | 0.68 ± 0.08 per (θ, G) bin | doc (S1 `decide`) |
| Teff, log g, [Fe/H] | proper scores vs incumbent | test labels | log, energy and variogram scores, CRPS | better than the incumbent overall; no [Fe/H] × G cell worse beyond max(0.02, 2 SE) in bias or max(0.01, 2 SE) in NMAD | doc (S2 `_decide`) |
| log g | seismic held-out anchors, per stage | APOKASC-3, APO-K2, dwarfs, LEGACY; TESS Hon+2021 and K2 GAP (re-stage) | bias, NMAD | NMAD ≤ 1.05 × incumbent; \|bias\| ≤ \|incumbent\| + 0.01 dex | doc (`decide`) |
| log g | residual vs covariates | test labels | median residual vs Teff, log g, [Fe/H], G, E(B−V) | no bin with \|bias\| > 0.1 dex for n ≥ 50 | doc (plan 04 F02) |
| log g | Kiel tightness | measured-label ridge | R_K = σ_K / σ_ref | R_K < 1 with the CI excluding 1, and no log g MAE regression | doc (plan 04 F01) |
| [Fe/H] | frozen core bias per [Fe/H]ref × G | high-res core | median offset | \|bias\| ≤ 0.10 dex in every bin up to G = 16; metal-poor bias at G > 16 halved relative to the incumbent | doc (S1 `decide`; design 2026-11-20 §7 S3a) |
| [Fe/H] | bulk precision | core, [Fe/H] ≥ −1, G ≤ 16 | NMAD | ≤ incumbent + 0.01 dex | doc (S1 `decide`) |
| [Fe/H] | metal-poor selection | core, [Fe/H] < −2.5 | purity, completeness per G | purity ≥ 0.9 at stated completeness; LR-test false-positive rate controlled per θ | partly doc (design 2026-11-20 §3), threshold proposed |
| [Fe/H] | wide binaries, all pairs and held-out pool | El-Badry 2021 with R_chance < 0.1; neither star labelled | Δ[Fe/H] median per companion Teff and per G; z distribution | \|median Δ\| ≤ 0.05 dex per companion-Teff bin; z-NMAD 0.9–1.1 | proposed |
| [Fe/H] | open clusters | HR23 with Prob ≥ 0.7; held-out clusters | giant − dwarf, faint − bright slope, χ²/dof about the cluster mean | \|giant − dwarf\| ≤ 0.05 dex; χ²/dof ≈ 1 (median 0.8–1.25) | proposed (plan: "never require zero scatter") |
| [Fe/H] | globular clusters | Vasiliev & Baumgardt 2021 members; Harris 2010 | median(pred − Harris) per Harris bin; within-cluster χ²/dof | \|median\| ≤ 0.10 dex per bin | proposed |
| [α/Fe] | native α per pipeline; GC α-enhancement; same α in OCs and pairs; thick/thin separation at fixed [Fe/H]; GBS and Casamiquela 2026 | APOGEE [α/M], GALAH [α/Fe] and [Mg/Fe] (never interchanged) | bias, NMAD, coverage per stratum | coverage ±0.03; same per-cell rule as [Fe/H] | proposed |
| distance | parallax calibration per G and predicted-parallax S/N | Gaia parallax, L21 zero point, no cut | PIT coverage; mean pull | coverage ±0.03 per bin; \|mean pull\| ≤ max(0.05, 3 SE) per G bin | doc (S3 `_decide`) |
| distance | bright geometric baseline | Hipparcos and VLBI | parallax MAE | must not regress against raw Gaia (0.96 mas on 4,166 Hipparcos stars) | doc (plan 04 F08) |
| distance | clusters and pairs | OC members (geometric distance from members' parallaxes); GC distances (Baumgardt & Vasiliev 2021); wide pairs | χ²/dof of members' s; Δμ vs geometric; component Δs z-scores | χ²/dof ≈ 1; \|Δμ bias\| ≤ 0.03 mag; z-NMAD 0.9–1.1 | proposed |
| distance | beyond the parallax horizon | RR Lyrae and BHB standard candles; GC distances | Δμ per G and \|b\| | descriptive, then bias ≤ 0.05 mag | proposed |
| distance | competitors | Bailer-Jones 2021, StarHorse 2023, GSP-Phot, SHBoost, Zhang 2023 | paired pull against the parallax likelihood, on exact-ID overlap | descriptive; no catalogue is truth | proposed (comparator contract) |
| E(B−V) | calibration per G and \|b\| | empirical labels (EIV) | PIT coverage, pull sd, κ_E | coverage ±0.03 per bin (fails today at 0.90) | doc for S3 reporting; acceptance proposed |
| E(B−V) | independent dust | Bayestar19 or Edenhofer 2024 at the predicted distance (to ingest); Harris GC E(B−V); GSP-Phot E(BP−RP) | residual vs distance, \|b\| and SFD bin | bias ≤ 0.02 mag at E < 0.3 | proposed |
| E(B−V) | co-natal | OC and GC members | χ²/dof of members' E | ≈ 1 after a differential-reddening term | proposed |
| age (C0) | consensus gates | held-out methods, clusters, synthetic truth | bias per class; cluster z-scores | held-out-method bias ≤ 0.03 dex per class; cluster z ~ N(0, 1); maps recover synthetic truth | doc (design age §7) |
| age (C1 head) | distributional head | consensus measurement model | held-out log score; coverage per Mondrian stage stratum | beats single-teacher (StarHorse-only) heads; coverage ±0.03 per stratum | doc (design age §7) |
| age | held-out clusters; wide-pair (τ₁ − τ₂)/σ per class and G; seismic ceiling (0.045 dex); [Fe/H] < −1 and age > 10 Gyr slices; fraction > 13.8 Gyr; LDB, EB, WD+MS and GC arbitration; chemistry-cell giant vs subgiant | clusters, El-Badry pairs, APOKASC-3 gold, LEGACY, benchmark tables | z-NMAD; offsets ± SE | z-NMAD ≈ 1; n < 20 descriptive; no fixed gauge pass mark (the user's decision) | doc (design age §6) |
| all, no-XP tier | modality dropout | same panels with the XP inputs hidden (S3 `noxp` mechanism) | every metric above, per G | reported separately; the coverage rules apply per tier | proposed |
| all | competitors on exact-ID overlap | Andrae (with `in_training_sample`), Zhang, SHBoost, Gaia Net, Ye, GSP-Phot, SoS DR3 | paired ΔMAE against a common spectroscopic reference; block bootstrap; catalogue-wide agreement maps | descriptive; tail-only wins labelled as such | doc (comparator contract; plan 04 §5) |
| C (population) | PPI vs NPMLE agreement per cell; propensity and positivity | labelled stars vs the parent catalogue | effective sample size per cell; interval coverage over repeated label splits | cells with π(x) ≈ 0 reported as unsupported | doc (design 2026-11-22 §5, §9) |

### 5.4 First steps, cheapest first

1. Commit the ARC-only scripts (`consistency_analysis.py`, `binary_heldout.py`, `make_targets.py`, `rescore_analysis.py`, `estimator_table.py`, `tail_prob.py`) as functions of one `evaluate_model.py`. Add membership-probability and R_chance cuts, and split labelled from unlabelled stars.
2. Score GBS v3 and RAVE DR6 on v10 and on every S1/S2 candidate. Re-stage TESS and K2 GAP.
3. Encode the consistency stars (`consistency_table_20261118/stars.fits`) through S1 and S2, so that the co-natal checks score A and B0, not only the posterior head.
4. Re-run Andrae and GSP-Phot (local HATS) on the v10/S2 overlap. Add SHBoost (on ARC), Zhang 2023 and SoS DR3.
5. S3: add GC distances, wide-pair Δs, and Bailer-Jones and StarHorse comparisons. Fix the E(B−V) error scale before claiming coverage.
6. Add the stage, sky/extinction and no-XP strata to the S1 and S2 reports, and run the S/N-degradation test.
7. Start α QA when α returns to the label set. Start age C1 QA when the head exists.

## Appendix: reproducing the v10 co-natal numbers (this audit)

```bash
export PATH=$HOME/.pixi/bin:$PATH PYTHONNOUSERSITE=1; unset PYTHONPATH
A=https://ws-uv.canfar.net/arc/files/projects/astroai/msa
C="--cert $HOME/.ssl/cadcproxy.pem"
mkdir -p qa_data qa_tools && cd qa_data
curl -sSL $C "$A/runs/multisurvey_finetune_20261121_v10/consistency_v10.npz" -o consistency_v10.npz
curl -sSL $C "$A/analysis/consistency/consistency_targets.npz" -o consistency_targets.npz
curl -sSL $C "$A/analysis/consistency/diag_data_vizier_cfa.tar.gz" | tar xz
curl -sSL $C "$A/runs/consistency_table_20261118/labelled.npz" -o labelled.npz
curl -sSL $C "$A/runs/binary_pool_20261120_pairs/pool.npz" -o pool.npz
cd ../qa_tools
curl -sSL $C "$A/analysis/scripts/consistency_analysis.py" -O
curl -sSL $C "$A/analysis/scripts/binary_heldout.py" -O
cd ..
pixi run python -I qa_tools/consistency_analysis.py qa_data/consistency_v10.npz \
  qa_data/consistency_targets.npz qa_data/diag_data --labelled qa_data/labelled.npz
pixi run python -I qa_tools/binary_heldout.py qa_data/pool.npz v10=qa_data/consistency_v10.npz
```

`consistency_analysis.py` also writes `consistency_v10_clusters.csv` next to the predictions. The v7 reference output is `/arc/projects/astroai/msa/analysis/consistency/consistency_v7.txt`.
