# QA of the v10 chain with the single evaluator (2026-11-24)

First run of `scripts/evaluate_model.py` (qa.md §5) on the saved v10 predictions,
read-only. `report.md` is the evaluator's report, `metrics.json` every number,
`plots/` the figures, `clusters_*.csv` the per-cluster tables. The same files and
the staged inputs are on ARC under `/arc/projects/astroai/msa/runs/qa_v10_20261124/`
(`out/`, `inputs/`, with `inputs/sha256.txt` and `inputs/run_eval.sh`).

## What was evaluated

| set | file (ARC `runs/`) | product |
|---|---|---|
| `posterior_v1test` | `multisurvey_finetune_20261121_v10/multi_v10_test.npz` (129,927 stars of the v1-table test, never in the v5 table) | posterior Student-t head, B (16/50/84 %) |
| `posterior_core` | `multisurvey_finetune_20261121_v10/multi_v10_core.npz` (39,073 frozen-core rows of the v5 table) | posterior head |
| `s1_all_ens` | `likelihood_head_s1_v10_ens_20261122/s1_predictions.npz` | S1 arm `all`, K = 5: MLE and Neyman sets (A) |
| `s2_flow` | `distributional_heads_s2_v10_20261122/s2_predictions_flow.npz` | S2 flow (B0) and its Mondrian conformal sets |
| `s3_gmix` | `distance_extinction_s3_v21_20261123/s3_predictions_gmix.npz` | S3 v2.1 distance and E(B−V); parallax and E PITs |
| co-natal | `multisurvey_finetune_20261121_v10/consistency_v10.npz` (1,209,544 stars) | posterior head only (S1/S2 were not run on these stars) |

Rows: `msa-split-v1` TEST only (frozen hash + `overrides.parquet`, sha256
`46ad0f6f…4b55e`); rows of the v5 table's train and valid splits are dropped
(`--exclude-table-splits 0,1`, data.md §3.10): v10 and its heads saw their labels.
QA-reserved objects (GBS, globular clusters) are scored report-only, and only when
not seen. Every model here started from the July encoder, which saw the inputs of
every XP star: TEST is label-fresh, not input-fresh.

## Inputs and how they were made

- **References.** `posterior_*`: the table label in the prediction file (one
  survey's native value, not the common scale; coverage is of that label).
  S1/S2: common-scale labels from `likelihood_head_s1_v10_ens_20261122/labels.npz`
  (TEST and reserved rows: `inputs/labels_v5_common_scale_test.parquet`). Core:
  `multisurvey_table_20261117_v5/records.npz` with the v10 `fe_h_core` list
  (`qa/core.py`, the order of the former ARC `rescore_analysis.py`).
- **Covariates** (G, BP−RP, l, b, RUWE, has_xp) and **GSP-Phot**: Gaia archive, for
  the 309,722 TEST or reserved ids (`scripts/stage_qa_covariates.py`). SFD E(B−V)
  only exists for the posterior sets (their files carry the table's `EBV`).
- **Competitors**: rows of the TEST ids staged on CANFAR by
  `evaluate_model.py --stage-competitors` (launcher step `qa-v10-stage`, attempt 3,
  session f8txmamq): Andrae 2023 `table-1.fits.gz` (in-training stars removed),
  SHBoost (129 parts), StarHorse 2023 spectroscopic runs. Zhang/Green/Rix 2023,
  Survey of Surveys DR3, Bailer-Jones 2021, Bayestar/Edenhofer, Ye 2025 and Gaia Net
  are not on ARC (listed as missing).
- **Co-natal**: VizieR dumps of `analysis/consistency/diag_data_vizier_cfa.tar.gz`
  (El-Badry+2021 with R_chance, HR23 members and clusters, Harris), labelled list
  `consistency_table_20261118/labelled.npz`, held-out pool
  `binary_pool_20261120_pairs/pool.npz`.
- **Physical**: seismic catalogues of `/arc/projects/k-pop/catalogues/`; GBS v3 with
  the exact DR3 ids (`/arc/projects/astroai/msa/catalogues/gaia_fgk_benchmarks_v3_dr3ids.parquet`).

## Reading it

- S1 and S2 sets cover the common-scale label at 0.66 (68 %) and 0.92 (95 %):
  slightly under nominal everywhere, with the metal-poor, hot (BP−RP < 0.5) and
  low-latitude strata failing. Their own sets carry no membership for rows without
  all three labels; those rows are not scored (they read as uncovered otherwise).
- The posterior head's 16–84 % interval covers its native-survey test label at
  0.34–0.53: the interval has no label-noise or survey-offset term.
- S3: parallax 68 % coverage 0.720 (over-covers), E(B−V) 0.898 (the label errors
  are too large; qa.md K7).
- Frozen core under msa-split-v1 TEST minus v5 train/valid holds 296 rows, 12 below
  −2.5: every metal-poor cell is descriptive. A decision-grade core needs a model
  trained on FIT.
- Co-natal checks reproduce the audit's v10 numbers on the TEST groups: fainter
  companions read 0.09 dex more metal-poor, cool held-out companions 0.26–0.31 dex
  too metal-poor, open-cluster giants +0.33 dex above dwarfs.
- 74 of the 192 GBS stars are rows of the v5 fine-tune table (63 in train), because
  the table builder excluded the benchmarks by their float-rounded ids; only the
  unseen ones are scored (4–5 stars).
