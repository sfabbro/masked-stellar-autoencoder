# 01 — Current state snapshot (authoritative "where we are")

Compiled 2026-10-22 ~15:30 UTC by `state-audit`; read-only audit at HEAD `1c51c85`. Local disk: **85 GB**
artefacts, **138 GB** free. NB: while this audit ran, another agent added uncommitted
`density-precision-certificate-*` configs/scripts/tests and modified `batch_scripts/canfar_entrypoint.sh` +
`scripts/launch_stellar_improvement.py`; those are **not** part of this audit (my only write is this file).
**Evidence legend** — `[V]` verified locally this session (SHA/row/array recompute or HEAD match); `[P]`
producer-only (manifest/receipt, not re-derived); `[R]` citation-only/remote on CANFAR (not inspected).
Primary sources `[V]`: `.../density-review-20261022/{point,distribution,interval,capability,inventory}.csv`
+ `metrics.json` + `validation.json` (136/182/486/43 rows, `passed`); `docs/reports/msa-density-review-20261022.html`;
evaluated library `4e81a72`, 67 input SHAs (`metrics.json`). Cohorts (never mix):

| cohort id | n | tail FeH<-2 | meaning |
|---|---|---|---|
| `native3` (complete-case joint Teff/logg/FeH test) | **97,695** | **141** (14 below −3) | production target cohort of all 10M screens |
| `scalarFeH` (enriched-cohort identity test) | **100,620** | **185** | scalar FeH screens / XP-mask evidence |
| `frozenScalarFeH` | 100,620 | 185 | frozen-encoder head probes |

## 1. Current best per label

### 1a. Point metrics, `native3`, verified means `[V]`

All `native3` rows use the same 97,695 ids; MAE/RMSE/bias/NMAD/R² are recomputed inside each subset. `scalarFeH` rows are labelled.

| label | subset | method (run) | MAE | RMSE | bias | NMAD | R² | tail n |
|---|---|---|---|---|---|---|---|---|
| Teff [K] | global | **Long noise Gaussian 270M (selected99M)** `multi_gaussian_noise_recon_30e_v1`/fp0qj7pm | **58.83** | 91.12 | +2.89 | 61.23 | 0.9832 | — |
| Teff | global | Huber 10M `multi_huber_recon_s42_v1` (best verified *matched-budget*) | 71.62 | 104.74 | +5.48 | 76.83 | 0.9778 | — |
| Teff | global | Gaussian s42 10M `multi_gaussian_recon_s42_v1` | 70.29 | 109.34 | −4.35 | 71.80 | 0.9758 | — |
| Teff | FeH<−2 | **Long noise Gaussian 270M(99M)** | **161.34** | 209.32 | **+155.06** | 118.72 | 0.6488 | 141 |
| Teff | FeH<−2 | Huber 10M | 164.17 | 203.29 | +151.01 | 117.34 | 0.6687 | 141 |
| logg [dex] | global | **Long noise Gaussian 270M(99M)** | **0.1200** | 0.1816 | −0.002 | 0.1224 | 0.9711 | — |
| logg | global | Huber 10M | 0.1438 | 0.2074 | −0.007 | 0.1534 | 0.9623 | — |
| logg | FeH<−2 | **Long noise Gaussian 270M(99M)** | **0.1998** | 0.2911 | +0.1307 | 0.2090 | 0.8011 | 141 |
| logg | FeH<−2 | Huber 10M | 0.2263 | 0.3225 | +0.1423 | 0.2187 | 0.7559 | 141 |
| FeH [dex] | global | **Long noise Gaussian 270M(99M)** | **0.0758** | 0.1160 | −0.002 | 0.0773 | 0.8698 | — |
| FeH | global | Huber 10M | 0.0915 | 0.1356 | +0.000 | 0.0953 | 0.8220 | — |
| FeH | global | (cohort `scalarFeH`) Residual Huber | 0.0854 | — | — | — | 0.8464 | 100,620 |
| FeH | FeH<−2 | **Long noise Gaussian 270M(99M)** | **0.3931** | 0.5365 | **+0.3877** | 0.2249 | −7.48 | 141 |
| FeH | FeH<−2 | Huber 10M | 0.4640 | 0.6032 | +0.4629 | 0.2297 | −9.72 | 141 |
| FeH | FeH<−2 | (cohort `scalarFeH`) Residual Huber | 0.5158 | — | — | — | −3.18 | 185 |

Reading rules: the Long arm has the lowest **MAE** in every cell but runs a **different budget/objective**
(270M presentations, 30×9M, noise-aware diagonal Gaussian+recon, checkpoint selected at 99M on reserved tune
mean observed NLL `[P]`); it is not an isolated winner, its density is *not* verified (1b), and it is *not*
uniformly better (tail Teff RMSE/R² still favour Huber: 203.3/0.669 vs 209.3/0.649). Its metal-poor
Teff/logg/FeH bias is huge (+155 K, +0.13 dex, +0.39 dex) — systematically under-predicting the tail.
All tail R² are negative for FeH because tail variance is tiny (`metrics.json:limits`).
`Forward inversion XP-only` tail FeH 0.2943 is **`[P]` only** (`FAILED solver replay; producer diagnostic only`,
`inventory.csv:point_verified=False`) — not usable evidence.

### 1b. Distribution metrics, `native3` `[V]` where score_status allows

Only Gaussian arms have *verified* NLL/CRPS/predictive-quantiles. Flows expose joint density values only, so
`nll`/`crps`/`pit_*` are `null` for them; only Dawid–Sebastiani (moment) scores exist, and those are **not
comparable** to Gaussian NLL (joint K·dex·dex vs marginal).

| label | subset | method | NLL (nat) | CRPS | PIT KS | mean pred sd | status |
|---|---|---|---|---|---|---|---|
| Teff | global | **Gaussian s42 10M** | **5.8319** | **50.4475** | 0.0556 | 98.76 | verified density |
| Teff | global | Long noise 270M(99M) | 5.7440 | 43.2786 | 0.0570 | 83.10 | `[P]` diagnostic, density replay FAILED |
| Teff | FeH<−2 | Gaussian s42 / Long | 7.1086 / 7.1462 | 139.15 / 119.77 | 0.429 / 0.521 | 218.4 / 104.5 | verified / `[P]` |
| logg | global | **Gaussian s42 10M** | **−0.3714** | **0.1027** | 0.0537 | 0.194 | verified density |
| logg | global | Long noise 270M(99M) | −0.4086 | 0.0876 | 0.0391 | 0.155 | `[P]` diagnostic |
| logg | FeH<−2 | Gaussian s42 / Long | 0.8754 / 0.1429 | 0.2835 / 0.1471 | 0.342 / 0.212 | 0.547 / 0.209 | verified / `[P]` |
| FeH | global | **Gaussian s42 10M** | **−0.7656** | **0.0697** | 0.0551 | 0.136 | verified density |
| FeH | global | Long noise 270M(99M) | −0.9758 | 0.0548 | 0.0343 | 0.096 | `[P]` diagnostic |
| FeH | FeH<−2 | Gaussian s42 / Long | 2.6437 / 2.0541 | 0.5497 / 0.2881 | 0.667 / 0.541 | 0.617 / 0.285 | verified / `[P]` |

No flow arm (MAF/cINN/NSF/BPF) beats Gaussian s42 on global point MAE for any label
(FeH MAE: MAF 0.1053, cINN 0.1005, NSF 0.1011, BPF 0.1034 vs s42 0.0967; same ordering on Teff/logg), yet flows improve joint NLL
(MAF 4.26752 vs Gaussian 4.69485 joint-native `[P]`, progress md 2026-10-22T09:09/14:20).
Density-level-set CP (NSF/NSF+contrastive only) global 90% → cov 0.8897/0.8904; tail 141 → 0.1844/0.2128 `[V]`.

### 1c. Interval coverage (`interval.csv`, conformal CP — independent of density verification) `[V]`

| label | subset | method | CP90 marginal cov (nom 0.90) | mean width | joint Bonferroni box (nom 0.90) |
|---|---|---|---|---|---|
| Teff | global | Huber 10M | 0.8978 | 301.1 K | 0.9278 |
| Teff | global | Gaussian s42 | 0.8968 | 292.1 K | 0.9254 |
| Teff | global | Long noise 270M(99M) | 0.8843 | 232.2 K | 0.9187 |
| Teff | FeH<−2 | Huber / s42 / Long | 0.5674 / 0.4184 / 0.4752 | — | 0.2837 / 0.0284 / 0.2908 |
| logg | global, FeH<−2 | Huber / s42 / Long | 0.8990 / 0.8956 / 0.8931 · tail 0.7234 / 0.4894 / 0.7305 | — | — |
| FeH | global | Huber / s42 / Long | 0.8954 / 0.8977 / 0.8877 | 0.383 / 0.415 / 0.304 dex | — |
| FeH | FeH<−2 | Huber / s42 / Long | **0.1844 / 0.0000 / 0.2128** | — | 0.0000 (s42) |

Marginal coverage is well calibrated globally (~0.89) and **catastrophically uncalibrated in the metal-poor
tail** on every label (FeH: 0–21%, Teff: 42–57%, logg: 49–73%). Tail FeH 90% interval width (0.30–0.42 dex)
is smaller than the observed tail bias (0.39–0.46 dex): the model is confidently wrong, not noisy.

## 2. Other-label evidence

### 2a. Distance — pilots only, no production head

| item | value | source |
|---|---|---|
| fit / identity-test / Hipparcos-ref n | 4,452 / 22,932 / 4,166 (159 non-positive Hip parallaxes retained) | `[V]` `distance_feature_cohort_v1/manifest.json`, `distance_value_frozen_v3/manifest.json` |
| local disk baseline (raw Gaia DR3 parallax) | **MAE 0.960 mas**, RMSE 1.3215, bias −0.111, N=4,166 | `[V]` `distance_value_frozen_v3/verified-comparison/comparison.md` |
| parallax-free ridge head | MAE **42.30 mas**, RMSE 695.9 (extreme extrapolations, max 41,403 mas) | `[V]` same |
| parallax-assisted ridge head | MAE **3.226 mas**, RMSE 11.76, bias −0.949 | `[V]` same |
| native StarHorse kpc agreement | free 0.0905 kpc / assisted 0.0314 kpc (N=22,932) | `[V]` same |

**Both heads are worse than doing nothing** on the only independent geometric reference; the report itself says
"no further distance wave is justified by these results alone" (`comparison.md`). Regression by posterior-distance
bin is severe: assisted MAE 0.018 kpc (0–0.5 bin) → 0.302 kpc (2–4 kpc bin) `[V]` `conditional-metrics.csv`.
No production distance head exists in any 10M/long run — all manifests carry
`scope: ... no distance/age/alpha/astrometry` (e.g. `multi_gaussian_noise_recon_30e_v1/manifest.json`, `[V]`).

### 2b. Age — plumbing verified, no science result

| item | value | source |
|---|---|---|
| fit cohort / identity test | **4,440** / 22,830 strict StarHorse native ages | `[V]` `native-age-comparison/comparison.md`, `age_feature_cohort_v1/age-cohort-report.json` |
| native MAE (frozen / SFT / SFT+recon) | 1.1427 / 1.0357 / **1.0222 Gyr**; bias +0.048 / −0.207 / −0.239 | `[V]` same (checkpoints local, 119 MB each) |
| external seismic, 50 LEGACY stars, 6 pipelines | frozen 2.66–3.29 Gyr; SFT 1.79–2.43; recon **1.72–2.36 Gyr**, **bias +1.68 to +2.34 Gyr** | `[V]` `native-age-comparison/conditional-metrics.csv` |
| 8–14 Gyr native bin · external refs | recon MAE 1.475 Gyr, bias −1.189 · 37 quality ages of 66 ids, 50 feature-available | `[V]` same, `age-cohort-report.json` |
| density/NLL for age | **none exists** (no quantile/flow age arm) | `[V]` absence in all CSVs |

Age predictions are ~1.7–2.3 Gyr *too old* vs seismic ages for the same stars — a signed, physics-scale bias,
not a precision issue. StarHorse posterior ages are model/parallax-conditioned targets, not truth.

### 2c. Alpha — no native labels, no training, no metrics

- Production target list is exactly `teff, logg, fe_h` (`[V]` `multi_huber_recon_s42_v1/native-provenance.json:target_names`).
- `joint-native-labels.h5` contains only `quantity == 'FeH'` (737,001 measurements) plus raw GALAH/APOGEE
  Teff/logg/FeH columns — **no alpha column** (`[V]` h5 inspection, `joint-native-preparation-v2/`).
- 151-job ledger (`experiments/xp/launch-stellar-improvement-20261015.json`) contains **zero** alpha runs `[V]`.
- APOGEE `alpha/M` is not alpha/Fe and cross-survey scale is unaudited (BRIEF). **Alpha is the only goal label with
  no cohort, no checkpoint, no point metric and no density — the single total-evidence gap among the six labels.**

### 2d. XP-off / photometry-only evidence

| probe | result | source |
|---|---|---|
| frozen-encoder head, all XP blocked, FeH, n=20,000 tune | MAE **0.14482 → 0.16205** (+0.0172 dex, +11.9%) | `[V]` `photometry-ablation.csv`, `photo_value_no_xp/manifest.json` |
| full 10M Huber *training* with all XP masked, n=100,620 | MAE **0.08231 (parent86 unmasked) → 0.09208**; tail −2 → **0.44884 → 0.58814**; below −3 (n=14) 1.914 | `[V]` `targeted-parent-verified-comparison/comparison.md`, `target_e72_xpmask_s42_10m_v1/manifest.json` |
| XP noise+mask | 0.09462 all / 0.61698 tail | `[V]` `target_e72_xpnoise_mask_s42_10m_v1/manifest.json` |
| leave-one-survey-out ablation, all (n=20,000) | SkyMapper +0.0163 dex (largest), **XP +0.0172**, 2MASS +0.0021, WISE +0.0002, SDSS −0.0006, PS1 −0.0015 | `[V]` `photometry-ablation.csv` |
| same, metal-poor subset (n=544) | SkyMapper +0.0979, XP +0.0223, SDSS +0.0094, PS1 −0.0098 | `[V]` same |

So an XP-free model is trainable but currently costs ~12% global FeH accuracy and **~31% in the tail**; no XP-off
run has been done for Teff/logg joint densities, and no all-Gaia inference store exists.

## 3. Asset inventory

| asset | local path (SHA verified) | remote | status |
|---|---|---|---|
| parent86 (production parent) | `.../main-checkpoints/checkpoint-86.pth` sha `d1ed550e…fb3`, 409,873,692 B `[V]` | `/arc/projects/k-pop/msa_runs/pretrain-e21-full100-maskhalf-20261010-ram128-mb16k/pretrain/checkpoints/msa_pretrain_checkpoint_86.pth` | **preserved, refit-required** |
| parent85 | `.../main-checkpoints/checkpoint-85.pth` sha `cca5775e…4f3` `[V]` | same dir `_85.pth` | preserved (85 not used by current runs) |
| parent72 | `.../prediction-comparison-e72/parent/checkpoint-72.pth` sha `9de961ce…6ec` `[V]`, 409,873,692 B | same dir `_72.pth` (path in `configs/multilabel-*`) | preserved locally; no manifest SHA found `[P]` gap |
| parent57 | **absent locally** | same dir `_57.pth`, sha `9b033ae6…b2ca` (`docs/xp-lr-batch-experiments.md:3`) | ARC-only `[R]` |
| fp0qj7pm selected 99M | `.../multi_gaussian_noise_recon_30e_v1/checkpoints/checkpoint-000099000000.pth` sha `f96e5487…d79` `[V]`, 410,386,779 B | ARC `.../multi_gaussian_noise_recon_30e_v1` | **retrievable**, 30/30 periodic states + final local (12 GB) |
| dd27e718 states | `.../residual_huber_e86_150m_30e_v1/checkpoints/`: 50,003,968 sha `[P]`, 1.35B sha `dcb5ff32…152` `[P]`, 2.25B sha `a0191929…c24` `[V receipt]` | ARC `.../residual_huber_e86_150m_30e_v1` (all 32) | **only 3 of 32 planned states local** (29 missing); ARC copies `[R]` |
| 10M screen checkpoints | 17/18 arms carry a local `checkpoint.pth` (~391 MB) `[V]` | ARC per-run dirs | preserved |
| label registry v2 | `.../label_registry_v2/labels.h5` 116,172,002 B `[V]`; 737,001 quality-pass FeH (GALAH 550,854 / APOGEE 210,990 / Li 291) | ARC same name | preserved, refit-required |
| joint-native labels v2 | `.../joint-native-preparation-v2/joint-native-labels.h5` 119,732,554 B, sidecar sha `3998fe21…85b` `[V]` | ARC same name | preserved; `training_launch_ready: false` (zero-point not verified) |
| enriched feature cohort | `.../enriched_feature_cohort_v1/enriched-feature-cohort.h5` 663,449,752 B, sha `321f2723…120` `[V]` | ARC same name | **production cohort for every 10M run**, preserved |
| distance · age cohorts | `distance_feature_cohort_v1/` (27,809,808 B) · `age_feature_cohort_v1/age-feature-cohort.h5` 23,711,055 B sha `20ee3749…14f` `[V]` | ARC same names | preserved |
| 10M benchmark | **absent locally** | `/arc/projects/k-pop/msa_runs/xp-benchmark-10m-20261012-hdf5-r3/benchmark.h5` (`configs/multilabel-screen-20261020.yaml:7`) | ARC-only `[R]` |
| 150M unique-train cache | **absent locally** | `/arc/projects/k-pop/msa_runs/stellar-improvement-20261015/residual-huber-e86-150m-cache-v1` (~87 GB, `configs/residual-huber-150m.yaml:8`) | ARC-only `[R]`, **required for refits of the long arm** |

Not preserved / lost: 29 of 32 `dd27e718` full states locally (ARC only); parent57 locally; no
aggregate checkpoint for the 3-seed Gaussian ensemble (`configs/multilabel-screen-20261020.yaml:27`
`aggregate_checkpoint: null`). Everything needed to *evaluate* results is local; everything needed to
*reproduce the 150M training* except parent86 is remote.

## 4. Run-ledger status

`experiments/xp/launch-stellar-improvement-20261015.json`: **151 jobs** = 138 `accepted`, 10
`submission_ambiguous` (immutable, never resubmitted), 3 `accepted_recovered` `[V]`. "accepted" includes
retained failures (progress md: forward-inversion, contrastive, density-strict arms) — acceptance is not success.
Task mix: catalog 30, stellar-labels 28, prediction-comparison 26, targeted-prediction 20, multilabel-screen 14,
training 13, others 20.

| run | id | state | latest evidence |
|---|---|---|---|
| `residual_huber_e86_150m_30e_v1` | **dd27e718** | **RUNNING** `[P]` | 2.25 B / 4.5 B presentations, 68,670 updates, 15/30 passes, `checkpoint-002250000000.pth` sha `a0191929…`; receipt 2026-10-22T14:20Z (`stellar-improvement-progress.md:990`, `density-runtime-20261022T1420/large-latest-checkpoint.json`) |
| `multi_gaussian_noise_recon_30e_v1` | **fp0qj7pm** | **COMPLETE** `[V]` | 270 M presentations / 8,250 updates, 16,572 s (4.6 h) runtime; selected 99 M (3,025 updates) sha `f96e5487…`, `all_test_reload_byte_equal: true` (`manifest.json`) |
| density wave NSF NLL/contrastive/NLL+0.1 | lnxaovw5 / x0m40mgn / pu0bbxpb | COMPLETE, producer PASS, **density strict replay failed** `[P]` | `density-wave-20261021-launch-state.json`, `density-collection-20261022/` |

**dd27e718 trajectory** (`[P]`, from progress md receipts): 50 M (21T01:48Z) → 1.35 B/41,202 upd (21T22:52Z) →
1.65 B/50,358 (22T03:58Z) → 1.80 B/54,936 (22T08:05Z) → 1.95 B/59,514 (22T09:09Z) → 2.10 B/64,092 (22T11:08Z)
→ **2.25 B/68,670 (22T14:20Z)**. Observed rate **≈58–62 M presentations/h** (1.30 B in 21.1 h; 0.60 B in 10.4 h).
**Estimated remaining: ≈36–39 h for the remaining 2.25 B → completion ≈2026-10-24 03:00–06:00 UTC**, well inside
the 14-day session deadline (~2026-11-03). Consumed ≈38 h so far → projected total ≈75 h × (8 CPU, 137 GB, 1×H100 MIG 1g.12GB). No new submission/retry is required or permitted.

## 5. Gap analysis (current → human goal), ranked by expected impact

| # | gap | quantified distance | blocks |
|---|---|---|---|
| 1 | **Metal-poor tail cohort is statistically empty** | tail n=**141** (14 below −3) in `native3`, 185 in `scalarFeH`, 29 in enriched `strict_metalpoor`. 141 stars → SE(MAE) ≈0.05 dex and one star moves tail coverage 0.7 pp. No CI can separate methods | "best for oldest/most metal-poor", cross-survey tail calibration |
| 2 | **Alpha absent** | 0 labels, 0 runs, 0 metrics. Needs APOGEE DR17/DR20 alpha/Fe (elemental), GALAH DR4/DR5 `alpha_fe`, DESI, ESO/RVS — and a cross-survey scale audit | 1 of 6 goal labels; no Kiel colouring or elemental-abundance science possible |
| 3 | **Age is a 4,440-star, point-only, sign-biased result** | fit n=4,440; external n=50 with MAE 1.72–2.36 Gyr and **bias +1.68…+2.34 Gyr**; no NLL/CRPS/interval anywhere; native QA only | "best age", "oldest stars", Kiel density |
| 4 | **Distance has no production head; best pilot loses to raw Gaia** | assisted ridge MAE 3.23 mas vs raw Gaia 0.96 mas (N=4,166); extrapolation to 41,403 mas; no head in any 10M/long run | "distance of the furthest stars"; distance-conditioned metallicity/age |
| 5 | **No all-Gaia input store or inference path** | `docs/pipeline.md:55` HATS/leave-one-survey-out unwired; legacy loader needs ~117 GB RAM per 10M rows (1.8 B rows unscalable); XP-off costs +12% global / +31% tail FeH MAE | "all Gaia source_id incl. no XP" |
| 6 | **Distributional best-in-class unprovable: flow density certification blocked** | only Gaussian s42/s44/no-recon densities verified; NSF/BPF/MAF/cINN/noisy-NSF all carry retained strict float32/CPU-vs-GPU replay failures; the best point arm (fp0qj7pm) has `saved density replay failed` → NLL/CRPS `[P]`-only | "best distribution metrics", any NLL ranking claim |
| 7 | **Tail conditional coverage uncalibrated** | FeH CP90 tail 0.1844/0.0000/0.2128; joint Bonferroni box tail 0.0284 (Gaussian s42) vs nominal 0.90 | scientific usability of UQ for metal-poor stars |
| 8 | **Cross-survey label scale/merge unsolved** | GALAH–APOGEE median −0.0045 / scatter 0.048 dex; DESI–APOGEE +0.0024 / 0.0746 vs quoted 0.0102 (66.8% >3σ); no <−2 test pairs for some pairs | merging SDSS DR20/DESI/GALAH/ESO/RVS/LAMOST into one training set |
| 9 | **Compute/latency** | current long run needs ≈75 h for 4.5 B presentations on 1 GPU; a 1.8 B-row all-Gaia inference pass is ≈400× the current 97 k-measurement evaluation | all-Gaia catalogue production |
| 10 | **Observability & preservation** | 29 of 32 long-run states and the 87 GB 150M cache exist only on ARC; benchmark.h5 ARC-only; no collection/QA harness for end-product catalogues | reproducibility of any final refit |

## 6. Reproducibility check `[V]`

- `pixi run pytest tests/ -q` → **570 passed, 4 skipped** (30.77 s), re-run 2026-10-22 ~15:25 UTC. Matches BRIEF.
- Selected long checkpoint **is retrievable locally**: `checkpoint-000099000000.pth` sha `f96e5487…d79`
  matches `manifest.json:checkpoint_selection`; its parent (`checkpoint-86.pth`, sha `d1ed550e…`) is present and
  hash-matching; sidecars `conformal.npz`/`predictions.npz`/`final_reconstruction.npz`/`independent-long-*` present.
- Density scoring regenerable from local arrays: 67 input SHAs + 5 metric-source SHAs, `validation.json` counts,
  `reproduce.py` present.
- **Citation-only / remote `[R]`**: `benchmark.h5`, 87 GB 150M cache, all ARC run outputs (dd27e718 full
  32 states, density-wave outputs), parent57 checkpoint. Producer receipts are preserved in progress md but
  the artefacts were not inspected.
- Two provenance aporias: (i) `joint-native-preparation-v2/manifest.json` says `training_launch_ready: false` /
  `distance_quality_zero_point_verified: false` yet every 10M run used that cohort; (ii) production roles
  (19,299 fit / 97,695 test) are the complete-case subset of h5 roles (20,216 / 100,634) — apply the mask.
- `evaluated_torchregress_commit = 4e81a72` ≠ MSA pin `1d735c6a` (`pyproject.toml`); newer local fixes
  `2db6fa1`, `8973bcd` also unpinned. Density numbers are pinned to the *evaluated* commit.

## What this implies for the plan

1. **Fix the tail cohort before more model search.** 141 stars cannot support "best metal-poor"; ingesting SDSS DR20 /
   DESI / GALAH DR5 / ESO / Gaia-RVS / LAMOST DR12 with an audited scale grows the <−2 tail by orders of magnitude.
2. **Alpha is a data-acquisition + training programme** — zero labels, zero runs. Do it first: only goal label with no evidence.
3. **Freeze fp0qj7pm's selected 99M state as the point baseline; stop chasing 10M arms** (18 exist, all worse globally and in tail).
4. **Certify distribution before claiming it**: either get CUDA-float64 replay + versioned rescoring for one flow arm, or
   scope the deliverable to verified Gaussian density + conformal CP. Never rank NLL from `[P]` rows.
5. **Replace frozen-ridge distance with a jointly trained multi-task head**, judged against raw Gaia (0.96 mas) on the 4,166
   Hipparcos reference — not on StarHorse kpc; the current pilot is a clean negative result (3.23 mas).
6. **Age needs a bigger cohort and a distribution head**: 4,440 fits with a +2 Gyr external bias is plumbing, not science.
7. **Design the all-Gaia path (HATS + batched inference) early**; the in-RAM 140-feature loader is the binding constraint,
   and XP-off costs +12% global / +31% tail FeH MAE today.
8. **Make tail-Mondrian coverage a first-class gate**: global CP90 is 0.89 while tail joint box coverage is 0.03–0.29.
9. **Collect what a refit needs**: 29 of 32 dd27e718 states, the 87 GB 150M cache and parent57 are ARC-only.
10. **Re-pin torchregress before new density work** (`4e81a72` evaluated vs `1d735c6a` pinned; `2db6fa1`/`8973bcd` unpinned).
