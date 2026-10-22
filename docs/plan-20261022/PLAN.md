# MSA end-goal plan — best Gaia stellar parameters for every source

**Prepared:** 2026-10-22 (Lead synthesis of four independent audits:
[01-current-state.md](01-current-state.md) · [02-failures.md](02-failures.md) ·
[03-data-labels.md](03-data-labels.md) · [04-plots-metrics.md](04-plots-metrics.md)).

> **Scope update 2026-10-22 (later same day).** This is the umbrella programme plan.
> The **current round is the training-architecture plan in
> [TRAINING_PLAN.md](TRAINING_PLAN.md)**: labels are being ingested by the human in
> parallel, the ARC quota is confirmed, and the round is deliberately label-light. The
> data/label/DR4/all-Gaia workstreams below remain the umbrella path and are not changed
> by that round.
**Repo:** `/Users/fabbros/src/sfabbro/masked-stellar-autoencoder`, branch
`codex/xp-reconstruction-experiments`, HEAD `0476628`, CI 572 passed / 4 skipped
(re-verified 2026-10-22 15:35 UTC). A parallel session is already executing part of
workstream R0 (see R0.3); this plan adopts and extends it.
**Live run:** `dd27e718` = `residual_huber_e86_150m_30e_v1`, 150M rows × 30 passes
(4.5B presentations), half-budget at 2.25B on 2026-10-22 14:20 UTC, ETA
**2026-10-24 ~03–06 UTC**; do not duplicate or resubmit.
**Related release date:** Gaia DR4 goes live **2026-12-02 12:00 CET** (XP spectra for
1,824,229,813 sources; `ap_rvs` 35,825,564 rows with FLAME ages; `ap_xp` 999,939,280).
This is the single largest external fact in the plan: the "stars without XP" constraint
is a DR3-only constraint, and the DR4 migration is the natural end state.

---

## 0. Executive plan

**Goal (restated as a measurable deliverable).** One catalogue containing every Gaia
`source_id` exactly once, with point estimates and calibrated predictive distributions for
`teff`, `logg`, `fe_h`, `alpha`, `distance`, `age`; demonstrably best (paired, pre-registered
tests) on the bulk **and** on the oldest / most metal-poor stars and on `logg`; with a
quantitatively tight Kiel diagram; and with honest per-source validity/tier flags.

**Six strategic moves.** In dependency order:

| # | Move | Why it is the right move | When |
|---|------|--------------------------|------|
| 1 | **Repair the measurement stack before training anything new.** Float64 scoring + shift-invariant gates, pin `torchregress@8973bcd`, versioned re-score of every completed arm, fix CP/eval/infer code defects, retire the method failures. **No retraining.** | 19 failure entries (F1–F19) are now classified: six are precision/contract fixes (no retraining), four are method failures that must be demoted, and the rest are code-path/definition gaps that directly block distance/age/alpha and all-Gaia inference. Without this, no density/NLL claim is admissible. | Week 0–1 |
| 2 | **Grow the metal-poor tail with labels already on disk, then calibrate it.** DESI DR1 already holds 2,976,271 unique labelled stars incl. **78,419 at FeH<−2 / 3,829 at <−3** and **~2.9M `ALPHAFE` rows** in the checksummed FITS; add LAMOST DR12 (8.38M AFGK LRS + 3.22M MRS), Gaia DR3 GSP-Spec, SDSS DR20 Astra/BOSS-CLAM/MineSweeper, GALAH DR4 leftovers, ESO/4MOST when accessible. | The test tail is **141 stars** (14 below −3) — one star moves tail coverage 0.7 pp; no method can be separated. (The joint-native v2 mask gives 145 and 0; reconcile in L1.1 before any tail claim.) The 60× tail increase needs ingestion + a cross-survey calibration model, not new acquisition. | Week 0–4 |
| 3 | **Replace the negative distance pilots with a joint distance–extinction model.** Train on raw signed Gaia parallax with a declared observation likelihood; add 3D dust, PM and position as *separately ablated* context/prior tiers; validate against independent geometric references and in parallax-space, never against StarHorse kpc. | Existing frozen-encoder "assisted" head is **3.23 mas MAE vs 0.960 mas for raw Gaia** on 4,166 Hipparcos stars — a clean negative result. No production distance head exists. Distance also gates extinction correction, hence far-star FeH/age. | Week 2–7 |
| 4 | **Build the all-Gaia data plane and a no-XP tier.** DR3 input store by HEALPix nside 64 (compact photometry schema ~120–170 GB zstd parquet for 1.8B rows; XP tier 219M); streaming training/inference reader; XP-model distillation for the no-XP tier; per-tier calibration; exactly-once catalogue QA. Design the schema DR4-ready. | The legacy 140-feature loader is in-RAM (~117 GB per 10M rows) and cannot scale. XP-off currently costs **+12% global / +31% tail FeH MAE**; 83,491 no-XP rows (25,113 labelled) already exist to validate a fix. | Week 1–8 |
| 5 | **Train one canonical model family, not another grid.** New parent from the running 150M continuation → joint full-covariance Gaussian over Teff/logg/FeH (+ noise convolution) + raw-parallax distance likelihood + `alpha` head + `age` distribution head + stage-aware `logg`; certified flow only if it beats the Gaussian on joint NLL *without* point regression; 3-seed ensemble for the finalist. | The current best point arm (long noise-aware Gaussian + reconstruction, selected 99M) is 58.83 K / 0.1200 dex / 0.0758 dex but has a **+155 K / +0.13 / +0.39 tail bias** and its density replay failed. Flows improve joint NLL but worsen point MAE and are replay-blocked. We need one coherent head, calibrated, not another 18-arm screen. | Week 3–8 |
| 6 | **Ship the end-product dashboard with gates, then the catalogue.** F01–F14: Kiel density + three tightness definitions, PIT/coverage, distance/age validation, metal-poor/old zooms, catalogue QA, arm comparison; DR3 v1 by **2026-12-18**, DR4 v2 by **2027-03-31**. | No Kiel/HR/CMD figure exists in the repo at all. The metric layer is 80% built (density-review schema + `reproduce.py` + visual-QA pattern); the MVP is ~2 days from saved arrays. | Week 1 + 7–9 |

**Timeline at a glance**

| Phase | Window | Exit deliverable |
|---|---|---|
| **P0 — Repair, certify, freeze baseline** | 22–29 Oct 2026 | Certificates for all arms; `8973bcd` pinned; incumbent frozen; canonical cohort + new locked-test design; tail-count reconciliation |
| **P1 — Labels v3 + tail references** | 22 Oct–14 Nov | DESI α/FeH, LAMOST DR12, Gaia DR3 GSP-Spec, SDSS DR20 Astra ingested + calibrated; new locked tail test frozen; age/distance reference v2 |
| **P2 — All-Gaia data plane v1** | 26 Oct–21 Nov | DR3 input store + streaming reader + no-XP pilot; prototype throughput measured; quota confirmed |
| **P3 — Model programme v3** | 2 Nov–5 Dec | Joint distance+3-label+α/age model, dust/PM/position ablations, finalist + 2nd seed, locked test read once |
| **P4 — DR3 catalogue v1 + release report** | 7–18 Dec | 1.81B-row catalogue with per-tier flags + F01–F14 + certificate/ledger |
| **P5 — DR4 migration + final product** | 2 Dec 2026–31 Mar 2027 | DR4 catalogue (2.0B sources, XP for ~1.82B, 35.8M GSP-Spec/FLAME labels) |

The **critical path** is: repair (P0) → labels v3 calibration + new tail test (P1) → model v3
(P3) → all-Gaia inference (P4). The all-Gaia data plane (P2) runs in parallel and can start
on a 1% sky prototype before the quota is final.

---

## 1. Where we are (condensed; full evidence in 01)

| Label | Current best (verified) | Tail (FeH<−2, n=141) | Blocking gap |
|---|---|---|---|
| Teff | 58.83 K MAE (long noise Gaussian, selected 99M; different budget) | 161.34 K, bias **+155 K** | Tail cohort empty; bias |
| logg | 0.1200 dex | 0.1998 dex, bias +0.13 | Tail cohort; dwarf/giant branches |
| FeH | 0.0758 dex | 0.3931 dex, bias **+0.39**; CP90 coverage **21%** | Tail cohort; cross-survey scale |
| alpha | **none** (0 labels, 0 runs) | — | No label definition/ingestion |
| distance | no production head; best pilot **worse than raw Gaia** | — | Negative pilot; no likelihood model |
| age | 1.02–1.14 Gyr MAE vs StarHorse; external seismic **1.72–2.36 Gyr, bias +1.7…+2.3 Gyr** | — | 4,440 fits; model-conditioned targets |
| Density | only Gaussian s42 verified; flows improve NLL but are replay-failed | FeH CP90 0.000–0.213 | Float64 certificate |
| All-Gaia | no input store, no streaming inference path; XP-off +12%/+31% | — | Data plane + no-XP tier |

Tail counts differ by mask definition: **141 at FeH<−2 and 14 below −3** on the `native3`
complete-case mask, but **145 and 0** on the joint-native v2 mask; L1.1 reconciles this before
any tail number is published.

Details: [01-current-state.md](01-current-state.md) §1–§6; failure register:
[02-failures.md](02-failures.md) §1; label inventory: [03-data-labels.md](03-data-labels.md) §1;
plot gap: [04-plots-metrics.md](04-plots-metrics.md) §1.

**Honest reading of "best".** The 58.83 K arm is *not* a matched-budget winner and its
density/CP audit failed. The matched-budget incumbent for comparisons is the 10M Huber arm
(71.62 K / 0.1438 / 0.0915), with the long arm as a higher-budget challenger. The plan
freezes both and requires the new model to beat the long arm on a **paired, grouped-bootstrap,
locked test** before any "best" claim.

---

## 2. Definition of done — the release scorecard

Frozen **before** the final locked-test read; stored as `scorecard.json` with input hashes.

### 2.1 Product requirements

1. **Coverage:** every Gaia DR3 `source_id` (1,811,709,771) appears exactly once; zero
   duplicates; partition manifest covers the full sky; per-source tier flag
   (`xp_full`, `xp_partial`, `no_xp`, `gaia_only`), quality flags, and model/config/scaler
   SHAs.
2. **Columns:** `source_id`, `ra`, `dec`, `l`, `b`, `G`, `bp_rp`, tier/quality flags,
   `teff_med` [+q16/q84, q02/q98 for distance/age], `logg_*`, `fe_h_*`, `alpha_*`,
   `distance_kpc_*`, `age_gyr_*`, `parallax_mas_*` (consistent 1/d),
   `nll/joint flags`, `model_version`, `inference_version`.
3. **Distributions:** quantiles + covariance / joint level-set information per source where
   the head supports it; documented as *observed-label predictive distributions*, not
   physical-truth posteriors.
4. **Validity flags:** no silent fallback; per-source flags for label domain, extrapolation,
   crowding/RUWE, extinction-map validity, no-XP tier, and zero-point provenance.

### 2.2 Point gates (all must hold on the locked test, paired vs frozen incumbent)

| Label | Primary gate | Secondary gate |
|---|---|---|
| Teff | ΔMAE vs incumbent, grouped-bootstrap 95% CI upper bound < 0; no bias increase > 10 K | tail (FeH<−2) ΔMAE CI upper < 0 |
| logg | ΔMAE CI upper < 0; no branch-ratio distortion (dwarf/giant fractions within 2%) | tail ΔMAE CI upper < 0 |
| FeH | ΔMAE CI upper < 0 | tail ΔMAE CI upper < 0 **and** tail bias reduced by ≥ 0.1 dex |
| alpha | first release; MAE vs ≥ 2 separately calibrated survey layers with CI; no cross-survey offset claim unless the overlap calibration passes | α-poor/α-rich branch separation |
| distance | no regression vs raw Gaia in high-S/N bins; improvement in low-S/N and d>3 kpc bins on independent references; parallax-space bias CI | no catastrophic outliers beyond declared flag |
| age | MAE and signed bias vs seismic/cluster references both improve; coverage reported | age>10 Gyr slice reported with n |

### 2.3 Distribution/calibration gates

- The finalist must carry a **`prediction-certificate.json`** (float64 CPU vs float64 CUDA,
  full roles, shift-invariant gates; schema in [02-failures.md](02-failures.md) §5-P1).
- NLL/CRPS/PIT/rank histogram per label and per pre-registered slice.
- Marginal coverage within **±0.02** of nominal at 0.68/0.90/0.95; **Mondrian** coverage by
  FeH, logg, G, distance-proxy and tier within ±0.05 for strata with n≥200.
- Joint 3-label level-set coverage and joint NLL with Jacobian in native units.
- Metal-poor tail conditional coverage reported with n; **no physical-truth coverage claim**.

### 2.4 Kiel gate

Three definitions, reported separately, never composited (full formulas in
[04-plots-metrics.md](04-plots-metrics.md) §3):

1. **Ridge residual**: cross-fitted `logg(Teff, FeH)` reference ridge; `R_K=σ_K/σ_ref`,
   `R_N=σ_K/σ_noise`; require `R_K<1` with grouped-bootstrap CI excluding 1 **and** no
   logg-MAE regression (anti-shrinkage gate).
2. **Distributional concentration**: Teff-quantile-matched reference Kiel density,
   energy distance, 90% HDR overlap, branch ratio; no collapse onto the ridge.
3. **Independent-logg residual**: APOKASC3 `loggSeis` gold RGB now (15,808 stars local),
   clusters/isochrones later; NMAD per cluster, cluster-level bootstrap.

### 2.5 Evidence gates

- No density/NLL/level-set ranking from producer-only rows; failed rows stay visible.
- Immutable outputs never edited; repairs write `rescored-*`/`certificate-*` siblings with
  `supersedes` provenance.
- Every number has n, units, slice definition, and input SHAs.
- Locked test read once; no test-based model/checkpoint selection.

---

## 3. Non-negotiable principles

1. **Evidence culture** already in the repo: verified vs producer-only vs citation-only;
   preserve failures verbatim; no invented numbers or universals.
2. **No label leakage:** target aliases and astrometry stay out of the point head for
   Teff/logg/FeH unless a declared ablation proves gain without shortcut; distance/age
   context tiers carry an explicit "additional information" label.
3. **No naive label merging:** source-tagged layers, per-survey flags, calibrated offsets,
   correlated repeats never counted as independent stars.
4. **Locked roles:** fit/tune/calibration/**final test**, source-ID based, with survey and
   sky holdouts; the old 97,695 test stays a development cohort, a **new** final test is
   built from fresh labels.
5. **Immutable, versioned outputs**; one owner per output path; no duplicate CANFAR IDs.
6. **DR3 now, DR4 later.** Do not over-engineer DR3-only no-XP coverage; the DR4 schema is
   the end state and the registry is already parameterised by `n_bp`/`n_rp`.
7. **No broad grids.** Every comparison is matched (same parent, rows, masks, optimizer
   budget, tuning protocol); a second seed only for the finalist.
8. **Coordinate with in-flight work:** the committed bounded density precision
   certificate (`65e2243`, `0476628`; job `wetmolvn`, completed 2026-10-22 ~15:37 UTC)
   is adopted and extended — not duplicated; the 8 `trr-tabular-*` sessions belong to
   torchregress and are not touched.

---

## 4. Workstreams

Task IDs: **R** = repair/certify, **L** = labels, **D** = data plane, **M** = model,
**P** = plots/release, **O** = operations. Each task lists the deliverable, main files, and
acceptance gate.

### WS0 — Repair & certify the prediction/scoring stack (owner: Lead; 1 week)

| ID | Task | Deliverable / files | Gate |
|---|---|---|---|
| R0.1 | Re-declare the scoring contract: f64 CPU vs f64 CUDA first; shift-invariant quantities only; dtype-derived `atol`; one batch size recorded | `docs/plan-20261022/SCORING_CONTRACT.md` (or `docs/scoring-contract.md`), `tests/test_scoring_contract.py` | Contract tests pass on synthetic distributions; no old tolerance changed |
| R0.2 | Production float64 scoring module used by all runners; `kind=="huber"` raises instead of NaN density; batch/dtype/device in receipt | `scripts/scoring.py` (extracted from `run_multilabel_screen.py:275-346`), `prediction-certificate.json` | One certificate per family ≥1 checkpoint |
| R0.3 | Extend the committed bounded certificate (`density_fp64_precision_certificate_v1`, session `wetmolvn`, Completed 2026-10-22 ~15:37 UTC; 128 rows × 3 NSF checkpoints at the old pin; output still to be fetched and independently verified; explicitly "no full-role recertification") to **all** families (Gaussian, noisy-Gaussian, NSF, MAF, cINN, BPF, noisy-NSF, long-selected, anchored-NSF), **full roles**, at the new pin | `configs/density-precision-certificate-20261022.yaml` extension, `scripts/run_density_precision_certificate.py` extension | Every family has a certificate; F1/F2 resolved under R0.1 |
| R0.4 | Pin bump `torchregress@8973bcd` (not HEAD; numerically identical, BPF identity tails + honoured capacity), refresh env lock, add provenance test | `pyproject.toml`, `tests/test_library_provenance.py`, `pixi.lock` | CI green; installed commit == pin; BPF tail identity (f(x)=x beyond bound, continuity 1e-12) |
| R0.5 | Versioned re-score of the whole inventory, inference only, into `rescored-v2-f64-8973bcd/`; never overwrite old dirs | 12 short native arms + long selected (99M) + forward + 3 density-wave arms; `reconciliation.json` | Old vs new verdicts explained field-by-field; old FAIL kept |
| R0.6 | Code defects: `conformal.py` finite-sample rank `⌈(n+1)(1−α)⌉` and two-sided level; `eval_ensemble.py` positional label mapping + −9999 sentinel + f64; `infer_msa.py` hardcoded 6 labels, silent feature drop; `multilabel_experiments.py` dtype clamps; density_calibration as arithmetic reference | files + tests | New tests + legacy-config regression <1e-6 in f64 |
| R0.7 | Retire/demote: forward inversion (F5/F6), scalar contrastive (F7), NRE grid MLE (F8); annotate ledger; keep artifacts | `docs/`, ledger note | No product path depends on them |
| R0.8 | Freeze the incumbent: tag `msa-incumbent-20261022`; certificate for long-selected 99M and Huber 10M | git tag, certificate JSONs | Reproducible from hashes |

**Estimate:** 5–7 working days local CPU + one short GPU job (≈0.5 GPU-day). No retraining.
**Exit:** CI green including new gates; certificates exist; re-score reconciliation done;
incumbent frozen.

### WS1 — Labels v3: bigger, calibrated, tail-validated (owner: data/science; 3–4 weeks)

| ID | Task | Details | Gate |
|---|---|---|---|
| L1.1 | Canonical cohort + tail reconciliation | Explain 141 vs 145 test / 14 vs 0 below −3 (different complete-case masks); define `native3_v3` and `tail_v3`; one canonical mask module | Counts reproduced from published HDF5 by two independent paths |
| L1.2 | DESI re-ingest | Re-read checksummed `mwsall-pix-iron.fits`; add `ALPHAFE/ALPHAFE_ERR`; keep `RVS_WARN/CHISQ/SN_R`; publish `desi-labels-v2.h5` | Publisher SHA; flag semantics; units; new versioned file |
| L1.3 | DESI FeH into the training pool | Policy change + calibration layer; source-tagged rows (do not merge values) | Calibration gate L1.7; tail counts |
| L1.4 | LAMOST DR12 | LRS AFGK 8,376,135 rows + MRS 3,224,901; Gaia DR3 crossmatch; flags; LAMOST→native offset layer | Publisher/column check; exact ID match; offset report |
| L1.5 | Gaia DR3 GSP-Spec | `ap_rvs`/`ap_chem` Teff/logg/FeH/α + quality flags; DR4-ready schema | TAP query status; calibration pages; flags |
| L1.6 | SDSS DR20 | Astra summary files (11 pipelines) + BOSS-CLAM α + MineSweeper; **no new APOGEE** | DR20 paper/Astra flags; per-pipeline selection; no averaging |
| L1.7 | Cross-survey calibration model | Per-survey affine offsets `y_s=a_s+b_s y` stratified by Teff/logg/regime; noise inflation from overlaps; hierarchical noisy-label objective; tail gate requiring ≥2 independent refs | Overlap report with n; DESI σ 7× underquoted handled; no global correction claimed below −2 |
| L1.8 | Age anchors v2 | APOKASC3 (15,808), Kepler LEGACY (66), globular/open cluster memberships + literature ages; DR4 FLAME later | Reference IDs held out of fit/tune; model-dependence stated |
| L1.9 | Distance references v2 | Hipparcos nearby negative control; clusters/seismic for distant; StarHorse only as model-dependent training target | No Gaia-as-truth for beyond-Gaia claims; parallax-space scoring |
| L1.10 | Tail validation set | GC members + Li2025 + Pristine weak candidates + DESI high-SN; independent of training labels | ≥2 independent layers agree within stated tolerance before any tail accuracy claim |
| L1.11 | Alpha definition | GALAH `alpha_fe` anchor; APOGEE α/M as separate calibrated label; Gaia RVS α; state clearly they are not the same quantity | Cross-survey α report; block α claims otherwise |

**Deliverable:** `label-registry-v3.h5` + per-source release HDF5s + calibration report +
`cohorts-v3/` split manifest + tail-reference report.
**Exit:** v3 labels ingested/audited; calibration report; new locked test frozen (fit/tune/
cal/test, survey/sky holdouts); DESI α and FeH in; tail test n≥1,000 at FeH<−2.

### WS2 — All-Gaia data plane and inference (owner: data engineering; 4–8 weeks)

| ID | Task | Details | Gate |
|---|---|---|---|
| D2.1 | Prototype input store | 1–2 HEALPix nside-64 tiles (≈36k stars each); schema: compact no-XP (~26 photo + 6 astro + id) and XP (~110+ coeffs); zstd parquet | Measured I/O, RAM, build throughput; schema review |
| D2.2 | Full DR3 build | Gaia `source` 1.81B + XP 219M + external photometry (WISE/2MASS/PS1/SDSS/SkyMapper) + 2D SFD + 3D dust + astrometry; partition manifest with SHA | Counts exactly 1,811,709,771; no duplicates; coverage map |
| D2.3 | No-XP tier | Validate current model on 25,113 labelled no-XP rows; then XP-teacher distillation / no-XP pretraining mixture; separate calibration | No-XP pilot ΔMAE vs XP tier measured and reported |
| D2.4 | Streaming reader + inference path | `pipeline/` reader/model/infer/calibrate: batch iteration over partitions, checkpoint/scaler reload, f64 scoring, per-partition output, resume | Unit tests; 1M-row smoke; no in-RAM full table |
| D2.5 | Production all-Gaia inference | Tiered forward, cross-tier consistency, per-tier calibration, exactly-once output, partition QA | Catalogue QA gates; runtime/throughput record |
| D2.6 | Catalogue QA | sky counts, tier fractions, parameter distributions vs reference, missingness, duplicates, per-healpix sanity | No cell >5σ; 0 duplicates; NaN policy explicit |
| D2.7 | DR4 migration design | DR3↔DR4 crosswalk via `dr3_neighbourhood`; registry abstraction; DR4 AP/XP ingestion plan | Design signed off before 2026-12-02 |

**Storage estimate (must be confirmed, not assumed):** ~1.0 TB dense for 1.8B × 140 f32;
~120–170 GB zstd parquet for the no-XP compact schema; XP tier ~100 GB; output catalogue
~0.3–1 TB; checkpoints ~50 GB. All persistent data on `/arc/projects/k-pop`, never
`/scratch` (session-only, ~200 GiB typical). **Human decision required on quota.**

### WS3 — Model programme (owner: Lead + science; 4–6 weeks)

| ID | Task | Details | Gate |
|---|---|---|---|
| M3.1 | Freeze incumbent + matched controls | long selected 99M, Huber 10M, Gaussian s42, no-recon control; same cohort/roles | Certificate + frozen tag |
| M3.2 | Joint continuation v1 (template) | `configs/joint-continuation-30e.yaml` prepared but never accepted: full-covariance Gaussian over linear-Kelvin Teff/logg/FeH with measurement-error convolution + separate raw-parallax distance point loss; parent86; run on current v2 labels as template after R0 | Matched budget; exact state reload; distance parity vs raw Gaia in high-SN; no distance claim from this run |
| M3.3 | Pretraining v2 (all-Gaia mixture) | From parent86/long; XP+no-XP mixture, masked photometry self-supervision; astrometry/dust as auxiliary *pretraining* targets (not point-head inputs); 1–3 epochs; new parent | Matched comparison vs XP-only pretrain on downstream no-XP + bulk; staged data hash |
| M3.4 | Supervised v3 | Multi-head: quantile point + joint full-cov Gaussian + distance likelihood + α head + age log-Gaussian; source-tagged noisy-label loss; per-survey calibration; stage-aware logg (mixture/classifier) | Pre-registered; v2 vs v3 paired; no test selection |
| M3.5 | Dust/PM/position ablations | Tier 1 context-only (3D dust at predicted distance, 2–3 iterations); Tier 2 joint distance–AV likelihood with map prior; Tier 3 declared 141st input slot; falsifiers from [03](03-data-labels.md) §5 | Gain only if d>3 kpc bins improve on independent refs and calibration does not degrade |
| M3.6 | Alpha head | Native α source definition; source-tagged training; validate on two independent survey layers; no α claim until overlap scale passes | α MAE + branch separation + calibration |
| M3.7 | Age head | `log(age/Gyr)`, positive output, asymmetric posteriors preserved; train on StarHorse + seismic + clusters; evaluate vs held-out seismic/cluster refs; report bias by age/FeH | Bias and MAE improve; no "true age" claim |
| M3.8 | Certified flow finalist | Only after R0.2/R0.3: flow vs full-cov Gaussian on joint NLL; admit only if point MAE does not regress | Certificate; paired NLL CI |
| M3.9 | Ensemble + calibration | 3 seeds of finalist; ensemble mean/quantiles; CQR per slice (Mondrian); tune/final calibration split | Coverage gates §2.3 |
| M3.10 | No-XP student | Distill XP teacher into photometry-only student; train on no-XP rows; validate on labelled no-XP and true no-XP+spectroscopy set | No-XP tier gate; no regression in XP tier |

**Bounded experiment matrix** (anything not listed needs a written decision):

| Axis | Arms |
|---|---|
| Labels | v2 native3 (control) · +DESI FeH calibrated · +LAMOST DR12 · +SDSS DR20 · +Gaia RVS/DR4 (later) |
| Head | quantile (control) · diag-Gaussian (control) · full-cov Gaussian (candidate) · certified flow (challenger) |
| Pre-training | XP-only (parent86/long) · all-Gaia mixture · mixture + distillation |
| Distance context | none (control) · 3D-dust context · joint distance–AV · +PM/position |
| Alpha/age | off (control) · on (candidate) |
| Seed | 42 · 43 (finalist only) |

### WS4 — Metrics, plots and release dashboard (owner: reporting; starts week 1)

| ID | Task | Details | Gate |
|---|---|---|---|
| P4.1 | `src/masked_stellar_autoencoder/reporting/` | `metrics.py` (point/dist/coverage + grouped bootstrap), `slices.py` (16 pre-registered slices), `kiel.py` (3 tightness definitions), `figures.py` | Unit tests; synthetic truth |
| P4.2 | `scripts/make_release_report.py` | Single entry point; reads only versioned CSV/JSON + hashes; writes HTML via tracked template (`augment_catalog.py` pattern); PNG+PDF | <2 min CPU; byte-identical regeneration |
| P4.3 | F01–F14 implementation | From [04](04-plots-metrics.md) §4: Kiel density/tightness, residuals, HR/CMD, age–metallicity, α plane (gated), PIT/coverage, width–precision, distance validation, age vs seismic, tail zoom, catalogue QA, comparison matrix, provenance, ledger | Visual QA JSON per figure |
| P4.4 | MVP dashboard (2 days) | F01/F02 cohort version, F06/F07, F10, F12, partial QA from saved arrays; APOKASC3 local | Visually checked; limits stated |
| P4.5 | Final release report | All-Gaia tier + XP/no-XP panels; certificate/ledger footer; failures visible | Scorecard JSON + gates |

**Kiel-tightness anti-shrinkage rule:** a "tighter" claim requires `R_K<1` (CI excludes 1)
**and** no logg-MAE regression **and** recovered branch ratios; otherwise it is reported as
an artefact. No composite "tightness index".

### WS5 — Operations, resources, governance (owner: Lead; continuous)

| ID | Task | Details |
|---|---|---|
| O5.1 | Run ledger + preservation | Every run: unique ID, source SHA, config hash, library commit, certificate; `dd27e718` all 32 states preserved on ARC and key states local (~13 GB); 87 GB 150M cache preserved; no duplicate IDs |
| O5.2 | CANFAR resources | GPU sessions (MIG 1g.12GB now; consider full H100 / multiple parallel sessions for P2/P3/P4); CPU jobs for ingestion; /arc quota confirmation; scratch staging; monitor via `canfar ps` + `vcp` |
| O5.3 | Change control | Branch policy (push `origin`, never `upstream`); weekly scorecard review; locked-test read approval; tag releases |
| O5.4 | Reproducibility | Pinned env (`pixi.lock`), pinned library, container image, input SHA manifest, `prediction-certificate.json` in every release |

---

## 5. Phases, milestones and gates

### P0 — Repair, certify, freeze baseline (22–29 Oct 2026)

1. R0.1–R0.3 scoring contract + f64 certificates (extend the committed/pending
   `wetmolvn` certificate; do not duplicate the submission).
2. R0.4 pin `8973bcd` + CI; R0.5 versioned re-score; R0.6 code fixes; R0.7 retire.
3. L1.1 canonical cohort + tail reconciliation; L1.2 DESI α re-ingest starts (CPU).
4. O5.1 preserve dd27e718 on completion (~24 Oct); certificate the selected states.
5. P4.4 MVP dashboard from saved arrays.

**Exit gate:** CI green; certificates for every family; `msa-incumbent-20261022` tag;
written cohort definition with reconciled counts; MVP dashboard visually checked.

### P1 — Labels v3 + tail references (22 Oct–14 Nov)

- L1.2–L1.6 ingestion jobs (DESI α, DESI FeH pool, LAMOST DR12, Gaia DR3 GSP-Spec, SDSS DR20
  Astra, GALAH leftovers).
- L1.7 calibration; L1.8 age anchors; L1.9 distance refs; L1.10 tail validation; L1.11 α.
- Freeze **new locked test** (fresh labels, not the reused 97k) and `cohorts-v3/`.

**Exit gate:** label registry v3 + calibration report; α training set exists; tail test
n≥1,000 at FeH<−2 with ≥2 independent references; new locked test hashes recorded.

### P2 — All-Gaia data plane v1 (26 Oct–21 Nov)

- D2.1 prototype (week 1), quota decision, D2.2 full build (weeks 2–3), D2.3 no-XP pilot,
  D2.4 streaming reader, D2.5 inference smoke on 1% sky, D2.6 QA, D2.7 DR4 design.

**Exit gate:** throughput/space measured against the prototype; 1% sky inference matches
cohort predictions; no duplicates; quota confirmed; DR4 design signed off.

### P3 — Model programme v3 (2 Nov–5 Dec)

- M3.1 freeze; M3.2 joint continuation template on current labels; M3.3 pretraining v2;
  M3.4 supervised v3; M3.5 dust/PM/position ablations; M3.6 α; M3.7 age; M3.8 certified
  flow; M3.9 ensemble; M3.10 no-XP student.

**Exit gate:** finalist + 2nd seed selected on tune; certificates; locked test read **once**;
scorecard passes for Teff/logg/FeH (+α if labels pass; distance/age reported with honest
gates); Kiel gate.

### P4 — DR3 catalogue v1 + release (7–18 Dec)

- D2.5 full 1.81B inference, D2.6 QA, P4.3/P4.5 report, release tag, catalogue manifest.

**Exit gate:** exactly-once 1,811,709,771 rows; QA gates; F01–F14; scorecard/certificate/
ledger; catalogue deposited on `/arc/projects` with VOSpace publication.

### P5 — DR4 migration and final product (2 Dec 2026–31 Mar 2027)

- DR4 AP/XP ingestion after release; DR3↔DR4 crosswalk; retrain/re-finetune with 35.8M
  GSP-Spec/FLAME labels and 1.82B XP; DR4 catalogue (2.0B sources); final validation.
- DR4 is a **new product version**, never a silent overwrite.

---

## 6. Compute, data and effort

### 6.1 Compute (assumption-based; measure before committing)

| Item | Estimate | Notes |
|---|---|---|
| Current 150M pretrain | ≈75 h on 1×H100 MIG 1g.12GB (8 CPU/137 GB) | dd27e718, ETA 24 Oct; 60M presentations/h observed |
| Supervised arms (M3.2/M3.4) | ≈5 h/arm on MIG (270M presentations) | 8–12 arms = 40–60 GPU-h; finalist 3 seeds = 15 GPU-h |
| Pretraining v2 | ≈60–100 h on MIG per 1–2 all-Gaia epochs | data staging dominates; consider full H100 or streamed partitions |
| All-Gaia inference | ≈1–2 GPU-h compute for 1.8B, but **I/O bound**; 1–3 days wall with 8–16 parallel workers/sessions | partition per healpix tile; output 0.3–1 TB |
| Label ingestion | CPU only (DESI re-read hours; LAMOST/DR20 days) | 4–16 core jobs |
| Total to DR3 v1 | ≈15–25 MIG-GPU-days over 8 weeks | plus CPU/storage |

### 6.2 Storage

- DR3 input store: ~120–170 GB (no-XP compact) + ~100 GB (XP tier) + manifests.
- Output catalogue: ~0.3–1 TB; checkpoints ~50 GB; existing artefacts 85 GB local/ARC.
- **Hard requirement:** `/arc/projects/k-pop` quota confirmation before D2.2. `/scratch` is
  session-only and must not hold products.

### 6.3 People/roles

- **Science lead (you):** decisions, final test read, claims.
- **Model/lead agent:** P0, P3 implementation and certificates.
- **Data engineer/agent:** P1 ingestion + calibration, P2 data plane.
- **Reporting/agent:** P4 metrics/plots/dashboard.
- **Parallel workstreams:** torchregress (library fixes) and the in-flight density
  certificate; coordinate, don't duplicate.

---

## 7. Risk register

| # | Risk | Trigger | Mitigation |
|---|---|---|---|
| 1 | Metal-poor labels prove uncalibratable (no cross-survey pairs below −2) | overlap report still has 0 tail pairs after L1.2–L1.6 | Build independent tail references first (GCs, Li2025, DESI high-SN); report tail without a calibrated absolute scale until ≥2 refs agree; do not average surveys |
| 2 | Distance model still loses to raw Gaia | M3.2/M3.5 high-SN bins regress | Ship feature-only distance as a *prior/assist*; declare high-SN fallback to raw Gaia; claim gains only in low-SN/distant bins vs independent refs |
| 3 | Age bias persists | seismic/GC bias not reduced | Report age as model-conditioned; gate age claims on independent refs; add DR4 FLAME later; do not advertise "true ages" |
| 4 | All-Gaia quota/IO infeasible | prototype throughput < 1% sky/day or quota < 1 TB | Reduce to tiered release (XP tier first, all-sky later); compress; partition parallelism; negotiate quota early |
| 5 | Density certification still fails on GPU | f64 CUDA vs CPU delta > gate | Ship verified Gaussian density only; flow as producer-only appendix; do not rank NLL |
| 6 | DR4 schema/source-ID mapping surprises | Dec 2 release differs from announced schema | Build crosswalk design now; DR3 product stands alone; DR4 is a new version |
| 7 | Locked test contaminated by repeated use | model selection creeps onto test | New fresh-label test; certificates; one read; audit who read what |
| 8 | Parallel sessions collide in repo/CANFAR | duplicate job IDs, overwritten outputs | Run ledger + unique output paths; `git status` check; coordinate with in-flight certificate work |
| 9 | Label systematics masquerade as model gain | e.g. DESI σ underquoted by 7× | Hierarchical noisy-label loss; per-survey calibration; report offsets; no naive merge |
| 10 | Schedule slips into DR4 window | P4 > 18 Dec | Deliver DR3 v1 as rehearsal; freeze scope; DR4 v2 is the scientific end state |

---

## 8. Decisions needed from you (now)

1. **ARC quota**: approve ~1 TB feature store + ~0.3–1 TB output on `/arc/projects/k-pop`
   (or name the allocation). Without it D2.2 cannot start.
2. **DESI policy change**: approve moving the 2.98M DESI FeH and 2.9M DESI α into the
   training pool with source-tagged calibration. This is the largest single scientific jump
   available and it changes every downstream number.
3. **DR3 vs DR4**: confirm DR3 product v1 by 18 Dec + DR4 product v2 by 31 Mar 2027 (recommended),
   or wait for DR4 (simpler data path, but no product until 2027).
4. **ESO/4MOST access and proprietary status**; whether to spend effort before ESO is back online.
5. **GPU allocation**: keep MIG slices, or request a full H100 / additional parallel sessions
   for P2–P4 (recommended for the pretrain v2 + all-Gaia inference phase).
6. **Locked test definition**: approve building a fresh final test from labels v3 rather than
   reusing the 97,695 development cohort.
7. **Accept per-source validity flags** for the ~88% of DR3 sources with no XP and no external
   labels, rather than a single accuracy number for all Gaia.

---

## 9. Immediate next 10 working days (ordered)

1. **Approve decisions 1–7** (or answer the subset you can); freeze the plan version.
2. **R0.1–R0.3**: write scoring contract; extend the in-flight certificate to all families,
   full roles, at the new pin; extend `tests/`.
3. **R0.4**: bump `torchregress@8973bcd`, refresh lock, full CI, provenance test.
4. **R0.5–R0.6**: versioned re-score; fix `conformal.py`, `eval_ensemble.py`,
   `infer_msa.py`; add the missing regression tests.
5. **R0.7–R0.8**: demote forward inversion/scalar contrastive; freeze incumbent tag;
   certificate the long-selected 99M state.
6. **L1.1–L1.2**: canonical cohort module + tail-count reconciliation; DESI α/FeH re-ingest
   CPU job (checksum-verified, new versioned output).
7. **L1.3–L1.6**: queue LAMOST DR12, Gaia DR3 GSP-Spec, SDSS DR20 Astra acquisition jobs;
   write the per-survey audit gates (checksum, ID match, units, flags).
8. **D2.1**: build the 1–2-tile HEALPix prototype; measure throughput/RAM/disk; bring the
   quota number to the decision.
9. **P4.4**: ship the MVP dashboard (Kiel cohort version, PIT/coverage, tail, comparison
   matrix) from the saved arrays.
10. **O5.1**: on completion of `dd27e718` (≈24 Oct), preserve all 32 states + receipts;
    archive the 150M cache and benchmark references.

---

## Appendix A — evidence index

| What | Where |
|---|---|
| Current metrics/assets/gaps | [01-current-state.md](01-current-state.md) |
| Failure register + repair wave | [02-failures.md](02-failures.md) |
| Label inventory + survey gaps + all-Gaia scale | [03-data-labels.md](03-data-labels.md) |
| Metric definitions, Kiel definitions, F01–F14 | [04-plots-metrics.md](04-plots-metrics.md) |
| Verified point/density/coverage tables | `experiments/xp/results-stellar-improvement/density-review-20261022/{point,distribution,interval,capability}.csv`, `metrics.json`, `validation.json` |
| Accepted-job ledger (151 jobs) | `experiments/xp/launch-stellar-improvement-20261015.json` |
| Run progress / failure history | `experiments/xp/stellar-improvement-progress.md` |
| Main method docs | `docs/pipeline.md`, `docs/METHODOLOGY.md`, `docs/stellar-prediction-final-design-20261017.md`, `docs/stellar-improvement-programme.md`, `docs/multilabel-programme-20261020.md`, `docs/multilabel-long-20261020.md`, `docs/residual-huber-150m-20261020.md`, `docs/joint-continuation-30e.md` |
| Label sources pin list | `configs/stellar-catalog-sources.yaml` |
| Cohort/preparation configs | `configs/joint-native-preparation.yaml`, `configs/multilabel-long-20261020.yaml`, `configs/residual-huber-150m.yaml` |
| In-flight certificate work (parallel session, committed) | `configs/density-precision-certificate-20261022.yaml`, `configs/density-precision-certificate-launch-20261022.yaml`, `scripts/run_density_precision_certificate.py`, `tests/test_density_precision_certificate.py`, `experiments/xp/monitor/independent_nsf_kernel.py`; commits `65e2243`/`0476628`; job `wetmolvn` |

## Appendix B — release decisions log (to be filled as we go)

| Date | Decision | Basis | Owner |
|---|---|---|---|
| 2026-10-22 | Plan drafted; repair wave P0 adopted; DR3 v1 + DR4 v2 sequence recommended | this document + four audits | Lead |
| 2026-10-__ | Quota confirmed | D2.1 measurements | you |
| 2026-10-__ | DESI pool policy approved | L1.7 calibration report | you |
| 2026-11-__ | v3 locked test frozen | L1.1–L1.10 gates | you |
| 2026-12-__ | DR3 v1 release approved | scorecard + certificates | you |
