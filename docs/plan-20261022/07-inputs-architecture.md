# 07 — Inputs and encoder/head architecture (agent `inputs-arch`, 2026-10-22)

Scope: the **best training buildable now** with data already on disk. Labels are out of
scope (parallel ingestion). No code/config/ledger edited; only this file written.
`[V]` verified this session by reading the local file / recomputing the count ·
`[P]` producer-only (manifest, receipt, or another agent's report) · `[N]` needs
verification before use · `[P+]` proposed design (no physical prior invented).

## 1. Verified input inventory

### 1a. The 140 slots (`configs/pretrain.canfar.yaml:6-30,31-56`, `pipeline/registry.py:9-50`,
checkpoint `multi_huber_recon_s42_v1/checkpoint.pth:feature_columns` `[V]`)

| group | slots | idx | observed in 756,836-row cohort `[V]` | error column `[V]` | sky |
|---|---|---|---|---|---|
| Gaia G,BP,RP | 3 | 2-4 | 673,345 (89.0%) | **null** (raw flux errors, not mags) | all-sky |
| XP bp_1-55, rp_1-55 | 110 | 5-114 | 673,345 (89.0%), all-or-nothing | 110 (`bpe_*`/`rpe_*`) | all-sky |
| WISE W1,W2 (CatWISE) | 2 | 0-1 | 654,211 (86.4%) | **null** | all-sky |
| 2MASS J,H,KS | 3 | 131-133 | 671,963 (88.8%) | 3 | all-sky |
| SkyMapper U..Z | 6 | 115-120 | 591,671 any / 556,494 all | 6 | south |
| PS1 G,R,I,Z,Y | 5 | 126-130 | 221,559 (29.3%) | 5 | dec ≳ −30° |
| SDSS U,G,R,I,Z | 5 | 121-125 | 77,323 (10.2%) | 5 | ~1/3 sky |
| **PARALLAX, EBV, pmra, pmdec, RA, DEC** | 6 | 134-139 | **673,345 — exactly the XP rows** | PARALLAX+pm only | all-sky |
| recon_cols (decoder width) | **136** | — | 134 photometry+XP + PARALLAX + EBV | — | — |

- **Error channels are 132/140**: `error_observed` never fires for W1,W2,G,BP,RP,EBV,RA,DEC `[V]`.
  XP errors are stored **in raw archive units** while coefficients are G-normalized; divide by
  `10**((8.5−G)/2.5)` before any covariance/uncertainty loss (`docs/xp-benchmark-usage.md:29-31`);
  `input_measurement.standardized_xp_errors` already implements both conventions explicitly.
- **The current "physical core" is 134 slots**: `spectrophotometric_inputs(include_reddening=False)`
  sends PARALLAX/EBV/pm/RA/DEC to the −9999 sentinel in **both** supervised and reconstruction calls
  (`training/spectrophotometric_inputs.py:59-71`; policy `docs/stellar-prediction-final-design-20261017.md:29,33-47`;
  `docs/multilabel-long-20261020.md:19`, `configs/joint-continuation-30e.yaml:35`). `run_joint_continuation.py:81-84`
  **raises** if `include_positions`/`include_proper_motions` are set — the extension is not a config switch.
- Missingness is already encoded: `MaskedPeriodicEmbeddings` carries a frozen
  `missing_periodic_encoding` buffer (140×96) so −9999 is a learned token, not a median `[V]`.

### 1b. What the six "withheld" slots actually contain `[V]`

| slot | measurement | caveat that gates use |
|---|---|---|
| PARALLAX | median 0.928 mas, min −7.77, max 273.8; **6,184 ≤ 0 (0.92%)** | zero-point not applied (`configs/distance-gaia-baseline-verification.json:zero_point_correction_applied=false`); formal ≠ total calibrated |
| e_parallax | median 0.0158 mas, p1 0.0093 / p99 0.207 | **verify mas vs µas** against one raw Gaia row before the likelihood `[N]` |
| pmra / pmdec (+e_*) | median −1.77 / −2.26 mas/yr; err median 0.0162 / 0.0144 | no epoch propagation in the 140 table |
| RA / DEC | full sky; 21.2% \|dec\|<20° | no error column; l/b and HEALPix are derived bookkeeping |
| EBV (2D SFD) | median 0.112, **max 132.7899, 2.56% > 2** | not a stellar E(B−V); unit/map/uncertainty provenance **fails audit** (`audit_joint_native_readiness.py`; `docs/joint-continuation-30e.md:57-62`) |
| `distance_fit_eligible` | **0 / 756,836** | hard-coded `np.zeros(...,bool)` (`scripts/materialize_joint_native_labels.py:222`): policy flag, not a data finding |

### 1c. The no-XP population is **label-only**, not photometry-only `[V]` — corrects BRIEF-v2:62

83,491 `input_source==3` rows (= "ID absent from both benchmark and legacy FITS",
`scripts/stellar_enriched_features.py:155,333`) have **0/140 features observed, 0 astrometry,
and 0 membership in fit/tune/cal/test**. 64,994 carry native labels (63,930 all-3 quality-pass),
11,070 PIGS, 7,457 Li2025. So: *the XP-off training pool does not exist on disk yet*; it needs a
feature re-join (Gaia DR3 TAP for G/BP/RP/ϖ/pm/RUWE on 83k IDs — GB-scale query, not a new survey).
All XP-off evidence in the repo is therefore **synthetic masking**, not genuine no-XP stars `[V]`.

### 1d. Usable today vs needs acquisition

| input | status | note |
|---|---|---|
| 134 photometry+XP, 140-slot matrix | **usable now** | `paired-cohort.npz:input` (756,836×140 f32) + `visible` |
| 6 withheld value slots, raw | **usable now** (local h5 rebuild) | `features/values`; npz has scaled only |
| e_parallax/e_pmra/e_pmdec | **usable now** (local h5 rebuild) | not in `paired-cohort.npz`; in `reported_feature_errors` |
| XP covariance (1,485 packed per band) | **usable on ARC only** | 10M `benchmark.h5`: `xp_correlations`, `xp_band_metadata`, `xp_covariance_available` (`docs/xp-benchmark-dataset.md:13`) |
| RUWE / neighbours / astrometric_params_solved | **eval-only, 99,525 rows** | `distance_feature_cohort_v1/hipparcos_reference/{gaia_ruwe,number_of_gaia_neighbors,gaia_astrometric_params_solved}`; **absent from the 140 slots** |
| parallax zero point (per-star/patch) | **absent** | needs. Gaia DR3 ZP recipe or a declared constant |
| 3D dust (Bayestar/Edenhofer/Green) | **absent** | only `dustmaps>=1.0` + `SFDQuery` pinned (`pyproject.toml:18`, `data/pretraining-partial-table-maker.py:168`) |
| per-star A_V + uncertainty + validity | **absent** | Tier 3 blocker |
| alpha labels | **0 rows** | head designed, disabled |

## 2. Tiered input policy

Rule enforced throughout: **a quantity is either an input, a likelihood term, or a target — never two.**
Parallax appears in exactly one role per arm (§4a is the only place this rule is deliberately broken,
and it is then a declared measurement-assisted arm).

| Tier | exact features | normalization | missing handling | leakage / shortcut risk | falsifier |
|---|---|---|---|---|---|
| **T0** feature-only | the 134 core slots as today | parent86 frozen RobustScaler; XP G-normalized (`pogson_div_v1`) then per-order median/IQR + asinh (`docs/pipeline.md:25-33`) | structural mask → −9999 token | survey-selection bias, not leakage | XP-off cost must not grow (today +11.9% global / +31% tail FeH MAE `[V]`) |
| **T1** context for distance/age heads only (never Teff/logg/FeH/α) | ϖ[mas], e_ϖ, zp, pmra, pmdec, e_pmra, e_pmdec, l, b (from RA/DEC), HEALPix nside=64 index, SFD E(B−V), + optional RUWE/neighbour bits (eval-only) → **14 continuous + 8 mask/quality bits** | per-feature median/IQR **fit on fit-role only**; ϖ and pms as `ϖ/max(e_ϖ,ε)` clipped ±10 is the alternative (METHODOLOGY.md:11-14) | mask bits concatenated (never median-fill); absent ruwe → bit 0 | (i) ϖ and pm only for stars that also have labels → spatial/survey selection; (ii) SFD max 132.79 ⇒ garbage rows; (iii) context concatenated to latent can be *inverted* by the head | shuffle T1 features **within sky patches (nside=8)** and retrain: any retained gain is architecture, not information. Hipparcos nearby control **must not** improve |
| **T2** measurement-assisted distance/age | T0 + T1 **minus ϖ, e_ϖ, zp** + raw signed ϖ likelihood | — | non-positive ϖ retained, never inverted | double counting if ϖ is also in T1 | profile-likelihood vs unchanged-measurement fallback must be reported per ϖ/σ bin; fallback must be *exact* for SNR>20 |
| **T3** joint distance–A_V | T2 + named 3D map queried at the **predicted** distance, iterated 2-3×, with published map σ and per-star validity | — | map-miss → masked | circularity if the map was built from these stars' photometry | the five falsifiers in `03-data-labels.md:151-155`; negative control = Hipparcos must not improve; gain must be confined to d>3 kpc, \|b\|<15° |

**Buildable now**: T0, T1, T2 (all features present locally; likelihood code exists:
`signed_parallax_nll` in `training/parallax_likelihood.py:24`, `signed_log_distance_nll` in
`training/joint_prediction.py:242`). T1's E(B−V) component only **after** the provenance gate (§6 E2).
**Waits for 3D dust**: T3. **Waits for labels**: α in every tier.

## 3. Encoder / backbone

| decision | verified basis | choice `[P+]` |
|---|---|---|
| backbone | parent86 = **34,155,392 params / 136.6 MB fp32**; with AdamW = the 409,873,692 B checkpoint `[V]`. RTDL TabResnet: 140→PE(96→16)→2240→2048,2048,1024,1024,512,512,256,**256** latent | keep. Do **not** replace the backbone this round |
| 10M-screen comparability | 17/18 arms already load parent86 and vary only head+loss `[V]` | **freeze the data/encoder identity**: all new arms start from parent86 with fresh matched Adam, 10M presentations, recon weight **0.01**, encoder lr 1.25e-6 / head 1e-3 / 500 warmup. One knob per arm |
| continue 150M state | `dd27e718` at 2.25B/4.5B presentations, 15/30 passes, `checkpoint-002250000000.pth` `[P]` | **separate, non-matched arm only** (E6). Its encoder has seen 10× more rows ⇒ cannot be the control |
| frozen vs fine-tuned | existing frozen-head probes (`frozen-head-screen.yaml`, `prediction_vir.py`) | **frozen for E1/E2/E8** (latent-only, CPU-cheap, no reconstruction); **fine-tuned for E3/E4/E6** |
| context injection point | input-width change breaks `dense_resnet.0.linear.weight (140,96,16)` + `dense_resnet.2.weight (2240,2048)` = **4.80M params (14.1%)** `[V]` | concatenate context **after** the latent (256→384), so the checkpoint loads unmodified |
| mask indicators | already present in-backbone (`missing_periodic_encoding`, 140×96) `[V]` | **not needed** for the 140 slots; **required** for each new context channel |
| per-band error channels (140→272) | 132 error columns exist `[V]`; prior art `configs/pretrain.xp-covariance.yaml` + `docs/xp-covariance-experiments.md` | **defer** to E7, only if E5 shows error weighting matters. Cost: +4.3M first-layer params, full checkpoint break |
| covariance-aware tokenization | `xp_correlations` 1,485/band available on ARC `[V]`; `V=diag(σ)R diag(σ)+0.1I` per band, `MultivariateGaussianLoss` wired `[P]` | **re-run with corrected error units** (E5). 106.9 GB scratch for 9M `[P]`. No architecture change: it is a reconstruction loss change |
| width/depth, memory | 34.2M params; production-cohort 756k×140 f32 = 424 MB; 9M×140 = 5.04 GB `[V]`-arithmetic | keep width/depth. Latent 256 is the head bottleneck, not the encoder |

## 4. Head architecture

All proposed heads **attach to the frozen/shared 256-d latent**, take `[latent ‖ context(128)] = 384`
where context is allowed, and are `<1%` of backbone size. Current 3-label head for reference:
`MultiLabelHead` = **42,755 params** (256→128 residual→3/6/16 outputs) `[V]`.

| module | inputs | outputs | loss | params `[P+]` | coexists with 3-label head |
|---|---|---|---|---|---|
| **C** context conditioner | 14 cont + 8 bits = 22 | 128 | none (trained through heads) | 22·64+64 + 64·64+64 + 64·128+128 = **13,696** | shared by B/D/E; T0 heads ignore it |
| **A** 3-label core (exists) | 256 | μ, full 3×3 `scale_tril` | `masked_gaussian_measurement_nll` (exact observed-subvector, `target_dim≤8`, `joint_prediction.py:60-61,126`) `[V]` | 45,116 | **this is the anchor**; every other head is additive |
| **B** distance point | `[z‖c]`384 | `s=log(d/kpc)`, `log σ_s` | `signed_log_distance_nll` (E3/`[V]` math) or convolved-parallax NLL | 384·128+128 + 16,512 + 258 = **66,050** | additive; separate optimizer group |
| **D** age log-age | 256 (context optional, declared) | μ, log σ on `log10(age/Gyr)` | heteroscedastic Gaussian on **log** age; σ_target from StarHorse p16/p84, not invented | **49,794** | additive, `age_enabled` gate |
| **E** alpha | 256 | μ, log σ on [α/Fe] | NLL with **source-tagged affine scale** `y_s=a_s+b_s y` per survey×regime | **49,794** | designed, **disabled** until ≥1 audited α source |
| **F** joint multi-target | 256 | 3×3 core + separate s + separate age; **block-structured** | sum of A+B+D terms, each with its own observed-count denominator | ≈ **1.6×10^5** | superset; only where co-occurrence is identified |
| **G** stage-aware | 256 | `K=4` stage logits (from **measured** logg×Teff bins) + per-stage 3×3 | A + stage cross-entropy, or a K-mixture over A | gate 256·64+64+64·4+4 = 16,708 + 4×3,612 = **31,156** | additive auxiliary; never replaces A |

Design notes (each is a guard against a known failure):
1. **B must not be a Gaussian on `d`.** `s=log(d/kpc)` matches the orientation of the error metric
   (fractional distance), keeps `d>0` automatic, and is the coordinate the likelihood expects
   (`exp(−s)` mas). B's feature-only predictive distribution is population-dependent — publish it
   as a predictive PDF, and T2 as a **confidence set via profile likelihood, not a posterior**.
2. **Convolved-parallax NLL is the primary distance objective; delta-method is only a control.**
   `p(ϖ_obs|z)=∫N(ϖ|exp(−s)+zp,σ_ϖ²)p(s|z)ds` has no closed form for Gaussian `p(s|z)`; use a fixed
   1-D quadrature in `s` and require agreement with the first-order approximation in the
   narrow-error limit before launch (design-doc gate, `docs/stellar-prediction-final-design-20261017.md:75`).
3. **F is block-structured, not a 6×6 Gaussian.** Today's co-occurrence makes the full 6×6
   unidentifiable: α has 0 rows; age complete-case is 4,440 of 19,299 fit; distance roles are disjoint
   from the 3-label roles. Only the 3×3 stellar core is identified (19,299 complete-case fit rows).
   Publish the pairwise co-occurrence matrix and add a cross-block only at n ≥ 1,000 co-occurring stars.
4. **G exists because a single Gaussian cannot represent the RGB/dwarf bifurcation** and because the
   metal-poor tail is *confidently wrong* (FeH CP90 tail coverage 0.0000–0.2128 vs nominal 0.90 `[V]`).
   Use measured-logg stages (a measurement, not a prior). Cheaper alternative to a full mixture.
5. **Width must be difficulty-linked**: predict `log σ` and require a positive slope of width vs
   |residual| decile and vs G/e_ϖ/EBV (`04-plots-metrics.md` F07). A saturating σ means "uninformative".

## 5. No-XP / all-Gaia readiness (design only)

| smallest experiment | what it measures | cost | blocker |
|---|---|---|---|
| N1 input-masking mixture `mask_mixture_xp_full_frac: 0.0 → 0.20` | does mixed-XP-drop pretraining shrink the XP-off penalty? | knob already implemented (`METHODOLOGY.md:93-95`), one 10M arm | none |
| N2 synthetic XP-off supervised pilot | already measured: FeH MAE 0.08231→0.09208, tail 0.44884→0.58814, <−3 (n=14) 1.914 `[V]` | **done**; reuse as the baseline | none |
| N3 distillation XP-teacher→XP-masked student | fraction of the +11.9% recoverable | one 10M arm, MSE on standardized outputs + own-label target | **gate on tail, not global**: the teacher's own tail bias is +0.39 dex; naive distillation propagates it |
| N4 genuine no-XP cohort | the only honest XP-off validation | Gaia DR3 TAP query for 83,491 IDs | needs the feature re-join (§1c); **not** a new survey |

The 1.8B path is noted, not planned: 140 f32 = 560 B/row ⇒ ~1.0 TB dense; the legacy loader caches
~117 GB per 10M rows `[P]` and cannot be used; HEALPix nside=64 partitions are the schema
(`StellarBatch.healpix` is already first-class, `pipeline/batch.py:27`). Gaia DR4 (2 Dec 2026) publishes
XP for 1.82B sources and resets the whole constraint.

## 6. Pretraining / mixture for this round

| question | decision `[P+]` | matched comparison |
|---|---|---|
| use the running 150M job? | **Do not touch it.** Take the 24 Oct final state as backbone candidate only | E6: same heads/inputs, parent86 vs 150M state, 10M presentations |
| continue pretraining with mixed XP/no-XP? | yes, but **synthetic** mixture (N1); genuine no-XP rows carry no features | parent86, 9M rows, recon-only, 0.0 vs 0.2 |
| astrometry as masked reconstruction target? | **yes, cheapest representation experiment**: score existing decoder channel 134 (PARALLAX). No architecture change — 136 outputs already exist; today the channel is simply excluded | recon scored-features ∈ {134, 135} at identical budget |
| `distance_fit_eligible` role exclusion | **required**: mask channel-134 recon targets for stars in the distance fit/tune/test roles, else the encoder has reconstructed the very ϖ used to judge B and "feature-only" is false | same arm with/without role exclusion |
| distinguish "pretraining with extra information" from "head input leakage" | *pretraining*: ϖ is a self-supervised target on unlabelled stars → representation learning, legitimate, but must be role-excluded and disclosed. *Head input*: ϖ in T1 while the same ϖ is a T2 likelihood term → double counting. *Label*: ϖ as a regression target = the product itself | leak detectors: (i) shuffle ϖ channel, (ii) T2 with ϖ deliberately corrupted ×1.01 — a head that tracks the corruption is leaking |

## 7. What to run first (8 experiments; campaign sizes them)

| # | experiment | info gain | cost | depends on |
|---|---|---|---|---|
| **E1** | T1 context + head B on **frozen latents**, two arms: 1a ϖ∈context/no-likelihood, 1b ϖ∉context/parallax-likelihood. Eval 22,932 identity + 4,166 Hipparcos vs **raw-Gaia 0.960 mas** | **highest**: the previous frozen-ridge pilot was 3.23 mas — worse than doing nothing. This decides whether astrometry/dust help at all | CPU-feasible (latent extraction over 756,836 rows once) | none |
| **E2** | E(B−V) provenance gate + within-patch-shuffle falsifier, context-only | decides whether the only dust channel available this round is admissible | low | none |
| **E8** | G stage-aware + difficulty-scaled width on the selected 3-label arm (frozen encoder) | attacks tail coverage (0.00–0.21 vs 0.90) at head level, no retraining | low | none |
| **E3** | recon scored-features +PARALLAX (134→135) w/ role exclusion, 10M arm | does distance representation improve without leakage? | medium | E1 result informs |
| **E4** | N1 mixed-XP pretraining (0→0.2), 10M arm | shrinks the XP-off penalty that blocks the all-Gaia product | medium | none |
| **E5** | covariance-aware XP recon **re-run with corrected units** (diagonal, then full-cov) | only route to improved faint-end Teff/logg and honest σ; prior results invalidated by the unit bug | high (107 GB scratch, 20M presentations) | none |
| **E6** | parent86 vs 150M-state backbone at fixed heads | is 4.5B presentations worth more than 9M×100? | medium | after 24 Oct |
| **E7** | per-band error channels (140→272) | only if E5 proves error weighting matters | high, checkpoint break | **E5 win required** |

Deferred with reason: T3 joint distance–A_V (no 3D map), alpha head (0 labels), genuine no-XP cohort
(feature re-join), 1.8B production (not this round). Every checkbox in
`docs/reports/msa-density-failures-and-missing-experiments-20261022.md` is either covered by E1-E8 or
is a loss/metrics item owned by the campaign agent — flag for the Lead's cross-check.

## What the Lead must put in the plan

1. **State the corrected no-XP fact**: the 83,491 no-XP rows are label-only with zero features and zero
   astrometry and are in no role split (§1c). BRIEF-v2's "25,113 labelled no-XP rows" is not
   reproducible from the local sidecar. Any no-XP claim this round must say "synthetic XP masking".
2. **Adopt the 134-slot T0 core + post-latent context concat as the only checkpoint-compatible
   extension path**, and forbid input-width changes in matched arms (4.80M params + checkpoint break).
3. **Adopt the one-role rule**: ϖ/E(B−V)/pm are input **or** likelihood **or** target, never two; and
   name E1a's deliberate violation as measurement-assisted.
4. **Fund E1+E2+E8 first** (frozen encoder, CPU-cheap, design-deciding) before any new 10M GPU arm.
5. **Require the role-aware parallax-recon exclusion** in E3, and the shuffle/corruption leak detectors
   in §6, as launch gates — not as post-hoc checks.
6. **Require `gaia_quality_and_zero_point_verified` (+ `distance_input_no_copy_verified`,
   `correlated_teacher_policy_verified`) before any distance NLL**, and
   `reddening_provenance_verified` before EBV is scored anywhere (`run_joint_continuation.py:96-115`).
7. **Declare the joint head block-structured** (3×3 core now; s, age, α as separate terms) and publish
   the pairwise label co-occurrence matrix; a 6×6 covariance is unidentifiable today.
8. **Do not spend GPU on E7/E6 before E5/E1 report**; both are conditional, not scheduled.
9. **Book the Gaia DR3 re-join of 83,491 label-only IDs** as the enabler for real XP-off validation.
