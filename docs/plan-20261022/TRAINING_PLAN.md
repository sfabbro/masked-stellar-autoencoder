# MSA training-architecture plan (round 2)

**Prepared:** 2026-10-22 (Lead synthesis of three design tracks:
[05-heads-losses-metrics.md](05-heads-losses-metrics.md) ·
[06-experiment-campaign.md](06-experiment-campaign.md) ·
[07-inputs-architecture.md](07-inputs-architecture.md)).
**Scope for this round:** design and select the best **training architecture** — inputs,
encoder/context branches, prediction heads, losses, calibration and metrics — using the data
and checkpoints already available. Labels are being ingested by the human in parallel; this
round is deliberately **label-light** and must not block on them. No all-Gaia production build,
no DR4 migration, no new survey acquisition in this round (the umbrella plan
[PLAN.md](PLAN.md) still owns those).

**Working assets:** HEAD `0476628`, CI 572 passed / 4 skipped. Parent86 is local and
hash-verified (`d1ed550e…`, 409,873,692 B). The big-training run `dd27e718` is still Running at
~2.3B/4.5B presentations (ETA 24 Oct 03–06 UTC); its latest local full state is
`checkpoint-002250000000.pth` (`a0191929…`), final state expected 24 Oct. The bounded FP64
certificate job `wetmolvn` Completed 22 Oct ~15:37 UTC; its outputs are not yet fetched or
independently verified. `/arc/projects/k-pop` quota is confirmed sufficient for this round.
The failure/missing-experiment inventory is
[msa-density-failures-and-missing-experiments-20261022.md](../reports/msa-density-failures-and-missing-experiments-20261022.md);
every checkbox there is mapped to a run in [06](06-experiment-campaign.md) §6 (24 done, 1
retired with reason, 0 silently dropped).

---

## 0. Definition of success for this round

A **selected, reproducible architecture** with a verified metric contract, not a new broad
grid:

1. **W0 certificate gate closed.** Every head family that enters a ranking has a float64
   CPU-vs-CUDA certificate over full roles at `torchregress@8973bcd`; all completed arms are
   versioned re-scored; old failures stay visible; no density/NLL ranking is quoted from
   producer-only rows.
2. **One head/loss family wins on evidence.** A matched 10M screen (same parent, rows, masks,
   optimizer budget, seed) selects among quantile/expectile, diagonal Gaussian, full-covariance
   Gaussian, low-rank Gaussian, MDN, and certified flows; the winner has paired, grouped-bootstrap
   deltas vs the diagonal-Gaussian control, and a verified distribution score (NLL/CRPS/energy/
   PIT/coverage), not just point MAE.
3. **Distance head is statistically correct and honest.** A latent `s=log(d/kpc)` head trained
   through the **raw signed parallax likelihood**, with a feature-only predictive PDF and a
   measurement-assisted profile-likelihood confidence set as **separate named products**; it
   must beat raw Gaia only in the declared low-S/N / distant bins and must not regress the
   Hipparcos nearby control. No StarHorse distance targets, no `1/ϖ`, negative parallaxes kept.
4. **Age head is a calibrated distribution**, not a point guess: log-age multi-quantile/split-normal
   primary, APOKASC3 fully held out, bias and MAE reported in Gyr **and** log Gyr vs seismic
   references, >10 Gyr slice with n, "StarHorse-conditioned, not truth" stated.
5. **Alpha interface exists and is testable**, source-tagged and gauge-fixed, but **no α number**
   is published until an audited α source and the five gates in [05](05-heads-losses-metrics.md) §6 pass.
6. **Metric contract emitted by every run**: `prediction-certificate.json`, `metrics.csv`,
   `joint.csv`, `distance.csv`, `age.csv`, `schema.json` with `density_kind`,
   `loss_denominator`, per-row Jacobian, exact conformal rank, grouped bootstrap and
   failed-verdict provenance ([05](05-heads-losses-metrics.md) §7.1).
7. **A written architecture decision** (head + inputs + parent + budget) that the later
   labels-v3 / all-Gaia rounds can consume without re-litigating the design.

**Out of scope / explicitly deferred:** new label ingestion, α science, joint distance–A_V
(no 3D dust map), genuine no-XP training (needs a small Gaia DR3 re-join, see §3.4), all-Gaia
1.8B inference, DR4, and any "physical truth" claim.

---

## 1. Load-bearing design decisions

| # | Decision | Why (evidence) |
|---|---|---|
| D1 | **W0 is the launch gate** — no new training run before the all-family FP64 certificate + full-inventory versioned rescoring | 19 failure entries: six are precision/contract, four are method failures; the bounded `wetmolvn` certificate (3 NSF checkpoints × 128 rows) cannot clear full-role failures, and the current review mixes three library versions (`d94a969` producers, `1d735c6` density arms, `4e81a72` evaluation) |
| D2 | **Matched unit = 10M presentations = 306 updates = 32 passes over the 19,299 fit labels**; 8,250 updates = 875 passes | 306 updates cannot move the 141-star tail; the tail needs new labels/inputs, not more compute |
| D3 | **Parent86 is the frozen control** for all matched 10M arms; the 150M/4.5B state is a **separate, non-matched** arm only | 17/18 existing arms load parent86; the big state saw 10× more rows and cannot be a control |
| D4 | **Input tiers T0/T1/T2**; the 134-slot photometry+XP core is the only input to the Teff/logg/FeH/α heads; a 22-feature context vector (parallax+error+zp, PMs, sky position/HEALPix, SFD EBV, quality bits) is concatenated **after the 256-d latent** (256→384) for distance/age heads only | An input-width change costs +4.80M params (14.1%) and breaks every checkpoint; the context conditioner is 13.7k params and loads the parent unmodified |
| D5 | **One-role rule:** parallax, EBV and PM are each **input OR likelihood OR target, never two**, except the explicitly named measurement-assisted arm | Prevents double counting and the feature-only distance claim becoming false |
| D6 | **Joint head is block-structured:** a 3×3 Teff/logg/FeH core (identified on 19,299 complete-case rows) + separate distance, age and α terms. A 6×6 joint covariance is unidentifiable today (α has 0 rows, age complete-case 4,440, distance roles disjoint) | Publishes only what is identified; the missing-data marginal is exact for the block |
| D7 | **Build order H1→H6:** full-cov Gaussian (masked, Σ+R) → MDN K=4/8 → distance head → age multi-quantile → certified flows → α interface | Cheapest correct joint first; flows only after the certificate + `8973bcd`; forward inversion retired |
| D8 | **Distance is two products forever:** predictive PDF `p(s|x)` and profile-likelihood confidence set `C_α`; a normalized `p·L` is a posterior under a learned prior, not prior-free inference | Statistically correct naming; avoids the "Bayesian distance" overclaim |
| D9 | **Age is `log(age/Gyr)`** with multi-quantile primary; APOKASC3 held out entirely; report Gyr and log Gyr; no "true age" | StarHorse posteriors are model/parallax-conditioned; asymmetric percentiles are the native label |
| D10 | **α is interface-only** this round: source-tagged output layer, gauge constraint, per-source κ_o; five gates before any number | APOGEE α/M ≠ GALAH α/Fe; zero α rows ingested; tail has no overlap pairs |
| D11 | **Per-row Jacobian, `density_kind`, `loss_denominator`** in every density path; current single-scalar Jacobian is wrong for log-distance/log-age | `run_multilabel_screen.py` subtracts one scalar (log w) and omits `+log|T'|`; with log d/log age the native density misses the `−log d`/`−log age` term per row (correct offset `log w + log d`) |
| D12 | **Metrics:** exact conformal rank `⌈(n+1)(1−α)⌉`; joint rank-score region primary, Bonferroni box secondary; grouped two-way bootstrap; proper scores from samples for flows | Current CP uses plug-in quantile (+6.0e-4 at n=2046); 91.9%/29.1% joint numbers are Bonferroni-only |

**Corrected no-XP fact (must replace earlier wording).** The 83,491 "no-XP" rows in the
enriched cohort are **label-only**: 0/140 features observed, 0 astrometry, and 0 membership in
fit/tune/cal/test. 64,994 carry native labels. There is **no genuine XP-off training or validation
cohort on disk**; all XP-off evidence in the repo is synthetic masking. A genuine cohort needs a
small Gaia DR3 TAP re-join of those 83k IDs (GB-scale, no new survey) — recommended as a cheap
side task, not a blocker for this round.

---

## 2. Target vector, transforms and metrics

Full specification in [05](05-heads-losses-metrics.md) §1–§7. Summary:

| # | Target | Model coordinate | Transform | Error model | Role this round |
|---|---|---|---|---|---|
| 0 | Teff | `T` (K) | identity | symmetric K | trainable |
| 1 | logg | dex | identity | symmetric dex | trainable |
| 2 | FeH | dex | identity | symmetric dex | trainable |
| 3 | distance | `s=log(d/kpc)` | log | raw signed `ϖ ± σ_ϖ` (+zp) | trainable, new |
| 4 | age | `τ=log(age/Gyr)` | log | asymmetric 5/16/50/84/95 pct | trainable, small cohort |
| 5 | alpha | dex | identity | symmetric + per-source κ | interface only |

Standardization fitted on fit only; frozen order; no positional inversion (`eval_ensemble` today
hardcodes index 5=parallax, index 0=Teff — must become registry-driven).

**Native-unit density.** `log p_Y(y|x) = log p_U(u|x) − Σ_obs log w_j + Σ_obs log|T'_j(y_j)|`
with `T'=1` (identity), `T'=1/d` (log kpc), `T'=1/age` (log Gyr). Per row and per observed
coordinate, not one scalar.

**Two estimands never mixed.** `p_obs(y_obs|x)=∫p_intr(y|x)N(y_obs;y,R)dy`. `p_intr` is not
identified from `p_obs` without `R`; do not call it truth or score it against noisy labels. Every
NLL/CRPS row carries `density_kind ∈ {intrinsic, observed_label}`; today `observed_lp` equals
intrinsic for non-noise-aware arms and saved variance is always intrinsic — fix by saving both
accessors explicitly.

**Metric contract** (emitted by every run; details [05](05-heads-losses-metrics.md) §7.1):
`metrics.csv` one row per model × label × stratum with point errors, CRPS/NLL, PIT, coverage,
width/interval score, calibration slope, grouped CI and `evidence_status`; `joint.csv` with joint
coverage, rank-score vs Bonferroni, region volume/level-set method, energy/variogram;
`distance.csv` in mas and kpc with `frac_ci_empty`/`frac_ci_upper_unbounded` and `ci_kind`;
`age.csv` in Gyr and log Gyr with seismic reference and >10 Gyr slice. Tail FeH<−2 always
reported separately; n<20 descriptive only.

---

## 3. Inputs and architecture

Full inventory and falsifiers in [07](07-inputs-architecture.md) §1–§6. Summary:

### 3.1 Input tiers (one-role rule enforced)

| Tier | Features | Used by | Status |
|---|---|---|---|
| **T0** feature-only | 134 photometry+XP slots, parent scaler, structural missingness token | Teff/logg/FeH/α heads | usable now |
| **T1** context | 14 continuous: `ϖ`, `e_ϖ`, zp, `pmra`, `pmdec`, `e_pmra`, `e_pmdec`, `l`, `b`, HEALPix nside-64 index, SFD E(B−V), optional RUWE/neighbour bits; +8 mask/quality bits; fit-only normalization | distance/age heads only, concatenated after the 256-d latent → 384 | usable now except EBV until provenance gate |
| **T2** measurement-assisted | T1 minus `ϖ,e_ϖ,zp` + raw signed parallax likelihood | distance/age product, separate name | likelihood code exists |
| **T3** joint distance–A_V | T2 + 3D dust prior at predicted distance | future | deferred (no 3D map) |

**Falsifiers before any input claim:** shuffle T1 features within nside-8 sky patches and
retrain (retained gain = architecture, not information); Hipparcos nearby control must not
improve; EBV shuffle must erase any EBV gain; T2 with ϖ corrupted ×1.01 must show the head
tracking the corruption if it is leaking. The EBV column (`max 132.79`, 2.56% >2) is **not**
admissible as physical extinction until the provenance gate; 2D SFD is a line-of-sight total,
not `A_V(d)`.

### 3.2 Encoder/backbone

Keep the 34.2M-param parent86 backbone; do not replace it this round. All matched 10M arms start
from parent86 with fresh matched Adam, encoder lr 1.25e-6, head lr 1e-3, 500 warmup, recon weight
0.01 (or 0.03 only where matched). The 2.25B/4.5B state is a separate arm (E6/W4) after 24 Oct.
The decoder (136 outputs) stays; PARALLAX channel is excluded from reconstruction scoring today
and can be enabled as a **declared representation experiment** with role exclusion.

### 3.3 Heads

| Module | Inputs | Output | Loss | Adds |
|---|---|---|---|---|
| A 3-label core | latent 256 | μ + full 3×3 `scale_tril` (observed-label `Σ+R`) | masked Gaussian measurement NLL (exact observed-subvector, already implemented) | anchor |
| B distance | `[latent‖context]` 384 | `s=log(d/kpc)`, `log σ_s` or a 1-D flow/MDN | convolved raw-parallax NLL: residual-space GH default + prior-space GH fallback (delta-method as control only) | new |
| C age | 256 (+declared context) | log-age quantiles / split-normal / flow | pinball + crossover (primary), split-normal NLL (control), flow (third) | new |
| D alpha | 256 | source-tagged μ, log σ | NLL with per-source affine scale, gauge-fixed | interface |
| E joint | 256 | block-structured 3×3 core + separate s/age | sum of A+B+C terms with own denominators | superset, later |
| F stage-aware | 256 | K=4 stage logits from measured logg×Teff + per-stage 3×3 | A + stage CE or K-mixture | attacks Kiel branches/coverage |
| G difficulty width | within A/E | `log σ` linked to observed precision | positive width-vs-|residual| slope gate | attacks tail coverage |

### 3.4 No-XP / all-Gaia readiness (design only)

- This round: synthetic XP masking only; the existing 25k-label claim is not reproducible.
- Smallest honest experiments: input-masking mixture `mask_mixture_xp_full_frac 0→0.20` (knob
  exists), distillation XP-teacher→XP-masked student (gate on tail, not global: teacher tail bias
  +0.39 dex propagates), and the Gaia DR3 re-join of 83k label-only IDs for a genuine cohort.
- The 1.8B production path is not built this round; the HEALPix nside-64 schema and registry are
  the future path.

---

## 4. Losses and likelihoods (what is trainable now)

1. **Reconstruction regularizer** `0.01×masked MAE (+0.1 spectral)`; starting value, not optimum;
   scored features = 134 (or 135 with PARALLAX as declared experiment).
2. **Point/quantile:** Huber control; multi-quantile `(0.05,0.16,0.5,0.84,0.95)` + crossover;
   expectile as a shape control (not intervals).
3. **Gaussian:** diagonal control; **full covariance** with exact masked observed-subvector
   marginal; low-rank as a cheaper factor control; analytic label-error convolution `Σ+R`
   (exact for Gaussian).
4. **MDN:** K=4 diagonal then K=8 full; mixture NLL; tail/multimodality for Kiel branches.
5. **Flows:** NSF/cINN/MAF/BPF (and NFM/CFM after `8973bcd`) only after W0 certificate; 3-D
   label-noise convolution is **MC-only** — gate `|NLL_MC64−NLL_MC512|/|NLL_MC512| ≤ 1e-3`
   ex ante (the GH non-convergence lesson); fixed saved QMC nodes at evaluation.
6. **SLS/SLSConformal:** implement/train after the Gaussian/MDN screen; conformal post-fit.
7. **Label-noise model:** per-source `R_s=κ_s²diag(σ_quoted²)+b_sI`, κ_s **fitted** from measured
   overlaps (DESI−APOGEE ⇒ κ_DESI≈7.3; GALAH−APOGEE scatter 0.048), never an invented floor.
8. **Tail weighting/DRO:** secondary declared arm only; primary estimand is population-weighted;
   at n=141 no weighting buys precision — report Poisson-limited.
9. **Ensemble as mixture:** `p_ens=(1/M)Σp_m`; score with mixture NLL/CRPS/energy; no spread-only
   uncertainty.
10. **No input-EIV claim:** no per-star input σ enters the encoder; attenuation slope is a
    diagnostic, never a correction divisor.

---

## 5. Distance head (highest-risk new module)

**Detailed loss specification:** [08-losses-age-distance-alpha.md](08-losses-age-distance-alpha.md) §1.

**Model.** Latent `s=log(d/kpc)`; predictive `p_θ(s|x,context)`.

**Training loss (marginalized over s, through raw signed parallax):**

```
L_A = −Σ_i log ∫ p_θ(s|x_i) · N(ϖ_i | z_i + exp(−s), σ_eff,i) ds
σ_eff² = κ_ϖ² σ_ϖ,quoted² + σ_zp²(G,colour)
```

No StarHorse distance targets, no `1/ϖ`, negative parallaxes retained. Gaussian `p_θ`: **two fixed
quadrature rules** — residual-space GH in `h=e^{−s}` (default; exact in the narrow-error limit and
representable while `σ_eff < h*/6.63`) with prior-space GH in `s` as the fallback; `rule="auto"`
selects per row. The literal "GH-32 in `s`" alone is **not adequate** (measured error 1.7e-3 at
`σ_eff=0.1h*`, divergent at `0.01h*`). Certificate records `quadrature.rule`,
`residual_rule_rows`, `min_prior_resolution`; flow/MDN use QMC-64 fixed nodes. Report the
prior-vs-likelihood mass fraction (how much the answer is data-driven) and per-ϖ/σ-bin fallback.
Do not implement the naive prior-density×likelihood form (double counts the prior).

**Products (named separately forever):**
- **A. predictive PDF** `p_θ(s|x)` — a normalized predictive distribution (population-dependent).
- **B. assisted profile-likelihood confidence set** `C_α={s: −2[ℓ(s)−ℓ(ŝ)]≤χ²₁(1−α)}`,
  `ℓ(s)=log N(ϖ_obs; z+e^{−s}, σ_eff)`; may be empty or unbounded; never cosmetically capped.
- **B′ forbidden:** normalized `p_θ·L/Z` may be reported only as a posterior under a learned
  population prior.

**Minimum numerical tests before launch:** narrow-error limit to 1e-8; float64 gradcheck;
negative-parallax finite and correctly shaped; empty-batch differentiable zero; GH16→256
relative convergence <1e-3; 1-D Gaussian vs `scipy.integrate.quad` to 1e-10; unit identity
`s=0⟹1 mas⟹1 kpc`; batch-1 vs batch-128 identical; measured ϖ never enters `1/x` or `exp`
except through `h(s)`.

**Validation:** Hipparcos 4,166 (nearby **negative control**; raw Gaia 0.960 mas, previous
frozen ridge 3.23 mas — no beyond-Gaia claim); clusters/seismic later; score in parallax space
and kpc; never bin by inverted noisy ϖ; hold out all 99,574 reference IDs.

**Pipeline gate:** `gaia_quality_and_zero_point_verified`, `distance_input_no_copy_verified`,
`correlated_teacher_policy_verified` before any distance NLL.

---

## 6. Age and alpha heads

**Detailed loss specification:** [08-losses-age-distance-alpha.md](08-losses-age-distance-alpha.md) §2–§3.

**Age data and systematics:** [09-age-data-and-systematics.md](09-age-data-and-systematics.md).
The current StarHorse labels are **MSTO/SGB only** (APOGEE DR17 + GALAH DR3, 4,440 strict)
and show a +1.7–2.3 Gyr bias against 50 held-out LEGACY seismic stars. New sources:
**APOKASC-3** (15,808 giants; already local; 8,895 in the enriched cohort but only 672 in
the 10M benchmark → train giant ages on the 150M cache), **Willett et al. 2026**
(17,000 Kepler/K2/TESS asteroseismic ages, Zenodo 20428851, 44 MB; Gaia-ID column currently
unusable — crossmatch via KIC/TESS/APOGEE IDs), open/globular-cluster coeval anchors, and
**Gaia DR4 FLAME** after 2 Dec 2026 (`ap_rvs` 35.8M, `ap_xp` 1B; crosswalk via
`dr3_neighbourhood`). APOKASC3 and Willett differ by 0.2–0.4 Gyr median and up to 3.3 Gyr
NMAD for the oldest stars, so age must be trained with a **multi-teacher hierarchical
loss + meta-correction layer** and reported on a **named age scale with a systematic
envelope**, never as truth.

**Age.** `τ=log(age/Gyr)`. Primary: multi-quantile `(0.16,0.5,0.84)` (+0.05/0.95) with
crossover, because it matches the reported asymmetric percentiles and is proper. Controls:
split-normal in τ (documented approximation), heteroscedastic Gaussian on τ, flow third. Declare
the label-noise-on-spread bias (fitting to StarHorse percentiles makes widths narrower than the
true predictive spread). APOKASC3 `table4.dat` (15,808) **fully held out** of fit/tune/cal;
50 LEGACY stars descriptive only; report bias/MAE in Gyr and log Gyr, coverage, >10 Gyr slice.
No "true age" claim.

**Alpha.** Interface only: source-tagged output layer, gauge (`a=0,b=1` for GALAH or Σn a=0),
per-source `κ_o`, joint 2-D (FeH,α) when labels exist. Five gates before any number: native α
column with flags; overlap offsets/scatter with n; source-tagged beats single-α on held-out
overlap stars; cal-split coverage; 2-D level-set coverage. Until then F05 stays "NOT RUN".

---

## 7. Calibration and distribution metrics

- **Split conformal:** exact finite-sample rank `q̂=s_(⌈(n+1)(1−α)⌉)`; two-sided score
  `max(q_lo−y, y−q_hi)` at one level or per-tail α/2; **raise** when the rank exceeds n; no silent
  zero offsets. Fix `training/conformal.py` (plug-in quantile today; +6.02e-4 at n=2046).
- **Joint region:** rank-score (Mahalanobis radius or `−log p̂`) primary → finite-sample marginal
  joint coverage; Bonferroni box secondary; report region volume and level-set integration method.
  Relabel the existing 91.9%/29.1% joint numbers as Bonferroni-only.
- **Mondrian** by pre-registered strata (FeH<−2, logg branch, G, tier); report the exact discrete
  achievable coverage at n=141 (index 128/141; Poisson-limited).
- **Proper scores:** NLL only within the same family and `density_kind`; CRPS from samples for
  flows (never a Gaussian proxy); energy and variogram for dependence; PIT/rank/SBC per
  coordinate; interval score + width + region size alongside coverage.
- **Uncertainty:** grouped paired bootstrap by unique `source_id` **and** HEALPix nside-64 block
  (duplicates exist), 1,000 reps, seed recorded; multiplicity stated (Bonferroni over
  pre-registered slices or BH-FDR); n<20 descriptive.
- **Locked test:** the current 97,695 test stays a **development** cohort for this round; final
  claims move to the labels-v3 locked test when it exists. Test read once per wave family.

---

## 8. Experiment campaign (58 named runs; full registry in [06](06-experiment-campaign.md))

**Cost basis:** measured 10M/306-update arms = 403–2,530 s by family; 270M = 16,572 s; fitted
`t≈342+60.1·(P/1e6)` s ⇒ 59.8M presentations/h (matches `dd27e718` 58–62M/h). No manifest
carries peak GPU, so W0 makes it mandatory. Memory class: full-cov/MDN/quantile ≤6 GiB; NSF/BPF/
MDN-8 8–12 GiB; lower `micro_batch` before ever upsizing the GPU.

### Wave summary

| Wave | Scope | Run IDs | GPU-h | Exit gate |
|---|---|---|---|---|
| **W0** | adoption/verification of `wetmolvn`; all-family full-role FP64 certificate at `8973bcd`; full-inventory versioned rescore; BPF corrected map; recon-source + CUDA mask provenance; TTT; weighted/Mondrian CP | 10 (0 train) | 8.2 (0.34 d) | certificate per family + `reconciliation.json` + incumbent tag; **no training before this closes** |
| **W1** | head/loss screen at 10M, parent86, native3, 306 updates, recon 0.03, seed 42, `ttt=false`: quantile, expectile, full-cov, low-rank, MDN4/8, SLS/SLSConformal, NFM/CFM, no-recon controls; diagonal/Huber/NSF/cINN/MAF/BPF reused | 12 new (11 train + 1 post-fit) | 3.5 (0.15 d) | family advances only with W0 certificate **and** no point-MAE regression vs diagonal control |
| **W2** | matched comparisons: clean vs noise-aware NSF; NLL vs contrastive vs combined + Mondrian density-CP; distributional ensemble as predictive mixture; energy/variogram; marginal CRPS/PIT/quantile; region size vs coverage; replicate seeds | 10 (5 train, 5 analytic) | 2.8 (0.12 d) | one certificate-backed density ranking; no Gaussian proxy for flows |
| **W3** | target heads at 20M unique rows (staged from 150M cache), latest/selected parent: distance feature-only vs assisted, context PM/position/EBV, zero-point; age log-NLL + quantile; distance CP in parallax space; 6-target partial and α head blocked/deferred | 8 (7 train, 1 post-fit) | 4.7 (0.20 d) + CPU staging | distance no high-S/N regression vs raw Gaia; gain only in declared bins; age bias/MAE improve vs seismic; α no claim |
| **W4** | inputs/architecture at 20M: XP-off, mask indicators, survey LOO, recon weight 0.01, EBV, parent 2.25B/4.5B, 150M pass, ensemble seeds, astrometry context for age only | 11 | 9.0 (0.38 d) | dust/PM/position claim resolved by falsifiers; parent/budget question answered |
| **W5** | finalists: winning head+inputs at 20M ×3 seeds; 150M if tune says so; Mondrian CQR/level-set calibration; single locked-test read + scorecard + F01–F14 | 7 (5 train, 2 read/report) | 8.8 (0.37 d) | finalist + certificate + scorecard; coverage ±0.02 marginal / ±0.05 Mondrian n≥200 |

**Totals:** 58 run IDs (39 training, 19 inference/analytic) ≈37 GPU-h ≈1.5 GPU-day raw; budget
**3 GPU-days** with scoring/re-runs; ~16 GB new checkpoints. Critical path: W0 → W1/W2 →
**W3.0 plumbing (2–5 Nov)** → W3 → W4 → W5. W4-parent/W5 wait on `dd27e718` (24 Oct); W0–W2 can
use a second session from 23 Oct 18 UTC.

### Dated sequence

| Window (2026) | Work | Exit |
|---|---|---|
| 22–23 Oct | W0 CPU: recon-source, CUDA-mask provenance, tail-count reconciliation, scoring contract; fetch `wetmolvn` | W0.1 verified |
| 23–24 Oct | `w0_cert_allfamilies_v1` (fresh GPU session; not before 23 Oct 18 UTC); 24 Oct `dd27e718` completes → preserve 32 states | certificate per family |
| 24–26 Oct | versioned rescore + BPF map + TTT + weighted/Mondrian CP | reconciliation; freeze incumbent |
| 27–29 Oct | W1 screen (12 arms, 2 sessions × 6) | family ranking |
| 30 Oct–1 Nov | W1 analysis; W2 matched comparisons + seeds + analytic runs | one density ranking |
| 2–5 Nov | W3.0 plumbing: CANFAR joint-continuation stage, 20M staging keys, new heads, target spaces | staging hash + smoke |
| 6–8 Nov | W3 distance (4) + age (2) | distance/age gates |
| 10–13 Nov | W4 ablations (11 runs, 3 sessions) | input/parent verdicts |
| 16–20 Nov | W5 20M ×3 seeds; promote 150M if tune says so | finalist + calibration |
| 23–26 Nov | locked-test read once, scorecard, F01–F14 | campaign close |
| 27–30 Nov | buffer: for-cause re-runs only | ledger + architecture decision |

---

## 9. Code implementation plan (launch-blocking gaps)

| # | Change | Files | Tests/gate |
|---|---|---|---|
| C1 | Per-row/per-observed Jacobian; `density_kind`; `loss_denominator`; remove hardcoded `/3` for d≠3 | `scripts/run_multilabel_screen.py:337,322-326`; `training/multilabel_experiments.py:186` | synthetic log-distance/log-age density equals analytic; certificate keys |
| C2 | Target registry `TargetSpec` (unit, transform, positivity, error kind, source tag, status) and registry-driven inversion | new `training/targets.py`; `eval_ensemble.py`, `infer_msa.py` | reorder labels test; legacy config regression <1e-6 |
| C3 | Head-family union: `mvgaussian`, `lowrank`, `mdn`, `quantile`, `expectile`, `sls` (+ `slsconformal` post-fit); flow/nfm/cfm after pin | `training/multilabel_experiments.py:84-151`; `scripts/run_multilabel_screen.py:876-884` | one synthetic test per family; finite loss; exact NLL on known Σ/π/τ; `--smoke` |
| C4 | Post-latent context conditioner (22→128) + distance head B + age head C + α interface D; block-structured joint E | `training/joint_prediction.py`, new `training/context_heads.py`; `run_joint_continuation.py` stage | checkpoint loads unmodified; gradcheck; context ablation |
| C5 | Distance likelihood: convolved raw-parallax NLL (residual-space GH default + prior-space GH fallback), zero point, profile-likelihood CI; tests from §5 | `training/parallax_likelihood.py` (exists), new `training/distance_heads.py` | §5 nine tests; no-pollution check; `residual_rule_rows`/`min_prior_resolution` in certificate |
| C6 | Metrics/reporting module + schema; exact conformal rank; joint rank-score; grouped bootstrap; energy/variogram/region size; certificate | new `src/masked_stellar_autoencoder/reporting/`; `scripts/make_release_report.py`; fix `training/conformal.py` | synthetic exchangeable coverage ≥1−α over 2,000 reps; plug-in version shown to fail |
| C7 | CANFAR stages: joint-continuation; 10M/20M staging; peak-GPU/dtype/batch receipt; library pin + provenance test | `batch_scripts/canfar_entrypoint.sh`, `scripts/launch_stellar_improvement.py`, `scripts/stage_joint_reconstruction.py`, `pyproject.toml` | smoke + staging hash; `peak_allocated_bytes` present; library commit test |
| C8 | Fix observed/intrinsic aliasing; flows' complete-case selection printed; MC convolution convergence gate | `run_multilabel_screen.py:322-326`; `multilabel_experiments.py:201-218` | certificate `density_kind`; complete-case n printed |
| C9 | Retire/demote: forward inversion, scalar contrastive, NRE grid, old BPF verdict | ledger/docs | no product path depends on them |

**Owners** (proposed): C1/C2/C4/C5 model Lead; C3 model runner; C6 reporting; C7 platform; C8/C9
verification. Each C-item lands with tests before the wave that needs it; no silent schema change.

---

## 10. Risks and mitigations

| Risk | Trigger | Mitigation |
|---|---|---|
| W0 certificate still fails for some families | f64 CUDA vs CPU > gate | ship certified Gaussian/MDN only; flows stay producer-only appendix; do not rank NLL |
| Head screen shows no gain over diagonal | paired ΔNLL/MAE CI includes 0 | keep diagonal + stage-aware/difficulty-width; close the family claim honestly |
| Distance still loses to raw Gaia in high S/N | W3 high-S/N bins regress | ship feature-only as prior, assisted confidence set with raw-Gaia fallback; claim only low-S/N/d>3 kpc gains |
| Age bias persists | seismic bias not reduced | report model-conditioned age; rely on labels/DR4 FLAME later; no truth claim |
| GPU memory for flows | peak >12 GiB | lower `micro_batch`; measured peak before upsizing; BPF inverse chunking |
| W3.0 plumbing slips | staging/heads not ready by 5 Nov | W3.0 is critical path and starts immediately after W2; scope frozen |
| Labels arrive late | no labels-v3 by 20 Nov | campaign is label-light; W5 runs on native3 with the 141-star tail caveat; labels swap in for a later round |
| `dd27e718` final state delayed | no 4.5B state by 24 Oct | W0–W2 use parent86; W4-parent can use 2.25B state; final promoted later |
| Test reuse inflates claims | multiple reads of 97,695 | development cohort only; final claims on labels-v3 locked test; one read per wave |

---

## 11. Decisions needed from you

1. **Ratify the W0-first gate and the 58-run campaign** (3 GPU-days, 12 GiB MIG, second session
   from 23 Oct 18 UTC; no training before the all-family certificate).
2. **Ratify the scope/out-of-scope list**: forward inversion retired/demoted; α interface only;
   joint 6-target and joint distance–A_V deferred; no all-Gaia/DR4 in this round.
3. **Approve the architecture shape**: block-structured joint head (3×3 core + separate
   s/age/α), post-latent context for distance/age only, parent86 as matched control and the
   150M/4.5B state as a separate arm.
4. **Approve the no-XP correction**: this round uses synthetic XP masking only; if you want a
   genuine XP-off validation cohort, approve the small Gaia DR3 re-join of the 83,491 label-only
   IDs (GB-scale CPU task, not a new survey).
5. **Confirm the current 97,695 test stays a development cohort**, with final claims moved to the
   labels-v3 locked test when your ingestion lands.
6. **Priority if time slips** (recommended order): W0 → W1/W2 → distance/age → input ablations;
   α and 6-target are label-gated and can wait.

---

## 12. Immediate next actions (next 5 days)

1. **Fetch and verify `wetmolvn`** outputs; reconcile the three cases against its producer log;
   extend the certificate to all families/full roles at `8973bcd`.
2. **Write the scoring contract** and fix C1/C2/C8 (per-row Jacobian, `density_kind`,
   `loss_denominator`, observed/intrinsic aliasing, hardcoded `/3`) with synthetic tests.
3. **Start W0 CPU runs**: recon-source audit, CUDA-mask provenance, tail-count reconciliation
   (141 vs 145 / 14 vs 0), weighted-CP provenance, Mondrian metric layer.
4. **Add head-family union C3** and the 22-feature context conditioner C4 with smoke tests; add
   the distance likelihood C5 and its §5 numerical tests.
5. **Build the 20M staging path C7** and the joint-continuation CANFAR stage; record
   `peak_allocated_bytes`/dtype/device/batch in every manifest.
6. **On 24 Oct:** preserve all 32 `dd27e718` states and receipts; archive the 150M cache and
   benchmark references; re-point the separate non-matched arm to the final state.
7. **27 Oct:** launch W1 (12 matched 10M arms across two sessions); no run is scored before its
   W0 certificate.

---

## Appendix — run-ID registry (summary; full tables in [06](06-experiment-campaign.md) §4)

- **W0 (10, inference/repair):** `w0_cert_adopt_wetmolvn_v1`, `w0_cert_allfamilies_v1`,
  `w0_rescore_inventory_v1`, `w0_rescore_verify_v1`, `w0_bpf_map_v1`,
  `w0_recon_source_v1`, `w0_mask_provenance_v1`, `w0_ttt_fullrole_v1`,
  `w0_weighted_cp_v1`, `w0_mondrian_cp_v1`.
- **W1 (12 new: 11 train + 1 post-fit):** `w1_quantile_s42`, `w1_expectile_s42`,
  `w1_fullcov_s42`, `w1_lowrank_s42`, `w1_mdn4_s42`, `w1_mdn8_s42`, `w1_sls_s42`,
  `w1_slsconformal_v1`, `w1_nfm_s42`, `w1_cfm_s42`, `w1_fullcov_norecon_s42`,
  `w1_mdn4_norecon_s42`; diagonal/Huber/NSF/cINN/MAF/BPF re-used.
- **W2 (10: 5 train + 5 analytic):** `w2_nsf_noise_norecon_s42`,
  `w2_clean_vs_noise_nsf_v1`, `w2_objective_threeway_v1`, `w2_ensemble_mixture_v1`,
  `w2_sample_metrics_v1`, `w2_region_size_v1`, `w2_fullcov_s43`, `w2_fullcov_s44`,
  `w2_mdn4_s43`, `w2_mdn4_s44`.
- **W3 (8 + 1 deferred):** `w3_dist_featonly_20m`, `w3_dist_assist_20m`,
  `w3_dist_context_20m`, `w3_dist_zp_20m`, `w3_dist_cp_v1`, `w3_age_lognll_20m`,
  `w3_age_quantile_20m`, `w3_joint6_partial_20m`; deferred `w3_dust_av_20m`.
- **W4 (11):** `w4_xp_off_20m`, `w4_mask_ind_20m`, `w4_loo_galah_20m`,
  `w4_loo_apogee_20m`, `w4_recon001_20m`, `w4_ebv_20m`, `w4_parent_latest_20m`,
  `w4_long_150m`, `w4_ens_20m_s43`, `w4_ens_20m_s44`, `w4_astro_ctx_age_20m`.
- **W5 (7):** `w5_final_20m_s42`, `w5_final_20m_s43`, `w5_final_20m_s44`,
  `w5_final_150m_s42` (+`s43` only if tune improves), `w5_calibrate_v1`,
  `w5_lockedtest_v1`.

Every run: one GPU (12 GiB MIG), 8 CPU, 64–137 GB RAM, own output dir, never overwritten;
`prediction-certificate.json` required before scoring; cohorts never mixed; seed 42 except
declared finalist/seeds; old failures retained.
