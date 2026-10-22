# CKMfitter/Rfit, profile likelihood and theory-error conformal for MSA

**Prepared:** 2026-10-22. Assessment of the proposed frequentist programme for the six
stellar parameters (and correlated/marginalized two-parameter products). This is a design
note, not a claim that the full physical-model pipeline is implemented.

---

## 0. Short answer

The philosophy is **right for the problem and worth adopting — but as a second, independent
inference track and as a calibration/validation layer, not as a drop-in replacement for the
current label-trained network.**

The current MSA head is a **discriminative conditional-density model**
`p(labels | photometry/XP, context)` trained on noisy, model-dependent survey labels. Profile
likelihood and Rfit are tools for a **generative physical model** with an explicit likelihood
`p(observables | stellar parameters, physics)`; applying them requires an isochrone/seismic
forward model and cannot be mimicked by profiling the ML network's output. The productive
architecture is the CKMfitter/UTfit pattern: **two independent tracks that cross-check each
other** — an ML predictive track for all-Gaia scale and a physical profile-likelihood track
for stars with rich data and for anchors — with disagreements treated as systematics, never
averaged away.

Where I would port the ideas immediately (cheap, high value):

1. **Separate statistical and theoretical errors explicitly** in every product; never let a
   prior silently shrink a theoretical error.
2. **Model-family ensembles** (MIST/PARSEC/GARSTEC/BASTI, different mixing lengths/overshoot)
   to define a theory dispersion `σ_theo(x)`.
3. **Mondrian conformal by evolutionary stage** (MS/TO/SGB/RGB/RC/AGB) plus tier
   (`xp_full`, `xp_partial`, `no_xp`), pre-registered.
4. **Theory-inflated nonconformity scores** with cross-fitted `γ`; report the empirical
   conformal interval and the theory envelope **separately**.
5. **Profile-likelihood cross-checks** on APOKASC3/Willett/cluster anchors for age, mass,
   distance and `A_V`, including 2D profile contours for correlated pairs, compared with the
   ML predictive contours.
6. **Model-rejection flags** (a high `χ²_min` should be published, not normalized away).

What I would not do:

- Treat a hard Rfit boundary as physical truth in all regimes, or use `Δχ²=1/4` blindly
  without simulation calibration.
- Call synthetic-grid conformal sets "guaranteed coverage" on real stars.
- Replace the all-Gaia predictive product with per-star grid profiling (compute-prohibitive).
- Profile the ML network's output as if it were a generative likelihood without a forward
  model; the object would not be a profile likelihood.

---

## 1. What is directly useful in the proposal

### 1.1 No-prior, no-normalization discipline

The prior-volume warning is real and relevant to stellar ages. Bayesian isochrone fits with
wide priors on mass, age, `[Fe/H]`, `α`, `A_V` and evolutionary stage can report
overconfident ages, especially near the MSTO, the RGB bump and the RC, where the mapping from
observables to age is multivalued. A profile-likelihood scan over nuisance parameters is
invariant to grid-point density and does not manufacture information by normalization. That
is a valuable **safeguard**, not just an alternative convention.

### 1.2 Separate statistical and theoretical errors

MSA currently convolves only quoted **statistical** errors (`R` in the likelihood). It does
not have a principled channel for "changing the mixing length changes age by 1 Gyr". Making
that explicit — a theory nuisance vector `ξ_theo` with a bounded/penalized range, plus a
model-family ensemble — is the single most valuable part of the proposal for our age problem
(see [09](09-age-data-and-systematics.md)).

### 1.3 Model rejection

If a star's best-fit `χ²_min` is unacceptable, a frequentist analysis can say "the model does
not describe this star" and publish a rejection/flag. The current ML product always returns
a normalized posterior; a model-rejection/out-of-distribution flag is a needed complement
(and is easy to add at the head level as an OOD/energy score).

### 1.4 Profile likelihood for correlations

Profiling nuisance parameters gives **1D and 2D confidence regions** with a clear frequentist
interpretation, e.g. `(age, mass)` or `(distance, A_V)` at fixed `(Teff, logg, [Fe/H])`.
That is exactly the kind of correlated/marginalized statement users need. A 2D profile scan
is far cheaper than a full 6D marginalization once the forward model exists.

---

## 2. Where the proposal does not transfer directly

### 2.1 Our model is not a generative physical model

`p_θ(labels | features)` is a **conditional density estimator**, trained against survey
labels. There is no `p(observables | θ_physics)` inside it: `θ` are network weights, not
stellar parameters, and the labels are noisy, model-dependent measurements. Profiling the
network output over "mass/[Fe/H]" is not defined unless we add a physical forward model and a
likelihood. Hence:

- **Profile likelihood/Rfit = a new physical-inference track** (isochrones + seismic +
  spectroscopy + astrometry), not a modification of the ML head.
- The ML head remains the scalable predictive track for 1.8B sources, with empirical
  calibration.

### 2.2 Rfit's hard theoretical boundary is an idealization

CKMfitter's theoretical errors are **bounded nuisance parameters** motivated by
renormalization-scale variation, with a well-defined range. Stellar "theory error" is:

- continuous and model-family dependent (MIST vs PARSEC vs GARSTEC vs BASTI; mixing length,
  overshoot, opacities, helium, `ΔY/ΔZ`, rotation, mass loss);
- itself uncertain (the true theory error is unknown);
- not independent across parameters (age, mass, radius, `[Fe/H]`, `α` shift together).

The step-function penalty in the proposal is implementable but has consequences: the
likelihood is flat inside the box (so the ML/grid point estimate is not unique), the
boundary is non-differentiable, and a too-small `Δ` produces spurious rejection while a
too-large `Δ` erases information. A better implementation is:

```
χ²(θ) = Σ_i [ (y_obs,i − y_model,i(θ))² / σ_exp,i² ] · w( |y_obs,i − y_model,i(θ)| / Δ_i )
```

with `w(r)=1` for `r ≤ 1` and a smooth penalty beyond (or a flat plateau plus a weak
outer quadratic), and with `Δ_i` taken from the **model-family ensemble spread** rather than
an arbitrary number. Report the result for several `Δ` choices (sensitivity analysis), and
never silently choose the one that gives the tightest interval.

### 2.3 Wilks' theorem is asymptotic and assumes regularity

`Δχ² ≤ 1` (68%) / `≤ 4` (95%) for one parameter holds asymptotically for a regular,
nested, correctly specified likelihood, with the nuisance parameters in the interior and no
boundary/identifiability issues. Stellar fits violate several assumptions:

- boundaries (age > 0, mass > 0, `A_V ≥ 0`, stage transitions);
- multimodality and near-degeneracies (age–mass–`[Fe/H]`, RGB vs RC);
- grid discreteness/interpolation error;
- model misspecification (the "true" model is not in the grid).

**Therefore:** calibrate the threshold by **Monte Carlo fake-star experiments through the
exact pipeline** (inject known parameters, sample observables with realistic errors and
model-family variations, run the profile scan, measure the coverage of `Δχ²` thresholds),
and report the calibrated values. Use `Δχ²` as a diagnostic, not as an axiom. Where the
profile is multimodal, report all modes and the look-elsewhere/CLs-like penalty.

### 2.4 The prior-volume critique applies to our labels too

Our labels come from StarHorse/isochrone pipelines with their own priors (IMF, SFR, age
prior, extinction prior). The ML model inherits those choices. A profile-likelihood physical
track is a useful **cross-check of the label scale**, but it does not remove the label prior;
the correction-learning/hierarchical teacher model in [09](09-age-data-and-systematics.md) is
still required.

---

## 3. Assessment of the three conformal-with-theory-error methods

### Method 1 — adaptive variance scaling (`σ_exp + γ σ_theo`)

Useful and implementable; this is the closest to a principled compromise. Caveats:

- `σ_theo(x)` must come from an **independent model ensemble**, not from the same grid used
  to train the predictor; otherwise the score is self-referential.
- `γ` fitted on the calibration set and then used on the test set breaks exchangeability;
  fit `γ` by **cross-fitting/nested calibration** (e.g. split calibration into folds; choose
  `γ` on one fold, evaluate on another), or treat the result as a heuristic and report the
  sensitivity to `γ`.
- The resulting intervals are **locally adaptive** and can be wider in theoretically volatile
  regions (subgiant, RC), which is desirable; but conditional coverage is not guaranteed.
- Report empirical coverage by stage/tier and compare with the theory envelope.

### Method 2 — Mondrian by evolutionary phase

Statistically the cleanest of the three **if** (a) strata are pre-registered, (b) calibration
stars are exchangeable within a stratum, and (c) each stratum has enough calibration stars.
Our data: `n=141` at FeH<−2 overall; per-stage tail strata can be tiny. Over-stratifying
produces unstable intervals. Recommended: stage strata (MS/TO/SGB/RGB/RC/AGB) with a
minimum `n` (e.g. 100–200) and **pooling adjacent stages** when below; report the achievable
discrete coverage.

### Method 3 — synthetic cross-grid calibration

This is the weakest as a coverage claim. Comparing Grid Base to a perturbed grid and taking
the 1−ε quantile of the discrepancy yields a **theory-shift envelope**, not a conformal
guarantee on real stars: synthetic and real data are not exchangeable, and the perturbation
family is not a random sample from the true model space. It is nevertheless valuable as:

- a **conservative theory envelope** to add to the width;
- a sensitivity study ("if the true physics lies anywhere in the tested family, the interval
  must be at least this wide");
- a way to define `σ_theo(x)` for Method 1.

Do not label its output as "95% coverage of the truth".

### Summary table

| Theoretical error type | Recommended use |
|---|---|
| continuous/region-dependent | Method 1 with independent `σ_theo` + cross-fitted `γ`; report empirical coverage |
| phase-dependent | Method 2 (pre-registered stage strata, minimum n, pooling) |
| unknown/large model spread | Method 3 as a **theory envelope/sensitivity**, not coverage |

---

## 4. Recommended architecture: two tracks + one envelope

### Track A — ML predictive product (all Gaia)

- 6D joint density `p(Teff, logg, FeH, α, distance, age | x)` with block structure
  ([TRAINING_PLAN](TRAINING_PLAN.md) §3, [08](08-losses-age-distance-alpha.md)).
- Empirical calibration: Mondrian by stage/tier + grouped conformal + joint rank-score
  regions; per-star samples and covariance/copula published.
- Theory envelope: `σ_theo(x)` from model-family ensembles, added as a declared width and
  reported separately from the empirical conformal interval.
- OOD/model-rejection score.

### Track B — physical profile-likelihood inference (anchors/validation)

- Forward model: isochrone grid (MIST/GARSTEC/PARSEC/BASTI) + asteroseismic constraints
  (`ν_max`, `Δν`, individual frequencies where available) + spectroscopy (`Teff`, `logg`,
  `[Fe/H]`, `α`) + Gaia astrometry/photometry + extinction (3D dust when available).
- Rfit-like bounded theory nuisance; profile over mass, `[Fe/H]`, `α`, `A_V`, stage; compute
  1D/2D profile likelihood ratios; calibrate thresholds by fake-star simulations.
- Use on APOKASC3, Willett, clusters, and stars with seismic/spectroscopic data; compare to
  Track A with paired plots; publish both.
- Optional emulator: train a surrogate on grid simulations to extend Track B to larger
  samples, but never present the surrogate as a substitute for the full scan without a
  validation certificate.

### Cross-check governance (CKMfitter/UTfit pattern)

- The two tracks are independent; disagreements are **systematics diagnostics**, not
  averaged.
- A public table: Track A interval, Track B profile interval, theory envelope, model-rejection
  flag, and a "consistent / tension" marker.
- No claim that one is "more correct"; report both and the comparison.

### Correlated/marginalized two-parameter products

For each pair (e.g. `age–mass`, `distance–A_V`, `Teff–logg`, `FeH–α`):

- Track A: 2D marginal contours from joint samples; conditional quantiles; correlation
  matrix/copula; joint rank-score region with finite-sample coverage.
- Track B: 2D profile-likelihood contours at calibrated thresholds.
- Use **profile** for "what values are allowed after removing nuisance parameters" and
  **marginal** for "what the predictive distribution says"; label them explicitly.
- Publish per-star samples (or a compressed covariance + marginals) so users can marginalize
  to any subset of the six without re-running the model.

---

## 5. Concrete implementation plan (incremental, no big-bang)

| Step | Work | Cost | Gate |
|---|---|---|---|
| R1 | Add stage/tier Mondrian strata + cross-fitted theory-inflated scores to the reporting package; `σ_theo` from a model-family ensemble; report empirical vs theory envelopes separately | CPU (reporting) | synthetic coverage by stratum; no over-stratification at n<100 |
| R2 | Profile-likelihood cross-check on anchors: isochrone forward model + seismic/spectroscopy likelihood; 1D/2D profiles for age, mass, distance, `A_V`; compare to Track A | CPU/GPU moderate | fake-star calibration of Δχ² thresholds; cluster holdout |
| R3 | Model-rejection/OOD score for Track A; publish `χ²`-like flags | head-only | injected OOD test |
| R4 | Two-track release table and plots (ML interval vs profile interval vs theory envelope) | reporting | independent verification |
| R5 | Emulator/surrogate of Track B for larger samples (optional) | GPU | surrogate certification against full scans |

**Start with R1 and R2**, not with a full 6D Rfit grid. R1 is cheap and immediately improves
the trustworthiness of the current 6D product; R2 is the real scientific cross-check and
also gives the anchor ages/corrections needed by [09](09-age-data-and-systematics.md).

---

## 6. What to tell users (uncertainty contract)

For each of the six parameters and for pairs:

- a **predictive interval** with empirical (Mondrian/grouped conformal) calibration and a
  declared **theory envelope**;
- a **profile-likelihood interval/contour** where Track B is available;
- a joint **sample set or covariance** for correlations;
- flags: `tier`, `stage`, `model_rejection`, `theory_envelope_flag`, `anchor_scale`;
- no claim of "physical truth coverage"; the theory envelope is a sensitivity bound over the
  tested model family, not a posterior.

This is more honest than a single "95% CI" and matches the CKMfitter/UTfit expectation that
users see both the statistical and theoretical error content.

---

## 7. Risks

- **Compute:** full grid profiling for 1.8B stars is infeasible; restrict Track B to anchors
  and rich-data stars, use an emulator elsewhere only after certification.
- **Model family:** isochrone grids differ; the theory envelope can dominate the error and
  must be reported, not hidden.
- **Wilks:** simulation-calibrate thresholds; do not assume Gaussian asymptotics.
- **Conformal theory methods:** Method 3 is not real-data coverage; Method 1 needs
  cross-fitting; Method 2 needs enough stars per stratum.
- **Label priors:** Track B does not remove the priors in the survey labels; the hierarchical
  teacher model remains necessary.
- **Selection:** both tracks are affected by the same selection function; report it.
