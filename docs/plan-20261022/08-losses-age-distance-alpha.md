# Prediction losses for distance, age and alpha

Detailed specification for the three new target heads in
[TRAINING_PLAN.md](TRAINING_PLAN.md) §5–§6. Read with
[05-heads-losses-metrics.md](05-heads-losses-metrics.md) and
[07-inputs-architecture.md](07-inputs-architecture.md).

**Statistical framing (applies to all three).** We model the **observed-label
predictive distribution**

```
p_obs(m | x) = ∫ p_θ(y | x) p(m | y, R) dy ,
```

where `y` is the physical/latent target, `m` the survey measurement, and `R` the
measurement-error model. Training minimizes `−log p_obs`; evaluation scores the same
object. `p_θ(y|x)` is **not** a posterior over truth and is not identified from
`p_obs` without `R`. Every reported number carries `density_kind ∈ {intrinsic,
observed_label}`; no claim of "physical truth" is made. `R` is built from quoted
errors plus a **fitted** survey-specific inflation `κ` (never an invented floor);
`κ` is estimated on the fit role only.

---

## 0. Common notation and loss composition

| Symbol | Meaning |
|---|---|
| `x_i` | 134 photometry+XP features (T0), plus optionally 22 context features (T1) for distance/age |
| `y_i` | latent target (per-head; see below) |
| `m_i, R_i` | measured label(s) and measurement covariance (diagonal unless stated) |
| `θ` | network parameters (shared encoder + head) |
| `N_obs` | number of observed label coordinates used in the denominator (never rows when partial) |

Total loss for one optimizer step:

```
L = L_obs + λ_rec · L_recon + λ_spec · L_spec
L_obs = Σ_targets  w_t · L_t(observed labels of target t)
```

Each `L_t` has its own observed-count denominator; `w_t` are declared fixed weights
(uncertainty weighting is an ablation). `L_recon` uses the masked reconstruction loss
over the 134 scored observable slots; `λ_rec` starts at 0.01 (0.03 only in matched
arms). `L_spec` is the relative spectral auxiliary (0.1 where enabled). No target is
imputed; missing labels are omitted from `L_obs` (partial-label marginalization where
the head supports it).

---

## 1. Distance

### 1.1 Coordinate and measurement model

Target: `s = log(d / kpc)` so `d = e^s > 0` automatically. Gaia parallax in mas is
`π(s) = e^{-s}` because `1 kpc ⟺ 1 mas` (`1 pc ⟺ 1000 mas`; this is the only unit
conversion allowed and is asserted in tests). **Measured parallax is never inverted.**

Observation model per star:

```
π_obs,i = z_i + e^{-s_i} + ε_i ,     ε_i ~ N(0, σ_eff,i²)
σ_eff,i² = κ_π² · σ_π,quoted,i² + σ_zp,i² + σ_floor,i²
```

- `σ_π,quoted` is the formal Gaia error in mas (verify mas vs µas against one raw row
  before the first launch).
- `z_i` is the Gaia parallax zero point: a declared constant or a published
  magnitude/colour-dependent recipe (`z0 + z_G(G) + z_C(BP−RP)`); if any part is fitted,
  fit it on the fit role only with a penalty and never on verification references.
- `κ_π` is fitted from residual excess variance on the fit role (e.g. robust estimate from
  high-S/N stars); `σ_floor` is used only if declared as a model term and reported.
- Negative and zero measured parallaxes are **retained**; eligibility is only
  `finite(π_obs) & finite(σ_eff) & σ_eff > 0`. No S/N cut, no positivity cut, no error
  floor invented for convenience.

### 1.2 Predictive distribution

Three interchangeable head families over `s`:

1. **Gaussian:** `p_θ(s|x) = N(μ_θ(x), σ_θ²(x))`, with `log σ` output.
2. **MDN:** `p_θ(s|x) = Σ_{k=1}^K w_k N(μ_k, σ_k²)`; mask/logsumexp stable; K=4 first.
3. **Flow:** normalized conditional density `p_θ(s|x)` (NSF/1-D flow) only after the W0
   certificate; fixed saved QMC nodes at evaluation.

Context: the feature-only head receives the 256-d latent. The context-conditioned head
receives `[z ‖ c]` with `c` the 22-feature T1 vector, but **`π` is input XOR likelihood
XOR target**: if `π` is used in the likelihood (all training below), it must not also be
a context input for the same arm, except the explicitly named measurement-assisted arm.

### 1.3 Primary loss — marginalized over distance

```
L_dist = − (1/N_obs) Σ_i log ∫ p_θ(s | x_i) · N(π_obs,i | z_i + e^{-s}, σ_eff,i) ds
```

**Gaussian head — two fixed quadrature rules.** A literal "GH-32 in `s`" rule is
**not adequate** for bright/near stars: with `σ_s = 0.3`, `h* = e^{−μ} = 1` mas it
errs by 1.7e-3 at `σ_eff = 0.1 h*`, 1.6e-4 at `0.14 h*`, and diverges (NLL 29.1 vs
−0.285) at `0.01 h*` because fixed `s` nodes cannot resolve a narrow likelihood in
`h = e^{−s}`. Implement **two fixed rules, both saved with the certificate**:

1. **Prior-space GH in `s`** (`rule="gh_prior"`): `s = μ + σ_s u`,
   ```
   ∫ = (1/√π) Σ_k w_k · N(π_obs | z + e^{−(μ+σ_s u_k)}, σ_eff)
   ```
   used only where the prior can resolve the likelihood
   (`σ_eff / (σ_s h*)` above a declared threshold).
2. **Residual-space GH in `h`** (`rule="gh_residual"`, default via `rule="auto"`):
   GH nodes in the standardized residual `h_k = (π_obs − z) − σ_eff r_k`,
   `s_k = −log h_k`, with exact Jacobian `1/h_k`:
   ```
   ∫ = (1/√π) Σ_k w_k · p_s(s_k) · (1/h_k) · N(π_obs | z + h_k, σ_eff) / N(r_k | 0,1)   [normalized form]
   ```
   This rule is exact in the narrow-error limit (`σ_eff → 0`: 1.9e-11 at
   `σ_eff = 1e-6`) and representable while `σ_eff < h*/6.63`; per-row `rule="auto"`
   selects it when valid and falls back to `gh_prior` otherwise.
   **Do not** write the naive "prior density × likelihood" formula without the
   Jacobian: `∫ p(s) N(π | z+e^{−s}, σ) ds = E_{s∼p}[N(...)]`; a naive version
   double-counts the prior density.

**Mandatory certificate fields:** `quadrature.rule`, `residual_rule_rows`,
`min_prior_resolution = min σ_eff/(σ_s h*)` over prior rows, and the GH16→GH256
convergence check run **on the actual `σ_eff/h*` population** (prior-branch error at
the switch point was measured at 5e-9 for `σ_s=0.2`, 5e-5 for 0.3, 8e-3 for 0.5,
0.27 for 0.8 — flag prior rows where the ratio is below the declared threshold).
A "GH-32" statement alone is not sufficient.

**MDN head.** Apply the same GH rule to every component and combine with
`logsumexp(log w_k + log integral_k)`.

**Flow head.** `L_dist ≈ −(1/N_obs) Σ_i log [ (1/M) Σ_{m=1}^M N(π_obs,i |
z_i + e^{−s_m}, σ_eff,i) ]` with `s_m` fixed QMC/antithetic draws from `p_θ(·|x_i)`.
**Gate:** `|NLL_M64 − NLL_M512| / |NLL_M512| ≤ 1e-3` declared before the run, M≥64;
otherwise the flow distance head is not admitted.

### 1.4 Controls (declared, not substitutes)

- **Delta-method control (first order in `σ_s`):**

  ```
  r_i = (π_obs,i − z_i − e^{−μ_i}) / σ_tot,i
  σ_tot,i² = σ_eff,i² + (e^{−μ_i} σ_s,i)²
  L_delta = (1/N_obs) Σ_i [ 0.5 r_i² + log σ_tot,i + 0.5 log(2π) ]
  ```

  Valid only when `e^{−μ} σ_s ≪ σ_eff`; compare with the quadrature loss in the
  narrow-error limit. It is a **control**, never the final loss.
- **Point control:** the existing `signed_log_distance_nll` (mean `e^{−μ}`, no `σ_s`
  term) for the point-only arm.

### 1.5 Feature-only and measurement-assisted products

These are **different statistical objects** and are named separately in every table:

1. **Feature-only predictive PDF** `p_θ(s|x)`, trained by `L_dist` with **no π in the
   input**. Point = median/mean of `p_θ`; intervals from its quantiles.
   This is population-dependent (it inherits the training selection and the model), so it
   is called a **predictive distribution**, not a physical distance posterior.
2. **Measurement-assisted confidence set.** With the same `p_θ` as a learned population
   prior, `p(s | π_obs, x) ∝ p_θ(s|x) N(π_obs | z+e^{−s}, σ_eff)` may be reported only
   as a **posterior under a learned prior**. For a frequentist product, report the
   **profile-likelihood confidence set**

   ```
   C_α = { s : −2 [ ℓ(s) − ℓ(ŝ) ] ≤ χ²₁(1−α) } ,
   ℓ(s) = log N(π_obs | z + e^{−s}, σ_eff)
   ```

   and report `frac_ci_empty`, `frac_ci_upper_unbounded`; never cosmetically cap an
   unbounded interval.
3. **Forbidden rename:** `p_θ(s|x) N(π|...) / Z` is **not** prior-free frequentist
   inference; it is a posterior under a learned prior.

At inference, high-S/N stars keep an **unchanged-measurement fallback** (raw Gaia
parallax) and the assisted product must be at least as good as that fallback on the
declared high-S/N bin; otherwise the fallback ships.

### 1.6 Missingness, selection, and auxiliary labels

- Partial labels are handled per row: a star enters `L_dist` iff `π_obs` and `σ_π` are
  finite and positive; all other rows are omitted with `N_obs` updated.
- Selection is declared: the eligible population is "Gaia parallax with finite positive
  formal error"; report counts, G/quality/parallax-S/N compositions, and the fraction of
  the benchmark excluded. No inverse-probability weighting unless a selection model is
  fitted and validated.
- **StarHorse distances are not training targets** for this head. If used at all, they
  are a separate auxiliary noisy-label head with an asymmetric likelihood and are
  reported as model-dependent; they never replace the raw-parallax loss.
- 3-D dust / `A_V` is not part of this loss this round (no 3-D map ingested); 2-D SFD
  `E(B−V)` may enter only as declared context and cannot be treated as `A_V(d)`.

### 1.7 Evaluation

- **Parallax space (primary):** PIT of `π_obs` under the predictive `p_obs(π|x)` obtained
  by convolving `p_θ`; 68/90/95 coverage; CRPS/NLL in mas; bias.
- **Distance space (secondary):** MAE/bias/fractional error **only** against independent
  references — Hipparcos (nearby **negative control**: raw Gaia 0.960 mas; previous
  frozen ridge 3.23 mas), clusters/seismic later. Never score against StarHorse or any
  catalogue built from the same Gaia parallax.
- Bins: `π/σ` (precision), G, E(B−V), sky/HEALPix, **not** bins of inverted noisy `π`.
- Report the prior-vs-likelihood dominance: e.g. the width ratio `σ_eff / (e^{−μ}σ_s)` or
  the likelihood-to-prior information fraction per bin, so flat-likelihood stars are not
  presented as data-driven.

### 1.8 Required numerical tests before launch

1. Narrow-error limit: `σ_eff → 0` gives `L_dist → −log p_θ(s*) + log π_obs` with
   `s* = log(1/π_obs)`; agreement to 1e-8 at `σ_eff = 1e-6`.
2. `float64` `gradcheck` on 3 synthetic rows through the GH integral.
3. Negative parallax (`π_obs = −5 mas`): finite loss, monotone decreasing in `s` beyond
   the mode, argmin at large `d`.
4. Empty/all-invalid batch returns a differentiable zero with `N_obs=0`.
5. GH16→GH256 relative NLL convergence `< 1e-3` on a synthetic Gaussian latent.
6. 1-D Gaussian latent vs `scipy.integrate.quad` to 1e-10 (float64).
7. Unit identity: `s=0 ⟺ π=1 mas ⟺ d=1 kpc`; `s=ln 10 ⟺ 0.1 mas ⟺ 10 kpc`.
8. Assert measured `π` never enters `1/x` or `exp` except through `e^{−s}` in the
   likelihood; no `1/π_obs` anywhere.
9. Batch-1 vs batch-128 float64 identical (same rows/order).

---

## 2. Age

### 2.1 Coordinate and labels

Target: `τ = log(age / Gyr)` (natural log; convert with `age = e^τ`). Positivity is
automatic; **never** fit a Gaussian directly on Gyr (the age distribution is
right-skewed and asymmetric).

Labels: StarHorse posterior percentiles `age_5, age_16, age_50, age_84, age_95` in Gyr
(APOGEE/GALAH). Transform each to `τ_p = log(age_p / Gyr)`. These are **model- and
parallax-conditioned** posterior percentiles, not truth; `τ_50` is the central label and
`τ_16 < τ_50 < τ_84` encode asymmetry. Records with invalid/non-monotone percentiles are
dropped with a printed count and `null_reason`.

### 2.2 Primary loss — multi-quantile in `τ`

Predict `q_p(x)` for `p ∈ {0.05, 0.16, 0.50, 0.84, 0.95}` (drop 0.05/0.95 if the fit
cohort is too small). Pinball loss per percentile, using the **same-percentile label**
as its target:

```
L_age = Σ_{i,p} w_p · ρ_p( τ_p,i − q_p(x_i) )      [ρ_p(r) = max(p r, (p−1) r)]
        + λ_cross Σ_{i,p<p'} max(0, q_p − q_p')²
```

- `w_p` equal by default; `w_50` may dominate in a declared ablation. The exact
  per-percentile targets are noisy quantile labels; this is a quantile-regression
  approximation and must be stated in the certificate (`label_model:
  starhorse_quantile_targets`).
- Crossover penalty `λ_cross` enforces monotonicity; after a run, quantiles are
  rearranged monotonically before reporting (report the rearrangement rate).
- **Do not** substitute `τ_50` as the target for all quantiles; that fits a point with a
  distributional loss and destroys asymmetry.

### 2.3 Controls

- **Split-normal likelihood** (asymmetric label model): with
  `σ_l = τ_50 − τ_16`, `σ_r = τ_84 − τ_50`:
  ```
  p(τ | τ_50, σ_l, σ_r) = √(2/π)/(σ_l + σ_r)
                          · exp( −(τ − τ_50)² / (2 σ_{l,r}²) ) ,  τ ≶ τ_50
  L_split = −Σ_i log p( τ_i^{obs} | τ_50,i, σ_l,i, σ_r,i )    (τ_i^{obs} = τ_50,i)
  ```
  Its 16/84 quantiles are only approximately `τ_50 ∓ σ`; report the approximation error
  and do not claim exact quantile reproduction.
- **Heteroscedastic Gaussian on `τ`:** `σ_label² = ((τ_84−τ_16)/2)²`,
  `σ_tot² = σ_θ² + κ_age² σ_label²`, loss `0.5((τ_50−μ)/σ_tot)² + log σ_tot`. Ignores
  asymmetry; a control only.
- **Flow/MDN on `τ`:** `p_θ(τ|x)` trained with MC convolution over the split-normal label
  distribution; admissible only after W0 and a convergence gate; motivated by possible
  young/old bimodality.

### 2.4 Label-noise / errors-in-variables

Fitting network quantiles to StarHorse quantiles treats the label distribution as known.
The quoted spread is itself uncertain, so predicted widths can be **too narrow**
(errors-in-variables on the spread). Mitigation: a fitted inflation `κ_age ≥ 1` on the
label widths, estimated from independent repeats or seismic reference residuals on the
fit role; report `κ_age` in the certificate. No physical ensemble spread may be used as a
substitute. Define the estimand as `p(age_obs | x)` and state calibration is only as good
as the StarHorse labels.

### 2.5 Multi-teacher correction learning (new data; full note in [09](09-age-data-and-systematics.md))

The current labels are **MSTO/SGB** StarHorse ages (APOGEE DR17/GALAH DR3), while the new
giant labels (APOKASC-3, Willett 2026 Kepler/K2/TESS) are a different evolutionary regime
and a different age scale. Treat each catalogue/pipeline `s` as a teacher with its own bias
and scale; never merge their values:

```
t_i ~ p_pop(tau | x_i; psi)                         # learned population prior / flow
y_{i,s} | t_i, x_i ~ N( a_s(x_i) + b_s(x_i) t_i,    # teacher offset + scale
                       sigma_s^2(x_i) + kappa_s^2 sigma_quoted,is^2 )
L_age = sum_i log sum/ integrate over t_i  +  cluster coeval term
```

- `a_s(x), b_s(x)` are smooth functions of Teff/logg/FeH/alpha/stage (small MLPs or
  stage-binned affine); `kappa_s` is fitted from overlap scatter (APOKASC3 vs Willett
  measured 0.73–1.25 Gyr NMAD).
- **Bridging**: zero overlap exists between StarHorse MSTO/SGB and APOKASC3 giants; use
  **cluster coevality** (members share one age) and globular/open-cluster anchors to
  identify the relative scale.
- **Predict-time correction layer** `tau_corr = tau_base + g_phi(tau_base, x, target_scale)`
  (monotone, small) is fitted on anchors/new-survey overlaps; when a new catalogue arrives,
  re-fit only `g_phi` (few-shot), not the full model. This is the practical "correction
  learning as we predict".
- **Uncertainty**: `Var(age) = Var_model + Var_within_teacher + Var_between_teacher +
  Var_scale`. Conformal within a teacher cannot cover teacher/scale systematics; report the
  between-teacher spread as a **systematic envelope**, not truth coverage.
- **Anchor policy**: APOKASC3/Willett/cluster/LEGACY IDs are held out of fit/tune/cal;
  leave-one-teacher-out and cluster-holdout evaluation; "age on a named scale", never truth.

### 2.6 External anchors (held out)

- **APOKASC3** `table4.dat` (15,808 stars) is **entirely held out** of fit/tune/cal and
  used only as an external comparison.
- Kepler LEGACY 66 stars: descriptive only (six correlated pipelines, not truth).
- Cluster memberships/ages: later external validation; do not train on them in this round
  unless an explicit cross-validation policy is declared.
- All reference IDs are banned from fit/tune; report the ban in the manifest.

### 2.7 Evaluation and metrics

- Bias/MAE in **Gyr and log Gyr**, PIT and coverage at 0.68/0.90 in `τ`, interval width,
  CRPS/NLL where a full distribution exists.
- **Age > 10 Gyr slice** with `n`; n<20 descriptive only.
- Seismic external comparison: predicted vs reference age with asymmetric errors, per
  pipeline; never average the six correlated pipelines into "truth".
- No "true age" claim; no age PDF at ages where the training labels are absent.

### 2.7 Required numerical tests

1. Positivity/round-trip: `τ → e^τ → log` stable; non-finite/negative ages rejected.
2. Split-normal normalizes to 1 (quadrature), its 16/84 quantiles compared with
   `τ_50 ∓ σ` and the approximation error reported.
3. Multi-quantile recovers exact Gaussian quantiles on synthetic data; no crossings after
   the penalty; rearrangement rate = 0.
4. `float64` gradcheck through pinball/split-normal losses.
5. Batch-1 vs batch-128 identical; missing `τ_16/τ_84` → drop with printed count.
6. `κ_age` fitted on synthetic excess-spread data recovers the true inflation.

---

## 3. Alpha

### 3.1 Observables and source tagging

Alpha is **not one quantity** across surveys:

| Source tag | Observable | Notes |
|---|---|---|
| `galah` | `[α/Fe]` (elemental) | anchor; 550,854 native rows ingested |
| `apogee` | `[α/M]` | different denominator; must not be renamed `[α/Fe]` |
| `gaia_rvs` | GSP-Spec `alphafe` | DR3/DR4; calibration/flags required |
| `lamost` | MRS α (where available) | scale audit required |

The head predicts `α_s` per source tag; the shared latent is the same but the output
layer and noise model are source-specific. No source is averaged into a universal truth.

### 3.2 Calibration / hierarchical label model

For source `o`:

```
α_obs,o = a_o + b_o · α_true + ε_o ,     ε_o ~ N(0, κ_o² σ_{quoted,o}²)
```

- `(a_o, b_o)` are fitted on **overlap stars** (stars with both source `o` and the
  anchor source), using fit-role rows only; `κ_o` is fitted from paired scatter. The
  DESI−APOGEE scatter (0.0746 dex vs quoted 0.0102 ⇒ `κ_DESI ≈ 7.3`) is the template.
- **Gauge:** only relative offsets/scales are identifiable. Fix the anchor source
  `a=0, b=1` (recommended: GALAH) or impose `Σ n_o a_o = 0`; document the choice. Never
  report an absolute α zero point.
- **Overlap scarcity:** the metal-poor tail has essentially no overlap pairs; the
  calibration there is an extrapolation. Report the overlap counts per regime and the
  extrapolation distance; no α claim below the calibrated domain.

### 3.3 Losses

Single-source, source-tagged:

```
L_alpha,o = (1/N_o) Σ_{i∈o} [ 0.5 (α_obs,i − a_o − b_o μ_α(x_i))² / σ_tot²
                             + 0.5 log σ_tot² ]
σ_tot² = b_o² σ_α²(x_i) + κ_o² σ_quoted,i²
```

- `α_obs` is the calibrated **observed** value; if the network is trained directly on
  `α_obs`, then `a_o,b_o` must be applied to the prediction (not to the label) to avoid
  double calibration. State the direction explicitly in the manifest.
- Robust/quantile variant for outlier flags; never silently clip.
- **Joint (FeH, α) model:** 2-D Gaussian/MDN/flow over `(FeH, α)` (or conditional
  `α | FeH`) so the FeH–α degeneracy lives in the covariance; marginal α MAE alone is
  misleading. The core 3×3 Teff/logg/FeH block remains separate (block-structured joint
  head); a cross-block is added only when n≥1,000 co-occurring stars.
- If multiple sources observe the same star, one latent `α_true` with source-specific
  likelihood is the preferred hierarchical form; amortized variational inference is an
  option only if the simpler calibrated-source loss is insufficient.

### 3.4 Gates before any α number

1. A native α column per source with units, quality flags and provenance is ingested.
2. Overlap offsets/scatter measured with `n`; `κ_o` fitted and reported.
3. The source-tagged head beats a single-α baseline on held-out **overlap-consistent**
   stars (paired CI).
4. Calibration coverage on the cal split (marginal and by source).
5. 2-D (FeH, α) joint level-set coverage.
Until all five, F05 remains "NOT RUN"; no α number is published.

### 3.5 Metrics and tests

- Per-source MAE/bias, cross-source consistency on overlaps, coverage, CRPS/NLL where a
  density exists, α-poor/α-rich branch separation, joint (FeH, α) coverage.
- Tests: synthetic affine recovery `(a_o,b_o,κ_o)`; gauge invariance under global
  shift/scale; source-tag routing; non-finite/absent labels dropped with counts; joint
  correlation recovery.

---

## 4. Shared calibration and metric contract

For every head the campaign emits (see [05](05-heads-losses-metrics.md) §7.1):

- `prediction-certificate.json`: run/source/library/checkpoint/cohort SHAs; labels,
  units, transforms; `density_kind`; `loss_denominator`; per-row Jacobian;
  `noise_model{kind, source_tagged, κ}`; `partial_label_policy`; conformal method/rank;
  integration nodes/seed/SHA; n per label; failed-verdict provenance.
- `metrics.csv`: point metrics, CRPS, NLL (same family and `density_kind` only), PIT,
  coverage marginal/Mondrian, interval score/width, grouped two-way bootstrap CI,
  multiplicity, `evidence_status`.
- `distance.csv`: mas and kpc metrics, `ci_kind(predictive|profile)`, reference type,
  `frac_ci_empty`, `frac_ci_upper_unbounded`.
- `age.csv`: Gyr and log-Gyr metrics, >10 Gyr slice, seismic reference.
- `joint.csv`: joint rank-score coverage (primary), Bonferroni box (secondary), region
  volume/level-set method, energy/variogram scores.

Conformal: exact finite-sample rank `q̂ = s_(⌈(n+1)(1−α)⌉)`; two-sided score
`max(q_lo−y, y−q_hi)` at one level or per-tail α/2; raise when the rank exceeds `n`.
Mondrian by pre-registered strata; report the discrete achievable coverage at small `n`.
Tail FeH<−2 always reported separately with `n`; n<20 descriptive only.

---

## 5. Implementation map

| Piece | Where |
|---|---|
| Masked Gaussian observed-subvector NLL | `src/masked_stellar_autoencoder/training/joint_prediction.py` (`masked_gaussian_measurement_nll`) — reuse, do not re-derive |
| Point log-distance NLL (control) | `joint_prediction.py` (`signed_log_distance_nll`) and `training/parallax_likelihood.py` |
| Full-cov/low-rank Gaussian | torchregress `MultivariateGaussianLoss`, `LowRankGaussianLoss` |
| MDN | torchregress `MDNLoss` / `MixtureDensityLoss` |
| Flows | torchregress `nflows` (NSF/cINN/MAF/BPF), after W0 + `8973bcd` |
| Quantile/expectile | torchregress `MultiQuantileLoss`/`MultiExpectileLoss`; MSA `MultiLabelHead` |
| SLS/SLSConformal | torchregress `SLSLoss`/`SLSConformal` |
| Distance head (new) | `training/distance_heads.py` (this round) |
| Age head (new) | `training/age_heads.py` (this round) |
| Alpha interface (new) | `training/alpha_heads.py` (this round) |
| Context conditioner (new) | `training/context_heads.py` (this round) |
| Metric contract | `src/masked_stellar_autoencoder/reporting/` (this round) |
| Runner integration | `scripts/run_joint_continuation.py` distance/age hooks; `scripts/run_multilabel_screen.py` density scoring |

**Gate:** no distance/age/α loss is quoted in a plan or report until its §1.8/§2.7/§3.5
tests pass and its head is wired into a runner with a certificate. The first
distance/age runs are gated on the W0 certificate and W3.0 plumbing; α remains an
interface until labels arrive.
