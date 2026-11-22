# Data-driven inference from survey catalogues (design note, 2026-11-22)

Scope: what information we can extract from Gaia-scale catalogues (spectra,
photometry, astrometry, kinematics, positions) **without prior physics**: no
galaxy model, no isochrones or atmospheres, no assumed population distributions.
Everything that is not an instrument fact is learned from the data, and every
estimate should improve as more stars (labelled or not) are added. We keep
competitive multi-target distributional predictions (fast, calibrated,
non-Gaussian) as a first-class product.

This note builds on `design-20261120-likelihood-inference.md` and on the
22 November results (S1 parallax ensemble, XP forward model run 2).

## 1. Ground rules

What counts as "no prior physics", operationally:

| allowed | not allowed |
|---|---|
| instrument facts: XP covariances, photometric and astrometric errors, the parallax zero point (Lindegren+21, an empirical calibration), survey-to-survey label maps learned from overlaps | Galactic structure models, potentials and orbits, isochrones, synthetic spectra, IMF or star-formation histories |
| emulators and heads learned from labelled stars | parametric population priors (Gaussian MDFs, exponential discs, ...) |
| population distributions *estimated* from the data, nonparametrically, with stated sampling assumptions | using a predictive distribution as if it were a measurement |
| geometric frame changes (equatorial to Galactic, heliocentric velocities); solar motion constants enter only as measured offsets | dust-map priors (SFD is used today only as an upper bound on E; flagged) |

The spectroscopic labels themselves are the one external input. They come from
model atmospheres, but we use them only as noisy, survey-calibrated
measurements on a common scale. They are never a functional form.

## 2. Three products, never mixed

Most of our bias problems came from using one product for another's job
(posterior shrinkage read as a measurement; balanced-sampling priors; free
survey offsets absorbing a head bias). We keep three products apart:

**A. Measurement (likelihood).** `L(theta | x_star) = p(x_star | theta)` up to a
constant, using only the star's own data.
- Implementations: the S1 ratio head and the XP forward model.
- Population-free, as long as it is evaluated inside its training support.
- Combines by multiplication across data sets (XP x photometry x parallax x RVS).
- Supports Neyman or profile intervals, and is the input to deconvolution (C2).
- It is never the best point estimate for a faint star; it is the unbiased input.

**B. Prediction (distribution).** `p(theta | x_star, context)` against a stated
reference population.
- The best accuracy and the best-calibrated individual intervals, but it depends
  on the population and on the selection.
- `context` can include kinematics and position (section 4). That makes B much
  better (for example, metal-poor purity) and also makes it circular for
  chemo-dynamics, so B is never used directly for population science.

**C. Population estimands.** Functionals of the joint distribution of (theta,
observables) in cells of observable space: MDF quantiles, metal-poor fractions,
mean [Fe/H] against v_phi, gradients, joint chemistry-kinematics histograms.
Two routes, which should agree:
- **C1, prediction-powered inference (PPI).** Any predictor, B included, on the
  full catalogue, rectified by the labelled stars. Valid intervals whatever the
  predictor's errors, under a sampling assumption about the labelled stars
  (section 5).
- **C2, deconvolution (empirical Bayes, g-modelling).** The population
  distribution `g(theta)` in a cell is estimated from the per-star likelihoods
  (A) by nonparametric maximum likelihood (Kiefer-Wolfowitz NPMLE: convex, on a
  grid, no parametric form, consistent as N grows). This is a frequentist
  estimator. The empirical-Bayes posteriors it implies give a B-type product
  whose "prior" is the cell's estimated population, with known frequentist
  behaviour and improving with N.

C1 needs representative labels (positivity); C2 needs a correct likelihood
(support, nuisances). Their agreement is our strongest end-to-end QA.

## 3. Information inventory

What each catalogue channel can tell us, from data alone:

| channel | observables | information (data-driven) | caveats |
|---|---|---|---|
| XP spectra | 2 x 55 coefficients + full covariance | Teff, log g, [Fe/H], E(B-V); weaker [alpha/Fe], carbon; peculiar spectra as residual outliers | the emulator or head must stay inside its training support; E-Teff-[Fe/H] degeneracy |
| Photometry | G, BP, RP, WISE, PS1, SDSS, SkyMapper (2MASS once re-matched) | SED shape, E, flux excess (binaries) | band-dependent systematics, learned scatter about 0.5 mag today (too large; see section 8) |
| Parallax | parallax, error, zero point | distance, hence absolute magnitude, hence log g given Teff via the learned M_b(theta) | useless faint and far (parallax/error < 3 for most halo giants) |
| Proper motion | mu_a*, mu_d, covariance | **reduced proper motion** H = G + 5 log10(mu) + 5: a luminosity-kinematics proxy needing no parallax, valid to G ~ 20; tangential velocity v_t = 4.74 mu / parallax when the parallax is good | H mixes luminosity and velocity; only meaningful jointly with colour |
| Radial velocity | Gaia RVS (~33M), surveys | full 3-D heliocentric velocity, (v_R, v_phi, v_z) by geometry | no orbits or actions (they need a potential) |
| Position | l, b, distance | spatial strata; the 3-D correlation of E between neighbours (an empirical dust field) | neighbours' E is population information, so it belongs in B or C, not A |
| Astrometric quality | RUWE, excess noise, ipd fractions, NSS | unresolved-binary flags (photometric-excess binaries bias log g and Teff) | |
| Variability | flux scatter, transits, Gaia variability classes | pulsators and eclipsing binaries (bad single-star likelihoods) | |
| Co-moving pairs and clusters | astrometry alone | **label-free replicates**: components should share [Fe/H] within the method's errors | the shared composition is the one physical assumption, used only for validation |

Three channels are under-used today and come with no physics:
- reduced proper motion, for faint stars;
- the astrometric and variability flags, as covariates that tell a head when its
  single-star model fails;
- co-moving pairs, as a scale-free precision test.

## 4. Kinematics and position as information

Including kinematics (H, v_t, (v_R, v_phi, v_z) where available) and position in
product B helps exactly where XP is weakest: faint stars, the metal-poor tail,
and the dwarf/giant degeneracy (H breaks it without a parallax). The cost is
circularity: B then encodes the correlations between chemistry and kinematics
present in the labelled set. Rules:

1. **A never sees kinematics or position.** The likelihood stays a property of
   the star's spectrum and photometry.
2. **B comes in two variants.** `B0` uses intrinsic data only, for individual
   catalogues and cross-matches. `B1` adds context, for targeting, selection of
   rare objects, and as a C1 predictor.
3. **Population science uses C1 or C2, never raw B.** PPI keeps B1's precision
   gain (a better predictor gives narrower rectified intervals) without
   inheriting its circularity, because the rectifier measures and removes the
   predictor's bias in each cell.

## 5. Prediction-powered inference with selected labels

PPI's validity rests on the labelled stars being a random sample of the
population being estimated. Ours are not: spectroscopic targeting selects on
magnitude, colour, sky position and, for the metal-poor catalogues, on
metallicity-sensitive photometry.

**Selection on observables (data-driven).** Let `s = 1` if a star is labelled.
We estimate the propensity `pi(x) = P(s = 1 | x)` with a classifier, labelled
stars against the parent catalogue, on observables x (G, colours, l, b,
parallax/error, H, SkyMapper u/v when present). Then:
- inverse-propensity-weighted PPI (rectifier terms weighted by `1/pi`) gives
  valid intervals for parent-population estimands, provided selection depends
  only on x (missing at random given x);
- weighted conformal (Tibshirani et al. 2019; torchregress
  `NonExchangeableConformalRegressor`) with weights `(1 - pi)/pi` gives B's
  intervals coverage on the parent population rather than on the labelled stars.

**Positivity.** Where `pi(x)` is about 0 there is no label information. The
answer there is "unsupported", reported with the effective sample size per cell,
never an extrapolation. Faint metal-poor halo giants (G > 17) are the obvious
case.

**Selection on unobservables.** Surveys preselected on photometry we do not have
(Pristine CaHK, objective-prism metal-poor candidates) violate missing at
random. For C1, the rectifier uses only surveys whose targeting is a function
of our observables (APOGEE, GALAH, LAMOST, Gaia-ESO selection functions are
documented in magnitudes and colours). The targeted metal-poor catalogues still
train and validate A and B, but stay out of the PPI rectifier unless their
targeting photometry is in x.

**Label noise.** PPI estimates functionals of the *label* distribution. Means are
unaffected (labels on the common scale, zero-mean noise). Quantiles and tail
fractions are broadened by label noise; for those, C2 (deconvolution) or
rectifiers corrected for label noise are needed.

Estimands in scope: mean and quantiles of [Fe/H] per cell (MDFs); the fraction
with [Fe/H] < -2 or -2.5 per cell; the slope of [Fe/H] against R, z or v_phi
(PPI-OLS); 2-D histograms of chemistry against kinematics with simultaneous
intervals.

torchregress already has `ppi_mean_ci`, `ppi_quantile_ci`, `ppi_ols_ci` and
`ppi_pp_mean_ci`. Missing upstream: propensity weights (IPW-PPI), stratified
PPI over cells with simultaneous intervals, and a positivity diagnostic.

## 6. Competitive multi-target distributional predictions (product B)

Requirements: joint (Teff, log g, [Fe/H]), later with E and [alpha/Fe];
amortized (one forward pass per star); non-Gaussian (dwarf/giant bimodality at
the faint end, metal-poor tails, boundaries); calibrated marginally and
conditionally by stratum; fast summaries for 2 x 10^8 stars.

**Heads (all in torchregress, on the frozen v10 latent, then fine-tuned):**
1. A conditional normalizing flow: a small rational-quadratic spline flow on 3-D
   (`NormalizingFlowLoss`). It gives exact densities, multimodality and sampling.
2. A mixture of multivariate Student-t (`student_t_mixture`): a cheaper baseline
   with closed-form summaries.
3. The current single Student-t, as reference.

Each is run as a K = 5 ensemble (mixture of the members), since ensembling was
the largest single gain in S1.

**Training target and reference population:**
- **Label noise:** fit `p(y_obs | x) = integral of p(theta | x) N(y_obs | theta,
  Sigma_label) dtheta` (errors-in-variables; torchregress `eiv`,
  `noise_aware_density`), so that intervals cover the true parameter, not the
  noisy label. Today our coverage is of labels.
- **Reference population:** stated explicitly. The default is the parent
  catalogue, reached by propensity weighting of the labelled stars (section 5).
  This replaces balanced [Fe/H] sampling, which silently imposes a sampling
  prior. Balanced sampling can stay as a training-efficiency device if the
  weights undo it.

**Calibration:**
- split-conformal level sets of `-log p` (`ConformalizedDensity`), so that
  coverage is exact marginally;
- Mondrian (stratified) conformal over G and colour strata and over predicted
  [Fe/H] bins, for coverage within each stratum;
- propensity-weighted conformal for the parent population.

**Scores (all exist in torchregress):**
- log score (NLL) on the true-parameter proxy;
- energy and variogram scores (multivariate, proper);
- PIT per marginal;
- HPD-coverage curves (`highest_posterior_density_coverage`) per stratum;
- bias and NMAD of the mode and median per [Fe/H] x G bin (the existing tables).

**Outputs at scale:** per star, the mode or modes, per-parameter 16/50/84
quantiles, the 68 % and 95 % HPD membership test (via the flow, on demand), and a
bimodality flag (mass of the secondary mode). The flow itself is tiny, so the
latent plus head weights reproduce any density on demand; we do not store
samples.

**What "competitive" means, measured:**
- Against the current Student-t head and the S1 ratio head (on frozen latents):
  better log and energy scores overall.
- No worse bias or NMAD in any [Fe/H] x G bin.
- HPD coverage within ±0.03 of nominal in every stratum, including G > 16 and
  [Fe/H] < -2.5.
- At least 10^5 stars/s per GPU for summaries.

## 7. Improving with more stars

- **Representation:** masked SSL on all of Gaia, including the low-S/N stars that
  never got spectra. The latent improves with N; heads are cheap to refit.
- **Predictions:** new labelled surveys enter through the anchored survey maps.
  Conformal guarantees hold at any calibration size; strata get finer as n
  grows.
- **Population estimands:** PPI interval width shrinks as `residual sd /
  sqrt(n_lab) + sd / sqrt(N)`. A better predictor (B1, with kinematics) lowers
  the residual sd, and more catalogue stars lower the second term. The NPMLE in
  C2 converges as N grows. Cell resolution can follow the effective sample size
  (adaptive binning), so maps sharpen as data accumulate.
- **Label-free QA scales with N:** the number of co-moving pairs grows with the
  catalogue, which tightens the precision check without new labels.

## 8. Lessons from 22 November that this design absorbs

- **S1 parallax and ensemble.** Parallax as a profiled nuisance with its
  measurement term closes most of the log g gap (seismic NMAD 0.124 -> 0.088),
  and the 5-head ensemble improves every metric. Training only on parallax/error
  >= 10 removed 70-80 % of the metal-poor stars, and the bias returned.
  - Lesson: no product may be trained on a sample selected on a quantity
    correlated with the target unless the selection is undone by weights.
  - Next step: train the 4-D heads on all labelled stars, keeping the parallax
    measurement term.
- **XP forward model.** Training works (XP chi^2/dof ~ 1), and the likelihood is
  right: from the labels it converges to Teff r = 0.97 (71 K). The search fails
  on real stars because the start ranking fails. Lowering E's bound shows the
  likelihood wanders into (theta, E) regions the emulator never saw.
  - Lesson: A must carry an explicit training support, as S1 does, in (theta, E)
    for the forward model.
  - Running now: likelihood scans (`scripts/scan_xp_forward_likelihood.py`).
- **The learned photometric scatter is about 0.5 mag in every band.** Absolute
  magnitudes are poorly emulated, likely because of binaries and label noise in
  log g. This argues for the binary flags (section 3) as covariates and for
  errors-in-variables training.

## 9. Plan (ordered; each step ends with the QA plots listed)

| # | step | QA plots |
|---|---|---|
| 1 | Forward-model likelihood scans, then the support restriction and start fix | likelihood against Teff around the labels for representative stars; (Teff, E) NLL maps; recovery after the fix (MLE against label, coverage by [Fe/H] x G) |
| 2 | S1 4-D heads on all labelled stars, 5-head ensemble | bias and NMAD against reference [Fe/H] per G bin for posterior / S1 / S1-plx; Neyman coverage per bin |
| 3 | Selection propensity `pi(x)` on the parent catalogue | propensity in (G, colour) and on the sky; effective sample size per cell; overlap histogram |
| 4 | Product B: flow vs Student-t mixture vs Student-t (B0; then B1 with H, v_t) | log and energy scores; HPD-coverage curves per stratum; example faint bimodal stars; bias and NMAD per bin |
| 5 | C1: IPW-PPI estimands (MDF quantiles, metal-poor fraction, [Fe/H] against v_phi) | PPI against naive plug-in against held-out truth; interval coverage over repeated label splits |
| 6 | C2: NPMLE deconvolution from the A likelihoods | deconvolved MDFs against PPI per cell |
| 7 | Co-moving-pair consistency for A, B0 and B1 | distribution of delta-[Fe/H] between components; implied scatter against reported uncertainties |

Generic statistics go upstream to torchregress: IPW and stratified PPI, NPMLE
g-modelling, Mondrian and weighted density conformal, multivariate flow heads
with errors-in-variables training. MSA keeps the feature engineering (reduced
proper motion, velocities, quality flags, propensity features) and the
experiments.
