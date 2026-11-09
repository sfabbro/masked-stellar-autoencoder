# Mapping and distributional-model audit — 6 November 2026

The present three-survey common gravity is not physically calibrated on the
APOKASC sample. The completed audit points primarily to its equal-weight
reference convention and underestimated predictive uncertainty, rather than a
large spline-linearization error or a demonstrated Jacobian collapse.

The intended large-run output is **six flexible target distributions conditional
on Gaia inputs**. Full joint modeling is desirable but not a prerequisite;
population corner plots must show cross-star correlations/nonlinearities, and
within-star corner plots must use genuine joint samples where available. Six
independent marginal samplers cannot establish within-star dependence.
The [requirements](../../configs/final-six-density-requirements-20261106.json)
cover distributions conditioned on Gaia inputs. Optional physical-density/Jacobian
utilities are scoring tools, not an eligibility gate for SLS or conformal region
methods. A target-on-target conditioning API is outside the requested scope.

## What happened to logg

The frozen three-target model has a neural encoder plus a four-component
full-covariance conditional mixture. Its survey corrections are additive cubic
splines of the three atmospheric parameters. The fitting likelihood and mapped
PDF export use an affine Taylor approximation around a per-star Gaussian
reference-head mean. That reference is not generally the final mixture mean.

The relative spline head enforces equal-weight, target-wise zero-sum
corrections and derivatives across APOGEE, DESI and GALAH. Therefore, for the
implemented affine component mapping,

\[
 E[\theta\mid x]
 =\tfrac13\{E[y_{\rm APOGEE}\mid x]+E[y_{\rm DESI}\mid x]+E[y_{\rm GALAH}\mid x]\}.
\]

The trained audit confirms this equality on all 734 seismic devtest objects,
with maximum logg discrepancy **6.3e-7 dex** from saved-export numerical drift.
This is an arithmetic average of *model-implied survey projections*, including
surveys that did not observe the star. It is not an average of three actual
measurements and it is not a physical anchor.

On these same objects, the common gravity median residual is +0.229 dex. The
APOGEE projection is -0.026 dex, the GALAH projection +0.039 dex, and the DESI
projection +0.681 dex relative to seismic logg. **No native DESI or GALAH labels
are present in this particular subset.** The displaced DESI projection can be
an extrapolation or calibration-model problem; it does not measure a +0.681 dex
bias in the DESI catalogue itself. But the equal-weight convention gives it a
one-third role in defining the common coordinate here regardless of the lack
of an actual DESI measurement.

The native APOGEE label differs from seismic logg by only +0.013 dex median and
0.049 dex MAE. The common model's 0.228 dex MAE is therefore not explained by a
comparably large native APOGEE label bias on this sample. APOGEE's seismic
calibration pedigree and historical parent/teacher exposure remain explicit;
these are fine-tuning holdouts, not certified wholly independent stars.

### Trained geometry and approximation checks

Job `survey_mapping_geometry_audit_20261106_v2` completed on CANFAR CUDA.
It verifies the selected checkpoint and original saved arrays by SHA and uses
the exact calibration source at torchregress revision
`2777b0de88121768434725ef227893cd35595d2a`. There is no fitting or encoder replay.
For each of the 734 stars it draws 1,024 whole three-target vectors from the
saved mixture, evaluates the exact nonlinear splines, and compares with the
trained affine observation model on identical draws.

| Mapping | Median absolute per-star logg mean difference, nonlinear vs affine | p95 | Minimum sampled standardized Jacobian singular value |
|---|---:|---:|---:|
| APOGEE | 0.00080 dex | 0.00700 | 0.633 |
| DESI | 0.00124 dex | 0.01640 | 0.420 |
| GALAH | 0.00058 dex | 0.00875 | 0.806 |

No sampled Jacobian has nonpositive determinant or nonpositive own-logg
sensitivity. This is not a global invertibility proof. Large tail-draw
approximation errors exist (up to 0.73 dex); the maximum per-star mean error
is 0.059 dex. Nevertheless, replacing affine propagation by direct spline
propagation barely changes the median mapping biases. It does not remove
the 0.229 dex common-scale problem on this sample.

The Gaussian reference mean differs from the final mixture mean by 0.081 dex
median in logg (p95 0.285). This is a real approximation design consideration,
but the direct propagation control shows it is not the dominant explanation
of this particular median gravity offset.

An initial audit invocation used tighter floating-point tolerances than the
original prediction/export replay and stopped on a maximum 0.00030 K,
9.1e-7 dex logg discrepancy. V2 uses the export's original tolerances
(0.02 K, 2e-5 dex, 2e-5 dex), records measured drift, and passes. The failed
receipt and manifest remain preserved; no scientific integration threshold or
calibration criterion was relaxed.

### External-anchor control isolates bias from width

Job `seismic_anchor_controls_20261106_v1` fits scalar calibration controls on
5,296 FIT seismic matches only, selects the family using log score on 689 TUNE
matches, and reports CAL and devtest separately. No existing model or survey
prediction is modified.

| Control, same 734 devtest stars | Median residual | MAE | 90% coverage including quoted seismic error |
|---|---:|---:|---:|
| Original common gravity | +0.2287 | 0.2282 | 14.31% |
| FIT-only offset (-0.2223 dex) | +0.0064 | 0.0877 | 51.63% |
| FIT-only affine correction | +0.0065 | 0.0880 | 51.09% |
| FIT-only affine plus discrepancy scatter, TUNE-selected | +0.0115 | 0.0880 | 90.74% |

The selected control has slope 0.9867, intercept -0.1825 dex, and extra
logg scatter 0.0862 dex. This scalar Gaussian discrepancy kernel is an external
calibration diagnostic, not a proposal to make the six-target stellar density
Gaussian. It shows two separate problems: removing the reference offset is
insufficient because the original common predictive spread remains too narrow.
It is not a general correction for every stellar population or all six targets.

An invertible affine reparameterization of the common coordinates can be
exactly compensated by inverse-composing the survey maps, preserving every
survey likelihood component. A test demonstrates this. Independent anchors,
not native survey fit alone, must decide which coordinate deserves physical
interpretation. Adding extra scatter is a separate recalibration operation and
does not have that likelihood-invariance property.

## Why the mapping needs redesign, beyond three parameters

A shared latent representation with multiple observation models is a sensible
starting point. The current choice of an equal-weight gauge, extrapolation
convention, fixed noise floors and local linearization is a particular
approximation, not a unique physically identified calibration.

The replacement design must have

\[
 \theta=(T_{\rm eff},\log g,[\mathrm{Fe/H}],\alpha,\mathrm{age},\mathrm{distance}),
 \qquad y_{is}^{\rm obs}\sim
 p_s\big(y_{is}^{\rm obs}\mid M_{is}h_s(\theta_i,q_{is}),\psi_s\big).
\]

Here each survey/source has its own quantity definitions and observation map
`h_s`, known measurement conditions `q`, observed-target mask `M`, and
calibration/noise parameters `psi`. A source measuring only Teff/logg/FeH
constrains those outputs without fabricated age/alpha/distance labels. A source
with age, distance or alpha information must constrain those corresponding
coordinates through an appropriate source-specific mapping. There is no
three-target design limit.

Important implementation choices:

- Anchor supported physical coordinates with independently justified reference
  measurements. Keep a reference-survey convention only as an explicitly named
  baseline, not as physical truth. Do not silently recenter archived outputs.
- Start with identifiable, regularized observation functions. Compare offset,
  monotone own-coordinate corrections, and supported cross-parameter effects
  before unconstrained source NNs. A measurement map need not be globally
  invertible if the physics is degenerate, but its loss of information must be
  represented rather than interpreted as a successful calibration.
- Treat source/domain support as an observed constraint. Spline knot-range
  checks alone do not establish that a particular survey supports that kind
  of star. The current zero-sum rule gives unsupported survey projections a
  role in defining the common coordinate.
- Carry uncertainty in fitted mappings and residual discrepancy into prediction
  QA. The current source-map coefficients are point estimates; narrow stellar
  PDFs do not include all uncertainty about the common reference.
- Distinguish direct observations from posterior teachers. Age/distance
  catalogue posteriors incorporate priors and shared data and are not
  independent Gaussian measurement errors. Alpha/M and alpha/Fe, or M/H and
  Fe/H, remain separate quantities unless an explicit justified map exists.
- Use parallax information once in a coherent model. An amortized predictor
  conditioned on measured parallax is different from a model conditioned on
  XP/photometry followed by a parallax likelihood update. Multiplying a second
  supposedly independent parallax factor after already conditioning on the
  same measurement is not an automatic improvement.

For a likelihood-trained distribution, the conceptual objective is the
observed-data integral over the six latent targets, with independent anchor
measurement factors and a reconstruction term. For sample-based training, an
alternative is to draw the six-vector, propagate it through the actual
nonlinear source maps, add declared observation noise and score the observed
source coordinates. That avoids Taylor propagation and a failed partial-flow
NLL estimator, but it still requires identified scales, source support and
honest handling of missing/selected labels.

The **current trained mapped-survey checkpoint covers three atmospheric
parameters**. The **current six-target pilots use pooled labels**. Neither is
already the expanded six-target mapped-survey training described above.

## Scientific role of NSF, contrastive flows, SLS and simpler controls

| Method | Actual output relevant here | Scientific role and limitation |
|---|---|---|
| Unrestricted NSF | Flexible conditional six-vector samples and joint density | Main flexible-density candidate. General missing-label noisy likelihoods require numerical integration; exact full-vector log density does not solve that integral. Marginal distributions can be inspected from genuine predictive samples. |
| Contrastive flow + ContrastiveFlowConformal | A conditional positive flow plus conformal level-set membership | Explicit candidate for density/representation training plus calibrated regions. Compare NLL/sample-proper-score training with a controlled contrastive auxiliary term; pure ranking is not itself proof of density accuracy. Conformal calibration is an additional region procedure. |
| SLS | Direct conditional multivariate prediction regions/frontier | Relevant candidate for efficient multimodal uncertainty regions and visual inspection. It need not expose normalized log density to be considered. Compare coverage and region size/volume; do not invent an NLL from its frontier score. |
| Gaussian | Simple conditional uncertainty baseline | Numerical and statistical control for input/mapping changes; not a mandated final family. |
| Full-covariance mixture | Flexible multimodal baseline with tractable missing/noisy affine observations | Integration/control baseline, not evidence that arbitrary nonlinear survey maps were integrated exactly or a default winner. |

A synthetic six-target interface check ran the actual local torchregress APIs:
NSF and ContrastiveFlowLoss provide finite six-target draws and log density;
ContrastiveFlowConformal calibrates and evaluates membership; SLS evaluates a
finite six-dimensional frontier. The wrapper/frontier objects do not themselves
provide sampling/log-density APIs. This is only an interface check, not a
stellar training or performance comparison.

Four unrestricted six-target NSF arms **are training**, using masked sample
energy scores at two learning rates and with/without measured parallax. At
readback they have **7,525–8,750 / 10,000 updates**. They are not six scalar
flows or a Gaussian copula. The Gaussian/mixture results were available first
because those runs finished earlier; that is not a scientific family-selection
argument. Earlier scalar NSF plus a contrastive auxiliary objective also
exists, but that was not an unrestricted six-target joint demonstration.

The reason to retain analytic baselines is concrete: earlier unrestricted NSF
partial/noisy NLL integration failed selected-checkpoint numerical gates.
In two preserved runs median relative ESS was 0.00622, with replica log-density
gap p95 of 11–13 nats against a 0.10 threshold. That rejects those numerical
likelihood estimates; it does **not** reject the NSF family. The current sample
energy objective avoids claiming that failed integral was exact.

Energy scores are useful but can be relatively insensitive to dependence errors;
include pairwise/dependence-sensitive diagnostics and corners where joint draws
are available. A conformal region's marginal coverage under exchangeability
is not a universal star-by-star conditional-coverage guarantee or a correction
for biased physical labels. The SLS paper itself distinguishes its conditional
adaptivity aim from a finite-sample distribution-free conditional guarantee.

## Current seismic comparison and decision

On all 734 available seismic devtest stars, the completed six-target controls
have the following gravity scores. These are a different pooled-label recipe
from the frozen mapped-survey model, so they do not isolate a causal family or
mapping fix.

| Six-target pilot | logg MAE | Median residual | Quoted-seismic-noise 90% coverage |
|---|---:|---:|---:|
| Gaussian, XP + photometry | 0.0963 | +0.0429 | 89.78% |
| Gaussian, plus measured parallax | 0.0768 | +0.0352 | 90.33% |
| Mixture, XP + photometry | 0.0999 | +0.0473 | 95.78% |
| Mixture, plus measured parallax | 0.0731 | -0.0294 | 90.19% |

The paired parallax controls support retaining measured parallax in the
candidate input profile. They do not choose the final distribution family.
The same worker will score the four selected NSF checkpoints when ready.

The historical pool contains only 277 all-six FIT stars, 39 TUNE, 31 CAL and
28 devtest before the smaller PDF-panel selection. Millions of partial labels
remain usable, but data-supported cross-target dependence and six-dimensional
region calibration cannot be certified from catalogue size alone. This is a
reason for honest visual support flags and targeted QA, not a reason to refuse
six flexible marginals or exclude NSF/SLS.

Before the large training decision: compare source-map conventions and physical
anchors with the same data/inputs/head; inspect gravity and other-target
residuals by population; compare flexible-method proper scores or region
coverage/efficiency as appropriate; preserve dedicated calibration and external
identities. The final family is not selected and final large training has not
started.

## Evidence and references

- [Trained geometry audit](msa-mapping-density-audit-20261106/survey_mapping_geometry_audit_20261106_v2-latest.json).
- [FIT/TUNE anchor controls](msa-mapping-density-audit-20261106/seismic_anchor_controls_20261106_v1-latest.json).
- [Full 734-star pilot seismic PDF results](msa-mapping-density-audit-20261106/decision_seismic_pdf_qa_20261106_v1-latest.json).
- [Synthetic method-interface proof](msa-mapping-density-audit-20261106/method-capabilities.json).
- [Original seismic reference/join audit](msa-andrae-seismic-20261106.md).
- [Neural Spline Flows](https://arxiv.org/abs/1906.04032).
- [Super-Level-Set Regression](https://arxiv.org/abs/2605.06210).
- [Limits of distribution-free conditional predictive inference](https://arxiv.org/abs/1903.04684).
- [Dependence-sensitive variogram scores](https://repository.library.noaa.gov/view/noaa/22327/).
- [Pinned torchregress conformal implementation](https://github.com/sfabbro/torchregress/blob/2777b0de88121768434725ef227893cd35595d2a/src/torchregress/losses/conformal.py).

Verification: scoped pipeline and new mapping/coordinate tests pass (nine
checks). Synthetic examples demonstrate biased but zero-sum maps, possible
Jacobian collapse despite a zero-sum derivative constraint, and exact
survey-component invariance under an affine common-coordinate change. These
examples explain possible failures; the trained audit separately establishes
what actually occurred on the measured sample.


## Corner plots and additional failures exposed by the distributions

Saved producer-checksummed arrays now provide two complementary inspections:

- [Population corner](msa-mapping-density-audit-20261106/mixture-parallax-population-corner.png):
  six predicted medians for **24,607 distinct development stars**. This is the
  already-exported fixed pilot panel, not the available training population or
  a full Gaia inference census. The Teff/logg branch, abundance trends and
  nonlinear age/temperature relation are visible. Population patterns alone
  do not establish within-star uncertainty dependence or physical correctness.
- [Within-star mixture corner](msa-mapping-density-audit-20261106/mixture-parallax-within-star-corner.png)
  and [Gaussian control](msa-mapping-density-audit-20261106/gaussian-parallax-within-star-corner.png):
  **8,192 genuine whole-vector draws** for the same predeclared Gaia source
  `4185355646116022528`, using identical axis limits across models. Orange
  lines mark available native atmospheric references. There are no independent
  age/distance/alpha reference points for this star in the saved bundle.

These plots are diagnostic, not selected as success examples. The displayed
metal-poor star has a strong hot/high-gravity mode displaced from its native
atmospheric references. Its mixture corner shows multimodal/correlated
structure, but that structure is not automatically correct physics.

The age distributions also expose a support problem hidden by median-only
QA. Across the ten predeclared mixture corner stars, **0.23%–28.1%** of the
saved age draws exceed **20 Gyr**; maximum sampled ages reach about **1,014
Gyr**. Gaussian controls also have excessive tails (up to about 420 Gyr in
these saved draws). The current log-age parameterization guarantees positivity
but not physically plausible upper support. These remain posterior-teacher
scale predictions, not physically calibrated stellar ages. No samples were
clipped to improve the metrics. Physical-support choices and teacher outliers
need explicit treatment before calling the large-run age output physical.

Plots show central 99.6% per-axis display ranges; all arrays and full-sample
age-tail statistics are retained in the
[plot manifest](msa-mapping-density-audit-20261106/corner-plot-manifest.json).
Age and distance use log10 display coordinates to expose structure and tails.
The four NSF runs will need the same visual and support QA when completed.


## Completed full Andrae input census

All **174,922,161 / 174,922,161** released Andrae IDs match the full
**218,131,884-row** raw input. **Zero** Andrae stars are absent from that raw
input. The reconstruction cache contains **120,181,886** of them;
**54,740,275** are present in the raw input but outside the 150M cache.
The publisher checksum and exact integer IDs were checked. The completed
inference index stores original raw shard/row locations and comparator values,
with a SHA-bound output artifact. These are input-availability counts, not
completed model-prediction counts or independent reference truth counts.
The earlier 2.4M comparator overlap was solely the existing prediction-export
scope. All released Andrae stars are available for a full inference/comparison
pass using the full raw input.


A CANFAR saved-array worker, `decision_corner_plots_20261106_v1`, is queued
to verify producer hashes and render matching within-star corners for all eight
existing runs when their immutable arrays are ready, including the four NSF
variants. It also records age-tail fractions for every predeclared corner star.
It performs no model execution or training and continues independently of the
laptop.


## Is the displayed bimodality representative?

No population-representativeness claim is justified by that illustration.
The displayed star has native reference [Fe/H] **-3.228**, Teff **4,730 K**,
and logg **2.278**. All ten predeclared corner stars have native reference
[Fe/H] below **-2**: they are a metal-poor stress-test set, not a random sample.
The previous one-star caption should have identified that selection explicitly.

Across all ten saved mixture+parallax examples, three show clear separated logg
modes under a descriptive histogram diagnostic checked at two smoothing widths.
The other seven have predominantly single-peaked logg marginals under that
check; this is not proof that their full six-dimensional distributions are
unimodal. The count is not an estimate of prevalence across Gaia or the full
development population.

For the displayed star, the logg peaks are approximately **2.6** and **3.66**
dex. Splitting at the intervening valley gives about **37% / 63%** of draws
in the lower/higher branch. These are draw-mass fractions, not fitted mixture
component weights. The same star's Gaussian control is single-peaked around
**3.53 dex**, also displaced from the native logg reference. Thus suppressing
bimodality does not by itself correct the prediction. The branches could reflect
input degeneracy, a learned source/label mixture, sparse tail supervision or a
model failure; their astrophysical validity is not established by this plot.

[All ten gravity marginals and Gaussian controls](msa-mapping-density-audit-20261106/all-ten-stress-logg-gallery.png),
[diagnostic definitions and per-star results](msa-mapping-density-audit-20261106/stress-bimodality-audit.json).

A new CUDA evaluation, `representative_corner_draws_20261106_v1`, selects
**1,024** input-eligible prepared-development identities uniformly without
replacement, with seed fixed before any head evaluation or label inspection.
Predeclared stress stars are separately flagged. It saves 1,024 genuine whole
six-vector draws per star for all eight selected heads as checkpoints finish.
The paired analysis worker `representative_corner_analysis_20261106_v1`
verifies SHA, records descriptive gravity-mode frequency, renders identical
random-star galleries across models and saves full corners for two model-blind
random identities per model. This estimates behavior within the eligible
spectroscopic/teacher development cohort, not an all-Gaia incidence rate.
Both are evaluation only, with no training or role changes.

Verification: the peak diagnostic separates a synthetic single Gaussian from a
clear two-mode distribution; selection tests verify reproducibility, held-out
role/input eligibility, exact IDs and separate stress membership. Both pass.


### Random-panel result now available

On the **same 1,024 model-blind randomly selected eligible development stars**,
the descriptive criterion flags separated logg modes in **7** mixture+parallax
predictions (**0.684%**) versus **207** mixture without-parallax predictions
(**20.215%**). Both Gaussian controls have zero flags, as expected for their
unimodal logg marginals. These compare input-profile training recipes on the
same star identities, not a measured astrophysical binary/degeneracy fraction.
The criterion concerns well-separated logg peaks, not every possible form of
joint multimodality. The cohort is not a random draw from all Gaia.

Only 293 random-panel stars have a native reference Fe/H: one is below -2,
and 292 are at least -2. The other 731 have no native Fe/H reference in the
prepared table. For mixture+parallax the flag counts are 1/1, 4/292 and 2/731,
respectively. For mixture without parallax they are 1/1, 4/292 and 202/731.
The single metal-poor random case is too small for a metal-poor prevalence
estimate. The large change in the group without native Fe/H motivates
checking source/label coverage, parallax availability/SNR and calibration, not
assuming an astrophysical interpretation of every branch.

[Random single-star example 1](msa-mapping-density-audit-20261106/random-corners/decision6_mdn_parallax_20261106_v1-4694907817311533952-corner.png),
[random single-star example 2](msa-mapping-density-audit-20261106/random-corners/decision6_mdn_parallax_20261106_v1-1405433012875924224-corner.png),
[12 random gravity marginals with parallax](msa-mapping-density-audit-20261106/random-corners/decision6_mdn_parallax_20261106_v1-random-logg-gallery.png),
[same stars without parallax](msa-mapping-density-audit-20261106/random-corners/decision6_mdn_photxp_20261106_v1-random-logg-gallery.png).


The source-only visibility audit confirms the original ten are all reserved
**external** reference stars (none marked as known supervised-ancestor exposed).
Six have all 110 XP coefficients and measured parallax; four have neither XP
nor measured parallax in the staged inputs. Two of the three clearly bimodal
stress cases are in that missing-XP/missing-parallax group. **The originally
shown star itself does have all 110 XP coefficients and measured parallax**,
so its bimodality cannot be attributed to missing XP or parallax. Its extreme
metallicity and displaced dominant mode still require explicit tail/model QA.
The ten external stress stars are absent from the randomly sampled devtest
panel by construction; their saved stress results remain separate.


## Explicit noisy, correlated and occasionally wrong label requirement

The replacement observation model must separate source bias, within-source
correlated measurement errors, shared cross-source dependencies and catastrophic
failures. The current prototype explicitly assumes diagonal catalogue covariance
and fixed floors; its full stellar predictive covariance does not supply the
missing label-error covariance. Full covariance submatrices should be retained
where published. Elsewhere, covariance is unknown and requires regularized
source/domain estimation and controls using repeats, overlaps and anchors;
quoted scalar errors must not be presented as a known full covariance.
Robust heavy-tailed or inlier/outlier measurement models are planned controls,
separate from the choice of stellar conditional density family. Repeated
spectra, shared Gaia information, teachers and calibration dependencies require
joint blocks/shared effects rather than independent multiplication. Age and
distance posterior-teacher widths remain distinct from measurement errors.
These requirements are recorded in the ablation plan; they are not yet an
implemented replacement likelihood.


## Completed-wave update

All eight controlled 10,000-update arms and their selected-state PDF, seismic,
random-panel and corner QAs have now completed. Earlier pending/running
paragraphs describe their original observation times. NSF+parallax is a
promising candidate with 0.06977 dex seismic logg MAE, but its nominal 90%
coverage is 82.43%, versus mixture+parallax 0.07309 dex / 90.19%. No final
family is selected. See the [completed-wave lessons](msa-completed-wave-20261106.md).
