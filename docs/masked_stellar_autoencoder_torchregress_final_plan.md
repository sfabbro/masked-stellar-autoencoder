# Multi-Survey Stellar-Parameter Inference and Homogenization

**Final scientific architecture and implementation plan**
**Repositories:** `sfabbro/masked-stellar-autoencoder` (MSA) and `sfabbro/torchregress`
**Date:** 26 October 2026
**Status:** Implementation specification; decisions, mathematical invariants, dependencies, ablations, and acceptance criteria

---

## 0. Executive decision

Train **one survey-independent, probabilistic predictor of stellar parameters** from Gaia XP spectra and heterogeneous photometry. Treat each spectroscopic survey's released stellar parameters as **noisy, potentially biased observations** of those underlying quantities, rather than either (i) interchangeable ground truth to pool into a single target or (ii) separate physical targets to predict with unrelated MLPs. Use star-level overlaps to learn **relative** survey calibration, and independent physical calibrators to define and validate the **absolute** scale.

The primary model is

\[
 x_i\overset{f_\phi}{\longrightarrow}z_i
 \overset{p_\psi}{\longrightarrow}p_\psi(\boldsymbol\theta_i\mid x_i),
 \qquad
 \mathbf y_{is}\sim p_s(\mathbf y\mid\boldsymbol\theta_i,\mathbf q_{is}).
\]

The encoder continues to participate in the **original masked reconstruction task**. Survey-specific components belong to the **training-time observation likelihood**, not to the canonical inference output. At catalog inference, the default output is **one estimate per Gaia source, not one per survey**. Survey-equivalent predicted labels are optional diagnostic products.

**Three non-negotiable principles:**

1. **Relative consistency is not absolute accuracy.** Overlap stars identify differences between survey scales, not physical truth. Any absolute physical-scale claim must specify its calibrators, domain, assumptions, and systematic uncertainty.
2. **Separate three kinds of systematic effects.** A common survey-consensus offset, relative survey offsets, and quality-/pipeline-dependent effects must not be conflated.
3. **Do not combine unrelated architectural hypotheses prematurely.** Establish whether multi-survey modeling improves stellar parameters while keeping the present MAE encoder/decoder; only then test a physics-structured decoder bottleneck.

### 0.1 Deliverables

- **MSA:** a star-indexed crossmatch and calibration dataset; an unchanged reconstruction path plus a canonical probabilistic atmospheric head; joint training on all available heterogeneous survey measurements; independent physical anchors; inference and diagnostics; separate distance and age modules.
- **`torchregress`:** reusable multi-source observation mappings, normalized per-target gauge constraints, differentiable B-spline/Jacobian functions, Cholesky-stable marginal likelihoods with arbitrary observation masks, robust observation mixtures, nonlinear forward-observation losses, and optional hierarchical-group consistency objectives.
- **Scientific output:** a versioned all-sky stellar-parameter catalog with posterior summaries and covariance, coverage/OOD flags, physical-calibration provenance, and domain-of-validity indicators; evaluation versus pooled SFT and independent survey-head baselines.

### 0.2 Priority and scope

**Core (must implement):** `T_eff`, `log g`, `[Fe/H]`; optionally a *well-defined* scalar alpha parameter if the source labels permit it; Gaussian canonical predictive distribution; per-survey smooth calibrations; missing-label likelihood; physical anchor likelihood; overlap/anchor diagnostics; original MAE reconstruction.
**Separate core branches:** parallax-based distance inference and a clearly model-dependent age product; neither can redefine the atmospheric scale through catalog labels.
**Extensions contingent on evidence:** robust mixtures, calibration-function uncertainty, element-resolved abundances, cluster hierarchical constraints, mixture/flow parameter posteriors, and a structured physical reconstruction bottleneck.

---

## 1. Scientific estimands: define the quantities before fitting

Let the default canonical atmospheric vector be

\[
\boldsymbol\theta_i=(T_{\mathrm{eff},i},\log g_i,
[\mathrm{Fe/H}]_i,[\alpha/\mathrm{Fe}]_i)^\top.
\]

- **Temperature:** physical effective temperature (K); distinguish spectroscopic temperature estimates from independent interferometric/bolometric calibration.
- **Gravity:** `log10(g/[cm s^-2])`; seismic scaling-law values and benchmark mass/radius estimates have method-dependent systematics.
- **Metallicity:** explicitly **iron abundance `[Fe/H]`**, with specified solar reference. APOGEE's global `[M/H]` and other fitted atmosphere metallicities are *not* automatically interchangeable with `[Fe/H]`.
- **Alpha:** not automatically the same observable across surveys. In APOGEE, global `ALPHA_M` / `[alpha/M]` has element- and temperature-dependent sensitivity; another survey may report `[alpha/Fe]` synthesized from selected elemental abundances. **The canonical alpha target must have a documented operational definition.**

**Phase-1 recommendation:** implement the generic likelihood with `D` configurable, but demonstrate full physical validity for **three canonical dimensions** first. Enable alpha only after an explicit source-field mapping audit. It is acceptable to run exploratory four-dimensional models, but do not release them as a physically homogenized alpha scale without this audit.

**Alpha extension:** represent selected individual abundances, e.g. `[Mg/Fe]`, `[Si/Fe]`, `[Ca/Fe]`, `[Ti/Fe]`, and derive a documented aggregate from their posterior. Survey-reported composite alpha then has its own element-dependent observation map. This is preferable to applying an affine correction between intrinsically different aggregates.

Record for every label: survey, data release, pipeline, exact source column, units/solar scale, quality mask, uncertainty provenance, and known dependency on external priors or catalogs. **Never silently overwrite one survey's value with another.**

---

## 2. Statistical model and what is identifiable

### 2.1 Encoder and canonical predictive distribution

For masked XP coefficients and heterogeneous multi-survey photometry `x_i` (including measurement uncertainties and input-availability masks), obtain a learned representation

\[
 z_i=f_\phi(x_i),\quad
 p_\psi(\boldsymbol\theta_i\mid x_i)
 =\mathcal N(\boldsymbol\mu_i,\mathbf V_i),\quad
 \mathbf V_i=\mathbf L_i\mathbf L_i^\top.
\]

Use a numerically stable Cholesky parameterization and a target normalization transform. Recommended normalized parameterization: `log(T_eff/K)` or carefully standardized `T_eff`, plus standardized dex quantities. Define and persist the normalization map. All derivatives, calibration functions, covariances, and published outputs must be transformed consistently back to physical units. A full covariance is useful because atmospheric-label degeneracies correlate parameter errors.

The Gaussian is a first baseline—not a claim that all atmospheric posteriors are unimodal. Check multimodality, heteroscedasticity, and conditional coverage; introduce conditional mixtures/flows only if the diagnostics demonstrate a meaningful need.

**Interpretation:** `V_i` characterizes the canonical predictive distribution conditional on the supplied inputs and fitted model. It does **not** by itself include uncertainty in the fitted survey-calibration functions, and one neural network cannot cleanly distinguish epistemic versus aleatoric uncertainty merely by outputting a covariance.

### 2.2 Generative survey observation model

For a measurement `y_is` from survey/pipeline instance `s`, with available dimensions `O_is` and measurement conditions `q_is`, define

\[
 \boxed{\quad
 \mathbf y_{is,O}
 =\mathbf P_O\left[
 \boldsymbol\theta_i
 +\mathbf c(\boldsymbol\theta_i)
 +\boldsymbol\delta_s(\boldsymbol\theta_i)
 +\mathbf u_s(\boldsymbol\theta_i,\mathbf q_{is})
 \right]
 +\boldsymbol\epsilon_{is,O}.\quad}
\]

Here `P_O` selects the actually observed label dimensions. For labels with a different mathematical definition (notably `[M/H]`, `[alpha/M]`, and composite alpha), replace the leading identity mapping with an explicit observation map `h_s(theta, ...)`; **do not force mismatched definitions into a Gaussian `[Fe/H]` / alpha residual.**

| Term | Meaning | Identification / restriction |
|---|---|---|
| `theta_i` | Canonical physical parameters | Constrained by physical anchors where they exist; uncertain elsewhere |
| `c(theta)` | **Common** difference between survey-consensus and physical scale | Smooth, low capacity, shrinkage; requires physical anchors |
| `delta_s(theta)` | **Relative** survey/pipeline bias | Fixed-weight zero-sum at a reference observing condition |
| `u_s(theta,q_is)` | Conditional mean bias from SNR/flags/release etc. | Exactly zero at a documented `q_s^ref`; include only when supported |
| `epsilon_is` | Measurement error/outlier process | Catalog covariance + survey-dependent intrinsic scatter; robust extension |

For a shared target dimension `d`, with a fixed set of comparable surveys `S_d` and strictly positive fixed weights `w_{s,d}` summing to one,

\[
\boxed{\sum_{s\in S_d}w_{s,d}\,\delta_{s,d}(\boldsymbol\theta)=0}
\qquad\text{for every }\boldsymbol\theta\text{ in the calibrated domain}.
\]

Also impose

\[
\boxed{\mathbf u_s(\boldsymbol\theta,\mathbf q_s^{\rm ref})=0.}
\]

The weights are **not** the empirical number of labeled stars. Use fixed, versioned per-target weights (uniform among physically comparable sources as a transparent default). Do not include surveys with undefined measurements of that target; do not give a disconnected or uncalibrated source fictitious support merely to satisfy the algebra. The zero-sum constraint is a coordinate convention, not a property of nature.

**Important:** do *not* claim that `sum_s w_s b_s(theta,q_is)=0` for arbitrary SNR or quality conditions. Only the relative baseline component `delta_s` has zero weighted sum. The derivatives of `delta_s` satisfy the corresponding weighted zero-Jacobian identity wherever the functions are differentiable; the total quality-conditioned biases need not.

### 2.3 Gauge non-identifiability and physical anchoring

Without independent physical anchors, the observations identify a **survey-consensus coordinate** and **relative survey differences**. They do not determine the absolute scale of `theta`. More generally, a flexible reparameterization of `theta` can be absorbed by flexible survey observation functions. A zero-sum constraint can choose a consensus coordinate but cannot turn that coordinate into physical truth.

The split into `c(theta)` and `delta_s(theta)` makes this distinction explicit:

- In a **consensus-only** fit, fix `c=0` as a *convention*, and label the output `theta_consensus`, not `theta_physical`.
- In a **physically anchored** fit, fit `c` with strong shrinkage and independent anchor measurements. The canonical `theta` is then a **physically calibrated estimate** only where the anchor information identifies the scale.
- Outside anchored/interconnected regions, absolute offsets, slopes, or nonlinear warps may remain weakly identifiable. Propagate this uncertainty and publish a **physical-anchor coverage flag**. Do not assume a flexible warp is identified because an NN can fit it.

Prefer low-dimensional smooth `c` (constant/linear first; low-order spline only if justified), a regularizing prior centered on zero, and conservative extrapolation. `c` and `delta_s` must be estimated jointly with anchors or coherently reoptimized after an anchored initialization; simply turning on a new `c` layer while freezing a consensus-trained model is not sufficient.

**No absolute calibration exists for some quantities in the same sense as temperature.** `[Fe/H]` depends on abundance-analysis conventions, atomic data, solar reference, atmosphere models and 3D/NLTE treatment. Label the resulting scale as `externally calibrated to [named analysis/standards]`, not model-independent physical truth. The same warning is stronger for composite alpha.

### 2.4 Anchor likelihoods—not hard labels

For external measurement `a_ij` of some target dimensions, use an explicit measurement mapping and uncertainty:

\[
 \mathbf a_{ij}\sim
 p_{\mathrm{anchor},j}(\mathbf a\mid\boldsymbol\theta_i,\mathbf r_{ij}).
\]

Available sources, with distinct status:

1. `T_eff`: angular-diameter + bolometric-flux benchmark temperatures (strongest fundamental route); broader IRFM values can supplement but are not equivalently independent of photometric calibration/extinction.
2. `log g`: asteroseismic gravity, dynamically constrained binaries, mass/radius benchmarks; propagate scaling-law/model error and potential correlations with temperature and parallax.
3. `[Fe/H]`: high-quality benchmark/high-resolution abundance analyses with consistent methodology and explicit solar/NLTE/3D caveats. These are *spectroscopic standards*, not fundamental-free truths.
4. Alpha: individual high-resolution elemental abundances with a documented solar baseline and aggregation rule; often insufficiently independent to anchor a universal scalar.

Include anchor likelihood factors **in the same latent-parameter marginalization** as survey measurements when possible, selecting only the observed dimensions. Do not both insert an anchor as an ordinary survey target and multiply its likelihood again. If a benchmark combines Gaia parallax, photometry, or the same spectra used elsewhere, document correlations and avoid double-counting shared data/calibration systematics.

Use partial anchors: a temperature-only interferometric measurement is useful even if it has no independent gravity or metallicity. Give anchors method/source identifiers and systematic floors/correlated calibration blocks where available. Reserve some independent anchors entirely for validation; design the allocation by *coverage and independence*, not an arbitrary fraction of a tiny sample.

### 2.5 Noise, covariance and shared systematics

For survey `s`, initially use

\[
\mathbf R_{is,O}=\mathbf P_O
 (\mathbf C^{\rm cat}_{is}+\mathbf C^{\rm floor}_s(\theta_i,q_{is}))
 \mathbf P_O^\top,
\]

with a PSD learned scatter-floor matrix and sensible lower bounds. Distinguish catalog random uncertainties, repeatability scatter, shared survey zero-point uncertainty, and uncertainty in the fitted bias function. The last two are *correlated across stars*, not independent additions to every per-star variance.

Do not assume independent labels solely because they come from different public catalogs: releases, data-reduction teams, or label-transfer pipelines may reuse reference stars and priors. Baseline conditional independence is acceptable only when explicitly stated; add joint blocks or hierarchical shared effects where source dependencies materially alter results.

**Critical variance identifiability:** a flexible predictor covariance `V_i` and overly flexible survey noise floors can explain the same residuals. Regularize floors, use repeat observations/overlaps/independent anchors to assess them, and audit PIT/coverage, sharpness, and residual dependence by survey and regime. Uncertainty inflation is not an acceptable way to hide bias.

### 2.6 Survey overlap, selection and extrapolation

Each survey has its own magnitude, sky, extinction, stellar-population, and quality selection. Survey overlap stars are not representative of all XP stars. The graph of pairwise overlap must be resolved **per target and stellar-parameter region**. Connectivity in total star counts does not imply identifiability for metal-poor giants or hot stars.

- Use overlaps to measure **relative bias** only where the relevant star types overlap.
- Smooth calibration heads may extrapolate numerically, but such extrapolations require **flags and increasing calibration uncertainty**.
- Do not adversarially erase survey identity from the full latent distribution: different surveys sample genuinely different stellar populations.
- If correcting population-average risks, use carefully specified target-population weighting or stratified diagnostics, not indiscriminate inverse-propensity weights on a poorly identified sampling process.
- Treat crossmatch quality, crowding, unresolved binarity and photometric contamination as potential outlier/OOD sources.

---

## 3. Multi-source marginal likelihood: exact within a linearized model

### 3.1 Linearization and observed-dimension handling

For a fixed star, stack all *actually available* scalar or vector label observations (including compatible anchors if independent) in `y \in R^K`. There is no requirement that all surveys observe all `D` targets. Let `mu` and `V` be the canonical prediction.

Define the nonlinear stacked observation mean `h(theta)`, with individual components according to the previous section. Linearize at `mu`:

\[
 h(\theta)\approx h(\mu)+\mathbf H(\theta-\mu),
 \quad \mathbf H=\left.\frac{\partial h}{\partial\theta}\right|_{\mu},
\]

and take `epsilon ~ N(0,R)` for the observed components only. The covariance `R` may be block diagonal across surveys in the baseline, but the API must allow future cross-survey covariance blocks. **No missing dimension is represented as a numerical zero with a gigantic variance**: build the exact selected subproblem.

Thus

\[
 y\mid x\ \dot\sim\ \mathcal N\!\left(h(\mu),\ \mathbf R+\mathbf H\mathbf V\mathbf H^\top\right).
\]

The dot denotes equality **under the first-order approximation**, not an exact nonlinear marginal.

### 3.2 Stable information-form expression

Let `r=y-h(mu)`, `Lambda_0=V^{-1}`, `Lambda=Lambda_0+H^T R^{-1}H`, and `eta=H^T R^{-1}r`. Then for `K` observed scalars,

\[
\boxed{\begin{aligned}
-\log p(y\mid x)\ \dot=\ \frac12\Big[&K\log(2\pi)
+r^\top R^{-1}r-\eta^\top\Lambda^{-1}\eta\\
&+\log|R|+\log|\Lambda|-\log|\Lambda_0|\Big].
\end{aligned}}
\]

Equivalent implementation: Cholesky factorization of the stacked predictive covariance `R+HVH^T` if `K` is small. Select the faster stable path by shape (`D` typically 3–4, active observations modest). **Use triangular solves and Cholesky log-determinants, never explicit matrix inverses.** Compute log determinants/solves in at least fp32 (fp64 for reference tests); make AMP behavior explicit; symmetrize numerical covariance and use controlled jitter.

The model must compute `H=I+J_c+J_delta_s+J_u_s` only when the survey quantity really maps to the corresponding canonical target. For nonidentity observation maps, compute their full Jacobian. When splines are applied in standardized coordinates, apply the input/output Jacobian factors to yield derivatives in consistent units.

### 3.3 Approximation diagnostics

Compare the linearized NLL against Gauss–Hermite quadrature, unscented integration, or quasi-Monte-Carlo for representative cases with wide parameter uncertainty and high nonlinearity. Gradients must agree within a documented tolerance in the narrow-posterior regime. If approximations fail for a scientifically important regime, use more accurate integration **for that regime** or a mixture posterior, not an unacknowledged linearization.

### 3.4 Robust measurement likelihoods

**Do not apply pseudo-Huber / Student-t transforms directly to Gaussian marginal Mahalanobis terms and call the result an exact marginal likelihood.** This destroys the associated Gaussian normalizer and information-form derivation.

Implement robustness independently as a normalized observation model, e.g. for each survey measurement block

\[
 p_s(y_s\mid\theta)
 =(1-\epsilon_s)\,\mathcal N(y_s;h_s(\theta),R_s)
 +\epsilon_s\,\mathcal N(y_s;h_s(\theta),\kappa_sR_s),\quad \kappa_s>1.
\]

Under the linearized shared latent `theta`, the likelihood of each **joint assignment of inlier/outlier blocks** is a Gaussian integral computable by the previous method. Marginalize assignments with `logsumexp` (exact for a small number of measured survey blocks; approximate/prune if necessary). A Student-t hierarchical scale-mixture likelihood is a separate, potentially approximate option.

An anchor may itself need an outlier model, but anchor reliability should be curated rather than hidden behind a large learned outlier probability.

### 3.5 Systematic calibration uncertainty

A point-estimated `c`/`delta_s`/`u_s` is insufficient for credible physical-scale error bars. A practical staged strategy:

- First learn point estimates, retaining model ensembles/bootstraps split by star and calibration sample.
- Later approximate spline-coefficient posterior covariance, e.g. Laplace/low-rank covariance, bootstrap, or a compact Bayesian layer.
- Propagate the **correlated global calibration component** separately from `V_i` and per-star measurement noise. For linearized calibration coefficients `beta`, `Cov_cal(h_i,h_j) ~ J_{beta,i} Cov(beta) J_{beta,j}^T`.
- Report systematic ranges by parameter-domain cell where full star-to-star covariance is too large to publish.

Avoid claiming an ensemble gives formal posterior epistemic uncertainties without calibration tests.

---

## 4. Calibration function parameterization

### 4.1 Recommended first model

Use **low-capacity additive B-splines**, starting with constant offsets and then gradually adding smooth dependencies on stellar parameters. The common correction `c`, relative survey corrections `delta_s`, and condition-dependent corrections `u_s` are distinct modules/parameter blocks. Each takes `theta` in a documented normalized coordinate system.

For relative offsets, define unrestricted smooth `a_{s,d}(theta)` and center **only this part**:

\[
 \delta_{s,d}(\theta)=a_{s,d}(\theta)
 -\sum_{t\in S_d}w_{t,d}\,a_{t,d}(\theta).
\]

This exactly satisfies the zero-sum constraint and weighted-zero Jacobian. The per-target survey set/weight matrix is fixed for a trained model version.

For measurement-conditions bias, define an unrestricted low-capacity `v_s(theta,q)` and reference-center **within the same survey**:

\[
 u_s(\theta,q)=v_s(\theta,q)-v_s(\theta,q_s^{\rm ref}).
\]

This makes `u_s(theta,q_s^ref)=0` by construction, without falsely constraining the sum of heterogeneous SNR effects. For the first release, **permit SNR to affect variance before permitting SNR to affect the mean**, unless residual plots show a reproducible mean trend. Treat pipeline/release identifiers as categorical, not continuous numeric SNR-like variables; differing releases of the same spectra are usually correlated, not independent repeated labels.

### 4.2 Capacity, boundary behavior, and monotonicity

- Start with offsets + a small number of one-dimensional spline terms; add selected interactions only when residual structure and overlap density justify them. Avoid high-dimensional tensor-product splines by default.
- Center/standardize each parameter; regularize coefficient magnitude, curvature and interactions. Calibrate only over surveyed/anchored support.
- Provide **explicit extrapolation behavior** outside the spline knot domain: e.g. boundary-value or linearly continued correction with bounded slope and an OOD flag/variance increment. **Do not silently `clamp`** and interpret the resulting zero derivative as learned physical behavior.
- Optionally constrain `I+J_hbias` to be locally well conditioned / physically monotone within the support, preventing severe foldings in the survey observation map; regularize, rather than assert that zero-sum removes all nonlinear reparameterization degeneracy.
- Reuse or extend existing `torchregress/utils/bspline.py` primitives after verifying their actual behavior in the checked-out repository.

### 4.3 Differentiable Jacobians

The B-spline derivative recurrence for degree `p` is

\[
 \frac{d}{dx}B_{k,p}(x)
 =p\left[\frac{B_{k,p-1}(x)}{t_{k+p}-t_k}
 -\frac{B_{k+1,p-1}(x)}{t_{k+p+1}-t_{k+1}}\right],
\]

with zero-contribution conventions for repeated-knot zero denominators. Implement basis *and analytic derivative* together. For the additive case, contract derivative bases with centered spline coefficients to produce `(B,S,D_out,D_in)` Jacobians; transform to physical coordinates via the chain rule.

Do not differentiate a sum across surveys when individual Jacobians are needed. Do not call `.detach()` on the canonical mean before evaluating a bias; that severs essential gradients. Finite-difference and autograd reference tests should verify the analytic Jacobian and gradient flow **through both the canonical head and calibration coefficients**.

---

## 5. `masked-stellar-autoencoder`: architecture and data flow

### 5.1 Baseline architecture (the required first implementation)

```text
Gaia XP coefficients (+ covariance/quality) + heterogeneous photometry
                       + input masks and metadata
                              |
                      shared MAE encoder z
                        /             \
                       /               \
   original masked reconstruction       canonical p(theta | x)
             decoder                         N(mu, V)
        L_reconstruction                      |
                               +-------------+--------------+
                               |             |              |
                        survey mappings   independent    optional distance,
                       & likelihoods      anchors       age / dust heads
                        L_multi-source    L_anchor       separate objectives
                               |
                shared latent marginal likelihood
                       (NOT independent heads)
```

The survey mappings act on `theta` at training/evaluation; they are not unconstrained survey-specific predictors `z -> y_s`. The **encoder need not be survey-invariant across all populations**. Masked-reconstruction training should continue on the large unlabeled population, with a controlled labeled/unlabeled minibatch mix.

**Do not replace the decoder input with `[theta,dust,distance,z_nuis]` in this baseline.** That is a second scientific hypothesis with separate ablations. It can either leak all physics through `z_nuis` or force unspecified astrophysics into `theta` if the nuisance channel is artificially restricted.

### 5.2 Alternative structured physical bottleneck (research extension)

Only after the baseline wins, test

\[
 z\longrightarrow(\theta,\mu_d,A_V,R_V,z_{\rm nuis})
 \longrightarrow\widehat{x}_{\rm masked}.
\]

Require explicit ablations for (i) unconstrained reconstruction decoder, (ii) structured bottleneck, (iii) reduced nuisance dimensionality, (iv) stopped-gradient versus coupled paths, and (v) reconstruction at different mask fractions and photometric depth. Test whether `theta` is causally used by the decoder (interventions/ablations), and whether physical calibration actually improves on held-out stars. Reconstruction alone does **not** identify the physical interpretation of latent coordinates.

If adding dust, distance or elemental abundance latent components, audit degeneracies with flux scaling, temperature, metal line blanketing, and the photometric selection function.

### 5.3 Data architecture: unique star, sparse observations

**One Gaia source ID = one unique object record.** Use the unique star as the unit of splitting, weighting, and encoder presentation. Store XP and photometry once; store spectroscopic labels as a separate sparse relational table keyed by `source_id`, `survey_id`, `release_id`, `pipeline_id`, and measurement type. Keep multiple reductions of the same input spectrum identifiable so they are not naively counted as independent observations.

Suggested on-disk layout (HDF5, Parquet + HDF5, or Arrow; choose from existing stack and benchmark I/O):

```text
stars:
  source_id[N]                       uint64
  xp_coefficients[N, C]             plus XP quality/covariance references
  photometry[N, P], phot_err[N, P], phot_mask[N, P]
  input_metadata[...]               bandpass, flux conventions, etc.

survey_measurements (sparse, sorted/indexed by source_id):
  source_id[M], survey_id[M], release_id[M], pipeline_id[M]
  target_values[M, D], target_err[M, D], target_mask[M, D]
  optional_covariance[M, D, D]       only if genuinely available
  q[M, Q], quality_bits[M], field_definition_id[M]
  match_probability[M], provenance_id[M]

anchors (sparse, multiple methods may exist per star):
  source_id[A], anchor_method[A], anchor_source[A]
  target_values[A, D], target_covariance[A, D, D], target_mask[A, D]
  independence/correlation metadata, quality, provenance

astrometry:
  source_id, parallax_mas, parallax_sigma_mas, zp_recipe_inputs,
  astrometric_solution_type, astrometric_quality, spatial_sys_group

groups:
  source_id, cluster_id, membership_probability, group_source

splits and provenance:
  split_key, training/validation/test/anchor_calibration/anchor_test,
  crossmatch_version, label_definition_version, file hashes
```

A sparse label table is preferable to storing `(10^8,S,D)` dense label arrays with repeated missing values. **Batch collation** can still provide padded `(B,K,D)` survey tensors, `(B,K,D)` dimension masks, row/source IDs, and covariance blocks for active measurements. The loss operates on the *observed scalar subspace*, not on a single survey-level mask.

Guard against ambiguous or chance crossmatches, nonunique stars, multiple sources in crowded fields and binaries. Preserve missingness rather than replacing it with zero or an artificial label. Keep raw and quality-filtered views so rejected labels can be audited.

### 5.4 Split discipline and sampling

Use deterministic splits by Gaia `source_id`, and treat duplicate observations/releases as the same leakage group. For independent scientific validation, also create **cluster-wise**, **benchmark-family / observation-source-wise**, and where useful sky-region/quality-domain holdouts. Do not rely only on random star splits when physical calibrators share pipelines or measurement inputs.

Separate:

1. General train/validation/test stars;
2. Physically anchored **training** stars;
3. Independently anchored **model-selection validation** stars;
4. Unused **final benchmark test** stars;
5. A distinct **conformal calibration** sample if conformal intervals are attempted.

Do not prescribe `30%` benchmark holdout or `20%` cluster holdout as universal numbers: tens of fundamental benchmarks cannot support robust multivariate calibration over the entire stellar domain. Choose a fixed, preregistered split based on sample size, independence and coverage. Preserve all survey labels on the same star within the same partition.

For training, labeled stars are rare relative to unlabeled XP stars. Use a **controlled dual-stream sampler**: large unlabeled batches for MAE and stratified labeled batches for the marginalized label likelihood. Oversample poorly represented astrophysical regimes thoughtfully, while keeping validation on defined reference populations. Balance gradients/losses by per-star sampling weights, not by raw counts of millions of unlabeled observations or by pretending all survey records are independent stars. Overlap stars should receive adequate sampling but not unrestricted multiplication of weight.

### 5.5 Photometry, masked inputs and data leakage

Retain per-band availability and photometric-depth information; avoid making the representation quality limited by the shallowest survey. Test optically rich, IR rich, missing-data, and low-SNR input regimes separately. If SNR/quality of a survey **label** is used only by its observation head, that information need not be an encoder input.

Check whether any input feature is itself derived from the target spectroscopic label, a catalog distance, a reddening map with target-dependent training, or Gaia astrophysical-parameter products. Such leakage can make held-out label scores look artificially good. Store an explicit feature-provenance audit.

Gaia XP coefficients have correlated measurement errors; the reconstruction objective should use the available covariance/quality information where feasible, or document the simplification. Missing and masked coefficients/bands are not equivalent to zero flux. Cross-survey photometry may have correlated calibration errors.

### 5.6 Distances and parallax: a separate nonlinear observation model

Use distance modulus `mu_d = 5 log10(d/10 pc)` with

\[
\boxed{d(\mathrm{pc})=10^{(\mu_d+5)/5},\qquad
\varpi_{\rm true}(\mathrm{mas})=10^{2-0.2\mu_d}.}
\]

A corrected Gaia-parallax observation model is

\[
\varpi_{\rm obs}\sim
\mathcal N\left(10^{2-0.2\mu_d}+zp(\mathrm{colour},G,\beta,\mathrm{solution}),
\sigma_\varpi^2+\sigma_{\varpi,\rm sys}^2\right).
\]

Define the sign convention for `zp` explicitly; if the stored quantity is already corrected, never apply the correction again. Gaia's zero point depends on magnitude, colour, ecliptic latitude and astrometric solution type, and correlated spatial systematics may matter. Negative measured parallaxes are valid observations and must remain in the likelihood.

If `p(mu_d|x)` is broad, integrate the nonlinear mapping over it using quadrature/Monte Carlo rather than assuming `E[parallax]=parallax(E[mu_d])`. The first-order variance formula

\[
 \sigma^2_{\varpi,\mathrm{pred}}\approx
 (0.2\ln 10)^2\varpi^2\,\sigma^2_{\mu_d}
\]

is only a local approximation. For data beyond parallax sensitivity, a declared **distance/population prior** becomes important; report its influence. Extinction-distance-temperature degeneracies require monitoring.

**Double-use rule:** do not feed Gaia parallax into a direct distance predictor *and* treat the same parallax as an independent training-time observation target under a naive conditional likelihood. Choose input-feature conditioning versus forward generative supervision consistently, or use a coherent joint model. Derived distance catalogs based on the same Gaia parallax are **not independent distance labels**.

### 5.7 Ages: isolated, model-dependent prediction

Age has survey-, population- and isochrone-dependent definitions and strong priors. Do not put pooled catalog ages on the same footing as measured atmospheric quantities. Train a separate age posterior conditioned on stable stellar predictions and, if justified, luminosity/astrometry/abundances. Prefer a flexible conditional density/mixture/flow or age posterior summary over a Gaussian age MSE. Track the source's stellar-evolution assumptions and training priors. Avoid age-loss gradients redefining the physical temperature/metallicity scale until an explicit experiment justifies this coupling.

---

## 6. `torchregress`: reusable primitives, module contracts and tests

**Package principle:** implement generic probabilistic multi-instrument regression primitives, not Gaia-, APOGEE-, or star-specific code. Application-specific source registries, benchmark tables, units and selection functions belong in MSA. Proposed paths below follow the uploaded draft and **must be reconciled against the actual checked-out repositories before editing**.

### 6.1 Core modules

| Proposed module | Responsibility | Contract / tests |
|---|---|---|
| `utils/bspline.py` | `bspline_basis_with_derivative`, boundary modes | Value/derivative vs SciPy, finite differences, repeated knots, AMP, extrapolation |
| `constraints/calibration_heads.py` | Per-target fixed-weight `RelativeSurveySplineHead` | `delta[B,S,D]`, `J_delta[B,S,D,D]`; hard zero-sum per target, gradients intact |
| `constraints/calibration_heads.py` | `CommonScaleSplineHead` and `ConditionBiasHead` | `c[B,D]`, `u[B,S,D]`; `u(theta,q_ref)=0`; separately regularized |
| `losses/multi_source.py` | `MarginalizedMultiSourceGaussianNLLLoss` | Arbitrary per-parameter masks, optional anchor blocks; Cholesky-stable, correct constants |
| `losses/robust_multi_source.py` | `MarginalizedGaussianMixtureNLLLoss` | Exact logsumexp for small assignment sets; normalized probability model |
| `losses/forward_observation.py` | `NonlinearGaussianObservationNLL` / parallax adapter | Nonlinear expectation/integration, negative parallaxes, gradients, units |
| `losses/group_hierarchical.py` | Optional cluster consistency / latent-group model | Includes normalization/log variance, membership weights, intrinsic dispersion |
| `metrics/multi_source.py` | Survey residual/PIT/coverage metrics, per-regime stratification | Per-target coverage, held-out residual trends, no label leakage |

**Avoid one overgrown `ZeroSumBSplineCalibrationHead` that simultaneously learns survey offsets, condition biases, physical consensus corrections, and noise floors.** Composition makes constraints and identifiability auditable.

### 6.2 Suggested generic interfaces

```python
class RelativeSurveySplineHead(nn.Module):
    def forward(self, theta: Tensor) -> tuple[Tensor, Tensor]: ...

class CommonScaleSplineHead(nn.Module):
    def forward(self, theta: Tensor) -> tuple[Tensor, Tensor]: ...

class ConditionBiasHead(nn.Module):
    def forward(self, theta: Tensor, q: Tensor,
                q_ref: Tensor) -> tuple[Tensor, Tensor]: ...

class MultiSourceGaussianMarginalNLL(nn.Module):
    def forward(self, mu: Tensor, chol: Tensor,
                observations: "PackedObservations") -> Tensor: ...

class PackedObservations:
    # Conceptual typed structure; design for padded batch or ragged packed layout.
    # Contains observed y, mask/index, mapped means/Jacobians,
    # independent/correlated covariance blocks, source metadata.
    ...
```

**Interface semantics:** `mu`/`chol` are canonical **normalized coordinates**, while observed values and covariances must be converted into a declared common coordinate for the likelihood. A generic `ObservationMap` interface may evaluate mapped mean `h(mu)`, Jacobian `H`, and covariance `R` before passing them to the generic Gaussian marginal NLL. This prevents the loss module from hardcoding an identity map or star-specific field meanings.

For efficiency, choose either (A) pack all observed scalars per star into contiguous ragged arrays, or (B) pad on a per-minibatch maximum with a compact selected-subproblem implementation. Both must produce identical NLLs and gradients. Benchmark throughput and GPU memory before finalizing the interface.

### 6.3 Mathematical invariants: required unit tests

- **Gauge:** per target/dense grid, `sum_s w_sd delta_sd(theta) = 0` and `sum_s w_sd d(delta_sd)/dtheta = 0` to numerical precision; weights stay fixed, positive and normalized over the comparable subset.
- **Reference condition:** `u_s(theta,q_ref_s)=0` identically; random `q_s` does *not* need cross-survey zero-sum.
- **Physical-scale offset:** mock data where all surveys have +100 K bias, anchors require a -100 K shift from consensus; recover it using `c` without violating the relative-survey gauge.
- **Gauge limitation:** when anchors are removed, confirm that the absolute scale is convention-dependent; no test should claim physical recovery from overlaps alone.
- **Jacobian:** spline analytic vs autograd/finite differences; chain rule through normalization; no `.detach()` gradient break.
- **Marginal likelihood:** compare information-form and direct stacked Gaussian NLL and dense numerical quadrature for linear mock models; include the `K log(2pi)` term and correct determinants.
- **Missing labels:** arbitrary missing dimensions/surveys and anchor-only stars; duplicating a measurement mistakenly as two independent rows must demonstrably change likelihood (to catch provenance errors).
- **Noise:** PSD covariances, Cholesky jitter, floor limits, full 4x4 covariance, correlated blocks where supported, gradients in fp32/fp64.
- **Robustness:** mixture marginalized likelihood equals exhaustive sum of small synthetic mixture assignments; probability normalizes, logsumexp stable, outliers do not force huge canonical bias.
- **Parallax:** `mu_d=10 -> parallax=1 mas`; negative observed parallax finite; known zero-point sign; quadrature agrees with high-sample MC on broad posteriors.
- **Extrapolation:** spline functions finite and gradients defined at/outside boundaries; OOD flag is triggered and calibration uncertainty does not shrink outside support.
- **Gradient checks:** optimize synthetic recoverable star+survey examples end-to-end and confirm gradients reach encoder, canonical head, `c`, `delta_s`, and optional condition-bias parameters.

### 6.4 Design constraints

- Stabilize uncertainty parameters with a positive-diagonal Cholesky mapping and bounded floor regularization; keep floor estimation distinct from bias-calibration coefficients.
- Register new losses in the existing public registry only after unit tests; keep API backwards compatible with current Gaussian/uncertain-target functionality.
- Use configurable `D`, `S`, per-target survey masks and per-target weights; never assume all surveys publish four comparable quantities.
- Return per-star NLL and diagnostics in addition to aggregate reduction; dataset weighting and objective balancing remain application decisions.
- Add tests under `tests/constraints` and `tests/losses`; run local CI and bump package version when the API stabilizes, not before the scientific interface is settled.

---

## 7. Joint objectives, staged optimization and practical training

### 7.1 Primary objective

For survey labels and independent anchors under the shared latent, write

\[
\mathcal L_{\mathrm{meas}}
=-\sum_{i\in\mathcal I_{\mathrm{label}}}
\log\!\int p_\psi(\theta_i\mid x_i)
\left[\prod_{s\in\mathcal S_i}p_s(y_{is}\mid\theta_i,q_{is})\right]
\left[\prod_{j\in\mathcal A_i}p_j(a_{ij}\mid\theta_i)\right]
\,d\theta_i.
\]

The product assumes conditional independence of factors *after accounting for their documented shared effects*. Combine correlated factors into joint likelihood blocks instead. Compute with the linearized Gaussian analytic method for Phase 1; robust extension as needed. The prediction `p(theta|x)` is the **conditional distribution produced by the encoder**, not a free latent variable separately optimized for each star.

Add the pre-existing reconstruction objective as **auxiliary multitask training** rather than claiming an exact single generative joint likelihood of spectra and survey labels:

\[
 \boxed{\mathcal L
 =\lambda_{\rm rec}\mathcal L_{\rm MAE}
 +\lambda_{\rm meas}\mathcal L_{\rm meas}
 +\lambda_{\rm smooth}\mathcal R_{\rm smooth}
 +\lambda_{\rm scale}\mathcal R_{c}
 +\lambda_{\rm cond}\mathcal R_u
 +\lambda_{\rm aux}\mathcal L_{\rm distance/age/cluster}.}
\]

Here optional distance/age/cluster terms are separately gated; neither a separate `L_anchor` nor `L_labels` should double-count the same physical measurements already included in `L_meas`.

### 7.2 Training progression

**Phase A — Audit & reproduce existing MSA.** Version/checksum existing pooled-label baseline, input normalization, reconstruction quality, and label-selection policy. Build overlap graph and simple robust descriptive residual maps *without* fitting the end-to-end model. Verify sources/units and alpha compatibility.

**Phase B — Survey consensus.** Retain original reconstruction decoder; add canonical Gaussian head and low-capacity `delta_s` with `c=0` and initially `u_s=0`. Train on every star with one or more compatible survey measurements; ensure overlap stars constrain relative biases. This output is a **consensus**, not yet a physical calibrated scale.

**Phase C — Physical anchoring.** Add independent anchor likelihoods and restricted `c`. Jointly fine-tune canonical head and survey calibration functions; use shrinkage and held-out anchor validation. Add anchors gradually by target dimension and regime to discover conflicts. Do not change consensus weights to conceal anchor disagreements; weights define the gauge. Publish residual physically unanchored domains.

**Phase D — Noise and reliability.** Validate reported uncertainties and learned scatter floors; introduce normalized robust outlier mixtures if needed. Estimate calibration uncertainty with source-level bootstrap/ensemble or coefficient covariance. Extend survey mean bias to SNR/quality only if residual trends robustly demand it.

**Phase E — Distance, age and clusters.** Add correctly specified parallax likelihood and separate distance posterior; add prior-explicit age only after atmospheric scale is stable. Treat cluster homogeneity first as a diagnostic, then as weak hierarchical regularization if validated.

**Phase F — Research architecture.** Compare a physical structured decoder bottleneck, element-resolved abundance heads, flexible conditional posteriors, and other more expressive designs against the unchanged-decoder baseline on frozen test sets.

### 7.3 Stability and efficiency at survey scale

- Warm-start from existing SSL/MAE checkpoint. Initialize `delta`, `c`, and `u` to zero; initialize `V_i` and noise floors conservatively, with explicit lower bounds.
- Initially freeze most of the encoder while calibrating the common head and low-capacity survey functions. Then unfreeze progressively; recheck reconstruction metrics and original latent utility.
- Use balanced labeled/unlabeled training streams, controlled class/stellar-regime resampling, and instrument/source-aware minibatches so overlap stars occur often enough to calibrate relative offsets.
- Normalize losses **per unique star** with appropriate sampling correction; avoid treating multiple pipeline reductions as independent evidence. Tune loss mixing through gradient magnitudes and held-out metrics rather than arbitrary loss magnitudes.
- Accumulate likelihood statistics per star and survey domain; use distributed reduction carefully so rare anchors cannot be swamped by millions of unlabeled reconstruction examples.
- Expect sparse label observations to make the multi-source NLL cheap relative to encoder throughput; benchmark grouping/Cholesky kernels before premature optimization. Cache survey spline basis/Jacobians if warranted.
- Never initialize learned quality-bias terms from numeric categorical flag encodings. Use explicit embeddings or per-release indices; split shared releases/pipelines appropriately.

---

## 8. Clusters as constraints, without pathological variance inflation

Clusters can test metallicity trends with temperature, gravity and evolutionary phase. They do not automatically provide identical surface abundances for all member stars: diffusion, multiple populations, membership contamination and intrinsic abundance spreads matter, especially for globular clusters.

**Default:** use open/globular clusters for **held-out trend diagnostics** rather than penalizing every star toward an arbitrary fixed cluster mean. Measure residual `[Fe/H]` against `T_eff`, `log g`, location and survey, with membership probability and evolutionary-stage stratification.

If a cluster likelihood becomes useful, introduce latent cluster metallicity `m_c` and intrinsic spread `tau_c`, for example

\[
 \theta_{i,\rm FeH}\mid m_c,\tau_c
 \sim\mathcal N(m_c,\tau_c^2),
\]

with priors on `m_c`, `tau_c`, membership-mixture contamination, and proper variance/log normalization. A consistency term computed from predictive Gaussians must include the log-variance term; otherwise the network can reduce its penalty by inflating its output uncertainties. Even with that term, evaluate whether this additional objective double-counts information and changes the effective parameter/population prior.

Do not hardcode `sigma_intrinsic=0.02 dex` as universal, and do not claim a cluster loss forces residual trends uniquely into survey heads. Test alternative diffusion-aware or stage-conditioned models before strong cluster regularization.

---

## 9. Scientific evaluation, ablations and acceptance gates

### 9.1 Mandatory comparisons

- **A: Pooled SFT + reconstruction:** current MSA with pre-merged labels, reproduced as an immutable baseline.
- **B: Independent survey heads:** each survey predicts its released labels from the common encoder, with the same parameter count/optimization budget where practical; no common physical output without an extra aggregation rule.
- **C: Consensus hierarchical:** common `p(theta|x)` + gauge-constrained `delta_s`, no physical anchors; explicit consensus-only output.
- **D: Physically anchored hierarchical:** C + `c` + independently anchored likelihoods, the **proposed primary model**.
- **E: D + extensions:** robust mixture, quality biases, calibration uncertainty, structured decoder and/or element-resolved alpha, **one controlled addition at a time**.

### 9.2 Split-safe experiments

1. **Held-out *stars* from overlapping surveys:** train the global heads using other overlap stars. Withhold the selected test stars' labels from training; predict both canonical `theta` and each survey's observable labels. This is a valid transfer test.
2. **Hold out one survey measurement on overlap stars while retaining the star's other-survey labels:** useful to test conditional label reconstruction/overlap calibration *only if the test measurement is excluded from training and the intended evaluation conditions are stated*. It is not an independent test of inference from `x` if the model has already used the star's other label during training.
3. **Full leave-one-survey-out source:** without that survey's calibration head learned or external anchor calibration, predicting that survey's labels is not identified. An entirely unseen survey requires measured bridges/anchors or explicit adaptation. Do not claim zero-shot source calibration from a nonexistent head.
4. **External physical-anchor test:** final unseen benchmark/seismic/high-quality metallicity standards, separate from those used to fit `c`, with uncertainty-aware residuals.
5. **Population/domain shift:** metal-poor and extremely metal-poor regimes, hot/cool stars, dwarfs/giants, high extinction, large distance, low SNR, sky/crowding and photometry depth/missingness.
6. **Independent cluster test:** metallicity residual trend along evolutionary loci, not merely raw overall within-cluster dispersion.
7. **Calibration and posterior:** NLL, CRPS and componentwise PIT where independently validated truths are informative; interval coverage/sharpness; covariance-standardized residuals; residual correlations across parameters/surveys.
8. **MAE capability:** masked XP/photometry reconstruction, robustness under missing bands, representation transfer, and performance versus the pretrained encoder. The supervised extension must not silently damage the original foundation representation.
9. **Outlier and selection:** failure on crossmatch ambiguity, binary contamination, peculiar stars, survey quality flags and calibration extrapolation; reliability and abstention/OOD flag evaluation.

### 9.3 Acceptance criteria (define quantitative thresholds from baselines)

The primary model is accepted only if all hold:

- It produces **one common parameter distribution per star** regardless of which catalog provided its labels; only optional observation predictions are survey specific.
- It reduces **held-out** cross-survey systematic residual trends relative to A and B (binned conditional biases and robust dispersions, not only global RMSE).
- Physical anchoring improves or maintains external reference agreement without degrading previously good regions; where anchors are weak, uncertainty/support flags reflect that limitation.
- Posterior coverage is defensible against available independent physical calibrators and does not improve merely by indiscriminate variance inflation.
- The relative-bias zero-sum, quality reference-condition, unit, mask, and marginal NLL invariants pass all unit tests.
- The MAE reconstruction and useful latent representations are not materially degraded beyond a prespecified tolerated difference.
- Results are stable across seeds/overlap resamples and do not hinge on APOGEE or another single survey (test leave-one-source-out **calibration fits where the overlap graph remains connected**).
- No unsupported extrapolation is presented as physically calibrated; physical-scale and OOD flags accompany predictions.

Pre-register numerical go/no-go limits after reproducing A, not after inspecting the final test sample. Archive frozen model checkpoints, feature lists, random seeds, exact overlap/anchor splits and calibration reference definitions.

### 9.4 Conformal calibration

If desired, apply scalar or multivariate conformal methods to a **disjoint calibration sample** whose reference values and error distributions are understood. Ordinary split conformal gives marginal coverage under exchangeability, not guaranteed conditional coverage in rare stellar regimes or covariate shift. Tiny fundamental benchmark samples are insufficient for fine-grained multivariate or extreme-tail claims. Report empirical coverage per regime and caveat coverage when the calibration population differs from full Gaia XP.

---

## 10. Engineering execution by repository

### 10.1 `masked-stellar-autoencoder`: concrete backlog

| Priority | Work item | Candidate paths from prior draft | Completion condition |
|---|---|---|---|
| P0 | Audit schema, label definitions, provenance, overlap counts | `data/fine-tune-dataset-maker.ipynb`, new `scripts/map_survey_overlaps.py` | Versioned report with overlap graph per target/domain; no silent label merges |
| P0 | Star/survey sparse crossmatch builder | `data/build_multisurvey_h5.py` | Reproducible unique `source_id` joins; ambiguity & duplicate tests |
| P0 | Split and calibrator manifests | `pipeline/splits.py` | Deterministic leakage-safe tests/anchor/cluster splits |
| P0 | Sparse label dataset and padded collator | `training/finetune_data.py`, `hdf5_io.py`, `pipeline/batch.py` | Handles missing labels per parameter, partial anchors, covariances |
| P1 | Canonical Gaussian head preserving decoder | `models/model.py`, `models/blocks.py` | `mu`, `chol`, physical-unit adapters; original MAE path intact |
| P1 | Shared observation map + loss integration | `training/finetune_msa.py` | Gradients to encoder and survey biases; single latent marginal likelihood |
| P1 | Config/checkpoint versioning | `configs/finetune.yaml` and model schema | Target/source registry, weights, knots, anchors, scaler/version persisted |
| P2 | Anchor ingestion/calibration and reference reports | new `data/build_anchors.py`, `scripts/evaluate_anchors.py` | Independent reference sets, per-regime calibration report |
| P2 | Distance/parallax branch | `models/model.py`, `training/finetune_msa.py` | Correct units/sign/nonlinear likelihood and no target leakage |
| P2 | Uncertainty/domain diagnostics | `scripts/evaluate_multisurvey.py` | Coverage, residual trends, support and source sensitivity |
| P3 | Age, cluster, alpha-element and physical decoder extensions | modular training/model files | Each justified by separate ablation |

### 10.2 `torchregress`: concrete backlog

| Priority | Work item | Completion condition |
|---|---|---|
| P1 | B-spline values+derivatives and boundary policy | Analytic + numerical reference tests |
| P1 | Per-target relative calibration and shared/common correction heads | Exact algebraic invariants and correct Jacobians |
| P1 | Generic observation-map interface and packed masks | Arbitrary observed dimension selection, multiple sources/anchors |
| P1 | Marginal Gaussian NLL | Direct covariance equivalence and end-to-end gradient tests |
| P2 | Normalized robust source mixtures | Exact small-K test, performance profile |
| P2 | Nonlinear forward-observation likelihood with quadrature | Parallax test and general reusable interface |
| P3 | Hierarchical-group module | No uncertainty inflation, cluster mock recovery |
| P3 | Calibration uncertainties and diagnostics | Bootstrap/ensemble propagation & tests |

The earlier proposed directories are **integration targets rather than verified source-tree facts**. Coding agents should inspect actual package APIs, tests, registries, and data formats before modifying source. Preserve existing public interfaces wherever possible; avoid speculative renaming or premature version bumps.

### 10.3 Milestones and dependencies

```text
M0: Schema + label semantics + overlap/anchor diagnostics
  -> M1: generic spline/Jacobian + marginal Gaussian unit tests in torchregress
  -> M2: MSA canonical head + unchanged MAE + multi-source consensus fit
  -> M3: physical-anchor/common-bias fit and honest physical-scale catalog
  -> M4: uncertainty/OOD and cross-survey validation; frozen comparison to baselines
  -> M5: optional robust likelihood and nonlinear parallax/distance branch
  -> M6: only if warranted, alpha element vector, age, cluster regularization,
         physically structured decoder, flexible posterior
```

Keep distinct experiment IDs for consensus, physically anchored, and physically bottlenecked models. Do not block the atmospheric homogenization result on age/distance or on the structured decoder experiment.

---

## 11. Risks and explicit mitigation

| Risk | Symptom | Response |
|---|---|---|
| False absolute scale from overlap-only consensus | Inter-survey agreement but benchmark offset | Model `c` separately; label consensus vs anchored physical |
| Nonidentifiability in sparse domains | Large drift with seed/survey set | Restrict spline capacity, increase calibration uncertainty, flag unsupported domain |
| Survey quantities not comparable | Systematic alpha/metallicity trends persist | Audit definitions, element-wise maps, exclude incompatible columns |
| Noise-scale degeneracy | Huge posterior/survey scatter, improved NLL without better means | Priors/floors, repeat observations and independent coverage checks |
| Anchor circularity | Spectroscopic labels appear as supposedly independent standards | Method provenance, shared covariance blocks, independent held-out anchors |
| Source selection leakage | Excellent random-split test, poor new-survey/sky performance | Star/cluster/source/sky stratification and domain-shift tests |
| Correlated catalogs counted twice | Implausibly narrow uncertainty | Release/pipeline identifiers and joint/shared systematic likelihoods |
| Structured decoder shortcuts | `z_nuis` carries physics; `theta` ignored | Preserve original MAE first, intervention tests, bottleneck ablations |
| Out-of-support spline behavior | False high-confidence metal-poor predictions | Controlled boundary mode, OOD/support flags, growing scale uncertainty |
| Wrong parallax zero-point/distance mapping | Factors of 1000 or biased distance | Explicit unit tests and sign convention; retain negative parallaxes |
| Inconsistent loss/gradient implementation | Zero-sum fails with `q`, encoder stops learning | Separation of `delta/u/c`, no detach, finite-difference gradient tests |
| Cluster penalty encourages variance inflation | Wide posteriors, compressed cluster means | Proper normalized hierarchy, membership probabilities, use diagnostics first |

---

## 12. Reference sources and domain cautions

These are supporting scientific references, not substitutes for inspecting actual data-release field definitions and pipeline documentation.

1. **Gaia-ESO survey homogenization**: *The Gaia-ESO Survey: Homogenisation of stellar parameters and elemental abundances*, A&A 676, A129 (2023). Survey overlaps, benchmark stars, clusters and seismic calibrators; explicitly demonstrates practical compromises in shared calibration. https://doi.org/10.1051/0004-6361/202345910
2. **Fundamental benchmark temperatures and gravities**: *Gaia FGK benchmark stars: Effective temperatures and surface gravities*, A&A (2015). Interferometric/bolometric approach and limits of benchmark coverage. https://doi.org/10.1051/0004-6361/201526319
3. **Updated Gaia FGK benchmark set**: *Gaia FGK benchmark stars: Fundamental Teff and log g of the third version*, A&A (2024). Improved but finite and nonuniform parameter-space coverage. https://doi.org/10.1051/0004-6361/202347136
4. **Contemporary comparison**: Turchi et al., *Survey of Surveys II: Stellar parameters for 23 million stars*, A&A 700, A195 (2025). A useful external homogenization/prediction baseline, not a universal physical ground truth. https://doi.org/10.1051/0004-6361/202555695
5. **Gaia parallax zero point**: Lindegren et al., *Gaia Early Data Release 3: Parallax bias versus magnitude, colour, and position*, A&A (2021). Corrections depend on color, magnitude, position and solution type. https://doi.org/10.1051/0004-6361/202039653
6. **APOGEE DR17 definitions**: SDSS documentation, *Using APOGEE Stellar Parameters*. `[M/H]` versus `[Fe/H]` and parameter-dependent composite `[alpha/M]` sensitivity. https://www.sdss4.org/dr17/irspec/parameters/
7. **Gaia DR3 XP input statistics**: *Gaia Data Release 3: Processing and validation of BP/RP low-resolution spectral data*, A&A (2023), including continuous XP coefficients, errors and coefficient correlations. https://doi.org/10.1051/0004-6361/202243680

---

## 13. Final architectural contract for coding agents

**Implement a single physical/consensus latent predictor of stellar parameters; keep the existing MAE reconstruction path; train against every compatible, properly identified survey measurement by marginalizing the uncertain canonical parameters.**

The observation model must explicitly separate **common consensus-to-physical correction `c`**, **zero-sum relative survey correction `delta_s`**, **reference-centered quality correction `u_s`**, and **survey scatter/outliers**. Physical anchors are noisy observations and must join the same statistically coherent likelihood without double counting. Calibration and epistemic/systematic uncertainty must not disappear behind a single Cholesky predictor covariance.

**Milestone to deliver first:** a three-parameter (`T_eff`, `log g`, `[Fe/H]`) common predictor with fixed original MAE decoder, label-mask-safe exact *linearized-Gaussian* marginalized loss, audited APOGEE/GALAH/LAMOST/other overlap labels, held-out physical anchor validation, and a documented consensus-versus-physical-scale distinction. Add alpha only after defining it; add nonlinear distance, prior-dependent age, robust mixture likelihood, cluster constraints and physical decoder bottleneck only as separate, measured extensions.

**Success is not merely smoother differences between survey catalogues.** Success is a reproducible, calibrated, survey-independent stellar-parameter posterior, honest about where the physical scale is actually constrained and where it is not, that improves external physical validation without sacrificing the robustness and scalability of the pretrained masked stellar autoencoder.
