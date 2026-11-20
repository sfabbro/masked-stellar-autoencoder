# Unbiased stellar parameters: architecture, loss and validation (2026-11-20)

Status: design proposal after the v7–v9 rounds. Nothing here is launched. Numbers come from
the frozen held-out [Fe/H] core (7.7k stars with a high-resolution reference, 646 below −2;
`analysis/frozen_core/` on arc), the 18k held-out wide-binary pairs, and the 130k-star test
split. Scripts and outputs: `/arc/projects/astroai/msa/analysis/`.

## 1. What the evidence says

| finding | evidence |
|---|---|
| Every **posterior** head is pulled toward the training population, whatever its family | Student-t, 3-component Student-t, 3-component Gaussian (full and diagonal): bias below −3 between +0.54 and +0.89 dex; mixtures do not move the median (e01–e03) |
| The pull **grows with noise** | v7 head, [Fe/H] < −2, held-out: +0.10 (G < 13.5), +0.33 (13.5–15), +0.41 (15–16), +0.37 (G > 16) |
| Re-weighting the population only moves the prior | balanced sampling (d02, v8): −0.03 to −0.09 dex in the metal-poor bins, at a cost in bulk precision; a different estimand, needing a population correction |
| **The latent already holds the information** | likelihood (ratio) head on frozen v7 latents, held-out: bias +0.11 / −0.01 / +0.03 below −3 / −3..−2.5 / −2.5..−2 (three seeds: ±0.02), Neyman coverage 0.59–0.76 per bin for a 0.68 target, bulk NMAD 0.119 vs 0.142 |
| Likelihood heads need **no re-weighting** | θ-balanced sampling (power 0.5, 1) leaves the bias unchanged (all bins within ±0.06) and raises the scatter below −3 from 0.25 to 0.40–0.46 (over-fitting the few metal-poor stars) |
| The **faint end** fails for both heads | [Fe/H] < −2 at G > 16 (84 stars): ratio MLE +0.25 / NMAD 0.53 (vs ≤ ±0.04 up to G = 16); bulk at G > 16: NMAD 0.22 (MLE), 0.26 (head) |
| Auxiliary constraints added as **separately weighted loss terms** destabilise the scale | the binary pool was a mean over a 10 %-of-batch sample added with weight 1 (each pair ≈ 10 labelled stars). Fixed v7 targets (v8): scale −0.05 dex; the model's own primary (v9): −0.17 dex, though the binary offsets became 0.00 |
| Current best | v8 without the pool: matches v7 on Teff (IRFM NMAD 36.7 K), E(B−V) and bulk [Fe/H] (test NMAD 0.114); best metal-poor selection (95 stars at P > 0.9, 95 % pure) |

The unlabelled reconstruction pool is already a uniform 3 % sample of the XP catalogue
(6.36M stars, 5M used, weight 0.2), so it already contains mostly faint stars never selected
for spectroscopy, and the encoder was pretrained on the whole catalogue. More unlabelled stars
of the same kind are therefore not the obvious missing piece; §5 tests what is.

## 2. Diagnosis: three different problems

1. **Estimand.** A network trained to predict labels learns p(θ | x) under the training
   population (and its selection). For a rare, noisy star the correct posterior median *is*
   pulled toward the bulk. No density family fixes that; only changing the estimand does.
   We want a per-star measurement: the likelihood p(x | θ) (or a statistic with frequentist
   guarantees conditional on θ). Population studies then put their own prior or a
   hierarchical model on top (interim-prior logic: Hogg, Myers & Bovy 2010).
2. **Domain shift in noise and nuisance.** A likelihood learned from labelled stars is
   p(x | θ) averaged over the *labelled* stars' distances, extinctions and S/N. Labelled stars
   are bright; most targets are not. Up to G = 16 the shift is harmless (MLE bias ≤ 0.04); above
   it the estimate breaks. Either the information is not there at that S/N (an XP limit), or
   the likelihood was never trained there (a supervision gap).
3. **Label systems.** Survey labels are mapped to a common scale anchored by IRFM Teff, seismic
   log g and the high-resolution [Fe/H] core. [Fe/H] below ~4500 K is a gauge freedom (nothing
   anchors it). Constraints that close it (binaries, clusters) must enter as *data*, with the
   same per-record weight as every other record, never as an extra weighted loss or with
   model-made labels.

## 3. Architecture

```
x (XP + photometry + astrometry, with errors) ──► encoder E (pretrained masked AE, 34M params)
                                                     │ z (256)
                         ┌───────────────────────────┼─────────────────────────────┐
                         ▼                           ▼                             ▼
              likelihood head s(θ, z)        proposal head q(θ | z)          reconstruction
              CNRE: s = log p(z|θ) + c(z)    Student-t (current head)        (unlabelled +
              PRIMARY PRODUCT                interim prior stated;            labelled stars)
                                             init / convenience only
```

- **Encoder.** Keep the backbone (the latent holds the information; §1). Change only what
  it is trained on (§4). Covariance-aware reconstruction (the XP correlation matrices are
  on arc; E5 in `plan-20261022/07`) is the principled way to make it noise-aware and stays on
  the list, conditional on §5.
- **Likelihood head (primary).** `torchregress.losses.NeuralRatioLoss(criterion="cnre")`.
  Parameters θ = (Teff, log g, [Fe/H]) first, then E(B−V) and parallax as nuisance parameters
  (profiled, not marginalised under the labelled stars' distribution, which is exactly the
  domain shift of §2.2). For speed the head embeds z once per star, e(z), and evaluates a small
  network on [θ, e(z)] per candidate.
- **Proposal head.** The current Student-t head, trained on the same latent. It initialises
  the MLE search and gives an "interim posterior" product whose prior is the stated training
  population. It must not be the published parameter.
- **Survey maps.** Unchanged: they calibrate survey labels onto the common scale. The
  likelihood head trains on the high-resolution core plus survey labels mapped to the common
  scale. Survey label noise (0.05–0.15 dex) widens the learned likelihood slightly; that is an
  errors-in-variables term to handle later, not now.
- **Binaries and clusters.** Validation by default. If used in training, a companion receives
  its primary's *survey records* (one record per companion, map evaluated at the primary's
  parameters, same likelihood as all records): no separate weight, no model-made label, no
  moving target.
- **Inference and calibration.** `torchregress.inference.maximize_score` (batched multistart
  projected Newton) for the MLE, `profile_interval` for profile intervals, and
  `NeymanConfidenceSet` with critical values learned on held-out labelled stars as a function of
  θ *and* G (coverage must hold per S/N, not only per θ). Metal-poor selection becomes a
  likelihood-ratio test of [Fe/H] < −2.5 with a false-positive rate controlled per θ, not a
  prior-dependent probability.
- **Outputs per star.** MLE, profile 68/95 % intervals, Neyman 68/95 % sets, observed
  information at the MLE (a Gaussian likelihood summary for hierarchical population work), a
  goodness-of-fit / out-of-support flag, and the interim posterior with its prior named.

**Alternative for the faint end: a data-driven forward model.** Zhang, Green & Rix (2023)
fit a network that predicts XP coefficients from (Teff, log g, [Fe/H], extinction, distance)
and evaluate the likelihood with the published XP covariances. That likelihood is exact in XP
space, so faint stars are handled by the noise model rather than by the labelled stars'
S/N distribution. It needs every nuisance modelled explicitly and does not reuse our encoder.
Laroche & Speagle (2023, 2024) show a label-independent generative model of XP with intrinsic
scatter. If §5 finds a supervision gap the ratio head plus augmentation suffices; if it finds
that the encoder loses information at low S/N, the forward model is the principled fix.

## 4. Losses

L = L_CNRE(labelled, natural population) + λ_rec · L_rec(unlabelled and labelled)
  + L_maps(survey records, as now) [+ L_q(proposal head, stop-gradient into the encoder)]

- **No population re-weighting** in L_CNRE: the population cancels in the ratio, and
  re-weighting only adds variance (§1).
- **Auxiliary information enters as data.** Each record, star or pair has its natural
  per-sample weight. No term gets its own multiplier to "make it count".
- **S/N degradation augmentation** for the likelihood head: add Gaussian noise to a bright
  labelled star's inputs so that its errors match a fainter magnitude (error-column scaling,
  diagonal first, full XP covariance later). The likelihood is then trained, with known θ,
  where the targets are. This is the labelled counterpart of "more faint unlabelled stars".
  Only *adding* noise is valid.
- **λ_rec** is chosen on the representation tests of §5 (latent invariance under degradation,
  faint-end metrics), not on training loss.

### 4.1 Why the earlier covariance experiments did not help, and where the covariance belongs

What was run (records: `docs/xp-covariance-experiments.md`, `experiments/xp/monitor/
compare_covariance.py`, progress log 2026-10-22; the run outputs were lost with the old
`msa_runs` tree, only the verdicts survive):

- **XP reconstruction** (epoch-27 continuation, 20M presentations, 612 steps, single seed): a
  diagonal Gaussian NLL with `v = 0.1 + σ²` and a per-band full covariance
  `V = diag(σ) R diag(σ) + 0.1 I`, against the MAE control. "Neither improves XP coefficient MAE."
- **Label heads** (W1): full-covariance Gaussian 71.5 K / 0.147 / 0.098 dex, held-out NLL −4.16;
  rank-2 low-rank Gaussian 69.8 K / 0.145 / 0.097, NLL −4.72, tail 0.86; diagonal 70.3 K /
  0.143 / 0.097, NLL −4.69, tail 0.79. Noise-aware NSF and corrected-score EIV: worse point MAE.

Reasons, from implementation to fundamentals:

1. **The variance floor erased the noise model.** `0.1` (variance, RobustScaler units) is added
   to every coefficient. Wherever σ² ≪ 0.1, i.e. most coefficients of all but the faintest
   stars, the diagonal and full-covariance losses are a constant-weight squared error. The
   experiment mostly compared MSE with MAE, not "with noise model" with "without". The floor
   should be fractional (proportional to the coefficient amplitude: calibration errors scale
   with flux) and learned (a model-scatter term, as in the scatter-VAE of Laroche & Speagle),
   not an absolute constant.
2. **The metric was the control's own loss on noisy targets.** MAE against noisy hidden
   coefficients rewards the MAE-trained control by construction, and rewards reproducing
   noise. A noise-aware objective that down-weights noisy entries *should* score worse there.
   The right metrics are downstream (labels, especially faint), whitened-residual calibration
   (χ²/dof per G bin), or MAE on high-S/N entries only.
3. **Optimiser state and length.** Switching objective while inheriting the MAE run's AdamW
   moments mis-scales every step (MAE gradients are bounded, Gaussian-NLL gradients are not);
   612 steps barely move a 34M-parameter encoder.
4. **Correlated noise leaks through partial masking.** With independent 20 % masking, visible
   coefficients carry noise correlated with the hidden targets; the network can predict part
   of the hidden *noise* from the visible inputs, and the marginal `V_SS` is then the wrong
   covariance (the conditional is `V_SS − V_SV V_VV⁻¹ V_VS`). Whole-band masking avoids it.
5. **The full-covariance label head was parameterised in the slow form.** It predicted the
   covariance Cholesky factor (exp diagonal). torchregress later found that this form does not
   even fit the training data on a 16-target benchmark and switched the default to the
   precision factor (log-density linear in the factor). A full covariance nests the diagonal,
   so a fitted one cannot lose 0.5 nats to it on held-out data: −4.16 vs −4.69 is an
   optimisation failure, not evidence against covariance.
6. **Fundamentals (expected, not bugs).**
   - With a correct model, weighting by the true noise covariance is efficient
     (Gauss–Markov / Cramér–Rao). With model error larger than the noise (bright stars: the
     decoder cannot reproduce XP to the precision of their errors), inverse-noise weighting
     concentrates the loss on the stars and directions the model fits worst. GLS with a
     misspecified covariance can be worse than unweighted least squares.
   - Published XP covariances describe random errors, not calibration systematics, which
     dominate for bright stars; hence a fractional floor.
   - In a *representation* loss, inverse-variance weighting moves capacity toward bright stars
     and low-order coefficients, the opposite of what the faint end needs.
   - Low-rank approximations keep the largest-variance directions, but a Mahalanobis or
     likelihood term is governed by the *smallest* eigenvalues (the precision). Truncating the
     covariance approximates the precision worst exactly where it matters; approximate the
     precision directly (or low-rank plus a well-chosen diagonal, in precision form).
   - Point estimates (means) barely depend on how residual correlations among a few labels
     are weighted; covariance heads were never expected to improve point MAE, only calibrated
     joint regions.

Where the covariance belongs:

- **In the likelihood, with model scatter:** `Σ_tot = Σ_XP + Σ_model`, Σ_model learned and
  fractional (low-rank plus diagonal, precision-parameterised), as in a forward model of XP or
  a scatter-VAE. That is where a noise model can only help.
- **In S/N degradation augmentation (§4):** inject noise with the *full* covariance difference
  to simulate a fainter version of a bright star. Correlated noise is part of what makes
  faint XP faint; diagonal injection under-represents it. Cheap to test inside S2
  (diagonal vs full injection).
- **Not as an inverse-variance weight in the encoder's reconstruction loss**, unless the loss
  has a learned fractional scatter term, fresh optimiser state, whole-band masking and
  downstream metrics.

## 5. Validation protocol (fixed before any run)

One script, `scripts/evaluate_model.py` (to consolidate the scratch tools), producing one table:

- **Bias conditional on the truth.** Median offset and NMAD per reference [Fe/H] bin, crossed
  with G bins (≤ 13.5, 13.5–15, 15–16, 16–17.65). The G > 16, [Fe/H] < −2 cell is the target.
- **Coverage per θ and per G.** Binned and smooth diagnostics
  (`torchregress.inference.binned_conditional_coverage`, `smooth_conditional_coverage`).
- **Anchors.** IRFM Teff, seismic log g, Gaia benchmark stars, E(B−V).
- **Consistency.** Held-out binary offsets by companion Teff; open-cluster giant − dwarf;
  globular clusters against Harris.
- **Selection.** Purity and completeness of metal-poor selection against the reference.

Decision rules are written next to each step below and not changed after seeing results.

## 6. Compute

- **Training.** The likelihood head adds K + 1 = 9 small head passes per labelled star; the
  encoder pass is shared, so the cost is close to the current fine-tune.
- **Inference.** Measured with `maximize_score` and three `profile_interval` calls: about
  1 min per 10k stars (P = 3) on 4 CPU threads, mostly candidate scoring with a generic MLP.
  Embedding z once per star removes about 40 % of that; on a GPU, 220M stars become an
  O(10) GPU-hour job (to be benchmarked, not assumed).
- **Storage.** About 100 bytes per star (MLE, intervals, information matrix, flags).

## 7. Decisive steps, cheapest first

| step | what | cost | decision rule |
|---|---|---|---|
| S1 | 3-parameter ratio head (Teff, log g, [Fe/H]) on **frozen v8-without-pool latents** of all labelled table rows (core + mapped survey labels, ~5.9M stars); MLE, profile and Neyman via `torchregress.inference`; full §5 table | one GPU job (encoder pass over ~6M rows, head training, batched MLE/profile; CPU would take hours to days) | adopt the likelihood product if, up to G = 16, \|bias\| ≤ 0.10 in every [Fe/H] bin, Neyman coverage 0.68 ± 0.08 per θ and G bin, bulk NMAD ≤ posterior + 0.01; Teff/log g no worse than the head on the anchors |
| S2 | **S/N degradation** of bright labelled stars to G = 16–17.6 noise (diagonal and full XP covariance injection), through the frozen encoder, scored with the S1 head and compared with real faint labelled stars | one GPU job, frozen encoder | degraded-bright ≈ real faint ⇒ information or representation limit ⇒ S3b. Degraded fine but real faint biased ⇒ domain mismatch (extinction, distance, selection) ⇒ add nuisance parameters, S3a |
| S3a | one fine-tune: CNRE head end-to-end + degradation augmentation, encoder recon on the current unlabelled pool | 1 GPU run | G > 16 metal-poor bias halved without loss elsewhere |
| S3b | covariance-aware reconstruction on a magnitude-stratified XP sample, or the XP forward model (§3) | design first | as S3a |

**Stop:** full-data runs to compare head families or loss weights; extra weighted loss terms for
auxiliary data; posterior-only products presented as parameters.

## References

- Zhang, Green & Rix 2023, MNRAS 524, 1855, "Parameters of 220 million stars from Gaia BP/RP
  spectra" (arXiv:2303.03420).
- Laroche & Speagle 2023 (arXiv:2307.06378) and 2024, ApJ (arXiv:2404.07316), "Closing the
  stellar labels gap".
- Andrae, Rix & Chandra 2023 (arXiv:2302.02611), XP metallicities with an added very-metal-poor
  training set.
- Miller, Weniger & Forré 2022 (arXiv:2210.06170), contrastive neural ratio estimation.
- Dalmasso et al. (arXiv:2107.03920), likelihood-free frequentist inference (LF2I);
  Masserano et al. 2023 (arXiv:2205.15680), WALDO.
- Hogg, Myers & Bovy 2010, ApJ 725, 2166, hierarchical inference with interim priors.
- Earlier notes: `plan-20261022/05`, `07`, `10`, `forward-eiv-screen-20261018.md`.
