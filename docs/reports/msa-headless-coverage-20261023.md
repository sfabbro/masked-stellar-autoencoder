# CANFAR coverage and method definitions — 23 October 2026

52 new headless jobs were accepted, including failed predecessors and distinct repairs. Acceptance is not scientific verification. All models ran on CANFAR; local native-data smokes and the old local verification queue were stopped when requested. Local activity thereafter was code/unit checks, submission, metadata and file preservation.

## Coverage

- Four SLS/SLSConformal full/low-rank region fits, seeds45/46, reconstruction .03. SLS estimates joint sets, not a normalized PDF; NLL is not applicable.
- Fourteen matched native3 NSF objectives, clean/quoted-label-noise NLL, NLL+0.1contrastive, NLL+1contrastive at seeds47/48, plus pure-contrastive seed47 controls. Each scans10M reconstruction rows.
- Matched FP64 per-object Gauss–Newton and L-BFGS forward inversion on the same bounded subsets, two FIT-defined starts and150iteration caps. Method-specific stopping criteria differ; this is not a full-cohort or global-optimum claim.
- Three original FlowEB attempts, three versioned integration repairs and one remaining-resolution continuation. These compose a joint flow prior and forward likelihood; there is no invented public FlowEB class.
- Twelve frozen-parent age/distance Gaussian, MDN and NSF heads: native APOKASC age and StarHorse age/distance posterior targets, two seeds. These are source-specific reported-label/posterior estimands, not independent physical truth or raw-parallax likelihood.
- Four MAE-inspired TTT policies: encoder+decoder SGD3/20steps at20%masking, LayerNorm SGD20, and encoder+decoder SGD20 at75%masking. Prediction head fixed, fresh stateless XP masks, exact per-object reset. Hash512global TEST and the separate141metal-poor stratum are not pooled into a population estimate. Adapted calibration is separate.
- Existing DESIalpha projection, failed-and-repaired feature preparation, and six source-audited alpha heads. Counts sum52 including the retained failures.

## What NLL + contrastive means

The clean path uses torchregress `ContrastiveFlowLoss` around a normalized NSF. MSA combines likelihood with weighted InfoNCE: `NLL + lambda * contrastive`, lambda0.1or1, one source-distinct cyclic FIT-context negative, temperature1, margin0; the joint objective is divided by target count.

`ContrastiveFlowConformal` is a separate post-fit prediction-set calibrator. It does not replace the training objective or automatically account for measurement noise.

The noisy contrastive path explicitly integrates the SAME quoted target-error model under both positive and negative contexts before the likelihood ratio, retaining both gradients. This is an explicit MSA composition matching the clean public loss in the zero-error limit, not a claim that simply wrapping `ContrastiveFlowLoss` handles both contexts. Native target errors are diagonal; XP/input noise and physical-truth covariance are different quantities. Catalogue posterior widths are not blindly treated as independent measurement errors.

## TTT and the paper

The previous three-step LayerNorm/Adam/L1 procedure was not the paper's recipe. The new four jobs test masked-MSE/SGD adaptation with fixed heads and per-object reset. The paper uses an MAE-pretrained encoder/decoder and separately trained fixed task head, whereas these jobs reuse an MSA model with a jointly trained Gaussian head. They are tabular adaptation tests, not an exact image-MAE reproduction. No benefit is assumed in advance.

Reference: [Test-Time Training with Masked Autoencoders](https://arxiv.org/abs/2209.07522), especially Method and Implementation Details.

## Alpha source and submissions

All6372607original source rows passed association/quality/unit checks. The prepared feature set has2715FIT/969TUNE/490CAL/433TEST. Alpha stays signed native DESI alpha/Fe in dex; it is not APOGEEalpha/M. DESI warns that alpha belowFeH−2 is not recommended: that regime is excluded fromFIT/TUNE and only diagnostic in evaluation. The six direct-label baselines preserve quoted errors; no physical deconvolution claim is made.

- alpha_desi_gaussian_s45_v1: `fch6tul4`.
- alpha_desi_mdn_s45_v1: `sgt10wv3`.
- alpha_desi_nsf_s45_v1: `jl6lpjmy`.
- alpha_desi_gaussian_s46_v1: `e07j1vvv`.
- alpha_desi_mdn_s46_v1: `dn0712bu`.
- alpha_desi_nsf_s46_v1: `ds8g30wi`.

## Failures retained and repair scope

Original FlowEB512node integration failed at .04012/.03302 versus the fixed .01 bound. Seed42 passed the stronger adaptive gate at16384 nodes; seed43 passed at32768 with two successive refinement errors .000714/.002234 and independent-scramble error .001495. Both MAP predictions remain a separate downstream gate. The posterior importance-convergence failure remains unresolved; no successful normalized/global posterior is claimed. Priors and decoders were not refitted by these repairs.

Alpha feature preparation initially failed on a local-staging checkpoint path. Its distinctv2 job uses the canonical SHA-pinned ARC parent and passed the full source audit. Original failure evidence remains preserved.

## Publication and result preservation

Torchregress `6bd8b93b35fe4fe216319870e72e94eb78429106` is committed and published on `origin/codex/density-cdf-api-20261023`; main was preserved.4038tests passed/730skipped, lint/type checks/strict docs passed, and six bounded synthetic MSA head API checks passed. Unrelated `Claude outputs/` remains untracked.

MSA sources are a361552(first40, library1d735), e71a79e and8d4d8fa(repairs/alpha, library6bd8b93). Latest source gate885passed/4skipped plus17focused checks. No local dependency installation occurred.

All315required first-batch artifacts and all six alpha heads'18artifacts+6manifests are preserved with SHA/NPZ CRC checks. Actual library/source pins were checked for alpha. This is preservation and producer verification, not independent full predictive-science verification. New rankings require remaining source/numerical/objective/conformal checks on CANFAR and paired global/metal-poor reports with visual QA. Historical strict failures are not erased.

The authoritative acceptance IDs are in `experiments/xp/launch-stellar-improvement-20261015.json`. Durable state is in `experiments/xp/results-stellar-improvement/coverage-wave-20261023/current-handoff.json` and `alpha-wave-20261023-v2/preservation/state.json`.
