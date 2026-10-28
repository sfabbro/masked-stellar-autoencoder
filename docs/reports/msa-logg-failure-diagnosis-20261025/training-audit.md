# Read-only log g training audit

No models/checkpoints were executed or loaded, no CANFAR jobs/automations were launched, and no repository files were changed. Source/config and already-preserved numeric arrays/JSON only.

## Most supportable training-side finding

The large experiment is **not trained on hundreds of millions of distinct labelled stars**. The 450M budget is three reconstruction passes over the existing 150M cache. Current supervised log g FIT is only **19,236 unique identities**, reused repeatedly. The expansion deliberately changed TEST only, leaving FIT/TUNE/CAL byte-identical. Thus larger TEST reveals a generalization problem; it did not improve supervised coverage. This is a concrete coverage/training-design limitation, not proof of corrupt labels or a numerical bug.

Evidence:
- `configs/five-wave-training-20261024.yaml:14`–29 declares 150M reconstruction rows, 450M presentations, eight supervised batches/update, encoder LR1.25e-6, recon weight0.03, 25% XP dropout, and macro6 half-natural/half-XP-masked TUNE choice.
- `experiments/xp/results-stellar-improvement/five-job-large-wave-20261024/preparation-latest-manifest.json` observed_per_role_target.FIT = [19236,19236,19236,2279,2827,2827].
- `scripts/prepare_five_job_source.py:91`–99 explicitly preserves every FIT/TUNE/CAL field byte-identically.
- Actual final manifests record **26,196,040 log g presentations**, averaging **1,361.824 draws per unique FIT log g star** over the whole training run (includes head warmup). These are repeated presentations, not independent labels. Same count for all4 arms.

## Directly measured support in the original native FIT parent

Local exact pinned `multi_sls_full_r03_s45_10m_v1/paired-cohort.npz` has **19,299** original native FIT stars: 14,610 GALAH, 4,689 APOGEE. Its log g support is extremely thin above4.75: **7 total** (5 in[4.75,5),2>=5); **zero below0**. It has only **25 Fe/H<-2** stars. This is the source parent of final FIT, which is reduced to19,236 by63 cross-source exclusions. **These bin counts are upper bounds for final FIT, not exact final counts.** `/tmp/msa-logg-parent-fit-distribution.json` retains the full histograms, quantiles and source counts for parent FIT/TEST. Parent log g 95th/99th percentiles=4.51697/4.61396. Thus the most extreme high-gravity tail is extrapolation. The broad original FIT/TEST histogram shapes are similar; this does not by itself explain all bad regions near4–4.5.

## Sampler and loss weighting

`run_final_six_recipes.py:50`–60 chooses48 candidates for each *label availability*, then unique-concatenates. It does **not** balance gravity bins, HRD strata, label source, Fe/H, uncertainty, or modality. Repeated stars in abundant regions remain abundant; rare tails are not deliberately upsampled.

`final_six_recipes.py:154`–180 sums per-coordinate NLL (and optional per-label InfoNCE), then divides by total observed coordinates. This is not macro per-label equal weighting. Actual final count fractions are Teff23.94%, logg23.94%, FeH23.94%, alpha4.96%, age11.61%, distance11.61%. **Do not claim global log g is underweighted**: it is ~24% of coordinate terms, although numerical gradients also depend on scale/errors and cannot be inferred from counts alone.

## Selection and optimization

The selected checkpoint minimizes global macro6 median MAE averaged half natural and half masked-XP TUNE (`run_five_wave_training.py:220`–263; selection at1033–1040), not local log g residuals or gravity-tail performance. Saved JSON shows NSF selected50M, whereas its normalized *natural* log g TUNE MAE improves .112203 at50M to .110174 at150M (~1.8%); mixed .113918→.111579. This is a real selection tradeoff, too small alone to explain gross local residuals. Gaussian selected150M; later natural log g deteriorates mildly while masked-XP improves.

Encoder gradients are globally clipped at1 (`run_five_wave_training.py:989`–994). Saved all13734-step metrics show100% clipping in each density arm; median preclip encoder norms Gaussian55.45, NSF85.65, mixed82.99 (quantile94.98%,median4.91). **This is a diagnostic, not evidence clipping caused the errors or reconstruction dominated.** Separate supervised/recon gradient norms and angles are not saved, so their relative influence cannot be established from logged losses/norms. Adam's behavior is not equivalent to simply dividing learning rate by the clipping factor.

## Source/measurement semantics

Known GALAH log g is parallax-conditioned (`configs/joint-native-preparation.yaml` notes; native preparation manifest field). Current input policy is W1,W2,G,BP,RP+110XP; parallax is not an input (`prepare_six_target_wave.py:296`–303 and `run_five_wave_training.py:807`–813). Thus exact recovery of the reference scale can have an information mismatch, especially without XP; not proof the labels themselves are wrong. No source-conditional offset model is used. Parent FIT quoted log g error medians differ: APOGEE.02584dex versus GALAH.10435dex; the likelihood includes these as measurement error, but that does not model systematic source-scale differences.

## Checks that did not reveal an obvious implementation bug

Target names/order and immutable prepared source/scaler/feature-order contracts are checked on load (`run_five_wave_training.py:124`–176). Only age/distance are log-transformed; log g stays native dex, is standardized with FIT-only mean/std, and quoted error is divided by the same std (`final_six_recipes.py:25`–70). Masks and finite-safe missing targets are used consistently. Gaussian variance is intrinsic variance plus quoted-error variance; scalar NSF includes quoted-error convolution. I found no direct off-by-one, log-g exponentiation, units, mask-broadcast, or source-target order error in this path. This is static/source review, not a fresh encoder/head numerical replay or independent full training trajectory certificate.

## Next diagnostic priority (no launch performed)

Use the parent's saved same-star XP vs masked-XP and Andrae log g bins to locate the failure. If bad regions persist with XP and Andrae succeeds on those same labels, label corruption alone is unlikely. Check shared features/target distributions and supervised source scales; expand/stratify legitimate FIT labels only in a subsequently authorized training design. Do not launch another head sweep or just increase reconstruction epochs on this evidence.
