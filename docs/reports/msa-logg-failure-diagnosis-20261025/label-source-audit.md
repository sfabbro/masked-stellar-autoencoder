# Read-only log g label/source audit

No native model/checkpoint execution, no jobs/automations/network acquisitions, no repository edits. Examined source/config/proof metadata and completed metric CSV only.

## What is verified

- Expanded native labels are the same hash-bound native sidecar used by training (full-label-source-preparation-20261025-v2.yaml native SHA3998...). Expanded logg is not silently copied from StarHorse/Andrae or exponentiated: `prepare_full_label_evaluation.py:117-136` reads targets/value, errors, original survey string, and retains columns0/1/2 unchanged.
- Original projection uses GALAH `logg` and APOGEE `LOGG` directly in dex; native source IDs/rows and sourceFeH/errors are compared exactly (`src/masked_stellar_autoencoder/training/joint_native_labels.py:112-130`). Masked invalid values remain raw but never become observed labels.
- GALAH quality requires flag_sp==0 and flag_sp_fit==0. APOGEE requires ASPCAPFLAG==0,STARFLAG==0 and metadata-located parameter flags excluding bad/fixed bits (`joint_native_labels.py:132-161`). These are explicit quality cuts, not mere finiteness.
- Existing independent original-source audit reports PASS on all original source target values/errors/flags/rows and masks: `experiments/xp/results-stellar-improvement/joint-native-preparation-v2-quality-v1/native-label-quality-independent-verification.json`, with531918completeGALAH/185859completeAPOGEE rows. This is provenance/mask proof, NOT proof astrophysical labels are unbiased.
- Expanded assembly joins integer sourceIDs by sorted search and assigns values/errors/source codes in the same selected/destination indices (`prepare_full_label_evaluation.py:466-490`). No sourceID float conversion identified.

## Concrete mismatch / limitation

GALAH logg is explicitly parallax-conditioned (`joint_native_labels.py:168`, materialize_joint_native_labels.py limitations). But native six-target FIT inputs allow only W1,W2,G,BP,RP and110XP coefficients (`prepare_six_target_wave.py:296-305`). Expanded feature lookup applies the same photo/XP-only allowed mask (`evaluation_feature_lookup.py:211-214,248-254`): PARALLAX is masked. A true no-XP prediction therefore has only five apparent magnitudes to infer a parallax-informed gravity target. Photometry plus apparent brightness does not identify the dwarf/giant distance/luminosity degeneracy. This is an explicit input/target estimand and modality limitation, not an accidental wrong-column proof.

No label homogenization/survey calibration is implemented. GALAH and APOGEE provide different native scales, with GALAH parallax conditioning. The source table differences below are evidence that source and modality must be separated; not a causal source-scale test on matched same objects.

## Actual global logg MAE (dex) by source/modality

Gaussian:
- XP all639531:.12230; GALAH479616:.10907; APOGEE159915:.16199.
- Genuine noXP all30534:.32056; GALAH30262:.31983; APOGEE272:.40111.

Other heads:
- Quantile XP.12606/noXP.29045.
- NSF XP.12388/noXP.32586.
- Mixed XP.12368/noXP.32876.

The noXP failure is shared across all heads, not unique to flow likelihood arithmetic, and true noXP evaluation is99.1%GALAH (30262/30534). Overall XP errors also mix source distributions: APOGEE higherMAE thanGALAH in every head. These marginal source differences may reflect different stellar domains as well as scale/quality and cannot isolate either explanation.

A fresh runnable consistency check `/tmp/msa-logg-label-source-check.py` confirms source counts sum toall_sources and weighted per-source MAE/bias reproduce global numbers for all8arm/modality cases to1e-12. It is table consistency, not independent per-star reconstruction.

## Training/evaluation cohort evidence

Actual source preparation manifest reports only19236observedFITlogg,2116TUNE,2046CAL versus101191originalTESTlogg; expanding TEST adds labels but preserves FIT/TUNE/CAL bytes exactly. Source policy explicitly append_test_only and preserves historicalsmallFIT. Thus “4.41Mevaluation stars” and450Mreconstructionpresentations do not mean large supervised gravity coverage. The training audit agent is checking actual sampling/density coverage separately.

The source preparation manifest itself states `no_xp_caveat`: model trained onXP-richinputs, noXP evaluation modality-shifted. Artificial25% whole-XP masking is a useful stress augmentation but does not guarantee the real noXP stellar population is exchangeable with FIT.

## Not established / next discriminant

No wrong-logg-column, unit conversion or sourceID misalignment was found in this bounded audit. Do not assert labels are clean merely because original values were faithfully projected. Need logg-binned paired residuals and catastrophic fractions on same source-qualified stars; inspect high-error objects and calibration/flag/evolution-stage/source mixtures. Match FIT/TUNE/TEST distributions in logg/Teff/FeH/G/XP availability. Most useful future correction is adequate source-qualified supervised coverage plus modality-aware calibration/abstention and, if legitimate astrometry provenance can be established, a controlled parallax-input comparison; not another arbitrary density-head swap.
